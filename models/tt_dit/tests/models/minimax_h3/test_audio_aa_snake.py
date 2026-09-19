# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The fused anti-aliased SnakeBeta kernel (`layers/audio_aa_snake.py`) against the `Activation1d` chain it
replaces, one chip, at the H3 vocoder's per-device shapes (600 latents, stereo, T-shard 8).

Staged by the kernel's compile-time STAGE: 0 copies the input through the tile gathers (the block-as-tile identity
and the address patching across two calls), 1 runs the two resamplers without the activation, 2 is the full
activation. Every stage must be bit-identical to its reference (`torch.equal`); both forms are timed.
    AA_STAGES=0 pytest models/tt_dit/tests/models/minimax_h3/test_audio_aa_snake.py -s
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn

from ....layers.audio_aa_snake import FusedActivation1d
from ....layers.audio_ops import SnakeBeta
from ....layers.audio_resample import Activation1d, DownSample1d, UpSample1d
from ....parallel.config import ParallelFactor
from ....parallel.manager import CCLManager
from ....pipelines.minimax_h3.pipeline_minimax_h3 import resolve_mesh_preset

SINGLE_DEVICE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]
# The pipeline's audio layout: T-shard 8 over mesh axis 1 (the rows of the 4x8 mesh hold replicas).
MESH = [
    pytest.param(
        (4, 8),
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True, "l1_small_size": 65536},
        id="mesh4x8_tshard8",
    )
]

# (channels, pack, rows): the tensor is (2, rows, pack * channels), i.e. band 0, 1, 4, the packed bands 5-6 and act_post.
SHAPES = [
    pytest.param(256, 1, 1875, id="band1_c256"),
    pytest.param(512, 1, 375, id="band0_c512"),
    pytest.param(32, 1, 15000, id="band4_c32"),
    pytest.param(16, 2, 15000, id="band5_c16_k2"),
    pytest.param(8, 4, 15000, id="band6_c8_k4"),
    pytest.param(8, 1, 60000, id="post_c8"),
]
STAGES = [int(s) for s in os.environ.get("AA_STAGES", "0,1,2").split(",") if s.strip()]
# Per-device rows on the mesh; the last one leaves cores idle (32 tiles for 60 cores per batch item).
MESH_SHAPES = SHAPES + [pytest.param(256, 1, 128, id="band1_c256_spare_cores")]


def _best(fn, mesh_device, n=5):
    out = fn()
    ttnn.synchronize_device(mesh_device)
    best = float("inf")
    for _ in range(n):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(mesh_device)
        best = min(best, time.perf_counter() - t0)
    return best, out


def _device(x, mesh_device):
    return ttnn.from_torch(x, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize(("channels", "pack", "rows"), SHAPES)
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_fused_matches_chain(mesh_device, channels, pack, rows, stage):
    torch.manual_seed(0)
    batch = 2
    x = torch.randn(batch, rows * pack, channels) * 0.5
    log_alpha = torch.randn(channels) * 0.3
    log_beta = torch.randn(channels) * 0.3
    state = {"act.alpha": log_alpha.clone(), "act.beta": log_beta.clone()}
    common = dict(mesh_device=mesh_device, dtype=ttnn.float32)

    fused = FusedActivation1d(channels=channels, stage=stage, **common)
    fused.load_torch_state_dict(dict(state))

    if stage == 0:
        reference = lambda xd: xd  # noqa: E731
    elif stage == 1:
        up = UpSample1d(ratio=2, window="kaiser", **common)
        down = DownSample1d(ratio=2, **common)
        up.load_torch_state_dict({})
        down.load_torch_state_dict({})
        reference = lambda xd: down(up(xd))  # noqa: E731
    else:
        act = Activation1d(channels=channels, activation=SnakeBeta(channels, alpha_logscale=True, **common), **common)
        act.load_torch_state_dict(dict(state))
        reference = act

    def run(x_cpu):
        x_ref = _device(x_cpu, mesh_device)
        x_fused = x_ref if pack == 1 else _device(x_cpu.reshape(batch, rows, pack * channels), mesh_device)
        t_ref, out_ref = _best(lambda: reference(x_ref), mesh_device)
        t_fused, out_fused = _best(lambda: fused(x_fused), mesh_device)
        ref = ttnn.to_torch(out_ref).float().reshape(batch, rows * pack, channels)
        got = ttnn.to_torch(out_fused).float().reshape(batch, rows * pack, channels)
        return t_ref, t_fused, ref, got

    # Two inputs: the second call hits the program cache with new buffer addresses.
    for tag, x_cpu in (("a", x), ("b", x * 0.5 + 0.1)):
        t_ref, t_fused, ref, got = run(x_cpu)
        diff = (got.double() - ref.double()).abs()
        n_diff = int((got != ref).sum())
        rel = float(diff.norm() / ref.double().norm().clamp_min(1e-30))
        logger.info(
            f"AA_RESULT stage={stage} c={channels} k={pack} rows={rows} input={tag}: chain {t_ref * 1e3:.3f} ms, "
            f"fused {t_fused * 1e3:.3f} ms ({t_ref / max(t_fused, 1e-9):.1f}x); differing values {n_diff} / {ref.numel()}, "
            f"max |diff| {float(diff.max()):.3e}, rel-RMSE {rel:.3e}"
        )
        if n_diff:
            idx = torch.nonzero(got != ref)[:8]
            for b_i, t_i, c_i in idx.tolist():
                logger.info(f"  first mismatches: [{b_i},{t_i},{c_i}] got {got[b_i, t_i, c_i].item()!r} ref {ref[b_i, t_i, c_i].item()!r}")
        assert torch.equal(got, ref), f"stage {stage}: {n_diff} values differ (max |diff| {float(diff.max()):.3e})"


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(("channels", "pack", "rows"), MESH_SHAPES)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH, indirect=["mesh_device", "device_params"])
def test_fused_matches_chain_t_sharded(mesh_device, channels, pack, rows):
    """T-sharded like the vocoder: the halo pages from the neighbours, the per-stick clamp at the two sequence ends (also
    for packed rows, where the halo holds whole rows), the per-device flags and the idle cores, against the unpacked
    ``Activation1d`` on the same shards. Bit-identical on every device (``torch.equal`` over the gathered sequence)."""
    torch.manual_seed(0)
    mesh_rows, mesh_cols = tuple(mesh_device.shape)
    pc = ParallelFactor(factor=mesh_cols, mesh_axis=1)
    preset = resolve_mesh_preset((mesh_rows, mesh_cols), required=False)
    ccl = CCLManager(mesh_device, num_links=preset.get("num_links", 1), topology=preset.get("topology", ttnn.Topology.Linear))
    batch = 2
    total = rows * mesh_cols
    x = torch.randn(batch, total * pack, channels) * 0.5
    state = {"act.alpha": torch.randn(channels) * 0.3, "act.beta": torch.randn(channels) * 0.3}
    common = dict(mesh_device=mesh_device, dtype=ttnn.float32, parallel_config=pc, ccl_manager=ccl)

    fused = FusedActivation1d(channels=channels, **common)
    fused.load_torch_state_dict(dict(state))
    act = Activation1d(
        channels=channels,
        activation=SnakeBeta(channels, alpha_logscale=True, mesh_device=mesh_device, dtype=ttnn.float32, parallel_config=pc),
        **common,
    )
    act.load_torch_state_dict(dict(state))

    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(mesh_rows, mesh_cols), dims=(None, 1))
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(mesh_rows, mesh_cols), dims=[0, 1])

    def shard(t):
        return ttnn.from_torch(t, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32, mesh_mapper=mapper)

    def gather(t):
        # (mesh_rows * batch, total_rows, width): the mesh rows are replicas and must agree.
        full = ttnn.to_torch(t, mesh_composer=composer).float()
        full = full.reshape(mesh_rows, batch, -1, full.shape[-1])
        for r in range(1, mesh_rows):
            assert torch.equal(full[r], full[0]), f"mesh row {r} differs from row 0"
        return full[0].reshape(batch, total * pack, channels)

    for tag, x_cpu in (("a", x), ("b", x * 0.5 + 0.1)):
        x_ref = shard(x_cpu)
        x_fused = x_ref if pack == 1 else shard(x_cpu.reshape(batch, total, pack * channels))
        t_ref, out_ref = _best(lambda: act(x_ref), mesh_device)
        t_fused, out_fused = _best(lambda: fused(x_fused), mesh_device)
        ref, got = gather(out_ref), gather(out_fused)
        diff = (got.double() - ref.double()).abs()
        n_diff = int((got != ref).sum())
        logger.info(
            f"AA_MESH c={channels} k={pack} rows/device={rows} input={tag}: chain {t_ref * 1e3:.3f} ms, fused {t_fused * 1e3:.3f} ms "
            f"({t_ref / max(t_fused, 1e-9):.1f}x); differing values {n_diff} / {ref.numel()}, max |diff| {float(diff.max()):.3e}"
        )
        if n_diff:
            idx = torch.nonzero(got != ref)[:8]
            for b_i, t_i, c_i in idx.tolist():
                logger.info(f"  first mismatches: [{b_i},{t_i},{c_i}] got {got[b_i, t_i, c_i].item()!r} ref {ref[b_i, t_i, c_i].item()!r}")
        assert torch.equal(got, ref), f"{n_diff} values differ on the mesh (max |diff| {float(diff.max()):.3e})"
