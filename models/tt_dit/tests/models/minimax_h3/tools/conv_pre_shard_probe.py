# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Why conv_pre stays replicated on the 4x8 mesh: T-sharded against replicated at its shape (2048 -> 1024, k7; 8 KB fp32
sticks) and smaller ones, plus the halo exchange alone: neighbor_pad returns wrong halo rows for sticks over 4 KB."""

import time

import pytest
import torch
from loguru import logger

import ttnn

from .....layers.audio_ops import _AlignedOutConv1d, _t_neighbor_pad
from .....models.audio_vae.minimax_h3.blockings_minimax_h3_audio import register_h3_audio_blockings
from .....parallel.config import ParallelFactor
from .....parallel.manager import CCLManager
from .....pipelines.minimax_h3.pipeline_minimax_h3 import resolve_mesh_preset

MESH = [
    pytest.param(
        (4, 8),
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True, "l1_small_size": 65536},
        id="mesh4x8",
    )
]
# (C_in, C_out, k, T total): conv_pre at 600 latents, and the polyphase ups shapes (stride*out outputs, K' = 3)
SHAPES = [
    (2048, 1024, 7, 600),  # conv_pre: 8 KB sticks, wrong halo rows on several shards
    (1024, 5 * 512, 3, 600),
    (512, 5 * 256, 3, 3000),
    (128, 2 * 64, 3, 30000),
    (2048, 1024, 3, 600),  # conv_pre's stick size (8 KB) with a one-row halo: is it the stick size?
    (1024, 5 * 512, 7, 600),  # a three-row halo with 4 KB sticks: is it the halo width?
    (2048, 1024, 5, 600),  # two-row halo, 8 KB sticks
]


def _timed(mesh_device, fn, n=5):
    out = fn()
    ttnn.synchronize_device(mesh_device)
    best = float("inf")
    for _ in range(n):
        ttnn.synchronize_device(mesh_device)
        mark = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(mesh_device)
        best = min(best, time.perf_counter() - mark)
    return out, best * 1e3


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH, indirect=["mesh_device", "device_params"])
def test_conv_sharded_matches_replicated(mesh_device):
    register_h3_audio_blockings()
    torch.manual_seed(0)
    mesh_rows, mesh_cols = tuple(mesh_device.shape)
    pc = ParallelFactor(factor=mesh_cols, mesh_axis=1)
    preset = resolve_mesh_preset((mesh_rows, mesh_cols), required=False)
    ccl = CCLManager(mesh_device, num_links=preset.get("num_links", 1), topology=preset.get("topology", ttnn.Topology.Linear))
    shard_map = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(mesh_rows, mesh_cols), dims=(None, 1))
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(mesh_rows, mesh_cols), dims=[0, 1])
    common = dict(mesh_device=mesh_device, dtype=ttnn.float32, split_mode="kernel")
    for c_in, c_out, k, t_total in SHAPES:
        state = {"weight": torch.randn(c_out, c_in, k) * (1.0 / (c_in * k) ** 0.5), "bias": torch.randn(c_out) * 0.1}
        full = _AlignedOutConv1d(c_in, c_out, kernel_size=k, **common)
        full.load_torch_state_dict({key: v.clone() for key, v in state.items()})
        shard = _AlignedOutConv1d(c_in, c_out, kernel_size=k, parallel_config=pc, ccl_manager=ccl, **common)
        shard.load_torch_state_dict({key: v.clone() for key, v in state.items()})
        x = torch.randn(2, t_total, c_in)
        x_rep = ttnn.from_torch(x, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
        x_sh = ttnn.from_torch(x, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32, mesh_mapper=shard_map)
        y_full, ms_full = _timed(mesh_device, lambda: full(x_rep))
        try:
            y_sh, ms_sh = _timed(mesh_device, lambda: shard(x_sh))
        except Exception as exc:  # noqa: BLE001
            logger.info(f"CONVSHARD ({c_in}->{c_out},k{k},T{t_total}): sharded form FAILED {type(exc).__name__}: {str(exc)[:160]}")
            continue
        ref = ttnn.to_torch(ttnn.get_device_tensors(y_full)[0]).float()
        got = ttnn.to_torch(y_sh, mesh_composer=composer).float().reshape(mesh_rows, 2, t_total, -1)[0]
        ref = ref[..., : got.shape[-1]]
        d = (got - ref).abs()
        rows_per = t_total // mesh_cols
        per_shard = [float(d[:, s * rows_per : (s + 1) * rows_per].max()) for s in range(mesh_cols)]
        bad_rows = torch.nonzero(d.amax(dim=(0, 2)) > 1e-3).flatten().tolist()
        logger.info(
            f"CONVSHARD ({c_in}->{c_out},k{k},T{t_total}): replicated {ms_full:.3f} ms, sharded {ms_sh:.3f} ms; bit-identical "
            f"{bool(torch.equal(got, ref))}, max |diff| {float(d.max()):.3e}, per-shard max {['%.1e' % v for v in per_shard]}, "
            f"rows > 1e-3: {len(bad_rows)} (first {bad_rows[:6]}, shard boundaries every {rows_per})"
        )
        for t in (x_rep, x_sh, y_full, y_sh):
            ttnn.deallocate(t)


@pytest.mark.timeout(900)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH, indirect=["mesh_device", "device_params"])
def test_neighbor_pad_exact(mesh_device):
    """The T-halo exchange against a host reference at 1024 and 2048 fp32 channels (4 KB and 8 KB sticks), three rows of
    zeros each side: the model's `_t_neighbor_pad` and the raw CCL op side by side."""
    torch.manual_seed(0)
    mesh_rows, mesh_cols = tuple(mesh_device.shape)
    pc = ParallelFactor(factor=mesh_cols, mesh_axis=1)
    preset = resolve_mesh_preset((mesh_rows, mesh_cols), required=False)
    ccl = CCLManager(mesh_device, num_links=preset.get("num_links", 1), topology=preset.get("topology", ttnn.Topology.Linear))
    shard_map = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(mesh_rows, mesh_cols), dims=(None, 1))
    rows, pad = 75, 3
    for c in (1024, 2048):
        x = torch.randn(2, rows * mesh_cols, c)
        x_sh = ttnn.from_torch(x, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32, mesh_mapper=shard_map)
        padded_full = torch.nn.functional.pad(x, (0, 0, pad, pad))  # zeros at the global ends
        expect = [padded_full[:, s * rows : s * rows + rows + 2 * pad] for s in range(mesh_cols)]
        forms = {"_t_neighbor_pad": lambda: _t_neighbor_pad(x_sh, pad_left=pad, pad_right=pad, parallel_config=pc, ccl_manager=ccl, padding_mode="zeros")}
        sem = ccl.get_np_ping_pong_semaphore(pc.mesh_axis)
        forms["raw neighbor_pad"] = lambda: ccl.neighbor_pad_persistent_buffer(
            x_sh, dims=[1], pad_left=[pad], pad_right=[pad], padding_mode="zeros", axes=[pc.mesh_axis], neighbor_sems=[sem], num_links=[max(1, min(2, ccl.num_links))]
        )
        for name, fn in forms.items():
            y = fn()
            ttnn.synchronize_device(mesh_device)
            shards = ttnn.get_device_tensors(y)
            coords = list(y.tensor_topology().mesh_coords())
            worst = 0.0
            bad = []
            for coord, shard in zip(coords, shards):
                col = int(coord[1])
                got = ttnn.to_torch(shard).float()
                d = float((got - expect[col]).abs().max())
                worst = max(worst, d)
                if d > 0:
                    bad.append((int(coord[0]), col, f"{d:.1e}"))
            logger.info(f"NEIGHBORPAD C={c} ({c * 4} B sticks) {name}: max |diff| vs host {worst:.3e}; bad (row, col, diff) {bad[:8]}")
            # not deallocated: both forms hand back the manager's persistent buffer
        ttnn.deallocate(x_sh)
