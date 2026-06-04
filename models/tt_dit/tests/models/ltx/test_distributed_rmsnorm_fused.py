# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
A/B parity test for the LTX distributed RMSNorm fused-op swap.

``DistributedRMSNorm.forward`` has two code paths, selected by the ``LTX_FUSED_AGRMS`` env var:
  * default        -> wan ``rmsnorm_pre_allgather`` -> ``all_gather`` -> ``rmsnorm_post_allgather``
  * LTX_FUSED_AGRMS -> the single fused ``ttnn.all_gather_rms_norm`` op (this PR)

This test builds the norm EXACTLY as the LTX model does (same ctor kwargs / call signature), feeds the
same sharded activation, and runs ``.forward`` both ways, asserting the two outputs match. It covers the
two ways LTX invokes the norm:
  * "block"  : norm1/2/3, audio_norm* -> non-affine, num_heads_per_device=1 (output keeps input shape).
  * "qk"     : norm_q/norm_k          -> affine (weight), num_heads_per_device = num_heads/TP (head-split,
               output reshaped to (1, num_heads, B*N, head_dim)).

RoPE is NOT part of either norm path (LTX applies it as a separate rotary op afterward), so it is not
exercised here. Mesh is (2,4) with the LTX SP/TP axis assignments; the gather runs over the TP axis.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.layers.normalization import DistributedRMSNorm
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.tensor import bf16_tensor_2dshard
from models.tt_dit.utils.test import line_params
from tests.ttnn.utils_for_testing import assert_with_pcc

NUM_HEADS = 32  # LTX video and audio both use 32 attention heads


@pytest.mark.parametrize("device_params", [line_params], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.parametrize(
    "sp_axis, tp_axis",
    [(1, 0), (0, 1)],
    ids=["sp1tp0_tp2", "sp0tp1_tp4"],
)
@pytest.mark.parametrize("dim", [4096, 2048], ids=["video4096", "audio2048"])
@pytest.mark.parametrize("case", ["block", "qk"], ids=["block", "qk"])
@pytest.mark.parametrize("seq_len", [512, 4096, 16384], ids=["seq512", "seq4k", "seq16k"])
def test_distributed_rmsnorm_fused_parity(mesh_device, sp_axis, tp_axis, dim, case, seq_len):
    if mesh_device.get_num_devices() < 8:
        pytest.skip("needs 8 devices for a 2x4 mesh")

    tp = tuple(mesh_device.shape)[tp_axis]
    sp = tuple(mesh_device.shape)[sp_axis]
    affine = case == "qk"
    num_heads_per_device = (NUM_HEADS // tp) if case == "qk" else 1
    eps = 1e-6
    B = 1

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    # Build the norm exactly as LTX does (mesh_axis = TP axis; the reduction-dim hidden is TP-sharded).
    norm = DistributedRMSNorm(
        embedding_dim=dim,
        norm_eps=eps,
        norm_elementwise_affine=affine,
        bias=False,
        mesh_axis=tp_axis,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
    )

    torch.manual_seed(1234)
    if affine:
        # Random weight (gamma), loaded the same way LTX loads norm_q/norm_k weights.
        torch_weight = (torch.rand(dim) * 2 - 1).to(torch.bfloat16)
        norm.load_torch_state_dict({"weight": torch_weight})

    # Activation: (1, B, N, dim), sequence sharded over the SP axis, hidden over the TP axis (= the norm's
    # mesh_axis). Each device holds (1, B, N/sp, dim/tp), matching the norm's expected last dim dim/tp.
    # Enumerated input: every 32-wide width-tile gets a distinct integer (tile_index+1), bf16-EXACT since
    # dim/32 <= 128. After the RMS norm each width-tile becomes a distinct constant, so if the writer
    # scatters tiles to the wrong head/position the misplacement is obvious and exact (not hidden in noise).
    if os.environ.get("PARITY_ENUMERATED", "1") in ("1", "true", "True"):
        d_tile = (torch.arange(dim) // 32 + 1).to(torch.float32)  # (dim,) 1-based per-tile index
        x = d_tile.view(1, 1, 1, dim).expand(1, B, seq_len, dim).contiguous().to(torch.bfloat16)
    else:
        x = (torch.randn(1, B, seq_len, dim) * 4 - 1).to(torch.bfloat16)
    x_tt = bf16_tensor_2dshard(x, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})

    def run(fused: bool):
        prev = os.environ.get("LTX_FUSED_AGRMS")
        prev_qk = os.environ.get("LTX_FUSED_AGRMS_QK")
        os.environ["LTX_FUSED_AGRMS"] = "1" if fused else "0"
        # Q/K head-split fusion is opt-in via LTX_FUSED_AGRMS_QK; set it so the qk case actually exercises
        # the fused op (otherwise the "fused" run falls through to wan and we compare wan-vs-wan).
        os.environ["LTX_FUSED_AGRMS_QK"] = "1" if fused else "0"
        try:
            return norm(x_tt, num_heads_per_device=num_heads_per_device)
        finally:
            for _k, _p in (("LTX_FUSED_AGRMS", prev), ("LTX_FUSED_AGRMS_QK", prev_qk)):
                if _p is None:
                    os.environ.pop(_k, None)
                else:
                    os.environ[_k] = _p

    out_ref = run(fused=False)  # wan pre/AG/post
    out_fused = run(fused=True)  # ttnn.all_gather_rms_norm

    # On-device tensor SPECS (the value/PCC/allclose comparison below gathers to torch and CANNOT see these):
    # if op and wan differ in memory_config / padded(tile) shape / dtype / layout, downstream model ops read
    # the op's (value-correct) output wrong even though parity passes.
    def _spec(t):
        try:
            pad = tuple(t.padded_shape)
        except Exception:
            pad = "?"
        return f"shape={tuple(t.shape)} pad={pad} dtype={t.dtype} layout={t.layout} mem={t.memory_config()}"

    logger.info(f"  SPEC wan: {_spec(out_ref)}")
    logger.info(f"  SPEC op : {_spec(out_fused)}")

    # Output sharding: block -> (1,B,N,dim) [seq/SP, hidden/TP]; qk head-split -> (1,H,M,E) [heads/TP, seq/SP].
    if case == "qk":
        concat_dims = [0, 0]
        concat_dims[sp_axis] = 2  # seq (M)
        concat_dims[tp_axis] = 1  # heads
    else:
        concat_dims = [0, 0]
        concat_dims[sp_axis] = 2  # seq
        concat_dims[tp_axis] = 3  # hidden
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=tuple(concat_dims))

    ref = ttnn.to_torch(out_ref, mesh_composer=composer).float()
    fused = ttnn.to_torch(out_fused, mesh_composer=composer).float()

    assert list(ref.shape) == list(fused.shape), f"shape mismatch: {tuple(ref.shape)} vs {tuple(fused.shape)}"
    diff = (ref - fused).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    # Element-wise tolerance. Same algorithm, different bf16 kernels -> a few % is expected; a layout/
    # permutation/value bug blows this up far past it (and PCC can stay ~1.0 through such bugs). Report the
    # reference's own spread too, so a "heavy match" isn't just trivially low-variance data.
    atol, rtol = 0.05, 0.02
    frac_bad = (diff > atol).float().mean().item()
    is_allclose = torch.allclose(ref, fused, rtol=rtol, atol=atol)
    logger.info(
        f"distributed_rmsnorm parity[{case} dim={dim} tp={tp} sp={sp} H/dev={num_heads_per_device} "
        f"shape={tuple(fused.shape)}] ref[std={ref.std():.3f} min={ref.min():.2f} max={ref.max():.2f}] "
        f"max_abs={max_abs:.4f} mean_abs={mean_abs:.5f} frac>|{atol}|={frac_bad:.4f} allclose={is_allclose}"
    )
    # Structural dump (enumerated input): with per-tile-distinct input, after norm each tile is a distinct
    # constant. For qk the per-head row-0 col-0 value should increase monotonically with head index (head h
    # holds tiles starting at h*head_dim_tiles); for block, row-0 sampled every 32 cols should be the tile
    # ramp. A wrong scatter shows up as out-of-order / mismatched values here.
    if case == "qk":
        logger.info(f"  qk per-head[row0,col0] ref  ={[round(v, 4) for v in ref[0, :, 0, 0].tolist()]}")
        logger.info(f"  qk per-head[row0,col0] fused={[round(v, 4) for v in fused[0, :, 0, 0].tolist()]}")
    else:
        logger.info(f"  block per-tile[row0] ref  ={[round(v, 4) for v in ref[0, 0, 0, ::32][:16].tolist()]}")
        logger.info(f"  block per-tile[row0] fused={[round(v, 4) for v in fused[0, 0, 0, ::32][:16].tolist()]}")

    # Both paths implement the same RMSNorm (+ gamma, + head-split); differences should be only kernel numerics.
    assert_with_pcc(ref, fused, pcc=0.999)
    assert is_allclose, (
        f"allclose FAILED (rtol={rtol} atol={atol}): max_abs={max_abs:.4f} mean_abs={mean_abs:.5f} "
        f"frac_bad={frac_bad:.4f} -- PCC passed but element-wise mismatch => layout/value divergence"
    )


@pytest.mark.parametrize("device_params", [line_params], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.parametrize("sp_axis, tp_axis", [(1, 0)], ids=["sp1tp0_tp2"])  # LTX bh_2x4sp1tp0
@pytest.mark.parametrize("dim", [4096, 2048], ids=["video4096", "audio2048"])
@pytest.mark.parametrize("case", ["block", "qk"], ids=["block", "qk"])
@pytest.mark.parametrize("seq_len", [4096, 16384], ids=["seq4k", "seq16k"])
def test_distributed_rmsnorm_perf(mesh_device, sp_axis, tp_axis, dim, case, seq_len):
    """Op-level perf: time the full norm (wan pre+all_gather+post) vs the fused all_gather_rms_norm op."""
    import time

    if mesh_device.get_num_devices() < 8:
        pytest.skip("needs 8 devices for a 2x4 mesh")

    tp = tuple(mesh_device.shape)[tp_axis]
    affine = case == "qk"
    num_heads_per_device = (NUM_HEADS // tp) if case == "qk" else 1
    B = 1

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    norm = DistributedRMSNorm(
        embedding_dim=dim,
        norm_eps=1e-6,
        norm_elementwise_affine=affine,
        bias=False,
        mesh_axis=tp_axis,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
    )
    torch.manual_seed(1234)
    if affine:
        norm.load_torch_state_dict({"weight": (torch.rand(dim) * 2 - 1).to(torch.bfloat16)})

    x = (torch.randn(1, B, seq_len, dim) * 4 - 1).to(torch.bfloat16)
    x_tt = bf16_tensor_2dshard(x, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})

    def run(fused):
        os.environ["LTX_FUSED_AGRMS"] = "1" if fused else "0"
        os.environ["LTX_FUSED_AGRMS_QK"] = "1" if fused else "0"  # so the head-split (qk) path is actually fused
        return norm(x_tt, num_heads_per_device=num_heads_per_device)

    def time_path(fused, iters=50, warmup=5):
        for _ in range(warmup):
            run(fused)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = run(fused)
        ttnn.synchronize_device(mesh_device)
        return (time.perf_counter() - t0) / iters * 1e6  # us / norm

    wan_us = time_path(False)
    fused_us = time_path(True)
    os.environ.pop("LTX_FUSED_AGRMS", None)
    logger.info(
        f"PERF[{case} dim={dim} seq={seq_len} tp={tp}] "
        f"wan(pre+AG+post)={wan_us:.1f}us  fused(no-mux)={fused_us:.1f}us  speedup={wan_us / fused_us:.2f}x"
    )
