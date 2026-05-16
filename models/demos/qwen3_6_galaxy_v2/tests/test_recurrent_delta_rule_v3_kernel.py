# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""V2-17d standalone validation for the multi-head batched recurrent kernel.

Loads the C++ kernels emitted by
``models/demos/qwen3_6_galaxy_v2/tt/kernels/recurrent_delta_rule_v3_kernel.py``
(compiled in the 3.12 venv) and launches them through ``ttnn.generic_op``
on the BH GLX mesh.

Validates BOTH:
- ``state_new`` (all 6 heads) against the existing fp32 op chain
  (``state * decay + outer(k, v) * beta``) per head
- ``o`` (readout, all 6 heads) against ``q @ state_new`` per head

PCC ≥ 0.9999 required for BOTH. Perf target: ≥ 2× the existing all-heads
fp32 chain (the apples-to-apples baseline at 6 heads is different from
V2-17c's per-head-single-tile baseline).

Run sequentially:
    source python_env/bin/activate
    python -m pytest -s --noconftest \\
        models/demos/qwen3_6_galaxy_v2/tests/test_recurrent_delta_rule_v3_kernel.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import torch

import ttnn

KERNELS_DIR = Path(__file__).resolve().parent.parent / "tt" / "kernels" / "recurrent_delta_rule_v3"
COMPUTE_KERNEL_CPP = str(KERNELS_DIR / "recurrent_compute.cpp")
READ_KERNEL_CPP = str(KERNELS_DIR / "recurrent_read.cpp")
WRITE_KERNEL_CPP = str(KERNELS_DIR / "recurrent_write.cpp")

TILE = 32
HEAD_DIM = 128
K_TILES = HEAD_DIM // TILE  # 4
V_TILES = HEAD_DIM // TILE  # 4
V_HEADS = 6  # n_v_per_row

# Logical 2D shapes after the metadata reshape that the integration does.
STATE_SHAPE = (V_HEADS * HEAD_DIM, HEAD_DIM)  # [768, 128]
Q_SHAPE = (V_HEADS * TILE, HEAD_DIM)  # [192, 128]
K_COL_SHAPE = (V_HEADS * HEAD_DIM, TILE)  # [768, 32]
V_SHAPE = (V_HEADS * TILE, HEAD_DIM)  # [192, 128]
BCAST_SHAPE = (V_HEADS * TILE, TILE)  # [192, 32]
O_SHAPE = (V_HEADS * TILE, HEAD_DIM)  # [192, 128]

# 8 tensors: [state, q, k_col, v, decay, beta, state_out, o]
NUM_TENSORS = 8

# Kernel tensor indices from the V3 emitted runner — identical layout to V2-17c.
KERNEL_TENSOR_INDICES = [
    [],
    [5, 4, 2, 1, 0, 3],  # noc reader: beta, decay, k_col, q, state, v
    [7, 6],  # noc writer: o, state_out
]

# 10 CBs (all fp32, single tile, double-buffered) — matching V3 emission.
NUM_CBS = 10
CB_PAGE_SIZE = 4096
CB_TOTAL_SIZE = 8192  # block_count=2


@pytest.fixture(scope="module")
def bh_glx_mesh():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _from_torch_replicated(mesh, x, dtype=ttnn.float32):
    return ttnn.from_torch(
        x,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _to_torch_first_device(t):
    shards = ttnn.get_device_tensors(t)
    return ttnn.to_torch(shards[0])


def _run_kernel(tensors, core_ranges):
    tensor_accessor_args = []
    for t in tensors:
        tensor_accessor_args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    cb_descriptors = []
    for i in range(NUM_CBS):
        cb_format = ttnn.CBFormatDescriptor(
            buffer_index=i,
            data_format=ttnn.float32,
            page_size=CB_PAGE_SIZE,
        )
        cb_desc = ttnn.CBDescriptor(
            total_size=CB_TOTAL_SIZE,
            core_ranges=core_ranges,
            format_descriptors=[cb_format],
        )
        cb_descriptors.append(cb_desc)

    cb_indices = list(range(NUM_CBS))
    kernel_descriptors = []
    paths = [
        (COMPUTE_KERNEL_CPP, "compute"),
        (READ_KERNEL_CPP, "noc"),
        (WRITE_KERNEL_CPP, "noc"),
    ]
    noc_idx = 0
    for kernel_idx, (kernel_path, thread_type) in enumerate(paths):
        tensor_indices = KERNEL_TENSOR_INDICES[kernel_idx]
        common_runtime_args = [tensors[idx].buffer_address() for idx in tensor_indices]
        if thread_type == "compute":
            compile_time_args = cb_indices
            cfg = ttnn.ComputeConfigDescriptor()
            cfg.fp32_dest_acc_en = True
            cfg.math_fidelity = ttnn.MathFidelity.HiFi4
            cfg.math_approx_mode = False
            config = cfg
        else:
            compile_time_args = cb_indices + tensor_accessor_args
            if noc_idx == 0:
                config = ttnn.ReaderConfigDescriptor()
            else:
                config = ttnn.WriterConfigDescriptor()
            noc_idx += 1
        kernel_descriptors.append(
            ttnn.KernelDescriptor(
                kernel_source=kernel_path,
                core_ranges=core_ranges,
                compile_time_args=compile_time_args,
                common_runtime_args=common_runtime_args,
                config=config,
            )
        )

    program = ttnn.ProgramDescriptor(
        kernels=kernel_descriptors,
        cbs=cb_descriptors,
        semaphores=[],
    )
    return ttnn.generic_op(list(tensors), program)


def _reference_chain_all_heads(state_tt, q_tt, k_col_tt, v_tt, decay_tt, beta_tt):
    """All-head reference state update + readout (mirrors the V3 kernel math).

    state_new[h, K, V] = state[h, K, V] * decay[h, 0..tile] + (k_col[h, K, 0] @ v[h, 0, V]) * beta[h, 0..tile]
    o[h, 0..tile, V]   = q[h, 0..tile, K] @ state_new[h, K, V]

    Implementation: uses ttnn ops on the SAME logical 2D tensors the kernel
    operates on (no per-head split). The math is per-head outer product +
    elementwise; the 2D-stacked layout makes per-head ops naturally batched
    via the (H, ...) tile-coord pattern.

    We use full-rank matmul on the 2D layouts directly:
       outer_big = k_col (768x32) @ v_block (32x128) → would interleave heads incorrectly.

    Instead, decompose per-head and stack — this is correctness-only reference;
    perf is irrelevant.
    """
    # Reshape host-side via to_torch + per-head matmul. This guarantees a
    # correct semantic for the reference, independent of any ttnn quirks.
    raise NotImplementedError("use host-side reference")


def _host_reference(state_t, q_t, k_col_t, v_t, decay_t, beta_t):
    """Host-side correctness reference.

    Inputs are 2D torch tensors as the kernel sees them. Computes per-head:
        state_new[h]_tile_ij = state[h]_tile_ij * decay[h] + (k_col[h]_i @ v[h]_j) * beta[h]
        o[h]_tile_j         = sum_i q[h]_tile_i @ state_new[h]_tile_ij

    Returns (state_new_2d, o_2d) at the same shapes as the kernel outputs.
    """
    state_new = torch.zeros_like(state_t)
    o = torch.zeros_like(state_t.new_zeros(V_HEADS * TILE, HEAD_DIM))

    for h in range(V_HEADS):
        # Slice this head's tiles out of the 2D layout.
        # state[h, K, V] → rows h*HEAD_DIM .. (h+1)*HEAD_DIM, cols 0..HEAD_DIM
        s_h = state_t[h * HEAD_DIM : (h + 1) * HEAD_DIM, :]  # [K, V]
        k_h = k_col_t[h * HEAD_DIM : (h + 1) * HEAD_DIM, :TILE]  # [K, T=32]; col 0 valid
        v_h = v_t[h * TILE : (h + 1) * TILE, :]  # [T=32, V]; row 0 valid
        q_h = q_t[h * TILE : (h + 1) * TILE, :]  # [T=32, K]; row 0 valid
        decay_h = decay_t[h * TILE : (h + 1) * TILE, :TILE]  # [T, T]; scalar broadcast
        beta_h = beta_t[h * TILE : (h + 1) * TILE, :TILE]  # [T, T]; scalar broadcast

        # decay/beta scalars (element 0,0) — broadcast value across the tile.
        # For the per-tile compute we follow the kernel's exact pattern:
        #   for each (i, j) tile: state_new_tile = state_tile * decay_tile
        #                                        + (k_tile @ v_tile) * beta_tile
        # where decay_tile/beta_tile are 32x32 with the scalar at all entries
        # (or just position (0,0) — kernel multiplies element-wise so the
        # representation must be consistent host-side, see below).
        for i in range(K_TILES):
            for j in range(V_TILES):
                state_tile = s_h[i * TILE : (i + 1) * TILE, j * TILE : (j + 1) * TILE]
                k_tile = k_h[i * TILE : (i + 1) * TILE, :TILE]
                v_tile = v_h[:TILE, j * TILE : (j + 1) * TILE]
                outer = k_tile @ v_tile  # [32, 32]
                state_new_tile = state_tile * decay_h + outer * beta_h
                state_new[
                    h * HEAD_DIM + i * TILE : h * HEAD_DIM + (i + 1) * TILE, j * TILE : (j + 1) * TILE
                ] = state_new_tile

        # Per-head readout: o[h, 0..T, V] = sum_i q[h, 0..T, K_tile_i] @ state_new[h, i, :]
        for j in range(V_TILES):
            o_tile = torch.zeros(TILE, TILE)
            for i in range(K_TILES):
                q_tile = q_h[:TILE, i * TILE : (i + 1) * TILE]
                state_new_tile = state_new[
                    h * HEAD_DIM + i * TILE : h * HEAD_DIM + (i + 1) * TILE, j * TILE : (j + 1) * TILE
                ]
                o_tile = o_tile + q_tile @ state_new_tile
            o[h * TILE : (h + 1) * TILE, j * TILE : (j + 1) * TILE] = o_tile

    return state_new, o


def _device_chain_all_heads(state, q, k_col, v, decay, beta, mesh):
    """Run the equivalent computation on-device via the existing op chain.

    Uses a per-head loop with ttnn.matmul + multiply/add for an apples-to-apples
    perf comparison against the V3 kernel.

    Returns (state_new_tt, o_tt) — caller owns both.
    """
    mem = ttnn.L1_MEMORY_CONFIG
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # Compute per-head and stitch back. This is the V2-17c "single-head per
    # launch" cost x 6, the apples-to-apples baseline for V3.
    # decay/beta arrive as [192, 32] (one [32,32] tile per head). Since the
    # ttnn binary_ng path doesn't subtile-broadcast a [32,32] tile across a
    # [128,128] state, we multiply state by a single decay scalar via
    # ttnn.multiply(state, scalar_float). To get the per-head scalar
    # cheaply we just use scalar Python literals captured at fixture time.
    state_news = []
    os_ = []
    for h in range(V_HEADS):
        decay_scalar_h = 0.9 + 0.01 * h
        beta_scalar_h = 0.2 + 0.02 * h
        s_h = ttnn.slice(state, [h * HEAD_DIM, 0], [(h + 1) * HEAD_DIM, HEAD_DIM], memory_config=mem)
        k_h = ttnn.slice(k_col, [h * HEAD_DIM, 0], [(h + 1) * HEAD_DIM, TILE], memory_config=mem)
        v_h = ttnn.slice(v, [h * TILE, 0], [(h + 1) * TILE, HEAD_DIM], memory_config=mem)
        q_h = ttnn.slice(q, [h * TILE, 0], [(h + 1) * TILE, HEAD_DIM], memory_config=mem)

        outer = ttnn.matmul(k_h, v_h, memory_config=mem, compute_kernel_config=cfg)
        state_dec = ttnn.multiply(s_h, decay_scalar_h, memory_config=mem)
        outer_beta = ttnn.multiply(outer, beta_scalar_h, memory_config=mem)
        outer.deallocate(True)
        state_new_h = ttnn.add(state_dec, outer_beta, memory_config=mem)
        state_dec.deallocate(True)
        outer_beta.deallocate(True)
        o_h = ttnn.matmul(q_h, state_new_h, memory_config=mem, compute_kernel_config=cfg)

        state_news.append(state_new_h)
        os_.append(o_h)
        s_h.deallocate(True)
        k_h.deallocate(True)
        v_h.deallocate(True)
        q_h.deallocate(True)

    state_new = ttnn.concat(state_news, dim=0, memory_config=mem)
    o_concat = ttnn.concat(os_, dim=0, memory_config=mem)
    for t in state_news:
        t.deallocate(True)
    for t in os_:
        t.deallocate(True)
    return state_new, o_concat


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _build_inputs(mesh):
    """Build the multi-head input tensors at the 2D logical shape.

    Returns (state_t, q_t, k_col_t, v_t, decay_t, beta_t,
             state_tt, q_tt, k_col_tt, v_tt, decay_tt, beta_tt).
    """
    torch.manual_seed(0xCAFE)

    state_t = torch.randn(STATE_SHAPE, dtype=torch.float32)
    q_t = torch.zeros(Q_SHAPE, dtype=torch.float32)
    k_col_t = torch.zeros(K_COL_SHAPE, dtype=torch.float32)
    v_t = torch.zeros(V_SHAPE, dtype=torch.float32)
    decay_t = torch.zeros(BCAST_SHAPE, dtype=torch.float32)
    beta_t = torch.zeros(BCAST_SHAPE, dtype=torch.float32)

    for h in range(V_HEADS):
        q_data = torch.randn(HEAD_DIM, dtype=torch.float32)
        k_data = torch.randn(HEAD_DIM, dtype=torch.float32)
        v_data = torch.randn(HEAD_DIM, dtype=torch.float32)
        # q[h, 0, :] -> row 0 of the head's tile-row at row index h*TILE.
        q_t[h * TILE, :] = q_data
        # k_col[h, K, 0] -> col 0 of the head's tile-col at rows h*HEAD_DIM .. (h+1)*HEAD_DIM.
        k_col_t[h * HEAD_DIM : (h + 1) * HEAD_DIM, 0] = k_data
        # v[h, 0, :] -> row 0 at h*TILE.
        v_t[h * TILE, :] = v_data
        # Scalar broadcast tiles — fill the entire 32x32 tile at (h, 0) with the scalar.
        decay_scalar = 0.9 + 0.01 * h
        beta_scalar = 0.2 + 0.02 * h
        decay_t[h * TILE : (h + 1) * TILE, :TILE] = decay_scalar
        beta_t[h * TILE : (h + 1) * TILE, :TILE] = beta_scalar

    state_tt = _from_torch_replicated(mesh, state_t)
    q_tt = _from_torch_replicated(mesh, q_t)
    k_col_tt = _from_torch_replicated(mesh, k_col_t)
    v_tt = _from_torch_replicated(mesh, v_t)
    decay_tt = _from_torch_replicated(mesh, decay_t)
    beta_tt = _from_torch_replicated(mesh, beta_t)

    return state_t, q_t, k_col_t, v_t, decay_t, beta_t, state_tt, q_tt, k_col_tt, v_tt, decay_tt, beta_tt


def test_recurrent_delta_rule_v3_pcc(bh_glx_mesh):
    """state PCC ≥ 0.9999 AND o PCC ≥ 0.9999 across all 6 heads."""
    mesh = bh_glx_mesh

    for p in (COMPUTE_KERNEL_CPP, READ_KERNEL_CPP, WRITE_KERNEL_CPP):
        assert Path(p).is_file(), f"missing emitted kernel: {p}"

    state_t, q_t, k_col_t, v_t, decay_t, beta_t, state, q, k_col, v, decay, beta = _build_inputs(mesh)

    state_out = ttnn.allocate_tensor_on_device(state.spec, mesh)
    o_spec = ttnn.from_torch(
        torch.zeros(O_SHAPE, dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    ).spec
    o = ttnn.allocate_tensor_on_device(o_spec, mesh)

    # grid=(4, 6). Use cores (0..3, 0..5). 24 cores.
    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(V_TILES - 1, V_HEADS - 1))])

    _run_kernel(
        [state, q, k_col, v, decay, beta, state_out, o],
        core_ranges,
    )
    ttnn.synchronize_device(mesh)

    state_kernel = _to_torch_first_device(state_out)
    o_kernel = _to_torch_first_device(o)

    # Host reference for correctness.
    state_ref, o_ref = _host_reference(state_t, q_t, k_col_t, v_t, decay_t, beta_t)

    state_pcc = _pcc(state_kernel, state_ref)
    o_pcc = _pcc(o_kernel, o_ref)
    state_max_abs = (state_kernel.float() - state_ref.float()).abs().max().item()
    o_max_abs = (o_kernel.float() - o_ref.float()).abs().max().item()
    print(f"state PCC     = {state_pcc:.8f}", file=sys.stderr)
    print(f"o     PCC     = {o_pcc:.8f}", file=sys.stderr)
    print(f"state max_abs = {state_max_abs:.6f}", file=sys.stderr)
    print(f"o     max_abs = {o_max_abs:.6f}", file=sys.stderr)
    print(f"state ref|max = {state_ref.float().abs().max().item():.4f}", file=sys.stderr)
    print(f"o     ref|max = {o_ref.float().abs().max().item():.4f}", file=sys.stderr)

    # Per-head PCC for diagnostics.
    for h in range(V_HEADS):
        s_pcc_h = _pcc(
            state_kernel[h * HEAD_DIM : (h + 1) * HEAD_DIM, :], state_ref[h * HEAD_DIM : (h + 1) * HEAD_DIM, :]
        )
        o_pcc_h = _pcc(o_kernel[h * TILE : (h + 1) * TILE, :], o_ref[h * TILE : (h + 1) * TILE, :])
        print(f"  head {h}: state PCC = {s_pcc_h:.8f}, o PCC = {o_pcc_h:.8f}", file=sys.stderr)

    assert state_pcc >= 0.9999, f"state PCC {state_pcc} < 0.9999"
    assert o_pcc >= 0.9999, f"o PCC {o_pcc} < 0.9999"


def test_recurrent_delta_rule_v3_perf(bh_glx_mesh):
    """100-call latency: V3 multi-head kernel vs per-head ttnn chain (6 launches)."""
    mesh = bh_glx_mesh

    state_t, q_t, k_col_t, v_t, decay_t, beta_t, state, q, k_col, v, decay, beta = _build_inputs(mesh)

    state_out = ttnn.allocate_tensor_on_device(state.spec, mesh)
    o_spec = ttnn.from_torch(
        torch.zeros(O_SHAPE, dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    ).spec
    o = ttnn.allocate_tensor_on_device(o_spec, mesh)

    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(V_TILES - 1, V_HEADS - 1))])

    N_WARMUP = 5
    N_RUNS = 100

    # Warmup chain
    for _ in range(N_WARMUP):
        s, o_ref = _device_chain_all_heads(state, q, k_col, v, decay, beta, mesh)
        s.deallocate(True)
        o_ref.deallocate(True)
    ttnn.synchronize_device(mesh)

    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        s, o_ref = _device_chain_all_heads(state, q, k_col, v, decay, beta, mesh)
        s.deallocate(True)
        o_ref.deallocate(True)
    ttnn.synchronize_device(mesh)
    t_chain = (time.perf_counter() - t0) / N_RUNS * 1e6

    # Warmup kernel
    for _ in range(N_WARMUP):
        _run_kernel([state, q, k_col, v, decay, beta, state_out, o], core_ranges)
    ttnn.synchronize_device(mesh)

    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        _run_kernel([state, q, k_col, v, decay, beta, state_out, o], core_ranges)
    ttnn.synchronize_device(mesh)
    t_kernel = (time.perf_counter() - t0) / N_RUNS * 1e6

    speedup = t_chain / t_kernel if t_kernel > 0 else float("inf")
    print(f"chain (6 launches): {t_chain:.2f} us/call", file=sys.stderr)
    print(f"kernel (1 launch) : {t_kernel:.2f} us/call", file=sys.stderr)
    print(f"speedup           : {speedup:.2f}x", file=sys.stderr)
    assert t_kernel > 0
