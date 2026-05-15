# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Validate the tt-lang-emitted recurrent DeltaNet kernel.

This test loads the C++ kernels emitted by
`models/demos/qwen3_6_galaxy_v2/tt/kernels/recurrent_delta_rule_kernel.py`
(compiled in the 3.12 venv) and launches them through `ttnn.generic_op` on
the BH GLX mesh.

The state-update output (state_new) is compared tile-by-tile against the
existing fp32 op chain (state * decay + outer(k, v) * beta).

NOTE: This proof-of-concept kernel handles a SINGLE HEAD per launch with
single-core grid=(1,1). It validates the tt-lang fused matmul + elementwise
pattern at fp32. See the kernel docstring for design rationale.

Run sequentially:
    source python_env/bin/activate
    python -m pytest -s --noconftest \
        models/demos/qwen3_6_galaxy_v2/tests/test_recurrent_delta_rule_kernel.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import torch

import ttnn

KERNELS_DIR = Path(__file__).resolve().parent.parent / "tt" / "kernels" / "recurrent_delta_rule"
COMPUTE_KERNEL_CPP = str(KERNELS_DIR / "recurrent_compute.cpp")
READ_KERNEL_CPP = str(KERNELS_DIR / "recurrent_read.cpp")
WRITE_KERNEL_CPP = str(KERNELS_DIR / "recurrent_write.cpp")

TILE = 32
K_TILES = 4  # head_dim / 32
V_TILES = 4  # head_dim / 32
K_DIM = K_TILES * TILE
V_DIM = V_TILES * TILE

# Tensor signature must match the kernel author script:
#   inputs:  [state, q, k, v, decay, beta]
#   outputs: [state_out, o]
NUM_TENSORS = 8

# From runner: reader tensor indices [5, 4, 2, 0, 3]
#               writer tensor indices [7, 6]
KERNEL_TENSOR_INDICES = [
    [],  # compute
    [5, 4, 2, 0, 3],  # noc reader: beta, decay, k, state, v
    [7, 6],  # noc writer: o, state_out
]


# CBs are fp32 (page_size 4096 = 32*32*4). state_out CB has block_count=4
# (from the kernel: `state_out_dfb = ... block_count=4`).
def _cb_configs():
    cfgs = []
    # CB 0 (state)   block_count=2 -> 8192
    cfgs.append(((1, 1), 2, ttnn.float32, 4096, 8192))
    # CB 1 (q) - unused; create a placeholder anyway
    cfgs.append(((1, 1), 2, ttnn.float32, 4096, 8192))
    # CB 2 (k)
    cfgs.append(((1, 1), 2, ttnn.float32, 4096, 8192))
    # CB 3 (v)
    cfgs.append(((1, 1), 2, ttnn.float32, 4096, 8192))
    # CB 4 (decay)
    cfgs.append(((1, 1), 2, ttnn.float32, 4096, 8192))
    # CB 5 (beta)
    cfgs.append(((1, 1), 2, ttnn.float32, 4096, 8192))
    # CB 6 (state_out) -- block_count=4
    cfgs.append(((1, 1), 4, ttnn.float32, 4096, 16384))
    # CB 7 (o)
    cfgs.append(((1, 1), 2, ttnn.float32, 4096, 8192))
    # CB 8 (o_acc - internal)
    cfgs.append(((1, 1), 2, ttnn.float32, 4096, 8192))
    return cfgs


CB_CONFIGS = _cb_configs()


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
    for i, (_, _, dtype, page_size, total_size) in enumerate(CB_CONFIGS):
        cb_format = ttnn.CBFormatDescriptor(
            buffer_index=i,
            data_format=dtype,
            page_size=page_size,
        )
        cb_desc = ttnn.CBDescriptor(
            total_size=total_size,
            core_ranges=core_ranges,
            format_descriptors=[cb_format],
        )
        cb_descriptors.append(cb_desc)

    cb_indices = list(range(len(CB_CONFIGS)))
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
            # fp32 matmul accuracy: enable fp32_dest_acc_en + HiFi4
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


def _reference_state_update(state_tt, k_col_tt, v_tt, decay_scalar, beta_scalar):
    """Reference: state_new = state * decay + outer(k, v) * beta  (fp32).

    state_tt:  [K, V] fp32 tilized
    k_col_tt:  [K, 1] fp32 tilized (only col 0 valid; shape [K_DIM, TILE])
    v_tt:      [1, V] fp32 tilized (only row 0 valid; shape [TILE, V_DIM])
    decay_scalar, beta_scalar: python float
    """
    mem = ttnn.L1_MEMORY_CONFIG
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # outer = k_col @ v_row : [K_DIM, V_DIM]
    outer = ttnn.matmul(k_col_tt, v_tt, memory_config=mem, compute_kernel_config=cfg)

    # state_new = state * decay + outer * beta (both scalar)
    state_dec = ttnn.multiply(state_tt, decay_scalar, memory_config=mem)
    outer_beta = ttnn.multiply(outer, beta_scalar, memory_config=mem)
    outer.deallocate(True)
    state_new = ttnn.add(state_dec, outer_beta, memory_config=mem)
    state_dec.deallocate(True)
    outer_beta.deallocate(True)
    return state_new


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def test_recurrent_delta_rule_kernel_pcc(bh_glx_mesh):
    """Task 3 (simplified PoC): state-update kernel PCC >= 0.9999."""
    mesh = bh_glx_mesh

    for p in (COMPUTE_KERNEL_CPP, READ_KERNEL_CPP, WRITE_KERNEL_CPP):
        assert Path(p).is_file(), f"missing emitted kernel: {p}"

    torch.manual_seed(0xCAFE)

    state_shape = (K_DIM, V_DIM)  # [128, 128]
    row_shape = (TILE, K_DIM)  # [32, 128] — head row (q, v)
    col_shape = (K_DIM, TILE)  # [128, 32] — head col (k)
    scalar_shape = (TILE, TILE)  # [32, 32]

    state_t = torch.randn(state_shape, dtype=torch.float32)
    k_data = torch.randn(K_DIM, dtype=torch.float32)
    v_data = torch.randn(V_DIM, dtype=torch.float32)
    # k as column [K, 1]: only col 0 valid
    k_t = torch.zeros(col_shape, dtype=torch.float32)
    k_t[:, 0] = k_data
    # v as row [1, V]: only row 0 valid
    v_t = torch.zeros(row_shape, dtype=torch.float32)
    v_t[0, :] = v_data
    # q is unused in the simplified kernel; provide zeros
    q_t = torch.zeros(row_shape, dtype=torch.float32)

    decay_scalar = 0.95
    beta_scalar = 0.3

    # The kernel reads decay/beta as 32x32 tiles, multiplying the WHOLE state
    # tile by the scalar — so we fill the full tile (all 32x32 elements) with
    # the same scalar value, matching ttnn.multiply(state, scalar) semantics.
    decay_tile = torch.full(scalar_shape, decay_scalar, dtype=torch.float32)
    beta_tile = torch.full(scalar_shape, beta_scalar, dtype=torch.float32)

    state = _from_torch_replicated(mesh, state_t)
    q = _from_torch_replicated(mesh, q_t)
    k = _from_torch_replicated(mesh, k_t)
    v = _from_torch_replicated(mesh, v_t)
    decay = _from_torch_replicated(mesh, decay_tile)
    beta = _from_torch_replicated(mesh, beta_tile)

    # Allocate outputs.
    state_out = ttnn.allocate_tensor_on_device(state.spec, mesh)
    o_spec = ttnn.from_torch(
        torch.zeros(row_shape, dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    ).spec
    o = ttnn.allocate_tensor_on_device(o_spec, mesh)

    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    # --- Run fused kernel ---
    _run_kernel(
        [state, q, k, v, decay, beta, state_out, o],
        core_ranges,
    )
    ttnn.synchronize_device(mesh)

    state_kernel = _to_torch_first_device(state_out)

    # --- Reference path (existing op chain) ---
    state_ref_tt = _reference_state_update(state, k, v, decay_scalar, beta_scalar)
    ttnn.synchronize_device(mesh)
    state_ref = _to_torch_first_device(state_ref_tt)

    state_pcc = _pcc(state_kernel, state_ref)
    max_abs = (state_kernel.float() - state_ref.float()).abs().max().item()
    print(f"state PCC     = {state_pcc:.8f}", file=sys.stderr)
    print(f"state max_abs = {max_abs:.6f}", file=sys.stderr)
    print(f"state ref|max = {state_ref.float().abs().max().item():.4f}", file=sys.stderr)
    print(f"state krn|max = {state_kernel.float().abs().max().item():.4f}", file=sys.stderr)

    assert state_pcc >= 0.9999, f"state PCC {state_pcc} < 0.9999"


def test_recurrent_delta_rule_kernel_perf(bh_glx_mesh):
    """Task 4: 100-call latency comparison kernel vs op chain (state update only)."""
    mesh = bh_glx_mesh

    torch.manual_seed(7)

    state_shape = (K_DIM, V_DIM)
    row_shape = (TILE, K_DIM)
    col_shape = (K_DIM, TILE)
    scalar_shape = (TILE, TILE)

    state_t = torch.randn(state_shape, dtype=torch.float32)
    k_data = torch.randn(K_DIM, dtype=torch.float32)
    v_data = torch.randn(V_DIM, dtype=torch.float32)
    k_t = torch.zeros(col_shape, dtype=torch.float32)
    k_t[:, 0] = k_data
    v_t = torch.zeros(row_shape, dtype=torch.float32)
    v_t[0, :] = v_data
    q_t = torch.zeros(row_shape, dtype=torch.float32)
    decay_scalar = 0.95
    beta_scalar = 0.3
    decay_tile = torch.full(scalar_shape, decay_scalar, dtype=torch.float32)
    beta_tile = torch.full(scalar_shape, beta_scalar, dtype=torch.float32)

    state = _from_torch_replicated(mesh, state_t)
    q = _from_torch_replicated(mesh, q_t)
    k = _from_torch_replicated(mesh, k_t)
    v = _from_torch_replicated(mesh, v_t)
    decay = _from_torch_replicated(mesh, decay_tile)
    beta = _from_torch_replicated(mesh, beta_tile)
    state_out = ttnn.allocate_tensor_on_device(state.spec, mesh)
    o_spec = ttnn.from_torch(
        torch.zeros(row_shape, dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    ).spec
    o = ttnn.allocate_tensor_on_device(o_spec, mesh)

    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    N_WARMUP = 5
    N_RUNS = 100

    # Warmup ref chain
    for _ in range(N_WARMUP):
        s = _reference_state_update(state, k, v, decay_scalar, beta_scalar)
        s.deallocate(True)
    ttnn.synchronize_device(mesh)

    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        s = _reference_state_update(state, k, v, decay_scalar, beta_scalar)
        s.deallocate(True)
    ttnn.synchronize_device(mesh)
    t_chain = (time.perf_counter() - t0) / N_RUNS * 1e6

    # Warmup kernel
    for _ in range(N_WARMUP):
        _run_kernel([state, q, k, v, decay, beta, state_out, o], core_ranges)
    ttnn.synchronize_device(mesh)

    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        _run_kernel([state, q, k, v, decay, beta, state_out, o], core_ranges)
    ttnn.synchronize_device(mesh)
    t_kernel = (time.perf_counter() - t0) / N_RUNS * 1e6

    speedup = t_chain / t_kernel if t_kernel > 0 else float("inf")
    print(f"chain   : {t_chain:.2f} us/call", file=sys.stderr)
    print(f"kernel  : {t_kernel:.2f} us/call", file=sys.stderr)
    print(f"speedup : {speedup:.2f}x", file=sys.stderr)
    assert t_kernel > 0
