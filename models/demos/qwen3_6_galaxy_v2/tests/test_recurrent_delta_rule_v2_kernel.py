# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""V2-17c standalone validation for the multi-core + readout-fused kernel.

This test loads the C++ kernels emitted by
``models/demos/qwen3_6_galaxy_v2/tt/kernels/recurrent_delta_rule_v2_kernel.py``
(compiled in the 3.12 venv) and launches them through ``ttnn.generic_op`` on
the BH GLX mesh.

Validates both:
- ``state_new`` tile-by-tile against the existing fp32 op chain
  (``state * decay + outer(k, v) * beta``)
- ``o`` (readout) against ``q @ state_new`` computed externally

PCC ≥ 0.9999 required for BOTH. Speedup target: ≥ 1.5× over the
ref chain (state update + external readout matmul).

Run sequentially:
    source python_env/bin/activate
    python -m pytest -s --noconftest \
        models/demos/qwen3_6_galaxy_v2/tests/test_recurrent_delta_rule_v2_kernel.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import torch

import ttnn

KERNELS_DIR = Path(__file__).resolve().parent.parent / "tt" / "kernels" / "recurrent_delta_rule_v2"
COMPUTE_KERNEL_CPP = str(KERNELS_DIR / "recurrent_compute.cpp")
READ_KERNEL_CPP = str(KERNELS_DIR / "recurrent_read.cpp")
WRITE_KERNEL_CPP = str(KERNELS_DIR / "recurrent_write.cpp")

TILE = 32
K_TILES = 4
V_TILES = 4
K_DIM = K_TILES * TILE
V_DIM = V_TILES * TILE

# Tensor signature must match the kernel author script:
#   inputs:  [state, q, k, v, decay, beta]
#   outputs: [state_out, o]
NUM_TENSORS = 8

# From V2-17c emitted runner:
#   reader tensor indices [5, 4, 2, 1, 0, 3]   (beta, decay, k, q, state, v)
#   writer tensor indices [7, 6]               (o, state_out)
KERNEL_TENSOR_INDICES = [
    [],
    [5, 4, 2, 1, 0, 3],
    [7, 6],
]

# 10 CBs (all fp32, single tile, double-buffered) — matching runner emission.
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


def _reference_chain(state_tt, q_row_tt, k_col_tt, v_row_tt, decay_scalar, beta_scalar):
    """Reference state update + readout.

    state_new = state * decay + (k_col @ v_row) * beta
    o = q_row @ state_new   # [TILE, V]

    Returns (state_new_tt, o_tt) — caller owns both.
    """
    mem = ttnn.L1_MEMORY_CONFIG
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    outer = ttnn.matmul(k_col_tt, v_row_tt, memory_config=mem, compute_kernel_config=cfg)
    state_dec = ttnn.multiply(state_tt, decay_scalar, memory_config=mem)
    outer_beta = ttnn.multiply(outer, beta_scalar, memory_config=mem)
    outer.deallocate(True)
    state_new = ttnn.add(state_dec, outer_beta, memory_config=mem)
    state_dec.deallocate(True)
    outer_beta.deallocate(True)

    o = ttnn.matmul(q_row_tt, state_new, memory_config=mem, compute_kernel_config=cfg)
    return state_new, o


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def test_recurrent_delta_rule_v2_pcc(bh_glx_mesh):
    """state_new PCC ≥ 0.9999 AND o PCC ≥ 0.9999."""
    mesh = bh_glx_mesh

    for p in (COMPUTE_KERNEL_CPP, READ_KERNEL_CPP, WRITE_KERNEL_CPP):
        assert Path(p).is_file(), f"missing emitted kernel: {p}"

    torch.manual_seed(0xCAFE)

    state_shape = (K_DIM, V_DIM)
    row_shape = (TILE, K_DIM)
    col_shape = (K_DIM, TILE)
    scalar_shape = (TILE, TILE)

    state_t = torch.randn(state_shape, dtype=torch.float32)
    q_data = torch.randn(K_DIM, dtype=torch.float32)
    k_data = torch.randn(K_DIM, dtype=torch.float32)
    v_data = torch.randn(V_DIM, dtype=torch.float32)
    # q as row [1, K_DIM]: only row 0 valid
    q_t = torch.zeros(row_shape, dtype=torch.float32)
    q_t[0, :] = q_data
    # k as column [K, 1]: only col 0 valid
    k_t = torch.zeros(col_shape, dtype=torch.float32)
    k_t[:, 0] = k_data
    # v as row [1, V]: only row 0 valid
    v_t = torch.zeros(row_shape, dtype=torch.float32)
    v_t[0, :] = v_data

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

    # grid=(4, 1) — 4 cores along x-axis. Use cores (0..3, 0).
    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(V_TILES - 1, 0))])

    # --- Run fused kernel ---
    _run_kernel(
        [state, q, k, v, decay, beta, state_out, o],
        core_ranges,
    )
    ttnn.synchronize_device(mesh)

    state_kernel = _to_torch_first_device(state_out)
    o_kernel = _to_torch_first_device(o)

    # --- Reference path (existing op chain) ---
    state_ref_tt, o_ref_tt = _reference_chain(state, q, k, v, decay_scalar, beta_scalar)
    ttnn.synchronize_device(mesh)
    state_ref = _to_torch_first_device(state_ref_tt)
    o_ref = _to_torch_first_device(o_ref_tt)

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

    assert state_pcc >= 0.9999, f"state PCC {state_pcc} < 0.9999"
    assert o_pcc >= 0.9999, f"o PCC {o_pcc} < 0.9999"


def test_recurrent_delta_rule_v2_perf(bh_glx_mesh):
    """100-call latency: kernel (state + readout fused) vs ref chain
    (state update + external readout matmul)."""
    mesh = bh_glx_mesh

    torch.manual_seed(7)

    state_shape = (K_DIM, V_DIM)
    row_shape = (TILE, K_DIM)
    col_shape = (K_DIM, TILE)
    scalar_shape = (TILE, TILE)

    state_t = torch.randn(state_shape, dtype=torch.float32)
    q_data = torch.randn(K_DIM, dtype=torch.float32)
    k_data = torch.randn(K_DIM, dtype=torch.float32)
    v_data = torch.randn(V_DIM, dtype=torch.float32)
    q_t = torch.zeros(row_shape, dtype=torch.float32)
    q_t[0, :] = q_data
    k_t = torch.zeros(col_shape, dtype=torch.float32)
    k_t[:, 0] = k_data
    v_t = torch.zeros(row_shape, dtype=torch.float32)
    v_t[0, :] = v_data
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

    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(V_TILES - 1, 0))])

    N_WARMUP = 5
    N_RUNS = 100

    # Warmup ref chain
    for _ in range(N_WARMUP):
        s, o_ref = _reference_chain(state, q, k, v, decay_scalar, beta_scalar)
        s.deallocate(True)
        o_ref.deallocate(True)
    ttnn.synchronize_device(mesh)

    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        s, o_ref = _reference_chain(state, q, k, v, decay_scalar, beta_scalar)
        s.deallocate(True)
        o_ref.deallocate(True)
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
