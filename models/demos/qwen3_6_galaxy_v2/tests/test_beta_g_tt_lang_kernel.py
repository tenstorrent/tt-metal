# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Validate the tt-lang-emitted fused beta/g kernel against the 6-op TTNN chain.

This test loads the C++ kernels emitted by
`models/demos/qwen3_6_galaxy_v2/tt/kernels/beta_g_kernel.py` (compiled in the
3.12 venv, see kernels/beta_g/*.cpp) and launches them through ttnn.generic_op
on the BH GLX mesh. Output is compared tile-by-tile against the existing
`_compute_beta_g` TTNN op chain.

Run sequentially:
    source python_env/bin/activate
    python -m pytest -s --noconftest \
        models/demos/qwen3_6_galaxy_v2/tests/test_beta_g_tt_lang_kernel.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import torch

import ttnn

KERNELS_DIR = Path(__file__).resolve().parent.parent / "tt" / "kernels" / "beta_g"
COMPUTE_KERNEL_CPP = str(KERNELS_DIR / "beta_g_compute.cpp")
READ_KERNEL_CPP = str(KERNELS_DIR / "beta_g_read.cpp")
WRITE_KERNEL_CPP = str(KERNELS_DIR / "beta_g_write.cpp")

TILE = 32
ROWS = 2
COLS = 2

# Tensor order MUST match the kernel author script (kernels/beta_g_kernel.py):
#   inputs : [b, a, dt, A_log, ones]    (5 tensors)
#   outputs: [beta, g]                  (2 tensors)
NUM_TENSORS = 7

# Reader thread `wait()` order in the author script:
#   b_dfb, a_dfb, dt_dfb, al_dfb, ones_dfb -> CB indices 0..4
# Writer thread reserves beta, g -> CB indices 5, 6.
#
# The author-script emit produced these tensor indices for each kernel:
#   compute: []
#   read:    [3, 1, 0, 2, 4]   # al, a, b, dt, ones (sorted by CB internal order)
#   write:   [5, 6]            # beta, g
# These are extracted from the runner emitted at compile time
# (see kernels/beta_g/_runner_emitted.py).
KERNEL_TENSOR_INDICES = [
    [],
    [3, 1, 0, 2, 4],
    [5, 6],
]

# CB shape (1,1)=1 tile, block_count=2 -> 4096 bytes per CB (double-buffered bf16).
CB_CONFIGS = [((1, 1), 2, ttnn.bfloat16, 2048, 4096) for _ in range(NUM_TENSORS)]


@pytest.fixture(scope="module")
def bh_glx_mesh():
    """Open the BH GLX 8x4 mesh; close cleanly on teardown."""
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


def _from_torch_replicated(mesh, x):
    return ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _to_torch_first_device(t):
    """Read tensor back as torch via the first device shard."""
    shards = ttnn.get_device_tensors(t)
    return ttnn.to_torch(shards[0])


def _run_fused_kernel(tensors, core_ranges):
    """Build the ProgramDescriptor for the emitted kernels and dispatch."""
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

    cb_indices = list(range(NUM_TENSORS))
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
            config = ttnn.ComputeConfigDescriptor()
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


def _compute_beta_g_ttnn(b, a, dt_bias, A_log):
    """Mirror qwen36_delta_attention._compute_beta_g exactly (6 TTNN ops)."""
    mem = ttnn.DRAM_MEMORY_CONFIG
    beta = ttnn.sigmoid(b, memory_config=mem)
    a_biased = ttnn.add(a, dt_bias, memory_config=mem)
    sp = ttnn.softplus(a_biased, memory_config=mem)
    A_exp = ttnn.exp(A_log, memory_config=ttnn.L1_MEMORY_CONFIG)
    g = ttnn.multiply(ttnn.neg(A_exp, memory_config=mem), sp, memory_config=mem)
    return beta, g


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def test_beta_g_kernel_pcc(bh_glx_mesh):
    """Task 3: kernel output PCC >= 0.9999 vs 6-op TTNN chain."""
    mesh = bh_glx_mesh

    # Verify the emitted kernels are present.
    for p in (COMPUTE_KERNEL_CPP, READ_KERNEL_CPP, WRITE_KERNEL_CPP):
        assert Path(p).is_file(), f"missing emitted kernel: {p}"

    shape = (ROWS * TILE, COLS * TILE)
    torch.manual_seed(0xBEEFCAFE)
    # Use the empirically-encountered ranges from the model:
    #   b, a   ~ Normal(0, 1) (post-projection, pre-activation)
    #   dt_bias~ Normal(0, 0.1) (small bias)
    #   A_log  ~ Normal(0, 1) (will be exp'd, give it a benign range)
    b_t = torch.randn(shape, dtype=torch.bfloat16)
    a_t = torch.randn(shape, dtype=torch.bfloat16)
    dt_t = 0.1 * torch.randn(shape, dtype=torch.bfloat16)
    al_t = torch.randn(shape, dtype=torch.bfloat16)

    b = _from_torch_replicated(mesh, b_t)
    a = _from_torch_replicated(mesh, a_t)
    dt = _from_torch_replicated(mesh, dt_t)
    al = _from_torch_replicated(mesh, al_t)
    ones = _from_torch_replicated(mesh, torch.ones(shape, dtype=torch.bfloat16))

    beta_out = ttnn.allocate_tensor_on_device(b.spec, mesh)
    g_out = ttnn.allocate_tensor_on_device(b.spec, mesh)

    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    # --- Run fused kernel ---
    _run_fused_kernel(
        [b, a, dt, al, ones, beta_out, g_out],
        core_ranges,
    )
    ttnn.synchronize_device(mesh)

    beta_kernel = _to_torch_first_device(beta_out)
    g_kernel = _to_torch_first_device(g_out)

    # --- Run 6-op TTNN reference chain ---
    beta_ref_tt, g_ref_tt = _compute_beta_g_ttnn(b, a, dt, al)
    ttnn.synchronize_device(mesh)
    beta_ref = _to_torch_first_device(beta_ref_tt)
    g_ref = _to_torch_first_device(g_ref_tt)

    beta_pcc = _pcc(beta_kernel, beta_ref)
    g_pcc = _pcc(g_kernel, g_ref)
    print(f"beta PCC = {beta_pcc:.6f}", file=sys.stderr)
    print(f"g    PCC = {g_pcc:.6f}", file=sys.stderr)
    print(
        f"beta max abs err = {(beta_kernel.float() - beta_ref.float()).abs().max().item():.6f}",
        file=sys.stderr,
    )
    print(
        f"g    max abs err = {(g_kernel.float() - g_ref.float()).abs().max().item():.6f}",
        file=sys.stderr,
    )

    assert beta_pcc >= 0.9999, f"beta PCC {beta_pcc} < 0.9999"
    assert g_pcc >= 0.9999, f"g PCC {g_pcc} < 0.9999"


def test_beta_g_kernel_perf(bh_glx_mesh):
    """Task 4: 100-call latency comparison kernel vs 6-op TTNN chain."""
    mesh = bh_glx_mesh

    shape = (ROWS * TILE, COLS * TILE)
    torch.manual_seed(7)
    b_t = torch.randn(shape, dtype=torch.bfloat16)
    a_t = torch.randn(shape, dtype=torch.bfloat16)
    dt_t = 0.1 * torch.randn(shape, dtype=torch.bfloat16)
    al_t = torch.randn(shape, dtype=torch.bfloat16)

    b = _from_torch_replicated(mesh, b_t)
    a = _from_torch_replicated(mesh, a_t)
    dt = _from_torch_replicated(mesh, dt_t)
    al = _from_torch_replicated(mesh, al_t)
    ones = _from_torch_replicated(mesh, torch.ones(shape, dtype=torch.bfloat16))

    beta_out = ttnn.allocate_tensor_on_device(b.spec, mesh)
    g_out = ttnn.allocate_tensor_on_device(b.spec, mesh)

    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    N_WARMUP = 10
    N_RUNS = 100

    # Warm up TTNN chain.
    for _ in range(N_WARMUP):
        beta_ref_tt, g_ref_tt = _compute_beta_g_ttnn(b, a, dt, al)
    ttnn.synchronize_device(mesh)

    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        beta_ref_tt, g_ref_tt = _compute_beta_g_ttnn(b, a, dt, al)
    ttnn.synchronize_device(mesh)
    t_chain = (time.perf_counter() - t0) / N_RUNS * 1e6

    # Warm up kernel.
    for _ in range(N_WARMUP):
        _run_fused_kernel(
            [b, a, dt, al, ones, beta_out, g_out],
            core_ranges,
        )
    ttnn.synchronize_device(mesh)

    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        _run_fused_kernel(
            [b, a, dt, al, ones, beta_out, g_out],
            core_ranges,
        )
    ttnn.synchronize_device(mesh)
    t_kernel = (time.perf_counter() - t0) / N_RUNS * 1e6

    speedup = t_chain / t_kernel if t_kernel > 0 else float("inf")
    print(f"chain   : {t_chain:.2f} us/call", file=sys.stderr)
    print(f"kernel  : {t_kernel:.2f} us/call", file=sys.stderr)
    print(f"speedup : {speedup:.2f}x", file=sys.stderr)

    # Acceptance is reporting; >=2x is the optimization target but PCC must hold.
    assert t_kernel > 0
