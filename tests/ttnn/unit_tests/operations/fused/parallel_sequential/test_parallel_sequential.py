# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for Sequential/Parallel kernel fusion.

~30 test methods covering all orthogonal dimensions:
  - Infrastructure (CB extraction, source structure, named args)
  - Sequential execution (chain lengths, core positions, grids, repetition)
  - Sharded execution (block/width, 3-phase, bias+residual)
  - Matmul fusion (all orderings, multicore, N-RMS tail, fp32 mismatch)
  - Branching topology (2/3-way split, nested, symmetric, asymmetric, slice)
  - Parallel execution (independent chains, matmul+chain, full grid, disjoint trees)
  - Sequential/Parallel API (inline, add, branching)
  - Cross-op compilation (10 kernel pairs)
"""

import functools
import os
import re

import pytest
import torch
import ttnn


from models.common.utility_functions import comp_pcc, skip_with_llk_assert, skip_with_watcher


def stress_test_program_cache(fn):
    """Decorator (#41622): run test 5x to exercise both program cache paths.

    Run 1:   normal (program cache miss — builds and caches the program).
    Runs 2-3: program cache enabled (hit — C++ patches the cached Program via
              override_runtime_arguments with fresh tensor addresses).
    Runs 4-5: program cache cleared then re-enabled (miss — Program{descriptor}
              is reconstructed from the cached descriptor with stale addresses,
              exercising patch_stale_descriptor).

    The fusion build cache persists across all runs (same device), so runs 2-5
    hit it with stale CBDescriptor.buffer pointers and runtime arg addresses.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Run 1: normal (program cache miss — builds and caches)
        fn(*args, **kwargs)

        device = kwargs.get("device")
        if device is None:
            for arg in args:
                if hasattr(arg, "disable_and_clear_program_cache"):
                    device = arg
                    break
        if device is None:
            return

        # Runs 2-3: program cache hits (C++ patches cached Program)
        for _ in range(2):
            fn(*args, **kwargs)

        # Runs 4-5: program cache misses (Program rebuilt from stale descriptor)
        device.disable_and_clear_program_cache()
        try:
            for _ in range(2):
                fn(*args, **kwargs)
        finally:
            device.enable_program_cache()

    return wrapper


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def torch_layer_norm(x, weight, bias=None, eps=1e-5):
    """Reference LayerNorm."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    out = (x - mean) / torch.sqrt(var + eps)
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out


def torch_rms_norm(x, weight, eps=1e-5):
    """Reference RMSNorm."""
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = x / rms
    if weight is not None:
        out = out * weight
    return out


# ---------------------------------------------------------------------------
# Compact helpers
# ---------------------------------------------------------------------------


def cores(x1, y1, x2=None, y2=None):
    """Shorthand for CoreRangeSet. cores(0,0) = single core, cores(0,0,3,0) = range."""
    if x2 is None:
        x2, y2 = x1, y1
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x1, y1), ttnn.CoreCoord(x2, y2))})


def check_pcc(golden, tt_tensor, pcc=0.98, label=""):
    """to_torch + comp_pcc + assert."""
    result = ttnn.to_torch(tt_tensor)
    passing, actual_pcc = comp_pcc(golden, result, pcc=pcc)
    assert passing, f"{label} PCC: {actual_pcc}"


def tt(t, device):
    """Move torch tensor to device."""
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


# ---------------------------------------------------------------------------
# Tensor bundle helpers (use with the ``device`` fixture from tt-metal conftest)
# ---------------------------------------------------------------------------


def make_small_norm_tensors(device):
    """Small tensors (1,1,32,128) for single/few-core norm tests."""
    torch.manual_seed(42)
    hidden = 128
    inp = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
    ws = [torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16) for _ in range(4)]
    bs = [torch.zeros(1, 1, 1, hidden, dtype=torch.bfloat16) for _ in range(2)]

    return {
        "torch_input": inp,
        "torch_weights": ws,
        "torch_biases": bs,
        "tt_input": tt(inp, device),
        "tt_weights": [tt(w, device) for w in ws],
        "tt_biases": [tt(b, device) for b in bs],
    }


def make_multi_norm_tensors(device):
    """Larger tensors (1,1,256,128) for multi-core and branching tests (NCHt=8)."""
    torch.manual_seed(42)
    hidden = 128
    inp = torch.randn(1, 1, 256, hidden, dtype=torch.bfloat16)
    ws = [torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16) for _ in range(8)]
    bs = [torch.zeros(1, 1, 1, hidden, dtype=torch.bfloat16) for _ in range(2)]

    return {
        "torch_input": inp,
        "torch_weights": ws,
        "torch_biases": bs,
        "tt_input": tt(inp, device),
        "tt_weights": [tt(w, device) for w in ws],
        "tt_biases": [tt(b, device) for b in bs],
    }


# ===========================================================================
# TestInfrastructure
# ===========================================================================


class TestInfrastructure:
    """Build-time infrastructure tests (CB extraction, source structure, named args)."""

    @stress_test_program_cache
    def test_cb_extraction(self, device):
        """extract_cb_info on LN descriptor, check c_0/c_16."""
        from models.experimental.ops.descriptors.fusion import extract_cb_info
        from models.experimental.ops.descriptors.normalization import layer_norm

        t = make_small_norm_tensors(device)
        desc = layer_norm.layer_norm(
            t["tt_input"], core_range_set=cores(0, 0), weight=t["tt_weights"][0], bias=t["tt_biases"][0], epsilon=1e-5
        )
        cb_info = extract_cb_info(desc.descriptor)
        assert len(cb_info) > 0
        assert 0 in cb_info, "Should have input CB (c_0)"
        assert 16 in cb_info, "Should have output CB (c_16)"

    @stress_test_program_cache
    def test_fused_source_structure(self, device):
        """2-phase LN->LN: verify phase namespaces and barrier code in fused source."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm

        t = make_small_norm_tensors(device)
        cr = cores(0, 0)
        ln1 = layer_norm.layer_norm(t["tt_input"], core_range_set=cr, weight=t["tt_weights"][0], epsilon=1e-5)
        ln2 = layer_norm.layer_norm(ln1.output_tensors[0], core_range_set=cr, weight=t["tt_weights"][1], epsilon=1e-5)

        fused = Sequential(ln1, ln2).build(device)
        # Check source of each kernel for phase namespaces and barrier
        for kernel in fused.descriptor.kernels:
            src = kernel.kernel_source
            if "namespace phase_0" in src and "namespace phase_1" in src:
                assert "barrier::" in src or "phase_0::run" in src
                return
        # At least one kernel should have both phases
        assert False, "No kernel contained both phase_0 and phase_1 namespaces"

    @stress_test_program_cache
    def test_named_args_phase_prefix(self, device):
        """2-phase fused: verify phase_1_ prefix on named CT args."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm

        t = make_small_norm_tensors(device)
        cr = cores(0, 0)
        ln_compute = ttnn.layernorm_default_compute_config(device.arch())

        ln = layer_norm.layer_norm(t["tt_input"], core_range_set=cr, weight=t["tt_weights"][0], epsilon=1e-5)
        rms = rms_norm.rms_norm(
            ln.output_tensors[0],
            core_range_set=cr,
            weight=t["tt_weights"][1],
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
        )

        fused = Sequential(ln, rms).build(device)
        # Check that at least one kernel has phase_1_ prefixed named args
        for kernel in fused.descriptor.kernels:
            for name, _ in kernel.named_compile_time_args:
                if name.startswith("phase_1_"):
                    return
        assert False, "No phase_1_ prefixed named compile-time args found"


# ===========================================================================
# TestSequentialExecution
# ===========================================================================


class TestSequentialExecution:
    """Core sequential chain execution tests."""

    @pytest.mark.parametrize("num_phases", [2, 3, 4])
    @stress_test_program_cache
    def test_norm_chain(self, device, num_phases):
        """Mixed LN/RMS chain of varying length on single core."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm

        t = make_small_norm_tensors(device)
        cr = cores(0, 0)
        ln_compute = ttnn.layernorm_default_compute_config(device.arch())

        ops, golden = [], t["torch_input"].float()
        prev = t["tt_input"]
        for i in range(num_phases):
            w_idx = i % len(t["tt_weights"])
            if i % 2 == 0:
                op = layer_norm.layer_norm(prev, core_range_set=cr, weight=t["tt_weights"][w_idx], epsilon=1e-5)
                golden = torch_layer_norm(golden, t["torch_weights"][w_idx].float())
            else:
                op = rms_norm.rms_norm(
                    prev,
                    core_range_set=cr,
                    weight=t["tt_weights"][w_idx],
                    epsilon=1e-5,
                    compute_kernel_config=ln_compute,
                )
                golden = torch_rms_norm(golden, t["torch_weights"][w_idx].float())
            ops.append(op)
            prev = op.output_tensors[0]

        fused = Sequential(*ops)
        [out] = fused.run(results=[ops[-1]])
        check_pcc(golden, out, pcc=0.97, label=f"{num_phases}-phase chain")

    @pytest.mark.parametrize("core_x", [3, 5, 7])
    @stress_test_program_cache
    def test_chain_on_nonzero_core(self, device, core_x):
        """2-phase LN->RMS on non-origin single core."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm

        t = make_small_norm_tensors(device)
        cr = cores(core_x, 0)
        ln_compute = ttnn.layernorm_default_compute_config(device.arch())

        ln = layer_norm.layer_norm(t["tt_input"], core_range_set=cr, weight=t["tt_weights"][0], epsilon=1e-5)
        rms = rms_norm.rms_norm(
            ln.output_tensors[0],
            core_range_set=cr,
            weight=t["tt_weights"][1],
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
        )

        fused = Sequential(ln, rms)
        [out] = fused.run(results=[rms])

        golden = torch_rms_norm(
            torch_layer_norm(t["torch_input"].float(), t["torch_weights"][0].float()),
            t["torch_weights"][1].float(),
        )
        check_pcc(golden, out, label=f"core ({core_x},0)")

    @pytest.mark.parametrize(
        "grid",
        [
            pytest.param((3, 0, 0, 0), id="1D_4core"),
            pytest.param((1, 1, 0, 0), id="2D_2x2"),
        ],
    )
    @stress_test_program_cache
    def test_multicore_chain(self, device, grid):
        """2-phase LN->RMS on multi-core grid (row or 2D)."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm

        x2, y2, x1, y1 = grid
        cr = cores(x1, y1, x2, y2)
        ln_compute = ttnn.layernorm_default_compute_config(device.arch())
        num_cores = (x2 - x1 + 1) * (y2 - y1 + 1)

        torch.manual_seed(42)
        hidden = 128
        torch_input = torch.randn(1, 1, 32 * num_cores, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)

        ln = layer_norm.layer_norm(tt(torch_input, device), core_range_set=cr, weight=tt(torch_w, device), epsilon=1e-5)
        rms = rms_norm.rms_norm(
            ln.output_tensors[0],
            core_range_set=cr,
            weight=tt(torch_w, device),
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
        )

        fused = Sequential(ln, rms)
        [out] = fused.run(results=[rms])

        golden = torch_rms_norm(torch_layer_norm(torch_input.float(), torch_w.float()), torch_w.float())
        check_pcc(golden, out, label="multicore chain")

    @stress_test_program_cache
    def test_repeated_execution(self, device):
        """Same fused op launched 3x — verify no stale state."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = make_small_norm_tensors(device)
        cr = cores(0, 0)
        rms1 = rms_norm.rms_norm(t["tt_input"], core_range_set=cr, weight=t["tt_weights"][0], epsilon=1e-5)
        rms2 = rms_norm.rms_norm(rms1.output_tensors[0], core_range_set=cr, weight=t["tt_weights"][1], epsilon=1e-5)

        fused = Sequential(rms1, rms2)

        golden = torch_rms_norm(
            torch_rms_norm(t["torch_input"].float(), t["torch_weights"][0].float()),
            t["torch_weights"][1].float(),
        )

        for i in range(3):
            [out] = fused.run()
            check_pcc(golden, out, label=f"run {i}")

    @stress_test_program_cache
    def test_single_op_passthrough(self, device):
        """Single op in Sequential — verify pass-through."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = make_small_norm_tensors(device)
        rms = rms_norm.rms_norm(t["tt_input"], core_range_set=cores(0, 0), weight=t["tt_weights"][0], epsilon=1e-5)

        fused = Sequential(rms)
        [out] = fused.run(results=[rms])

        golden = torch_rms_norm(t["torch_input"].float(), t["torch_weights"][0].float())
        check_pcc(golden, out, pcc=0.99, label="single op")


# ===========================================================================
# TestShardedExecution
# ===========================================================================


class TestShardedExecution:
    """Sharded (L1) execution tests."""

    @pytest.mark.parametrize(
        "shard_type",
        [
            pytest.param("block", id="block"),
            pytest.param("width", id="width"),
        ],
    )
    @stress_test_program_cache
    def test_sharded_chain(self, device, shard_type):
        """2-phase LN->RMS with block or width sharding."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from tests.ttnn.unit_tests.operations.fused.sharded_test_utils import (
            create_sharded_mem_config,
            torch_layer_norm as sh_ln_golden,
            rms_norm_golden,
        )

        torch.manual_seed(12345)
        two_stage = shard_type == "width"
        if two_stage:
            h, w, nch, ncw, bht, bwt = 256, 512, 2, 4, 8, 2
        else:
            h, w, nch, ncw, bht, bwt = 256, 320, 2, 5, 4, 2

        torch_input = torch.randn((h, w), dtype=torch.bfloat16)
        torch_w = torch.ones((w,), dtype=torch.bfloat16)

        sharded_mem = create_sharded_mem_config(h, w, nch, ncw, two_stage)
        tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded_mem)
        tt_w = ttnn.from_torch(torch_w, layout=ttnn.TILE_LAYOUT, device=device)

        cc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
        )
        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            block_h=bht,
            block_w=bwt,
            subblock_w=1,
            use_welford=False,
            inplace=False,
        )

        ln = layer_norm.layer_norm(tt_input, weight=tt_w, epsilon=1e-5, compute_kernel_config=cc, program_config=pc)
        rms = rms_norm.rms_norm(
            ln.output_tensors[0], weight=tt_w, epsilon=1e-5, compute_kernel_config=cc, program_config=pc
        )

        fused = Sequential(ln, rms)
        [out] = fused.run(results=[rms])

        golden = rms_norm_golden(sh_ln_golden(torch_input, weight=torch_w), torch_w)
        check_pcc(golden, out, label=f"sharded {shard_type}")

    @skip_with_watcher("Program too large for kernel config buffer. Will not fix.")
    @skip_with_llk_assert("Program too large for kernel config buffer. Will not fix.")
    @stress_test_program_cache
    def test_sharded_three_phase(self, device):
        """3-phase LN->RMS->LN block-sharded on 4x4 grid."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from tests.ttnn.unit_tests.operations.fused.sharded_test_utils import (
            create_sharded_mem_config,
            torch_layer_norm as sh_ln_golden,
            rms_norm_golden,
        )

        torch.manual_seed(12347)
        h, w = 128, 256

        torch_input = torch.randn((h, w), dtype=torch.bfloat16)
        torch_ws = [torch.ones((w,), dtype=torch.bfloat16) for _ in range(3)]

        sharded_mem = create_sharded_mem_config(h, w, 4, 4, False)
        tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded_mem)
        tt_ws = [ttnn.from_torch(w_, layout=ttnn.TILE_LAYOUT, device=device) for w_ in torch_ws]

        cc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
        )
        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            block_h=1,
            block_w=2,
            subblock_w=1,
            use_welford=False,
            inplace=False,
        )

        ln1 = layer_norm.layer_norm(
            tt_input, weight=tt_ws[0], epsilon=1e-5, compute_kernel_config=cc, program_config=pc
        )
        rms = rms_norm.rms_norm(
            ln1.output_tensors[0], weight=tt_ws[1], epsilon=1e-5, compute_kernel_config=cc, program_config=pc
        )
        ln2 = layer_norm.layer_norm(
            rms.output_tensors[0], weight=tt_ws[2], epsilon=1e-5, compute_kernel_config=cc, program_config=pc
        )

        fused = Sequential(ln1, rms, ln2)
        [out] = fused.run(results=[ln2])

        g = sh_ln_golden(torch_input, weight=torch_ws[0])
        g = rms_norm_golden(g, torch_ws[1])
        golden = sh_ln_golden(g, weight=torch_ws[2])
        check_pcc(golden, out, label="sharded 3-phase")

    @stress_test_program_cache
    def test_sharded_with_bias_residual(self, device):
        """LN(bias+residual)->RMS block-sharded, single-stage."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from tests.ttnn.unit_tests.operations.fused.sharded_test_utils import (
            create_sharded_mem_config,
            torch_layer_norm as sh_ln_golden,
            rms_norm_golden,
        )

        torch.manual_seed(12349)
        h, w = 256, 320

        torch_input = torch.randn((h, w), dtype=torch.bfloat16)
        torch_residual = torch.randn((h, w), dtype=torch.bfloat16)
        torch_w = torch.ones((w,), dtype=torch.bfloat16)
        torch_bias = torch.randn((w,), dtype=torch.bfloat16) * 0.1

        sharded_mem = create_sharded_mem_config(h, w, 2, 5, False)
        tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded_mem)
        tt_res = ttnn.from_torch(torch_residual, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded_mem)
        tt_w = ttnn.from_torch(torch_w, layout=ttnn.TILE_LAYOUT, device=device)
        tt_bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

        cc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
        )
        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            block_h=4,
            block_w=2,
            subblock_w=1,
            use_welford=False,
            inplace=False,
        )

        ln = layer_norm.layer_norm(
            tt_input,
            weight=tt_w,
            bias=tt_bias,
            residual_input_tensor=tt_res,
            epsilon=1e-5,
            compute_kernel_config=cc,
            program_config=pc,
        )
        rms = rms_norm.rms_norm(
            ln.output_tensors[0], weight=tt_w, epsilon=1e-5, compute_kernel_config=cc, program_config=pc
        )

        fused = Sequential(ln, rms)
        [out] = fused.run(results=[rms])

        golden = rms_norm_golden(
            sh_ln_golden(torch_input, residual=torch_residual, weight=torch_w, bias=torch_bias), torch_w
        )
        check_pcc(golden, out, label="sharded bias+residual")


# ===========================================================================
# TestMatmulFusion
# ===========================================================================


class TestMatmulFusion:
    """Matmul fusion tests — orderings, multicore, N-RMS tail, fp32 mismatch."""

    def _mm_config(self, gx=1, gy=1, in0_block_w=4, per_core_M=1, per_core_N=4):
        return ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=min(per_core_N, 4),
            per_core_M=per_core_M,
            per_core_N=per_core_N,
        )

    def _compute(self, fp32=False, approx=True):
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=approx,
            fp32_dest_acc_en=fp32,
        )

    @stress_test_program_cache
    def test_matmul_standalone(self, device):
        """Single matmul via descriptor API."""
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc

        torch.manual_seed(42)
        torch_a = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, 64, 128, dtype=torch.bfloat16)

        mm = matmul_desc(tt(torch_a, device), tt(torch_b, device))
        mm.launch()
        check_pcc(torch_a @ torch_b, mm.output_tensors[0], pcc=0.99, label="matmul standalone")

    @pytest.mark.parametrize(
        "ordering",
        [
            pytest.param("mm_rms", id="mm_rms"),
            pytest.param("rms_mm", id="rms_mm"),
            pytest.param("rms_mm_rms", id="rms_mm_rms"),
            pytest.param("mm_ln", id="mm_ln"),
            pytest.param("ln_mm", id="ln_mm"),
        ],
    )
    @stress_test_program_cache
    def test_matmul_norm_chain(self, device, ordering):
        """All matmul+norm orderings on single core."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc

        torch.manual_seed(42)
        hidden = 128
        cr = cores(0, 0)
        ln_compute = ttnn.layernorm_default_compute_config(device.arch())

        torch_a = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_bias = torch.zeros(1, 1, 1, hidden, dtype=torch.bfloat16)

        def mk_mm(inp):
            return matmul_desc(
                inp,
                tt(torch_b, device),
                core_range_set=cr,
                program_config=self._mm_config(),
                compute_kernel_config=self._compute(fp32="ln" in ordering, approx="ln" not in ordering),
            )

        def mk_rms(inp):
            return rms_norm.rms_norm(
                inp,
                core_range_set=cr,
                weight=tt(torch_w, device),
                epsilon=1e-5,
                compute_kernel_config=self._compute(fp32="ln" in ordering, approx="ln" not in ordering),
            )

        def mk_ln(inp):
            return layer_norm.layer_norm(
                inp,
                core_range_set=cr,
                weight=tt(torch_w, device),
                bias=tt(torch_bias, device),
                epsilon=1e-5,
            )

        # Build chain
        ops = []
        prev = tt(torch_a, device)
        for token in ordering.split("_"):
            if token == "mm":
                op = mk_mm(prev)
            elif token == "rms":
                op = mk_rms(prev)
            elif token == "ln":
                op = mk_ln(prev)
            ops.append(op)
            prev = op.output_tensors[0]

        fused = Sequential(*ops)
        [out] = fused.run(results=[ops[-1]])

        # Golden
        g = torch_a.float()
        for token in ordering.split("_"):
            if token == "mm":
                g = g @ torch_b.float()
            elif token == "rms":
                g = torch_rms_norm(g, torch_w.float())
            elif token == "ln":
                g = torch_layer_norm(g, torch_w.float(), torch_bias.float())

        check_pcc(g, out, pcc=0.97, label=ordering)

    @stress_test_program_cache
    def test_multicore_matmul_chain(self, device):
        """RMS->MM->RMS on 4x2 grid (8 cores)."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc

        torch.manual_seed(42)
        hidden = 128
        cr = cores(0, 0, 3, 1)

        torch_input = torch.randn(1, 1, 256, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)

        rms1 = rms_norm.rms_norm(tt(torch_input, device), core_range_set=cr, weight=tt(torch_w, device), epsilon=1e-5)
        mm = matmul_desc(
            rms1.output_tensors[0],
            tt(torch_b, device),
            core_range_set=cr,
            program_config=self._mm_config(gx=4, gy=2),
            compute_kernel_config=self._compute(),
        )
        rms2 = rms_norm.rms_norm(mm.output_tensors[0], core_range_set=cr, weight=tt(torch_w, device), epsilon=1e-5)

        fused = Sequential(rms1, mm, rms2)
        [out] = fused.run(results=[rms2])

        golden = torch_rms_norm(torch_rms_norm(torch_input.float(), torch_w.float()) @ torch_b.float(), torch_w.float())
        check_pcc(golden, out, label="multicore RMS->MM->RMS")

    @skip_with_watcher("Program too large for kernel config buffer. Will not fix.")
    @pytest.mark.parametrize("num_rms", [2, 3, 4])
    @stress_test_program_cache
    def test_matmul_followed_by_n_rms(self, device, num_rms):
        """MM then N consecutive RMS norms."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc

        torch.manual_seed(42)
        hidden = 128
        cr = cores(0, 0)

        torch_a = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)

        mm = matmul_desc(
            tt(torch_a, device),
            tt(torch_b, device),
            core_range_set=cr,
            program_config=self._mm_config(),
            compute_kernel_config=self._compute(),
        )
        ops = [mm]
        prev = mm.output_tensors[0]
        for _ in range(num_rms):
            rms = rms_norm.rms_norm(prev, core_range_set=cr, weight=tt(torch_w, device), epsilon=1e-5)
            ops.append(rms)
            prev = rms.output_tensors[0]

        fused = Sequential(*ops)
        [out] = fused.run(results=[ops[-1]])

        g = torch_a.float() @ torch_b.float()
        for _ in range(num_rms):
            g = torch_rms_norm(g, torch_w.float())
        check_pcc(g, out, pcc=0.97, label=f"MM->{num_rms}xRMS")

    @stress_test_program_cache
    def test_fp32_mismatch_error(self, device):
        """MM(fp32=off) + LN(fp32=on) should raise."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc

        torch.manual_seed(42)
        hidden = 128
        cr = cores(0, 0)

        torch_a = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)

        mm = matmul_desc(
            tt(torch_a, device),
            tt(torch_b, device),
            core_range_set=cr,
            program_config=self._mm_config(),
            compute_kernel_config=self._compute(fp32=False),
        )
        ln = layer_norm.layer_norm(
            mm.output_tensors[0],
            core_range_set=cr,
            weight=tt(torch_w, device),
            epsilon=1e-5,
        )

        with pytest.raises((ValueError, RuntimeError)):
            Sequential(mm, ln).build(device)


# ===========================================================================
# TestBranchingTopology
# ===========================================================================


class TestBranchingTopology:
    """Branching (tree) topology tests, including slice ops."""

    @stress_test_program_cache
    def test_two_branch_split(self, device):
        """Stem(8c) -> 2 branches(4c each)."""
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = make_multi_norm_tensors(device)
        stem = rms_norm.rms_norm(
            t["tt_input"], core_range_set=cores(0, 0, 7, 0), weight=t["tt_weights"][0], epsilon=1e-5
        )
        a = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(0, 0, 3, 0), weight=t["tt_weights"][1], epsilon=1e-5
        )
        b = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(4, 0, 7, 0), weight=t["tt_weights"][2], epsilon=1e-5
        )

        fused = Sequential(stem, Parallel(a, b))
        [out_a, out_b] = fused.run(results=[a, b])

        g_stem = torch_rms_norm(t["torch_input"].float(), t["torch_weights"][0].float())
        check_pcc(torch_rms_norm(g_stem, t["torch_weights"][1].float()), out_a, label="branch A")
        check_pcc(torch_rms_norm(g_stem, t["torch_weights"][2].float()), out_b, label="branch B")

    @stress_test_program_cache
    def test_three_way_split_with_slice(self, device):
        """Stem RMS(6c) -> 3 branches(2c each), using slice in one branch.

        Tree:
            RMS [0-5] -> RMS [0-1]
                      -> slice [2-3]
                      -> RMS [4-5]
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors.data_movement.slice import slice as slice_desc

        torch.manual_seed(42)
        hidden = 128
        torch_input = torch.randn(1, 1, 192, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)

        stem = rms_norm.rms_norm(
            tt(torch_input, device), core_range_set=cores(0, 0, 5, 0), weight=tt(torch_w, device), epsilon=1e-5
        )
        branch_a = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(0, 0, 1, 0), weight=tt(torch_w, device), epsilon=1e-5
        )
        branch_b = slice_desc(stem.output_tensors[0], [0, 0, 0, 0], [1, 1, 192, 64], core_range_set=cores(2, 0, 3, 0))
        branch_c = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(4, 0, 5, 0), weight=tt(torch_w, device), epsilon=1e-5
        )

        fused = Sequential(stem, Parallel(branch_a, branch_b, branch_c))
        [out_a, out_b, out_c] = fused.run()

        g_stem = torch_rms_norm(torch_input.float(), torch_w.float())
        check_pcc(torch_rms_norm(g_stem, torch_w.float()), out_a, pcc=0.97, label="A (RMS)")
        check_pcc(g_stem[:, :, :, :64], out_b, pcc=0.97, label="B (slice)")
        check_pcc(torch_rms_norm(g_stem, torch_w.float()), out_c, pcc=0.97, label="C (RMS)")

    @stress_test_program_cache
    def test_nested_split_with_slice(self, device):
        """Stem -> Parallel(Sequential(A, Parallel(A1, A2_slice)), B).

        Tree (8 cores):
            RMS [0-7] -> RMS [0-3] -> RMS [0-1]
                                    -> slice [2-3]
                      -> RMS [4-7]
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors.data_movement.slice import slice as slice_desc

        torch.manual_seed(42)
        hidden = 128
        torch_input = torch.randn(1, 1, 256, hidden, dtype=torch.bfloat16)
        ws = [torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16) for _ in range(4)]
        tt_ws = [tt(w, device) for w in ws]

        stem = rms_norm.rms_norm(
            tt(torch_input, device), core_range_set=cores(0, 0, 7, 0), weight=tt_ws[0], epsilon=1e-5
        )
        a_mid = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(0, 0, 3, 0), weight=tt_ws[1], epsilon=1e-5
        )
        a1 = rms_norm.rms_norm(a_mid.output_tensors[0], core_range_set=cores(0, 0, 1, 0), weight=tt_ws[2], epsilon=1e-5)
        a2_slice = slice_desc(a_mid.output_tensors[0], [0, 0, 0, 0], [1, 1, 256, 64], core_range_set=cores(2, 0, 3, 0))
        b = rms_norm.rms_norm(stem.output_tensors[0], core_range_set=cores(4, 0, 7, 0), weight=tt_ws[3], epsilon=1e-5)

        fused = Sequential(
            stem,
            Parallel(
                Sequential(a_mid, Parallel(a1, a2_slice)),
                b,
            ),
        )
        [out_a1, out_a2, out_b] = fused.run()

        g_stem = torch_rms_norm(torch_input.float(), ws[0].float())
        g_a = torch_rms_norm(g_stem, ws[1].float())
        check_pcc(torch_rms_norm(g_a, ws[2].float()), out_a1, pcc=0.97, label="A1 (RMS)")
        check_pcc(g_a[:, :, :, :64], out_a2, pcc=0.97, label="A2 (slice)")
        check_pcc(torch_rms_norm(g_stem, ws[3].float()), out_b, pcc=0.97, label="B (RMS)")

    @stress_test_program_cache
    def test_symmetric_binary_tree(self, device):
        """Stem -> 2 mid -> 4 leaves.

        Tree (8 cores):
            RMS [0-7] -> RMS [0-3] -> RMS [0-1] / RMS [2-3]
                      -> RMS [4-7] -> RMS [4-5] / RMS [6-7]
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = make_multi_norm_tensors(device)
        wt = t["tt_weights"]

        def rms(inp, cr, wi):
            return rms_norm.rms_norm(inp, core_range_set=cr, weight=wt[wi], epsilon=1e-5)

        root = rms(t["tt_input"], cores(0, 0, 7, 0), 0)
        left = rms(root.output_tensors[0], cores(0, 0, 3, 0), 1)
        right = rms(root.output_tensors[0], cores(4, 0, 7, 0), 2)
        ll = rms(left.output_tensors[0], cores(0, 0, 1, 0), 3)
        lr = rms(left.output_tensors[0], cores(2, 0, 3, 0), 4)
        rl = rms(right.output_tensors[0], cores(4, 0, 5, 0), 5)
        rr = rms(right.output_tensors[0], cores(6, 0, 7, 0), 6)

        fused = Sequential(
            root,
            Parallel(
                Sequential(left, Parallel(ll, lr)),
                Sequential(right, Parallel(rl, rr)),
            ),
        )
        outs = fused.run()

        ws = t["torch_weights"]
        g_root = torch_rms_norm(t["torch_input"].float(), ws[0].float())
        g_left = torch_rms_norm(g_root, ws[1].float())
        g_right = torch_rms_norm(g_root, ws[2].float())
        goldens = [
            torch_rms_norm(g_left, ws[3].float()),
            torch_rms_norm(g_left, ws[4].float()),
            torch_rms_norm(g_right, ws[5].float()),
            torch_rms_norm(g_right, ws[6].float()),
        ]
        for i, label in enumerate(["LL", "LR", "RL", "RR"]):
            check_pcc(goldens[i], outs[i], label=label)

    @stress_test_program_cache
    def test_asymmetric_deep_left(self, device):
        """Deep left + shallow right.

        Tree (8 cores):
            RMS [0-7] -> RMS [0-3] -> RMS [0-1] -> RMS [0-1]
                                    -> RMS [2-3]
                      -> RMS [4-7]
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = make_multi_norm_tensors(device)
        wt = t["tt_weights"]

        def rms(inp, cr, wi):
            return rms_norm.rms_norm(inp, core_range_set=cr, weight=wt[wi], epsilon=1e-5)

        root = rms(t["tt_input"], cores(0, 0, 7, 0), 0)
        left = rms(root.output_tensors[0], cores(0, 0, 3, 0), 1)
        ll = rms(left.output_tensors[0], cores(0, 0, 1, 0), 2)
        ll_deep = rms(ll.output_tensors[0], cores(0, 0, 1, 0), 3)
        lr = rms(left.output_tensors[0], cores(2, 0, 3, 0), 4)
        right = rms(root.output_tensors[0], cores(4, 0, 7, 0), 5)

        fused = Sequential(
            root,
            Parallel(
                Sequential(left, Parallel(Sequential(ll, ll_deep), lr)),
                right,
            ),
        )
        outs = fused.run()

        ws = t["torch_weights"]
        g_root = torch_rms_norm(t["torch_input"].float(), ws[0].float())
        g_left = torch_rms_norm(g_root, ws[1].float())
        goldens = [
            torch_rms_norm(torch_rms_norm(g_left, ws[2].float()), ws[3].float()),
            torch_rms_norm(g_left, ws[4].float()),
            torch_rms_norm(g_root, ws[5].float()),
        ]
        for i, label in enumerate(["LL(deep)", "LR", "Right"]):
            check_pcc(goldens[i], outs[i], label=label)

    @stress_test_program_cache
    def test_overlapping_branches_error(self, device):
        """Overlapping branch core ranges should raise ValueError."""
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = make_multi_norm_tensors(device)
        stem = rms_norm.rms_norm(
            t["tt_input"], core_range_set=cores(0, 0, 3, 0), weight=t["tt_weights"][0], epsilon=1e-5
        )
        # Overlapping: (0,0)-(2,0) and (1,0)-(3,0)
        a = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(0, 0, 2, 0), weight=t["tt_weights"][1], epsilon=1e-5
        )
        b = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(1, 0, 3, 0), weight=t["tt_weights"][2], epsilon=1e-5
        )

        with pytest.raises(ValueError, match="overlapping"):
            Sequential(stem, Parallel(a, b)).build(device)


# ===========================================================================
# TestParallelExecution
# ===========================================================================


class TestParallelExecution:
    """Independent parallel execution tests."""

    @pytest.mark.parametrize("n_chains", [2, 4])
    @stress_test_program_cache
    def test_parallel_chains(self, device, n_chains):
        """N independent fused LN->RMS chains on separate single cores."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm

        t = make_small_norm_tensors(device)
        ln_compute = ttnn.layernorm_default_compute_config(device.arch())

        torch_inputs = [torch.randn_like(t["torch_input"]) for _ in range(n_chains)]
        chains = []
        rms_tails = []
        for i in range(n_chains):
            cr = cores(i, 0)
            ln = layer_norm.layer_norm(
                tt(torch_inputs[i], device), core_range_set=cr, weight=t["tt_weights"][0], epsilon=1e-5
            )
            rms = rms_norm.rms_norm(
                ln.output_tensors[0],
                core_range_set=cr,
                weight=t["tt_weights"][1],
                epsilon=1e-5,
                compute_kernel_config=ln_compute,
            )
            rms_tails.append(rms)
            chains.append(Sequential(ln, rms))

        chain_outs = []
        for ch in chains:
            chain_outs.append(ch.run())

        for i in range(n_chains):
            golden = torch_rms_norm(
                torch_layer_norm(torch_inputs[i].float(), t["torch_weights"][0].float()),
                t["torch_weights"][1].float(),
            )
            check_pcc(golden, chain_outs[i][0], label=f"chain {i}")

    @stress_test_program_cache
    def test_matmul_plus_fused_chain(self, device):
        """Matmul + 3-phase norm chain on disjoint cores."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc

        t = make_small_norm_tensors(device)
        ln_compute = ttnn.layernorm_default_compute_config(device.arch())
        torch.manual_seed(42)

        # Matmul on default core
        torch_a = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, 64, 128, dtype=torch.bfloat16)
        mm = matmul_desc(tt(torch_a, device), tt(torch_b, device))

        # 3-phase chain on core (4,0)
        cr = cores(4, 0)
        ln1 = layer_norm.layer_norm(
            t["tt_input"], core_range_set=cr, weight=t["tt_weights"][0], bias=t["tt_biases"][0], epsilon=1e-5
        )
        rms = rms_norm.rms_norm(
            ln1.output_tensors[0],
            core_range_set=cr,
            weight=t["tt_weights"][1],
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
        )
        ln2 = layer_norm.layer_norm(
            rms.output_tensors[0], core_range_set=cr, weight=t["tt_weights"][2], bias=t["tt_biases"][1], epsilon=1e-5
        )
        fused = Sequential(ln1, rms, ln2)

        mm.launch()
        [out_ln2] = fused.run()

        check_pcc(torch_a @ torch_b, mm.output_tensors[0], pcc=0.99, label="matmul")

        g = torch_layer_norm(t["torch_input"].float(), t["torch_weights"][0].float(), t["torch_biases"][0].float())
        g = torch_rms_norm(g, t["torch_weights"][1].float())
        golden = torch_layer_norm(g, t["torch_weights"][2].float(), t["torch_biases"][1].float())
        check_pcc(golden, out_ln2, label="3-phase chain")

    @stress_test_program_cache
    def test_two_disjoint_trees_parallel(self, device):
        """Two independent branching trees launched in parallel on disjoint cores.

        Tree A (cores 0-3):
            RMS [0-3] -> slice [0-1]
                      -> RMS [2-3]

        Tree B (cores 4-7):
            RMS [4-7] -> RMS [4-5]
                      -> slice [6-7]
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors.data_movement.slice import slice as slice_desc

        torch.manual_seed(42)
        hidden = 128
        torch_input_a = torch.randn(1, 1, 128, hidden, dtype=torch.bfloat16)
        torch_input_b = torch.randn(1, 1, 128, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)

        # Tree A
        stem_a = rms_norm.rms_norm(
            tt(torch_input_a, device), core_range_set=cores(0, 0, 3, 0), weight=tt(torch_w, device), epsilon=1e-5
        )
        a_left = slice_desc(stem_a.output_tensors[0], [0, 0, 0, 0], [1, 1, 128, 64], core_range_set=cores(0, 0, 1, 0))
        a_right = rms_norm.rms_norm(
            stem_a.output_tensors[0], core_range_set=cores(2, 0, 3, 0), weight=tt(torch_w, device), epsilon=1e-5
        )
        tree_a = Sequential(stem_a, Parallel(a_left, a_right))

        # Tree B
        stem_b = rms_norm.rms_norm(
            tt(torch_input_b, device), core_range_set=cores(4, 0, 7, 0), weight=tt(torch_w, device), epsilon=1e-5
        )
        b_left = rms_norm.rms_norm(
            stem_b.output_tensors[0], core_range_set=cores(4, 0, 5, 0), weight=tt(torch_w, device), epsilon=1e-5
        )
        b_right = slice_desc(stem_b.output_tensors[0], [0, 0, 0, 0], [1, 1, 128, 64], core_range_set=cores(6, 0, 7, 0))
        tree_b = Sequential(stem_b, Parallel(b_left, b_right))

        [out_a_left, out_a_right] = tree_a.run()
        [out_b_left, out_b_right] = tree_b.run()

        g_stem_a = torch_rms_norm(torch_input_a.float(), torch_w.float())
        check_pcc(g_stem_a[:, :, :, :64], out_a_left, pcc=0.97, label="tree_a slice")
        check_pcc(torch_rms_norm(g_stem_a, torch_w.float()), out_a_right, pcc=0.97, label="tree_a RMS")

        g_stem_b = torch_rms_norm(torch_input_b.float(), torch_w.float())
        check_pcc(torch_rms_norm(g_stem_b, torch_w.float()), out_b_left, pcc=0.97, label="tree_b RMS")
        check_pcc(g_stem_b[:, :, :, :64], out_b_right, pcc=0.97, label="tree_b slice")

    @stress_test_program_cache
    def test_full_grid_stress(self, device):
        """7 items on full 8x8 grid: matmuls, chains, tree, single op.

        Layout:
          - Matmul 1:           (0,0)-(3,1)  8 cores
          - LN->RMS chain:      (4,0)-(7,1)  8 cores
          - RMS->LN chain:      (0,2)-(3,3)  8 cores
          - RMS->MM->RMS chain:  (4,2)-(7,3)  8 cores
          - Branching tree:      (0,4)-(7,5) stem -> (0,4)-(3,5) + (4,4)-(7,5)
          - Matmul 2:           (0,6)-(3,7)  8 cores
          - Single RMS:         (4,6)-(7,7)  8 cores
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc

        torch.manual_seed(42)
        hidden = 128
        ln_compute = ttnn.layernorm_default_compute_config(device.arch())
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_b = torch.zeros(1, 1, 1, hidden, dtype=torch.bfloat16)
        tt_w, tt_b = tt(torch_w, device), tt(torch_b, device)
        norm_shape = (1, 1, 256, hidden)

        def mm_config(gx=4, gy=2):
            return ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=1,
                per_core_N=4,
            )

        # Norm inputs (4 of them)
        torch_norms = [torch.randn(norm_shape, dtype=torch.bfloat16) for _ in range(4)]

        # MM1
        torch_mm1_a = torch.randn(1, 1, 256, 64, dtype=torch.bfloat16)
        torch_mm1_b = torch.randn(1, 1, 64, 128, dtype=torch.bfloat16)
        mm1 = matmul_desc(
            tt(torch_mm1_a, device),
            tt(torch_mm1_b, device),
            core_range_set=cores(0, 0, 3, 1),
            program_config=mm_config(),
        )

        # LN->RMS
        ln1 = layer_norm.layer_norm(
            tt(torch_norms[0], device), core_range_set=cores(4, 0, 7, 1), weight=tt_w, epsilon=1e-5
        )
        rms1 = rms_norm.rms_norm(
            ln1.output_tensors[0],
            core_range_set=cores(4, 0, 7, 1),
            weight=tt_w,
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
        )
        chain1 = Sequential(ln1, rms1)

        # RMS->LN
        rms2 = rms_norm.rms_norm(
            tt(torch_norms[1], device),
            core_range_set=cores(0, 2, 3, 3),
            weight=tt_w,
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
        )
        ln2 = layer_norm.layer_norm(
            rms2.output_tensors[0], core_range_set=cores(0, 2, 3, 3), weight=tt_w, bias=tt_b, epsilon=1e-5
        )
        chain2 = Sequential(rms2, ln2)

        # RMS->MM->RMS
        rms3a = rms_norm.rms_norm(
            tt(torch_norms[2], device), core_range_set=cores(4, 2, 7, 3), weight=tt_w, epsilon=1e-5
        )
        torch_mm3_b = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)
        mm3_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
        )
        mm3 = matmul_desc(
            rms3a.output_tensors[0],
            tt(torch_mm3_b, device),
            core_range_set=cores(4, 2, 7, 3),
            program_config=mm_config(),
            compute_kernel_config=mm3_compute,
        )
        rms3c = rms_norm.rms_norm(mm3.output_tensors[0], core_range_set=cores(4, 2, 7, 3), weight=tt_w, epsilon=1e-5)
        chain3 = Sequential(rms3a, mm3, rms3c)

        # Branching tree
        torch_stem_in = torch.randn(1, 1, 512, hidden, dtype=torch.bfloat16)
        stem_op = rms_norm.rms_norm(
            tt(torch_stem_in, device), core_range_set=cores(0, 4, 7, 5), weight=tt_w, epsilon=1e-5
        )
        br_a = rms_norm.rms_norm(stem_op.output_tensors[0], core_range_set=cores(0, 4, 3, 5), weight=tt_w, epsilon=1e-5)
        br_b = rms_norm.rms_norm(stem_op.output_tensors[0], core_range_set=cores(4, 4, 7, 5), weight=tt_w, epsilon=1e-5)
        tree = Sequential(stem_op, Parallel(br_a, br_b))

        # MM2
        torch_mm2_a = torch.randn(1, 1, 256, 64, dtype=torch.bfloat16)
        torch_mm2_b = torch.randn(1, 1, 64, 128, dtype=torch.bfloat16)
        mm2 = matmul_desc(
            tt(torch_mm2_a, device),
            tt(torch_mm2_b, device),
            core_range_set=cores(0, 6, 3, 7),
            program_config=mm_config(),
        )

        # Single RMS
        single = rms_norm.rms_norm(
            tt(torch_norms[3], device), core_range_set=cores(4, 6, 7, 7), weight=tt_w, epsilon=1e-5
        )

        # Launch all 7
        mm1.launch()
        [out_rms1] = chain1.run()
        [out_ln2] = chain2.run()
        [out_rms3c] = chain3.run()
        [out_br_a, out_br_b] = tree.run()
        mm2.launch()
        single.launch()

        w, b = torch_w.float(), torch_b.float()
        check_pcc(torch_mm1_a @ torch_mm1_b, mm1.output_tensors[0], pcc=0.99, label="MM1")
        check_pcc(torch_rms_norm(torch_layer_norm(torch_norms[0].float(), w), w), out_rms1, label="LN->RMS")
        check_pcc(torch_layer_norm(torch_rms_norm(torch_norms[1].float(), w), w, b), out_ln2, label="RMS->LN")
        check_pcc(
            torch_rms_norm(torch_rms_norm(torch_norms[2].float(), w) @ torch_mm3_b.float(), w),
            out_rms3c,
            label="RMS->MM->RMS",
        )

        g_stem = torch_rms_norm(torch_stem_in.float(), w)
        check_pcc(torch_rms_norm(g_stem, w), out_br_a, label="tree A")
        check_pcc(torch_rms_norm(g_stem, w), out_br_b, label="tree B")

        check_pcc(torch_mm2_a @ torch_mm2_b, mm2.output_tensors[0], pcc=0.99, label="MM2")
        check_pcc(torch_rms_norm(torch_norms[3].float(), w), single.output_tensors[0], pcc=0.99, label="single RMS")


# ===========================================================================
# TestSequentialParallelAPI
# ===========================================================================


class TestSequentialParallelAPI:
    """API surface tests for Sequential/Parallel."""

    @stress_test_program_cache
    def test_sequential_inline(self, device):
        """Sequential(rms, rms).run()."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = make_small_norm_tensors(device)
        cr = cores(0, 0)
        rms1 = rms_norm.rms_norm(t["tt_input"], core_range_set=cr, weight=t["tt_weights"][0], epsilon=1e-5)
        rms2 = rms_norm.rms_norm(rms1.output_tensors[0], core_range_set=cr, weight=t["tt_weights"][1], epsilon=1e-5)

        fused = Sequential(rms1, rms2)
        [out] = fused.run(results=[rms2])

        golden = torch_rms_norm(
            torch_rms_norm(t["torch_input"].float(), t["torch_weights"][0].float()),
            t["torch_weights"][1].float(),
        )
        check_pcc(golden, out, pcc=0.99, label="inline")

    @stress_test_program_cache
    def test_sequential_add_method(self, device):
        """s.add(op) matches inline construction."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = make_small_norm_tensors(device)
        cr = cores(0, 0)
        rms1 = rms_norm.rms_norm(t["tt_input"], core_range_set=cr, weight=t["tt_weights"][0], epsilon=1e-5)
        rms2 = rms_norm.rms_norm(rms1.output_tensors[0], core_range_set=cr, weight=t["tt_weights"][1], epsilon=1e-5)

        s = Sequential(rms1)
        s.add(rms2)
        [out] = s.run(results=[rms2])

        golden = torch_rms_norm(
            torch_rms_norm(t["torch_input"].float(), t["torch_weights"][0].float()),
            t["torch_weights"][1].float(),
        )
        check_pcc(golden, out, pcc=0.99, label="add method")

    @stress_test_program_cache
    def test_sequential_branching(self, device):
        """Sequential(stem, Parallel(a, b)).run() — API-level branching."""
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = make_multi_norm_tensors(device)
        stem = rms_norm.rms_norm(
            t["tt_input"], core_range_set=cores(0, 0, 3, 0), weight=t["tt_weights"][0], epsilon=1e-5
        )
        a = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(0, 0, 1, 0), weight=t["tt_weights"][1], epsilon=1e-5
        )
        b = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(2, 0, 3, 0), weight=t["tt_weights"][2], epsilon=1e-5
        )

        fused = Sequential(stem, Parallel(a, b))
        [out_a, out_b] = fused.run(results=[a, b])

        g_stem = torch_rms_norm(t["torch_input"].float(), t["torch_weights"][0].float())
        check_pcc(torch_rms_norm(g_stem, t["torch_weights"][1].float()), out_a, label="A")
        check_pcc(torch_rms_norm(g_stem, t["torch_weights"][2].float()), out_b, label="B")


# ===========================================================================
# TestCrossOpCompilation
# ===========================================================================


class TestCrossOpCompilation:
    """Verify merged pre-main from different ops produces compilable kernels.

    10 parametrized test cases covering all kernel pair combinations.
    """

    KERNEL_PATHS = {
        "layernorm": "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm.cpp",
        "rmsnorm_post": "ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_post_allgather.cpp",
        "matmul": "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm.cpp",
        "batchnorm": "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp",
        "untilize": "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp",
        "eltwise_sfpu": "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp",
        "typecast": "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp",
    }

    @staticmethod
    def _read_and_process(kernel_path):
        from models.experimental.ops.descriptors.fusion import inline_local_includes

        with open(kernel_path, "r") as f:
            source = f.read()
        kernel_dir = os.path.dirname(os.path.abspath(kernel_path))
        return inline_local_includes(source, kernel_dir)

    @staticmethod
    def _build_fused_source(sources_with_headers):
        from models.experimental.ops.descriptors.fusion import collect_includes, collect_defines

        all_combined = []
        for _, hdrs, s in sources_with_headers:
            hdr_text = "\n".join(c for _, c in hdrs)
            all_combined.append(hdr_text + "\n" + s)
        includes = collect_includes(all_combined)
        defines = collect_defines(all_combined)

        header_path_seen = set()
        file_scope_blocks = []
        for _, hdrs, _ in sources_with_headers:
            for path, content in hdrs:
                if path not in header_path_seen:
                    header_path_seen.add(path)
                    if content.strip():
                        file_scope_blocks.append(content.strip())

        _km_re = re.compile(r"\bvoid\s+kernel_main\s*\(")
        _skip = ("#include", "#define", "#pragma", "#undef")
        phase_pre_mains = {}
        for phase_idx, _, source in sources_with_headers:
            m = _km_re.search(source)
            pre_text = source[: m.start()] if m else source
            pre_lines = [line for line in pre_text.split("\n") if not line.strip().startswith(_skip)]
            phase_pre_mains[phase_idx] = "\n".join(pre_lines).strip()

        lines = ["// Auto-generated fused compute kernel - compilation test", ""]
        lines.extend(defines)
        lines.append("")
        lines.extend(includes)
        lines.append("")
        for block in file_scope_blocks:
            lines.append(block)
            lines.append("")
        for phase_idx, _, _ in sources_with_headers:
            ns = f"phase_{phase_idx}"
            pre_main = phase_pre_mains.get(phase_idx, "")
            lines.append(f"namespace {ns} {{")
            if pre_main.strip():
                lines.append(pre_main)
            lines.append("void run() {}")
            lines.append(f"}} // namespace {ns}")
            lines.append("")
        lines.append("void kernel_main() {}")
        return "\n".join(lines)

    @staticmethod
    def _compile_source(device, source):
        core = cores(0, 0)
        kernel = ttnn.KernelDescriptor()
        kernel.kernel_source = source
        kernel.source_type = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
        kernel.core_ranges = core
        kernel.config = ttnn.ComputeConfigDescriptor()

        cb = ttnn.CBDescriptor()
        fmt = ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.DataType.BFLOAT16, page_size=2048)
        cb.total_size = 2048
        cb.core_ranges = core
        cb.format_descriptors = [fmt]

        desc = ttnn.ProgramDescriptor()
        desc.kernels = [kernel]
        desc.cbs = [cb]

        dummy = ttnn.from_torch(torch.zeros(1, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        dummy2 = ttnn.from_torch(torch.zeros(1, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        try:
            ttnn.generic_op([dummy, dummy2], desc)
        except RuntimeError as e:
            if "build failed" in str(e):
                raise AssertionError(f"Compilation failed: {e}") from None
            raise

    @pytest.mark.parametrize(
        "pair",
        [
            pytest.param(("layernorm", "matmul"), id="ln_matmul"),
            pytest.param(("layernorm", "batchnorm"), id="ln_batchnorm"),
            pytest.param(("layernorm", "untilize"), id="ln_untilize"),
            pytest.param(("rmsnorm_post", "layernorm"), id="rms_ln"),
            pytest.param(("matmul", "batchnorm"), id="matmul_batchnorm"),
            pytest.param(("layernorm", "eltwise_sfpu"), id="ln_sfpu"),
            pytest.param(("layernorm", "typecast"), id="ln_typecast"),
            pytest.param(("matmul", "layernorm", "batchnorm"), id="3phase_mm_ln_bn"),
            pytest.param(("layernorm", "untilize", "eltwise_sfpu"), id="3phase_ln_ut_sfpu"),
            pytest.param(("matmul", "layernorm", "batchnorm", "untilize"), id="4phase_max"),
        ],
    )
    @stress_test_program_cache
    def test_cross_op_compilation(self, device, pair):
        """Parametrized cross-op compilation test."""
        phases = []
        for i, name in enumerate(pair):
            h, s = self._read_and_process(self.KERNEL_PATHS[name])
            phases.append((i, h, s))
        source = self._build_fused_source(phases)
        self._compile_source(device, source)


# ===========================================================================
# TestDocExample
# ===========================================================================


class TestDocExample:
    """Integration test matching the example in op_fusion.md."""

    @stress_test_program_cache
    def test_matmul_slice_ln_rms_tree(self, device):
        """Doc-example: matmul -> Parallel(slice->Parallel(matmul, LN), slice->RMS).

        Matches the tree topology from the overview image in op_fusion.md,
        using the full 8x8 grid (64 cores) split left/right then top/bottom:
            op1: matmul [1,1,1024,256]x[1,1,256,256]        on (0,0)-(7,7)  64 cores
            +- op2: slice left half -> [1,1,1024,128]        on (0,0)-(3,7)  32 cores
            |  +- op4: matmul [1,1,1024,128]x[1,1,128,128]  on (0,0)-(3,3)  16 cores
            |  +- op5: layer_norm [1,1,1024,128]             on (0,4)-(3,7)  16 cores
            +- op3: slice right half -> [1,1,1024,128]       on (4,0)-(7,7)  32 cores
               +- op6: rms_norm [1,1,1024,128]               on (4,0)-(7,7)  32 cores
        """
        import models.experimental.ops.descriptors as descriptors
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel

        torch.manual_seed(42)

        compute_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

        full = cores(0, 0, 7, 7)  # 8x8 = 64 cores
        left = cores(0, 0, 3, 7)  # 4x8 = 32 cores (cols 0-3)
        right = cores(4, 0, 7, 7)  # 4x8 = 32 cores (cols 4-7)
        left_top = cores(0, 0, 3, 3)  # 4x4 = 16 cores (left, rows 0-3)
        left_bot = cores(0, 4, 3, 7)  # 4x4 = 16 cores (left, rows 4-7)

        torch_a = torch.randn(1, 1, 1024, 256, dtype=torch.bfloat16)
        torch_b1 = torch.randn(1, 1, 256, 256, dtype=torch.bfloat16)
        torch_b4 = torch.randn(1, 1, 128, 128, dtype=torch.bfloat16)
        torch_ln_w = torch.ones(1, 1, 1, 128, dtype=torch.bfloat16)
        torch_ln_bias = torch.zeros(1, 1, 1, 128, dtype=torch.bfloat16)
        torch_rms_w = torch.ones(1, 1, 1, 128, dtype=torch.bfloat16)

        op1 = descriptors.matmul(
            tt(torch_a, device), tt(torch_b1, device), core_range_set=full, compute_kernel_config=compute_cfg
        )
        op2 = descriptors.slice(op1.output_tensors[0], [0, 0, 0, 0], [1, 1, 1024, 128], core_range_set=left)
        op3 = descriptors.slice(op1.output_tensors[0], [0, 0, 0, 128], [1, 1, 1024, 256], core_range_set=right)
        op4 = descriptors.matmul(
            op2.output_tensors[0], tt(torch_b4, device), core_range_set=left_top, compute_kernel_config=compute_cfg
        )

        op5 = descriptors.layer_norm(
            op2.output_tensors[0],
            core_range_set=left_bot,
            weight=tt(torch_ln_w, device),
            bias=tt(torch_ln_bias, device),
            epsilon=1e-5,
            compute_kernel_config=compute_cfg,
        )

        op6 = descriptors.rms_norm(
            op3.output_tensors[0],
            core_range_set=right,
            weight=tt(torch_rms_w, device),
            epsilon=1e-5,
            compute_kernel_config=compute_cfg,
        )

        fused = Sequential(
            op1,
            Parallel(
                Sequential(op2, Parallel(op4, op5)),
                Sequential(op3, op6),
            ),
        )
        [out_op4, out_op5, out_op6] = fused.run()

        g1 = torch.matmul(torch_a.float(), torch_b1.float())
        g_left = g1[:, :, :, :128]
        g_right = g1[:, :, :, 128:]

        check_pcc(torch.matmul(g_left, torch_b4.float()), out_op4, pcc=0.97, label="op4 matmul")
        check_pcc(
            torch_layer_norm(g_left, torch_ln_w.float(), torch_ln_bias.float()),
            out_op5,
            pcc=0.97,
            label="op5 LN",
        )
        check_pcc(torch_rms_norm(g_right, torch_rms_w.float()), out_op6, pcc=0.97, label="op6 RMS")


# ===========================================================================
# TestDeepSeekV3
# ===========================================================================


class TestDeepSeekV3:
    """DeepSeek V3 MLA Q/KV RMS norms — inline and persistent modes.

    Both tests use the same shapes/configs as MLA1D decode:
      Q:  [1,1,32,1536] width-sharded 4×4 (0,0)-(3,3), shard [32,96]
      KV: [1,1,32,512]  width-sharded 2×8 (5,0)-(6,7), shard [32,32]
    """

    @staticmethod
    def _setup_mla_norm_configs(device):
        """Shared setup: DeepSeek V3 MLA decode shapes, sharding, weights, configs."""
        from models.demos.deepseek_v3.tt.rms_norm.rms_norm_base import RMSNormBase
        from models.demos.deepseek_v3.utils.config_helpers import COMPUTE_KERNEL_CONFIG_HIFI4_NOFP32_ACC

        q_lora_rank = 1536
        kv_lora_rank = 512
        bsz = 32
        shard_height = ttnn.core.roundup(bsz, ttnn.TILE_SIZE)
        cc = COMPUTE_KERNEL_CONFIG_HIFI4_NOFP32_ACC

        q_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})
        q_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(q_cores, [shard_height, q_lora_rank // 16], ttnn.ShardOrientation.ROW_MAJOR),
        )
        q_pc = RMSNormBase._get_pc(q_mem)

        kv_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 7))})
        kv_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(kv_cores, [shard_height, kv_lora_rank // 16], ttnn.ShardOrientation.ROW_MAJOR),
        )
        kv_pc = RMSNormBase._get_pc(kv_mem)

        torch_q_w = torch.rand(1, 1, 1, q_lora_rank, dtype=torch.bfloat16)
        torch_kv_w = torch.rand(1, 1, 1, kv_lora_rank, dtype=torch.bfloat16)
        tt_q_w = ttnn.from_torch(torch_q_w, device=device, layout=ttnn.TILE_LAYOUT)
        tt_kv_w = ttnn.from_torch(torch_kv_w, device=device, layout=ttnn.TILE_LAYOUT)

        return dict(
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            bsz=bsz,
            q_cores=q_cores,
            kv_cores=kv_cores,
            q_mem=q_mem,
            kv_mem=kv_mem,
            q_pc=q_pc,
            kv_pc=kv_pc,
            cc=cc,
            torch_q_w=torch_q_w,
            torch_kv_w=torch_kv_w,
            tt_q_w=tt_q_w,
            tt_kv_w=tt_kv_w,
        )

    @stress_test_program_cache
    def test_q_kv_rms_norm_inline(self, device):
        """Inline mode: fresh descriptors + Parallel each iteration."""
        from models.experimental.ops.descriptors.fusion import Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        s = self._setup_mla_norm_configs(device)

        for iteration in range(3):
            torch_q_in = torch.rand(1, 1, s["bsz"], s["q_lora_rank"], dtype=torch.bfloat16)
            torch_kv_in = torch.rand(1, 1, s["bsz"], s["kv_lora_rank"], dtype=torch.bfloat16)
            tt_q = ttnn.from_torch(torch_q_in, device=device, layout=ttnn.TILE_LAYOUT, memory_config=s["q_mem"])
            tt_kv = ttnn.from_torch(torch_kv_in, device=device, layout=ttnn.TILE_LAYOUT, memory_config=s["kv_mem"])

            q_desc = rms_norm.rms_norm(
                tt_q,
                epsilon=1e-6,
                weight=s["tt_q_w"],
                memory_config=s["q_mem"],
                core_range_set=s["q_cores"],
                program_config=s["q_pc"],
                compute_kernel_config=s["cc"],
            )
            kv_desc = rms_norm.rms_norm(
                tt_kv,
                epsilon=1e-6,
                weight=s["tt_kv_w"],
                memory_config=s["kv_mem"],
                core_range_set=s["kv_cores"],
                program_config=s["kv_pc"],
                compute_kernel_config=s["cc"],
            )

            out_q, out_kv = Parallel(q=q_desc, kv=kv_desc).run()

            check_pcc(
                torch_rms_norm(torch_q_in.float(), s["torch_q_w"].float(), eps=1e-6),
                out_q,
                pcc=0.98,
                label=f"Q inline iter {iteration}",
            )
            check_pcc(
                torch_rms_norm(torch_kv_in.float(), s["torch_kv_w"].float(), eps=1e-6),
                out_kv,
                pcc=0.98,
                label=f"KV inline iter {iteration}",
            )

    @stress_test_program_cache
    def test_q_kv_rms_norm_persistent(self, device):
        """Persistent mode: lazy-init Parallel, update() + run() each iteration.

        Matches MLA1D decode: reuse branch :class:`OpDescriptor` objects with
        :meth:`~OpDescriptor.update`. Fusion program reuse comes from ``_BUILD_CACHE``;
        no ``FusedOp`` is retained on the container between calls.
        """
        from models.experimental.ops.descriptors.fusion import Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        s = self._setup_mla_norm_configs(device)

        cfg = {
            "q_norm": dict(epsilon=1e-6, weight=s["tt_q_w"], compute_kernel_config=s["cc"]),
            "kv_norm": dict(epsilon=1e-6, weight=s["tt_kv_w"], compute_kernel_config=s["cc"]),
            "fused_qkv_norm": None,
        }

        for i in range(10):
            torch.manual_seed(i)
            torch_q_in = torch.rand(1, 1, s["bsz"], s["q_lora_rank"], dtype=torch.bfloat16)
            torch_kv_in = torch.rand(1, 1, s["bsz"], s["kv_lora_rank"], dtype=torch.bfloat16)
            tt_q = ttnn.from_torch(torch_q_in, device=device, layout=ttnn.TILE_LAYOUT, memory_config=s["q_mem"])
            tt_kv = ttnn.from_torch(torch_kv_in, device=device, layout=ttnn.TILE_LAYOUT, memory_config=s["kv_mem"])

            fused = cfg["fused_qkv_norm"]
            if fused is None:
                fused = Parallel(
                    q=rms_norm.rms_norm(program_config=s["q_pc"], **cfg["q_norm"]),
                    kv=rms_norm.rms_norm(program_config=s["kv_pc"], **cfg["kv_norm"]),
                )
                cfg["fused_qkv_norm"] = fused
            fused.q.update(input_tensor=tt_q)
            fused.kv.update(input_tensor=tt_kv)
            out_q, out_kv = fused.run()

            check_pcc(
                torch_rms_norm(torch_q_in.float(), s["torch_q_w"].float(), eps=1e-6),
                out_q,
                pcc=0.98,
                label=f"Q persistent iter {i}",
            )
            check_pcc(
                torch_rms_norm(torch_kv_in.float(), s["torch_kv_w"].float(), eps=1e-6),
                out_kv,
                pcc=0.98,
                label=f"KV persistent iter {i}",
            )

    def test_q_kv_rms_norm_profile(self, device):
        """Profile fused-inline vs fused-persistent vs unfused (n trials × n iters)."""
        import statistics
        import time

        from models.experimental.ops.descriptors.fusion import Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        s = self._setup_mla_norm_configs(device)
        # Number of trials and iterations reduced to 1 for CI efficiency.
        # For reasonable perf estimates, increase these to e.g. 100.
        N_TRIALS = 1
        N_ITERS = 1

        torch_q_in = torch.rand(1, 1, s["bsz"], s["q_lora_rank"], dtype=torch.bfloat16)
        torch_kv_in = torch.rand(1, 1, s["bsz"], s["kv_lora_rank"], dtype=torch.bfloat16)
        tt_q_base = ttnn.from_torch(torch_q_in, device=device, layout=ttnn.TILE_LAYOUT, memory_config=s["q_mem"])
        tt_kv_base = ttnn.from_torch(torch_kv_in, device=device, layout=ttnn.TILE_LAYOUT, memory_config=s["kv_mem"])

        q_norm_kw = dict(epsilon=1e-6, weight=s["tt_q_w"], compute_kernel_config=s["cc"])
        kv_norm_kw = dict(epsilon=1e-6, weight=s["tt_kv_w"], compute_kernel_config=s["cc"])

        # --- warmup all three paths (2 iters each) so caches are hot ---
        for _ in range(2):
            q_d = rms_norm.rms_norm(tt_q_base, program_config=s["q_pc"], **q_norm_kw)
            kv_d = rms_norm.rms_norm(tt_kv_base, program_config=s["kv_pc"], **kv_norm_kw)
            Parallel(q=q_d, kv=kv_d).run()

        fused_persistent = Parallel(
            q=rms_norm.rms_norm(program_config=s["q_pc"], **q_norm_kw),
            kv=rms_norm.rms_norm(program_config=s["kv_pc"], **kv_norm_kw),
        )
        for _ in range(2):
            fused_persistent.q.update(tt_q_base)
            fused_persistent.kv.update(tt_kv_base)
            fused_persistent.run()

        for _ in range(2):
            ttnn.rms_norm(
                tt_q_base,
                weight=s["tt_q_w"],
                epsilon=1e-6,
                program_config=s["q_pc"],
                compute_kernel_config=s["cc"],
                memory_config=s["q_mem"],
            )
            ttnn.rms_norm(
                tt_kv_base,
                weight=s["tt_kv_w"],
                epsilon=1e-6,
                program_config=s["kv_pc"],
                compute_kernel_config=s["cc"],
                memory_config=s["kv_mem"],
            )
        ttnn.synchronize_device(device)

        # --- fused inline ---
        inline_times = []
        for _ in range(N_TRIALS):
            t0 = time.perf_counter()
            for _ in range(N_ITERS):
                q_d = rms_norm.rms_norm(tt_q_base, program_config=s["q_pc"], **q_norm_kw)
                kv_d = rms_norm.rms_norm(tt_kv_base, program_config=s["kv_pc"], **kv_norm_kw)
                Parallel(q=q_d, kv=kv_d).run()
            ttnn.synchronize_device(device)
            inline_times.append(time.perf_counter() - t0)

        # --- fused persistent (positional update to match real model usage) ---
        persistent_times = []
        for _ in range(N_TRIALS):
            t0 = time.perf_counter()
            for _ in range(N_ITERS):
                fused_persistent.q.update(tt_q_base)
                fused_persistent.kv.update(tt_kv_base)
                fused_persistent.run()
            ttnn.synchronize_device(device)
            persistent_times.append(time.perf_counter() - t0)

        # --- unfused (two consecutive ttnn.rms_norm) ---
        unfused_times = []
        for _ in range(N_TRIALS):
            t0 = time.perf_counter()
            for _ in range(N_ITERS):
                ttnn.rms_norm(
                    tt_q_base,
                    weight=s["tt_q_w"],
                    epsilon=1e-6,
                    program_config=s["q_pc"],
                    compute_kernel_config=s["cc"],
                    memory_config=s["q_mem"],
                )
                ttnn.rms_norm(
                    tt_kv_base,
                    weight=s["tt_kv_w"],
                    epsilon=1e-6,
                    program_config=s["kv_pc"],
                    compute_kernel_config=s["cc"],
                    memory_config=s["kv_mem"],
                )
            ttnn.synchronize_device(device)
            unfused_times.append(time.perf_counter() - t0)

        med_inline = statistics.median(inline_times)
        med_persistent = statistics.median(persistent_times)
        med_unfused = statistics.median(unfused_times)
        per_iter_inline_us = med_inline / N_ITERS * 1e6
        per_iter_persistent_us = med_persistent / N_ITERS * 1e6
        per_iter_unfused_us = med_unfused / N_ITERS * 1e6

        print(f"\n{'='*70}")
        print(f"  Q/KV RMS Norm Profile: {N_TRIALS} trials × {N_ITERS} iterations")
        print(f"{'='*70}")
        print(f"  {'Mode':<22} {'Median (s)':<14} {'Per-iter (µs)':<16} {'vs unfused':<12}")
        print(f"  {'-'*22} {'-'*14} {'-'*16} {'-'*12}")
        print(
            f"  {'Fused inline':<22} {med_inline:<14.4f} {per_iter_inline_us:<16.1f} {med_unfused/med_inline:<12.2f}x"
        )
        print(
            f"  {'Fused persistent':<22} {med_persistent:<14.4f} {per_iter_persistent_us:<16.1f} {med_unfused/med_persistent:<12.2f}x"
        )
        print(f"  {'Unfused (2× rms_norm)':<22} {med_unfused:<14.4f} {per_iter_unfused_us:<16.1f} {'1.00x':<12}")
        print(f"{'='*70}\n")


# ===========================================================================
# TestPersistentMode
# ===========================================================================


class TestPersistentMode:
    """End-to-end persistent mode: build once, update() inputs, run() repeatedly."""

    @stress_test_program_cache
    def test_persistent_parallel_pcc(self, device):
        """Persistent Parallel Q/KV norms: 20 iterations with different random data.

        Uses the full persistent API: descriptors created without activation,
        named kwargs on Parallel, update() + run() each iteration.
        """
        from models.experimental.ops.descriptors.fusion import Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        q_cores = cores(0, 0, 3, 3)
        kv_cores = cores(5, 0, 6, 7)
        q_shard_w, kv_shard_w = 96, 32
        q_total_w, kv_total_w = 16 * q_shard_w, 16 * kv_shard_w

        q_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(q_cores, [32, q_shard_w], ttnn.ShardOrientation.ROW_MAJOR),
        )
        q_pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(4, 4),
            subblock_w=q_shard_w // 32,
            block_h=1,
            block_w=q_shard_w // 32,
            inplace=False,
        )
        kv_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(kv_cores, [32, kv_shard_w], ttnn.ShardOrientation.ROW_MAJOR),
        )
        kv_pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(2, 8),
            subblock_w=kv_shard_w // 32,
            block_h=1,
            block_w=kv_shard_w // 32,
            inplace=False,
        )

        # Persistent setup: weights + descriptors created once (no activation)
        torch_q_w = torch.rand(1, 1, 1, q_total_w, dtype=torch.bfloat16)
        torch_kv_w = torch.rand(1, 1, 1, kv_total_w, dtype=torch.bfloat16)
        tt_q_w = ttnn.from_torch(torch_q_w, device=device, layout=ttnn.TILE_LAYOUT)
        tt_kv_w = ttnn.from_torch(torch_kv_w, device=device, layout=ttnn.TILE_LAYOUT)

        fused = Parallel(
            q=rms_norm.rms_norm(
                weight=tt_q_w,
                epsilon=1e-5,
                memory_config=q_mem,
                core_range_set=q_cores,
                program_config=q_pc,
            ),
            kv=rms_norm.rms_norm(
                weight=tt_kv_w,
                epsilon=1e-5,
                memory_config=kv_mem,
                core_range_set=kv_cores,
                program_config=kv_pc,
            ),
        )

        # 20 iterations with different random activations
        for i in range(20):
            torch.manual_seed(i)
            torch_q_in = torch.rand(1, 1, 32, q_total_w, dtype=torch.bfloat16)
            torch_kv_in = torch.rand(1, 1, 32, kv_total_w, dtype=torch.bfloat16)
            tt_q_in = ttnn.from_torch(torch_q_in, device=device, layout=ttnn.TILE_LAYOUT, memory_config=q_mem)
            tt_kv_in = ttnn.from_torch(torch_kv_in, device=device, layout=ttnn.TILE_LAYOUT, memory_config=kv_mem)

            fused.q.update(tt_q_in)
            fused.kv.update(tt_kv_in)
            [out_q, out_kv] = fused.run()

            check_pcc(
                torch_rms_norm(torch_q_in.float(), torch_q_w.float()),
                out_q,
                pcc=0.98,
                label=f"Q persistent iter {i}",
            )
            check_pcc(
                torch_rms_norm(torch_kv_in.float(), torch_kv_w.float()),
                out_kv,
                pcc=0.98,
                label=f"KV persistent iter {i}",
            )

    @stress_test_program_cache
    def test_persistent_sequential_pcc(self, device):
        """Persistent Sequential LN→RMS: 10 iterations with different random data.

        Only the first op's activation is updated — the intermediate tensor
        (LN output → RMS input) is auto-wired via _Placeholder identity.
        """
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm

        cr = cores(0, 0)
        cc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
        )

        torch.manual_seed(42)
        hidden = 128
        torch_w0 = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_w1 = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        tt_w0 = tt(torch_w0, device)
        tt_w1 = tt(torch_w1, device)

        # Persistent setup: wire topology via _Placeholder identity
        ln_desc = layer_norm.layer_norm(
            weight=tt_w0,
            epsilon=1e-5,
            core_range_set=cr,
            compute_kernel_config=cc,
        )
        rms_desc = rms_norm.rms_norm(
            ln_desc.output_tensors[0],  # _Placeholder — auto-wired by Sequential
            weight=tt_w1,
            epsilon=1e-5,
            core_range_set=cr,
            compute_kernel_config=cc,
        )
        fused = Sequential(ln=ln_desc, rms=rms_desc)

        for i in range(10):
            torch.manual_seed(i + 100)
            torch_in = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
            tt_in = tt(torch_in, device)

            # Only update the external input — internal wire handled automatically
            fused.ln.update(tt_in)
            [out] = fused.run(results=[rms_desc])

            golden = torch_rms_norm(
                torch_layer_norm(torch_in.float(), torch_w0.float()),
                torch_w1.float(),
            )
            check_pcc(golden, out, pcc=0.97, label=f"Sequential persistent iter {i}")

    @stress_test_program_cache
    def test_persistent_binary_tree_inline(self, device):
        """3-level binary tree (stem→2 mid→4 leaves) using inline descriptors
        with named kwargs, exercised through stress_test_program_cache.

        Tree (8 cores):
            RMS [0-7] → RMS [0-3] → RMS [0-1] / RMS [2-3]
                      → RMS [4-7] → RMS [4-5] / RMS [6-7]

        Uses named kwargs at every level so names are hoisted to top.
        Verifies PCC on all 4 leaf outputs.
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = make_multi_norm_tensors(device)
        wt = t["tt_weights"]

        def rms(inp, cr, wi):
            return rms_norm.rms_norm(inp, core_range_set=cr, weight=wt[wi], epsilon=1e-5)

        root = rms(t["tt_input"], cores(0, 0, 7, 0), 0)
        left = rms(root.output_tensors[0], cores(0, 0, 3, 0), 1)
        right = rms(root.output_tensors[0], cores(4, 0, 7, 0), 2)
        ll = rms(left.output_tensors[0], cores(0, 0, 1, 0), 3)
        lr = rms(left.output_tensors[0], cores(2, 0, 3, 0), 4)
        rl = rms(right.output_tensors[0], cores(4, 0, 5, 0), 5)
        rr = rms(right.output_tensors[0], cores(6, 0, 7, 0), 6)

        # Compose topology — matches the doc example pattern
        fused = Sequential(
            root=root,
            branches=Parallel(
                left_path=Sequential(left=left, leaves=Parallel(ll=ll, lr=lr)),
                right_path=Sequential(right=right, rms=Parallel(rl=rl, rr=rr)),
            ),
        )

        # Verify all names are hoisted to top level
        assert hasattr(fused, "root")
        assert hasattr(fused, "ll")
        assert hasattr(fused, "rr")

        # Run with explicit results — same as doc example
        out_ll, out_lr, out_rl, out_rr = fused.run(results=[ll, lr, rl, rr])

        ws = t["torch_weights"]
        g_root = torch_rms_norm(t["torch_input"].float(), ws[0].float())
        g_left = torch_rms_norm(g_root, ws[1].float())
        g_right = torch_rms_norm(g_root, ws[2].float())
        goldens = [
            torch_rms_norm(g_left, ws[3].float()),
            torch_rms_norm(g_left, ws[4].float()),
            torch_rms_norm(g_right, ws[5].float()),
            torch_rms_norm(g_right, ws[6].float()),
        ]
        for out, golden, label in zip([out_ll, out_lr, out_rl, out_rr], goldens, ["LL", "LR", "RL", "RR"]):
            check_pcc(golden, out, label=f"tree {label}")

    @stress_test_program_cache
    def test_persistent_matmul_deferred(self, device):
        """Matmul with deferred inputs: build once, update both inputs, run."""
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc

        torch.manual_seed(42)
        M, K, N = 32, 64, 32
        torch_a = torch.rand(1, 1, M, K, dtype=torch.bfloat16)
        torch_b = torch.rand(1, 1, K, N, dtype=torch.bfloat16)
        tt_a = ttnn.from_torch(torch_a, device=device, layout=ttnn.TILE_LAYOUT)
        tt_b = ttnn.from_torch(torch_b, device=device, layout=ttnn.TILE_LAYOUT)

        # Inline matmul first (for reference)
        inline_desc = matmul_desc(tt_a, tt_b)
        inline_desc.launch()
        ref_out = ttnn.to_torch(inline_desc.output_tensors[0])

        # Deferred matmul
        deferred = matmul_desc()
        assert deferred.program_cache_key is None
        deferred.update(tt_a, tt_b)
        assert deferred.program_cache_key is not None
        deferred.launch()
        deferred_out = ttnn.to_torch(deferred.output_tensors[0])

        golden = torch_a.float() @ torch_b.float()
        _, pcc_inline = comp_pcc(golden, ref_out, pcc=0.98)
        _, pcc_deferred = comp_pcc(golden, deferred_out, pcc=0.98)
        assert pcc_inline > 0.98, f"Inline matmul PCC: {pcc_inline}"
        assert pcc_deferred > 0.98, f"Deferred matmul PCC: {pcc_deferred}"

    @stress_test_program_cache
    def test_persistent_sequential_3chain_update_weight(self, device):
        """3-phase Sequential LN→RMS→LN with weight update on the middle op.

        Tests updating a non-first input (weight at index 1) on an intermediate
        op in a persistent Sequential chain.

        Wiring:
            ln1.input  = external (user updates)
            rms.input  = ln1.output (auto-wired via _Placeholder)
            rms.weight = external (user updates each iteration)
            ln2.input  = rms.output (auto-wired via _Placeholder)
        """
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm

        cr = cores(0, 0)
        cc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
        )
        hidden = 128
        torch_w0 = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_w2 = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        tt_w0 = tt(torch_w0, device)
        tt_w2 = tt(torch_w2, device)

        # Persistent setup: wire internal connections via _Placeholder.
        # Pass an initial weight to rms — it gets replaced each iteration.
        torch_w1_init = torch.rand(1, 1, 1, hidden, dtype=torch.bfloat16) + 0.5
        tt_w1_init = tt(torch_w1_init, device)

        ln1 = layer_norm.layer_norm(
            weight=tt_w0,
            epsilon=1e-5,
            core_range_set=cr,
            compute_kernel_config=cc,
        )
        rms_op = rms_norm.rms_norm(
            ln1.output_tensors[0],  # auto-wired from ln1
            weight=tt_w1_init,
            epsilon=1e-5,
            core_range_set=cr,
            compute_kernel_config=cc,
        )
        ln2 = layer_norm.layer_norm(
            rms_op.output_tensors[0],  # auto-wired from rms
            weight=tt_w2,
            epsilon=1e-5,
            core_range_set=cr,
            compute_kernel_config=cc,
        )
        fused = Sequential(ln1=ln1, rms=rms_op, ln2=ln2)

        for i in range(5):
            torch.manual_seed(i + 200)
            torch_in = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
            # Different weight each iteration for the middle op
            torch_w1 = torch.rand(1, 1, 1, hidden, dtype=torch.bfloat16) + 0.5
            tt_in = tt(torch_in, device)
            tt_w1 = tt(torch_w1, device)

            fused.ln1.update(tt_in)
            fused.rms.update(weight=tt_w1)  # update non-first input (index 1) by name
            [out] = fused.run(results=[ln2])

            g = torch_layer_norm(torch_in.float(), torch_w0.float())
            g = torch_rms_norm(g, torch_w1.float())
            g = torch_layer_norm(g, torch_w2.float())
            check_pcc(g, out, pcc=0.97, label=f"3chain weight-update iter {i}")

    @stress_test_program_cache
    def test_persistent_stem_parallel_branches(self, device):
        """Persistent Sequential(stem, Parallel(a, b)) with auto-wired branches.

        Wiring:
            stem.input = external (user updates)
            a.input    = stem.output (auto-wired via _Placeholder)
            b.input    = stem.output (auto-wired via _Placeholder)
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = make_multi_norm_tensors(device)
        wt = t["tt_weights"]
        ws = t["torch_weights"]

        # Persistent: stem deferred, branches auto-wired
        stem = rms_norm.rms_norm(
            core_range_set=cores(0, 0, 7, 0),
            weight=wt[0],
            epsilon=1e-5,
        )
        branch_a = rms_norm.rms_norm(
            stem.output_tensors[0],  # auto-wired from stem
            core_range_set=cores(0, 0, 3, 0),
            weight=wt[1],
            epsilon=1e-5,
        )
        branch_b = rms_norm.rms_norm(
            stem.output_tensors[0],  # auto-wired from stem (same _Placeholder)
            core_range_set=cores(4, 0, 7, 0),
            weight=wt[2],
            epsilon=1e-5,
        )
        fused = Sequential(
            stem=stem,
            branches=Parallel(a=branch_a, b=branch_b),
        )

        for i in range(5):
            torch.manual_seed(i + 300)
            torch_in = torch.randn_like(t["torch_input"])
            tt_in = ttnn.from_torch(torch_in, device=device, layout=ttnn.TILE_LAYOUT)

            fused.stem.update(tt_in)
            [out_a, out_b] = fused.run()

            g_stem = torch_rms_norm(torch_in.float(), ws[0].float())
            g_a = torch_rms_norm(g_stem, ws[1].float())
            g_b = torch_rms_norm(g_stem, ws[2].float())
            check_pcc(g_a, out_a, pcc=0.98, label=f"branch_a iter {i}")
            check_pcc(g_b, out_b, pcc=0.98, label=f"branch_b iter {i}")


# ===========================================================================
# TestAsymmetricBarrier
# ===========================================================================


class TestAsymmetricBarrier:
    """Tests for asymmetric barrier (narrow→wide) topologies.

    In narrow→wide topologies, extra cores in the wide phase get _NOOP_OP
    placeholder phases and must use SyncMode::WaitOnly (skip arrive, only
    wait on release).  These tests exercise edge cases that could cause hangs
    if the arrive threshold or sync mode dispatch is wrong.
    """

    @stress_test_program_cache
    def test_narrow_stem_wide_branches(self, device):
        """Narrow stem (2 cores) → wide branches (4+4 cores).

        6 cores get _NOOP_OP for the stem phase.  arrive_threshold=2 at
        the stem→branch barrier.  If all 8 cores arrive, threshold would
        be wrong → hang or early release.
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = make_multi_norm_tensors(device)
        # Stem: only 2 cores
        stem = rms_norm.rms_norm(
            t["tt_input"], core_range_set=cores(0, 0, 1, 0), weight=t["tt_weights"][0], epsilon=1e-5
        )
        # Branches: 4 cores each — wider than stem
        a = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(0, 0, 3, 0), weight=t["tt_weights"][1], epsilon=1e-5
        )
        b = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(4, 0, 7, 0), weight=t["tt_weights"][2], epsilon=1e-5
        )

        fused = Sequential(stem, Parallel(a, b))
        [out_a, out_b] = fused.run(results=[a, b])

        g_stem = torch_rms_norm(t["torch_input"].float(), t["torch_weights"][0].float())
        check_pcc(torch_rms_norm(g_stem, t["torch_weights"][1].float()), out_a, label="branch A")
        check_pcc(torch_rms_norm(g_stem, t["torch_weights"][2].float()), out_b, label="branch B")

    @stress_test_program_cache
    def test_single_core_stem_wide_branches(self, device):
        """Extreme asymmetry: 1-core stem → 4+4 core branches.

        7 cores get _NOOP_OP.  arrive_threshold=1, release_cores=8.
        Tests the most extreme arrive/release ratio.
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = make_multi_norm_tensors(device)
        stem = rms_norm.rms_norm(t["tt_input"], core_range_set=cores(0, 0), weight=t["tt_weights"][0], epsilon=1e-5)
        a = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(0, 0, 3, 0), weight=t["tt_weights"][1], epsilon=1e-5
        )
        b = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(4, 0, 7, 0), weight=t["tt_weights"][2], epsilon=1e-5
        )

        fused = Sequential(stem, Parallel(a, b))
        [out_a, out_b] = fused.run(results=[a, b])

        g_stem = torch_rms_norm(t["torch_input"].float(), t["torch_weights"][0].float())
        check_pcc(torch_rms_norm(g_stem, t["torch_weights"][1].float()), out_a, label="branch A")
        check_pcc(torch_rms_norm(g_stem, t["torch_weights"][2].float()), out_b, label="branch B")

    @stress_test_program_cache
    def test_narrow_wide_repeated_execution(self, device):
        """Narrow→wide with repeated execution to catch stale semaphores.

        Re-execution is the most common source of barrier bugs: stale
        arrive/release semaphore values from run N can deadlock run N+1
        if the threshold or reset is wrong.
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = make_multi_norm_tensors(device)
        stem = rms_norm.rms_norm(
            t["tt_input"], core_range_set=cores(0, 0, 1, 0), weight=t["tt_weights"][0], epsilon=1e-5
        )
        a = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(0, 0, 3, 0), weight=t["tt_weights"][1], epsilon=1e-5
        )
        b = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(4, 0, 7, 0), weight=t["tt_weights"][2], epsilon=1e-5
        )

        fused = Sequential(stem, Parallel(a, b))

        g_stem = torch_rms_norm(t["torch_input"].float(), t["torch_weights"][0].float())

        for run in range(5):
            [out_a, out_b] = fused.run()
            check_pcc(
                torch_rms_norm(g_stem, t["torch_weights"][1].float()),
                out_a,
                label=f"branch A run {run}",
            )
            check_pcc(
                torch_rms_norm(g_stem, t["torch_weights"][2].float()),
                out_b,
                label=f"branch B run {run}",
            )

    @stress_test_program_cache
    def test_multi_level_narrow_wide(self, device):
        """Narrow stem → mid-width → wide leaves.  Different arrive counts at each transition.

        Tree:
            RMS [0-1]  (2 cores, stem)
            → RMS [0-3] (4 cores, mid) — 2 extra noop cores at stem barrier
              → RMS [0-1] (leaf)
              → RMS [2-3] (leaf)
            → RMS [4-7] (4 cores, mid) — 4 extra noop cores at stem barrier
              → RMS [4-5] (leaf)
              → RMS [6-7] (leaf)

        Transition 0 (stem→mid): arrive=2, release=8
        Transition 1 (mid→leaf): arrive=4, release=4 (symmetric per group)
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = make_multi_norm_tensors(device)
        wt = t["tt_weights"]

        def rms(inp, cr, wi):
            return rms_norm.rms_norm(inp, core_range_set=cr, weight=wt[wi], epsilon=1e-5)

        stem = rms(t["tt_input"], cores(0, 0, 1, 0), 0)
        left = rms(stem.output_tensors[0], cores(0, 0, 3, 0), 1)
        right = rms(stem.output_tensors[0], cores(4, 0, 7, 0), 2)
        ll = rms(left.output_tensors[0], cores(0, 0, 1, 0), 3)
        lr = rms(left.output_tensors[0], cores(2, 0, 3, 0), 4)
        rl = rms(right.output_tensors[0], cores(4, 0, 5, 0), 5)
        rr = rms(right.output_tensors[0], cores(6, 0, 7, 0), 6)

        fused = Sequential(
            stem,
            Parallel(
                Sequential(left, Parallel(ll, lr)),
                Sequential(right, Parallel(rl, rr)),
            ),
        )
        outs = fused.run()

        ws = t["torch_weights"]
        g_stem = torch_rms_norm(t["torch_input"].float(), ws[0].float())
        g_left = torch_rms_norm(g_stem, ws[1].float())
        g_right = torch_rms_norm(g_stem, ws[2].float())
        goldens = [
            torch_rms_norm(g_left, ws[3].float()),
            torch_rms_norm(g_left, ws[4].float()),
            torch_rms_norm(g_right, ws[5].float()),
            torch_rms_norm(g_right, ws[6].float()),
        ]
        for i, label in enumerate(["LL", "LR", "RL", "RR"]):
            check_pcc(goldens[i], outs[i], label=label)

    @stress_test_program_cache
    def test_narrow_deep_left_wide_right(self, device):
        """Narrow stem with asymmetric depth AND width.

        Tree:
            RMS [0-1]  (2 cores, stem)
            → RMS [0-1] → RMS [0-1]   (deep left, same 2 cores)
            → RMS [2-7]               (wide right, 6 cores)

        Tests: stem barrier has arrive=2, release=8.  Right branch has
        6 noop-padded cores.  Left branch goes 2 levels deeper.
        Core groups will have very different phase counts AND noop patterns.
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = make_multi_norm_tensors(device)
        wt = t["tt_weights"]

        def rms(inp, cr, wi):
            return rms_norm.rms_norm(inp, core_range_set=cr, weight=wt[wi], epsilon=1e-5)

        stem = rms(t["tt_input"], cores(0, 0, 1, 0), 0)
        left1 = rms(stem.output_tensors[0], cores(0, 0, 1, 0), 1)
        left2 = rms(left1.output_tensors[0], cores(0, 0, 1, 0), 2)
        right = rms(stem.output_tensors[0], cores(2, 0, 7, 0), 3)

        fused = Sequential(
            stem,
            Parallel(
                Sequential(left1, left2),
                right,
            ),
        )
        [out_left, out_right] = fused.run(results=[left2, right])

        ws = t["torch_weights"]
        g_stem = torch_rms_norm(t["torch_input"].float(), ws[0].float())
        check_pcc(
            torch_rms_norm(torch_rms_norm(g_stem, ws[1].float()), ws[2].float()),
            out_left,
            label="deep left",
        )
        check_pcc(torch_rms_norm(g_stem, ws[3].float()), out_right, label="wide right")

    @stress_test_program_cache
    def test_narrow_wide_with_slice(self, device):
        """Narrow stem → wide branches with slice op in one branch.

        Tree:
            RMS [0-1]  (2 cores)
            → slice [0-3]  (4 cores, narrow→wide with data movement)
            → RMS [4-7]   (4 cores)

        Slice is a data movement op (no compute kernel), so this tests
        the noop+SyncMode interaction with non-compute phases.
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors.data_movement.slice import slice as slice_desc

        torch.manual_seed(42)
        hidden = 128
        torch_input = torch.randn(1, 1, 256, hidden, dtype=torch.bfloat16)
        torch_w0 = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_w1 = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)

        stem = rms_norm.rms_norm(
            tt(torch_input, device), core_range_set=cores(0, 0, 1, 0), weight=tt(torch_w0, device), epsilon=1e-5
        )
        branch_a = slice_desc(stem.output_tensors[0], [0, 0, 0, 0], [1, 1, 256, 64], core_range_set=cores(0, 0, 3, 0))
        branch_b = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(4, 0, 7, 0), weight=tt(torch_w1, device), epsilon=1e-5
        )

        fused = Sequential(stem, Parallel(branch_a, branch_b))
        [out_a, out_b] = fused.run(results=[branch_a, branch_b])

        g_stem = torch_rms_norm(torch_input.float(), torch_w0.float())
        check_pcc(g_stem[:, :, :, :64], out_a, pcc=0.97, label="A (slice)")
        check_pcc(torch_rms_norm(g_stem, torch_w1.float()), out_b, label="B (RMS)")

    @stress_test_program_cache
    def test_fully_disjoint_parent_children(self, device):
        """Fully disjoint: parent {0,1} → children {2-3, 4-7}.

        Parent cores share zero overlap with any child core.  Parent
        cores get _NOOP_OP exit phase, arrive at barrier, then exit.
        Child cores get _NOOP_OP wait phase, then run real work.
        This would deadlock without the exit-phase fix.
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = make_multi_norm_tensors(device)
        stem = rms_norm.rms_norm(
            t["tt_input"], core_range_set=cores(0, 0, 1, 0), weight=t["tt_weights"][0], epsilon=1e-5
        )
        a = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(2, 0, 3, 0), weight=t["tt_weights"][1], epsilon=1e-5
        )
        b = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(4, 0, 7, 0), weight=t["tt_weights"][2], epsilon=1e-5
        )

        fused = Sequential(stem, Parallel(a, b))
        [out_a, out_b] = fused.run(results=[a, b])

        g_stem = torch_rms_norm(t["torch_input"].float(), t["torch_weights"][0].float())
        check_pcc(torch_rms_norm(g_stem, t["torch_weights"][1].float()), out_a, label="branch A")
        check_pcc(torch_rms_norm(g_stem, t["torch_weights"][2].float()), out_b, label="branch B")

    @stress_test_program_cache
    def test_partial_disjoint_one_core_exits(self, device):
        """Partial disjoint: parent {0,1} → child {1-7}.

        Core 0 does real work in parent but isn't in any child — it gets
        an exit phase.  Core 1 overlaps and continues normally.  Cores
        2-7 get noop for parent.
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = make_multi_norm_tensors(device)
        stem = rms_norm.rms_norm(
            t["tt_input"], core_range_set=cores(0, 0, 1, 0), weight=t["tt_weights"][0], epsilon=1e-5
        )
        a = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(1, 0, 3, 0), weight=t["tt_weights"][1], epsilon=1e-5
        )
        b = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(4, 0, 7, 0), weight=t["tt_weights"][2], epsilon=1e-5
        )

        fused = Sequential(stem, Parallel(a, b))
        [out_a, out_b] = fused.run(results=[a, b])

        g_stem = torch_rms_norm(t["torch_input"].float(), t["torch_weights"][0].float())
        check_pcc(torch_rms_norm(g_stem, t["torch_weights"][1].float()), out_a, label="branch A")
        check_pcc(torch_rms_norm(g_stem, t["torch_weights"][2].float()), out_b, label="branch B")

    @stress_test_program_cache
    def test_disjoint_repeated_execution(self, device):
        """Fully disjoint with repeated execution to catch stale semaphores."""
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = make_multi_norm_tensors(device)
        stem = rms_norm.rms_norm(
            t["tt_input"], core_range_set=cores(0, 0, 1, 0), weight=t["tt_weights"][0], epsilon=1e-5
        )
        a = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(2, 0, 3, 0), weight=t["tt_weights"][1], epsilon=1e-5
        )
        b = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(4, 0, 7, 0), weight=t["tt_weights"][2], epsilon=1e-5
        )

        fused = Sequential(stem, Parallel(a, b))

        g_stem = torch_rms_norm(t["torch_input"].float(), t["torch_weights"][0].float())

        for run in range(5):
            [out_a, out_b] = fused.run()
            check_pcc(
                torch_rms_norm(g_stem, t["torch_weights"][1].float()),
                out_a,
                label=f"branch A run {run}",
            )
            check_pcc(
                torch_rms_norm(g_stem, t["torch_weights"][2].float()),
                out_b,
                label=f"branch B run {run}",
            )
