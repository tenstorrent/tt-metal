# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

import os
import re

import pytest
import torch
import ttnn


from models.common.utility_functions import comp_pcc


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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_tensors(device):
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


@pytest.fixture
def multi_tensors(device):
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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestInfrastructure:
    """Build-time infrastructure tests (CB extraction, source structure, named args)."""

    def test_cb_extraction(self, device, test_tensors):
        """extract_cb_info on LN descriptor, check c_0/c_16."""
        from models.experimental.ops.descriptors.fusion import extract_cb_info
        from models.experimental.ops.descriptors.normalization import layer_norm

        t = test_tensors
        desc = layer_norm.layer_norm(
            t["tt_input"], core_range_set=cores(0, 0), weight=t["tt_weights"][0], bias=t["tt_biases"][0], epsilon=1e-5
        )
        cb_info = extract_cb_info(desc.descriptor)
        assert len(cb_info) > 0
        assert 0 in cb_info, "Should have input CB (c_0)"
        assert 16 in cb_info, "Should have output CB (c_16)"

    def test_fused_source_structure(self, device, test_tensors):
        """2-phase LN->LN: verify phase namespaces and barrier code in fused source."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm

        t = test_tensors
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

    def test_named_args_phase_prefix(self, device, test_tensors):
        """2-phase fused: verify phase_1_ prefix on named CT args."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm

        t = test_tensors
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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestSequentialExecution:
    """Core sequential chain execution tests."""

    @pytest.mark.parametrize("num_phases", [2, 3, 4])
    def test_norm_chain(self, device, test_tensors, num_phases):
        """Mixed LN/RMS chain of varying length on single core."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm

        t = test_tensors
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

        fused = Sequential(*ops).build(device)
        fused.launch()
        check_pcc(golden, fused.output_tensors[0], pcc=0.97, label=f"{num_phases}-phase chain")

    @pytest.mark.parametrize("core_x", [3, 5, 7])
    def test_chain_on_nonzero_core(self, device, test_tensors, core_x):
        """2-phase LN->RMS on non-origin single core."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm

        t = test_tensors
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

        fused = Sequential(ln, rms).build(device)
        fused.launch()

        golden = torch_rms_norm(
            torch_layer_norm(t["torch_input"].float(), t["torch_weights"][0].float()),
            t["torch_weights"][1].float(),
        )
        check_pcc(golden, fused.output_tensors[0], label=f"core ({core_x},0)")

    @pytest.mark.parametrize(
        "grid",
        [
            pytest.param((3, 0, 0, 0), id="1D_4core"),
            pytest.param((1, 1, 0, 0), id="2D_2x2"),
        ],
    )
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

        fused = Sequential(ln, rms).build(device)
        fused.launch()

        golden = torch_rms_norm(torch_layer_norm(torch_input.float(), torch_w.float()), torch_w.float())
        check_pcc(golden, fused.output_tensors[0], label="multicore chain")

    def test_repeated_execution(self, device, test_tensors):
        """Same fused op launched 3x — verify no stale state."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = test_tensors
        cr = cores(0, 0)
        rms1 = rms_norm.rms_norm(t["tt_input"], core_range_set=cr, weight=t["tt_weights"][0], epsilon=1e-5)
        rms2 = rms_norm.rms_norm(rms1.output_tensors[0], core_range_set=cr, weight=t["tt_weights"][1], epsilon=1e-5)

        fused = Sequential(rms1, rms2).build(device)
        golden = torch_rms_norm(
            torch_rms_norm(t["torch_input"].float(), t["torch_weights"][0].float()),
            t["torch_weights"][1].float(),
        )

        for i in range(3):
            fused.launch()
            check_pcc(golden, fused.output_tensors[0], label=f"run {i}")

    def test_single_op_passthrough(self, device, test_tensors):
        """Single op in Sequential — verify pass-through."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = test_tensors
        rms = rms_norm.rms_norm(t["tt_input"], core_range_set=cores(0, 0), weight=t["tt_weights"][0], epsilon=1e-5)

        fused = Sequential(rms).build(device)
        fused.launch()

        golden = torch_rms_norm(t["torch_input"].float(), t["torch_weights"][0].float())
        check_pcc(golden, fused.output_tensors[0], pcc=0.99, label="single op")


# ===========================================================================
# TestShardedExecution
# ===========================================================================


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestShardedExecution:
    """Sharded (L1) execution tests."""

    @pytest.mark.parametrize(
        "shard_type",
        [
            pytest.param("block", id="block"),
            pytest.param("width", id="width"),
        ],
    )
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

        fused = Sequential(ln, rms).build(device)
        fused.launch()

        golden = rms_norm_golden(sh_ln_golden(torch_input, weight=torch_w), torch_w)
        check_pcc(golden, fused.output_tensors[0], label=f"sharded {shard_type}")

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

        fused = Sequential(ln1, rms, ln2).build(device)
        fused.launch()

        g = sh_ln_golden(torch_input, weight=torch_ws[0])
        g = rms_norm_golden(g, torch_ws[1])
        golden = sh_ln_golden(g, weight=torch_ws[2])
        check_pcc(golden, fused.output_tensors[0], label="sharded 3-phase")

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

        fused = Sequential(ln, rms).build(device)
        fused.launch()

        golden = rms_norm_golden(
            sh_ln_golden(torch_input, residual=torch_residual, weight=torch_w, bias=torch_bias), torch_w
        )
        check_pcc(golden, fused.output_tensors[0], label="sharded bias+residual")


# ===========================================================================
# TestMatmulFusion
# ===========================================================================


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
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

        fused = Sequential(*ops).build(device)
        fused.launch()

        # Golden
        g = torch_a.float()
        for token in ordering.split("_"):
            if token == "mm":
                g = g @ torch_b.float()
            elif token == "rms":
                g = torch_rms_norm(g, torch_w.float())
            elif token == "ln":
                g = torch_layer_norm(g, torch_w.float(), torch_bias.float())

        check_pcc(g, fused.output_tensors[0], pcc=0.97, label=ordering)

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

        fused = Sequential(rms1, mm, rms2).build(device)
        fused.launch()

        golden = torch_rms_norm(torch_rms_norm(torch_input.float(), torch_w.float()) @ torch_b.float(), torch_w.float())
        check_pcc(golden, fused.output_tensors[0], label="multicore RMS->MM->RMS")

    @pytest.mark.parametrize("num_rms", [2, 3, 4])
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

        fused = Sequential(*ops).build(device)
        fused.launch()

        g = torch_a.float() @ torch_b.float()
        for _ in range(num_rms):
            g = torch_rms_norm(g, torch_w.float())
        check_pcc(g, fused.output_tensors[0], pcc=0.97, label=f"MM->{num_rms}xRMS")

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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestBranchingTopology:
    """Branching (tree) topology tests, including slice ops."""

    def test_two_branch_split(self, device, multi_tensors):
        """Stem(8c) -> 2 branches(4c each)."""
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = multi_tensors
        stem = rms_norm.rms_norm(
            t["tt_input"], core_range_set=cores(0, 0, 7, 0), weight=t["tt_weights"][0], epsilon=1e-5
        )
        a = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(0, 0, 3, 0), weight=t["tt_weights"][1], epsilon=1e-5
        )
        b = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(4, 0, 7, 0), weight=t["tt_weights"][2], epsilon=1e-5
        )

        fused = Sequential(stem, Parallel(a, b)).build(device)
        fused.launch()

        g_stem = torch_rms_norm(t["torch_input"].float(), t["torch_weights"][0].float())
        check_pcc(torch_rms_norm(g_stem, t["torch_weights"][1].float()), fused.output_tensors[0], label="branch A")
        check_pcc(torch_rms_norm(g_stem, t["torch_weights"][2].float()), fused.output_tensors[1], label="branch B")

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

        fused = Sequential(stem, Parallel(branch_a, branch_b, branch_c)).build(device)
        fused.launch()

        g_stem = torch_rms_norm(torch_input.float(), torch_w.float())
        check_pcc(torch_rms_norm(g_stem, torch_w.float()), branch_a.output_tensors[0], pcc=0.97, label="A (RMS)")
        check_pcc(g_stem[:, :, :, :64], branch_b.output_tensors[0], pcc=0.97, label="B (slice)")
        check_pcc(torch_rms_norm(g_stem, torch_w.float()), branch_c.output_tensors[0], pcc=0.97, label="C (RMS)")

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
        ).build(device)
        fused.launch()

        g_stem = torch_rms_norm(torch_input.float(), ws[0].float())
        g_a = torch_rms_norm(g_stem, ws[1].float())
        check_pcc(torch_rms_norm(g_a, ws[2].float()), a1.output_tensors[0], pcc=0.97, label="A1 (RMS)")
        check_pcc(g_a[:, :, :, :64], a2_slice.output_tensors[0], pcc=0.97, label="A2 (slice)")
        check_pcc(torch_rms_norm(g_stem, ws[3].float()), b.output_tensors[0], pcc=0.97, label="B (RMS)")

    def test_symmetric_binary_tree(self, device, multi_tensors):
        """Stem -> 2 mid -> 4 leaves.

        Tree (8 cores):
            RMS [0-7] -> RMS [0-3] -> RMS [0-1] / RMS [2-3]
                      -> RMS [4-7] -> RMS [4-5] / RMS [6-7]
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = multi_tensors
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
        ).build(device)
        fused.launch()

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
            check_pcc(goldens[i], fused.output_tensors[i], label=label)

    def test_asymmetric_deep_left(self, device, multi_tensors):
        """Deep left + shallow right.

        Tree (8 cores):
            RMS [0-7] -> RMS [0-3] -> RMS [0-1] -> RMS [0-1]
                                    -> RMS [2-3]
                      -> RMS [4-7]
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = multi_tensors
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
        ).build(device)
        fused.launch()

        ws = t["torch_weights"]
        g_root = torch_rms_norm(t["torch_input"].float(), ws[0].float())
        g_left = torch_rms_norm(g_root, ws[1].float())
        goldens = [
            torch_rms_norm(torch_rms_norm(g_left, ws[2].float()), ws[3].float()),
            torch_rms_norm(g_left, ws[4].float()),
            torch_rms_norm(g_root, ws[5].float()),
        ]
        for i, label in enumerate(["LL(deep)", "LR", "Right"]):
            check_pcc(goldens[i], fused.output_tensors[i], label=label)

    def test_overlapping_branches_error(self, device, multi_tensors):
        """Overlapping branch core ranges should raise ValueError."""
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = multi_tensors
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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestParallelExecution:
    """Independent parallel execution tests."""

    @pytest.mark.parametrize("n_chains", [2, 4])
    def test_parallel_chains(self, device, test_tensors, n_chains):
        """N independent fused LN->RMS chains on separate single cores."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm

        t = test_tensors
        ln_compute = ttnn.layernorm_default_compute_config(device.arch())

        torch_inputs = [torch.randn_like(t["torch_input"]) for _ in range(n_chains)]
        fused_ops = []
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
            fused_ops.append(Sequential(ln, rms).build(device))

        for op in fused_ops:
            op.launch()

        for i in range(n_chains):
            golden = torch_rms_norm(
                torch_layer_norm(torch_inputs[i].float(), t["torch_weights"][0].float()),
                t["torch_weights"][1].float(),
            )
            check_pcc(golden, fused_ops[i].output_tensors[0], label=f"chain {i}")

    def test_matmul_plus_fused_chain(self, device, test_tensors):
        """Matmul + 3-phase norm chain on disjoint cores."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc

        t = test_tensors
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
        fused = Sequential(ln1, rms, ln2).build(device)

        mm.launch()
        fused.launch()

        check_pcc(torch_a @ torch_b, mm.output_tensors[0], pcc=0.99, label="matmul")

        g = torch_layer_norm(t["torch_input"].float(), t["torch_weights"][0].float(), t["torch_biases"][0].float())
        g = torch_rms_norm(g, t["torch_weights"][1].float())
        golden = torch_layer_norm(g, t["torch_weights"][2].float(), t["torch_biases"][1].float())
        check_pcc(golden, fused.output_tensors[0], label="3-phase chain")

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
        tree_a = Sequential(stem_a, Parallel(a_left, a_right)).build(device)

        # Tree B
        stem_b = rms_norm.rms_norm(
            tt(torch_input_b, device), core_range_set=cores(4, 0, 7, 0), weight=tt(torch_w, device), epsilon=1e-5
        )
        b_left = rms_norm.rms_norm(
            stem_b.output_tensors[0], core_range_set=cores(4, 0, 5, 0), weight=tt(torch_w, device), epsilon=1e-5
        )
        b_right = slice_desc(stem_b.output_tensors[0], [0, 0, 0, 0], [1, 1, 128, 64], core_range_set=cores(6, 0, 7, 0))
        tree_b = Sequential(stem_b, Parallel(b_left, b_right)).build(device)

        # Launch both in parallel
        tree_a.launch()
        tree_b.launch()

        # Verify tree A (read from individual leaf ops)
        g_stem_a = torch_rms_norm(torch_input_a.float(), torch_w.float())
        check_pcc(g_stem_a[:, :, :, :64], a_left.output_tensors[0], pcc=0.97, label="tree_a slice")
        check_pcc(torch_rms_norm(g_stem_a, torch_w.float()), a_right.output_tensors[0], pcc=0.97, label="tree_a RMS")

        # Verify tree B (read from individual leaf ops)
        g_stem_b = torch_rms_norm(torch_input_b.float(), torch_w.float())
        check_pcc(torch_rms_norm(g_stem_b, torch_w.float()), b_left.output_tensors[0], pcc=0.97, label="tree_b RMS")
        check_pcc(g_stem_b[:, :, :, :64], b_right.output_tensors[0], pcc=0.97, label="tree_b slice")

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
        chain1 = Sequential(ln1, rms1).build(device)

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
        chain2 = Sequential(rms2, ln2).build(device)

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
        chain3 = Sequential(rms3a, mm3, rms3c).build(device)

        # Branching tree
        torch_stem_in = torch.randn(1, 1, 512, hidden, dtype=torch.bfloat16)
        stem_op = rms_norm.rms_norm(
            tt(torch_stem_in, device), core_range_set=cores(0, 4, 7, 5), weight=tt_w, epsilon=1e-5
        )
        br_a = rms_norm.rms_norm(stem_op.output_tensors[0], core_range_set=cores(0, 4, 3, 5), weight=tt_w, epsilon=1e-5)
        br_b = rms_norm.rms_norm(stem_op.output_tensors[0], core_range_set=cores(4, 4, 7, 5), weight=tt_w, epsilon=1e-5)
        tree = Sequential(stem_op, Parallel(br_a, br_b)).build(device)

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
        chain1.launch()
        chain2.launch()
        chain3.launch()
        tree.launch()
        mm2.launch()
        single.launch()

        w, b = torch_w.float(), torch_b.float()
        check_pcc(torch_mm1_a @ torch_mm1_b, mm1.output_tensors[0], pcc=0.99, label="MM1")
        check_pcc(
            torch_rms_norm(torch_layer_norm(torch_norms[0].float(), w), w), chain1.output_tensors[0], label="LN->RMS"
        )
        check_pcc(
            torch_layer_norm(torch_rms_norm(torch_norms[1].float(), w), w, b), chain2.output_tensors[0], label="RMS->LN"
        )
        check_pcc(
            torch_rms_norm(torch_rms_norm(torch_norms[2].float(), w) @ torch_mm3_b.float(), w),
            chain3.output_tensors[0],
            label="RMS->MM->RMS",
        )

        g_stem = torch_rms_norm(torch_stem_in.float(), w)
        check_pcc(torch_rms_norm(g_stem, w), tree.output_tensors[0], label="tree A")
        check_pcc(torch_rms_norm(g_stem, w), tree.output_tensors[1], label="tree B")

        check_pcc(torch_mm2_a @ torch_mm2_b, mm2.output_tensors[0], pcc=0.99, label="MM2")
        check_pcc(torch_rms_norm(torch_norms[3].float(), w), single.output_tensors[0], pcc=0.99, label="single RMS")


# ===========================================================================
# TestSequentialParallelAPI
# ===========================================================================


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestSequentialParallelAPI:
    """API surface tests for Sequential/Parallel."""

    def test_sequential_inline(self, device, test_tensors):
        """Sequential(rms, rms).build().launch()."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = test_tensors
        cr = cores(0, 0)
        rms1 = rms_norm.rms_norm(t["tt_input"], core_range_set=cr, weight=t["tt_weights"][0], epsilon=1e-5)
        rms2 = rms_norm.rms_norm(rms1.output_tensors[0], core_range_set=cr, weight=t["tt_weights"][1], epsilon=1e-5)

        fused = Sequential(rms1, rms2).build(device)
        fused.launch()

        golden = torch_rms_norm(
            torch_rms_norm(t["torch_input"].float(), t["torch_weights"][0].float()),
            t["torch_weights"][1].float(),
        )
        check_pcc(golden, fused.output_tensors[0], pcc=0.99, label="inline")

    def test_sequential_add_method(self, device, test_tensors):
        """s.add(op) matches inline construction."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = test_tensors
        cr = cores(0, 0)
        rms1 = rms_norm.rms_norm(t["tt_input"], core_range_set=cr, weight=t["tt_weights"][0], epsilon=1e-5)
        rms2 = rms_norm.rms_norm(rms1.output_tensors[0], core_range_set=cr, weight=t["tt_weights"][1], epsilon=1e-5)

        s = Sequential(rms1)
        s.add(rms2)
        fused = s.build(device)
        fused.launch()

        golden = torch_rms_norm(
            torch_rms_norm(t["torch_input"].float(), t["torch_weights"][0].float()),
            t["torch_weights"][1].float(),
        )
        check_pcc(golden, fused.output_tensors[0], pcc=0.99, label="add method")

    def test_sequential_branching(self, device, multi_tensors):
        """Sequential(stem, Parallel(a, b)).build() — API-level branching."""
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = multi_tensors
        stem = rms_norm.rms_norm(
            t["tt_input"], core_range_set=cores(0, 0, 3, 0), weight=t["tt_weights"][0], epsilon=1e-5
        )
        a = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(0, 0, 1, 0), weight=t["tt_weights"][1], epsilon=1e-5
        )
        b = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(2, 0, 3, 0), weight=t["tt_weights"][2], epsilon=1e-5
        )

        fused = Sequential(stem, Parallel(a, b)).build(device)
        fused.launch()

        g_stem = torch_rms_norm(t["torch_input"].float(), t["torch_weights"][0].float())
        check_pcc(torch_rms_norm(g_stem, t["torch_weights"][1].float()), fused.output_tensors[0], label="A")
        check_pcc(torch_rms_norm(g_stem, t["torch_weights"][2].float()), fused.output_tensors[1], label="B")


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
        "untilize": "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/common.cpp",
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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestDocExample:
    """Integration test matching the example in op_fusion.md."""

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
        ).build(device)
        fused.launch()

        # Golden
        g1 = torch.matmul(torch_a.float(), torch_b1.float())
        g_left = g1[:, :, :, :128]
        g_right = g1[:, :, :, 128:]

        check_pcc(torch.matmul(g_left, torch_b4.float()), op4.output_tensors[0], pcc=0.97, label="op4 matmul")
        check_pcc(
            torch_layer_norm(g_left, torch_ln_w.float(), torch_ln_bias.float()),
            op5.output_tensors[0],
            pcc=0.97,
            label="op5 LN",
        )
        check_pcc(torch_rms_norm(g_right, torch_rms_w.float()), op6.output_tensors[0], pcc=0.97, label="op6 RMS")


# ===========================================================================
# TestDeepSeekV3
# ===========================================================================


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestDeepSeekV3:
    """DeepSeek V3 MLA block patterns using Parallel fusion."""

    def test_q_kv_rms_norm(self, device):
        """Parallel Q/KV RMS norms from the DeepSeek V3 MLA block.

        Q norm: 16 cores (0,0)-(3,3), width-sharded, shard [32, 96], total width 1536
        KV norm: 16 cores (5,0)-(6,7), width-sharded, shard [32, 32], total width 512
        """
        from models.experimental.ops.descriptors.fusion import Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        torch.manual_seed(42)

        # --- Q norm setup ---
        q_cores = cores(0, 0, 3, 3)  # 4x4 = 16 cores
        q_shard_w = 96
        q_total_w = 16 * q_shard_w  # 1536

        torch_q_input = torch.rand(1, 1, 32, q_total_w, dtype=torch.bfloat16)
        torch_q_weight = torch.rand(1, 1, 1, q_total_w, dtype=torch.bfloat16)

        q_shard_spec = ttnn.ShardSpec(q_cores, [32, q_shard_w], ttnn.ShardOrientation.ROW_MAJOR)
        q_mem = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=q_shard_spec,
        )
        q_pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(4, 4),
            subblock_w=q_shard_w // 32,
            block_h=1,
            block_w=q_shard_w // 32,
            inplace=False,
        )

        tt_q_input = ttnn.from_torch(torch_q_input, device=device, layout=ttnn.TILE_LAYOUT, memory_config=q_mem)
        tt_q_weight = ttnn.from_torch(torch_q_weight, device=device, layout=ttnn.TILE_LAYOUT)

        q_branch = rms_norm.rms_norm(
            tt_q_input,
            epsilon=1e-5,
            weight=tt_q_weight,
            memory_config=q_mem,
            core_range_set=q_cores,
            program_config=q_pc,
        )

        # --- KV norm setup ---
        kv_cores = cores(5, 0, 6, 7)  # 2x8 = 16 cores
        kv_shard_w = 32
        kv_total_w = 16 * kv_shard_w  # 512

        torch.manual_seed(123)
        torch_kv_input = torch.rand(1, 1, 32, kv_total_w, dtype=torch.bfloat16)
        torch_kv_weight = torch.rand(1, 1, 1, kv_total_w, dtype=torch.bfloat16)

        kv_shard_spec = ttnn.ShardSpec(kv_cores, [32, kv_shard_w], ttnn.ShardOrientation.ROW_MAJOR)
        kv_mem = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=kv_shard_spec,
        )
        kv_pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(2, 8),
            subblock_w=kv_shard_w // 32,
            block_h=1,
            block_w=kv_shard_w // 32,
            inplace=False,
        )

        tt_kv_input = ttnn.from_torch(torch_kv_input, device=device, layout=ttnn.TILE_LAYOUT, memory_config=kv_mem)
        tt_kv_weight = ttnn.from_torch(torch_kv_weight, device=device, layout=ttnn.TILE_LAYOUT)

        kv_branch = rms_norm.rms_norm(
            tt_kv_input,
            epsilon=1e-5,
            weight=tt_kv_weight,
            memory_config=kv_mem,
            core_range_set=kv_cores,
            program_config=kv_pc,
        )

        # --- Build and launch in parallel ---
        fused = Parallel(q_branch, kv_branch).build()
        fused.launch()

        # --- Verify ---
        check_pcc(
            torch_rms_norm(torch_q_input.float(), torch_q_weight.float()),
            q_branch.output_tensors[0],
            pcc=0.98,
            label="Q norm",
        )
        check_pcc(
            torch_rms_norm(torch_kv_input.float(), torch_kv_weight.float()),
            kv_branch.output_tensors[0],
            pcc=0.98,
            label="KV norm",
        )


# ===========================================================================
# TestAsymmetricBarrier
# ===========================================================================


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestAsymmetricBarrier:
    """Tests for asymmetric barrier (narrow→wide) topologies.

    In narrow→wide topologies, extra cores in the wide phase get _NOOP_OP
    placeholder phases and must use SyncMode::WaitOnly (skip arrive, only
    wait on release).  These tests exercise edge cases that could cause hangs
    if the arrive threshold or sync mode dispatch is wrong.
    """

    def test_narrow_stem_wide_branches(self, device, multi_tensors):
        """Narrow stem (2 cores) → wide branches (4+4 cores).

        6 cores get _NOOP_OP for the stem phase.  arrive_threshold=2 at
        the stem→branch barrier.  If all 8 cores arrive, threshold would
        be wrong → hang or early release.
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = multi_tensors
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

        fused = Sequential(stem, Parallel(a, b)).build(device)
        fused.launch()

        g_stem = torch_rms_norm(t["torch_input"].float(), t["torch_weights"][0].float())
        check_pcc(torch_rms_norm(g_stem, t["torch_weights"][1].float()), fused.output_tensors[0], label="branch A")
        check_pcc(torch_rms_norm(g_stem, t["torch_weights"][2].float()), fused.output_tensors[1], label="branch B")

    def test_single_core_stem_wide_branches(self, device, multi_tensors):
        """Extreme asymmetry: 1-core stem → 4+4 core branches.

        7 cores get _NOOP_OP.  arrive_threshold=1, release_cores=8.
        Tests the most extreme arrive/release ratio.
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = multi_tensors
        stem = rms_norm.rms_norm(t["tt_input"], core_range_set=cores(0, 0), weight=t["tt_weights"][0], epsilon=1e-5)
        a = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(0, 0, 3, 0), weight=t["tt_weights"][1], epsilon=1e-5
        )
        b = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(4, 0, 7, 0), weight=t["tt_weights"][2], epsilon=1e-5
        )

        fused = Sequential(stem, Parallel(a, b)).build(device)
        fused.launch()

        g_stem = torch_rms_norm(t["torch_input"].float(), t["torch_weights"][0].float())
        check_pcc(torch_rms_norm(g_stem, t["torch_weights"][1].float()), fused.output_tensors[0], label="branch A")
        check_pcc(torch_rms_norm(g_stem, t["torch_weights"][2].float()), fused.output_tensors[1], label="branch B")

    def test_narrow_wide_repeated_execution(self, device, multi_tensors):
        """Narrow→wide with repeated execution to catch stale semaphores.

        Re-execution is the most common source of barrier bugs: stale
        arrive/release semaphore values from run N can deadlock run N+1
        if the threshold or reset is wrong.
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = multi_tensors
        stem = rms_norm.rms_norm(
            t["tt_input"], core_range_set=cores(0, 0, 1, 0), weight=t["tt_weights"][0], epsilon=1e-5
        )
        a = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(0, 0, 3, 0), weight=t["tt_weights"][1], epsilon=1e-5
        )
        b = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(4, 0, 7, 0), weight=t["tt_weights"][2], epsilon=1e-5
        )

        fused = Sequential(stem, Parallel(a, b)).build(device)

        g_stem = torch_rms_norm(t["torch_input"].float(), t["torch_weights"][0].float())

        for run in range(5):
            fused.launch()
            check_pcc(
                torch_rms_norm(g_stem, t["torch_weights"][1].float()),
                fused.output_tensors[0],
                label=f"branch A run {run}",
            )
            check_pcc(
                torch_rms_norm(g_stem, t["torch_weights"][2].float()),
                fused.output_tensors[1],
                label=f"branch B run {run}",
            )

    def test_multi_level_narrow_wide(self, device, multi_tensors):
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

        t = multi_tensors
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
        ).build(device)
        fused.launch()

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
            check_pcc(goldens[i], fused.output_tensors[i], label=label)

    def test_narrow_deep_left_wide_right(self, device, multi_tensors):
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

        t = multi_tensors
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
        ).build(device)
        fused.launch()

        ws = t["torch_weights"]
        g_stem = torch_rms_norm(t["torch_input"].float(), ws[0].float())
        check_pcc(
            torch_rms_norm(torch_rms_norm(g_stem, ws[1].float()), ws[2].float()),
            fused.output_tensors[0],
            label="deep left",
        )
        check_pcc(torch_rms_norm(g_stem, ws[3].float()), fused.output_tensors[1], label="wide right")

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

        fused = Sequential(stem, Parallel(branch_a, branch_b)).build(device)
        fused.launch()

        g_stem = torch_rms_norm(torch_input.float(), torch_w0.float())
        check_pcc(g_stem[:, :, :, :64], branch_a.output_tensors[0], pcc=0.97, label="A (slice)")
        check_pcc(torch_rms_norm(g_stem, torch_w1.float()), branch_b.output_tensors[0], label="B (RMS)")

    def test_fully_disjoint_parent_children(self, device, multi_tensors):
        """Fully disjoint: parent {0,1} → children {2-3, 4-7}.

        Parent cores share zero overlap with any child core.  Parent
        cores get _NOOP_OP exit phase, arrive at barrier, then exit.
        Child cores get _NOOP_OP wait phase, then run real work.
        This would deadlock without the exit-phase fix.
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = multi_tensors
        stem = rms_norm.rms_norm(
            t["tt_input"], core_range_set=cores(0, 0, 1, 0), weight=t["tt_weights"][0], epsilon=1e-5
        )
        a = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(2, 0, 3, 0), weight=t["tt_weights"][1], epsilon=1e-5
        )
        b = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(4, 0, 7, 0), weight=t["tt_weights"][2], epsilon=1e-5
        )

        fused = Sequential(stem, Parallel(a, b)).build(device)
        fused.launch()

        g_stem = torch_rms_norm(t["torch_input"].float(), t["torch_weights"][0].float())
        check_pcc(torch_rms_norm(g_stem, t["torch_weights"][1].float()), fused.output_tensors[0], label="branch A")
        check_pcc(torch_rms_norm(g_stem, t["torch_weights"][2].float()), fused.output_tensors[1], label="branch B")

    def test_partial_disjoint_one_core_exits(self, device, multi_tensors):
        """Partial disjoint: parent {0,1} → child {1-7}.

        Core 0 does real work in parent but isn't in any child — it gets
        an exit phase.  Core 1 overlaps and continues normally.  Cores
        2-7 get noop for parent.
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = multi_tensors
        stem = rms_norm.rms_norm(
            t["tt_input"], core_range_set=cores(0, 0, 1, 0), weight=t["tt_weights"][0], epsilon=1e-5
        )
        a = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(1, 0, 3, 0), weight=t["tt_weights"][1], epsilon=1e-5
        )
        b = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(4, 0, 7, 0), weight=t["tt_weights"][2], epsilon=1e-5
        )

        fused = Sequential(stem, Parallel(a, b)).build(device)
        fused.launch()

        g_stem = torch_rms_norm(t["torch_input"].float(), t["torch_weights"][0].float())
        check_pcc(torch_rms_norm(g_stem, t["torch_weights"][1].float()), fused.output_tensors[0], label="branch A")
        check_pcc(torch_rms_norm(g_stem, t["torch_weights"][2].float()), fused.output_tensors[1], label="branch B")

    def test_disjoint_repeated_execution(self, device, multi_tensors):
        """Fully disjoint with repeated execution to catch stale semaphores."""
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = multi_tensors
        stem = rms_norm.rms_norm(
            t["tt_input"], core_range_set=cores(0, 0, 1, 0), weight=t["tt_weights"][0], epsilon=1e-5
        )
        a = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(2, 0, 3, 0), weight=t["tt_weights"][1], epsilon=1e-5
        )
        b = rms_norm.rms_norm(
            stem.output_tensors[0], core_range_set=cores(4, 0, 7, 0), weight=t["tt_weights"][2], epsilon=1e-5
        )

        fused = Sequential(stem, Parallel(a, b)).build(device)

        g_stem = torch_rms_norm(t["torch_input"].float(), t["torch_weights"][0].float())

        for run in range(5):
            fused.launch()
            check_pcc(
                torch_rms_norm(g_stem, t["torch_weights"][1].float()),
                fused.output_tensors[0],
                label=f"branch A run {run}",
            )
            check_pcc(
                torch_rms_norm(g_stem, t["torch_weights"][2].float()),
                fused.output_tensors[1],
                label=f"branch B run {run}",
            )
