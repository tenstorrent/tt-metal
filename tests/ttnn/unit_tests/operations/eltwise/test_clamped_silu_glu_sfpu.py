# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone device test for the fused clamped_silu_glu binary SFPU op (api/compute/clamped_silu_glu.h).

Drives the op through ttnn.generic_op with a minimal binary test kernel (no
production op wired), reaching both dst-accumulator modes, and compares against
the torch reference:

    gate_c = min(gate, limit)
    up_c   = clamp(up, -limit, limit)
    result = gate_c * sigmoid(gate_c) * up_c

This is DeepSeek-V4's routed-expert activation. Testing it here rather than only through
the fused expert FFN is what makes the sigmoid's vConstFloatPrgm0 requirement (see
ckernel_sfpu_clamped_silu_glu.h) fail against a direct golden instead of as an unexplained
PCC drop three matmuls downstream.
"""

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp
from models.common.utility_functions import is_blackhole, ulp

LIMIT = 10.0  # DeepSeek-V4 swiglu_limit, baked as ClampedSiluGluConfigDsV4
TILE_ELEMS = 32 * 32

# (ttnn dtype, tile page bytes). bfp8_b: 1 mantissa byte/datum + 1 shared exp byte / 16.
IN_DTYPES = {
    "bf16": (ttnn.bfloat16, TILE_ELEMS * 2),
    "bfp8_b": (ttnn.bfloat8_b, TILE_ELEMS + TILE_ELEMS // 16),
}
OUT_PAGE_BYTES = TILE_ELEMS * 2  # the output CB is bf16

# The result reaches host as bf16 either way -- rounded inside the op under bf16 dst, by the packer
# under fp32 dst -- so the bf16 arm is gated in ULP. Measured worst case 0.89 (bf16 dst) / 0.50
# (fp32 dst); the bound carries margin over that rather than being tuned to the edge, because the
# op's own budget is close to a ULP by construction: _sfpu_exp_21f_bf16_ rounds to bf16 internally
# and the deep-negative gate tail passes that straight through, then the result is rounded again.
# test_clamped_silu_glu_init_is_required, not this bound, is what holds the init in place.
#
# The bfp8_b arms carry no accuracy signal -- input quantization alone puts them past 300 ULP, so
# they gate data-format plumbing via PCC only and do not back up the bf16 bound.
BF16_ULP = 2
BF16_PCC = 0.999
BFP8_PCC = 0.99


def _max_ulp(actual, golden):
    """Worst-case error on the scale assert_with_ulp gates: |actual - golden| / ULP(golden), with
    ULP taken at bf16 spacing because that is the precision the op delivers."""
    return ((actual - golden).abs() / ulp(golden.to(torch.bfloat16)).to(torch.float32)).max().item()


def clamped_silu_glu_reference(gate, up):
    g = gate.to(torch.float32)
    u = up.to(torch.float32)
    gate_c = torch.clamp(g, max=LIMIT)
    up_c = torch.clamp(u, min=-LIMIT, max=LIMIT)
    return gate_c * torch.sigmoid(gate_c) * up_c


def _coverage_inputs(num_tiles, seed=0):
    """Sweeps that force the gate's upper clamp and BOTH of the up clamp's tails, plus the
    SiLU tail on the unclamped side, shuffled so every tile spans the range. gate and up are
    permuted independently over different endpoints so they do not correlate.
    """
    n = num_tiles * TILE_ELEMS
    torch.manual_seed(seed)
    gate = torch.cat([torch.linspace(-3 * LIMIT, 3 * LIMIT, n // 2), torch.randn(n - n // 2) * LIMIT])
    up = torch.cat([torch.linspace(-2 * LIMIT, 4 * LIMIT, n // 2), torch.randn(n - n // 2) * LIMIT])
    return gate[torch.randperm(n)].to(torch.bfloat16), up[torch.randperm(n)].to(torch.bfloat16)


def _run(device, gate_t, up_t, in_dtype, page_bytes, fp32_dest, dst_gate=0, dst_up=1, dst_out=0, skip_init=False):
    num_tiles = gate_t.numel() // TILE_ELEMS
    shape = [1, num_tiles, 32, 32]

    gate = ttnn.from_torch(
        gate_t.reshape(shape),
        dtype=in_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    up = ttnn.from_torch(
        up_t.reshape(shape),
        dtype=in_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    cb_gate, cb_up, cb_out = 0, 1, 16

    def cb(idx, fmt, page):
        return ttnn.CBDescriptor(
            total_size=2 * page,
            core_ranges=core,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page)],
        )

    cbs = [
        cb(cb_gate, in_dtype, page_bytes),
        cb(cb_up, in_dtype, page_bytes),
        cb(cb_out, ttnn.bfloat16, OUT_PAGE_BYTES),
    ]

    reader_rt = ttnn.RuntimeArgs()
    reader_rt[0][0] = [gate.buffer_address(), up.buffer_address(), num_tiles, 0]
    writer_rt = ttnn.RuntimeArgs()
    writer_rt[0][0] = [output.buffer_address(), num_tiles, 0]

    reader_cta = (
        ttnn.TensorAccessorArgs(gate).get_compile_time_args() + ttnn.TensorAccessorArgs(up).get_compile_time_args()
    )
    writer_cta = [cb_out] + ttnn.TensorAccessorArgs(output).get_compile_time_args()

    kernels = [
        # The situ_glu reader is a generic two-tensor (gate -> c_0, up -> c_1) reader; only its
        # comment names situ_glu, so it is reused rather than cloned.
        ttnn.KernelDescriptor(
            kernel_source="tests/tt_metal/tt_metal/test_kernels/dataflow/reader_situ_glu.cpp",
            core_ranges=core,
            compile_time_args=reader_cta,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
            core_ranges=core,
            compile_time_args=writer_cta,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source="tests/tt_metal/tt_metal/test_kernels/compute/clamped_silu_glu.cpp",
            core_ranges=core,
            compile_time_args=[num_tiles, dst_gate, dst_up, dst_out, int(skip_init)],
            runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=fp32_dest),
        ),
    ]

    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    out = ttnn.generic_op([gate, up, output], program)
    return ttnn.to_torch(out).reshape(gate_t.shape)


@pytest.mark.skipif(not is_blackhole(), reason="clamped_silu_glu SFPU op is implemented for Blackhole only")
@pytest.mark.parametrize("in_name", list(IN_DTYPES), ids=list(IN_DTYPES))
@pytest.mark.parametrize("fp32_dest", [False, True], ids=["bf16_dst", "fp32_dst"])
# (gate, up, out) dst placements. out aliasing gate is what the expert kernel does to save a slot;
# a separate out slot catches a kernel that ignores out_tile_idx; and "production" mirrors the
# fused call BINARY_ACT_TILE(j, c + j, j) -- a non-zero gate index and a gate->up stride that is
# not 1, which is the shape a kernel with hardcoded input indices would fail on.
#
# Every index must stay under 4: the dst register file holds 8 tiles under bf16 dst but only 4
# under fp32 dest accumulate (fused_swiglu.cpp derives the same kDstCapacity), and both dst modes
# are parametrized here.
@pytest.mark.parametrize(
    "dst_gate, dst_up, dst_out",
    [
        pytest.param(0, 1, 0, id="out_aliases_gate"),
        pytest.param(0, 1, 2, id="out_separate"),
        pytest.param(1, 3, 1, id="production"),
    ],
)
def test_clamped_silu_glu_sfpu(device, in_name, fp32_dest, dst_gate, dst_up, dst_out):
    in_dtype, page_bytes = IN_DTYPES[in_name]
    num_tiles = 8
    gate_t, up_t = _coverage_inputs(num_tiles)

    # Every clamp branch must fire, and the up tails must be counted separately -- a one-sided
    # clamp bug is invisible against a golden whose inputs never reach the other side. The up
    # counts are joined with gate > 0 because silu(gate_c) is ~4e-4 where gate is deeply negative,
    # so an up-tail element there is not observable in the output.
    g32, u32 = gate_t.to(torch.float32), up_t.to(torch.float32)
    gate_hi = (g32 > LIMIT).float().mean().item()
    # gate < -LIMIT is the only region where clamping the gate from below too would differ from
    # min(gate, LIMIT), and that mutation is invisible to both PCC gates -- so it is guarded here
    # rather than left to whatever the sweep endpoints happen to produce.
    gate_lo = ((g32 < -LIMIT) & (u32.abs() > 1.0)).float().mean().item()
    up_hi = ((u32 > LIMIT) & (g32 > 0)).float().mean().item()
    up_lo = ((u32 < -LIMIT) & (g32 > 0)).float().mean().item()
    logger.debug(
        f"clamp coverage: gate>{LIMIT}: {gate_hi:.1%}, gate<-{LIMIT}: {gate_lo:.1%}, "
        f"up>{LIMIT}: {up_hi:.1%}, up<-{LIMIT}: {up_lo:.1%} (up counts joint with gate>0)"
    )
    assert min(gate_hi, gate_lo, up_hi, up_lo) > 0.02, "stimulus does not observably reach every clamp branch"

    golden = clamped_silu_glu_reference(gate_t, up_t)
    actual = _run(device, gate_t, up_t, in_dtype, page_bytes, fp32_dest, dst_gate, dst_up, dst_out)

    is_bfp8 = in_name == "bfp8_b"
    # |gate_c * sigmoid(gate_c)| <= limit and |up_c| <= limit -> |result| <= their product.
    bound = LIMIT * LIMIT * (1.0 + (5e-2 if is_bfp8 else 2**-8))
    assert actual.to(torch.float32).abs().max().item() <= bound

    g = golden.to(torch.float32)
    a = actual.to(torch.float32)
    max_ulp = _max_ulp(a, g)
    logger.debug(f"{in_name} fp32_dst={fp32_dest}: max abs err {(a - g).abs().max().item():.4e}, max ULP {max_ulp:.2f}")

    if is_bfp8:
        # bfp8_b inputs quantize before the op runs, so the output carries hundreds of bf16 ULP
        # no matter how accurate the SFPU is; ULP only says something about the bf16 arm.
        assert_with_pcc(g, a, pcc=BFP8_PCC)
    else:
        assert_with_ulp(golden, actual, ulp_threshold=BF16_ULP)
        assert_with_pcc(g, a, pcc=BF16_PCC)


@pytest.mark.skipif(not is_blackhole(), reason="clamped_silu_glu SFPU op is implemented for Blackhole only")
def test_clamped_silu_glu_init_is_required(device):
    """clamped_silu_glu_tile_init() is load-bearing, not ceremony.

    The op's sigmoid reaches sfpu_reciprocal_iter, which reads vConstFloatPrgm0 and needs 2.0f
    there. The test kernel leaves a wrong value in that register, so skipping the init leaves the
    reciprocal's Newton step disabled. Comparing the two runs rather than testing either against a
    fixed bound keeps this independent of BF16_ULP, which is free to carry accuracy margin.
    """
    in_dtype, page_bytes = IN_DTYPES["bf16"]
    gate_t, up_t = _coverage_inputs(8)
    golden = clamped_silu_glu_reference(gate_t, up_t).to(torch.float32)

    with_init = _max_ulp(_run(device, gate_t, up_t, in_dtype, page_bytes, False).to(torch.float32), golden)
    without_init = _max_ulp(
        _run(device, gate_t, up_t, in_dtype, page_bytes, False, skip_init=True).to(torch.float32), golden
    )
    logger.debug(f"max ULP: with init {with_init:.2f}, without init {without_init:.2f}")
    assert without_init > with_init, "skipping clamped_silu_glu_tile_init() did not degrade the result"
