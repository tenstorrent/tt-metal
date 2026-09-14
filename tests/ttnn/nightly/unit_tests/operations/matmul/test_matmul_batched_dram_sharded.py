# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Nightly coverage for MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory.

The Sanity-tier tests in tests/ttnn/unit_tests/operations/matmul/test_matmul_deepseek.py pin this
factory's default path and a compact cross-section of its compute-path branches. This file widens
that cross-section along the axes that are too expensive to run per-PR: more shapes, the full
intermediate-buffer matrix (output dtype x dest accumulation x packer L1 accumulation), the
remaining fused activations, and deeper inner-dim blocking.

What is deliberately *not* covered here, because no caller can reach it:

  * The FUSE_BIAS branch. ttnn.matmul / ttnn.linear route bias to a post-processed add() whenever
    in1 is batched (ttnn/cpp/ttnn/operations/matmul/matmul.cpp, "Fused matmul+bias does not support
    batched weights"), and this factory's in1 is always batched - it is batch-sharded across DRAM
    banks. The bias buffer, its compile-time args and the compute kernel's whole FUSE_BIAS section
    are dead unless the device op is invoked directly.
  * untilize_out. Read only from MatmulMultiCoreReuseMultiCast1DProgramConfig, so it is always
    false for this factory's program config.
  * skip_compute / skip_write_back. Hardcoded false at the factory's only call site.

Adding tests for those would require either an owner decision or a device-op-level harness, so they
are recorded here rather than papered over with a test that cannot run.
"""

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.matmul.test_matmul_deepseek import run_batched_dram_sharded_matmul


# (name, batch, m, k, n). k is the contracted dimension; with a 32-wide tile, K in tiles is k / 32,
# which bounds how far the inner dimension can be split.
SHAPES = [
    ("small_k4", 12, 32, 128, 64),
    ("wide_n_k4", 24, 32, 128, 256),
    ("deep_k8", 12, 32, 256, 64),
]
SHAPE_IDS = [s[0] for s in SHAPES]


ACTIVATIONS = [
    (None, None),
    (ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), torch.relu),
    (ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU), torch.nn.functional.silu),
    (ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH), lambda t: torch.nn.functional.gelu(t, approximate="tanh")),
]
ACTIVATION_IDS = ["no_activation", "relu_packer", "silu_sfpu", "gelu_tanh_sfpu"]


@pytest.mark.parametrize("fused_activation, torch_activation", ACTIVATIONS, ids=ACTIVATION_IDS)
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("num_k_blocks", [1, 2], ids=["one_k_block", "two_k_blocks"])
def test_matmul_batched_dram_sharded_activations(device, shape, fused_activation, torch_activation, num_k_blocks):
    """Every fused activation, across shapes and with the inner-dim loop both off and on.

    RELU is folded into the packer (PACK_RELU); the others are SFPU ops whose type and parameters
    travel as compile-time args, so the two kinds fail in different ways and both need exercising.
    """
    _name, batch, m, k, n = shape

    run_batched_dram_sharded_matmul(
        device,
        batch=batch,
        m=m,
        k=k,
        n=n,
        in0_dtype=ttnn.bfloat16,
        in1_dtype=ttnn.bfloat8_b,
        out_dtype=ttnn.bfloat16,
        fused_activation=fused_activation,
        torch_activation=torch_activation,
        num_k_blocks=num_k_blocks,
        expected_pcc=0.99,
    )


@pytest.mark.parametrize(
    "out_dtype", [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32], ids=["out_bf16", "out_bfp8", "out_fp32"]
)
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["fp32_acc_off", "fp32_acc_on"])
@pytest.mark.parametrize("packer_l1_acc", [False, True], ids=["packer_acc_off", "packer_acc_on"])
@pytest.mark.parametrize("num_k_blocks", [1, 2], ids=["one_k_block", "two_k_blocks"])
def test_matmul_batched_dram_sharded_intermediate_buffer(
    device, out_dtype, fp32_dest_acc_en, packer_l1_acc, num_k_blocks
):
    """The intermediate-buffer matrix.

    The factory aliases intermed0 onto the output buffer only when their data formats match, and the
    intermediate format is derived from (packer_l1_acc && num_blocks > 1, fp32_dest_acc_en, output
    format). This parametrization walks that derivation end to end, so both the aliased clique and
    the separately sized and bound pair of buffers are exercised against every output dtype.
    """
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
    )

    run_batched_dram_sharded_matmul(
        device,
        batch=12,
        m=32,
        k=128,
        n=64,
        in0_dtype=ttnn.bfloat16,
        in1_dtype=ttnn.bfloat8_b,
        out_dtype=out_dtype,
        fused_activation=None,
        torch_activation=None,
        num_k_blocks=num_k_blocks,
        # A bfloat8_b output quantizes the result on the way out, which costs more accuracy than the
        # bfloat8_b weights alone.
        expected_pcc=0.98 if out_dtype == ttnn.bfloat8_b else 0.99,
        compute_kernel_config=compute_kernel_config,
    )


# Deep inner-dim blocking. These are the only cells in CI that exercise more than two accumulation
# passes. Four of them used to be wrong: packer L1 accumulation adds into intermed0 in place, which
# only lands on the same addresses each block when that buffer holds exactly one block of output
# tiles, and sharing storage with a multi-batch output shard made it a whole multiple of that. The
# factory no longer shares the two buffers in that case.
@pytest.mark.parametrize("num_k_blocks", [1, 2, 4, 8], ids=["k_blocks_1", "k_blocks_2", "k_blocks_4", "k_blocks_8"])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["fp32_acc_off", "fp32_acc_on"])
def test_matmul_batched_dram_sharded_k_blocking(device, num_k_blocks, fp32_dest_acc_en):
    """Deep inner-dim blocking, where the accumulation loop actually accumulates.

    K is 8 tiles here, so the block count sweeps from one pass to eight. This is the axis the
    committed Sanity tests could not reach at all: they set in0_block_w == K, so the loop the
    intermediate buffer exists to serve ran exactly once and never accumulated across iterations.
    """
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=True,
    )

    run_batched_dram_sharded_matmul(
        device,
        batch=12,
        m=32,
        k=256,  # 8 tiles on the contracted dimension
        n=64,
        in0_dtype=ttnn.bfloat16,
        in1_dtype=ttnn.bfloat8_b,
        out_dtype=ttnn.bfloat16,
        fused_activation=None,
        torch_activation=None,
        num_k_blocks=num_k_blocks,
        expected_pcc=0.99,
        compute_kernel_config=compute_kernel_config,
    )
