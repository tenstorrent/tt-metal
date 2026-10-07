# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Nightly coverage for MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory.

This file sweeps the factory's compute paths along the axes that are too expensive to run per-PR:
every fused activation across three shapes, the full intermediate-buffer matrix (output dtype x
dest accumulation x packer L1 accumulation x batches per core), and inner-dim blocking down to
single-tile blocks.

Shard depth is expressed as batches per core rather than as a batch, because the batch a given
depth needs is the DRAM bank count times that depth, and the bank count differs per arch (12 on
Wormhole, 8 on Blackhole). run_batched_dram_sharded_matmul derives the batch from the device.

What is deliberately *not* covered here, because no caller can reach it:

  * The FUSE_BIAS branch. ttnn.matmul / ttnn.linear route bias to a post-processed add() whenever
    in1 is batched (ttnn/cpp/ttnn/operations/matmul/matmul.cpp, "Fused matmul+bias does not support
    batched weights"), and this factory's in1 is always batched - it is batch-sharded across DRAM
    banks. The bias buffer, its compile-time args and the compute kernel's whole FUSE_BIAS section
    are dead unless the device op is invoked directly.
  * untilize_out. Read only from MatmulMultiCoreReuseMultiCast1DProgramConfig, so it is always
    false for this factory's program config.
  * skip_compute / skip_write_back. Hardcoded false at the factory's only call site.

"""

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.matmul.test_matmul_deepseek import run_batched_dram_sharded_matmul

# (name, batches_per_core, m, k, n). k is the contracted dimension; with a 32-wide tile, K in tiles
# is k / 32, which bounds how far the inner dimension can be split. The batch itself is derived
# from the device's DRAM bank count, so batches_per_core holds across archs (see the helper).
SHAPES = [
    ("small_k4", 2, 32, 128, 64),
    ("wide_n_k4", 3, 32, 128, 256),
    ("deep_k8", 2, 32, 256, 64),
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
@pytest.mark.parametrize("num_k_blocks", [1, 4], ids=["one_k_block", "four_k_blocks"])
def test_matmul_batched_dram_sharded_activations(device, shape, fused_activation, torch_activation, num_k_blocks):
    """Every fused activation, across shapes and with the inner-dim loop both off and on.

    RELU is folded into the packer (PACK_RELU); the others are SFPU ops whose type and parameters
    travel as compile-time args, so the two kinds fail in different ways and both need exercising.
    """
    _name, batches_per_core, m, k, n = shape

    run_batched_dram_sharded_matmul(
        device,
        batches_per_core=batches_per_core,
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
@pytest.mark.parametrize("num_k_blocks", [1, 4], ids=["one_k_block", "four_k_blocks"])
@pytest.mark.parametrize("batches_per_core", [1, 2], ids=["one_batch_per_core", "two_batches_per_core"])
def test_matmul_batched_dram_sharded_intermediate_buffer(
    device, out_dtype, fp32_dest_acc_en, packer_l1_acc, num_k_blocks, batches_per_core
):
    """The intermediate-buffer matrix.

    The factory aliases intermed0 onto the output buffer only when their data formats match, the
    output shard is a single block of output tiles, and the intermediate format is derived from
    (packer_l1_acc && num_blocks > 1, fp32_dest_acc_en, output format). This parametrization walks
    that derivation end to end, so both the aliased clique and the separately sized and bound pair
    of buffers are exercised against every output dtype.

    batches_per_core is what sizes the output shard against one block: at 1 the two buffers may
    still be aliased under packer L1 accumulation, at 2 they may not, and both sides of that
    condition need exercising.
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
        batches_per_core=batches_per_core,
        m=32,
        k=128,
        n=64,
        in0_dtype=ttnn.bfloat16,
        in1_dtype=ttnn.bfloat8_b,
        out_dtype=out_dtype,
        fused_activation=None,
        torch_activation=None,
        num_k_blocks=num_k_blocks,
        expected_pcc=0.98 if out_dtype == ttnn.bfloat8_b else 0.99,
        compute_kernel_config=compute_kernel_config,
    )


@pytest.mark.parametrize("num_k_blocks", [1, 2, 4, 8], ids=["k_blocks_1", "k_blocks_2", "k_blocks_4", "k_blocks_8"])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["fp32_acc_off", "fp32_acc_on"])
def test_matmul_batched_dram_sharded_k_blocking(device, num_k_blocks, fp32_dest_acc_en):
    """Deep inner-dim blocking, where the accumulation loop actually accumulates.

    K is 8 tiles here, so the block count sweeps from one pass to eight. This is the shape the
    dropped-block bug was found on: two batches per core with packer L1 accumulation on, which
    before the fix aliased intermed0 onto a two-block output shard and accumulated blocks / 2 + 1
    of the blocks.
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
        batches_per_core=2,
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
