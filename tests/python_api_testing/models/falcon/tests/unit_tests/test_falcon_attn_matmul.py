import pytest
from loguru import logger

import tt_lib as ttl
from models.utility_functions import comp_pcc, tt2torch_tensor
import torch


def run_falcon_attn_matmul_test(
    falcon_op,
    batch,
    seq_len,
    K,
    in0_dtype,
    in1_dtype,
    out_dtype,
    in0_mem_config,
    in1_mem_config,
    out_mem_config,
):
    pcc = 0.99

    if falcon_op == ttl.operations.primary.transformers.attn_matmul:
        q_len = 1
        kv_heads = 1
        q_heads = 71
        a_shape = [q_len, q_heads, batch, K]
        b_shape = [batch, kv_heads, K, seq_len]
        expected_output_shape = [1, q_heads, batch, seq_len]
    else:
        raise NotImplementedError(f"falcon matmul op is undefined!")

    torch.manual_seed(1234)
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)

    A = torch.randn(a_shape)
    B = torch.randn(b_shape) - 0.95

    a_t = (
        ttl.tensor.Tensor(A, in0_dtype)
        .to(ttl.tensor.Layout.TILE)
        .to(device, in0_mem_config)
    )
    b_t = (
        ttl.tensor.Tensor(B, in1_dtype)
        .to(ttl.tensor.Layout.TILE)
        .to(device, in1_mem_config)
    )

    out = falcon_op(
        a_t,
        b_t,
        compute_with_storage_grid_size=ttl.tensor.CoreCoord(12, 9),
        output_mem_config=out_mem_config,
        output_dtype=out_dtype,
    )

    # Check memory and dtype of inputs and outputs
    assert a_t.memory_config().buffer_type == in0_mem_config.buffer_type
    assert a_t.dtype() == in0_dtype
    assert b_t.memory_config().buffer_type == in1_mem_config.buffer_type
    assert b_t.dtype() == in1_dtype
    assert out.memory_config().buffer_type == out_mem_config.buffer_type
    assert out.dtype() == out_dtype
    logger.debug(
        f"in0 ({a_shape}): {a_t.memory_config().buffer_type} and {a_t.dtype()}"
    )
    logger.debug(
        f"in1 ({b_shape}): {b_t.memory_config().buffer_type} and {b_t.dtype()}"
    )
    logger.debug(
        f"out ({expected_output_shape}): {out.memory_config().buffer_type} and {out.dtype()}"
    )

    assert out.shape() == expected_output_shape
    pyt_got_back_rm = tt2torch_tensor(out)

    ref_bmm = torch.matmul(A.transpose(0, 2), B).transpose(0, 2)

    passing_pcc, output_pcc = comp_pcc(ref_bmm, pyt_got_back_rm, pcc)
    logger.info(f"Passing={passing_pcc}")
    logger.info(f"Output pcc={output_pcc}")
    ttl.device.CloseDevice(device)
    assert passing_pcc


# TODO: We could parametrize these separately for comprehensive testing
@pytest.mark.parametrize(
    "in0_mem_config, in1_mem_config, out_mem_config",
    (
        (
            ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
            ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
            ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        ),
    ),
    ids=["DRAM"],
)
@pytest.mark.parametrize(
    "out_dtype",
    (ttl.tensor.DataType.BFLOAT16,),
    ids=["out_BFLOAT16"],
)
@pytest.mark.parametrize(
    "in1_dtype",
    (ttl.tensor.DataType.BFLOAT16,),
    ids=["in1_BFLOAT16"],
)
@pytest.mark.parametrize(
    "in0_dtype",
    (ttl.tensor.DataType.BFLOAT16,),
    ids=["in0_BFLOAT16"],
)
@pytest.mark.parametrize(
    "falcon_op",
    (ttl.operations.primary.transformers.attn_matmul,),
    ids=["attn_matmul"],
)
@pytest.mark.parametrize(
    "batch, seq_len, K",
    ((32, 128, 64), (64, 2048, 64), (32, 64, 128), (64, 64, 2048)),
)
def test_falcon_matmul(
    falcon_op,
    batch,
    seq_len,
    K,
    in0_dtype,
    in1_dtype,
    out_dtype,
    in0_mem_config,
    in1_mem_config,
    out_mem_config,
    request,
):
    ttl.profiler.set_profiler_location(
        f"tt_metal/tools/profiler/logs/falcon_{request.node.callspec.id}"
    )
    run_falcon_attn_matmul_test(
        falcon_op,
        batch,
        seq_len,
        K,
        in0_dtype,
        in1_dtype,
        out_dtype,
        in0_mem_config,
        in1_mem_config,
        out_mem_config,
    )
