from libs import tt_lib as ttl
from loguru import logger


def convert_to_datatype_on_device(x, target_dtype, host, device):
    assert x.layout() == ttl.tensor.Layout.TILE
    if x.dtype() == target_dtype:
        return x

    logger.warning(f"Converting tensor {x.shape()} from {x.dtype()} to {target_dtype}!")

    mem_config = ttl.tensor.MemoryConfig(True, -1, x.buffer_type())
    x = x.to(host).to(ttl.tensor.Layout.ROW_MAJOR)
    x = (
        ttl.tensor.Tensor(
            x.data(),
            x.shape(),
            target_dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device, mem_config)
    )

    return x


def run_matmul_with_dataformat(matmul, target_dtype, device, output_mem_config, in0, in1, bias=None, gelu_activation=False):
    assert isinstance(in0, ttl.tensor.Tensor)
    assert isinstance(in1, ttl.tensor.Tensor)
    if bias is not None:
        assert isinstance(bias, ttl.tensor.Tensor)
    host = ttl.device.GetHost()

    assert in0.layout() == ttl.tensor.Layout.TILE
    assert in1.layout() == ttl.tensor.Layout.TILE
    assert in0.dtype() == in1.dtype()
    if bias is not None:
        assert bias.layout() == ttl.tensor.Layout.TILE
        assert in0.dtype() == bias.dtype()
    if in0.dtype() == target_dtype:
        matmul_args = [in0, in1]
        if bias is not None:
            matmul_args.append(bias)
        if gelu_activation:
            matmul_args.append(gelu_activation)
        return matmul(*matmul_args, mem_config=output_mem_config)

    src_dtype = in0.dtype()

    logger.warning(
        f"Converting tensor {in0.shape()} from {in0.dtype()} to {target_dtype}!"
    )
    mem_config = ttl.tensor.MemoryConfig(True, -1, in0.buffer_type())
    in0 = in0.to(host).to(ttl.tensor.Layout.ROW_MAJOR)
    in0 = (
        ttl.tensor.Tensor(
            in0.data(),
            in0.shape(),
            target_dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device, mem_config)
    )

    logger.warning(
        f"Converting tensor {in1.shape()} from {in1.dtype()} to {target_dtype}!"
    )
    mem_config = ttl.tensor.MemoryConfig(True, -1, in1.buffer_type())
    in1 = in1.to(host).to(ttl.tensor.Layout.ROW_MAJOR)
    in1 = (
        ttl.tensor.Tensor(
            in1.data(),
            in1.shape(),
            target_dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device, mem_config)
    )
    if bias is not None:
        logger.warning(
            f"Converting tensor {bias.shape()} from {bias.dtype()} to {target_dtype}!"
        )
        mem_config = ttl.tensor.MemoryConfig(True, -1, bias.buffer_type())
        bias = bias.to(host).to(ttl.tensor.Layout.ROW_MAJOR)
        bias = (
            ttl.tensor.Tensor(
                bias.data(),
                bias.shape(),
                target_dtype,
                ttl.tensor.Layout.ROW_MAJOR,
            )
            .to(ttl.tensor.Layout.TILE)
            .to(device, mem_config)
        )

    matmul_args = [in0, in1]
    if bias is not None:
        matmul_args.append(bias)
    if gelu_activation:
        matmul_args.append(gelu_activation)
    output = matmul(*matmul_args, mem_config=output_mem_config)

    logger.warning(
        f"Converting tensor {output.shape()} from {output.dtype()} to {src_dtype}!"
    )
    output = output.to(host).to(ttl.tensor.Layout.ROW_MAJOR)
    output = (
        ttl.tensor.Tensor(
            output.data(),
            output.shape(),
            src_dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device, output_mem_config)
    )

    return output
