import tt_lib as ttl
from loguru import logger


def convert_to_datatype_on_device(x, target_dtype, host, device):
    assert x.layout() == ttl.tensor.Layout.TILE
    if x.dtype() == target_dtype:
        return x

    logger.warning(f"Converting tensor {x.shape()} from {x.dtype()} to {target_dtype}!")

    mem_config = ttl.tensor.MemoryConfig(True, x.memory_config().buffer_type)
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
