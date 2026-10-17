import ttl

import ttnn

TILE = 32


@ttl.operation(grid=(1, 1))
def fused_relu2(x: ttnn.Tensor, y: ttnn.Tensor) -> None:
    n_tiles = x.shape[-1] // TILE
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)

    @ttl.datamovement()
    def read():
        for nt in range(n_tiles):
            with x_dfb.reserve() as x_blk:
                ttl.copy(x[0, nt], x_blk).wait()

    @ttl.compute()
    def compute():
        for _ in range(n_tiles):
            with x_dfb.wait() as x_blk:
                with y_dfb.reserve() as y_blk:
                    r = ttl.math.relu(x_blk)
                    y_blk.store(r * r)

    @ttl.datamovement()
    def write():
        for nt in range(n_tiles):
            with y_dfb.wait() as y_blk:
                ttl.copy(y_blk, y[0, nt]).wait()


def relu2(x):
    x2 = ttnn.reshape(x, [1, x.shape[-1]])
    y = ttnn.allocate_tensor_on_device(x2.shape, x2.dtype, x2.layout, x2.device())
    fused_relu2(x2, y)
    return ttnn.reshape(y, list(x.shape))
