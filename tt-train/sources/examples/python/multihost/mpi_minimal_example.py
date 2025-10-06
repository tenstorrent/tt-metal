import os
import sys
import numpy as np

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml

if __name__ == "__main__":
    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.initialize_distributed_context(*sys.argv)
    distributed_ctx = autograd_ctx.get_distributed_context()

    autograd_ctx.initialize_socket_manager(ttml.core.distributed.SocketType.MPI)
    socket_manager = autograd_ctx.get_socket_manager()

    rank = distributed_ctx.rank()
    world_size = distributed_ctx.size()

    assert world_size > 1, f"World size must be greater than 1, world size: {world_size}"

    if rank == 0:
        print("Rank 0 is sending data")
        tensor_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        tensor_data = tensor_data.reshape(1, 1, 2, 4)
        tensor = ttml.autograd.Tensor.from_numpy(
            tensor_data, layout=ttml.Layout.TILE, new_type=ttml.autograd.DataType.BFLOAT16
        )

        for dest_rank in range(1, world_size):
            socket_manager.send(tensor, distributed_ctx, dest_rank)
    else:
        print(f"Rank {rank} is receiving data")
        tensor_data = np.zeros((1, 1, 2, 4), dtype=np.float32)
        tensor = ttml.autograd.Tensor.from_numpy(
            tensor_data, layout=ttml.Layout.TILE, new_type=ttml.autograd.DataType.BFLOAT16
        )
        tensor = socket_manager.recv(tensor, distributed_ctx, 0)
        tensor_data = tensor.to_numpy().flatten()
        assert tensor_data.tolist() == [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
        ], f"Rank {rank} received data: {tensor_data} does not match expected data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]"
        print(f"Rank {rank} received data: {tensor_data}")
