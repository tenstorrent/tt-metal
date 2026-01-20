# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
This script is a minimal example of using the SocketManager to send and receive data from rank 0 to all other ranks through MPI.

Example usage:
```
export TT_MESH_ID=0
export TT_MESH_HOST_RANK=0
cd ${TT_METAL_HOME}/tt-train
# synchronize build and binaries to all machines (may require modifications to the all_machines_copy.sh script)
./sources/examples/nano_gpt/3tier/all_machines_copy.sh --run --sync
# run the script
mpirun -x TT_METAL_HOME -x TT_MESH_ID -x TT_MESH_HOST_RANK --host metal-wh-01,metal-wh-03,metal-wh-04,metal-wh-05,metal-wh-06 --tag-output python3 ${TT_METAL_HOME}/tt-train/sources/examples/python/multihost/mpi_minimal_example.py
```

Expected output:
```
[1,0]<stdout>:Rank 0 is sending data
[1,1]<stdout>:Rank 1 is receiving data
[1,2]<stdout>:Rank 2 is receiving data
[1,4]<stdout>:Rank 4 is receiving data
[1,3]<stdout>:Rank 3 is receiving data
[1,1]<stdout>:Rank 1 received data: [1. 2. 3. 4. 5. 6. 7. 8.]
[1,2]<stdout>:Rank 2 received data: [1. 2. 3. 4. 5. 6. 7. 8.]
[1,3]<stdout>:Rank 3 received data: [1. 2. 3. 4. 5. 6. 7. 8.]
[1,4]<stdout>:Rank 4 received data: [1. 2. 3. 4. 5. 6. 7. 8.]
```

"""

import os
import sys
import numpy as np

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttnn
import ttml

if __name__ == "__main__":
    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.initialize_distributed_context(*sys.argv)
    distributed_ctx = autograd_ctx.get_distributed_context()

    autograd_ctx.initialize_socket_manager(ttml.core.distributed.SocketType.MPI)
    socket_manager = autograd_ctx.get_socket_manager()

    rank = distributed_ctx.rank()
    world_size = distributed_ctx.size()

    assert (
        world_size > 1
    ), f"World size must be greater than 1, world size: {world_size}"

    if rank == 0:
        print("Rank 0 is sending data")
        tensor_data = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32
        )
        tensor_data = tensor_data.reshape(1, 1, 2, 4)
        tensor = ttml.autograd.Tensor.from_numpy(
            tensor_data, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16
        )

        for dest_rank in range(1, world_size):
            socket_manager.send(tensor, distributed_ctx, dest_rank)
    else:
        print(f"Rank {rank} is receiving data")
        tensor_data = np.zeros((1, 1, 2, 4), dtype=np.float32)
        tensor = ttml.autograd.Tensor.from_numpy(
            tensor_data, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16
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
