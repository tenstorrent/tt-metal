from typing import Optional, Tuple

import torch

torch.manual_seed(42)
import random
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random
from device_fixtures import default_device
import pdb


def run(
    embedding_specs,
    dtype,
    layout,
    *,
    device,
):
    device.enable_async(False)

    # Extract the weight and indices shape from embedding_specs
    weight_shape = embedding_specs["weight_shape"]
    indices_shape = embedding_specs["indices_shape"]
    padding_idx = embedding_specs.get("padding_idx", None)  # Optional padding index

    # Create random weight and indices tensors in PyTorch
    # weight = torch.arange(weight_shape, dtype=torch.bfloat16) # invalid combination of arguments
    weight = torch.arange(weight_shape[0] * weight_shape[1], dtype=torch.bfloat16).reshape(weight_shape)
    # indices needs to be of type uint32
    indices = torch.randint(0, weight_shape[0], indices_shape, dtype=torch.int32)

    # Create a PyTorch embedding layer and apply it
    torch_embedding = torch.nn.Embedding.from_pretrained(weight, padding_idx=padding_idx)
    torch_output_tensor = torch_embedding(indices)

    # Convert the weight and indices to ttnn tensor format
    ttnn_weight = ttnn.from_torch(weight, device=device, layout=layout, dtype=dtype)
    ttnn_indices = ttnn.from_torch(indices, device=device, layout=layout, dtype=ttnn.uint32)

    # Measure performance of the embedding operation in ttnn
    start_time = start_measuring_time()

    # Apply embedding in ttnn
    ttnn_output_tensor = ttnn.embedding(
        ttnn_indices,
        ttnn_weight,
        padding_idx=padding_idx,
        layout=layout,
        embeddings_type=ttnn.EmbeddingsType.GENERIC,  # Default embeddings type
        dtype=dtype,
        output_tensor=None,  # No preallocated output tensor
        memory_config=None,  # Default memory config
        queue_id=0,  # Default queue id
    )

    e2e_perf = stop_measuring_time(start_time)

    # Convert the ttnn tensor back to PyTorch for comparison
    ttnn_output_tensor = ttnn.to_torch(ttnn_output_tensor)

    check_tensor = torch_output_tensor == ttnn_output_tensor

    # pdb.set_trace()

    print("shape: ")
    print(torch_output_tensor.shape)

    # Loop through rows of tensor and print first false element, print PASS if row is correct
    # for i in range(check_tensor.shape[1]):
    #     if False in check_tensor[0, i]:
    #         print(f"Row {i}: FAIL")
    #         print(torch_output_tensor[0, i])
    #         print(ttnn_output_tensor[0, i])
    #         break
    #     else:
    #         print(f"Row {i}: PASS")

    # Compare the results and return performance and accuracy check
    result = check_with_pcc(torch_output_tensor, ttnn_output_tensor, 0.999)

    return [result, e2e_perf]


def get_devices(test_module):
    try:
        return test_module.mesh_device_fixture()
    except:
        return default_device()


if __name__ == "__main__":
    # Run the test

    # pcc errors
    # embedding_specs = {'weight_shape': [2, 768], 'indices_shape': [1, 8]}
    embedding_specs = {"weight_shape": [30528, 768], "indices_shape": [1, 32], "padding_idx": 0}
    # embedding_specs = {'weight_shape': [400, 10], 'indices_shape': [1, 24]} # Should output 1, 24, 10

    # page/buffer errors
    # embedding_specs = {"weight_shape": [77, 512], "indices_shape": [1, 8]} #SHould output 1, 7, 512

    dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT
    # layout = ttnn.ROW_MAJOR_LAYOUT
    device_id = 0
    device = ttnn.open_device(device_id=device_id)

    print(run(embedding_specs, dtype, layout, device=device))
