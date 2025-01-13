# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import random
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

TIMEOUT = 10
# seed for random
random.seed(0)

parameters = {
    "nightly": {
        "embedding_specs": [
            {"weight_shape": [1, 768], "indices_shape": [1, 10]},
            {"weight_shape": [1024, 768], "indices_shape": [1, 7]},
            {"weight_shape": [2, 1024], "indices_shape": [1, 256]},
            {"weight_shape": [2, 128], "indices_shape": [1, 12]},
            {"weight_shape": [2, 128], "indices_shape": [1, 14]},
            {"weight_shape": [2, 128], "indices_shape": [1, 9]},
            {"weight_shape": [2, 768], "indices_shape": [1, 193]},
            {"weight_shape": [2, 768], "indices_shape": [1, 25]},
            {"weight_shape": [2, 768], "indices_shape": [1, 8]},
            {"weight_shape": [2048, 768], "indices_shape": [1, 1]},
            {"weight_shape": [2048, 768], "indices_shape": [1, 45]},
            {"weight_shape": [2048, 768], "indices_shape": [2048]},
            {"weight_shape": [2050, 1024], "indices_shape": [1, 1]},
            {"weight_shape": [2050, 1024], "indices_shape": [1, 59]},
            {"weight_shape": [250002, 768], "indices_shape": [1, 10], "padding_idx": 1},
            {"weight_shape": [250880, 1536], "indices_shape": [1, 32]},
            {"weight_shape": [256008, 1024], "indices_shape": [1, 19], "padding_idx": 1},
            {"weight_shape": [262, 768], "indices_shape": [1, 2048]},
            {"weight_shape": [30000, 128], "indices_shape": [1, 12], "padding_idx": 0},
            {"weight_shape": [30000, 128], "indices_shape": [1, 14], "padding_idx": 0},
            {"weight_shape": [30000, 128], "indices_shape": [1, 9], "padding_idx": 0},
            {"weight_shape": [30522, 1024], "indices_shape": [1, 256], "padding_idx": 0},
            {"weight_shape": [30522, 768], "indices_shape": [1, 16], "padding_idx": 0},
            {"weight_shape": [30522, 768], "indices_shape": [1, 25], "padding_idx": 0},
            {"weight_shape": [30522, 768], "indices_shape": [1, 8]},
            {"weight_shape": [30528, 768], "indices_shape": [1, 8], "padding_idx": 0},
            {"weight_shape": [32, 12], "indices_shape": [1, 1]},
            {"weight_shape": [32, 12], "indices_shape": [10, 10]},
            {"weight_shape": [32, 12], "indices_shape": [2, 2]},
            {"weight_shape": [32, 12], "indices_shape": [1, 1]},
            {"weight_shape": [32, 16], "indices_shape": [1, 1]},
            {"weight_shape": [32, 16], "indices_shape": [10, 10]},
            {"weight_shape": [32, 16], "indices_shape": [2, 2]},
            {"weight_shape": [32, 16], "indices_shape": [1, 1]},
            {"weight_shape": [32, 6], "indices_shape": [1, 1]},
            {"weight_shape": [32, 6], "indices_shape": [15, 15]},
            {"weight_shape": [32, 6], "indices_shape": [17, 17]},
            {"weight_shape": [32, 6], "indices_shape": [2, 2]},
            {"weight_shape": [32, 6], "indices_shape": [1, 1]},
            {"weight_shape": [32, 8], "indices_shape": [1, 1]},
            {"weight_shape": [32, 8], "indices_shape": [10, 10]},
            {"weight_shape": [32, 8], "indices_shape": [2, 2]},
            {"weight_shape": [32, 8], "indices_shape": [1, 1]},
            {"weight_shape": [320, 64], "indices_shape": [24, 24]},
            {"weight_shape": [32000, 4096], "indices_shape": [1, 32], "padding_idx": 0},
            {"weight_shape": [32128, 1024], "indices_shape": [1, 10]},
            {"weight_shape": [32128, 1024], "indices_shape": [1, 1]},
            {"weight_shape": [32128, 512], "indices_shape": [1, 10]},
            {"weight_shape": [32128, 512], "indices_shape": [1, 15]},
            {"weight_shape": [32128, 512], "indices_shape": [1, 1]},
            {"weight_shape": [32128, 768], "indices_shape": [1, 10]},
            {"weight_shape": [32128, 768], "indices_shape": [1, 1]},
            {"weight_shape": [40, 768], "indices_shape": [1, 8]},
            {"weight_shape": [49408, 512], "indices_shape": [2, 7]},
            {"weight_shape": [50, 768], "indices_shape": [1, 50]},
            {"weight_shape": [50257, 768], "indices_shape": [1, 1]},
            {"weight_shape": [50257, 768], "indices_shape": [1, 45]},
            {"weight_shape": [50257, 768], "indices_shape": [1, 7]},
            {"weight_shape": [50272, 512], "indices_shape": [1, 1], "padding_idx": 1},
            {"weight_shape": [50272, 512], "indices_shape": [1, 59], "padding_idx": 1},
            {"weight_shape": [512, 1024], "indices_shape": [1, 256]},
            {"weight_shape": [512, 128], "indices_shape": [1, 12]},
            {"weight_shape": [512, 128], "indices_shape": [1, 14]},
            {"weight_shape": [512, 128], "indices_shape": [1, 9]},
            {"weight_shape": [512, 768], "indices_shape": [1, 16]},
            {"weight_shape": [512, 768], "indices_shape": [1, 25]},
            {"weight_shape": [512, 768], "indices_shape": [1, 8]},
            {"weight_shape": [51200, 1024], "indices_shape": [1, 1]},
            {"weight_shape": [51200, 1024], "indices_shape": [1, 5]},
            {"weight_shape": [514, 768], "indices_shape": [1, 10], "padding_idx": 1},
            {"weight_shape": [51865, 768], "indices_shape": [1, 1], "padding_idx": 50257},
            {"weight_shape": [51865, 768], "indices_shape": [1, 4], "padding_idx": 50257},
            {"weight_shape": [65024, 4544], "indices_shape": [1, 7]},
            {"weight_shape": [77, 512], "indices_shape": [1, 7]},
            {"weight_shape": [81, 768], "indices_shape": [1, 24], "padding_idx": 1},
        ],
        "dtype": [ttnn.bfloat16],
        "indices_layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "weight_layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "output_layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
    }
}


def run(
    embedding_specs,
    dtype,
    indices_layout,
    weight_layout,
    output_layout,
    *,
    device,
):
    device.enable_async(False)

    # Extract the weight and indices shape from embedding_specs
    weight_shape = embedding_specs["weight_shape"]
    indices_shape = embedding_specs["indices_shape"]
    padding_idx = embedding_specs.get("padding_idx", None)  # Optional padding index

    # Create random weight and indices tensors in PyTorch
    weight = torch_random(weight_shape, -0.1, 0.1, dtype=torch.bfloat16)
    indices = torch.randint(0, weight_shape[0], indices_shape, dtype=torch.int32)

    # Create a PyTorch embedding layer and apply it
    torch_embedding = torch.nn.Embedding.from_pretrained(weight, padding_idx=padding_idx)
    torch_output_tensor = torch_embedding(indices)

    # Convert the weight and indices to ttnn tensor format
    ttnn_weight = ttnn.from_torch(weight, device=device, layout=weight_layout, dtype=dtype)
    ttnn_indices = ttnn.from_torch(indices, device=device, layout=indices_layout, dtype=ttnn.uint32)

    # Measure performance of the embedding operation in ttnn
    start_time = start_measuring_time()

    # Apply embedding in ttnn
    ttnn_output_tensor = ttnn.embedding(
        ttnn_indices,
        ttnn_weight,
        padding_idx=padding_idx,
        layout=output_layout,
        embeddings_type=ttnn.EmbeddingsType.GENERIC,  # Default embeddings type
        dtype=dtype,
        output_tensor=None,  # No preallocated output tensor
        memory_config=None,  # Default memory config
        queue_id=0,  # Default queue id
    )

    e2e_perf = stop_measuring_time(start_time)

    # Convert the ttnn tensor back to PyTorch for comparison
    ttnn_output_tensor = ttnn.to_torch(ttnn_output_tensor)

    # Compare the results and return performance and accuracy check
    result = check_with_pcc(torch_output_tensor, ttnn_output_tensor, 0.999)

    return [result, e2e_perf]
