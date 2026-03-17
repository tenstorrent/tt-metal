# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtLlamaClassEmbedding(LightweightModule):
    """Class Embedding layer."""

    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        configuration,
    ):
        super().__init__()

        self.mesh_device = mesh_device

        # Add batch and ntoks dimensions
        class_embedding = state_dict[f"{state_dict_prefix}class_embedding"]
        class_embedding = class_embedding.reshape(1, 1, 1, *class_embedding.shape)

        self.class_embedding = ttnn.as_tensor(
            class_embedding,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,  # [INFO] TILE_LAYOUT shouldn't be used here because of the `dim=2` concat that this tensor is used in
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def forward(self, x):
        bsz = x.shape[1]
        # Broadcast class embedding to match input batch size
        class_embedding = ttnn.concat([self.class_embedding] * bsz, dim=1)  # Broadcast batch size
        x = ttnn.concat([class_embedding, x], dim=2)  # Output ROW_MAJOR_LAYOUT

        # Pad height to next multiple of 32 required by tilize
        ntok = x.shape[2]
        padded_ntok = ((ntok + 31) // 32) * 32
        if ntok != padded_ntok:
            output_shape = [x.shape[0], x.shape[1], padded_ntok, x.shape[3]]
            x = ttnn.tilize_with_val_padding(x, output_tensor_shape=output_shape, pad_value=0)
        else:
            x = ttnn.tilize(x)  # Convert back to TILE_LAYOUT
        x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)  # Convert back to TILE_LAYOUT

        return x
