# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import csv
import os

import ttnn
from models.common.lightweightmodule import LightweightModule

_cls_emb_collected = set()
if os.path.exists("llama_class_embedding_1d_performance.csv"):
    with open("llama_class_embedding_1d_performance.csv", "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                _cls_emb_collected.add(",".join(row))


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
        self._model_name = configuration.model_name if hasattr(configuration, "model_name") else "unknown"

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
        _file_exists = os.path.exists("llama_class_embedding_1d_performance.csv")
        with open("llama_class_embedding_1d_performance.csv", "a") as _f:
            if not _file_exists:
                _f.write(
                    "device_shape_x,device_shape_y,x_dtype,x_shape_0,x_shape_1,x_shape_2,x_shape_3,cls_emb_dtype,cls_emb_shape_0,cls_emb_shape_1,cls_emb_shape_2,cls_emb_shape_3,model_name\n"
                )
            _dev_shape = list(self.mesh_device.shape) if hasattr(self.mesh_device, "shape") else [1, 1]
            _entry = (
                f"{_dev_shape[0]},{_dev_shape[1]},"
                f"{x.dtype},{x.shape[0]},{x.shape[1]},{x.shape[2]},{x.shape[3]},"
                f"{self.class_embedding.dtype},{self.class_embedding.shape[0]},{self.class_embedding.shape[1]},{self.class_embedding.shape[2]},{self.class_embedding.shape[3]},"
                f"{self._model_name}"
            )
            if _entry not in _cls_emb_collected:
                _cls_emb_collected.add(_entry)
                _f.write(f"{_entry}\n")

        bsz = x.shape[1]
        # Broadcast class embedding to match input batch size
        class_embedding = ttnn.concat([self.class_embedding] * bsz, dim=1)  # Broadcast batch size
        x = ttnn.concat([class_embedding, x], dim=2)  # Output ROW_MAJOR_LAYOUT
        x = ttnn.tilize(x)  # Convert back to TILE_LAYOUT

        return x
