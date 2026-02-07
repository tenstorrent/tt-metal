# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import csv
import os

import ttnn
from models.common.lightweightmodule import LightweightModule

_emb_collected = set()
if os.path.exists("embedding_1d_performance.csv"):
    with open("embedding_1d_performance.csv", "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                _emb_collected.add(",".join(row))


class Embedding(LightweightModule):
    def __init__(
        self,
        mesh_device,
        args,
        weight_cache_path,
        state_dict,
        dtype,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self._model_name = args.model_name if hasattr(args, "model_name") else "unknown"
        self._cluster_shape = args.cluster_shape
        self._vocab_size = args.vocab_size
        self._dim = args.dim
        self._weights_dtype = dtype
        self._embed_scale = 1.0

        base_name = args.get_state_dict_prefix("", None) + "tok_embeddings.weight"
        torch_weight = state_dict[base_name].unsqueeze(0).unsqueeze(0)
        cache_name = None if args.dummy_weights else weight_cache_path / base_name
        self.weights = ttnn.as_tensor(
            torch_weight,
            dtype=dtype,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device=mesh_device, dims=(None, 3), mesh_shape=args.cluster_shape),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=args.get_model_config()["EMB_WEIGHTS_MEMCFG"],
            cache_file_name=cache_name,
        )

    def _log_csv(self, x):
        _file_exists = os.path.exists("embedding_1d_performance.csv")
        with open("embedding_1d_performance.csv", "a") as _f:
            if not _file_exists:
                _f.write(
                    "class_name,cluster_shape_x,cluster_shape_y,weights_dtype,embed_scale,x_shape_0,x_shape_1,weights_shape_0,weights_shape_1,weights_shape_2,weights_shape_3,model_name\n"
                )
            _entry = (
                f"{self.__class__.__name__},{self._cluster_shape[0]},{self._cluster_shape[1]},"
                f"{self._weights_dtype},{self._embed_scale},"
                f"{x.shape[0]},{x.shape[1]},"
                f"{self.weights.shape[0]},{self.weights.shape[1]},{self.weights.shape[2]},{self.weights.shape[3]},"
                f"{self._model_name}"
            )
            if _entry not in _emb_collected:
                _emb_collected.add(_entry)
                _f.write(f"{_entry}\n")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        self._log_csv(x)
        x = ttnn.embedding(x, self.weights, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return x


class ScaledEmbedding(Embedding):
    def __init__(self, mesh_device, args, weight_cache_path, state_dict, dtype, embed_scale: float = 1.0):
        super().__init__(mesh_device, args, weight_cache_path, state_dict, dtype)
        self.embed_scale = embed_scale
        self._embed_scale = embed_scale

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        self._log_csv(x)
        e = ttnn.embedding(x, self.weights, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        s = ttnn.multiply(e, self.embed_scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return s
