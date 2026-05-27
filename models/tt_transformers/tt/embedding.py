# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule


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
        self._dtype = dtype
        self._memory_config = args.get_model_config()["EMB_WEIGHTS_MEMCFG"]
        base_name = args.get_state_dict_prefix("", None) + "tok_embeddings.weight"
        torch_weight = state_dict[base_name].unsqueeze(0).unsqueeze(0)
        cache_name = None if args.dummy_weights else weight_cache_path / base_name
        self.weights = ttnn.as_tensor(
            torch_weight,
            dtype=self._dtype,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device=mesh_device, dims=(None, 3), mesh_shape=args.cluster_shape),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=self._memory_config,
            cache_file_name=cache_name,
        )

    def update(self, tensor: ttnn.Tensor) -> None:
        """In-place replace the on-device embedding values via ``ttnn.copy``.

        Strictly on device. ``tensor`` must already live on
        ``self.mesh_device``; this method handles any divergence in shape,
        layout, dtype, or ``memory_config`` using on-device ops only (no
        host roundtrip). Single-device assumption: mesh-sharding
        redistribution is not performed.

        Per-replica element count must equal ``vocab_size * hidden_size``
        (e.g. ``128256 * 2048`` for Llama-3.2-1B-Instruct; see
        ``tt-train/sources/examples/grpo_speedup/HF_LLAMA_FORMAT.md`` §3.1
        where the storage shape is ``(1, 1, V, H)``).

        Conversion pipeline (each step skipped when already matching):

        1. ``ttnn.to_layout``        -> ``ttnn.ROW_MAJOR_LAYOUT``
        2. ``ttnn.typecast``         -> ``self._dtype`` (e.g. ``ttnn.bfloat16``)
        3. ``ttnn.reshape``          -> ``self.weights.shape``
                                        (``(1, 1, vocab_size, hidden_size)``)
        4. ``ttnn.to_memory_config`` -> ``self._memory_config``
                                        (``EMB_WEIGHTS_MEMCFG``)
        5. ``ttnn.copy(input_a=converted, input_b=self.weights)``
           -- in-place. ``self.weights``' device buffer is preserved (no
           reallocation) so any captured trace remains valid.
        """
        converted = tensor

        if converted.layout != ttnn.ROW_MAJOR_LAYOUT:
            converted = ttnn.to_layout(converted, layout=ttnn.ROW_MAJOR_LAYOUT)

        if converted.dtype != self._dtype:
            converted = ttnn.typecast(converted, dtype=self._dtype)

        if tuple(converted.shape) != tuple(self.weights.shape):
            converted = ttnn.reshape(converted, list(self.weights.shape))

        if converted.memory_config() != self._memory_config:
            converted = ttnn.to_memory_config(converted, self._memory_config)

        ttnn.copy(input_a=converted, input_b=self.weights)

    def forward(self, x: ttnn.Tensor, memory_config=None) -> ttnn.Tensor:
        x = ttnn.embedding(x, self.weights, layout=ttnn.TILE_LAYOUT, memory_config=memory_config)
        return x


class ScaledEmbedding(Embedding):
    def __init__(self, mesh_device, args, weight_cache_path, state_dict, dtype, embed_scale: float = 1.0):
        super().__init__(mesh_device, args, weight_cache_path, state_dict, dtype)
        self.embed_scale = embed_scale

    def forward(self, x: ttnn.Tensor, memory_config=None) -> ttnn.Tensor:
        e = ttnn.embedding(x, self.weights, layout=ttnn.TILE_LAYOUT, memory_config=memory_config)
        s = ttnn.multiply(e, self.embed_scale, memory_config=memory_config)
        return s
