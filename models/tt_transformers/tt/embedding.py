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
        self.vocab_size = args.vocab_size
        self.padded_vocab_size = args.padded_vocab_size
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

    @staticmethod
    def _inplace_copy(src: ttnn.Tensor, dst: ttnn.Tensor, target_dtype) -> None:
        """Convert ``src`` to ``dst``'s layout/dtype/shape/memcfg, then
        ``ttnn.copy`` it into ``dst``. ``dst``'s device buffer is preserved
        (no reallocation) so any captured trace remains valid.
        """
        converted = src

        if converted.layout != dst.layout:
            converted = ttnn.to_layout(converted, layout=dst.layout)

        if converted.dtype != target_dtype:
            converted = ttnn.typecast(converted, dtype=target_dtype)

        if tuple(converted.shape) != tuple(dst.shape):
            converted = ttnn.reshape(converted, list(dst.shape))

        if converted.memory_config() != dst.memory_config():
            converted = ttnn.to_memory_config(converted, dst.memory_config())

        ttnn.copy(input_a=converted, input_b=dst)

    def update(self, *, embed_tokens: ttnn.Tensor) -> None:
        """In-place replace the embedding table via ``ttnn.copy``.

        HF-format input (see ``LLAMA_WEIGHT_TRANSFER.md``): ``embed_tokens`` is
        ``model.embed_tokens.weight``, shape ``(1, 1, vocab_size, hidden_size)``,
        bf16, TILE, DRAM-interleaved, replicated.

        No vocab padding: the assert requires ``vocab_size == padded_vocab_size``
        (always true for Llama-3.2-1B-Instruct); padding would need extending.

        Single-device-only: ``self.weights``' ``(None, 3)`` sharding is a no-op
        on a 1x1 mesh; a multi-device mesh needs a ``ttnn.mesh_partition(dim=3,
        cluster_axis=1)`` before the copy. Buffer address is preserved so any
        captured trace stays valid.
        """
        assert self.vocab_size == self.padded_vocab_size, (
            f"Embedding.update requires self.vocab_size == self.padded_vocab_size "
            f"(got {self.vocab_size} vs {self.padded_vocab_size}); "
            "padding the embedding table is not yet supported."
        )

        self._inplace_copy(embed_tokens, self.weights, self._dtype)

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
