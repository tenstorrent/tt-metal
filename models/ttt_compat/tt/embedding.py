# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import copy_to_buffer


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
        self.num_devices = args.num_devices
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
        assert self.num_devices == 1, (
            f"Embedding.update for num_devices > 1 is not yet implemented "
            f"(got num_devices={self.num_devices}); the multi-device path "
            "needs a ttnn.mesh_partition(dim=3, cluster_axis=1) into the "
            "(None, 3)-sharded buffer before copy."
        )
        assert self.vocab_size == self.padded_vocab_size, (
            f"Embedding.update requires self.vocab_size == self.padded_vocab_size "
            f"(got {self.vocab_size} vs {self.padded_vocab_size}); "
            "padding the embedding table is not yet supported."
        )

        copy_to_buffer(embed_tokens, self.weights, self._dtype)

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
