# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B LM head: the final `[hidden, vocab]` projection.

Column-parallel over the vocab dim: each TP column produces `vocab/tp` logits from a full-emb input.
The logits are left SHARDED — nothing in a prefill bring-up needs them gathered, and a full-emb
gather of a 128256-wide row is pure cost. (Bring-up ends at chunked prefill; sampling belongs to the
serving follow-on.)

## `tie_word_embeddings` is FALSE for this checkpoint

`lm_head.weight` is a real, separate `[128256, 4096]` tensor — not a view of `model.embed_tokens`.
Asserted rather than assumed: a model that ties them and a model that does not are indistinguishable
until the untied weights diverge from the embedding, and reusing the embedding here would be a
silent accuracy bug on this checkpoint.

## No vocab padding is needed

Most LM-head implementations pad the vocab up to a tile- and shard-aligned size before sharding,
because a ragged vocab lands padding inside one TP column. Llama-3.1's 128256 needs none of it:
128256 = 4008 tiles, and 128256 / 4 = 32064 = 1002 tiles per column. The alignment is asserted so a
different checkpoint fails loudly here instead of silently mis-slicing the last column.
"""

import ttnn
from models.demos.llama_3_1_8b_d_p.utils.general_utils import get_cache_file_name

from .dense_mlp import hifi4_compute_config


class LMHead:
    """Column-parallel vocab projection."""

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        mesh_config,
        *,
        weight_dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
    ):
        """
        Args:
            state_dict: `{"weight": [vocab, hidden]}` (HF `lm_head`), or `{}` for cache-only.
        """
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.hidden_size = hf_config.hidden_size
        self.vocab_size = hf_config.vocab_size
        self.weight_dtype = weight_dtype
        self.tensor_cache_path = tensor_cache_path
        self.state_dict = state_dict
        self.compute_kernel_config = hifi4_compute_config()

        assert not getattr(hf_config, "tie_word_embeddings", False), (
            "this checkpoint has tie_word_embeddings=false: lm_head is a separate weight, not the "
            "embedding table transposed"
        )
        assert self.vocab_size % mesh_config.tp == 0, (
            f"vocab {self.vocab_size} must divide tp {mesh_config.tp}"
        )
        assert (self.vocab_size // mesh_config.tp) % ttnn.TILE_SIZE == 0, (
            f"vocab/tp = {self.vocab_size // mesh_config.tp} is not tile-aligned; this head would "
            "need explicit vocab padding before the column-parallel shard"
        )
        self.vocab_local = self.vocab_size // mesh_config.tp
        self.weight = None
        self._load_weights()

    def _load_weights(self):
        """Tilize the column-parallel `[hidden, vocab]` weight onto the mesh.

        HF stores `lm_head.weight` as `[vocab, hidden]` and `ttnn.linear` wants `[hidden, vocab]`,
        so it is transposed on the way in. Sharding the OUTPUT (vocab) dim across TP is what makes
        this column-parallel.
        """
        weight = self.state_dict.get("weight") if self.state_dict else None
        if weight is not None:
            weight = weight.transpose(-1, -2).unsqueeze(0).unsqueeze(0)  # -> [1, 1, hidden, vocab]
        elif not self.tensor_cache_path:
            return

        self.weight = ttnn.as_tensor(
            weight,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=self.weight_dtype,
            mesh_mapper=self.mesh_config.column_parallel(self.mesh_device),
            cache_file_name=get_cache_file_name(self.tensor_cache_path, "lm_head"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, x):
        """x `[1, 1, tokens_local, hidden]` (full emb) -> logits `[1, 1, tokens_local, vocab/tp]`.

        The logits stay SHARDED across the TP columns. Gathering a 128256-wide row per token is pure
        cost in a prefill bring-up, and nothing here consumes it — sampling is a serving concern.
        """
        assert self.weight is not None, "lm_head weight was never built (no state_dict and no cache)"
        return ttnn.linear(
            x, self.weight, dtype=ttnn.bfloat16, compute_kernel_config=self.compute_kernel_config
        )
