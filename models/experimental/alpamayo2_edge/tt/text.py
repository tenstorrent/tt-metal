# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Cosmos3 Edge language model on device: tt_dit's `TransformerEncoder` with a ReLU² feed-forward.

The KV cache is tt_dit's `Cache`, with per-layer access for the action expert, which attends over the LM's
rotated K and raw V.
"""

from __future__ import annotations

import torch

import ttnn
from models.experimental.alpamayo2_edge.reference.config import TextConfig
from models.experimental.alpamayo2_edge.reference.text import KVCache
from models.experimental.alpamayo2_edge.tt.common import drop_unbuilt_layers, round_up
from models.tt_dit.blocks.rope import RopeConfig
from models.tt_dit.encoders.transformer import (
    LINEAR_DTYPE,
    MAX_CHUNK_SIZE,
    WORKAROUND_MIN_DECODE_CHUNK_SIZE,
    Cache,
    StateConversion,
    TransformerEncoder,
    TransformerEncoderConfig,
)
from models.tt_dit.layers.linear import ColParallelLinear, RowParallelLinear
from models.tt_dit.layers.module import Module
from models.tt_dit.utils import tensor

STATE_CONVERSION = StateConversion(
    rename=[
        (r"^embed_tokens\.", r"token_embedding."),
        (r"^layers\.([0-9]+)\.self_attn\.([qkvo])_proj\.", r"layers.\1.attn.\2_proj."),
        (r"^layers\.([0-9]+)\.mlp\.fc1\.", r"layers.\1.ff.linear_in."),
        (r"^layers\.([0-9]+)\.mlp\.fc2\.", r"layers.\1.ff.linear_out."),
        (r"^layers\.([0-9]+)\.input_layernorm\.", r"layers.\1.attn_norm."),
        (r"^layers\.([0-9]+)\.post_attention_layernorm\.", r"layers.\1.ff_norm."),
        (r"^norm\.", r"final_norm."),
    ],
)


class ReluSquaredFeedForward(Module):
    """`linear_out(relu(linear_in(x)) ** 2)`, unsharded, without biases."""

    def __init__(
        self, embed_size: int, hidden_size: int, *, mesh_device: ttnn.MeshDevice, dtype: ttnn.DataType = LINEAR_DTYPE
    ) -> None:
        super().__init__()
        self.linear_in = ColParallelLinear(
            embed_size, hidden_size, bias=False, mesh_device=mesh_device, mesh_axis=None, dtype=dtype
        )
        self.linear_out = RowParallelLinear(
            hidden_size, embed_size, bias=False, mesh_device=mesh_device, mesh_axis=None, dtype=dtype
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.square(ttnn.relu(self.linear_in.forward(x)))
        return self.linear_out.forward(x)


class AlpamayoCache(Cache):
    """tt_dit's prefill/decode cache with per-layer access.

    K (rotated) and V of layer i are `[B, kv_heads, size, head_dim]` bf16 tile tensors in DRAM.
    """

    def __init__(self, *, device: ttnn.MeshDevice, size: int, batch_size: int) -> None:
        # The decode kernel needs the cache length to be a multiple of its smallest correct k-chunk.
        if size % WORKAROUND_MIN_DECODE_CHUNK_SIZE != 0:
            msg = f"cache size must be a multiple of {WORKAROUND_MIN_DECODE_CHUNK_SIZE}, got {size}"
            raise ValueError(msg)
        super().__init__(device=device, size=size, batch_size=batch_size)

    @staticmethod
    def size_for(max_prompt: int, *, max_new_tokens: int, action_rows: int) -> int:
        """Rows for a prompt of up to `max_prompt` tokens, `max_new_tokens` CoT tokens and the expert's action rows.

        `Cache.prefill` writes the prompt padded to a multiple of 128 rows, and the action rows start at the CoT
        prefix rounded up to a tile.
        """
        rows = max(
            round_up(max_prompt, MAX_CHUNK_SIZE),
            round_up(max_prompt + max_new_tokens, ttnn.TILE_SIZE) + action_rows,
        )
        return round_up(rows, WORKAROUND_MIN_DECODE_CHUNK_SIZE)

    def layer_kv(self, layer: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if layer not in self._k_cache:
            msg = f"layer {layer} of the cache holds no K/V: it was not filled, or it was deallocated"
            raise KeyError(msg)
        return self._k_cache[layer], self._v_cache[layer]

    def deallocate(self) -> None:
        """Frees the K/V of every layer and the position; the cache cannot be used afterwards."""
        for kv in (*self._k_cache.values(), *self._v_cache.values()):
            ttnn.deallocate(kv)
        self._k_cache.clear()
        self._v_cache.clear()
        ttnn.deallocate(self._position)

    def repeat_to(self, batch: int) -> AlpamayoCache:
        """A copy of this single-sequence cache for `batch` samples, at the same position."""
        if self.batch_size != 1:
            msg = f"only a single-sequence cache can be repeated, this one holds {self.batch_size}"
            raise ValueError(msg)
        out = AlpamayoCache(device=self._device, size=self.size, batch_size=batch)
        for layer in self._k_cache:
            out._k_cache[layer] = ttnn.repeat(self._k_cache[layer], ttnn.Shape([batch, 1, 1, 1]))
            out._v_cache[layer] = ttnn.repeat(self._v_cache[layer], ttnn.Shape([batch, 1, 1, 1]))
        ttnn.copy(ttnn.repeat(self.position, ttnn.Shape([batch])), out.position)
        return out

    @classmethod
    def from_reference(cls, kv: KVCache, *, size: int, mesh_device: ttnn.MeshDevice) -> AlpamayoCache:
        """The reference cache's rows in a `size`-row cache laid out as the TT prefill leaves it.

        The cache is positioned after those rows.
        """
        rows = kv.seq_len
        if rows > size:
            msg = f"a {size}-row cache cannot hold the {rows} reference rows"
            raise ValueError(msg)
        cache = cls(device=mesh_device, size=size, batch_size=kv.k[0].shape[0])
        pad = (0, 0, 0, size - rows)
        for layer, (k, v) in enumerate(zip(kv.k, kv.v)):
            if k is None:
                continue
            cache._k_cache[layer] = tensor.from_torch(torch.nn.functional.pad(k, pad), device=mesh_device)
            cache._v_cache[layer] = tensor.from_torch(torch.nn.functional.pad(v, pad), device=mesh_device)
        cache.advance(rows)
        return cache


class TtLanguageModel(TransformerEncoder):
    """The Cosmos3 Edge LM: tt_dit's encoder with each layer's gated-SiLU feed-forward replaced by ReLU²."""

    def __init__(
        self,
        config: TransformerEncoderConfig,
        *,
        device: ttnn.MeshDevice,
        lm_head_dtype: ttnn.DataType = LINEAR_DTYPE,
    ) -> None:
        """
        Args:
            lm_head_dtype: Weight dtype of the vocabulary projection; tt_dit's linear default is bf8.
        """
        super().__init__(config, device=device)
        # Parameters are allocated when loaded, so the replaced modules cost nothing.
        for layer in self.layers:
            layer.ff = ReluSquaredFeedForward(config.embed_size, config.ff_size, mesh_device=device)
        if self.final_linear is not None and lm_head_dtype != LINEAR_DTYPE:
            self.final_linear = ColParallelLinear(
                config.embed_size,
                config.vocab_size,
                bias=False,
                mesh_device=device,
                mesh_axis=None,
                dtype=lm_head_dtype,
            )

    @staticmethod
    def config_from(tcfg: TextConfig) -> TransformerEncoderConfig:
        return TransformerEncoderConfig(
            embed_size=tcfg.hidden,
            ff_size=tcfg.mlp,
            head_size=tcfg.head_dim,
            norm_eps=tcfg.eps,
            num_heads=tcfg.heads,
            num_kv_heads=tcfg.kv_heads,
            num_layers=tcfg.layers,
            attn_qkv_bias=False,
            attn_out_bias=False,
            attn_qk_norm=False,
            vocab_size=tcfg.vocab,
            rope_config=RopeConfig(
                theta=tcfg.rope_theta, mrope_section=list(tcfg.mrope_section), mrope_interleaved=True
            ),
        )

    @staticmethod
    def convert_state(lm: dict[str, torch.Tensor], head: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Checkpoint `vlm.model.language_model.*` and `vlm.lm_head.*` tensors, prefixes stripped -> our keys."""
        return {**STATE_CONVERSION.convert(lm), **{f"final_linear.{k}": v for k, v in head.items()}}

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        super()._prepare_torch_state(state)
        drop_unbuilt_layers(state, "layers.", len(self.layers))
        # A config may leave out the final norm or head; their tensors are not loaded.
        for key in list(state):
            if (key.startswith("final_norm.") and self.final_norm is None) or (
                key.startswith("final_linear.") and self.final_linear is None
            ):
                del state[key]

    def prefill(
        self,
        tokens: ttnn.Tensor,
        *,
        positions: ttnn.Tensor,
        vision_embeds: ttnn.Tensor,
        vision_mask: ttnn.Tensor,
        cache: AlpamayoCache,
    ) -> ttnn.Tensor:
        """Fills `cache` with the prompt and returns the logits of its last token, `[1, vocab]`.

        Args:
            tokens: `[1, S]` uint32 prompt ids.
            positions: `[3, 1, S]` float32 M-RoPE positions.
            vision_embeds: `[V, hidden]` embeddings that replace the `V` rows `vision_mask` marks, in order.
            vision_mask: `[1, S]`, 1 on the image-token rows.
            cache: A single-sequence cache. The prefill writes rows up to S padded to a multiple of 128; the padding
                rows hold finite K/V of pad tokens, which the decode overwrites and the expert masks. They must stay
                finite: a masked inf or NaN still turns the attention output into NaN.
        """
        batch_size, seq_len = tokens.shape
        if batch_size != 1 or cache.batch_size != 1:
            msg = "the prefill runs a single sequence"
            raise ValueError(msg)
        if cache.size < round_up(seq_len, MAX_CHUNK_SIZE):
            msg = f"a {cache.size}-row cache cannot hold the prefill of {seq_len} tokens padded to {MAX_CHUNK_SIZE}"
            raise ValueError(msg)
        hidden = self.forward(
            tokens,
            positions=positions,
            vision_embeds=vision_embeds,
            vision_mask=vision_mask,
            cache=cache,
            skip_final_linear=True,
        )
        return self._last_token_logits(hidden, index=seq_len - 1)

    @property
    def device(self) -> ttnn.MeshDevice:
        return self._device

    def decode_step(self, tokens: ttnn.Tensor, *, cache: AlpamayoCache, rope_offset: ttnn.Tensor) -> ttnn.Tensor:
        """Appends one token per sequence at the cache position, advances it and returns the next logits `[B, vocab]`.

        A decoded text token sits at cache position + `rope_offset` on every M-RoPE axis, so the 1D rope table is
        exact. That table has `cache.size` rows: the caller keeps the position plus offset below it.

        Args:
            tokens: `[B]` uint32 ids.
            rope_offset: `[B]` int32 row-major, the prompt's `rope_deltas`.
        """
        pos_embeds = self._get_pos_embeds(start=0, sequence_length=cache.size, layout=ttnn.ROW_MAJOR_LAYOUT)
        logits = self._decode_step(tokens, pos_embeds=pos_embeds, cache=cache, rope_offset=rope_offset)
        cache.advance(1)
        return logits
