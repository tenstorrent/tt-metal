# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import math
import re
import warnings
from collections.abc import Container, Hashable, Mapping, Sequence
from dataclasses import dataclass

import torch

import ttnn
from models.tt_dit.blocks.rope import RopeConfig, RotaryEmbedding
from models.tt_dit.layers.embeddings import Embedding
from models.tt_dit.layers.linear import ColParallelLinear, RowParallelLinear
from models.tt_dit.layers.module import Module, ModuleList
from models.tt_dit.layers.normalization import RMSNorm
from models.tt_dit.parallel.config import EncoderParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import tensor
from models.tt_dit.utils.padding import torch_pad
from models.tt_dit.utils.tracing import Tracer, traced_function

MAX_CHUNK_SIZE = 128
# The decode kernel gets a 32-wide k-chunk wrong (https://github.com/tenstorrent/tt-metal/issues/56171),
# so the cache length must be a multiple of the smallest chunk it gets right.
WORKAROUND_MIN_DECODE_CHUNK_SIZE = 64
# With two k-chunks that are masked everywhere, the decode kernel subtracts their maxima, -inf -
# (-inf), which the SFPU does not evaluate to 0, and the output becomes all zeros. A finite value
# avoids that.
MASK_VALUE = -(2.0**127)
# Largest top-k the decode step selects on the device, the multi-core limit of `ttnn.topk`.
MAX_DEVICE_TOP_K = 64

LINEAR_DTYPE = ttnn.bfloat8_b
WEIGHT_CACHE_DTYPE = "bf8"


@dataclass
class GenerationOutput:
    tokens: torch.Tensor
    logits: torch.Tensor | None


@dataclass
class TransformerContext:
    device: ttnn.MeshDevice
    tp_axis: int | None
    ccl_manager: CCLManager | None
    sp_axis: int | None
    fsdp_axis: int | None


@dataclass(frozen=True, kw_only=True)
class TransformerEncoderConfig:
    """Architecture parameters for ``TransformerEncoder``."""

    embed_size: int
    ff_size: int
    head_size: int
    norm_eps: float
    num_heads: int
    num_kv_heads: int
    num_layers: int
    attn_qkv_bias: bool
    attn_out_bias: bool
    vocab_size: int
    rope_config: RopeConfig
    nope_layer_indices: Sequence[int] = ()
    attn_qk_norm: bool = False
    final_norm: bool = True
    final_linear: bool = True
    attn_qkv_dtype: ttnn.DataType = LINEAR_DTYPE


class TransformerEncoder(Module):
    """Transformer encoder model with causal self-attention and support for decode mode.

    Like `torch.nn.TransformerEncoder` it is an encoder model in the sense that it does not feature
    cross-attention. Confusingly it is commonly known as a 'decoder-only' transformer, since it is
    often used autoregressively to generate sequences.
    """

    def __init__(
        self,
        config: TransformerEncoderConfig,
        *,
        device: ttnn.MeshDevice,
        parallel_config: EncoderParallelConfig | None = None,
        ccl_manager: CCLManager | None = None,
    ) -> None:
        super().__init__()

        sp = parallel_config.sequence_parallel if parallel_config is not None else None
        if sp is not None and sp.factor == 1:
            sp = None

        fsdp = parallel_config.fsdp if parallel_config is not None else None
        if fsdp is not None and fsdp.factor == 1:
            fsdp = None

        ctx = TransformerContext(
            device=device,
            tp_axis=parallel_config.tensor_parallel.mesh_axis if parallel_config is not None else None,
            ccl_manager=ccl_manager,
            sp_axis=sp.mesh_axis if sp is not None else None,
            fsdp_axis=fsdp.mesh_axis if fsdp is not None else None,
        )

        if ctx.tp_axis is not None and ctx.ccl_manager is None:
            msg = "ccl_manager must be provided if tensor parallelism is used"
            raise ValueError(msg)

        if ctx.sp_axis is not None:
            if ctx.ccl_manager is None:
                msg = "ccl_manager must be provided if sequence parallelism is used"
                raise ValueError(msg)
            if ctx.sp_axis == ctx.tp_axis:
                msg = "sequence and tensor parallelism cannot share a mesh axis"
                raise ValueError(msg)

        if ctx.fsdp_axis is not None:
            if ctx.ccl_manager is None:
                msg = "ccl_manager must be provided if FSDP is used"
                raise ValueError(msg)
            if ctx.fsdp_axis == ctx.tp_axis:
                msg = "FSDP and tensor parallelism cannot share a mesh axis"
                raise ValueError(msg)

        self._nope_set = set(config.nope_layer_indices)
        for idx in self._nope_set:
            if not 0 <= idx < config.num_layers:
                msg = f"nope_layer_indices entry {idx} out of range [0, {config.num_layers})"
                raise ValueError(msg)

        self.pos_embedding = RotaryEmbedding(head_size=config.head_size, config=config.rope_config)

        self.token_embedding = Embedding(config.vocab_size, config.embed_size, device=ctx.device, mesh_axis=ctx.tp_axis)
        self.layers = ModuleList(
            TransformerEncoderLayer(
                head_size=config.head_size,
                embed_size=config.embed_size,
                ff_size=config.ff_size,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                norm_eps=config.norm_eps,
                attn_qkv_bias=config.attn_qkv_bias,
                attn_out_bias=config.attn_out_bias,
                attn_qk_norm=config.attn_qk_norm,
                attn_qkv_dtype=config.attn_qkv_dtype,
                cache_id=i,
                ctx=ctx,
            )
            for i in range(config.num_layers)
        )

        self.final_norm = (
            TransformerRmsNorm(config.embed_size, eps=config.norm_eps, ctx=ctx) if config.final_norm else None
        )

        # vocab_size is much greater than embed_size
        self.final_linear = (
            ColParallelLinear(
                config.embed_size,
                config.vocab_size,
                bias=False,
                mesh_device=ctx.device,
                mesh_axis=ctx.tp_axis,
                dtype=LINEAR_DTYPE,
            )
            if config.final_linear
            else None
        )

        self.config = config

        tp_factor = device.shape[ctx.tp_axis] if ctx.tp_axis is not None else 1

        self._device = ctx.device
        self._tp_axis = ctx.tp_axis
        self._sp_axis = ctx.sp_axis
        self._sp_factor = device.shape[ctx.sp_axis] if ctx.sp_axis is not None else 1
        self._local_vocab_size = config.vocab_size // tp_factor
        self._ccl_manager = ctx.ccl_manager
        self._cached_position_embeddings = {}
        self._decode_trace: _DecodeTrace | None = None

    # TODO: Remove the mask buffer generation from the function to prevent trace assertion errors.
    @traced_function(device=lambda self: self._device, clone_prep_inputs=False)
    def forward(
        self,
        tokens: ttnn.Tensor,
        *,
        mask: ttnn.Tensor | None = None,
        positions: ttnn.Tensor | None = None,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        vision_embeds: ttnn.Tensor | None = None,
        vision_mask: ttnn.Tensor | None = None,
        deepstack_embeds: Sequence[ttnn.Tensor] = (),
        cache: Cache | None = None,
        skip_final_linear: bool = False,
        output_hidden_states: bool | Container[int] = False,
    ) -> ttnn.Tensor | list[ttnn.Tensor]:
        """Run the stack over `tokens`, filling `cache` for decoding when given.

        Args:
            tokens: Token ids of shape (batch, sequence).
            mask: Attention mask of shape (batch, sequence), 1 where a token may be attended to.
            positions: float32 rope positions of shape (batch, sequence), or (axes, batch,
                sequence) with one row per multimodal rope axis.
            pos_embeds: The cos and sin of the rope for every token, in place of `positions`.
            cache: The k/v cache to fill with the sequence, for `generate`'s decode steps.
            vision_embeds: Embeddings of shape (num_vision_tokens, embed_size) that replace the
                token embeddings of the rows `vision_mask` marks, in sequence order.
            vision_mask: Mask of shape (batch, sequence) marking the vision rows with 1; like
                `mask` it covers the whole sequence.
            deepstack_embeds: One tensor like `vision_embeds` per leading layer, added to the
                vision rows after that layer.
            skip_final_linear: Leaves out the language-model head, returning the final states.
            output_hidden_states: Returns the input of every layer followed by the outputs of the
                final norm and, unless skipped, the head; or only the entries of that list at the
                given indices, so that the others are not kept alive.
        """
        if cache is not None:
            cache.reset()

        batch_size, seq_len = tokens.shape

        def keep_hidden_state(i: int) -> bool:
            return output_hidden_states is True or i in (output_hidden_states or ())

        if (vision_embeds is None) != (vision_mask is None):
            msg = "vision_embeds and vision_mask must be passed together"
            raise ValueError(msg)
        if deepstack_embeds and vision_mask is None:
            msg = "deepstack_embeds needs vision_mask"
            raise ValueError(msg)
        if len(deepstack_embeds) > len(self.layers):
            msg = f"got {len(deepstack_embeds)} deepstack_embeds for {len(self.layers)} layers"
            raise ValueError(msg)
        if vision_mask is not None and batch_size != 1:
            msg = "vision tokens are supported for a single sequence only"
            raise ValueError(msg)

        if self._sp_axis is not None:
            if cache is not None:
                msg = "the cache/decode path does not support sequence parallelism"
                raise ValueError(msg)

            if _padded_sequence_length(seq_len) != seq_len:
                msg = (
                    f"sequence parallelism currently requires an already padded sequence; "
                    f"got local length {seq_len}, expected {_padded_sequence_length(seq_len)}"
                )
                raise ValueError(msg)

        device = tokens.device()

        # There should be no need for a mask when SP is off, but
        # `ttnn.transformer.scaled_dot_product_attention` produces incorrect results when the
        # sequence length is not a multiple of the tile size.
        if mask is None and (seq_len % 32 != 0 or self._sp_axis is not None):
            mask = ttnn.ones(
                [batch_size, seq_len * self._sp_factor],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

        pos_embeds = self._prepare_pos_embeds(positions, pos_embeds, batch_size=batch_size, seq_len=seq_len)

        # padding is only required by `ttnn.transformer.scaled_dot_product_attention` when
        # using an attention mask
        padded_seq_len = seq_len if mask is None else _padded_sequence_length(seq_len)

        tokens = ttnn.pad(tokens, [(0, padded_seq_len - seq_len)], value=0)
        pos_embeds = tuple(ttnn.pad(x, [(0, padded_seq_len - seq_len), (0, 0)], value=0) for x in pos_embeds)

        if mask is not None:
            assert mask.shape == (batch_size, seq_len * self._sp_factor)

            attn_bias = self._prepare_attn_bias(
                mask,
                query_length=seq_len,
                query_pos=0,
                kv_length=mask.shape[1],
                device=device,
            )

            bias_padding = padded_seq_len - seq_len
            attn_bias = ttnn.pad(attn_bias, [(0, bias_padding), (0, bias_padding)], value=MASK_VALUE)
        else:
            attn_bias = None

        del mask

        x = self.token_embedding.forward(tokens)

        if self._tp_axis is not None:
            x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)
            # clone to move out of persistent buffer
            x = ttnn.clone(x)

        if vision_mask is not None:
            vision_index, vision_row_mask = self._vision_rows(vision_mask, padded_seq_len=padded_seq_len)
            x = ttnn.where(vision_row_mask, ttnn.embedding(vision_index, vision_embeds, layout=ttnn.TILE_LAYOUT), x)

        hidden_states = []

        for i, decoder_layer in enumerate(self.layers):
            if keep_hidden_state(i):
                hidden_states.append(x)

            x = decoder_layer.forward(
                x,
                attn_bias=attn_bias,
                pos_embeds=None if i in self._nope_set else pos_embeds,
                cache=cache,
            )

            if i < len(deepstack_embeds):
                x = x + ttnn.embedding(vision_index, deepstack_embeds[i], layout=ttnn.TILE_LAYOUT) * vision_row_mask

            if (i + 1) % 10 == 0:
                ttnn.ReadDeviceProfiler(self._device)

        if cache is not None:
            cache.advance(seq_len)

        if padded_seq_len != seq_len:
            x = x[:, :seq_len, :]
            hidden_states = [h[:, :seq_len, :] for h in hidden_states]

        if self.final_norm is not None:
            x = self.final_norm.forward(x)

        if keep_hidden_state(len(self.layers)):
            hidden_states.append(x)

        if not skip_final_linear and self.final_linear is not None:
            x = self.final_linear.forward(x)

            if keep_hidden_state(len(self.layers) + 1):
                hidden_states.append(x)

        return hidden_states if output_hidden_states is not False else x

    def _decode_step(
        self,
        tokens: ttnn.Tensor,
        *,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor],
        cache: Cache,
        rope_offset: ttnn.Tensor | None = None,
        attn_bias: ttnn.Tensor | None = None,
        device_top_k: _DeviceTopK | None = None,
    ) -> ttnn.Tensor:
        """Runs one decode step and returns the logits, or with `device_top_k`, its output."""
        rope_index = cache.position if rope_offset is None else cache.position + rope_offset
        rope_index = ttnn.reshape(ttnn.typecast(rope_index, ttnn.uint32), [-1, 1])
        cos, sin = pos_embeds
        cos = ttnn.embedding(rope_index, cos, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rope_index, sin, layout=ttnn.TILE_LAYOUT)
        cos = _shard_rope_decode(cos, device=self._device)
        sin = _shard_rope_decode(sin, device=self._device)

        x = self.token_embedding.forward(tokens)

        if self._tp_axis is not None:
            x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)
            # clone to move out of persistent buffer
            x = ttnn.clone(x)

        for i, decoder_layer in enumerate(self.layers):
            x = decoder_layer.forward(
                x,
                attn_bias=attn_bias,
                pos_embeds=None if i in self._nope_set else (cos, sin),
                cache=cache,
                decode=True,
            )

            if (i + 1) % 10 == 0:
                ttnn.ReadDeviceProfiler(self._device)

        if self.final_norm is not None:
            x = self.final_norm.forward(x, decode=True)

        x = self.final_linear.forward(x)

        if device_top_k is not None:
            x = device_top_k.forward(x)

        # Reading the shards one by one costs the host more than a decode step, and a tile-layout
        # read transfers the padding to 32 rows, so they are gathered and untilized here.
        if self._tp_axis is not None:
            x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        return ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

    def _last_token_logits(self, x: ttnn.Tensor, *, index: int) -> ttnn.Tensor:
        """Applies the lm head to row `index` of the normalized prefill states `[batch, seq, embed]`.

        The head runs over the tile-aligned block of rows holding `index` rather than the whole
        sequence, and that slice stays on the tile-aligned fast path.
        """
        _batch_size, seq_len, _embed_size = x.shape
        block = index // ttnn.TILE_SIZE * ttnn.TILE_SIZE

        x = x[:, block : min(block + ttnn.TILE_SIZE, seq_len), :]
        x = self.final_linear.forward(x)

        if self._tp_axis is not None:
            x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        return x[:, index - block]

    def _make_device_top_k(self, top_k: int | None) -> _DeviceTopK | None:
        return (
            _DeviceTopK(
                top_k,
                local_vocab_size=self._local_vocab_size,
                device=self._device,
                tp_axis=self._tp_axis,
            )
            if top_k is not None
            else None
        )

    def _get_decode_trace(self, *, batch_size: int, size: int, masked: bool, top_k: int | None) -> _DecodeTrace:
        """Returns the decode trace for these shapes, building it when the kept one differs."""
        # A trace fixes whether the step takes an attention bias and what it returns, so `masked`
        # and `top_k` are part of the key.
        key = (batch_size, size, masked, top_k)
        if self._decode_trace is not None and self._decode_trace.key == key:
            return self._decode_trace

        self._release_decode_trace()

        cache = Cache(device=self._device, size=size, batch_size=batch_size)
        # `ttnn.embedding` converts a tile-layout table on every call, so the step takes row-major ones.
        pos_embeds = self._get_pos_embeds(start=0, sequence_length=size, layout=ttnn.ROW_MAJOR_LAYOUT)
        device_top_k = self._make_device_top_k(top_k)

        self._decode_trace = _DecodeTrace(
            key=key,
            tracer=Tracer(
                functools.partial(self._decode_step, pos_embeds=pos_embeds, cache=cache, device_top_k=device_top_k),
                device=self._device,
                clone_prep_inputs=False,
            ),
            cache=cache,
            device_top_k=device_top_k,
        )

        return self._decode_trace

    def _release_decode_trace(self) -> None:
        """Releases the decode trace kept by `generate`, with its cache."""
        if self._decode_trace is not None:
            self._decode_trace.release()
            self._decode_trace = None

    def deallocate_weights(self) -> None:
        self._release_decode_trace()
        self._cached_position_embeddings.clear()
        super().deallocate_weights()

    def _get_pos_embeds(
        self,
        start: int,
        sequence_length: int,
        layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        cache_key = (start, sequence_length, layout)
        if cache_key in self._cached_position_embeddings:
            return self._cached_position_embeddings[cache_key]

        if layout != ttnn.TILE_LAYOUT:
            cos, sin = self._get_pos_embeds(start, sequence_length)
            cos = ttnn.to_layout(cos, layout)
            sin = ttnn.to_layout(sin, layout)
        else:
            positions = _make_positions(
                start=start,
                sequence_length=sequence_length * self._sp_factor,
                sp_axis=self._sp_axis,
                device=self._device,
            )
            cos, sin = self.pos_embedding.forward(positions, dtype=self.token_embedding.weight.dtype)

        if self._decode_trace is not None:
            warnings.warn(
                f"caching position embeddings {cache_key} while a decode trace is live",
                stacklevel=2,
            )

        self._cached_position_embeddings[cache_key] = (cos, sin)
        return cos, sin

    def _prepare_pos_embeds(
        self,
        positions: ttnn.Tensor | None,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor] | None,
        *,
        batch_size: int,
        seq_len: int,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Returns cos and sin for `positions`, for the given `pos_embeds`, or for a range from zero."""
        if positions is None:
            return pos_embeds if pos_embeds is not None else self._get_pos_embeds(start=0, sequence_length=seq_len)

        if pos_embeds is not None:
            msg = "positions and pos_embeds are mutually exclusive"
            raise ValueError(msg)

        section = self.config.rope_config.mrope_section
        expected_shape = (batch_size, seq_len * self._sp_factor)
        if section is not None:
            expected_shape = (len(section), *expected_shape)
        if tuple(positions.shape) != expected_shape:
            msg = f"positions must have shape {expected_shape}, got {tuple(positions.shape)}"
            raise ValueError(msg)
        if positions.dtype != ttnn.float32:
            msg = f"positions must be float32, got {positions.dtype}"
            raise ValueError(msg)

        axes = [positions] if section is None else [positions[i] for i in range(len(section))]
        if self._sp_axis is not None:
            axes = [ttnn.mesh_partition(p, dim=1, cluster_axis=self._sp_axis) for p in axes]

        return self.pos_embedding.forward(
            axes[0] if section is None else axes,
            dtype=self.token_embedding.weight.dtype,
        )

    def _vision_rows(self, mask: ttnn.Tensor, *, padded_seq_len: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Numbers the vision rows of the sequence."""
        _, full_len = mask.shape

        mask = ttnn.typecast(mask, ttnn.int32)
        mask = ttnn.pad(mask, [(0, padded_seq_len * self._sp_factor - full_len)], value=0)

        index = ttnn.cumsum(mask, dim=1) - 1

        if self._sp_axis is not None:
            mask = ttnn.mesh_partition(mask, dim=1, cluster_axis=self._sp_axis)
            index = ttnn.mesh_partition(index, dim=1, cluster_axis=self._sp_axis)

        index = ttnn.relu(index)
        index = ttnn.typecast(index, ttnn.uint32)
        index = ttnn.to_layout(index, ttnn.ROW_MAJOR_LAYOUT)

        mask = ttnn.typecast(mask, self.token_embedding.weight.dtype)
        mask = ttnn.unsqueeze(mask, 2)

        return index, mask

    def _prepare_attn_bias(
        self,
        mask: ttnn.Tensor,
        *,
        query_length: int,
        query_pos: int,
        kv_length: int,
        device: ttnn.MeshDevice,
    ) -> ttnn.Tensor:
        """Build the additive attention bias from a padding mask."""
        batch_size = mask.shape[0]

        # Reshape padding mask to [batch, 1, 1, kv_length]
        mask = ttnn.to_layout(mask, ttnn.ROW_MAJOR_LAYOUT)
        mask = ttnn.reshape(mask, [batch_size, 1, 1, kv_length])
        mask = ttnn.to_layout(mask, ttnn.TILE_LAYOUT)
        # Broadcast to [batch, 1, query_length, kv_length]
        mask = ttnn.expand(mask, [batch_size, 1, query_length, kv_length])

        col = _make_positions(start=0, sequence_length=kv_length, device=device)
        row = _make_positions(
            start=query_pos,
            sequence_length=query_length * self._sp_factor,
            device=device,
            sp_axis=self._sp_axis,
        )

        col = ttnn.reshape(col, [1, 1, 1, kv_length])
        row = ttnn.reshape(row, [1, 1, query_length, 1])

        causal = ttnn.typecast(ttnn.le(col, row), ttnn.bfloat16)
        mask = ttnn.logical_and(causal, mask)

        return ttnn.where(mask, 0.0, MASK_VALUE)

    def generate(
        self,
        tokens: torch.Tensor,
        *,
        mask: torch.Tensor | None,
        positions: torch.Tensor | None = None,
        vision_embeds: ttnn.Tensor | None = None,
        vision_mask: torch.Tensor | None = None,
        deepstack_embeds: Sequence[ttnn.Tensor] = (),
        max_length: int,
        cache_length: int | None = None,
        prefill_length: int | None = None,
        eos_tokens: int | Sequence[int] | None,
        top_k: int | None = None,
        top_p: float = 1,
        temperature: float = 1,
        return_logits: bool = False,
        guide: torch.Tensor | None = None,
        traced: bool = False,
    ) -> GenerationOutput:
        """Extends the prompt `tokens` by sampling one token per step on the host, after prefilling.

        Args:
            tokens: Token ids of the prompt, of shape (batch, sequence).
            mask: Attention mask of shape (batch, sequence), 1 where a token may be attended to.
            positions: Rope positions of the prompt, of shape (batch, sequence) or (axes, batch,
                sequence) with one row per multimodal rope axis; a plain range when omitted.
                Decoding continues from the largest position given.
            vision_embeds: Embeddings of shape (num_vision_tokens, embed_size) that replace the
                token embeddings of the rows `vision_mask` marks, in sequence order.
            vision_mask: Mask of shape (batch, sequence) marking the vision rows with 1.
            deepstack_embeds: One tensor like `vision_embeds` per leading layer, added to the
                vision rows after that layer.
            max_length: Length of the prompt and the generated tokens together.
            cache_length: Length the k/v cache and the decode trace are sized for, `max_length`
                when omitted. A fixed value keeps one trace across calls of different lengths.
            prefill_length: Length the prompt is padded to on the right for the prefill, so that
                prompts of different lengths share one set of compiled prefill kernels.
            eos_tokens: Ids that end a sequence; generation stops once every sequence has ended.
            top_k: Number of most likely tokens to sample among, or all of them when omitted.
            top_p: Probability mass of the most likely tokens to sample among.
            temperature: Divisor of the logits before sampling.
            return_logits: Returns the logits of every step, of shape (batch, steps, vocab).
            guide: Token ids of shape (batch, max_length) to take the generated tokens from
                instead of sampling, for teacher forcing.
            traced: Replays the decode step as a trace.
        """
        # The original Llama implementation starts generation after the shortest input, thereby
        # overwriting any padding tokens that are on the right, resuing that space. We use a
        # slightly simpler approach and start generation after the longest input, which is also what
        # the transformers library does.

        if self.final_linear is None:
            msg = "generation needs the language-model head"
            raise ValueError(msg)

        batch_size, input_length = tokens.shape
        device = self._device

        if cache_length is None:
            cache_length = max_length
        elif cache_length < max_length:
            msg = f"cache_length {cache_length} is shorter than max_length {max_length}"
            raise ValueError(msg)

        padded_seq_len = _padded_sequence_length(cache_length - 1)
        padded_seq_len = -(-padded_seq_len // WORKAROUND_MIN_DECODE_CHUNK_SIZE) * WORKAROUND_MIN_DECODE_CHUNK_SIZE

        if prefill_length is None:
            prefill_length = input_length
        elif not input_length <= prefill_length <= padded_seq_len:
            msg = (
                f"prefill_length {prefill_length} must be between the prompt length {input_length} and {padded_seq_len}"
            )
            raise ValueError(msg)

        padding = prefill_length - input_length
        prefill_tokens = torch.nn.functional.pad(tokens, [0, padding])
        if positions is not None:
            positions = torch.nn.functional.pad(positions, [0, padding])
        if vision_mask is not None:
            vision_mask = torch.nn.functional.pad(vision_mask, [0, padding])

        if mask is not None:
            assert mask.shape == tokens.shape
            mask = torch.nn.functional.pad(mask, [0, padded_seq_len - input_length], value=1)
            mask = tensor.from_torch(mask, device=device)

        if eos_tokens is not None:
            if isinstance(eos_tokens, int):
                eos_tokens = [eos_tokens]
            elif len(eos_tokens) == 0:
                eos_tokens = None

        eos_token_tensor = torch.tensor(eos_tokens, dtype=torch.uint32) if eos_tokens else None

        finished = torch.zeros([batch_size], dtype=torch.bool)
        prev_pos = 0

        logits = [] if return_logits else None

        top_k_on_device = top_k is not None and top_k <= MAX_DEVICE_TOP_K and not return_logits

        if guide is None and not top_k_on_device and torch.get_num_threads() > 1:
            warnings.warn(
                f"sampling the whole vocabulary on {torch.get_num_threads()} torch threads leads to "
                "poor performance; call torch.set_num_threads(1)",
                stacklevel=2,
            )

        if traced:
            trace = self._get_decode_trace(
                batch_size=batch_size,
                size=padded_seq_len,
                masked=mask is not None,
                top_k=top_k if top_k_on_device else None,
            )
            cache = trace.cache
            decode_step = trace.tracer
            device_top_k = trace.device_top_k
        else:
            device_top_k = self._make_device_top_k(top_k if top_k_on_device else None)
            cache = Cache(device=device, size=padded_seq_len, batch_size=batch_size)
            decode_step = functools.partial(
                self._decode_step,
                pos_embeds=self._get_pos_embeds(
                    start=0,
                    sequence_length=padded_seq_len,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                ),
                cache=cache,
                device_top_k=device_top_k,
            )

        tt_input_tokens = tensor.from_torch(prefill_tokens, dtype=ttnn.uint32, device=device)
        tt_positions = (
            tensor.from_torch(positions.float(), dtype=ttnn.float32, device=device) if positions is not None else None
        )
        tt_vision_mask = tensor.from_torch(vision_mask, device=device) if vision_mask is not None else None

        if positions is None:
            rope_offset = torch.zeros([batch_size], dtype=torch.int32)
        else:
            last = positions.transpose(0, -2).reshape(batch_size, -1).amax(dim=1)
            rope_offset = (last + 1 - input_length).to(torch.int32)

        tt_rope_offset = tensor.from_torch(
            rope_offset,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            on_host=traced,
        )

        decode_attn_bias = (
            self._prepare_attn_bias(
                mask,
                query_length=padded_seq_len,
                query_pos=0,
                kv_length=padded_seq_len,
                device=device,
            )
            if mask is not None
            else None
        )

        for pos in range(input_length, max_length):
            if prev_pos == 0:
                if tt_positions is None:
                    cos, sin = self._get_pos_embeds(start=0, sequence_length=padded_seq_len)
                    pos_embeds = (cos[:, :prefill_length], sin[:, :prefill_length])
                else:
                    pos_embeds = None

                # The prefill is currently not traced as the performance gain is small.
                x = self.forward(
                    tokens=tt_input_tokens,
                    mask=mask[:, :prefill_length] if mask is not None else None,
                    positions=tt_positions,
                    pos_embeds=pos_embeds,
                    cache=cache,
                    vision_embeds=vision_embeds,
                    vision_mask=tt_vision_mask,
                    deepstack_embeds=deepstack_embeds,
                    skip_final_linear=True,
                )
                # The prefill advanced the cache past its padding
                cache.advance(pos - prefill_length)
                output = self._last_token_logits(x, index=pos - 1)
            else:
                output = decode_step(
                    tt_input_tokens,
                    rope_offset=tt_rope_offset,
                    attn_bias=decode_attn_bias[:, :, prev_pos : prev_pos + 1, :]
                    if decode_attn_bias is not None
                    else None,
                )
                # Outside the step so it's not executed twice due to the tracer's preparation run.
                cache.advance(1)

            torch_output = tensor.to_torch(output).float()

            if logits is not None:
                logits.append(torch_output)

            if guide is not None:
                torch_new_tokens = guide[:, pos : pos + 1].float()
            elif device_top_k is not None and prev_pos != 0:
                values, indices = device_top_k.split(torch_output)
                picked = _sample(torch.softmax(values / temperature, 1), top_k=device_top_k.top_k, top_p=top_p)
                torch_new_tokens = torch.gather(indices, 1, picked.long()).to(torch.uint32)
            else:
                torch_prob = torch.softmax(torch_output / temperature, 1)
                torch_new_tokens = _sample(torch_prob, top_k=top_k, top_p=top_p)

            tokens = torch.cat([tokens, torch_new_tokens.to(tokens.dtype)], dim=1)

            tt_input_tokens = tensor.from_torch(tokens[:, -1], dtype=ttnn.uint32, device=device, on_host=traced)

            if eos_token_tensor is not None:
                finished |= (torch_new_tokens == eos_token_tensor).any(dim=1)
                if finished.all():
                    break

            prev_pos = pos

        if logits is not None:
            logits = torch.stack(logits, dim=1) if logits else torch.zeros([batch_size, 0, self.config.vocab_size])

        return GenerationOutput(tokens=tokens, logits=logits)


class TransformerEncoderLayer(Module):
    def __init__(
        self,
        *,
        head_size: int,
        embed_size: int,
        num_heads: int,
        num_kv_heads: int,
        ff_size: int,
        norm_eps: float,
        attn_qkv_bias: bool,
        attn_out_bias: bool,
        attn_qk_norm: bool,
        attn_qkv_dtype: ttnn.DataType,
        cache_id: Hashable,
        ctx: TransformerContext,
    ) -> None:
        super().__init__()

        self.attn = Attention(
            head_size=head_size,
            embed_size=embed_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            qkv_bias=attn_qkv_bias,
            out_bias=attn_out_bias,
            qk_norm=attn_qk_norm,
            qkv_dtype=attn_qkv_dtype,
            norm_eps=norm_eps,
            cache_id=cache_id,
            ctx=ctx,
        )
        self.ff = FeedForward(embed_size=embed_size, hidden_size=ff_size, ctx=ctx)
        self.attn_norm = TransformerRmsNorm(embed_size, eps=norm_eps, ctx=ctx)
        self.ff_norm = TransformerRmsNorm(embed_size, eps=norm_eps, ctx=ctx)

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        attn_bias: ttnn.Tensor | None = None,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor] | None,
        cache: Cache | None = None,
        decode: bool = False,
    ) -> ttnn.Tensor:
        residual = x
        x = self.attn_norm.forward(x, decode=decode)
        if decode:
            if cache is None:
                msg = "decode requires a cache"
                raise ValueError(msg)
            x = self.attn.forward_decode(x, attn_bias=attn_bias, pos_embeds=pos_embeds, cache=cache)
        else:
            x = self.attn.forward(x, attn_bias=attn_bias, pos_embeds=pos_embeds, cache=cache)
        x = x + residual

        residual = x
        x = self.ff_norm.forward(x, decode=decode)
        x = self.ff.forward(x)
        x = x + residual

        return x


class Attention(Module):
    def __init__(
        self,
        *,
        head_size: int,
        embed_size: int,
        num_heads: int,
        num_kv_heads: int,
        qkv_bias: bool,
        out_bias: bool,
        qk_norm: bool,
        qkv_dtype: ttnn.DataType,
        norm_eps: float,
        cache_id: Hashable,
        ctx: TransformerContext,
    ) -> None:
        super().__init__()

        if ctx.tp_axis is not None:
            assert ctx.ccl_manager is not None

        tp_factor = ctx.device.shape[ctx.tp_axis] if ctx.tp_axis is not None else 1
        group_count = num_kv_heads
        group_size = num_heads // num_kv_heads

        opt_group_count, opt_group_size, split_factor = _optimal_groups(group_count, group_size, tp_factor)
        padded_heads = opt_group_count * opt_group_size

        # heads are distributed across tensor parallel axis
        self.qkv_proj = ColParallelLinear(
            embed_size,
            (padded_heads + 2 * opt_group_count) * head_size,
            bias=qkv_bias,
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis,
            fsdp_mesh_axis=ctx.fsdp_axis,
            ccl_manager=ctx.ccl_manager,
            dtype=qkv_dtype,
        )
        self.o_proj = ColParallelLinear(
            padded_heads * head_size,
            embed_size,
            bias=out_bias,
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis,
            fsdp_mesh_axis=ctx.fsdp_axis,
            ccl_manager=ctx.ccl_manager,
            dtype=LINEAR_DTYPE,
        )

        self._hifi_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            # packer_l1_acc=True,
        )

        # Plain RMSNorm: TransformerRmsNorm's decode path width-shards over the embedding size and
        # expects an interleaved input, but the decode q and k are head-sharded with a head_size width.
        self.q_norm = RMSNorm(head_size, norm_eps=norm_eps, bias=False, mesh_device=ctx.device) if qk_norm else None
        self.k_norm = RMSNorm(head_size, norm_eps=norm_eps, bias=False, mesh_device=ctx.device) if qk_norm else None

        self._head_size = head_size
        self._group_count = group_count
        self._group_size = group_size
        self._num_local_heads = padded_heads // tp_factor
        self._num_local_kv_heads = opt_group_count // tp_factor
        self._group_size_padding = opt_group_size * split_factor - group_size
        self._group_count_padding = opt_group_count - group_count * split_factor
        self._split_factor = split_factor
        self._cache_id = cache_id
        self._tp_axis = ctx.tp_axis
        self._tp_factor = tp_factor
        self._sp_axis = ctx.sp_axis
        self._sp_factor = ctx.device.shape[ctx.sp_axis] if ctx.sp_axis is not None else 1
        self._device = ctx.device
        self._ccl_manager = ctx.ccl_manager

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        def _prepare_qkv(q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor) -> ttnn.Tensor:
            q = q.unflatten(0, [self._group_count, self._group_size, self._head_size])
            k = k.unflatten(0, [self._group_count, 1, self._head_size])
            v = v.unflatten(0, [self._group_count, 1, self._head_size])

            # pad group size
            q = torch_pad(q, self._group_size_padding, dim=1)

            # split groups
            s = self._split_factor
            q = q.flatten(0, 1).unflatten(0, [self._group_count * s, -1])
            k = k.repeat_interleave(s, dim=0)
            v = v.repeat_interleave(s, dim=0)

            # pad group count
            q = torch_pad(q, self._group_count_padding, dim=0)
            k = torch_pad(k, self._group_count_padding, dim=0)
            v = torch_pad(v, self._group_count_padding, dim=0)

            # fuse
            q = q.flatten(0, 1).unflatten(0, [self._tp_factor, self._num_local_heads])
            k = k.flatten(0, 1).unflatten(0, [self._tp_factor, self._num_local_kv_heads])
            v = v.flatten(0, 1).unflatten(0, [self._tp_factor, self._num_local_kv_heads])

            return torch.cat([q, k, v], dim=1).flatten(0, 2)

        if "q_proj.weight" in state and "k_proj.weight" in state and "v_proj.weight" in state:
            state["qkv_proj.weight"] = _prepare_qkv(
                state.pop("q_proj.weight"), state.pop("k_proj.weight"), state.pop("v_proj.weight")
            )

        if "q_proj.bias" in state and "k_proj.bias" in state and "v_proj.bias" in state:
            state["qkv_proj.bias"] = _prepare_qkv(
                state.pop("q_proj.bias"), state.pop("k_proj.bias"), state.pop("v_proj.bias")
            )

        if "o_proj.weight" in state:
            o = state["o_proj.weight"]

            o = o.unflatten(1, [self._group_count, self._group_size, self._head_size])

            # pad group size
            o = torch_pad(o, self._group_size_padding, dim=2)

            # split groups
            o = o.flatten(1, 2).unflatten(1, [self._group_count * self._split_factor, -1])

            # pad group count
            o = torch_pad(o, self._group_count_padding, dim=1)

            state["o_proj.weight"] = o.flatten(1, 3)

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        attn_bias: ttnn.Tensor | None,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor] | None,
        cache: Cache | None = None,
    ) -> ttnn.Tensor:
        batch_size, padded_q_seq_len, _ = x.shape

        if attn_bias is not None:
            kv_len = padded_q_seq_len * self._sp_factor
            expected_shape = (
                (1, 1, padded_q_seq_len, kv_len),
                (batch_size, 1, padded_q_seq_len, kv_len),
            )
            assert (
                attn_bias.shape in expected_shape
            ), f"unexpected attn_bias shape {tuple(attn_bias.shape)}, expected one of {list(expected_shape)}"

        x = self.qkv_proj.forward(x)

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            ttnn.unsqueeze(x, 1),
            num_heads=self._num_local_heads,
            num_kv_heads=self._num_local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # q shape: batch_size num_local_heads    padded_q_seq_len head_size
        # k shape: batch_size num_local_kv_heads padded_q_seq_len head_size
        # v shape: batch_size num_local_kv_heads padded_q_seq_len head_size

        if self.q_norm is not None:
            q = self.q_norm.forward(q, compute_kernel_config=self._hifi_compute_kernel_config)
        if self.k_norm is not None:
            k = self.k_norm.forward(k, compute_kernel_config=self._hifi_compute_kernel_config)

        if pos_embeds is not None:
            cos, sin = pos_embeds
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

        if self._sp_axis is not None:
            k = self._ccl_manager.all_gather_persistent_buffer(k, dim=2, mesh_axis=self._sp_axis, use_hyperparams=True)
            v = self._ccl_manager.all_gather_persistent_buffer(v, dim=2, mesh_axis=self._sp_axis, use_hyperparams=True)

        if cache is not None:
            cache.prefill(self._cache_id, k, v)

        kv_seq_len = k.shape[2]
        if attn_bias is not None:
            padded_kv_seq_len = -(-kv_seq_len // 32) * 32
            k = ttnn.pad(k, [(0, padded_kv_seq_len - kv_seq_len), (0, 0)], value=0)
            v = ttnn.pad(v, [(0, padded_kv_seq_len - kv_seq_len), (0, 0)], value=0)
        else:
            padded_kv_seq_len = kv_seq_len

        x = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            is_causal=attn_bias is None,
            program_config=self._sdpa_program_config(padded_q_seq_len, padded_kv_seq_len),
            compute_kernel_config=self._hifi_compute_kernel_config,
        )
        del q, k, v

        x = ttnn.transformer.concatenate_heads(x)

        if self._tp_axis is not None:
            x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        x = self.o_proj.forward(x)

        if self._tp_axis is not None:
            x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        return x

    def forward_decode(
        self,
        x: ttnn.Tensor,
        *,
        attn_bias: ttnn.Tensor | None,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor] | None,
        cache: Cache,
    ) -> ttnn.Tensor:
        if self._sp_axis is not None:
            msg = "decode mode does not support sequence parallelism"
            raise ValueError(msg)

        if len(x.shape) != 2:
            msg = "decode mode expects input shape of (batch_size, embed_size)"
            raise ValueError(msg)

        batch_size, _embed_size = x.shape
        seq_len = 1

        if attn_bias is not None:
            assert attn_bias.shape in (
                (1, 1, seq_len, cache.size),
                (batch_size, 1, seq_len, cache.size),
            )
            attn_bias = ttnn.repeat(attn_bias, [1, 1, self._num_local_heads, 1])

        x = self.qkv_proj.forward(x)

        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            x.reshape([1, 1, batch_size, -1]),
            num_heads=self._num_local_heads,
            num_kv_heads=self._num_local_kv_heads,
        )
        # q shape: 1 batch_size num_local_heads    head_size
        # k shape: 1 batch_size num_local_kv_heads head_size
        # v shape: 1 batch_size num_local_kv_heads head_size

        if self.q_norm is not None:
            q = _norm_in_dram(self.q_norm, q, compute_kernel_config=self._hifi_compute_kernel_config)
        if self.k_norm is not None:
            k = _norm_in_dram(self.k_norm, k, compute_kernel_config=self._hifi_compute_kernel_config)

        if pos_embeds is not None:
            cos, sin = pos_embeds
            q = ttnn.experimental.rotary_embedding_hf(q, cos, sin, is_decode_mode=True)
            k = ttnn.experimental.rotary_embedding_hf(k, cos, sin, is_decode_mode=True)

        k, v = cache.update(self._cache_id, k, v)

        # q shape: 1 batch_size num_local_heads               head_size
        # k shape:   batch_size num_local_kv_heads kv_seq_len head_size
        # v shape:   batch_size num_local_kv_heads kv_seq_len head_size

        # TODO: This is a bit inaccurate when supplied with a bias. When replaced with a manual SDPA
        # implementation, the bias works fine.
        x = ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            k,
            v,
            cur_pos_tensor=cache.position,
            attn_mask=attn_bias,
            is_causal=attn_bias is None,
            program_config=self._sdpa_decode_program_config(k.shape[2]),
            compute_kernel_config=self._hifi_compute_kernel_config,
        )
        del q, k, v

        memory_config = ttnn.create_sharded_memory_config(
            shape=[-(-self._num_local_heads // 32) * 32, self._head_size],
            core_grid=ttnn.CoreRangeSet({_num_to_corerange(batch_size)}),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        x = ttnn.to_memory_config(x, memory_config)
        x = ttnn.experimental.nlp_concat_heads_decode(x, num_heads=self._num_local_heads)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        if self._tp_axis is not None:
            x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        x = self.o_proj.forward(x)

        x = ttnn.squeeze(ttnn.squeeze(x, 0), 0)[:batch_size]

        if self._tp_axis is not None:
            x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        return x

    def _sdpa_program_config(self, q_len: int, kv_len: int) -> ttnn.SDPAProgramConfig:
        grid_size = self._device.compute_with_storage_grid_size()

        q_len = -(-q_len // 32) * 32
        q_chunk_size = min(q_len, MAX_CHUNK_SIZE)

        kv_len = -(-kv_len // 32) * 32
        kv_chunk_size = min(kv_len, MAX_CHUNK_SIZE)

        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,
            q_chunk_size=q_chunk_size,
            k_chunk_size=kv_chunk_size,
            exp_approx_mode=False,
        )

    def _sdpa_decode_program_config(self, kv_len: int) -> ttnn.SDPAProgramConfig:
        # The decode kernel requires a power-of-two k-chunk that divides the cache length.
        kv_chunk_size = MAX_CHUNK_SIZE
        while kv_len % kv_chunk_size != 0:
            kv_chunk_size //= 2

        if kv_chunk_size < WORKAROUND_MIN_DECODE_CHUNK_SIZE:
            msg = f"cache length must be a multiple of {WORKAROUND_MIN_DECODE_CHUNK_SIZE}, got {kv_len}"
            raise ValueError(msg)

        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self._device.compute_with_storage_grid_size(),
            q_chunk_size=32,
            k_chunk_size=kv_chunk_size,
            exp_approx_mode=False,
        )


class FeedForward(Module):
    def __init__(self, embed_size: int, hidden_size: int, ctx: TransformerContext) -> None:
        super().__init__()

        if ctx.tp_axis is not None:
            assert ctx.ccl_manager is not None

        # hidden_size is much greater than embed_size
        self.gate = ColParallelLinear(
            embed_size,
            hidden_size,
            bias=False,
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis,
            fsdp_mesh_axis=ctx.fsdp_axis,
            ccl_manager=ctx.ccl_manager,
            dtype=LINEAR_DTYPE,
        )
        self.linear_in = ColParallelLinear(
            embed_size,
            hidden_size,
            bias=False,
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis,
            fsdp_mesh_axis=ctx.fsdp_axis,
            ccl_manager=ctx.ccl_manager,
            dtype=LINEAR_DTYPE,
        )
        self.linear_out = RowParallelLinear(
            hidden_size,
            embed_size,
            bias=False,
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis,
            fsdp_mesh_axis=ctx.fsdp_axis,
            ccl_manager=ctx.ccl_manager,
            dtype=LINEAR_DTYPE,
        )

        self._act_fn = ttnn.silu

        self._ccl_manager = ctx.ccl_manager
        self._tp_axis = ctx.tp_axis

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self._act_fn(self.gate.forward(x)) * self.linear_in.forward(x)
        x = self.linear_out.forward(x)

        if self._tp_axis is not None:
            x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        return x


class TransformerRmsNorm(Module):
    def __init__(self, num_channels: int, *, eps: float, ctx: TransformerContext) -> None:
        super().__init__()

        self.inner = RMSNorm(
            num_channels,
            norm_eps=eps,
            bias=False,
            mesh_device=ctx.device,
        )

        self.eps = eps
        self._num_channels = num_channels
        self._grid_size = ctx.device.compute_with_storage_grid_size()

        self._compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        state["inner.weight"] = state.pop("weight")

    def forward(self, x: ttnn.Tensor, *, decode: bool = False) -> ttnn.Tensor:
        if not decode:
            return self.inner.forward(x, compute_kernel_config=self._compute_kernel_config)

        # Sharded config taken from tt_transformers (`ModelArgs.create_sharded_norm_config`).
        rows = x.padded_shape[-2]
        grid = self._decode_grid()
        block_w = self._num_channels // ttnn.TILE_SIZE // grid.num_cores

        memory_config = ttnn.create_sharded_memory_config(
            shape=[rows, block_w * ttnn.TILE_SIZE],
            core_grid=grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )
        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[grid.x, grid.y],
            subblock_w=max(w for w in (4, 3, 2, 1) if block_w % w == 0),
            block_h=rows // ttnn.TILE_SIZE,
            block_w=block_w,
            inplace=False,
        )

        x = ttnn.interleaved_to_sharded(x, memory_config)
        x = self.inner.forward(x, compute_kernel_config=self._compute_kernel_config, program_config=program_config)
        return ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)

    def _decode_grid(self) -> ttnn.CoreGrid:
        """Picks the core grid closest to 32 cores over which the channel tiles divide evenly."""
        tiles = self._num_channels // ttnn.TILE_SIZE
        candidates = []
        for rows in range(1, self._grid_size.y + 1):
            for cols in range(1, self._grid_size.x + 1):
                if tiles % (rows * cols) == 0:
                    candidates.append((abs(rows * cols - 32), -rows * cols, rows, cols))
        _, _, rows, cols = min(candidates)
        return ttnn.CoreGrid(y=rows, x=cols)


class Cache:
    def __init__(self, *, device: ttnn.MeshDevice, size: int, batch_size: int) -> None:
        self._k_cache = {}
        self._v_cache = {}

        self._device = device
        self._size = size
        self._batch_size = batch_size
        self._position = tensor.from_torch(
            torch.zeros([batch_size], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )

    @property
    def size(self) -> int:
        return self._size

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def position(self) -> ttnn.Tensor:
        """The `[batch]` int32 position on the device the decode kernels take."""
        return self._position

    def prefill(self, cache_id: Hashable, k: ttnn.Tensor, v: ttnn.Tensor) -> None:
        batch_size, local_kv_heads, _seq_len, head_dim = k.shape
        if batch_size != self._batch_size:
            msg = f"the cache holds {self._batch_size} sequences, got {batch_size}"
            raise ValueError(msg)

        if cache_id not in self._k_cache:
            self._k_cache[cache_id] = ttnn.zeros(
                [batch_size, local_kv_heads, self._size, head_dim],
                dtype=k.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self._device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self._v_cache[cache_id] = ttnn.zeros(
                [batch_size, local_kv_heads, self._size, head_dim],
                dtype=v.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self._device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        k_cache = self._k_cache[cache_id]
        v_cache = self._v_cache[cache_id]

        for batch_idx in range(batch_size):
            ttnn.fill_cache(k_cache, k[batch_idx : batch_idx + 1], batch_idx)
            ttnn.fill_cache(v_cache, v[batch_idx : batch_idx + 1], batch_idx)

    def update(self, cache_id: Hashable, k: ttnn.Tensor, v: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        one, _batch_size, _local_kv_heads, _head_dim = k.shape
        assert one == 1

        k_cache = self._k_cache[cache_id]
        v_cache = self._v_cache[cache_id]

        ttnn.experimental.paged_update_cache(k_cache, k, update_idxs_tensor=self._position)
        ttnn.experimental.paged_update_cache(v_cache, v, update_idxs_tensor=self._position)

        return k_cache, v_cache

    def advance(self, distance: int) -> None:
        ttnn.copy(self._position + distance, self._position)

    def reset(self) -> None:
        """Rewinds to the start for a new sequence; the cache tensors stay allocated."""
        ttnn.fill(self._position, 0, output_tensor=self._position)


@dataclass
class _DecodeTrace:
    key: tuple[int, int, bool, int | None]  # [batch size, cache size, masked, device top-k]
    tracer: Tracer
    cache: Cache
    device_top_k: _DeviceTopK | None

    def release(self) -> None:
        self.tracer.release_trace()


def _apply_rope(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
    n, _heads, seq, dim = x.shape

    assert cos.shape in ((n, seq, dim), (1, seq, dim))
    assert cos.shape == sin.shape

    return x * ttnn.unsqueeze(cos, 1) + _rotate_half(x) * ttnn.unsqueeze(sin, 1)


def _shard_rope_decode(x: ttnn.Tensor, *, device: ttnn.MeshDevice) -> ttnn.Tensor:
    """Shards the cos or sin rows of a decode step for the fused rope op."""
    batch, _one, head_size = x.shape

    grid = ttnn.num_cores_to_corerangeset(batch, device.compute_with_storage_grid_size(), row_wise=True)
    memory_config = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, head_size),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    x = ttnn.reshape(x, [1, batch, 1, head_size])
    return ttnn.interleaved_to_sharded(x, memory_config)


def _rotate_half(x: ttnn.Tensor) -> ttnn.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ttnn.concat([ttnn.neg(x2), x1], dim=-1)


def _norm_in_dram(
    norm: RMSNorm, x: ttnn.Tensor, *, compute_kernel_config: ttnn.DeviceComputeKernelConfig
) -> ttnn.Tensor:
    memory_config = x.memory_config()
    x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
    x = norm.forward(x, compute_kernel_config=compute_kernel_config)
    return ttnn.to_memory_config(x, memory_config)


def _make_positions(
    *, start: int, sequence_length: int, device: ttnn.MeshDevice, sp_axis: int | None = None
) -> ttnn.Tensor:
    pos = tensor.arange(start, start + sequence_length, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    pos = ttnn.unsqueeze(pos, 0)

    # If the attention mask had holes, i.e., contained zeros between ones, this would have to be
    # done instead:
    # mask = ttnn.typecast(mask, ttnn.float32)
    # # equivalent to: pos = mask.cumsum(1) - 1; pos.masked_fill_(mask == 0, 1)
    # pos = (ttnn.cumsum(mask, 1) - 2) * mask + 1
    # pos = pos[:, start:]

    if sp_axis is not None:
        pos = ttnn.mesh_partition(pos, dim=1, cluster_axis=sp_axis)

    return pos


class _DeviceTopK:
    """Selects the top-k logits of a decode step on the device, so that only those are read back.

    `ttnn.topk` is fast only on a power-of-two width with 16-bit indices, so each device's local
    vocabulary is split into chunks of at most `2**15` columns, each padded up to a power of two.
    """

    def __init__(self, top_k: int, *, local_vocab_size: int, device: ttnn.MeshDevice, tp_axis: int | None) -> None:
        if not 0 < top_k <= MAX_DEVICE_TOP_K:
            msg = f"top_k must be in [1, {MAX_DEVICE_TOP_K}], got {top_k}"
            raise ValueError(msg)

        chunk_size = 2**15
        tp_factor = device.shape[tp_axis] if tp_axis is not None else 1

        self.top_k = top_k
        self._local_vocab_size = local_vocab_size
        self._chunks = [
            (start, min(chunk_size, 1 << (min(chunk_size, local_vocab_size - start) - 1).bit_length()))
            for start in range(0, local_vocab_size, chunk_size)
        ]

        # Added to the output of `forward`, it turns each index within a chunk into a vocabulary
        # index: zero at the logits, and where the chunk starts in the vocabulary at the indices.
        starts = torch.tensor([start for start, _ in self._chunks])
        offsets = torch.zeros([tp_factor, len(starts), 2, top_k])
        offsets[:, :, 1] = (torch.arange(tp_factor).reshape(-1, 1) * local_vocab_size + starts).unsqueeze(-1)
        self._index_offsets = tensor.from_torch(
            offsets.reshape(1, -1), device=device, dtype=ttnn.float32, mesh_axes=[None, tp_axis]
        )

    def forward(self, logits: ttnn.Tensor) -> ttnn.Tensor:
        """Takes a device's local logits `[batch, local vocab]` and returns its candidates.

        The result is a float32 tensor `[batch, chunks * 2 * top_k]`. For every chunk, it holds the
        `top_k` largest logits, followed by their vocabulary indices. The indices are float32 so
        that one gather across the devices carries both.
        """
        results = []

        for start, width in self._chunks:
            chunk = logits[:, start : min(start + width, self._local_vocab_size)]
            if chunk.shape[-1] != width:
                # `ttnn.topk` is several times faster on a power-of-two width.
                chunk = ttnn.pad(chunk, [(0, 0), (0, width - chunk.shape[-1])], value=MASK_VALUE)
            values, indices = ttnn.topk(chunk, k=self.top_k, dim=-1)
            results += [ttnn.typecast(values, ttnn.float32), ttnn.typecast(indices, ttnn.float32)]

        return ttnn.concat(results, dim=-1) + self._index_offsets

    def split(self, output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits the candidates of all devices into logits and vocabulary indices, `[batch, n]` each."""
        batch_size = output.shape[0]
        output = output.reshape(batch_size, -1, 2, self.top_k)
        values = output[:, :, 0].reshape(batch_size, -1)
        indices = output[:, :, 1].long().reshape(batch_size, -1)
        return values, indices


def _sample(prob: torch.Tensor, *, top_k: int | None = None, top_p: float = 1, num_samples: int = 1) -> torch.Tensor:
    assert 0 < top_p <= 1

    if top_k is None:
        top_k = prob.shape[-1]
    else:
        assert top_k > 0
        top_k = min(top_k, prob.shape[-1])

    output_shape = [*prob.shape[:-1], num_samples]
    prob = prob.reshape(-1, prob.shape[-1]).float()

    values, indices = torch.topk(prob, k=top_k, dim=-1)
    values = values / values.sum(dim=1, keepdim=True)

    ignore = values.cumsum(1) - values >= top_p
    values[ignore] = 0

    picked = torch.multinomial(values, num_samples=num_samples, replacement=True)
    return torch.gather(indices, 1, picked).view(output_shape).to(torch.uint32)


def _optimal_groups(group_count: int, group_size: int, device_count: int) -> tuple[int, int, int]:
    # In order to distribute heads evenly on devices, three operations are possibly performed:
    # 1. Pad to increase group size.
    # 2. Pad to increase group count (= number of key/value heads).
    # 3. Split groups into smaller groups defined by a split factor.
    # For a particular split factor, padding sizes follow from the requirements that the padded
    # group size must be divisible by this factor and the new group count must be divisible by the
    # device count. We choose this factor such that memory requirments are minimized.

    best_split_factor = 1
    best_size = math.inf
    best_group_count = group_count
    best_group_size = group_size

    for s in range(1, group_size + 1):
        new_group_size = -(-group_size // s)  # = ceil(group_size / s)
        new_group_count = -(-group_count * s // device_count) * device_count

        # query heads + 2 * key/value heads
        size = new_group_size * new_group_count + 2 * new_group_count

        if size < best_size:
            best_size = size
            best_split_factor = s
            best_group_count = new_group_count
            best_group_size = new_group_size

    return best_group_count, best_group_size, best_split_factor


@dataclass
class StateConversion:
    rename: Sequence[tuple[str, str]] | None = None
    remove: Sequence[str] | None = None

    def convert(self, state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        in_ = dict(state_dict)
        out = {}

        compiled = [(re.compile(pattern), template) for (pattern, template) in self.rename or []]

        for k in list(in_):
            transformed = False
            for pattern, t in compiled:
                new_k, count = pattern.subn(t, k, count=1)
                if count == 1:
                    if transformed:
                        msg = f"multiple renames for key: {k}"
                        raise RuntimeError(msg)
                    if new_k in out:
                        msg = f"key collision: {new_k}"
                        raise RuntimeError(msg)
                    out[new_k] = in_.pop(k)
                    transformed = True

            for pattern in self.remove or []:
                if re.search(pattern, k):
                    if transformed:
                        msg = f"multiple renames/removes for key: {k}"
                        raise RuntimeError(msg)
                    in_.pop(k)
                    transformed = True

        if in_:
            warnings.warn(f"unprocessed keys remain: {', '.join(in_.keys())}", stacklevel=2)

        return {**in_, **out}


def _padded_sequence_length(sequence_length: int) -> int:
    if sequence_length < MAX_CHUNK_SIZE:
        # make sequence length a multiple of tile size
        return -(-sequence_length // 32) * 32

    # make sequence length a multiple of MAX_CHUNK_SIZE
    return -(-sequence_length // MAX_CHUNK_SIZE) * MAX_CHUNK_SIZE


# copied from tt_transformers
def _num_to_corerange(x: int) -> ttnn.CoreRange:
    assert x < 8 or x % 8 == 0
    num_x = min(x, 8)
    num_y = x // num_x
    assert num_x * num_y == x
    return ttnn.CoreRange(
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(num_x - 1, num_y - 1),
    )
