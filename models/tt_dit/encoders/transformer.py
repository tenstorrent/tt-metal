# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import ttnn

from ..blocks.rope import RopeConfig, RotaryEmbedding
from ..layers.embeddings import Embedding
from ..layers.linear import ColParallelLinear, RowParallelLinear
from ..layers.module import Module, ModuleList
from ..layers.normalization import RMSNorm
from ..parallel.config import EncoderParallelConfig
from ..parallel.manager import CCLManager
from ..utils import tensor

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

MAX_CHUNK_SIZE = 128


@dataclass
class GenerationOutput:
    tokens: ttnn.Tensor
    logits: list[ttnn.Tensor] | None


@dataclass
class TransformerContext:
    device: ttnn.MeshDevice
    tp_axis: int | None
    ccl_manager: CCLManager | None


class TransformerEncoder(Module):
    """Transformer encoder model with causal self-attention and support for decode mode.

    Like `torch.nn.TransformerEncoder` it is an encoder model in the sense that it does not feature
    cross-attention. Confusingly it is commonly known as a 'decoder-only' transformer, since it is
    often used autoregressively to generate sequences.
    """

    def __init__(
        self,
        *,
        embed_size: int,
        ff_size: int,
        head_size: int,
        norm_eps: float,
        num_heads: int,
        num_kv_heads: int,
        num_layers: int,
        attn_qkv_bias: bool,
        attn_out_bias: bool,
        vocab_size: int,
        rope_config: RopeConfig,
        device: ttnn.MeshDevice,
        parallel_config: EncoderParallelConfig | None = None,
        ccl_manager: CCLManager | None = None,
    ) -> None:
        super().__init__()

        ctx = TransformerContext(
            device=device,
            tp_axis=parallel_config.tensor_parallel.mesh_axis if parallel_config is not None else None,
            ccl_manager=ccl_manager,
        )

        if ctx.tp_axis is not None and ctx.ccl_manager is None:
            msg = "ccl_manager must be provided if tensor parallelism is used"
            raise ValueError(msg)

        self.pos_embedding = RotaryEmbedding(head_size=head_size, config=rope_config)

        self.token_embedding = Embedding(vocab_size, embed_size, device=ctx.device, mesh_axis=ctx.tp_axis)
        self.layers = ModuleList(
            TransformerEncoderLayer(
                head_size=head_size,
                embed_size=embed_size,
                ff_size=ff_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                norm_eps=norm_eps,
                attn_qkv_bias=attn_qkv_bias,
                attn_out_bias=attn_out_bias,
                cache_id=i,
                ctx=ctx,
            )
            for i in range(num_layers)
        )

        self.final_norm = TransformerRmsNorm(embed_size, eps=norm_eps, ctx=ctx)

        # vocab_size is much greater than embed_size
        self.final_linear = ColParallelLinear(
            embed_size, vocab_size, bias=False, mesh_device=ctx.device, mesh_axis=ctx.tp_axis
        )

        self.embed_size = embed_size
        self.ff_size = ff_size
        self.head_size = head_size
        self.norm_eps = norm_eps
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_layers = num_layers
        self.attn_qkv_bias = attn_qkv_bias
        self.attn_out_bias = attn_out_bias
        self.vocab_size = vocab_size
        self.rope_config = rope_config

        self._device = ctx.device
        self._tp_axis = ctx.tp_axis
        self._ccl_manager = ctx.ccl_manager

    def forward(
        self,
        tokens: ttnn.Tensor,
        *,
        mask: ttnn.Tensor | None = None,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        cache: Cache | None = None,
        skip_final_linear: bool = False,
        output_hidden_states: bool = False,
    ) -> ttnn.Tensor | list[ttnn.Tensor]:
        if cache is not None and cache.position != 0:
            (batch_size,) = tokens.shape
            seq_len = 1
        else:
            batch_size, seq_len = tokens.shape

        device = tokens.device()
        dtype = self.token_embedding.weight.dtype

        start_pos = cache.position if cache is not None else 0

        # There should be no need for a mask when start_pos is zero, but
        # `ttnn.transformer.scaled_dot_product_attention` produces incorrect results when the
        # sequence length is not a multiple of the tile size.
        if mask is None and start_pos == 0 and seq_len % 32 != 0:
            mask = ttnn.ones(
                [batch_size, start_pos + seq_len],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

        if pos_embeds is None:
            pos = _make_positions(start=start_pos, sequence_length=seq_len, device=device)
            pos_embeds = self.pos_embedding.forward(pos, dtype=dtype)

        # padding is only required by `ttnn.transformer.scaled_dot_product_attention` when
        # using an attention mask
        padded_seq_len = seq_len if mask is None or start_pos != 0 else _padded_sequence_length(seq_len)

        tokens = ttnn.pad(tokens, [(0, padded_seq_len - seq_len)], value=0)
        pos_embeds = tuple(ttnn.pad(x, [(0, padded_seq_len - seq_len), (0, 0)], value=0) for x in pos_embeds)

        if mask is not None:
            assert mask.shape[0] == batch_size
            if start_pos == 0:
                assert mask.shape[1] == seq_len

            attn_bias = _prepare_attn_bias(mask, query_length=seq_len, query_pos=start_pos)

            if start_pos == 0:
                bias_padding = padded_seq_len - seq_len
                attn_bias = ttnn.pad(attn_bias, [(0, bias_padding), (0, bias_padding)], value=-math.inf)
        else:
            attn_bias = None

        del mask

        x = self.token_embedding.forward(tokens)

        if self._tp_axis is not None:
            x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)
            # clone to move out of persistent buffer
            x = ttnn.clone(x)

        hidden_states = []

        for i, decoder_layer in enumerate(self.layers, start=1):
            if output_hidden_states:
                hidden_states.append(x)

            x = decoder_layer.forward(
                x,
                attn_bias=attn_bias,
                pos_embeds=pos_embeds,
                cache=cache,
            )

            if i % 10 == 0:
                ttnn.ReadDeviceProfiler(self._device)

        if cache is not None:
            cache.advance(seq_len)

        if padded_seq_len != seq_len:
            x = x[:, :seq_len, :]

        x = self.final_norm.forward(x)

        if output_hidden_states:
            hidden_states.append(x)

        if not skip_final_linear:
            x = self.final_linear.forward(x)

            if output_hidden_states:
                hidden_states.append(x)

        return hidden_states if output_hidden_states else x

    def generate(
        self,
        tokens: ttnn.Tensor,
        *,
        mask: ttnn.Tensor | None,
        max_length: int,
        eos_tokens: int | Sequence[int] | None,
        top_k: int | None = None,
        top_p: float = 1,
        temperature: float = 1,
        use_cache: bool = True,
        return_logits: bool = False,
        guide: torch.Tensor | None = None,
    ) -> GenerationOutput:
        # The original Llama implementation starts generation after the shortest input, thereby
        # overwriting any padding tokens that are on the right, resuing that space. We use a
        # slightly simpler approach and start generation after the longest input, which is also what
        # the transformers library does.

        batch_size, input_length = tokens.shape
        device = tokens.device()
        dtype = self.token_embedding.weight.dtype

        padded_seq_len = _padded_sequence_length(max_length - 1)

        if mask is not None:
            assert mask.shape == tokens.shape
            mask = ttnn.pad(mask, [(0, padded_seq_len - input_length)], value=1)

        if eos_tokens is not None:
            if isinstance(eos_tokens, int):
                eos_tokens = [eos_tokens]
            elif len(eos_tokens) == 0:
                eos_tokens = None

        eos_token_tensor = torch.tensor(eos_tokens, dtype=torch.uint32) if eos_tokens else None

        positions = _make_positions(start=0, sequence_length=padded_seq_len, device=device)
        cos, sin = self.pos_embedding.forward(positions, dtype=dtype)

        finished = torch.zeros([batch_size], dtype=torch.bool)
        cache = Cache(device=device, size=padded_seq_len) if use_cache else None
        prev_pos = 0

        logits = [] if return_logits else None

        for pos in range(input_length, max_length):
            current_logits = self.forward(
                tokens=tokens if prev_pos == 0 else tokens[:, -1],
                mask=mask[:, :pos] if prev_pos == 0 and mask is not None else mask,
                pos_embeds=(cos[:, prev_pos:pos], sin[:, prev_pos:pos]),
                cache=cache,
            )
            if prev_pos == 0:
                current_logits = current_logits[:, -1]

            if guide is not None:
                torch_new_tokens = guide[:, pos : pos + 1].float()
            else:
                torch_current_logits = tensor.to_torch(current_logits).float()
                torch_prob = torch.softmax(torch_current_logits / temperature, 1)
                torch_new_tokens = _sample(torch_prob, top_k=top_k, top_p=top_p)

            new_tokens = tensor.from_torch(torch_new_tokens, dtype=tokens.dtype, device=device)

            tokens = ttnn.concat([tokens, new_tokens], dim=1)

            if logits is not None:
                logits.append(ttnn.squeeze(current_logits, 1))

            if eos_token_tensor is not None:
                finished |= (torch_new_tokens == eos_token_tensor).any(dim=1)
                if finished.all():
                    break

            if cache is not None:
                prev_pos = pos

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
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor],
        cache: Cache | None = None,
    ) -> ttnn.Tensor:
        residual = x
        x = self.attn_norm.forward(x)
        x = self.attn.forward(x, attn_bias=attn_bias, pos_embeds=pos_embeds, cache=cache)
        x = x + residual

        residual = x
        x = self.ff_norm.forward(x)
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
        )
        self.o_proj = ColParallelLinear(
            padded_heads * head_size, embed_size, bias=out_bias, mesh_device=ctx.device, mesh_axis=ctx.tp_axis
        )

        self._sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            # packer_l1_acc=True,
        )

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
        self._device = ctx.device
        self._ccl_manager = ctx.ccl_manager

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        def _prepare_qkv(q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor) -> ttnn.Tensor:
            q = q.unflatten(0, [self._group_count, self._group_size, self._head_size])
            k = k.unflatten(0, [self._group_count, 1, self._head_size])
            v = v.unflatten(0, [self._group_count, 1, self._head_size])

            # pad group size
            q = _pad(q, self._group_size_padding, dim=1)

            # split groups
            s = self._split_factor
            q = q.flatten(0, 1).unflatten(0, [self._group_count * s, -1])
            k = k.repeat_interleave(s, dim=0)
            v = v.repeat_interleave(s, dim=0)

            # pad group count
            q = _pad(q, self._group_count_padding, dim=0)
            k = _pad(k, self._group_count_padding, dim=0)
            v = _pad(v, self._group_count_padding, dim=0)

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
            o = _pad(o, self._group_size_padding, dim=2)

            # split groups
            o = o.flatten(1, 2).unflatten(1, [self._group_count * self._split_factor, -1])

            # pad group count
            o = _pad(o, self._group_count_padding, dim=1)

            state["o_proj.weight"] = o.flatten(1, 3)

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        attn_bias: ttnn.Tensor | None,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor],
        cache: Cache | None = None,
    ) -> ttnn.Tensor:
        if cache is not None and cache.position != 0:
            return self.forward_decode(x, attn_bias=attn_bias, pos_embeds=pos_embeds, cache=cache)

        batch_size, padded_q_seq_len, _ = x.shape

        if attn_bias is not None:
            expected_shape = (
                (1, 1, padded_q_seq_len, padded_q_seq_len),
                (batch_size, 1, padded_q_seq_len, padded_q_seq_len),
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

        cos, sin = pos_embeds
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

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
            program_config=self._sdpa_program_config(padded_q_seq_len, padded_q_seq_len),
            compute_kernel_config=self._sdpa_compute_kernel_config,
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
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor],
        cache: Cache,
    ) -> ttnn.Tensor:
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

        cos, sin = pos_embeds
        q = _apply_rope_decode(q, cos, sin)
        k = _apply_rope_decode(k, cos, sin)

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
            cur_pos=[cache.position] * batch_size,
            attn_mask=attn_bias,
            is_causal=attn_bias is None,
            program_config=self._sdpa_program_config(seq_len, k.shape[2]),
            compute_kernel_config=self._sdpa_compute_kernel_config,
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


class FeedForward(Module):
    def __init__(self, embed_size: int, hidden_size: int, ctx: TransformerContext) -> None:
        super().__init__()

        if ctx.tp_axis is not None:
            assert ctx.ccl_manager is not None

        # hidden_size is much greater than embed_size
        self.gate = ColParallelLinear(
            embed_size, hidden_size, bias=False, mesh_device=ctx.device, mesh_axis=ctx.tp_axis
        )
        self.linear_in = ColParallelLinear(
            embed_size, hidden_size, bias=False, mesh_device=ctx.device, mesh_axis=ctx.tp_axis
        )
        self.linear_out = RowParallelLinear(
            hidden_size,
            embed_size,
            bias=False,
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis,
            ccl_manager=ctx.ccl_manager,
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

        self._compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        state["inner.weight"] = state.pop("weight")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        dtype = x.dtype
        if dtype not in (ttnn.bfloat4_b, ttnn.bfloat8_b):
            # reduce L1 memory requirements
            x = ttnn.clone(x, dtype=ttnn.bfloat8_b)

        x = self.inner.forward(x, compute_kernel_config=self._compute_kernel_config)

        if x.dtype != dtype:
            x = ttnn.clone(x, dtype=dtype)

        return x


class Cache:
    def __init__(self, *, device: ttnn.MeshDevice, size: int) -> None:
        self.k_cache = {}
        self.v_cache = {}

        self._position = 0
        self._device = device

        self.size = size

    def prefill(self, cache_id: Hashable, k: ttnn.Tensor, v: ttnn.Tensor) -> None:
        batch_size, local_kv_heads, _seq_len, head_dim = k.shape

        assert self._position == 0

        k_cache = ttnn.zeros(
            [batch_size, local_kv_heads, self.size, head_dim],
            dtype=k.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self._device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        v_cache = ttnn.zeros(
            [batch_size, local_kv_heads, self.size, head_dim],
            dtype=v.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self._device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.k_cache[cache_id] = k_cache
        self.v_cache[cache_id] = v_cache

        for batch_idx in range(batch_size):
            ttnn.fill_cache(k_cache, k[batch_idx : batch_idx + 1], batch_idx)
            ttnn.fill_cache(v_cache, v[batch_idx : batch_idx + 1], batch_idx)

    def update(self, cache_id: Hashable, k: ttnn.Tensor, v: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        one, batch_size, _local_kv_heads, _head_dim = k.shape
        assert one == 1

        k_cache = self.k_cache[cache_id]
        v_cache = self.v_cache[cache_id]

        pos = [self._position] * batch_size

        ttnn.experimental.paged_update_cache(k_cache, k, update_idxs=pos)
        ttnn.experimental.paged_update_cache(v_cache, v, update_idxs=pos)

        return k_cache, v_cache

    def advance(self, distance: int) -> None:
        self._position += distance

    @property
    def position(self) -> int:
        return self._position


def _apply_rope(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
    n, _heads, seq, dim = x.shape

    assert cos.shape in ((n, seq, dim), (1, seq, dim))
    assert cos.shape == sin.shape

    return x * ttnn.unsqueeze(cos, 1) + _rotate_half(x) * ttnn.unsqueeze(sin, 1)


def _apply_rope_decode(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
    one, n, _heads, dim = x.shape
    seq = 1

    assert one == 1
    assert cos.shape in ((n, seq, dim), (1, seq, dim))
    assert cos.shape == sin.shape

    memory_config = x.memory_config()
    x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
    x = x * ttnn.unsqueeze(cos, 0) + _rotate_half(x) * ttnn.unsqueeze(sin, 0)
    return ttnn.to_memory_config(x, memory_config)


def _rotate_half(x: ttnn.Tensor) -> ttnn.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ttnn.concat([ttnn.neg(x2), x1], dim=-1)


def _prepare_attn_bias(mask: ttnn.Tensor, *, query_length: int, query_pos: int) -> ttnn.Tensor:
    batch_size, kv_length = mask.shape

    # convert to causal attention mask
    mask = ttnn.to_layout(mask, ttnn.ROW_MAJOR_LAYOUT)
    mask = mask.reshape([batch_size, 1, 1, kv_length])
    mask = ttnn.to_layout(mask, ttnn.TILE_LAYOUT)
    mask = ttnn.expand(mask, [batch_size, 1, query_length, kv_length])
    mask = ttnn.tril(mask, diagonal=query_pos)

    return (mask - 1.0) * math.inf


def _make_positions(*, start: int, sequence_length: int, device: ttnn.MeshDevice) -> ttnn.Tensor:
    pos = ttnn.arange(start, start + sequence_length, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    return ttnn.unsqueeze(pos, 0)

    # If the attention mask had holes, i.e., contained zeros between ones, this would have to be
    # done instead:
    # mask = ttnn.clone(mask, dtype=ttnn.float32)
    # # equivalent to: pos = mask.cumsum(1) - 1; pos.masked_fill_(mask == 0, 1)
    # pos = (ttnn.cumsum(mask, 1) - 2) * mask + 1
    # return pos[:, start:]


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


def _pad(t: torch.Tensor, amount: int, *, dim: int) -> torch.Tensor:
    """Pad tensor with `amount` zeros on the end of dimension `dim`."""
    padding = [0] * (2 * t.ndim)
    padding[-(dim * 2 + 1)] = amount
    return torch.nn.functional.pad(t, padding)


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
