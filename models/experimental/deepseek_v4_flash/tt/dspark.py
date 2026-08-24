# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn DSpark drafter: ``LinearDecode`` + DRISC weight prefetcher.

Port of :mod:`models.experimental.deepseek_v4_flash.dspark`. Every learned
projection (``main_proj``, Q/K/V/O, SwiGLU, LM head, Markov ``W2``, confidence)
is a :class:`~.layers.LinearDecode` whose weight is streamed through one shared
decode GCB. Attention scores / RoPE / embeddings are activations, not weights,
so they stay off the prefetcher path.

The sequential Markov loop self-queues each step so it does not occupy the GCB
FIFO for ``gamma`` repeats at hoist time. ``prefetch_weights`` therefore covers
the parallel backbone (plus the LM head) in forward order.

K/V of context and the draft block are packed into one ``k_proj`` / ``v_proj``
call each so those weights are consumed once per stage, matching a single
hoisted prefetch request.
"""

from __future__ import annotations

from typing import Optional

import torch
import ttnn

from models.experimental.deepseek_v4_flash.dspark import (
    DSparkConfig,
    DSparkOutput,
    apply_rope,
    dspark_block_mask,
    prefix_survival,
    truncate_prefix,
)

from .common import DeepSeekV4Module
from .layers import (
    DeepSeekV4RMSNorm,
    LinearDecode,
    decode_gcb_page_bytes,
    make_shared_decode_gcb,
)
from .system_config import active_system_config
from .weight_cache import WeightCache, _as_cache


def _n_blocks_for(N: int, *, tile: int = 32, target: int = 32) -> int:
    """B-core count for a fully width-sharded ``LinearDecode`` of width ``N``.

    Prefetched weights that share a GCB must agree on this number. ``target``
    (32) matches the rest of the Flash decode grid; fall back to any divisor of
    ``N`` whose per-core shard is tile-aligned.
    """
    if N % target == 0 and (N // target) % tile == 0:
        return target
    n_blocks = min(target, N // tile)
    while n_blocks > 1 and (N % n_blocks != 0 or (N // n_blocks) % tile != 0):
        n_blocks -= 1
    if n_blocks < 1 or N % n_blocks or (N // n_blocks) % tile:
        raise ValueError(f"cannot width-shard N={N} onto a tile-aligned decode grid")
    return n_blocks


def dspark_decode_specs(config: DSparkConfig, n_blocks: int) -> list[dict]:
    """``decode_weight_layout`` kwargs for every DSpark ``LinearDecode``, GCB order.

    Backbone first (the order :meth:`DSparkModel.forward` runs them), then the
    two sequential heads that self-queue per draft step. All use the same
    ``n_blocks`` so they can share one GCB.
    """
    h, qkv, inter, vocab, rank = (
        config.hidden_size,
        config.qkv_dim,
        config.intermediate_size,
        config.vocab_size,
        config.dspark_markov_rank,
    )

    def spec(K, N):
        return {"K": K, "N": N, "n_blocks": n_blocks}

    specs = [spec(h * config.num_target_layers, h)]
    for _ in range(config.num_stages):
        specs.extend(
            [
                spec(h, qkv),
                spec(h, qkv),
                spec(h, qkv),
                spec(qkv, h),
                spec(h, inter),
                spec(h, inter),
                spec(inter, h),
            ]
        )
    specs.append(spec(h, vocab))
    specs.append(spec(rank, vocab))
    specs.append(spec(h + rank, vocab))  # confidence padded from 1 to vocab
    return specs


def _to_dram(x: ttnn.Tensor) -> ttnn.Tensor:
    if x.is_sharded():
        return ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
    return x


def _as_decode_act(x: ttnn.Tensor) -> ttnn.Tensor:
    """Fold ``x`` onto ``[1, 1, tokens, dim]``, DRAM-interleaved for ``LinearDecode``."""
    x = _to_dram(x)
    tokens, dim = x.shape[-2], x.shape[-1]
    if list(x.shape) != [1, 1, tokens, dim]:
        x = ttnn.reshape(x, [1, 1, tokens, dim])
    return x


def _from_torch_act(t: torch.Tensor, device) -> ttnn.Tensor:
    """``[B, S, D]`` torch -> DRAM TILE ``[1, 1, B*S, D]``."""
    if t.dim() == 3:
        t = t.reshape(1, 1, t.shape[0] * t.shape[1], t.shape[2])
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _to_torch_act(x: ttnn.Tensor, batch: int, seq: int, dim: int) -> torch.Tensor:
    return ttnn.to_torch(_to_dram(x)).float().reshape(batch, seq, dim)


class _PrefetchedLinear(DeepSeekV4Module):
    """``LinearDecode`` wrapper that always uses the shared DSpark GCB."""

    def __init__(
        self,
        weight,
        cache_file,
        *,
        device,
        dtype,
        K,
        N,
        n_blocks,
        global_cb,
        page_bytes,
    ):
        self.layer = LinearDecode(
            weight,
            device,
            cache_file,
            dtype=dtype,
            K=K,
            N=N,
            n_blocks=n_blocks,
            use_prefetcher=True,
            global_cb=global_cb,
            global_cb_page_bytes=page_bytes,
        )

    def prefetch_weights(self) -> None:
        self.layer.fetch_weights()

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.layer(_as_decode_act(x))


def _linear_kw(device, dtype, n_blocks, global_cb, page_bytes) -> dict:
    return dict(
        device=device,
        dtype=dtype,
        n_blocks=n_blocks,
        global_cb=global_cb,
        page_bytes=page_bytes,
    )


class DSparkAttention(DeepSeekV4Module):
    """Block queries, KV-injected context; Q/K/V/O via prefetched ``LinearDecode``.

    Projections run on device. RoPE and the score/softmax matmuls run on host so
    they match the PyTorch reference (interleaved-pair RoPE, bidirectional block
    mask) without depending on fused-partial-RoPE's GLM table layout.
    """

    def __init__(
        self, config: DSparkConfig, weights: dict, prefix: str, device, cache, dtype, n_blocks, global_cb, page_bytes
    ):
        self.device = device
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.qkv_dim = config.qkv_dim
        h, qkv = config.hidden_size, config.qkv_dim
        kw = _linear_kw(device, dtype, n_blocks, global_cb, page_bytes)
        self.q_proj = _PrefetchedLinear(weights[f"{prefix}.q_proj.weight"], cache.file("q_proj"), K=h, N=qkv, **kw)
        self.k_proj = _PrefetchedLinear(weights[f"{prefix}.k_proj.weight"], cache.file("k_proj"), K=h, N=qkv, **kw)
        self.v_proj = _PrefetchedLinear(weights[f"{prefix}.v_proj.weight"], cache.file("v_proj"), K=h, N=qkv, **kw)
        self.o_proj = _PrefetchedLinear(weights[f"{prefix}.o_proj.weight"], cache.file("o_proj"), K=qkv, N=h, **kw)

    def prefetch_weights(self) -> None:
        self.q_proj.prefetch_weights()
        self.k_proj.prefetch_weights()
        self.v_proj.prefetch_weights()
        self.o_proj.prefetch_weights()

    def _split(self, x: torch.Tensor, seq: int) -> torch.Tensor:
        return x.view(1, seq, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        context: ttnn.Tensor,
        attn_mask: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        block_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
    ) -> ttnn.Tensor:
        gamma = hidden_states.shape[-2]
        ctx_len = context.shape[-2]
        hidden = _as_decode_act(hidden_states)
        ctx = _as_decode_act(context)
        kv_in = ttnn.concat([ctx, hidden], dim=2)

        q_pt = _to_torch_act(self.q_proj(hidden), 1, gamma, self.qkv_dim)
        k_pt = _to_torch_act(self.k_proj(kv_in), 1, ctx_len + gamma, self.qkv_dim)
        v_pt = _to_torch_act(self.v_proj(kv_in), 1, ctx_len + gamma, self.qkv_dim)

        cos, sin = rope
        kv_pos = torch.cat([context_position_ids, block_position_ids], dim=1)
        query = apply_rope(self._split(q_pt, gamma), cos, sin, block_position_ids)
        key = apply_rope(self._split(k_pt, ctx_len + gamma), cos, sin, kv_pos)
        value = self._split(v_pt, ctx_len + gamma)

        scores = torch.matmul(query, key.transpose(-2, -1)) * (self.head_dim**-0.5)
        scores = scores + attn_mask.to(dtype=scores.dtype)
        attn = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        out = torch.matmul(attn, value).transpose(1, 2).contiguous().view(1, gamma, self.qkv_dim)
        return self.o_proj(_from_torch_act(out, self.device))


class DSparkMLP(DeepSeekV4Module):
    def __init__(self, config, weights, prefix, device, cache, dtype, n_blocks, global_cb, page_bytes):
        h, inter = config.hidden_size, config.intermediate_size
        kw = _linear_kw(device, dtype, n_blocks, global_cb, page_bytes)
        self.gate_proj = _PrefetchedLinear(
            weights[f"{prefix}.gate_proj.weight"], cache.file("gate_proj"), K=h, N=inter, **kw
        )
        self.up_proj = _PrefetchedLinear(weights[f"{prefix}.up_proj.weight"], cache.file("up_proj"), K=h, N=inter, **kw)
        self.down_proj = _PrefetchedLinear(
            weights[f"{prefix}.down_proj.weight"], cache.file("down_proj"), K=inter, N=h, **kw
        )

    def prefetch_weights(self) -> None:
        self.gate_proj.prefetch_weights()
        self.up_proj.prefetch_weights()
        self.down_proj.prefetch_weights()

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.down_proj(ttnn.multiply(ttnn.silu(self.gate_proj(x)), self.up_proj(x)))


class DSparkStage(DeepSeekV4Module):
    def __init__(
        self, config, weights, prefix, device, cache, dtype, n_blocks, global_cb, page_bytes, *, has_main, has_heads
    ):
        cache = _as_cache(cache)
        self.attn_norm = DeepSeekV4RMSNorm(
            weights[f"{prefix}.attn_norm.weight"],
            config.rms_norm_eps,
            device,
            cache.file("attn_norm"),
            sharded=True,
        )
        self.attn = DSparkAttention(
            config, weights, f"{prefix}.attn", device, cache.sub("attn"), dtype, n_blocks, global_cb, page_bytes
        )
        self.ffn_norm = DeepSeekV4RMSNorm(
            weights[f"{prefix}.ffn_norm.weight"],
            config.rms_norm_eps,
            device,
            cache.file("ffn_norm"),
            sharded=True,
        )
        self.mlp = DSparkMLP(
            config, weights, f"{prefix}.mlp", device, cache.sub("mlp"), dtype, n_blocks, global_cb, page_bytes
        )
        self.main_proj = None
        self.main_norm = None
        if has_main:
            fused = config.hidden_size * config.num_target_layers
            kw = _linear_kw(device, dtype, n_blocks, global_cb, page_bytes)
            self.main_proj = _PrefetchedLinear(
                weights[f"{prefix}.main_proj.weight"],
                cache.file("main_proj"),
                K=fused,
                N=config.hidden_size,
                **kw,
            )
            self.main_norm = DeepSeekV4RMSNorm(
                weights[f"{prefix}.main_norm.weight"],
                config.rms_norm_eps,
                device,
                cache.file("main_norm"),
                sharded=True,
            )
        self.norm = None
        if has_heads:
            self.norm = DeepSeekV4RMSNorm(
                weights[f"{prefix}.norm.weight"],
                config.rms_norm_eps,
                device,
                cache.file("norm"),
                sharded=True,
            )

    def prefetch_weights(self) -> None:
        if self.main_proj is not None:
            self.main_proj.prefetch_weights()
        self.attn.prefetch_weights()
        self.mlp.prefetch_weights()

    def fuse_context(self, stacked: ttnn.Tensor) -> ttnn.Tensor:
        return self.main_norm(self.main_proj(stacked))

    def forward(self, hidden_states, context, **attn_kwargs) -> ttnn.Tensor:
        residual = _to_dram(hidden_states)
        hidden_states = ttnn.add(residual, _to_dram(self.attn(self.attn_norm(residual), context, **attn_kwargs)))
        return ttnn.add(hidden_states, _to_dram(self.mlp(self.ffn_norm(hidden_states))))


class DSparkModel(DeepSeekV4Module):
    """ttnn DSpark drafter. Construct from a torch :class:`~dspark.DSparkModel` state dict."""

    def __init__(
        self,
        config: DSparkConfig,
        weights: dict,
        device,
        *,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
        n_blocks: Optional[int] = None,
        num_prefetch_pages: Optional[int] = None,
    ):
        if n_blocks is None:
            n_blocks = _n_blocks_for(config.hidden_size)
        if num_prefetch_pages is None:
            num_prefetch_pages = active_system_config().prefetcher.num_prefetch_pages
        cache = _as_cache(cache)
        self.config = config
        self.device = device
        self.n_blocks = n_blocks
        self.weight_dtype = weight_dtype

        specs = dspark_decode_specs(config, n_blocks)
        page_bytes = decode_gcb_page_bytes(specs, weight_dtype)
        global_cb = make_shared_decode_gcb(device, specs, weight_dtype, num_pages=num_prefetch_pages)
        self.global_cb = global_cb
        self.page_bytes = page_bytes

        self.embed_tokens = ttnn.as_tensor(
            weights["embed_tokens.weight"],
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache.file("embed_tokens"),
        )
        kw = _linear_kw(device, weight_dtype, n_blocks, global_cb, page_bytes)
        self.lm_head = _PrefetchedLinear(
            weights["lm_head.weight"], cache.file("lm_head"), K=config.hidden_size, N=config.vocab_size, **kw
        )

        self.mtp = []
        last = config.num_stages - 1
        for i in range(config.num_stages):
            self.mtp.append(
                DSparkStage(
                    config,
                    weights,
                    f"mtp.{i}",
                    device,
                    cache.sub(f"mtp.{i}"),
                    weight_dtype,
                    n_blocks,
                    global_cb,
                    page_bytes,
                    has_main=(i == 0),
                    has_heads=(i == last),
                )
            )

        last_prefix = f"mtp.{last}"
        self.markov_w1 = ttnn.as_tensor(
            weights[f"{last_prefix}.markov_head.markov_w1.weight"],
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache.file("markov_w1"),
        )
        self.markov_w2 = _PrefetchedLinear(
            weights[f"{last_prefix}.markov_head.markov_w2.weight"],
            cache.file("markov_w2"),
            K=config.dspark_markov_rank,
            N=config.vocab_size,
            **kw,
        )
        conf_w = weights[f"{last_prefix}.confidence_head.proj.weight"]
        padded = torch.zeros(config.vocab_size, config.hidden_size + config.dspark_markov_rank, dtype=conf_w.dtype)
        padded[0].copy_(conf_w[0])
        self.confidence_proj = _PrefetchedLinear(
            padded,
            cache.file("confidence_proj"),
            K=config.hidden_size + config.dspark_markov_rank,
            N=config.vocab_size,
            **kw,
        )

        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim))
        freqs = torch.outer(torch.arange(config.max_position_embeddings).float(), inv_freq)
        self.cos_cached = freqs.cos()
        self.sin_cached = freqs.sin()

    @classmethod
    def from_torch(cls, torch_model, device, **kwargs) -> "DSparkModel":
        weights = {k: v.detach().float() for k, v in torch_model.state_dict().items()}
        return cls(torch_model.config, weights, device, **kwargs)

    def prefetch_weights(self) -> None:
        """Queue backbone weights in the order :meth:`forward` consumes them."""
        for stage in self.mtp:
            stage.prefetch_weights()
        self.lm_head.prefetch_weights()

    def _embed(self, ids: torch.Tensor) -> ttnn.Tensor:
        ids_tt = ttnn.from_torch(
            ids.view(1, -1).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        emb = ttnn.embedding(ids_tt, self.embed_tokens, layout=ttnn.TILE_LAYOUT)
        return ttnn.reshape(emb, [1, 1, ids.numel(), self.config.hidden_size])

    def fuse_target_hiddens(self, target_hiddens: torch.Tensor) -> ttnn.Tensor:
        """``target_hiddens`` is ``[B, S, L, D]`` torch; returns fused ``[1,1,B*S,D]``."""
        batch, seq, layers, dim = target_hiddens.shape
        flat = target_hiddens.reshape(1, 1, batch * seq, layers * dim)
        x = ttnn.from_torch(flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        return self.mtp[0].fuse_context(x)

    def forward(
        self,
        target_hiddens: torch.Tensor,
        anchor_ids: torch.Tensor,
        *,
        greedy: bool = True,
        temperature: float = 1.0,
        min_survival: float | None = None,
        hoist_prefetch: bool = True,
    ) -> DSparkOutput:
        if not greedy:
            raise NotImplementedError("ttnn DSpark only implements greedy sampling")
        del temperature
        if anchor_ids.dim() == 2:
            anchor_ids = anchor_ids.squeeze(-1)
        batch = int(anchor_ids.shape[0])
        cfg = self.config
        gamma = cfg.dspark_block_size
        ctx_len = target_hiddens.shape[1]
        if batch != 1:
            raise NotImplementedError("ttnn DSpark decode currently supports batch=1")

        if hoist_prefetch:
            self.prefetch_weights()

        context = self.fuse_target_hiddens(target_hiddens)
        noise = torch.full((batch, gamma - 1), cfg.dspark_noise_token_id, dtype=torch.long)
        block_ids = torch.cat([anchor_ids.view(batch, 1), noise], dim=1)
        hidden = self._embed(block_ids)

        ctx_pos = torch.arange(ctx_len).unsqueeze(0)
        blk_pos = torch.arange(ctx_len, ctx_len + gamma).unsqueeze(0)
        rope = (self.cos_cached[: ctx_len + gamma], self.sin_cached[: ctx_len + gamma])
        attn_mask = dspark_block_mask(ctx_len, gamma, cfg.sliding_window, torch.device("cpu"), torch.float32)
        attn_kwargs = dict(
            attn_mask=attn_mask,
            rope=rope,
            block_position_ids=blk_pos,
            context_position_ids=ctx_pos,
        )
        for stage in self.mtp:
            hidden = stage(hidden, context, **attn_kwargs)
        hidden = self.mtp[-1].norm(_as_decode_act(hidden))

        base_logits = self.lm_head(hidden)
        draft_ids, logits, confidence = self._markov_sample(hidden, base_logits, anchor_ids)
        survival = prefix_survival(confidence)
        if min_survival is not None:
            lengths = truncate_prefix(confidence, min_survival)
            keep = torch.arange(gamma).view(1, -1) < lengths.unsqueeze(-1)
            draft_ids = torch.where(keep, draft_ids, torch.zeros_like(draft_ids))
        return DSparkOutput(
            draft_ids=draft_ids,
            logits=logits,
            base_logits=_to_torch_act(base_logits, batch, gamma, cfg.vocab_size),
            confidence=confidence,
            prefix_survival=survival,
            hidden_states=_to_torch_act(hidden, batch, gamma, cfg.hidden_size),
            context=_to_torch_act(context, batch, ctx_len, cfg.hidden_size),
            block_input_ids=block_ids,
        )

    def _markov_sample(
        self, hidden: ttnn.Tensor, base_logits: ttnn.Tensor, anchor_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        gamma, vocab = cfg.dspark_block_size, cfg.vocab_size
        hidden_pt = _to_torch_act(hidden, 1, gamma, cfg.hidden_size)
        base_pt = _to_torch_act(base_logits, 1, gamma, vocab)
        draft = torch.empty(1, gamma, dtype=torch.long)
        step_logits = torch.empty(1, gamma, vocab)
        conf = torch.empty(1, gamma)
        prev = anchor_ids.view(1)
        for k in range(gamma):
            prev_tt = ttnn.from_torch(
                prev.view(1, 1).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
            )
            markov_emb = ttnn.embedding(prev_tt, self.markov_w1, layout=ttnn.TILE_LAYOUT)
            markov_emb = ttnn.reshape(markov_emb, [1, 1, 1, cfg.dspark_markov_rank])
            bias = _to_torch_act(self.markov_w2(markov_emb), 1, 1, vocab).view(vocab)
            logits_k = base_pt[0, k] + bias
            next_id = int(logits_k.argmax())
            draft[0, k] = next_id
            step_logits[0, k] = logits_k
            h_k = _from_torch_act(hidden_pt[:, k : k + 1, :], self.device)
            conf_in = ttnn.concat([_as_decode_act(h_k), markov_emb], dim=-1)
            conf_logits = _to_torch_act(self.confidence_proj(conf_in), 1, 1, vocab).view(vocab)
            conf[0, k] = torch.sigmoid(conf_logits[0])
            prev = torch.tensor([next_id])
        return draft, step_logits, conf
