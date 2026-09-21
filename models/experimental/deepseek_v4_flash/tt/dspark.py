# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn DSpark drafter: ``LinearDecode`` + DRISC weight prefetcher.

Standalone port of :mod:`models.experimental.deepseek_v4_flash.dspark`; the only
importer in the repo is ``tests/test_dspark_ttnn.py`` -- this module is *not* part of the
decode model. Every learned projection (``main_proj``, Q/K/V/O, SwiGLU, LM head, Markov
``W2``, confidence) is a :class:`~.layers.LinearDecode` whose weight streams through one
shared decode GCB (:func:`~.layers.make_shared_decode_gcb`), so every construction here
passes ``use_prefetcher=True`` with an explicit ``global_cb`` / ``global_cb_page_bytes``.

Letters used below: ``B`` = users decoded per round (1 here), ``gamma`` =
``dspark_block_size`` = draft-block length in query rows, ``C`` = injected context length,
``L`` = ``num_target_layers`` (the fused context is ``L*D`` wide), ``D`` =
``hidden_size``, ``H``/``Dh`` = ``num_attention_heads``/``head_dim``, ``K``/``N`` =
matmul in/out features, ``V`` = ``vocab_size``.

Activations are DRAM ``[1, 1, rows, K]`` turned into TILE WIDTH_SHARDED L1 by
:func:`_matmul_a` (``use_rm_hs=False``): the ROW_MAJOR HEIGHT_SHARDED replica path is
Flash's M=1 decode layout and rejects M>8, while DSpark's block is a full tile of rows.
``main_proj`` is followed by its own :class:`~.layers.DeepSeekV4RMSNorm` (``main_norm``),
not a matmul epilogue. RoPE is :func:`~.attention._apply_rope`
(``fused_partial_rope``). Attention is ``scaled_dot_product_attention_decode`` with the
draft block as the decode batch and ``share_cache`` so every query sees the same
injected K/V.

The sequential Markov loop self-queues each step so it does not occupy the GCB FIFO for
``gamma`` repeats at hoist time; ``prefetch_weights`` therefore covers the parallel
backbone (plus the LM head) in forward order. Context and draft K/V are concatenated into
one ``k_proj`` / ``v_proj`` call each so those weights are consumed once per stage,
matching a single hoisted prefetch request.
"""

from __future__ import annotations

import math
import queue
import threading
from typing import Iterable, Optional

import torch
import ttnn

from models.experimental.deepseek_v4_flash.dspark import (
    DSparkConfig,
    DSparkOutput,
    dspark_block_mask,
    prefix_survival,
    truncate_prefix,
)

from .attention import (
    _apply_rope,
    _interleaved_rotate_matrix,
    make_rope_table,
)
from .common import DeepSeekV4Module, _HIFI4_SDPA
from .layers import (
    DeepSeekV4RMSNorm,
    LinearDecode,
    decode_gcb_page_bytes,
    make_shared_decode_gcb,
)
from .system_config import active_system_config
from .weight_cache import WeightCache, _as_cache, _load_weight


# Kept clear of the two cores the model pipeline hand-off sockets use. Explicit local
# coordinates also make the traced path usable on a single-device test mesh, where the
# usual idle-submesh placement does not exist.
_TRACE_INPUT_SOCKET_CORE = (0, 2)
_TRACE_OUTPUT_SOCKET_CORE = (0, 3)


def _n_blocks_for(N: int, *, tile: int = 32, target: int = 32) -> int:
    """B-core count for a fully width-sharded ``LinearDecode`` of width ``N``; ``[1, target]``.

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
    """``decode_weight_layout`` kwargs for every DSpark ``LinearDecode``, in GCB order.

    Backbone first -- the order :meth:`DSparkModel.forward` runs them, each consuming
    ``[1, 1, rows, K]`` and producing ``[1, 1, rows, N]`` -- then the two sequential heads
    that self-queue per draft step. All use the same ``n_blocks`` so they can share one
    GCB, and that single FIFO is what makes this order load-bearing rather than cosmetic.
    """
    h, qkv, inter, vocab, rank = (
        config.hidden_size,
        config.qkv_dim,
        config.intermediate_size,
        config.vocab_size,
        config.dspark_markov_rank,
    )

    def spec(K, N):
        """One ``{"K": K, "N": N, "n_blocks": n_blocks}`` spec for a ``[K, N]`` weight."""
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
    """``x`` (any ``[..]`` shape) in DRAM interleaved, layout unchanged; sharded input is
    moved first, an interleaved input passes straight through."""
    if x.is_sharded():
        return ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
    return x


def _as_decode_act(x: ttnn.Tensor) -> ttnn.Tensor:
    """``x`` folded onto ``[1, 1, tokens, dim]`` in DRAM interleaved, layout left as it was.

    Only the shape is normalised here; :func:`_matmul_a` is what shards it for the matmul.
    """
    x = _to_dram(x)
    tokens, dim = x.shape[-2], x.shape[-1]
    if list(x.shape) != [1, 1, tokens, dim]:
        x = ttnn.reshape(x, [1, 1, tokens, dim])
    return x


def _matmul_a(layer: LinearDecode, x: ttnn.Tensor) -> ttnn.Tensor:
    """TILE WIDTH_SHARDED L1 ``[1, 1, tokens, dim]`` for ``use_rm_hs=False``.

    The replicated ROW_MAJOR HEIGHT_SHARDED path is Flash's M=1 decode layout and rejects
    M>8, and DSpark's block is a full tile of tokens, so every backbone projection comes
    through here. Markov is M=1 but stays TILE too, to share the LM head's kernel -- the
    replica path disagreed with the reference at the vocab projection.
    """
    return layer.to_width_sharded_activation(_as_decode_act(x))


def _to_torch_act(x: ttnn.Tensor, batch: int, seq: int, dim: int) -> torch.Tensor:
    """Device ``x`` -> float32 torch ``[batch, seq, dim]`` (moved to DRAM first)."""
    return ttnn.to_torch(_to_dram(x)).float().reshape(batch, seq, dim)


def _first_row(x: ttnn.Tensor) -> ttnn.Tensor:
    """First row of ``x`` as ``[1, 1, 1, N]`` DRAM, so the next decode matmul sees M=1.

    A replicated matmul output carries more rows than the single run this result feeds.
    """
    x = _as_decode_act(x)
    if x.shape[-2] != 1:
        x = ttnn.slice(x, [0, 0, 0, 0], [1, 1, 1, x.shape[-1]])
    return x


class DSparkAttention(DeepSeekV4Module):
    """Block queries over KV-injected context; Q/K/V/O via prefetched ``LinearDecode``.

    RoPE is Flash's fused interleaved ``rotate_half``. The ``gamma`` draft tokens are the
    SDPA-decode batch (S=1 per query); ``share_cache`` reuses one K/V sequence (injected
    context + block). Q/K/V stay packed ``[1, 1, rows, H*Dh]`` through the projections and
    are reshaped only for the op. Q goes in as DRAM TILE because MHA cannot use the
    ROW_MAJOR Q path -- that path requires ``n_kv_heads == 1``, and here ``n_kv == H``.
    """

    def __init__(
        self, config: DSparkConfig, weights: dict, prefix: str, device, cache, dtype, n_blocks, global_cb, page_bytes
    ):
        """``prefix`` is the ``mtp.<i>.attn`` state-dict prefix; the four ``[N, K]`` weights
        it names become :class:`~.layers.LinearDecode`s that all draw on ``global_cb``
        (``page_bytes``-sized pages, ``n_blocks`` B cores, ``use_rm_hs=False``)."""
        self.device = device
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.qkv_dim = config.qkv_dim
        self.scaling = config.head_dim**-0.5
        self._sdpa_pcfg = active_system_config().attention.sdpa_program_config(device)
        h, qkv = config.hidden_size, config.qkv_dim
        kw = dict(
            dtype=dtype,
            n_blocks=n_blocks,
            use_prefetcher=True,
            global_cb=global_cb,
            global_cb_page_bytes=page_bytes,
            use_rm_hs=False,
        )
        self.q_proj = LinearDecode(weights[f"{prefix}.q_proj.weight"], device, cache.file("q_proj"), K=h, N=qkv, **kw)
        self.k_proj = LinearDecode(weights[f"{prefix}.k_proj.weight"], device, cache.file("k_proj"), K=h, N=qkv, **kw)
        self.v_proj = LinearDecode(weights[f"{prefix}.v_proj.weight"], device, cache.file("v_proj"), K=h, N=qkv, **kw)
        self.o_proj = LinearDecode(weights[f"{prefix}.o_proj.weight"], device, cache.file("o_proj"), K=qkv, N=h, **kw)

    def prefetch_weights(self) -> None:
        """Queue the ``[N, K]`` q/k/v/o prefetches in forward order -- one GCB FIFO, so
        order is a contract, not a convenience."""
        self.q_proj.fetch_weights()
        self.k_proj.fetch_weights()
        self.v_proj.fetch_weights()
        self.o_proj.fetch_weights()

    def _to_sdpa_kv(self, packed: ttnn.Tensor, seq: int) -> ttnn.Tensor:
        """Packed ``[1, 1, S, H*Dh]`` -> DRAM TILE ``[1, H, S, Dh]`` (share_cache batch=1)."""
        h, dh = self.num_heads, self.head_dim
        x = ttnn.reshape(_to_dram(packed), [1, seq, h, dh])
        return ttnn.permute(x, (0, 2, 1, 3))

    def _sdpa_decode(
        self,
        q_packed: ttnn.Tensor,
        k_packed: ttnn.Tensor,
        v_packed: ttnn.Tensor,
        mask: ttnn.Tensor,
        *,
        gamma: int,
        seq: int,
    ) -> ttnn.Tensor:
        """``q_packed`` ``[1, 1, gamma, H*Dh]``; returns packed attention ``[1, 1, gamma, H*Dh]``.

        Decode Q is ``[1, B, H, Dh]`` with ``B = gamma``. K/V must use the same B
        (``[B, H, S, Dh]``): ``share_cache`` keeps K batch at 1, but the op's output
        spec still takes B from K, so the result would be one query instead of
        ``gamma``. Replicating the injected cache matches Flash's ``[B, n_kv, S, Dh]``.
        ``n_kv == H`` is treated as GQA, which forbids a sharded output, so this stays
        DRAM TILE.
        """
        h, dh = self.num_heads, self.head_dim
        q = ttnn.reshape(_to_dram(q_packed), [1, gamma, h, dh])
        k = self._to_sdpa_kv(k_packed, seq)
        v = self._to_sdpa_kv(v_packed, seq)
        if gamma > 1:
            k = ttnn.repeat(k, ttnn.Shape([gamma, 1, 1, 1]))
            v = ttnn.repeat(v, ttnn.Shape([gamma, 1, 1, 1]))
        attn_mask = mask
        if mask.shape[-2] != h:
            attn_mask = ttnn.repeat(mask, ttnn.Shape([1, 1, h, 1]))
        out = ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            k,
            v,
            is_causal=False,
            attn_mask=attn_mask,
            scale=self.scaling,
            program_config=self._sdpa_pcfg,
            compute_kernel_config=_HIFI4_SDPA,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.reshape(_to_dram(out), [1, 1, gamma, h * dh])

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        context: ttnn.Tensor,
        attn_mask: ttnn.Tensor,
        cos_q: ttnn.Tensor,
        sin_q: ttnn.Tensor,
        cos_kv: ttnn.Tensor,
        sin_kv: ttnn.Tensor,
        rot: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """One DSpark attention block; returns ``[1, 1, gamma, D]`` after ``o_proj``.

        ``hidden_states`` ``[1, 1, gamma, D]`` and ``context`` ``[1, 1, C, D]`` are
        concatenated on the row dim into the ``[1, 1, C+gamma, D]`` KV input. ``cos_q`` /
        ``sin_q`` are ``[1, 1, gamma, Rd]`` and ``cos_kv`` / ``sin_kv`` ``[1, 1, C+gamma,
        Rd]`` (DRAM TILE); ``rot`` is the ``[Rd, Rd]`` interleaved rotate matrix; and
        ``attn_mask`` is the ``[1, 1, 1, C+gamma]`` additive BF16 TILE mask, repeated to
        ``H`` rows inside :meth:`_sdpa_decode`.
        """
        hidden = _as_decode_act(hidden_states)
        ctx = _as_decode_act(context)
        gamma = hidden.shape[-2]
        kv_in = ttnn.concat([ctx, hidden], dim=2)
        dh = self.head_dim
        q = _apply_rope(self.q_proj(_matmul_a(self.q_proj, hidden)), cos_q, sin_q, rot, dh, head_dim=dh)
        k = _apply_rope(self.k_proj(_matmul_a(self.k_proj, kv_in)), cos_kv, sin_kv, rot, dh, head_dim=dh)
        v = self.v_proj(_matmul_a(self.v_proj, kv_in))
        out = self._sdpa_decode(q, k, v, attn_mask, gamma=gamma, seq=kv_in.shape[-2])
        return self.o_proj(_matmul_a(self.o_proj, out))


class DSparkMLP(DeepSeekV4Module):
    def __init__(self, config, weights, prefix, device, cache, dtype, n_blocks, global_cb, page_bytes):
        """SwiGLU MLP: ``gate``/``up`` are ``[I, D]`` and ``down`` ``[D, I]`` torch weights
        (``[out, in]``), each a prefetched :class:`~.layers.LinearDecode` sharing
        ``global_cb`` / ``page_bytes`` over ``n_blocks`` B cores. ``prefix`` is
        ``mtp.<i>.mlp``."""
        h, inter = config.hidden_size, config.intermediate_size
        kw = dict(
            dtype=dtype,
            n_blocks=n_blocks,
            use_prefetcher=True,
            global_cb=global_cb,
            global_cb_page_bytes=page_bytes,
            use_rm_hs=False,
        )
        self.gate_proj = LinearDecode(
            weights[f"{prefix}.gate_proj.weight"], device, cache.file("gate_proj"), K=h, N=inter, **kw
        )
        self.up_proj = LinearDecode(
            weights[f"{prefix}.up_proj.weight"], device, cache.file("up_proj"), K=h, N=inter, **kw
        )
        self.down_proj = LinearDecode(
            weights[f"{prefix}.down_proj.weight"], device, cache.file("down_proj"), K=inter, N=h, **kw
        )

    def prefetch_weights(self) -> None:
        """Queue the ``[I, D]`` gate/up and ``[D, I]`` down prefetches in forward order
        (single GCB FIFO, so order matters)."""
        self.gate_proj.fetch_weights()
        self.up_proj.fetch_weights()
        self.down_proj.fetch_weights()

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """``x`` ``[1, 1, rows, D]`` -> ``silu(gate(x)) * up(x)`` through ``down``,
        ``[1, 1, rows, D]``; both matmul inputs go through :func:`_matmul_a`."""
        act = _matmul_a(self.gate_proj, x)
        gated = ttnn.multiply(ttnn.silu(self.gate_proj(act)), self.up_proj(act))
        return self.down_proj(_matmul_a(self.down_proj, gated))


class DSparkStage(DeepSeekV4Module):
    def __init__(
        self, config, weights, prefix, device, cache, dtype, n_blocks, global_cb, page_bytes, *, has_main, has_heads
    ):
        """One MTP stage: pre-norm attention and pre-norm MLP, plus the optional
        context-fusing ``main_proj`` / ``main_norm`` (only when ``has_main``, stage 0) and
        the final ``norm`` (only when ``has_heads``, last stage). ``prefix`` is
        ``mtp.<i>``; the ``[D]`` norm weights are ``sharded=True``
        :class:`~.layers.DeepSeekV4RMSNorm`s."""
        cache = _as_cache(cache)
        self.eps = config.rms_norm_eps
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
            self.main_proj = LinearDecode(
                weights[f"{prefix}.main_proj.weight"],
                device,
                cache.file("main_proj"),
                K=fused,
                N=config.hidden_size,
                dtype=dtype,
                n_blocks=n_blocks,
                use_prefetcher=True,
                global_cb=global_cb,
                global_cb_page_bytes=page_bytes,
                use_rm_hs=False,
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
        """Queue the ``[D, L*D]`` ``main_proj`` (when present), then attention, then MLP, in
        the order :meth:`forward` runs them."""
        if self.main_proj is not None:
            self.main_proj.fetch_weights()
        self.attn.prefetch_weights()
        self.mlp.prefetch_weights()

    def fuse_context(self, stacked: ttnn.Tensor) -> ttnn.Tensor:
        """``stacked`` ``[1, 1, rows, L*D]`` -> ``[1, 1, rows, D]`` via ``main_proj``,
        then ``main_norm`` when this stage owns one (stage 0 does)."""
        fused = self.main_proj(_matmul_a(self.main_proj, stacked))
        return fused if self.main_norm is None else self.main_norm(fused)

    def forward(self, hidden_states, context, **attn_kwargs) -> ttnn.Tensor:
        """``hidden_states`` ``[1, 1, gamma, D]`` and ``context`` ``[1, 1, C, D]``;
        returns the twice-residual-added ``[1, 1, gamma, D]`` (attention, then MLP).
        ``attn_kwargs`` is forwarded verbatim to :meth:`DSparkAttention.forward`."""
        residual = _as_decode_act(hidden_states)
        hidden_states = ttnn.add(residual, _to_dram(self.attn(self.attn_norm(residual), context, **attn_kwargs)))
        return ttnn.add(hidden_states, _to_dram(self.mlp(self.ffn_norm(hidden_states))))


class DSparkModel(DeepSeekV4Module):
    """ttnn DSpark drafter: context fusion, ``num_stages`` draft stages, LM head, Markov head.

    Standalone -- the decode model never builds this, only ``tests/test_dspark_ttnn.py``
    does. Construct from a torch :class:`~dspark.DSparkModel` state dict, normally via
    :meth:`from_torch`. Below: ``B`` = users per round (only 1 is supported), ``gamma`` =
    ``dspark_block_size`` draft rows, ``C`` = injected context length.
    """

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
        """Build every weight on ``device`` and size the one shared decode GCB.

        ``weights`` maps torch state-dict keys to tensors. ``n_blocks`` defaults to
        :func:`_n_blocks_for` at ``hidden_size``; ``num_prefetch_pages`` to the active
        profile's ``prefetcher.num_prefetch_pages`` and is the ring depth in pages. The
        GCB page size comes from :func:`~.layers.decode_gcb_page_bytes` over
        :func:`dspark_decode_specs`, so all of these layers can share one prefetch FIFO.

        ``embed_tokens`` and ``markov_w1`` stay ROW_MAJOR ``[V, D]`` / ``[V, rank]`` DRAM
        because ``ttnn.embedding`` takes row-major tables; ``cos_cached`` / ``sin_cached``
        are ``[1, 1, max_position_embeddings, Rd]`` TILE DRAM and ``rot`` is the
        ``[Rd, Rd]`` interleaved rotate matrix. ``confidence_proj`` is the ``[1, D+rank]``
        head weight zero-padded to ``[V, D+rank]`` so it can share the GCB page geometry
        with the LM head.
        """
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
        kw = dict(
            dtype=weight_dtype,
            n_blocks=n_blocks,
            use_prefetcher=True,
            global_cb=global_cb,
            global_cb_page_bytes=page_bytes,
            use_rm_hs=False,
        )
        self.lm_head = LinearDecode(
            weights["lm_head.weight"], device, cache.file("lm_head"), K=config.hidden_size, N=config.vocab_size, **kw
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
        self.markov_w2 = LinearDecode(
            weights[f"{last_prefix}.markov_head.markov_w2.weight"],
            device,
            cache.file("markov_w2"),
            K=config.dspark_markov_rank,
            N=config.vocab_size,
            **kw,
        )
        conf_w = weights[f"{last_prefix}.confidence_head.proj.weight"]
        padded = torch.zeros(config.vocab_size, config.hidden_size + config.dspark_markov_rank, dtype=conf_w.dtype)
        padded[0].copy_(conf_w[0])
        self.confidence_proj = LinearDecode(
            padded,
            device,
            cache.file("confidence_proj"),
            K=config.hidden_size + config.dspark_markov_rank,
            N=config.vocab_size,
            **kw,
        )

        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim))
        freqs = torch.outer(torch.arange(config.max_position_embeddings).float(), inv_freq)
        cos, sin = make_rope_table(freqs.cos(), freqs.sin())
        self.cos_cached = ttnn.from_torch(
            cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.sin_cached = ttnn.from_torch(
            sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.rot = _load_weight(_interleaved_rotate_matrix(config.head_dim), device, cache_file_name=cache.file("rot"))

        # Traced drafting is initialized lazily: the input context length is part of the
        # trace shape. These fields stay separate from the eager path because a trace owns
        # the addresses of every tensor participating in it.
        self._trace_input_socket = None
        self._trace_output_socket = None
        self._trace_id = None
        self._trace_output = None
        self._trace_context = None
        self._trace_context_len: Optional[int] = None
        self._trace_input_width: Optional[int] = None
        self._trace_input_rows: Optional[int] = None
        self._trace_input_page_bytes: Optional[int] = None
        self._trace_output_words: Optional[int] = None
        self._trace_noise = None
        self._trace_attn_mask = None
        self._trace_replay_queue: queue.Queue = queue.Queue()
        self._trace_replay_thread: Optional[threading.Thread] = None

    @classmethod
    def from_torch(cls, torch_model, device, **kwargs) -> "DSparkModel":
        """Build from a torch :class:`~dspark.DSparkModel`: ``torch_model.config`` plus every
        ``[..]`` state-dict tensor detached and cast to float32. Extra ``kwargs`` go to
        ``__init__``."""
        weights = {k: v.detach().float() for k, v in torch_model.state_dict().items()}
        return cls(torch_model.config, weights, device, **kwargs)

    def prefetch_weights(self) -> None:
        """Queue the backbone and head weights in the order :meth:`forward` consumes them, so
        every ``[K, N]`` slab is already in the GCB ring when its matmul asks for it."""
        for stage in self.mtp:
            stage.prefetch_weights()
        self.lm_head.fetch_weights()
        self.markov_w2.fetch_weights()
        self.confidence_proj.fetch_weights()

    def _prefetch_trace_weights(self) -> None:
        """Queue only the weights the device-only traced draft consumes.

        Same order as :meth:`prefetch_weights` minus the ``[V, D+rank]`` ``confidence_proj``:
        the traced body never runs the confidence head, so nothing in the trace would
        consume that request.
        """
        for stage in self.mtp:
            stage.prefetch_weights()
        self.lm_head.fetch_weights()
        self.markov_w2.fetch_weights()

    def _embed(self, ids: torch.Tensor) -> ttnn.Tensor:
        """``ids`` torch int tensor of ``n`` token IDs -> TILE ``[1, 1, n, D]``.

        ROW_MAJOR uint32 ids are uploaded for ``ttnn.embedding`` against
        ``embed_tokens``; the caller passes anchor + noise as one flat block."""
        ids_tt = ttnn.from_torch(
            ids.view(1, -1).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        emb = ttnn.embedding(ids_tt, self.embed_tokens, layout=ttnn.TILE_LAYOUT)
        return ttnn.reshape(emb, [1, 1, ids.numel(), self.config.hidden_size])

    def fuse_target_hiddens(self, target_hiddens: torch.Tensor) -> ttnn.Tensor:
        """``target_hiddens`` is ``[B, S, L, D]`` torch; returns TILE DRAM ``[1, 1, B*S, D]``."""
        batch, seq, layers, dim = target_hiddens.shape
        flat = target_hiddens.reshape(1, 1, batch * seq, layers * dim)
        x = ttnn.from_torch(flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        return self.mtp[0].fuse_context(x)

    def _fuse_target_hiddens_device(self, target_hiddens: ttnn.Tensor) -> ttnn.Tensor:
        """Fuse a device-resident ``[1, 1, S, L*D]`` context packet into ``[1, 1, S, D]``.

        The eager API uploads a torch tensor; the traced API receives this packet from an
        H2D socket, so everything after the receive has to stay on device.
        """
        return self.mtp[0].fuse_context(ttnn.to_layout(target_hiddens, ttnn.TILE_LAYOUT))

    def _embed_device(self, ids: ttnn.Tensor) -> ttnn.Tensor:
        """Device ROW_MAJOR uint32 ``ids`` ``[1, gamma]`` -> TILE ``[1, 1, gamma, D]``.

        The traced block always embeds exactly one draft block, so the row count is fixed
        at ``dspark_block_size`` and can be baked into the shape.
        """
        emb = ttnn.embedding(ids, self.embed_tokens, layout=ttnn.TILE_LAYOUT)
        return ttnn.reshape(emb, [1, 1, self.config.dspark_block_size, self.config.hidden_size])

    def _trace_input_page(self, target_hiddens: torch.Tensor, anchor_id: int) -> torch.Tensor:
        """One PCIe-aligned H2D page ``[1, 1, rows, width]`` float32 for a traced step.

        ``target_hiddens`` is ``[1, S, L, D]`` torch (a bare ``[S, L, D]`` is promoted) and
        must match the context length captured in :meth:`_capture_trace`; ``anchor_id``
        rides in the last word of the flat payload. The context is sent as FP32 so the
        anchor token can share the page without losing precision at the ~129K vocabulary --
        the device casts only the context slice to BF16 before entering the model.

        The payload is ``L*D + 1`` FP32 words rounded up to ``pcie_alignment`` and then
        capped at the H2D FIFO size, so a large hidden row spans several socket pages; the
        trace reassembles the flat stream.
        """
        if target_hiddens.dim() == 3:
            target_hiddens = target_hiddens.unsqueeze(0)
        if target_hiddens.dim() != 4 or target_hiddens.shape[0] != 1:
            raise ValueError("traced target_hiddens must have shape [1, S, L, D]")
        _, seq, layers, hidden = target_hiddens.shape
        expected_layers = self.config.num_target_layers
        if layers != expected_layers or hidden != self.config.hidden_size:
            raise ValueError(
                f"expected target_hiddens [1, S, {expected_layers}, {self.config.hidden_size}], "
                f"got {tuple(target_hiddens.shape)}"
            )
        if self._trace_context_len is None:
            self._trace_context_len = int(seq)
        if seq != self._trace_context_len:
            raise ValueError(
                f"traced DSpark context length is fixed at {self._trace_context_len}; got {seq}. "
                "Capture another trace for a different context length."
            )

        alignment = active_system_config().pipeline.pcie_alignment
        values = target_hiddens.detach().float().reshape(-1)
        # H2D FIFOs are deliberately small (typically one 4 KiB page), so a large hidden
        # row has to be split across socket pages.
        payload_width = layers * hidden + 1
        page_bytes = math.ceil(payload_width * 4 / alignment) * alignment
        fifo_bytes = active_system_config().pipeline.h2d_fifo_bytes
        page_bytes = min(page_bytes, (fifo_bytes // alignment) * alignment)
        width = page_bytes // 4
        if width <= 0:
            raise ValueError("H2D FIFO is smaller than one PCIe alignment")
        payload = torch.empty(values.numel() + 1, dtype=torch.float32)
        payload[:-1] = values
        payload[-1] = float(anchor_id)
        rows = math.ceil(payload.numel() / width)
        packet = torch.zeros(1, 1, rows, width, dtype=torch.float32)
        packet.view(-1)[: payload.numel()] = payload
        return packet

    def _trace_body(self) -> ttnn.Tensor:
        """Device-only DSpark block, run once for compile and once for trace capture.

        Parks on ``recv_async_h2d`` for the ``[1, 1, rows, width]`` FP32 page in
        ``_trace_input_socket`` -- so the matching host write must come *after* this is
        dispatched -- then does context fusion, ``num_stages`` draft stages, the LM head and
        the on-device Markov loop. Returns the draft block as ROW_MAJOR ``[1, 1, 1,
        _trace_output_words]`` token IDs, zero-padded past ``gamma`` to fill one int32 D2H
        page.
        """
        cfg = self.config
        ttnn.experimental.recv_async_h2d(self._trace_context, self._trace_input_socket)
        context_width = cfg.num_target_layers * cfg.hidden_size
        packet = ttnn.reshape(self._trace_context, [1, 1, 1, -1])
        payload_size = self._trace_context_len * context_width + 1
        payload = ttnn.slice(packet, [0, 0, 0, 0], [1, 1, 1, payload_size])
        context = ttnn.typecast(
            ttnn.reshape(
                ttnn.slice(payload, [0, 0, 0, 0], [1, 1, 1, payload_size - 1]),
                [1, 1, self._trace_context_len, context_width],
            ),
            ttnn.bfloat16,
        )
        anchor = ttnn.typecast(
            ttnn.reshape(
                ttnn.slice(
                    payload,
                    [0, 0, 0, payload_size - 1],
                    [1, 1, 1, payload_size],
                ),
                [1, 1],
            ),
            ttnn.uint32,
        )
        context = self._fuse_target_hiddens_device(context)

        block_ids = ttnn.concat([anchor, self._trace_noise], dim=1)
        hidden = self._embed_device(block_ids)

        total = self._trace_context_len + cfg.dspark_block_size
        rd = cfg.head_dim
        cos_kv = ttnn.slice(self.cos_cached, [0, 0, 0, 0], [1, 1, total, rd])
        sin_kv = ttnn.slice(self.sin_cached, [0, 0, 0, 0], [1, 1, total, rd])
        cos_q = ttnn.slice(self.cos_cached, [0, 0, self._trace_context_len, 0], [1, 1, total, rd])
        sin_q = ttnn.slice(self.sin_cached, [0, 0, self._trace_context_len, 0], [1, 1, total, rd])
        attn_kwargs = dict(
            attn_mask=self._trace_attn_mask,
            cos_q=cos_q,
            sin_q=sin_q,
            cos_kv=cos_kv,
            sin_kv=sin_kv,
            rot=self.rot,
        )
        for stage in self.mtp:
            hidden = stage(hidden, context, **attn_kwargs)
        hidden = self.mtp[-1].norm(_as_decode_act(hidden))
        base_logits = self.lm_head(_matmul_a(self.lm_head, hidden))

        # Unlike eager :meth:`forward`, which reads every Markov step back to the host, the
        # trace keeps the recurrent token on device and exports only the final block.
        prev = anchor
        draft = None
        for k in range(cfg.dspark_block_size):
            markov_emb = ttnn.embedding(prev, self.markov_w1, layout=ttnn.TILE_LAYOUT)
            markov_emb = ttnn.reshape(markov_emb, [1, 1, 1, cfg.dspark_markov_rank])
            bias = _first_row(self.markov_w2(_matmul_a(self.markov_w2, markov_emb)))
            base_k = ttnn.slice(_as_decode_act(base_logits), [0, 0, k, 0], [1, 1, k + 1, cfg.vocab_size])
            logits_k = ttnn.add(_as_decode_act(base_k), bias)
            next_id = ttnn.reshape(ttnn.argmax(logits_k, dim=-1, keepdim=True), [1, 1, 1, 1])
            draft = next_id if draft is None else ttnn.concat([draft, next_id], dim=3)
            prev = ttnn.reshape(next_id, [1, 1])

        output_words = self._trace_output_words
        if draft.shape[-1] < output_words:
            draft = ttnn.pad(draft, [(0, 0), (0, 0), (0, 0), (0, output_words - draft.shape[-1])], value=0)
        return ttnn.to_layout(draft, ttnn.ROW_MAJOR_LAYOUT)

    def _ensure_trace_replay_thread(self) -> None:
        """Start the daemon replay thread on first use; later calls are no-ops.

        The thread drains ``queue.Queue[bool]`` markers with non-blocking ``execute_trace``
        on ``cq_id=0`` and exits at the ``None`` sentinel :meth:`shutdown` pushes.
        """
        if self._trace_replay_thread is not None:
            return

        def _run() -> None:
            """One non-blocking ``execute_trace`` (``cq_id=0``) per queued marker, until
            ``None``. Dispatch only -- the ``[1, gamma]`` output read stays with the caller.
            """
            for _ in iter(self._trace_replay_queue.get, None):
                ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=False)

        self._trace_replay_thread = threading.Thread(target=_run, name="dspark-trace-replay", daemon=True)
        self._trace_replay_thread.start()

    def shutdown(self) -> None:
        """Stop the trace-replay thread started by :meth:`_ensure_trace_replay_thread`.

        The thread is a daemon whose target closes over ``self``, and nothing ever tells it
        to stop. CPython does not unwind a daemon thread's frame at shutdown, so the closure
        keeps this model -- and every tensor it owns, down to the ``[1, 1, rows, width]``
        trace input page -- alive until the process exits, where nanobind's ``Py_AtExit``
        leak check reports it. Ending the loop releases that reference so the model can be
        collected normally.

        Idempotent, and safe whether or not a traced replay ever ran. Not safe to call
        concurrently with :meth:`replay_traced`/:meth:`decode_traced_async`.
        """
        thread, self._trace_replay_thread = self._trace_replay_thread, None
        if thread is None:
            return
        # Drain what is already queued, then end the loop. The thread's only work is
        # non-blocking ``execute_trace`` dispatch, so joining cannot park on the device.
        self._trace_replay_queue.put(None)
        thread.join()

    def _capture_trace(self, target_hiddens: torch.Tensor, anchor_id: int) -> None:
        """Capture the drafting trace for one fixed ``(context length, gamma)`` shape.

        ``target_hiddens`` ``[1, S, L, D]`` torch fixes the context length recorded in
        ``_trace_context_len``; ``anchor_id`` only feeds the compile-run page. Sockets get
        ``_TRACE_INPUT_SOCKET_CORE`` / ``_TRACE_OUTPUT_SOCKET_CORE`` and one page each:
        input ``[1, 1, _trace_input_rows, _trace_input_width]`` FP32 ROW_MAJOR L1,
        output ``[1, 1, 1, _trace_output_words]`` int32. Every trace-side tensor
        (context, noise ids, attention mask, both sockets) is created here, because a trace
        owns the addresses of everything in it.
        """
        packet = self._trace_input_page(target_hiddens, anchor_id)
        alignment = active_system_config().pipeline.pcie_alignment
        self._trace_input_width = packet.shape[-1]
        self._trace_input_rows = packet.shape[-2]
        self._trace_input_page_bytes = self._trace_input_width * 4
        self._trace_output_words = math.ceil(self.config.dspark_block_size * 4 / alignment) * (alignment // 4)

        self._trace_input_socket = ttnn.H2DSocket(
            self.device,
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(*_TRACE_INPUT_SOCKET_CORE)),
            ttnn.BufferType.L1,
            active_system_config().pipeline.h2d_fifo_bytes,
            ttnn.H2DMode.HOST_PUSH,
        )
        self._trace_input_socket.set_page_size(self._trace_input_page_bytes)
        self._trace_output_socket = ttnn.D2HSocket(
            self.device,
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(*_TRACE_OUTPUT_SOCKET_CORE)),
            active_system_config().pipeline.d2h_fifo_bytes,
        )
        output_page_bytes = self._trace_output_words * 4
        self._trace_output_socket.set_page_size(output_page_bytes)
        self._trace_context = ttnn.zeros(
            [1, 1, self._trace_input_rows, self._trace_input_width],
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        self._trace_noise = ttnn.from_torch(
            torch.full((1, self.config.dspark_block_size - 1), self.config.dspark_noise_token_id, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        mask_row = dspark_block_mask(
            self._trace_context_len,
            self.config.dspark_block_size,
            self.config.sliding_window,
            torch.device("cpu"),
            torch.float32,
        )[:, :, :1]
        self._trace_attn_mask = ttnn.from_torch(
            mask_row, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

        # Compile every program before creating the trace. The compile run consumes the
        # input page and the output page exactly once, so both are written and read here.
        self._prefetch_trace_weights()
        compile_output = self._trace_body()
        self._trace_input_socket.write_tensor(packet)
        ttnn.experimental.send_async_d2h(compile_output, self._trace_output_socket)
        ttnn.synchronize_device(self.device)
        discarded = torch.empty((1, 1, 1, self._trace_output_words), dtype=torch.int32)
        self._trace_output_socket.read_tensor(discarded)
        compile_output.deallocate(True)

        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        output = self._trace_body()
        ttnn.experimental.send_async_d2h(output, self._trace_output_socket)
        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
        self._trace_id = trace_id
        self._trace_output = output

    def _write_trace_input(self, target_hiddens: torch.Tensor, anchor_id: int) -> None:
        """Push one page built from ``[1, S, L, D]`` ``target_hiddens`` + ``anchor_id`` into
        the H2D socket; raises if the traced path was never captured."""
        if self._trace_input_socket is None:
            raise RuntimeError("traced DSpark is not initialized")
        self._trace_input_socket.write_tensor(self._trace_input_page(target_hiddens, anchor_id))

    def replay_traced(self, target_hiddens: Optional[torch.Tensor] = None, anchor_id: Optional[int] = None) -> None:
        """Queue one trace execution without writing its input page.

        The first call also captures the trace and so needs ``target_hiddens``
        ``[1, S, L, D]`` and ``anchor_id``; later calls may pass neither.
        """
        if self._trace_id is None:
            if target_hiddens is None or anchor_id is None:
                raise RuntimeError("the first replay needs target_hiddens and anchor_id to capture the trace")
            self._capture_trace(target_hiddens, anchor_id)
        self._ensure_trace_replay_thread()
        self._trace_replay_queue.put(True)

    def write_step_packet(self, target_hiddens: torch.Tensor, anchor_id: int) -> None:
        """Write one page (``target_hiddens`` ``[1, S, L, D]``, ``anchor_id``) without
        dispatching device work -- dispatch first, e.g. via :meth:`decode_traced_async`."""
        self._write_trace_input(target_hiddens, anchor_id)

    def replay_traced_ahead(self, steps: Iterable[tuple[torch.Tensor, int]]) -> None:
        """Queue every trace execution before writing any input pages.

        ``steps`` is ``(target_hiddens [1, S, L, D], anchor_id)`` pairs. This mirrors
        :meth:`DeepSeekV4Model.replay_traced_ahead`: every trace is already parked in
        ``recv_async_h2d`` before the PCIe input writes begin, so each write lands on a
        trace that is waiting for it.
        """
        steps = list(steps)
        if not steps:
            return
        if self._trace_id is None:
            self._capture_trace(*steps[0])
        self._ensure_trace_replay_thread()
        for _ in steps:
            self._trace_replay_queue.put(True)

    def decode_traced_async(self, target_hiddens: torch.Tensor, anchor_id: int) -> None:
        """Dispatch one trace, then push the ``[1, S, L, D]`` page it is parked on.

        The order is the contract: dispatch first, write second.
        """
        if self._trace_id is None:
            self._capture_trace(target_hiddens, anchor_id)
        self.replay_traced()
        self.write_step_packet(target_hiddens, anchor_id)

    def read_decoded_output(self) -> torch.Tensor:
        """Read the oldest queued draft block from the D2H socket as ``[1, gamma]`` int32
        token IDs -- one page per dispatched replay, consumed in FIFO order."""
        if self._trace_output_socket is None or self._trace_output_words is None:
            raise RuntimeError("traced DSpark is not initialized")
        output = torch.empty((1, 1, 1, self._trace_output_words), dtype=torch.int32)
        self._trace_output_socket.read_tensor(output)
        return output.reshape(1, -1)[:, : self.config.dspark_block_size]

    def forward_traced(self, target_hiddens: torch.Tensor, anchor_id: int) -> torch.Tensor:
        """One blocking traced draft step: dispatch, write the ``[1, S, L, D]`` page, then
        return the ``[1, gamma]`` int32 token IDs from :meth:`read_decoded_output`."""
        self.decode_traced_async(target_hiddens, anchor_id)
        return self.read_decoded_output()

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
        """One eager draft round; only ``B == 1`` is supported.

        ``target_hiddens`` is ``[B, C, L, D]`` torch (the target model's tapped layer
        states) and ``anchor_ids`` ``[B]`` or ``[B, 1]``. Returns a
        :class:`~dspark.DSparkOutput`: ``draft_ids`` ``[B, gamma]``, ``logits`` /
        ``base_logits`` ``[B, gamma, V]`` (Markov-biased / backbone-only), ``confidence``
        and ``prefix_survival`` ``[B, gamma]``, ``hidden_states`` ``[B, gamma, D]``,
        ``context`` ``[B, C, D]`` and ``block_input_ids`` ``[B, gamma]``.

        ``greedy`` must be True (anything else raises) and ``temperature`` is ignored, so a
        non-greedy caller cannot silently get greedy output. ``min_survival`` zeroes draft
        positions past the :func:`~dspark.truncate_prefix` length; ``hoist_prefetch`` calls
        :meth:`prefetch_weights` before the pass.
        """
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

        total = ctx_len + gamma
        rd = cfg.head_dim
        cos_kv = ttnn.slice(self.cos_cached, [0, 0, 0, 0], [1, 1, total, rd])
        sin_kv = ttnn.slice(self.sin_cached, [0, 0, 0, 0], [1, 1, total, rd])
        cos_q = ttnn.slice(self.cos_cached, [0, 0, ctx_len, 0], [1, 1, total, rd])
        sin_q = ttnn.slice(self.sin_cached, [0, 0, ctx_len, 0], [1, 1, total, rd])
        mask_row = dspark_block_mask(ctx_len, gamma, cfg.sliding_window, torch.device("cpu"), torch.float32)[:, :, :1]
        attn_mask = ttnn.from_torch(mask_row, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        attn_kwargs = dict(
            attn_mask=attn_mask,
            cos_q=cos_q,
            sin_q=sin_q,
            cos_kv=cos_kv,
            sin_kv=sin_kv,
            rot=self.rot,
        )
        for stage in self.mtp:
            hidden = stage(hidden, context, **attn_kwargs)
        hidden = self.mtp[-1].norm(_as_decode_act(hidden))

        base_logits = self.lm_head(_matmul_a(self.lm_head, hidden))
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
        """Greedy sequential Markov head over the draft block.

        ``hidden`` is ``[1, 1, gamma, D]`` and ``base_logits`` ``[1, 1, gamma, V]`` (device,
        from the parallel backbone) and ``anchor_ids`` is ``[B]`` torch with B=1. Returns
        ``(draft [1, gamma] int64, step_logits [1, gamma, V] float32, confidence
        [1, gamma] float32)``: step ``k`` adds the Markov ``W2`` bias to row ``k`` of
        ``base_logits`` and takes the argmax, and the confidence head sigmoids element 0 of
        its zero-padded ``[V]`` output.

        Every step comes back to the host because the next step's embedding depends on the
        token just sampled; the traced body runs the equivalent loop entirely on device.
        """
        cfg = self.config
        gamma, vocab = cfg.dspark_block_size, cfg.vocab_size
        hidden = _as_decode_act(hidden)
        dim = cfg.hidden_size
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
            bias = _first_row(self.markov_w2(_matmul_a(self.markov_w2, markov_emb)))
            base_k = ttnn.slice(_as_decode_act(base_logits), [0, 0, k, 0], [1, 1, k + 1, vocab])
            logits_k = ttnn.add(_as_decode_act(base_k), bias)
            logits_pt = _to_torch_act(logits_k, 1, 1, vocab).view(vocab)
            next_id = int(logits_pt.argmax())
            draft[0, k] = next_id
            step_logits[0, k] = logits_pt
            h_k = ttnn.slice(hidden, [0, 0, k, 0], [1, 1, k + 1, dim])
            conf_in = ttnn.concat([_as_decode_act(h_k), markov_emb], dim=-1)
            conf_logits = _to_torch_act(
                _first_row(self.confidence_proj(_matmul_a(self.confidence_proj, conf_in))), 1, 1, vocab
            ).view(vocab)
            conf[0, k] = torch.sigmoid(conf_logits[0])
            prev = torch.tensor([next_id])
        return draft, step_logits, conf
