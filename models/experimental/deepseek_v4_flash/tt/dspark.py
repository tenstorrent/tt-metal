# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn DSpark drafter: ``LinearDecode`` + DRISC weight prefetcher.

Port of :mod:`models.experimental.deepseek_v4_flash.dspark`. Every learned
projection (``main_proj``, Q/K/V/O, SwiGLU, LM head, Markov ``W2``, confidence)
is a :class:`~.layers.LinearDecode` whose weight is streamed through one shared
decode GCB. Activations are the same ROW_MAJOR HEIGHT_SHARDED replica
``matmul_decode`` consumes in Flash attention (full K on every B core).
``main_proj`` fuses its RMSNorm in the matmul epilogue, matching ``q_a`` / ``kv``.
RoPE is :func:`~.attention._apply_rope` (``fused_partial_rope``). Attention is
``scaled_dot_product_attention_decode`` with the draft block as the decode batch
and ``share_cache`` so every query sees the same injected K/V.

The sequential Markov loop self-queues each step so it does not occupy the GCB
FIFO for ``gamma`` repeats at hoist time. ``prefetch_weights`` therefore covers
the parallel backbone (plus the LM head) in forward order.

K/V of context and the draft block are packed into one ``k_proj`` / ``v_proj``
call each so those weights are consumed once per stage, matching a single
hoisted prefetch request.
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


# Keep these cores separate from the two cores used by the model pipeline
# hand-off sockets.  The traced DSpark path is normally placed on an idle
# submesh, but using explicit local coordinates also makes it usable with a
# single-device test mesh.
_TRACE_INPUT_SOCKET_CORE = (0, 2)
_TRACE_OUTPUT_SOCKET_CORE = (0, 3)


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
    """Fold ``x`` onto ``[1, 1, tokens, dim]`` in DRAM, then ``LinearDecode`` replicates it."""
    x = _to_dram(x)
    tokens, dim = x.shape[-2], x.shape[-1]
    if list(x.shape) != [1, 1, tokens, dim]:
        x = ttnn.reshape(x, [1, 1, tokens, dim])
    return x


def _matmul_a(layer: LinearDecode, x: ttnn.Tensor) -> ttnn.Tensor:
    """TILE WIDTH_SHARDED L1 ``[1, 1, tokens, dim]`` for ``use_rm_hs=False``.

    The replicated ROW_MAJOR HEIGHT_SHARDED path is Flash's M=1 decode layout and
    rejects M>8. DSpark's block is a full tile of tokens, so every backbone
    projection uses this path. Markov is M=1 but stays TILE as well so it shares
    the same kernel as the LM head (the replica path disagreed with the reference
    at the vocab projection).
    """
    return layer.to_width_sharded_activation(_as_decode_act(x))


def _to_torch_act(x: ttnn.Tensor, batch: int, seq: int, dim: int) -> torch.Tensor:
    return ttnn.to_torch(_to_dram(x)).float().reshape(batch, seq, dim)


def _first_row(x: ttnn.Tensor) -> ttnn.Tensor:
    """Drop extra HEIGHT_SHARDED replica rows so a decode matmul is ``[1, 1, 1, N]``."""
    x = _as_decode_act(x)
    if x.shape[-2] != 1:
        x = ttnn.slice(x, [0, 0, 0, 0], [1, 1, 1, x.shape[-1]])
    return x


class DSparkAttention(DeepSeekV4Module):
    """Block queries, KV-injected context; Q/K/V/O via prefetched ``LinearDecode``.

    RoPE is Flash's fused interleaved ``rotate_half``. The ``gamma`` draft tokens are
    the SDPA-decode batch (S=1 per query); ``share_cache`` reuses one K/V sequence
    (injected context + block). Q is TILE height-sharded because MHA cannot use the
    ROW_MAJOR Q path (that path requires ``n_kv_heads == 1``).
    """

    def __init__(
        self, config: DSparkConfig, weights: dict, prefix: str, device, cache, dtype, n_blocks, global_cb, page_bytes
    ):
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
        self.gate_proj.fetch_weights()
        self.up_proj.fetch_weights()
        self.down_proj.fetch_weights()

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        act = _matmul_a(self.gate_proj, x)
        gated = ttnn.multiply(ttnn.silu(self.gate_proj(act)), self.up_proj(act))
        return self.down_proj(_matmul_a(self.down_proj, gated))


class DSparkStage(DeepSeekV4Module):
    def __init__(
        self, config, weights, prefix, device, cache, dtype, n_blocks, global_cb, page_bytes, *, has_main, has_heads
    ):
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
        if self.main_proj is not None:
            self.main_proj.fetch_weights()
        self.attn.prefetch_weights()
        self.mlp.prefetch_weights()

    def fuse_context(self, stacked: ttnn.Tensor) -> ttnn.Tensor:
        fused = self.main_proj(_matmul_a(self.main_proj, stacked))
        return fused if self.main_norm is None else self.main_norm(fused)

    def forward(self, hidden_states, context, **attn_kwargs) -> ttnn.Tensor:
        residual = _as_decode_act(hidden_states)
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

        # Traced drafting is initialized lazily because the input context length
        # is part of the trace shape.  These fields are deliberately kept
        # separate from the eager path: a trace owns the addresses of every
        # tensor participating in it.
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
        weights = {k: v.detach().float() for k, v in torch_model.state_dict().items()}
        return cls(torch_model.config, weights, device, **kwargs)

    def prefetch_weights(self) -> None:
        """Queue backbone weights in the order :meth:`forward` consumes them."""
        for stage in self.mtp:
            stage.prefetch_weights()
        self.lm_head.fetch_weights()
        self.markov_w2.fetch_weights()
        self.confidence_proj.fetch_weights()

    def _prefetch_trace_weights(self) -> None:
        """Queue only weights consumed by the device-only traced draft."""
        for stage in self.mtp:
            stage.prefetch_weights()
        self.lm_head.fetch_weights()
        self.markov_w2.fetch_weights()

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

    def _fuse_target_hiddens_device(self, target_hiddens: ttnn.Tensor) -> ttnn.Tensor:
        """Fuse a device-resident ``[1, 1, S, L*D]`` context packet.

        The eager API accepts a torch tensor and uploads it.  The traced API
        receives the packet from an H2D socket, so all work after the receive
        must stay on device.
        """
        return self.mtp[0].fuse_context(ttnn.to_layout(target_hiddens, ttnn.TILE_LAYOUT))

    def _embed_device(self, ids: ttnn.Tensor) -> ttnn.Tensor:
        """Embedding variant for IDs produced inside a trace."""
        emb = ttnn.embedding(ids, self.embed_tokens, layout=ttnn.TILE_LAYOUT)
        return ttnn.reshape(emb, [1, 1, self.config.dspark_block_size, self.config.hidden_size])

    def _trace_input_page(self, target_hiddens: torch.Tensor, anchor_id: int) -> torch.Tensor:
        """Build one PCIe-aligned H2D page for a traced drafting step.

        The context is sent as FP32 so the anchor token can share the page
        without losing precision for the 129K vocabulary.  The device casts
        only the context slice to BF16 before entering the model.
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
        # H2D FIFOs are deliberately small (typically one 4 KiB page), so a
        # large hidden row must be split into several socket pages. The trace
        # reassembles the flat stream before applying the model.
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
        """Device-only DSpark block used for compile and trace capture."""
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

        # The eager implementation reads each Markov step to the host.  In a
        # trace, keep the recurrent token on device and only export the final
        # draft block.
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
        if self._trace_replay_thread is not None:
            return

        def _run() -> None:
            for _ in iter(self._trace_replay_queue.get, None):
                ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=False)

        self._trace_replay_thread = threading.Thread(target=_run, name="dspark-trace-replay", daemon=True)
        self._trace_replay_thread.start()

    def shutdown(self) -> None:
        """Stop the trace-replay thread started by :meth:`_ensure_trace_replay_thread`.

        The thread is a daemon whose target closes over ``self``, and nothing ever tells it
        to stop. CPython does not unwind a daemon thread's frame at shutdown, so the closure
        keeps this model -- and every ttnn tensor it owns -- alive until the process exits,
        where nanobind's ``Py_AtExit`` leak check reports it. Stopping the thread releases
        that reference so the model can be collected normally.

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

        # Compile all programs before creating the trace.  The input page and
        # output page are consumed by the compile run exactly once.
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
        if self._trace_input_socket is None:
            raise RuntimeError("traced DSpark is not initialized")
        self._trace_input_socket.write_tensor(self._trace_input_page(target_hiddens, anchor_id))

    def replay_traced(self, target_hiddens: Optional[torch.Tensor] = None, anchor_id: Optional[int] = None) -> None:
        """Queue one trace execution without writing its input page."""
        if self._trace_id is None:
            if target_hiddens is None or anchor_id is None:
                raise RuntimeError("the first replay needs target_hiddens and anchor_id to capture the trace")
            self._capture_trace(target_hiddens, anchor_id)
        self._ensure_trace_replay_thread()
        self._trace_replay_queue.put(True)

    def write_step_packet(self, target_hiddens: torch.Tensor, anchor_id: int) -> None:
        """Write one traced DSpark input page without dispatching device work."""
        self._write_trace_input(target_hiddens, anchor_id)

    def replay_traced_ahead(self, steps: Iterable[tuple[torch.Tensor, int]]) -> None:
        """Queue all trace executions before writing any input pages.

        This mirrors :meth:`DeepSeekV4Model.replay_traced_ahead`: every trace is
        already parked in ``recv_async_h2d`` before PCIe input writes begin.
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
        """Dispatch one trace before pushing its H2D page."""
        if self._trace_id is None:
            self._capture_trace(target_hiddens, anchor_id)
        self.replay_traced()
        self.write_step_packet(target_hiddens, anchor_id)

    def read_decoded_output(self) -> torch.Tensor:
        """Read the oldest traced draft block as ``[1, gamma]`` token IDs."""
        if self._trace_output_socket is None or self._trace_output_words is None:
            raise RuntimeError("traced DSpark is not initialized")
        output = torch.empty((1, 1, 1, self._trace_output_words), dtype=torch.int32)
        self._trace_output_socket.read_tensor(output)
        return output.reshape(1, -1)[:, : self.config.dspark_block_size]

    def forward_traced(self, target_hiddens: torch.Tensor, anchor_id: int) -> torch.Tensor:
        """Run one traced draft block and return its token IDs."""
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
