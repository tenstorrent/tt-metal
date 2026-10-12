# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash DSpark drafter (the checkpoint's ``mtp.0..2``) on one 1xTP stage.

DSpark is a block drafter. With the rings holding every verified position up to ``f`` and ``b``
the token for ``f + 1``, one pass over the block ``[b, noise, ..., noise]`` at positions
``f + 1 .. f + block_size`` proposes the tokens for ``f + 2 .. f + block_size + 1``:

* Every stage keeps a ``window``-slot ring of ``main_kv = kv_norm(wkv(main_x))`` (RoPE'd at the
  row's position), one row per verified position. ``main_x = main_norm(main_proj(main_hidden))``
  and ``main_hidden`` is the stream mean at the input of the target layers (37, 38, 39), which
  the decode model stashes for every step (see :attr:`DeepSeekV41DSpark.stash`).
* A stage is a V4.1 block: mHC around attention and a 128-expert top-3 MoE, ``pre_mix`` carried
  from stage to stage. Its attention is MQA over ``[ring | block kv]``, non-causal inside the
  block, with the attention sink, inverse RoPE on the output and the grouped ``wo_a``.
* The head collapses the streams with the last ``pre_mix``, applies the tied ``lm_head`` and
  walks the block left to right: row ``i``'s logits get the Markov bias of row ``i``'s input
  token, and the argmax is row ``i + 1``'s token. The confidence head scores each row.

The block's ``block_size`` rows run together as ``[1, 1, rows, D]`` TILE tensors (``[1, rows,
hc, D]`` streams), the layout of the V4 prefill blocks. Under TP the query heads, the ``wo_a``
groups and the experts' ``I`` are split over the ranks; ``wo_b`` and the MoE all-reduce their
partials, and everything else is replicated.

Two traceable entry points share persistent inputs the host fills per round:
:meth:`DeepSeekV41DSpark.write_main` writes up to ``main_rows`` verified rows into every ring,
and :meth:`DeepSeekV41DSpark.draft` drafts the block.
"""

from types import SimpleNamespace

import torch
import ttnn

from models.experimental.deepseek_v41_flash.tt.config import dspark_expert_provider, dspark_weights
from models.experimental.deepseek_v41_flash.tt.decode.moe import DeepSeekV41PreloadedExperts
from models.experimental.deepseek_v4_flash.tt.common import (
    _HIFI4,
    _MASK_NEG,
    DeepSeekV4Module,
    width_sharded_l1_config,
)
from models.experimental.deepseek_v4_flash.tt.decode.attention import _interleaved_rotate_matrix, make_rope_table
from models.experimental.deepseek_v4_flash.tt.decode.hyperconnection import _HC_COLLAPSE_CORES
from models.experimental.deepseek_v4_flash.tt.decode.moe import _tp_all_reduce
from models.experimental.deepseek_v4_flash.tt.layers import Linear
from models.experimental.deepseek_v4_flash.tt.prefill.decoder_layer import (
    load_norm_gamma,
    mix_streams,
    rows_from_tokens,
    tokens_from_rows,
)
from models.experimental.deepseek_v4_flash.tt.prefill.hyperconnection import (
    DeepSeekV4PrefillHyperConnection,
    flatten_streams,
    wide_rms_norm,
)
from models.experimental.deepseek_v4_flash.tt.weight_cache import _as_cache, _load_weight, _materialize, _memo

# Key rows SDPA sees: the ring, the block, the sink's zero row, padding to a tile.
_KV_ROWS = 160
# Column width of one stash buffer. paged_update_cache double-buffers whole rows on one core, so a
# hidden-wide (5120) row needs ~1.75 MB of circular buffers, more than L1.
STASH_PIECE = 1024


def _rope_tail(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor, rot: ttnn.Tensor) -> ttnn.Tensor:
    """Interleaved RoPE on the trailing ``Rd`` columns of every row of ``x`` ``[1, 1, R, W]`` (TILE).

    ``cos`` / ``sin`` are ``[1, 1, R, Rd]`` (one position per row); passing ``-sin`` undoes the rotation.
    """
    r, w = x.shape[2], x.shape[3]
    rd = rot.shape[-1]
    nope = ttnn.slice(x, [0, 0, 0, 0], [1, 1, r, w - rd])
    tail = ttnn.slice(x, [0, 0, 0, w - rd], [1, 1, r, w])
    rotated = ttnn.matmul(tail, rot, compute_kernel_config=_HIFI4)
    tail = ttnn.add(ttnn.multiply(tail, cos), ttnn.multiply(rotated, sin))
    return ttnn.concat([nope, tail], dim=-1)


def _rms(x: ttnn.Tensor, gamma: ttnn.Tensor, eps: float) -> ttnn.Tensor:
    return ttnn.multiply(wide_rms_norm(x, eps), gamma)


def host_to_device(value: torch.Tensor, dst: ttnn.Tensor) -> None:
    """Overwrite ``dst`` in place with ``value``, replicated over every chip of ``dst``'s mesh.

    Goes through command queue 0, so it lands between whatever is queued there before and after.
    """
    host = ttnn.from_torch(
        value, dtype=dst.dtype, layout=dst.layout, mesh_mapper=ttnn.ReplicateTensorToMesh(dst.device())
    )
    ttnn.copy_host_to_device_tensor(host, dst)


class DeepSeekV41DSparkHyperConnection(DeepSeekV4PrefillHyperConnection):
    """The prefill mHC with V4.1's carried ``pre_mix``: the streams collapse with the predecessor's
    ``pre``, and this call's ``pre`` is returned fourth."""

    def forward(self, hidden_streams: ttnn.Tensor, pre_mix: ttnn.Tensor):
        """``[1, T, hc, D]`` streams and ``[1, T, 1, hc]`` ``pre_mix`` -> ``(post, comb, collapsed, pre)``, DRAM.

        ``fused_hyperconnection`` runs once per token, on the single-user program that splits a token's
        ``[hc, D]`` slab over 8 cores. The multi-token program stages a whole slab on one core (~640 KB at
        D = 5120), which does not fit beside the decode model's L1 buffers on the output stage.
        """
        t, hc, d = hidden_streams.shape[1], hidden_streams.shape[2], hidden_streams.shape[3]
        flat = flatten_streams(hidden_streams)
        normed = wide_rms_norm(flat, self.norm_eps)
        ttnn.deallocate(flat)
        fused_w = self.fn(normed)
        ttnn.deallocate(normed)
        w = fused_w.shape[-1]
        device = hidden_streams.device()
        collapse_config = width_sharded_l1_config(hc, d, device, num_cores=_HC_COLLAPSE_CORES)
        fused_w_config = width_sharded_l1_config(1, -(-w // ttnn.TILE_SIZE) * ttnn.TILE_SIZE, device, num_cores=1)
        per_token = []
        for i in range(t):
            streams = ttnn.to_memory_config(
                ttnn.slice(hidden_streams, [0, i, 0, 0], [1, i + 1, hc, d]), collapse_config
            )
            outputs = ttnn.experimental.deepseek.fused_hyperconnection(
                streams,
                fused_w=ttnn.to_memory_config(ttnn.slice(fused_w, [0, 0, i, 0], [1, 1, i + 1, w]), fused_w_config),
                pre_bias=self.pre_b,
                post_bias=self.post_b,
                comb_bias=self.comb_b,
                num_streams=hc,
                sinkhorn_iters=self.iters,
                pre_scale=self.pre_scale,
                post_scale=self.post_scale,
                comb_scale=self.comb_scale,
                eps=self.eps,
                pre_mix=ttnn.slice(pre_mix, [0, i, 0, 0], [1, i + 1, 1, hc]),
            )
            per_token.append([ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG) for out in outputs])
            ttnn.deallocate(streams)
        ttnn.deallocate(fused_w)
        return tuple(ttnn.concat(list(parts), dim=1) for parts in zip(*per_token))


def dspark_moe_config(config) -> SimpleNamespace:
    """``config`` with the drafter's routed-expert count and top-k."""
    return SimpleNamespace(
        **{
            **vars(config),
            "num_local_experts": config.dspark_n_routed_experts,
            "n_routed_experts": config.dspark_n_routed_experts,
            "num_experts_per_tok": config.dspark_num_experts_per_tok,
        }
    )


class DeepSeekV41DSparkStage(DeepSeekV4Module):
    """DSpark stage ``stage`` (``mtp.{stage}``): ring, attention, MoE and the two mHCs.

    ``weights`` comes from :func:`~..config.dspark_weights` and ``experts`` is the stage's
    :class:`~.moe.DeepSeekV41PreloadedExperts` at :func:`dspark_moe_config`. ``rot`` is the shared
    ``[Rd, Rd]`` rotate matrix. ``weight_dtype`` is for the projections; the router gate,
    norms, sinks and hyper-connections stay bf16.
    """

    def __init__(
        self, config, stage: int, weights: dict, experts, device, rot, cache=None, weight_dtype=ttnn.bfloat8_b
    ):
        cache = _as_cache(cache)
        tp = experts.tp_size
        self.device = device
        self.tp_size = tp
        self.rot = rot
        self.eps = config.rms_norm_eps
        self.window = config.sliding_window
        self.head_dim = config.head_dim
        self.local_heads = config.num_attention_heads // tp
        self.local_groups = config.o_groups // tp
        self.block = config.dspark_block_size
        self.scale = self.head_dim**-0.5
        self.limit = config.swiglu_limit
        self.experts = experts
        tag = f".tp{tp}"
        col = ttnn.ShardTensorToMesh(device, dim=1) if tp > 1 else None
        row = ttnn.ShardTensorToMesh(device, dim=0) if tp > 1 else None

        def gamma(name):
            return load_norm_gamma(weights[f"{name}.weight"], device, cache.file(name))

        self.wq_a = Linear(weights["attn.wq_a.weight"], device, cache.file("attn.wq_a"), dtype=weight_dtype)
        self.q_norm = gamma("attn.q_norm")
        self.wq_b = Linear(
            weights["attn.wq_b.weight"], device, cache.file("attn.wq_b" + tag), dtype=weight_dtype, mesh_mapper=col
        )
        self.wkv = Linear(weights["attn.wkv.weight"], device, cache.file("attn.wkv"), dtype=weight_dtype)
        self.kv_norm = gamma("attn.kv_norm")
        wo_a = _memo(weights["attn.wo_a.weight"])  # [groups * o_lora, heads * head_dim / groups]
        o_lora = config.o_lora_rank
        wo_a_file = cache.file("attn.wo_a" + tag)
        self.wo_a = _load_weight(
            _materialize(
                lambda: wo_a()
                .reshape(config.o_groups, o_lora, -1)
                .transpose(1, 2)
                .reshape(tp, self.local_groups, -1, o_lora)
                .contiguous(),
                wo_a_file,
                weight_dtype,
            ),
            device,
            cache_file_name=wo_a_file,
            dtype=weight_dtype,
            mesh_mapper=row,
        )
        self.wo_b = Linear(
            weights["attn.wo_b.weight"], device, cache.file("attn.wo_b" + tag), dtype=weight_dtype, mesh_mapper=row
        )
        # Row r = t * local_heads + h of the folded scores: the sink of head h at the sink's key column.
        sink = _memo(weights["attn.attn_sink"])
        sink_file = cache.file("attn.sink_bias" + tag)

        def sink_bias():
            bias = torch.zeros(tp, 1, self.block * self.local_heads, _KV_ROWS)
            heads = sink().float().reshape(tp, 1, self.local_heads).repeat(1, self.block, 1)
            bias[:, 0, :, self.window + self.block] = heads.reshape(tp, -1)
            return bias

        self.sink_bias = _load_weight(
            _materialize(sink_bias, sink_file, ttnn.bfloat16),
            device,
            cache_file_name=sink_file,
            mesh_mapper=row,
        )
        self.ring = ttnn.from_torch(
            torch.zeros(1, 1, self.window, self.head_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.attn_norm = gamma("attn_norm")
        self.ffn_norm = gamma("ffn_norm")
        self.attn_hc = DeepSeekV41DSparkHyperConnection(
            config,
            {"fn": weights["hc_attn_fn"], "base": weights["hc_attn_base"], "scale": weights["hc_attn_scale"]},
            device,
            cache=cache.sub("hc_attn"),
        )
        self.ffn_hc = DeepSeekV41DSparkHyperConnection(
            config,
            {"fn": weights["hc_ffn_fn"], "base": weights["hc_ffn_base"], "scale": weights["hc_ffn_scale"]},
            device,
            cache=cache.sub("hc_ffn"),
        )

        gate = _memo(weights["ffn.gate.weight"])
        bias = _memo(weights["ffn.gate.bias"])
        temp = getattr(config, "gate_temp", 1.0)
        self.gate = Linear(lambda: gate() / temp, device, cache.file("ffn.gate"), dtype=ttnn.bfloat16)
        # The op ranks on bf16 ``scores + bias``; shifting the bias by its max keeps it where bf16 is fine.
        bias_file = cache.file("ffn.gate.bias_shifted")
        self.gate_bias = _load_weight(
            _materialize(lambda: (bias() - bias().max()).reshape(1, 1, 1, -1), bias_file, ttnn.bfloat16),
            device,
            cache_file_name=bias_file,
        )
        self.shared_w1 = Linear(
            weights["ffn.shared_experts.w1.weight"],
            device,
            cache.file("ffn.shared_experts.w1" + tag),
            dtype=weight_dtype,
            mesh_mapper=col,
        )
        self.shared_w3 = Linear(
            weights["ffn.shared_experts.w3.weight"],
            device,
            cache.file("ffn.shared_experts.w3" + tag),
            dtype=weight_dtype,
            mesh_mapper=col,
        )
        self.shared_w2 = Linear(
            weights["ffn.shared_experts.w2.weight"],
            device,
            cache.file("ffn.shared_experts.w2" + tag),
            dtype=weight_dtype,
            mesh_mapper=row,
        )

    def _all_reduce(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Sum the TP partials of ``[1, 1, T, D]`` (TILE). The rows go ROW_MAJOR first: the direct
        reduce-scatter stages its input over every core's L1, and a 32-row padded tile takes ~640 KB per core
        on the output stage."""
        if self.tp_size == 1:
            return x
        summed = _tp_all_reduce(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), self.device)
        return ttnn.to_layout(summed, ttnn.TILE_LAYOUT)

    def _kv(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """``[1, 1, R, D]`` -> the RoPE'd ``kv_norm(wkv(x))`` ``[1, 1, R, head_dim]``."""
        return _rope_tail(_rms(self.wkv(x), self.kv_norm, self.eps), cos, sin, self.rot)

    def write_main(self, main_x, cos, sin, scatter, keep) -> None:
        """Write the ``main_kv`` of ``main_x`` ``[1, 1, R, D]`` into the ring: ``ring = ring * keep + scatter @ kv``.

        ``scatter`` ``[1, 1, window, R]`` is one-hot (row ``j`` -> its slot, zero for rows not written)
        and ``keep`` ``[1, 1, window, 1]`` zeroes the written slots. Both products are exact.
        """
        kv = self._kv(main_x, cos, sin)
        written = ttnn.matmul(scatter, kv, compute_kernel_config=_HIFI4)
        ttnn.copy(ttnn.add(ttnn.multiply(self.ring, keep), written), self.ring)

    def attention(self, x, q_cos, q_sin, q_neg_sin, kv_cos, kv_sin, bias_row) -> ttnn.Tensor:
        """The block's attention: ``x`` ``[1, 1, T, D]`` -> ``[1, 1, T, D]`` (all-reduced).

        The ``T * local_heads`` query rows are folded onto one axis and attend the shared keys
        ``[ring | block kv | sink | padding]``; ``bias_row`` ``[1, 1, 1, _KV_ROWS]`` masks the
        empty ring slots and the padding, and the sink row's score is its per-head bias.
        """
        t = x.shape[2]
        q = self.wq_b(_rms(self.wq_a(x), self.q_norm, self.eps))
        q = ttnn.reshape(q, [1, 1, t * self.local_heads, self.head_dim])
        q = _rope_tail(q, q_cos, q_sin, self.rot)
        kv = self._kv(x, kv_cos, kv_sin)
        kv = ttnn.pad(kv, [(0, 0), (0, 0), (0, _KV_ROWS - self.window - t), (0, 0)], 0.0)
        keys = ttnn.concat([self.ring, kv], dim=2)
        scores = ttnn.matmul(q, keys, transpose_b=True, dtype=ttnn.float32, compute_kernel_config=_HIFI4)
        scores = ttnn.add(ttnn.multiply(scores, self.scale), ttnn.add(self.sink_bias, bias_row))
        probs = ttnn.typecast(ttnn.softmax(scores, dim=-1, compute_kernel_config=_HIFI4), ttnn.bfloat16)
        o = ttnn.matmul(probs, keys, compute_kernel_config=_HIFI4)
        o = _rope_tail(o, q_cos, q_neg_sin, self.rot)
        o = ttnn.reshape(o, [1, t, self.local_groups, -1])
        o = ttnn.matmul(ttnn.permute(o, (0, 2, 1, 3)), self.wo_a, compute_kernel_config=_HIFI4)
        o = ttnn.reshape(ttnn.permute(o, (0, 2, 1, 3)), [1, 1, t, -1])
        return self._all_reduce(self.wo_b(o))

    def moe(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Routed experts plus the shared expert: ``[1, 1, T, D]`` -> ``[1, 1, T, D]`` (all-reduced).

        ``fused_experts`` runs once per row, on the ROW_MAJOR single-token path decode uses. A multi-row
        call would take TILE rows padded to 32, whose circular buffers do not fit beside the decode model's
        L1 buffers on the output stage.
        """
        t, d = x.shape[2], x.shape[3]
        experts = self.experts
        logits = ttnn.linear(x, self.gate.weight, dtype=ttnn.float32, compute_kernel_config=_HIFI4)
        scores = ttnn.typecast(ttnn.sqrt(ttnn.softplus(logits)), ttnn.bfloat16)
        ranking = ttnn.add(scores, self.gate_bias)
        e = scores.shape[3]
        routed = []
        for i in range(t):
            row = ttnn.to_layout(ttnn.slice(x, [0, 0, i, 0], [1, 1, i + 1, d]), ttnn.ROW_MAJOR_LAYOUT)
            out = ttnn.experimental.deepseek.moe.fused_experts(
                row,
                routing_scores=ttnn.slice(scores, [0, 0, i, 0], [1, 1, i + 1, e]),
                ranking_scores=ttnn.slice(ranking, [0, 0, i, 0], [1, 1, i + 1, e]),
                gate_up_weights=experts.gate_up_weights,
                down_weights=experts.down_weights,
                num_experts=experts.top_k,
                intermediate_size=experts.intermediate,
                swiglu_limit=experts.limit,
                top_k=experts.top_k,
                routed_scaling_factor=experts.routed_scaling_factor,
                routing_eps=experts.routing_eps,
                experts_block_size=min(experts.experts_block_size, experts.top_k),
            )
            routed.append(ttnn.to_layout(ttnn.reshape(out, [1, 1, 1, d]), ttnn.ROW_MAJOR_LAYOUT))
        routed = ttnn.to_layout(ttnn.concat(routed, dim=2), ttnn.TILE_LAYOUT)
        gate = ttnn.clamp(self.shared_w1(x), max=self.limit)
        up = ttnn.clamp(self.shared_w3(x), min=-self.limit, max=self.limit)
        shared = self.shared_w2(ttnn.multiply(ttnn.silu(gate), up))
        return self._all_reduce(ttnn.add(routed, shared))

    def forward(self, streams, pre_mix, rope: dict, bias_row) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """``(streams [1, T, hc, D], pre_mix [1, T, 1, hc])`` -> the same pair for the next stage."""
        post, comb, collapsed, attn_pre = self.attn_hc(streams, pre_mix)
        normed = _rms(rows_from_tokens(collapsed), self.attn_norm, self.eps)
        attn_out = self.attention(normed, *rope["q"], *rope["kv"], bias_row)
        streams = mix_streams(post, comb, attn_out, streams)
        post, comb, collapsed, ffn_pre = self.ffn_hc(streams, attn_pre)
        normed = _rms(rows_from_tokens(collapsed), self.ffn_norm, self.eps)
        streams = mix_streams(post, comb, self.moe(normed), streams)
        return streams, ffn_pre


class DeepSeekV41DSpark(DeepSeekV4Module):
    """The three DSpark stages plus ``main_proj``, the head and the per-round inputs, on ``device``.

    ``lm_head`` is the decode model's tied head (a :class:`~models.experimental.deepseek_v4_flash.tt.layers.Linear`
    on the same ``device``), ``rope_main`` the sliding ``(cos_half, sin_half)`` ``[max_seq, Rd / 2]`` tables.
    ``main_rows`` is how many verified rows one :meth:`write_main` takes and ``stash_rows`` the depth of
    :attr:`stash`, the ring the decode model writes each step's target-layer means into at
    ``pos % stash_rows`` (per target, ``[1, 1, stash_rows, STASH_PIECE]`` column pieces).

    ``expert_dtype`` defaults to bfloat4_b, like the decode MoE: ``fused_experts`` double-buffers a full-K
    weight slice per core (~700 KB at bfloat8_b), which does not fit below the decode prefetch rings on the
    output stage. bfloat8_b is more faithful (bfloat4_b flips more of the top-3-of-128 routes) and fits on
    an otherwise empty device.
    """

    def __init__(
        self,
        config,
        loader,
        device,
        lm_head,
        rope_main,
        cache=None,
        weight_dtype=ttnn.bfloat8_b,
        expert_dtype=ttnn.bfloat4_b,
        tp_size: int = 4,
        main_rows: int = 8,
        stash_rows: int = 256,
    ):
        cache = _as_cache(cache)
        self.config = config
        self.device = device
        self.lm_head = lm_head
        self.cos_half, self.sin_half = rope_main
        self.eps = config.rms_norm_eps
        self.hidden = config.hidden_size
        self.hc = config.hc_mult
        self.window = config.sliding_window
        self.block = config.dspark_block_size
        self.noise_id = config.dspark_noise_token_id
        self.targets = list(config.dspark_target_layer_ids)
        self.main_rows = main_rows
        self.stash_rows = stash_rows
        self.rope_dim = config.qk_rope_head_dim
        self.local_heads = config.num_attention_heads // tp_size
        replicate = ttnn.ReplicateTensorToMesh(device) if tp_size > 1 else None
        self._replicate = replicate

        self.rot = ttnn.from_torch(
            _interleaved_rotate_matrix(self.rope_dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        moe_config = dspark_moe_config(config)
        self.stages = []
        weights = [dspark_weights(loader, s) for s in range(config.num_nextn_predict_layers)]
        for s, w in enumerate(weights):
            stage_cache = cache.sub(f"mtp.{s}")
            experts = DeepSeekV41PreloadedExperts(
                moe_config,
                dspark_expert_provider(loader, s),
                device,
                dtype=expert_dtype,
                cache=stage_cache.sub("ffn"),
                tp_size=tp_size,
            )
            self.stages.append(
                DeepSeekV41DSparkStage(
                    config, s, w, experts, device, self.rot, cache=stage_cache, weight_dtype=weight_dtype
                )
            )
        first, last = weights[0], weights[-1]
        head_cache = cache.sub("mtp")
        self.main_proj = Linear(first["main_proj.weight"], device, head_cache.file("main_proj"), dtype=weight_dtype)
        self.main_norm = load_norm_gamma(first["main_norm.weight"], device, head_cache.file("main_norm"))
        self.norm = load_norm_gamma(last["norm.weight"], device, head_cache.file("norm"))
        markov_file = head_cache.file("markov_head.embed")
        self.markov_embed = _load_weight(
            _materialize(last["markov_head.embed.weight"], markov_file, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            device,
            cache_file_name=markov_file,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.markov_head = Linear(
            last["markov_head.head.weight"], device, head_cache.file("markov_head.head"), dtype=ttnn.bfloat16
        )
        conf = _memo(last["confidence_head.proj.weight"])  # [1, D + rank]
        self.conf_hidden = Linear(
            lambda: conf()[:, : self.hidden], device, head_cache.file("confidence.hidden"), dtype=ttnn.float32
        )
        self.conf_markov = Linear(
            lambda: conf()[:, self.hidden :], device, head_cache.file("confidence.markov"), dtype=ttnn.float32
        )
        self.host_embed = loader.get_tensor("embed.weight")  # [V, D] bf16, for the block's input rows

        def zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            host = torch.zeros(shape, dtype=torch.int32 if dtype == ttnn.uint32 else torch.float32)
            return ttnn.from_torch(
                host,
                dtype=dtype,
                layout=layout,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate,
            )

        w, r, t = main_rows, self.rope_dim, self.block
        self.stash = [
            [zeros((1, 1, stash_rows, STASH_PIECE)) for _ in range(self.hidden // STASH_PIECE)] for _ in self.targets
        ]
        self.inputs = {
            "gather": zeros((1, 1, w, stash_rows)),
            "scatter": zeros((1, 1, self.window, w)),
            "keep": zeros((1, 1, self.window, 1)),
            "main_cos": zeros((1, 1, w, r)),
            "main_sin": zeros((1, 1, w, r)),
            "x_embed": zeros((1, 1, t, self.hidden)),
            "token": zeros((1, 1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            "q_cos": zeros((1, 1, t * self.local_heads, r)),
            "q_sin": zeros((1, 1, t * self.local_heads, r)),
            "q_neg_sin": zeros((1, 1, t * self.local_heads, r)),
            "kv_cos": zeros((1, 1, t, r)),
            "kv_sin": zeros((1, 1, t, r)),
            "bias_row": zeros((1, 1, 1, _KV_ROWS)),
        }
        one_hot = torch.zeros(1, t, 1, self.hc)
        one_hot[..., 0] = 1
        self.pre_mix_init = ttnn.from_torch(
            one_hot, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=replicate
        )

    # ------------------------------------------------------------------ host side
    def _rope_rows(self, positions) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.as_tensor(positions, dtype=torch.long)
        return make_rope_table(self.cos_half[idx], self.sin_half[idx])

    def set_main_rows(self, positions, written) -> None:
        """Stage :meth:`write_main` for verified ``positions`` (at most ``main_rows``); ``written[j]`` is
        False for a row whose ring slot must not change (a rejected or padding row)."""
        w = self.main_rows
        positions = list(positions)
        assert 0 < len(positions) <= w, f"{len(positions)} rows for a {w}-row write"
        written = list(written) + [False] * (w - len(positions))
        positions = positions + [positions[-1]] * (w - len(positions))
        gather = torch.zeros(1, 1, w, self.stash_rows)
        scatter = torch.zeros(1, 1, self.window, w)
        for j, (p, keep_row) in enumerate(zip(positions, written)):
            gather[0, 0, j, p % self.stash_rows] = 1
            if keep_row:
                scatter[0, 0, p % self.window, j] = 1
        cos, sin = self._rope_rows(positions)
        host_to_device(gather, self.inputs["gather"])
        host_to_device(scatter, self.inputs["scatter"])
        host_to_device(1 - scatter.sum(-1, keepdim=True), self.inputs["keep"])
        host_to_device(cos, self.inputs["main_cos"])
        host_to_device(sin, self.inputs["main_sin"])

    def set_block(self, token: int, frontier: int) -> None:
        """Stage :meth:`draft` for ``token`` at position ``frontier + 1``, ``frontier`` being the last
        position in the rings."""
        t, heads = self.block, self.local_heads
        ids = torch.tensor([token] + [self.noise_id] * (t - 1))
        host_to_device(self.host_embed[ids].float().reshape(1, 1, t, self.hidden), self.inputs["x_embed"])
        host_to_device(torch.tensor([[token]], dtype=torch.int32), self.inputs["token"])
        cos, sin = self._rope_rows(range(frontier + 1, frontier + 1 + t))
        q_cos, q_sin = (x.repeat_interleave(heads, dim=2) for x in (cos, sin))
        host_to_device(q_cos, self.inputs["q_cos"])
        host_to_device(q_sin, self.inputs["q_sin"])
        host_to_device(-q_sin, self.inputs["q_neg_sin"])
        host_to_device(cos, self.inputs["kv_cos"])
        host_to_device(sin, self.inputs["kv_sin"])
        bias = torch.full((1, 1, 1, _KV_ROWS), _MASK_NEG)
        bias[..., : min(self.window, frontier + 1)] = 0
        bias[..., self.window : self.window + t + 1] = 0
        host_to_device(bias, self.inputs["bias_row"])

    # ------------------------------------------------------------------ device side (traceable)
    def write_main(self) -> None:
        """Gather the staged rows' target-layer means from :attr:`stash` and write their ``main_kv`` into every ring."""
        inp = self.inputs
        rows = ttnn.concat(
            [
                ttnn.matmul(inp["gather"], piece, compute_kernel_config=_HIFI4)
                for pieces in self.stash
                for piece in pieces
            ],
            dim=-1,
        )
        main_x = _rms(self.main_proj(rows), self.main_norm, self.eps)
        for stage in self.stages:
            stage.write_main(main_x, inp["main_cos"], inp["main_sin"], inp["scatter"], inp["keep"])

    def draft(self) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Draft the staged block: ``(ids [1, 1, block] uint32, confidence [1, 1, block, 1] fp32)``.

        Row ``i`` sits at position ``frontier + 1 + i`` and ``ids[i]``, its argmax, is the proposal for
        position ``frontier + 2 + i``; ``confidence[i]`` scores it.
        """
        inp, t = self.inputs, self.block
        streams = ttnn.to_layout(
            ttnn.repeat(tokens_from_rows(inp["x_embed"]), ttnn.Shape([1, 1, self.hc, 1])), ttnn.TILE_LAYOUT
        )
        pre_mix = self.pre_mix_init
        rope = {"q": (inp["q_cos"], inp["q_sin"], inp["q_neg_sin"]), "kv": (inp["kv_cos"], inp["kv_sin"])}
        for stage in self.stages:
            streams, pre_mix = stage.forward(streams, pre_mix, rope, inp["bias_row"])

        streams_f = ttnn.typecast(streams, ttnn.float32)
        weights = ttnn.typecast(pre_mix, ttnn.float32)
        collapsed = None
        for j in range(self.hc):
            term = ttnn.multiply(
                ttnn.slice(streams_f, [0, 0, j, 0], [1, t, j + 1, self.hidden]),
                ttnn.slice(weights, [0, 0, 0, j], [1, t, 1, j + 1]),
            )
            collapsed = term if collapsed is None else ttnn.add(collapsed, term)
        x = rows_from_tokens(ttnn.typecast(collapsed, ttnn.bfloat16))
        logits = self.lm_head(_rms(x, self.norm, self.eps))

        rank = self.markov_embed.shape[-1]
        token = inp["token"]
        ids, embeds = [], []
        for i in range(t):
            embed = ttnn.reshape(ttnn.embedding(token, self.markov_embed), [1, 1, 1, rank])
            embeds.append(embed)
            row = ttnn.slice(logits, [0, 0, i, 0], [1, 1, i + 1, logits.shape[-1]])
            row = ttnn.add(row, self.markov_head(ttnn.to_layout(embed, ttnn.TILE_LAYOUT)))
            best = ttnn.argmax(ttnn.to_layout(row, ttnn.ROW_MAJOR_LAYOUT), dim=-1)
            ids.append(ttnn.reshape(best, [1, 1, 1]))
            token = ttnn.reshape(best, [1, 1])
        markov = ttnn.to_layout(ttnn.concat(embeds, dim=2), ttnn.TILE_LAYOUT)
        conf = ttnn.add(
            ttnn.linear(x, self.conf_hidden.weight, dtype=ttnn.float32, compute_kernel_config=_HIFI4),
            ttnn.linear(markov, self.conf_markov.weight, dtype=ttnn.float32, compute_kernel_config=_HIFI4),
        )
        return ttnn.concat(ids, dim=-1), conf
