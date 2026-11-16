# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash DSpark drafter (the checkpoint's ``mtp.0..2``) on one 1xTP stage.

DSpark is a block drafter. With the rings holding every verified position up to ``f`` and ``b``
the token for ``f + 1``, one pass over the block ``[b, noise, ..., noise]`` at positions
``f + 1 .. f + block_size`` proposes the tokens for ``f + 2 .. f + block_size + 1``:

* Every stage keeps a ring of ``main_kv = kv_norm(wkv(main_x))`` (RoPE'd at the row's position), one
  row per verified position at slot ``pos % ring_rows``. ``main_x = main_norm(main_proj(main_hidden))``
  and ``main_hidden`` is the stream mean at the input of the target layers (37, 38, 39), which
  the decode model stashes for every step (see :attr:`DeepSeekV41DSpark.stash`).
* A stage is a V4.1 block: mHC around attention and a 128-expert top-3 MoE, ``pre_mix`` carried
  from stage to stage. Its attention is MQA over ``[ring | block kv]``, non-causal inside the
  block, with the attention sink, inverse RoPE on the output and the grouped ``wo_a``.
* The head collapses the streams with the last ``pre_mix``, applies the tied ``lm_head`` and
  walks the block left to right: row ``i``'s logits get the Markov bias of row ``i``'s input
  token, and the argmax is row ``i + 1``'s token. The confidence head scores each row. ``T`` times
  Gumbel noise, from uniforms drawn on device, is added before each argmax: it samples the row at
  temperature ``T`` (Gumbel-max), and ``T = 0`` drafts greedily.

The block's ``block_size`` rows run together as ``[1, 1, rows, D]`` TILE tensors (``[1, rows,
hc, D]`` streams), the layout of the V4 prefill blocks. Under TP the query heads, the ``wo_a``
groups and the experts' ``I`` are split over the ranks; ``wo_b`` and the MoE all-reduce their
partials, and everything else is replicated.

A stage's projections have the decode layers' shapes, so they are the decode layers'
``matmul_decode`` :class:`LinearDecode` layers, fed the ``[rows, K]`` ROW_MAJOR replica the decode
model feeds a batch of users (``rows`` is at most ``main_rows``, which the replica path caps at 8).
Under ``use_prefetcher`` ``wq_a``, ``wq_b``, ``wkv``, ``wo_b`` and the shared expert's ``w2``
stream through the drafter's own rings (:func:`dspark_prefetch_rings`): it runs on a pipeline stage
of its own, so its rings are sized for its dtypes and share no FIFO with a decode step. Each matmul
queues its own request as it runs, so a ring's requests are sent in the order its matmuls are queued.
The shared expert's ``w1`` / ``w3`` are not on a ring, so their matmuls read the DRAM ND-sharded
weight themselves. ``wo_a``
(the decode ``o_a_proj`` is single-row only), ``main_proj`` (whose slab fits no ring), the router
gate and the head keep plain matmuls.

The traceable entry points share persistent inputs the host fills per round:
:meth:`DeepSeekV41DSpark.write_main` writes up to ``main_rows`` verified rows into every ring,
and :meth:`DeepSeekV41DSpark.draft` drafts the block. A decode or verify step of the model instead hands
its rows and positions to :meth:`DeepSeekV41DSpark.write_rows` device to device, with nothing from the host,
and a decode step its logits to :meth:`DeepSeekV41DSpark.write_logits`. A draft's inputs are one INT32
packet (the block's ids and positions, the frontier's slot); the draft embeds the ids, looks up the RoPE
rows and builds the attention mask from it on device. :meth:`DeepSeekV41DSpark.accept_draft` runs a
round's accept rule over those logits and the last draft, and drafts on from its result without the host:
its packet carries the round's start, and the accepted count shifts it on device.

Each mHC is the decode block's (:class:`DeepSeekV4HyperConnection`): its ``fn`` cuts the ``hc * D`` K over
cores and folds the unweighted norm in, where a plain matmul runs the one-tile output on one core; then
the block's ``T`` rows go through one multi-token ``fused_hyperconnection`` (one token per core, the token's
``[hc, D]`` slab staged whole): the drafter's stage has no decode model's L1 buffers beside it.

A round writes every row it verified, the rejected ones too: the ring is ``window + main_rows - 1`` rows
or more, so a rejected row (at most ``main_rows - 1`` past the frontier) never shares a slot with one of
the ``window`` live positions, and the draft's ``bias_row`` masks every slot outside the live window. The
next round verifies from the slot after the frontier and so rewrites every rejected row.
"""

import math
import os
from types import SimpleNamespace

import torch
import ttnn

from models.experimental.deepseek_v41_flash.tt.config import dspark_expert_provider, dspark_weights
from models.experimental.deepseek_v41_flash.tt.decode.attention import DECODE_LAYOUTS as ATTENTION_LAYOUTS
from models.experimental.deepseek_v41_flash.tt.decode.attention import _rope_tail, _rows_on_one_core
from models.experimental.deepseek_v41_flash.tt.decode.decoder_layer import _gain_block
from models.experimental.deepseek_v41_flash.tt.decode.moe import DECODE_LAYOUTS as MOE_LAYOUTS
from models.experimental.deepseek_v41_flash.tt.decode.moe import DeepSeekV41PreloadedExperts
from models.experimental.deepseek_v4_flash.tt.common import (
    _HIFI4,
    _MASK_NEG,
    CCL_TOPOLOGY,
    DeepSeekV4Module,
    width_sharded_l1_config,
)
from models.experimental.deepseek_v4_flash.tt.decode.attention import (
    _decode_activation,
    _tp_cluster_axis,
    _interleaved_rotate_matrix,
    make_rope_table,
)
from models.experimental.deepseek_v4_flash.tt.decode.decode_prefetch import (
    PrefetchRing,
    check_decode_layout,
    ring_prefetch,
    tp_gate_up_layout,
)
from models.experimental.deepseek_v4_flash.tt.decode.hyperconnection import DeepSeekV4HyperConnection
from models.experimental.deepseek_v4_flash.tt.decode.moe import _tp_all_reduce
from models.experimental.deepseek_v4_flash.tt.embedding import DeepSeekV4Embedding
from models.experimental.deepseek_v4_flash.tt.layers import BatchedLinearDecode, Linear, LinearDecode
from models.experimental.deepseek_v4_flash.tt.prefill.decoder_layer import (
    load_norm_gamma,
    mix_streams,
    rows_from_tokens,
    tokens_from_rows,
)
from models.experimental.deepseek_v4_flash.tt.prefill.hyperconnection import wide_rms_norm
from models.experimental.deepseek_v4_flash.tt.weight_cache import _as_cache, _load_weight, _materialize, _memo

# Column width of one stash buffer. paged_update_cache double-buffers whole rows on one core, so a
# hidden-wide (5120) row needs ~1.75 MB of circular buffers, more than L1.
STASH_PIECE = 1024
# Depth of every drafter ring, the streaming floor. rings take the same L1 offset on every core, so the
# five rings stack: ~460 KB a core at bfloat8_b, below which fused_experts' bfloat8_b circular buffers
# (~840 KB) still fit.
DSPARK_RING_PAGES = 2
# Rows of the target's logits ring (one per decode step, at pos % LOGIT_ROWS): a whole tile, so the
# accept's one-hot gather has no padded K, and a divisor of ring_rows, so the slot follows from the ring's.
LOGIT_ROWS = 32
# The draft's Markov steps hold a vocabulary row as [MARKOV_GROUPS, V / MARKOV_GROUPS] (row-major in vocabulary
# order) rather than a [1, V] TILE row padded to 32 rows: each step's bias add, noise, untilize and argmax touch
# 1/32 of the tiles. The head writes the bias in that layout itself, from a block-diagonal copy of the embedding.
MARKOV_GROUPS = ttnn.TILE_SIZE
# DeepInfer's residual penalty: the residual keeps at least 1 - exp(-penalty) mass, so a rejection can draw.
DRAFT_PROB_PENALTY = 5e-5
# The greedy draft's block mode: the stages run on (BLOCK_ROWS, 32)-tiled row blocks (the block padded with zero rows)
# through the block ops (fused_hyperconnection, matmul_decode, width_sharded_all_reduce) instead of per-row loops.
BLOCK_ROWS = 8
DSPARK_BLOCK = os.environ.get("DEEPSEEK_V4_DSPARK_BLOCK_OPS", "1") == "1"
# The drafted block's rows (0: the checkpoint's dspark_block_size, 5). 4 by default: [token, noise x 3] for the 3 drafts a
# 4-row verify reads accepts more per round than the trained 5-row block (TOK=300: 2.25 vs 2.15 tokens/round on the
# default prompt, 3.25 vs 3.18 on gsm8k; 3 rows: 2.11), and fits the 4-row tiles and requests of the block ops.
DSPARK_BLOCK_SIZE = int(os.environ.get("DEEPSEEK_V4_DSPARK_BLOCK_SIZE", "4"))
# The greedy block draft's FFNs (router, routed experts and shared expert) on the target's persistent MoE server, as its
# on-demand requester (the model builds that; see DeepSeekV41Model._launch_moe_server). DEEPSEEK_V4_DSPARK_SERVER_MOE=0
# keeps them on the drafter's row (TP4 fused_experts and the shared expert's matmuls). TOK=300: draft 4.28 -> 2.80 ms,
# 143.5 -> 144.8 tok/s/u on the default prompt, 205.4 -> 215.5 on gsm8k (tokens/round 2.27 -> 2.08, 3.25 -> 3.08).
DSPARK_SERVER_MOE = os.environ.get("DEEPSEEK_V4_DSPARK_SERVER_MOE", "1") not in ("0", "", "false", "False")
# The greedy draft's Markov head: each step's bias matmul streams the rank's [G / TP * rank, V / G] slice of it. bfloat8_b
# keeps tokens/round (TOK=300: 2.27 / 3.25) and saves ~25 us a step; bfloat4_b drops the default prompt to 2.17.
DSPARK_MARKOV_DTYPE = {"bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b, "bf4": ttnn.bfloat4_b}[
    os.environ.get("DEEPSEEK_V4_DSPARK_MARKOV_DTYPE", "bf8")
]
# The TP4 path's router gate dtype (experiment: the persistent server's router reads it at bfloat8_b).
_GATE_DTYPE = {"bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b}[os.environ.get("DEEPSEEK_V4_DSPARK_GATE_DTYPE", "bf16")]
# Layer id of drafter stage 0 on that server (persistent_moe.DSPARK_LAYER_BASE).
DSPARK_SERVER_LAYER_BASE = 40
# The routed experts' dtype: bfloat4_b by default (half the bytes fused_experts streams; the same tokens per round as
# bfloat8_b at TOK=300 on both test prompts). Each dtype has
# its own cache files (the dtype is in their name).
DSPARK_EXPERT_DTYPE = {"bf8": ttnn.bfloat8_b, "bf4": ttnn.bfloat4_b}[
    os.environ.get("DEEPSEEK_V4_DSPARK_EXPERTS", "bf4")
]
# The uniforms' clamp, so both logs of a Gumbel draw stay finite (and 0 * Gumbel is 0 when greedy).
_UNIFORM_MIN, _UNIFORM_MAX = 2.0**-24, 1.0 - 2.0**-24


def dspark_prefetch_rings(tp_size: int) -> dict:
    """The drafter's rings by weight name, at the per-rank shapes of a 1x``tp_size`` stage.

    Each attention ring's page is pinned by a spec that is not a consumer (the same layout over fewer K
    rows): two whole slabs would not fit (q_a's ``[5120, 160]`` alone is ~870 KB at bfloat8_b). The
    pages are 80 tiles for q_a and 32 for kv, q_b and o_b at TP4. The shared down keeps whole-slab
    pages: its A is several ROW_MAJOR rows not height-sharded, which custom_mm takes in one K block only.
    """
    q_a, kv = ATTENTION_LAYOUTS["q_a_proj"], ATTENTION_LAYOUTS["kv_proj"]
    q_b = {**ATTENTION_LAYOUTS["q_b_proj"], "N": ATTENTION_LAYOUTS["q_b_proj"]["N"] // tp_size}
    o_b = {**ATTENTION_LAYOUTS["o_b_proj"], "K": ATTENTION_LAYOUTS["o_b_proj"]["K"] // tp_size}
    down = {"K": MOE_LAYOUTS["shared_down_proj"]["K"] // tp_size, "N": MOE_LAYOUTS["shared_down_proj"]["N"]}
    return {
        "q_a_proj": PrefetchRing("dspark.q_a", (q_a, {**q_a, "K": 512}), DSPARK_RING_PAGES),
        "kv_proj": PrefetchRing("dspark.kv", (kv, {**kv, "K": 1024}), DSPARK_RING_PAGES),
        "q_b_proj": PrefetchRing("dspark.q_b", (q_b, {**q_b, "K": 256}), DSPARK_RING_PAGES),
        "o_b_proj": PrefetchRing("dspark.o_b", (o_b, {**o_b, "K": 512}), DSPARK_RING_PAGES),
        "shared_down_proj": PrefetchRing("dspark.shared_down", (down,), DSPARK_RING_PAGES),
    }


def _rms(x: ttnn.Tensor, gamma: ttnn.Tensor, eps: float) -> ttnn.Tensor:
    return ttnn.multiply(wide_rms_norm(x, eps), gamma)


def _rows(x: ttnn.Tensor) -> ttnn.Tensor:
    """A ``matmul_decode`` result (``[1, 1, R, N]`` width-sharded over its B cores, ROW_MAJOR after the
    replica path) as the DRAM TILE rows the rest of a stage works on."""
    return ttnn.to_layout(ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG), ttnn.TILE_LAYOUT)


def _lookup_rows(idx: ttnn.Tensor, table: ttnn.Tensor) -> ttnn.Tensor:
    """Rows ``idx`` (uint32 ``[1, n]``) of the ROW_MAJOR ``table`` ``[L, W]``, as TILE ``[1, 1, n, W]``."""
    rows = ttnn.embedding(idx, table, layout=ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.to_layout(ttnn.reshape(rows, [1, 1, idx.shape[-1], table.shape[-1]]), ttnn.TILE_LAYOUT)


def _as_float(x: ttnn.Tensor, shape) -> ttnn.Tensor:
    """Integer ``x`` (ROW_MAJOR) as fp32 TILE of ``shape``; exact below 2**24."""
    return ttnn.typecast(ttnn.to_layout(ttnn.reshape(x, shape), ttnn.TILE_LAYOUT), ttnn.float32)


def _as_ids(x: ttnn.Tensor) -> ttnn.Tensor:
    """Whole-valued fp32 TILE ``x`` ``[1, 1, 1, n]`` as uint32 ROW_MAJOR ``[1, n]``."""
    return ttnn.reshape(ttnn.to_layout(ttnn.typecast(x, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [1, x.shape[-1]])


def _softmax_rows(x: ttnn.Tensor) -> ttnn.Tensor:
    """Softmax over the last dim of fp32 TILE ``x``."""
    e = ttnn.exp(ttnn.subtract(x, ttnn.max(x, dim=-1, keepdim=True)))
    return ttnn.multiply(e, ttnn.reciprocal(ttnn.sum(e, dim=-1, keepdim=True)))


def _exponentials(u: ttnn.Tensor) -> ttnn.Tensor:
    """Standard exponential draws ``-log(u)`` from uniforms ``u``, clamped so they are positive and finite."""
    return ttnn.neg(ttnn.log(ttnn.clamp(u, min=_UNIFORM_MIN, max=_UNIFORM_MAX)))


def host_to_device(value: torch.Tensor, dst: ttnn.Tensor) -> None:
    """Overwrite ``dst`` in place with ``value``, replicated over every chip of ``dst``'s mesh.

    Goes through command queue 0, so it lands between whatever is queued there before and after.
    """
    host = ttnn.from_torch(
        value, dtype=dst.dtype, layout=dst.layout, mesh_mapper=ttnn.ReplicateTensorToMesh(dst.device())
    )
    ttnn.copy_host_to_device_tensor(host, dst)


def _upload_packet(values: dict, fields: dict, dst: ttnn.Tensor) -> None:
    """Pack ``values`` (name -> ints) at their ``fields`` ``(offset, n)`` into the INT32 packet ``dst``."""
    flat = [0] * dst.shape[-1]
    for name, (offset, n) in fields.items():
        flat[offset : offset + n] = values[name]
    host_to_device(torch.tensor(flat, dtype=torch.int32).reshape(tuple(dst.shape)), dst)


def _packet_fields(pkt: ttnn.Tensor, fields: dict) -> dict:
    """The INT32 ROW_MAJOR ``[1, n]`` slices of packet ``pkt`` at its ``fields`` ``(offset, n)``."""
    return {
        name: ttnn.reshape(ttnn.slice(pkt, [0, 0, 0, offset], [1, 1, 1, offset + n]), [1, n])
        for name, (offset, n) in fields.items()
    }


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
    ``[Rd, Rd]`` rotate matrix. ``weight_dtype`` is for the attention projections and ``shared_dtype``
    for the shared expert; the router gate, norms, sinks and hyper-connections stay bf16.
    ``use_prefetcher`` streams the ringed projections (:func:`dspark_prefetch_rings`) through
    ``prefetch_buffers``, the mapping every stage on ``device`` shares, which builds each ring on first use.

    ``ring_rows`` (a multiple of a tile) is the depth of one session's ``main_kv`` ring. The attention's keys
    are ``kv_rows = ring_rows + TILE_SIZE``: the ring, then the block, the sink's zero row and padding to a tile.
    ``sessions`` slots each have their own ring, ``ring_rows`` rows of :attr:`ring` from ``slot * ring_rows``;
    with more than one, the attention copies the drafting slot's into :attr:`ring_view` first.
    """

    def __init__(
        self,
        config,
        stage: int,
        weights: dict,
        experts,
        device,
        rot,
        ring_rows: int,
        sessions: int = 1,
        cache=None,
        weight_dtype=ttnn.bfloat8_b,
        shared_dtype=ttnn.bfloat8_b,
        use_prefetcher: bool = False,
        prefetch_buffers=None,
        block_rows: int = 0,
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
        if ring_rows % ttnn.TILE_SIZE or self.block + 1 > ttnn.TILE_SIZE:
            raise ValueError(f"a {ring_rows}-row ring and a {self.block}-row block do not tile the keys")
        self.ring_rows = ring_rows
        self.kv_rows = ring_rows + ttnn.TILE_SIZE
        self.scale = self.head_dim**-0.5
        self.limit = config.swiglu_limit
        self.experts = experts
        # fused_experts' experts per block (0: the experts' own setting). A block is one gather / broadcast barrier;
        # the arithmetic is the same at any size, only L1 bounds it.
        self.experts_block = 0
        tag = f".tp{tp}"
        row = ttnn.ShardTensorToMesh(device, dim=0) if tp > 1 else None
        # A LinearDecode weight is [K, N] (or [1, 1, K, N] when prefetched): columns are the last dim.
        replicate = ttnn.ReplicateTensorToMesh(device) if tp > 1 else None
        col_k_n = ttnn.ShardTensorToMesh(device, dim=-1) if tp > 1 else None
        row_k_n = ttnn.ShardTensorToMesh(device, dim=-2) if tp > 1 else None
        rings = dspark_prefetch_rings(tp) if use_prefetcher else {}

        def gamma(name):
            return load_norm_gamma(weights[f"{name}.weight"], device, cache.file(name))

        self.wq_a = LinearDecode(
            weights["attn.wq_a.weight"],
            device,
            cache.file("attn.wq_a.decode"),
            dtype=weight_dtype,
            mesh_mapper=replicate,
            **ATTENTION_LAYOUTS["q_a_proj"],
            **ring_prefetch(rings, "q_a_proj", prefetch_buffers, device, weight_dtype),
            rectangle_b_grid=True,
            use_rm_hs=True,
            dram_sharded_weight=True,
        )
        self.q_norm = gamma("attn.q_norm")
        self.wq_b = LinearDecode(
            weights["attn.wq_b.weight"],
            device,
            cache.file("attn.wq_b" + tag + ".decode"),
            dtype=weight_dtype,
            mesh_mapper=col_k_n,
            **{**ATTENTION_LAYOUTS["q_b_proj"], "N": ATTENTION_LAYOUTS["q_b_proj"]["N"] // tp},
            **ring_prefetch(rings, "q_b_proj", prefetch_buffers, device, weight_dtype),
            rectangle_b_grid=True,
            use_rm_hs=True,
            dram_sharded_weight=True,
        )
        self.wkv = LinearDecode(
            weights["attn.wkv.weight"],
            device,
            cache.file("attn.wkv.decode"),
            dtype=weight_dtype,
            mesh_mapper=replicate,
            **ATTENTION_LAYOUTS["kv_proj"],
            **ring_prefetch(rings, "kv_proj", prefetch_buffers, device, weight_dtype),
            use_rm_hs=True,
            dram_sharded_weight=True,
        )
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
        self.wo_b = LinearDecode(
            weights["attn.wo_b.weight"],
            device,
            cache.file("attn.wo_b" + tag + ".decode"),
            dtype=weight_dtype,
            mesh_mapper=row_k_n,
            **{**ATTENTION_LAYOUTS["o_b_proj"], "K": ATTENTION_LAYOUTS["o_b_proj"]["K"] // tp},
            **ring_prefetch(rings, "o_b_proj", prefetch_buffers, device, weight_dtype),
            rectangle_b_grid=True,
            use_rm_hs=True,
            dram_sharded_weight=True,
        )
        # Row r = t * local_heads + h of the folded scores: the sink of head h at the sink's key column.
        sink = _memo(weights["attn.attn_sink"])
        sink_file = cache.file(
            f"attn.sink_bias.ring{ring_rows}" + tag + (f".blk{self.block}" if DSPARK_BLOCK_SIZE else "")
        )

        def sink_bias():
            bias = torch.zeros(tp, 1, self.block * self.local_heads, self.kv_rows)
            heads = sink().float().reshape(tp, 1, self.local_heads).repeat(1, self.block, 1)
            bias[:, 0, :, ring_rows + self.block] = heads.reshape(tp, -1)
            return bias

        self.sink_bias = _load_weight(
            _materialize(sink_bias, sink_file, ttnn.bfloat16),
            device,
            cache_file_name=sink_file,
            mesh_mapper=row,
        )
        self.sessions = sessions

        def ring(rows):
            return ttnn.from_torch(
                torch.zeros(1, 1, rows, self.head_dim),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.ring = ring(sessions * ring_rows)
        # One session's rows, which the draft's attention reads; with one session the ring itself.
        self.ring_view = ring(ring_rows) if sessions > 1 else self.ring

        self.attn_norm = gamma("attn_norm")
        self.ffn_norm = gamma("ffn_norm")
        self.attn_hc = DeepSeekV4HyperConnection(
            config,
            {"fn": weights["hc_attn_fn"], "base": weights["hc_attn_base"], "scale": weights["hc_attn_scale"]},
            device,
            cache=cache.sub("hc_attn"),
        )
        self.ffn_hc = DeepSeekV4HyperConnection(
            config,
            {"fn": weights["hc_ffn_fn"], "base": weights["hc_ffn_base"], "scale": weights["hc_ffn_scale"]},
            device,
            cache=cache.sub("hc_ffn"),
        )

        gate = _memo(weights["ffn.gate.weight"])
        bias = _memo(weights["ffn.gate.bias"])
        temp = getattr(config, "gate_temp", 1.0)
        self.gate = Linear(lambda: gate() / temp, device, cache.file("ffn.gate"), dtype=_GATE_DTYPE)
        # The op ranks on bf16 ``scores + bias``; shifting the bias by its max keeps it where bf16 is fine.
        bias_file = cache.file("ffn.gate.bias_shifted")
        self.gate_bias = _load_weight(
            _materialize(lambda: (bias() - bias().max()).reshape(1, 1, 1, -1), bias_file, ttnn.bfloat16),
            device,
            cache_file_name=bias_file,
        )
        # Gate/up's N / 64 B cores (9 at TP4) divide no ring; their ROW_MAJOR width-sharded outputs are
        # already the A that down cuts K over.
        hidden, local_inter = config.hidden_size, config.moe_intermediate_size // tp
        self.shared_w1 = LinearDecode(
            weights["ffn.shared_experts.w1.weight"],
            device,
            cache.file("ffn.shared_experts.w1" + tag + ".decode"),
            dtype=shared_dtype,
            mesh_mapper=col_k_n,
            **tp_gate_up_layout(tp, hidden, local_inter, layouts=MOE_LAYOUTS),
            rectangle_b_grid=True,
            use_rm_hs=True,
            dram_sharded_weight=True,
        )
        self.shared_w3 = LinearDecode(
            weights["ffn.shared_experts.w3.weight"],
            device,
            cache.file("ffn.shared_experts.w3" + tag + ".decode"),
            dtype=shared_dtype,
            mesh_mapper=col_k_n,
            **tp_gate_up_layout(tp, hidden, local_inter, layouts=MOE_LAYOUTS),
            rectangle_b_grid=True,
            use_rm_hs=True,
            dram_sharded_weight=True,
        )
        self.shared_w2 = LinearDecode(
            weights["ffn.shared_experts.w2.weight"],
            device,
            cache.file("ffn.shared_experts.w2" + tag + ".decode"),
            dtype=shared_dtype,
            mesh_mapper=row_k_n,
            K=local_inter,
            N=hidden,
            num_inputA_cores=max(1, local_inter // 64),
            **ring_prefetch(rings, "shared_down_proj", prefetch_buffers, device, shared_dtype),
            rectangle_b_grid=True,
            use_rm_hs=False,
            dram_sharded_weight=True,
        )
        self.block_rows = block_rows
        # The target's persistent MoE server, when it runs this stage's block FFN (DeepSeekV41DSpark's moe_server).
        self.moe_server = None
        self.moe_requester = 1
        self.server_layer = None
        if block_rows:
            self._init_block(config, weights, cache, tag, weight_dtype)

    # ------------------------------------------------------------------ block mode
    def _init_block(self, config, weights, cache, tag, weight_dtype) -> None:
        """What :meth:`forward_block` runs on besides the per-row path's layers: the block hyper-connections (fused
        ``fn`` dot), the norms' gains as ``(rows, 32)`` blocks, the grouped ``wo_a`` as a batched ``matmul_decode``,
        and the epilogues (fused norms, output multicasts) the block path swaps onto the shared projections."""
        rows, device, tp = self.block_rows, self.device, self.tp_size

        def hc(name, prefix):
            return DeepSeekV4HyperConnection(
                config,
                {
                    "fn": weights[f"hc_{prefix}_fn"],
                    "base": weights[f"hc_{prefix}_base"],
                    "scale": weights[f"hc_{prefix}_scale"],
                },
                device,
                cache=cache.sub(name),
                row_major_streams=True,
                fused_fn=True,
            )

        self.attn_hc_blk = hc("hc_attn", "attn")
        self.ffn_hc_blk = hc("hc_ffn", "ffn")
        self.attn_norm_blk = _gain_block(self.attn_norm, rows, device)
        self.ffn_norm_blk = _gain_block(self.ffn_norm, rows, device)
        self.streams_config = width_sharded_l1_config(
            config.hc_mult * rows, config.hidden_size, device, num_cores=8, tile_height=rows
        )

        def box(*layers):
            grid = layers[0].b_core_grid()
            for layer in layers[1:]:
                grid = grid.merge(layer.b_core_grid())
            return ttnn.CoreRangeSet({grid.bounding_box()})

        # The collapsed rows' replicas: attention's input on a rectangle around wq_a's and wkv's B cores, the
        # shared expert's around w1's and w3's.
        self.attn_in_grid = box(self.wq_a, self.wkv)
        self.ffn_in_grid = box(self.shared_w1, self.shared_w3)

        o_lora = self.o_lora = config.o_lora_rank
        in_per_group = config.num_attention_heads * self.head_dim // config.o_groups
        layout = check_decode_layout("o_a_proj", in_per_group, o_lora, batch=config.o_groups)
        b_blocks, n_blocks = layout["b_blocks"] // tp, layout["n_blocks"] * tp
        wo_a = _memo(weights["attn.wo_a.weight"])
        self.wo_a_blk = BatchedLinearDecode(
            wo_a,
            device,
            cache.file(f"attn.wo_a.batched{tag}.b{b_blocks}n{n_blocks}"),
            dtype=weight_dtype,
            batch=self.local_groups,
            global_batch=config.o_groups,
            K=in_per_group,
            N=o_lora,
            b_blocks=b_blocks,
            n_blocks=n_blocks,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=3) if tp > 1 else None,
            preprocess=lambda w: w.reshape(config.o_groups, o_lora, in_per_group).transpose(1, 2).contiguous(),
            dram_sharded_weight=True,
        )
        self.wo_a_blk.set_output_core_grid(box(self.wo_b))
        self.o_core = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        self.o_block_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(self.o_core, [rows, self.local_heads * self.head_dim], ttnn.ShardOrientation.ROW_MAJOR),
        )
        # Per-row norms of (rows, 32) blocks, which sharded_rmsnorm also un-tiles into ROW_MAJOR rows.
        self.q_norm_blk = _gain_block(self.q_norm, rows, device)
        self.kv_norm_blk = _gain_block(self.kv_norm, rows, device)

    def _rows_normed(self, x: ttnn.Tensor, gain: ttnn.Tensor, n: int) -> ttnn.Tensor:
        """The width-sharded ``(rows, 32)`` block ``x`` RMS-normed per row (``gain`` a block gain), as its first ``n``
        rows ROW_MAJOR ``[1, 1, n, W]`` in DRAM: sharded_rmsnorm un-tiles as it writes."""
        rows = ttnn.experimental.deepseek.sharded_rmsnorm(
            x, gain, epsilon=self.eps, memory_config=ttnn.DRAM_MEMORY_CONFIG, output_layout=ttnn.ROW_MAJOR_LAYOUT
        )
        if n < rows.shape[2]:
            rows = ttnn.slice(rows, [0, 0, 0, 0], [1, 1, n, rows.shape[-1]])
        return rows

    def to_block(self, x: ttnn.Tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor:
        """ROW_MAJOR rows ``[1, 1, n, W]`` -> the ``(rows, 32)`` block ``[1, 1, rows, W]``, zero rows past ``n``."""
        n, rows = x.shape[2], self.block_rows
        if n < rows:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, rows - n), (0, 0)], 0.0)
        return ttnn.tilize(x, tile=ttnn.Tile((rows, ttnn.TILE_SIZE)), memory_config=memory_config)

    def attention_block(self, replica, t, q_cos, q_sin, q_neg_sin, kv_cos, kv_sin, bias_row, ring_base=None):
        """:meth:`attention` from the normed block's ``(rows, 32)`` replica on :attr:`attn_in_grid`: the all-reduced
        ``(rows, 32)`` block ``[1, 1, rows, D]`` width-sharded over ``wo_b``'s cores (zero rows past ``t``).

        ``wq_a``, ``wkv`` and the grouped ``wo_a`` / ``wo_b`` take the block (one in0 tile per K step for every row);
        ``wq_b`` still takes the ``t`` normed q_a rows ROW_MAJOR, as the attention core reads ``q`` in 32x32 tiles."""
        if self.sessions > 1:
            ttnn.fill_cache(
                self.ring_view, self.ring, 0, input_row_offset_tensor=ring_base, num_input_rows=self.ring_rows
            )
        q_a = self._rows_normed(self.wq_a(replica), self.q_norm_blk, t)
        kv = self._rows_normed(self.wkv(replica), self.kv_norm_blk, t)
        q = _rows(self.wq_b(_decode_activation(self.wq_b, q_a)))
        q = ttnn.reshape(q, [1, 1, t * self.local_heads, self.head_dim])
        q = _rope_tail(q, q_cos, q_sin, self.rot)
        kv = _rope_tail(ttnn.to_layout(kv, ttnn.TILE_LAYOUT), kv_cos, kv_sin, self.rot)
        kv = ttnn.pad(kv, [(0, 0), (0, 0), (0, self.kv_rows - self.ring_rows - t), (0, 0)], 0.0)
        keys = ttnn.concat([self.ring_view, kv], dim=2)
        scores = ttnn.matmul(q, keys, transpose_b=True, dtype=ttnn.float32, compute_kernel_config=_HIFI4)
        scores = ttnn.add(ttnn.multiply(scores, self.scale), ttnn.add(self.sink_bias, bias_row))
        probs = ttnn.typecast(ttnn.softmax(scores, dim=-1, compute_kernel_config=_HIFI4), ttnn.bfloat16)
        o = ttnn.matmul(probs, keys, compute_kernel_config=_HIFI4)
        o = _rope_tail(o, q_cos, q_neg_sin, self.rot)
        # [t * H, Dh] row t * H + h is [t, H * Dh] row t: the (rows, 32) block o_a's gather reads on core (0, 0).
        o = ttnn.reshape(ttnn.to_layout(o, ttnn.ROW_MAJOR_LAYOUT), [1, 1, t, self.local_heads * self.head_dim])
        o = self.to_block(o, self.o_block_config)
        dest = ttnn.CoreRangeSet({self.wo_a_blk.b_core_grid().bounding_box()})
        gathered = ttnn.experimental.deepseek.all_gather_for_matmul(o, dest, num_groups=self.local_groups)
        ttnn.deallocate(o)
        y = self.wo_a_blk(gathered)
        ttnn.deallocate(gathered)
        y = ttnn.experimental.view(y, [1, 1, y.shape[-2], self.local_groups * self.o_lora])
        out = self.wo_b(y)
        ttnn.deallocate(y)
        if self.tp_size == 1:
            return out
        reduced = ttnn.experimental.deepseek.width_sharded_all_reduce(
            out, cluster_axis=_tp_cluster_axis(self.device), num_links=2, topology=CCL_TOPOLOGY
        )
        ttnn.deallocate(out)
        return reduced

    def _tap(self, name: str, x: ttnn.Tensor, n: int) -> None:
        """With :attr:`debug` a dict (eager runs only), keep the first ``n`` rows of ``x`` on host under ``name``."""
        if getattr(self, "debug", None) is not None:
            host = ttnn.to_torch(ttnn.get_device_tensors(x)[0]).float()
            self.debug[name] = host.reshape(-1, host.shape[-1])[:n]

    def forward_block(self, streams, pre_mix, mix, rope: dict, bias_row, t: int, ring_base=None, out_rows=None):
        """:meth:`forward` on ``(rows, 32)`` blocks: ``streams`` ``[1, 1, hc * rows, D]`` stream-major in
        :attr:`streams_config`, ``pre_mix`` ``[1, rows, 1, hc]``, ``t`` the block's real rows (the rest stay zero)
        and ``mix`` the previous stage's FFN ``(post, comb, out)``, which this stage's attention HC folds in
        (``None`` for the first). Returns ``(streams, ffn_pre, mix)``: this stage's FFN mix is left to the next HC
        (or the head's). The MoE runs ROW_MAJOR on the first ``out_rows`` (default ``t``) rows."""
        outs = self.attn_hc_blk(
            streams,
            pre_mix=pre_mix,
            mix=mix,
            norm=(self.attn_norm_blk, self.eps, True),
            replica_grid=self.attn_in_grid,
        )
        if mix is None:
            post, comb, collapsed, attn_pre, replica = outs
        else:
            post, comb, collapsed, attn_pre, streams, replica = outs
        ttnn.deallocate(collapsed)
        self._tap("attn_in", replica, t)
        attn_out = self.attention_block(replica, t, *rope["q"], *rope["kv"], bias_row, ring_base)
        self._tap("attn_out", attn_out, t)
        ttnn.deallocate(replica)
        if self.moe_server is not None:
            # The FFN HC folds the attention mix and the FFN norm and writes the request row (the block's (4 x 32)
            # tiles, one DRAM page); the server answers routed + shared, all-reduced, in that same layout, which the
            # next HC takes as its mix's sublayer output.
            post2, comb2, row, ffn_pre, streams = self.ffn_hc_blk(
                streams, pre_mix=attn_pre, mix=(post, comb, attn_out), norm=(self.ffn_norm_blk, self.eps)
            )
            for x in (post, comb, attn_out, attn_pre):
                ttnn.deallocate(x)
            self._tap("moe_in", row, 1)
            out = self.moe_server.request(self.moe_requester, self.server_layer, row)
            ttnn.deallocate(row)
            self._tap("moe_out", out, 1)
            return streams, ffn_pre, (post2, comb2, out)
        post2, comb2, collapsed, ffn_pre, streams = self.ffn_hc_blk(
            streams, pre_mix=attn_pre, mix=(post, comb, attn_out)
        )
        n = t if out_rows is None else min(out_rows, t)
        normed = self._rows_normed(collapsed, self.ffn_norm_blk, n)
        self._tap("moe_in", normed, n)
        out = self.moe(ttnn.to_layout(normed, ttnn.TILE_LAYOUT), x_rows=normed, rows_out=True)
        self._tap("moe_out", out, n)
        return streams, ffn_pre, (post2, comb2, self.to_block(out))

    def _all_reduce(self, x: ttnn.Tensor, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
        """Sum the TP partials of ``[1, 1, T, D]`` (any layout, DRAM or sharded) into DRAM rows in ``layout``. The
        rows go ROW_MAJOR first: the direct reduce-scatter stages its input over every core's L1, and a 32-row
        padded tile takes ~640 KB per core on the output stage."""
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        if self.tp_size > 1:
            x = _tp_all_reduce(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), self.device)
        return ttnn.to_layout(x, layout)

    def _kv(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """``[1, 1, R, D]`` -> the RoPE'd ``kv_norm(wkv(x))`` ``[1, 1, R, head_dim]``."""
        kv = _rows(self.wkv(_decode_activation(self.wkv, x)))
        return _rope_tail(_rms(kv, self.kv_norm, self.eps), cos, sin, self.rot)

    def write_main(self, main_x, cos, sin, slots) -> None:
        """Write row ``j`` of the ``main_kv`` of ``main_x`` ``[1, 1, R, D]`` into row ``slots[j]`` of
        :attr:`ring` (INT32 ``[R]``, the session's base included; ``-1`` skips the row)."""
        kv = _rows_on_one_core(self._kv(main_x, cos, sin))
        ttnn.experimental.paged_update_cache(self.ring, kv, update_idxs_tensor=slots)
        ttnn.deallocate(kv)

    def attention(self, x, q_cos, q_sin, q_neg_sin, kv_cos, kv_sin, bias_row, ring_base=None) -> ttnn.Tensor:
        """The block's attention: ``x`` ``[1, 1, T, D]`` -> ``[1, 1, T, D]`` (all-reduced).

        The ``T * local_heads`` query rows are folded onto one axis and attend the shared keys
        ``[ring | block kv | sink | padding]``; ``bias_row`` ``[1, 1, 1, kv_rows]`` masks the ring
        slots outside the window and the padding, and the sink row's score is its per-head bias.
        ``ring_base`` (INT32 ROW_MAJOR ``[1]``, with several sessions) is the drafting slot's first row of
        :attr:`ring`.
        """
        if self.sessions > 1:
            ttnn.fill_cache(
                self.ring_view, self.ring, 0, input_row_offset_tensor=ring_base, num_input_rows=self.ring_rows
            )
        t = x.shape[2]
        # wkv's B grid contains wq_a's, so its replica of x feeds both.
        x = _decode_activation(self.wkv, x)
        q_a = _rms(_rows(self.wq_a(_decode_activation(self.wq_a, x))), self.q_norm, self.eps)
        q = _rows(self.wq_b(_decode_activation(self.wq_b, q_a)))
        q = ttnn.reshape(q, [1, 1, t * self.local_heads, self.head_dim])
        q = _rope_tail(q, q_cos, q_sin, self.rot)
        kv = self._kv(x, kv_cos, kv_sin)
        kv = ttnn.pad(kv, [(0, 0), (0, 0), (0, self.kv_rows - self.ring_rows - t), (0, 0)], 0.0)
        keys = ttnn.concat([self.ring_view, kv], dim=2)
        scores = ttnn.matmul(q, keys, transpose_b=True, dtype=ttnn.float32, compute_kernel_config=_HIFI4)
        scores = ttnn.add(ttnn.multiply(scores, self.scale), ttnn.add(self.sink_bias, bias_row))
        probs = ttnn.typecast(ttnn.softmax(scores, dim=-1, compute_kernel_config=_HIFI4), ttnn.bfloat16)
        o = ttnn.matmul(probs, keys, compute_kernel_config=_HIFI4)
        o = _rope_tail(o, q_cos, q_neg_sin, self.rot)
        o = ttnn.reshape(o, [1, t, self.local_groups, -1])
        o = ttnn.matmul(ttnn.permute(o, (0, 2, 1, 3)), self.wo_a, compute_kernel_config=_HIFI4)
        o = ttnn.reshape(ttnn.permute(o, (0, 2, 1, 3)), [1, 1, t, -1])
        return self._all_reduce(self.wo_b(_decode_activation(self.wo_b, o)))

    def moe(self, x: ttnn.Tensor, x_rows=None, rows_out: bool = False) -> ttnn.Tensor:
        """Routed experts plus the shared expert: ``[1, 1, T, D]`` -> ``[1, 1, T, D]`` (all-reduced).

        ``fused_experts`` takes the block's ``T`` rows ROW_MAJOR in one call: it computes them in one
        ``bit_ceil(T)``-row tile row (TILE rows would be padded to 32, whose circular buffers do not fit
        beside the decode model's L1 buffers on the output stage) and fetches each expert of the rows'
        union once. ``num_experts`` is the union's bound ``T * top_k``, so the program, and the trace
        holding it, does not depend on how much the rows' selections overlap.
        """
        t, d = x.shape[2], x.shape[3]
        experts = self.experts
        logits = ttnn.linear(x, self.gate.weight, dtype=ttnn.float32, compute_kernel_config=_HIFI4)
        scores = ttnn.typecast(ttnn.sqrt(ttnn.softplus(logits)), ttnn.bfloat16)
        ranking = ttnn.add(scores, self.gate_bias)
        num_experts = min(t * experts.top_k, scores.shape[3])
        # x_rows: x ROW_MAJOR when the caller has it already; rows_out returns ROW_MAJOR rows.
        x_rows = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT) if x_rows is None else x_rows
        out = ttnn.experimental.deepseek.moe.fused_experts(
            x_rows,
            routing_scores=scores,
            ranking_scores=ranking,
            gate_up_weights=experts.gate_up_weights,
            down_weights=experts.down_weights,
            num_experts=num_experts,
            intermediate_size=experts.intermediate,
            swiglu_limit=experts.limit,
            top_k=experts.top_k,
            routed_scaling_factor=experts.routed_scaling_factor,
            routing_eps=experts.routing_eps,
            experts_block_size=min(self.experts_block or experts.experts_block_size, num_experts),
        )
        routed = ttnn.to_layout(ttnn.reshape(out, [1, 1, t, d]), ttnn.TILE_LAYOUT)
        x_rep = _decode_activation(self.shared_w1, x_rows)
        gate = ttnn.clamp(self.shared_w1(x_rep), max=self.limit)
        up = ttnn.clamp(self.shared_w3(_decode_activation(self.shared_w3, x_rep)), min=-self.limit, max=self.limit)
        shared = _rows(self.shared_w2(ttnn.multiply(ttnn.silu(gate), up)))
        return self._all_reduce(ttnn.add(routed, shared), ttnn.ROW_MAJOR_LAYOUT if rows_out else ttnn.TILE_LAYOUT)

    def forward(
        self, streams, pre_mix, rope: dict, bias_row, ring_base=None, out_rows=None
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """``(streams [1, T, hc, D], pre_mix [1, T, 1, hc])`` -> the same pair for the next stage.

        ``out_rows`` (the last stage of a greedy draft) keeps only the block's first ``out_rows`` rows after the
        attention: the later rows' FFN feeds nothing the kept rows read, so it is skipped (fewer routed experts)."""
        post, comb, collapsed, attn_pre = self.attn_hc(streams, pre_mix)
        normed = _rms(rows_from_tokens(collapsed), self.attn_norm, self.eps)
        self._tap("attn_in", normed, normed.shape[2])
        attn_out = self.attention(normed, *rope["q"], *rope["kv"], bias_row, ring_base)
        self._tap("attn_out", attn_out, attn_out.shape[2])
        streams = mix_streams(post, comb, attn_out, streams)
        _, t, hc, d = streams.shape
        if out_rows is not None and out_rows < t:
            streams = ttnn.slice(streams, [0, 0, 0, 0], [1, out_rows, hc, d])
            attn_pre = ttnn.slice(attn_pre, [0, 0, 0, 0], [1, out_rows, 1, attn_pre.shape[-1]])
        post, comb, collapsed, ffn_pre = self.ffn_hc(streams, attn_pre)
        normed = _rms(rows_from_tokens(collapsed), self.ffn_norm, self.eps)
        self._tap("moe_in", normed, normed.shape[2])
        moe_out = self.moe(normed)
        self._tap("moe_out", moe_out, moe_out.shape[2])
        streams = mix_streams(post, comb, moe_out, streams)
        return streams, ffn_pre


class DeepSeekV41DSpark(DeepSeekV4Module):
    """The three DSpark stages plus ``main_proj``, the head and the per-round inputs, on ``device``.

    ``lm_head`` is the decode model's tied head (a :class:`~..lm_head.DeepSeekV41LMHead`, or any callable
    returning the replicated ``[.., V]`` logits, on the same ``device``), ``rope_main`` the sliding
    ``(cos_half, sin_half)`` ``[max_seq, Rd / 2]`` tables.
    ``main_rows`` is how many verified rows one :meth:`write_main` takes, which sizes the rings
    (``ring_rows``, ``window + main_rows - 1`` rounded up to a tile), and ``stash_rows`` the depth of
    :attr:`stash`, the ring the decode model writes each step's target-layer means into at
    ``pos % stash_rows`` (per target, ``[1, 1, stash_rows, STASH_PIECE]`` column pieces).

    ``expert_dtype`` defaults to :data:`DSPARK_EXPERT_DTYPE` (bfloat4_b: half the bytes of bfloat8_b at the same
    tokens per round). ``fused_experts`` double-buffers a full-K weight slice per core (~700 KB at bfloat8_b),
    which fits below the drafter's own rings on its stage.

    ``weight_dtype`` (attention projections and ``main_proj``) and ``shared_dtype`` (shared expert) size
    the drafter's rings: it runs on a pipeline stage of its own, so with ``use_prefetcher`` it builds its
    five rings (:func:`dspark_prefetch_rings`) on ``device``. The caller owns the prefetcher session, as for
    the decode layers.

    The stash (and the one-hot that gathers from it) lives on ``stash_device``, the decode stage holding
    the target layers. When that is not ``device``, ``link`` is a D2D ``(sender, receiver)`` socket pair
    from it to ``device``: :meth:`send_main` gathers the staged rows there and sends them, and
    :meth:`write_main` receives them.

    ``sessions`` is the decode model's session slots. Each has its own rows of every ring (:meth:`ring_row`),
    so a session's rings hold its own positions whatever the others decode between its rounds, and a draft
    reads the slot :meth:`set_block` names. The rest is one draft's: the stash (every step's, whichever its
    session), the logits ring, the uniforms and :attr:`out` hold the last steps' and the last draft's, so a
    caller reads a draft before the next one and, with several sessions, drafts no other session between a
    :meth:`draft` and the :meth:`accept_draft` that accepts against it.
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
        expert_dtype=DSPARK_EXPERT_DTYPE,
        shared_dtype=ttnn.bfloat8_b,
        tp_size: int = 4,
        main_rows: int = 8,
        stash_rows: int = 256,
        use_prefetcher: bool = False,
        stash_device=None,
        link=None,
        sessions: int = 1,
        greedy_drafts=None,
        block_mode: bool = DSPARK_BLOCK,
        moe_server=None,
        moe_requester: int = 1,
    ):
        if sessions < 1:
            raise ValueError(f"sessions must be at least 1, got {sessions}")
        stash_device = device if stash_device is None else stash_device
        if (link is None) != (stash_device is device):
            raise ValueError("a stash on another device needs a link to the drafter's, and only then")
        cache = _as_cache(cache)
        if DSPARK_BLOCK_SIZE and DSPARK_BLOCK_SIZE != config.dspark_block_size:
            config = SimpleNamespace(**{**vars(config), "dspark_block_size": DSPARK_BLOCK_SIZE})
        # Every ring is private, so the mapping starts empty and the stages fill it as they are built.
        prefetch_buffers = {} if use_prefetcher else None
        self.config = config
        self.device = device
        self.stash_device = stash_device
        self.link = link
        self.lm_head = lm_head
        self.cos_half, self.sin_half = rope_main
        self.eps = config.rms_norm_eps
        self.hidden = config.hidden_size
        self.hc = config.hc_mult
        self.window = config.sliding_window
        self.block = config.dspark_block_size
        self.noise_id = config.dspark_noise_token_id
        # How many proposals draft(greedy=True) computes (the rest of out["ids"] repeats the last): a verify of
        # k + 1 rows reads k, and the Markov steps and the last stage's FFN past them feed nothing it reads.
        self.greedy_drafts = self.block if greedy_drafts is None else int(greedy_drafts)
        if not 0 < self.greedy_drafts <= self.block:
            raise ValueError(f"greedy_drafts must be in [1, {self.block}], got {greedy_drafts}")
        self.targets = list(config.dspark_target_layer_ids)
        self.main_rows = main_rows
        self.stash_rows = stash_rows
        self.ring_rows = -(-(self.window + main_rows - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        self.sessions = sessions
        # draft(greedy=True) runs the stages on (BLOCK_ROWS, 32) row blocks (forward_block).
        self.block_mode = bool(block_mode) and self.block <= BLOCK_ROWS
        # The block's tile height: the smallest of 4 and 8 rows that holds it.
        self.block_rows = 4 if self.block <= 4 else BLOCK_ROWS
        # What set_sampling last staged (the inputs start greedy).
        self.temperature = 0.0
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
                    config,
                    s,
                    w,
                    experts,
                    device,
                    self.rot,
                    self.ring_rows,
                    sessions=sessions,
                    cache=stage_cache,
                    weight_dtype=weight_dtype,
                    shared_dtype=shared_dtype,
                    use_prefetcher=use_prefetcher,
                    prefetch_buffers=prefetch_buffers,
                    block_rows=self.block_rows if self.block_mode else 0,
                )
            )
        # The block draft's FFNs on the target's persistent MoE server (see DSPARK_SERVER_MOE): 4-row requests.
        self.moe_server = moe_server
        if moe_server is not None:
            if not self.block_mode or self.block_rows != 4:
                raise ValueError("the persistent MoE serves the drafter 4-row blocks: a block of at most 4 rows")
            for s, stage in enumerate(self.stages):
                stage.moe_server = moe_server
                stage.moe_requester = moe_requester
                stage.server_layer = DSPARK_SERVER_LAYER_BASE + s
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
        # [V, rank] -> [groups * rank, V / groups]: row c * rank + k holds column k for vocabulary slice c, so
        # the block-diagonal embedding [groups, groups * rank] times it is the bias as [groups, V / groups].
        groups, rank, vocab = MARKOV_GROUPS, self.markov_embed.shape[-1], config.vocab_size
        if vocab % groups:
            raise ValueError(f"the Markov head splits V = {vocab} into {groups} rows")
        head_file = head_cache.file("markov_head.head.groups")
        head = _materialize(last["markov_head.head.weight"], head_file, ttnn.bfloat16)
        self.markov_head = _load_weight(
            None
            if head is None
            else head.reshape(groups, vocab // groups, rank).transpose(1, 2).reshape(-1, vocab // groups),
            device,
            cache_file_name=head_file,
        )
        self.markov_blocks = ttnn.from_torch(
            torch.kron(torch.eye(groups), torch.ones(1, rank)).reshape(1, 1, groups, groups * rank),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=replicate,
        )
        # The greedy draft (draft(greedy=True)) runs each Markov step on this rank's slice of the vocabulary: the
        # lm_head's V / TP columns (before its all-gather) are groups [r * G / TP, (r + 1) * G / TP) of the grouped
        # layout, so the rank's rows of markov_head and a G / TP block-diagonal embedding give its slice of the bias,
        # and only the biased [G / TP, V / G] rows are gathered for the argmax: a quarter of the bias matmul's weight
        # reads and no vocab-wide logits gather. The products and sums are the full matmul's (the other ranks' rows
        # only added zeros), so the rows, and the ids, are the same.
        self.head_tp = getattr(lm_head, "tp_size", 1) if hasattr(lm_head, "linear") else 1
        self.markov_head_local = self.markov_blocks_local = None
        if self.head_tp > 1:
            if groups % self.head_tp:
                raise ValueError(f"{groups} Markov groups do not split over TP{self.head_tp}")
            local_groups = groups // self.head_tp
            host = ttnn.to_torch(ttnn.get_device_tensors(self.markov_head)[0])
            self.markov_head_local = ttnn.from_torch(
                host,
                dtype=DSPARK_MARKOV_DTYPE,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(device, dim=-2),
            )
            self.markov_blocks_local = ttnn.from_torch(
                torch.kron(torch.eye(local_groups), torch.ones(1, rank)).reshape(
                    1, 1, local_groups, local_groups * rank
                ),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=replicate,
            )
        conf = _memo(last["confidence_head.proj.weight"])  # [1, D + rank]
        self.conf_hidden = Linear(
            lambda: conf()[:, : self.hidden], device, head_cache.file("confidence.hidden"), dtype=ttnn.float32
        )
        self.conf_markov = Linear(
            lambda: conf()[:, self.hidden :], device, head_cache.file("confidence.markov"), dtype=ttnn.float32
        )
        # The backbone's token table, for the block's input rows: replicated, as ttnn.embedding reads it whole.
        self.embed = DeepSeekV4Embedding(loader, device, cache=cache)
        cos, sin = make_rope_table(self.cos_half, self.sin_half)
        self.rope_tables = [
            ttnn.from_torch(
                x.reshape(x.shape[-2], x.shape[-1]),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                mesh_mapper=replicate,
            )
            for x in (cos, sin)
        ]

        def zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, on=device):
            host = torch.zeros(shape, dtype=torch.int32 if dtype in (ttnn.uint32, ttnn.int32) else torch.float32)
            return ttnn.from_torch(
                host,
                dtype=dtype,
                layout=layout,
                device=on,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(on) if tp_size > 1 else None,
            )

        w, r, t = main_rows, self.rope_dim, self.block
        # The block's INT32 packet (set_block): its ids, the key and query rows' positions, the ring slot of
        # the frontier, how many positions up to it the window holds and the session's first ring row.
        self._block_fields, offset = {}, 0
        fields = {"ids": t, "pos": t, "q_pos": t * self.local_heads, "frontier_slot": 1, "live": 1, "ring_base": 1}
        for name, n in fields.items():
            self._block_fields[name] = (offset, n)
            offset += n
        kv_rows = self.stages[0].kv_rows
        cols = torch.arange(kv_rows, dtype=torch.float32).reshape(1, 1, 1, kv_rows)
        self.bias_consts = {
            name: ttnn.from_torch(
                value, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=replicate
            )
            for name, value in {
                "cols": cols,
                "ring_neg": torch.where(cols < self.ring_rows, _MASK_NEG, 0.0),
                "block": torch.where(cols < self.ring_rows + t + 1, 0.0, _MASK_NEG),
            }.items()
        }
        self.stash = [
            [zeros((1, 1, stash_rows, STASH_PIECE), on=stash_device) for _ in range(self.hidden // STASH_PIECE)]
            for _ in self.targets
        ]
        # Lands where the stash's gathered rows are received, when they come over the link.
        self.main_in = zeros((1, 1, w, len(self.targets) * self.hidden)) if link is not None else None
        self.inputs = {
            "gather": zeros((1, 1, w, stash_rows), on=stash_device),
            "main_slots": zeros((w,), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            "main_cos": zeros((1, 1, w, r)),
            "main_sin": zeros((1, 1, w, r)),
            "block": zeros((1, 1, 1, -(-offset // ttnn.TILE_SIZE) * ttnn.TILE_SIZE), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
        }
        # The round's INT32 packet (set_round), for accept_draft: the next block's positions and frontier
        # fields as if no draft were accepted, the verified rows' logits-ring slots and which rows were drafts.
        self._round_fields, offset = {}, 0
        fields = {
            "pos": t,
            "q_pos": t * self.local_heads,
            "frontier_slot": 1,
            "live": 1,
            "ring_base": 1,
            "q_rows": t,
            "keep": t,
        }
        for name, n in fields.items():
            self._round_fields[name] = (offset, n)
            offset += n
        vocab = config.vocab_size
        if self.ring_rows % LOGIT_ROWS or t > LOGIT_ROWS:
            raise ValueError(f"the logits ring's {LOGIT_ROWS} rows must divide {self.ring_rows} and hold {t}")
        round_w = -(-offset // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        self.inputs.update(
            {
                "round": zeros((1, 1, 1, round_w), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                # The draft's Gumbel noise, in the Markov steps' layout.
                "draft_uniform": zeros((1, t, MARKOV_GROUPS, vocab // MARKOV_GROUPS), ttnn.float32),
                # Rows [0, T) race the residual draws, [T, 2T) column 0 the accepts.
                "uniform": zeros((1, 1, 2 * t, vocab), ttnn.float32),
                "temp": zeros((1, 1, 1, 1), ttnn.float32),
                "inv_temp": zeros((1, 1, 1, 1), ttnn.float32),
            }
        )
        # The target's logits of the last LOGIT_ROWS decode steps, at slot pos % LOGIT_ROWS (write_logits).
        self.logit_ring = zeros((1, 1, LOGIT_ROWS, vocab))
        # The last draft, which the next accept_draft accepts against: ids, confidence, the rows' logits, and
        # accept_draft's [accepted drafts, next token].
        self.out = {
            "ids": zeros((1, 1, t), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            "conf": zeros((1, 1, t, 1), ttnn.float32),
            "rows": zeros((1, 1, t, vocab), layout=ttnn.ROW_MAJOR_LAYOUT),
            "accept": zeros((1, 1, 1, 2), ttnn.float32),
        }
        arange = torch.arange(max(vocab, LOGIT_ROWS), dtype=torch.float32)
        first = torch.zeros(1, 1, 1, t)
        first[..., 0] = 1
        self.sample_consts = {
            name: ttnn.from_torch(
                value, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=replicate
            )
            for name, value in {
                "vocab_cols": arange[:vocab].expand(1, 1, t, vocab).contiguous(),
                "ring_cols": arange[:LOGIT_ROWS].expand(1, 1, t, LOGIT_ROWS).contiguous(),
                "ring_rows": arange[:LOGIT_ROWS].reshape(1, 1, LOGIT_ROWS, 1),
                "first": first,
                "noise_ids": torch.full((1, 1, 1, t), float(self.noise_id)),
                "pick_accepted": torch.tensor([1.0, 0.0]).reshape(1, 1, 1, 2),
                "pick_token": torch.tensor([0.0, 1.0]).reshape(1, 1, 1, 2),
            }.items()
        }
        one_hot = torch.zeros(1, t, 1, self.hc)
        one_hot[..., 0] = 1
        self.pre_mix_init = ttnn.from_torch(
            one_hot, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=replicate
        )
        if self.block_mode:
            one_hot = torch.zeros(1, self.block_rows, 1, self.hc)
            one_hot[..., 0] = 1
            self.pre_mix_block = ttnn.from_torch(
                one_hot, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=replicate
            )
            self.norm_blk = _gain_block(self.norm, self.block_rows, device)
        if use_prefetcher:
            # The DRISC senders read a weight straight from DRAM when its request arrives, off the command
            # queue; a session that is already open only fenced the uploads queued before it started.
            ttnn.synchronize_device(device)

    # ------------------------------------------------------------------ host side
    def _rope_rows(self, positions) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.as_tensor(positions, dtype=torch.long)
        return make_rope_table(self.cos_half[idx], self.sin_half[idx])

    def ring_row(self, slot: int, pos: int) -> int:
        """The row of every ring that holds session slot ``slot``'s position ``pos``."""
        if not 0 <= slot < self.sessions:
            raise ValueError(f"session slot {slot} is not one of the drafter's {self.sessions}")
        return slot * self.ring_rows + pos % self.ring_rows

    def set_main_rows(self, positions, slot: int = 0) -> None:
        """Stage :meth:`write_main` for session slot ``slot``'s verified ``positions`` (at most ``main_rows``,
        consecutive or not), rejected rows included: :meth:`set_block`'s mask hides every row past the
        frontier. The rows come from the stash, so no other session may have stepped since them."""
        w = self.main_rows
        positions = list(positions)
        n = len(positions)
        assert 0 < n <= w, f"{n} rows for a {w}-row write"
        slots = [self.ring_row(slot, p) for p in positions] + [-1] * (w - n)
        positions = positions + [positions[-1]] * (w - n)
        gather = torch.zeros(1, 1, w, self.stash_rows)
        gather[0, 0, torch.arange(w), torch.tensor(positions) % self.stash_rows] = 1
        cos, sin = self._rope_rows(positions)
        host_to_device(gather, self.inputs["gather"])
        host_to_device(torch.tensor(slots, dtype=torch.int32), self.inputs["main_slots"])
        host_to_device(cos, self.inputs["main_cos"])
        host_to_device(sin, self.inputs["main_sin"])

    def set_block(self, token: int, frontier: int, slot: int = 0) -> None:
        """Stage :meth:`draft` for ``token`` at position ``frontier + 1`` of session slot ``slot``, ``frontier``
        being the last verified position. The draft attends the slot's ring rows of the ``window`` positions up
        to it; rows a round wrote past it are masked.

        Only the block's integers go up, as one packet; :meth:`draft` builds its inputs from them."""
        t = self.block
        base = self.ring_row(slot, 0)
        if frontier + t >= self.cos_half.shape[0]:
            raise ValueError(f"positions up to {frontier + t} past the {self.cos_half.shape[0]}-row RoPE table")
        positions = list(range(frontier + 1, frontier + 1 + t))
        values = {
            "ids": [token] + [self.noise_id] * (t - 1),
            "pos": positions,
            "q_pos": [p for p in positions for _ in range(self.local_heads)],
            "frontier_slot": [frontier % self.ring_rows],
            "live": [min(self.window, frontier + 1)],
            "ring_base": [base],
        }
        _upload_packet(values, self._block_fields, self.inputs["block"])

    def set_round(self, start: int, drafts: int, slot: int = 0) -> None:
        """Stage :meth:`accept_draft` for the round of session slot ``slot`` that verified ``[next, the last
        draft's first drafts]`` at positions ``start .. start + drafts``. The decode steps of those positions must
        have handed the drafter their logits (:meth:`write_logits`), and the last :meth:`draft` /
        :meth:`accept_draft` must be the one that proposed those drafts.

        The packet holds the next block's fields for a frontier of ``start`` (no draft accepted); the trace
        adds the accepted count on device."""
        t = self.block
        base = self.ring_row(slot, 0)
        if not 0 < drafts < t:
            raise ValueError(f"a device accept takes 1 to {t - 1} drafts (one row is the bonus draw's), got {drafts}")
        if start + drafts + t >= self.cos_half.shape[0]:
            raise ValueError(f"positions up to {start + drafts + t} past the {self.cos_half.shape[0]}-row RoPE table")
        positions = list(range(start + 1, start + 1 + t))
        values = {
            "pos": positions,
            "q_pos": [p for p in positions for _ in range(self.local_heads)],
            "frontier_slot": [start % self.ring_rows],
            "live": [start + 1],
            "ring_base": [base],
            "q_rows": [(start + j) % LOGIT_ROWS for j in range(t)],
            "keep": [int(j < drafts) for j in range(t)],
        }
        _upload_packet(values, self._round_fields, self.inputs["round"])

    def set_sampling(self, temperature: float) -> None:
        """Draft (and accept) at ``temperature``; 0, the initial state, is greedy."""
        if temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        self.temperature = temperature
        inv = 1.0 / temperature if temperature > 0 else 0.0
        for name, value in (("temp", temperature), ("inv_temp", inv)):
            host_to_device(torch.full((1, 1, 1, 1), value), self.inputs[name])

    def refresh_uniforms(self, seed: int, accept: bool = True) -> None:
        """Queue a fresh draw of the uniforms the next draft and (with ``accept``) accept sample with, in place
        on device (eager ops: a trace would replay one seed). ``seed`` must be in ``[1, 2**31 - 1)``; every rank
        draws the same."""
        if not 0 < seed < 2**31 - 1:
            raise ValueError(f"seed {seed} is not in [1, 2**31 - 1): ttnn.uniform takes 0 as 'pick one per core'")
        ttnn.uniform(self.inputs["draft_uniform"], 0.0, 1.0, seed)
        if accept:
            # A Lehmer step of the seed: nonzero, and a stream apart from the draft's.
            ttnn.uniform(self.inputs["uniform"], 0.0, 1.0, seed * 48271 % (2**31 - 1))

    # ------------------------------------------------------------------ device side (traceable)
    def _block_inputs(
        self,
        ids: ttnn.Tensor,
        pos: ttnn.Tensor,
        q_pos: ttnn.Tensor,
        frontier_slot: ttnn.Tensor,
        live: ttnn.Tensor,
        embed: bool = True,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, dict, ttnn.Tensor]:
        """A block's inputs from its fields (``ids`` / ``pos`` / ``q_pos`` uint32 ``[1, n]``, ``frontier_slot``
        and ``live`` fp32 TILE ``[1, 1, 1, 1]``): ``(x_embed [1, 1, T, D], token [1, 1] uint32, rope, bias_row
        [1, 1, 1, kv_rows])``, ``rope`` as :meth:`DeepSeekV41DSparkStage.forward` takes it.

        The rows' embeddings and RoPE rows are looked up by id and position. ``bias_row`` opens ring column
        ``c`` when ``(frontier - c) mod ring_rows`` is below the live count, the block and sink columns always.
        """
        x_embed = _lookup_rows(ids, self.embed.embedding_weight) if embed else None
        token = ttnn.slice(ids, [0, 0], [1, 1])
        kv_cos, kv_sin = (_lookup_rows(pos, table) for table in self.rope_tables)
        q_cos, q_sin = (_lookup_rows(q_pos, table) for table in self.rope_tables)
        rope = {"q": (q_cos, q_sin, ttnn.neg(q_sin)), "kv": (kv_cos, kv_sin)}

        consts = self.bias_consts
        back = ttnn.subtract(frontier_slot, consts["cols"])
        back = ttnn.add(back, ttnn.multiply(ttnn.ltz(back), float(self.ring_rows)))
        dead = ttnn.ge(back, live)
        bias_row = ttnn.typecast(ttnn.add(consts["block"], ttnn.multiply(consts["ring_neg"], dead)), ttnn.bfloat16)
        return x_embed, token, rope, bias_row

    def _gather_main(self) -> ttnn.Tensor:
        """The staged rows' target-layer means ``[1, 1, main_rows, targets * D]``, on :attr:`stash_device`."""
        return ttnn.concat(
            [
                ttnn.matmul(self.inputs["gather"], piece, compute_kernel_config=_HIFI4)
                for pieces in self.stash
                for piece in pieces
            ],
            dim=-1,
        )

    def send_main(self) -> None:
        """On :attr:`stash_device`: gather the staged rows and send them over :attr:`link` to the
        :meth:`write_main` that follows on :attr:`device`."""
        rows = self._gather_main()
        ttnn.experimental.send_direct_async(rows, self.link[0])
        rows.deallocate(True)

    def write_main(self) -> None:
        """Take the staged rows' target-layer means (gathered from :attr:`stash` here, or received from
        :meth:`send_main`) and write their ``main_kv`` into every ring."""
        inp = self.inputs
        if self.link is None:
            rows = self._gather_main()
        else:
            ttnn.experimental.recv_direct_async(self.main_in, self.link[1])
            rows = self.main_in
        self._write_rings(rows, inp["main_cos"], inp["main_sin"], inp["main_slots"])

    def write_rows(self, rows: ttnn.Tensor, positions: ttnn.Tensor, slots: ttnn.Tensor) -> None:
        """Write the ``main_kv`` of ``R`` rows handed over device to device into every ring, with no host input:
        ``rows`` ``[1, 1, R, targets * D]`` ROW_MAJOR are their target-layer means (a decode or verify step's
        taps), ``positions`` uint32 ``[1, R]`` their positions and ``slots`` INT32 ``[R]`` their ring slots."""
        cos, sin = (_lookup_rows(positions, table) for table in self.rope_tables)
        self._write_rings(ttnn.to_layout(rows, ttnn.TILE_LAYOUT), cos, sin, slots)

    def write_logits(self, logits: ttnn.Tensor, ring_row: ttnn.Tensor) -> None:
        """Keep one decode step's target logits ``[1, 1, 1, V]`` bf16 ROW_MAJOR for :meth:`accept_draft`, at
        slot ``pos % LOGIT_ROWS`` of :attr:`logit_ring`; ``ring_row`` INT32 ``[1, 1, 1, 1]`` is the step's
        ``pos % ring_rows``. Every product is with 0 or 1, so the row lands exact."""
        slot = _as_float(ring_row, [1, 1, 1, 1])
        slot = ttnn.subtract(slot, ttnn.multiply(ttnn.floor(ttnn.multiply(slot, 1.0 / LOGIT_ROWS)), float(LOGIT_ROWS)))
        rows = self.sample_consts["ring_rows"]
        hit = ttnn.typecast(ttnn.eq(rows, slot), ttnn.bfloat16)
        keep = ttnn.typecast(ttnn.ne(rows, slot), ttnn.bfloat16)
        row = ttnn.repeat(ttnn.to_layout(logits, ttnn.TILE_LAYOUT), ttnn.Shape([1, 1, LOGIT_ROWS, 1]))
        ring = self.logit_ring
        ttnn.copy(ttnn.add(ttnn.multiply(ring, keep), ttnn.multiply(row, hit)), ring)

    def _write_rings(self, rows: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor, slots: ttnn.Tensor) -> None:
        main_x = _rms(self.main_proj(rows), self.main_norm, self.eps)
        for stage in self.stages:
            stage.write_main(main_x, cos, sin, slots)

    def draft(self, greedy: bool = False) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Draft the block :meth:`set_block` staged: :attr:`out`'s ``(ids [1, 1, block] uint32, confidence [1, 1,
        block, 1] fp32, logits [1, 1, block, V] bf16 ROW_MAJOR)``.

        Row ``i`` sits at position ``frontier + 1 + i`` and ``ids[i]``, the argmax of its Markov-biased
        ``logits[i]`` plus ``temperature`` times Gumbel noise (from the staged uniforms), is the proposal for
        position ``frontier + 2 + i``: a sample from ``softmax(logits[i] / temperature)``, the argmax when greedy.
        ``confidence[i]`` scores it.

        ``greedy`` drafts at temperature 0 whatever :meth:`set_sampling` staged, with the same ids, and leaves
        ``logits`` as they were: it skips the noise and the logits rows, and with a TP ``lm_head`` runs the Markov
        steps on each rank's slice of the vocabulary (see :attr:`head_tp`).
        """
        f = _packet_fields(self.inputs["block"], self._block_fields)
        ids, pos, q_pos = (ttnn.typecast(f[name], ttnn.uint32) for name in ("ids", "pos", "q_pos"))
        scalar = [1, 1, 1, 1]
        return self._draft(
            ids,
            pos,
            q_pos,
            _as_float(f["frontier_slot"], scalar),
            _as_float(f["live"], scalar),
            f["ring_base"],
            greedy=greedy,
        )

    def accept_draft(self, sampled: bool) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Accept the round :meth:`set_round` staged against the last draft, then draft from the result, all on
        device. :attr:`out` ``["accept"]`` gets ``[m, token]``, ``m`` the leading drafts accepted and ``token``
        the next input, at position ``start + m + 1``; the draft from it lands in :attr:`out` as :meth:`draft`'s.

        ``sampled`` is DeepInfer's speculative sampling at the staged temperature without top-p / top-k (``q_j``
        the target's ``softmax(logits_j / temperature)``, ``p_j`` the drafter's): draft ``d_j`` is accepted when
        ``u_j p_j(d_j) < q_j(d_j)``, and row ``m`` draws ``token`` from ``max(q_m - exp(-DRAFT_PROB_PENALTY)
        p_m, 0)`` (``q_m`` when every draft was accepted) by an exponential race. Greedy accepts while ``d_j``
        is the target's argmax and takes the argmax of row ``m``.
        """
        inp, consts, t = self.inputs, self.sample_consts, self.block
        vocab = self.config.vocab_size
        f = _packet_fields(inp["round"], self._round_fields)
        column = [1, 1, t, 1]
        keep = _as_float(f["keep"], column)
        gather = ttnn.typecast(ttnn.eq(consts["ring_cols"], _as_float(f["q_rows"], column)), ttnn.bfloat16)
        target = ttnn.matmul(gather, self.logit_ring, compute_kernel_config=_HIFI4)
        drafted = _as_float(self.out["ids"], column)
        if sampled:
            inv = inp["inv_temp"]
            q = _softmax_rows(ttnn.multiply(ttnn.typecast(target, ttnn.float32), inv))
            p = ttnn.typecast(ttnn.to_layout(self.out["rows"], ttnn.TILE_LAYOUT), ttnn.float32)
            p = ttnn.multiply(_softmax_rows(ttnn.multiply(p, inv)), keep)
            one_hot = ttnn.eq(consts["vocab_cols"], drafted)
            q_d = ttnn.sum(ttnn.multiply(q, one_hot), dim=-1, keepdim=True)
            p_d = ttnn.sum(ttnn.multiply(p, one_hot), dim=-1, keepdim=True)
            u = ttnn.slice(inp["uniform"], [0, 0, t, 0], [1, 1, 2 * t, 1])
            hits = ttnn.multiply(ttnn.lt(ttnn.multiply(u, p_d), q_d), ttnn.gtz(p_d))
            residual = ttnn.relu(ttnn.subtract(q, ttnn.multiply(p, math.exp(-DRAFT_PROB_PENALTY))))
            race = _exponentials(ttnn.slice(inp["uniform"], [0, 0, 0, 0], [1, 1, t, vocab]))
            score = ttnn.multiply(residual, ttnn.reciprocal(race))
        else:
            score = target
        best = _as_float(ttnn.argmax(ttnn.to_layout(score, ttnn.ROW_MAJOR_LAYOUT), dim=-1), column)
        if not sampled:
            hits = ttnn.multiply(ttnn.eq(drafted, best), keep)

        run = accepted = None
        for j in range(t):
            hit = ttnn.slice(hits, [0, 0, j, 0], [1, 1, j + 1, 1])
            run = hit if run is None else ttnn.multiply(run, hit)
            accepted = run if accepted is None else ttnn.add(accepted, run)
        token = None
        for j in range(t):
            pick = ttnn.multiply(ttnn.eq(accepted, float(j)), ttnn.slice(best, [0, 0, j, 0], [1, 1, j + 1, 1]))
            token = pick if token is None else ttnn.add(token, pick)
        result = ttnn.add(ttnn.multiply(consts["pick_accepted"], accepted), ttnn.multiply(consts["pick_token"], token))
        ttnn.copy(result, self.out["accept"])

        ids = _as_ids(
            ttnn.add(consts["noise_ids"], ttnn.multiply(consts["first"], ttnn.subtract(token, float(self.noise_id))))
        )
        pos, q_pos = (
            _as_ids(ttnn.add(_as_float(f[name], [1, 1, 1, f[name].shape[-1]]), accepted)) for name in ("pos", "q_pos")
        )
        rows = float(self.ring_rows)
        frontier_slot = ttnn.add(_as_float(f["frontier_slot"], [1, 1, 1, 1]), accepted)
        frontier_slot = ttnn.subtract(frontier_slot, ttnn.multiply(ttnn.ge(frontier_slot, rows), rows))
        live = ttnn.clamp(ttnn.add(_as_float(f["live"], [1, 1, 1, 1]), accepted), min=1.0, max=float(self.window))
        return self._draft(ids, pos, q_pos, frontier_slot, live, f["ring_base"])

    def _draft(
        self,
        ids: ttnn.Tensor,
        pos: ttnn.Tensor,
        q_pos: ttnn.Tensor,
        frontier_slot: ttnn.Tensor,
        live: ttnn.Tensor,
        ring_base: ttnn.Tensor,
        greedy: bool = False,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        inp, t = self.inputs, self.block
        if greedy and self.block_mode:
            return self._draft_block(ids, pos, q_pos, frontier_slot, live, ring_base)
        x_embed, token, rope, bias_row = self._block_inputs(ids, pos, q_pos, frontier_slot, live)
        streams = ttnn.to_layout(
            ttnn.repeat(tokens_from_rows(x_embed), ttnn.Shape([1, 1, self.hc, 1])), ttnn.TILE_LAYOUT
        )
        pre_mix = self.pre_mix_init
        # INT32 ROW_MAJOR [1]: fill_cache reads it on device, so the trace drafts whichever slot the packet names.
        ring_base = ttnn.reshape(ring_base, [1]) if self.sessions > 1 else None
        last = len(self.stages) - 1
        for s, stage in enumerate(self.stages):
            out_rows = self.greedy_drafts if greedy and s == last else None
            streams, pre_mix = stage.forward(streams, pre_mix, rope, bias_row, ring_base, out_rows=out_rows)

        # pre_mix [1, T, 1, hc] @ streams [1, T, hc, D] per token, over hc padded to a tile with real zeros:
        # the matmul reads the tiles' padding, which other ops leave unspecified.
        pad = ttnn.TILE_SIZE - self.hc
        collapsed = ttnn.matmul(
            ttnn.pad(pre_mix, [(0, 0), (0, 0), (0, 0), (0, pad)], 0.0),
            ttnn.pad(streams, [(0, 0), (0, 0), (0, pad), (0, 0)], 0.0),
            dtype=ttnn.bfloat16,
            compute_kernel_config=_HIFI4,
        )
        x = rows_from_tokens(collapsed)
        normed = _rms(x, self.norm, self.eps)
        if greedy:
            self._debug_head(normed)
            ids, embeds = self._markov_greedy(normed, token)
            return self._finish_draft(x, ids, embeds, None)
        logits = self.lm_head(normed)

        rank, vocab = self.markov_embed.shape[-1], logits.shape[-1]
        groups, wide = MARKOV_GROUPS, vocab // MARKOV_GROUPS
        grouped = ttnn.to_layout(
            ttnn.reshape(ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT), [1, t, groups, wide]), ttnn.TILE_LAYOUT
        )
        noise = ttnn.multiply(ttnn.neg(ttnn.log(_exponentials(inp["draft_uniform"]))), inp["temp"])
        ids, embeds, rows = [], [], []
        for i in range(t):
            embed = ttnn.reshape(ttnn.embedding(token, self.markov_embed), [1, 1, 1, rank])
            embeds.append(embed)
            blocks = ttnn.to_layout(ttnn.repeat(embed, ttnn.Shape([1, 1, 1, groups])), ttnn.TILE_LAYOUT)
            bias = ttnn.matmul(
                ttnn.multiply(self.markov_blocks, blocks), self.markov_head, compute_kernel_config=_HIFI4
            )
            row = ttnn.add(ttnn.slice(grouped, [0, i, 0, 0], [1, i + 1, groups, wide]), bias)
            rows.append(row)
            noisy = ttnn.add(
                ttnn.typecast(row, ttnn.float32), ttnn.slice(noise, [0, i, 0, 0], [1, i + 1, groups, wide])
            )
            # Over the whole [groups, wide] row: its flat index is the vocabulary id.
            best = ttnn.argmax(ttnn.to_layout(noisy, ttnn.ROW_MAJOR_LAYOUT))
            ids.append(ttnn.reshape(best, [1, 1, 1]))
            token = ttnn.reshape(best, [1, 1])
        rows = ttnn.reshape(ttnn.to_layout(ttnn.concat(rows, dim=1), ttnn.ROW_MAJOR_LAYOUT), [1, 1, t, vocab])
        return self._finish_draft(x, ids, embeds, rows)

    def _finish_draft(self, x, ids: list, embeds: list, rows) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """The confidence head over the block's rows ``x`` and Markov inputs ``embeds``, then :attr:`out` (``rows``
        None keeps the last logits). Fewer ids than the block (a greedy draft of :attr:`greedy_drafts`) pad
        :attr:`out` ``["ids"]`` with the last and leave the confidences as they were."""
        conf = None
        if len(ids) == self.block and x is not None:
            markov = ttnn.to_layout(ttnn.concat(embeds, dim=2), ttnn.TILE_LAYOUT)
            conf = ttnn.add(
                ttnn.linear(x, self.conf_hidden.weight, dtype=ttnn.float32, compute_kernel_config=_HIFI4),
                ttnn.linear(markov, self.conf_markov.weight, dtype=ttnn.float32, compute_kernel_config=_HIFI4),
            )
        else:
            ids = ids + [ids[-1]] * (self.block - len(ids))
        out = self.out
        for name, value in (("ids", ttnn.concat(ids, dim=-1)), ("conf", conf), ("rows", rows)):
            if value is not None:
                ttnn.copy(ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG), out[name])
        return out["ids"], out["conf"], out["rows"]

    def _debug_head(self, normed: ttnn.Tensor) -> None:
        """With :attr:`debug_rows` a list (eager runs only), append the head's normed input rows to it, on host."""
        if getattr(self, "debug_rows", None) is not None:
            self.debug_rows.append(ttnn.to_torch(ttnn.get_device_tensors(normed)[0]).float().reshape(-1, self.hidden))

    def _draft_block(self, ids, pos, q_pos, frontier_slot, live, ring_base):
        """The greedy draft on ``(BLOCK_ROWS, 32)`` row blocks (:meth:`DeepSeekV41DSparkStage.forward_block`): the
        block's ``T`` rows padded with zero rows, which stay zero. Each stage's FFN mix is folded into the next
        stage's attention HC, and the last one's into the head's collapse (one more block HC, its RMSNorm the head's
        norm, which un-tiles the rows)."""
        t, rows, hc, d = self.block, self.block_rows, self.hc, self.hidden
        _, token, rope, bias_row = self._block_inputs(ids, pos, q_pos, frontier_slot, live, embed=False)
        first, last = self.stages[0], self.stages[-1]
        embedded = ttnn.reshape(
            ttnn.embedding(ids, self.embed.embedding_weight, layout=ttnn.ROW_MAJOR_LAYOUT), [1, 1, t, d]
        )
        if rows > t:
            embedded = ttnn.pad(embedded, [(0, 0), (0, 0), (0, rows - t), (0, 0)], 0.0)
        stacked = ttnn.repeat(embedded, ttnn.Shape([1, 1, hc, 1]))
        streams = ttnn.tilize(stacked, tile=ttnn.Tile((rows, ttnn.TILE_SIZE)), memory_config=first.streams_config)
        ttnn.deallocate(stacked)
        ring_base = ttnn.reshape(ring_base, [1]) if self.sessions > 1 else None
        pre_mix, mix = self.pre_mix_block, None
        k = self.greedy_drafts
        for stage in self.stages:
            out_rows = k if stage is last else None
            streams, pre_mix, mix = stage.forward_block(streams, pre_mix, mix, rope, bias_row, t, ring_base, out_rows)
        collapsed = last.ffn_hc_blk(streams, pre_mix=pre_mix, mix=mix)[2]
        normed = ttnn.to_layout(last._rows_normed(collapsed, self.norm_blk, k), ttnn.TILE_LAYOUT)
        self._debug_head(normed)
        ids, embeds = self._markov_greedy(normed, token)
        return self._finish_draft(None, ids, embeds, None)

    def _markov_greedy(self, normed: ttnn.Tensor, token: ttnn.Tensor) -> tuple[list, list]:
        """The greedy Markov steps from the head's input ``normed`` ``[1, 1, T, D]`` and the block's token: ``(ids,
        embeds)``, as :meth:`_draft`'s loop at temperature 0 (the noise is ``0 *`` finite there, an exact no-op).

        With a TP ``lm_head`` each rank keeps its ``V / TP`` logits columns and biases them with its rows of the
        Markov head; the ranks' biased rows, gathered in rank order, are the full grouped row."""
        t, rank = normed.shape[2], self.markov_embed.shape[-1]
        tp = self.head_tp
        if tp > 1:
            logits = self.lm_head.linear(normed)
            head, blocks_const = self.markov_head_local, self.markov_blocks_local
        else:
            logits = self.lm_head(normed)
            head, blocks_const = self.markov_head, self.markov_blocks
        groups = MARKOV_GROUPS // tp
        wide = logits.shape[-1] // groups
        grouped = ttnn.to_layout(
            ttnn.reshape(ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT), [1, t, groups, wide]), ttnn.TILE_LAYOUT
        )
        ids, embeds = [], []
        for i in range(self.greedy_drafts):
            embed = ttnn.reshape(ttnn.embedding(token, self.markov_embed), [1, 1, 1, rank])
            embeds.append(embed)
            blocks = ttnn.to_layout(ttnn.repeat(embed, ttnn.Shape([1, 1, 1, groups])), ttnn.TILE_LAYOUT)
            bias = ttnn.matmul(ttnn.multiply(blocks_const, blocks), head, compute_kernel_config=_HIFI4)
            row = ttnn.add(ttnn.slice(grouped, [0, i, 0, 0], [1, i + 1, groups, wide]), bias)
            if tp > 1:
                row = ttnn.all_gather(
                    row,
                    dim=2,
                    cluster_axis=_tp_cluster_axis(self.device),
                    num_links=1,
                    topology=CCL_TOPOLOGY,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            # Over the whole [MARKOV_GROUPS, V / MARKOV_GROUPS] row: its flat index is the vocabulary id. ROW_MAJOR, the
            # row is already in vocabulary order, and argmax over one [1, V] row runs ~3x faster than over 32 rows.
            flat = ttnn.reshape(ttnn.to_layout(row, ttnn.ROW_MAJOR_LAYOUT), [1, 1, 1, wide * MARKOV_GROUPS])
            best = ttnn.argmax(flat)
            ids.append(ttnn.reshape(best, [1, 1, 1]))
            token = ttnn.reshape(best, [1, 1])
        return ids, embeds
