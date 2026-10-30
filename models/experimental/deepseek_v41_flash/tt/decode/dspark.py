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

Each mHC takes the block's ``T`` rows in one multi-token ``fused_hyperconnection`` (one token per core,
the token's ``[hc, D]`` slab staged whole): the drafter's stage has no decode model's L1 buffers beside it.

A round writes every row it verified, the rejected ones too: the ring is ``window + main_rows - 1`` rows
or more, so a rejected row (at most ``main_rows - 1`` past the frontier) never shares a slot with one of
the ``window`` live positions, and the draft's ``bias_row`` masks every slot outside the live window. The
next round verifies from the slot after the frontier and so rewrites every rejected row.
"""

import math
from types import SimpleNamespace

import torch
import ttnn

from models.experimental.deepseek_v41_flash.tt.config import dspark_expert_provider, dspark_weights
from models.experimental.deepseek_v41_flash.tt.decode.attention import DECODE_LAYOUTS as ATTENTION_LAYOUTS
from models.experimental.deepseek_v41_flash.tt.decode.attention import _rope_tail, _rows_on_one_core
from models.experimental.deepseek_v41_flash.tt.decode.moe import DECODE_LAYOUTS as MOE_LAYOUTS
from models.experimental.deepseek_v41_flash.tt.decode.moe import DeepSeekV41PreloadedExperts
from models.experimental.deepseek_v4_flash.tt.common import _HIFI4, _MASK_NEG, DeepSeekV4Module
from models.experimental.deepseek_v4_flash.tt.decode.attention import (
    _decode_activation,
    _interleaved_rotate_matrix,
    make_rope_table,
)
from models.experimental.deepseek_v4_flash.tt.decode.decode_prefetch import (
    PrefetchRing,
    ring_prefetch,
    tp_gate_up_layout,
)
from models.experimental.deepseek_v4_flash.tt.decode.moe import _tp_all_reduce
from models.experimental.deepseek_v4_flash.tt.embedding import DeepSeekV4Embedding
from models.experimental.deepseek_v4_flash.tt.layers import Linear, LinearDecode
from models.experimental.deepseek_v4_flash.tt.prefill.decoder_layer import (
    load_norm_gamma,
    mix_streams,
    rows_from_tokens,
    tokens_from_rows,
)
from models.experimental.deepseek_v4_flash.tt.prefill.hyperconnection import (
    DeepSeekV4PrefillHyperConnection,
    wide_rms_norm,
)
from models.experimental.deepseek_v4_flash.tt.weight_cache import _as_cache, _load_weight, _materialize, _memo

# Column width of one stash buffer. paged_update_cache double-buffers whole rows on one core, so a
# hidden-wide (5120) row needs ~1.75 MB of circular buffers, more than L1.
STASH_PIECE = 1024
# Depth of every drafter ring, the streaming floor. GCBs take the same L1 offset on every core, so the
# five rings stack: ~460 KB a core at bfloat8_b, below which fused_experts' bfloat8_b circular buffers
# (~840 KB) still fit.
DSPARK_GCB_PAGES = 2
# Rows of the target's logits ring (one per decode step, at pos % LOGIT_ROWS): a whole tile, so the
# accept's one-hot gather has no padded K, and a divisor of ring_rows, so the slot follows from the ring's.
LOGIT_ROWS = 32
# DeepInfer's residual penalty: the residual keeps at least 1 - exp(-penalty) mass, so a rejection can draw.
DRAFT_PROB_PENALTY = 5e-5
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
        "q_a_proj": PrefetchRing("dspark.q_a", (q_a, {**q_a, "K": 512}), DSPARK_GCB_PAGES),
        "kv_proj": PrefetchRing("dspark.kv", (kv, {**kv, "K": 1024}), DSPARK_GCB_PAGES),
        "q_b_proj": PrefetchRing("dspark.q_b", (q_b, {**q_b, "K": 256}), DSPARK_GCB_PAGES),
        "o_b_proj": PrefetchRing("dspark.o_b", (o_b, {**o_b, "K": 512}), DSPARK_GCB_PAGES),
        "shared_down_proj": PrefetchRing("dspark.shared_down", (down,), DSPARK_GCB_PAGES),
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
    packet = torch.zeros(tuple(dst.shape), dtype=torch.int32)
    for name, (offset, n) in fields.items():
        packet[..., offset : offset + n] = torch.tensor(values[name], dtype=torch.int32)
    host_to_device(packet, dst)


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

    ``ring_rows`` (a multiple of a tile) is the depth of the ``main_kv`` ring. The attention's keys are
    ``kv_rows = ring_rows + TILE_SIZE``: the ring, then the block, the sink's zero row and padding to a tile.
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
        cache=None,
        weight_dtype=ttnn.bfloat8_b,
        shared_dtype=ttnn.bfloat8_b,
        use_prefetcher: bool = False,
        prefetch_buffers=None,
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
        sink_file = cache.file(f"attn.sink_bias.ring{ring_rows}" + tag)

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
        self.ring = ttnn.from_torch(
            torch.zeros(1, 1, ring_rows, self.head_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.attn_norm = gamma("attn_norm")
        self.ffn_norm = gamma("ffn_norm")
        self.attn_hc = DeepSeekV4PrefillHyperConnection(
            config,
            {"fn": weights["hc_attn_fn"], "base": weights["hc_attn_base"], "scale": weights["hc_attn_scale"]},
            device,
            cache=cache.sub("hc_attn"),
        )
        self.ffn_hc = DeepSeekV4PrefillHyperConnection(
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

    def _all_reduce(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Sum the TP partials of ``[1, 1, T, D]`` (any layout, DRAM or sharded) into DRAM TILE rows. The rows
        go ROW_MAJOR first: the direct reduce-scatter stages its input over every core's L1, and a 32-row
        padded tile takes ~640 KB per core on the output stage."""
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        if self.tp_size == 1:
            return ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        summed = _tp_all_reduce(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), self.device)
        return ttnn.to_layout(summed, ttnn.TILE_LAYOUT)

    def _kv(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """``[1, 1, R, D]`` -> the RoPE'd ``kv_norm(wkv(x))`` ``[1, 1, R, head_dim]``."""
        kv = _rows(self.wkv(_decode_activation(self.wkv, x)))
        return _rope_tail(_rms(kv, self.kv_norm, self.eps), cos, sin, self.rot)

    def write_main(self, main_x, cos, sin, slots) -> None:
        """Write row ``j`` of the ``main_kv`` of ``main_x`` ``[1, 1, R, D]`` into ring slot ``slots[j]``
        (INT32 ``[R]``, ``-1`` skips the row)."""
        kv = _rows_on_one_core(self._kv(main_x, cos, sin))
        ttnn.experimental.paged_update_cache(self.ring, kv, update_idxs_tensor=slots)
        ttnn.deallocate(kv)

    def attention(self, x, q_cos, q_sin, q_neg_sin, kv_cos, kv_sin, bias_row) -> ttnn.Tensor:
        """The block's attention: ``x`` ``[1, 1, T, D]`` -> ``[1, 1, T, D]`` (all-reduced).

        The ``T * local_heads`` query rows are folded onto one axis and attend the shared keys
        ``[ring | block kv | sink | padding]``; ``bias_row`` ``[1, 1, 1, kv_rows]`` masks the ring
        slots outside the window and the padding, and the sink row's score is its per-head bias.
        """
        t = x.shape[2]
        q_a = _rms(_rows(self.wq_a(_decode_activation(self.wq_a, x))), self.q_norm, self.eps)
        q = _rows(self.wq_b(_decode_activation(self.wq_b, q_a)))
        q = ttnn.reshape(q, [1, 1, t * self.local_heads, self.head_dim])
        q = _rope_tail(q, q_cos, q_sin, self.rot)
        kv = self._kv(x, kv_cos, kv_sin)
        kv = ttnn.pad(kv, [(0, 0), (0, 0), (0, self.kv_rows - self.ring_rows - t), (0, 0)], 0.0)
        keys = ttnn.concat([self.ring, kv], dim=2)
        scores = ttnn.matmul(q, keys, transpose_b=True, dtype=ttnn.float32, compute_kernel_config=_HIFI4)
        scores = ttnn.add(ttnn.multiply(scores, self.scale), ttnn.add(self.sink_bias, bias_row))
        probs = ttnn.typecast(ttnn.softmax(scores, dim=-1, compute_kernel_config=_HIFI4), ttnn.bfloat16)
        o = ttnn.matmul(probs, keys, compute_kernel_config=_HIFI4)
        o = _rope_tail(o, q_cos, q_neg_sin, self.rot)
        o = ttnn.reshape(o, [1, t, self.local_groups, -1])
        o = ttnn.matmul(ttnn.permute(o, (0, 2, 1, 3)), self.wo_a, compute_kernel_config=_HIFI4)
        o = ttnn.reshape(ttnn.permute(o, (0, 2, 1, 3)), [1, 1, t, -1])
        return self._all_reduce(self.wo_b(_decode_activation(self.wo_b, o)))

    def moe(self, x: ttnn.Tensor) -> ttnn.Tensor:
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
        out = ttnn.experimental.deepseek.moe.fused_experts(
            ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT),
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
            experts_block_size=min(experts.experts_block_size, num_experts),
        )
        routed = ttnn.to_layout(ttnn.reshape(out, [1, 1, t, d]), ttnn.TILE_LAYOUT)
        x_rep = _decode_activation(self.shared_w1, x)
        gate = ttnn.clamp(self.shared_w1(x_rep), max=self.limit)
        up = ttnn.clamp(self.shared_w3(_decode_activation(self.shared_w3, x_rep)), min=-self.limit, max=self.limit)
        shared = _rows(self.shared_w2(ttnn.multiply(ttnn.silu(gate), up)))
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

    ``lm_head`` is the decode model's tied head (a :class:`~..lm_head.DeepSeekV41LMHead`, or any callable
    returning the replicated ``[.., V]`` logits, on the same ``device``), ``rope_main`` the sliding
    ``(cos_half, sin_half)`` ``[max_seq, Rd / 2]`` tables.
    ``main_rows`` is how many verified rows one :meth:`write_main` takes, which sizes the rings
    (``ring_rows``, ``window + main_rows - 1`` rounded up to a tile), and ``stash_rows`` the depth of
    :attr:`stash`, the ring the decode model writes each step's target-layer means into at
    ``pos % stash_rows`` (per target, ``[1, 1, stash_rows, STASH_PIECE]`` column pieces).

    ``expert_dtype`` defaults to bfloat8_b, not the decode MoE's bfloat4_b: it flips fewer of the bf16-ranked
    top-3-of-128 routes. ``fused_experts`` double-buffers a full-K weight slice per core (~700 KB at bfloat8_b),
    which fits below the drafter's own rings on its stage.

    ``weight_dtype`` (attention projections and ``main_proj``) and ``shared_dtype`` (shared expert) size
    the drafter's rings: it runs on a pipeline stage of its own, so with ``use_prefetcher`` it builds its
    five GCBs (:func:`dspark_prefetch_rings`) on ``device``. The caller owns the prefetcher session, as for
    the decode layers.

    The stash (and the one-hot that gathers from it) lives on ``stash_device``, the decode stage holding
    the target layers. When that is not ``device``, ``link`` is a D2D ``(sender, receiver)`` socket pair
    from it to ``device``: :meth:`send_main` gathers the staged rows there and sends them, and
    :meth:`write_main` receives them.
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
        expert_dtype=ttnn.bfloat8_b,
        shared_dtype=ttnn.bfloat8_b,
        tp_size: int = 4,
        main_rows: int = 8,
        stash_rows: int = 256,
        use_prefetcher: bool = False,
        stash_device=None,
        link=None,
    ):
        stash_device = device if stash_device is None else stash_device
        if (link is None) != (stash_device is device):
            raise ValueError("a stash on another device needs a link to the drafter's, and only then")
        cache = _as_cache(cache)
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
        self.targets = list(config.dspark_target_layer_ids)
        self.main_rows = main_rows
        self.stash_rows = stash_rows
        self.ring_rows = -(-(self.window + main_rows - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
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
                    cache=stage_cache,
                    weight_dtype=weight_dtype,
                    shared_dtype=shared_dtype,
                    use_prefetcher=use_prefetcher,
                    prefetch_buffers=prefetch_buffers,
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
        # the frontier and how many positions up to it the window holds.
        self._block_fields, offset = {}, 0
        for name, n in {"ids": t, "pos": t, "q_pos": t * self.local_heads, "frontier_slot": 1, "live": 1}.items():
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
        fields = {"pos": t, "q_pos": t * self.local_heads, "frontier_slot": 1, "live": 1, "q_rows": t, "keep": t}
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
                # Rows [0, T) Gumbel the draft, [T, 2T) race the residual draws, [2T, 3T) column 0 the accepts.
                "uniform": zeros((1, 1, 3 * t, vocab), ttnn.float32),
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
        if use_prefetcher:
            # The DRISC senders read a weight straight from DRAM when its request arrives, off the command
            # queue; a session that is already open only fenced the uploads queued before it started.
            ttnn.synchronize_device(device)

    # ------------------------------------------------------------------ host side
    def _rope_rows(self, positions) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.as_tensor(positions, dtype=torch.long)
        return make_rope_table(self.cos_half[idx], self.sin_half[idx])

    def set_main_rows(self, positions) -> None:
        """Stage :meth:`write_main` for verified ``positions`` (at most ``main_rows``, consecutive or not),
        rejected rows included: :meth:`set_block`'s mask hides every row past the frontier."""
        w = self.main_rows
        positions = list(positions)
        n = len(positions)
        assert 0 < n <= w, f"{n} rows for a {w}-row write"
        slots = [p % self.ring_rows for p in positions] + [-1] * (w - n)
        positions = positions + [positions[-1]] * (w - n)
        gather = torch.zeros(1, 1, w, self.stash_rows)
        gather[0, 0, torch.arange(w), torch.tensor(positions) % self.stash_rows] = 1
        cos, sin = self._rope_rows(positions)
        host_to_device(gather, self.inputs["gather"])
        host_to_device(torch.tensor(slots, dtype=torch.int32), self.inputs["main_slots"])
        host_to_device(cos, self.inputs["main_cos"])
        host_to_device(sin, self.inputs["main_sin"])

    def set_block(self, token: int, frontier: int) -> None:
        """Stage :meth:`draft` for ``token`` at position ``frontier + 1``, ``frontier`` being the last
        verified position. The draft attends the ring slots of the ``window`` positions up to it; rows a
        round wrote past it are masked.

        Only the block's integers go up, as one packet; :meth:`draft` builds its inputs from them."""
        t = self.block
        if frontier + t >= self.cos_half.shape[0]:
            raise ValueError(f"positions up to {frontier + t} past the {self.cos_half.shape[0]}-row RoPE table")
        positions = list(range(frontier + 1, frontier + 1 + t))
        values = {
            "ids": [token] + [self.noise_id] * (t - 1),
            "pos": positions,
            "q_pos": [p for p in positions for _ in range(self.local_heads)],
            "frontier_slot": [frontier % self.ring_rows],
            "live": [min(self.window, frontier + 1)],
        }
        _upload_packet(values, self._block_fields, self.inputs["block"])

    def set_round(self, start: int, drafts: int) -> None:
        """Stage :meth:`accept_draft` for the round that verified ``[next, the last draft's first drafts]`` at
        positions ``start .. start + drafts``. The decode steps of those positions must have handed the drafter
        their logits (:meth:`write_logits`), and the last :meth:`draft` / :meth:`accept_draft` must be the one
        that proposed those drafts.

        The packet holds the next block's fields for a frontier of ``start`` (no draft accepted); the trace
        adds the accepted count on device."""
        t = self.block
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
            "q_rows": [(start + j) % LOGIT_ROWS for j in range(t)],
            "keep": [int(j < drafts) for j in range(t)],
        }
        _upload_packet(values, self._round_fields, self.inputs["round"])

    def set_sampling(self, temperature: float) -> None:
        """Draft (and accept) at ``temperature``; 0, the initial state, is greedy."""
        if temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        inv = 1.0 / temperature if temperature > 0 else 0.0
        for name, value in (("temp", temperature), ("inv_temp", inv)):
            host_to_device(torch.full((1, 1, 1, 1), value), self.inputs[name])

    def refresh_uniforms(self, seed: int) -> None:
        """Queue a fresh draw of the uniforms the next draft or accept samples with, in place on device (an
        eager op: a trace would replay one seed). ``seed`` must be nonzero; every rank draws the same."""
        if seed == 0:
            raise ValueError("ttnn.uniform takes seed 0 as 'pick one per core on the host'")
        ttnn.uniform(self.inputs["uniform"], 0.0, 1.0, seed)

    # ------------------------------------------------------------------ device side (traceable)
    def _block_inputs(
        self, ids: ttnn.Tensor, pos: ttnn.Tensor, q_pos: ttnn.Tensor, frontier_slot: ttnn.Tensor, live: ttnn.Tensor
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, dict, ttnn.Tensor]:
        """A block's inputs from its fields (``ids`` / ``pos`` / ``q_pos`` uint32 ``[1, n]``, ``frontier_slot``
        and ``live`` fp32 TILE ``[1, 1, 1, 1]``): ``(x_embed [1, 1, T, D], token [1, 1] uint32, rope, bias_row
        [1, 1, 1, kv_rows])``, ``rope`` as :meth:`DeepSeekV41DSparkStage.forward` takes it.

        The rows' embeddings and RoPE rows are looked up by id and position. ``bias_row`` opens ring column
        ``c`` when ``(frontier - c) mod ring_rows`` is below the live count, the block and sink columns always.
        """
        x_embed = _lookup_rows(ids, self.embed.embedding_weight)
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

    def draft(self) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Draft the block :meth:`set_block` staged: :attr:`out`'s ``(ids [1, 1, block] uint32, confidence [1, 1,
        block, 1] fp32, logits [1, 1, block, V] bf16 ROW_MAJOR)``.

        Row ``i`` sits at position ``frontier + 1 + i`` and ``ids[i]``, the argmax of its Markov-biased
        ``logits[i]`` plus ``temperature`` times Gumbel noise (from the staged uniforms), is the proposal for
        position ``frontier + 2 + i``: a sample from ``softmax(logits[i] / temperature)``, the argmax when greedy.
        ``confidence[i]`` scores it.
        """
        f = _packet_fields(self.inputs["block"], self._block_fields)
        ids, pos, q_pos = (ttnn.typecast(f[name], ttnn.uint32) for name in ("ids", "pos", "q_pos"))
        scalar = [1, 1, 1, 1]
        return self._draft(ids, pos, q_pos, _as_float(f["frontier_slot"], scalar), _as_float(f["live"], scalar))

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
            u = ttnn.slice(inp["uniform"], [0, 0, 2 * t, 0], [1, 1, 3 * t, 1])
            hits = ttnn.multiply(ttnn.lt(ttnn.multiply(u, p_d), q_d), ttnn.gtz(p_d))
            residual = ttnn.relu(ttnn.subtract(q, ttnn.multiply(p, math.exp(-DRAFT_PROB_PENALTY))))
            race = _exponentials(ttnn.slice(inp["uniform"], [0, 0, t, 0], [1, 1, 2 * t, vocab]))
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
        return self._draft(ids, pos, q_pos, frontier_slot, live)

    def _draft(
        self, ids: ttnn.Tensor, pos: ttnn.Tensor, q_pos: ttnn.Tensor, frontier_slot: ttnn.Tensor, live: ttnn.Tensor
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        inp, t = self.inputs, self.block
        x_embed, token, rope, bias_row = self._block_inputs(ids, pos, q_pos, frontier_slot, live)
        streams = ttnn.to_layout(
            ttnn.repeat(tokens_from_rows(x_embed), ttnn.Shape([1, 1, self.hc, 1])), ttnn.TILE_LAYOUT
        )
        pre_mix = self.pre_mix_init
        for stage in self.stages:
            streams, pre_mix = stage.forward(streams, pre_mix, rope, bias_row)

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

        rank, vocab = self.markov_embed.shape[-1], logits.shape[-1]
        gumbel = ttnn.neg(ttnn.log(_exponentials(ttnn.slice(inp["uniform"], [0, 0, 0, 0], [1, 1, t, vocab]))))
        noise = ttnn.multiply(gumbel, inp["temp"])
        ids, embeds, rows = [], [], []
        for i in range(t):
            embed = ttnn.reshape(ttnn.embedding(token, self.markov_embed), [1, 1, 1, rank])
            embeds.append(embed)
            row = ttnn.slice(logits, [0, 0, i, 0], [1, 1, i + 1, vocab])
            row = ttnn.add(row, self.markov_head(ttnn.to_layout(embed, ttnn.TILE_LAYOUT)))
            rows.append(row)
            noisy = ttnn.add(ttnn.typecast(row, ttnn.float32), ttnn.slice(noise, [0, 0, i, 0], [1, 1, i + 1, vocab]))
            best = ttnn.argmax(ttnn.to_layout(noisy, ttnn.ROW_MAJOR_LAYOUT), dim=-1)
            ids.append(ttnn.reshape(best, [1, 1, 1]))
            token = ttnn.reshape(best, [1, 1])
        markov = ttnn.to_layout(ttnn.concat(embeds, dim=2), ttnn.TILE_LAYOUT)
        conf = ttnn.add(
            ttnn.linear(x, self.conf_hidden.weight, dtype=ttnn.float32, compute_kernel_config=_HIFI4),
            ttnn.linear(markov, self.conf_markov.weight, dtype=ttnn.float32, compute_kernel_config=_HIFI4),
        )
        out = self.out
        for name, value in (
            ("ids", ttnn.concat(ids, dim=-1)),
            ("conf", conf),
            ("rows", ttnn.to_layout(ttnn.concat(rows, dim=2), ttnn.ROW_MAJOR_LAYOUT)),
        ):
            ttnn.copy(ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG), out[name])
        return out["ids"], out["conf"], out["rows"]
