# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Multichip (tensor-parallel) TTNN decoder layer for ``meta-models/Muse-Glimmer-30B``.

``MultichipDecoder`` takes the single-chip :class:`OptimizedDecoder` and fractures
every projection weight across the four Blackhole chips of this host's
``P300_X2`` mesh (2 x P300 boards, 4 dies, opened as ``ttnn.MeshShape(1, 4)`` with
``FABRIC_1D_RING``), so one decoder layer's weights are streamed from four DRAM
subsystems at once instead of one.

The plan in one table (per-layer, per device, at the shipped precision policy):

===============  ==================  ==========  ==========================  ===============
tensor           full ``[K, N]``     mesh axis   per device                  padding
===============  ==================  ==========  ==========================  ===============
``wqkv``         6656 x 4608         column      6656 x **1280**             KV replicated
``attn_gate``    6656 x 4096         column      6656 x 1024                 none
``o_proj``       4096 x 6656         row         1024 x 6656                 none
``mlp_gate``     6656 x 19968        column      6656 x **5120**            4992 -> 5120
``mlp_up``       6656 x 19968        column      6656 x **5120**            4992 -> 5120
``mlp_down``     19968 x 6656        row         **5120** x 6656            4992 -> 5120
4 x RMSNorm      6656                replicated  6656                        none
K/V cache        blocks x 2 x 64 x 128  by head  blocks x **1** x 64 x 128   KV replicated
page table       batch x blocks      replicated  batch x blocks              none
===============  ==================  ==========  ==========================  ===============

Two rows are not a plain division and both are deliberate:

**KV replication.** The model has 32 query heads and only ``num_key_value_heads =
2``, so 2 KV heads cannot be split four ways.  Each device takes 8 contiguous
query heads and the *one* KV head those heads read under GQA (group size
32/2 = 16, so devices 0-1 share KV head 0 and devices 2-3 share KV head 1).  The
fused QKV weight therefore grows from 4608 to ``4 x 1280 = 5120`` columns -- the
K/V blocks appear twice -- and each device's KV cache holds one head instead of
two.  Per-device cache bytes still **halve** against the single-chip layer;
they do not quarter, and the reason is the model's GQA ratio, not the mesh.
The alternative -- splitting the KV *sequence* across a device pair and merging
the two SDPA partials -- is recorded as a rejected alternative in
``doc/multichip_decoder/README.md``.

**MLP intermediate padding.** ``19968 / 4 = 4992`` is 156 tiles, which is *not* a
multiple of the 8 DRAM banks a width-sharded weight needs (156/8 = 19.5), and it
shares no useful core count with the 208 hidden-size tiles.  Padding each
device's slice to 5120 (160 tiles) with zero columns costs 2.6 % of the MLP
weight bytes and buys two things: an exact one-shard-per-bank DRAM weight, and a
**single** L1 core grid for the whole decode step -- the single-chip layer needed
a 16-core boundary grid plus a 26-core MLP grid and paid two reshards per token.
``silu(0) * 0 = 0`` and the matching zero rows of ``mlp_down`` make the padding
numerically inert; ``test_mlp_padding_is_inert`` pins that.

Collectives
-----------

Both row-parallel projections (``o_proj``, ``mlp_down``) produce a full-width
*partial* sum that has to be reduced.  This layer keeps the **residual stream
replicated** and reduces once per sublayer, i.e. **two reductions per layer** and
no conversion at the layer boundary, so the stacked model pays nothing to hand
one layer's output to the next.  Which spelling of the reduction is used is a
measurement rather than a contract -- a ring all-reduce *is* a reduce-scatter plus
an all-gather -- and the two modes disagree, so decode uses the pair and prefill
uses the fused op (:data:`DEFAULT_DECODE_CCL_MODE`).

That is a measured choice, not an assumption.  ``bench/topology_probe.py``
measures four boundary contracts through the *next consuming op* (the norm, the
residual add and the following matmul) rather than stopping at the collective;
``doc/multichip_decoder/README.md`` has the table.  The short version is that on
a 4-device ring the fractured (reduce-scatter) residual moves exactly the same
bytes as the all-reduce -- ``1.5 S`` per sublayer either way -- while needing an
extra all-gather *and* a distributed-norm stats gather per sublayer.

The collective **payload dtype** is a separate knob and it is split by mode:
prefill reduces in BFP8 (11 % of the prefill layer, at 1.2e-4 of PCC) and decode
reduces in BF16, because the decode accuracy budget is 7.9e-5 wide and the decode
gain is 1.6 %.  See :data:`DEFAULT_PREFILL_CCL_DTYPE`.

Layout contract
---------------

``prefill_forward`` and ``decode_forward`` take and return a **replicated**
``hidden_states`` (identical on all four devices, bit-identical by construction
because the collective is what produces it) and are otherwise the single-chip
contract: same paged KV semantics, same arbitrary logical sequence lengths, same
131072-token capability, same ``sliding_kv_tail`` hand-off -- except that the
sliding tail is now a per-device tensor carrying that device's single KV head,
``[1, 1, tail, 128]`` instead of ``[1, 2, tail, 128]``.  The decode step keeps
the activation width-sharded in L1 on one 16-core grid; see
:data:`MULTICHIP_BOUNDARY_CORES` for why 16 is both the largest legal grid and
the measured winner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import (
    DEFAULT_PREFILL_CHUNK_SIZE,
    LAYER_KIND_FULL,
    LAYER_KIND_SLIDING,
    MODEL_ID,
    PREFILL_SDPA_MAX_SEQ,
    TILE_SIZE,
    MuseGlimmerLayerConfig,
    PagedAttentionConfig,
    _get_layer_tensor,
    _require_muse_glimmer_text_config,
    _rope_cos_sin,
    _text_config,
    reference_layer_indices,
    resolve_layer_kind,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.fused_decoder import _FusedNorm, norm_compute_kernel_config
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import (
    DEFAULT_PRECISION,
    PREFILL_MCAST2D,
    PREFILL_MINIMAL_BLOCKS,
    PREFILL_NORM_SHARD_MAX_ROWS,
    OptimizedDecoder,
    PrecisionPolicy,
    _OptimizedMLP,
    _override_precision,
    decode_matmul_program_config,
    dram_sharded_weight_memcfg,
    prefill_mcast2d_program_config,
)

__all__ = [
    "CCL_TOPOLOGY",
    "DEFAULT_DECODE_CCL_DTYPE",
    "DEFAULT_DECODE_CCL_RS_WORKERS",
    "DEFAULT_PREFILL_CCL_RS_WORKERS",
    "DEFAULT_DECODE_CCL_MODE",
    "DEFAULT_L1_SMALL_SIZE",
    "DEFAULT_MESH_SHAPE",
    "DEFAULT_PREFILL_CCL_DTYPE",
    "DEFAULT_PREFILL_CCL_MODE",
    "FABRIC_CONFIG",
    "FABRIC_PACKET_PAYLOAD_BYTES",
    "MODEL_ID",
    "LAYER_KIND_FULL",
    "LAYER_KIND_SLIDING",
    "MULTICHIP_BOUNDARY_CORES",
    "MULTICHIP_DECODE_MATMUL",
    "MULTICHIP_DECODE_SDPA",
    "MULTICHIP_PREFILL_MCAST2D",
    "MULTICHIP_PREFILL_MINIMAL_BLOCKS",
    "P150X2_DECODE_MATMUL",
    "P150X2_PREFILL_MCAST2D",
    "P150X2_PREFILL_MINIMAL_BLOCKS",
    "PREFILL_FRACTURED_NORM_MIN_ROWS",
    "MeshPlan",
    "MultichipDecoder",
    "ROW_PARALLEL_ROLES",
    "mesh_plan",
    "multichip_decode_matmul",
    "multichip_prefill_mcast2d",
    "multichip_prefill_minimal_blocks",
    "minimal_matmul_subblocks",
    "open_multichip_mesh",
    "reference_layer_indices",
    "resolve_layer_kind",
]


#: The mesh this stage targets.  ``ttnn.GetNumAvailableDevices() == 4`` on this
#: host (``ClusterType::P300_X2``: two P300 boards, four Blackhole dies, all
#: MMIO).  Auto-discovery reports an intra-mesh degree histogram of ``{2: 4}``,
#: i.e. every die has exactly two Ethernet neighbours -- a 4-die **ring** -- so a
#: ``1x4`` mesh view is a physical line whose ends are also adjacent and
#: ``ttnn.Topology.Ring`` is legal.  ``2x2`` opens as well and is the shape the
#: ``p300_x2`` mesh-graph descriptor declares; it is rejected for this layer in
#: ``doc/multichip_decoder/README.md`` because every collective here spans all
#: four devices and a 4-device ring moves ``1.5 S`` where two 2-device
#: collectives move ``2 S``.
DEFAULT_MESH_SHAPE = (1, 4)

#: Fabric configuration that must be set **before** ``open_mesh_device``.  Without
#: it every CCL op fails at ``control_plane.cpp:2222`` (``fabric_context_ !=
#: nullptr``) -- setup evidence, not hardware evidence.
FABRIC_CONFIG = ttnn.FabricConfig.FABRIC_1D_RING

#: Fabric packet payload, in bytes.
#:
#: The runtime asks for this one directly -- every CCL dispatch logs *"Fabric
#: packet size 4352 B is suboptimal for transporting 2048 B pages. Configure
#: 8192 B packet size to maximize throughput"*
#: (``ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:39-71``) -- and the advice is
#: about the **page** size, so it flips with the payload dtype: a BF16 page is
#: 2048 B and wants 8192, a BFP8 page is 1088 B and wants 4352.  This layer
#: reduces decode in BF16 and prefill in BFP8, so both were measured
#: (``logs/fabric_packet_probe.log``; decode traced, prefill warmed):
#:
#: =============================================  =========================  =========================
#: op / payload                                   4352 B                     8192 B
#: =============================================  =========================  =========================
#: **decode, BF16**: reduce_scatter + all_gather  19.03 + 16.77 = 35.80 us   19.23 + 15.59 = 34.82 us
#: prefill, BF16: all_reduce                      2197.80                    **1928.31**
#: **prefill, BFP8 (shipped)**: all_reduce        **1563.71**                1581.06
#: prefill, BFP8: reduce_scatter w=4              **759.92**                 775.76
#: =============================================  =========================  =========================
#:
#: So the runtime's advice is right for the decode payload and wrong for the
#: shipped prefill one.  8192 ships because the decode step is the per-token
#: metric: it saves 1.96 us per token per layer (two sublayers x 0.98), against
#: 34.7 us spread over a whole 8192-token prefill chunk.  The consequence is that
#: the warning still fires on the prefill collectives, now recommending 4352;
#: that is the 1.1 % *prefill-collective* row above -- not the reducer margin,
#: which is a different 1.1 % -- measured and taken knowingly.
#: 15232 (the Blackhole maximum) was measured too and adds nothing over 8192.
FABRIC_PACKET_PAYLOAD_BYTES = 8192

#: Topology passed to the reducing collectives.  ``ttnn.all_gather`` ignores and
#: deprecates its ``topology`` argument on this build; ``reduce_scatter`` and
#: ``all_reduce`` still take one.
CCL_TOPOLOGY = ttnn.Topology.Ring

#: Projections whose weight is fractured over the *input* (K) dimension, so the
#: local matmul returns a partial sum that a collective has to reduce.
ROW_PARALLEL_ROLES = ("o_proj", "mlp_down")

#: Cores for every width-sharded L1 tensor in the decode step.
#:
#: **One** grid for the whole step, which the single-chip layer could not manage:
#: it needed a 16-core boundary grid plus a 26-core MLP working grid and paid two
#: reshards per token.  Here the per-device MLP intermediate is 160 tiles after
#: padding and the hidden size is 208, so a single core count can serve both.
#:
#: The candidate set is bounded by two hard divisibility rules, not by taste. A
#: matmul's *input* shard must not be padded -- a padded in0 would feed the
#: reduction columns that are not in the weight -- so the core count has to
#: divide every width that is an ``in0``: 208 (hidden, for QKV/gate/MLP), 32
#: (the gated attention output, for ``o_proj``) and 160 (the MLP intermediate,
#: for ``mlp_down``).  ``gcd`` of those admits only 1, 2, 4, 8 and 16, and the
#: sharded RMSNorm additionally needs the count to divide 208.  So 16 is the
#: largest legal grid, and it is the measured winner
#: (``logs/layer_ab_geometry1.log``, ``logs/layer_ab_geometry3.log``, traced
#: decode ms/token, sliding / full):
#:
#: ================================  ===============  ===============
#: candidate                         sliding          full
#: ================================  ===============  ===============
#: 8 boundary + 8 MLP                0.5018           0.4700
#: 8 boundary + 16 MLP (2 reshards)  0.4886           0.4578
#: **16 everywhere**                 **0.4669**       **0.4358**
#: 4 everywhere                      L1 clash         L1 clash
#: ================================  ===============  ===============
#:
#: The one tensor 16 does *not* divide is the 40-tile QKV projection output (2.5
#: tiles per core), so ``per_core_N`` rounds up to 3 and the op -- which writes on
#: its own storage-core layout rather than the requested grid -- returns it on
#: ``ceil(40/3) = 14`` cores of 3 tiles.  42 tiles for 40, and the following
#: ``sharded_to_interleaved`` drops the pad.  It is an output, never an ``in0``,
#: so the padding cannot reach a reduction;
#: ``test_qkv_output_shard_is_padded_not_wrong`` pins both the padding and the
#: core count, and 2 wasted tiles on one projection is a small price for the 7 %
#: the wider grid buys the rest of the step.
MULTICHIP_BOUNDARY_CORES = 16

#: ``(role, weight dtype) -> (cores, in0_block_w)`` for the decode DRAM-sharded
#: matmuls, re-swept for the per-device shapes.  ``in0_block_w`` must divide the
#: activation's per-core K-tile count, and every per-device K changed, so the
#: single-chip table is not merely suboptimal here -- several entries are
#: illegal.  Legal values per role at 8 cores:
#:
#: ============  =========  ==============  =========================
#: role          K          K-tiles / core  legal ``in0_block_w``
#: ============  =========  ==============  =========================
#: ``wqkv``      6656       13              1, 13
#: ``attn_gate`` 6656       13              1, 13
#: ``o_proj``    **1024**   2               1, 2
#: ``mlp_gate``  6656       13              1, 13
#: ``mlp_up``    6656       13              1, 13
#: ``mlp_down``  **5120**   10              1, 2, 5, 10
#: ============  =========  ==============  =========================
#:
#: Every shipped value is the largest legal one and the measured winner
#: (``logs/layer_ab_geometry*.log``): dropping ``mlp_down`` from 10 to 5 costs
#: 0.9 %, to 2 costs 4.4 %, and dropping gate/up from 13 to 1 costs 37 %.  This
#: is the same shape of result the single-chip stage found -- ``in0_block_w`` is
#: the field that matters most on this part -- with the values moved because
#: every per-device K moved.
MULTICHIP_DECODE_MATMUL: dict[tuple[str, ttnn.DataType], tuple[int, int]] = {
    ("wqkv", ttnn.bfloat8_b): (MULTICHIP_BOUNDARY_CORES, 13),
    ("attn_gate", ttnn.bfloat8_b): (MULTICHIP_BOUNDARY_CORES, 13),
    # $optimize OPT-011 was tried here and **not** taken, on a shipped-path
    # measurement rather than an inherited assumption.  ``o_proj``'s per-device K
    # is 1024 = 32 tiles, i.e. 2 tiles per core on the 16-core grid, which caps
    # ``in0_block_w`` at 2 and is why ``tt-perf-report`` marks this row SLOW at
    # 62 % of peak DRAM.  A narrower working shard is the only way to widen it,
    # its in0 is the gated attention output rather than the residual, and
    # ``decode_forward`` reshards for it -- so the candidate is real and cheap.
    # Measured against the shipped default in one invocation
    # (``logs/final_layer_ab.log``): 8 cores / ``in0_block_w=4`` gives 0.4541 /
    # 0.4236 against ``tp4``/``tp4b``/``tp4c``'s 0.4547 / 0.4546 / 0.4544 and
    # 0.4237 / 0.4238 / 0.4238 -- a 0.11 % win on ``sliding`` and inside the noise
    # on ``full``.  It is 0.11 % / 0.05 %, and
    # it costs an extra reshard op, the single-grid invariant three structural
    # tests assert, and 13 % of the multichip-vs-single-chip PCC headroom
    # -- shipping it moved the worst check below 0.999183 against a 0.999 bar,
    # into the 1.83e-4 that check has left, though that run was not committed so
    # it is the weakest of the three.  For a layer whose job is to be a stacking
    # baseline this is not a good trade.  4 cores is neutral or worse
    # and 2 and 1 fail L1; see ``logs/ab_oproj_workshard.log`` for the exact
    # circular-buffer errors.
    ("o_proj", ttnn.bfloat8_b): (MULTICHIP_BOUNDARY_CORES, 2),
    ("mlp_gate", ttnn.bfloat8_b): (MULTICHIP_BOUNDARY_CORES, 13),
    ("mlp_up", ttnn.bfloat8_b): (MULTICHIP_BOUNDARY_CORES, 13),
    ("mlp_down", ttnn.bfloat8_b): (MULTICHIP_BOUNDARY_CORES, 10),
    ("wqkv", ttnn.bfloat4_b): (MULTICHIP_BOUNDARY_CORES, 13),
    ("attn_gate", ttnn.bfloat4_b): (MULTICHIP_BOUNDARY_CORES, 13),
    ("o_proj", ttnn.bfloat4_b): (MULTICHIP_BOUNDARY_CORES, 2),
    ("mlp_gate", ttnn.bfloat4_b): (MULTICHIP_BOUNDARY_CORES, 13),
    ("mlp_up", ttnn.bfloat4_b): (MULTICHIP_BOUNDARY_CORES, 13),
    ("mlp_down", ttnn.bfloat4_b): (MULTICHIP_BOUNDARY_CORES, 10),
    # BF16 weights are not a shipped policy but stay reachable through
    # ``weight_dtype=``; these are the largest legal values that fit L1 at BF16.
    ("wqkv", ttnn.bfloat16): (MULTICHIP_BOUNDARY_CORES, 1),
    ("attn_gate", ttnn.bfloat16): (MULTICHIP_BOUNDARY_CORES, 1),
    ("o_proj", ttnn.bfloat16): (MULTICHIP_BOUNDARY_CORES, 2),
    ("mlp_gate", ttnn.bfloat16): (MULTICHIP_BOUNDARY_CORES, 1),
    ("mlp_up", ttnn.bfloat16): (MULTICHIP_BOUNDARY_CORES, 1),
    ("mlp_down", ttnn.bfloat16): (MULTICHIP_BOUNDARY_CORES, 5),
}

#: Legal initial decode geometry for the two-device P150x2 profile.
#:
#: Tensor parallelism halves the MLP width to 9984 (312 tiles).  The P150x4
#: table's 16-core MLP shard is therefore illegal: 16 does not divide 312.
#: Twenty-six cores divide both the 208 hidden tiles and 312 local-intermediate
#: tiles, keep every activation shard exact, and reuse the proven single-chip
#: working grid.  These values are deliberately conservative bring-up choices;
#: the target-specific latency sweep is the authority for later tuning.
P150X2_DECODE_MATMUL: dict[tuple[str, ttnn.DataType], tuple[int, int]] = {
    ("wqkv", ttnn.bfloat4_b): (MULTICHIP_BOUNDARY_CORES, 13),
    ("attn_gate", ttnn.bfloat4_b): (MULTICHIP_BOUNDARY_CORES, 13),
    ("o_proj", ttnn.bfloat4_b): (MULTICHIP_BOUNDARY_CORES, 4),
    ("mlp_gate", ttnn.bfloat4_b): (26, 8),
    ("mlp_up", ttnn.bfloat4_b): (26, 8),
    ("mlp_down", ttnn.bfloat4_b): (26, 12),
    ("wqkv", ttnn.bfloat8_b): (MULTICHIP_BOUNDARY_CORES, 13),
    ("attn_gate", ttnn.bfloat8_b): (MULTICHIP_BOUNDARY_CORES, 13),
    ("o_proj", ttnn.bfloat8_b): (MULTICHIP_BOUNDARY_CORES, 4),
    ("mlp_gate", ttnn.bfloat8_b): (26, 4),
    ("mlp_up", ttnn.bfloat8_b): (26, 4),
    ("mlp_down", ttnn.bfloat8_b): (26, 6),
    ("wqkv", ttnn.bfloat16): (MULTICHIP_BOUNDARY_CORES, 1),
    ("attn_gate", ttnn.bfloat16): (MULTICHIP_BOUNDARY_CORES, 1),
    ("o_proj", ttnn.bfloat16): (MULTICHIP_BOUNDARY_CORES, 2),
    ("mlp_gate", ttnn.bfloat16): (26, 1),
    ("mlp_up", ttnn.bfloat16): (26, 1),
    ("mlp_down", ttnn.bfloat16): (26, 3),
}

#: Decode SDPA ``(grid_x, grid_y, q_chunk, k_chunk, max_cores_per_head_batch)``;
#: ``None`` grid entries mean the whole device compute grid.
#:
#: The last field is new here and it is not cosmetic.  ``SdpaDecode`` gives each
#: (batch, KV head) pair ``max_cores_per_head_batch`` cores and then reduces
#: across them in a binary tree bounded to 6 rounds
#: (``sdpa_decode_program_factory.cpp:239-245``), so the *default* 16 was already
#: the effective cap on the single-chip layer's 2 KV heads.  With **one** local
#: KV head per device the same default would hand this layer 16 cores where 64 is
#: still legal, and the local SDPA reads half a cache rather than a quarter (KV
#: replication), so the cap is where the multichip SDPA wins or loses.
MULTICHIP_DECODE_SDPA = (None, None, 0, 0, 32)

#: Payload dtype for the two reducing collectives, per mode.
#:
#: Prefill moves ``2 x 1.5 x seq x 6656 x sizeof(dtype)`` bytes per layer, which
#: at 8192 rows is 327 MB of BF16 against ~9.4 ms of per-device compute -- the
#: collective is 20 % of the prefill layer, so halving the payload is worth 11 %
#: of the whole thing.  Decode moves 416 KiB and gains 1.6 %.  Both are measured on
#: the **released checkpoint** (``logs/layer_ab_real_ccl.log``,
#: ``logs/layer_ab_geometry_final.log``):
#:
#: =========  ==================  ==================  =========================
#: mode       payload BF16        payload BFP8        real-weight PCC cost
#: =========  ==================  ==================  =========================
#: prefill    21.16 / 20.88 ms    18.82 / 18.35 ms    1.1e-4 / 1.2e-4
#: decode     0.4661 / 0.4351     0.4585 / 0.4275     0.8e-4 / 1.8e-4
#: =========  ==================  ==================  =========================
#:
#: The split is not a compromise for its own sake, and the decode half of it was
#: settled by running the candidate rather than by arguing from the prefill
#: number.  ``$optimize`` OPT-012 forbids rejecting a faster reduced-precision
#: candidate on synthetic evidence, so the BFP8 **decode** payload was measured
#: twice on the released checkpoint:
#:
#: * ``logs/real_weight_ccl_dtype_gate.log`` -- an eight-step decode off a real
#:   3000-token prefill, both layer kinds: worst 0.995354 against 0.995440 for
#:   BF16, i.e. it **passes**, and costs 0.9e-4;
#: * ``logs/real_weight_decode_bfp8_experiment.log`` -- the suite's *whole*
#:   real-weight surface with the payload flipped: worst **0.9950028** against the
#:   0.995 bar, on ``decode[sliding] step=6 pos=3006``.
#:
#: So it passes, by 2.8e-6, where the shipped BF16 payload passes by 1.05e-4 --
#: it spends 97 % of the layer's remaining accuracy budget to buy 1.6 % of the
#: decode step.  This layer is a *stacking* baseline: the full model composes 52
#: of them, and ``test_two_layers_stack`` measures how error composes (0.9936 per
#: layer becomes 0.971975 for two -- ``logs/full_test_run.log``).  A margin of
#: three parts per million is not a
#: margin the next stage can build on, and 1.6 % is not what it costs to keep it.
#: Prefill is the opposite case on both counts -- ~2.1e-3 of headroom, 1.2e-4 of
#: cost, 11 % of the window -- so prefill takes it.
DEFAULT_PREFILL_CCL_DTYPE = ttnn.bfloat8_b
DEFAULT_DECODE_CCL_DTYPE = None

#: How the full-width partial is reduced, per mode.  A ring all-reduce *is* a
#: reduce-scatter followed by an all-gather, so both forms move the same bytes --
#: and on device they run the same pair, since ``ttnn.all_reduce`` decomposes into
#: ``reduce_scatter_minimal_async`` + ``all_gather_async``
#: (``ttnn/cpp/ttnn/operations/ccl/all_reduce/all_reduce.cpp``).  The difference is
#: one *host* dispatch against two, which is why the two measure so nearly alike
#: at a bandwidth-bound payload.
#:
#: All rows from one invocation of the A/B harness at the shipped configuration
#: (``logs/layer_ab_reducer_final.log``, traced decode ms/token and 8192-token
#: prefill ms, sliding / full); an earlier version of this table quoted
#: ``logs/layer_ab_ccl_mode.log``, which predates the per-payload worker counts:
#:
#: =====================  ==========================  ==========================
#: reducer                traced decode               prefill 8192
#: =====================  ==========================  ==========================
#: ``all_reduce``         0.4624 / 0.4310             19.13 / 18.64
#: ``rs_ag``              0.4571 / 0.4259             19.43 / 18.34
#: ``rs_ag`` prefill @1w  0.4570 / 0.4259             21.04 / 20.65
#: **shipped** (split)    **0.4573 / 0.4258**         **18.89 / 18.36**
#: =====================  ==========================  ==========================
#:
#: The pair wins decode by 1.10 % (``sliding``) / 1.21 % (``full``), and prefill is
#: a wash once both payloads are
#: tuned: see :data:`DEFAULT_PREFILL_CCL_RS_WORKERS` for the 0.24 % op-level gap
#: and ``logs/layer_ab_reducer_final.log`` for the whole-layer A/B that measures
#: every reducer candidate in one invocation.  The two modes are still split
#: because decode is the per-token metric and the fused op costs nothing to keep
#: on prefill.  (An earlier version of this comment recorded the prefill gap as
#: "12 % slower" from ``logs/layer_ab_ccl_workers.log``, which predates
#: :data:`DEFAULT_PREFILL_CCL_RS_WORKERS` and therefore measured a prefill payload
#: with the decode-tuned single worker.)  The mechanism is that at 32 rows
#: the collective is latency-bound and the fused op's internal barrier is worth
#: less than the two dispatches cost, while at 8192 rows it is bandwidth-bound and
#: one dispatch wins.  (The fractured-residual contract is decided by a different
#: number -- the cost of the *distributed norm* it forces, 14.90 us against the
#: shipped full-width norm's 8.11 us; see the README and
#: ``bench/fractured_decode_probe.py``.)
DEFAULT_PREFILL_CCL_MODE = "all_reduce"
DEFAULT_DECODE_CCL_MODE = "rs_ag"

#: ``num_workers_per_link`` for ``ttnn.reduce_scatter``.
#:
#: The single largest non-matmul win in this stage, and it is one integer.  At the
#: decode payload (416 KiB) the reduce-scatter is pure fixed cost -- 2.7 GB/s against
#: the **120.6 GB/s** the same fabric reaches on the 8192-row BF16 all-gather at
#: the shipped packet size (``logs/fabric_packet_probe.log``, 678.41 us for
#: 81,788,928 B) -- and the op's default worker
#: count *costs* time there because each worker's setup and sync is paid whether or
#: not there is payload for it.  Traced, at the real decode shape, both memory
#: configs (``logs/ccl_tuning_probe.log``):
#:
#: ===========================  ==========  ==============
#: candidate                    DRAM        L1 sharded
#: ===========================  ==========  ==============
#: ``reduce_scatter`` default   33.55 us    33.78 us
#: ``num_workers_per_link=1``   **20.93**   **21.19**
#: ``num_workers_per_link=2``   34.30       34.59
#: ``num_workers_per_link=4``   34.55       34.62
#: ``num_links=1``              27.82       28.11
#: ``Topology.Linear``          39.88       40.09
#: ``ttnn.all_reduce``          46.38       50.97
#: ``ttnn.all_gather``          16.39       17.22
#: ===========================  ==========  ==============
#:
#: ``chunks_per_sync`` (2/5/10/20) and ``num_buffers_per_channel`` (2/4/8) move it
#: by less than 1 %.  The ``use_l1_small_for_semaphores=True`` row of that sweep
#: reads *"Out of Memory: Not enough space to allocate 1760 B L1_SMALL buffer"* --
#: which is **not** a property of the part, as an earlier version of this comment
#: claimed: the probe reaches that case as roughly its 19th distinct CCL program
#: and had simply exhausted its own ``L1_SMALL`` region (see
#: :data:`DEFAULT_L1_SMALL_SIZE`).  The flag is what this layer ships, and it
#: works.  The Linear row is also the measurement that earns the ``1x4`` mesh
#: view: on a ``2x2`` view every collective is demoted to Linear
#: (``ccl_common.cpp:98-110``), which is 19 % slower here.

#: ``num_workers_per_link`` is swept **per payload**, not once.  At the 416 KiB
#: decode payload one worker wins by 39 % (above); at the 57.9 MB prefill payload
#: the same op is bandwidth-bound and wants four -- 1814.9 us at one worker
#: against **759.9** at four, on the shipped BFP8 payload
#: (``logs/fabric_packet_probe.log``).  That is also why the prefill reducer is
#: not the pair.  At the **shipped** 8192 B packet, ``reduce_scatter(w=4) +
#: all_gather`` is 775.8 + 809.2 = 1584.9 us against ``all_reduce``'s **1581.1**,
#: i.e. the same to 0.24 %; at the op default 4352 B the same rows read 759.9 +
#: 810.2 = 1570.1 against 1563.7, the same to 0.4 %.  Either way the pair buys
#: nothing on prefill, so prefill keeps the single dispatch.  An earlier version
#: of this stage recorded that gap as "12 % slower", which was the decode-tuned
#: one-worker value applied to a prefill payload.
DEFAULT_DECODE_CCL_RS_WORKERS = 1
#: The same knob at the prefill payload, where the op is bandwidth-bound rather
#: than latency-bound and wants four workers instead of one (1814.9 -> 759.9 us on
#: the shipped BFP8 payload).  It only applies if a caller selects the
#: reduce-scatter pair for prefill; the shipped prefill reducer is the fused op.
DEFAULT_PREFILL_CCL_RS_WORKERS = 4

#: ``(role, weight dtype) -> ((max_rows, (grid_y, in0_block_w)), ...)`` ascending,
#: for the prefill 2D-multicast matmul; ``None`` past the last band means
#: "``minimal_matmul``".  Re-swept for the per-device shapes: ``in0_block_w`` must
#: divide the *full* K in tiles here (the activation is DRAM interleaved, not
#: sharded), and ``mlp_down``'s K is now 160 tiles, so the single-chip table's
#: ``in0_block_w = 26`` is illegal on that row.
MULTICHIP_PREFILL_MCAST2D: dict[tuple[str, ttnn.DataType], tuple[tuple[int, tuple[int, int]], ...]] = {
    ("wqkv", ttnn.bfloat8_b): ((128, (2, 13)), (256, (4, 13)), (512, (8, 26)), (1024, (8, 16))),
    ("attn_gate", ttnn.bfloat8_b): ((128, (2, 13)), (256, (4, 16)), (512, (8, 26)), (1024, (8, 13))),
    ("o_proj", ttnn.bfloat8_b): ((128, (2, 8)), (256, (4, 8)), (512, (8, 16)), (1024, (8, 16))),
    ("mlp_gate", ttnn.bfloat4_b): ((64, (2, 13)), (128, (4, 13)), (256, (8, 13)), (512, (8, 8))),
    ("mlp_up", ttnn.bfloat4_b): ((64, (2, 13)), (128, (4, 13)), (256, (8, 13)), (512, (8, 8))),
    ("mlp_down", ttnn.bfloat4_b): ((64, (2, 20)), (128, (4, 20)), (512, (8, 20))),
}

#: ``(role, weight dtype) -> ((min_rows, blocks | None), ...)`` descending, for
#: prefill ``minimal_matmul``.  ``K_block_size`` has to divide the per-device K in
#: tiles, which moved for ``o_proj`` (32) and ``mlp_down`` (160).
MULTICHIP_PREFILL_MINIMAL_BLOCKS: dict[
    tuple[str, ttnn.DataType], tuple[tuple[int, tuple[int, int, int] | None], ...]
] = {
    ("wqkv", ttnn.bfloat8_b): ((8192, None), (2048, (8, 8, 16)), (TILE_SIZE, (2, 8, 24))),
    ("attn_gate", ttnn.bfloat8_b): ((8192, None), (2048, (8, 8, 16)), (TILE_SIZE, (2, 8, 16))),
    ("o_proj", ttnn.bfloat8_b): ((8192, (8, 8, 8)), (2048, (4, 8, 24)), (TILE_SIZE, (2, 8, 24))),
    ("mlp_gate", ttnn.bfloat4_b): ((8192, (4, 8, 32)), (2048, (8, 8, 16)), (TILE_SIZE, (4, 16, 16))),
    ("mlp_up", ttnn.bfloat4_b): ((8192, (4, 8, 32)), (2048, (8, 8, 16)), (TILE_SIZE, (4, 16, 16))),
    ("mlp_down", ttnn.bfloat4_b): ((8192, (8, 10, 16)), (2048, None), (TILE_SIZE, (4, 10, 24))),
}

# P150x2 keeps the single-chip prefill blocks wherever the K dimension is
# unchanged or still divisible.  o_proj is the exception: its local attention
# width is 2048 (64 tiles), so use the already-proven multichip entries whose
# K_block_size is 8 instead of the single-chip table's occasional 13.
P150X2_PREFILL_MCAST2D = dict(PREFILL_MCAST2D)
P150X2_PREFILL_MINIMAL_BLOCKS = {
    **PREFILL_MINIMAL_BLOCKS,
    **{key: value for key, value in MULTICHIP_PREFILL_MINIMAL_BLOCKS.items() if key[0] == "o_proj"},
    ("o_proj", ttnn.bfloat4_b): MULTICHIP_PREFILL_MINIMAL_BLOCKS[("o_proj", ttnn.bfloat8_b)],
}


def multichip_decode_matmul(tp: int) -> dict[tuple[str, ttnn.DataType], tuple[int, int]]:
    """Return the legal decode geometry for a qualified tensor-parallel width."""
    if tp == 2:
        return P150X2_DECODE_MATMUL
    if tp == 4:
        return MULTICHIP_DECODE_MATMUL
    raise ValueError(f"Muse-Glimmer multichip decode supports tp=2 or tp=4, got tp={tp}")


def multichip_prefill_mcast2d(tp: int) -> dict:
    if tp == 2:
        return P150X2_PREFILL_MCAST2D
    if tp == 4:
        return MULTICHIP_PREFILL_MCAST2D
    raise ValueError(f"Muse-Glimmer multichip prefill supports tp=2 or tp=4, got tp={tp}")


def multichip_prefill_minimal_blocks(tp: int) -> dict:
    if tp == 2:
        return P150X2_PREFILL_MINIMAL_BLOCKS
    if tp == 4:
        return MULTICHIP_PREFILL_MINIMAL_BLOCKS
    raise ValueError(f"Muse-Glimmer multichip prefill supports tp=2 or tp=4, got tp={tp}")


def minimal_matmul_subblocks(m_block: int, n_block: int, *, prefer_wide: bool) -> tuple[int, int]:
    """Largest legal ``(subblock_h, subblock_w)`` for a ``minimal_matmul`` block pair.

    The single-chip layer picked ``(2, 4)`` or ``(4, 2)`` from the output width
    against the row count, which is safe there and **not** here: ``minimal_matmul``
    requires ``M_block_size % subblock_h == 0``
    (*"M_block_size (2) must be divisible by subblock_h (4)"*), and tensor
    parallelism narrowed every per-device output width, so the same table entry
    that took the ``(2, 4)`` branch on one chip takes the ``(4, 2)`` branch here
    and is rejected.  It is a real fault rather than a slow path: it fires on
    every batched prefill, whose per-user rows exceed the fractured ``wqkv``
    output width of 1280.

    So the inherited pair is kept wherever it is legal -- every measured entry in
    the geometry tables was swept with it -- and only the illegal combinations
    fall through to the largest legal pair, with the shape breaking ties in the
    direction the inherited heuristic intended.
    """
    inherited = (2, 4) if prefer_wide else (4, 2)
    if m_block % inherited[0] == 0 and n_block % inherited[1] == 0:
        return inherited
    best = (1, 1)
    for height in range(1, min(m_block, 8) + 1):
        if m_block % height:
            continue
        for width in range(1, min(n_block, 8 // height) + 1):
            if n_block % width:
                continue
            better_area = height * width > best[0] * best[1]
            same_area = height * width == best[0] * best[1]
            if better_area or (same_area and ((width > best[1]) if prefer_wide else (height > best[0]))):
                best = (height, width)
    return best


#: ``l1_small_size`` for a mesh that runs collectives.
#:
#: Not a tuning knob -- without it this layer eventually fails to run. Every CCL
#: dispatch creates a global semaphore, and
#: ``all_gather_multicast_factory.cpp:36-43`` puts it in ``L1_SMALL`` **only if
#: the mesh has one**, otherwise in the main L1 pool with the warning *"Allocating
#: semaphores in L1, which may fragment L1 and reduce headroom for subsequent op
#: allocations"*.  The semaphore belongs to the cached program, so it is never
#: freed, and it is allocated at the top of L1: after one decode the largest
#: contiguous free block drops from 1,461,376 to 1,434,048 B per bank, and the
#: *next* 256-row prefill -- whose sharded RMSNorm wants two 213 KB L1 tensors --
#: fails with
#:
#:     Statically allocated circular buffers in program 20 clash with L1 buffers
#:     on core range [0-0 - 7-1]. L1 buffer allocated at 1119552 and static
#:     circular buffer region ends at 1137536
#:
#: The size is bounded on both sides and the ceiling is the decode step's own
#: budget: its circular buffers end at 1,137,536 B and its live L1 tensors take
#: 316,544 B from the top, so it needs 1,454,080 of the 1,461,376 B pool and has
#: **7,296 B** to give away.  The whole ladder was measured -- the faithful probe
#: is "build both layer kinds, 256-row prefill then decode on each", which is the
#: sequence that first exposed this (``logs/fabric_packet_probe.log`` for the
#: isolated view, and the ladder below from the probe plus the graph-audit tests):
#:
#: ======  ==================  =============================================
#: bytes   CCL programs        result
#: ======  ==================  =============================================
#: 32768   128                 the *first* 256-row prefill fails: the
#:                             inherited sharded prefill norm wants two
#:                             213 KB L1 tensors and sits at that edge
#: 8192    32                  896 B over the decode budget: *"L1 buffer
#:                             allocated at 1136640 and static circular
#:                             buffer region ends at 1137536"*
#: 7168    28                  passes, 128 B from the ceiling
#: **6144**  **24**            **shipped**: passes, 1,152 B of margin
#: 4096    16                  passes; the region itself is the constraint
#: 2048    8                   the region fills mid-suite: *"Out of Memory:
#:                             Not enough space to allocate 1760 B
#:                             L1_SMALL buffer across 110 banks"*
#: ======  ==================  =============================================
#:
#: 6144 is the largest value that keeps a four-figure margin under a ceiling that
#: was itself derived from one profile, and it holds 24 distinct CCL programs
#: (256 B each, released by ``clear_program_cache()``).  A stacked model needs two
#: CCL shapes per layer kind, so 24 is not a limit it can reach; a *test session*
#: that builds hundreds of distinct CCL programs can, which is why the suite also
#: bounds its program cache.
DEFAULT_L1_SMALL_SIZE = 6144

#: Which implementation carries the two reducing collectives, per mode.
#:
#: ``"wrapper"`` is ``ttnn.reduce_scatter`` / ``ttnn.all_gather`` / ``ttnn.all_reduce``:
#: the composite ops, which create their own per-program global semaphores and
#: allocate their own output and staging buffers.  ``"async"`` calls the
#: primitives those wrappers lower to -- ``ttnn.experimental.reduce_scatter_minimal_async``
#: and ``ttnn.experimental.all_gather_async`` -- with semaphores this layer owns
#: and, optionally, staging buffers it owns ($optimize OPT-009).
#:
#: The reason to want it is not the buffers.  It is that the *all-gather* half of
#: the reduction has no tuning surface through ``ttnn.all_gather``: the multichip
#: stage's sweep records it as "not tunable" at 16.39 us, next to a reduce-scatter
#: that one integer moved from 33.55 to 20.93.  The async op takes
#: ``num_workers_per_link``, ``chunks_per_sync``, ``num_buffers_per_channel`` and
#: ``use_optimal_ccl_for_llama``.
#:
#: **Split by payload.**  At the 57.9 MB prefill payload the async pair is
#: **15.2 %** faster than ``ttnn.all_reduce`` -- 1348.0 against 1588.7 us, at the
#: shipped BFP8 payload and packet size
#: (``doc/optimized_multichip_decoder/logs/prefill_ccl_probe.log``).  At the 416 KiB
#: decode payload it is **0.2 % slower** than the composite wrappers -- 0.4554 /
#: 0.4246 against 0.4545 / 0.4236 ms/token, three non-overlapping rounds each
#: (``doc/optimized_multichip_decoder/logs/final_layer_ab.log``) -- because there
#: the collective is pure fixed cost and the async op's extra synchronization
#: outweighs its tuning surface.  So prefill takes it and decode does not.
#:
#: Both primitives are given a ``barrier_semaphore``, the all-gather's included.
#: It costs 0.13 % of the prefill collective (1348.0 against 1346.3 without) and
#: it is kept because every other model in the tree passes one and the op takes
#: one -- **not** because this stage established what happens without it.  A
#: fabric ERISC watcher assert was observed once, on a build that also carried the
#: persistent staging buffers :data:`DEFAULT_CCL_PERSISTENT_BUFFERS` rejects, and
#: it has **not** reproduced with the barrier removed on its own: see
#: ``logs/watcher_no_ag_barrier.log``, which is watcher-clean, and the full
#: account in ``doc/optimized_multichip_decoder/work_log.md`` section 10.1.
DEFAULT_DECODE_CCL_IMPL = "wrapper"
DEFAULT_PREFILL_CCL_IMPL = "async"

#: ``num_workers_per_link`` / ``chunks_per_sync`` / ``num_buffers_per_channel``
#: for the async all-gather, per mode.  ``None`` keeps the op's own default.
DEFAULT_DECODE_CCL_AG_WORKERS: int | None = 1
DEFAULT_PREFILL_CCL_AG_WORKERS: int | None = None

#: Give the async reduce-scatter this layer's own staging buffers instead of
#: letting it allocate them per dispatch ($optimize OPT-009).
#:
#: **Off**, and the reason is correctness, not performance.  It is worth 0.36 % of
#: traced decode (0.4523 -> 0.4507 ms/token, ``logs/ab_ccl_async.log``) and it is
#: the only knob in this stage that failed a correctness bisect.
#: ``reduce_scatter_minimal_async_create_intermediate_buffer`` returns an
#: uninitialised chunk-paged staging pair, and the ring algorithm reads the penult
#: intermediate before writing it on the first invocation: with the buffers first
#: used inside decode, decode step 0 measured PCC 0.7395 (``full``) / 0.7749
#: (``sliding``) against the identical layer without them, while steps 1-3 were
#: bit-identical (``logs/regression_bisect.log``).  That is what broke
#: ``test_multichip_matches_single_chip[12345-4-full]`` at 0.7268.
#:
#: Two fixes were implemented and measured -- a warm-up collective through the
#: fresh buffers, then that plus a ``synchronize_device`` and eager allocation at
#: build time (:meth:`MultichipDecoder._prewarm_decode_ccl_buffers`).  Each moved
#: the fault rather than removing it: the first left the *default* configuration
#: wrong on the first ``sliding`` decode step -- that run was not committed -- and
#: the second left the wrapper-prefill/async-decode combination at 0.9605 on the
#: first ``full`` one
#: (``logs/regression_bisect_fixed.log``).  A fault that moves between arms and
#: between runs of the same arm is a race, and an intermittently wrong first token
#: is not a defect a *stacking baseline* may ship for half a percent -- 52 layers compose
#: it, and the full-model and vLLM stages both decode without prefilling through
#: this layer.
#:
#: The knob, the warm-up and the bisect harness are all kept: the async ops
#: themselves are clean in every arm and carry the win, and this is a TTNN
#: first-use contract worth re-testing when the op changes.
DEFAULT_CCL_PERSISTENT_BUFFERS = False

#: Run the two prefill RMSNorms that follow a reduction at the **fractured** width.
#:
#: Both row-parallel prefill projections feed straight into a norm -- ``o_proj``
#: into ``post_attention_layernorm``, ``mlp_down`` into
#: ``post_feedforward_layernorm``.  With this on, those reductions stop at the
#: reduce-scatter, the norm runs distributed over the 1664-wide shard
#: (``rms_norm_pre_all_gather`` -> stats ``all_gather`` -> ``rms_norm_post_all_gather``),
#: and the all-gather that the reduction would have done anyway carries the
#: *normalised* tensor back to full width.
#:
#: The arithmetic is unchanged: a distributed RMSNorm combines per-device partial
#: sums through the stats gather, so it normalises over all 6656 channels exactly
#: as the full-width norm does.  The collective bytes are unchanged too -- the same
#: reduce-scatter and the same all-gather, plus a stats gather of 32 columns.  What
#: changes is that the norm and its reads run at a quarter of the width, and the
#: layer's public prefill contract does not move at all: DRAM-interleaved
#: replicated in, DRAM-interleaved replicated out.
#:
#: Measured on the two-sublayer chain at the real 8192-row shape
#: (``doc/optimized_multichip_decoder/bench/fractured_prefill_probe.py``):
#: **4443.9 us against 5902.1**, i.e. 24.7 % of the chain and ~8 % of the prefill
#: layer.  This is the lever ``doc/multichip_decoder/README.md`` limitation 1 left
#: for "the stage that owns the layer stack"; it needs no second residual contract,
#: so this stage takes it.
#:
#: **Only above** :data:`PREFILL_FRACTURED_NORM_MIN_ROWS`.  The saving is
#: proportional to the rows the norm reads, while the statistics gather it adds is
#: a fixed, latency-bound collective, so at small row counts the trade inverts:
#: measured on the 128-row window it costs **13.4 % / 13.6 %** of device time
#: against 8.98 % / 9.27 % saved on the 8192-row one.  The threshold is the same
#: boundary the inherited norm already switches on -- below it the full-width norm
#: is L1-sharded and cheap, above it it is DRAM-interleaved and is the expensive
#: thing this replaces.
DEFAULT_PREFILL_FRACTURED_NORM = True

#: Rows below which the reduction is finished before the norm, as it was.  Equal to
#: :data:`PREFILL_NORM_SHARD_MAX_ROWS` by construction, not by coincidence: that is
#: exactly where the full-width prefill norm stops being a cheap sharded kernel.
PREFILL_FRACTURED_NORM_MIN_ROWS = PREFILL_NORM_SHARD_MAX_ROWS

#: Hand the next layer the decode residual width-sharded in L1 instead of DRAM
#: interleaved.  See ``OptimizedDecoder.decode_forward``; measured by
#: ``bench/boundary_probe.py`` at 1.9-6.3 us per layer with a bit-identical
#: output (PCC 1.000000000).
DEFAULT_SHARDED_DECODE_IO = True


def open_multichip_mesh(
    mesh_shape: tuple[int, int] = DEFAULT_MESH_SHAPE,
    *,
    trace_region_size: int = 0,
    l1_small_size: int = DEFAULT_L1_SMALL_SIZE,
    packet_payload_bytes: int = FABRIC_PACKET_PAYLOAD_BYTES,
    fabric_config: ttnn.FabricConfig | None = FABRIC_CONFIG,
) -> ttnn.MeshDevice:
    """Open the target mesh with the fabric CCL needs, and somewhere to put its semaphores.

    All three arguments are load-bearing rather than tuning.
    ``ttnn.set_fabric_config`` has to happen before the mesh is opened; a mesh
    opened without it looks healthy and then fails inside the first collective.
    ``l1_small_size`` decides where the per-program CCL semaphores go, and both
    too-large and too-small fail (:data:`DEFAULT_L1_SMALL_SIZE`).
    ``packet_payload_bytes`` is the one the runtime asks for in a warning on every
    dispatch (:data:`FABRIC_PACKET_PAYLOAD_BYTES`).
    """
    if fabric_config is not None:
        router = ttnn.FabricRouterConfig()
        router.max_packet_payload_size_bytes = packet_payload_bytes
        ttnn.set_fabric_config(fabric_config, router_config=router)
    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*mesh_shape),
        trace_region_size=trace_region_size,
        l1_small_size=l1_small_size,
    )


#: Global CCL semaphores per open mesh, keyed by ``id(mesh_device)``.  Shared by
#: every layer on that mesh; see :meth:`MultichipDecoder._ccl_semaphores`.
_CCL_SEMAPHORES: dict[int, dict] = {}


def close_multichip_mesh(mesh: ttnn.MeshDevice, *, fabric_config: ttnn.FabricConfig | None = FABRIC_CONFIG) -> None:
    """Close a mesh opened by :func:`open_multichip_mesh` and drop the fabric.

    Both ``id(mesh)``-keyed semaphore caches are dropped, not just this module's.  Round 5
    of the stage review pointed out that the model's own cache
    (``tt.model._MODEL_CCL_SEMAPHORES``, for the terminal all-gathers) was never cleared,
    which leaves 4 global semaphores per closed mesh alive **and** exposes the ``id()``-reuse
    hazard that :meth:`MuseGlimmerGenerator._kv_cache_signature` explicitly rejects for
    exactly this reason: a freed ``MeshDevice`` wrapper's id can be handed to the next one,
    and the next mesh would then find and reuse semaphores belonging to a dead device.
    """
    _CCL_SEMAPHORES.pop(id(mesh), None)
    # Imported here rather than at module scope: ``tt.model`` imports this module.
    from models.autoports.meta_models_muse_glimmer_30b.tt.model import _MODEL_CCL_SEMAPHORES

    _MODEL_CCL_SEMAPHORES.pop(id(mesh), None)
    ttnn.close_mesh_device(mesh)
    if fabric_config is not None:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@dataclass(frozen=True)
class MeshPlan:
    """How one layer's tensors map onto the mesh.

    Everything the forward pass needs to know about the parallelisation lives
    here, computed once from the HF config and the mesh, so the runtime path
    never re-derives a head count.
    """

    tp: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    #: Query heads owned by each device.
    local_heads: int
    #: KV heads owned by each device: ``1`` when the model has fewer KV heads
    #: than devices, in which case each is held by ``tp / num_key_value_heads``
    #: devices.
    local_kv_heads: int
    kv_replicated: bool
    #: Per-device MLP intermediate width after zero padding.
    local_intermediate: int

    @property
    def local_qkv_width(self) -> int:
        return (self.local_heads + 2 * self.local_kv_heads) * self.head_dim

    @property
    def local_attn_width(self) -> int:
        return self.local_heads * self.head_dim

    def kv_head_of_device(self, device: int) -> int:
        """Which global KV head device ``device`` owns.

        GQA assignment: the query heads a device owns all read the same KV head
        exactly when the group size is a multiple of ``local_heads``, which
        :func:`mesh_plan` checks.
        """
        return (device * self.local_heads) * self.num_key_value_heads // self.num_attention_heads


def mesh_plan(text_config: Any, tp: int, *, dram_banks: int = 8) -> MeshPlan:
    """Derive and validate the parallelisation plan for ``tp`` devices."""
    n_heads = int(text_config.num_attention_heads)
    n_kv = int(text_config.num_key_value_heads)
    head_dim = int(text_config.head_dim)
    hidden = int(text_config.hidden_size)
    intermediate = int(text_config.intermediate_size)

    if n_heads % tp:
        raise ValueError(f"num_attention_heads={n_heads} must be divisible by the {tp}-device mesh")
    local_heads = n_heads // tp
    if hidden // TILE_SIZE % tp:
        raise ValueError(f"hidden_size={hidden} must be a multiple of {tp * TILE_SIZE} for a fractured residual/norm")

    kv_replicated = n_kv < tp
    if kv_replicated:
        if tp % n_kv:
            raise ValueError(f"a {tp}-device mesh cannot replicate {n_kv} KV heads evenly")
        local_kv = 1
        group = n_heads // n_kv
        if group % local_heads:
            raise ValueError(
                f"each device's {local_heads} query heads must fall inside one GQA group of {group}; "
                f"{group} % {local_heads} != 0"
            )
    else:
        if n_kv % tp:
            raise ValueError(f"num_key_value_heads={n_kv} must be divisible by the {tp}-device mesh")
        local_kv = n_kv // tp

    # A width-sharded DRAM weight needs one shard per bank, so the per-device
    # MLP width is padded up to a multiple of ``32 * dram_banks``.
    stride = TILE_SIZE * dram_banks
    local_intermediate = math.ceil(intermediate / tp / stride) * stride

    return MeshPlan(
        tp=tp,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        head_dim=head_dim,
        hidden_size=hidden,
        intermediate_size=intermediate,
        local_heads=local_heads,
        local_kv_heads=local_kv,
        kv_replicated=kv_replicated,
        local_intermediate=local_intermediate,
    )


def fused_qkv_weight(wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor, plan: MeshPlan) -> torch.Tensor:
    """Per-device ``[q | k | v]`` blocks concatenated along the output dim.

    ``wq/wk/wv`` are already ``[in, out]``.  The result is
    ``[hidden, tp * local_qkv_width]``, so a plain ``ShardTensorToMesh(dim=-1)``
    hands each device a contiguous fused QKV of exactly the width
    ``nlp_create_qkv_heads*`` expects for ``(local_heads, local_kv_heads)``.  This
    is the layout ``models/tt_transformers/tt/attention.py:250-276`` builds, with
    the GQA-assigned KV replication of
    ``models/demos/gemma4/tt/attention/weights.py:55-95`` for the case where the
    model has fewer KV heads than devices.
    """
    head_dim = plan.head_dim
    q_width = plan.local_heads * head_dim
    kv_width = plan.local_kv_heads * head_dim
    blocks = []
    for device in range(plan.tp):
        q = wq[:, device * q_width : (device + 1) * q_width]
        if plan.kv_replicated:
            kv_head = plan.kv_head_of_device(device)
            k = wk[:, kv_head * head_dim : (kv_head + 1) * head_dim]
            v = wv[:, kv_head * head_dim : (kv_head + 1) * head_dim]
        else:
            k = wk[:, device * kv_width : (device + 1) * kv_width]
            v = wv[:, device * kv_width : (device + 1) * kv_width]
        blocks.append(torch.cat([q, k, v], dim=-1))
    return torch.cat(blocks, dim=-1).contiguous()


def _pad_per_device(tensor: torch.Tensor, dim: int, tp: int, padded: int) -> torch.Tensor:
    """Zero-pad each device's slice of ``tensor`` along ``dim`` to ``padded``.

    The slices have to be padded *individually* and re-concatenated, because a
    mesh mapper splits the whole tensor evenly: padding the end would give the
    last device all the padding.
    """
    per_device = tensor.shape[dim] // tp
    if padded == per_device:
        return tensor
    pieces = []
    for device in range(tp):
        piece = tensor.narrow(dim, device * per_device, per_device)
        pad = [0] * (2 * tensor.dim())
        # torch.nn.functional.pad's spec counts dims from the last one.
        pad[2 * (tensor.dim() - 1 - (dim % tensor.dim())) + 1] = padded - per_device
        pieces.append(torch.nn.functional.pad(piece, pad))
    return torch.cat(pieces, dim=dim).contiguous()


class _MultichipMLP(_OptimizedMLP):
    """SwiGLU MLP whose ``mlp_down`` output is a partial sum to be reduced.

    Both forwards are inherited; the reduction rides on
    :meth:`MultichipDecoder._decode_projection` /
    :meth:`MultichipDecoder._prefill_projection`, which is where every
    row-parallel projection in this layer is reduced, so there is exactly one
    place that knows about the collective.
    """


class MultichipDecoder(OptimizedDecoder):
    """Tensor-parallel TTNN implementation of ``MuseGlimmerTextDecoderLayer``."""

    def __init__(
        self,
        *,
        plan: MeshPlan,
        ccl_dtype: ttnn.DataType | None = None,
        prefill_ccl_dtype: ttnn.DataType | None = DEFAULT_PREFILL_CCL_DTYPE,
        decode_ccl_dtype: ttnn.DataType | None = DEFAULT_DECODE_CCL_DTYPE,
        ccl_mode: str | None = None,
        prefill_ccl_mode: str = DEFAULT_PREFILL_CCL_MODE,
        decode_ccl_mode: str = DEFAULT_DECODE_CCL_MODE,
        ccl_rs_workers: int | None = None,
        prefill_ccl_rs_workers: int = DEFAULT_PREFILL_CCL_RS_WORKERS,
        decode_ccl_rs_workers: int = DEFAULT_DECODE_CCL_RS_WORKERS,
        ccl_impl: str | None = None,
        prefill_ccl_impl: str = DEFAULT_PREFILL_CCL_IMPL,
        decode_ccl_impl: str = DEFAULT_DECODE_CCL_IMPL,
        ccl_ag_workers: int | None = None,
        prefill_ccl_ag_workers: int | None = DEFAULT_PREFILL_CCL_AG_WORKERS,
        decode_ccl_ag_workers: int | None = DEFAULT_DECODE_CCL_AG_WORKERS,
        ccl_persistent_buffers: bool = DEFAULT_CCL_PERSISTENT_BUFFERS,
        ccl_chunks_per_sync: int | None = None,
        ccl_buffers_per_channel: int | None = None,
        ccl_num_links: int | None = None,
        ccl_ag_barrier: bool = True,
        prefill_fractured_norm: bool = DEFAULT_PREFILL_FRACTURED_NORM,
        prefill_fractured_norm_min_rows: int = PREFILL_FRACTURED_NORM_MIN_ROWS,
        **kwargs,
    ) -> None:
        kwargs.setdefault("decode_matmul", multichip_decode_matmul(plan.tp))
        kwargs.setdefault("boundary_cores", MULTICHIP_BOUNDARY_CORES)
        self.plan = plan
        #: Payload dtype for the two reducing collectives, per mode.  ``None``
        #: keeps the activation dtype.  ``ccl_dtype`` sets both, which is what the
        #: A/B harness passes when it sweeps the payload as one knob.
        self.prefill_ccl_dtype = ccl_dtype if ccl_dtype is not None else prefill_ccl_dtype
        self.decode_ccl_dtype = ccl_dtype if ccl_dtype is not None else decode_ccl_dtype
        self.prefill_ccl_mode = ccl_mode or prefill_ccl_mode
        self.decode_ccl_mode = ccl_mode or decode_ccl_mode
        self.prefill_ccl_rs_workers = ccl_rs_workers or prefill_ccl_rs_workers
        self.decode_ccl_rs_workers = ccl_rs_workers or decode_ccl_rs_workers
        for mode in (self.prefill_ccl_mode, self.decode_ccl_mode):
            if mode not in ("all_reduce", "rs_ag"):
                raise ValueError(f"a CCL mode must be 'all_reduce' or 'rs_ag', got {mode!r}")
        self.prefill_ccl_impl = ccl_impl or prefill_ccl_impl
        self.decode_ccl_impl = ccl_impl or decode_ccl_impl
        for impl in (self.prefill_ccl_impl, self.decode_ccl_impl):
            if impl not in ("wrapper", "async"):
                raise ValueError(f"a CCL impl must be 'wrapper' or 'async', got {impl!r}")
        self.prefill_ccl_ag_workers = ccl_ag_workers if ccl_ag_workers is not None else prefill_ccl_ag_workers
        self.decode_ccl_ag_workers = ccl_ag_workers if ccl_ag_workers is not None else decode_ccl_ag_workers
        self.ccl_persistent_buffers = ccl_persistent_buffers
        self.ccl_chunks_per_sync = ccl_chunks_per_sync
        self.ccl_buffers_per_channel = ccl_buffers_per_channel
        self.ccl_num_links = ccl_num_links
        #: Pass ``all_gather_async`` its barrier semaphore.  **Always true in any
        #: shipped configuration**; the knob exists only so the arm without it can
        #: be reproduced from committed code.  Dropping it is worth 0.13 % of the
        #: prefill collective and is not taken -- see
        #: :data:`DEFAULT_DECODE_CCL_IMPL` and
        #: ``doc/optimized_multichip_decoder/logs/watcher_no_ag_barrier.log``.
        self.ccl_ag_barrier = ccl_ag_barrier
        #: See :data:`DEFAULT_PREFILL_FRACTURED_NORM`.  Requires the async prefill
        #: collective, because it needs the reduce-scatter and the all-gather as
        #: separate ops with the norm between them.
        self.prefill_fractured_norm = prefill_fractured_norm and self.prefill_ccl_impl == "async"
        #: Rows at or below which the reduction is finished before the norm.  A
        #: knob rather than a constant so the ungated arm -- the one that measures
        #: why the gate exists -- runs from committed code.
        self.prefill_fractured_norm_min_rows = prefill_fractured_norm_min_rows
        #: Set by ``_prefill_projection`` when it hands back a *scattered* partial
        #: instead of a reduced one, and consumed by the very next
        #: ``_prefill_norm``.  The pairing is structural: both row-parallel prefill
        #: projections return directly into a norm.
        self._pending_scatter: ttnn.Tensor | None = None
        self._ccl_sems = None
        self._ccl_slot = 0
        decode_sdpa = kwargs.pop("decode_sdpa", None) or MULTICHIP_DECODE_SDPA
        super().__init__(decode_sdpa=decode_sdpa[:4], **kwargs)
        gx, gy, q_chunk, k_chunk = self.decode_sdpa
        self.max_cores_per_head_batch = decode_sdpa[4] if len(decode_sdpa) > 4 else 16
        # Rebuilt with the cap: with one local KV head the default 16 cores per
        # (batch, head) leaves most of the grid idle, and 64 is still inside the
        # 6-round tree-reduction bound.
        self.decode_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
            max_cores_per_head_batch=self.max_cores_per_head_batch,
        )
        self.prefill_mcast2d = multichip_prefill_mcast2d(plan.tp)
        self.prefill_minimal_blocks = multichip_prefill_minimal_blocks(plan.tp)
        if self.mesh_device.get_num_devices() != plan.tp:
            raise ValueError(f"plan targets {plan.tp} devices but the mesh has {self.mesh_device.get_num_devices()}")
        self._prewarm_decode_ccl_buffers()

    def _prewarm_decode_ccl_buffers(self) -> None:
        """Allocate and warm the decode-shape persistent CCL staging, at build time.

        The decode collective has exactly one payload shape -- one tile-row by
        ``hidden_size``, because ``nlp_create_qkv_heads_decode`` caps the decode
        batch at 32 and the activation is tile-padded to that
        ($optimize OPT-005) -- so it can be built once here instead of on the
        first token.  Doing it here rather than lazily is what keeps the warm-up
        (and its ``synchronize_device``) out of ``decode_forward``, which matters
        for two reasons: a traced capture must not contain either, and a
        decode-only caller would otherwise pay a corrupted first token.  See
        :meth:`_warm_persistent_buffers` for why the warm-up exists at all.
        """
        if not (self.ccl_persistent_buffers and self.decode_ccl_impl == "async"):
            return
        dtype = self.decode_ccl_dtype or self.activation_dtype
        probe = ttnn.zeros(
            [1, 1, TILE_SIZE, self.config.hidden_size],
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for slot in range(2):
            self._ccl_persistent_buffers(probe, slot)
        ttnn.deallocate(probe)

    # ------------------------------------------------------------------ setup

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        hf_config: Any,
        layer_idx: int,
        mesh_device: ttnn.MeshDevice,
        max_batch_size: int = 1,
        max_seq_len: int | None = None,
        page_block_size: int = 64,
        max_num_blocks: int | None = None,
        weight_dtype: ttnn.DataType | None = None,
        activation_dtype: ttnn.DataType | None = None,
        kv_cache_dtype: ttnn.DataType | None = None,
        prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE,
        precision: PrecisionPolicy = DEFAULT_PRECISION,
        decode_matmul: dict | None = None,
        boundary_cores: int = MULTICHIP_BOUNDARY_CORES,
        decode_sdpa: tuple | None = None,
        decode_fused_activation: bool | None = None,
        sharded_decode_io: bool = DEFAULT_SHARDED_DECODE_IO,
        ccl_dtype: ttnn.DataType | None = None,
        prefill_ccl_dtype: ttnn.DataType | None = DEFAULT_PREFILL_CCL_DTYPE,
        decode_ccl_dtype: ttnn.DataType | None = DEFAULT_DECODE_CCL_DTYPE,
        ccl_mode: str | None = None,
        prefill_ccl_mode: str = DEFAULT_PREFILL_CCL_MODE,
        decode_ccl_mode: str = DEFAULT_DECODE_CCL_MODE,
        ccl_rs_workers: int | None = None,
        prefill_ccl_rs_workers: int = DEFAULT_PREFILL_CCL_RS_WORKERS,
        decode_ccl_rs_workers: int = DEFAULT_DECODE_CCL_RS_WORKERS,
        ccl_impl: str | None = None,
        prefill_ccl_impl: str = DEFAULT_PREFILL_CCL_IMPL,
        decode_ccl_impl: str = DEFAULT_DECODE_CCL_IMPL,
        ccl_ag_workers: int | None = None,
        prefill_ccl_ag_workers: int | None = DEFAULT_PREFILL_CCL_AG_WORKERS,
        decode_ccl_ag_workers: int | None = DEFAULT_DECODE_CCL_AG_WORKERS,
        ccl_persistent_buffers: bool = DEFAULT_CCL_PERSISTENT_BUFFERS,
        ccl_chunks_per_sync: int | None = None,
        ccl_buffers_per_channel: int | None = None,
        ccl_num_links: int | None = None,
        ccl_ag_barrier: bool = True,
        prefill_fractured_norm: bool = DEFAULT_PREFILL_FRACTURED_NORM,
        prefill_fractured_norm_min_rows: int = PREFILL_FRACTURED_NORM_MIN_ROWS,
        rope_cache: dict[str, ttnn.Tensor] | None = None,
        **kwargs,
    ) -> "MultichipDecoder":
        """Same contract as ``OptimizedDecoder.from_state_dict`` on a mesh.

        The state dict is the *whole* layer's; this method does the fracturing,
        so a caller does not have to know the mesh plan.

        ``rope_cache`` lets a *stack* own the four RoPE tables instead of one per
        layer; ``None`` keeps the single-layer behaviour of building its own.  See
        :func:`build_rope_cache`.
        """
        if kwargs:
            raise TypeError(f"Unexpected MultichipDecoder.from_state_dict kwargs: {sorted(kwargs)}")
        tp = mesh_device.get_num_devices()
        if tp < 2:
            raise ValueError(
                "MultichipDecoder is the multi-chip stage; use OptimizedDecoder on a 1x1 mesh. "
                f"Got a {tp}-device mesh."
            )

        precision = _override_precision(precision, weight_dtype, activation_dtype, kv_cache_dtype)
        text_config = _text_config(hf_config)
        _require_muse_glimmer_text_config(text_config)
        layer_kind = resolve_layer_kind(hf_config, layer_idx)
        dram_banks = mesh_device.dram_grid_size().x
        plan = mesh_plan(text_config, tp, dram_banks=dram_banks)

        max_seq_len = int(max_seq_len or text_config.max_position_embeddings)
        if max_seq_len > text_config.max_position_embeddings:
            raise ValueError(
                f"max_seq_len={max_seq_len} exceeds the HF-advertised context {text_config.max_position_embeddings}"
            )
        if page_block_size % TILE_SIZE != 0:
            raise ValueError(f"page_block_size must be a multiple of {TILE_SIZE}, got {page_block_size}")
        if max_seq_len % TILE_SIZE != 0:
            raise ValueError(f"max_seq_len must be a multiple of {TILE_SIZE}, got {max_seq_len}")
        blocks_per_seq = (max_seq_len + page_block_size - 1) // page_block_size
        if max_num_blocks is None:
            max_num_blocks = max_batch_size * blocks_per_seq
        if max_num_blocks < max_batch_size * blocks_per_seq:
            raise ValueError(
                f"max_num_blocks={max_num_blocks} cannot hold max_batch_size={max_batch_size} x "
                f"{blocks_per_seq} blocks of {page_block_size} tokens"
            )
        if prefill_chunk_size % page_block_size or prefill_chunk_size % TILE_SIZE:
            raise ValueError(
                f"prefill_chunk_size={prefill_chunk_size} must be a multiple of the page block size "
                f"({page_block_size}) and the tile height ({TILE_SIZE})"
            )
        if prefill_chunk_size > PREFILL_SDPA_MAX_SEQ // 2:
            raise ValueError(
                f"prefill_chunk_size={prefill_chunk_size} is too large: the sliding-window prefill "
                f"slice (chunk + window) must stay below the {PREFILL_SDPA_MAX_SEQ}-token SDPA limit"
            )

        # The layer config carries **local** head counts: every inherited forward
        # (head creation, RoPE, cache update, SDPA, head concat, the sliding-tail
        # shape check, the persistent Q filler) then computes the per-device shape
        # without a second head-count concept in the runtime path.
        config = MuseGlimmerLayerConfig(
            layer_idx=layer_idx,
            layer_kind=layer_kind,
            hidden_size=plan.hidden_size,
            intermediate_size=plan.local_intermediate,
            num_attention_heads=plan.local_heads,
            num_key_value_heads=plan.local_kv_heads,
            head_dim=plan.head_dim,
            rms_norm_eps=text_config.rms_norm_eps,
            post_norm_eps=text_config.post_norm_eps,
            qk_scale_factor=text_config.qk_scale_factor,
            sliding_window=text_config.sliding_window if layer_kind == LAYER_KIND_SLIDING else None,
            rope_theta=(float(text_config.layer_rope_theta[layer_idx]) if layer_kind == LAYER_KIND_SLIDING else None),
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            paged_attention_config=PagedAttentionConfig(
                block_size=page_block_size,
                max_num_blocks=max_num_blocks,
            ),
            prefill_chunk_size=prefill_chunk_size,
        )

        replicate = ttnn.ReplicateTensorToMesh(mesh_device)
        norm_ck = norm_compute_kernel_config(mesh_device.arch())

        def to_mesh(tensor: torch.Tensor, *, dtype, layout=ttnn.TILE_LAYOUT, mapper=None, memory_config=None):
            return ttnn.from_torch(
                tensor,
                device=mesh_device,
                layout=layout,
                dtype=dtype,
                memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper if mapper is not None else replicate,
            )

        def norm(name: str, eps: float) -> _FusedNorm:
            weight = _get_layer_tensor(state_dict, layer_idx, f"{name}.weight").to(torch.float32)
            folded = (1.0 + weight).to(torch.bfloat16)
            tile = to_mesh(folded.reshape(1, 1, 1, plan.hidden_size), dtype=ttnn.bfloat16)
            # The per-device quarter of the same weight, for the distributed form
            # of this norm (DEFAULT_PREFILL_FRACTURED_NORM).  Fractured over the
            # hidden dimension with the same mapper the column-parallel weights
            # use, so device d holds exactly the channels its reduce-scatter shard
            # carries.
            local = ttnn.from_torch(
                folded.reshape(1, 1, 1, plan.hidden_size),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
            )
            row_major = to_mesh(
                folded.reshape(1, 1, plan.hidden_size // TILE_SIZE, TILE_SIZE),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            fused = _FusedNorm(tile, row_major, eps, norm_ck)
            fused.local_weight = local
            return fused

        def linear_weight(suffix: str) -> torch.Tensor:
            return _get_layer_tensor(state_dict, layer_idx, suffix).to(torch.float32).transpose(-2, -1).contiguous()

        def projection(tensor: torch.Tensor, role: str, shard_dim: int) -> ttnn.Tensor:
            """One DRAM width-sharded per-device weight, shared by prefill and decode."""
            k, n = int(tensor.shape[-2]), int(tensor.shape[-1])
            local_k = k // tp if shard_dim == -2 else k
            local_n = n // tp if shard_dim == -1 else n
            return ttnn.from_torch(
                tensor.reshape(1, 1, k, n),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=precision.weight_dtype(role),
                memory_config=dram_sharded_weight_memcfg(local_k, local_n, mesh_device),
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=shard_dim),
            )

        wqkv = fused_qkv_weight(
            linear_weight("self_attn.q_proj.weight"),
            linear_weight("self_attn.k_proj.weight"),
            linear_weight("self_attn.v_proj.weight"),
            plan,
        )

        mlp = _MultichipMLP(
            gate=projection(
                _pad_per_device(linear_weight("mlp.gate_proj.weight"), -1, tp, plan.local_intermediate),
                "mlp_gate",
                -1,
            ),
            up=projection(
                _pad_per_device(linear_weight("mlp.up_proj.weight"), -1, tp, plan.local_intermediate),
                "mlp_up",
                -1,
            ),
            down=projection(
                _pad_per_device(linear_weight("mlp.down_proj.weight"), -2, tp, plan.local_intermediate),
                "mlp_down",
                -2,
            ),
            activation_dtype=precision.activation_dtype,
        )

        cache_shape = (max_num_blocks, plan.local_kv_heads, page_block_size, plan.head_dim)
        k_cache = to_mesh(torch.zeros(cache_shape), dtype=precision.kv_cache_dtype)
        v_cache = to_mesh(torch.zeros(cache_shape), dtype=precision.kv_cache_dtype)

        cos_cache = sin_cache = cos_cache_tile = sin_cache_tile = None
        if config.uses_rope:
            if rope_cache is not None:
                # A 52-layer stack has 39 sliding layers with the same theta, and the
                # four tables are 134 MB per layer at full context; building them per
                # layer would spend 5.2 GB of device DRAM on 39 copies of one tensor.
                # The stack owns them instead and hands the same four in every time --
                # see :func:`build_rope_cache`, which is what checks they match.
                cos_cache = rope_cache["cos"]
                sin_cache = rope_cache["sin"]
                cos_cache_tile = rope_cache["cos_tile"]
                sin_cache_tile = rope_cache["sin_tile"]
            else:
                cos, sin = _rope_cos_sin(max_seq_len, plan.head_dim, config.rope_theta)
                cos_cache = to_mesh(cos.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
                sin_cache = to_mesh(sin.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
                cos_cache_tile = to_mesh(
                    cos.to(torch.bfloat16).reshape(1, 1, max_seq_len, plan.head_dim), dtype=ttnn.bfloat16
                )
                sin_cache_tile = to_mesh(
                    sin.to(torch.bfloat16).reshape(1, 1, max_seq_len, plan.head_dim), dtype=ttnn.bfloat16
                )

        return cls(
            plan=plan,
            config=config,
            mesh_device=mesh_device,
            input_layernorm=norm("input_layernorm", config.rms_norm_eps),
            post_attention_layernorm=norm("post_attention_layernorm", config.post_norm_eps),
            pre_feedforward_layernorm=norm("pre_feedforward_layernorm", config.rms_norm_eps),
            post_feedforward_layernorm=norm("post_feedforward_layernorm", config.post_norm_eps),
            mlp=mlp,
            wqkv=projection(wqkv, "wqkv", -1),
            w_attn_gate=projection(linear_weight("self_attn.gate_proj.weight"), "attn_gate", -1),
            wo=projection(linear_weight("self_attn.o_proj.weight"), "o_proj", -2),
            k_cache=k_cache,
            v_cache=v_cache,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            cos_cache_tile=cos_cache_tile,
            sin_cache_tile=sin_cache_tile,
            activation_dtype=precision.activation_dtype,
            kv_cache_dtype=precision.kv_cache_dtype,
            precision=precision,
            decode_matmul=decode_matmul if decode_matmul is not None else multichip_decode_matmul(tp),
            boundary_cores=boundary_cores,
            decode_sdpa=decode_sdpa,
            ccl_dtype=ccl_dtype,
            prefill_ccl_dtype=prefill_ccl_dtype,
            decode_ccl_dtype=decode_ccl_dtype,
            ccl_mode=ccl_mode,
            prefill_ccl_mode=prefill_ccl_mode,
            decode_ccl_mode=decode_ccl_mode,
            ccl_rs_workers=ccl_rs_workers,
            prefill_ccl_rs_workers=prefill_ccl_rs_workers,
            decode_ccl_rs_workers=decode_ccl_rs_workers,
            ccl_impl=ccl_impl,
            prefill_ccl_impl=prefill_ccl_impl,
            decode_ccl_impl=decode_ccl_impl,
            ccl_ag_workers=ccl_ag_workers,
            prefill_ccl_ag_workers=prefill_ccl_ag_workers,
            decode_ccl_ag_workers=decode_ccl_ag_workers,
            ccl_persistent_buffers=ccl_persistent_buffers,
            ccl_chunks_per_sync=ccl_chunks_per_sync,
            ccl_buffers_per_channel=ccl_buffers_per_channel,
            ccl_num_links=ccl_num_links,
            ccl_ag_barrier=ccl_ag_barrier,
            prefill_fractured_norm=prefill_fractured_norm,
            prefill_fractured_norm_min_rows=prefill_fractured_norm_min_rows,
            sharded_decode_io=sharded_decode_io,
            **({} if decode_fused_activation is None else {"decode_fused_activation": decode_fused_activation}),
        )

    # ------------------------------------------------------- precision readback

    def precision_report(self) -> dict[str, Any]:
        """The single-chip report plus the collective payload dtypes.

        ``_row_parallel_dtype`` is called rather than restated, so the reported
        payload is the one the two row-parallel matmuls are actually asked for.
        """
        report = super().precision_report()
        report["ccl"] = {
            "prefill_payload_dtype": str(self._row_parallel_dtype("mlp_down", prefill=True)),
            "decode_payload_dtype": str(self._row_parallel_dtype("mlp_down", prefill=False)),
            "prefill_ccl_dtype_requested": str(self.prefill_ccl_dtype),
            "decode_ccl_dtype_requested": str(self.decode_ccl_dtype),
            "prefill_ccl_impl": self.prefill_ccl_impl,
            "decode_ccl_impl": self.decode_ccl_impl,
            "prefill_ccl_mode": self.prefill_ccl_mode,
            "decode_ccl_mode": self.decode_ccl_mode,
        }
        # The companion settings a precision artifact may carry
        # (``ALLOWED_DECODER_OVERRIDES``), read off the built layer so the
        # propagation check can see whether they landed.
        report["decoder_overrides"] = {
            "prefill_fractured_norm": self.prefill_fractured_norm,
            "prefill_ccl_mode": self.prefill_ccl_mode,
            "decode_ccl_mode": self.decode_ccl_mode,
            "prefill_ccl_impl": self.prefill_ccl_impl,
            "decode_ccl_impl": self.decode_ccl_impl,
        }
        return report

    # ------------------------------------------------------------- collectives

    def _row_parallel_dtype(self, role: str, *, prefill: bool) -> ttnn.DataType:
        """Output dtype of a row-parallel matmul, i.e. the CCL payload dtype.

        Asking the matmul for the payload dtype directly means the reduced
        precision costs **no** extra op: there is no typecast either side of the
        collective, and the residual add that consumes the reduced tensor takes
        its dtype from the (BF16) residual, so the layer's output contract is
        unchanged.  ``test_layer_output_dtype_is_the_activation_dtype`` pins that.
        """
        if role not in ROW_PARALLEL_ROLES:
            return self.activation_dtype
        override = self.prefill_ccl_dtype if prefill else self.decode_ccl_dtype
        return override if override is not None else self.activation_dtype

    def _all_reduce(self, tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig, *, prefill: bool) -> ttnn.Tensor:
        """Sum a full-width partial across the mesh, into ``memory_config``.

        One reduction per sublayer, and the residual stream stays replicated, so
        a stacked model needs no conversion between layers.  A sharded input is
        converted to interleaved inside the op and the result is written back
        into the requested config, so this can be handed the width-sharded L1
        decode layout directly and hands back the boundary layout.

        Which of the two forms is used is per mode, because they do not agree.  A
        ring all-reduce *is* a reduce-scatter plus an all-gather, so both forms
        move the same bytes and the only question is one fused dispatch against
        two.  At the 416 KiB decode payload the collective is latency-bound and the
        pair wins by 1.10 / 1.21 %; at the 57.9 MB prefill payload, with each payload's own
        ``num_workers_per_link``, the two are within 0.24 % and prefill keeps the
        single dispatch (``logs/layer_ab_reducer_final.log``,
        ``logs/fabric_packet_probe.log``).  See
        :data:`DEFAULT_DECODE_CCL_MODE`.
        """
        impl = self.prefill_ccl_impl if prefill else self.decode_ccl_impl
        if impl == "async":
            return self._all_reduce_async(tensor, memory_config, prefill=prefill)
        if (self.prefill_ccl_mode if prefill else self.decode_ccl_mode) == "rs_ag":
            scattered = ttnn.reduce_scatter(
                tensor,
                dim=3,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                # Pinned to DRAM rather than left to default from the input's
                # config: the decode input is the width-sharded L1 boundary
                # layout, and an L1 intermediate would be 425 KB of the pool for
                # a buffer that is written and read once.  (This was *not* what
                # fixed the L1 exhaustion this layer hit -- that was the
                # semaphores, see DEFAULT_L1_SMALL_SIZE -- but it is the right
                # place for it and it is measured in the shipped numbers.)
                intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=CCL_TOPOLOGY,
                num_workers_per_link=self.prefill_ccl_rs_workers if prefill else self.decode_ccl_rs_workers,
                # Only ``all_gather`` reads the mesh's L1_SMALL bank size on its
                # own; reduce_scatter needs to be told.  Without this the
                # per-program semaphore lands in the main L1 pool and fragments
                # it -- see DEFAULT_L1_SMALL_SIZE for the measurement.
                use_l1_small_for_semaphores=True,
            )
            ttnn.deallocate(tensor)
            reduced = ttnn.all_gather(scattered, dim=3, memory_config=memory_config)
            ttnn.deallocate(scattered)
            return reduced
        reduced = ttnn.all_reduce(
            tensor,
            memory_config=memory_config,
            topology=CCL_TOPOLOGY,
        )
        ttnn.deallocate(tensor)
        return reduced

    # --------------------------------------------- the async / persistent form

    def _ccl_semaphores(self) -> dict:
        """Global semaphores for the async collectives, shared across the mesh.

        Seven of them, created **once per mesh** and reused by every shape and
        every layer (``_CCL_SEMAPHORES``, dropped by ``close_multichip_mesh``), in
        the mesh's ``L1_SMALL`` region.  The composite wrappers instead create one
        per *program* and leave it there for the life of the program cache, which
        is what bounds a mesh to 24 distinct CCL programs
        (:data:`DEFAULT_L1_SMALL_SIZE`); these are a fixed 1,792 B rather than a
        per-program cost, but they are not free -- they are 7 fewer wrapper CCL
        program slots, which is why the suite's clearing floor had to move.  See
        ``doc/optimized_multichip_decoder/README.md``.

        They must be in ``L1_SMALL``.  Twelve of them in the main L1 pool -- the
        first implementation here -- sit at the top of it for the life of the mesh,
        and the decode step has only 7,296 B of headroom there; that made the next
        sharded-norm program fail with *"Statically allocated circular buffers in
        program 8764 clash with L1 buffers"* in 7 of the 104 acceptance tests.
        """
        key = id(self.mesh_device)
        sems = _CCL_SEMAPHORES.get(key)
        if sems is None:
            grid = self.mesh_device.compute_with_storage_grid_size()
            crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})

            def sem():
                return ttnn.create_global_semaphore(self.mesh_device, crs, 0, ttnn.BufferType.L1_SMALL)

            # Seven semaphores, not fourteen.  ``reduce_scatter_minimal_async``
            # takes three and ``all_gather_async`` two, and each is given its own
            # barrier; see DEFAULT_DECODE_CCL_IMPL for what the all-gather's
            # barrier is and is not known to buy.
            #
            # They are *not* double-buffered, because every collective in this
            # layer is ordered by data dependency rather than by convention: each
            # consumes the previous one's output.  That includes the two gathers
            # inside a fractured prefill norm, where the statistics gather's
            # result is the post_all_gather's input and the post_all_gather's
            # result is the payload gather's input.  A future edit that breaks
            # that chain has to double-buffer these.  The count matters because this region is
            # shared with the wrapper collectives' one-semaphore-per-program
            # allocations, and every 256 B here is one fewer distinct wrapper CCL
            # program the mesh can hold before they spill into the main L1 pool
            # and fragment it.  At fourteen, that test's prefill failed under
            # watcher with "Statically allocated circular buffers ... clash with
            # L1 buffers"; at seven it passes.
            sems = {
                "rs": [sem() for _ in range(3)],
                "ag": [sem() for _ in range(2)],
                "rs_barrier": sem(),
                "ag_barrier": sem(),
            }
            _CCL_SEMAPHORES[key] = sems
        return sems

    def _ccl_persistent_buffers(self, tensor: ttnn.Tensor, slot: int) -> list | None:
        """``persistent_output_buffers`` for ``reduce_scatter_minimal_async``.

        ``[intermediate, output, penult]`` -- the staging pair comes from the op's
        own sizing helper so the layouts are guaranteed to match the contiguous
        ring fast path, and the scattered output is this layer's.  Cached per
        ``(rows, width, dtype, slot)``; ``slot`` alternates so the layer's two
        reductions never share a buffer.

        Only the *scatter* half is made persistent.  The all-gather output is left
        to the op, because it is the tensor this method returns and callers
        deallocate it -- handing them a buffer this layer intends to reuse next
        token would be a use-after-free waiting for a scheduler change.

        Allocated with ``ttnn.zeros`` rather than ``ttnn.from_torch`` so that
        first use inside ``decode_forward`` is not a host round trip: the runtime
        fallback audit (``test_no_host_fallback_in_forward``) traps exactly that.
        """
        key = (int(tensor.shape[-2]), int(tensor.shape[-1]), tensor.dtype, slot)
        cache = self.__dict__.setdefault("_ccl_bufs", {})
        if key not in cache:
            staging = ttnn.experimental.reduce_scatter_minimal_async_create_intermediate_buffer(
                tensor, dim=3, topology=CCL_TOPOLOGY
            )
            shape = list(tensor.shape)
            shape[-1] //= self.plan.tp
            out = ttnn.zeros(
                shape,
                dtype=tensor.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            cache[key] = [staging[0], out, staging[1]]
            self._warm_persistent_buffers(tensor, cache[key], slot)
        return cache[key]

    def _warm_persistent_buffers(self, tensor: ttnn.Tensor, buffers: list, slot: int) -> None:
        """Run one throwaway reduce-scatter through freshly allocated staging.

        ``reduce_scatter_minimal_async_create_intermediate_buffer`` allocates the
        chunk-paged staging pair but does not initialise it, and the ring
        algorithm reads the penult intermediate before it has written it on the
        **first** invocation.  Measured
        (``bench/regression_bisect.py``, ``logs/regression_bisect.log``): with
        persistent buffers first used inside decode, decode step 0 scores PCC
        0.7395 (``full``) / 0.7749 (``sliding``) against the same layer without
        them, while steps 1-3 are bit-identical (1.000000) -- a first-use fault,
        not a numerical one.  It stayed hidden in the whole-layer A/B and in the
        HF-reference suite because both warm the layer before they measure or
        assert, and it surfaced in
        ``test_multichip_matches_single_chip[12345-4-full]``, which compares the
        very first decode step against a single-chip TTNN baseline at 0.999.

        Warming here turns every *real* first use into a second use.  It costs one
        collective per distinct ``(rows, width, dtype, slot)``, once, and the
        result is discarded.  A decode-only caller -- vLLM, or a generator that
        never prefills through this layer -- is exactly the case that would
        otherwise ship the corrupted step, so this cannot be left to the accident
        that prefill happens to run first.
        """
        sems = self._ccl_semaphores()
        # Not deallocated: the op writes into -- and returns -- ``buffers[1]``,
        # which is the persistent output this layer keeps.
        ttnn.experimental.reduce_scatter_minimal_async(
            tensor,
            persistent_output_buffers=buffers,
            dim=3,
            multi_device_global_semaphore=sems["rs"],
            barrier_semaphore=sems["rs_barrier"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=CCL_TOPOLOGY,
            num_workers_per_link=self.decode_ccl_rs_workers,
        )
        # Drain it.  The warm-up and the real call that follows share a semaphore
        # set, and without a barrier between them the real call can observe the
        # warm-up's counts: leaving this out left the first
        # ``sliding`` decode step wrong (``logs/regression_bisect_fixed.log``) where the
        # drained version is bit-identical.  Only ever reached outside trace
        # capture, because :meth:`_prewarm_decode_ccl_buffers` has already run.
        ttnn.synchronize_device(self.mesh_device)

    def _all_reduce_async(self, tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig, *, prefill: bool):
        """``reduce_scatter_minimal_async`` + ``all_gather_async``, tuned per half.

        Same algebra and the same bytes as :meth:`_all_reduce`'s ``rs_ag`` form.
        What it adds is a tuning surface on the *all-gather*, which
        ``ttnn.all_gather`` does not expose, and caller-owned semaphores and
        staging buffers.
        """
        sems = self._ccl_semaphores()
        slot = self._ccl_slot
        self._ccl_slot = (slot + 1) % 2
        buffers = self._ccl_persistent_buffers(tensor, slot) if self.ccl_persistent_buffers else None
        ag_workers = self.prefill_ccl_ag_workers if prefill else self.decode_ccl_ag_workers
        tune = {
            k: v
            for k, v in (
                ("chunks_per_sync", self.ccl_chunks_per_sync),
                ("num_buffers_per_channel", self.ccl_buffers_per_channel),
                ("num_links", self.ccl_num_links),
            )
            if v is not None
        }
        scattered = ttnn.experimental.reduce_scatter_minimal_async(
            tensor,
            persistent_output_buffers=buffers,
            dim=3,
            multi_device_global_semaphore=sems["rs"],
            barrier_semaphore=sems["rs_barrier"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=CCL_TOPOLOGY,
            num_workers_per_link=self.prefill_ccl_rs_workers if prefill else self.decode_ccl_rs_workers,
            **tune,
        )
        ttnn.deallocate(tensor)
        reduced = ttnn.experimental.all_gather_async(
            scattered,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=sems["ag"],
            barrier_semaphore=sems["ag_barrier"] if self.ccl_ag_barrier else None,
            memory_config=memory_config,
            topology=CCL_TOPOLOGY,
            **({} if ag_workers is None else {"num_workers_per_link": ag_workers}),
            **tune,
        )
        if buffers is None:
            ttnn.deallocate(scattered)
        return reduced

    # ------------------------------------------------------------ projections

    def _decode_projection(
        self,
        x_sharded: ttnn.Tensor,
        weight: ttnn.Tensor,
        *,
        role: str,
        rows: int,
        activation: ttnn.UnaryOpType | None = None,
    ) -> ttnn.Tensor:
        """As the single-chip version, plus the reduction a row-parallel role needs.

        The two row-parallel projections are the only place a collective appears
        in the decode step.  ``mlp_down``'s reduction also *replaces* the
        single-chip layer's second reshard: the collective writes its result
        straight into the boundary layout.
        """
        cores, in0_block_w = self.decode_matmul[role]
        n = int(weight.shape[-1])
        program_config = self._decode_program_config(rows, n, cores, in0_block_w, activation)
        out = ttnn.linear(
            x_sharded,
            weight,
            dtype=self._row_parallel_dtype(role, prefill=False),
            memory_config=self._sharded_memcfg(rows, n, cores),
            program_config=program_config,
            compute_kernel_config=self.decode_compute_kernel_config_by_role[role],
        )
        if role not in ROW_PARALLEL_ROLES:
            return out
        return self._all_reduce(out, self._sharded_memcfg(rows, n, self.boundary_cores), prefill=False)

    def _decode_program_config(self, rows, n, cores, in0_block_w, activation):
        return decode_matmul_program_config(
            rows, n, cores, in0_block_w, activation=activation if self.decode_fused_activation else None
        )

    def _prefill_projection(self, x: ttnn.Tensor, weight: ttnn.Tensor, *, role: str) -> ttnn.Tensor:
        """Three prefill kernels by row count, then the row-parallel reduction.

        The kernel selection is the single-chip one; it is re-spelled rather than
        delegated because the two geometry tables are per-stage instance state
        here (every per-device K and N moved, so several single-chip entries are
        illegal, not merely suboptimal) while the single-chip helper reads the
        module-level dicts.
        """
        rows = int(x.shape[-2])
        dtype = weight.dtype
        if rows == TILE_SIZE:
            cores, _ = self.decode_matmul[role]
            x_sharded = ttnn.interleaved_to_sharded(x, self._sharded_memcfg(rows, int(x.shape[-1]), cores))
            # ``_decode_projection`` already reduces a row-parallel role, and it
            # returns the boundary-grid layout, so the reduction is not repeated
            # below.
            out_sharded = self._decode_projection(x_sharded, weight, role=role, rows=rows)
            ttnn.deallocate(x_sharded)
            out = ttnn.sharded_to_interleaved(out_sharded, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(out_sharded)
            return out

        mcast = self._prefill_mcast2d_spec(role, rows, dtype)
        if mcast is not None:
            grid_y, in0_block_w = mcast
            out = ttnn.linear(
                x,
                weight,
                dtype=self._row_parallel_dtype(role, prefill=True),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.prefill_compute_kernel_config_by_role[role],
                program_config=prefill_mcast2d_program_config(
                    rows, int(weight.shape[-1]), grid_y, in0_block_w, self.dram_banks
                ),
            )
        else:
            blocks = self._minimal_matmul_blocks(role, rows, dtype)
            config = None
            if blocks is not None:
                m_block, k_block, n_block = blocks
                subblock_h, subblock_w = minimal_matmul_subblocks(
                    m_block, n_block, prefer_wide=int(weight.shape[-1]) >= rows
                )
                config = ttnn.MinimalMatmulConfig(
                    M_block_size=m_block,
                    K_block_size=k_block,
                    N_block_size=n_block,
                    subblock_h=subblock_h,
                    subblock_w=subblock_w,
                    compute_with_storage_grid_size=self.device_grid,
                )
            out = ttnn.experimental.minimal_matmul(
                x,
                weight,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=self._row_parallel_dtype(role, prefill=True),
                compute_kernel_config=self.prefill_compute_kernel_config_by_role[role],
                config=config,
            )
        if role not in ROW_PARALLEL_ROLES:
            return out
        if self.prefill_fractured_norm and rows > self.prefill_fractured_norm_min_rows:
            # Stop at the scatter and hand the 1664-wide partial to the norm that
            # is structurally guaranteed to consume it next; ``_prefill_norm``
            # finishes the reduction with the gather it would have done anyway.
            self._pending_scatter = self._prefill_reduce_scatter(out)
            return self._pending_scatter
        return self._all_reduce(out, ttnn.DRAM_MEMORY_CONFIG, prefill=True)

    def _prefill_reduce_scatter(self, tensor: ttnn.Tensor) -> ttnn.Tensor:
        sems = self._ccl_semaphores()
        scattered = ttnn.experimental.reduce_scatter_minimal_async(
            tensor,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=sems["rs"],
            barrier_semaphore=sems["rs_barrier"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=CCL_TOPOLOGY,
            num_workers_per_link=self.prefill_ccl_rs_workers,
        )
        ttnn.deallocate(tensor)
        return scattered

    def _prefill_norm(self, norm, x: ttnn.Tensor) -> ttnn.Tensor:
        """The inherited prefill norm, or the distributed one over a scattered partial.

        See :data:`DEFAULT_PREFILL_FRACTURED_NORM`.  ``_prefill_projection`` sets
        ``_pending_scatter`` when it hands back a reduce-scattered partial rather
        than a reduced tensor; this consumes it, normalises at the fractured width
        with the same arithmetic a full-width RMSNorm would use, and finishes the
        reduction with the all-gather the reduction owed anyway.

        Identity of the tensor is checked rather than its width, so a norm that is
        *not* the paired consumer cannot accidentally take the fractured path.
        """
        pending, self._pending_scatter = self._pending_scatter, None
        if pending is None or x is not pending:
            return super()._prefill_norm(norm, x)
        stats = ttnn.rms_norm_pre_all_gather(x, compute_kernel_config=norm.compute_kernel_config, dtype=ttnn.bfloat16)
        gathered_stats = self._prefill_all_gather(stats)
        ttnn.deallocate(stats)
        normed_local = ttnn.rms_norm_post_all_gather(
            x,
            gathered_stats,
            epsilon=norm.eps,
            weight=norm.local_weight,
            compute_kernel_config=norm.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(gathered_stats)
        ttnn.deallocate(x)
        out = self._prefill_all_gather(normed_local)
        ttnn.deallocate(normed_local)
        return out

    def _prefill_all_gather(self, tensor: ttnn.Tensor) -> ttnn.Tensor:
        sems = self._ccl_semaphores()
        return ttnn.experimental.all_gather_async(
            tensor,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=sems["ag"],
            barrier_semaphore=sems["ag_barrier"] if self.ccl_ag_barrier else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=CCL_TOPOLOGY,
            **({} if self.prefill_ccl_ag_workers is None else {"num_workers_per_link": self.prefill_ccl_ag_workers}),
        )

    def _prefill_mcast2d_spec(self, role: str, rows: int, dtype: ttnn.DataType) -> tuple[int, int] | None:
        for max_rows, spec in self.prefill_mcast2d.get((role, dtype), ()):
            if rows <= max_rows:
                return spec
        return None

    def _minimal_matmul_blocks(self, role: str, rows: int, dtype: ttnn.DataType) -> tuple[int, int, int] | None:
        for min_rows, blocks in self.prefill_minimal_blocks.get((role, dtype), ()):
            if rows >= min_rows:
                return blocks
        return None

    # ------------------------------------------------------------- properties

    @property
    def local_kv_cache_bytes(self) -> int:
        """Per-device K+V cache bytes, for the capability contract.

        Computed from the shipped cache dtype rather than read off the buffer:
        a BFLOAT8_B tile is 1024 mantissa bytes plus 64 exponent bytes, i.e.
        1.0625 B per element, which no single element size expresses.
        """
        per_element = {
            ttnn.bfloat16: 2.0,
            ttnn.float32: 4.0,
            ttnn.bfloat8_b: 1.0625,
            ttnn.bfloat4_b: 0.5625,
        }[self.kv_cache_dtype]
        elements = 1
        for dim in self.k_cache.shape:
            elements *= int(dim)
        return int(2 * elements * per_element)
