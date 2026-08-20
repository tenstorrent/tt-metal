# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Multichip TTNN decoder layer for Qwen3-Coder-30B-A3B-Instruct, 4 Blackhole dies.

Stages 03 and 04. Stage 04 optimized this file **in place**; the parallelisation
below is stage 03's and is unchanged, and what stage 04 changed is where the
activations live inside a layer:

* **Both residual RMSNorms are width-sharded over 8 cores** rather than running
  on one (``decode_residual_norm``). 19.82 -> 4.92 us each, and *more* accurate
  than the call they replace. The shard spec is deliberately
  ``_width_sharded_l1(2048)``, so the first norm's output feeds the qkv
  projection with no conversion at all.
* **The router projection reads that L1 shard** instead of DRAM-interleaved.
  24.62 -> **5.85 us** at the shipped 8-core norm shard (the same sweep's 4-core
  leg reads 4.30, but 4 cores is not what ships -- the norm shards over
  ``_NORM_SHARD_CORES = 8``), output **bit-identical**, which is what keeps the
  four dies agreeing on the top-8. In the layer it is row 182, 6.241 us.
* **The two collectives use caller-owned persistent buffers**
  (``_decode_ccl_buffers``), so nothing in the forward path allocates inside the
  trace.
* **Decode collectives use one ethernet link, not two** (``NUM_LINKS_DECODE``).
  Stage 03 measured this at 0.6% and kept 2 for a single code path; against the
  stage-04 layer it is **1.22%**, over six passes with the leg order alternating
  so that a position effect cannot be read as a link effect. Prefill keeps both.

Decode layer device time 414.661 -> 362.828 us on device 0 (1.143x); traced
decode at ctx 128, 0.4767 -> 0.4286 ms (1.112x), and 0.4700 -> 0.4282 measured
before and after in one process by ``probes/layer_levers.py``. The inter-layer contract is untouched: a layer
takes and returns a replicated ``[1, 1, B, 2048]`` bf16 TILE DRAM tensor with no
collective, gather or reshard between layers. Everything is in
``doc/optimized_multichip_decoder/``.

The single-chip baseline is ``optimized_decoder.py`` -- every program
config, dtype and fidelity constant it measured is imported rather than
re-derived, and the multichip path is the *same graph* with three changes:

1. **Attention is tensor-parallel by 4.** Each die owns 8 Q heads, 1 K head and
   1 V head, so ``wqkv`` is ``[2048, 1280]`` per die and ``wo`` is
   ``[1024, 2048]``. Both still satisfy ``_dram_sharded_ok`` (1280 = 5x256,
   1024 = 4x256), so stage 02's DRAM-sharded decode projections survive intact.
2. **Experts are expert-parallel by 4.** Each die owns 32 whole experts. M, N
   and K of both ``sparse_matmul`` calls are unchanged, which is the entire
   reason EP was chosen over splitting ``moe_intermediate`` -- see
   ``doc/multichip_decoder/mesh_plan.md`` section 2.
3. **Two all-reduces per layer**, one after ``wo`` and one after the expert
   reduce, so the residual stream stays a replicated ``[1, 1, B, 2048]``. That
   makes the layer's input contract identical to its output contract and lets 48
   of them stack with no boundary conversion.

Router, both RMSNorms and the residual are replicated. Routing needs a global
view of 128 logits for top-8, and a 128-wide ``topk`` occupies one core, so
there is nothing to fracture; the price is that 25.18% of the single-die decode
layer is replicated work -- 129.09 us of 512.65, the two residual RMSNorms
(40.23) plus the router block (88.86) -- which caps decode at 3.97x even at
infinite dies.

Mesh and fabric
---------------
1x4, ``FabricConfig.FABRIC_1D_RING`` before mesh open, ``Topology.Ring`` and
``num_links=2`` on every collective. All three are deliberate:
``tt_ccl.default_topology()`` returns ``Topology.Linear`` for a 4-device mesh
(it only special-cases 8-device T3K/Galaxy), and the cluster descriptor for this
host -- ``ClusterType.P300_X2``, two p300 boards -- shows a genuine closed
4-ring with two ethernet links on every hop. Measured cost of taking the default
instead: 1.21x at decode size, 1.79x at 2 MB
(``doc/multichip_decoder/mesh_plan.md`` section 5).

Both all-reduces, in both modes, are reduce-scatter followed by all-gather. The
design phase expected decode to want AG-of-partials instead, on a standalone
sweep that measured 19.96 us against RS+AG's 23.69 at ``[1,1,32,2048]``; the
shipped decode tensor has **one** logical row rather than 32, which makes
``ttnn.sum`` pull a ``FillPad``, and measured on the real layer the order
reverses. See ``all_reduce`` for the profile rows and the A/B.

The ``nnz`` contract, which is a device hang if you get it wrong
--------------------------------------------------------------
``ttnn.sparse_matmul`` bakes ``nnz`` into the kernel as a compile-time arg and
requires ``count_nonzero(sparsity) == nnz`` exactly;
``sparse_matmul_device_operation.cpp:205-211`` says a mismatch *deadlocks the
device* (tt-metal #45943), silently unless the watcher is on.

A TTNN mesh op is SPMD -- one program, one ``nnz``, four dies. Under EP the
number of locally-live experts is the number of the global top-8 that landed in
this die's 32-expert window: data-dependent, different on every die, anywhere in
0..8. There is no single correct value, so **decode must pass ``nnz=None``**,
which switches the sender to reading the sparsity page at runtime. That costs
0.79 us per slot per matmul, measured at the shipped shapes: the pair costs
158.01 us dynamic against 107.73 with an exact nnz, **1.47x**
(``probes/nnz_cost_probe.py``). That is 50 us over 32 slots -- affordable only
because EP already cut E from 128 to 32, where the same rate would cost 200. The
two decisions are coupled.

**Prefill keeps an exact ``nnz``.** Its sparsity is per 32-token tile and with
32 tokens x top-8 = 256 selections over 128 experts essentially every expert is
live, so the shipped path uses an all-ones mask; under EP that means all 32
local experts are live on every die, deterministically, and
``nnz = 32 * group_size`` is exact and identical across dies.

Rejected alternatives, with the measurement, are in
``doc/multichip_decoder/mesh_plan.md`` section 6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import torch

import ttnn
from models.common.modules.tt_ccl import TT_CCL

from .functional_decoder import (
    AttentionConfig,
    AttentionWeights,
    DecoderLayerConfig,
    KVCache,
    MoEConfig,
    apply_rope_llama,
    attention_prefill,
    rope_transformation_matrix,
)

# The four precision constants this module used to import from here are gone:
# every one of them is now read off the ``PrecisionConfig`` threaded through the
# functions below, so importing the import-time default would have been the bug
# this stage exists to remove. ``tt/precision.py`` holds the values.
from .optimized_decoder import (
    _DRAM_BANKS,
    OptimizedWeights,
    _attention_compute_kernel_config,
    _bank_row,
    _dram_sharded_ok,
    _expert_compute_kernel_config,
    _ones_column,
    _tuned_sparse_matmul_config,
    _width_sharded_l1,
    attention_decode_optimized,
    moe_prefill_optimized,
)
from .precision import DEFAULT_PRECISION, PrecisionConfig  # noqa: F401  (re-exported)
from .weight_mapping import hf_to_meta_channels, permute_head_vector_to_meta, permute_wqkv_to_meta

# The target mesh. This module deliberately supports exactly one shape: the
# goal is the best use of *this* machine, and every constant below -- the head
# split, the 32-expert window, the ring topology, the two links -- is chosen
# against it. A 1x2 or 1x8 mesh would want different answers, not a scaled
# version of these ones.
MESH_SHAPE = (1, 4)
NUM_DEVICES = 4

# Ring, not Linear. See the module docstring; this must be passed explicitly
# because tt_ccl.default_topology() returns Linear for a 4-device mesh.
TOPOLOGY = ttnn.Topology.Ring

# Both ethernet links on every hop are used **in prefill**, where the payload is
# large enough to be bandwidth-bound: 1.84x at 2 MB.
NUM_LINKS = 2

# Decode uses **one**. Stage 03 measured 1 link at 0.4738 ms against 2 links'
# 0.4766, called 0.6% noise-level, and kept 2 for a single code path. Against
# the stage-04 layer -- where the collectives are a larger share because
# everything around them got smaller, and where they no longer allocate -- the
# gap is 1.22% and output is bit-identical (``probes/links_probe.py``, six
# passes with the **leg order alternating**, so that a position effect cannot be
# read as a link effect):
#
#     posA  2 links 0.4342  0.4341  0.4340     1 link 0.4290  0.4288  0.4286
#     posB  2 links 0.4341  0.4337  0.4339     1 link 0.4291  0.4283  0.4287
#
#     mean  2 links 0.43400      1 link 0.42875      1.22%
#
# Each configuration reads the same at both positions, which is what rules out
# the alternative explanation -- that the leg running first in a pass is simply
# slower. That control was added because review found ``_links`` had stopped
# honouring an explicit ``num_links=2``, leaving the probe unable to tell its
# own legs apart; the figure survived the repair, its reproducibility did not
# and now does. 5.25 us on the layer against a leg-against-itself spread of
# 0.5-0.8 us. A decode
# collective moves 128 KB per die and is latency-bound, so the second link buys
# no bandwidth and costs the split and merge. ``all_reduce`` branches on the
# same ``S <= 32`` test ``_decode_ccl_buffers`` uses, so prefill keeps both.
NUM_LINKS_DECODE = 1


def _links(x: ttnn.Tensor, ctx: "MeshContext") -> int:
    """``ctx.num_links`` for prefill, ``ctx.decode_num_links`` for decode.

    The two counts are **separate fields** rather than one field plus a
    "differs from the default means override" test. That test was the first
    spelling here and it is not expressible: a caller asking explicitly for
    ``num_links=2`` at decode passes ``ctx.num_links == NUM_LINKS``, which the
    test read as "no override" and silently gave 1 link. ``links_probe.py``
    builds its two-link leg exactly that way, so the probe that established
    ``NUM_LINKS_DECODE`` could not have been re-run against it -- review caught
    this. Two fields make each mode's count independently settable and the
    probe's legs actually different.
    """
    return ctx.decode_num_links if int(x.shape[-2]) <= 32 else ctx.num_links


# Decode's two expert intermediates under EP:
#
#     batch * 32 experts * 32 padded rows * (2*768 + 2048) cols * 2 B
#         = batch * 7,340,032 B = batch * 7.34 MB
#
# a quarter of the single-die figure, because EP fractures the expert dimension
# these tensors are indexed by. Stage 02's 40 MB threshold sat between its batch
# 1 (29.4 MB) and batch 2 (58.8 MB) and its own comment calls it "asserted, not
# measured"; inherited here it would have admitted batch 5 by accident. Swept
# instead (``probes/l1_budget_probe.py``, eager, ms per decode step):
#
#     batch   intermediates      L1        DRAM
#         1          7.34 MB   1.7336    1.6924
#         2         14.68      1.7816    1.7930
#         4         29.36      1.8887    1.9753
#         8         58.72      2.4969    2.9082
#        16        117.44      3.3040    4.0386
#        32        234.88   allocator refuses  (bank_manager.cpp:462)
#
# L1 wins from batch 2 to 16 and stops being allocatable at 32, so the threshold
# goes between 117.44 and 234.88 MB. The batch-1 row of that sweep reads the
# other way, but it is an *eager* measurement where host dispatch is most of the
# 1.7 ms; the warmed traced A/B that decides the shipped configuration says L1,
# clearly -- **0.4766 ms against DRAM's 0.5128, 7.6%**
# (``probes/decode_levers.py``). Batch 1 is also the latency target.
_DECODE_EXPERT_L1_BUDGET_BYTES = 128 * 1024 * 1024


def _decode_expert_memory_config(batch: int, local_moe: MoEConfig) -> ttnn.MemoryConfig:
    padded_rows = batch * local_moe.num_experts * 32
    nbytes = padded_rows * (2 * local_moe.moe_intermediate_size + local_moe.hidden_size) * 2
    return ttnn.L1_MEMORY_CONFIG if nbytes <= _DECODE_EXPERT_L1_BUDGET_BYTES else ttnn.DRAM_MEMORY_CONFIG


# --- Meta-ordered rotary for decode (stage 04) --------------------------------


def _meta_rope(ctx: MeshContext, cos_cache: ttnn.Tensor, sin_cache: ttnn.Tensor, head_dim: int):
    """Return a ``rope(t, cos, sin, token_index)`` callable using the llama op.

    **Measured, and not adopted.** Kept runnable rather than deleted, on the
    same principle as ``router_forward_threshold``: the finding is the useful
    part, and a future stage that changes prefill should not have to rediscover
    it. Nothing on the shipped path calls this --
    ``decoder_layer_decode_multichip`` ships the HF op, and
    ``upload_multichip_weights`` builds the Meta weights only under
    ``meta_rope=True``, so the shipped upload pays no DRAM for it.

    ``rotary_embedding_llama`` costs **1.26 us** against the shipped HF op's
    3.84 at the per-die decode shape, with ``max|diff|`` exactly 0.0 and PCC
    1.0000000 (``probes/rope_probe.py``). Both run on one core: the llama
    decode factory shards over *batch*, not heads, so at batch 1 none of the
    3.05x is parallelism -- it is the activation living in L1 and a kernel that
    multiplies by a resident 32x32 matrix instead of gathering a cos/sin row out
    of a DRAM cache. Same lever as the router projection, different op.

    Two things are hoisted out of the forward path, both on the first (eager)
    call and cached on ``ctx``:

    * the **Meta cos/sin** for this ``token_index``, read off the HF device
      cache once, permuted on the host and uploaded already sharded. This hoist
      is what makes *this* wiring unreplayable, and it is a property of the
      wiring rather than of the op: ``rotary_embedding_llama`` takes cos/sin as
      tensors and no position argument at all
      (``rotary_embedding_llama_nanobind.cpp:38-44``), so it can be driven from
      a position tensor inside a trace. The shipped
      ``rotary_embedding(..., token_index)`` genuinely cannot, which is why
      stage 05 moved decode to ``rotary_embedding_hf``. An earlier revision of
      this docstring claimed neither spelling could; that was wrong.
    * the **transformation matrix**, which is position-independent.

    The Meta *channel order* is not established here at all: it is a property of
    ``ctx``-independent weights, applied once by
    ``weight_mapping.permute_wqkv_to_meta`` at upload.

    **Why it is not adopted.** RoPE runs *before* K is written to the cache, so
    the cache inherits the rotary's channel convention. Prefill is untouched by
    this lever and writes HF-ordered keys; a Meta-ordered decode Q then scores
    against them, and the dot products are meaningless.
    ``probes/rope_layer_probe.py``:

        fresh KV cache          PCC 0.9999697    the rotary itself is right
        prefill-primed cache    PCC 0.1932974    the cache convention is not

    The op-level probe looked clean precisely because its cache was fresh. So
    the lever is not decode-local: adopting it means adopting the llama rotary
    in **prefill** as well, permuting the interleaved ``wqkv`` prefill copy, and
    changing the KV cache's channel convention -- which
    ``test_per_die_kv_heads_stitched`` compares against a single-chip cache and
    which ``doc/context_contract.json`` describes. That is a whole-layer change,
    not the in-place decode optimization this stage is.

    A second cost, smaller and independent: the qkv weight dtype
    (``PrecisionConfig.attention_qkv_dtype``) is ``bfloat8_b`` by default, and
    bfloat8_b's 16-element blocks share an exponent, so permuting
    channels **regroups the blocks** and requantizes. The two paths therefore
    are not bit-identical in the layer even where the ops are -- attention out
    ``max|diff|`` 1.221e-04 on a fresh cache, and the K cache differs by
    3.125e-01 after permuting back. "Bit-identical" is a property of the op at
    fixed input, not of the layer at permuted weights.
    """
    st = ctx.rope_meta

    def trans_mat(batch: int):
        # One 32x32 copy **per batch core**, because the decode factory shards
        # over batch: at batch 1 that is a single tile on a single core, at
        # batch 32 it is 32 of them. Keyed by batch for that reason.
        t = st.get(("tm", batch))
        if t is None:
            t = st[("tm", batch)] = ttnn.from_torch(
                rope_transformation_matrix().repeat(1, 1, batch, 1),
                device=ctx.mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=_head_shard(32, 32, batch),
                mesh_mapper=ttnn.ReplicateTensorToMesh(ctx.mesh),
            )
        return t

    def rope(t: ttnn.Tensor, _cos, _sin, token_index: int) -> ttnn.Tensor:
        batch = int(t.shape[1])
        key = (int(token_index), batch, head_dim)
        pair = st.get(key)
        if pair is None:
            if "host" not in st:
                # Read the HF cos/sin cache back once and permute on the host.
                # Replicated, so die 0's copy is the whole tensor.
                comp = ttnn.ConcatMeshToTensor(ctx.mesh, dim=0)
                st["host"] = (
                    ttnn.to_torch(cos_cache, mesh_composer=comp)[:1].float(),
                    ttnn.to_torch(sin_cache, mesh_composer=comp)[:1].float(),
                )
            perm = hf_to_meta_channels(head_dim)
            mem = _head_shard(32, head_dim, batch)
            up = []
            for c in st["host"]:
                row = c[:, :, token_index : token_index + 1, :]
                row = row.expand(1, 1, 32, head_dim).contiguous()[..., perm]
                up.append(
                    ttnn.from_torch(
                        row.expand(1, batch, 32, head_dim).contiguous(),
                        device=ctx.mesh,
                        layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat16,
                        memory_config=mem,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(ctx.mesh),
                    )
                )
            pair = st[key] = tuple(up)
        sharded = ttnn.to_memory_config(t, _head_shard(32, head_dim, int(t.shape[1])))
        out = apply_rope_llama(sharded, pair[0], pair[1], trans_mat(batch))
        ttnn.deallocate(sharded)
        return out

    return rope


def _head_shard(rows: int, cols: int, batch: int) -> ttnn.MemoryConfig:
    """The height-sharded L1 config ``nlp_create_qkv_heads_decode`` emits and
    ``rotary_embedding_llama``'s decode factory requires: one core per user,
    each holding that user's whole ``[32 padded heads, head_dim]`` block."""
    gx = min(batch, 8)
    while batch % gx:
        gx -= 1
    gy = batch // gx
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))}),
            [rows, cols],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


# --- mesh plumbing ------------------------------------------------------------


@dataclass
class MeshContext:
    """Mesh, CCL semaphores and the collective parameters, owned explicitly.

    ``TT_CCL`` is instantiated directly rather than through
    ``tt_ccl.get_tt_ccl()``: that helper caches by ``mesh_device.id()`` in a
    module-global dict, and the pytest ``mesh_device`` fixture is
    function-scoped, so a later mesh can be handed a recycled id and inherit
    semaphores belonging to a closed device. Ownership here is per-caller and
    dies with the caller.

    Global semaphores are hardware resources allocated at construction time,
    which is also what makes this trace-safe: nothing in the forward path
    allocates one.
    """

    mesh: ttnn.MeshDevice
    ccl: TT_CCL
    num_devices: int = NUM_DEVICES
    num_links: int = NUM_LINKS
    topology: ttnn.Topology = TOPOLOGY
    # Links for a *decode* collective, separately settable. See ``_links``.
    decode_num_links: int = NUM_LINKS_DECODE
    # Stage 04. Meta-ordered rotary state for the decode path: the 32x32
    # transformation matrix, the host-side Meta cos/sin caches, and the
    # per-position sharded cos/sin pair. Keyed by ``token_index``, which is a
    # Python int here exactly as it is for the shipped HF op -- the rotary
    # position is baked into a traced program either way, so the gather is
    # hoisted out of the forward path rather than run per token. Allocated on a
    # miss, so the first call at each position must be eager, which is the same
    # discipline ``ccl_buffers`` below already imposes. See ``_meta_rope``.
    rope_meta: dict = field(default_factory=dict)
    # Stage 04. Persistent collective buffers, keyed by (logical shape, padded
    # shape, dtype), so that neither the reduce-scatter nor the all-gather
    # allocates inside the trace. See ``_decode_ccl_buffers``. Owned by the
    # context and therefore by the caller, exactly like the semaphores above.
    ccl_buffers: dict = field(default_factory=dict)


def mesh_context(mesh_device) -> MeshContext:
    """Build the CCL context for the 4-die mesh, asserting the shape."""
    n = mesh_device.get_num_devices()
    assert n == NUM_DEVICES, (
        f"multichip_decoder targets exactly {NUM_DEVICES} dies (the full P300_X2 mesh); got {n}. "
        "Smaller meshes are out of scope by design -- the head split, expert window and ring "
        "topology are all chosen against the 4-die shape."
    )
    return MeshContext(mesh=mesh_device, ccl=TT_CCL(mesh_device))


def all_reduce(x: ttnn.Tensor, ctx: MeshContext, precision: PrecisionConfig = DEFAULT_PRECISION) -> ttnn.Tensor:
    """All-reduce a ``[1, 1, ., H]`` partial as reduce-scatter then all-gather.

    **One spelling for both modes, and that is a change from the plan.**
    ``mesh_plan.md`` §5 chose AG-of-partials-plus-local-sum for decode on a
    standalone sweep that measured 19.96 us against RS+AG's 23.69 at
    ``[1,1,32,2048]``. That sweep used the wrong shape. The shipped decode tensor
    is ``[1,1,1,2048]`` -- **one** logical row padded to a tile -- and
    ``ttnn.sum`` over a tensor whose last two dims are not both tile-aligned
    drags a ``FillPad`` behind it (``fill_pad.cpp:17-24``), which is precisely
    the hazard stage 02 removed from the router. Read off
    ``ops_perf_multichip_decode_agsum.csv`` -- a profile of this layer with the
    plan's spelling, kept precisely because the shipped path no longer produces
    those rows (``probes/profile_layer.py decode-agsum``) -- the local sum is not
    one op but four:

        AllGatherAsync 22.31 us + FillPad 5.89 + FastReduceNC 2.44 + Slice 1.32
            = 31.96 us   (attention all-reduce)
        AllGatherAsync 18.65 + FillPad 5.64 + FastReduceNC 2.43 + Slice 1.31
            = 28.04 us   (expert all-reduce)

    against the 19.96 the probe promised for each. Measured on the whole traced
    layer at ctx 128, median of 100 (``probes/allreduce_ab.py``):

        AG(dim 0) + ttnn.sum   0.4801 ms
        reduce-scatter + all-gather   **0.4760 ms**   <- adopted

    0.9%, which is small -- but it is also three fewer ops, one code path
    instead of two, and it is the direction the standalone probe got backwards.
    A third leg that kept the single collective and reshaped the logical shape
    up to the padded 32 rows to dodge the ``FillPad`` **did not run**:
    ``reshape_common.cpp:50`` rejects it, ``new_volume == old_volume``.

    Prefill was RS+AG already and for the reason that still holds: 76.85 us
    against AG-of-partials' 121.72 at ``[1,1,512,2048]``, because past ~128 KB
    per device the collective is bandwidth-bound and RS+AG moves a quarter of
    the bytes on each of its two hops.

    The scatter axis is dim 3 (hidden, 2048), which is independent of the
    sequence length -- that is what keeps non-aligned S working through the
    collective without any padding of its own.

    ``precision.ccl_dtype`` is ``None`` on the shipped path, which means "run
    the collective at whatever dtype the partial arrives in" -- no cast, no
    extra op, the behaviour every stage-02..06 number was measured at. A named
    dtype casts in before the reduce-scatter and back out after the all-gather,
    so a sweep can price a narrower wire without touching the arithmetic that
    feeds it. The cast is deliberately *outside* the buffer cache key's reach
    only in the sense that the cache keys on ``x.dtype`` already -- casting
    first means the cached buffers are allocated at the wire dtype, which is the
    point.
    """
    # The cast allocates a *new* tensor and leaves ``x`` alone: every caller
    # deallocates the partial it passed in, so freeing it here would be a double
    # free the moment ``ccl_dtype`` was set.
    wire_dtype = precision.ccl_dtype
    restore_dtype = None
    cast_in = None
    if wire_dtype is not None and x.dtype != wire_dtype:
        restore_dtype = x.dtype
        cast_in = ttnn.typecast(x, wire_dtype)
        x = cast_in
    bufs = _decode_ccl_buffers(x, ctx)
    num_links = _links(x, ctx)
    scattered = ttnn.experimental.reduce_scatter_minimal_async(
        x,
        persistent_output_buffers=None if bufs is None else bufs[0],
        dim=3,
        multi_device_global_semaphore=ctx.ccl.get_and_cycle_rs_semaphore_handles(),
        barrier_semaphore=ctx.ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=num_links,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ctx.topology,
    )
    gathered = ttnn.experimental.all_gather_async(
        scattered,
        persistent_output_buffer=None if bufs is None else bufs[1],
        dim=3,
        multi_device_global_semaphore=ctx.ccl.get_and_cycle_ag_semaphore_handles(),
        barrier_semaphore=ctx.ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=num_links,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ctx.topology,
    )
    if bufs is None:
        ttnn.deallocate(scattered)
        out = gathered
    else:
        # ``gathered`` *is* the persistent buffer, which the caller is about to
        # deallocate. Hand back a copy so the buffer survives the next token.
        out = ttnn.clone(gathered, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if restore_dtype is not None:
        cast_out = ttnn.typecast(out, restore_dtype)
        ttnn.deallocate(out)
        out = cast_out
    if cast_in is not None:
        ttnn.deallocate(cast_in)
    return out


def _decode_ccl_buffers(x: ttnn.Tensor, ctx: MeshContext):
    """``([rs intermediate, rs output, penult], ag output)`` for a decode-shaped
    ``x``, allocated once per (logical shape, padded shape, dtype) and cached on
    the context.

    ``None`` for anything taller than one 32-row tile: prefill runs at a
    different ``S`` on every call, so caching there would allocate a set per
    sequence length for a lever worth 0.2% at decode and nothing measurable at
    prefill (the prefill collective is bandwidth-bound, not allocation-bound).

    **Allocated on the first call at each shape, so that call must be eager.**
    ``ttnn.from_torch`` inside ``begin_trace_capture`` raises "Writes are not
    supported during trace capture" and leaves the trace open, which is a hung
    mesh and a ``tt-smi -r``. Every harness here runs the layer once before
    capturing, which is also what the semaphores in ``MeshContext`` already
    require; the constraint is not new, only wider.

    All 48 layers of the stacked model share the cache, and so does every token.
    That is safe because the trace serialises the collectives and each result is
    cloned out before the next one starts -- but it is exactly the property a
    future change has to preserve, so it is exercised by
    ``test_multichip_decode_20_steps_deterministic`` running 20 tokens through
    the same buffers and by ``test_two_layers_stacked``.

    The layer's *two* all-reduces do **not** share a set; see the key below.

    Measured: 0.4343 / 0.4337 ms against the allocating path's 0.4348 / 0.4346,
    over two interleaved passes (``probes/layer_levers3.py``), and 0.4335 /
    0.4333 against 0.4348 / 0.4346 in ``probes/layer_levers2.py``.
    """
    if int(x.shape[-2]) > 32:
        return None
    # The key must carry the **logical** shape, not just the padded one.
    #
    # The layer's two all-reduces are both [1,1,batch,2048] and *do* share one
    # set, correctly: the attention partial is ``batch`` rows because
    # ``_concat_heads_decode`` slices the padded tile back before ``wo``, and the
    # expert partial is ``batch`` by construction. What collides is the priming
    # prefill: at ``S <= 32`` it takes this branch too, and a 32-token prefill
    # and a decode at ``batch < 32`` have the same *padded* shape, one 32-row
    # tile. A persistent output buffer imposes *its* logical shape on the op's
    # result, so keyed on the padded shape alone the decode layer inherited the
    # prefill's 32 rows and silently returned a 32-row tensor. Not hypothetical
    # -- six decode tests caught it (``work_log.md`` section 5).
    key = (tuple(int(v) for v in x.shape), tuple(int(v) for v in x.padded_shape), str(x.dtype))
    entry = ctx.ccl_buffers.get(key)
    if entry is None:
        interm, penult = ttnn.experimental.reduce_scatter_minimal_async_create_intermediate_buffer(
            x, dim=3, topology=ctx.topology, cluster_axis=None
        )
        shape = list(x.shape)
        shape[3] //= ctx.num_devices

        def zeros(s):
            return ttnn.from_torch(
                torch.zeros(s),
                device=ctx.mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=x.dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(ctx.mesh),
            )

        rs_bufs = [interm, zeros(shape)] + ([penult] if penult is not None else [])
        entry = (rs_bufs, zeros(list(x.shape)))
        ctx.ccl_buffers[key] = entry
    return entry


# Kept as names so callers read as prefill/decode; both are the same collective.
all_reduce_prefill = all_reduce
all_reduce_decode = all_reduce


# --- configuration ------------------------------------------------------------


@dataclass(frozen=True)
class MeshDecoderConfig:
    """The global layer config plus the per-die views the kernels actually see.

    ``local_attention`` carries 8 Q heads and 1 KV head; ``local_moe`` carries
    32 experts. Every op below is handed a *local* config, which is what makes
    the multichip layer literally the single-chip code at a quarter of the
    shape rather than a reimplementation of it.
    """

    global_config: DecoderLayerConfig
    local_attention: AttentionConfig
    local_moe: MoEConfig
    num_devices: int = NUM_DEVICES

    @classmethod
    def from_hf(cls, hf_config, num_devices: int = NUM_DEVICES) -> "MeshDecoderConfig":
        return cls.from_global(DecoderLayerConfig.from_hf(hf_config), num_devices)

    @classmethod
    def from_global(cls, config: DecoderLayerConfig, num_devices: int = NUM_DEVICES) -> "MeshDecoderConfig":
        a, m = config.attention, config.moe
        assert a.num_attention_heads % num_devices == 0, f"{a.num_attention_heads} Q heads / {num_devices}"
        assert a.num_key_value_heads % num_devices == 0, (
            f"{a.num_key_value_heads} KV heads / {num_devices} -- this is the hard cap on the TP factor; "
            "TP=8 would need KV-head replication and would also take wqkv's N to 640, which is not a "
            "multiple of 8 banks x 32 = 256, so the DRAM-sharded attention path would silently vanish"
        )
        assert m.num_experts % num_devices == 0, f"{m.num_experts} experts / {num_devices}"
        local_attention = AttentionConfig(
            hidden_size=a.hidden_size,
            num_attention_heads=a.num_attention_heads // num_devices,
            num_key_value_heads=a.num_key_value_heads // num_devices,
            head_dim=a.head_dim,
            rms_norm_eps=a.rms_norm_eps,
        )
        local_moe = MoEConfig(
            hidden_size=m.hidden_size,
            num_experts=m.num_experts // num_devices,
            num_experts_per_tok=m.num_experts_per_tok,
            moe_intermediate_size=m.moe_intermediate_size,
            norm_topk_prob=m.norm_topk_prob,
        )
        return cls(
            global_config=config,
            local_attention=local_attention,
            local_moe=local_moe,
            num_devices=num_devices,
        )


# --- weights ------------------------------------------------------------------


def head_interleaved_wqkv(wqkv: torch.Tensor, config: AttentionConfig, num_devices: int) -> torch.Tensor:
    """Permute the fused QKV columns so a contiguous 4-way split is the TP split.

    **This is the one weight transform that a naive port gets wrong.** The
    checkpoint's fused weight is ``[Wq(4096) | Wk(512) | Wv(512)]``, so a plain
    ``ShardTensorToMesh(dim=-1)`` hands die 0 nothing but Q heads and die 3
    nothing but K and V. Die *d* must instead own Q heads ``8d..8d+7``, K head
    ``d`` and V head ``d``, laid out as ``[Q_local | K_local | V_local]`` --
    which is what ``nlp_create_qkv_heads_decode(num_heads=8, num_kv_heads=1)``
    reads on the other side.

    Rebuilding the tensor in that order here means the runtime split stays a
    plain contiguous shard, and the failure mode -- which produces no shape
    error, only a wrong answer -- cannot come back through a different mapper.

    ``wqkv`` is ``[..., hidden, (n_heads + 2*n_kv) * head_dim]``; the return has
    the same shape with its last dim permuted.
    """
    n_heads, n_kv, hd = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    q_per, kv_per = n_heads // num_devices, n_kv // num_devices
    q_end, k_end = n_heads * hd, n_heads * hd + n_kv * hd

    cols = []
    for d in range(num_devices):
        cols.append(wqkv[..., d * q_per * hd : (d + 1) * q_per * hd])
        cols.append(wqkv[..., q_end + d * kv_per * hd : q_end + (d + 1) * kv_per * hd])
        cols.append(wqkv[..., k_end + d * kv_per * hd : k_end + (d + 1) * kv_per * hd])
    out = torch.cat(cols, dim=-1)
    assert out.shape == wqkv.shape
    return out


@dataclass
class MultichipWeights:
    """Everything one multichip decoder layer reads.

    ``experts`` is an ``OptimizedWeights`` whose tensors are mesh-sharded: the
    two expert weights on the expert dimension, ``wqkv``/``wo`` on the
    head-interleaved column and Q-head row split respectively. The dataclass is
    reused unchanged so ``attention_decode_optimized`` and
    ``moe_prefill_optimized`` can be called directly.

    ``expert_window`` is the only genuinely *device-varying* constant in the
    layer: a one-hot ``[1, 1, 128, 32]`` matrix, different on every die, that
    slices this die's 32-expert column window out of the replicated dense
    routing vector. See ``router_forward_multichip``.
    """

    input_layernorm: ttnn.Tensor
    post_attention_layernorm: ttnn.Tensor
    router: ttnn.Tensor
    expert_window: ttnn.Tensor
    experts: OptimizedWeights
    # Stage 04. ``ttnn.rms_norm``'s sharded program factory reads its weight as a
    # ROW_MAJOR ``[1, 1, dim/32, 32]`` tensor rather than the tiled ``[1,1,1,dim]``
    # the interleaved factory takes, so decode carries a second copy of each of
    # the two residual norm vectors. 4 KB each against the layer's 95.5 MB.
    input_layernorm_rm: ttnn.Tensor | None = None
    post_attention_layernorm_rm: ttnn.Tensor | None = None
    # Stage 04. The same ``OptimizedWeights`` with the Q and K channels of
    # ``wqkv_decode`` -- and of ``q_norm``/``k_norm``, which Qwen3 applies
    # between the head split and RoPE -- reordered to the Meta convention
    # ``rotary_embedding_llama`` requires. Decode-only: ``wqkv`` (prefill's
    # interleaved copy), ``wo`` and the expert weights are the *same objects*,
    # not copies, so this costs one extra DRAM-sharded qkv (11.14 MB/4 per die)
    # and two 128-element vectors, and prefill cannot reach it.
    experts_meta: OptimizedWeights | None = None


# SDPA-decode's tree reduction is capped at 6 rounds, i.e. 64 cores per KV head
# (``sdpa_decode_program_factory.cpp:245``). With no program config the op sets
# ``max_cores_per_head = num_cores_available``, so at TP=4 -- one KV head per die
# -- batch 1 asks for all 110 worker cores on that single head and the op raises
#
#     Tree reduction max 6 rounds (64 cores/head), got 110 cores/head
#
# This is a *new* failure created by the head split: at the single-chip 4 KV
# heads the same arithmetic gives 27 cores/head. It only bites the contiguous
# cache path -- the paged one runs at the default -- and only at small batch,
# because ``num_cores_per_head`` divides by the batch. Capping the per-head core
# budget at the op's own limit fixes it without giving up any parallelism the op
# would have been allowed to use.
_SDPA_MAX_CORES_PER_HEAD = 64

# The **paged** path -- the one the full model actually runs -- had no program
# config at all until stage 06, because the cap above was added to clear a
# ``TT_FATAL`` the paged path never raised. Running at the op default is not
# free: with no config the op picks its own ``k_chunk_size`` and core split, and
# the result is a decode cost that is **linear in ``cur_pos``** rather than
# flat. Measured at the shipped per-die decode shapes -- 8 Q heads, 1 KV head,
# head_dim 128, page 32, batch 1, **bfloat16** cache -- with PCC taken against a
# float32 reference built from the same cache the kernel reads, not against the
# default leg (``probes/sdpa_sweep_confirm.py``, median of 5 blocks of 50):
#
#     cur_pos      default   k256/c16   speedup   default PCC   k256/c16 PCC
#         127     23.72 us   19.00 us     1.25x      0.999734       0.999707
#        1023    120.51      22.02        5.47x      0.999714       0.999703
#        4095    451.85      30.75       14.69x      0.999519       0.999655
#        8191    893.14      38.15       23.41x      0.999024       0.999590
#       16383   1777.60      49.83       35.67x      0.993199       0.999577
#       32767   3545.05      74.13       47.82x      0.989703       0.999692
#
# Two things in that table, not one. The speed column is the expected one. The
# **PCC columns are the surprise**: the default's accuracy *decays with depth* --
# 0.9932 at 16k and 0.9897 at 32k, through this project's 0.995 layer bar --
# while the configured path holds 0.9996-0.9997 flat from 127 to 32767. So this
# is not a speed-for-accuracy trade. At the context this model advertises the
# config is strictly better on both axes, and the shipped default was the *less*
# accurate of the two.
#
# **The cache dtype is why this took two passes, and it is the lesson.** The
# stage-06 lever analysis recommended ``k_chunk_size=512`` on the strength of
# probes that allocated the cache as ``bfloat8_b``; ``create_mesh_kv_cache``
# allocates ``ttnn.bfloat16`` (see below, ~line 1167). Re-run at the real dtype,
# 512 loses its edge -- and, far worse, **512 is wrong in-model**:
# ``test_multichip_decode_batch`` (128-position paged cache, cur_pos 32) returns
# PCC **-0.04 to -0.17** against HF with it, nondeterministically in 2-3 of its 4
# batch sizes, across nine runs. Sweeping that real test pins the boundary
# exactly -- ``k_chunk`` in {32, 64, 128, 256} passes 4/4 at every
# ``max_cores_per_head_batch`` in {16, 32, 64}; only 512 fails -- so
# ``max_cores`` is innocent and ``k_chunk`` is the whole effect.
#
# No standalone construction reproduces it. ``probes/sdpa_kchunk_rule_probe.py``
# re-runs the op at bfloat16, at the failing 128-deep cache, with an 8-user paged
# page table laid out exactly as ``create_mesh_kv_cache`` lays it out, and reads
# PCC 0.9997 at k512 at every depth from 128 to 4096;
# ``probes/sdpa_shallow_cache_probe.py`` finds nothing either. The leading
# explanation is **L1 pressure**: standalone the op owns the whole of L1, while
# in-model it is co-resident with the layer's sharded activations, expert
# weights and CCL buffers, and a 512-deep bf16 K chunk is exactly the size that
# stops fitting. That the boundary is dtype-linked is independently visible --
# ``k1024/c64`` fails to *build* at bfloat16 (``program.cpp:1722``) and builds
# fine at bfloat8_b. It is recorded as unexplained-in-detail rather than argued;
# what is measured is that 512 is unsafe in-model and 256 is not.
#
# So: the sweep was redone at bfloat16, 6 x 4 points at five positions
# (``probes/sdpa_sweep_probe.py``), finalists re-timed over nine positions, and
# the choice restricted to the in-model-safe ``k_chunk <= 256``. **256/16 is the
# uniform winner** -- fastest of the safe configs at cur_pos 4095 and above,
# within 0.6% at 511-2047, and its worst point is +6.4% at cur_pos 127 (19.00 vs
# 17.86 us for 256/8, i.e. 0.05 ms on a 20 ms iteration). There is no
# context-dependence worth a runtime switch, so it is **fixed**; a traced decode
# could not vary it per step anyway. ``q_chunk_size`` stays 32: decode has one
# query row and 32 is the tile height.
_SDPA_PAGED_K_CHUNK = 256
_SDPA_PAGED_MAX_CORES_PER_HEAD = 16


def _paged_cache_depth(kv_cache) -> int:
    """Positions allocated **per user** in a paged cache.

    ``page_table`` is ``[max_batch, blocks_per_seq]`` and every block holds
    ``block_size`` positions, so this is the length of the logical sequence the
    cache can hold for one user -- which is the quantity ``k_chunk_size`` has to
    respect. See ``_sdpa_k_chunk``.
    """
    return int(kv_cache.page_table.shape[-1]) * int(kv_cache.block_size)


def _sdpa_k_chunk(kv_cache) -> int:
    """``_SDPA_PAGED_K_CHUNK``, clamped to what the cache can actually supply.

    **``k_chunk_size`` must not exceed the cache's per-user allocated depth**, and
    exceeding it does not raise -- it silently returns garbage. This is the whole
    reason the first adoption of this lever failed its gates, and it is worth
    stating precisely because nothing in the op signature hints at it.

    How it presents: ``test_multichip_decode_batch`` allocates a **128**-position
    paged cache. At ``k_chunk_size=256`` it returns PCC **-0.10 to +0.06** against
    HF -- noise, not a degraded answer -- but *only when another test has run
    before it in the same process*; run alone it passes. Run the same test after
    ``test_router_windows_partition_global_routing`` and it fails 4/4, at every
    ``max_cores_per_head_batch``. Sweeping ``k_chunk`` through that reproducer
    puts the boundary exactly at the cache depth:

        k_chunk  32   64   128  ->  7 passed, at max_cores in {8, 16, 32, 64}
        k_chunk 256          ->  4 failed

    That order-dependence is the tell, and it is what makes the bug so easy to
    miss: the op reads a full ``k_chunk`` past the end of the cache buffer, and
    whether that hurts depends on what the allocator last left there. On a fresh
    device it is zeros and the softmax mask hides it; after another test has
    allocated and freed tensors it is live garbage. **Every standalone probe
    misses this by construction** -- ``probes/sdpa_shallow_cache_probe.py`` and
    ``probes/sdpa_kchunk_rule_probe.py`` both reproduce the shapes, the dtype, the
    128-deep cache and the multi-user page table exactly, and both read PCC 0.9997
    at k512, because in a probe the cache is the only thing allocated. This is the
    same shape of miss as the stage-04 ``rotary_embedding_llama`` rejection: a
    probe that structurally cannot see the state interaction.

    So the clamp is not defensive coding, it is the operating range. At the
    shipped ``max_context_len`` (4096 and up, contract 262144) it never binds and
    the config is the tuned 256; at the tests' 128-deep caches it drops to 128,
    which the sweep prices at +6% on the op at cur_pos 127 and 0% past 511.
    """
    depth = _paged_cache_depth(kv_cache)
    # The ``max(32, ...)`` floor exists because SDPA will not take a chunk below
    # one tile. It is the one input that could make this function *violate* the
    # invariant in its own first line, and only when ``block_size < 32``, which
    # this model never configures (the block size is 32 and the page table is at
    # least one block per user). Assert it rather than leave a silent hole: a
    # shallower cache than one tile would return a chunk deeper than the cache.
    assert depth >= 32, (
        f"paged cache depth {depth} is below one tile, so the 32-row floor below would return a "
        "k_chunk_size deeper than the per-user allocated depth -- which SDPA reads past without "
        "raising. Raise block_size (currently "
        f"{int(kv_cache.block_size)}) or the page table width ({int(kv_cache.page_table.shape[-1])})."
    )
    chunk = min(_SDPA_PAGED_K_CHUNK, max(32, depth))
    # SDPA wants a power-of-two chunk; take the largest one that still fits.
    return 1 << (chunk.bit_length() - 1)


#: Program configs are immutable and there are at most a handful of distinct
#: ones, but they are built **per layer per call** -- 48 times a token on the
#: decode path and 48 times a prefill chunk. Each build calls
#: ``device.compute_with_storage_grid_size()``, which is a device query, not a
#: Python attribute. On the traced decode path that is capture-only and free; on
#: the *untraced* paths (``run_teacher_forcing``, ``run_prefill_check``, eager
#: decode) it is 96 device queries per token of pure host time. Memoised on the
#: grid size rather than the device handle so the cache survives device reopen.
_SDPA_CONFIG_CACHE: dict = {}


def _cached_sdpa_config(grid, q_chunk, k_chunk, max_cores=None):
    key = (grid.x, grid.y, q_chunk, k_chunk, max_cores)
    cfg = _SDPA_CONFIG_CACHE.get(key)
    if cfg is None:
        kwargs = {} if max_cores is None else {"max_cores_per_head_batch": max_cores}
        cfg = _SDPA_CONFIG_CACHE[key] = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=q_chunk, k_chunk_size=k_chunk, **kwargs
        )
    return cfg


def _sdpa_program_config(device, kv_cache=None):
    """Program config for SDPA-decode; ``kv_cache`` selects the tuned paged form.

    Both spellings are the same op family and the same maths; they differ only
    in chunking and core budget. Neither touches dtype, fidelity, the KV cache
    layout, or any collective -- this is a program config on a call the model
    already makes.
    """
    paged = kv_cache is not None and kv_cache.is_paged
    return _cached_sdpa_config(
        device.compute_with_storage_grid_size(),
        32,
        _sdpa_k_chunk(kv_cache) if paged else 32,
        _SDPA_PAGED_MAX_CORES_PER_HEAD if paged else _SDPA_MAX_CORES_PER_HEAD,
    )


# Prefill has the *same* gap and it is larger in absolute terms:
# ``attention_prefill`` also called SDPA with no program config, and the op
# default is quadratic-with-a-bad-constant in S. Same shapes, bfloat16
# (``doc/optimized_full_model/probes/sdpa_prefill_confirm.py``):
#
#       S    default   q128/k128   q256/k256
#     128    23.92 us    25.72 us    32.68 us
#     512    58.96       54.14       87.18
#    1024   230.58       88.36      127.54
#    2048   741.08      216.03      207.28
#    4096  2850.25      882.61      451.04
#    8192 10938.43     2907.58     1956.15
#   16384 44456.67    11364.22     6527.48
#
# The winner *is* length-dependent here, unlike decode, and the two legs cross
# at S ~= 2048. Prefill could pick at call time -- it is eager, not traced, so
# the branch is a Python ``if`` and there is no captured trace to invalidate.
#
# **It is nevertheless NOT adopted.** ``decoder_layer_prefill_multichip`` passes
# ``sdpa_program_config=None`` and prefill runs at the op default. Two measured
# reasons, in this order:
#
# 1. **It costs accuracy on the one gate that can see it.** With this config
#    wired, ``run_teacher_forcing`` reads top-1 **0.980** against a baseline of
#    **0.990** on the same tree (top-5 and top-100 stay 1.000). Bisected: the
#    *decode* config alone holds 0.990, the *prefill* config alone drops it to
#    0.980, so the flip is this and not the lever above
#    (``logs/run_teacher_forcing_leg_prefill.log`` /
#    ``logs/run_teacher_forcing_leg_decode.log``). One greedy token in a hundred
#    is small, but the stage bar is "do not spend accuracy for speed", and here
#    there is no speed to buy it with, which is reason 2.
# 2. **At the length actually being served it is a loss, not a win.** The
#    readiness reference prompt is **158 tokens**. The table above says the
#    config is *behind* the default below S ~= 384, and the measured TTFTs agree:
#    3448.79 ms baseline against 3445.31 ms with prefill configured -- noise. The
#    6.8x is real but it lives at S >= 4096, which nothing in the current gate
#    set exercises.
#
# So this is a **verified-fast, accuracy-ungated** lever, left wired and
# documented rather than taken. What it needs before adoption is a readiness
# reference with a multi-thousand-token prompt, so the regime where it pays
# (S >= 4096, 6.3-6.8x on the SDPA op, and 48 of them per prefill) is the same
# regime the accuracy gate covers. The seam in ``attention_prefill`` exists for
# exactly that -- same pattern as ``_meta_rope``, which is also built, measured
# and not adopted.
#
# **Arbitrary S keeps working**, so nothing here is blocked on alignment. This
# was checked and not assumed: S in
# {1, 3, 31, 33, 100, 129, 255, 257, 1000, 1023, 1025, 2049, 4095, 4097, 5000}
# all build and run under both chunkings with PCC identical to the default's to
# five decimals. Prefill is not chunked in this model -- ``prefill_forward``
# feeds each user's whole logical length in -- so that property is load-bearing
# for the stage contract, not a nicety.
#
# ``q512/k512`` is **rejected** outright: it fails to build at *every* length
# including 128 (``program.cpp:1722``), so it is a resource limit, not an
# alignment rule.
_SDPA_PREFILL_CROSSOVER = 2048


def _sdpa_prefill_program_config(device, seq_len: int):
    """Built and measured; **not wired in**. See the note above before adopting."""
    q_chunk = 256 if seq_len >= _SDPA_PREFILL_CROSSOVER else 128
    return _cached_sdpa_config(device.compute_with_storage_grid_size(), q_chunk, q_chunk)


# --- decode residual RMSNorm, width-sharded (stage 04) ------------------------
#
# The stage-03 decode layer spent 40.21 us -- 9.7% -- in its two residual
# RMSNorms, and the profile says why: both run on **one core**
# (``../doc/multichip_decoder/ops_perf_multichip_decode.csv.gz``, device 0, rows
# 134 and 159, 20.081 and 20.127 us, ``CORE COUNT`` 1). A 2048-wide bf16 norm
# over one 32-row tile is 128 KB in 20 us, i.e. 6.5 GB/s, which is a single
# core's share of L1 bandwidth and nothing else.
#
# ``ttnn.rms_norm`` has a sharded program factory that splits the row across a
# core grid. Feeding it the same L1 width-shard the DRAM-sharded qkv projection
# already wants -- 8 cores, one per DRAM bank, ``[32, 256]`` -- gives
# (``probes/norm_accuracy_probe.py``, trace slope, median of 30):
#
#     interleaved, no compute config (shipped)   19.82 us   max|err vs fp64| 6.711e-02
#     sharded  4 cores, HiFi4 fp32acc             7.53          1.439e-02
#     sharded  8 cores, default                   4.26          3.586e-02
#     sharded  8 cores, HiFi4 fp32acc             4.92          1.686e-02
#                                        i2s 0.51 us, s2i 0.53 us
#
# 8 cores at HiFi4 with fp32 accumulation is **4.0x faster and 4.0x more
# accurate** than the shipped call, which passes no compute config at all and so
# accumulates the sum of squares in bf16. The reference is torch fp64 over the
# bf16-rounded inputs the device actually sees, so "more accurate" is against
# the mathematical answer rather than against the other kernel.
#
# 16 cores and beyond do not pay: the norm itself stops improving (a 2048-wide
# row is 64 tiles, so 8 cores already hold 8 tiles each) while the resharding at
# both ends grows with the core count.
_NORM_SHARD_CORES = _DRAM_BANKS


def _norm_shard_config(dim: int) -> ttnn.MemoryConfig:
    """The L1 width-shard the sharded norm reads and writes.

    Deliberately ``_width_sharded_l1(dim)``'s spec: at ``dim == hidden_size``
    this is bit-for-bit the memory config ``attention_decode_optimized`` reshards
    its input into, so the first norm's output feeds the qkv projection with no
    conversion at all.
    """
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(_bank_row(_NORM_SHARD_CORES), [32, dim // _NORM_SHARD_CORES], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _norm_program_config(dim: int):
    block_w = dim // _NORM_SHARD_CORES // 32
    subblock_w = next(w for w in (4, 3, 2, 1) if block_w % w == 0)
    return ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[_NORM_SHARD_CORES, 1],
        subblock_w=subblock_w,
        block_h=1,  # decode's padded M is exactly one 32-row tile; batch is capped at 32
        block_w=block_w,
        inplace=False,
    )


def _norm_compute_config(device, precision: PrecisionConfig = DEFAULT_PRECISION):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=precision.norm_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def decode_residual_norm(
    x: ttnn.Tensor, weight_rm: ttnn.Tensor, eps: float, precision: PrecisionConfig = DEFAULT_PRECISION
) -> ttnn.Tensor:
    """One residual-stream RMSNorm at decode shape, width-sharded across 8 cores.

    Takes a DRAM-interleaved ``[1, 1, B, H]`` (B <= 32, padded to one tile) and
    returns an **L1 width-sharded** tensor in ``_norm_shard_config(H)``. Callers
    that need it interleaved say so; the first norm's consumer does not.
    """
    dim = int(x.shape[-1])
    assert int(x.shape[-2]) <= 32, (
        f"decode_residual_norm shards a single 32-row tile; got {int(x.shape[-2])} rows. "
        "Prefill uses the interleaved rms_norm."
    )
    mc = _norm_shard_config(dim)
    xs = ttnn.to_memory_config(x, mc)
    out = ttnn.rms_norm(
        xs,
        weight=weight_rm,
        epsilon=eps,
        program_config=_norm_program_config(dim),
        memory_config=mc,
        # ``precision`` rather than the default: this is the only site
        # ``norm_fidelity`` reaches. It was called with the module default until
        # the stage-07 review, which meant the field was a documented knob with
        # no effect and ``R21_norm_hifi2`` measured nothing. The prefill norms
        # (``decoder_layer_prefill_multichip``) pass no compute config at all and
        # still take the op default -- ``norm_fidelity`` is a decode-path field,
        # which is the path the stage ranks on.
        compute_kernel_config=_norm_compute_config(x.device(), precision),
    )
    ttnn.deallocate(xs)
    return out


def _exact_matmul_config(device, precision: PrecisionConfig = DEFAULT_PRECISION):
    """HiFi4, so the one-hot window matmul is a copy rather than an approximation.

    The matmul default is LoFi, which keeps ~5 mantissa bits, and that is fine
    for everything else in this layer -- but here the operand is 0/1 and the
    intent is to *select* a routing weight, not to compute with it. Measured, the
    LoFi spelling moved the stitched windows 9.77e-4 away from the single-chip
    dense routing (one bf16 ulp at these magnitudes) where HiFi4 reproduces them
    bit-for-bit. The tensor is 4 tiles by 1, so exactness is free.
    """
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=precision.router_window_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


def _expert_window_matrix(mesh_device, num_experts: int, num_devices: int) -> ttnn.Tensor:
    """Per-die one-hot selector ``[1, 1, E, E/num_devices]``.

    A TTNN mesh op is SPMD: one program on four dies, so ``ttnn.slice`` cannot
    take a different start offset per die and there is no way to ask for
    "columns 32d..32d+31" directly. The device-varying constant is built the
    only way a mesh tensor can vary by device -- a leading dim of ``num_devices``
    sharded on dim 0 -- and applied as a matmul.

    The matmul is exact, not approximate: the operand is 0/1, the accumulator is
    fp32 and the output is bf16, so each selected weight is copied bit-for-bit.
    K = 128 is 4 tiles and N = 32 is 1, so it is the cheapest op in the router
    block, and it *replaces* work rather than adding it -- the divide that
    follows now runs over 32 columns instead of 128.
    """
    local = num_experts // num_devices
    sel = torch.zeros(num_devices, 1, num_experts, local)
    for d in range(num_devices):
        for j in range(local):
            sel[d, 0, d * local + j, j] = 1.0
    return ttnn.from_torch(
        sel,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )


def upload_multichip_weights(
    torch_weights: dict[str, torch.Tensor],
    mesh_device,
    config: MeshDecoderConfig,
    expert_dtype=None,
    meta_rope: bool = False,
    precision: PrecisionConfig = DEFAULT_PRECISION,
) -> MultichipWeights:
    """Shard and upload one layer's weights across the mesh.

    Per-die footprint at the shipped dtypes, which is the table
    ``doc/multichip_decoder/mesh_plan.md`` section 2 computes:

        gate_up  [1, 32, 2048, 1536] bfloat4_b   56.623 MB
        down     [1, 32,  768, 2048] bfloat4_b   28.312 MB
        wqkv     [2048, 1280] bfloat8_b x2 copies 5.570 MB
        wo       [1024, 2048] bfloat8_b x2 copies 4.456 MB
        router   [2048, 128]  bf16               0.524 MB
        norms + qk-norms                         0.009 MB
                                          total 95.49 MB / layer / die

    Every division is exact -- 2048/4, 32/4, 4/4, 128/4 -- so this scheme needs
    **zero load-time padding**, and the contract's allowance for it goes unused.
    """
    a = config.global_config.attention
    n = config.num_devices
    # ``expert_dtype`` is the stage-04 spelling and still wins when given (the
    # multichip tests sweep it); otherwise every dtype below comes from
    # ``precision``, whose defaults are the values this docstring's table was
    # measured at.
    gate_up_dtype = expert_dtype if expert_dtype is not None else precision.experts_gate_up_dtype
    down_dtype = expert_dtype if expert_dtype is not None else precision.experts_down_dtype

    def replicate(t: torch.Tensor, tensor_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor:
        return ttnn.from_torch(
            t.contiguous().float(),
            dtype=tensor_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=memory_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def shard(t: torch.Tensor, dim: int, tensor_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor:
        return ttnn.from_torch(
            t.contiguous().float(),
            dtype=tensor_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=memory_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim),
        )

    def as_4d(t: torch.Tensor, pad_to_4d: bool = False) -> torch.Tensor:
        if pad_to_4d:
            t = t.reshape(1, 1, 1, -1)
        while t.dim() < 4:
            t = t.unsqueeze(0)
        return t

    wqkv = head_interleaved_wqkv(as_4d(torch_weights["wqkv"]), a, n)
    wo = as_4d(torch_weights["wo"])

    # Per-die shapes, used by both the shard spec and the assertions below.
    k_qkv, n_qkv = int(wqkv.shape[-2]), int(wqkv.shape[-1]) // n
    k_o, n_o = int(wo.shape[-2]) // n, int(wo.shape[-1])
    assert _dram_sharded_ok(k_qkv, n_qkv), (
        f"per-die wqkv [{k_qkv}, {n_qkv}] is not bank-divisible; the DRAM-sharded decode "
        "projections would silently fall back to interleaved and give back stage 02's 1.11x"
    )
    assert _dram_sharded_ok(k_o, n_o), f"per-die wo [{k_o}, {n_o}] is not bank-divisible"

    def dram_sharded(t: torch.Tensor, dim: int, k: int, n_local: int, tensor_dtype) -> ttnn.Tensor:
        """Width-shard the per-die weight one shard per DRAM bank, then mesh-shard it.

        Two independent shardings compose here and it is worth being explicit
        about which is which: ``mesh_mapper`` fractures the tensor *across dies*
        (TP), while ``memory_config`` fractures each die's piece across that
        die's 8 DRAM banks (stage 02's decode projection layout). The shard spec
        is therefore written in per-die elements, not global ones.
        """
        return shard(
            t,
            dim,
            tensor_dtype,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(_bank_row(_DRAM_BANKS), [k, n_local // _DRAM_BANKS], ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )

    experts = OptimizedWeights(
        # [E, 2I, H] -> [1, E, H, 2I], sharded on the expert dim.
        gate_up_proj=shard(torch_weights["experts_gate_up"].transpose(-2, -1).unsqueeze(0), 1, gate_up_dtype),
        # [E, H, I] -> [1, E, I, H], sharded on the expert dim.
        down_proj=shard(torch_weights["experts_down"].transpose(-2, -1).unsqueeze(0), 1, down_dtype),
        attention=AttentionWeights(
            # Column split (head-interleaved, see head_interleaved_wqkv).
            wqkv=shard(wqkv, -1, precision.attention_qkv_dtype),
            # Row split by Q head. Contiguous *because* the Q head assignment
            # above is contiguous per die: die d owns rows 1024d..1024d+1023.
            wo=shard(wo, -2, precision.attention_wo_dtype),
            q_norm=replicate(as_4d(torch_weights["q_norm"], pad_to_4d=True), precision.norm_weight_dtype),
            k_norm=replicate(as_4d(torch_weights["k_norm"], pad_to_4d=True), precision.norm_weight_dtype),
        ),
        wqkv_decode=dram_sharded(wqkv, -1, k_qkv, n_qkv, precision.attention_qkv_dtype),
        wo_decode=dram_sharded(wo, -2, k_o, n_o, precision.attention_wo_dtype),
    )

    # Stage 04. The Meta-ordered decode twin, built **only when asked**. It is
    # not the shipped path (see ``_meta_rope``); it exists so
    # ``probes/rope_layer_probe.py`` can re-measure the rejection rather than
    # cite it. Off by default, so the shipped upload pays no extra DRAM.
    experts_meta = None
    if meta_rope:
        # The channel permutation is applied to the *pre-interleave* wqkv, which
        # is safe because it reorders channels **within** a head and
        # ``head_interleaved_wqkv`` only reorders whole heads -- the two commute.
        # V is untouched, and so are ``wo`` and every expert weight, which are
        # shared objects here rather than copies.
        wqkv_meta = head_interleaved_wqkv(
            permute_wqkv_to_meta(
                as_4d(torch_weights["wqkv"]),
                n_heads=a.num_attention_heads,
                n_kv_heads=a.num_key_value_heads,
                head_dim=a.head_dim,
            ),
            a,
            n,
        )
        experts_meta = replace(
            experts,
            attention=replace(
                experts.attention,
                q_norm=replicate(
                    as_4d(permute_head_vector_to_meta(torch_weights["q_norm"], head_dim=a.head_dim), pad_to_4d=True),
                    precision.norm_weight_dtype,
                ),
                k_norm=replicate(
                    as_4d(permute_head_vector_to_meta(torch_weights["k_norm"], head_dim=a.head_dim), pad_to_4d=True),
                    precision.norm_weight_dtype,
                ),
            ),
            wqkv_decode=dram_sharded(wqkv_meta, -1, k_qkv, n_qkv, precision.attention_qkv_dtype),
        )

    router = torch_weights["router"]

    def norm_row_major(t: torch.Tensor) -> ttnn.Tensor:
        """The same vector the tiled copy holds, in the layout the sharded
        ``rms_norm`` program factory reads: ROW_MAJOR ``[1, 1, dim/32, 32]``."""
        flat = t.reshape(-1)
        return ttnn.from_torch(
            flat.reshape(1, 1, flat.numel() // 32, 32).contiguous().float(),
            dtype=precision.norm_weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    return MultichipWeights(
        input_layernorm=replicate(torch_weights["input_layernorm"].reshape(1, 1, 1, -1), precision.norm_weight_dtype),
        post_attention_layernorm=replicate(
            torch_weights["post_attention_layernorm"].reshape(1, 1, 1, -1), precision.norm_weight_dtype
        ),
        router=replicate(router.T.contiguous().reshape(1, 1, router.shape[1], router.shape[0]), precision.router_dtype),
        expert_window=_expert_window_matrix(mesh_device, config.global_config.moe.num_experts, n),
        experts=experts,
        experts_meta=experts_meta,
        input_layernorm_rm=norm_row_major(torch_weights["input_layernorm"]),
        post_attention_layernorm_rm=norm_row_major(torch_weights["post_attention_layernorm"]),
    )


def create_mesh_kv_cache(
    mesh_device,
    config: MeshDecoderConfig,
    max_batch: int,
    max_seq_len: int,
    block_size: int | None = None,
    precision: PrecisionConfig = DEFAULT_PRECISION,
) -> KVCache:
    """Allocate the *local* KV cache: 1 KV head per die, not 4.

    This is where the TP factor buys capacity rather than speed. Per die the
    cache is ``[.., 1, .., 128]`` instead of ``[.., 4, .., 128]``, i.e. 512 B
    per token per layer instead of 2048 -- 6.44 GB at the advertised 262144
    context over 48 layers, against 25.77 GB on one die. One die cannot hold
    this model at full context; four can, with room to spare. See
    ``doc/context_contract.json``.

    The buffers are *replicated at allocation* because they are zeros, and
    diverge the moment the first token is written -- each die holds a different
    KV head. The page table is genuinely identical on every die: paging is a
    logical-to-physical block mapping and does not depend on which head lives
    where.
    """
    local = config.local_attention
    if block_size is None:
        shape = (max_batch, local.num_key_value_heads, max_seq_len, local.head_dim)
        page_table = None
    else:
        blocks_per_seq = math.ceil(max_seq_len / block_size)
        shape = (max_batch * blocks_per_seq, local.num_key_value_heads, block_size, local.head_dim)
        page_table = ttnn.from_torch(
            torch.arange(max_batch * blocks_per_seq, dtype=torch.int32).reshape(max_batch, blocks_per_seq),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    k, v = (
        ttnn.from_torch(
            torch.zeros(shape),
            dtype=precision.kv_cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        for _ in range(2)
    )
    return KVCache(k=k, v=v, page_table=page_table, block_size=block_size or 0)


def build_local_sparsity(mesh_device, local_moe: MoEConfig) -> ttnn.Tensor:
    """All-ones prefill sparsity over this die's 32 experts, replicated."""
    return ttnn.from_torch(
        torch.ones(1, 1, 1, local_moe.num_experts, dtype=torch.bfloat16),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


# --- router -------------------------------------------------------------------


def router_forward_threshold(
    x: ttnn.Tensor,
    w_router: ttnn.Tensor,
    window: ttnn.Tensor,
    config: MoEConfig,
    local_moe: MoEConfig,
) -> ttnn.Tensor:
    """``router_forward_multichip`` with the dense vector built by a threshold
    comparison instead of a scatter, so nothing leaves TILE layout.

    Stage 03 inherited stage 02's routing tail, whose shape is
    ``topk -> untilize(zeros) / untilize(indices) / untilize(values) ->
    scatter -> tilize``: ``ttnn.scatter`` only accepts ROW_MAJOR, and every
    consumer of the dense vector (both matmuls, the divide) needs TILE. Stage 02
    recorded that round trip as *not removable* on those grounds. It is
    removable -- by not scattering.

    ``topk(sorted=True)`` already returns the 8th-largest logit in column 7, and
    the top-8 set is exactly ``{j : logit_j >= that}``. So the same dense vector
    is

        dense = exp(logits - top_max) * (logits >= top_logits[..., 7])

    computed over all 128 columns, entirely in TILE. The surviving values are
    ``ttnn.exp`` of the same fp32 inputs the scatter path fed it, so the result
    is bit-identical -- **unless two logits tie exactly at rank 8**, in which
    case this selects both and the scatter path selects one. With fp32 logits
    accumulated over K=2048 that does not happen on real weights, and
    ``test_router_windows_partition_global_routing`` asserts the equality at
    ``max |diff| = 0.0``, so a tie would fail loudly rather than drift.

    **Measured, and rejected.** It removes rows 190-197 of the stage-04 decode
    profile -- ``zeros_like`` 1.210, two ``typecast`` 2.537, three ``untilize``
    4.654, ``scatter`` 3.030, ``tilize`` 5.576 = **17.007 us** -- and is
    nonetheless **0.8% slower on the layer**: 0.4382 / 0.4382 ms against the
    shipped 0.4348 / 0.4346 over two interleaved passes
    (``doc/optimized_multichip_decoder/probes/layer_levers2.py``). Widening the
    softmax's ``sub`` and ``exp`` from 8 columns to 128, plus the ``ge`` and the
    ``mul``, costs more than the layout conversions save. The output is
    bit-identical on all four dies (``max|diff| 0.000e+00``), which is also the
    evidence that no two logits tie at rank 8.

    Kept rather than deleted because the arithmetic is the useful part: stage
    02 recorded this round trip as *not removable*, and it is.
    """
    assert config.norm_topk_prob
    e = config.num_experts

    logits = ttnn.linear(x, w_router, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    top_logits, _idx = ttnn.topk(logits, k=config.num_experts_per_tok, dim=-1, largest=True, sorted=True)

    rows = top_logits.shape[2]
    top_max = ttnn.slice(top_logits, [0, 0, 0, 0], [1, 1, rows, 1])
    cutoff = ttnn.slice(top_logits, [0, 0, 0, config.num_experts_per_tok - 1], [1, 1, rows, config.num_experts_per_tok])

    # exp(l - max) over the whole row; every entry is in (0, 1], so nothing can
    # overflow and the losers underflow towards zero before the mask even runs.
    weights = ttnn.exp(ttnn.sub(logits, top_max))
    dense = ttnn.typecast(ttnn.mul(weights, ttnn.ge(logits, cutoff)), ttnn.bfloat16)

    total = ttnn.matmul(dense, _ones_column(x.device(), e), dtype=ttnn.bfloat16)
    local = ttnn.matmul(dense, window, dtype=ttnn.bfloat16, compute_kernel_config=_exact_matmul_config(x.device()))
    guarded = ttnn.maximum(total, 1e-30)
    normalised = ttnn.div(local, guarded)
    assert int(normalised.shape[-1]) == local_moe.num_experts
    for t in (logits, top_logits, _idx, top_max, cutoff, weights, dense, total, local, guarded):
        ttnn.deallocate(t)
    return normalised


def router_forward_multichip(
    x: ttnn.Tensor,
    w_router: ttnn.Tensor,
    window: ttnn.Tensor,
    config: MoEConfig,
    local_moe: MoEConfig,
    precision: PrecisionConfig = DEFAULT_PRECISION,
) -> ttnn.Tensor:
    """Replicated global routing, returning this die's ``[1, 1, S, 32]`` window.

    Identical arithmetic to ``optimized_decoder.router_forward_optimized`` --
    selection on raw fp32 logits, softmax over the 8 survivors, neither keepdim
    reduction spelled as a ttnn reduction -- with one op added and one op made
    four times narrower.

    **Why the whole router is replicated.** Top-8 of 128 needs the global logit
    vector, and there is nothing worth fracturing anyway: the router matmul has
    N = 128 = 4 tiles so it can use 4 cores, and ``ttnn.topk`` over a single
    128-wide row occupies exactly 1. Splitting N four ways would give each die
    one tile and one core, and would additionally need a collective *inside* the
    routing path to reassemble the logits before the top-k. So each die computes
    the full 128-way routing on the bit-identical replicated activation and
    takes its own window out of the result -- no collective, at the price of
    88.9 us of decode device time that four dies pay in full.

    **The correctness assumption this makes, stated plainly.** The four windows
    are a partition of the global top-8 only if all four dies agree on which 8
    experts won. The inputs are bit-identical and the program is the same, so
    ``ttnn.topk`` should return identical indices -- but that is a *tie-breaking
    determinism* claim, and if it ever failed the layer would be silently wrong
    with no shape error and only a PCC drift to show for it. It is asserted
    directly by ``test_topk_is_identical_across_dies`` rather than argued.
    """
    assert config.norm_topk_prob, (
        "router selects on raw logits, which relies on the softmax denominator "
        "cancelling during top-k renormalisation; that only holds when norm_topk_prob is True"
    )

    logits = ttnn.linear(x, w_router, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return _router_tail(logits, window, config, local_moe, x.device(), precision)


def _router_tail(
    logits,
    window,
    config: MoEConfig,
    local_moe: MoEConfig,
    device,
    precision: PrecisionConfig = DEFAULT_PRECISION,
) -> ttnn.Tensor:
    """Top-8, softmax over the survivors, this die's 32-expert window.

    Split out of ``router_forward_multichip`` so a probe can vary where the
    logits are produced without duplicating the tail.
    """
    top_logits, top_indices = ttnn.topk(logits, k=config.num_experts_per_tok, dim=-1, largest=True, sorted=True)

    top_max = ttnn.slice(top_logits, [0, 0, 0, 0], [1, 1, top_logits.shape[2], 1])
    exp_logits = ttnn.exp(ttnn.sub(top_logits, top_max))

    zeros = ttnn.typecast(ttnn.zeros_like(logits), ttnn.bfloat16)
    dense = ttnn.scatter(zeros, dim=-1, index=top_indices, src=ttnn.typecast(exp_logits, ttnn.bfloat16))

    # The denominator is the sum over all 128 -- which is the sum over the 8
    # survivors, since the scatter fills a field of exact zeros -- and must stay
    # global: normalising within a window would renormalise each die's share to
    # 1 and the four contributions would sum to 4.
    total = ttnn.matmul(dense, _ones_column(device, config.num_experts), dtype=ttnn.bfloat16)
    local = ttnn.matmul(
        dense, window, dtype=ttnn.bfloat16, compute_kernel_config=_exact_matmul_config(device, precision)
    )
    # Same clamp as the single-chip router, and for the same reason: after the
    # scatter the divide runs over whole tiles, and the tile row-padding has a
    # zero numerator *and* a zero denominator, which unguarded ttnn.div returns
    # as +inf. Every real row's denominator is >= 1 (sorted=True makes column 0
    # of exp_logits exactly exp(0) = 1), so the clamp cannot touch one.
    guarded = ttnn.maximum(total, 1e-30)
    normalised = ttnn.div(local, guarded)
    assert int(normalised.shape[-1]) == local_moe.num_experts
    for t in (logits, top_logits, top_indices, top_max, exp_logits, dense, total, local, guarded):
        ttnn.deallocate(t)
    return normalised


# --- experts ------------------------------------------------------------------


def moe_decode_multichip(
    x: ttnn.Tensor,
    routing: ttnn.Tensor,
    weights: OptimizedWeights,
    local_moe: MoEConfig,
    precision: PrecisionConfig = DEFAULT_PRECISION,
) -> ttnn.Tensor:
    """Decode expert pass over this die's 32 experts. Returns a *partial* sum.

    Structurally ``optimized_decoder.moe_decode_optimized`` with the local
    expert count, and one difference that is not cosmetic: **``nnz`` is
    ``None``**.

    Stage 02 passes ``nnz = top_k * batch``, exact because every one of the
    global top-8 is computed on the single die. Under EP the count of live
    experts in *this* die's window is data-dependent -- 0 to 8, mean 2, and
    different on each die -- while a mesh op is SPMD and compiles one kernel for
    all four. Passing any host-computed value would deadlock the board the first
    time the routing was unbalanced (``sparse_matmul_device_operation.cpp``
    205-211, tt-metal #45943), silently unless the watcher is on. ``nnz=None``
    switches the in0 sender to reading the sparsity page at runtime and
    multicasting a per-slot valid flag; the loop still visits all 32 slots but
    only reads weights and does math for the live ones.

    Measured cost of dynamic mode, decode M=1, bfp4/LoFi, trace-slope:

        E=128 nnz=8 (single-die baseline)   139.45 + 125.20 = 264.65 us
        E=32  nnz=None (this path)           60.67 +  63.29 = 123.96 us  2.13x
        E=32  nnz=8  (exact, illegal here)   82.72 +  58.21 = 140.93 us
        E=128 nnz=None (dynamic at full E)  243.08 + 249.03 = 492.11 us  0.54x

    **That 2.13x did not survive.** Re-measured at the shipped shapes -- E=32,
    M=1, bfloat4_b, LoFi, L1 output, stage 02's tuned block widths -- dynamic
    mode costs 158.01 us against an exact ``nnz``'s 107.73, **1.47x**, and the
    multichip decode profile reads 82.65 us for the pair against the single
    chip's 92.06, i.e. **1.11x, not 2.13x**
    (``probes/nnz_cost_probe.py``, ``doc/multichip_decoder/work_log.md`` section
    8). The sweep above was a DRAM-out, random-weight microbenchmark whose E=128
    baseline read 264.65 where the profiled layer reads 92.06; only ratios were
    taken from it, and the ratio was still wrong, because the overhead it hid is
    additive rather than proportional.

    The E=32/nnz=8 row is also the answer to "why not capacity padding": the
    only capacity that can never be exceeded is 8, and building a fixed-count
    sparsity on device needs a second ``topk`` over the local 32 -- **26.32 us**
    on one core in the decode profile, more than the ~26 us it would save.
    Any smaller capacity drops experts, which changes the model output.
    """
    batch = x.shape[2]
    n_experts = local_moe.num_experts
    hidden_size = local_moe.hidden_size
    inter = local_moe.moe_intermediate_size

    sparsity = ttnn.to_layout(routing, ttnn.ROW_MAJOR_LAYOUT)
    expert_memory_config = _decode_expert_memory_config(batch, local_moe)
    output_tile = ttnn.Tile([32, 32])
    compute_config = _expert_compute_kernel_config(x.device(), precision)
    gate_up_config = _tuned_sparse_matmul_config(1, 2 * inter, hidden_size, precision.experts_gate_up_in0_block_w)
    down_config = _tuned_sparse_matmul_config(1, hidden_size, inter, precision.experts_down_in0_block_w)

    x_batched = ttnn.reshape(x, (1, batch, 1, hidden_size))
    fused = ttnn.sparse_matmul(
        x_batched,
        weights.gate_up_proj,
        sparsity=sparsity,
        nnz=None,  # see docstring -- a host-computed nnz deadlocks the board here
        memory_config=expert_memory_config,
        output_tile=output_tile,
        program_config=gate_up_config,
        compute_kernel_config=compute_config,
        dtype=precision.activation_dtype,
    )
    packed_width = fused.shape[-1]
    fused = ttnn.reshape(fused, (batch, n_experts, packed_width))

    half = packed_width // 2
    gate = ttnn.slice(fused, [0, 0, 0], [batch, n_experts, half])
    up = ttnn.slice(fused, [0, 0, half], [batch, n_experts, packed_width])
    ttnn.deallocate(fused)

    down_input = ttnn.reshape(ttnn.mul(ttnn.silu(gate), up), (batch, n_experts, 1, half))
    ttnn.deallocate(gate)
    ttnn.deallocate(up)

    down = ttnn.sparse_matmul(
        down_input,
        weights.down_proj,
        sparsity=sparsity,
        nnz=None,
        memory_config=expert_memory_config,
        output_tile=output_tile,
        program_config=down_config,
        is_input_a_sparse=True,
        is_input_b_sparse=False,  # selects batch_length_A = B * E; see the single-chip docstring
        compute_kernel_config=compute_config,
        dtype=precision.activation_dtype,
    )
    ttnn.deallocate(down_input)

    # The multiply by the routing weight is what makes a skipped slot harmless:
    # a die holding none of the global top-8 multiplies 32 untouched output
    # slots by exact zero and contributes an exact zero to the all-reduce.
    # test_expert_window_can_be_empty pins that, because "untouched" would not
    # be enough if the op left a NaN there.
    states = ttnn.reshape(down, (batch, n_experts, hidden_size))
    states = ttnn.mul(states, ttnn.reshape(routing, (batch, n_experts, 1)))
    states = ttnn.unsqueeze_to_4D(ttnn.sum(states, dim=1))
    return ttnn.reshape(states, (1, 1, batch, hidden_size), (1, 1, max(32, batch), hidden_size))


# --- the layer ----------------------------------------------------------------


def decoder_layer_prefill_multichip(
    x: ttnn.Tensor,
    weights: MultichipWeights,
    config: MeshDecoderConfig,
    ctx: MeshContext,
    cos_cache: ttnn.Tensor,
    sin_cache: ttnn.Tensor,
    sparsity: ttnn.Tensor,
    kv_cache: KVCache | None = None,
    user_id: int = 0,
    precision: PrecisionConfig = DEFAULT_PRECISION,
) -> ttnn.Tensor:
    """Prefill one layer on the mesh. ``x`` / return replicated ``[1, 1, S, 2048]``.

    ``S`` is arbitrary. Nothing in the multichip path adds an alignment
    constraint: the collectives scatter on dim 3 (hidden, 2048, fixed), and the
    only padding in play is ``moe_prefill_optimized``'s internal chunk padding,
    which is the single-chip behaviour and is sliced back inside that function.
    """
    eps = config.global_config.rms_norm_eps

    normed = ttnn.rms_norm(x, weight=weights.input_layernorm, epsilon=eps)
    attn_partial = attention_prefill(
        normed,
        weights.experts.attention,
        config.local_attention,
        cos_cache,
        sin_cache,
        kv_cache,
        user_id,
        # ``None`` at the shipped precision, which is the op default and what
        # every prefill number was measured at; see
        # ``optimized_decoder._attention_compute_kernel_config``.
        compute_kernel_config=_attention_compute_kernel_config(x.device(), precision),
        activation_dtype=precision.activation_dtype,
        # NOT adopted -- see _sdpa_prefill_program_config. The seam is wired and
        # the config is built and measured; passing it costs a top-1 point on
        # run_teacher_forcing, so prefill stays at the op default.
        sdpa_program_config=None,
    )
    ttnn.deallocate(normed)
    attn_out = all_reduce_prefill(attn_partial, ctx, precision)
    ttnn.deallocate(attn_partial)
    hidden = ttnn.add(x, attn_out)
    ttnn.deallocate(attn_out)

    normed = ttnn.rms_norm(hidden, weight=weights.post_attention_layernorm, epsilon=eps)
    routing = router_forward_multichip(
        normed, weights.router, weights.expert_window, config.global_config.moe, config.local_moe, precision
    )
    moe_partial = moe_prefill_optimized(normed, routing, weights.experts, config.local_moe, sparsity, precision)
    ttnn.deallocate(normed)
    ttnn.deallocate(routing)
    moe_out = all_reduce_prefill(moe_partial, ctx, precision)
    ttnn.deallocate(moe_partial)

    out = ttnn.add(hidden, moe_out)
    ttnn.deallocate(hidden)
    ttnn.deallocate(moe_out)
    return out


def decoder_layer_decode_multichip(
    x: ttnn.Tensor,
    weights: MultichipWeights,
    config: MeshDecoderConfig,
    ctx: MeshContext,
    cos_cache: ttnn.Tensor,
    sin_cache: ttnn.Tensor,
    kv_cache: KVCache,
    current_pos: ttnn.Tensor,
    token_index: int,
    rope=None,
    precision: PrecisionConfig = DEFAULT_PRECISION,
    active_mask: ttnn.Tensor | None = None,
) -> ttnn.Tensor:
    """Decode one token per user on the mesh. ``x`` / return ``[1, 1, B, 2048]``.

    Input and output layouts are the same replicated tensor, which is the point:
    48 of these stack with no boundary conversion, and the stacked model pays
    the two all-reduces per layer and nothing else. **That is the inter-layer
    residual layout contract**, and stage 04 keeps it unchanged while moving
    every *intra*-layer boundary it can into L1 shards -- see
    ``doc/optimized_multichip_decoder/README.md``.

    Both residual norms are width-sharded (``decode_residual_norm``). The first
    one's output is already in ``attention_decode_optimized``'s qkv input shard,
    so it crosses into attention with no conversion; the second one's output
    feeds the router projection sharded and the expert path interleaved.
    """
    eps = config.global_config.rms_norm_eps

    normed = decode_residual_norm(x, weights.input_layernorm_rm, eps, precision)
    # ``rope`` is the stage-05 seam. It defaults to ``None`` and therefore to
    # ``_apply_rope`` -- ``ttnn.experimental.rotary_embedding`` with a **Python
    # int** ``token_index``, which is what every stage-03/04 number was measured
    # at and what the single-layer tests still exercise. That spelling cannot be
    # replayed: the position is a compile-time argument, so a captured trace
    # rotates every later token at the position it was captured at. The full
    # model therefore passes ``model._rope_decode``, which is
    # ``ttnn.experimental.rotary_embedding_hf(is_decode_mode=True)`` reading a
    # **per-user cos/sin pair gathered on device** from a position tensor the
    # trace itself advances. Same HF ``rotate_half`` channel convention, so the
    # KV cache convention, prefill, and every weight are untouched -- which is
    # exactly what stage 04's rejected ``rotary_embedding_llama`` lever could not
    # offer (README limitation 4).
    # The rotary stays the **HF** op. ``rotary_embedding_llama`` is 3.05x faster
    # standalone and bit-identical there, but it cannot be adopted for decode
    # alone: it needs Meta channel order, and the KV cache prefill already wrote
    # is in HF order, so SDPA would score a Meta-ordered Q against HF-ordered
    # keys. Measured, not argued -- PCC 0.193 against a prefill-primed cache
    # where a fresh cache reads 0.99997 (``probes/rope_layer_probe.py``). See
    # ``_meta_rope`` and ``README.md`` limitation 4.
    attn_partial = attention_decode_optimized(
        normed,
        weights.experts,
        config.local_attention,
        cos_cache,
        sin_cache,
        kv_cache,
        current_pos,
        token_index,
        # Both paths are configured now. The contiguous one needs the 64-core cap
        # to clear a TT_FATAL; the paged one -- what the full model runs -- takes
        # the swept k256/c16 config (k clamped to the cache depth), which is flat
        # in cur_pos where the op default is linear in it. See
        # _sdpa_program_config and _sdpa_k_chunk.
        sdpa_program_config=_sdpa_program_config(x.device(), kv_cache),
        rope=rope,
        precision=precision,
    )
    ttnn.deallocate(normed)
    attn_out = all_reduce_decode(attn_partial, ctx, precision)
    ttnn.deallocate(attn_partial)
    hidden = ttnn.add(x, attn_out)
    ttnn.deallocate(attn_out)

    normed_sharded = decode_residual_norm(hidden, weights.post_attention_layernorm_rm, eps, precision)
    # The router projection reads the shard directly -- N = 128 is 4 tiles, so
    # the matmul uses 4 cores either way, but a width-sharded L1 in0 turns a
    # 24.62 us DRAM-interleaved read into 5.85 us of L1 with bit-identical
    # output (``probes/norm_router_probe.py``, max|diff| exactly 0.0).
    routing = router_forward_multichip(
        normed_sharded, weights.router, weights.expert_window, config.global_config.moe, config.local_moe, precision
    )
    if active_mask is not None:
        # Zero the routing weights of every slot that holds no live request.
        #
        # ``routing`` *is* ``sparse_matmul``'s sparsity tensor, and its nonzero
        # count is the amount of expert math the op does: with ``nnz=None`` the
        # kernel reads the sparsity page at runtime and only fetches weights and
        # multiplies for the live ``(row, expert)`` pairs. A serving decode batch
        # is padded to the configured ``max_num_seqs`` with inactive rows, and an
        # inactive row's garbage hidden state still routes to a full top-8 -- so
        # without this a 32-slot server does 32 rows of expert work no matter how
        # many users are actually connected. See
        # ``doc/optimized_vllm/probes/batch_decode_control.py``.
        #
        # ``active_mask`` is derived **on device** from ``current_pos`` inside the
        # same traced graph (``Qwen3CoderModel._decode_active_mask``), so it can
        # never be stale: ``ttnn.plus_one(..., skip_negative_entries=True)`` leaves
        # an inactive row at ``-1`` forever, and a row that changes hands only does
        # so through a host reinstall of ``current_pos``.
        gated = ttnn.mul(routing, active_mask)
        ttnn.deallocate(routing)
        routing = gated
    # ``sparse_matmul``'s in0 is DRAM-interleaved, so the expert path pays one
    # sharded-to-interleaved (0.53 us) rather than the norm paying 15.
    normed = ttnn.sharded_to_interleaved(normed_sharded, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(normed_sharded)
    moe_partial = moe_decode_multichip(normed, routing, weights.experts, config.local_moe, precision)
    ttnn.deallocate(normed)
    ttnn.deallocate(routing)
    moe_out = all_reduce_decode(moe_partial, ctx, precision)
    ttnn.deallocate(moe_partial)

    out = ttnn.add(hidden, moe_out)
    ttnn.deallocate(hidden)
    ttnn.deallocate(moe_out)
    return out


# Bytes one 32x32 tile occupies, per dtype. Spelled out because
# ``Tensor.element_size()`` raises for the block-float types -- their storage is
# a byte (or nibble) per element *plus* a shared exponent per 16-element face
# row, i.e. 1024 + 64 for bfloat8_b and 512 + 64 for bfloat4_b -- and the
# expert weights, which are the whole point of measuring this, are block-float.
_TILE_BYTES = {
    str(ttnn.float32): 4096,
    str(ttnn.bfloat16): 2048,
    str(ttnn.bfloat8_b): 1088,
    str(ttnn.bfloat4_b): 576,
}


def _tensor_bytes(t: ttnn.Tensor) -> int | None:
    """Device bytes one mesh-sharded tensor occupies **per die**, or ``None``.

    A mesh tensor's shape is already the *local* (per-die) shape, so this is the
    allocation a dtype change actually moves -- which is the observable
    ``tests/test_precision_config.py`` asserts on. ``None`` for a dtype with no
    entry above rather than a wrong number.
    """
    tile_bytes = _TILE_BYTES.get(str(t.dtype))
    if tile_bytes is None:
        return None
    shape = [int(v) for v in t.padded_shape]
    tiles = math.prod(shape[:-2]) * math.ceil(shape[-2] / 32) * math.ceil(shape[-1] / 32)
    return tiles * tile_bytes


def fallback_audit(
    weights: MultichipWeights,
    config: MeshDecoderConfig,
    batch: int,
    precision: PrecisionConfig = DEFAULT_PRECISION,
) -> dict:
    """Every runtime fallback the imported single-chip code can still take.

    Three of stage 02's helpers choose a slower path silently rather than
    raising, and all three have different inputs under TP/EP than they were
    tuned against, so "it still passes PCC" would not notice any of them:

    * ``_dram_sharded_usable`` -- falls back to the interleaved ``attention_decode``
      if the weight dims were not bank-divisible at upload or the batch exceeds
      32. Per-die N is now 1280 rather than 5120 and per-die K 1024 rather than
      4096, and 1280 = 5x256 is only one factor of two away from failing.
    * ``_tuned_sparse_matmul_config`` -- silently lowers ``in0_block_w`` to the
      largest divisor of K in tiles. EP leaves K alone (2048 and 768), so the
      tuned 16 and 12 must survive; if they did not, this would be scheme A's
      regression arriving by the back door.
    * ``_decode_expert_memory_config`` -- moves the expert intermediates from L1
      to DRAM past a byte budget, which EP shrank 4x.

    Since stage 07 it also reports what the *precision config actually put on
    the device*: the dtypes read back off the uploaded tensors (not the config's
    own fields -- those would only prove the dataclass round-trips), the block
    widths the program configs resolved to, and the fidelities the compute
    configs carry. That is what ``tests/test_precision_config.py`` asserts
    against when it constructs at a non-default value.

    Returned as data so a test can assert on it and the work log can quote it.
    """
    a = config.local_attention
    m = config.local_moe
    k_qkv = int(weights.experts.wqkv_decode.shape[-2]) if weights.experts.wqkv_decode is not None else None
    n_qkv = int(weights.experts.wqkv_decode.shape[-1]) if weights.experts.wqkv_decode is not None else None
    k_o = int(weights.experts.wo_decode.shape[-2]) if weights.experts.wo_decode is not None else None
    n_o = int(weights.experts.wo_decode.shape[-1]) if weights.experts.wo_decode is not None else None
    gate_up = _tuned_sparse_matmul_config(
        1, 2 * m.moe_intermediate_size, m.hidden_size, precision.experts_gate_up_in0_block_w
    )
    down = _tuned_sparse_matmul_config(1, m.hidden_size, m.moe_intermediate_size, precision.experts_down_in0_block_w)
    return {
        "batch": batch,
        "dram_sharded_qkv": (k_qkv, n_qkv),
        "dram_sharded_wo": (k_o, n_o),
        "dram_sharded_taken": weights.experts.wqkv_decode is not None
        and weights.experts.wo_decode is not None
        and batch <= 32,
        "gate_up_in0_block_w": gate_up.in0_block_w,
        "down_in0_block_w": down.in0_block_w,
        "expert_intermediate_buffer": "L1"
        if _decode_expert_memory_config(batch, m) == ttnn.L1_MEMORY_CONFIG
        else "DRAM",
        "local_heads": (a.num_attention_heads, a.num_key_value_heads),
        "local_experts": m.num_experts,
        # Stage 04. Not a fallback in the "silently slower path" sense -- a
        # mismatch here raises rather than degrades -- but it is the same class
        # of risk, so it is reported as data: if the sharded norm's output shard
        # ever stops being *exactly* the one the DRAM-sharded qkv projection
        # wants, TTNN inserts a reshard between them and the layer gets slower
        # with no error at all. That single equality is what removed stage-03
        # row 135 from the profile.
        "norm_shard_cores": _NORM_SHARD_CORES,
        "norm_shard_feeds_qkv_directly": _norm_shard_config(m.hidden_size) == _width_sharded_l1(m.hidden_size),
        "decode_ccl_buffers_persistent": True,
        # -- what the precision config actually produced on device -------------
        # Read off the uploaded tensors, so these differ from
        # ``precision.<field>`` if any of the threading above is broken.
        "device_experts_gate_up_dtype": str(weights.experts.gate_up_proj.dtype),
        "device_experts_down_dtype": str(weights.experts.down_proj.dtype),
        "device_attention_qkv_dtype": str(weights.experts.attention.wqkv.dtype),
        "device_attention_wo_dtype": str(weights.experts.attention.wo.dtype),
        "device_attention_qkv_decode_dtype": (
            None if weights.experts.wqkv_decode is None else str(weights.experts.wqkv_decode.dtype)
        ),
        "device_router_dtype": str(weights.router.dtype),
        "device_norm_weight_dtype": str(weights.input_layernorm.dtype),
        # Bytes one layer's expert weights occupy per die -- the allocation-size
        # consequence of the two expert dtypes, in a form a sweep can diff.
        "device_expert_bytes_per_die": (
            _tensor_bytes(weights.experts.gate_up_proj) + _tensor_bytes(weights.experts.down_proj)
        ),
        "expert_math_fidelity": str(precision.experts_fidelity),
        "attention_math_fidelity": None if precision.attention_fidelity is None else str(precision.attention_fidelity),
        "router_window_math_fidelity": str(precision.router_window_fidelity),
        "ccl_dtype": str(precision.effective_ccl_dtype),
        "activation_dtype": str(precision.activation_dtype),
    }


__all__ = [
    "MESH_SHAPE",
    "NUM_DEVICES",
    "NUM_LINKS",
    "NUM_LINKS_DECODE",
    "TOPOLOGY",
    "MeshContext",
    "MeshDecoderConfig",
    "MultichipWeights",
    "all_reduce",
    "all_reduce_decode",
    "all_reduce_prefill",
    "build_local_sparsity",
    "create_mesh_kv_cache",
    "decoder_layer_decode_multichip",
    "decoder_layer_prefill_multichip",
    "fallback_audit",
    "head_interleaved_wqkv",
    "mesh_context",
    "decode_residual_norm",
    "moe_decode_multichip",
    "router_forward_multichip",
    "router_forward_threshold",
    "upload_multichip_weights",
]
