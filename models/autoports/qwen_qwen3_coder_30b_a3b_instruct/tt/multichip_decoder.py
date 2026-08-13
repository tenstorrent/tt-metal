# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Multichip TTNN decoder layer for Qwen3-Coder-30B-A3B-Instruct, 4 Blackhole dies.

Stage 03. The single-chip baseline is ``optimized_decoder.py`` -- every program
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
from dataclasses import dataclass

import torch

import ttnn
from models.common.modules.tt_ccl import TT_CCL

from .functional_decoder import (
    AttentionConfig,
    AttentionWeights,
    DecoderLayerConfig,
    KVCache,
    MoEConfig,
    attention_prefill,
)
from .optimized_decoder import (
    _DRAM_BANKS,
    ATTENTION_WEIGHT_DTYPE,
    EXPERT_IN0_BLOCK_W_DOWN,
    EXPERT_IN0_BLOCK_W_GATE_UP,
    EXPERT_WEIGHT_DTYPE,
    OptimizedWeights,
    _bank_row,
    _dram_sharded_ok,
    _expert_compute_kernel_config,
    _ones_column,
    _tuned_sparse_matmul_config,
    attention_decode_optimized,
    moe_prefill_optimized,
)

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

# Both ethernet links on every hop are used. Worth 1.04x at decode size (it is
# latency-bound there) and 1.84x at 2 MB, i.e. free money in prefill.
NUM_LINKS = 2

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


def mesh_context(mesh_device) -> MeshContext:
    """Build the CCL context for the 4-die mesh, asserting the shape."""
    n = mesh_device.get_num_devices()
    assert n == NUM_DEVICES, (
        f"multichip_decoder targets exactly {NUM_DEVICES} dies (the full P300_X2 mesh); got {n}. "
        "Smaller meshes are out of scope by design -- the head split, expert window and ring "
        "topology are all chosen against the 4-die shape."
    )
    return MeshContext(mesh=mesh_device, ccl=TT_CCL(mesh_device))


def all_reduce(x: ttnn.Tensor, ctx: MeshContext) -> ttnn.Tensor:
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
    """
    scattered = ttnn.experimental.reduce_scatter_minimal_async(
        x,
        persistent_output_buffers=None,
        dim=3,
        multi_device_global_semaphore=ctx.ccl.get_and_cycle_rs_semaphore_handles(),
        barrier_semaphore=ctx.ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=ctx.num_links,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ctx.topology,
    )
    gathered = ttnn.experimental.all_gather_async(
        scattered,
        dim=3,
        multi_device_global_semaphore=ctx.ccl.get_and_cycle_ag_semaphore_handles(),
        barrier_semaphore=ctx.ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=ctx.num_links,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ctx.topology,
    )
    ttnn.deallocate(scattered)
    return gathered


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


def _sdpa_program_config(device):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=32,
        k_chunk_size=32,
        max_cores_per_head_batch=_SDPA_MAX_CORES_PER_HEAD,
    )


def _exact_matmul_config(device):
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
        math_fidelity=ttnn.MathFidelity.HiFi4,
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
    dtype = expert_dtype if expert_dtype is not None else EXPERT_WEIGHT_DTYPE

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

    def dram_sharded(t: torch.Tensor, dim: int, k: int, n_local: int) -> ttnn.Tensor:
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
            ATTENTION_WEIGHT_DTYPE,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(_bank_row(_DRAM_BANKS), [k, n_local // _DRAM_BANKS], ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )

    experts = OptimizedWeights(
        # [E, 2I, H] -> [1, E, H, 2I], sharded on the expert dim.
        gate_up_proj=shard(torch_weights["experts_gate_up"].transpose(-2, -1).unsqueeze(0), 1, dtype),
        # [E, H, I] -> [1, E, I, H], sharded on the expert dim.
        down_proj=shard(torch_weights["experts_down"].transpose(-2, -1).unsqueeze(0), 1, dtype),
        attention=AttentionWeights(
            # Column split (head-interleaved, see head_interleaved_wqkv).
            wqkv=shard(wqkv, -1, ATTENTION_WEIGHT_DTYPE),
            # Row split by Q head. Contiguous *because* the Q head assignment
            # above is contiguous per die: die d owns rows 1024d..1024d+1023.
            wo=shard(wo, -2, ATTENTION_WEIGHT_DTYPE),
            q_norm=replicate(as_4d(torch_weights["q_norm"], pad_to_4d=True), ttnn.bfloat16),
            k_norm=replicate(as_4d(torch_weights["k_norm"], pad_to_4d=True), ttnn.bfloat16),
        ),
        wqkv_decode=dram_sharded(wqkv, -1, k_qkv, n_qkv),
        wo_decode=dram_sharded(wo, -2, k_o, n_o),
    )

    router = torch_weights["router"]
    return MultichipWeights(
        input_layernorm=replicate(torch_weights["input_layernorm"].reshape(1, 1, 1, -1), ttnn.bfloat16),
        post_attention_layernorm=replicate(
            torch_weights["post_attention_layernorm"].reshape(1, 1, 1, -1), ttnn.bfloat16
        ),
        router=replicate(router.T.contiguous().reshape(1, 1, router.shape[1], router.shape[0]), ttnn.bfloat16),
        expert_window=_expert_window_matrix(mesh_device, config.global_config.moe.num_experts, n),
        experts=experts,
    )


def create_mesh_kv_cache(
    mesh_device,
    config: MeshDecoderConfig,
    max_batch: int,
    max_seq_len: int,
    block_size: int | None = None,
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
            dtype=ttnn.bfloat16,
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


def router_forward_multichip(
    x: ttnn.Tensor,
    w_router: ttnn.Tensor,
    window: ttnn.Tensor,
    config: MoEConfig,
    local_moe: MoEConfig,
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
    top_logits, top_indices = ttnn.topk(logits, k=config.num_experts_per_tok, dim=-1, largest=True, sorted=True)

    top_max = ttnn.slice(top_logits, [0, 0, 0, 0], [1, 1, top_logits.shape[2], 1])
    exp_logits = ttnn.exp(ttnn.sub(top_logits, top_max))

    zeros = ttnn.typecast(ttnn.zeros_like(logits), ttnn.bfloat16)
    dense = ttnn.scatter(zeros, dim=-1, index=top_indices, src=ttnn.typecast(exp_logits, ttnn.bfloat16))

    # The denominator is the sum over all 128 -- which is the sum over the 8
    # survivors, since the scatter fills a field of exact zeros -- and must stay
    # global: normalising within a window would renormalise each die's share to
    # 1 and the four contributions would sum to 4.
    total = ttnn.matmul(dense, _ones_column(x.device(), config.num_experts), dtype=ttnn.bfloat16)
    local = ttnn.matmul(dense, window, dtype=ttnn.bfloat16, compute_kernel_config=_exact_matmul_config(x.device()))
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
    compute_config = _expert_compute_kernel_config(x.device())
    gate_up_config = _tuned_sparse_matmul_config(1, 2 * inter, hidden_size, EXPERT_IN0_BLOCK_W_GATE_UP)
    down_config = _tuned_sparse_matmul_config(1, hidden_size, inter, EXPERT_IN0_BLOCK_W_DOWN)

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
        dtype=ttnn.bfloat16,
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
        dtype=ttnn.bfloat16,
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
        normed, weights.experts.attention, config.local_attention, cos_cache, sin_cache, kv_cache, user_id
    )
    ttnn.deallocate(normed)
    attn_out = all_reduce_prefill(attn_partial, ctx)
    ttnn.deallocate(attn_partial)
    hidden = ttnn.add(x, attn_out)
    ttnn.deallocate(attn_out)

    normed = ttnn.rms_norm(hidden, weight=weights.post_attention_layernorm, epsilon=eps)
    routing = router_forward_multichip(
        normed, weights.router, weights.expert_window, config.global_config.moe, config.local_moe
    )
    moe_partial = moe_prefill_optimized(normed, routing, weights.experts, config.local_moe, sparsity)
    ttnn.deallocate(normed)
    ttnn.deallocate(routing)
    moe_out = all_reduce_prefill(moe_partial, ctx)
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
) -> ttnn.Tensor:
    """Decode one token per user on the mesh. ``x`` / return ``[1, 1, B, 2048]``.

    Input and output layouts are the same replicated tensor, which is the point:
    48 of these stack with no boundary conversion, and the stacked model pays
    the two all-reduces per layer and nothing else.
    """
    eps = config.global_config.rms_norm_eps

    normed = ttnn.rms_norm(x, weight=weights.input_layernorm, epsilon=eps)
    attn_partial = attention_decode_optimized(
        normed,
        weights.experts,
        config.local_attention,
        cos_cache,
        sin_cache,
        kv_cache,
        current_pos,
        token_index,
        # Only the contiguous cache needs the cap; the paged path runs at the op
        # default, which is the configuration every published decode number here
        # was measured at. See _sdpa_program_config.
        sdpa_program_config=None if kv_cache.is_paged else _sdpa_program_config(x.device()),
    )
    ttnn.deallocate(normed)
    attn_out = all_reduce_decode(attn_partial, ctx)
    ttnn.deallocate(attn_partial)
    hidden = ttnn.add(x, attn_out)
    ttnn.deallocate(attn_out)

    normed = ttnn.rms_norm(hidden, weight=weights.post_attention_layernorm, epsilon=eps)
    routing = router_forward_multichip(
        normed, weights.router, weights.expert_window, config.global_config.moe, config.local_moe
    )
    moe_partial = moe_decode_multichip(normed, routing, weights.experts, config.local_moe)
    ttnn.deallocate(normed)
    ttnn.deallocate(routing)
    moe_out = all_reduce_decode(moe_partial, ctx)
    ttnn.deallocate(moe_partial)

    out = ttnn.add(hidden, moe_out)
    ttnn.deallocate(hidden)
    ttnn.deallocate(moe_out)
    return out


def fallback_audit(weights: MultichipWeights, config: MeshDecoderConfig, batch: int) -> dict:
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

    Returned as data so a test can assert on it and the work log can quote it.
    """
    a = config.local_attention
    m = config.local_moe
    k_qkv = int(weights.experts.wqkv_decode.shape[-2]) if weights.experts.wqkv_decode is not None else None
    n_qkv = int(weights.experts.wqkv_decode.shape[-1]) if weights.experts.wqkv_decode is not None else None
    k_o = int(weights.experts.wo_decode.shape[-2]) if weights.experts.wo_decode is not None else None
    n_o = int(weights.experts.wo_decode.shape[-1]) if weights.experts.wo_decode is not None else None
    gate_up = _tuned_sparse_matmul_config(1, 2 * m.moe_intermediate_size, m.hidden_size, EXPERT_IN0_BLOCK_W_GATE_UP)
    down = _tuned_sparse_matmul_config(1, m.hidden_size, m.moe_intermediate_size, EXPERT_IN0_BLOCK_W_DOWN)
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
    }


__all__ = [
    "MESH_SHAPE",
    "NUM_DEVICES",
    "NUM_LINKS",
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
    "moe_decode_multichip",
    "router_forward_multichip",
    "upload_multichip_weights",
]
