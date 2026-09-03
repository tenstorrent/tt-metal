# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Kimi-K2.6 single-MoE-layer op2op harness -- weight-free, version-portable.

WHAT THIS IS
    A standalone replay of the op sequence one Kimi-K2.6 chunked-prefill MoE layer issues for
    chunk 0 on an 8x4 Blackhole galaxy, with every tensor created RANDOMLY (no checkpoint, no TTNN
    weight cache, no golden trace).  Nothing here is numerically meaningful -- the point is the
    DISPATCH SHAPE of the layer: the same ops, in the same order, at the same shapes, dtypes,
    layouts, memory configs, matmul program configs and CCL topologies.  That makes per-op
    "op to op latency" comparable across two tt-metal revisions, which is how a host-dispatch
    regression is found.

    This file deliberately imports NOTHING from models/ -- only ttnn, torch, pytest and the root
    conftest's `mesh_device` / `device_params` fixtures.  It is meant to be dropped, byte-identical,
    into an older checkout and run there.

SHAPES (measured from a real run; see the table in `_report_reference` at the bottom)
    mesh 8x4, sp_axis=0 (8), tp_axis=1 (4); chunk 5120 tokens -> 640 per SP chip.
    hidden 7168 -> 1792 per TP chip; 64 heads -> 16 per TP chip; q_lora 1536; kv_lora 512;
    qk_nope 128; qk_rope 64; v_head 128; 384 routed experts, top-8; MoE intermediate 2048.

WHAT IS SKIPPED, AND WHY
    Ops that do not exist in BOTH revisions under comparison are skipped rather than substituted,
    so the two runs execute the identical program.  Against 2026-01-01 (tt-metal 6223e8c20a0) the
    absent ops are:
        ring_mla                                  (chunked MLA attention)
        deepseek_prefill.rotary_embedding_indexed (block-cyclic RoPE)
        deepseek_prefill.update_padded_kv_cache   (chunked KV write)
        deepseek_prefill.zero_padded_kv_cache     (migration pad zero)
        deepseek_prefill.moe_grouped_topk         (device gate)
        deepseek_prefill.masked_bincount          (routing histogram)
        deepseek_prefill.offset_cumsum            (expert region offsets)
        deepseek_prefill.dispatch                 (token dispatch)
        deepseek_prefill.unified_routed_expert_moe (fused expert FFN)
        deepseek_prefill.combine                  (token combine)
        deepseek_prefill.post_combine_reduce      (weighted top-k sum)
        experimental.high_bw_all_gather           (see below)
    Every skip is announced at runtime by `_skip`, so the log says what was left out.

    high_bw_all_gather is the one deliberate SUBSTITUTION rather than a skip: the layer issues four
    of them and dropping them would remove the TP gather from the graph entirely.  They are replaced
    by `ttnn.experimental.all_gather_async` at the same shape/axis, which exists in both revisions.
    Every such swap is announced by `_subst`.

USAGE
    Wall-clock only (no profiler), 5 measured iterations:
        pytest tests/ttnn/unit_tests/test_kimi_moe_layer_op2op.py -k moe_layer -s

    Per-op device + op2op breakdown (this is the regression signal):
        python -m tracy -v -r --op-support-count 20000 -o /tmp/op2op \
            -m pytest tests/ttnn/unit_tests/test_kimi_moe_layer_op2op.py -s
        # then read <out>/reports/*/ops_perf_results_*.csv.  Rows are in execution order; one
        # logical op = consecutive rows until OP CODE changes or a DEVICE ID repeats (GLOBAL CALL
        # COUNT is a per-device runtime id and cannot be used to group).  The `iter_<n>_start` /
        # `iter_<n>_end` signposts bracket each iteration; iteration 0 is the JIT-compile one and
        # must be discarded.
    Do NOT pass --sync-host-device: it serialises every op and inflates op2op several-fold.

    FABRIC: defaults to FABRIC_2D with all-Linear collectives, because the 2026-01-01 revision binds
    FabricConfig only up to FABRIC_2D -- production's FABRIC_2D_TORUS_XY (Ring on the TP axis) cannot
    be expressed there at all, so it is not a valid A/B point.  KIMI_OP2OP_FABRIC=torus_xy switches a
    modern build to the production fabric; those numbers are only comparable to other torus runs.

    A/B against an older tree: copy this file there, and give each tree its OWN kernel cache --
        TT_METAL_KERNEL_CACHE=/some/dir/<tree-name>
    The JIT build key does not include the tree root, so two checkouts otherwise silently trade
    compiled kernels and the comparison measures the wrong binaries.

Env knobs:
    KIMI_OP2OP_ITERS         measured iterations after the warmup (default 5)
    KIMI_OP2OP_VERBOSE       1 = print every op's in/out shapes + memory configs (default 1)
    KIMI_OP2OP_INCLUDE_TOPK  1 = also run ttnn.topk + ttnn.softmax where the real layer runs
                             moe_grouped_topk.  Both exist in either revision, but they are NOT
                             what the model dispatches, so this is off by default.
    KIMI_OP2OP_FABRIC        "2d" (default) or "torus_xy"; see FABRIC above.
    KIMI_OP2OP_INDEX_DTYPE   "bfloat16" (default) or "uint16" for the routing-index stand-in; see
                             the INDEX_DTYPE comment for why uint16 is not the default.
    KIMI_OP2OP_SYNC_PER_OP   1 = synchronize after each op so its per-op number includes its own
                             device time.  Serialises the pipeline (absolute times inflate), but
                             attributes device time per op with no profiler.  See SYNC_PER_OP.
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from tracy import signpost

# ---------------------------------------------------------------------------------------------
# Kimi-K2.6 geometry for chunk 0 on mesh 8x4 (sp_axis=0, tp_axis=1)
# ---------------------------------------------------------------------------------------------
SP, TP = 8, 4
SP_AXIS, TP_AXIS = 0, 1
NUM_LINKS = 2

CHUNK = 5 * 1024  # global tokens per chunk
SEQ = CHUNK // SP  # 640 local tokens per SP chip

EMB = 7168
EMB_TP = EMB // TP  # 1792

NUM_HEADS = 64
HEADS_TP = NUM_HEADS // TP  # 16
Q_LORA = 1536
KV_LORA = 512
QK_NOPE = 128
QK_ROPE = 64
QK_HEAD = QK_NOPE + QK_ROPE  # 192
V_HEAD = 128

N_ROUTED_EXPERTS = 384
TOPK = 8
MOE_INTERMEDIATE = 2048
SHARED_HIDDEN_TP = MOE_INTERMEDIATE // TP  # 512

RMS_EPS = 1e-5

# Kimi's q_b_proj output width per TP chip: heads_tp * qk_head_dim
Q_B_N = HEADS_TP * QK_HEAD  # 3072
# o_proj input width per TP chip: heads_tp * v_head_dim
O_PROJ_K = HEADS_TP * V_HEAD  # 2048

TILE = 32
# 11x10 on Blackhole: the full grid is 12x10 but di/dt throttling caps the model at 11 columns.
MLA_GRID = ttnn.CoreCoord(11, 10)
SHARED_EXPERT_GRID = ttnn.CoreCoord(11, 9)
GATE_GRID = ttnn.CoreCoord(6, 10)


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


ITERS = _env_int("KIMI_OP2OP_ITERS", 5)
VERBOSE = _env_int("KIMI_OP2OP_VERBOSE", 1) == 1
INCLUDE_TOPK = _env_int("KIMI_OP2OP_INCLUDE_TOPK", 0) == 1


# --- fabric / CCL topology -------------------------------------------------------------------
# Production Kimi runs FABRIC_2D_TORUS_XY, which rings the TP axis.  The 2026-01-01 revision's
# python bindings expose FabricConfig only up to FABRIC_2D -- the torus values exist in the C++ enum
# but are not bound -- so a torus run is simply not expressible there.  Default therefore to the
# configuration BOTH revisions can express (FABRIC_2D, all-Linear collectives), which is what makes
# the A/B measure the same program on both sides.  Set KIMI_OP2OP_FABRIC=torus_xy on a revision that
# supports it to measure the production fabric instead (that number is NOT comparable to a 2d one).
FABRIC_CHOICE = os.environ.get("KIMI_OP2OP_FABRIC", "2d").lower()


def _resolve_fabric():
    """Return (fabric_config, sp_topology, tp_topology) for FABRIC_CHOICE.

    Falls back to FABRIC_2D with a warning when the requested torus config is not bound in this
    build, so the same command line runs on either revision rather than erroring at import.
    """
    if FABRIC_CHOICE in ("torus_xy", "torus"):
        cfg = getattr(ttnn.FabricConfig, "FABRIC_2D_TORUS_XY", None)
        if cfg is not None:
            # per_axis_topology(FABRIC_2D_TORUS_XY): rows (SP) Linear, cols (TP) Ring.
            return cfg, ttnn.Topology.Linear, ttnn.Topology.Ring
        logger.warning("KIMI_OP2OP_FABRIC=torus_xy but FABRIC_2D_TORUS_XY is not bound here; using FABRIC_2D")
    elif FABRIC_CHOICE != "2d":
        logger.warning(f"unknown KIMI_OP2OP_FABRIC={FABRIC_CHOICE!r}; using 2d")
    return ttnn.FabricConfig.FABRIC_2D, ttnn.Topology.Linear, ttnn.Topology.Linear


FABRIC_CONFIG, SP_TOPOLOGY, TP_TOPOLOGY = _resolve_fabric()


# --- index/score stand-in dtype ---------------------------------------------------------------
# The model's routing indices are UINT16 (that is what the gate's top-k emits and what the untilize
# and the L1->DRAM copy then carry).  UINT16 untilizes on both revisions, but the 2026-01-01
# ttnn.copy rejects it outright -- "ttnn.copy only supports float, bfloat and int32 inputs but got
# DataType::UINT16" -- so a UINT16 run cannot execute the same program on both sides.  Default to
# bfloat16, which is byte-for-byte the same width (so untilize and copy move the same bytes) and is
# accepted everywhere.  Set KIMI_OP2OP_INDEX_DTYPE=uint16 for a single-revision run that wants the
# production dtype exactly.
_INDEX_DTYPE_NAME = os.environ.get("KIMI_OP2OP_INDEX_DTYPE", "bfloat16").lower()
INDEX_DTYPE = ttnn.uint16 if _INDEX_DTYPE_NAME == "uint16" else ttnn.bfloat16

# --- per-op attribution mode -------------------------------------------------------------------
# Default (off): no sync inside the sequence, so the per-op numbers are pure HOST enqueue cost and
# the iteration wall time is honest.  On this op subset that split turns out to be ~8% host / ~92%
# device, i.e. a revision-to-revision wall change lives on the device side and the host table alone
# will not find it.
# SYNC_PER_OP=1: synchronize after every op, so each op's number becomes enqueue + that op's own
# device execution.  This SERIALISES the pipeline, so absolute times inflate and the iteration total
# is no longer a fair wall-clock -- but it attributes device time per op without a profiler, and the
# inflation is the same on both revisions, so the A/B ranking still holds.  Use it to find which op
# moved, then confirm the magnitude with an un-synced run or with tracy.
SYNC_PER_OP = os.environ.get("KIMI_OP2OP_SYNC_PER_OP", "0") == "1"


# ---------------------------------------------------------------------------------------------
# op logging / availability
# ---------------------------------------------------------------------------------------------
class Recorder:
    """Numbers each op as it is issued, times its host dispatch, and prints its in/out shapes.

    The numbering matches the order of the non-signpost op groups in a tracy ops CSV, so op N here
    is op N there.  Skips and substitutions are recorded too, so a log from an old revision states
    exactly which ops were left out.

    PER-OP TIMING: `call(label, fn, *ins)` brackets the ttnn call with perf_counter.  With no
    device sync in the sequence, that measures the HOST cost of getting the op onto the command
    queue -- i.e. exactly the quantity a dispatch regression shows up in, and the one that dominates
    this (deliberately device-light) op subset.  It is not device time: read the tracy CSV for that.
    An op whose enqueue blocks on a full command queue absorbs the stall, which is where the time
    genuinely went.  Two perf_counter calls per op perturb nothing measurable.
    """

    def __init__(self):
        self.n = 0
        self.skipped = []
        self.substituted = []
        self.lines = []
        self.host_ns = []  # per-op time, index-aligned with self.lines
        self.mesh = None  # set by the driver so SYNC_PER_OP can synchronize

    def call(self, label, fn, *ins):
        """Issue one op through `fn`, timing it, then record/print it.

        Under SYNC_PER_OP the timer also covers a device synchronize, so the number includes this
        op's device execution (see the SYNC_PER_OP comment for what that does and does not mean).
        """
        t0 = time.perf_counter_ns()
        out = fn()
        if SYNC_PER_OP and getattr(self, "mesh", None) is not None:
            ttnn.synchronize_device(self.mesh)
        dt = time.perf_counter_ns() - t0
        self.host_ns.append(dt)
        return self.op(label, out, *ins)

    @staticmethod
    def _desc(t):
        if t is None:
            return "None"
        if not isinstance(t, ttnn.Tensor):
            return repr(t)
        mc = t.memory_config()
        mem = f"{mc.buffer_type.name}_{mc.memory_layout.name}"
        return f"{list(t.shape)} {t.dtype.name} {t.layout.name} {mem}"

    def op(self, label, out, *ins):
        i = self.n
        self.n += 1
        line = f"[{i:3d}] {label}"
        self.lines.append((i, label))
        if VERBOSE:
            logger.info(line)
            for k, t in enumerate(ins):
                logger.info(f"        in{k} : {self._desc(t)}")
            outs = out if isinstance(out, (tuple, list)) else (out,)
            for k, t in enumerate(outs):
                logger.info(f"        out{k}: {self._desc(t)}")
        return out

    def skip(self, name, reason):
        if name not in [s for s, _ in self.skipped]:
            self.skipped.append((name, reason))
            logger.warning(f"      SKIP {name} -- {reason}")

    def subst(self, original, replacement):
        key = (original, replacement)
        if key not in self.substituted:
            self.substituted.append(key)
            logger.warning(f"      SUBST {original} -> {replacement}")


def _available(dotted):
    """True when ttnn exposes `dotted` (e.g. "experimental.high_bw_all_gather") in this build."""
    obj = ttnn
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return False
    return True


# The ops the real layer runs that must be absent-tolerant.  Presence is probed at runtime, so this
# same file runs unchanged on a revision that has them and on one that does not.
MODEL_ONLY_OPS = [
    ("transformer.ring_mla", "chunked MLA attention"),
    ("experimental.deepseek_prefill.rotary_embedding_indexed", "block-cyclic RoPE"),
    ("experimental.deepseek_prefill.update_padded_kv_cache", "chunked KV cache write"),
    ("experimental.deepseek_prefill.zero_padded_kv_cache", "migration pad zero"),
    ("experimental.deepseek_prefill.moe_grouped_topk", "device MoE gate (top-k + score)"),
    ("experimental.deepseek_prefill.masked_bincount", "routing histogram"),
    ("experimental.deepseek_prefill.offset_cumsum", "expert region offsets"),
    ("experimental.deepseek_prefill.dispatch", "MoE token dispatch"),
    ("experimental.deepseek_prefill.unified_routed_expert_moe", "fused routed-expert FFN"),
    ("experimental.deepseek_prefill.combine", "MoE token combine"),
    ("experimental.deepseek_prefill.post_combine_reduce", "weighted top-k sum"),
]

# Ops this harness itself needs.  If one is missing the test cannot run at all, so say so clearly
# instead of dying inside a call.
REQUIRED_OPS = [
    "linear",
    "matmul",
    "rms_norm",
    "rms_norm_pre_all_gather",
    "rms_norm_post_all_gather",
    "slice",
    "concat",
    "add",
    "multiply_",
    "typecast",
    "reshape",
    "to_layout",
    "to_memory_config",
    "all_gather",
    "reduce_scatter",
    "experimental.all_gather_async",
    "experimental.reduce_scatter_minimal_async",
    "experimental.nlp_create_qkv_heads",
    "experimental.nlp_concat_heads",
    "experimental.fast_reduce_nc",
]


# ---------------------------------------------------------------------------------------------
# tensor factories -- everything random, replicated across the mesh
# ---------------------------------------------------------------------------------------------
def _rand(mesh, shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=None, zeros=False):
    """A per-chip tensor of `shape`, replicated across the mesh.

    Replication (not sharding) is intentional: the harness measures dispatch, and every op below is
    driven by the PER-CHIP shape plus an explicit cluster_axis, so how the data got there is
    irrelevant.  Replicating keeps the factory trivial and identical on both revisions.
    """
    t = torch.zeros(shape) if zeros else torch.randn(shape)
    return ttnn.from_torch(
        t,
        device=mesh,
        dtype=dtype,
        layout=layout,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _rand_int(mesh, shape, dtype, layout=ttnn.ROW_MAJOR_LAYOUT, high=8, memory_config=None):
    """Integer tensor; falls back to bfloat16 if this build's from_torch cannot make `dtype`.

    Only the byte width matters for dispatch cost (uint16 and bfloat16 are both 2 bytes), so the
    fallback keeps an older revision runnable without changing what is being measured.
    """
    t = torch.randint(0, high, shape, dtype=torch.int32)
    try:
        return ttnn.from_torch(
            t,
            device=mesh,
            dtype=dtype,
            layout=layout,
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
    except Exception as e:  # pragma: no cover - build-dependent
        logger.warning(f"from_torch({dtype}) unsupported here ({e}); falling back to bfloat16")
        return ttnn.from_torch(
            t.to(torch.float32),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=layout,
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )


# ---------------------------------------------------------------------------------------------
# CCL plumbing -- mirrors models/demos/deepseek_v3_d_p/tt/tt_ccl.py:TT_CCL
# ---------------------------------------------------------------------------------------------
class Ccl:
    """Persistent global semaphores for the async collectives.

    Counts and the two-deep round-robin match the model's TT_CCL exactly (all-gather wants 2
    semaphores, reduce-scatter 3, plus one barrier), because an under-provisioned pool changes what
    the op does rather than just how fast it does it.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        grid = mesh.compute_with_storage_grid_size()
        self.cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        # index 0 = cluster_axis 0, 1 = cluster_axis 1, 2 = no cluster axis
        self._barrier = [[], [], []]
        self._ag = [[], [], []]
        self._rs = [[], [], []]
        self._bi, self._ai, self._ri = [0, 0, 0], [0, 0, 0], [0, 0, 0]
        for axis in range(3):
            for _ in range(2):
                self._barrier[axis].append(ttnn.create_global_semaphore(mesh, self.cores, 0))
                self._ag[axis].append([ttnn.create_global_semaphore(mesh, self.cores, 0) for _ in range(2)])
                self._rs[axis].append([ttnn.create_global_semaphore(mesh, self.cores, 0) for _ in range(3)])

    def barrier(self, cluster_axis):
        i = 2 if cluster_axis is None else cluster_axis
        cur = self._bi[i]
        self._bi[i] = (cur + 1) % 2
        return self._barrier[i][cur]

    def ag(self, cluster_axis):
        i = 2 if cluster_axis is None else cluster_axis
        cur = self._ai[i]
        self._ai[i] = (cur + 1) % 2
        return self._ag[i][cur]

    def rs(self, cluster_axis):
        i = 2 if cluster_axis is None else cluster_axis
        cur = self._ri[i]
        self._ri[i] = (cur + 1) % 2
        return self._rs[i][cur]


def _all_gather_async(ccl, t, dim, cluster_axis, topology, memory_config=None):
    return ttnn.experimental.all_gather_async(
        t,
        dim=dim,
        cluster_axis=cluster_axis,
        multi_device_global_semaphore=ccl.ag(cluster_axis),
        barrier_semaphore=ccl.barrier(cluster_axis),
        num_links=NUM_LINKS,
        topology=topology,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
    )


def _reduce_scatter_async(ccl, t, dim, cluster_axis, topology, memory_config=None, persistent=None):
    return ttnn.experimental.reduce_scatter_minimal_async(
        t,
        persistent_output_buffers=persistent,
        dim=dim,
        multi_device_global_semaphore=ccl.rs(cluster_axis),
        barrier_semaphore=ccl.barrier(cluster_axis),
        num_links=NUM_LINKS,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=topology,
        cluster_axis=cluster_axis,
    )


# ---------------------------------------------------------------------------------------------
# matmul program configs -- copied verbatim from the model's tuned tables for seq_len_local=640
# (models/demos/deepseek_v3_d_p/tt/mla/mla_config.py @ num_heads=64, q_lora_rank=1536, chunked;
#  tt/moe/tt_shared_expert.py:get_bh_program_configs; tt/moe/tt_moe_gate_prefill.py)
# ---------------------------------------------------------------------------------------------
def _mm_2d(grid, in0_block_w, osh, osw, pcM, pcN, out_block_h=None, out_block_w=None, fuse_batch=False, act=None):
    kwargs = dict(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=osh,
        out_subblock_w=osw,
        per_core_M=pcM,
        per_core_N=pcN,
        transpose_mcast=False,
        fuse_batch=fuse_batch,
        fused_activation=act,
    )
    if out_block_h is not None:
        kwargs["out_block_h"] = out_block_h
    if out_block_w is not None:
        kwargs["out_block_w"] = out_block_w
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(**kwargs)


def _mm_1d(grid, in0_block_w, osh, osw, pcM, pcN, out_block_h=None, out_block_w=None, fuse_batch=False, act=None):
    kwargs = dict(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=osh,
        out_subblock_w=osw,
        per_core_M=pcM,
        per_core_N=pcN,
        fuse_batch=fuse_batch,
        mcast_in0=False,
        fused_activation=act,
    )
    if out_block_h is not None:
        kwargs["out_block_h"] = out_block_h
    if out_block_w is not None:
        kwargs["out_block_w"] = out_block_w
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(**kwargs)


def _mm_batched(grid, in0_block_w, osh, osw, pcM, pcN):
    """Batched per-head matmul (wkv_b1 / wkv_b2): in0 batch 1, in1 batch num_heads."""
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=osh,
        out_subblock_w=osw,
        per_core_M=pcM,
        per_core_N=pcN,
    )


def _program_configs():
    return {
        # MLA, seq_len_local = 640
        "q_a_proj": _mm_2d(MLA_GRID, 8, 1, 5, 2, 5, out_block_h=2, out_block_w=5),
        "q_b_proj": _mm_2d(MLA_GRID, 8, 1, 3, 2, 9, out_block_h=2, out_block_w=9),
        "wkv_b1": _mm_batched(MLA_GRID, 2, 2, 4, 4, 16),
        "kv_a_proj": _mm_2d(MLA_GRID, 14, 2, 1, 2, 2, out_block_h=2, out_block_w=2),
        "wkv_b2": _mm_batched(MLA_GRID, 2, 4, 1, 4, 4),
        "o_proj": _mm_2d(MLA_GRID, 8, 1, 7, 2, 21, out_block_h=2, out_block_w=21),
        # MoE gate: interleaved in0 -> the 2D table entry (640, 1792, 384)
        "gate": _mm_2d(GATE_GRID, 8, 1, 2, 2, 2, out_block_h=2, out_block_w=2, fuse_batch=True),
        # Shared expert: m_tiles = 640/32 = 20, num_cores = 20 -> per_core_M = 1;
        # gate/up N tiles = 512/32 = 16; down N tiles = 7168/32 = 224.
        "shared_gate": _mm_1d(
            SHARED_EXPERT_GRID, 4, 1, 8, 1, SHARED_HIDDEN_TP // TILE, act=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
        ),
        "shared_up": _mm_1d(SHARED_EXPERT_GRID, 4, 1, 8, 1, SHARED_HIDDEN_TP // TILE),
        "shared_down": _mm_1d(SHARED_EXPERT_GRID, 1, 1, 8, 1, EMB // TILE),
    }


# ---------------------------------------------------------------------------------------------
# persistent (per-mesh, built once) state
# ---------------------------------------------------------------------------------------------
class LayerState:
    """Weights, per-op program configs and CCL handles -- built once, reused every iteration.

    Built once on purpose: the model builds these at construction too, so rebuilding them per
    iteration would add host work the real layer never pays and pollute the op2op measurement.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        self.ccl = Ccl(mesh)
        self.pc = _program_configs()

        arch = mesh.arch()
        self.ck_hifi2 = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.ck_hifi4_fp32 = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        # The gate's own compute config (HiFi4 + fp32 dest, no packer L1 acc).
        self.ck_gate = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        bf8 = ttnn.bfloat8_b
        # --- MLA weights (per TP chip) ---
        self.w_q_a = _rand(mesh, (1, 1, EMB_TP, Q_LORA), bf8)
        self.w_q_b = _rand(mesh, (1, 1, Q_LORA, Q_B_N), bf8)
        self.w_kv_a = _rand(mesh, (1, 1, EMB_TP, KV_LORA + QK_ROPE), bf8)
        self.w_wkv_b1 = _rand(mesh, (1, HEADS_TP, QK_NOPE, KV_LORA), bf8)
        self.w_wkv_b2 = _rand(mesh, (1, HEADS_TP, KV_LORA, V_HEAD), bf8)
        self.w_o = _rand(mesh, (1, 1, O_PROJ_K, EMB), bf8)

        # --- norm weights: [1, 1, dim/32, 32] ROW_MAJOR, as the model caches them ---
        self.w_attn_norm = _rand(mesh, (1, 1, EMB_TP // TILE, TILE), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self.w_ffn_norm = _rand(mesh, (1, 1, EMB_TP // TILE, TILE), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self.w_q_a_norm = _rand(mesh, (1, 1, Q_LORA // TILE, TILE), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self.w_kv_a_norm = _rand(mesh, (1, 1, KV_LORA // TILE, TILE), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)

        # --- MoE weights (per chip) ---
        self.w_gate = _rand(mesh, (1, 1, EMB_TP, N_ROUTED_EXPERTS), ttnn.bfloat16)
        self.w_sh_gate = _rand(mesh, (1, 1, EMB, SHARED_HIDDEN_TP), bf8)
        self.w_sh_up = _rand(mesh, (1, 1, EMB, SHARED_HIDDEN_TP), bf8)
        self.w_sh_down = _rand(mesh, (1, 1, SHARED_HIDDEN_TP, EMB), bf8)

        # --- stand-ins for the outputs of the model-only ops this harness skips -------------------
        # Built ONCE, here, and never re-created inside the measured loop.  Creating them per
        # iteration cost ~110 ms of host torch.randn per iteration (the two big ones are 5.2M and
        # 4.6M elements), which swamped the ttnn time being measured and, worse, would have made the
        # A/B a comparison of the two checkouts' torch/numpy builds.  Reusing one buffer keeps the
        # DRAM addresses fixed across iterations, which is also what the persistent-buffer ops want.
        self.syn_attn_out = _rand(mesh, (1, HEADS_TP, SEQ, KV_LORA), ttnn.bfloat16)  # ring_mla output
        self.syn_combined = _rand(mesh, (1, 1, SEQ, EMB), ttnn.bfloat16)  # post_combine_reduce output
        self.syn_hist = _rand_int(mesh, (1, 1, 1, N_ROUTED_EXPERTS), ttnn.uint32, high=64)  # bincount out
        if INDEX_DTYPE == ttnn.uint16:
            self.syn_idx_tile = _rand_int(
                mesh, (1, 1, SEQ, TOPK), ttnn.uint16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
            )
        else:
            self.syn_idx_tile = _rand(
                mesh, (1, 1, SEQ, TOPK), INDEX_DTYPE, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
            )
        self.syn_scores_tile = _rand(
            mesh, (1, 1, SEQ, TOPK), ttnn.bfloat16, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
        )

        # Shared-expert reduce_scatter persistent intermediate.  The model's shared expert runs this
        # collective Linear with a stable-address intermediate; the Linear path wants a doubled
        # leading dim for the forward/backward halves, hence the [2] + shape.
        self.sh_rs_intermediate = _rand(mesh, (2, 1, SEQ, EMB), ttnn.bfloat16, zeros=True)

    @property
    def sp_topology(self):
        return SP_TOPOLOGY

    @property
    def tp_topology(self):
        return TP_TOPOLOGY


# ---------------------------------------------------------------------------------------------
# the layer
# ---------------------------------------------------------------------------------------------
def _mla(rec, st, x):
    """attn_norm -> MLA (minus ring_mla / RoPE / KV write) -> o_proj -> TP reduce-scatter -> residual."""
    ccl, pc = st.ccl, st.pc
    tp_t, dram, l1 = st.tp_topology, ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG

    signpost(header="MLA_START")

    # --- distributed attn_norm: pre_all_gather -> TP gather of stats -> post_all_gather ---
    stats = rec.call(
        "rms_norm_pre_all_gather (attn_norm)",
        lambda: ttnn.rms_norm_pre_all_gather(x, dtype=ttnn.bfloat16),
        x,
    )
    rec.subst("experimental.high_bw_all_gather", "experimental.all_gather_async")
    g_stats = rec.call(
        "all_gather_async (attn_norm stats, TP)",
        lambda: _all_gather_async(ccl, stats, dim=3, cluster_axis=TP_AXIS, topology=tp_t),
        stats,
    )
    attn_norm_out = rec.call(
        "rms_norm_post_all_gather (attn_norm)",
        lambda: ttnn.rms_norm_post_all_gather(x, g_stats, epsilon=RMS_EPS, weight=st.w_attn_norm, dtype=ttnn.bfloat16),
        x,
        g_stats,
        st.w_attn_norm,
    )
    ttnn.deallocate(stats)
    ttnn.deallocate(g_stats)

    # --- q_a stem: q_a_proj (K-sharded, partial) -> TP reduce-scatter -> TP all-gather -> q_a norm ---
    qr = rec.call(
        "linear q_a_proj",
        lambda: ttnn.linear(
            attn_norm_out,
            st.w_q_a,
            compute_kernel_config=st.ck_hifi2,
            program_config=pc["q_a_proj"],
            memory_config=l1,
            dtype=ttnn.bfloat16,
        ),
        attn_norm_out,
        st.w_q_a,
    )
    qr = rec.call(
        "reduce_scatter_minimal_async (q_a, TP)",
        lambda: _reduce_scatter_async(ccl, qr, dim=3, cluster_axis=TP_AXIS, topology=tp_t),
        qr,
    )
    rec.subst("experimental.high_bw_all_gather", "experimental.all_gather_async")
    qr = rec.call(
        "all_gather_async (q_a latent, TP)",
        lambda: _all_gather_async(ccl, qr, dim=3, cluster_axis=TP_AXIS, topology=tp_t),
        qr,
    )
    qr = rec.call(
        "rms_norm (q_a_layernorm)",
        lambda: ttnn.rms_norm(
            qr, weight=st.w_q_a_norm, epsilon=RMS_EPS, memory_config=l1, compute_kernel_config=st.ck_hifi2
        ),
        qr,
        st.w_q_a_norm,
    )

    # --- q stem: q_b_proj -> heads -> split nope/rope -> wkv_b1 -> (RoPE) -> concat ---
    tt_q = rec.call(
        "linear q_b_proj",
        lambda: ttnn.linear(
            qr,
            st.w_q_b,
            compute_kernel_config=st.ck_hifi2,
            program_config=pc["q_b_proj"],
            memory_config=l1,
            dtype=ttnn.bfloat16,
        ),
        qr,
        st.w_q_b,
    )
    ttnn.deallocate(qr)

    tt_q = rec.call(
        "nlp_create_qkv_heads",
        lambda: ttnn.experimental.nlp_create_qkv_heads(
            tt_q, num_heads=HEADS_TP, num_kv_heads=0, transpose_k_heads=False, memory_config=dram
        )[0],
        tt_q,
    )

    q_nope = rec.call("slice q_nope", lambda: ttnn.slice(tt_q, [0, 0, 0, 0], [1, HEADS_TP, SEQ, QK_NOPE]), tt_q)
    q_rope = rec.call("slice q_rope", lambda: ttnn.slice(tt_q, [0, 0, 0, QK_NOPE], [1, HEADS_TP, SEQ, QK_HEAD]), tt_q)
    ttnn.deallocate(tt_q)

    q_nope = rec.call(
        "linear wkv_b1 (batched per head)",
        lambda: ttnn.linear(
            q_nope,
            st.w_wkv_b1,
            compute_kernel_config=st.ck_hifi2,
            program_config=pc["wkv_b1"],
            memory_config=l1,
            dtype=ttnn.bfloat16,
        ),
        q_nope,
        st.w_wkv_b1,
    )
    rec.skip("deepseek_prefill.rotary_embedding_indexed (q_rope)", "excluded: absent in the baseline revision")

    tt_q = rec.call("concat q [nope|rope]", lambda: ttnn.concat([q_nope, q_rope], dim=-1), q_nope, q_rope)
    ttnn.deallocate(q_nope)
    ttnn.deallocate(q_rope)

    # --- kv stem: kv_a_proj (partial) -> TP gather on dim 1 -> fast_reduce_nc -> split/norm/(RoPE) ---
    tt_kv = rec.call(
        "linear kv_a_proj_with_mqa",
        lambda: ttnn.linear(
            attn_norm_out,
            st.w_kv_a,
            compute_kernel_config=st.ck_hifi2,
            program_config=pc["kv_a_proj"],
            memory_config=dram,  # the TP gather streams its source from DRAM
            dtype=ttnn.bfloat16,
        ),
        attn_norm_out,
        st.w_kv_a,
    )
    rec.subst("experimental.high_bw_all_gather", "experimental.all_gather_async")
    tt_kv = rec.call(
        "all_gather_async (kv stem, TP, dim=1)",
        lambda: _all_gather_async(ccl, tt_kv, dim=1, cluster_axis=TP_AXIS, topology=tp_t),
        tt_kv,
    )
    tt_kv = rec.call(
        "fast_reduce_nc (kv partials)",
        lambda: ttnn.experimental.fast_reduce_nc(tt_kv, dims=[1], output=None, compute_kernel_config=st.ck_hifi4_fp32),
        tt_kv,
    )

    kv_nope = rec.call("slice kv_nope", lambda: ttnn.slice(tt_kv, [0, 0, 0, 0], [1, 1, SEQ, KV_LORA]), tt_kv)
    kv_rope = rec.call(
        "slice kv_rope",
        lambda: ttnn.slice(tt_kv, [0, 0, 0, KV_LORA], [1, 1, SEQ, KV_LORA + QK_ROPE]),
        tt_kv,
    )
    ttnn.deallocate(tt_kv)

    kv_nope = rec.call(
        "rms_norm (kv_a_layernorm)",
        lambda: ttnn.rms_norm(
            kv_nope, weight=st.w_kv_a_norm, epsilon=RMS_EPS, memory_config=dram, compute_kernel_config=st.ck_hifi2
        ),
        kv_nope,
        st.w_kv_a_norm,
    )
    rec.skip("deepseek_prefill.rotary_embedding_indexed (k_pe)", "excluded: absent in the baseline revision")

    kvpe = rec.call("concat kvpe [nope|rope]", lambda: ttnn.concat([kv_nope, kv_rope], dim=-1), kv_nope, kv_rope)
    ttnn.deallocate(kv_rope)
    kvpe = rec.call("typecast kvpe -> bfloat8_b (cache dtype)", lambda: ttnn.typecast(kvpe, ttnn.bfloat8_b), kvpe)

    rec.skip("deepseek_prefill.update_padded_kv_cache", "excluded: absent in the baseline revision")
    rec.skip("transformer.ring_mla", "excluded: absent in the baseline revision")
    ttnn.deallocate(kvpe)
    ttnn.deallocate(tt_q)

    # ring_mla would have produced [1, heads_tp, seq, kv_lora]; the pre-built stand-in stands in for
    # it so the epilogue (wkv_b2 -> concat_heads -> o_proj -> reduce-scatter) runs at the right shape.
    attn_out = st.syn_attn_out

    # --- epilogue: wkv_b2 -> concat heads -> o_proj -> TP reduce-scatter -> residual add ---
    v_out = rec.call(
        "linear wkv_b2 (batched per head)",
        lambda: ttnn.linear(
            attn_out,
            st.w_wkv_b2,
            compute_kernel_config=st.ck_hifi2,
            program_config=pc["wkv_b2"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        ),
        attn_out,
        st.w_wkv_b2,
    )
    v_out = rec.call(
        "nlp_concat_heads",
        lambda: ttnn.experimental.nlp_concat_heads(v_out, memory_config=dram),
        v_out,
    )
    v_out = rec.call(
        "linear o_proj",
        lambda: ttnn.linear(
            v_out,
            st.w_o,
            compute_kernel_config=st.ck_hifi2,
            program_config=pc["o_proj"],
            memory_config=l1,
            dtype=ttnn.bfloat16,
        ),
        v_out,
        st.w_o,
    )
    mla_out = rec.call(
        "reduce_scatter_minimal_async (o_proj, TP)",
        lambda: _reduce_scatter_async(ccl, v_out, dim=3, cluster_axis=TP_AXIS, topology=tp_t),
        v_out,
    )
    ttnn.deallocate(attn_norm_out)

    signpost(header="MLA_END")
    out = rec.call("add (post-MLA residual)", lambda: ttnn.add(x, mla_out), x, mla_out)
    ttnn.deallocate(mla_out)
    return out


def _moe(rec, st, x):
    """ffn_norm -> gate (+TP all-reduce) -> routing prep -> shared expert -> reduce -> residual."""
    ccl, pc = st.ccl, st.pc
    tp_t, sp_t = st.tp_topology, st.sp_topology
    dram, l1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG

    # --- distributed ffn_norm ---
    stats = rec.call(
        "rms_norm_pre_all_gather (ffn_norm)",
        lambda: ttnn.rms_norm_pre_all_gather(x, dtype=ttnn.bfloat16),
        x,
    )
    rec.subst("experimental.high_bw_all_gather", "experimental.all_gather_async")
    g_stats = rec.call(
        "all_gather_async (ffn_norm stats, TP)",
        lambda: _all_gather_async(ccl, stats, dim=3, cluster_axis=TP_AXIS, topology=tp_t),
        stats,
    )
    h = rec.call(
        "rms_norm_post_all_gather (ffn_norm)",
        lambda: ttnn.rms_norm_post_all_gather(x, g_stats, epsilon=RMS_EPS, weight=st.w_ffn_norm, dtype=ttnn.bfloat16),
        x,
        g_stats,
        st.w_ffn_norm,
    )
    ttnn.deallocate(stats)
    ttnn.deallocate(g_stats)

    signpost(header="MOE_START")

    # --- gate matmul + TP all-reduce.  The model calls all_reduce_async, which lowers to exactly
    #     this reduce-scatter + all-gather pair; issuing the pair directly keeps the same two device
    #     ops without depending on the composite's kwargs, which differ between revisions.
    logits = rec.call(
        "matmul moe_gate",
        lambda: ttnn.matmul(
            h, st.w_gate, compute_kernel_config=st.ck_gate, program_config=pc["gate"], memory_config=l1
        ),
        h,
        st.w_gate,
    )
    logits = rec.call(
        "reduce_scatter_minimal_async (gate all-reduce, TP)",
        lambda: _reduce_scatter_async(ccl, logits, dim=3, cluster_axis=TP_AXIS, topology=tp_t, memory_config=l1),
        logits,
    )
    logits = rec.call(
        "all_gather_async (gate all-reduce, TP)",
        lambda: _all_gather_async(ccl, logits, dim=3, cluster_axis=TP_AXIS, topology=tp_t, memory_config=l1),
        logits,
    )

    # --- gate top-k + routing setup: all device ops here are model-only ---
    rec.skip("deepseek_prefill.moe_grouped_topk", "excluded: absent in the baseline revision")
    if INCLUDE_TOPK:
        # Not what the model dispatches (it fuses selection+scoring into moe_grouped_topk), but both
        # ops exist in either revision, so this is offered as an opt-in stand-in.
        vals, idx = ttnn.topk(logits, k=TOPK, dim=-1, sorted=True)
        rec.op("topk (stand-in, KIMI_OP2OP_INCLUDE_TOPK=1)", (vals, idx), logits)
        sc = rec.call("softmax (stand-in)", lambda: ttnn.softmax(vals, dim=-1, numeric_stable=True), vals)
        ttnn.deallocate(vals)
        ttnn.deallocate(idx)
        ttnn.deallocate(sc)
    ttnn.deallocate(logits)

    rec.skip("deepseek_prefill.masked_bincount", "excluded: absent in the baseline revision")

    # The routing setup's expert-histogram all-gather is plain ttnn.all_gather and DOES exist in
    # both, so it stays -- fed a random histogram at the real shape.
    hist = st.syn_hist
    g_hist = rec.call(
        "all_gather (expert histogram, SP)",
        lambda: ttnn.all_gather(hist, dim=2, cluster_axis=SP_AXIS, num_links=NUM_LINKS, topology=sp_t),
        hist,
    )
    ttnn.deallocate(g_hist)
    rec.skip("deepseek_prefill.offset_cumsum", "excluded: absent in the baseline revision")

    # --- the two untilizes that hand indices/scores to dispatch in ROW_MAJOR ---
    idx_t = st.syn_idx_tile
    idx_rm = rec.call(
        "to_layout indices TILE->ROW_MAJOR (untilize_with_unpadding)",
        lambda: ttnn.to_layout(idx_t, ttnn.ROW_MAJOR_LAYOUT),
        idx_t,
    )
    sc_t = st.syn_scores_tile
    sc_rm = rec.call(
        "to_layout scores TILE->ROW_MAJOR (untilize_with_unpadding)",
        lambda: ttnn.to_layout(sc_t, ttnn.ROW_MAJOR_LAYOUT),
        sc_t,
    )

    # --- TP all-gather of x: both the shared expert and dispatch need the full emb_dim ---
    x_full = rec.call(
        "all_gather_async (x -> full emb_dim, TP)",
        lambda: _all_gather_async(ccl, h, dim=-1, cluster_axis=TP_AXIS, topology=tp_t),
        h,
    )
    ttnn.deallocate(h)

    # --- shared expert: gate(+SiLU) / up / multiply / down / TP reduce-scatter ---
    gate_out = rec.call(
        "matmul shared_expert gate_proj (SiLU fused)",
        lambda: ttnn.matmul(x_full, st.w_sh_gate, program_config=pc["shared_gate"], compute_kernel_config=st.ck_hifi2),
        x_full,
        st.w_sh_gate,
    )
    up_out = rec.call(
        "matmul shared_expert up_proj",
        lambda: ttnn.matmul(x_full, st.w_sh_up, program_config=pc["shared_up"], compute_kernel_config=st.ck_hifi2),
        x_full,
        st.w_sh_up,
    )
    rec.call("multiply_ (SwiGLU)", lambda: ttnn.multiply_(gate_out, up_out), gate_out, up_out)
    ttnn.deallocate(up_out)
    sh_full = rec.call(
        "matmul shared_expert down_proj",
        lambda: ttnn.matmul(
            gate_out, st.w_sh_down, program_config=pc["shared_down"], compute_kernel_config=st.ck_hifi2
        ),
        gate_out,
        st.w_sh_down,
    )
    ttnn.deallocate(gate_out)
    # rank-3 here, matching the model: dim=-1 is dim 2, and the Linear path takes the persistent
    # [2]+shape intermediate.
    sh_full = ttnn.reshape(sh_full, (1, SEQ, EMB))
    shared_out = rec.call(
        "reduce_scatter_minimal_async (shared expert, TP, Linear + persistent)",
        lambda: _reduce_scatter_async(
            ccl,
            sh_full,
            dim=-1,
            cluster_axis=TP_AXIS,
            topology=ttnn.Topology.Linear,
            persistent=[st.sh_rs_intermediate],
        ),
        sh_full,
    )

    # --- dispatch / routed experts / combine: all model-only ---
    rec.skip("deepseek_prefill.dispatch", "excluded: absent in the baseline revision")

    # The two L1->DRAM copies dispatch leaves behind are plain to_memory_config and DO exist in both.
    sc_dram = rec.call("to_memory_config scores L1->DRAM (copy)", lambda: ttnn.to_memory_config(sc_rm, dram), sc_rm)
    idx_dram = rec.call("to_memory_config indices L1->DRAM (copy)", lambda: ttnn.to_memory_config(idx_rm, dram), idx_rm)
    ttnn.deallocate(sc_rm)
    ttnn.deallocate(idx_rm)

    rec.skip("deepseek_prefill.unified_routed_expert_moe", "excluded: absent in the baseline revision")
    rec.skip("deepseek_prefill.combine", "excluded: absent in the baseline revision")

    # The reduce module's leading reshape is plain ttnn.reshape (ReshapeView) and stays.
    resh = rec.call("reshape (combine weights view)", lambda: ttnn.reshape(sc_dram, (1, 1, SEQ * TOPK, 1)), sc_dram)
    ttnn.deallocate(resh)
    ttnn.deallocate(sc_dram)
    ttnn.deallocate(idx_dram)

    rec.skip("deepseek_prefill.post_combine_reduce", "excluded: absent in the baseline revision")

    # post_combine_reduce would have produced [1, 1, seq, emb]; the pre-built stand-in stands in for
    # it so the reduce module's closing reduce_scatter (plain ttnn.reduce_scatter) still runs.
    combined = st.syn_combined
    routed_out = rec.call(
        "reduce_scatter (routed output, TP)",
        lambda: ttnn.reduce_scatter(combined, dim=-1, cluster_axis=TP_AXIS, num_links=NUM_LINKS, topology=tp_t),
        combined,
    )

    # --- final: routed + shared, then the block residual ---
    shared_out = ttnn.reshape(shared_out, (1, 1, SEQ, EMB_TP))
    routed_out = ttnn.reshape(routed_out, (1, 1, SEQ, EMB_TP))
    ffn_out = rec.call("add (routed + shared)", lambda: ttnn.add(routed_out, shared_out), routed_out, shared_out)
    ttnn.deallocate(routed_out)
    ttnn.deallocate(shared_out)
    out = rec.call("add (post-FFN residual)", lambda: ttnn.add(x, ffn_out), x, ffn_out)
    ttnn.deallocate(ffn_out)
    signpost(header="MOE_END")
    return out


def _forward(rec, st, x):
    x = _mla(rec, st, x)
    out = _moe(rec, st, x)
    ttnn.deallocate(x)
    return out


# ---------------------------------------------------------------------------------------------
# reference numbers from the real model, for orientation
# ---------------------------------------------------------------------------------------------
def _report_reference():
    logger.info(
        "reference (real Kimi-K2.6, L10 chunk 0, warm iteration, 8x4 BH, untraced eager, tracy with "
        "--sync-host-device): one MoE layer = 58 device ops, device kernel 11.25 ms, op2op 27.56 ms. "
        "The 58 include the 12 model-only ops this harness skips; the biggest of those by device time "
        "are UnifiedRoutedExpertFfn ~2.88 ms, Combine ~1.87 ms, Dispatch ~1.17 ms and RingJointSDPA "
        "~1.07 ms, so the subset here is dispatch-bound by design."
    )


# ---------------------------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (SP, TP),
            {
                # No fabric_router_config: the baseline revision's conftest does not accept it, and
                # the payload-size override must be identical on both sides or the CCLs differ.
                "fabric_config": FABRIC_CONFIG,
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
                "l1_small_size": 768,
                "trace_region_size": 0,
            },
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(0)
def test_kimi_moe_layer_op2op(mesh_device, device_params):
    """Replay one Kimi MoE layer's op sequence on random tensors and report per-iteration wall time."""
    if tuple(mesh_device.shape) != (SP, TP):
        pytest.skip(f"this harness targets mesh {SP}x{TP}, got {tuple(mesh_device.shape)}")

    missing = [op for op in REQUIRED_OPS if not _available(op)]
    if missing:
        pytest.skip(f"this ttnn build lacks ops the harness itself needs: {missing}")

    logger.info(
        f"=== ttnn build: mesh {tuple(mesh_device.shape)}, arch {mesh_device.arch()}, "
        f"fabric {FABRIC_CONFIG.name}, topology SP={SP_TOPOLOGY.name} TP={TP_TOPOLOGY.name}, "
        f"index dtype {INDEX_DTYPE.name} ==="
    )
    for name, why in MODEL_ONLY_OPS:
        logger.info(f"  model op {name:58s} present={_available(name)}  ({why})")
    _report_reference()

    st = LayerState(mesh_device)
    x0 = _rand(mesh_device, (1, 1, SEQ, EMB_TP), ttnn.bfloat16)

    # Iteration 0 is discarded: it pays the JIT compile for every program in the sequence.
    logger.info("--- warmup iteration (JIT compile; discarded) ---")
    signpost(header="iter_0_start")
    _warm_rec = Recorder()
    _warm_rec.mesh = mesh_device
    warm = _forward(_warm_rec, st, x0)
    ttnn.synchronize_device(mesh_device)
    signpost(header="iter_0_end")
    ttnn.deallocate(warm)

    rec = Recorder()
    rec.mesh = mesh_device
    times = []
    per_op = None  # per-op host dispatch time, accumulated across measured iterations
    for it in range(1, ITERS + 1):
        r = rec if it == 1 else Recorder()  # print the shape table once, on iteration 1 only
        r.mesh = mesh_device
        signpost(header=f"iter_{it}_start")
        t0 = time.perf_counter()
        out = _forward(r, st, x0)
        ttnn.synchronize_device(mesh_device)
        dt = time.perf_counter() - t0
        signpost(header=f"iter_{it}_end")
        ttnn.deallocate(out)
        times.append(dt)
        if per_op is None:
            per_op = [[ns] for ns in r.host_ns]
        else:
            assert len(r.host_ns) == len(per_op), "op count changed between iterations"
            for slot, ns in zip(per_op, r.host_ns):
                slot.append(ns)
        logger.info(f"iteration {it}: {dt * 1e3:.2f} ms wall ({r.n} device ops)")

    # --- per-op host dispatch table.  This is the regression signal: the op subset here is
    # deliberately device-light, so a revision-to-revision wall-clock change lands almost entirely
    # in these numbers, and the table says WHICH op moved.
    mode = "enqueue + own device time (SYNC_PER_OP=1, serialised)" if SYNC_PER_OP else "host enqueue only"
    logger.info(f"=== per-op time, {mode} (min / mean over measured iterations) ===")
    logger.info(f"{'idx':>4}  {'min us':>9}  {'mean us':>9}   op")
    order = sorted(range(len(per_op)), key=lambda i: -sum(per_op[i]) / len(per_op[i]))
    labels = {i: lbl for i, lbl in rec.lines}
    for i in range(len(per_op)):
        samples = per_op[i]
        logger.info(
            f"{i:4d}  {min(samples) / 1e3:9.1f}  {sum(samples) / len(samples) / 1e3:9.1f}   {labels.get(i, '?')}"
        )
    logger.info("=== top 10 ops by mean per-op time ===")
    for rank, i in enumerate(order[:10], 1):
        samples = per_op[i]
        logger.info(f"  {rank:2d}. [{i:3d}] {sum(samples) / len(samples) / 1e3:9.1f} us  {labels.get(i, '?')}")

    logger.info(f"=== skipped ({len(rec.skipped)}) ===")
    for name, why in rec.skipped:
        logger.info(f"  {name:52s} {why}")
    logger.info(f"=== substituted ({len(rec.substituted)}) ===")
    for a, b in rec.substituted:
        logger.info(f"  {a} -> {b}")

    best, mean = min(times), sum(times) / len(times)
    host_sum = sum(sum(s) / len(s) for s in per_op) / 1e6
    logger.info(
        f"=== {rec.n} ops/iteration | wall best {best * 1e3:.2f} ms, mean {mean * 1e3:.2f} ms "
        f"over {len(times)} iterations | summed per-op {host_sum:.2f} ms "
        f"({100.0 * host_sum / (mean * 1e3):.0f}% of wall) ==="
    )
    assert rec.n > 0, "no ops were issued"
