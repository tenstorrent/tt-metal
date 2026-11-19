# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill megakernel (PMK) FAMILY: the persistent per-layer prefill program skeleton of the sparse_kda kind, laid out
per the Stage 16 contract (doc/ttnn/16-prefill-production/prefill_megakernel.md: §2 core roles, §3 phase table
P0..P16, §4 L1 per core, §4b hand-off map, §4c PfStreamPhase) and 7a's prefill_mk_collectives.md (§3 FabricStream
AG / RS, §5 mHC phases).

What this module is: the PLAN + the PLUG. It builds the program descriptor of ONE persistent program per sparse_kda layer
at prefill width (v1: the 2,048 bucket, sequence-parallel: 256 SP rows per chip at TP8; TP2 = 1,024 SP rows) with
  - the explicit phase table (phase, cores, CBs in / out, the existing kernel body it wraps, the fabric collective),
  - the L1 census per core role from REAL tile sizes (fail loudly over the served budget: CB region
    [308,352, 1,334,720) = 1,026,368 B, i.e. the 1,264,512 worker_l1 contract minus Stage 14's binding L1 buffers),
  - the sub-chunking policy (rows resident per MM core, passes over the 2,048 rows, chunk store in L1 or DRAM scratch),
  - the semaphore / counter allocation (NUM_SEMAPHORES = 16 on this tree: ids 0..15 only),
  - stub phase bodies that compile as kernels (kernels/pmk/pmk_stub_*.cpp) behind the SAME interface the real bodies
    plug into: `Phase` (core role, CB ids in / out, kernel sources per RISC, compile-time / runtime args, hand-offs)
    and `register_phase()`.
The real phase bodies are written by the other pieces (mhc, coll, kda, moe, cache, attn) and registered here; the
composite kernels are generated the way kernels/mk/owner_*.cpp compose Stage 14's phases: every body is the unmodified
standalone kernel file included in its own namespace with its own argument bases (PH_CT0 / ph_rt0, kernels/mk/
phase_args.hpp), the main loop runs the core's phases in order with the hand-off waits / signals between them.

Row geometry (contract §1): SP rows = C / N per chip (mHC, norms, router rows); the KDA phases are head-parallel over
ALL C rows (8 heads per chip at TP8); the MoE is XSL (every chip applies its 1/N expert slice to every routed row);
four static collectives per layer (AG x̂, RS o, AG x̂_f, RS FFN).

No device is opened here. `PmkSparseKdaLayer.plan()` is pure python (no ttnn import); `descriptor()` needs ttnn and a
(mock) device to build the ProgramDescriptor with the stub / registered kernels.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field

TILE = 32
# ------------------------------------------------------------------------------------------------------------ L1 contract
L1_TOTAL = 1_572_864  # Blackhole Tensix L1
WORKER_L1_SERVED = 1_280_896  # cf3h-j (coordinator 2026-10-25T18:27:30Z: +16 KB, every CB base 308,352 -> 291,968); the served stack's worker_l1_size (the -O3 owners' kernel-config ring; 7a / 14 / 16)
CB_BASE = L1_TOTAL - WORKER_L1_SERVED  # 308,352: the first CB byte
CB_END_CEILING = 1_334_720  # Stage 14's sparse reader CB end = the binding L1 buffer floor (contract §4)
CB_BUDGET = CB_END_CEILING - CB_BASE  # 1,026,368 B of static CBs per core
STAGE16_CB_END_RULING = 1_330_000  # coordinator 2026-10-21 20:23Z (TPPrefill programs); the stricter of the two
TILE_BYTES = {"bf16": 2048, "bf8": 1088, "bf4": 576, "fp32": 4096, "u32": 4096, "u16": 2048}
NUM_SEMAPHORES = 16  # tt_metal/impl/buffers/semaphore.hpp (program semaphores: ids 0..15)
FID_VALUE = {"LoFi": 0, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}  # ckernel::MathFidelity (the PMK_PART_FID values)


def merge_compute(computes: list, default, flex=None, part_fid: dict | None = None):
    """one TRISC composite = one compute config: merge the phases' configs (Stage 16 integration). fp32_dest_acc_en is
    OR-ed; unpack_to_dest_mode is a per-CB union (two phases asking different modes for one CB index = error); math
    fidelity, math_approx_mode and dst_full_sync_en must agree (they are kernel-wide compile settings: a conflict is a
    hard error, never a silent pick). Each entry: (pid, a ttnn.ComputeConfigDescriptor or a dict with those keys).
    THE PART REGIME (Stage 14, the per-layer kernel framework 2.3; part_fid a dict to fill): flex = {part key:
    dict(dst_any, fid_mode)}; dst_any = the part runs in either DST sync mode (<= 4 fp32 DST tiles per acquire);
    fid_mode "template" = its matmuls take the fidelity as a template argument (kernels/pmk/pmk_fid.hpp: PfStream's
    member compute), "any" = its result does not depend on the fidelity (copy / SFPU add: the RS reducers). The
    kernel-wide fidelity / sync are the rigid parts' (else the highest fidelity / full sync if any part asked it), and
    part_fid[key] = the fidelity value of every "template" part that differs (-> its PMK_PART_FID). Without a
    conflict the result is exactly the old rule's."""
    import ttnn  # noqa: PLC0415

    if not computes:
        return default
    get = lambda c, k, d=None: (c.get(k, d) if isinstance(c, dict) else getattr(c, k, d))  # noqa: E731
    fname = lambda c: str(get(c, "math_fidelity")).split(".")[-1]  # noqa: E731
    flex = dict(flex) if isinstance(flex, dict) else {k: dict(dst_any=True, fid_mode="template") for k in (flex or ())}
    fmode = lambda k: (flex.get(k) or {}).get("fid_mode")  # noqa: E731
    fid = {fname(c) for k, c in computes if get(c, "math_fidelity") is not None and fmode(k) != "any"}
    f_kernel = None
    if len(fid) > 1:
        rigid = {fname(c) for k, c in computes if fmode(k) is None and get(c, "math_fidelity") is not None}
        if part_fid is None or len(rigid) > 1 or not fid <= set(FID_VALUE):
            raise ValueError(
                f"PMK composite: phases need different math fidelities {sorted(fid)}: {[p for p, _ in computes]}"
            )
        f_kernel = next(iter(rigid)) if rigid else max(fid, key=FID_VALUE.get)
        for k, c in computes:
            if get(c, "math_fidelity") is not None and fname(c) != f_kernel and fmode(k) == "template":
                part_fid[k] = FID_VALUE[fname(c)]
    elif fid:
        f_kernel = next(iter(fid))
    sync = {bool(get(c, "dst_full_sync_en", False)) for _, c in computes}
    if len(sync) > 1:
        dany = lambda k: bool((flex.get(k) or {}).get("dst_any"))  # noqa: E731
        rigid_s = {bool(get(c, "dst_full_sync_en", False)) for k, c in computes if not dany(k)}
        # a dst_any part that asked FULL sync (more than 4 fp32 DST tiles) cannot go half
        flex_full = any(bool(get(c, "dst_full_sync_en", False)) for k, c in computes if dany(k))
        if part_fid is None or len(rigid_s) > 1 or (rigid_s == {False} and flex_full):
            raise ValueError(
                f"PMK composite: phases disagree on dst_full_sync_en: {[(p, get(c, 'dst_full_sync_en')) for p, c in computes]}"
            )
        sync = rigid_s if rigid_s else {True}
    approx = {bool(get(c, "math_approx_mode", False)) for _, c in computes}
    if len(approx) > 1:
        raise ValueError(f"PMK composite: phases disagree on math_approx_mode: {[p for p, _ in computes]}")
    fp32 = any(bool(get(c, "fp32_dest_acc_en", False)) for _, c in computes)
    modes = [ttnn.UnpackToDestMode.Default] * 64
    owner = {}
    for pid, c in computes:
        um = get(c, "unpack_to_dest_mode")
        if um is None:
            continue
        items = um.items() if isinstance(um, dict) else enumerate(list(um))
        for i, m in items:
            if m == ttnn.UnpackToDestMode.Default:
                continue
            if modes[int(i)] != ttnn.UnpackToDestMode.Default and modes[int(i)] != m:
                raise ValueError(f"PMK composite: CB {i} unpack-to-dest conflict between {owner[int(i)]} and {pid}")
            modes[int(i)], owner[int(i)] = m, pid
    f0 = get(computes[0][1], "math_fidelity") if f_kernel is None else f_kernel
    if isinstance(f0, str):
        f0 = getattr(ttnn.MathFidelity, f0)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=f0 if f0 is not None else ttnn.MathFidelity.HiFi2,
                                       math_approx_mode=approx.pop(), fp32_dest_acc_en=fp32,
                                       dst_full_sync_en=sync.pop())  # fmt: skip
    cfg.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
    return cfg


KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
REL_KDIR = os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels")
REPO_ROOT = KDIR[: -len(REL_KDIR)] if KDIR.endswith(REL_KDIR) else os.getcwd()  # the tree a repo-root include is under

# -------------------------------------------------------------------------------------------- CB map (contract §4b)
# owners / members (Stage 14's map, so the existing phase kernels keep their indices)
CB_MHC_PRE = range(0, 21)  # 7a's PF set (mhc_pre_pf_common.hpp) uses 0..19
CB_MHC_POST = range(21, 29)
CB_PARK = 29  # forwarded weight blocks (the streamers' multicast target)


def ZONES(risc: str = "dm0") -> bool:
    """GLM53_PMK_ZONES=1 (Stage 16 W2, 2026-10-24): a device-profiler zone around every phase body and window entry of
    the PMK DATA-MOVEMENT composites (=all: the TRISC composites too -- each of their zones is three sites, unpack /
    math / pack, and the 16-bit location hashes collided: a15prof 21:32Z, PMK_P4 vs PMK_P2 in chlkc_pack.cpp; =math:
    the TRISC composites zone on the math thread only, zone_call; GLM53_PMK_ITEM_ZONES=1 adds the kda item steps).
    api/debug/kernel_profiler.hpp: no-ops unless the profiler build is on -> per-phase device time of ONE fused program
    in a tracy capture; check a build's locations with doc/.../pmk/w2_a15prof/zone_check.py before the device run"""
    v = os.environ.get("GLM53_PMK_ZONES", "0")
    return v in ("all", "math") or (v == "1" and risc != "c")


def PHASE_PROBE() -> tuple | None:
    """GLM53_PMK_PHASE_PROBE=<probe tensor address>:<L1 scratch address> (W7 step 6 / S5.3, coordinator 20:26:11Z (3):
    the P11 / P15 spans inside the fused program): every DATA-MOVEMENT composite marks the wall clock at its start and
    after each phase call, plus the marks a body emits itself (PMK_PP_MARK(); declared by its part's `probe_marks`, e.g.
    program B's plan / sweep / combine), <= 32 marks, then writes them as one 128-B row of a DRAM uint32 tensor [2048, 32]
    at row ((noc0 y & 31) x 32 + (noc0 x & 31)) x 2 + RISC (dm0 0, dm1 1), staged through 128 B of the L1 scratch
    (+ 128 RISC) -- an in-program probe under the trace; the labels of each (role, RISC) row are PmkLayer.probe_labels
    """
    v = os.environ.get("GLM53_PMK_PHASE_PROBE", "")
    if not v:
        return None
    a, b = v.split(":")
    return int(a), int(b)


def ZSALT() -> str:
    return os.environ.get("GLM53_PMK_ZONE_SALT", "")  # renames every zone (a fresh location hash on a collision)


def zone_call(risc: str, name: str, body: str) -> str:
    """`body` inside a device-profiler zone `name` (ZONES(risc) true). GLM53_PMK_ZONES=math (W2 a15prof3): the TRISC
    composites zone on the MATH thread only (one site per zone instead of three; unpack / pack run the bare body)"""
    z = f'{{ DeviceZoneScopedN("{name}"); {body} }}'
    if risc == "c" and os.environ.get("GLM53_PMK_ZONES", "0") == "math":
        return f"#if defined(TRISC_MATH)\n    {z}\n#else\n    {body}\n#endif"
    return f"    {z}"


CB_ARG = 30  # stack per-layer args (kernels/mk/stack.hpp); GLM53_PMK_ARGS=table: the arg rows (pmk::load_arg_table)
# GLM53_PMK_ARGS=table (trace-shrink ruling 2026-10-23 16:33:48Z): the DM composites of these roles take a 2-word header
# (the program's DRAM argument table, this core's row) instead of their full runtime-arg vector; lane / tail (fabric sends:
# the post-build hooks append connection args) and every TRISC composite (no NoC) keep their args in the launch
ARG_TABLE_ROLES = ("member", "root", "mm_only")
ARG_PAGE = 1024  # one row = one interleaved DRAM page (<= 256 args: MAX_RT_ARGS)
CB_PROBE = 31
CB_KDA_PREP = range(32, 50)  # family refinement: 35 = XS landing, 38..42 the MM slots -> prepare uses CB_KDA_PREP_IDX
CB_KDA_SCAN = range(50, 64)
# MM / reader cores
CB_INVH, CB_OUT32, CB_STAGE, CB_IN0, CB_PART, CB_OUT, CB_RING0 = 36, 38, 39, 40, 41, 42, 43
RING_BYTES_MAX = 160 * 1024  # mk_readers: a ring whose 3 slots exceed this gets 2
BLOCK_BYTES_MAX = 52 * 1024  # a weight block larger than this halves its K block (PARK = 2 blocks on every MM core)
CB_MHC_X = 2  # mhc_pre_pf CB["X"]: the member's resident residual block (aliased X32 = 3)
CB_MHC_X32, CB_MHC_H16 = 3, 12
MAX_RT_ARGS = 256  # per-core runtime-arg floor the audit enforces (tt-metal's portable max_runtime_args floor is higher)  # pmk_mhc's format views of X and H (same allocation, same data)
CB_MHC_FN, CB_MHC_W, CB_MHC_H, CB_MHC_GATHER, CB_MHC_OUT = 4, 5, 8, 0, 11
CB_POST_Y = 21  # mhc_post Y (H6 / H10 land here)
CB_LANE0 = 48  # the fabric lanes' CBs (lane cores only): 48..63 (fabric_ar's CB_L1 0.. are remapped by coll)
CB_TAIL0 = 32  # router tail CBs on the tail cores (RTAIL_CB0)

# ------------------------------------------------- semaphores (ids 0..15 only; Stage 16's binding map, 2026-10-22 09:37Z)
# MM / member / root / tail cores. Lane cores are DEDICATED (all 64 CBs / 16 semaphores are the coll piece's).
SEM_MHC = (0, 1, 2, 3)  # mhc_pre_pf's own (SEMAPHORES = 4)
SEM_HN = 4  # H1 / H7: member -> lanes, x̂ block written (raised on the LANE core: its own id space, coll's choice)
SEM_QKV = 5  # H3: P3 / P4 -> P5, chunks ready             (experts reuse 5..9 after the attention half)
SEM_KDA = 6  # H3b: P5 prepare -> P6 scan, (chunk, head) terms ready
SEM_ON = 7  # H4: P6 -> P7, O chunks done
SEM_A = 8  # H5: P7 -> P8 lanes, owner blocks complete (raised on the lane core)
SEM_M = 9  # H9: P13 / P14 -> P15 lanes (raised on the lane core)
SEM_PARK_VALID, SEM_PARK_READY = 10, 11  # weight multicast: streamer -> consumers VALID, consumers -> streamer READY
SEM_LOGITS = 12  # H8: router -> tail cores
SEM_TAIL = 13  # H8: tail -> MM cores (routes / row lists)
SEM_FS_REDUCED = 14  # H6 / H10: FabricStream RS -> members, owner block reduced
SEM_FS_ARRIVE = 15  # H2 / H8: FabricStream AG -> MM cores, x̂ row block landed (CB 35)
SEM_IN0_FREE = 8  # H7b on the MEMBERS (id 8 = Stage 14's A, free here: H5 goes through the lane words): the P7 consumer
# releases IN0 40 (= XS 35 = FN 4) -> the mhc P10 part may prefetch FN / W into the allocation; cumulative: +1 after
# P7 (k = 1), +1 after P14 (k = 2: the stack program's next-layer P1 prefetch)
# SEM 12 is RESERVED on the members for XS-FREE (the mhc piece dropped its SEM_C 12: the root's round-2 message implies
# the coefficients landed); id 12 = LOGITS only on the tail cores
SEM_XS_FREE = (
    12  # XS-FREE credit on the MEMBERS (id 12 = LOGITS only on the tail cores): the holder of block G raises it on
)
# its writers after its own P1 / P10 FN use (CB 4 shares the allocation with XS 35 / IN0 40); a local writer waits
# wait_min(k) (k = 1 before its P2 slice write, 2 before P11); the lane aggregates 64 credits per use and forwards ONE
# per-chip credit to every peer lane before that lane writes remote XS (FabricPhase FAR_CREDIT_NEXT style)
CB_XS_LAND = 35  # FabricStream's landing CB on the MM / member cores = the stationary row block (aliased onto IN0 40)
# 7a's FabricStreamC1 (tt/fabric_stream.py, kernels/fabric_stream/fs_common.hpp): per-lane sync tensor words (16 B each,
# same shard address on every lane core / chip). Producer -> lane readiness is NOT a cumulative lane semaphore (blocks
# complete out of order: TP2's P1 runs 32 blocks in group waves): a producer writes the source CB address once (XSADDR /
# OUTADDR) + the consumer semaphore address (SEMADDR), then NoC-atomic +1 on LREADY + k per 16-tile slice; k = the
# block's index in its lane's list, lane h owns the blocks with b % 2 == h. XSFREE: +1 per member per use (the H0 lane
# side); the lane forwards ONE per-chip credit to each peer lane before its remote XS writes.
FS_LANES = 2
FS_DEFAULT_WORDS = dict(
    WORD=16,
    ARR=0,
    DONE=16,
    GEN=17,
    XSFREE=18,
    LREADY=24,
    XSADDR=40,
    SEMADDR=41,
    GEN_NC=42,
    GEN_B=43,
    CREDIT=44,
    OUTADDR=52,
    OWNRDY=64,
    GEN_C=80,
    SYNC_WORDS=96,
)  # fs_common.hpp @ 11777c1cfc6+ (7a, 10:14Z)
FS_HEADER_CANDIDATES = (
    os.path.join(KDIR, "fabric_stream", "fs_common.hpp"),  # this tree (after the coll merge)
    "/home/alex/glm53/tt-metal-7a-d0ar/models/autoports/zai_org_glm_5_3_flash/tt/kernels/fabric_stream/fs_common.hpp",
)


def fs_words(path: str | None = None) -> dict:
    """7a's FabricStream sync-word indices parsed from kernels/fabric_stream/fs_common.hpp (`constexpr uint32_t NAME =
    N;`), as fabric_stream.py does since 11777c1cfc6: the first header found (this tree, then 7a's d0 tree) overrides
    the documented defaults; a word the header lacks keeps its default (XSFREE = 18, 7a 2026-10-22 10:14Z). The
    result carries `_source` (the header path or "defaults") for the plan's report."""
    import re

    words = dict(FS_DEFAULT_WORDS)
    words["_source"] = "defaults"
    for cand in ([path] if path else []) + list(FS_HEADER_CANDIDATES):
        if cand and os.path.exists(cand):
            with open(cand) as f:
                txt = f.read()
            for name, val in re.findall(r"constexpr\s+uint32_t\s+([A-Z_0-9]+)\s*=\s*(\d+)\s*;", txt):
                words[name] = int(val)
            words["_source"] = cand
            break
    return words


FS_WORDS = fs_words()
FS_WORD = FS_WORDS["WORD"]
FS_W_LREADY, FS_W_XSADDR, FS_W_SEMADDR, FS_W_OUTADDR = (FS_WORDS[k] for k in ("LREADY", "XSADDR", "SEMADDR", "OUTADDR"))
FS_W_XSFREE = FS_WORDS["XSFREE"]  # 18 (7a 10:14Z; 42 is GEN_NC, the lane NCRISC's launch counter)
CB_KDA_PREP_IDX = (32, 33, 34, 36, 37)  # prepare on the non-member MM cores: 32..34 + the free MM slots 36 (INVH) / 37


def _kb(b: int) -> str:
    return f"{b / 1024:,.1f} KB"


# ------------------------------------------------------------------------------------------------------ geometry
@dataclass
class PmkGeom:
    """The sparse_kda layer's shapes per chip. `from_config(cfg, tp)` reads the HF text config; `tp` = chips."""

    c_rows: int = 2048  # the bucket
    tp: int = 8
    hidden: int = 4096
    n_streams: int = 4
    heads: int = 64
    head_dim: int = 128
    conv_k: int = 4
    lowrank_n: int = 128 + 128 + 64  # [f_a | g_a | b] (kda.py); the b part is per head -> TP-sharded
    n_experts: int = 288
    top_k: int = 8
    moe_i: int = 2048  # routed experts' intermediate (XSL: 1/tp per chip)
    shared_i: int = 2048  # shared expert intermediate (TP: 1/tp per chip)
    router_pad: int = 320  # the router tail's COL_TILES * 32
    w_qkv: str = "bf8"
    w_lr: str = "bf8"
    w_o: str = "bf8"
    w_router: str = "bf8"
    w_shared: str = "bf8"
    w_experts: str = "bf4"
    # Stage 17 (contract B4: dense_kda = KDA attention + the dense MLP, layers 0-2): the FFN half is ONE dense SwiGLU
    # MLP, TP-sliced (gate_up hidden x 2 dense_i / tp, down dense_i / tp x hidden), no router / experts / tail
    kind: str = "sparse_kda"  # "sparse_kda" | "dense_kda"
    dense_i: int = 12288  # the dense layers' intermediate (HF intermediate_size)
    w_dense: str = "bf8"

    @classmethod
    def from_config(cls, cfg, tp: int, c_rows: int = 2048, kind: str = "sparse_kda") -> "PmkGeom":
        g = lambda n, d: int(getattr(cfg, n, d))  # noqa: E731
        assert kind in ("sparse_kda", "dense_kda"), kind
        return cls(c_rows=c_rows, tp=tp, kind=kind, dense_i=g("intermediate_size", 12288),
                   hidden=g("hidden_size", 4096), n_streams=g("hc_mult", 4),
                   heads=g("linear_num_heads", 64), head_dim=g("linear_head_dim", 128),
                   conv_k=g("linear_conv_kernel_dim", 4), lowrank_n=2 * g("linear_head_dim", 128) + g("linear_num_heads", 64),
                   n_experts=g("n_routed_experts", 288), top_k=g("num_experts_per_tok", 8),
                   moe_i=g("moe_intermediate_size", 2048),
                   shared_i=g("moe_intermediate_size", 2048) * g("n_shared_experts", 1))  # fmt: skip

    # per chip
    @property
    def sp_rows(self) -> int:
        assert self.c_rows % (TILE * self.tp) == 0, (self.c_rows, self.tp)
        return self.c_rows // self.tp

    @property
    def sp_blocks(self) -> int:  # 32-row blocks of the SP rows
        return self.sp_rows // TILE

    @property
    def c_blocks(self) -> int:  # 32-row blocks of the bucket = KDA chunks
        return self.c_rows // TILE

    @property
    def heads_r(self) -> int:
        return self.heads // self.tp

    @property
    def width_r(self) -> int:
        return self.heads_r * self.head_dim

    @property
    def qkv_n(self) -> int:
        return 3 * self.width_r

    @property
    def lr_n(self) -> int:  # f_a | g_a (per layer, replicated) | b (per head: TP shard)
        return 2 * self.head_dim + self.heads_r

    @property
    def xsl_w(self) -> int:
        return self.moe_i // self.tp

    @property
    def shared_r(self) -> int:
        return self.shared_i // self.tp

    @property
    def dense(self) -> bool:
        return self.kind == "dense_kda"

    @property
    def dense_r(self) -> int:
        return self.dense_i // self.tp

    @property
    def ht(self) -> int:
        return self.hidden // TILE

    @property
    def res_cols(self) -> int:
        return self.n_streams * self.hidden


@dataclass
class GeomDS:
    """A DSLinear stand-in for pure-python planning: (role, K, N, weight dtype); `n_pad` as DSLinear pads it (TILE x
    banks x readers). `from_dslinear(ds)` wraps a real one (the mock / device build)."""

    role: str
    k: int
    n: int
    w_dtype: str
    readers: int = 3
    banks: int = 8
    out_dtype: str = "bf16"

    @property
    def n_pad(self) -> int:
        q = TILE * self.banks * self.readers
        return -(-self.n // q) * q

    @classmethod
    def from_dslinear(cls, ds, w_dtype: str | None = None) -> "GeomDS":
        dt = w_dtype or {"DataType.BFLOAT8_B": "bf8", "DataType.BFLOAT4_B": "bf4", "DataType.BFLOAT16": "bf16"}.get(
            str(ds.weight.dtype), "bf8"
        )
        out = cls(role=ds.role, k=int(ds.k), n=int(ds.n), w_dtype=dt, readers=int(ds.readers))
        out._n_pad = int(ds.n_pad)
        return out


# ----------------------------------------------------------------------------------------- the phase interface
@dataclass
class HandOff:
    """One row of the hand-off map (contract §4b). `counter` = a program semaphore id (0..15) or ("word", index) for a
    monotonic L1 counter word (the fabric counters: never reset); `expected` = the count a consumer waits for with
    noc_semaphore_wait_min (never equality), per layer."""

    hid: str
    name: str
    producer: str
    consumer: str
    payload_bytes: int
    lives_in: str
    counter: object
    expected: int  # increments THIS hand-off adds per layer on the consumer's counter
    unit: str = "32-row block"
    # a program semaphore id is shared by several hand-offs per layer (14 = H6 then H10, 15 = H2 then H8, 12 = H0 twice):
    # the counters are cumulative and reset once per layer, so a later use waits for base + expected, base = the sum of
    # the earlier hand-offs' increments on that id in program (consumer-phase) order. Stage 14 found the bug (P16's wait
    # on 14 passed on P8's increments); `PmkSparseKdaPlan._assign_wait_bases` sets it, the composite emits wait_value.
    wait_base: int = 0

    @property
    def wait_value(self) -> int:
        return self.wait_base + self.expected


def core_rects(cores) -> list:
    """the cores as disjoint rectangles (x0, y0, x1, y1) covering EXACTLY them (trace-light lever (1), GLM53_PMK_RECT_MERGE:
    tt-metal pays the CB configs / semaphores / launch message / binary relays per CoreRange of a launch, so one range per
    core cost them per core): contiguous x runs per row, stacked over consecutive rows with the same run"""
    rows = {}
    for x, y in {(int(a), int(b)) for a, b in cores}:
        rows.setdefault(y, []).append(x)
    runs = {}
    for y in sorted(rows):
        xs = sorted(rows[y])
        r0 = xs[0]
        for a, b in zip(xs, xs[1:] + [None]):
            if b is None or b != a + 1:
                runs.setdefault(y, []).append((r0, a))
                if b is not None:
                    r0 = b
    out, open_ = [], {}  # (x0, x1) -> (y0, y_last)
    for y in sorted(runs):
        nxt = {}
        for run in runs[y]:
            if run in open_ and open_[run][1] == y - 1:
                nxt[run] = (open_[run][0], y)
            else:
                nxt[run] = (y, y)
        for run, (y0, y1) in open_.items():
            if run not in nxt or nxt[run][0] != y0:
                out.append((run[0], y0, run[1], y1))
        open_ = nxt
    out += [(run[0], y0, run[1], y1) for run, (y0, y1) in open_.items()]
    covered = {(x, y) for x0, y0, x1, y1 in out for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}
    assert covered == {(int(a), int(b)) for a, b in cores} and sum(
        (x1 - x0 + 1) * (y1 - y0 + 1) for x0, y0, x1, y1 in out
    ) == len(covered)
    return sorted(out, key=lambda r: (r[1], r[0]))


@dataclass
class CBUse:
    """A phase's use of one CB index on one core role: (format, page bytes, pages); `live` = the phase ids during
    which the index holds data (liveness: two indices aliased onto one allocation must never be live together)."""

    index: int
    fmt: str
    pages: int
    live: tuple
    page: int | None = None
    note: str = ""
    remote_writers: tuple = ()  # phase ids in which OTHER cores (local members, lanes, peer chips) write this CB
    # the phases whose BODIES read / write the CB (default: live). An incremental program allocates a CB only if one of
    # its users is registered: the mhc state is LIVE P1 -> P16 but used only by P1 / P10 / P16 (a KDA-only unit must
    # not allocate it)
    users: tuple = ()
    # never aliased / viewed: ALIASES / VIEWS are index-based and role-agnostic (the mhc / MM slots); another piece's
    # CB that happens to take one of those indices on some role must not join a group (13:3xZ: the kda scan CBs at
    # MM-only 0..13 fell into the X / X32 VIEW pair -> scan W (o_norm + ones) shared its allocation with the state ring
    # SR -> `on` collapsed; ST / PIN into H / H16)
    solo: bool = False
    # UNIFORM ADDRESS (assembly, first silicon 21:20Z): a CB that other cores address by their OWN write pointer (the
    # mhc's GATHER 0 / MSG 1 / PCOEF 22: member -> root sends, root -> member unicasts; "all"-role in pmk_mhc.Plan.cbs)
    # must sit at the same L1 address on every role: allocated FIRST, in index order, with the same size, on every role
    uniform: bool = False
    # PINNED (assembly TP8 resident residual, PMK_TP8_RESIDENT): allocated FIRST, in EVERY program of the layer (its
    # data must survive the programs that do not use it: the L1 is never cleared between the launches of one trace, so
    # a CB at the same address, not overlapped by any other CB of the intermediate programs, keeps its contents; the
    # consumer re-attaches it with a reserve + push and no DRAM read). The engine's PINNED groups (one allocation,
    # several format views) go before the UNIFORM slots on every role, so a role without the datum carries a
    # placeholder of the same size to keep the uniform addresses aligned.
    pinned: bool = False

    @property
    def page_bytes(self) -> int:
        return self.page if self.page is not None else TILE_BYTES[self.fmt]

    @property
    def nbytes(self) -> int:
        return self.page_bytes * self.pages


@dataclass
class ActSource:
    """PfStreamPhase.in0 (contract §4c): the stationary activation store (H2 / H3 / H8) + its row-block counter."""

    handoff: str  # "H2" | "H3" | "H8" | "H4"
    cb: int  # the CB index the rows sit in on the consumer core
    counter: int  # semaphore id waited with wait_min per row block
    order: str = "local-first"  # the row-block order the consumer walks (contract §5: local blocks, then arrivals)


@dataclass
class MMPhaseBase:
    """Stage 14's mk_readers.MMPhase fields, ttnn-free (ds = GeomDS or DSLinear)."""

    ds: object
    kb: int
    wait: tuple | None = None
    signal: int | None = None
    math_fidelity: str | None = None  # "HiFi2" / "LoFi" / None = the DSLinear's
    ring_kb: int | None = None
    k_split: tuple | None = None


@dataclass
class PfStreamPhase(MMPhaseBase):
    """Contract §4c: Stage 14's MMPhase extended to prefill rows. Streamer = BRISC of a bank streamer (reader_w.cpp
    lineage: blocks in program order into the (format, block) ring, never waits for activations, never rewound,
    multicast once to the consumer rectangles' PARK slots); consumers = MM cores, activation-stationary over ALL their
    row sub-chunks (every weight byte read from DRAM once per layer; `passes` > 1 = re-streaming, reported)."""

    m_tiles: int = 0  # rows / 32 of the phase
    rows_of: dict = field(default_factory=dict)  # MM core (x, y) -> its stationary row blocks
    consumers: list = field(default_factory=list)  # multicast rectangles (lists of (x, y)) per streamer
    nb: int = 0  # output tiles per N-block (consumer accumulator = rows_of x nb tiles, fp32, PART 41)
    in0: ActSource | None = None
    out: HandOff | None = None
    passes: int = 1  # sub-chunking: weight re-streams per layer (1 = the contract's no-re-read rule)

    def geo(self, streamers: int) -> dict:
        """ring geometry as mk_readers.ReaderPhases: per bank shard width, readers in the phase, N tiles per
        streamer, block bytes"""
        ds = self.ds
        nt = int(getattr(ds, "n_pad", 0) or ds.n_pad) // TILE
        kt = int(ds.k) // TILE
        banks = int(getattr(ds, "banks", 8))
        shard_w = nt // banks
        rpb = max(1, streamers // banks)
        rp = max(r for r in range(1, rpb + 1) if shard_w % r == 0)
        nr = shard_w // rp
        tb = TILE_BYTES[ds.w_dtype if isinstance(ds, GeomDS) else GeomDS.from_dslinear(ds).w_dtype]
        return dict(nb=kt // self.kb, kb=self.kb, nr=nr, rp=rp, tb=tb, block_bytes=self.kb * nr * tb,
                    kt=kt, nt=nt, w_bytes=kt * nt * tb, ring_key=(ds.w_dtype if isinstance(ds, GeomDS) else "w",
                                                                   self.kb * nr))  # fmt: skip


@dataclass
class PhaseBody:
    """What a piece registers for a phase: the kernel source per RISC (dm0 = BRISC, dm1 = NCRISC, c = TRISC), its
    compile-time args, runtime args per core (x, y), defines, the compute config, and its CB uses per role."""

    dm0: str | None = None
    dm1: str | None = None
    c: str | None = None
    ct: list = field(default_factory=list)
    rt: dict = field(default_factory=dict)
    defines: list = field(default_factory=list)
    compute: dict = field(default_factory=lambda: dict(math_fidelity="HiFi2", fp32_dest_acc_en=True))
    cbs: dict = field(default_factory=dict)  # role -> [CBUse]
    io: list = field(default_factory=list)  # tensors the body reads / writes (generic_op io)
    status: str = "stub"
    # per composite (role, risc) parts (Stage 16 integration, 11:1xZ): {(role, risc): dict(src, ct, rt, defines,
    # compute)}. A piece whose kernels / args differ per role or RISC (mhc member vs root, PfStream streamer dm0 vs
    # member dm1 / c) registers them here; a (role, risc) missing from `parts` falls back to the fields above.
    parts: dict = field(default_factory=dict)
    # semaphore initial values this body needs ({sid: value}); e.g. the kda items' H3 counter PRE-RAISED to its full
    # count while the H3 producers (P3) are stubs (the incremental unit stages the store; the body's own wait passes)
    sem_init: dict = field(default_factory=dict)

    def part(self, role: str, risc: str) -> dict:
        """the (src, ct, rt, defines, compute) this body contributes to the (role, risc) composite"""
        q = self.parts.get((role, risc))
        if q is not None:
            return dict(src=q.get("src"), ct=list(q.get("ct", [])), rt=dict(q.get("rt", {})),
                        defines=list(q.get("defines", [])), compute=q.get("compute"), config=q.get("config"),
                        noc_xy=q.get("noc_xy"), nonzero=q.get("nonzero"), idle_ok=bool(q.get("idle_ok", False)),
                        fab_at_arg=q.get("fab_at_arg"), noc_words=bool(q.get("noc_words", False)),
                        kdefines=list(q.get("kdefines", [])), regime_flex=bool(q.get("regime_flex", False)),
                        dst_any=bool(q.get("dst_any", False)), fid_mode=q.get("fid_mode"))  # fmt: skip
        if self.parts:  # a parts-style body runs only on the (role, risc) composites it names
            return dict(src=None, ct=[], rt={}, defines=[], compute=None, config=None, kdefines=[])
        return dict(src=getattr(self, risc), ct=list(self.ct), rt=dict(self.rt), defines=list(self.defines),
                    compute=self.compute if risc == "c" else None, config=None, kdefines=[])  # fmt: skip

    def subparts(self, role: str, risc: str, rank: int = 0) -> list:
        """(assembly, 2026-10-22) the ORDERED sub-parts this body contributes to the (role, risc) composite: a
        parts[(role, risc)] entry may be ONE part dict or a LIST of them (several bodies of one piece inside one phase,
        e.g. the P3 projections qkv -> low-rank -> f_b -> g_b as consecutive PfStream phases, or a signal kernel after
        the phase body); each gets its own namespace ph<pid>_<i> and argument bases. A part may carry `rt_by_rank`
        {mesh rank: rt dict} (the fabric lanes' per-chip args): `rt` is resolved for `rank`, `rts` lists every rank's
        dict (the width = the max over all of them, so ph_rt0 is the same on every rank)."""
        q = self.parts.get((role, risc))
        if q is None:
            one = self.part(role, risc)
            one["rts"] = [one["rt"]]
            return [one]
        qs = list(q) if isinstance(q, (list, tuple)) else [q]
        out = []
        for i, sub in enumerate(qs):
            by_rank = sub.get("rt_by_rank")
            if by_rank:
                rt = by_rank.get(rank)
                if rt is None:
                    raise KeyError(f"PMK part {sub.get('name', i)} ({role}, {risc}): no runtime args for rank {rank}")
                rts = list(by_rank.values())
            else:
                rt = sub.get("rt", {})
                rts = [rt]
            out.append(dict(src=sub.get("src"), ct=list(sub.get("ct", [])), rt=dict(rt), rts=rts,
                            defines=list(sub.get("defines", [])), compute=sub.get("compute"), config=sub.get("config"),
                            noc_xy=sub.get("noc_xy"), nonzero=sub.get("nonzero"), idle_ok=bool(sub.get("idle_ok", False)),
                            kdefines=list(sub.get("kdefines", [])),
                            fab_at_arg=sub.get("fab_at_arg"), noc_words=bool(sub.get("noc_words", False)),
                            regime_flex=bool(sub.get("regime_flex", False)), dst_any=bool(sub.get("dst_any", False)),
                            fid_mode=sub.get("fid_mode"), rt_alias=sub.get("rt_alias"), noc_sem=sub.get("noc_sem"),
                            probe_marks=list(sub.get("probe_marks", ())),
                            name=str(sub.get("name", i)) if len(qs) > 1 else ""))  # fmt: skip
        return out


@dataclass
class Phase:
    pid: str
    name: str
    owner: str  # the piece that writes the body
    roles: tuple  # core roles that run it
    rows: int
    units: str  # the work decomposition
    wraps: tuple  # the existing kernel bodies it wraps (paths relative to the tree)
    waits: tuple = ()  # HandOff ids consumed
    signals: tuple = ()  # HandOff ids produced
    collective: str | None = None  # "AG" | "RS" | None
    sub_chunks: int = 1
    stream: PfStreamPhase | None = None  # weight-streaming phases
    body: PhaseBody = field(default_factory=PhaseBody)
    note: str = ""


# ------------------------------------------------------------------------------------------------ placement
@dataclass
class Placement:
    """Contract §2 on an (gx x gy) worker grid: 64 members (8 SP blocks x 8), 8 roots, 32 MM-only, 2 lanes, 4 tail;
    24 bank streamers = BRISC of MM-only / root cores (never members: §4 'tightest core'); 32 KDA scan = the MM-only
    cores. The DS bank-adjacent streamer placement needs the device (mk_readers.reader_assignments); the plan takes
    the first 24 non-member MM cores and the job reports the device's assignment when it has one."""

    gx: int = 11
    gy: int = 10
    lanes_n: int = 2
    tail_n: int = 4
    members_n: int = 64
    roots_n: int = 8
    streamers_n: int = 24
    avoid: tuple = ()  # the device's bank-adjacent DS streamer cores (mk_readers.reader_assignments): never members
    # "rect" (7a's placement ruling, coordinator 11:43Z; every 7a silicon row): the 64 members are ONE 8 x 8 multicast
    # rectangle (PfStream multicasts each weight block once to it; pf_stream._rectangle asserts it); "colmajor" = the
    # family's first placement (members at the end of the free cells: 6 columns + 4 cells, two multicasts per block)
    layout: str = "rect"

    def __post_init__(self):
        if self.layout == "rect" and self.gx >= 11 and self.gy >= 10 and (
            self.members_n, self.roots_n, self.lanes_n, self.tail_n) == (64, 8, 2, 4):  # fmt: skip
            return self._rect()
        self._colmajor()

    def _rect(self):
        """members = logical x [gx - 8, gx), y [0, 8): G = 8 y + (x - x0) -> group b = row b, slice j = column offset;
        roots = (x0 - 1, b), each beside its group's row; streamers = x [0, x0), y [0, 8) (the BRISC of the roots +
        the left MM-only block, as 7a's silicon rows); lanes (0, 8), (1, 8); tail = the next 4 cells of rows >= 8 left
        of the rectangle; MM-only = the left block x [0, x0 - 1) rows 0..7 + rows >= 8 (32 at gx 11, gy 10). The DS
        'avoid' cores are x 0..2 on these chips (never in the rectangle): the prefill program's members own their L1"""
        x0 = self.gx - 8
        self.members = [(x, y) for y in range(8) for x in range(x0, self.gx)]
        self.roots = [(x0 - 1, y) for y in range(8)]
        low = [(x, y) for y in range(8, self.gy) for x in range(self.gx)]  # rows >= 8, row-major
        self.lanes = [(0, 8), (1, 8)]
        rest_low = [c for c in low if c not in self.lanes]
        left_low = [c for c in rest_low if c[0] < x0]
        self.tail = left_low[: self.tail_n]
        self.mm_only = [(x, y) for y in range(8) for x in range(x0 - 1)] + [c for c in rest_low if c not in self.tail]
        self.groups = [self.members[8 * g : 8 * g + 8] for g in range(8)]
        self.streamers = [(x, y) for y in range(8) for x in range(x0)][: self.streamers_n]
        self.kda_scan = self.mm_only[:32]
        self.mm_grid = self.members + self.roots + self.mm_only
        # 7a's ruling: the decode DS-streamer 'avoid' rule does NOT apply to the prefill program's members (its own
        # program and L1); the overlap is recorded for the interleaved decode + prefill step (P4), where it matters
        self.avoid_overlap = sorted(set(map(tuple, self.avoid)) & set(self.members))

    def _colmajor(self):
        cells = [(x, y) for x in range(self.gx) for y in range(self.gy)]  # column-major, as rect_placement
        need = self.lanes_n + self.tail_n + self.members_n + self.roots_n
        if len(cells) < need + 1:
            raise ValueError(f"PMK placement: {len(cells)} cores < {need + 1}")
        avoid = [tuple(c) for c in self.avoid if tuple(c) in cells]
        free = [c for c in cells if c not in avoid]
        i = 0
        self.lanes = free[i : i + self.lanes_n]
        i += self.lanes_n
        self.tail = free[i : i + self.tail_n]
        i += self.tail_n
        rest = free[i:]
        # members at the END of the free cells (whole columns -> a group of 8 = most of a column); the streamer
        # cores (avoid) and the remaining free cells are roots + MM-only
        self.members = rest[-self.members_n :]
        others = avoid + rest[: -self.members_n]
        self.roots = others[-self.roots_n :]
        self.mm_only = others[: -self.roots_n]
        self.groups = [self.members[8 * g : 8 * g + 8] for g in range(self.members_n // 8)]
        self.streamers = (avoid or (self.mm_only + self.roots))[: self.streamers_n]
        self.avoid_overlap = []
        self.kda_scan = self.mm_only[:32] if len(self.mm_only) >= 32 else self.mm_only
        self.mm_grid = self.members + self.roots + self.mm_only

    def role_cores(self) -> dict:
        return dict(member=self.members, root=self.roots, mm_only=self.mm_only, lane=self.lanes, tail=self.tail,
                    streamer=self.streamers, kda_scan=self.kda_scan, mm=self.mm_grid)  # fmt: skip

    @property
    def total(self) -> int:
        return len(self.lanes) + len(self.tail) + len(self.members) + len(self.roots) + len(self.mm_only)


# -------------------------------------------------------------------------------------------- the family
class PmkSparseKdaPlan:
    """The PLAN ENGINE: ONE persistent program per sparse_kda layer at prefill width (contract §3). Build: `plan()` (pure python:
    phases, hand-offs, CB slots, census, sub-chunk policy), `register_phase(pid, body)` (a piece plugs its kernels),
    `descriptor(device, io)` (ttnn ProgramDescriptor with the composite kernels: stubs where no body is registered)."""

    # alias pairs (contract §4 'everything is a CB'; one allocation with two indices = CBDescriptor with two format
    # descriptors): an mHC CB live only in P1 / P10 / P16 shares its allocation with an MM slot live only in the
    # matmul phases. Liveness is checked in plan().
    ALIASES = (
        (CB_MHC_FN, CB_XS_LAND, CB_IN0),
        (CB_MHC_W, CB_PART),
        (CB_MHC_H, CB_MHC_H16, CB_OUT),
        (CB_MHC_X, CB_MHC_X32),
    )
    # VIEWS: indices that ARE the same data in one allocation (pmk_mhc's X / X32 and H / H16 format views), exempt from
    # the liveness rule (they are live together by construction)
    VIEWS = (frozenset((CB_MHC_X, CB_MHC_X32)), frozenset((CB_MHC_H, CB_MHC_H16)))

    def __init__(self, geom: PmkGeom, place: Placement | None = None, *, kb: int = 8, kb_x: int = 32,
                 chunk_store: str = "auto", stationary: str = "auto", ring_slots: int = 3,
                 worker_l1: int = WORKER_L1_SERVED, cb_end_ceiling: int = CB_END_CEILING):  # fmt: skip
        self.g = geom
        self.place = place or Placement()
        self.probe_labels = {}  # GLM53_PMK_PHASE_PROBE: (role, risc) -> the labels of its marks (start, then each pid)
        self.kb, self.kb_x, self.ring_slots = kb, kb_x, ring_slots
        # GLM53_PMK_CB_END (cf3h-k B1, 2026-10-26): the CONSTRUCTION's own CB-end floor -- the served build's persistent L1
        # (12's i4kx L1_REPORT: the decode MK experts' persist buffers, lowest at 1,289,856 on every core) sits below the
        # unit harness's ceiling; the census then holds every role to it and the window hosting ('auto') sizes to it
        cb_end_ceiling = int(os.environ.get("GLM53_PMK_CB_END", "0") or 0) or cb_end_ceiling
        self.worker_l1, self.cb_end_ceiling = worker_l1, cb_end_ceiling
        # the stricter of Stage 14's binding CB end and the Stage 16 ruling bounds the static CBs of every role
        self.cb_budget = min(cb_end_ceiling, STAGE16_CB_END_RULING) - (L1_TOTAL - worker_l1)
        self.chunk_store_mode = chunk_store  # "l1" | "dram" | "auto"
        self.stationary_mode = stationary  # "members64" | "mm96" | "auto"
        self.phases: list[Phase] = []
        self.handoffs: dict[str, HandOff] = {}
        self.cbs: dict[str, dict[int, CBUse]] = {}  # role -> index -> merged use
        self.sems: dict[int, set] = {}  # id -> roles declaring it
        self.census: dict = {}
        self.policy: dict = {}
        self.rings: dict = {}
        self.planned = False

    # ------------------------------------------------------------------------------------------------ helpers
    def _ds(self, role, k, n, dt) -> GeomDS:
        return GeomDS(role, k, n, dt, readers=max(1, self.place.streamers_n // 8))

    def _use(
        self,
        role: str,
        index: int,
        fmt: str,
        pages: int,
        live,
        page=None,
        note="",
        remote_writers=(),
        users=None,
        solo=False,
    ):
        """declare a CB use on a role, merging with earlier uses of the index: one (format, page) per index, the
        size = the largest use; remote_writers = the phases in which other cores write it (cross-core hazard)"""
        u = CBUse(
            index,
            fmt,
            pages,
            tuple(live),
            page,
            note,
            tuple(remote_writers),
            tuple(users if users else live),
            bool(solo),
        )
        d = self.cbs.setdefault(role, {})
        if index in d:
            o = d[index]
            if (o.fmt, o.page_bytes) != (u.fmt, u.page_bytes):
                raise ValueError(f"PMK CB {index} on {role}: ({o.fmt}, {o.page_bytes}) vs ({u.fmt}, {u.page_bytes}) "
                                 f"[{o.note} / {note}]: one (format, page) per index")  # fmt: skip
            d[index] = CBUse(index, fmt, max(o.pages, pages), tuple(sorted(set(o.live) | set(u.live))), page,
                             f"{o.note} | {note}", tuple(sorted(set(o.remote_writers) | set(u.remote_writers))),
                             tuple(sorted(set(o.users) | set(u.users))), o.solo or u.solo)  # fmt: skip
        else:
            d[index] = u
        return u

    def _kda_cbs(self):
        """the KDA phase's CBs from the kda PIECE's own plan (tt/pmk_kda.KdaPlan: the single source of truth, as the mhc
        map): the item CBs (P4 conv / gates + P5 prepare, one kernel set) on the MEMBERS = the H x CPH item cores (64 in
        both geometries), the scan CBs (P6 scan + gated norm, P9 state) on the MM-only cores = the kda_scan cores. Their
        indices are the composites' FREE indices (the members also host the mhc / MM / landing CBs): the kernels take
        them as PK_ITEM_MAP / PK_SCAN_MAP (kda_common.hpp), kept here as self.kda_map. Item CBs are live P4 / P5, scan
        CBs P6 / P9 (users P4 / P6: the registered bodies)."""
        from models.autoports.zai_org_glm_5_3_flash.tt import pmk_kda as _kda  # noqa: PLC0415

        g = self.g
        geo = _kda.KdaGeometry(rows=g.c_rows, heads=g.heads_r, head_dim=g.head_dim, hidden=g.hidden)
        kp = _kda.plan(geo)
        used_m = set(self.cbs.get("member", {}))
        used_s = set(self.cbs.get("mm_only", {})) | set(self.cbs.get("streamer", {}))
        free_m = [i for i in range(64) if i not in used_m]
        free_s = [i for i in range(64) if i not in used_s]
        if len(free_m) < len(kp.item) or len(free_s) < len(kp.scan):
            raise ValueError(f"PMK KDA: {len(free_m)} free member / {len(free_s)} free MM-only CB indices for "
                             f"{len(kp.item)} item / {len(kp.scan)} scan CBs")  # fmt: skip
        imap, smap = free_m[: len(kp.item)], free_s[: len(kp.scan)]
        self.kda_geo, self.kda_map = geo, dict(item=imap, scan=smap)
        for spec, idx in zip(kp.item, imap):
            self._use(
                "member",
                idx,
                spec.fmt,
                spec.pages,
                ("P4", "P5"),
                note=f"kda item {spec.name}",
                users=("P4",),
                solo=True,
            )
        for spec, idx in zip(kp.scan, smap):
            self._use(
                "mm_only",
                idx,
                spec.fmt,
                spec.pages,
                ("P6", "P9"),
                note=f"kda scan {spec.name}",
                users=("P6",),
                solo=True,
            )

    def _sem(self, sid: int, *roles):
        assert 0 <= sid < NUM_SEMAPHORES, sid
        self.sems.setdefault(sid, set()).update(roles)

    def _handoff(self, **kw) -> HandOff:
        h = HandOff(**kw)
        self.handoffs[h.hid] = h
        return h

    # -------------------------------------------------------------------------------------------------- plan
    def plan(self) -> "PmkSparseKdaPlan":
        """plan, then trim the streamers' rings (3 -> 2 slots, largest block first: the inverse of Stage 14's deep-ring
        lever) while a streamer-carrying role (root / MM-only) is over budget; the trims are recorded in the policy"""
        self.ring_trims = []
        self._plan_once()
        while any(not self.census[r]["fits"] for r in ("root", "mm_only")):
            deep = [(k, r) for k, r in self.rings.items() if r["slots"] > 2]
            if not deep:
                break
            k, r = max(deep, key=lambda kr: kr[1]["block_bytes"])
            self.ring_trims.append((k, r["block_bytes"]))
            self._plan_once()
        self.policy["ring_trims"] = [f"{k[0]}x{k[1]} ({_kb(b)}) 3 -> 2 slots" for k, b in self.ring_trims]
        return self

    def _plan_once(self) -> "PmkSparseKdaPlan":
        g, P = self.g, self.place
        self.phases, self.handoffs, self.cbs, self.sems, self.rings, self.credits = [], {}, {}, {}, {}, {}
        mm_n = len(P.mm_grid)
        sp_b, c_b = g.sp_blocks, g.c_blocks
        ht, hd, hr = g.ht, g.head_dim, g.heads_r
        cpc = ht // 8  # member column tiles per stream (P = 8 members per SP block group)
        xt = g.n_streams * cpc  # residual tiles per member per 32-row block
        n_streamers = len(P.streamers)
        tp8_forms = (
            g.sp_rows * g.res_cols * 4 <= len(P.members) * xt * TILE_BYTES["fp32"]
        )  # the residual fits the members
        # -------------------------------------------------------------------- stationary row split (contract §4c)
        # members64: the 64 members each hold ONE 32-row block of the gathered x̂ (64 blocks at the 2,048 bucket);
        # mm96: 64 members + 32 MM-only with an (N / 3)-split (2 blocks per core, 512 KiB). The census decides.
        blocks_per_member = -(-c_b // len(P.members))
        self.policy = dict(blocks=c_b, members=len(P.members), blocks_per_member=blocks_per_member,
                           mm_compute_cores=len(P.members),  # P3 / P7 / P13 / P14 compute = the 64 stationary members only
                           p3_us_bound=round(2 * g.c_rows * g.hidden * (g.qkv_n + g.lr_n) / (297e12 * len(P.members) / 110) * 1e6),
                           compute_note="the other 40 MM cores only stream (24), prepare (40) and scan (32): contract §3's 290 us "
                                        "assumed 104 compute cores; at 64 members P3 is ~320 us at the HiFi2 peak (§4c)",
                           residual="l1 (resident in the members)" if tp8_forms else "dram (7a's MhcPrePf streams it, XBUF = 1)",
                           geometry=f"TP{g.tp}: SP rows {g.sp_rows}, heads/chip {hr}, XSL width {g.xsl_w}",
                           collective_axes=2 if g.tp >= 4 else 1)  # fmt: skip
        nb = 16  # output tiles per N-block per consumer (PART 41 = 1 block x nb fp32 = 64 KB; OUT 42 = nb bf16)

        def mm_phase(role_name, k, n, dt, m_tiles, handoff_in, out, kb=None, fid="HiFi2"):
            ds = self._ds(role_name, k, n, dt)
            ph = PfStreamPhase(ds=ds, kb=kb or self.kb, m_tiles=m_tiles, nb=nb,
                               in0=ActSource(handoff_in, CB_IN0, SEM_FS_ARRIVE), out=out, math_fidelity=fid,
                               rows_of={c: [i] for i, c in enumerate(P.members)})  # fmt: skip
            ph.consumers = [P.groups[s_ % len(P.groups)] for s_ in range(n_streamers)]
            # ring cap (mk_readers.RING_BYTES_MAX): halve the K block while two slots would exceed it
            while ph.kb > 1 and ph.geo(n_streamers)["block_bytes"] > BLOCK_BYTES_MAX:
                ph.kb //= 2
            return ph

        # --------------------------------------------------------------------------------- hand-offs (§4b)
        bf16_blk = TILE * g.hidden * 2  # one 32-row block of a [*, 4096] bf16 tensor
        H0 = self._handoff(hid="H0", name="XS-FREE credit", producer="holder member of block G, after its own P1 / P10 FN use",
                           consumer="the 8 local slice writers of G (group b members, local blocks) + the local lane (-> one per-chip "
                                    "credit forwarded to every peer lane before it writes remote XS)",
                           payload_bytes=0, lives_in=f"member semaphore {SEM_XS_FREE} (local writers) + lane word XSFREE ({FS_W_XSFREE}) "
                                    f"on lane G % {FS_LANES} (+1 per member per use; the lane forwards one per-chip credit to its peer lane)",
                           counter=(SEM_XS_FREE, f"lane word XSFREE: {len(P.members) // FS_LANES} per lane per use"), expected=1,
                           unit="credit per use (cumulative: P2 waits 1, P11 waits 2)")  # fmt: skip
        H0f = self._handoff(hid="H0f", name="XS-FREE credit (FFN use)", producer="holder member after its P10 FN use", consumer=H0.consumer,
                            payload_bytes=0, lives_in=H0.lives_in, counter=H0.counter, expected=1,
                            unit="credit (the second use of id 12 in the layer: P11 waits base 1 + 1 = 2)")  # fmt: skip
        self.credits[CB_XS_LAND] = ("H0", ("P1", "P10"))
        H1 = self._handoff(hid="H1", name="x̂ slice -> holder XS", producer="P1 member (b, j): its 16 col-tile slice of local block b",
                           consumer=f"member holding global block {g.sp_blocks} r + b (XS {CB_XS_LAND}); the lane pulls complete blocks from XS",
                           payload_bytes=sp_b * 8 * cpc * 2048,
                           lives_in=f"holder member XS {CB_XS_LAND} (= IN0 {CB_IN0}); the writer waits XS-FREE (H0) first",
                           counter=f"lane h = b % {FS_LANES}: word LREADY + k (k = index of b in lane h's list)",
                           expected=8, unit="slice: +1 the lane word per slice (the holder's FS_ARRIVE incs are H2's 8)")  # fmt: skip
        H2 = self._handoff(hid="H2", name="x̂ store (remote blocks)", producer="P2 lanes (peer chips' blocks, after the per-chip XS-FREE credit)",
                           consumer="P3: the 64 holder members", payload_bytes=c_b * bf16_blk,
                           lives_in=f"holder member XS landing {CB_XS_LAND} = IN0 {CB_IN0} (stationary row block)",
                           counter=SEM_FS_ARRIVE, expected=8,
                           unit="slice incs per block on EVERY holder (local: the 8 H1 writers; remote: the lane's fused +1 per 16-tile slice)")  # fmt: skip
        chunk_bytes = c_b * TILE * (3 * g.width_r * 2 + g.width_r * 4 + hr * 4)
        H3 = self._handoff(hid="H3", name="q / k / v / g / β chunks", producer="P3 / P4", consumer="P5",
                           payload_bytes=chunk_bytes, lives_in="chunk store (L1 over the MM grid, or DRAM scratch: policy)",
                           counter=SEM_QKV, expected=c_b, unit="KDA chunk (32 rows)")  # fmt: skip
        H3b = self._handoff(hid="H3b", name="W / U / K̃ / Q̃ / intra terms", producer="P5 prepare", consumer="P6 scan",
                            payload_bytes=c_b * hr * (4 * TILE * hd * 2 + TILE * TILE * 4), lives_in=f"prep OUT {CB_KDA_PREP_IDX[3]}/{CB_KDA_PREP_IDX[4]} -> scan {50 + 1}",
                            counter=SEM_KDA, expected=c_b, unit="chunk of this scan core's head (per core)")  # fmt: skip
        H4 = self._handoff(hid="H4", name="O", producer="P6 scan cores", consumer="P7 MM cores",
                           payload_bytes=c_b * TILE * g.width_r * 2, lives_in=f"scan OUT 53 -> MM IN0 {CB_IN0}",
                           counter=SEM_ON, expected=c_b * hr * (hd // TILE), unit="(chunk, head, v-col tile)")  # fmt: skip
        H5 = self._handoff(hid="H5", name="o_proj partial", producer="P7 MM cores", consumer="P8 lanes",
                           payload_bytes=c_b * bf16_blk, lives_in=f"MM OUT {CB_OUT} (OUTADDR written once) -> lane send CBs",
                           counter=f"lane h = b % {FS_LANES}: word LREADY + k per 16-tile slice (OUTADDR / SEMADDR written once)",
                           expected=g.tp, unit="owner block (SP rows)")  # fmt: skip
        H6 = self._handoff(hid="H6", name="y", producer="P8 lanes", consumer="P10 members",
                           payload_bytes=sp_b * TILE * g.hidden * 4, lives_in=f"member POST {CB_POST_Y}", counter=SEM_FS_REDUCED,
                           expected=sp_b // 8, unit="owner block reduced per member (its 32 x 512 fp32 slice; 1 block at TP8, 4 at TP2)")  # fmt: skip
        H7b = self._handoff(hid="H7b", name="IN0 free after P7", producer="P7 consumer on the member (its last IN0 40 pop: the O chunks "
                            "of o_proj; after P14 (dense_kda: P13): the last FFN x rows)",
                            consumer="the mhc P10 part's FN / W prefetch into FN 4 (= XS 35 = IN0 40); the stack's next-layer P1 prefetch",
                            payload_bytes=0, lives_in=f"member semaphore {SEM_IN0_FREE} (cumulative, reset once per layer)",
                            counter=SEM_IN0_FREE, expected=1, unit="release: k = 1 after P7 (P10 waits), 2 after P14 / dense P13 (next P1 waits)")  # fmt: skip
        H7 = self._handoff(hid="H7", name="x̂_f slice -> holder XS", producer="P10 member (b, j)", consumer=H1.consumer,
                           payload_bytes=H1.payload_bytes, lives_in=H1.lives_in + " (k = 2)", counter=H1.counter, expected=H1.expected,
                           unit=H1.unit)  # fmt: skip
        if g.dense:  # dense_kda (Stage 17): the gathered x̂_f only (no routes)
            H8 = self._handoff(hid="H8", name="x̂_f store", producer="P11 lanes", consumer="P13 (dense MLP)",
                               payload_bytes=H2.payload_bytes, lives_in=f"member XS landing {CB_XS_LAND} = IN0 {CB_IN0}",
                               counter=SEM_FS_ARRIVE, expected=8, unit="slice incs per block on every holder (as H2)")  # fmt: skip
        else:
            H8 = self._handoff(hid="H8", name="x̂_f store + routes", producer="P11 lanes / P12 tail", consumer="P12 / P13 / P14",
                               payload_bytes=H2.payload_bytes + g.c_rows * g.top_k * (2 + 2),
                               lives_in=f"member XS landing {CB_XS_LAND} = IN0 {CB_IN0} + tail LIST (ids, weights, per-expert row lists)",
                               counter=SEM_FS_ARRIVE, expected=8, unit="slice incs per block on every holder (as H2); the routes: LOGITS 12 / TAIL 13 (tail cores)")  # fmt: skip
        H9 = self._handoff(hid="H9", name="FFN partial", producer="P13 MM cores" if g.dense else "P13 / P14 MM cores", consumer="P15 lanes",
                           payload_bytes=c_b * bf16_blk, lives_in=f"MM OUT {CB_OUT} (OUTADDR written once)",
                           counter=f"lane h = b % {FS_LANES}: word LREADY + k per 16-tile slice", expected=g.tp,
                           unit="owner block (SP rows)")  # fmt: skip
        H10 = self._handoff(hid="H10", name="y_f", producer="P15 lanes", consumer="P16 members", payload_bytes=H6.payload_bytes,
                            lives_in=H6.lives_in, counter=H6.counter, expected=H6.expected, unit=H6.unit)  # fmt: skip
        mmr = ("member", "root", "mm_only")
        sems = ((SEM_FS_ARRIVE, mmr), (SEM_FS_REDUCED, ("member",)), (SEM_XS_FREE, ("member",)), (SEM_IN0_FREE, ("member",)), (SEM_QKV, ("member",)),
                (SEM_KDA, ("member", "mm_only")), (SEM_ON, mmr), (SEM_PARK_VALID, mmr), (SEM_PARK_READY, ("root", "mm_only")))  # fmt: skip
        if not g.dense:  # the router tail (sparse_kda only)
            sems += ((SEM_LOGITS, ("tail",)), (SEM_TAIL, mmr))
        for sid, roles in sems:
            self._sem(sid, *roles)
        for sid in SEM_MHC:
            self._sem(sid, "member", "root")
        # the kda piece (pmk/kda/README §1): H3 SEM_QKV on the ITEM cores (= the members), the chunk baton / terms-ready
        # SEM_KDA on items + scan, the scan's norm partials (PK_SEM_RMS 0) and chunk-done aggregation (PK_SEM_ONAGG 1)
        # on the scan cores (= MM-only; ids 0 / 1 are the mhc's on members / roots: disjoint cores)
        self._sem(0, "mm_only")
        self._sem(1, "mm_only")
        # (no lane semaphores: producer -> lane readiness is the FabricStream words LREADY + k; lanes are coll's)

        # ---------------------------------------------------------------------- stream phases + rings (§4c)
        qkv = mm_phase("kda_qkv", g.hidden, g.qkv_n, g.w_qkv, c_b, "H2", H3)
        lr = mm_phase("kda_lowrank", g.hidden, g.lr_n, g.w_lr, c_b, "H2", H3)
        o = mm_phase("kda_o", g.width_r, g.hidden, g.w_o, c_b, "H4", H5)
        if g.dense:  # dense_kda: the dense MLP's two stream phases (bf8, HiFi2), every gathered row
            dgu = mm_phase("dense_gate_up", g.hidden, 2 * g.dense_r, g.w_dense, c_b, "H8", H9)
            ddn = mm_phase("dense_down", g.dense_r, g.hidden, g.w_dense, c_b, "H8", H9)
            xffn = None
            self.streams = [qkv, lr, o, dgu, ddn]
        else:
            router = mm_phase("router", g.hidden, g.n_experts, g.w_router, sp_b, "H8", H8, kb=4)
            router.ds.out_dtype = "fp32"
            sgu = mm_phase("shared_gate_up", g.hidden, 2 * g.shared_r, g.w_shared, c_b, "H8", H9)
            sdn = mm_phase("shared_down", g.shared_r, g.hidden, g.w_shared, c_b, "H8", H9)
            xffn = g.xsl_w // TILE <= 8  # §1b: xffn at W <= 8 tiles (TP8); moe_fused_swiglu at the pair's W = 32
            xgu = mm_phase("xsl_gate_up", g.hidden, 2 * g.xsl_w, g.w_experts, -(-g.c_rows * g.top_k // TILE), "H8", H9,
                           kb=self.kb_x, fid="LoFi")  # fmt: skip
            xgu.in0 = ActSource("H8", CB_IN0, SEM_FS_ARRIVE, "row lists")
            xdn = mm_phase(
                "xsl_down", g.xsl_w, g.hidden, g.w_experts, xgu.m_tiles, "H8", H9, kb=self.kb_x // 4, fid="LoFi"
            )
            xdn.in0 = xgu.in0
            self.streams = [qkv, lr, o, router, sgu, sdn, xgu, xdn]
        for s_ in self.streams:
            geo = s_.geo(n_streamers)
            key = geo["ring_key"]
            r = self.rings.setdefault(key, dict(block_bytes=geo["block_bytes"], phases=[], slots=self.ring_slots))
            r["phases"].append(s_.ds.role)
            r["slots"] = self.ring_slots if self.ring_slots * geo["block_bytes"] <= RING_BYTES_MAX else 2
            if any(k_ == key for k_, _ in getattr(self, "ring_trims", [])):
                r["slots"] = 2
        if CB_RING0 + len(self.rings) > CB_KDA_SCAN.start:
            raise ValueError(f"PMK: {len(self.rings)} rings reach the KDA scan CBs at {CB_KDA_SCAN.start}")
        park_block = max(r["block_bytes"] for r in self.rings.values())

        # ----------------------------------------------------------------------------------------- CB uses
        U = self._use
        mhc_live = ("P1", "P10", "P16")
        mm_live = ("P2", "P3", "P7", "P11", "P13") if g.dense else ("P2", "P3", "P7", "P11", "P12", "P13", "P14")
        layer_live = tuple(f"P{i}" for i in range(1, 17))  # live from P1 through P16 (the mhc state between its parts)
        # the mhc piece's kernels are split into per-phase PARTS (P1 = pre(sub 0), P10 = post(sub 0) + pre(sub 1), P16 =
        # post(sub 1)); their state persists between parts: the resident X block, the pre / post / comb coefficient
        # planes (PRE 7, PCOEF 23, COEF 24, CROW 25) and the root's planes (14..19) are LIVE P1 -> P10 -> P16. FN (4)
        # and W (5) are NOT in that set: they share their allocations with the MM slots (XS 35 / IN0 40, PART 41) and are
        # RELOADED at P10 / P16 (fnᵀ 128 KB + w 64 KB per member per reload = 12 MB per chip per layer, ~25 us of DRAM)
        res_live = layer_live
        # the mHC CBs come from the mhc PIECE's own map (tt/pmk_mhc.Plan.cbs(): the indices / pages its kernels use),
        # the single source of truth (Stage 16 integration 11:2xZ: the census had 7a's standalone mhc_pre_pf /
        # mhc_post_pf7a maps -> PCOEF 8 vs 32 / 128 KB, W fp32 vs bf16, TOKEN / PSTAGE missing). Persistent state
        # (X, PRE, PCOEF, the root planes) is live P1 -> P16; FN / W are reloaded per part (alias with XS / PART); Y is
        # the H6 / H10 landing; OUT the H1 / H7 block (read by the lanes in P2 / P11).
        from models.autoports.zai_org_glm_5_3_flash.tt import pmk_mhc as _mhc  # noqa: PLC0415

        mp = _mhc.Plan(sp_rows=g.sp_rows, y_lane=(int(g.tp) > 2))
        persistent = {"X", "PRE", "PCOEF", "RED", "PLANE", "COEF", "SB", "INVR", "SCR"}
        for name, idx, page, pages, roles in mp.cbs():
            if name in ("OUT", "Y"):
                fmt_ = "fp32" if page == 4096 else "bf16"
            else:
                fmt_ = {4096: "fp32", 2048: "bf16"}.get(page, "fp32")
            live = (layer_live if name in persistent else ("P8", "P10", "P15", "P16") if name == "Y"
                    else mhc_live + ("P2", "P11") if name == "OUT" else mhc_live)  # fmt: skip
            pg = None if TILE_BYTES[fmt_] == page else page
            for role in ("member", "root") if roles == "all" else (roles,):
                users = mhc_live + (("P2", "P11") if name == "OUT" else ("P8", "P15") if name == "Y" else ())
                U(role, idx, fmt_, pages, live, page=pg, note=f"mhc {name}", users=users)
                view = {"X": CB_MHC_X32, "H": CB_MHC_H16}.get(name)
                if view is not None:
                    U(role, view, fmt_, pages, live, page=pg, note=f"mhc {name} view", users=users)
        # MM slots on every MM core (members, roots, MM-only): IN0 store, PART, OUT, PARK, probe / arg. USERS (what a
        # program allocates, 14:0xZ): the landing / stationary rows = the AG landings (P2 / P11) + the stream phases
        # that read the rows; PART / OUT / PARK = the stream phases only (the AGs never touch them: P0..P2 or P10..P11
        # programs carried 102 KB of PARK + the accumulators for nothing)
        stream_users = tuple(q for q in mm_live if q not in ("P2", "P11"))
        store_users = mm_live
        store_pages = blocks_per_member * ht  # one 32-row block of x̂ = 128 bf16 tiles = 256 KiB
        for role in mmr:
            if role == "member":
                U(role, CB_XS_LAND, "bf16", store_pages, mm_live, note="XS landing (H1 slices / H2 lanes; credit H0)",
                  remote_writers=("P2", "P11"), users=store_users)  # fmt: skip
                U(role, CB_IN0, "bf16", store_pages, mm_live, note="x̂ / x̂_f stationary row block = the landing",
                  users=store_users)  # fmt: skip
            else:
                U(role, CB_IN0, "bf16", 2 * self.kb, mm_live, note="row K-blocks (P14 x rows / P7 O chunks)",
                  users=stream_users)  # fmt: skip
            U(role, CB_PART, "fp32", nb, mm_live, note=f"fp32 accumulators 1 block x nb={nb} (packer L1 acc)",
              users=stream_users)  # fmt: skip
            U(role, CB_OUT, "bf16", nb, mm_live, note="output tiles -> hand-off (H5 / H9 blocks)", users=stream_users)
            U(role, CB_PARK, "u32", 2, mm_live, page=park_block, note="PARK: 2 forwarded weight blocks (max ring block)",
              users=stream_users)  # fmt: skip
            U(role, CB_PROBE, "u32", 1, ("P0",), page=128, note="probe scratch")
            U(role, CB_ARG, "u32", 1, ("P0",), page=1024, note="stack per-layer args")
        # (the KDA CBs come from the kda PIECE's plan, declared last at the free indices: _kda_cbs below)
        # streamers (BRISC of MM-only / roots): the rings (43+), carried across phases and layers, never rewound
        for i, (key, r) in enumerate(self.rings.items()):
            U("streamer", CB_RING0 + i, "u32", r["slots"], ("P0",) + mm_live, page=r["block_bytes"],
              note=f"ring {key[0]} block {key[1]} tiles: {', '.join(r['phases'])}")  # fmt: skip
        # lanes (coll, dedicated cores): landing 4 x 64 KB + send / reduce (contract §4: ~300 KB; 7a §3 chunked)
        for i in range(4):
            U("lane", CB_LANE0 + i, "bf16", 32, ("P2", "P8", "P11", "P15"), note=f"lane landing slot {i} (64 KB)")
        U("lane", CB_LANE0 + 4, "bf16", 32, ("P2", "P8", "P11", "P15"), note="lane send staging")
        U("lane", CB_LANE0 + 5, "fp32", 16, ("P8", "P15"), note="lane RS reduce ring (fp32)")
        U("lane", CB_PROBE, "u32", 1, ("P0",), page=128, note="probe")
        # tail (4): router tail CBs + the routing lists widened to C rows + the LPT plan (sparse_kda only)
        if not g.dense:
            for i, fmt, n, nm in ((0, "fp32", 2, "LOG"), (1, "fp32", 1, "SCO"), (2, "fp32", 1, "CHO"), (3, "bf16", 1, "IDS"),
                                  (4, "bf16", 1, "W")):  # fmt: skip
                U("tail", CB_TAIL0 + i, fmt, n, ("P12",), note=f"router tail {nm}")
            list_bytes = g.c_rows * g.top_k * 4 + g.c_rows * g.top_k * 2 + g.n_experts * 4 * 2 + 4096
            U("tail", CB_TAIL0 + 5, "u32", 1, ("P12", "P13", "P14"), page=-(-list_bytes // 64) * 64,
              note="routes: ids / weights / per-expert row lists / histogram / LPT plan")  # fmt: skip
        U("tail", CB_PROBE, "u32", 1, ("P0",), page=128, note="probe")
        self._kda_cbs()

        # ------------------------------------------------------------------------------------------ phases
        W = lambda *p: tuple(os.path.join(REL_KDIR, q) for q in p)  # noqa: E731
        ph = self.phases.append
        ag_units = (
            f"{sp_b} local blocks pushed to {g.tp - 1} peer(s) over {self.policy['collective_axes']} axis(es); "
            f"{c_b} blocks land in the holder members' XS {CB_XS_LAND} (local first)"
        )
        ph(
            Phase(
                "P0",
                "ring prologue",
                "family",
                ("streamer",),
                0,
                "qkv + low-rank weight blocks into the rings " "during the previous layer's P15 / P16",
                W("mk/reader_w.cpp"),
                note="stack program only",
            )
        )
        ph(Phase("P1", "mHC pre (attention)", "mhc", ("member", "root"), g.sp_rows,
                 f"{len(P.groups)} groups x (8 members + root): group b = SP blocks b, b + 8, ..; member j = col tiles [{cpc} j, +{cpc}) x 4 streams"
                 + ("" if tp8_forms else "; residual streamed from DRAM (XBUF = 1)"),
                 W("mhc_pre_pf/stream.cpp", "mhc_pre_pf/sync.cpp", "mhc_pre_pf/compute.cpp"), signals=("H0", "H1")))  # fmt: skip
        ph(Phase("P2", "AG(x̂)", "coll", ("lane", "member"), g.c_rows, ag_units + "; members: wait XS-FREE (H0, k = 1), write the "
                 "slice into the holder's XS (H1); lanes: pull complete local blocks, forward the per-chip credit, push to peers",
                 W("fabric_ar/lane_send.cpp", "fabric_ar/lane_recv.cpp"), waits=("H0", "H1"), signals=("H2",), collective="AG"))  # fmt: skip
        ph(Phase("P3", "q / k / v + f_a / g_a / β projections", "kda", ("member", "root", "mm_only", "streamer"), g.c_rows,
                 f"activation-stationary on the {len(P.members)} members ONLY (each holds {blocks_per_member} x 32-row block; ~320 us at "
                 f"the HiFi2 peak, not §3's 290); streamers multicast W K-blocks (kb {qkv.kb}) to their groups (PARK VALID / READY); "
                 "accumulate 1 block x nb in PART",
                 W("mk/reader_w.cpp", "mk/reader_a.cpp", "mk/reader_c.cpp", "mhc_pre_pf/stream.cpp (PF_MCAST loader)"),
                 waits=("H2",), signals=("H3",), stream=qkv, note="low-rank = a second PfStreamPhase on the same rows"))  # fmt: skip
        ph(Phase("P4", "conv1d k=4 + SiLU, q / k L2-norm, f_b / g_b up", "kda", ("member", "root", "mm_only"), g.c_rows,
                 "P3 epilogue per 32-row chunk; 3-row carry per segment (zeros or the slot's conv state)",
                 ("ttnn/.../experimental/kda/fused_recurrent_kda (qkv_causal_conv1d_silu kernels)",), signals=("H3",)))  # fmt: skip
        ph(Phase("P5", "KDA chunk prepare", "kda", ("root", "mm_only"), g.c_rows,
                 f"{c_b} chunks x {hr} heads = {c_b * hr} independent tasks over the {len(P.roots) + len(P.mm_only)} non-member MM cores",
                 ("ttnn/.../experimental/kda (prepare_chunk_recurrence compute)",), waits=("H3",), signals=("H3b",)))  # fmt: skip
        ph(Phase("P6", "KDA inter-chunk scan", "kda", ("kda_scan",), g.c_rows,
                 f"{hr * (hd // TILE)} chains = {hr} heads x {hd // TILE} v-col tiles on {len(P.kda_scan)} cores; {c_b} steps; "
                 "chains reset / load at segment starts", ("ttnn/.../experimental/kda (recurrent_chunk_scan compute)",
                 "kda_rec/compute.cpp (state update, decode lineage)"), waits=("H3b",), signals=("H4",)))  # fmt: skip
        ph(Phase("P7", "gated RMSNorm + o_proj partial", "kda", ("member", "root", "mm_only", "streamer"), g.c_rows,
                 "per chunk under P6: sigmoid_gated_rms_norm then O @ Wo (this chip's heads); emitted in RS owner order",
                 ("ttnn/.../experimental/kda (sigmoid_gated_rms_norm)",) + W("mk/reader_w.cpp", "mk/reader_a.cpp", "mk/reader_c.cpp"),
                 waits=("H4",), signals=("H5", "H7b"), stream=o))  # fmt: skip
        ph(Phase("P8", "RS(o) -> owners", "coll", ("lane",), g.c_rows, f"{g.tp} owner blocks of {g.sp_rows} rows, fixed-order "
                 "fp32 accumulation at the owner (bitwise-reproducible); owner order = chunk order",
                 W("fabric_ar/lane_send.cpp", "fabric_ar/lane_recv.cpp", "fabric_ar/lane_reduce.cpp"), waits=("H5",),
                 signals=("H6",), collective="RS"))  # fmt: skip
        ph(Phase("P9", "KDA cache writes", "cache", ("kda_scan",), 0, f"final S ({hr} x {hd} x {hd} fp32) + conv carry -> "
                 "slot state, per segment", W("prefill/state_io.cpp (STATE_OUT)"), waits=("H4",)))  # fmt: skip
        ph(Phase("P10", "mHC post (attention) + mHC pre (FFN), fused", "mhc", ("member", "root"), g.sp_rows,
                 "out_s = post_s y + Σ C x_i on the block, then the FFN pre on the new residual (one pass)",
                 W("mhc_post_pf7a/reader.cpp", "mhc_post_pf7a/compute.cpp", "mhc_post_pf7a/writer.cpp",
                   "mhc_pre_pf/stream.cpp", "mhc_pre_pf/sync.cpp", "mhc_pre_pf/compute.cpp"), waits=("H6", "H7b"), signals=("H0f", "H7")))  # fmt: skip
        ph(Phase("P11", "AG(x̂_f)", "coll", ("lane", "member"), g.c_rows, "as P2 (XS-FREE k = 2)", W("fabric_ar/lane_send.cpp",
                 "fabric_ar/lane_recv.cpp"), waits=("H0f", "H7"), signals=("H8",), collective="AG"))  # fmt: skip
        if g.dense:  # dense_kda (Stage 17, contract B4): ONE dense SwiGLU MLP slice instead of P12 / P13 / P14
            ph(Phase("P13", "dense MLP, 1/tp slice", "dense", ("member", "root", "mm_only", "streamer"), g.c_rows,
                     f"gate_up {g.hidden} x {2 * g.dense_r}, SwiGLU, down {g.dense_r} x {g.hidden}: two PfStreamPhases "
                     f"(the P13 shared-expert mechanism at the dense width; activation-stationary on the members)",
                     W("mk/reader_w.cpp", "mk/reader_a.cpp", "mk/reader_c.cpp", "swiglu/compute.cpp"), waits=("H8",),
                     signals=("H9", "H7b"), stream=dgu,
                     note="dense_down streams after dense_gate_up (same rings); its last IN0 pop is the layer's 2nd IN0 release (H7b)"))  # fmt: skip
        else:  # sparse_kda: router (P12), shared expert (P13), routed experts (P14)
            ph(Phase("P12", "router + top-8 + histogram + LPT plan", "moe", ("member", "root", "mm_only", "streamer", "tail"), g.c_rows,
                     f"fp32 logits {g.c_rows} x {g.hidden} @ {g.hidden} x {g.n_experts} (redundant per chip) -> tail: sigmoid + bias, "
                     "top-8, norm x 2.5, per-expert row lists, LPT expert-slice -> group plan",
                     W("mk/reader_w.cpp", "mk/reader_a.cpp", "mk/reader_c.cpp", "router_tail/reader.cpp", "router_tail/compute.cpp",
                       "router_tail/writer.cpp", "prefill/moe_plan_mc_count.cpp", "prefill/moe_plan_mc_runs.cpp"),
                     waits=("H8",), signals=("H8",), stream=router))  # fmt: skip
            ph(Phase("P13", "shared expert, 1/tp slice", "moe", ("member", "root", "mm_only", "streamer"), g.c_rows,
                     f"gate_up {g.hidden} x {2 * g.shared_r}, SwiGLU (clamp 10), down {g.shared_r} x {g.hidden}: two PfStreamPhases",
                     W("mk/reader_w.cpp", "mk/reader_a.cpp", "mk/reader_c.cpp", "swiglu/compute.cpp"), waits=("H8",),
                     signals=("H9",), stream=sgu, note="shared_down streams after gate_up (same rings)"))  # fmt: skip
            slice_kb = (
                (xgu.ds.k // TILE * (xgu.ds.n_pad // TILE) + xdn.ds.k // TILE * (xdn.ds.n_pad // TILE))
                * TILE_BYTES[g.w_experts]
                // 1024
            )
            if xffn:
                p14 = Phase("P14", "routed experts, XSL slice (xffn)", "moe", ("member", "root", "mm_only", "streamer"), g.c_rows * g.top_k,
                            f"expert-major: each of {g.n_experts} slices ({slice_kb:,} KB bfp4, W = {g.xsl_w // TILE} tiles) streamed once by its group and "
                            f"applied to its row list (avg {g.c_rows * g.top_k / g.n_experts:.0f} rows, LoFi); weighted partials accumulate per row in L1",
                            W("prefill/moe_xffn_reader.cpp", "prefill/moe_xffn_compute.cpp", "prefill/moe_xffn_writer.cpp"),
                            waits=("H8",), signals=("H9", "H7b"), stream=xgu)  # fmt: skip
            else:
                p14 = Phase("P14", "routed experts, XSL slice (moe_fused_swiglu)", "moe", ("member", "root", "mm_only", "streamer"), g.c_rows * g.top_k,
                            f"W = {g.xsl_w // TILE} tiles > 8: xffn would re-read weights per tile row (§1b) -> today's TP2 path "
                            f"(moe_fused_swiglu: K over rows, hidden over columns per expert); {g.n_experts} slices of {slice_kb:,} KB",
                            ("ttnn/.../moe_fused_swiglu kernels (the moe piece decides)",) + W("prefill/moe_combine_rm_reader.cpp"),
                            waits=("H8",), signals=("H9", "H7b"), stream=xgu, note="the moe piece decides the TP2 body")  # fmt: skip
            ph(p14)
        ph(Phase("P15", "RS(FFN partials) -> owners", "coll", ("lane",), g.c_rows,
                 "as P8; the dense MLP partials" if g.dense else "as P8; routed + shared partials summed in one pass",
                 W("fabric_ar/lane_send.cpp", "fabric_ar/lane_recv.cpp", "fabric_ar/lane_reduce.cpp"), waits=("H9",),
                 signals=("H10",), collective="RS"))  # fmt: skip
        ph(Phase("P16", "mHC post (FFN)", "mhc", ("member", "root"), g.sp_rows, "fused into the next layer's P1 (stack) or "
                 "written to the DRAM residual at a program boundary", W("mhc_post_pf7a/reader.cpp", "mhc_post_pf7a/compute.cpp",
                 "mhc_post_pf7a/writer.cpp"), waits=("H10",)))  # fmt: skip

        self._assign_wait_bases()
        self._liveness_check()
        self.census = self._census()
        self._policy()
        self.planned = True
        return self

    def _assign_wait_bases(self):
        """cumulative counters: for every program semaphore id, the hand-offs that wait on it in consumer-phase order
        get wait_base = the sum of the earlier ones' increments (reset once per layer, never between hand-offs)"""
        order = {p.pid: n for n, p in enumerate(self.phases)}
        first_wait = {}
        for p in self.phases:
            for hid in p.waits + self._drains_before(p.pid):
                first_wait.setdefault(hid, order[p.pid])
        by_id = {}
        for h in self.handoffs.values():
            if isinstance(h.counter, int) or (isinstance(h.counter, tuple) and isinstance(h.counter[0], int)):
                sid = h.counter if isinstance(h.counter, int) else h.counter[0]
                by_id.setdefault(sid, []).append(h)
        self._wait_order = {}
        for sid, hs in by_id.items():
            hs.sort(key=lambda h: (first_wait.get(h.hid, 999), h.hid))
            self._wait_order[sid] = [h.hid for h in hs]
            base = 0
            for h in hs:
                h.wait_base = base
                base += h.expected

    def _wait_value(self, h) -> int:
        """the value a consumer of hand-off h waits for in THIS program. Default: h.wait_value (the layer-wide cumulative
        base). wait_base_scope (G v2a, 2026-10-25 gv2w hang): a program semaphore restarts at 0 every launch, so the base
        counts only the earlier hand-offs on h's counter whose producer runs in this program -- the DSA MoE-half program
        has P11's AG but no P2's, and H8's layer-wide base 8 (H2's slices) left P12's members waiting FS_ARRIVE >= 16"""
        if not getattr(self, "wait_base_scope", False) or self.scope is None:
            return h.wait_value
        sid = h.counter if isinstance(h.counter, int) else h.counter[0]
        base = 0
        for hid in (getattr(self, "_wait_order", {}) or {}).get(sid, ()):
            if hid == h.hid:
                break
            if self._has_live_producer(hid):
                base += self.handoffs[hid].expected
        return base + h.expected

    def _h0f_risc(self) -> str:
        """the member RISC that raises P10's XS-FREE credit H0f: dm1 (the default, as P1's H0), or dm0 when P10 STORES the
        resident residual from CB_X on member dm0 (TP8: pmk_x_io 'xstore', the last P10 part there) and the credit's
        consumers (P11's slice writers) run in this program. CB_X / X32 share the window allocation with XS 35 / IN0 40,
        so the credit must follow the store's last read of X: raised from dm1 it let remote x-hat_f slices land in X
        while the store still read it (the TP8 x1 race of w7tp8 / w7tp8c / w7tp8d, 2026-10-25: x1 corrupt on chips 0-4,
        replay to replay, everything P10 computed in L1 exact)"""
        p10 = next((p for p in self.phases if p.pid == "P10"), None)
        if p10 is None or p10.body.status != "body" or not self.in_scope("P10") or not self.in_scope("P11"):
            return "dm1"
        q = p10.body.parts.get(("member", "dm0"))
        qs = list(q) if isinstance(q, (list, tuple)) else ([q] if q is not None else [])
        return "dm0" if any(isinstance(x, dict) and x.get("name") == "xstore" for x in qs) else "dm1"

    def _drain_hids(self, role: str, pid: str) -> tuple:
        """the RECLAIM DRAIN hand-offs `role` waits before entering phase pid's reclaim window (live producers only)"""
        rec = (self.win_reclaim or {}).get(role)
        if rec is None or rec["pid"] != pid or not self.in_scope(pid):
            return ()
        return tuple(h for h in (self.reclaim_drain or {}).get(role, ()) if self._has_live_producer(h))

    def _drains_before(self, pid: str) -> tuple:
        out = ()
        for role in self.reclaim_drain or {}:
            out += tuple(h for h in self._drain_hids(role, pid) if h not in out)
        return out

    def wait_schedule(self) -> dict:
        """{semaphore id: [(hand-off, consumer phase, wait value)] in program order}: the unit test's view"""
        order = {p.pid: n for n, p in enumerate(self.phases)}
        out = {}
        for p in self.phases:
            for hid in p.waits + self._drains_before(p.pid):
                h = self.handoffs[hid]
                sid = (
                    h.counter
                    if isinstance(h.counter, int)
                    else (h.counter[0] if isinstance(h.counter, tuple) and isinstance(h.counter[0], int) else None)
                )
                if sid is not None:
                    out.setdefault(sid, []).append((hid, p.pid, self._wait_value(h)))
        return {k: sorted(v, key=lambda t: order[t[1]]) for k, v in out.items()}

    def _wait_reach_check(self, init: dict):
        """every wait this program emits must be reachable IN this program: value <= the semaphore's initial value + the
        expected counts of the hand-offs on its counter whose first producer is live here (the gv2w hang, 2026-10-25:
        H8 waited FS_ARRIVE 16 with only P11's 8 slices in the program). Raises under wait_base_scope; otherwise logs
        (the served programs predate the check)"""
        reach = {}
        for h in self.handoffs.values():
            sid = h.counter if isinstance(h.counter, int) else (
                h.counter[0] if isinstance(h.counter, tuple) and isinstance(h.counter[0], int) else None)  # fmt: skip
            if sid is not None and self._has_live_producer(h.hid):
                reach[sid] = reach.get(sid, int(init.get(sid, 0))) + int(h.expected)
        bad = []
        for sid, ws in self.wait_schedule().items():
            for hid, pid, v in ws:
                if self.in_scope(pid) and self._has_live_producer(hid) and v > reach.get(sid, int(init.get(sid, 0))):
                    bad.append(f"{hid} in {pid}: wait {sid} >= {v} > reach {reach.get(sid, 0)}")
        if bad:
            msg = f"PMK program {sorted(self.scope) if self.scope else 'all'}: unreachable waits: " + "; ".join(bad)
            if getattr(self, "wait_base_scope", False):
                raise ValueError(msg)
            print("WARNING " + msg, flush=True)

    # ------------------------------------------------------------------------------------------- analysis
    def _liveness_check(self):
        """(1) two indices of one allocation are never live in the same phase of this core's order; (2) an index that
        OTHER cores write (remote_writers: the local slice writers, the lanes, peer chips) is a cross-core hazard the
        phase order of one core cannot see: a chip / member that finishes early would overwrite a lagging core's
        partner CB. Such an index needs a CREDIT hand-off (self.credits[index] = (HandOff id, phases after which the
        holder raises it)) covering the partner's last live phase before every remote-write phase."""
        order = {p.pid: n for n, p in enumerate(self.phases)}
        for grp in self.ALIASES:
            for role, d in self.cbs.items():
                have = [i for i in grp if i in d and not d[i].solo]
                for n, a in enumerate(have):
                    for b in have[n + 1 :]:
                        both = set(d[a].live) & set(d[b].live)
                        view = any({a, b} <= v for v in self.VIEWS)  # one datum, two format views
                        if both and not view and not (a == CB_XS_LAND or b == CB_XS_LAND):  # the landing IS the store
                            raise ValueError(f"PMK alias CB {a} / {b} on {role}: both live in {sorted(both)}")
                for a in have:
                    for w in d[a].remote_writers:
                        partners = [b for b in have if b != a and not (a == CB_XS_LAND and b == CB_IN0)]
                        for b in partners:
                            before = [l for l in d[b].live if order.get(l, 99) < order.get(w, 99)]
                            if not before:
                                continue
                            last = max(before, key=lambda l: order[l])
                            hid, after = self.credits.get(a, (None, ()))
                            if hid is None or last not in after:
                                raise ValueError(
                                    f"PMK cross-core hazard: CB {a} on {role} is written by other cores in {w} while "
                                    f"CB {b} (same allocation) was live in {last}: needs a credit raised after {last} "
                                    f"(credits = {self.credits})"
                                )

    def _slots(self, role: str, active: set | None = None) -> list:
        """[(indices, bytes, fmt/page, note)]: the allocations of a role (aliased indices share one). active = the
        pids with a registered body: only the CB uses live in one of them are allocated (an INCREMENTAL unit's
        program; a stub touches no CB). None = the full layer's census."""
        d = dict(self.cbs.get(role, {}))
        if role in ("root", "mm_only") and "streamer" in self.cbs:
            d.update(self.cbs["streamer"])  # the streamer's rings sit on MM-only / root cores
        d_all = dict(d)
        if active is not None:
            d = {i: u for i, u in d.items() if set(u.users or u.live) & active}
        out, done = [], set()
        for grp in getattr(self, "PINNED", ()):  # PINNED groups first, in every program (active or not)
            have = [i for i in grp if i in d_all and getattr(d_all[i], "pinned", False)]
            if have:
                us = [d_all[i] for i in have]
                out.append((tuple(have), max(u.nbytes for u in us), " | ".join(f"{u.fmt}/{u.page_bytes}" for u in us),
                            "PINNED " + " <-> ".join(u.note for u in us)))  # fmt: skip
                done |= set(have)
        for i in sorted(
            i for i, u in d.items() if getattr(u, "uniform", False) and i not in done
        ):  # the uniform-address CBs next
            u = d[i]
            out.append(((i,), u.nbytes, f"{u.fmt}/{u.page_bytes}", "UNIFORM " + u.note))
            done.add(i)
        for grp in self.ALIASES:
            have = [i for i in grp if i in d and not d[i].solo and i not in done]
            if len(have) > 1:
                us = [d[i] for i in have]
                out.append((tuple(have), max(u.nbytes for u in us), " | ".join(f"{u.fmt}/{u.page_bytes}" for u in us),
                            "ALIAS " + " <-> ".join(u.note for u in us)))  # fmt: skip
                done |= set(have)
        for i in sorted(d):
            if i not in done:
                u = d[i]
                out.append(((i,), u.nbytes, f"{u.fmt}/{u.page_bytes}", u.note))
        return out

    scope = None  # the phase subset of the PROGRAM being built (None = the whole layer as one program)

    def in_scope(self, pid: str) -> bool:
        return self.scope is None or pid in self.scope

    def active_phases(self) -> set | None:
        """the pids with a registered body (in the current program's scope), or None when the scope is the whole
        layer and every phase with rows has a body (the full layer: the full census)"""
        act = {p.pid for p in self.phases if p.body.status == "body" and self.in_scope(p.pid)}
        if self.scope is not None:
            return act
        full = {p.pid for p in self.phases if p.rows or p.pid in ("P9",)}
        return None if full <= act else act

    # PHASE WINDOWS (Stage 14, the per-layer kernel framework, per_layer_kernel_design.md 2.1 / 2.2; the A3 + A4 toy):
    # a program whose scope runs several phase families back to back on a role whose CB union does not fit. The
    # assembly sets window_plan = {frozenset(program scope): [window 1 pids, window 2 pids, ..]} and window_roles. On a
    # windowed role a CB used in exactly ONE window is a window CB: window 1's are declared in place (contiguous, after
    # the role's shared CBs, from PMK_WIN_BASE), every later window's ONE PAGE each (their formats); the composites
    # re-lay them out at the window entry into window 1's dead region (kernels/pmk/pmk_window.hpp: the core-local
    # barrier of the role's three composites on three counter-only barrier CBs, then each RISC's own interface rewrite).
    # A CB of several windows (or of none, the probe), the PINNED / UNIFORM / ALIAS groups and the other roles stay as
    # before. CB addresses: the allocator aligns every CB to the DRAM alignment (program.cpp allocate_circular_buffers).
    window_plan = None
    window_roles = ("member",)
    # NO-CARRY allocations (A15, 2026-10-24): CB indices whose data never crosses a window boundary even between ADJACENT
    # windows (the assembly vouches: the mhc X / H / OUT / Y views are dead at P1's end and fresh in P10; XS / IN0 / OUT 42
    # are filled inside their window, remote writers reach XS through c1_stub_prod's in-window XSADDR publish) -> placed
    # in EACH window that uses them instead of SHARED in place
    WIN_NO_CARRY = ()
    # WIN_TRIM_SCOPES (cf3h-k member shave, 2026-10-26): program scopes whose LATER windows place an alias group by the
    # members that window uses (the others get killed at its entry); window 1 keeps the whole group (the launch layout)
    WIN_TRIM_SCOPES = ()
    WIN_ALIGN = 64
    WIN_BAR_PAGE, WIN_BAR_PAGES = 16, 8  # a barrier CB: counters only (one push per window entry, never popped)

    # PER-ROLE window lists (M-C, A15 at TP8, 2026-10-24): {role: {frozenset(program scope): [window pids, ..]}}. A
    # windowed role that holds no CB of some windows (the TP8 MM-only cores: no P1 / P10 CB) takes its own list
    # (P2 P3 | P4 P6 | P7 P8); a role / scope without an entry takes window_plan's. The core barrier is per core (the
    # three composites of ONE core), so two roles may window the same program differently.
    window_plan_role = None
    # THE RECLAIM WINDOW (Stage 16 M-B, 2026-10-24): {role: dict(pid, allocs, fresh)} -- a piece whose CB set is
    # larger than any window region (program B: 1.1 MB of receiver CBs at TP2) runs as the program's LAST window,
    # laid out from the CB BASE over every dead allocation (the shared ones included; only the barrier CBs survive),
    # its allocs [([(index, page), ..], bytes)] placed from the CB base around the PROTECTED shared allocations (live in
    # the window: CB_ARG), reusing earlier windows' indices of the same format
    # where it must (pmk_window.hpp enter_reclaim balances / resyncs their page counters); `fresh` = its indices no
    # earlier allocation declares (CBUse, declared like a later window's: minimal / hosted). The piece registers no
    # CBUse for them (one (format, page) per index stays the engine's rule for every other window).
    win_reclaim = None
    # RECLAIM DRAIN {role: (hand-off id, ..)} (W7 step 1, 2026-10-25): hand-offs whose REMOTE writes land in a CB the
    # reclaim window lays out over (P11's x̂_f packets from the peer chip's lanes into the holders' XS 35 = B's CBs 49 / 59)
    # -> the role's data-movement RISCs wait them BEFORE enter_reclaim; the core barrier alone sees only its own traffic,
    # and late peer packets overwrote B's CBs on device 1 (w7s1w hang 14:48Z: device 1's members stuck inside B)
    reclaim_drain = None
    # RECLAIM PROTECT {role: (CB index, ..)} (W7 (I1), 2026-10-25): per-window allocations of the window right before the
    # reclaim whose BYTES the reclaim keeps (XS 35: B gathers the routed x-hat_f rows from the holders' XS, lever 2) --
    # holes at their window offset, like the shared allocations live in the reclaim window
    reclaim_protect = None
    # the DRAM arg-table row of THIS engine's programs (GLM53_PMK_ARGS=table): ARG_PAGE (256 args) by default; a fused
    # program whose member BRISC composite carries more (M-B: A15 + program B's receiver args, ~500) takes 2,048
    arg_page = ARG_PAGE

    def windows(self, role: str | None = None) -> list | None:
        if not self.window_plan or self.scope is None:
            return None
        if role is not None and self.window_plan_role:
            wr = (self.window_plan_role.get(role) or {}).get(frozenset(self.scope))
            if wr:
                return [set(x) for x in wr]
        w = self.window_plan.get(frozenset(self.scope))
        return [set(x) for x in w] if w else None

    def _win_index(self, pid: str, wins: list):
        for w, ws in enumerate(wins):
            if pid in ws:
                return w
        return None

    def _win_layout(self, role: str) -> dict | None:
        """the windowed role's declaration of the current program: decl = [(CBUse list, declared bytes)] in order,
        tabs = {W >= 1: [(index, offset, bytes, page)]}, base = window 1's first index, bar = (D0, D1, T) indices,
        span = window 1's bytes, need = {W: bytes}; None = the role is not windowed in this program"""
        wins = self.windows(role)
        if not wins or role not in self.window_roles:
            return None
        act = self.active_phases()
        d = dict(self.cbs.get(role, {}))
        if role in ("root", "mm_only") and "streamer" in self.cbs:
            d.update(self.cbs["streamer"])
        al = lambda b: -(-int(b) // self.WIN_ALIGN) * self.WIN_ALIGN  # noqa: E731
        trim = frozenset(self.scope) in set(self.WIN_TRIM_SCOPES or ())
        shared, per = [], {w: [] for w in range(len(wins))}
        for idx, nbytes, _, note in self._slots(role, act):
            uses = [d[i] for i in idx]
            ws = set()
            for u in uses:
                us = set(u.users or u.live) & (act if act is not None else set(u.users or u.live))
                ws |= {w for w in (self._win_index(q, wins) for q in us) if w is not None}
            fixed = any(getattr(u, "pinned", False) or getattr(u, "uniform", False) for u in uses)
            # SHARED (in place, never overwritten): PINNED / UNIFORM, used in every window (the arg rows, probes), or used
            # in two ADJACENT windows (its data may cross that boundary). Otherwise the allocation (an ALIAS group as one:
            # every index at the same offset) is placed in EACH window that uses it: across a gap window its data cannot
            # survive anyway (PfStream's PARK / OUT / XS-IN0 in P3 and P7 with the kda items between, stage 2)
            adjacent = any(w in ws and w + 1 in ws for w in range(len(wins))) and not any(
                u.index in self.WIN_NO_CARRY for u in uses
            )
            if ws and not fixed and len(ws) < len(wins) and not adjacent:
                for w in sorted(ws):
                    us_w = uses
                    if trim and w > 0 and len(uses) > 1:  # the members window w uses (VIEWS share their users)
                        us_w = [u for u in uses if any(self._win_index(q, wins) == w for q in
                                                       set(u.users or u.live) & (act if act is not None else set(u.users or u.live)))]  # fmt: skip
                    if us_w is uses or len(us_w) == len(uses):
                        per[w].append((uses, nbytes))
                    else:
                        per[w].append((us_w, max(int(u.nbytes) for u in us_w)))
            else:
                shared.append((uses, nbytes))
        rec0 = (self.win_reclaim or {}).get(role)
        rw0 = self._win_index(rec0["pid"], wins) if (rec0 is not None and rec0.get("allocs")) else None
        if not any(per[w] for w in range(1, len(wins))) and not (rw0 is not None and rw0 > 0):
            # nothing to re-lay out (an incremental program: the later windows' bodies are not registered yet). A role
            # whose only later window is the RECLAIM window still re-lays out (G v2a's MM-only role, windows (P12, P13) |
            # (P14): register_g strips every engine CB use out of P14, and returning here left G's reclaim CBs unlaid ->
            # get_write_ptr(CB_PLAN) = 0 -> plan_count's first read into L1 0x540: the gv2w2 watcher NOC error 22:06:53Z)
            return None
        if not per[0]:
            if act is not None and not (set(wins[0]) & set(act)):  # incremental registration: window 1's phases are not
                return None  # registered yet (a later window's are: P10 before P2 / P3, the a2fp10 bisect scope)
            raise ValueError(
                f"PMK windows on {role}: window 1 ({sorted(wins[0])}) has no CB of its own to host the others"
            )
        # WIN_CARRY_FIRST (G v2a: XS 35 carries P11's landed block from window 1 (P10, P11) into P12's and P13's windows):
        # those allocations go FIRST in every window, so they sit at the same window offset everywhere (_p13_carry_check)
        cf = set(getattr(self, "WIN_CARRY_FIRST", ()) or ())
        if cf:
            for w in per:
                per[w].sort(key=lambda e_: 0 if any(u.index in cf for u in e_[0]) else 1)
        used = {u.index for us, _ in shared for u in us} | {u.index for w in per for us, _ in per[w] for u in us}
        in0 = {us[0].index for us, _ in per[0]}
        free = [i for i in range(63, -1, -1) if i not in used]
        if len(free) < 4:
            raise ValueError(
                f"PMK windows on {role}: 4 free CB indices for the barrier / pad CBs (used {sorted(used)})"
            )
        bar = tuple(free[:3])
        span = sum(al(b) for _, b in per[0])
        tab0, off0 = [], 0  # window 1's launch layout (the CBs' own offsets from the region base), for checks
        for us, b in per[0]:
            for u in us:
                tab0.append((u.index, off0, int(b), int(u.page_bytes)))
            off0 += al(b)
        tabs, need = {}, {}
        for w in range(1, len(wins)):
            off, tab = 0, []
            for us, b in per[w]:
                for u in us:
                    tab.append((u.index, off, int(b), int(u.page_bytes)))
                off += al(b)
            tabs[w], need[w] = tab, off
        rec = (self.win_reclaim or {}).get(role)
        rw = self._win_index(rec["pid"], wins) if rec is not None else None
        # WIN_RECLAIM2 (B7g, 2026-10-25): a SECOND reclaim window right after the first, e.g. P15's RS8 reducer CBs after
        # program B's on the MM-only / root roles -- the region then holds max(B, RS8) instead of their sum (the B7g mock:
        # 877 + 576 KB over the MM-only budget); pmk_window.hpp enter_reclaim<W> is per window index (counter-only barriers)
        rec2 = (getattr(self, "win_reclaim2", None) or {}).get(role)
        rw2 = self._win_index(rec2["pid"], wins) if (rec2 is not None and rw is not None) else None
        reclaim = {}
        protect = set()
        if (
            rw is not None and rw > 0
        ):  # the reclaim window: offsets from the CB base = window 1's base - the shared bytes
            shared_b = sum(al(b) for _, b in shared)
            unreg_tail = all(not (set(w_) & set(act if act is not None else set(w_))) for w_ in wins[rw + 1 :])
            assert rw == len(wins) - 1 or (rw2 is not None and rw2 == len(wins) - 1 and rw == len(wins) - 2) or unreg_tail, (
                f"PMK windows on {role}: the reclaim window ({rec['pid']}) must be the program's last (or followed by "
                f"the second reclaim window only)", wins)  # fmt: skip
            assert not tabs.get(rw), (f"PMK windows on {role}: engine CBs in the reclaim window", tabs.get(rw))
            if rw2 is not None:
                assert not tabs.get(rw2), (
                    f"PMK windows on {role}: engine CBs in the second reclaim window",
                    tabs.get(rw2),
                )
            # PROTECTED: a shared allocation live in the reclaim window (the arg-table rows CB_ARG: rta_l1_base points
            # into it for the whole launch) keeps its bytes -- the reclaim allocations are placed around it
            holes, off = [], 0
            for us, b in shared:
                if any(rec["pid"] in tuple(u.users or u.live) for u in us):
                    holes.append((off, off + al(b)))
                    protect |= {u.index for u in us}
                off += al(b)
            for i_p in (self.reclaim_protect or {}).get(role, ()):  # (W7 (I1)) kept in place from the previous window
                ent = [t for t in tabs.get(rw - 1, ()) if int(t[0]) == int(i_p)]
                assert ent, (
                    f"PMK windows on {role}: reclaim-protected CB {i_p} not laid out in window {rw}",
                    tabs.get(rw - 1),
                )
                o_, n_ = int(ent[0][1]), int(ent[0][2])
                holes.append((shared_b + o_, shared_b + o_ + al(n_)))
                grp = next((set(g_) for g_ in self.ALIASES if int(i_p) in g_), {int(i_p)})
                protect |= grp
                self.reclaim_protected = dict(getattr(self, "reclaim_protected", None) or {})
                self.reclaim_protected[(role, int(i_p))] = (shared_b + o_, n_)
            # first-fit decreasing over the gaps between the protected holes (a sequential fill left up to one large
            # allocation's bytes unused before a hole: M-B mock 06:20Z, +115 KB)
            tab, gaps = [], []
            lo_ = 0
            for hlo, hhi in sorted(holes):
                if hlo > lo_:
                    gaps.append([lo_, hlo])
                lo_ = max(lo_, hhi)
            gaps.append([lo_, None])  # the open end (the pad extends the region to it)
            for ents, total in sorted(rec["allocs"], key=lambda a: -int(a[1])):
                for g_ in gaps:
                    if g_[1] is None or g_[1] - g_[0] >= al(total):
                        at = g_[0]
                        g_[0] += al(total)
                        break
                for i_, page_ in ents:
                    tab.append((int(i_), at, int(total), int(page_)))
            cur = max(t[1] + al(t[2]) for t in tab) if tab else 0
            tabs[rw] = sorted(tab, key=lambda t: t[1])
            clash = (set(free[:4]) | protect) & {t[0] for t in tabs[rw]}
            assert (
                not clash
            ), f"PMK windows on {role}: reclaim indices {sorted(clash)} on the barrier / pad / protected indices"
            need[rw] = cur - shared_b  # relative to window 1's base (the pad extends the region to it)
            reclaim[rw] = shared_b
            if rw2 is not None:
                holes2, off = [], 0
                for us, b in shared:
                    if any(rec2["pid"] in tuple(u.users or u.live) for u in us):
                        holes2.append((off, off + al(b)))
                        protect |= {u.index for u in us}
                    off += al(b)
                tab2, gaps2, lo_ = [], [], 0
                for hlo, hhi in sorted(holes2):
                    if hlo > lo_:
                        gaps2.append([lo_, hlo])
                    lo_ = max(lo_, hhi)
                gaps2.append([lo_, None])
                for ents, total in sorted(rec2["allocs"], key=lambda a: -int(a[1])):
                    for g_ in gaps2:
                        if g_[1] is None or g_[1] - g_[0] >= al(total):
                            at = g_[0]
                            g_[0] += al(total)
                            break
                    for i_, page_ in ents:
                        tab2.append((int(i_), at, int(total), int(page_)))
                tabs[rw2] = sorted(tab2, key=lambda t: t[1])
                clash2 = (set(free[:4]) | protect) & {t[0] for t in tabs[rw2]}
                assert (
                    not clash2
                ), f"PMK windows on {role}: reclaim-2 indices {sorted(clash2)} on the barrier / pad / protected indices"
                need[rw2] = (max(t[1] + al(t[2]) for t in tab2) if tab2 else 0) - shared_b
                reclaim[rw2] = shared_b
        # a later window larger than window 1 (TP8 MM-only: the o_proj rings + RS8 reducers 589,824 B vs the scan's 331,776):
        # a PAD CB right after window 1's CBs extends the region to the largest window (counters / data never touched)
        pad = max([0] + list(need.values())) - span

        padcb = [CBUse(index=free[3], fmt="u32", pages=pad // self.WIN_ALIGN, live=(), page=self.WIN_ALIGN,
                       note="window region pad")] if pad > 0 else []  # fmt: skip
        barcb = [CBUse(index=i, fmt="bf16", pages=self.WIN_BAR_PAGES, live=(), page=self.WIN_BAR_PAGE, note="window barrier")
                 for i in bar]  # fmt: skip
        def minimal(
            us, b
        ):  # a later window's allocation declared for its formats only: the smallest total every page divides
            m = 1
            for u in us:
                m = m * int(u.page_bytes) // math.gcd(m, int(u.page_bytes))
            return m if m <= b else b

        later, seen = [], set(
            in0
        )  # a later-window allocation declared ONCE (minimal), unless window 1 declares it in place
        if trim:  # a trimmed group's later windows may hold different subsets: declare each index once
            seen |= {u.index for us, _ in per[0] for u in us}
        for w in range(1, len(wins)):
            for us, b in per[w]:
                if trim:
                    new = [u for u in us if u.index not in seen]
                    if new:
                        seen |= {u.index for u in new}
                        later.append((new, minimal(new, b if len(new) == len(us) else max(int(u.nbytes) for u in new))))
                elif us[0].index not in seen:
                    seen.add(us[0].index)
                    later.append((us, minimal(us, b)))
        if rw is not None and rw > 0:  # the reclaim window's own (fresh) indices: declared once for their formats
            for u in list(rec.get("fresh") or ()) + (list(rec2.get("fresh") or ()) if rw2 is not None else []):
                if u.index not in seen:
                    seen.add(u.index)
                    later.append(([u], minimal([u], u.nbytes)))
        # HOSTED later windows (GLM53_PMK_WIN_HOST, TP8 stage 2: the kda items' ~22 CBs of window 2 at one page each cost
        # 68.7 KB over the member budget behind the projections' 724 KB window 1): a later-window allocation becomes extra
        # FORMAT descriptors of a window-1 allocation whose size every one of its pages divides (same address: an alias the
        # firmware sets up at launch and nobody touches before its window's entry re-lays it out), 0 declared bytes.
        # "auto" hosts only when the minimal declaration is over the budget (the gated layouts stay byte-identical)
        w1 = [[list(us), b] for us, b in per[0]]
        mode = os.environ.get("GLM53_PMK_WIN_HOST", "auto")
        tail = [([u], u.nbytes) for u in padcb], [([u], u.nbytes) for u in barcb]
        flat = sum(al(b) for _, b in shared + [(us, b) for us, b in w1] + tail[0] + later + tail[1])
        hosted = []
        if mode == "1" or (mode == "auto" and flat > self.cb_budget):
            rest = []
            for us, b in later:
                host = max(
                    (h for h in w1 if all(h[1] % int(u.page_bytes) == 0 for u in us)), key=lambda h: h[1], default=None
                )
                if host is None:
                    rest.append((us, b))
                else:
                    host[0].extend(us)
                    hosted += [u.index for u in us]
            later = rest
        decl = (shared + [(us, b) for us, b in w1] + tail[0] + later + tail[1])  # fmt: skip
        # KILL lists (A15 watched first light 04:01Z): at window W's entry the data-movement RISCs zero the interfaces of
        # every WINDOWED index not in W (window 1's in-place CBs, the later windows' one-page / hosted declarations):
        # their stale launch / earlier-window ranges overlap W's region, and the watcher's CB sanitizer judges a
        # transfer by the FIRST interface (lowest index) containing its start (sanitize.h debug_valid_cb_addr: a
        # correct write into a re-laid CB tripped 'overflows a circular buffer' on a stale FN 4). Diagnostic only:
        # no code of window W touches them, and a later window re-lays any of them out again
        windowed = {u.index for w in per for us, _ in per[w] for u in us}
        kill = {w: sorted(windowed - {t[0] for t in tabs[w]}) for w in tabs}
        if reclaim:  # the reclaim window: EVERY declared index but the barriers and its own is dead
            declared = {u.index for us, _ in decl for u in us}
            # every re-laid index needs a launch declaration: the TRISC's per-CB formats / tile dims are compile-time
            # tables of the declared CBs (an undeclared index compiles format Invalid: mbw2's CRM on CB_PROBE 31)
            for rw_ in reclaim:
                undeclared = {t[0] for t in tabs[rw_]} - declared
                assert not undeclared, (f"PMK windows on {role}: reclaim indices {sorted(undeclared)} have no launch "
                                        f"declaration (pmk_b_window fresh list)")  # fmt: skip
                kill[rw_] = sorted(declared - set(bar) - protect - {t[0] for t in tabs[rw_]})
        return dict(decl=decl, tabs=tabs, tab0=tab0, base=per[0][0][0][0].index, bar=bar, span=span, pad=max(pad, 0), need=need,
                    bytes=sum(al(b) for _, b in decl), windows=[sorted(w) for w in wins], hosted=hosted,
                    kill=kill, reclaim=reclaim)  # fmt: skip

    def _census(self, active: set | None = None) -> dict:
        out = {}
        for role in ("member", "root", "mm_only", "lane", "tail"):
            slots = self._slots(role, active)
            total = sum(b for _, b, _, _ in slots)
            wl = self._win_layout(role) if active is not None and active == self.active_phases() else None
            if wl is not None:  # the windowed declaration: window 1 in place + one page per later-window CB + barriers
                total = wl["bytes"]
            cb_end = CB_BASE + total
            out[role] = dict(cores=len(self.place.role_cores()[role]), cb_bytes=total, cb_end=cb_end,
                             budget=self.cb_budget, spare=self.cb_budget - total, fits=total <= self.cb_budget,
                             slots=slots)  # fmt: skip
        return out

    def _policy(self):
        g, c = self.g, self.census
        worst = max(c.values(), key=lambda r: r["cb_bytes"])
        # stationary: members64 if the member fits with its block(s); else sub-chunk the 2,048 rows (passes)
        passes = 1
        member = c["member"]
        if not member["fits"]:
            over = member["cb_bytes"] - member["budget"]
            block = g.ht * TILE_BYTES["bf16"]
            passes = min(self.policy["blocks_per_member"], max(2, -(-over // block) + 1))
        chunk_store = self.chunk_store_mode
        chunk_bytes = self.handoffs["H3"].payload_bytes
        per_core = -(-chunk_bytes // len(self.place.mm_grid))
        if chunk_store == "auto":
            spare = min(c["member"]["spare"], c["mm_only"]["spare"], c["root"]["spare"])
            chunk_store = "l1" if per_core <= spare else "dram"
        self.policy.update(
            stationary="members64 (one 32-row block per member)" if passes == 1 else f"members64 x {passes} passes (sub-chunked: {g.c_rows // passes} rows resident)",
            passes=passes, re_stream_bytes=(passes - 1) * sum(s_.geo(len(self.place.streamers))["w_bytes"] for s_ in self.streams[:3]),
            chunk_store=chunk_store, chunk_store_bytes=chunk_bytes, chunk_store_per_core=per_core,
            chunk_store_dram_us=round(2 * chunk_bytes / 480e3, 1) if chunk_store == "dram" else 0.0,
            worst_role=[k for k, v in c.items() if v is worst][0], worst_cb_end=worst["cb_end"],
            fits=all(v["fits"] for v in c.values()))  # fmt: skip
        for s in self.streams:
            s.passes = passes

    programs = None  # the layer's programs (list of phase-id sets, launched in order in one trace); None = one program

    def check_l1(self):
        """per PROGRAM of the layer (each its own static CB set) when the layer is split; else the one program"""
        if self.programs and self.scope is None:
            for sc in self.programs:
                self.scope = set(sc)
                try:
                    self._check_l1_one()
                finally:
                    self.scope = None
            return
        self._check_l1_one()

    def _check_l1_one(self):
        """fail loudly: every role's static CBs under the served budget (CB end <= ceiling) and the ruling.
        PMK_L1_CHECK=0 (host-only compile checks ONLY: a device refuses an over-budget program anyway) downgrades the
        failure to a warning so the composite kernels can still be JIT-checked on the mock."""
        # what the PROGRAM allocates: the registered phases' CBs (= the full census once every phase has a body, so
        # the finished layer is held to the full plan); an incremental unit's full plan over the ceiling is a note
        act = self.active_phases()
        census = self.census if act is None else self._census(act)
        if act is not None and not getattr(self, "_plan_noted", False):
            plan_over = {k: v["cb_end"] for k, v in self.census.items() if not v["fits"]}
            if plan_over:
                print(f"PMK L1 NOTE: the FULL plan is over the ceiling {plan_over}; this incremental program allocates "
                      f"only the registered phases' CBs ({sorted(act)})", flush=True)  # fmt: skip
                self._plan_noted = True
        if os.environ.get("PMK_L1_CHECK", "1") == "0":
            over = {k: v["cb_end"] for k, v in census.items() if not v["fits"]}
            if over:
                print(f"PMK L1 WARNING (PMK_L1_CHECK=0, host-only): over the ceiling: {over}", flush=True)
            return
        bad = {k: v for k, v in census.items() if not v["fits"]}
        if bad:
            lines = [
                f"{k}: {_kb(v['cb_bytes'])} of CBs, CB end {v['cb_end']:,} > ceiling {min(self.cb_end_ceiling, STAGE16_CB_END_RULING):,} "
                f"(budget {_kb(v['budget'])}, over by {_kb(-v['spare'])})"
                for k, v in bad.items()
            ]
            raise ValueError("PMK L1 census over the served worker_l1 contract:\n  " + "\n  ".join(lines))
        for k, v in census.items():
            if v["cb_end"] > STAGE16_CB_END_RULING:
                raise ValueError(f"PMK L1: {k} CB end {v['cb_end']:,} > Stage 16 ruling {STAGE16_CB_END_RULING:,}")

    # --------------------------------------------------------------------------------------------- reports
    def phase_table(self) -> str:
        rows = [
            "| # | phase | owner | cores | rows | CBs in -> out | waits -> signals | coll | wraps |",
            "|---|---|---|---|---:|---|---|---|---|",
        ]
        rc = self.place.role_cores()
        for p in self.phases:
            n = sum(len(rc[r]) for r in p.roles if r not in ("streamer", "mm"))
            cbs_in = sorted(
                {
                    u.index
                    for r in p.roles
                    for u in self.cbs.get(r, {}).values()
                    if p.pid in u.live
                    and any(w in u.note for w in ("store", "landing", "Y =", "prep", "rows", "ring", "PARK"))
                }
            )
            cbs_out = (
                sorted({u.index for r in p.roles for u in self.cbs.get(r, {}).values() if p.pid in u.live})
                if p.rows
                else []
            )
            rows.append(f"| {p.pid} | {p.name} | {p.owner} | {'+'.join(p.roles)} ({n}) | {p.rows:,} | "
                        f"{cbs_in} -> {cbs_out[:12]}{'…' if len(cbs_out) > 12 else ''} | {','.join(p.waits) or '-'} -> "
                        f"{','.join(p.signals) or '-'} | {p.collective or '-'} | {'; '.join(os.path.relpath(w, REL_KDIR) if w.startswith(REL_KDIR) else w for w in p.wraps)} |")  # fmt: skip
        return "\n".join(rows)

    def handoff_table(self) -> str:
        rows = [
            "| hand-off | producer -> consumer | payload / chip | lives in | counter | increments x unit | wait_min value (base + inc) |",
            "|---|---|---|---|---|---|---|",
        ]
        for h in self.handoffs.values():
            rows.append(f"| {h.hid} {h.name} | {h.producer} -> {h.consumer} | {h.payload_bytes / 2**20:,.2f} MiB | {h.lives_in} | "
                        f"{h.counter} | {h.expected} x {h.unit} | {h.wait_value} ({h.wait_base} + {h.expected}) |")  # fmt: skip
        return "\n".join(rows)

    def census_table(self) -> str:
        out = [f"CB region [{CB_BASE:,}, {min(self.cb_end_ceiling, STAGE16_CB_END_RULING):,}) = {_kb(self.cb_budget)} per core "
               f"(worker_l1 {self.worker_l1:,}; Stage 14 binding CB end {self.cb_end_ceiling:,}; Stage 16 ruling CB end <= {STAGE16_CB_END_RULING:,})"]  # fmt: skip
        for role, c in self.census.items():
            out.append(f"\n{role} ({c['cores']} cores): {_kb(c['cb_bytes'])} of static CBs, CB end {c['cb_end']:,}, "
                       f"spare {_kb(c['spare'])} -> {'FITS' if c['fits'] else 'OVER BUDGET'}")  # fmt: skip
            for idx, b, fp, note in c["slots"]:
                out.append(f"  CB {','.join(str(i) for i in idx):>6}  {_kb(b):>12}  {fp:<22} {note}")
        p = self.policy
        out.append(
            f"\npolicy: geometry = {p['geometry']}; residual = {p['residual']}; collective axes = {p['collective_axes']}"
        )
        out.append(
            f"policy: matmul compute cores = {p['mm_compute_cores']} (P3 bound ~{p['p3_us_bound']} us); {p['compute_note']}"
        )
        out.append(
            "policy: mhc state live P1 -> P10 -> P16 (X 2, PRE 7, PCOEF 23, COEF 24, CROW 25; root planes 14..19); FN 4 / W 5 "
            "reloaded at P10 / P16 (their allocations are the XS / PART aliases). Member levers if a body grows: drop post Y2 22 "
            "(8 KB, unused with one partial), mhc PART 6 (8 KB) into PRE's page, GATHER 0 (64 KB) only on the roots if the "
            "member gather goes through the root"
        )
        out.append(f"policy: ring trims = {p.get('ring_trims') or 'none'}")
        out.append(f"policy: stationary = {p['stationary']}; passes = {p['passes']} (re-streamed weight bytes "
                   f"{p['re_stream_bytes'] / 2**20:,.1f} MiB); chunk store = {p['chunk_store']} ({p['chunk_store_bytes'] / 2**20:,.1f} MiB, "
                   f"{_kb(p['chunk_store_per_core'])} per MM core; DRAM round trip ~{p['chunk_store_dram_us']} us under P5 / P6); "
                   f"worst core = {p['worst_role']} (CB end {p['worst_cb_end']:,}); fits = {p['fits']}")  # fmt: skip
        out.append("rings (streamer BRISC, carried across phases / layers, never rewound): " + "; ".join(
            f"CB {CB_RING0 + i}: {k[0]} x {k[1]} tiles = {_kb(r['block_bytes'])} x {r['slots']} ({', '.join(r['phases'])})"
            for i, (k, r) in enumerate(self.rings.items())))  # fmt: skip
        out.append("semaphores: " + ", ".join(f"{i}:{'/'.join(sorted(r))}" for i, r in sorted(self.sems.items()))
                   + f" (NUM_SEMAPHORES = {NUM_SEMAPHORES}: ids >= 16 are impossible; the contract's new counters take the free ids "
                   f"the map is Stage 16's 09:37Z one: FS_ARRIVE={SEM_FS_ARRIVE} FS_REDUCED={SEM_FS_REDUCED} XS_FREE={SEM_XS_FREE} (reserved on members) IN0_FREE={SEM_IN0_FREE} (H7b, members); "
                   f"producer -> lane readiness = FabricStream words LREADY({FS_W_LREADY}) + k, XSADDR {FS_W_XSADDR} / SEMADDR {FS_W_SEMADDR} / "
                   f"OUTADDR {FS_W_OUTADDR}, XSFREE {FS_W_XSFREE} (words from {os.path.basename(FS_WORDS['_source'])}: {FS_WORDS['_source']}); lanes dedicated)")  # fmt: skip
        return "\n".join(out)

    # ------------------------------------------------------------------------------------------ plug-in
    def register_phase(self, pid: str, body: PhaseBody):
        """A piece plugs its real kernels into phase `pid`: the body's CB uses re-run the census (fail loudly), its
        sources replace the stubs in the composite kernels."""
        p = next(q for q in self.phases if q.pid == pid)
        for role, uses in body.cbs.items():
            for u in uses:
                self._use(role, u.index, u.fmt, u.pages, u.live, u.page, f"{pid}: {u.note}")
        body.status = "body"
        p.body = body
        self._liveness_check()
        self.census = self._census()
        self._policy()
        self.check_l1()
        return p

    # ---------------------------------------------------------------------------------------- descriptor
    # the sub-roles each composite hosts: the KDA scan cores are the 32 MM-only cores, the streamers' BRISC sits on
    # MM-only / root cores (kernels/mk/reader_w.cpp lineage), the lanes and the tail are their own cores
    HOSTS = {"member": ("member",), "root": ("root", "streamer"), "mm_only": ("mm_only", "kda_scan", "streamer"),
             "lane": ("lane",), "tail": ("tail",)}  # fmt: skip

    def _hosts(self, role: str, risc: str, p: "Phase") -> bool:
        """does the (role, risc) composite run phase p? streamer phases only on the BRISC (dm0) of their hosts"""
        for r in p.roles:
            if r == "streamer":
                if r in self.HOSTS[role] and risc == "dm0":
                    return True
            elif r in self.HOSTS[role]:
                return True
        return False

    def wait_roles(self, p) -> set:
        """(Stage 14's engine fix d614a11a21, ported) the roles that wait on phase p's hand-offs: a weight-streaming
        phase's in0 ROW cores only (its rows_of: the members in members64; the streamers never wait for activations,
        contract §4c), else every role of the phase ("streamer" = the root / MM-only BRISC)"""
        if p.stream is not None and getattr(p.stream, "rows_of", None):
            rc = self.place.role_cores()
            rows = {tuple(c) for c in p.stream.rows_of}
            return {r for r in ("member", "root", "mm_only") if rows & {tuple(c) for c in rc[r]}}
        # COMPOSITE roles: a sub-role (kda_scan) waits in the composite that hosts it (mm_only) -- the raw phase roles
        # dropped P6 / P9's H3b / H4 waits on the MM-only cores (test_pmk_waits, 11:5xZ)
        hosted = {R for R, hs in self.HOSTS.items() for r in p.roles if r != "streamer" and r in hs}
        return hosted | ({"root", "mm_only"} if "streamer" in p.roles else set())

    def consumer_roles(self, hid: str) -> set:
        """(14's d614a11a21, the counter side) the roles whose composite waits on `hid` = wait_roles of every phase
        that waits on it: a producer raises the counter on exactly these roles' cores (never on a streamer BRISC that
        does not wait for activations)"""
        rs = set()
        for p in self.phases:
            if hid in p.waits:
                rs |= self.wait_roles(p)
        return rs

    def consumer_cores(self, hid: str) -> list:
        """the (x, y) cores a producer of `hid` raises: the consumer roles' cores, or the streams' rows_of where the
        waiting phase is a weight-streaming phase (its in0 row cores)"""
        rc, out = self.place.role_cores(), []
        for p in self.phases:
            if hid not in p.waits:
                continue
            if p.stream is not None and getattr(p.stream, "rows_of", None):
                cs = [tuple(c) for c in p.stream.rows_of]
            else:
                cs = [tuple(c) for r in sorted(self.wait_roles(p)) for c in rc[r]]
            out += [c for c in cs if c not in out]
        return out

    def _has_live_producer(self, hid: str) -> bool:
        """a hand-off is waited on only if a REGISTERED body produces it (a stub raises nothing). The lane / fabric
        producers (collective phases) count as live once their phase has a body."""
        prods = [p for p in self.phases if hid in p.signals]
        if not prods:
            return False
        # the FIRST producer raises the hand-off's counter (later 'producers' add another part on other counters:
        # H8 = P11's AG slices on FS_ARRIVE + P12's routes on LOGITS / TAIL; waiting FS_ARRIVE without P11 registered
        # and in this program deadlocks). A producer in ANOTHER program of the layer finished before this program
        # started: its data is in DRAM (the cross-program hand-off contract), nothing to wait for here.
        first = prods[0]
        return first.body.status == "body" and self.in_scope(first.pid)

    def _rt_width(self, p: "Phase", role: str, risc: str) -> int:
        """runtime args of phase p on the (role, risc) composite: the max over the role's cores (every core is padded
        to it, so ph_rt0 is the same on every core of the composite)"""
        if p.body.status != "body":
            return 0
        return sum(self._sub_width(sub, role) for sub in p.body.subparts(role, risc))

    def _alias_active(self, sub: dict) -> bool:
        """RT ALIAS (W7, 2026-10-25): a part that reads ANOTHER part's runtime args of the same composite (rt_alias = that
        part's key 'P<id>[_<name>]', e.g. P11's C1 send = P2's: identical args, the same connection) -- active only when
        the aliased phase is in this program; then the part has no args of its own and its ph_rt0 is the alias's"""
        al = sub.get("rt_alias")
        return bool(al) and self.in_scope(str(al).split("_")[0])

    def _sub_width(self, sub: dict, role: str) -> int:
        if self._alias_active(sub):
            return 0
        cores = self.place.role_cores()[role]
        return max((len(rt.get(c, [])) for rt in sub["rts"] for c in cores), default=0)

    def _ns(self, pid: str, sub: dict) -> str:
        return f"ph{pid[1:]}" + (f"_{sub['name']}" if sub.get("name") else "")

    def composite_source(self, role: str, risc: str, arg_table: bool = False, part_fid: dict | None = None) -> str:
        """The composite kernel text of a core role for one RISC: every phase the role runs, its body included in
        its own namespace with its own argument bases (kernels/mk/owner_*.cpp's pattern), then the main loop:
        wait (wait_min on the hand-off counters) -> phase_main -> signal. Stubs where no body is registered."""
        rc = self.place.role_cores()
        mine = [p for p in self.phases if self._hosts(role, risc, p) and self.in_scope(p.pid)]
        mine = [p for p in mine if p.rows or p.pid in ("P9",)]
        probe = PHASE_PROBE() if risc != "c" else None
        labels = ["start"]
        hdr = [
            "// generated by tt/pmk_layer.py: PMK composite kernel, role " + role + ", " + risc,
            "#define MK_PHASED 1",
            '#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/opt_size.hpp"',
            "#include <cstdint>",
        ] + (['#include "api/debug/kernel_profiler.hpp"', f'#define PMK_ZS "{ZSALT()}"']
             + (["#define PMK_ITEM_ZONES 1"] if risc == "c" and os.environ.get("GLM53_PMK_ITEM_ZONES", "0") == "1" else [])
             if ZONES(risc) else [])  # fmt: skip
        if risc == "c":
            hdr += [
                '#include "api/compute/common.h"',
                '#include "api/compute/compute_kernel_api.h"',
                '#include "api/compute/cb_api.h"',
                '#include "api/compute/tile_move_copy.h"',
            ]
        else:
            hdr += ['#include "api/dataflow/dataflow_api.h"']
        hdr += [
            '#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"',
            '#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"',
            f'#include "{os.path.join(KDIR, "pmk", "pmk_common.hpp")}"',
        ]
        part_fid = dict(part_fid or {})
        if part_fid:  # the part regime: a part at its own fidelity (merge_compute -> PMK_PART_FID)
            hdr.append(f'#include "{os.path.join(KDIR, "pmk", "pmk_fid.hpp")}"')
        wl, wins = self._win_layout(role), self.windows(role)
        if wl is not None:  # PHASE WINDOWS: the barrier / re-layout helpers + this program's window tables
            hdr += [f"#define PMK_WIN_BASE {wl['base']}", f"#define PMK_WIN_CB_D0 {wl['bar'][0]}",
                    f"#define PMK_WIN_CB_D1 {wl['bar'][1]}", f"#define PMK_WIN_CB_T {wl['bar'][2]}",
                    f'#include "{os.path.join(KDIR, "pmk", "pmk_window.hpp")}"']  # fmt: skip
            for W, tab in sorted(wl["tabs"].items()):
                vals = ", ".join(str(v) for e_ in tab for v in e_) or "0, 0, 0, 0"
                hdr.append(
                    f"constexpr uint32_t pmk_win_tab{W}[] = {{{vals}}};  // window {W + 1}: (cb, offset, bytes, page)"
                )
                kv = ", ".join(str(v) for v in wl["kill"][W]) or "0"
                hdr.append(
                    f"constexpr uint32_t pmk_win_kill{W}[] = {{{kv}}};  // window {W + 1}: stale windowed CBs zeroed"
                )
        # hoist every registered body source's #include lines to the top level: a header first included INSIDE a
        # phase namespace would put its symbols (and the API's) in that namespace, and with #pragma once the next
        # part's namespace would not see them. Hoisted, every in-namespace include is a no-op.
        hoisted = []
        for p in mine:
            if p.body.status != "body":
                continue
            for sub in p.body.subparts(role, risc):
                src_ = sub["src"]
                if not (src_ and os.path.exists(src_)):
                    continue
                if True:
                    for ln in open(src_).read().splitlines():
                        t = ln.strip()
                        if not t.startswith("#include"):
                            continue
                        inc = t.split(None, 1)[1].strip() if " " in t else ""
                        path = None
                        if inc.startswith('"') and inc.endswith('"'):
                            name_ = inc[1:-1]
                            near = os.path.join(os.path.dirname(os.path.abspath(src_)), name_)
                            root = os.path.join(REPO_ROOT, name_)
                            path = near if os.path.exists(near) else root if os.path.exists(root) else None
                            if (
                                path == near and "/" not in name_
                            ):  # a header next to the body: absolute path at top level
                                t = f'#include "{near}"'
                        if path is not None:  # a model header: hoist only if include-GUARDED (pfs_common.hpp is
                            txt = open(
                                path
                            ).read()  # deliberately unguarded: one copy per phase namespace, PH_CT0-based)
                            if "#pragma once" not in txt and not re.search(r"#ifndef\s+\w+\s*\n\s*#define\s+\w+", txt):
                                continue
                        if t not in hdr and t not in hoisted:
                            hoisted.append(t)
        hdr = hdr + hoisted
        body, main, ct_base, rt_base = [], [], 0, 0
        cur_w = 0
        rt_bases = {}  # part key -> its ph_rt0 (RT ALIAS)

        def enter_windows(upto):
            for W in range(cur_w + 1, upto + 1):
                rb = (wl.get("reclaim") or {}).get(W)
                if rb is not None:  # the reclaim window: from the CB base, counters balanced (pmk_window.hpp)
                    for hid in self._drain_hids(role, self.win_reclaim[role]["pid"]) if risc != "c" else ():
                        h = self.handoffs[hid]
                        assert isinstance(h.counter, int), ("RECLAIM DRAIN: an integer counter", hid, h.counter)
                        main.append(f"    pmk::wait_min({h.counter}, {self._wait_value(h)});  // RECLAIM DRAIN {hid} {h.name} "
                                    f"(base {h.wait_base} + {h.expected}): its remote writes land in CBs the reclaim lays out over")  # fmt: skip
                    main.append(f"    pmk_win::enter_reclaim<{W}>(pmk_win_tab{W}, {len(wl['tabs'][W])}, pmk_win_kill{W}, "
                                f"{len(wl['kill'][W])}, pmk_win_b - {rb});  // RECLAIM WINDOW {W + 1} {sorted(wins[W])}: "
                                f"core barrier + counters balanced + re-layout from the CB base")  # fmt: skip
                    continue
                ent = f"pmk_win::enter<{W}>(pmk_win_tab{W}, {len(wl['tabs'][W])}, pmk_win_kill{W}, {len(wl['kill'][W])}, pmk_win_b);"
                main.append((zone_call(risc, f"PMK_WIN{W + 1}{ZSALT()}", ent) if ZONES(risc) else f"    {ent}")
                            + f"  // PHASE WINDOW {W + 1} {sorted(wins[W])}: core barrier + CB re-layout")  # fmt: skip

        for p in mine:
            if wl is not None:
                w_ = self._win_index(p.pid, wins)
                if w_ is not None and w_ > cur_w:
                    enter_windows(w_)
                    cur_w = w_
            subs = (
                p.body.subparts(role, risc)
                if p.body.status == "body"
                else [dict(src=None, ct=[], defines=[], rts=[{}])]
            )
            calls = []
            for sub in subs:
                src = sub["src"]
                if src is None:
                    src = os.path.join(KDIR, "pmk", "pmk_stub_c.cpp" if risc == "c" else "pmk_stub_dm.cpp")
                ns = self._ns(p.pid, sub)
                pdefs = [(str(k), str(v)) for k, v in sub["defines"]]
                key_ = p.pid + (f"_{sub['name']}" if sub.get("name") else "")
                if key_ in part_fid:
                    pdefs.append(("PMK_PART_FID", str(part_fid[key_])))
                key_ = p.pid + (f"_{sub['name']}" if sub.get("name") else "")
                my_rt0 = rt_bases.get(str(sub.get("rt_alias")), rt_base) if self._alias_active(sub) else rt_base
                rt_bases[key_] = my_rt0
                body += (
                    # ph_rt0 constexpr (a phase's base is fixed at composition; the kda / cache bodies build constexpr arg
                    # offsets from it: 'ph_rt0 + 13' -- a mutable global broke kda_scan_w.cpp, mock 13:02Z)
                    [
                        f"namespace {ns} {{",
                        f"constexpr uint32_t PH_CT0 = {ct_base};",
                        f"constexpr uint32_t ph_rt0 = {my_rt0};",
                    ]
                    # the body's defines live only around its own #include: push / undef / define, then undef / pop, so a
                    # composite-wide -D (PMK_REV) or a hoisted header's default (PMK_MHC_PART) is restored afterwards
                    + [ln for k, v in pdefs for ln in (f'#pragma push_macro("{k}")', f"#undef {k}", f"#define {k} {v}")]
                    + ["#define kernel_main phase_main", f'#include "{src}"', "#undef kernel_main"]
                    + [ln for k, _ in reversed(pdefs) for ln in (f"#undef {k}", f'#pragma pop_macro("{k}")')]
                    + [f"}}  // namespace {ns}"]
                )
                ct_base += len(sub["ct"])
                rt_base += self._sub_width(sub, role) if p.body.status == "body" else 0
                zn = "PMK_" + p.pid + (f"_{sub['name']}" if sub.get("name") else "") + ZSALT()
                calls.append(
                    (zone_call(risc, zn, f"{ns}::phase_main();") if ZONES(risc) else f"    {ns}::phase_main();")
                    + f"  // {p.pid} {p.name} [{p.body.status}]"
                    + (f" part {sub['name']}" if sub.get("name") else "")
                )
            if risc != "c" and p.body.status == "body" and role in self.wait_roles(p):
                # waits only around registered bodies (stubs raise nothing), and only on the roles that consume
                for hid in p.waits:
                    h = self.handoffs[hid]
                    if not self._has_live_producer(hid):
                        # incremental integration: every producer of this hand-off is still a stub -> nothing raises
                        # its counter; a wait would hang. The body reads its input from where the unit staged it.
                        main.append(f"    // {hid} {h.name}: no registered producer yet -> no wait")
                        continue
                    sid = (
                        h.counter
                        if isinstance(h.counter, int)
                        else (h.counter[0] if isinstance(h.counter, tuple) and isinstance(h.counter[0], int) else None)
                    )
                    if sid is not None and role != "lane":
                        main.append(
                            f"    pmk::wait_min({sid}, {self._wait_value(h)});  // {hid} {h.name} (layer base {h.wait_base} + {h.expected})"
                        )
            main += calls
            if probe:
                for sub in subs:  # the marks the phase's bodies emit themselves (PMK_PP_MARK), in call order
                    labels.extend(str(m_) for m_ in sub.get("probe_marks", ()))
                main.append(f"    pmk_pp::mark();  // GLM53_PMK_PHASE_PROBE: after {p.pid}")
                labels.append(p.pid)
            if risc != "c":
                for hid in p.signals:
                    h = self.handoffs[hid]
                    credit_risc = "dm1"
                    if hid == "H0f":
                        hid = "H0"  # the same raise code (the member side + the lane word), second use
                        credit_risc = self._h0f_risc()  # dm0 after P10's resident-X store (the TP8 x1 race)
                    if hid == "H7b" and role == "member" and risc == "dm0" and p.body.status == "body":
                        # IN0 free (H7b): the body's TRISC has popped its last IN0 page (the body orders that, e.g. by
                        # draining its OUT CB first); this core's own counter +1 -> the NCRISC's P10 prefetch may proceed
                        main.append(f"    mk::signal_self({SEM_IN0_FREE});  // H7b IN0 40 released (cumulative)")
                    elif hid == "H0" and role == "member" and risc == credit_risc:
                        # XS-FREE credit (H0): after this member's FN use, +1 on its block's local writers + the lane
                        # (their NoC (x, y) are the tail runtime args at PMK_XSFREE_ARG: descriptor() appends them)
                        main.append("#if defined(PMK_XSFREE_ARG)")
                        main.append(
                            "    noc_async_read_barrier();  // this member's FN reads are done before the credit"
                        )
                        # count-prefixed (11:3xZ): [n, x0, y0, ..] -- a holder of a REMOTE block has n = 0 local
                        # writers; the old fixed PMK_XSFREE_N loop sent its zero padding to NoC (0, 0)
                        main.append(
                            f"    mk::signal_handoff({SEM_XS_FREE}, PMK_XSFREE_ARG + 1, "
                            "get_arg_val<uint32_t>(PMK_XSFREE_ARG));  // H0 XS-FREE credit"
                        )
                        main.append("#endif")
                        main.append("#if defined(PMK_XSFREE_LANE)  // the lane side: +1 on lane (G % 2)'s XSFREE word")
                        main.append(
                            "    noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(PMK_XSFREE_LANE), "
                            "get_arg_val<uint32_t>(PMK_XSFREE_LANE + 1), get_arg_val<uint32_t>(PMK_XSFREE_LANE + 2)), 1);"
                        )
                        main.append("    noc_async_atomic_barrier();")
                        main.append("#endif")
                    elif isinstance(h.counter, int):
                        main.append(
                            f"    // {hid} {h.name}: the body raises counter {h.counter} on its consumers per {h.unit}"
                        )
        if (
            wl is not None and cur_w < len(wins) - 1
        ):  # the windows this composite has no phase in: its barrier counts still
            enter_windows(len(wins) - 1)  # go up (the role's other composites wait for them)
        pro = (
            ["    pmk::load_arg_table();  // GLM53_PMK_ARGS=table: this core's row -> rta_l1_base"]
            if arg_table and risc != "c"
            else []
        )
        if wl is not None:  # the window region's base, read once before any re-layout / kill (pmk_window.hpp relayout)
            pro = pro + [
                "    const uint32_t pmk_win_b = pmk_win::base_bytes();  // window 1's region base (launch layout)"
            ]
        pp_hdr, pp_end = [], []
        if probe:
            ri = 0 if risc == "dm0" else 1
            assert len(labels) <= 32, (role, risc, len(labels), labels)
            self.probe_labels[(role, risc)] = list(labels)
            pp_hdr = ["namespace pmk_pp {  // GLM53_PMK_PHASE_PROBE (tt/pmk_layer.PHASE_PROBE)",
                      "inline uint32_t g_m[32];", "inline uint32_t g_n = 0;",
                      "FORCE_INLINE void mark() {", "    if (g_n < 32) {", "        g_m[g_n++] = static_cast<uint32_t>(get_timestamp());",
                      "    }", "}", "}  // namespace pmk_pp", "#undef PMK_PP_MARK", "#define PMK_PP_MARK() pmk_pp::mark()"]  # fmt: skip
            pro = ["    pmk_pp::mark();  // GLM53_PMK_PHASE_PROBE: start"] + pro
            pp_end = ["    {  // GLM53_PMK_PHASE_PROBE: the marks -> this RISC's row (128 B staged in the L1 scratch)",
                      "        noc_async_full_barrier();",
                      f"        volatile tt_l1_ptr uint32_t* s_ = reinterpret_cast<volatile tt_l1_ptr uint32_t*>({probe[1]}u + {128 * ri}u);",
                      "        for (uint32_t i_ = 0; i_ < 32; ++i_) {", "            s_[i_] = i_ < pmk_pp::g_n ? pmk_pp::g_m[i_] : 0u;", "        }",
                      f"        const uint32_t slot_ = ((my_y[0] & 31u) * 32u + (my_x[0] & 31u)) * 2u + {ri}u;",
                      f"        const InterleavedAddrGen<true> pg_{{.bank_base_address = {probe[0]}u, .page_size = 128}};",
                      f"        noc_async_write({probe[1]}u + {128 * ri}u, pg_.get_noc_addr(slot_), 128);",
                      "        noc_async_write_barrier();", "    }"]  # fmt: skip
        tail = ["void kernel_main() {", "    PMK_LAYER_BEGIN();"] + pro + main + pp_end + ["    PMK_LAYER_END();", "}"]
        # the probe header goes BEFORE the bodies: they call PMK_PP_MARK() (program B's plan / sweep / combine marks)
        return "\n".join(hdr + [""] + pp_hdr + [""] + body + [""] + tail) + "\n"

    def xs_writers(self, rank: int) -> dict:
        """member (x, y) -> the local cores that write its XS block and get its XS-FREE credit: the holder of global
        block G = member G; a LOCAL block (sp_blocks * rank <= G < + sp_blocks) is written by the 8 members of group
        (G - sp_blocks * rank) mod 8 (group b handles SP blocks b, b + 8, ..: 8 at TP8, 32 at TP2); every holder also
        credits the first lane (it forwards one per-chip credit to the peers)"""
        P, g = self.place, self.g
        lo = g.sp_blocks * rank
        out = {}
        for G, m in enumerate(P.members):
            writers = list(P.groups[(G - lo) % len(P.groups)]) if lo <= G < lo + g.sp_blocks else []  # group b' mod 8
            out[m] = writers
        return out

    def xs_credit_lane(self, G: int) -> tuple:
        """the lane whose XSFREE word the holder of global block G credits: lane G % 2 (lane h writes the peers' blocks
        with b % 2 == h, which land on holders with G % 2 == h since blocks per chip is even)"""
        return self.place.lanes[G % FS_LANES]

    def member_map(self, rank: int) -> dict:
        """Export for 7a (FabricStreamC1 / the RS consumer): member (x, y) -> dict(G = the global block it holds (XS 35),
        local = G is one of this chip's blocks, lane = G % 2, k = G's index in that lane's local list (None if remote),
        group b, slice j, col_tiles = [16 j, 16 j + 16) of every stream (the member's mHC columns), owner_blocks = the
        LOCAL SP blocks whose reduced y this member receives in POST 21 (group b handles SP blocks b, b + 8, ..: 1 at
        TP8, 4 at TP2 = 64 members vs 32 owned blocks per chip), one 64 KB slice per block in k order."""
        P, g = self.place, self.g
        lo = g.sp_blocks * rank
        cpc = g.ht // 8
        out = {}
        for G, m in enumerate(P.members):
            local = lo <= G < lo + g.sp_blocks
            h = G % FS_LANES
            lane_list = [b for b in range(g.sp_blocks) if b % FS_LANES == h]
            b, j = divmod(G, 8) if False else (G // 8, G % 8)  # member (b, j): group b = G // 8, slice j = G % 8
            out[m] = dict(G=G, local=local, lane=h, k=lane_list.index(G - lo) if local else None, group=b, slice=j,
                          col_tiles=(cpc * j, cpc * (j + 1)), owner_blocks=[sb for sb in range(g.sp_blocks) if sb % 8 == b],
                          credit_lane=self.xs_credit_lane(G))  # fmt: skip
        return out

    def post_layout(self) -> dict:
        """Export for 7a's RS consumer: POST 21 (H6 / H10) per member = the reduced y of ONE owner block restricted to
        the member's column slice: 32 rows x 512 cols fp32 = 16 fp32 tiles (4,096 B each) = 64 KB, tile order = the
        block's column tiles 16 j .. 16 j + 15 (row-major tile grid, one row tile); one block at a time (TP2: the
        member's 4 owner blocks arrive in k order and are consumed block by block)."""
        cpc = self.g.ht // 8
        return dict(cb=CB_POST_Y, tiles=cpc, tile_bytes=TILE_BYTES["fp32"], bytes=cpc * TILE_BYTES["fp32"], dtype="fp32",
                    rows=TILE, cols=cpc * TILE, order="column tiles 16 j .. 16 j + 15 of the owner block's [32, 4096] y",
                    blocks_per_member=self.g.sp_blocks // 8, reduced_counter=SEM_FS_REDUCED)  # fmt: skip

    def _arg_table(self, device, rc) -> dict:
        """GLM53_PMK_ARGS=table: this program's (scope's) DRAM argument table, allocated at its first build -- rows =
        every core of every DM composite of ARG_TABLE_ROLES (dm0 then dm1, the role's core order), one ARG_PAGE page
        each, sharded per chip (each rank's rows on its own chip: the address is the same everywhere)"""
        import torch

        import ttnn

        key = tuple(sorted(self.scope, key=lambda q: int(q[1:]))) if self.scope else ("all",)
        tables = self.__dict__.setdefault("arg_tables", {})
        if key in tables:
            return tables[key]
        base, n = {}, 0
        for role in ARG_TABLE_ROLES:
            for risc in ("dm0", "dm1"):
                base[(role, risc)] = n
                n += len(rc.get(role) or [])
        n_dev = int(device.get_num_devices()) if hasattr(device, "get_num_devices") else 1
        t = ttnn.from_torch(torch.zeros(n_dev * n, self.arg_page // 4, dtype=torch.int32), dtype=ttnn.uint32,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0) if n_dev > 1 else None)  # fmt: skip
        tables[key] = dict(
            key=key, t=t, addr=int(t.buffer_address()), base=base, nrows=n, n_dev=n_dev, rows={}, host=None
        )
        return tables[key]

    def _arg_table_commit(self, tb: dict, rank: int):
        """after rank `rank`'s build: the FIRST build of the program writes every rank's rows once (when the last rank
        is built -- the compile launch, before any capture: no host write inside a capture); every later build (the
        captured forward) must produce the same rows (the args are constant per program: else the table is stale)"""
        import torch

        import ttnn

        if tb["host"] is not None:
            got = tb["rows"].get(rank, {})
            if got != tb["host"][rank]:
                diff = sorted(k for k in set(got) | set(tb["host"][rank]) if got.get(k) != tb["host"][rank].get(k))
                raise ValueError(f"PMK_ARGS=table {tb['key']}: rank {rank} rows changed after the table fill "
                                 f"(rows {diff[:6]}): the args are not constant for this program")  # fmt: skip
            return
        if rank != tb["n_dev"] - 1:
            return
        w = self.arg_page // 4
        full = torch.zeros(tb["n_dev"] * tb["nrows"], w, dtype=torch.int64)
        for r_, rows in tb["rows"].items():
            for i_, v in rows.items():
                full[r_ * tb["nrows"] + i_, : len(v)] = torch.tensor(v, dtype=torch.int64)
        host = ttnn.from_torch(full.to(torch.int32) if int(full.max()) < 2**31 else (full - (full >= 2**31).long() * 2**32).to(torch.int32),
                               dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                               mesh_mapper=ttnn.ShardTensorToMesh(tb["t"].device(), dim=0) if tb["n_dev"] > 1 else None)  # fmt: skip
        ttnn.copy_host_to_device_tensor(host, tb["t"])
        tb["host"] = {r_: dict(rows) for r_, rows in tb["rows"].items()}
        tb["rows"] = {}
        self.arg_table_report = dict(tables=len(self.arg_tables), rows_used=sum(len(v) for v in tb["host"].values()),
                                     bytes_per_chip=tb["nrows"] * self.arg_page)  # fmt: skip

    def descriptor(self, device, io: list, *, probe=None, rank: int = 0, fs_sync_addr: int = 0):
        """ttnn.ProgramDescriptor of the layer program on `device` for mesh rank `rank` (row-major index: the XS-FREE
        writer lists depend on which blocks are local): composite kernels per role and RISC, the CBs of the census
        (aliases as multi-format CBDescriptors), the semaphores. `io` = the tensors the registered bodies touch (the
        stubs touch none; the job passes a few small ones so generic_op has io). fs_sync_addr = the FabricStream
        lanes' sync tensor address (7a's FabricStreamC1.sync): the members' XSFREE credits go to that lane word."""
        import ttnn

        if not self.planned:
            self.plan()
        use_table = os.environ.get("GLM53_PMK_ARGS", "trace") == "table"
        self._use_table = use_table
        # the arg rows' L1 (2 slots dm0 / dm1 + the 64-B alignment slack, live in every phase) exists ONLY in table
        # builds: the plan's own CB 30 use is kept aside and restored for a trace build, so a process that builds
        # both (the A/B unit) gives the trace build exactly its original CBs (pair 16:53Z: the table use leaked into
        # the next trace build -> a program never compiled -> 'Cannot load new binaries during trace capture')
        if not hasattr(self, "_arg_cb_orig"):
            self._arg_cb_orig = {r_: self.cbs.get(r_, {}).get(CB_ARG) for r_ in ARG_TABLE_ROLES}
        for role_ in ARG_TABLE_ROLES:
            d_ = self.cbs.setdefault(role_, {})
            if self._arg_cb_orig[role_] is None:
                d_.pop(CB_ARG, None)
            else:
                d_[CB_ARG] = self._arg_cb_orig[role_]
        if use_table:
            all_p = tuple(p.pid for p in self.phases)
            for role_ in ARG_TABLE_ROLES:
                o_ = self.cbs.get(role_, {}).get(CB_ARG)
                if o_ is not None and o_.page_bytes != self.arg_page:  # a wider row (M-B: 2 KB) replaces the plan's
                    self.cbs[role_].pop(CB_ARG)  # 1 KB use (the stack programs' per-layer args: not in a fused layer)
                if self.place.role_cores().get(role_):
                    self._use(role_, CB_ARG, "u32", 3, all_p, page=self.arg_page, users=all_p, solo=True,
                              note="GLM53_PMK_ARGS=table: arg rows (dm0 / dm1 slots)")  # fmt: skip
        self.check_l1()
        fmt = {
            "bf16": ttnn.bfloat16,
            "bf8": ttnn.bfloat8_b,
            "bf4": ttnn.bfloat4_b,
            "fp32": ttnn.float32,
            "u32": ttnn.uint32,
            "u16": ttnn.uint16,
        }
        rc = self.place.role_cores()
        if os.environ.get("GLM53_PMK_RECT_MERGE", "0") == "1":  # trace-light lever (1): one CoreRange per RECTANGLE
            crs = lambda cores: ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for x0, y0, x1, y1 in core_rects(cores)]
            )  # noqa: E731
        else:
            crs = lambda cores: ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores]
            )  # noqa: E731
        kernels, cbs, sems, audit, kidx = [], [], [], [], {}
        self.layouts = (
            {}
        )  # (role, risc) -> [(pid[_part], PH_CT0, ph_rt0, part)] of THIS program (post-build hooks: fab_at)
        tb = self._arg_table(device, rc) if use_table else None
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)  # noqa: E731
        cfgs = dict(dm0=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default),
                    dm1=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default),
                    c=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                                                   fp32_dest_acc_en=True))  # fmt: skip
        for role in ("member", "root", "mm_only", "lane", "tail"):
            cores = rc[role]
            if not cores:
                continue
            for risc in ("dm0", "dm1", "c"):
                rt_d = {c_: [] for c_ in cores}  # plain dict first (ttnn.RuntimeArgs raises on an unset core)
                ct, computes, layout, tails, flex = [], [], [], {}, {}
                mine = [p for p in self.phases if self._hosts(role, risc, p) and (p.rows or p.pid in ("P9",))
                        and self.in_scope(p.pid)]  # fmt: skip
                for p in mine:
                    if p.body.status != "body":
                        continue
                    for sub in p.body.subparts(role, risc, rank):
                        key = p.pid + (f"_{sub['name']}" if sub.get("name") else "")
                        if self._alias_active(sub):  # RT ALIAS: the aliased part's ph_rt0, no args of its own
                            rt0_ = next((e_[2] for e_ in layout if e_[0] == str(sub.get("rt_alias"))), None)
                            assert (
                                rt0_ is not None
                            ), f"PMK {role}/{risc} {key}: rt_alias {sub.get('rt_alias')} not earlier in the composite"
                            layout.append((key, len(ct), rt0_, sub))
                            ct += sub["ct"]
                            continue
                        layout.append((key, len(ct), len(rt_d[cores[0]]), sub))  # (pid[_part], PH_CT0, ph_rt0, part)
                        ct += sub["ct"]
                        width = self._sub_width(sub, role)
                        for c_ in cores:  # every core padded to the part's width: ph_rt0 equal on all cores / ranks
                            args = [int(a) for a in sub["rt"].get(c_, [])]
                            rt_d[c_] += args + [0] * (width - len(args))
                        if risc == "c" and sub["compute"] is not None:
                            computes.append((key, sub["compute"]))
                            if sub.get("regime_flex") or sub.get("dst_any") or sub.get("fid_mode"):
                                flex[key] = dict(dst_any=bool(sub.get("regime_flex") or sub.get("dst_any")),
                                                 fid_mode=("template" if sub.get("regime_flex") else sub.get("fid_mode")))  # fmt: skip
                defines = [("PMK_ROLE_" + role.upper(), "1"), ("PMK_REV", "2")]
                # kernel-level defines of the registered parts (a hoisted, include-guarded header reads them BEFORE any
                # phase namespace: e.g. the kda CB index maps PK_ITEM_MAP / PK_SCAN_MAP); one value per name
                kd = {}
                for p in mine:
                    if p.body.status != "body":
                        continue
                    for sub in p.body.subparts(role, risc, rank):
                        for k, v in sub.get("kdefines") or []:
                            if kd.get(k, v) != v:
                                raise ValueError(f"PMK {role}/{risc}: kernel define {k} = {kd[k]} vs {v} ({p.pid})")
                            kd[k] = v
                defines += sorted((str(k), str(v)) for k, v in kd.items())
                if role == "member" and (risc == "dm1" or (risc == "dm0" and self._h0f_risc() == "dm0")):
                    # the XS-FREE credit targets: NoC (x, y) of the writers + lane (dm0 too when it raises P10's H0f)
                    noc = lambda xy: device.worker_core_from_logical_core(ttnn.CoreCoord(*xy))  # noqa: E731
                    wr = self.xs_writers(rank)
                    n_rt = max((len(v) for v in rt_d.values()), default=0)
                    nmax = max(len(v) for v in wr.values())
                    for c_ in cores:  # [n, x0, y0, .., zero pad to nmax pairs]: n = 0 for a remote block's holder
                        tail = [int(u) for xy in wr[c_] for w in [noc(xy)] for u in (w.x, w.y)]
                        rt_d[c_] = (rt_d[c_] + [0] * (n_rt - len(rt_d[c_])) + [len(tail) // 2] + tail
                                    + [0] * (2 * nmax - len(tail)))  # fmt: skip
                    defines += [("PMK_XSFREE_ARG", str(n_rt)), ("PMK_XSFREE_N", str(nmax))]
                    tails["xsfree"] = (n_rt, nmax)
                    if fs_sync_addr:  # the lane side: lane (G % 2)'s NoC (x, y) + its XSFREE word address
                        base = n_rt + 1 + 2 * nmax
                        for G, c_ in enumerate(cores):
                            lw = noc(self.xs_credit_lane(G))
                            rt_d[c_] = rt_d[c_] + [int(lw.x), int(lw.y), int(fs_sync_addr) + FS_W_XSFREE * FS_WORD]
                        defines.append(("PMK_XSFREE_LANE", str(base)))
                        tails["lane"] = base
                in_table = tb is not None and role in ARG_TABLE_ROLES and risc != "c" and bool(mine)
                part_fid = {}  # the part regime (merge_compute): {part key: PMK_PART_FID}
                cfg_c = merge_compute(computes, cfgs["c"], flex=flex, part_fid=part_fid) if risc == "c" else None
                src_ = self.composite_source(role, risc, arg_table=in_table, part_fid=part_fid)
                if (
                    os.environ.get("GLM53_PMK_KEEP_SRC", "0") == "1"
                ):  # the generated composites per program (review / dumps)
                    tag_ = "+".join(sorted(self.scope, key=lambda q: int(q[1:]))) if self.scope else "all"
                    self.__dict__.setdefault("sources", {})[(tag_, role, risc, rank)] = src_
                audit.append((role, risc, rt_d, ct, layout, tails, src_))
                self.layouts[(role, risc)] = list(layout)
                rt_launch = rt_d
                if in_table:  # the full rows into the table; the launch carries (table address, row)
                    base = tb["base"][(role, risc)]
                    width = max((len(v) for v in rt_d.values()), default=0)
                    if 4 * width > self.arg_page:
                        raise ValueError(f"PMK {role}/{risc}: {width} args > the {self.arg_page // 4}-word table row")
                    rows_r = tb["rows"].setdefault(rank, {})
                    rt_launch = {}
                    for i_, c_ in enumerate(cores):
                        rows_r[base + i_] = list(rt_d[c_])
                        rt_launch[c_] = [int(tb["addr"]), base + i_]
                    defines += [("PMK_ARG_TABLE", "1"), ("PMK_ARG_CB", str(CB_ARG)), ("PMK_ARG_PAGE", str(self.arg_page)),
                                ("PMK_ARG_SLOT", "0" if risc == "dm0" else "1")]  # fmt: skip
                rt = ttnn.RuntimeArgs()
                for (x, y), v in rt_launch.items():
                    rt[x][y] = v
                if risc == "c":
                    cfg = cfg_c
                else:  # a DM composite: DM_DYNAMIC_NOC if any part asks (PfStream's streamer multicasts on NoC 1).
                    # tt-metal: "KernelGroup must have the same noc mode for all kernels" -> BOTH DM composites of the
                    # role go dynamic when either RISC's parts ask (assembly, 2x4 mock 19:48Z: the streamer BRISC on
                    # the roots / MM-only cores beside the static mhc root sync / kda scan writer on their NCRISC)
                    dyn = any(getattr(sub.get("config"), "noc_mode", None) == ttnn.NOC_MODE.DM_DYNAMIC_NOC
                              for p in mine if p.body.status == "body" for sub in p.body.subparts(role, risc, rank))  # fmt: skip
                    other = "dm1" if risc == "dm0" else "dm0"
                    dyn = dyn or any(
                        getattr(sub.get("config"), "noc_mode", None) == ttnn.NOC_MODE.DM_DYNAMIC_NOC
                        for p in self.phases if p.body.status == "body" and self._hosts(role, other, p)
                        and (p.rows or p.pid in ("P9",)) and self.in_scope(p.pid)
                        for sub in p.body.subparts(role, other, rank))  # fmt: skip
                    cfg = (ttnn.DataMovementConfigDescriptor(processor=cfgs[risc].processor, noc=cfgs[risc].noc,
                                                             noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC)
                           if dyn else cfgs[risc])  # fmt: skip
                kidx[(role, risc)] = len(kernels)
                kernels.append(ttnn.KernelDescriptor(kernel_source=src_,
                                                     source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                                     core_ranges=crs(cores), compile_time_args=ct, runtime_args=rt,
                                                     defines=defines, config=cfg))  # fmt: skip
            wl = self._win_layout(
                role
            )  # PHASE WINDOWS: window 1 in place, later windows one page each, the barrier CBs
            decl = (wl["decl"] if wl is not None else
                    [([self.cbs[role].get(i) or self.cbs["streamer"][i] for i in idx], nbytes)
                     for idx, nbytes, _, _ in self._slots(role, self.active_phases())])  # fmt: skip
            for rec_ in ((self.win_reclaim or {}).get(role), (getattr(self, "win_reclaim2", None) or {}).get(role)):
                if not (rec_ is not None and rec_.get("allocs") and self.in_scope(rec_["pid"]) and any(
                        p_.pid == rec_["pid"] and p_.body.status == "body" and role in p_.roles for p_ in self.phases)):  # fmt: skip
                    continue
                # a registered reclaim body whose allocations are never laid out runs on the launch-stale CB interfaces
                # of indices this program does not declare (get_write_ptr = 0 / a previous program's): refuse on the host
                w_r = self._win_index(rec_["pid"], self.windows(role)) if wl is not None else None
                laid = set() if (wl is None or w_r not in (wl.get("reclaim") or {})) else {
                    int(t[0]) for t in (wl.get("tabs") or {}).get(w_r, ())}  # fmt: skip
                want_ = {int(i_) for ents, _t in rec_["allocs"] for i_, _pg in ents}
                if not (want_ <= laid):
                    raise ValueError(
                        f"PMK {role}: the reclaim window of {rec_['pid']} is not laid out in program "
                        f"{sorted(self.scope or ())} (indices {sorted(want_ - laid)} never re-laid; wl "
                        f"{'None' if wl is None else sorted((wl.get('reclaim') or {}).keys())})"
                    )
            if wl is not None:
                self.window_report = dict(getattr(self, "window_report", {}) or {})
                self.window_report[role] = {
                    k: wl[k] for k in ("windows", "base", "bar", "span", "pad", "need", "bytes", "tabs", "hosted")
                }
            for uses, nbytes in decl:
                cbs.append(ttnn.CBDescriptor(total_size=nbytes, core_ranges=crs(cores), format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=u.index, data_format=fmt[u.fmt], page_size=u.page_bytes)
                    for u in uses]))  # fmt: skip
        init = {}
        for p in self.phases:
            if p.body.status == "body" and self.in_scope(p.pid):
                for sid, v in (p.body.sem_init or {}).items():
                    if init.get(sid, v) != v:
                        raise ValueError(f"PMK semaphore {sid}: initial {init[sid]} vs {v} ({p.pid})")
                    init[sid] = int(v)
        self._wait_reach_check(init)
        for sid, roles in sorted(self.sems.items()):
            cores = [c for r in roles for c in rc[r]]
            sems.append(ttnn.SemaphoreDescriptor(id=sid, core_ranges=crs(cores), initial_value=init.get(sid, 0)))
        self.audit_args(device, audit, rank)  # raises before any launch (mock and device alike)
        if tb is not None:
            self._arg_table_commit(tb, rank)
        n_pmk = len(sems)
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        self.kernel_index = dict(kidx)  # (role, risc) -> index in prog.kernels (the fabric connections need it)
        for hook in getattr(self, "post_build", []):  # assembly: e.g. FabricStream.connect(prog, r, c, lane dm0 index)
            hook(prog, rank, dict(kidx))
        self.fabric_word_audit = self.check_fabric_words(device, prog, n_pmk, audit, rank)
        return prog

    @staticmethod
    def check_fabric_words(device, prog, n_pmk: int, comps: list, rank: int) -> dict:
        """THE FABRIC WORDS CHECK (coordinator 2026-10-23 17:59:48Z; after three TP8 hangs in the first op fabric CCL after
        a PMK sequence). Every fabric connection opened by this program keeps its TEARDOWN ack and producer CURSOR in its
        core's semaphores (fabric_stream.fabric_connect: ids >= FAB_SEM_BASE, the free ids below are placeholders).
        Raises, before any launch, when (1) a connection's word is not on the reserved ids, (2) a PMK semaphore shares an
        id with a connection word on that core, or (3) a PMK part declares a NoC target (part['noc_xy']) on a core that
        holds connection words, unless the part is a fabric sender itself (fab_at_arg: its own coordinates for the
        peers' sync words) or declares noc_words (its targets are FabricStream sync words, never semaphores). The
        PfStream multicast rectangles are exact member rectangles (pf_stream.PfStream asserts it), so no lane is inside
        one. Returns {core: [connection word ids]}."""
        import ttnn
        from models.autoports.zai_org_glm_5_3_flash.tt.fabric_stream import FAB_SEM_BASE, _sem_cores

        sems = list(prog.semaphores)
        fab, pads, errs = {}, {}, []
        for s_ in sems[n_pmk:]:
            for c_ in _sem_cores(s_):
                (fab if int(s_.id) >= FAB_SEM_BASE else pads).setdefault(c_, []).append(int(s_.id))
        for c_, ids in pads.items():  # placeholders only where a connection lives (fabric_connect declares them)
            if c_ not in fab:
                errs.append(
                    f"core {c_}: semaphores {sorted(ids)} appended after the build with no fabric connection there"
                )
        own = {}
        for s_ in sems[:n_pmk]:
            for c_ in _sem_cores(s_):
                own.setdefault(c_, set()).add(int(s_.id))
        for c_, ids in fab.items():
            if len(ids) != len(set(ids)) or own.get(c_, set()) & set(ids):
                errs.append(f"core {c_}: connection words {sorted(ids)} vs PMK semaphores {sorted(own.get(c_, ()))}")
        if fab:
            noc = {}
            for c_ in fab:
                w = device.worker_core_from_logical_core(ttnn.CoreCoord(*c_))
                noc[(int(w.x), int(w.y))] = c_
            for role, risc, rt_d, ct, layout, tails, src in comps:
                for pid, ct0, rt0, part in layout:
                    if part.get("fab_at_arg") is not None or part.get("noc_words"):
                        continue
                    for c_, offs in (part.get("noc_xy") or {}).items():
                        args = rt_d[tuple(c_)]
                        for i in offs:
                            xy = tuple(int(a) for a in args[rt0 + i : rt0 + i + 2])
                            if (
                                xy in noc
                                and part.get("noc_sem") is not None
                                and int(part["noc_sem"]) not in fab[noc[xy]]
                            ):
                                continue  # (W7 P15) a semaphore +1 whose id is not a connection word on that core
                            if xy in noc:
                                errs.append(f"rank {rank} {role}/{risc} {pid} {part.get('name', '')} core {tuple(c_)}: NoC target "
                                            f"{xy} = core {noc[xy]} holding fabric connection words {sorted(fab[noc[xy]])}")  # fmt: skip
        if errs:
            raise ValueError(
                f"PMK fabric connection words (rank {rank}): "
                + "; ".join(errs[:8])
                + (f" (+{len(errs) - 8})" if len(errs) > 8 else "")
            )
        return {f"{c_}": sorted(v) for c_, v in fab.items()}

    @staticmethod
    def rt_bytes(comps: list) -> dict:
        """the per-launch RUNTIME-ARG bytes of one program (trace-shrink ruling 2026-10-23 16:33:48Z: every core's args
        are written into the trace on every launch): per (role, RISC) composite cores x width (every core is padded to
        the composite's width), per registered part its padded width x the composite's cores, and the composite tails
        (XS-FREE / lane triples) as their own entry"""
        out = dict(total=0, composites={}, parts={})
        for role, risc, rt_d, ct, layout, tails, _src in comps:
            n = len(rt_d)
            width = max((len(v) for v in rt_d.values()), default=0)
            words = sum(len(v) for v in rt_d.values())
            out["composites"][f"{role}/{risc}"] = dict(cores=n, width=width, bytes=4 * words)
            out["total"] += 4 * words
            offs = sorted({rt0 for _, _, rt0, _ in layout} | {width})
            tail0 = min((v if isinstance(v, int) else v[0] for v in tails.values()), default=width)
            for pid, _ct0, rt0, _part in layout:
                nxt = min((o for o in offs if o > rt0), default=rt0)
                nxt = min(nxt, max(tail0, rt0))
                k = f"{role}/{risc}/{pid}"
                out["parts"][k] = out["parts"].get(k, 0) + 4 * (nxt - rt0) * n
            if tails:
                out["parts"][f"{role}/{risc}/tails"] = 4 * max(0, width - tail0) * n
        out["top_parts"] = sorted(out["parts"].items(), key=lambda kv: -kv[1])[:8]
        return out

    def addr_hits(self, addrs: dict, spans: dict | None = None) -> dict:
        """{name: {"program:rank role/risc pid": n cores}} of the runtime args equal to a tensor's DRAM address (addrs:
        name -> buffer address), or inside [address, address + span) when spans[name] is given (a base + offset arg).
        An arg is attributed to the part whose ph_rt0 is the largest <= its index (tails after the last part count under
        that part; the post-build fabric args are not in these rows). The
        2026-10-25 x1 race (w7tp8*: the in-place residual corrupted after P10 at TP8): which phases hold its address"""
        out = {nm: {} for nm in addrs}
        for key, comps in (getattr(self, "arg_rows", {}) or {}).items():
            for role, risc, rt_d, layout in comps:
                for core, args in rt_d.items():
                    for i, v in enumerate(args):
                        for nm, a in addrs.items():
                            hit = v == a if not (spans and nm in spans) else a <= v < a + int(spans[nm])
                            if not hit:
                                continue
                            own = [(r0, p) for p, _c0, r0, _part in layout if r0 <= i]  # the part laid last below i
                            pid = max(own)[1] if own else "head"
                            k = f"{key} {role}/{risc} {pid}"
                            out[nm][k] = out[nm].get(k, 0) + 1
        return out

    def audit_args(self, device, comps: list, rank: int) -> dict:
        """RUNTIME-ARG AUDIT of every composite (coordinator 11:27Z, after the 11:25Z watcher NOC assert: an atomic
        increment to NoC (0, 0) from a member NCRISC = a zero (x, y) argument). Host-side, before any launch, on the
        mock and on a device alike. Per (role, RISC) composite and per core:
          1. the generated source's per-namespace bases (PH_CT0, ph_rt0) equal the descriptor's concatenation offsets
             for every registered body (a body reads its args where they were written);
          2. each registered part's own per-core args sit unchanged at [ph_rt0, ph_rt0 + len) (padding never shifts a
             later phase), its compile-time args at [PH_CT0, ..);
          3. the part's rt keys are exactly the composite's cores: a key outside = args for a core the body does not
             run on; a composite core with NO args = the body runs there on all-zero args (unless part['idle_ok']);
          4. declared NoC coordinates (part['noc_xy'][core] = arg offsets of (x, y) pairs) are worker NoC coordinates
             of this device, declared addresses (part['nonzero'][core]) are non-zero;
          5. the engine's own tails: the XS-FREE credit [n, x0, y0, ..] (n <= PMK_XSFREE_N, every pair a worker) and
             the lane triple (x, y, word address);
          6. no core exceeds the runtime-arg floor (MAX_RT_ARGS).
        Returns (and stores as self.arg_audit) the summary; raises ValueError listing the violations."""
        import ttnn

        P = self.place
        valid = set()
        for x in range(P.gx):
            for y in range(P.gy):
                w = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
                valid.add((int(w.x), int(w.y)))
        errs, checked = [], dict(composites=0, cores=0, parts=0, noc_pairs=0, xsfree_pairs=0)
        rc = P.role_cores()
        for role, risc, rt_d, ct, layout, tails, src in comps:
            checked["composites"] += 1
            cores = [tuple(c) for c in rc[role]]
            bases = {m.group(1): (int(m.group(2)), int(m.group(3))) for m in re.finditer(
                r"namespace ph(\w+) \{\nconstexpr uint32_t PH_CT0 = (\d+);\nconstexpr uint32_t ph_rt0 = (\d+);", src)}  # fmt: skip
            tag = f"rank {rank} {role}/{risc}"
            # a table composite's launch carries 2 words; its row is bounded by the engine's arg page instead
            lim = (self.arg_page // 4 if getattr(self, "_use_table", False) and role in ARG_TABLE_ROLES and risc != "c"
                   else MAX_RT_ARGS)  # fmt: skip
            for c in cores:
                if len(rt_d[c]) > lim:
                    errs.append(f"{tag} {c}: {len(rt_d[c])} runtime args > {lim}")
            for pid, ct0, rt0, part in layout:
                checked["parts"] += 1
                if bases.get(pid[1:]) != (ct0, rt0):
                    errs.append(
                        f"{tag} {pid}: source bases {bases.get(pid[1:])} != descriptor (PH_CT0 {ct0}, ph_rt0 {rt0})"
                    )
                if ct[ct0 : ct0 + len(part["ct"])] != [int(a) for a in part["ct"]]:
                    errs.append(f"{tag} {pid}: compile-time args not at PH_CT0 {ct0}")
                keys = {tuple(k) for k in part["rt"]}
                if not keys:
                    continue
                outside = sorted(keys - set(cores))
                if outside:
                    errs.append(
                        f"{tag} {pid}: args for {len(outside)} core(s) the composite does not run on, e.g. {outside[:3]}"
                    )
                missing = [c for c in cores if c not in keys]
                if missing and not part.get("idle_ok"):
                    errs.append(
                        f"{tag} {pid}: the body runs on {len(missing)} core(s) with NO args (all zero), e.g. {missing[:3]}"
                    )
                for c in cores:
                    checked["cores"] += 1
                    own = [int(a) for a in part["rt"].get(c, [])]
                    if rt_d[c][rt0 : rt0 + len(own)] != own:
                        errs.append(f"{tag} {pid} {c}: own args not at ph_rt0 {rt0}")
                        continue
                    for i in (part.get("noc_xy") or {}).get(c, ()):
                        checked["noc_pairs"] += 1
                        xy = tuple(rt_d[c][rt0 + i : rt0 + i + 2])
                        if xy not in valid:
                            errs.append(f"{tag} {pid} {c}: arg {rt0 + i} (body offset {i}) NoC {xy} is not a worker")
                    for i in (part.get("nonzero") or {}).get(c, ()):
                        if rt_d[c][rt0 + i] == 0:
                            errs.append(f"{tag} {pid} {c}: arg {rt0 + i} (body offset {i}) is a zero address")
                    fa = part.get("fab_at_arg")
                    nfab = sum(1 for _, _, _, q in layout if q.get("fab_at_arg") is not None)
                    if (
                        fa is not None and own and nfab > 1
                    ):  # two fabric phases on one composite (stage 2): the post-build
                        checked["fab_at_patched"] = checked.get("fab_at_patched", 0) + 1  # hook re-points each slot
                        fa = None
                    if (
                        fa is not None and own
                    ):  # a fabric send part: its fab_at slot = THIS composite's width, where the
                        # post-build connect() appends the connection args (01:22Z: a second sub-part on the tail BRISC
                        # shifted them; the send kernel read zeros and opened a connection to NoC (0, 0))
                        checked["fab_at"] = checked.get("fab_at", 0) + 1
                        if rt_d[c][rt0 + fa] != len(rt_d[c]):
                            errs.append(
                                f"{tag} {pid} {c}: fab_at {rt_d[c][rt0 + fa]} != composite width {len(rt_d[c])}"
                            )
            if "xsfree" in tails:
                a0, nmax = tails["xsfree"]
                for c in cores:
                    n = rt_d[c][a0]
                    if n > nmax:
                        errs.append(f"{tag} {c}: XS-FREE count {n} > PMK_XSFREE_N {nmax}")
                    for r in range(min(n, nmax)):
                        checked["xsfree_pairs"] += 1
                        xy = tuple(rt_d[c][a0 + 1 + 2 * r : a0 + 3 + 2 * r])
                        if xy not in valid:
                            errs.append(f"{tag} {c}: XS-FREE pair {r} NoC {xy} is not a worker")
            if "lane" in tails:
                b0 = tails["lane"]
                for c in cores:
                    if tuple(rt_d[c][b0 : b0 + 2]) not in valid or rt_d[c][b0 + 2] == 0:
                        errs.append(f"{tag} {c}: lane triple {rt_d[c][b0 : b0 + 3]} invalid")
        self.arg_audit = dict(rank=rank, errors=len(errs), **checked)
        self.arg_audit["rt_bytes"] = self.rt_bytes(comps)
        key = str(rank) if self.scope is None else f"{'+'.join(sorted(self.scope, key=lambda q: int(q[1:])))}:{rank}"
        self.arg_audits = dict(getattr(self, "arg_audits", {}), **{key: self.arg_audit})  # every (program,) rank
        # the per-core runtime args by phase (addr_hits: which phases carry a given tensor address)
        self.arg_rows = dict(
            getattr(self, "arg_rows", {}),
            **{key: [(role, risc, rt_d, layout) for role, risc, rt_d, _ct, layout, _t, _s in comps]},
        )
        if errs:
            raise ValueError(f"PMK runtime-arg audit FAILED ({len(errs)}): " + "; ".join(errs[:16]))
        return self.arg_audit


# ------------------------------------------------------------------------------- the harness-facing builder
class PmkSparseKdaLayer:
    """The BUILDER CONTRACT of tests/s16/pmk_harness.py (jobs/pmk_layer_unit.py, jobs/pmk_layer_mock.py): one sparse_kda
    layer of a TPPrefill as ONE persistent program. `prog(x, ci)` launches that program (the compile launch on the first
    call; the harness captures it); `counters["eager"]` stays 0 (the builder launches nothing else); `plan()` = the
    phase table + the L1 census (l1_max_cb_end); `probe_read()` / `taps()` as documented there. v1 skeleton: the phase
    bodies are the stubs until the pieces register theirs (register_phase), so y = x (in place, untouched) -- the unit
    row reports the numerics honestly; the program, its CBs, semaphores and hand-off waits are the real layout."""

    impl = "pmk:family-skeleton"

    def __init__(self, pf, li: int, c: int, sp: bool, probe: bool = False, *, geom: PmkGeom | None = None,
                 place: Placement | None = None, programs: list | None = None):  # fmt: skip
        self.pf, self.li, self.c, self.sp, self.probe = pf, int(li), int(c), bool(sp), bool(probe)
        self.counters = dict(eager=0, calls=0)
        mesh = pf.mesh
        tp = int(pf.ndev)
        if not sp and tp > 1:
            raise ValueError("PMK v1 is sequence-parallel (contract §0): a replicated bucket has no PMK program")
        self.geom = geom or PmkGeom.from_config(pf.cfg, tp, c)
        if place is None:
            grid = mesh.compute_with_storage_grid_size()
            avoid = ()
            try:
                from models.autoports.zai_org_glm_5_3_flash.tt.mk_readers import reader_assignments

                avoid = tuple((int(cc.x), int(cc.y)) for cc, _, _ in reader_assignments(mesh, 3, "MESH_ROW_PLUS_X"))
            except Exception:  # noqa: BLE001 -- no device query: the plan's own placement
                pass
            place = Placement(gx=int(grid.x), gy=int(grid.y), avoid=avoid)
        self.engine = PmkSparseKdaPlan(self.geom, place).plan()
        # the layer as SEVERAL persistent programs back to back in the caller's trace (ruling 13:5xZ: v1 = A P0..P11
        # + B P12..P16; incremental units: a piece per program). Cross-program hand-offs go through DRAM: the engine
        # emits no wait for a producer outside the program (_has_live_producer); each program has its own CB census
        # an entry is a phase-id set (a family program) or a CALLABLE = a piece-built persistent program launched as is
        # (7a's FabricStreamAG / PfAg / PfRs, moe2's program B): fn(x, ci) issues its own generic_op inside the trace
        self.external = {i: pp for i, pp in enumerate(programs or []) if callable(pp)}
        self.engine.programs = [set(pp) for pp in programs if not callable(pp)] if programs else None
        self.order = list(programs) if programs else None
        self.engine.check_l1()
        self.desc = None
        self.fs_sync_addr = 0  # coll sets it (FabricStreamC1.sync.buffer_address()) when its lanes are registered

    def __call__(self, x, ci, **_):
        import ttnn

        io = self._build_descs(x, ci)
        fam = iter(self.descs)
        for entry in self.order or [None]:  # the layer's programs, in order (the harness captures them in one trace)
            if callable(entry):
                entry(x, ci)
            else:
                ttnn.generic_op(io, next(fam))
        self.counters["calls"] += 1
        return x  # in place: the stub phases leave the residual untouched; P16 overwrites it when registered

    def prebuild(self, x, ci) -> None:
        """the descriptors of the current engine.programs WITHOUT a launch: a program's FIRST build writes its DRAM arg
        table (GLM53_PMK_ARGS=table, a host write) -- a harness builds every program it will capture first, outside any
        trace (a sub-scope program first built inside its capture: TT_FATAL 'Writes are not supported during trace
        capture'); a later build of the same program only re-checks its rows"""
        self._build_descs(x, ci)

    def _build_descs(self, x, ci) -> list:
        import ttnn

        pl = self.pf.layers[self.li]
        scratch = getattr(pl, "scratch", None)
        io = [x] + [t for t in (getattr(scratch, "S", None), getattr(scratch, "C", None)) if t is not None]
        seen = {id(t) for t in io}
        for p in self.engine.phases:  # the registered bodies' tensors (their addresses are baked into the args)
            for t in p.body.io if p.body.status == "body" else ():
                if id(t) not in seen:
                    io.append(t)
                    seen.add(id(t))
        io.append(io.pop(0))  # generic_op's last io is the output: the residual x (updated in place)
        dev = x.device() or self.pf.mesh  # (x.device() is None for some harness inputs: pmk-int 10:07Z)
        n = int(dev.get_num_devices()) if hasattr(dev, "get_num_devices") else 1
        e = self.engine
        scopes = list(e.programs) if e.programs else [None]
        self.descs, self.launch_census = [], {}
        try:
            for sc in scopes:  # (the family programs; the external entries are launched in self.order below)
                e.scope = set(sc) if sc is not None else None
                if n > 1:  # one program per chip: the XS-FREE writer lists depend on the rank (row-major mesh index)
                    R, C_ = int(dev.shape[0]), int(dev.shape[1])
                    desc = ttnn.MeshProgramDescriptor()
                    for r in range(R):
                        for c_ in range(C_):
                            m = ttnn.MeshCoordinate(r, c_)
                            desc[ttnn.MeshCoordinateRange(m, m)] = e.descriptor(dev, io, rank=r * C_ + c_,
                                                                                fs_sync_addr=self.fs_sync_addr)  # fmt: skip
                else:
                    desc = e.descriptor(dev, io, rank=0, fs_sync_addr=self.fs_sync_addr)
                # LAUNCH CENSUS (mandatory on a device, never bypassed): the CBs this program actually allocates
                # (registered bodies in its scope) must end under the ceiling. 10-22 11:25Z: never launch over it.
                launch = e._census(e.active_phases())
                tag = "all" if sc is None else "+".join(sorted(sc, key=lambda q: int(q[1:])))
                self.launch_census[tag] = {k: v["cb_end"] for k, v in launch.items()}
                over = {k: v["cb_end"] for k, v in launch.items() if not v["fits"]}
                if over and os.environ.get("GLM53_MOCK") != "1":
                    raise ValueError(f"PMK launch census over the CB ceiling on a DEVICE: {over} (program {tag})")
                self.descs.append(desc)
        finally:
            e.scope = None
        if len(self.descs) == 1:
            self.launch_census = next(iter(self.launch_census.values()))
        self.desc = self.descs[0]
        return io

    def register_phase(self, pid: str, body: PhaseBody):
        return self.engine.register_phase(pid, body)

    def plan(self) -> dict:
        e = self.engine
        rc = e.place.role_cores()
        phases = []
        for p in e.phases:
            roles = [r for r in p.roles if r in e.census]
            cbs = sorted({u.index for r in p.roles for u in e.cbs.get(r, {}).values() if p.pid in u.live})
            phases.append(dict(name=f"{p.pid} {p.name}", cores=sum(len(rc[r]) for r in p.roles if r not in ("streamer", "mm")),
                               roles=list(p.roles), l1_bytes={r: e.census[r]["cb_bytes"] for r in roles},
                               cbs=cbs, owner=p.owner, status=p.body.status, waits=list(p.waits), signals=list(p.signals),
                               collective=p.collective, rows=p.rows, wraps=list(p.wraps)))  # fmt: skip
        return dict(impl=self.impl, geometry=e.policy["geometry"], phases=phases,
                    l1_max_cb_end=max(v["cb_end"] for v in e.census.values()),
                    l1=  {k: {kk: vv for kk, vv in v.items() if kk != "slots"} for k, v in e.census.items()},
                    policy=e.policy, handoffs={h.hid: dict(name=h.name, counter=h.counter, expected=h.expected, unit=h.unit,
                                                           lives_in=h.lives_in) for h in e.handoffs.values()},
                    rings={f"{k[0]}x{k[1]}": v["block_bytes"] for k, v in e.rings.items()},
                    semaphores={i: sorted(r) for i, r in e.sems.items()}, phase_table=e.phase_table(),
                    census_table=e.census_table())  # fmt: skip

    def probe_read(self):
        return None  # PROBE CB 31 rows: wired when the first body with MK_MARK lands (open question in the README)

    def taps(self) -> dict:
        return {}  # H1.xhat / H6.y / x1 / H10.y_f: the mhc / coll bodies expose them (open question in the README)


def build_layer(pf, li: int, c: int, sp: bool, probe: bool = False):
    """the family's builder for a TPPrefill layer by kind (sparse_kda; dense_kda = Stage 17's phase set: the same
    attention half, the FFN half one dense MLP slice; sparse_dsa comes from 14)"""
    from models.autoports.zai_org_glm_5_3_flash.tt.config import DENSE_KDA, SPARSE_KDA

    kind = pf.layers[li].kind
    if kind not in (SPARSE_KDA, DENSE_KDA):
        raise NotImplementedError(f"PMK family builds sparse_kda / dense_kda layers; L{pf.layers[li].L} is {kind}")
    geom = PmkGeom.from_config(pf.cfg, int(pf.ndev), c, kind="dense_kda" if kind == DENSE_KDA else "sparse_kda")
    return PmkSparseKdaLayer(pf, li, c, sp, probe=probe, geom=geom)
