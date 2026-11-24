# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PMK ASSEMBLY: program A of the sparse_kda layer (ruling 2026-10-22 13:50Z: A = P0..P11, B = P12..P16) COMPOSED from
the delivered pieces on the family engine (tt/pmk_layer.py: phase registration, base-aware waits, audit_args,
per-program census), as SEVERAL persistent programs in one trace (closure 1, coordinator ~20:30Z: the member L1 holds
one phase family at a time; one compute regime per program):

  A1 {P1}        mhc pre (attn) (tt/pmk_mhc.py part P1; DRAM x-hat / y / coefficient pages: the modes of its silicon
                 units; TP8: F_X_LOAD of the resident block)                                -> hn[0], posts[0], combs[0]
  A2 {P2, P3}    7a FabricStreamC1 (pair) / C1X (2x4) lanes + c1_stub_prod on the holders (DRAM x-hat rows -> XS 35,
                 FS_ARRIVE + LREADY per slice) = tt/pf_ag.PfAgN's silicon path; then the proj piece's PfStream
                 (adj_pmk rt 1, kb 16, nb 4, PARK 4, HiFi2: DECISIONS 19:16Z) as four stream phases on the 64 members:
                 qkv (N 3W), low-rank [f_a | g_a | b | 0] (N 512), f_b, g_b (K 512 over the low-rank output, fp32
                 outputs, their own PARK: nbk must stay even)                              -> u, lr, f, gate_out (DRAM)
  A3 {P4, P6}    the kda piece's items (members) / scan (MM-only), H3 pre-raised (the stores are in DRAM)
                                                                                           -> on (DRAM), S / C
  A4 {P7, P8}    PfStream o_proj (x = on) with the H5 readiness signals + FabricStreamRSR lanes-reduce (pair)
                                                                                           -> y fp32 (DRAM)
  A5 {P10, P11}  [PCOEF rebuilt from DRAM posts[0] / combs[0] (kernels/pmk/asm/pmk_pcoef_dram.cpp)] mhc post (attn)
                 + pre (ffn) [TP8: X loaded before / stored after (pmk_x_io.cpp)] -> hn[1], posts[1], combs[1]; then
                 the AG of hn[1] into XS (C1, use 1 of this program)                        -> the A -> B contract
Hand-offs inside a program are the family's counters (§4b); across programs DRAM (the engine emits no wait for a
producer outside the program). Member L1 per program (TP2): A1 ~740 KB, A2 ~790, A3 700, A4 ~560, A5 ~836 -- all under
the 997.7 KB budget with ONE alias group: X (2 / 3) <-> XS 35 / IN0 40 (both 256 KB: exact capacities; X live P1 / P10,
XS live P2 / P3 / P7 / P11). Host mocks only here; the device units: tests/s16/jobs/pmk_asm_a_unit.py."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import os
import struct

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import fabric_stream as FS
from models.autoports.zai_org_glm_5_3_flash.tt import pf_stream as PF
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_b_window as PBW
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_layer as PMK
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_mhc as MHC
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_sparse_kda as INT
from models.autoports.zai_org_glm_5_3_flash.tt.fs_compose import _rt_dict

TILE = 32
ASM_KDIR = os.path.join(PMK.KDIR, "pmk", "asm")
SCOPES = (("P1",), ("P2", "P3"), ("P4", "P6"), ("P7", "P8"), ("P10", "P11"))
# PMK_A3_MERGE=1: the kda items / scan (A3) inside A2 (one launch, no DRAM round trip of u / lr / f / gate_out; the in-
# program H3 barrier below). Costs the member TRISC regime: PfStream must run HiFi4 full-sync (the items' regime).
SCOPES_A3_MERGE = (("P1",), ("P2", "P3", "P4", "P6"), ("P7", "P8"), ("P10", "P11"))
# the mhc-only program A (attn=False: the sparse_dsa layers of the served stack, 14b 2026-10-23): A1 + A5 around the
# layer's op attention (MLA / DSA until 16's attention entry lands); no streams, no kda, no RS, X never pinned
SCOPES_MHC = (("P1",), ("P10", "P11"))
# PMK_A_FOLD=1 (A lever 2, coordinator 19:18:59Z): the two mHC programs folded into their neighbours -- A1 into A2 and A5
# into A4 (5 -> 3 launches per layer; P10 consumes the attention output A4 writes, so it can only follow it)
SCOPES_FOLD = (("P1", "P2", "P3"), ("P4", "P6"), ("P7", "P8", "P10", "P11"))
# PMK_A34_MERGE=1 (Stage 14, the per-layer kernel framework's first fused program, coordinator 2026-10-23 23:35:19Z): A3
# + A4 = {P4, P6, P7, P8} in ONE program. The member CBs of the two families TIME-SHARE L1 (the engine's PHASE WINDOWS:
# the kda items' CBs in place, o_proj's re-laid out into their dead region after a core barrier); the member TRISC runs
# the items' regime (HiFi4, fp32 full-sync DST) with o_proj's matmuls at their own HiFi2 (the part regime,
# kernels/pmk/pmk_fid.hpp); P7 waits H4 in-program (the scan leader's SEM_ON, +1 per chunk, on every member).
SCOPES_A34_MERGE = (("P1",), ("P2", "P3"), ("P4", "P6", "P7", "P8"), ("P10", "P11"))
A34_SCOPE = ("P4", "P6", "P7", "P8")
A34_WINDOWS = (("P4", "P6"), ("P7", "P8"))
# PMK_A2_FUSE=1 (stage 2 of the per-layer kernel, coordinator 2026-10-24 00:13:51Z): A2 + A3 + A4 = {P2 .. P8} in ONE program,
# three member windows (the projections | the kda items | o_proj); P3's PfStream rings re-placed for P7 (fresh ring), the
# in-program H3 barrier (the A3 merge's parts, without its HiFi4 regime: the part regime keeps P3 / P7 at HiFi2), and two
# fabric phases on the lanes in sequence (C1 / C1X in P2, RSR / RS8 in P8: serial open / close, fab_at patched per program)
A2F_SCOPE = ("P2", "P3", "P4", "P6", "P7", "P8")
A2F_WINDOWS = (("P2", "P3"), ("P4", "P6"), ("P7", "P8"))
SCOPES_A2_FUSE = (("P1",), A2F_SCOPE, ("P10", "P11"))
# PMK_A15_FUSE=1 (M-A of the per-layer megakernel, operator / coordinator 2026-10-24 03:30:35Z): A1 + A2 + A3 + A4 + A5 =
# {P1 .. P10} in ONE program (P11 off: the served default), five windows (mhc pre | projections | kda items | o_proj |
# mhc post) on the members and the roots. Beyond stage 2 it needs the hand-offs the program boundaries gave for free:
# H1 (P1's x-hat stores -> the P2 holders' SEM_HN: an explicit fan-out part, as P10's H7), H6 (the TP2 RSR lanes' y rows
# -> every member's FS_REDUCED: a fan-out part after the lanes' writes), the mhc round counters CUMULATIVE across P1 ->
# P10 (no fresh-program fold) and the roots' FINAL rendezvous counted twice
A15_SCOPE = ("P1",) + A2F_SCOPE + ("P10",)
# the M-A bisect scopes (pmk_fuse3_unit variants p1a2f / a2fp10): A1 or A5 fused onto stage 2, the other separate
P1A2F_SCOPE, A2FP10_SCOPE = ("P1",) + A2F_SCOPE, A2F_SCOPE + ("P10",)
A15_WINDOWS = (("P1",),) + A2F_WINDOWS + (("P10",),)
SCOPES_A15_FUSE = (A15_SCOPE,)
# M-B (coordinator 2026-10-24 05:12:04Z; tt/pmk_b_window.py): A15 + program B (P14 = B's plan / routed experts / combine,
# its own kernels wrapped) in ONE program; B is the LAST window, a reclaim window from the CB base. The MM-only cores are
# windowed too (B's streamer rings beside A's): P2 P3 | P4 P6 | P7 P8 | P14
A15B_SCOPE = A15_SCOPE + ("P14",)
A15B_WINDOWS = A15_WINDOWS + (("P14",),)
# W7 / M-B2 step 1 (coordinator 2026-10-25 13:40:49Z; pmk/w7/README.md): + P11 = the AG of x-hat_f (hn[1]) into the
# holders' XS through the C1 lanes, in P10's window (A5 = {P10, P11} as in the chain); the lanes run P2 | P8 | P11 as
# three serial fabric phases (C1's protocol counts uses by its GEN word: two C1 uses in one launch are two generations)
A15B11_SCOPE = A15_SCOPE + ("P11", "P14")
A15B11_WINDOWS = A15_WINDOWS[:-1] + (("P10", "P11"),) + (("P14",),)
# the W7 step-1 bisect (the a15b11 hang 14:48Z: device 1 stalled after every fabric phase finished): A15 + P11 WITHOUT B
A15P11_SCOPE = A15_SCOPE + ("P11",)
A15P11_WINDOWS = A15_WINDOWS[:-1] + (("P10", "P11"),)
# W7 step 2 (pmk/w7/README.md section 7): + P13 = the shared expert (pfs_shared's two PfStream phases: gate_up with the
# SwiGLU epilogue into ACT, down -> y_s ROW_MAJOR = B's PMK_SHARED term) on the members, stationary in0 = the x-hat_f block
# P11 gathered into XS (waits H8); its OWN window between P10 / P11 and B's reclaim (P10's mhc CBs are dead there; XS is
# re-laid at the same offset, so its data carries: asserted in _p13_carry_check)
A15B13_SCOPE = A15B11_SCOPE[:-1] + ("P13", "P14")
A15B13_WINDOWS = A15B11_WINDOWS[:-1] + (("P13",),) + (("P14",),)
P13_SLOTS = (5, 6)  # the shared phases' PfStream sync slots (A's P3 phases 0..3, o_proj 4)
# W7 step 3 (section 8): + P12 = the router in-program -- the fp32 logits as a PfStream phase (in0 = XS, sync slot 7) and
# the router tail (router_tail/*.cpp prefill mode: sigmoid + bias, top-8, B's weights) on each member over its 32 rows,
# writing B's PMK_IDS_U16 / PMK_WT_BF16 inputs ([C, 8] uint16 / bf16 ROW_MAJOR); its own member window before P13's
A15B12_SCOPE = A15B11_SCOPE[:-1] + ("P12", "P13", "P14")
A15B12_WINDOWS = A15B11_WINDOWS[:-1] + (("P12",),) + (("P13",),) + (("P14",),)
P12_SLOT = 7
# the router tail's CBs on the member (64 / 64 used): kda item indices idle outside P4 / P5, LOG / BIAS on the two whose
# kda mode is already UnpackToDestFp32 (F 14, BETA 15: copy_tile inputs), CHOICE fp32 (MASK 16), SCORES bf16 (TAPS 18),
# the writer's 32-B id / weight scratch (CONST 17, data movement only)
P12_TAIL_CB = dict(LOG=14, BIAS=15, CHOICE=16, IDX=17, SCORES=18)
# W7 step 4 (section 9): + P15 = the RS of program B's y partial to the row owners in-program (TP2: a second
# FabricStreamRSR on A's lanes, its own sync shard / landing / fp32 y; the lanes wait every member's after-P14 signal on
# lane semaphore SEM_P15 -- B's y rows are written -- then send / reduce; P15's fabric connections = P8's (fab_at))
A15B15_SCOPE = A15B12_SCOPE + ("P15",)
A15B15_WINDOWS = A15B12_WINDOWS
SEM_P15 = 5  # lanes: ids < FAB_SEM_BASE are free there (the members' id 5 is their own L1 word)
# G v2a (W7 step 5 on the DSA layers, pmk/w7/gv2_design.md sec 4): a sparse_dsa layer's MoE half in ONE program -- A5 (P10)
# + P11 (the AG of hn[1] into XS: already run by the mhc-only program, unread until now) + P12 (router) + P13 (shared) +
# program G (P14, tt/pmk_gang.PmkGang via pmk_b_window.register_g); P11x feeds G's x_ext; the RS stays the served op
DSA_G_SCOPE = ("P10", "P11", "P12", "P13", "P14")
DSA_G_WINDOWS = (("P10", "P11"), ("P12",), ("P13",), ("P14",))
# G v2b (coordinator 01:52:14Z): the DSA layers' RS folded into G v2a as P15 (lanes / RS8 cores; the members' windows
# are G v2a's -- P15 runs after G on the lanes, the TP8 reducers' CBs in a second reclaim window on the windowed roles)
DSA_G15_SCOPE = DSA_G_SCOPE + ("P15",)
# W7 at TP8: link h of a C1X lane h is used P2 (C1X) -> P8 stage 2 (tail) -> P11 (C1X) -> P15 stage 2: the tail hands it
# BACK to the lane after P8 (rsl_send FS_RSL_REL_SEM -> the lane's SEM_LINK_ACQ), P11 waits it (c1x_send FS_C1X_ACQ_SEM)
SEM_LINK_ACQ = 6
SEM_P15_TP8 = 4
XS_SENTINEL = 0x7E5BA5E1  # W7 (I1): B's xs_base until the layout gives the holders' XS address  # TP8: lanes + the stage-1 reducers' roles (MM-only / root / tail): id 4 is free on all of them
P12_PARK_FP32 = (
    48  # the logits PARK for fp32 router weights: kda item OUT (fp32, 44 pages = a park-2 ring of 16-tile blocks)
)
# TP8 RS (FabricStreamRS8: stage 1 the board pairs on the 2 family lanes + 8 reducers, stage 2 the row on 2 more lanes +
# 8 reducers): the reducers are the 16 MM-only cores that carry no PfStream streamer (free in A4: the scan ran in A3),
# the second lane pair = 2 of the 4 tail cores. CB indices on the MM-only role (free beside the kda scan map 0..13):
RS8_CB = dict(A=14, B=15, Y1=16, T0=17, Y2=21, SEND2=0)  # T0..T0+3 = 17..20 (stage 2's 4 source slots)
PARK_TP8_RESIDENT = (
    3  # PMK_TP8_RESIDENT pins 256 KB of X in A2 too: PARK 4 would end at 1,381,504 > the 1,330,000 ruling
)
# FABRIC PACKET HEADERS (homelab-1 00:25Z, the first TP8 run: BRISC assert packet_header_pool.h:51 on the lane of device
# 2): the per-RISC PacketHeaderPool is MEM_PACKET_HEADER_POOL_SIZE / MaxDMProcessorsPerCoreType = 144 B x 24 / 2 = 12
# headers (blackhole/dev_mem_map.h); a send kernel allocates NHDR (4) data + 1 credit header per ACTIVE direction: the
# pair's one direction = 5, the 2x4 C1X's 3 directions = 15. The composite gives every send part a PRIVATE L1 CB for the
# data headers (FS_HDR_CB) and leaves only the credit headers in the pool; `_fabric_header_audit` asserts both per rank.
CB_HDR = 5  # lane / tail cores: NHDR x 4 directions x 144 B = 2,304 B -> one 4 KB page
FS_POOL_HEADERS = 12
FS_NHDR, FS_HDR_BYTES = 4, 144
# PMK_A_INPLACE=1 (Stage 12's image line, pmk_image_design.md L1): qkv / o_proj stream the RESIDENT DSLinear weights
# (ds_qkv / ds_o: DRAM WIDTH_SHARDED (K, n_pad / 8), 8 shards = 8 banks, bf8) in place -- no second copy, no bank
# padding, bitwise vs the copied layout (tests/s16/pmk_pfs_inplace_test.py). lowrank / f_b / g_b keep their bf8 stream
# copies: ds_lowrank's shard width (2 tiles) forces NB 2 (twice the blocks) and bf16 weights are unpack-bound (the
# 01:43Z pair cell: +330 us); f_b / g_b are skipped DS roles in the served config anyway (w_fb / w_gb bf16 interleaved).


def p13_kb_down(kt_d: int) -> int:
    """P13 down's K block: the largest of (KB_SMALL, 4, 2) that leaves an even K-block count (pf_stream's lane parity)"""
    return next(k for k in (KB_SMALL, 4, 2) if kt_d % k == 0 and (kt_d // k) % 2 == 0)


def p13_slice(i: int) -> int:
    """PMK_P13_SLICE (debug, default 0 = off): P13 runs only the first N columns of the chip's shared slice (gate / up
    halves alike, down rows alike) -- the TP8 slice geometry (i = 256: gate_up 16 tiles on 4 streams, down kt 8 at kb 4)
    on the pair, to reproduce the w7tp8 P13 NaN (2026-10-25 18:09Z); the unit's fp32 reference slices alike"""
    v = int(os.environ.get("PMK_P13_SLICE", "0"))
    assert v == 0 or (v % PMK.TILE == 0 and 0 < v <= i), (v, i)
    return v or i


def _knob(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)) or 0)


def _gated_text(text: str, gate_off: int, tag: str) -> str:
    """a 7a kernel TEXT wrapped so that its body runs only on the cores whose runtime args carry a 1 at ph_rt0 + gate_off
    (the engine runs every part of a (role, RISC) composite on ALL the role's cores with zero-padded args: a reducer
    kernel with a compile-time tile count would hang on the non-reducer cores -- README conflict #5)"""
    assert text.count("void kernel_main() {") == 1, tag
    inner = text.replace("void kernel_main() {", f"static void fs_gated_{tag}() {{", 1)
    return inner + (
        f"\nvoid kernel_main() {{\n    if (get_arg_val<uint32_t>(ph_rt0 + {gate_off}) != 0) {{\n"
        f"        fs_gated_{tag}();\n    }}\n}}\n"
    )


# member CB indices this assembly adds (free in the family map; reserved BEFORE the kda item map is drawn)
CB_XK = 46  # the K = 512 phases' stationary in0 (16 bf16 tiles): its own ring (contract §10.1)
CB_OUT32 = 44  # fp32 OUT of the f_b / g_b phases (the kda items read f / gate_out as fp32)
# W7 step 5 D3 (P11x, PMK_XEXT): the x_ext untilize's ROW_MAJOR staging CB on the members (P12's window; 54..59 are
# program B's in its window, 60 is free program-wide) and its block width (fp32 DEST: 4 tiles)
CB_XEXT, XEXT_BW = 60, 4
CB_PARK2 = 47  # the K = 512 phases' PARK (their K-block is half the others': a shared PARK needs one block size)
P13_DN_NB = 8  # PMK_P13_DN=bf16x8 (cf3h-k, LR_EXACT): P13 down's N block on the bf16 PARK2 (KB 4 x NB 8 = 32 tiles)
NB, NW, KB, PARK = 4, 32, 16, 4  # the projections ruling: adj_pmk rt 1, kb 16 nb 4, PARK 4 (half-sync DST, HiFi2)
KB_SMALL = 8  # the K = 512 phases: kt 16 / kb 8 = 2 K-blocks (even, as pf_stream asserts)
FBGB_NW = 8  # GLM53_PMK_FBGB_JOINT: f_b / g_b stream width (N = 32 tiles at TP8 -> 4 streams each, 8 in the joint copy)
LR_NB_INPLACE, LR_KB_INPLACE = (
    2,
    16,
)  # LR_EXACT's in-place lowrank: ds_lowrank shard 2 tiles (n_pad 512 / 8) -> NB 2, KB 16 = 32-tile blocks
FIDELITY = "HiFi2"
SEM_HN = PMK.SEM_HN
# GLM53_PMK_ROOT_FINAL=1 (coordinator 2026-10-24 00:41:21Z, the TP8 root-core late-write class): the mhc roots end A1 / A5
# with a FINAL rendezvous -- every member's mhc sync RISC (the only member -> root sender: GATHER partials + SEM_G1 / G2,
# kernels/pmk/mhc_sync.cpp) +1 its root's SEM_ROOT_FINAL after its own write / read barriers (pmk_sig.cpp, sent last),
# and the root's sync RISC waits for all of its members (pmk_wait.cpp) before the root core can complete. Id 9 = SEM_M
# (H9, raised on the lane cores only): free on the roots in program A.
SEM_ROOT_FINAL = 9
# THE LINK HAND-OVER (fused A2F at TP8; fuse8 hung 2026-10-24 02:36Z): C1X lane h (P2) and RS8 stage 2's lane h (P8, a
# tail core) both send on link h along axis 1 -- the same EDM sender channels -- and tt-fabric serves one worker per
# channel at a time. In one program the tails reached rsl_send at launch start and opened beside the C1X lanes: the
# lanes' peer credits were lost (every lane at c1x_send's credit wait, routers idle). C1X lane h +1s this semaphore on
# stage-2 lane h after its drain-closes (c1x_send.cpp FS_C1X_REL_SEM); rsl_send opens after it (FS_RSL_AFTER_SEM). Id 8
# on the tail cores (their only PMK id is SEM_LOGITS 12; 8 = SEM_A is a lane id: other cores).
SEM_LINK_REL = 8
NORM = {"attn": "input_layernorm.weight", "ffn": "post_attention_layernorm.weight"}


def _rtail_text(name: str) -> str:
    """a router_tail kernel's TEXT with router_tail_common.hpp inlined (the content hash of _src_file covers the header)"""
    d = os.path.join(PMK.KDIR, "router_tail")
    hdr = open(os.path.join(d, "router_tail_common.hpp")).read().replace("#pragma once", "")
    inc = '#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/router_tail/router_tail_common.hpp"'
    txt = open(os.path.join(d, name)).read()
    assert txt.count(inc) == 1, (name, "the router_tail header include")
    return txt.replace(inc, hdr)


def _rtail_writer_row_mask(txt: str) -> str:
    """GLM53_PMK_ROW_MASK=1 (coordinator 2026-10-26 12:10:42Z, the in-program VALID-ROW MASK of W7 / G): the router tail
    writer's text with rows >= valid_end -> the SENTINEL expert N_EXPERTS (e = id - lo >= E_loc on every chip: plan_count /
    plan_dest keep no such pair, so a padding row is in no dispatch; G's PMK_EP_SENTINEL gives it dest 0xFFFFFFFF) with
    weight 0 -- the op path's GLM53_PF_MOE_PADMASK keep semantics. valid_end = the chunk's [1] uint32 device tensor at
    runtime arg RTAIL_VE_RT (request-agnostic traces). A text patch of the PMK copy only: router_tail/writer.cpp's other
    users (decode megakernels, the op router tail) keep their kernel text, and the knob off keeps this one's."""
    subs = (
        (
            "    MK_MARK();  // tail probe 0: start\n",
            "    MK_MARK();  // tail probe 0: start\n"
            "    uint32_t valid_end;  // GLM53_PMK_ROW_MASK: the chunk's valid rows (rows >= valid_end -> sentinel expert, weight 0)\n"
            "    {\n"
            "        const InterleavedAddrGen<true> ve_g{.bank_base_address = get_arg_val<uint32_t>(ph_rt0 + RTAIL_VE_RT), .page_size = 4};\n"
            "        const uint32_t s = get_write_ptr(CB_IDX);  // the scratch's first 64 B (the RM_OUT rows start at + 64)\n"
            "        noc_async_read(ve_g.get_noc_addr(0), s, 4);\n"
            "        noc_async_read_barrier();\n"
            "        invalidate_l1_cache();\n"
            "        valid_end = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s)[0];\n"
            "    }\n",
        ),
        (
            "                out[j] = static_cast<uint16_t>(ids[j]);\n",
            "                out[j] = static_cast<uint16_t>(t >= valid_end ? N_EXPERTS : ids[j]);\n",
        ),
        (
            "                    wout[j] = f32_to_bf16_rne(sv[j] * inv);\n",
            "                    wout[j] = t >= valid_end ? static_cast<uint16_t>(0) : f32_to_bf16_rne(sv[j] * inv);\n",
        ),
    )
    for old, new in subs:
        assert txt.count(old) == 1, ("GLM53_PMK_ROW_MASK: router_tail/writer.cpp changed", old)
        txt = txt.replace(old, new)
    return txt


def _src_file(name: str, text: str) -> str:
    """a kernel TEXT (7a's kernel_text: the header inlined) as a file the engine's composite can #include; the name
    carries the content hash (SOURCE_CODE composites are hashed by their text: an edited include must change it)"""
    h = hashlib.sha1(text.encode()).hexdigest()[:10]
    d = os.path.join(ASM_KDIR, "gen")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{name}_{h}.cpp")
    if not os.path.exists(p):
        with open(p, "w") as f:
            f.write(text)
    return p


def _host_shards(t, n: int) -> list:
    """the per-chip host copies of a mesh tensor (mesh order = rank order)"""
    ds = ttnn.get_device_tensors(t) if n > 1 else [t]
    assert len(ds) == n, (len(ds), n)
    return [ttnn.to_torch(d).float() for d in ds]


def adj_streamers(mesh, members: list, fallback: list, hosts: list, n: int = 24) -> list:
    """the proj piece's adj_pmk placement: the 24 bank-adjacent cores (mk_readers.reader_assignments, 3 per bank);
    one inside the member rectangle is swapped for the nearest free STREAMER HOST (a root / MM-only core: the only
    composites that carry the PfStream streamer part -- d0 first silicon 20:47Z: a swap onto a lane / tail core left
    its stream without a streamer, the TOKEN chain never advanced, 128 members + 42 streamers hung); the mock (no
    DRAM-adjacent query) takes the family's streamer list"""
    hosts_ = [tuple(h) for h in hosts]
    try:
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_readers import reader_assignments

        asg = reader_assignments(mesh, 3, "MESH_ROW_PLUS_X")
        cores = [((int(c.x), int(c.y)), int(b), int(i)) for c, b, i in asg]
    except Exception:  # noqa: BLE001
        return [tuple(c) for c in fallback][:n]
    # BANK-ADJACENT ORDER (PMK_A_INPLACE): PfStream serves stream st (DRAM bank st % 8) with streamer cores 2 st and
    # 2 st + 1, so the list goes two cores per bank FIRST (bank 0's readers 0 / 1, bank 1's, .. bank 7's), then every
    # bank's third reader: the in-place streams (8 = the banks) and the copied layout's first 8 streams read the bank
    # their core is adjacent to (ttnn's list is bank-major 3 per bank, which put stream st's lanes on banks 2 st / 3)
    cores.sort(key=lambda t: (t[2] >= 2, t[1], t[2]))
    free = [h for h in hosts_ if h not in {c for c, _, _ in cores}]
    out = []
    for c, _, _ in cores:
        if c not in hosts_:
            c = min(free, key=lambda f: abs(f[0] - c[0]) + abs(f[1] - c[1]))
            free.remove(c)
        out.append(c)
    out = out[:n]
    assert set(out) <= set(hosts_) and len(set(out)) == len(out), (out, hosts_)
    return out


class PmkProgramA:
    """program A of one sparse_kda layer as SCOPES programs on the family engine; `prog(x, ci)` launches them in
    order (the harness captures them in one trace)."""

    impl = "pmk:assembly-program-A(5 programs)"

    def __init__(self, pf, li: int, c: int, sp: bool, x, state, valid_end, *, probe: bool = False,
                 mhc_source=None, terms: str = "fp32", scopes=SCOPES, p3_phases=("qkv", "lowrank", "fb", "gb"),
                 mhc_probe: bool = False, pfs_park: int = PARK, lowrank_idle_streamers: bool = False,
                 tp8_resident: bool | None = None, a3_merge: bool | None = None, a_inplace: bool | None = None,
                 a2_heads: int | None = None, park2: int | None = None, park_align: bool | None = None,
                 reuse: "PmkProgramA | None" = None, attn: bool = True, a5_ag: bool | None = None,
                 a34_merge: bool | None = None, a2_fuse: bool | None = None, o_rects: int | None = None,
                 a15_fuse: bool | None = None, b_window: dict | None = None, b_p11: bool | None = None,
                 a15_p11: bool = False, b_p13: bool | None = None, b_p12: bool | None = None,
                 b_p15: bool | None = None, b_i1: bool | None = None, g_window: dict | None = None,
                 moe_share: "PmkProgramA | None" = None):  # fmt: skip
        """b_window (M-B, tt/pmk_b_window.py): dict(b=ProgramB(compute_only=members), x_ext, ids_rm, wt, x_tile, y_s) ->
        the fused program is A15 + B (A15B_SCOPE), B staged on the caller's inputs.
        x: the PERSISTENT residual [1, 1, SP, 16384] fp32 DRAM (read / written in place; its address is baked into
        the mhc args); state: the KDA FusedKDAState of one slot; valid_end: the chunk's valid-row tensor.
        tp8_resident (PMK_TP8_RESIDENT, default 1; TP8 only): the member's residual block X stays PINNED in L1 from A1
        to A5 (no reload before the P10 part; stored once after it). a3_merge (PMK_A3_MERGE, default 0): A3 inside A2.
        a2_heads (the TP8 A2 bisect on the pair, coordinator 15:45:26Z): the K = 512 phases f_b / g_b on the chip's
        heads 0..a2_heads-1 only (W = a2_heads x 128: 8 = TP8's per-chip f_b / g_b = ONE stream of 32 tiles on 2
        streamers, the lowrank's); their stores are their own tensors (stores()), so A1 / A2 scopes only.
        reuse (the stack driver, Stage 12 item L3): another layer's program A of the same geometry whose TRANSIENT
        hand-off tensors this one shares -- hns / posts / combs / ys (ys[0] = the shared RS object's y:
        the pair's RSR / TP8's RS8, with c1 and the PfStream sync), the stream phases' outputs u / lr / f / gate_out / o-partial and the
        kda scratch; layers run sequentially in one trace, so one set per kind suffices. Weights (streams, mhc fn / sb /
        w) and the KDA state stay per layer.
        attn=False (the mhc-only program A, SCOPES_MHC): A1 + A5 only -- the layer's attention runs as the op graph
        between them (sparse_dsa), so the residual block X is reloaded by A5 (never pinned across a foreign launch)."""
        self.reuse = reuse
        # cf3h-k2 v2 (coordinator 16:02:20Z, the k2 capture-phase DRAM low): another group's program whose MoE-half
        # per-launch transients this program takes when its own reuse chain has none (the first W7 layer <- the G owner)
        self.moe_share = moe_share
        self.attn = bool(attn)
        self.pf, self.li, self.c, self.sp, self.probe = pf, int(li), int(c), bool(sp), bool(probe)
        self.x, self.state, self.valid_end = x, state, valid_end
        n_dev = int(pf.mesh.get_num_devices())
        self.tp8_resident = bool(_knob("PMK_TP8_RESIDENT", 1) if tp8_resident is None else tp8_resident) and n_dev == 8
        self.a3_merge = bool(_knob("PMK_A3_MERGE", 0) if a3_merge is None else a3_merge)
        self.a_inplace = bool(_knob("PMK_A_INPLACE", 0) if a_inplace is None else a_inplace)
        self.a_fold = bool(_knob("PMK_A_FOLD", 0))
        # a34: the scan signals the members (SEM_ON) and the engine windows the {P4, P6, P7, P8} program; on whenever a
        # scope holds P6 and P7 together (a fused program without the signal would hang in P7's H4 wait)
        self.b_window = dict(b_window) if b_window else None
        # W7 step 1: P11 in the fused A15 + B program (GLM53_PMK_B_P11 / b_p11); B's inputs stay pre-staged until P11b
        self.b_p11 = self.b_window is not None and bool(_knob("PMK_B_P11", 0) if b_p11 is None else b_p11)
        self.a15_p11 = bool(a15_p11) and self.b_window is None  # the bisect: A15 + P11, no B
        # W7 step 2: P13 (the shared expert) in-program on top of step 1 (GLM53 knob PMK_B_P13 / b_p13); B's y_s = its out
        self.b_p13 = self.b_p11 and bool(_knob("PMK_B_P13", 0) if b_p13 is None else b_p13)
        # W7 step 3: P12 (the router) in-program on top of step 2; B's ids / weights = its tail's (PMK_IDS_U16 lever)
        self.b_p12 = self.b_p13 and bool(_knob("PMK_B_P12", 0) if b_p12 is None else b_p12)
        # W7 step 5 D3 (P11x, PMK_XEXT=1; coordinator 18:48:11Z / 19:11:41Z): the members store their x-hat_f block (XS)
        # as ROW_MAJOR rows of x_ext [C + 32, H] inside P12's phase (kernels/pmk/xext_*.cpp); program B then reads THAT
        # x_ext (lever x_l1 0) instead of the harness-staged copy -- the TP8 fallback for (I1) and the gang's x input
        self.xext_on = self.b_p12 and bool(_knob("PMK_XEXT", 0))
        self.xext = None
        # W7 step 4: P15 (the RS of B's y) in-program on top of step 3 (TP2 RSR first; TP8 RS8 later)
        self.b_p15 = self.b_p12 and bool(_knob("PMK_B_P15", 0) if b_p15 is None else b_p15)
        # W7 (I1) x once per token: program B gathers the routed rows from the holders' XS (lever 2, x_l1), the reclaim
        # keeps XS in place (TP8; the pair's member L1 cannot hold XS + B)
        # PMK_B_I1_PAIR=1 (debug, 2026-10-25 19:08Z): (I1) on the pair too -- w7tp8c's B y was wrong at TP8 (PCC 0.11-0.17
        # vs the chain; everything else bitwise): the pair runs the same holders' XS gather (64 members x 32 rows)
        self.b_i1 = (
            self.b_p11
            and (n_dev == 8 or bool(_knob("PMK_B_I1_PAIR", 0)))
            and bool(_knob("PMK_B_I1", 0) if b_i1 is None else b_i1)
        )
        # G v2a (g_window = dict(g=PmkGang(..., wt_bf16=True, ids_u16=True, ys=True))): a sparse_dsa layer's MoE half with
        # program G -- attn=False, P11 / P12 / P13 in-program, P11x feeding G's x_ext, no P15 (the RS stays the op), no (I1)
        self.g_window = dict(g_window) if g_window else None
        if self.g_window is not None:
            assert not attn and self.b_window is None, "G v2a: attn=False (a sparse_dsa layer), no program B"
            self.b_p11 = self.b_p13 = self.b_p12 = True
            self.b_p15 = bool(self.g_window.get("p15", False))  # G v2b: P15 = the RS of G's y in-program
            self.b_i1 = False
            self.xext_on = True
        self.a15_fuse = (
            bool(_knob("PMK_A15_FUSE", 0) if a15_fuse is None else a15_fuse)
            or any(set(A15_SCOPE) <= set(s_) for s_ in scopes)
            or self.b_window is not None
        )
        self.a2_fuse = (
            bool(_knob("PMK_A2_FUSE", 0) if a2_fuse is None else a2_fuse)
            or self.a15_fuse
            or any(set(A2F_SCOPE) <= set(s_) for s_ in scopes)
        )
        self.a34 = (
            bool(_knob("PMK_A34_MERGE", 0) if a34_merge is None else a34_merge)
            or self.a2_fuse
            or any({"P6", "P7"} <= set(s_) for s_ in scopes)
        )
        if not self.attn:
            self.tp8_resident = self.a3_merge = self.a_inplace = self.a34 = self.a2_fuse = self.a15_fuse = False
            if tuple(tuple(s_) for s_ in scopes) == SCOPES:
                scopes = SCOPES_MHC
                # GLM53_PMK_DSA_P11=0 (the DSA L3 race bisect, 2026-10-23 17:0xZ): A5 without P11 -- the fabric AG of hn[1],
                # which no consumer reads on a sparse_dsa layer (the op ffn takes the fp32 x-hat_f filler)
                if os.environ.get("GLM53_PMK_DSA_P11", "1") == "0":
                    scopes = (("P1",), ("P10",))
            if (
                self.g_window is not None
            ):  # G v2a: the MoE-half program only (A1 runs in its own program before the attention)
                scopes = (DSA_G15_SCOPE,) if self.b_p15 else (DSA_G_SCOPE,)
            assert all(set(s_) <= ({"P1", "P10", "P11"} | (set(DSA_G15_SCOPE) if self.g_window is not None else set()))
                       for s_ in scopes), f"attn=False: mhc programs only ({scopes})"  # fmt: skip
        # the H3 hand-off in-program (P3's stores -> the kda items in the same launch): the A3 merge and stage 2
        self.h3_inprog = self.a3_merge or self.a2_fuse
        # PMK_O_RECTS=4 (the overlap lever, coordinator 00:50:22Z (2)): o_proj as 4 row-group RECTANGLES (members
        # 16 h .. 16 h + 15 = rows 2 h / 2 h + 1 of the 8 x 8 grid = KDA chunks 16 h .. 16 h + 15), each streamed by its
        # own 2 ROOTS (the cores beside those rows: free during the scan) and started when ITS chunks are scanned
        # (PFS_WAIT_RT: member wait SEM_ON >= 16 (h + 1)) instead of after the whole scan. Fused programs only (a34).
        self.o_rects = int(_knob("PMK_O_RECTS", 0) if o_rects is None else o_rects) if self.a34 else 0
        assert self.o_rects in (0, 4), ("o_proj row groups: 4 rectangles x 2 roots", self.o_rects)
        self.root_final = os.environ.get("GLM53_PMK_ROOT_FINAL", "0") == "1"
        # GLM53_PMK_A_LR_EXACT (_build_streams): the low-rank / f_b / g_b streams in bf16 at HiFi4 -- read before _cbs:
        # their PARK (PARK2, 47) is then declared bf16 (the member's unpacker takes the CB's format)
        self.lr_exact = os.environ.get("GLM53_PMK_A_LR_EXACT", "0") == "1"
        # the link hand-over (SEM_LINK_REL): the fused A2F at TP8 holds C1X and RS8 stage 2 on the same axis-1 links
        self.link_handover = (
            (self.a2_fuse or (self.g_window is not None and self.b_p15))
            and n_dev == 8
            and os.environ.get("GLM53_PMK_LINK_HANDOVER", "1") == "1"
        )
        if self.a3_merge and tuple(scopes) == SCOPES:
            scopes = SCOPES_A3_MERGE
        elif self.attn and self.a_fold and tuple(tuple(s_) for s_ in scopes) == SCOPES:
            scopes = SCOPES_FOLD
        # (the twin merges: PMK_A_FOLD's 3-launch scopes and the fused A3 + A4 / A2F / A15 programs are exclusive)
        assert not (self.a_fold and (self.a34 or self.a2_fuse or self.a15_fuse)), \
            "PMK_A_FOLD with PMK_A34_MERGE / PMK_A2_FUSE / the A15 fusion"  # fmt: skip
        if self.a15_fuse and tuple(tuple(s_) for s_ in scopes) == SCOPES:
            # M-C (2026-10-24 05:31:53Z ruling): TP8 = RS8 + the link hand-over; H6 through the root-0 aggregator
            assert not self.a3_merge and n_dev in (2, 8), ("A15: the TP2 pair (M-A) or TP8 (M-C)", self.a3_merge, n_dev)
            scopes = (
                ((A15P11_SCOPE,) if self.a15_p11 else SCOPES_A15_FUSE)
                if self.b_window is None
                else (
                    (A15B15_SCOPE,)
                    if self.b_p15
                    else (A15B12_SCOPE,)
                    if self.b_p12
                    else (A15B13_SCOPE,)
                    if self.b_p13
                    else (A15B11_SCOPE,)
                    if self.b_p11
                    else (A15B_SCOPE,)
                )
            )
        elif self.a2_fuse and not self.a3_merge and tuple(tuple(s_) for s_ in scopes) == SCOPES:
            scopes = SCOPES_A2_FUSE
        elif self.a34 and not self.a3_merge and tuple(tuple(s_) for s_ in scopes) == SCOPES:
            scopes = SCOPES_A34_MERGE
        # GLM53_PMK_A5_P11=0 / a5_ag=False: A5 without P11 (the AG of hn[1] into the members' XS through the C1 lanes). In
        # the served chain nothing reads it: the router and program B take the fp32 x-hat_f filler gathered by an op.
        # The sparse_dsa L3 race (17:06-17:15Z, 10-replay probe): with P11 L3 drifted every replay (0.99912-0.99995);
        # without it L3 is bitwise-stable at 0.999952
        if (
            a5_ag is None
        ):  # DEFAULT OFF since the 17:21Z proof (10 replays bitwise on L0-L4 + logits); =1 re-enables P11
            a5_ag = os.environ.get("GLM53_PMK_A5_P11", "0") == "1" or self.b_p11 or self.a15_p11
        if not a5_ag:
            scopes = tuple(tuple(p_ for p_ in s_ if p_ != "P11") for s_ in scopes if any(p_ != "P11" for p_ in s_))
        self.scopes = [tuple(s) for s in scopes]
        # the in-program hand-offs the program boundaries used to give, per pair of phases sharing a program (A15 and the
        # bisect scopes): H1 (P1 -> P2), H6 (P8 -> P10), the mhc rounds cumulative + no PCOEF rebuild (P1 + P10)
        self.f_p1p2 = any({"P1", "P2"} <= set(s_) for s_ in self.scopes)
        self.f_p8p10 = any({"P8", "P10"} <= set(s_) for s_ in self.scopes)
        self.f_p1p10 = any({"P1", "P10"} <= set(s_) for s_ in self.scopes)
        self.a2_heads = int(a2_heads) if a2_heads else None
        assert self.a2_heads is None or all(set(s) <= {"P1", "P2", "P3"} for s in self.scopes), (
            "a2_heads narrows f / gate_out below the kda phase's store: A1 / A2 scopes only",
            self.scopes,
        )
        self.p3_phases = tuple(p3_phases)  # the bisect knob: which of P3's stream phases A2 runs (silicon localisation)
        self.mhc_probe = bool(mhc_probe)  # the mhc root pass's in-program taps (z / pass / iters per block): F_PROBE
        self.park = int(
            pfs_park
        )  # PfStream PARK depth (2..4): credit-ahead slots per member (L1: 68 KB per slot at kb 16 nb 4)
        if self.tp8_resident and self.park > PARK_TP8_RESIDENT:
            self.park = PARK_TP8_RESIDENT  # the pinned X (256 KB) sits in A2 too: PARK 4 is over the 1,330,000 ruling
        # PARK ALIGNMENT (Stage 14, the A3 TP8 defect 2026-10-23): pfs_stream.cpp lands block q at the PUBLISHED PARK
        # pointer (the publisher's write pointer at the phase start) + (q mod P) slots, q = QBASE + v launch-global,
        # while the members' ring takes block v at (blocks pushed through that CB earlier in the program + v) mod P:
        # the two agree only if every phase starts on slot 0 of its PARK CB and QBASE = 0 mod P (_qbase). At TP2 (P 4)
        # every block count is a multiple of 4 and they always did; the TP8 resident P 3 put f_b at QBASE 224 = 2 mod 3
        # (blocks read from the wrong / never-written slots: f = inf on all 8 chips, 04:09Z u-pmk-asm-a2-L4-tp8), g_b on
        # slot 1 (16 = 1 mod 3: the last slot's writes past the CB end) and o_proj at 256 = 1 mod 3. park_align (knob
        # PMK_PARK_ALIGN, default 1) aligns QBASE and asserts the slot-0 starts; the K = 512 phases keep their own PARK2
        # depth park2 (PMK_PARK2, default 4: 16 = 0 mod 4 leaves g_b on slot 0; 4 x 34,816 B fits the TP8 A2 census).
        # park_align 0 = the previous build (PARK2 at the PARK depth, QBASE unaligned) for the A/B.
        self.park_align = bool(_knob("PMK_PARK_ALIGN", 1) if park_align is None else park_align)
        # A15 + LR_EXACT at TP8 (twin v3): the bf16 PARK2 at depth 4 overfills the member L1 (2x4 mock: CB end
        # 1,375,296 > 1,330,000) -> depth 2 by default there (PMK_PARK2 still overrides)
        p2_def = 2 if (self.a15_fuse and self.lr_exact and n_dev == 8) else PARK
        self.park2 = int(_knob("PMK_PARK2", p2_def) if park2 is None else park2) if self.park_align else self.park
        assert 2 <= self.park2 <= 4, self.park2
        # the A3 merge puts the kda items' TRISC regime (HiFi4, fp32 DST, FULL sync) on the PfStream phases: NB 8 (the
        # full-sync DST holds 8 fp32 tiles: PFS_DST_FULL) at half the K block keeps the PARK block at 64 tiles
        # GLM53_PMK_A_FID (default FIDELITY = HiFi2): the projections' PfStream math fidelity, read per instance (a module
        # constant would keep the first job's value in a resident server) -- the A lever (3) row, coordinator 19:14:58Z
        fid_env = os.environ.get("GLM53_PMK_A_FID", FIDELITY)
        assert fid_env in ("LoFi", "HiFi2", "HiFi3", "HiFi4"), fid_env
        self.nb_, self.kb_, self.kb_small_, self.fidelity_ = (
            (8, 8, 4, "HiFi4") if self.a3_merge else (NB, KB, KB_SMALL, fid_env)
        )
        # lowrank's two streamers on idle NON-adjacent MM-only cores: its ring fills while the members run qkv (the adj
        # cores' BRISC runs qkv's stream first); sized ~0 of the layer (the members bound P3: proj README), a knob only
        self.lowrank_idle = bool(lowrank_idle_streamers)
        self.counters = dict(eager=0, calls=0)
        self.layer = PMK.PmkSparseKdaLayer(pf, li, c, sp, probe=probe, programs=[set(s) for s in self.scopes])
        self.engine = e = self.layer.engine
        if self.a34:  # the fused programs' phase windows (any program whose scope is exactly A34_SCOPE / A2F_SCOPE)
            e.window_plan = {frozenset(A34_SCOPE): [list(w_) for w_ in A34_WINDOWS],
                             frozenset(A2F_SCOPE): [list(w_) for w_ in A2F_WINDOWS],
                             frozenset(A15_SCOPE): [list(w_) for w_ in A15_WINDOWS],
                             frozenset(A15B_SCOPE): [list(w_) for w_ in A15B_WINDOWS],
                             frozenset(A15B11_SCOPE): [list(w_) for w_ in A15B11_WINDOWS],
                             frozenset(A15B13_SCOPE): [list(w_) for w_ in A15B13_WINDOWS],
                             frozenset(A15B12_SCOPE): [list(w_) for w_ in A15B12_WINDOWS],
                             frozenset(A15B15_SCOPE): [list(w_) for w_ in A15B15_WINDOWS],
                             frozenset(A15P11_SCOPE): [list(w_) for w_ in A15P11_WINDOWS],
                             frozenset(P1A2F_SCOPE): [["P1"]] + [list(w_) for w_ in A2F_WINDOWS],
                             frozenset(A2FP10_SCOPE): [list(w_) for w_ in A2F_WINDOWS] + [["P10"]]}  # fmt: skip
            # TP8 (2x4 mock 00:24Z): the MM-only union (scan + o_proj streamer rings + the RS8 reducers) ends at 1,438,848
            # -> the MM-only cores time-share too (the reducers fill their CBs from DRAM, no remote writer)
            e.window_roles = ("member", "mm_only") if n_dev == 8 else ("member",)
            # cf3h-k member shave (coordinator 11:23:49Z): W7's later windows place an alias group by the members THAT
            # window uses (window P2 P3 held the H group's 64 KB for OUT 42's 32 KB); pmk_layer WIN_TRIM_SCOPES
            e.WIN_TRIM_SCOPES = (frozenset(A15B15_SCOPE),)
            if n_dev == 8:  # M-C: the MM-only cores hold no P1 / P10 CB -> their own list P2 P3 | P4 P6 | P7 P8 (the
                # members keep the five windows of A15 / the bisect scopes)
                a2w = [list(w_) for w_ in A2F_WINDOWS]
                e.window_plan_role = {"mm_only": {frozenset(A15_SCOPE): a2w, frozenset(P1A2F_SCOPE): a2w,
                                                  frozenset(A2FP10_SCOPE): a2w}}  # fmt: skip
            if (
                self.a15_fuse and n_dev == 8 and (self.lr_exact or self.park2 < PARK)
            ):  # PARK2 2: member dm0 258 > 256 args
                assert (
                    os.environ.get("GLM53_PMK_ARGS", "") == "table"
                ), "A15 + LR_EXACT at TP8 needs GLM53_PMK_ARGS=table"
                e.arg_page = 2048
            if self.b_window is not None:  # M-B: the MM-only cores windowed at every TP (B's streamer rings), + P14;
                # the member BRISC composite carries A's + B's receiver args (~500): GLM53_PMK_ARGS=table with 2 KB rows
                # (cf3h-k) under LR_EXACT in place the member BRISC carries 828 words (the low-rank's NB 2 streams): 4 KB rows
                e.arg_page = (
                    (4096 if self.lr_exact else 3072) if self.b_p13 else 2048
                )  # W7 step 2: + P13's two PfStream parts -> 586 words on the
                # member BRISC (mock 15:39Z: '586 args > the 512-word table row'); CB_ARG 3 pages: +3 KB per table role
                if "mm_only" not in e.window_roles:
                    e.window_roles = tuple(e.window_roles) + ("mm_only",)
                wr = dict(e.window_plan_role or {})
                wr["mm_only"] = dict(wr.get("mm_only") or {})
                wr["mm_only"][frozenset(A15B_SCOPE)] = [list(w_) for w_ in A2F_WINDOWS] + [["P14"]]
                wr["mm_only"][frozenset(A15B11_SCOPE)] = [list(w_) for w_ in A2F_WINDOWS] + [
                    ["P14"]
                ]  # (P11: lanes + members)
                # P13's streamers reuse o_proj's ring on the MM-only cores: P13 joins the P7 P8 window there (its own
                # window would make ring 43 span the adjacent P7 P8 | P13 windows -> shared, and window 1 would lose its
                # only CB: 'window 1 has no CB of its own', mock 15:36Z)
                wr["mm_only"][frozenset(A15B13_SCOPE)] = [list(w_) for w_ in A2F_WINDOWS[:-1]] + [
                    list(A2F_WINDOWS[-1]) + ["P13"],
                    ["P14"],
                ]
                wr["mm_only"][frozenset(A15B12_SCOPE)] = [list(w_) for w_ in A2F_WINDOWS[:-1]] + [
                    list(A2F_WINDOWS[-1]) + ["P12", "P13"],
                    ["P14"],
                ]
                wr["mm_only"][frozenset(A15B15_SCOPE)] = wr["mm_only"][frozenset(A15B12_SCOPE)]
                if not list(getattr(self.b_window["b"], "streamers", []) or []):
                    # B7g (the self-stream form, b_inside_fused_plan.md I-1): B's compute cores include the ROOTS, whose
                    # composite is idle after P13 -> windowed as [every earlier phase] | [P14] (P14 = B's reclaim window;
                    # a root RS8 reducer's P15 CBs go into that reclaim like the MM-only ones)
                    if "root" not in e.window_roles:
                        e.window_roles = tuple(e.window_roles) + ("root",)
                    wr["root"] = dict(wr.get("root") or {})
                    for sc_ in (A15B_SCOPE, A15B11_SCOPE, A15B13_SCOPE, A15B12_SCOPE, A15B15_SCOPE):
                        wr["root"][frozenset(sc_)] = [[q for q in sc_ if q not in ("P14", "P15")], ["P14"]]
                e.window_plan_role = wr
            if self.a15_fuse or any(
                set(s_) in (set(P1A2F_SCOPE), set(A2FP10_SCOPE)) for s_ in scopes
            ):  # the alias groups X / X32 / XS / IN0, H / H16 / OUT 42, mhc OUT 11 and mhc Y 21 span the
                # ADJACENT windows P1 | P2 and P7 | P10 without carrying data across them (see PmkSparseKdaLayer.WIN_NO_CARRY)
                e.WIN_NO_CARRY = (PMK.CB_MHC_X, PMK.CB_MHC_X32, PMK.CB_XS_LAND, PMK.CB_IN0, PMK.CB_MHC_H, PMK.CB_MHC_H16,
                                  PMK.CB_OUT, MHC.CB["OUT"], MHC.CB["Y"])  # fmt: skip
            if (
                self.a15_fuse and os.environ.get("GLM53_PMK_A15_ROOT_WIN", "0") == "1"
            ):  # the roots' union (10 indices, no
                # collisions) is expected to fit unwindowed; =1 windows them like the members (the uniform GATHER / MSG /
                # PCOEF stay shared in place either way: the members address them by their own write pointer)
                e.window_roles = tuple(e.window_roles) + ("root",)
        if (
            self.g_window is not None
        ):  # G v2a: the member / MM-only windows of the MoE half (the A15B12 tail's), G = the reclaim
            # window; MM-only: P12 / P13's streamer rings, then G; the member BRISC carries P10..P13's + G's args -> table rows
            e.window_plan = {frozenset(sc_): [list(w_) for w_ in DSA_G_WINDOWS] for sc_ in (DSA_G_SCOPE, DSA_G15_SCOPE)}
            e.window_roles = ("member", "mm_only")
            e.window_plan_role = {
                "mm_only": {frozenset(sc_): [["P12", "P13"], ["P14"]] for sc_ in (DSA_G_SCOPE, DSA_G15_SCOPE)}
            }
            e.arg_page = 3072
            e.WIN_NO_CARRY = (PMK.CB_MHC_X, PMK.CB_MHC_X32, PMK.CB_XS_LAND, PMK.CB_IN0, PMK.CB_MHC_H, PMK.CB_MHC_H16,
                              PMK.CB_OUT, MHC.CB["OUT"], MHC.CB["Y"])  # fmt: skip
            e.WIN_CARRY_FIRST = (PMK.CB_XS_LAND,)  # XS carries P11's block into P12's / P13's windows: first in each
            e.wait_base_scope = (
                True  # the program's waits count only its own producers (H8 = P11's 8 slices, no P2 here)
            )
        self.geom = g = self.layer.geom
        mesh = pf.mesh
        self.mesh = mesh
        self.n = int(mesh.get_num_devices())
        self.R, self.C = int(mesh.shape[0]), int(mesh.shape[1])
        P = e.place
        self.members = [tuple(m) for m in P.members]
        self.roots = [tuple(r) for r in P.roots]
        self.mm_only = [tuple(m) for m in P.mm_only]
        self.streamers = adj_streamers(mesh, self.members, P.streamers, self.roots + self.mm_only)
        assert set(self.streamers) <= set(self.roots) | set(self.mm_only), "every streamer on a root / MM-only core"
        self.lanes = [tuple(l) for l in P.lanes]
        self.tail = [tuple(t) for t in P.tail]
        self.report = {}
        self.rs8_roles = {}
        if self.n == 8:
            self._place_rs8()
        # B7g (the self-stream B on the member / MM-only / root cores): P15's RS8 reducer CBs get their OWN reclaim window
        # after B's on every windowed reducer role (pmk_layer WIN_RECLAIM2: the region holds max(B, RS8), not the sum)
        self.b7g = self.b_window is not None and not list(getattr(self.b_window["b"], "streamers", []) or [])
        if (self.b7g or self.g_window is not None) and self.b_p15 and self.n == 8:  # (G v2b: G's reclaim, then P15's)
            wr = dict(e.window_plan_role or {})
            key = frozenset(DSA_G15_SCOPE if self.g_window is not None else A15B15_SCOPE)
            dflt = DSA_G_WINDOWS if self.g_window is not None else A15B15_WINDOWS
            for role in sorted(set(self.rs8_roles.get(1, [])) | set(self.rs8_roles.get(2, []))):
                if role not in e.window_roles:
                    continue
                wr[role] = dict(wr.get(role) or {})
                base_w = wr[role].get(key) or [list(w_) for w_ in dflt]
                if ["P15"] not in base_w:
                    wr[role][key] = [list(w_) for w_ in base_w] + [["P15"]]
            e.window_plan_role = wr
        self.noc = lambda xy: (lambda w: (int(w.x), int(w.y)))(mesh.worker_core_from_logical_core(ttnn.CoreCoord(*xy)))
        self.bl = g.sp_blocks
        self.pl = pf.layers[li]
        self.kda = getattr(self.pl, "kda", None)  # None on a sparse_dsa layer (attn=False)
        self.W = g.width_r
        if self.b_p13:
            # PARK2 (CB 47) is SHARED by the K = 512 phases (P3: f_b / g_b) and P13 down, and the engine lays it ONCE at
            # park2 x the larger block, so its users must share one block size (pf_stream fused_parts asserts the same):
            # the members' ring wraps at the CB end while the streamer lands block q at slot q mod park2 of ITS block. At
            # TP2 both blocks are 8 x NB; at TP8 (shared slice 256 -> kt_d 8) P13 down takes kb 4 and the K = 512 phases
            # kept kb 8: P13 down read 16-tile blocks out of a 32-tile-slot ring (w7tp8 2026-10-25 18:09Z: y_s, B's y and
            # P15 NaN on all 8 chips). The K = 512 phases (kt 16) therefore take P13 down's K block. PMK_PARK2_ONE_BLOCK
            # = 0: the previous build (the pair A/B of the root cause).
            # (cf3h-k, 2026-10-26) GLM53_PMK_A_LR_EXACT declares PARK2 bf16 (the low-rank / f_b / g_b streams; in place the
            # low-rank lands 32-tile blocks), and P13 down is bf8 with 16-tile blocks at TP8 (kt 8 -> kb 4: a 32-tile block
            # would be ONE block per stream, odd per lane) -- it cannot share CB 47 ('one (format, page) per index'): it
            # takes its OWN bf8 PARK index (laid only in P13's window) and the K = 512 phases keep their block (no override).
            # PMK_P13_OWN_PARK (default = LR_EXACT) / PMK_PARK2_ONE_BLOCK (the shared form) stay knobs for the A/B
            # PMK_P13_DN: "share" (bf8, the K = 512 phases take P13 down's block: the pre-LR_EXACT form), "own" (bf8 on its
            # own index: costs the member index program B's window needs under the served profile -> 'no free CB index
            # for WD'), "bf16x8" (default under LR_EXACT: P13 down's copy in bf16 -- lossless, the device weights are
            # exact in bf16 -- at NB 8 x KB 4 = PARK2's 32-tile bf16 block, on CB 47 itself: no new index; NB 8 asks full
            # DST sync, which W7's member composite already runs for the KDA items)
            # (G v2a, attn=False: no low-rank / f_b / g_b on CB 47 -> LR_EXACT does not touch it: the bf8 shared form)
            self.p13_dn = os.environ.get("PMK_P13_DN", "bf16x8" if (self.lr_exact and self.attn) else "share")
            assert self.p13_dn in ("share", "own", "bf16x8"), self.p13_dn
            self.p13_own_park = self.p13_dn == "own"
            self.park2_one_block = self.p13_dn == "share" and bool(_knob("PMK_PARK2_ONE_BLOCK", 1))
            if self.park2_one_block:
                kb_d = p13_kb_down(p13_slice(int(g.shared_r)) // PMK.TILE)
                assert (512 // PMK.TILE) % kb_d == 0 and ((512 // PMK.TILE) // kb_d) % 2 == 0, kb_d
                self.kb_small_ = kb_d
        self._handoffs_and_sems()
        self._cbs()
        self._build_mhc(mhc_source)
        self._build_collectives()
        if self.attn:
            self._build_streams()
            self._build_kda(terms)
        elif self.g_window is not None:  # G v2a: the MoE half's PfStream phases only (P12 router, P13 shared)
            self._build_moe_streams()
            self.scr, self.kphase = {}, None
        else:
            self.mms3, self.mm_o, self.sync, self.scr, self.kphase = [], None, None, {}, None
        # PMK_P15_STREAM=1 (non-B lever (2)): P15 streamed behind the combine -- B's combine raises each finished row
        # block's P15 readiness word (H9 TABLE) and P15 reads a block as soon as its h_split units landed. Decided BEFORE
        # _register(): the pair's P15 registers there (TP8's after B) -- set after it, P15 kept the rendezvous (ready=False,
        # the members' after-B waits) while B raised the table: the 00:44-01:17Z TP2 arms never streamed P15
        b_streamers = self.b_window is not None and bool(list(getattr(self.b_window["b"], "streamers", []) or []))
        self.p15_stream = bool(
            self.b_p15
            and (self.b_window is not None or self.g_window is not None)
            and _knob("PMK_P15_STREAM", 0)
            and (
                (self.n == 2 and (b_streamers or self.g_window is not None))
                or (self.n == 8 and self.rs8_15 is not None)
            )
        )
        if (
            self.p15_stream and self.b_window is not None
        ):  # (review 02:57Z) only comb_writer raises H9: the combine-split
            # BRISC rebuild, B's bisect stages and the no-body knob leave every readiness word at 0 -> P15 would hang
            b_ = self.b_window["b"]
            assert not getattr(
                getattr(b_, "lv", None), "comb_split", False
            ), "PMK_P15_STREAM with PMK_COMB_SPLIT: no H9 raise"
            assert (
                int(getattr(b_, "bw_stage", -1)) < 0 and os.environ.get("GLM53_PMK_B_STAGE", "") == ""
            ), "P15 stream: B bisect"
            assert (
                os.environ.get("GLM53_PMK_B_NOBODY", "0") != "1"
            ), "PMK_P15_STREAM with GLM53_PMK_B_NOBODY: no combine"
        self._register()
        if self.g_window is not None:  # G v2a: program G as P14, the reclaim window (tt/pmk_b_window.register_g)
            g_ = self.g_window["g"]
            assert g_.wt_bf16 and g_.ids_u16 and g_.ys, "G v2a: G reads P12's tail ids / weights and P13's y_s"
            h9_of = self._p15_h9 if self.n == 2 else self._p15_h9_rs8
            h9_kw = {"h9_by_rank": h9_of(g_.scratch(self.c)["y"])} if self.p15_stream else {}
            self.report["g_window"] = PBW.register_g(self, g_, self.c, x_ext=self.xext, ids_rm=self.rt_ids, wt=self.rt_wt,
                                                     y_s=self.sh_dn.y, **h9_kw)  # fmt: skip
            # the peer lanes' P11 packets land in the holders' XS, which G's reclaim window lays out over: every member waits
            # H8 (every slice of its block landed) before enter_reclaim (program B's w7s1w lesson)
            self.engine.reclaim_drain = {"member": ("H8",)}
            self.report["wait_schedule"] = self.engine.wait_schedule()
            self._p13_carry_check()
            if (
                self.b_p15 and self.n == 8
            ):  # G v2b at TP8: the second RS8 after G (its reducers' reclaim CBs follow G's)
                self._register_p15_tp8(self.layer, g_.scratch(self.c)["y"])
        if self.b_window is not None:  # M-B: program B as P14, the reclaim window (tt/pmk_b_window.py)
            bw = self.b_window
            y_s = self.sh_dn.y if self.b_p13 else bw.get("y_s")  # W7 step 2: B's PMK_SHARED term = P13's y_s rows
            feed = self.b_p12 and _knob("PMK_P12_FEED_B", 1) and _knob("PMK_P12_TAIL", 1)  # (=0: the cost split A/B)
            ids_rm, wt = (self.rt_ids, self.rt_wt) if feed else (bw["ids_rm"], bw["wt"])  # W7 step 3: the tail's
            if self.b_p12:
                assert bw[
                    "b"
                ].lv.ids_u16, "W7 step 3: program B reads the tail's [C, 8] uint16 ids / bf16 weights (ids_u16)"
            xs_kw = {}
            if self.b_i1:  # W7 (I1): no XS fill (P11 landed it); the XS address is patched once the layout is known
                assert bw["b"].lv.x_l1, "W7 (I1): program B with lever 2 (x_l1)"
                assert list(bw["b"].xs_holders(self.c)) == list(
                    self.members[: self.c // TILE]
                ), "B's holder G = member G"
                self.engine.reclaim_protect = {"member": (PMK.CB_XS_LAND,)}
                xs_kw = dict(x_tile=None, xs_base=XS_SENTINEL)
            if self.xext_on:  # D3: B gathers its rows from the in-program x_ext (lever x_l1 0)
                assert not self.b_i1 and not bw["b"].lv.x_l1, "PMK_XEXT: program B reads x_ext (x_l1 0, no (I1))"
            reg_b = (
                PBW.register_b if list(getattr(bw["b"], "streamers", []) or []) else PBW.register_b_self
            )  # B7g: no streamers
            assert (
                self.p15_stream is False or reg_b is PBW.register_b or self.n == 8
            ), "P15 stream at TP2: the streamer B"
            h9_of = self._p15_h9 if self.n == 2 else self._p15_h9_rs8
            h9_kw = {"h9_by_rank": h9_of(bw["b"].scratch(self.c)["y"])} if self.p15_stream else {}
            self.report["b_window"] = reg_b(self, bw["b"], self.c, x_ext=self.xext if self.xext_on else bw["x_ext"], ids_rm=ids_rm,
                                            wt=wt, y_s=y_s, **({"x_tile": bw.get("x_tile")} | xs_kw), **h9_kw)  # fmt: skip
            self._b_fab_shift()
            if self.b_p11 and _knob(
                "PMK_RECLAIM_DRAIN", 1
            ):  # W7 step 1: the peer lanes' P11 packets land in the holders' XS 35, which B's reclaim window
                # lays out over (B's CBs 49 / 59): every member waits H8 (FS_ARRIVE base 8 + 8: every slice of its block
                # landed, local or remote) before enter_reclaim (w7s1w 14:48Z: device 1's members hung inside B)
                self.engine.reclaim_drain = {"member": ("H8",)}
                self.report["wait_schedule"] = self.engine.wait_schedule()
            if self.b_p13:
                self._p13_carry_check()
            if self.b_p15 and self.n == 8:
                self._register_p15_tp8(self.layer, bw["b"].scratch(self.c)["y"])
            if self.b_i1:
                self._i1_patch_xs_base()
        self.handoff = ProgramAHandoff(self)
        self.report["knobs"] = dict(tp8_resident=self.tp8_resident, a3_merge=self.a3_merge, a34=self.a34, root_final=self.root_final,
                                    o_rects=self.o_rects, a_inplace=self.a_inplace, park=self.park, nb=self.nb_,
                                    kb=self.kb_, fidelity=self.fidelity_, scopes=[list(s) for s in self.scopes])  # fmt: skip

    def _b_fab_shift(self):
        """mb8 (M-B1 at TP8): program B's tail BRISC part (P14: B's streamers on tail cores) follows P8's sub-parts in the
        tail composite, and connect() appends RS8 stage 2's connection args after the WHOLE composite -> every fab_at
        slot of P8's tail BRISC sub-parts moves by B's part width (2x4 mock 19:33Z: 'fab_at 54 != composite width')"""
        e = self.engine
        if self.rs8 is None or not e.in_scope("P8"):
            return
        p14 = next(q for q in e.phases if q.pid == "P14").body
        bt = (p14.parts or {}).get(("tail", "dm0")) if p14 is not None else None
        if bt is None:
            return
        width = lambda part: max(
            len(v)
            for rt in (part["rt_by_rank"].values() if "rt_by_rank" in part else [part["rt"]])
            for v in rt.values()
        )  # noqa: E731
        wb = sum(width(x) for x in (bt if isinstance(bt, list) else [bt]))
        subs = next(q for q in e.phases if q.pid == "P8").body.parts.get(("tail", "dm0"))
        for sub in subs if isinstance(subs, list) else [subs]:
            fa = sub.get("fab_at_arg")
            if fa is None:
                continue
            seen = set()  # once per list object (ranks may share them)
            for d in sub["rt_by_rank"].values() if "rt_by_rank" in sub else [sub["rt"]]:
                for v in d.values():
                    if id(v) not in seen:
                        seen.add(id(v))
                        v[fa] += wb
        self.report["b_fab_shift"] = wb

    # ------------------------------------------------------------------------------------------ the family map
    def _handoffs_and_sems(self):
        e, g = self.engine, self.geom
        h1, h7 = e.handoffs["H1"], e.handoffs["H7"]
        for h in (h1, h7):  # H1 / H7: DRAM x-hat rows -> the holders' SEM_HN (8 writer signals per holder per use)
            h.counter, h.expected = SEM_HN, 8
            h.lives_in = "DRAM hn[s] (mhc F_XH_DRAM) -> c1_stub_prod reads the holder's block into XS 35"
            h.unit = "writer signal (pmk_sig after the mhc part: 8 per holder; the AG of hn[0] is in another program)"
        p7 = next(p for p in e.phases if p.pid == "P7")
        p10 = next(p for p in e.phases if p.pid == "P10")
        p7.signals = tuple(s for s in p7.signals if s != "H7b")  # no FN prefetch in this cut: no IN0-free gate
        p10.waits = tuple(w for w in p10.waits if w != "H7b")
        if self.f_p8p10 and self.n == 2:  # A15 at TP2: H6 in-program = every RSR lane +1 each member's FS_REDUCED after
            # ITS y rows landed (pmk_sig after rs_send; the y stays in DRAM) -> P10 waits for all lanes
            h6 = e.handoffs["H6"]
            h6.counter, h6.expected = PMK.SEM_FS_REDUCED, len(self.lanes)
            h6.unit = "lane signal (pmk_sig after the lane's RSR y writes; y in DRAM) on every member and root"
        elif self.f_p8p10:  # A15 at TP8 (M-C): the 8 RS8 stage-2 reducers +1 root 0's FS_REDUCED after their red_write2
            # (pmk_sig; y in DRAM); root 0's NCRISC (pmk_agg) waits 8, then SETs every member's (one multicast over the
            # 8 x 8 rectangle) and every root's copy to the H6 wait value -> P10 waits ONE unit
            h6 = e.handoffs["H6"]
            h6.counter, h6.expected = PMK.SEM_FS_REDUCED, 1
            h6.unit = (
                "aggregator SET (root 0 after the 8 RS8 stage-2 reducers' pmk_sig; y in DRAM) on every member and root"
            )
        e._assign_wait_bases()
        if not self.f_p1p10:
            # the counters are per PROGRAM (semaphores start at 0 at every launch): the second uses of 4 / 12 are the FIRST
            # in A5 (H7 after P10 -> P11, H0f after P10 -> P11)
            e.handoffs["H7"].wait_base = 0
            e.handoffs["H0f"].wait_base = 0
        # P1 + P10 in one launch: the layer's cumulative bases stand (H1 then H7 on SEM_HN, H0 then H0f)
        if self.f_p8p10:
            e._sem(PMK.SEM_FS_REDUCED, "member")
            e._sem(PMK.SEM_FS_REDUCED, "root")  # P10's wait roles are member + root (the mhc root part waits H6 too)
        e._sem(SEM_HN, "member")
        e._sem(PMK.SEM_PARK_READY, "member")  # PfStream VALID1 (11) on the members (VALID0 = 10 is there already)
        if self.h3_inprog:  # the in-program H3 barrier: the members +1 root 0's SEM_QKV, root 0 SETs the members' copy
            e._sem(INT.KDA.SEM_QKV, "root")
        if self.root_final:  # the roots' FINAL rendezvous counter (A1 / A5)
            e._sem(SEM_ROOT_FINAL, "root")
        if self.a34:  # H4 in-program: the scan leader raises SEM_ON on every member once per chunk (kda_scan_w.cpp)
            e._sem(INT.KDA.SEM_ON, "member")
        if self.link_handover:
            e._sem(SEM_LINK_REL, "tail")
            if self.n == 8 and (self.b_p11 or self.a15_p11):  # W7 at TP8: the tails hand link h back to the lanes
                e._sem(SEM_LINK_ACQ, "lane")
        if self.n == 8:  # TP8: P8 = FabricStreamRS8 on the lanes + 2 tail cores + 16 MM-only reducers (no H5 counter
            # wait: the RS kernels poll the producers' readiness WORDS themselves; the H6 y lands in DRAM for A5)
            p8 = next(p for p in e.phases if p.pid == "P8")
            p8.roles = tuple(["lane", "tail"] + sorted({r for rs in self.rs8_roles.values() for r in rs} - {"tail"}))
            if self.f_p8p10 and "root" not in p8.roles:  # M-C: the H6 aggregator on root 0's NCRISC
                p8.roles = p8.roles + ("root",)
            p8.waits = ()

    def _place_rs8(self):
        """the TP8 RS cores: lanes2 = tail[0:2]; the 16 reducers from the cores idle in A4 (its program = o_proj +
        RS): the MM-only cores without a PfStream streamer first, then the other 2 tail cores, then the roots without a
        streamer (the device's bank-adjacent assignment puts 17-19 of the 24 streamers on MM-only cores, so 13-15 of
        them are free: d0 pair 20:47Z / the 2x4 mock). A reducer kernel is emitted on every role that hosts one of its
        cores (gated per core); the stage's CBs are declared on those roles."""
        st = set(self.streamers)
        self.lanes2 = self.tail[:2]
        pool = [c for c in self.mm_only if c not in st] + self.tail[2:] + [r for r in self.roots if r not in st]
        if _knob(
            "PMK_RS8_POOL_TAILS_FIRST", 0
        ):  # (G v2b P15 tile defect A/B, 2026-10-26) the tail cores own stage-1 blocks
            # 0 / 1 instead of (on silicon, 13-15 free MM-only cores) stage-2 blocks 6 / 7: does the defect follow them?
            pool = self.tail[2:] + [c for c in pool if c not in set(self.tail[2:])]
        assert len(pool) >= 16, (len(pool), "RS8 needs 16 idle cores in A4 (MM-only / tail / roots without a streamer)")
        self.red1, self.red2 = pool[:8], pool[8:16]
        role_of = (
            lambda c: "mm_only" if c in set(self.mm_only) else "tail" if c in set(self.tail) else "root"
        )  # noqa: E731
        self.rs8_role = {c: role_of(c) for c in self.red1 + self.red2}
        self.rs8_roles = {1: sorted({role_of(c) for c in self.red1}), 2: sorted({role_of(c) for c in self.red2})}
        self.report["rs8"] = dict(lanes1=self.lanes, reducers1=self.red1, lanes2=self.lanes2, reducers2=self.red2,
                                  roles=self.rs8_roles)  # fmt: skip

    def _free_member_cb(self) -> int:
        """a member CB index no declared CB and no fixed index constant (PMK / this module / the mhc / the router tail)
        takes: the highest below the window barriers (63 / 62 / 61); later free-index users (B / G reclaims, RS8 P15) see it
        as used, and a fixed-index clash raises in _use ('one (format, page) per index')"""
        import sys as _sys  # noqa: PLC0415

        fixed = {
            v
            for m_ in (PMK, _sys.modules[__name__])
            for k, v in vars(m_).items()
            if k.startswith("CB_") and isinstance(v, int)
        }
        fixed |= {int(v) for v in MHC.CB.values()} | {int(v) for v in P12_TAIL_CB.values()}
        used = set(self.engine.cbs.get("member", {}))
        for i in range(60, 15, -1):
            if i not in used and i not in fixed:
                return i
        raise AssertionError(("no free member CB index", sorted(used)))

    def _cbs(self):
        """the family's placeholder CBs replaced by the pieces' real ones; the alias group X <-> XS (exact 256 KB
        both); the kda item map re-drawn after the reservations (it takes the first free member indices)"""
        e, g = self.engine, self.geom
        bt, bt2, tb = self.kb_ * self.nb_, self.kb_small_ * self.nb_, PMK.TILE_BYTES["bf8"]
        mm_live = ("P3", "P7")
        mhc_live = ("P1", "P10", "P16")
        for role in ("member", "mm_only"):
            d = e.cbs[role]
            for i in [i for i, u in d.items() if u.note.startswith("kda ")]:
                d.pop(i)
        for role in ("member", "root", "mm_only"):
            d = e.cbs[role]
            for i in (PMK.CB_PARK, PMK.CB_PART):
                d.pop(i, None)
            if role != "member":
                for i in (PMK.CB_IN0, PMK.CB_OUT):
                    d.pop(i, None)
        # X (2 / 3): live in its mhc parts only (TP2 streamed; TP8 loaded / stored at the program boundaries by this
        # assembly), so it may share the XS allocation
        for i in (PMK.CB_MHC_X, PMK.CB_MHC_X32):
            u = e.cbs["member"][i]
            e.cbs["member"][i] = dataclasses.replace(u, live=mhc_live, users=mhc_live)
        e.ALIASES = ((PMK.CB_MHC_X, PMK.CB_MHC_X32, PMK.CB_XS_LAND, PMK.CB_IN0), (PMK.CB_MHC_W, PMK.CB_PART),
                     (PMK.CB_MHC_H, PMK.CB_MHC_H16, PMK.CB_OUT))  # fmt: skip
        # the mhc's "all"-role CBs are addressed across cores by each core's own write pointer: same address on members
        # and roots = allocated first, in index order, same size (d0 first silicon 21:20Z: the members' X group sat
        # before GATHER 0 while the roots' did not -> the (M, S) partials landed in the wrong root CB -> garbage z)
        for name in ("GATHER", "MSG", "PCOEF"):
            idx = MHC.CB[name]
            for role in ("member", "root"):
                u = e.cbs[role][idx]
                e.cbs[role][idx] = dataclasses.replace(u, uniform=True, solo=True)
                um, ur = e.cbs["member"][idx], e.cbs["root"][idx]
            assert (um.nbytes, um.fmt, um.page_bytes) == (ur.nbytes, ur.fmt, ur.page_bytes), (name, um, ur)
        if self.tp8_resident:
            # PINNED X / X32: one allocation, FIRST on the members in EVERY program (A1 loads it, A5 attaches + stores it,
            # A2..A4 leave the range alone); the roots carry a same-size placeholder at the same index so the UNIFORM
            # mhc CBs (GATHER / MSG / PCOEF) that follow stay at one address on both roles
            all_p = tuple(p.pid for p in e.phases)
            for i in (PMK.CB_MHC_X, PMK.CB_MHC_X32):
                u = e.cbs["member"][i]
                e.cbs["member"][i] = dataclasses.replace(u, live=all_p, users=all_p, solo=True, pinned=True)
            assert PMK.CB_MHC_X not in e.cbs["root"], e.cbs["root"].get(PMK.CB_MHC_X)
            e._use("root", PMK.CB_MHC_X, "fp32", MHC.X_TILES, all_p, note="PINNED placeholder (aligns the uniform CBs with the members)",
                   users=all_p, solo=True)  # fmt: skip
            e.cbs["root"][PMK.CB_MHC_X] = dataclasses.replace(e.cbs["root"][PMK.CB_MHC_X], pinned=True)
            e.PINNED = ((PMK.CB_MHC_X, PMK.CB_MHC_X32),)
        e._use(
            "member",
            PMK.CB_PARK,
            "bf8",
            self.park * bt,
            mm_live,
            note=f"PfStream PARK {self.park} x {bt} bf8 tiles",
            users=mm_live,
        )
        # GLM53_PMK_A_LR_EXACT: low-rank / f_b / g_b stream bf16 tiles (the attention half's; G v2a, attn=False, has none:
        # its only PARK2 user is P13 down, bf8 -- the w7lv5 form)
        p2_dt = "bf16" if (self.lr_exact and self.attn) else "bf8"
        e._use("member", CB_PARK2, p2_dt, self.park2 * bt2, ("P3",), note=f"PfStream PARK (K = 512 phases{' + the low-rank, bf16' if self.lr_exact else ''}) {self.park2} x {bt2}",
               users=("P3",), solo=True)  # fmt: skip
        e._use("member", CB_XK, "bf16", 512 // TILE, ("P3",), note="PfStream in0 of the K = 512 phases (f_b / g_b)",
               users=("P3",), solo=True)  # fmt: skip
        e._use("member", CB_OUT32, "fp32", 2 * self.nb_, ("P3",), note="PfStream fp32 OUT (f / gate_out stores)", users=("P3",),
               solo=True)  # fmt: skip
        st = e.cbs.get("streamer", {})
        for i in [i for i in st if i >= PMK.CB_RING0]:
            st.pop(i)
        e._use("streamer", PMK.CB_RING0, "bf8", PF.RING * bt, mm_live, note=f"PfStream ring {PF.RING} x {bt} bf8 tiles",
               users=mm_live)  # fmt: skip
        ln = e.cbs.get("lane", {})
        for i in list(ln):
            ln.pop(i)
        coll_live = ("P2", "P8", "P11") + (("P15",) if self.b_p15 else ())
        e._use(
            "lane", 0, "bf16", 4 * 16, coll_live, note="FabricStream STAGE (C1 / RS send)", users=coll_live, solo=True
        )
        e._use("lane", CB_HDR, "u32", 1, coll_live, note="fabric send DATA packet headers (private; the pool holds 12 per RISC)",
               users=coll_live, solo=True)  # fmt: skip
        if self.n == 2:  # the pair's lanes-reduce RSR (+ P15: the FFN partial's RS, W7 step 4)
            rsl = ("P8",) + (("P15",) if self.b_p15 else ())
            e._use("lane", 1, "bf16", 2 * 16, rsl, note="RS reduce A", users=rsl, solo=True)
            e._use("lane", 2, "bf16", 2 * 16, rsl, note="RS reduce B", users=rsl, solo=True)
            e._use("lane", 3, "fp32", 2 * 16, rsl, note="RS reduce Y (fp32)", users=rsl, solo=True)
        elif self.n == 8:  # RS8: stage-1 reducers (bf16 A / B, fp32 Y) and stage-2 reducers (4 fp32 source slots, Y)
            # on every role that hosts one (every core of the role gets the set; the gated parts use it on 8 + 8
            # cores), the stage-2 lanes' fp32 send ring on the tail role
            I = RS8_CB
            # r1 / the stage-2 source slots / SEND (GLM53_PMK_RS8_S2_BF16; _cbs runs before the RS8 object exists:
            # _build_collectives asserts that the object, possibly a shared owner's, matches)
            self.s2_bf16 = _knob("GLM53_PMK_RS8_S2_BF16", 0) == 1
            s2f = "bf16" if self.s2_bf16 else "fp32"
            r8w = ("P8",) + (("P15",) if self.b_p15 else ())  # W7 step 4 at TP8: the second RS8 on the same cores
            r8_of = lambda role: ("P8",) if role in e.window_roles else r8w  # noqa: E731  (windowed: reclaim allocs)
            for role in self.rs8_roles[1]:
                r8 = r8_of(role)
                e._use(role, I["A"], "bf16", 2 * 16, r8, note="RS8 s1 reduce A", users=r8, solo=True)
                e._use(role, I["B"], "bf16", 2 * 16, r8, note="RS8 s1 reduce B", users=r8, solo=True)
                e._use(role, I["Y1"], s2f, 2 * 16, r8, note=f"RS8 s1 reduce Y ({s2f} r1)", users=r8, solo=True)
            for role in self.rs8_roles[2]:
                r8 = r8_of(role)
                for j in range(4):
                    e._use(
                        role,
                        I["T0"] + j,
                        s2f,
                        2 * 8,
                        r8,
                        note=f"RS8 s2 source slot {j}",
                        users=r8,
                        solo=True,
                    )
                e._use(role, I["Y2"], "fp32", 2 * 8, r8, note="RS8 s2 reduce Y (fp32 y)", users=r8, solo=True)
            r8 = r8w
            e._use("tail", I["SEND2"], s2f, 2 * 8, r8, note=f"RS8 s2 lane SEND ({s2f})", users=r8, solo=True)
            e._use(
                "tail",
                CB_HDR,
                "u32",
                1,
                r8,
                note="fabric send DATA packet headers (private)",
                users=r8,
                solo=True,
            )
        if self.b_p12:  # W7 step 3: P12's CBs = idle member indices of the matching format (P12's own window)
            p12 = ("P12",)
            if self.xext_on:  # P11x's ROW_MAJOR staging (2 x BW tile-sized pages), P12's window
                e._use("member", CB_XEXT, "bf16", 2 * XEXT_BW, p12, note="P11x x_ext untilize rows (2 x BW pages)",
                       users=p12, solo=True)  # fmt: skip
            dt12 = str(self.pl.moe.router.weights[320].dtype).split(".")[-1]
            pk = {"FLOAT32": (P12_PARK_FP32, "fp32"), "BFLOAT16": (CB_XK, "bf16"), "BFLOAT8_B": (PMK.CB_PARK, "bf8")}[
                dt12
            ]
            e._use("member", pk[0], pk[1], 2 * 8 * 2, p12, note="P12 logits PARK (park 2 x kb 8 x nb 2)", users=p12,
                   solo=pk[0] != PMK.CB_PARK)  # fmt: skip
            e._use("member", CB_OUT32, "fp32", 2 * 2, p12, note="P12 logits OUT (fp32)", users=p12, solo=True)
            for nm_, (idx_, fmt_) in dict(LOG=(14, "fp32"), BIAS=(15, "fp32"), CHOICE=(16, "fp32"), IDX=(17, "fp32"),
                                          SCORES=(18, "bf16")).items():  # fmt: skip
                assert P12_TAIL_CB[nm_] == idx_
                e._use("member", idx_, fmt_, 1, p12, note=f"P12 router tail {nm_}", users=p12, solo=True)
            for idx in (PMK.CB_XS_LAND,):  # the logits phase's in0 = the x-hat_f block
                u = e.cbs["member"][idx]
                e.cbs["member"][idx] = dataclasses.replace(u, live=tuple(sorted(set(u.live) | set(p12))),
                                                           users=tuple(sorted(set(u.users or u.live) | set(p12))))  # fmt: skip
            e._use("streamer", PMK.CB_RING0, "bf8", PF.RING * bt, p12, note="P12 PfStream ring", users=p12)
        if self.b_p13:  # W7 step 2: P13's PfStream CBs = A's indices, re-laid in P13's own window (member 64 / 64)
            p13 = ("P13",)
            kt_d = p13_slice(int(g.shared_r)) // PMK.TILE
            kb_d = p13_kb_down(kt_d)
            e._use("member", PMK.CB_PARK, "bf8", self.park * bt, p13, note="P13 gate_up PARK", users=p13)
            if getattr(self, "p13_dn", "share") == "bf16x8":  # PARK2's own format and block (see __init__)
                kb_d = bt2 // P13_DN_NB  # PARK2's block / NB 8 (TP8: kt 8 -> 2 K-blocks; TP2: kt 32 -> 8)
                assert self.lr_exact and kb_d * P13_DN_NB == bt2 and kt_d % (2 * kb_d) == 0, (kt_d, kb_d, bt2, "bf16x8")
                self.p13_dn_kb = kb_d
                self.cb_p13_park = CB_PARK2
                e._use("member", CB_PARK2, "bf16", self.park2 * kb_d * P13_DN_NB, p13, note="P13 down PARK (bf16 x NB 8 on PARK2)",
                       users=p13, solo=True)  # fmt: skip
            elif getattr(self, "p13_own_park", False):  # its own bf8 index (PARK2 is bf16 under LR_EXACT): see __init__
                self.cb_p13_park = self._free_member_cb()
                e._use("member", self.cb_p13_park, "bf8", self.park2 * kb_d * self.nb_, p13,
                       note="P13 down PARK (own index: PARK2 is bf16 under LR_EXACT)", users=p13, solo=True)  # fmt: skip
            else:
                assert not self.park2_one_block or kb_d * self.nb_ == bt2, (
                    kb_d,
                    self.nb_,
                    bt2,
                    "PARK2 shared: one block size",
                )
                self.cb_p13_park = CB_PARK2
                e._use(
                    "member",
                    CB_PARK2,
                    "bf8",
                    self.park2 * kb_d * self.nb_,
                    p13,
                    note="P13 down PARK",
                    users=p13,
                    solo=True,
                )
            # ACT: gate_up's SwiGLU output = down's stationary in0 (kt_d bf16 tiles: 32 at TP2 > P3's 16 -> +32 KB there)
            e._use("member", CB_XK, "bf16", max(512 // TILE, kt_d), p13, note="P13 ACT (SwiGLU -> down in0)", users=p13,
                   solo=True)  # fmt: skip
            assert PF.CB_X == PMK.CB_XS_LAND and PF.CB_OUT == PMK.CB_OUT, (
                PF.CB_X,
                PMK.CB_XS_LAND,
                PF.CB_OUT,
                PMK.CB_OUT,
            )
            for idx in (
                PMK.CB_XS_LAND,
                PMK.CB_OUT,
            ):  # gate_up's in0 = the x-hat_f block (P11's XS 35); down's OUT rows 42
                u = e.cbs["member"][idx]
                e.cbs["member"][idx] = dataclasses.replace(u, live=tuple(sorted(set(u.live) | set(p13))),
                                                           users=tuple(sorted(set(u.users or u.live) | set(p13))))  # fmt: skip
            e._use("streamer", PMK.CB_RING0, "bf8", PF.RING * bt, p13, note="P13 PfStream ring", users=p13)
        e._kda_cbs()
        e._liveness_check()
        e.census = e._census()
        e._policy()

    # ------------------------------------------------------------------------------------------------ pieces
    def _build_mhc(self, source):
        from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, hf_config
        from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource

        mesh, g, L = self.mesh, self.geom, self.pl.L
        cfg = hf_config()
        source = source or FdTileCacheSource(CACHE_DIR, MANIFEST)
        SP = g.sp_rows
        self.mp = mp = MHC.Plan(sp_rows=SP, y_lane=False, probe=self.mhc_probe)
        self.taps = None
        if self.mhc_probe:  # the root's taps per (row block, sublayer): z planes, the pass, iters 1, iters 2
            self.taps = ttnn.zeros((1, 1, mp.n_blocks * mp.nsub * MHC.N_TAPS * TILE, TILE), ttnn.float32, ttnn.TILE_LAYOUT, mesh,
                                   ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
            self.engine._use("root", MHC.CB["DBG"], "fp32", MHC.N_TAPS, ("P1", "P10", "P16"), note="mhc DBG taps",
                             users=("P1", "P10", "P16"), solo=True)  # fmt: skip
        subs = [MHC.SubWeights.from_source(mesh, source, cfg, L, w, NORM[w]) for w in ("attn", "ffn")]
        self.mhc = MHC.PmkMhc(mesh, subs, mp, hc_eps=cfg.hc_eps, x_eps=cfg.rms_norm_eps, h_eps=cfg.rms_norm_eps,
                              iters=cfg.hc_sinkhorn_iters, placement=INT.family_mhc_placement(self.layer))  # fmt: skip
        empty = lambda shape, dt: ttnn.empty(shape, dt, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)  # noqa: E731
        if self.reuse is not None:  # the shared transient set (same geometry: asserted)
            r = self.reuse
            assert (r.geom.sp_rows, r.geom.c_rows, r.mp.nsub) == (SP, g.c_rows, mp.nsub), "reuse: geometry differs"
            self.hns, self.posts, self.combs = r.hns, r.posts, r.combs
            if self.taps is None:
                self.taps = r.taps if (self.mhc_probe and r.taps is not None) else None
        else:
            self.hns = [empty((1, 1, SP, MHC.HIDDEN), ttnn.bfloat16) for _ in range(2)]
            self.posts = [empty((1, 1, SP, MHC.N_STREAMS), ttnn.float32) for _ in range(2)]
            self.combs = [empty((1, 1, SP, MHC.N_STREAMS * MHC.N_STREAMS), ttnn.float32) for _ in range(2)]
        self.mhc_flags = MHC.F_XH_DRAM | MHC.F_Y_DRAM | MHC.F_PC_DRAM | MHC.F_INIT
        self.mhc_flags |= (MHC.F_X_LOAD | MHC.F_X_STORE) if mp.resident else 0
        self.mhc_flags |= MHC.F_PROBE if self.mhc_probe else 0

    def _build_collectives(self):
        mesh, bl = self.mesh, self.bl
        cls = FS.FabricStreamC1X if self.n == 8 else FS.FabricStreamC1
        r = self.reuse
        # the collectives' L1 sync shards (one address on EVERY core) are shared with the reused layer: per layer they
        # pushed the lowest L1 buffer down ~4.6 KB per layer until program B's static CBs clashed (14b 45-layer pair
        # mock r1: 'L1 buffer allocated at 1363328 and static circular buffer region ends at 1477248' at L~20). Their
        # protocol state is the same across consecutive launches of one program (replays) and across A2 / A5 (P2 / P11
        # already share c1) -- layers of one kind run the same programs in sequence
        self.c1 = r.c1 if r is not None else cls(mesh, bl, mm_cores=self.members, lanes=self.lanes)
        self.rsr = self.rs8 = None
        if r is not None and self.attn and (r.rsr is not None or r.rs8 is not None):
            self.rsr, self.rs8 = r.rsr, r.rs8
        elif self.n == 2 and self.attn:  # the pair's RS in the lanes-reduce form (no reducer cores)
            self.rsr = FS.FabricStreamRSR(mesh, bl, lanes=self.lanes, hidden=MHC.HIDDEN, nb=16)
        elif (
            self.n == 8 and self.attn
        ):  # the 2x4 RS: stage 1 (axis 0, the board pairs) on the family lanes + 8 reducers, stage 2
            # (axis 1, the row) on 2 tail cores + 8 reducers (placement: _place_rs8)
            # GLM53_PMK_RS8_S2_BF16=1 (rs8tl 2026-10-25 11:45Z, coordinator 11:47:25Z): stage 2 (the row RS, the RS
            # tail) moves r1 as bf16 tiles -- the exact fp32 pair sums rounded once to bf16, the owner adds in fp32
            self.rs8 = FS.FabricStreamRS8(mesh, lanes1=self.lanes, reducers1=self.red1, lanes2=self.lanes2,
                                          reducers2=self.red2, hidden=MHC.HIDDEN, nb1=16, nb2=8,
                                          s2_bf16=getattr(self, "s2_bf16", False),
                                          order=os.environ.get("GLM53_PMK_RS8_ORDER", ""))  # fmt: skip
        if self.rs8 is not None:  # a shared owner's RS8 must carry this program's CB formats (_cbs)
            assert self.rs8.s2_bf16 == getattr(self, "s2_bf16", False), ("GLM53_PMK_RS8_S2_BF16 vs the shared RS8",
                                                                         self.rs8.s2_bf16)  # fmt: skip
        # (after the chain above: an if / elif inserted inside it skipped the TP8 RS8 -> P8 stayed a stub, mock 17:3xZ)
        self.rsr15 = self.rs8_15 = None
        if (
            r is not None
            and self.b_p15
            and (getattr(r, "rsr15", None) is not None or getattr(r, "rs8_15", None) is not None)
        ):
            # cf3h-k (14b, the W7 stack): consecutive W7 layers share P15's RS object as they share P8's -- its L1 sync
            # shard (one address on every core) per layer would lower the L1 floor ~4.6 KB per layer (see c1 above)
            self.rsr15, self.rs8_15 = r.rsr15, r.rs8_15
        elif (
            self.b_p15 and self.n == 2
        ):  # W7 step 4: the FFN partial's RS (own sync shard / landing r / fp32 y = FFN rows)
            self.rsr15 = FS.FabricStreamRSR(mesh, bl, lanes=self.lanes, hidden=MHC.HIDDEN, nb=16)
        elif (
            self.b_p15
        ):  # TP8: a second RS8 on P8's cores (lanes + reducers1, tails + reducers2), its own state / buffers
            self.rs8_15 = FS.FabricStreamRS8(mesh, lanes1=self.lanes, reducers1=self.red1, lanes2=self.lanes2,
                                             reducers2=self.red2, hidden=MHC.HIDDEN, nb1=16, nb2=8,
                                             s2_bf16=getattr(self, "s2_bf16", False),
                                             order=os.environ.get("GLM53_PMK_RS8_ORDER", ""))  # fmt: skip
        own = self.rsr.y if self.rsr is not None else self.rs8.y if self.rs8 is not None else None
        if (
            r is not None
        ):  # ys[0] is the RS landing: the RS object's own y (pair RSR / TP8 RS8: per layer), else shared zeros
            self.ys = [own if own is not None else r.ys[0], r.ys[1]]
        else:
            self.ys = [own if own is not None else self._zeros((1, 1, self.geom.sp_rows, MHC.HIDDEN), ttnn.float32),
                       self._zeros((1, 1, self.geom.sp_rows, MHC.HIDDEN), ttnn.float32)]  # fmt: skip

    def _zeros(self, shape, dt):
        return ttnn.zeros(shape, dt, ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG)

    def _host_stream_weights(self, kda, n: int, W: int, g, K: int) -> tuple:
        """the per-chip host weights (the TP shards) of the five stream phases: qkv, low-rank, f_b, g_b, o_proj"""
        wq = [w.reshape(K, -1)[:, : 3 * W].contiguous() for w in _host_shards(kda.w_qkv, n)]
        lr = [w.reshape(K, -1) for w in _host_shards(kda.w_lowrank, n)]
        lr_n = 2 * g.head_dim + g.heads_r
        assert all(w.shape[1] >= lr_n for w in lr), ([w.shape for w in lr], lr_n)
        wl = []
        for w in lr:  # [f_a | g_a | b | 0] padded to 512 columns (the item reader's LRNT = 16, BTILE = 8)
            z = torch.zeros(K, 512)
            z[:, :lr_n] = w[:, :lr_n]
            wl.append(z)
        Wk = self.a2_heads * g.head_dim if self.a2_heads else W  # the K = 512 phases' width (a2_heads: TP8's per chip)
        assert Wk <= W, (Wk, W)
        fb = [w.reshape(g.head_dim, -1)[:, :Wk] for w in _host_shards(kda.w_fb, n)]
        gb = [w.reshape(g.head_dim, -1)[:, :Wk] for w in _host_shards(kda.w_gb, n)]
        wfb, wgb = [], []
        for f_, g_ in zip(fb, gb):  # K = 512 (the low-rank output row): W_fb at rows 0..127, W_gb at rows 128..255
            zf, zg = torch.zeros(512, Wk), torch.zeros(512, Wk)
            zf[: g.head_dim] = f_
            zg[g.head_dim : 2 * g.head_dim] = g_
            wfb.append(zf)
            wgb.append(zg)
        wo = [w.reshape(W, -1)[:, : g.hidden].contiguous() for w in _host_shards(kda.w_o, n)]
        return wq, wl, wfb, wgb, wo

    def _fbgb_streams(self, mk, wfb, wgb, kb_small: int, fbg_kw: dict, wof) -> tuple:
        """the f_b / g_b PfStream phases. GLM53_PMK_FBGB_JOINT=1 (default; the coordinator's 22:46:05Z DRAM reclaim): ONE
        joint copy of f_b | g_b at NW = FBGB_NW (8) -- 2 NST streams, every bank whole streams (TP8: 8 streams, one per bank;
        TP2: 16, two per bank) -> no bank padding. The v1 copies (=0) were one stream each at TP8 (N = 32 tiles, NW 32):
        8 x their bytes, 1.0 MiB/bank per copy per layer of which 0.125 is data (the served census: 66 MiB/bank over 33
        layers). g_b's streams follow f_b's in the joint layout; its streamers rotate so stream j sits on bank-adjacent
        cores (BANK-ADJACENT ORDER: two per bank first). The output is the same per tile (the K order is unchanged)."""
        kw = dict(
            kb=kb_small,
            cb_x=CB_XK,
            cb_out=CB_OUT32,
            cb_park=CB_PARK2,
            park=self.park2,
            out_dtype=ttnn.float32,
            **fbg_kw,
        )
        src = wof("mm_fb").get("weight_of")
        if (
            src is not None and getattr(src, "joint", None) is None
        ):  # the owner's v1 copies (a GroupStack chunk instance)
            return mk(wfb, **kw, **wof("mm_fb")), mk(wgb, **kw, **wof("mm_gb"))
        if src is None and os.environ.get("GLM53_PMK_FBGB_JOINT", "1") != "1":
            return mk(wfb, **kw), mk(wgb, **kw)
        from models.autoports.zai_org_glm_5_3_flash.tt import mm_stream as MS  # noqa: PLC0415

        if src is not None:  # a chunk instance of a joint owner: the same joint copy, the same streamer rotation
            sj = src.joint
            nst = int(sj["lay"]["cores"]) // 2
            s16 = list(self.streamers[:16])
            rot = (2 * nst) % len(s16)
            fb = mk(wfb, nw=int(sj["lay"]["nt"]), joint=dict(sj, base=0), **kw)
            gb = mk(
                wgb,
                nw=int(sj["lay"]["nt"]),
                joint=dict(sj, base=nst),
                streamers=s16[rot:] + s16[:rot] + list(self.streamers[16:]),
                **kw,
            )
            return fb, gb

        k_, n_ = int(wfb[0].shape[-2]), int(wfb[0].shape[-1])
        # the FEWEST streams per phase that still fill the 8 banks in the joint copy (2 NST >= 8): NW = min(v1's NW, N / 4
        # tiles) -- TP2 (N 128 tiles) keeps v1's per-phase geometry (NW 32, 4 streams: the pair A/B read +0.4..2.0 ms with
        # 8 streams per phase, the longer streamer TOKEN chain), TP8 (N 32 tiles) goes NW 32 -> 8 (1 -> 4 streams)
        nw = min(NW, max(FBGB_NW, 2 * self.nb_, (n_ // TILE) // 4))
        assert (n_ // TILE) % nw == 0 and (nw // self.nb_) % 2 == 0, (n_, nw, self.nb_, "even blocks per lane")
        nst = (n_ // TILE) // nw
        joint = MS.stream_layout(k_, 2 * n_, 2 * nst)
        w_dtype = kw.get("w_dtype", ttnn.bfloat8_b)
        st = torch.stack([MS.stream_tiles(torch.cat([f_.float(), g_.float()], dim=1), joint).reshape(1, -1, TILE)
                          for f_, g_ in zip(wfb, wgb)], dim=0)  # fmt: skip
        if w_dtype == ttnn.bfloat16 and torch.equal(st.to(torch.bfloat16).float(), st):
            st = st.to(torch.bfloat16)  # exact: a bf16 host tensor (pf_stream's own upload rule)
        jt = ttnn.from_torch(st, dtype=w_dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                             mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0))  # fmt: skip
        self.fbgb_joint = dict(weight=jt, lay=joint, bytes=int(joint["pages_per_bank"]) * PF.BANKS * PMK.TILE_BYTES[
            "bf16" if w_dtype == ttnn.bfloat16 else "bf8"], nw=nw, nst=nst)  # fmt: skip
        s16 = list(self.streamers[:16])
        rot = (2 * nst) % len(s16)
        s_gb = s16[rot:] + s16[:rot] + list(self.streamers[16:])
        fb = mk(wfb, nw=nw, joint=dict(weight=jt, lay=joint, base=0), **kw)
        gb = mk(wgb, nw=nw, joint=dict(weight=jt, lay=joint, base=nst), streamers=s_gb, **kw)
        fb.weight_bytes = self.fbgb_joint["bytes"]  # the joint copy, reported once (on f_b)
        return fb, gb

    def _build_streams(self):
        """the PfStream phases (proj's tt/pf_stream.py, rt 1): per-chip host weights (the TP shards) in the
        mk-stream-v1 layout"""
        kda, n, W, g = self.kda, self.n, self.W, self.geom
        K = g.hidden
        # the host references of A2's stores (the unit's store-vs-reference checks, coordinator 15:45:26Z): per chip.
        # ONLY for the units (probe=True or GLM53_PMK_KEEP_HOST_W=1): fp32 host copies are ~480 MiB per KDA layer at TP8,
        # ~16 GB over a served stack's 34 layers (the 45-layer group mock's 40 GB memcg OOM, 2026-10-23 22:26Z)
        keep = self.probe or os.environ.get("GLM53_PMK_KEEP_HOST_W", "0") == "1"
        r = self.reuse
        # the GroupStack's chunk instances of ONE layer (reuse = that layer's p0) also share p0's stream WEIGHT copies (the
        # outputs are shared with any reused layer below; the weights only within a layer)
        wsrc = r if (r is not None and int(r.li) == int(self.li) and r.a_inplace == self.a_inplace) else None
        if wsrc is not None and not keep:
            # a shared copy (PfStreamMM weight_of) reads only w's shape: shape-only stand-ins, no device readback. The
            # 45-layer group re-read every chunk instance's weights (15 x 5 per layer: ~4 min per layer on the g45h mock)
            meta = lambda nm: [torch.empty((getattr(wsrc, nm).k, getattr(wsrc, nm).n), device="meta")] * n  # noqa: E731
            wq, wl, wfb, wgb, wo = meta("mm_qkv"), meta("mm_lr"), meta("mm_fb"), meta("mm_gb"), meta("mm_o")
        else:
            wq, wl, wfb, wgb, wo = self._host_stream_weights(kda, n, W, g, K)
        self.host_w = dict(qkv=wq, lowrank=wl, fb=wfb, gb=wgb) if keep else None
        fid = getattr(ttnn.MathFidelity, self.fidelity_)
        NB_, KB_, KB_SMALL_ = self.nb_, self.kb_, self.kb_small_
        idle = [c for c in self.mm_only if c not in set(self.streamers)][:2] if self.lowrank_idle else []
        self.lr_streamers = idle if len(idle) == 2 else self.streamers
        sync_cores = self.streamers + [c for c in self.lr_streamers if c not in self.streamers]
        if self.o_rects:  # o_proj's root streamers hold their own sync shard
            sync_cores += [c for c in self.roots if c not in sync_cores]
        # the streamers' L1 sync shards shared with the reused layer (same cores, same slots)
        need_slots = 8 if self.b_p12 else 7 if self.b_p13 else 5
        # (cf3h-k, 14b) the shard is shared only when it HAS this program's slots: a W7 layer reusing an A15 owner's
        # transient set (5 slots) must not stream P12 / P13 through slots past the shard
        share_sync = (r is not None and r.sync is not None and set(r.sync_cores) == set(sync_cores)
                      and int(getattr(r, "sync_slots", 5)) >= need_slots)  # fmt: skip
        self.sync = r.sync if share_sync else PF.PfStreamMM.make_sync(self.mesh, sync_cores, slots=need_slots)
        self.sync_slots = int(getattr(r, "sync_slots", 5)) if share_sync else need_slots  # one shard per streamer core
        self.sync_cores = list(sync_cores)
        wof = lambda nm: {"weight_of": getattr(wsrc, nm)} if wsrc is not None else {}  # noqa: E731
        mk = lambda w, **kw: PF.PfStreamMM(
            self.mesh,
            w,
            members=self.members,
            streamers=kw.pop("streamers", self.streamers),
            nb=kw.pop("nb", NB_),
            nw=kw.pop("nw", NW),
            kb=kw.pop("kb", KB_),
            park=kw.pop("park", self.park),
            fidelity=kw.pop("fidelity", fid),
            rt=1,
            probe=self.probe,
            sync=self.sync,
            **kw,
        )  # noqa: E731
        # GLM53_PMK_A_LR_EXACT=1 (the real-weight 32K drift, coordinator 2026-10-24 05:05:19Z; the PMK vs op audit): the
        # KDA low-rank (f_a | g_a | b), f_b and g_b projections as the OP computes them -- bf16 weights at HiFi4 with fp32
        # accumulation (prefill_tp.py pf_mm / ds_fb.ckc / ds_gb.ckc; pmk_kda.py's own notes) instead of bfp8 stream copies at
        # the program fidelity (HiFi2). The decay g and beta path compounds through the recurrent state chunk to chunk.
        # bf16 tiles are 2 KiB: the low-rank's K-block 16 -> 8 so its ring still fits CB 43 (f_b / g_b already run KB 8)
        lrx = self.lr_exact
        # the low-rank moves onto PARK2 (bf16, its K-block 8 = PARK2's block of 32 tiles); PARK 29 stays bfp8 (qkv / o_proj)
        lr_kw = dict(w_dtype=ttnn.bfloat16, fidelity=ttnn.MathFidelity.HiFi4, kb=KB_SMALL_, cb_park=CB_PARK2,
                     park=self.park2) if lrx else {}  # fmt: skip
        fbg_kw = dict(w_dtype=ttnn.bfloat16, fidelity=ttnn.MathFidelity.HiFi4) if lrx else {}
        self.report["lr_exact"] = lrx
        if self.a_inplace:
            ds = {k_: getattr(kda, k_, None) for k_ in ("ds_qkv", "ds_lowrank", "ds_o")}
            assert all(v is not None for v in ds.values()), ("PMK_A_INPLACE needs the DSLinears (ds_decode)", ds)
            for k_ in ("ds_qkv", "ds_o"):
                assert ds[k_].weight.dtype == ttnn.bfloat8_b, (k_, ds[k_].weight.dtype, "the bf8 PARK 29")
            shard_w = lambda t: (int(t.shape[-1]) // TILE) // 8  # noqa: E731
            assert ds["ds_qkv"].n_pad == 3 * W and ds["ds_o"].n_pad == g.hidden, (ds["ds_qkv"].n_pad, ds["ds_o"].n_pad)
            mkr = lambda t, **kw: mk(None, resident=t, nw=shard_w(t), **kw)  # noqa: E731
            self.mm_qkv = mkr(ds["ds_qkv"].weight)
            self.mm_o = mkr(ds["ds_o"].weight, **self._o_rect_kw())
            # lowrank / f_b / g_b stay today's bf8 stream copies at the unchanged geometry (pair cell 01:43Z,
            # 4a933521b8: in place the lowrank's shard width 2 forces NB 2 = twice the blocks (member span 63 -> 147
            # us) and the bf16 copies are unpack-bound (f_b 254 -> 322, g_b 261 -> 310): +330 us on 9,681; qkv / o in
            # place were neutral-to-better, -26 us on qkv from the bank-adjacent streamer order)
            if lrx:
                # LR_EXACT in place (the v1.1 mock's DRAM OOM, coordinator 06:05:02Z): the bf16 lowrank copy is ONE stream
                # (2,048 pages in one bank, every bank padded to it: 32 MiB per layer per chip) -> stream the image's own
                # bf16 ds_lowrank (the op's weight itself, 0 copy) at 4a933521b8's geometry (shard 2 -> NB 2, KB 16 =
                # PARK2's 32-tile block). f_b / g_b stay 247de8ed9b7's bf16 copies at the v1 geometry (8 MiB each at TP8):
                # 16 MiB per layer per chip vs v1's 25.5 MiB of bf8 copies
                assert ds["ds_lowrank"].weight.dtype == ttnn.bfloat16 and ds["ds_lowrank"].n_pad == 512, \
                    (ds["ds_lowrank"].weight.dtype, ds["ds_lowrank"].n_pad)  # fmt: skip
                self.mm_lr = mkr(ds["ds_lowrank"].weight, nb=LR_NB_INPLACE, kb=LR_KB_INPLACE, w_dtype=ttnn.bfloat16,
                                 fidelity=ttnn.MathFidelity.HiFi4, cb_park=CB_PARK2, park=self.park2)  # fmt: skip
            else:
                self.mm_lr = mk(wl, nw=16, streamers=self.lr_streamers, **lr_kw, **wof("mm_lr"))
            self.mm_fb, self.mm_gb = self._fbgb_streams(mk, wfb, wgb, KB_SMALL_, fbg_kw, wof)
            assert not lrx or (self.mm_lr.bt == self.mm_fb.bt and self.mm_lr.tb == self.mm_fb.tb), \
                (self.mm_lr.bt, self.mm_fb.bt, self.mm_lr.tb, self.mm_fb.tb, "one PARK2 slot size")  # fmt: skip
        else:
            self.mm_qkv = mk(wq, **wof("mm_qkv"))
            self.mm_lr = mk(wl, nw=16, streamers=self.lr_streamers, **lr_kw, **wof("mm_lr"))
            self.mm_fb, self.mm_gb = self._fbgb_streams(mk, wfb, wgb, KB_SMALL_, fbg_kw, wof)
            self.mm_o = mk(wo, **self._o_rect_kw(), **wof("mm_o"))
        ring_bytes = PF.RING * KB * NB * PMK.TILE_BYTES["bf8"]  # the streamers' ring region (CB 43, declared bf8)
        for m_ in (self.mm_qkv, self.mm_lr, self.mm_fb, self.mm_gb, self.mm_o):
            assert PF.RING * m_.bt * m_.tb <= ring_bytes, (m_.k, m_.n, m_.bt, m_.tb, "a block ring must fit CB 43")
        names = ("qkv", "lowrank", "f_b", "g_b", "o_proj")
        copies = {
            nm: m_.weight_bytes for nm, m_ in zip(names, (self.mm_qkv, self.mm_lr, self.mm_fb, self.mm_gb, self.mm_o))
        }
        self.report["dram_weights"] = dict(inplace=self.a_inplace, copy_bytes=copies, copy_total=sum(copies.values()),
                                           dtypes={nm: str(m_.w_dtype).split(".")[-1] for nm, m_ in
                                                   zip(names, (self.mm_qkv, self.mm_lr, self.mm_fb, self.mm_gb, self.mm_o))})  # fmt: skip
        self.mms3 = [self.mm_qkv, self.mm_lr, self.mm_fb, self.mm_gb]
        self.sh_gu = self.sh_dn = None
        if self.b_p13:
            self._build_p13(mk, ring_bytes)
        self.rt_mm = None
        if self.b_p12:
            self._build_p12(ring_bytes)
        if (
            self.reuse is not None
        ):  # the stream outputs (u / lr / f / gate_out / the o partial) shared with the reused layer
            for mm, rm in zip(self.mms3 + [self.mm_o], self.reuse.mms3 + [self.reuse.mm_o]):
                assert tuple(mm.y.shape) == tuple(rm.y.shape) and mm.y.dtype == rm.y.dtype, (mm.y.shape, rm.y.shape)
                ttnn.deallocate(mm.y)
                mm.y = rm.y
        self._share_moe_scratch()
        self.report["streams"] = {name: dict(k=m.k, n=m.n, nst=m.nst, streamers=m.ns, blocks=m.blocks, footprint=m.footprint())
                                  for name, m in zip(("qkv", "lowrank", "f_b", "g_b", "o_proj"), self.mms3 + [self.mm_o])}  # fmt: skip
        self.report["streamers"] = self.streamers
        self.report["lowrank_streamers"] = self.lr_streamers
        self.report["park"] = self.park
        self.report["p3_phases"] = list(self.p3_phases)

    def _build_moe_streams(self):
        """G v2a (attn=False, a sparse_dsa layer): _build_streams without A's streams -- the streamers' sync shard (P12's
        slot 7, P13's 5 / 6), the PfStream factory, then P13 (the shared expert) and P12 (the router)"""
        fid = getattr(ttnn.MathFidelity, self.fidelity_)
        NB_, KB_ = self.nb_, self.kb_
        self.lr_streamers = list(self.streamers)
        self.sync_cores = list(self.streamers)
        r = (
            self.reuse
        )  # cf3h-k (14b): consecutive G layers share the streamers' sync shard (one L1 address on every core)
        if (
            r is not None
            and getattr(r, "g_window", None) is not None
            and r.sync is not None
            and set(r.sync_cores) == set(self.sync_cores)
        ):
            self.sync = r.sync
        else:
            self.sync = PF.PfStreamMM.make_sync(self.mesh, self.sync_cores, slots=8)
        self.sync_slots = 8
        mk = lambda w, **kw: PF.PfStreamMM(  # noqa: E731
            self.mesh, w, members=self.members, streamers=kw.pop("streamers", self.streamers), nb=kw.pop("nb", NB_),
            nw=kw.pop("nw", NW), kb=kw.pop("kb", KB_), park=kw.pop("park", self.park), fidelity=fid, rt=1,
            probe=self.probe, sync=self.sync, **kw)  # fmt: skip
        ring_bytes = PF.RING * KB * NB * PMK.TILE_BYTES["bf8"]
        self.mms3, self.mm_o = [], None
        self.host_w = None
        self.qbase_end = 0  # no stream phase before P12 in this launch (_register_p12 continues from here)
        self.sh_gu = self.sh_dn = None
        self._build_p13(mk, ring_bytes)
        self.rt_mm = None
        self._build_p12(ring_bytes)
        self._share_moe_scratch()
        self.report["streamers"] = self.streamers

    def _share_moe_scratch(self):
        """cf3h-k (14b, the W7 / G stack's DRAM, 14's review B2): the MoE half's per-launch scratch -- P12's router logits
        (rt_mm.y), the tail's ids / weights, P11x's xext, P13's gate_up / down outputs -- is written before it is read in
        every launch, so consecutive W7 (or G) layers share the reused layer's (same shapes); per layer stay the weights
        and the router bias. The 45-layer w7g mock ran out of DRAM at the 31st W7 layer without it (xext 4.26 MB/bank)
        """
        r = self.reuse
        if (
            r is None or getattr(r, "rt_mm", None) is None
        ):  # (v2) no MoE half up the reuse chain: the other group's owner
            r = getattr(self, "moe_share", None)
        if r is None:
            return
        shared = []
        pairs = [("rt_ids", self, r), ("rt_wt", self, r), ("xext", self, r)]
        for nm_, a_, b_ in pairs:
            t_, u_ = getattr(a_, nm_, None), getattr(b_, nm_, None)
            if (
                t_ is not None
                and u_ is not None
                and t_ is not u_
                and tuple(t_.shape) == tuple(u_.shape)
                and t_.dtype == u_.dtype
            ):
                ttnn.deallocate(t_)
                setattr(a_, nm_, u_)
                shared.append(nm_)
        for nm_ in ("rt_mm", "sh_gu", "sh_dn"):
            m_, rm_ = getattr(self, nm_, None), getattr(r, nm_, None)
            if (
                m_ is not None
                and rm_ is not None
                and getattr(m_, "y", None) is not None
                and getattr(rm_, "y", None) is not None
                and m_.y is not rm_.y
                and tuple(m_.y.shape) == tuple(rm_.y.shape)
                and m_.y.dtype == rm_.y.dtype
            ):
                ttnn.deallocate(m_.y)
                m_.y = rm_.y
                shared.append(f"{nm_}.y")
        self.report["moe_scratch_shared"] = shared

    def _build_kda(self, terms: str):
        # H3 is a DRAM hand-off between programs: the items start with SEM_QKV at its full count (h3_preraised); no
        # P7 consumers to signal (H4 = the `on` store in DRAM, read by the next program)
        # merged (A3 in A2): SEM_QKV starts at 0 and the P3 barrier parts SET it to NCH after every member's stores
        cons = [ttnn.CoreCoord(int(x_), int(y_)) for x_, y_ in self.members] if self.a34 else None
        self.kphase = INT.kda_phase(self.layer, self.kda, consumers=cons, nprod=1, h3_preraised=not self.h3_inprog)
        if (
            self.a34
        ):  # P7's H4 wait in the fused program: the leader's +1 per chunk (dead chunks included) on each member
            h4 = self.engine.handoffs["H4"]
            h4.expected, h4.wait_base = int(self.kphase.geo.n_chunks), 0
            h4.unit = "chunk (the scan leader's aggregate, kda_scan_w.cpp chunk_done)"
            if self.o_rects:  # the member readers wait per rectangle (PFS_WAIT_RT): no composite-wide H4 wait
                p7 = next(p for p in self.engine.phases if p.pid == "P7")
                p7.waits = tuple(w for w in p7.waits if w != "H4")
        assert self.kphase.geo.terms_dtype == terms, (self.kphase.geo.terms_dtype, terms)
        if (
            self.reuse is not None
        ):  # the kda scratch (terms, on, ...) shared; u / lr / f / gate_out = the shared stream ys
            scr = dict(self.reuse.scr)
        else:
            scr = self.kphase.scratch()
            own = (
                ("u", "lr") if self.a2_heads else ("u", "lr", "f", "gate_out")
            )  # a2_heads: f / gate_out narrower than A3's
            for k_ in own:  # the H3 store IS the stream phases' output tensors
                ttnn.deallocate(scr[k_])
        scr["u"], scr["lr"] = self.mm_qkv.y, self.mm_lr.y
        if not self.a2_heads:
            scr["f"], scr["gate_out"] = self.mm_fb.y, self.mm_gb.y
        self.scr = scr

    def stores(self) -> dict:
        """A2's four stores (the stream phases' output tensors; = the kda phase's H3 store unless a2_heads)"""
        return dict(u=self.mm_qkv.y, lr=self.mm_lr.y, f=self.mm_fb.y, gate_out=self.mm_gb.y)

    # ------------------------------------------------------------------------------------------ parts helpers
    def _sig(self, targets_by_core: dict, sem: int, cores: list, name: str) -> dict:
        rt = {}
        for c_ in cores:
            t = [v for xy in targets_by_core.get(c_, []) for v in xy]
            rt[c_] = [1 if t else 0, sem, len(t) // 2] + t
        w = max(len(v) for v in rt.values())
        for c_ in rt:
            rt[c_] += [0] * (w - len(rt[c_]))
        return dict(name=name, src=os.path.join(ASM_KDIR, "pmk_sig.cpp"), ct=[], rt=rt, defines=[],
                    noc_xy={c_: tuple(3 + 2 * i for i in range(rt[c_][2])) for c_ in rt})  # fmt: skip

    def _build_p13(self, mk, ring_bytes: int):
        """W7 step 2: the shared expert's 1/tp slice as pfs_shared.SharedP13's two PfStream phases on A's members and
        streamers. gate_up: K = hidden, N = 2 i (the (gate_j, up_j) tile pairs of interleave_gate_up), in0 = the x-hat_f
        block in XS 35 (P11's AG), SwiGLU epilogue (clamp = the MLP's limit) packed into ACT = CB_XK; down: K = i, in0 =
        ACT (xsrc), PARK = CB_PARK2, out = y_s [C, H] bf16 ROW_MAJOR (epi 2) = program B's PMK_SHARED input. 8 streams
        per phase (one per DRAM bank) on 16 of the 24 adj streamers."""
        from models.autoports.zai_org_glm_5_3_flash.tt import pfs_shared as PSH

        sh = self.pl.shared
        n, K, H = self.n, self.geom.hidden, self.geom.hidden
        i = int(sh.intermediate)
        gu = [w.reshape(K, -1) for w in _host_shards(sh.gate_up, n)]
        assert all(w.shape[1] == 2 * i for w in gu), ([w.shape for w in gu], i)
        ie = p13_slice(
            i
        )  # PMK_P13_SLICE (debug, default off): the first ie gate / up columns and down rows of the slice
        w_gu = [PSH.interleave_gate_up(w[:, :ie], w[:, i : i + ie]) for w in gu]
        w_d = [w.reshape(i, -1)[:ie, :H].contiguous() for w in _host_shards(sh.down, n)]
        i = ie

        def nw_of(nt, nb_=None):  # 8 streams (one per bank) when the width allows, else the most <= 12 (24 streamers)
            nb_ = self.nb_ if nb_ is None else nb_
            for nst in (8, 4, 12, 6, 2):
                if nt % nst == 0 and (nt // nst) % nb_ == 0:
                    return nt // nst
            raise AssertionError(("P13: no stream split of", nt))

        nt_gu, nt_d = 2 * i // PMK.TILE, H // PMK.TILE
        kt_d = i // PMK.TILE
        bf16x8 = getattr(self, "p13_dn", "share") == "bf16x8"
        kb_d = self.p13_dn_kb if bf16x8 else p13_kb_down(kt_d)  # bf16x8: PARK2's block / NB 8 (_cbs)
        assert not self.park2_one_block or kb_d == self.kb_small_, (
            kb_d,
            self.kb_small_,
            "PARK2: one block size (__init__)",
        )
        nw_gu, nw_d = nw_of(nt_gu), nw_of(nt_d, P13_DN_NB if bf16x8 else None)
        st_gu, st_d = self.streamers[: 2 * (nt_gu // nw_gu)], self.streamers[: 2 * (nt_d // nw_d)]
        self.sh_gu = mk(w_gu, nw=nw_gu, streamers=st_gu, epi=1, limit=float(sh.limit), cb_out=CB_XK)
        dn_kw = dict(nb=P13_DN_NB, w_dtype=ttnn.bfloat16) if bf16x8 else {}
        self.sh_dn = mk(w_d, nw=nw_d, kb=kb_d, streamers=st_d, epi=2, xsrc=True, cb_x=CB_XK,
                        cb_park=getattr(self, "cb_p13_park", CB_PARK2), park=self.park2, **dn_kw)  # fmt: skip
        for m_ in (self.sh_gu, self.sh_dn):
            assert PF.RING * m_.bt * m_.tb <= ring_bytes, (m_.k, m_.n, m_.bt, m_.tb, "a block ring must fit CB 43")
        assert self.sh_dn.kt == kt_d and self.sh_gu.cb_out == self.sh_dn.cb_x == CB_XK
        self.report["p13"] = dict(i=i, gu=dict(nt=nt_gu, nw=nw_gu, kb=self.sh_gu.kb, streamers=len(st_gu)),
                                  dn=dict(nt=nt_d, nw=nw_d, kb=kb_d, streamers=len(st_d)), act_tiles=kt_d,
                                  weight_bytes=self.sh_gu.weight_bytes + self.sh_dn.weight_bytes)  # fmt: skip

    def _build_p12(self, ring_bytes: int):
        """W7 step 3: the router logits as a PfStream phase on A's members (in0 = XS: the x-hat_f block; N = 320 = the 288
        experts zero-padded, K = hidden; weights = the router's own [K, 320] in its dtype, replicated per chip; fp32 out
        tiles -> rt_logits [C, 320] DRAM) + the router tail's tensors (packed bias with -1e9 on the padded experts; ids
        [C, 8] uint16 / weights [C, 8] bf16 ROW_MAJOR = program B's PMK_IDS_U16 / PMK_WT_BF16 inputs)"""
        from models.autoports.zai_org_glm_5_3_flash.tt import router_tail as RT

        router = self.pl.moe.router
        w320, b320 = router.weights[RT.COLS], router.biases[RT.COLS]
        dt = str(w320.dtype).split(".")[-1]
        park = {"FLOAT32": P12_PARK_FP32, "BFLOAT16": CB_XK, "BFLOAT8_B": PMK.CB_PARK}.get(dt)
        assert park is not None, ("P12: router weight dtype", dt)
        K = self.geom.hidden
        w = [t_.reshape(K, -1)[:, : RT.COLS].contiguous() for t_ in _host_shards(w320, self.n)]
        bias = _host_shards(b320, self.n)[0].reshape(-1)[: RT.COLS]
        nb_, nw_, kb_ = (
            2,
            2,
            8,
        )  # nt 10: nb | nw, rt nb <= 8, parity (nw / nb) (kt / kb) / 2 even; 5 streams = 10 streamers
        fid = ttnn.MathFidelity.HiFi4
        # PMK_P12_RECTS=R (the non-B lever (1)-lite, coordinator 23:38:15Z): the router logits stream as R member RECTANGLES
        # (rows of the 8 x 8 member grid = the holders of one source chip's blocks at TP8), each fed by its own 2 streamers
        # and gated by ITS members' H8 only -- the local blocks' rectangle routes while P11's remote slices are in flight,
        # instead of the 64-member lockstep waiting for the last block; per member the same K loop (bitwise logits)
        p12_rects = int(_knob("PMK_P12_RECTS", 0))
        rk = {}
        if p12_rects:
            per = len(self.members) // p12_rects
            assert per * p12_rects == len(self.members) and 2 * p12_rects <= len(self.streamers), (
                p12_rects,
                len(self.streamers),
            )
            rk = dict(rects=[self.members[per * h_ : per * (h_ + 1)] for h_ in range(p12_rects)],
                      streamers=self.streamers[: 2 * p12_rects])  # fmt: skip
        self.report["p12_rects"] = p12_rects
        self.rt_mm = PF.PfStreamMM(self.mesh, w, members=self.members, **(rk or {"streamers": self.streamers[: 2 * (RT.COLS // PMK.TILE // nw_)]}),
                                   nb=nb_, nw=nw_, kb=kb_, park=2, w_dtype=w320.dtype, fidelity=fid, rt=1, probe=False,
                                   sync=self.sync, cb_park=park, cb_out=CB_OUT32, out_dtype=ttnn.float32)  # fmt: skip
        assert PF.RING * self.rt_mm.bt * self.rt_mm.tb <= ring_bytes, (
            self.rt_mm.bt,
            self.rt_mm.tb,
            "the logits ring must fit CB 43",
        )
        mk = lambda t_, dt_, lay_: ttnn.from_torch(t_, dtype=dt_, layout=lay_, device=self.mesh,  # noqa: E731
                                                   memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                                   mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))  # fmt: skip
        self.rt_bias = mk(RT.packed_bias(bias).reshape(1, 1, PMK.TILE, PMK.TILE), ttnn.float32, ttnn.TILE_LAYOUT)
        # ONE ROW_MAJOR page each, [1, C x 8]: program B's PMK_IDS_U16 / PMK_WT_BF16 readers address the whole tensor as
        # page 0 (plan_count / plan_dest / comb_reader: page_size = C x 2 x K); a [C, 8] tensor is C interleaved pages
        # (s3w 16:15Z: B's y garbage + non-deterministic with [C, 8])
        self.rt_ids = mk(torch.zeros(1, self.c * RT.TOP_K, dtype=torch.int32), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT)
        if (
            self.xext_on
        ):  # P11x's output: rows [0, C) written in-program, rows [C, C + 32) stay zero (the gather's pad row)
            self.xext = mk(torch.zeros(self.c + 32, K), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self.rt_wt = mk(torch.zeros(1, self.c * RT.TOP_K), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self.rt_scaling, self.rt_eps = float(self.pl.moe.scaling), 1e-20
        self.report["p12"] = dict(w_dtype=dt, park=park, nb=nb_, nw=nw_, kb=kb_, streams=RT.COLS // PMK.TILE // nw_,
                                  weight_bytes=self.rt_mm.weight_bytes, scaling=self.rt_scaling)  # fmt: skip

    def _p13_carry_check(self):
        """P13 reads the x-hat_f block P11 left in XS 35 one window earlier: XS must sit at the SAME window offset in both
        windows (re-laying out moves only the interfaces; the bytes stay)"""
        e = self.engine
        old = e.scope
        e.scope = set(self.scopes[0] if self.g_window is not None else A15B12_SCOPE if self.b_p12 else A15B13_SCOPE)
        try:
            wl = e._win_layout("member")
            wins = e.windows("member")
        finally:
            e.scope = old
        w11, w13 = e._win_index("P11", wins), e._win_index("P13", wins)
        assert w13 == w11 + (2 if self.b_p12 else 1), (wins, w11, w13)
        off = {}
        for w in range(w11, w13 + 1):  # P11's window, P12's (step 3), P13's
            tab_w = wl["tab0"] if w == 0 else wl["tabs"].get(w, ())  # (G v2a: P11 in window 1 = the launch layout)
            ent = [t for t in tab_w if int(t[0]) == FS.CB_XS]
            assert ent, (f"P13 carry: XS {FS.CB_XS} not laid out in window {w + 1}", wl["tabs"].get(w))
            off[w] = int(ent[0][1])
        assert len(set(off.values())) == 1, ("P13 carry: XS moves between the P11 and P13 windows", off)
        self.report["p13_carry"] = dict(xs_offset=off[w13], windows=[w11 + 1, w13 + 1])

    def _qbase(self, qbase: int, mm, pushed: dict, name: str) -> int:
        """the launch-global block base of a PfStream phase (see park_align in __init__): aligned UP to lcm(2, P) (P =
        the phase's PARK depth; 2 = the lanes' parity) after asserting the phase starts on slot 0 of its PARK CB; then
        its blocks are counted against that CB. park_align 0: the running value (the previous build)."""
        P = int(mm.park)
        w0 = pushed.get(mm.cb_park, 0) % P
        if self.park_align:
            assert w0 == 0, (f"PfStream {name}: would start on slot {w0} of its {P}-slot PARK CB {mm.cb_park} "
                             f"({pushed.get(mm.cb_park, 0)} blocks before it in the program): the streamer would land blocks off "
                             f"the members' ring")  # fmt: skip
            step = P * 2 // math.gcd(P, 2)
            qbase = -(-qbase // step) * step
        pushed[mm.cb_park] = pushed.get(mm.cb_park, 0) + mm.blocks
        self.report.setdefault("qbase", {})[name] = dict(qbase=qbase, park=P, cb=mm.cb_park, start_slot=w0, blocks=mm.blocks,
                                                         streamer_slot0=(qbase % P), aligned=(w0 == 0 and qbase % P == 0))  # fmt: skip
        return qbase

    def _pfs_parts(self, pt: dict, mm, name: str, rt_by_rank: dict | None = None) -> dict:
        """one PfStream phase's parts() -> the engine's per-(role, risc) part dicts (member rd / c / wr, the streamer
        BRISC on the roots and the MM-only streamers; rt_by_rank: {rank: pt['rt']} when the args differ per chip)"""
        srcs = pt["sources"]
        files = {k: _src_file(f"pfs_{k}", v) for k, v in dict(st=srcs["streamer"]["dm0"], rd=srcs["member"]["dm1"],
                                                             c=srcs["member"]["c"], wr=srcs["member"]["dm0"]).items()}  # fmt: skip
        cc = ttnn.ComputeConfigDescriptor(**pt["compute"], math_approx_mode=False)
        dyn = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                noc=ttnn.NOC.RISCV_0_default, noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC)  # fmt: skip
        sub = lambda cores: {c_: pt["rt"][c_] for c_ in cores if c_ in pt["rt"]}  # noqa: E731

        def rts(cores, trim=None):
            cut = (lambda v: list(v)[:trim]) if trim else (lambda v: v)  # noqa: E731
            if rt_by_rank is None:
                return dict(rt={c_: cut(v) for c_, v in sub(cores).items()})
            return dict(rt_by_rank={r: {c_: cut(v[c_]) for c_ in cores if c_ in v} for r, v in rt_by_rank.items()})

        # COMPUTE-ARG TRIM (Stage 14, the trace-light PMK 2026-10-23): the member TRISC body pfs_member_c.cpp reads ONE
        # runtime arg, ph_rt0 + RT_PROBE; the phase's rt list is the readers' / writer's (43 / 18 words per phase), and a
        # TRISC's args stay per-core RTAs in the trace even with the DRAM arg table (TRISC cannot NoC-read): the compute
        # part keeps words 0 .. RT_PROBE only (A2 member/c 97 -> 16 words per core, A4 43 -> 4). PMK_PFS_C_TRIM=0: full.
        c_trim = PF._W["RT_PROBE"] + 1 if os.environ.get("PMK_PFS_C_TRIM", "1") == "1" else None

        streamer_roles = {}
        for role, cores in (
            ("root", self.roots),
            ("mm_only", self.mm_only),
            ("tail", [tuple(t) for t in self.engine.place.tail]),
        ):
            # the stream's OWN streamers (o_proj row groups stream from the roots only: PMK_O_RECTS)
            if set(cores) & set(tuple(c_) for c_ in mm.all_streamers):
                streamer_roles[(role, "dm0")] = dict(name=name, src=files["st"], ct=pt["ct"], defines=pt["defines"],
                                                     config=dyn, idle_ok=True, **rts(cores))  # fmt: skip
        out = {
            ("member", "dm1"): dict(name=name, src=files["rd"], ct=pt["ct"], defines=pt["defines"], **rts(self.members)),
            ("member", "c"): dict(name=name, src=files["c"], ct=pt["ct"], defines=pt["defines"], compute=cc,
                                  dst_any=True, fid_mode="template", **rts(self.members, trim=c_trim)),
            ("member", "dm0"): dict(name=name, src=files["wr"], ct=pt["ct"], defines=pt["defines"], **rts(self.members)),
        }  # fmt: skip
        out.update(streamer_roles)
        return out

    def _c1_parts(
        self,
        use: int,
        hn,
        fab_at: int,
        release: bool = False,
        lane_ack: bool = False,
        c1x_use: int = 1,
        acquire: bool = False,
    ) -> tuple:
        """(member dm1 part = c1_stub_prod per rank, lane dm0 = c1 send per rank, lane dm1 = c1 read per rank).
        release: lane h hands link h over to RS8 stage 2's lane h after its closes (SEM_LINK_REL: the fused A2F at TP8)
        lane_ack: a LOCAL holder's last-slice FS_ARRIVE +1 comes from its lane after the lane read the block out of XS
        (c1_stub_prod C1P_LANE_ACK / c1_read C1R_ACK; still SLICES per holder per use): FS_ARRIVE complete = XS free
        """
        c1, nl, bl = self.c1, len(self.lanes), self.bl
        fsync = c1.sync.buffer_address()
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        prod_ct = [c1.tpb, c1.slice_t, FS.CB_XS, FS.SEM_FS_ARRIVE] + acc(hn)
        prod, send, read = {}, {}, {}
        for rank in range(self.n):
            r, c = divmod(rank, self.C)
            p = c1._pos(r, c)
            rt = {}
            for j, m in enumerate(self.members):
                b = j - p * bl
                xl = self.noc(self.lanes[j % nl])
                xsf = [use, xl[0], xl[1]]
                if 0 <= b < bl:
                    lane = self.noc(self.lanes[b % nl])
                    me = self.noc(m)
                    rt[m] = [1, hn.buffer_address(), b, lane[0], lane[1], fsync, b // nl, me[0], me[1]] + xsf
                else:
                    rt[m] = [0, 0, 0, 0, 0, fsync, 0, 0, 0] + xsf
            prod[rank] = rt
            ks, _, _ = c1.parts(r, c, xsfree_n=len(self.members) // nl, fab_at=fab_at)
            send[rank] = _rt_dict(ks[0].runtime_args, self.lanes)
            if release:  # (rel_x, rel_y) of stage-2 lane h before the fab_at slot (c1x_send.cpp FS_C1X_REL_SEM)
                assert self.n == 8 and len(self.lanes2) == nl, (self.n, self.lanes2, nl)
                for h, l_ in enumerate(self.lanes):
                    v = send[rank][l_]
                    send[rank][l_] = v[:-1] + list(self.noc(self.lanes2[h])) + v[-1:]
            read[rank] = _rt_dict(ks[1].runtime_args, self.lanes)
            send_ct, read_ct = list(ks[0].compile_time_args), list(ks[1].compile_time_args)
            send_defs, send_cfg, read_cfg = [tuple(d) for d in ks[0].defines], ks[0].config, ks[1].config
            read_defs = [tuple(d) for d in ks[1].defines]  # the per-hop drain's kill switch (FS_NO_ROW_BARRIER)
        name = f"c1u{use}"
        if lane_ack:
            assert (
                self.n == 2
            ), "lane_ack: c1_read.cpp (TP2); at TP8 (I1) keeps XS live through B (no reclaim overwrite)"
            read_defs = read_defs + [("C1R_ACK", "1")]
        if c1x_use > 1:  # (W7 P11 at TP8) c1x_read's partner / drain polls count this launch's earlier C1 uses
            assert self.n == 8
            read_defs = read_defs + [("C1X_USE", str(c1x_use))]
        mem = dict(name=name, src=_src_file("c1_stub_prod", c1._srcs["c1_stub_prod.cpp"]), ct=prod_ct,
                   defines=[("C1P_LANE_ACK", "1")] if lane_ack else [],
                   rt_by_rank=prod, nonzero={m: (5,) for m in self.members})  # fmt: skip
        sname = "c1x_send.cpp" if self.n == 8 else "c1_send.cpp"
        rname = "c1x_read.cpp" if self.n == 8 else "c1_read.cpp"
        send_defs = send_defs + [("FS_HDR_CB", str(CB_HDR))]
        n_snd = len(next(iter(send[0].values())))
        noc_xy = (1, n_snd - 3) if release else (1,)
        if release:
            assert sname == "c1x_send.cpp", sname
            send_defs = send_defs + [("FS_C1X_REL_SEM", str(SEM_LINK_REL))]
        if acquire:  # (W7 P11 at TP8) RS8 stage 2's tail held link h in between: wait its release back
            assert release and sname == "c1x_send.cpp"
            send_defs = send_defs + [("FS_C1X_ACQ_SEM", str(SEM_LINK_ACQ)), ("FS_C1X_ACQ_N", "1")]
        snd = dict(name=name, src=_src_file("c1_send", c1._srcs[sname]), ct=send_ct, defines=send_defs, rt_by_rank=send,
                   config=send_cfg, fab_at_arg=n_snd - 1, noc_xy={l: noc_xy for l in self.lanes},
                   nonzero={l: (0,) for l in self.lanes})  # fmt: skip
        assert len({len(v) for d in send.values() for v in d.values()}) == 1, "the C1 send args are rank-invariant"
        rd = dict(name=name, src=_src_file("c1_read", c1._srcs[rname]), ct=read_ct, defines=read_defs, rt_by_rank=read,
                  config=read_cfg)  # fmt: skip
        return mem, snd, rd

    def _h1_targets(self, rank: int) -> dict:
        """member (b, j) -> the holders of its group's LOCAL blocks on this chip and their twins on the peers: 8
        targets, so every holder counts 8 per use (H7: the P10 x-hat_f rows are complete before P11's stub reads)"""
        g, bl, n = self.geom, self.bl, self.n
        lo = bl * rank
        out = {}
        for G, m in enumerate(self.members):
            b = G // 8
            tg = []
            for sb in range(g.sp_blocks):
                if sb % 8 != b:
                    continue
                for q in range(n):
                    tg.append(self.noc(self.members[(lo + sb + q * bl) % (n * bl)]))
            out[m] = tg
        return out

    def _rs_signals(self, rank: int) -> dict:
        """member G -> the H5 word of its row block (lanes-reduce RSR: lane b % nl, LREADY + b // nl for a peer-owned
        block, OWNRDY + b // nl for an own block)"""
        rsr, nl, bl = self.rsr, len(self.lanes), self.bl
        r, c = divmod(rank, self.C)
        p = c if rsr.axis == 1 else r
        sync = rsr.sync.buffer_address()
        out = {}
        for G, m in enumerate(self.members):
            o, b = divmod(G, bl)
            word = (FS._FS["OWNRDY"] if o == p else FS._FS["LREADY"]) + b // nl
            x, y = self.noc(self.lanes[b % nl])
            out[m] = (x, y, sync + word * FS.WORD)
        return out

    def _rs8_signals(self, rank: int) -> dict:
        """member G -> its H5 word (7a's pf_rs8 map): G's mesh row r' = G // 32, b = G % 32; r' == r: the stage-1 reducer
        b % 8 adds it (OWNRDY + b // 8), else the stage-1 lane b % 2 sends it to the partner (LREADY + b // 2)"""
        rs1 = self.rs8.rs1
        sync = rs1.sync.buffer_address()
        nl, nr, bl = len(rs1.lanes), len(rs1.reducers), rs1.bl
        r, c = divmod(rank, self.C)
        out = {}
        for G, m in enumerate(self.members):  # the index = the block's position in its reducer's / lane's list
            rr, b = divmod(G, bl)
            core, word = (rs1.reducers[b % nr], FS._FS["OWNRDY"] + rs1.red_list(c, b % nr).index(b)) if rr == r else \
                (rs1.lanes[b % nl], FS._FS["LREADY"] + rs1.lane_list(c, b % nl).index(b))  # fmt: skip
            x, y = self.noc(core)
            out[m] = (x, y, sync + word * FS.WORD)
        return out

    def _rs8_parts(
        self,
        fab1: int,
        fab2: int,
        *,
        rs8=None,
        src=None,
        ready: bool = True,
        sfx: str = "",
        after_n: int = 1,
        h6: bool = True,
        handback: bool | None = None,
        role_remap: dict | None = None,
        rdy_mult: int = 1,
    ) -> dict:
        """the TP8 RS as composite parts: {(role, risc): part | [parts]}. Stage 1 (FabricStreamRSR axis 0 with
        reducers): rs_send / rs_read on the family lanes, red_read / rs_reduce / red_write on reducers1; stage 2
        (FabricStreamRSL): rsl_send / rsl_read on lanes2 (tail cores), rsl_red_read / rsl_reduce / red_write on
        reducers2. Every part that runs on a SUBSET of its role's cores is GATED (a 1 after its own args). fab1 / fab2 =
        where connect() appends the fabric args (the lane / tail BRISC composite widths; two-pass)."""
        rs8, I = rs8 if rs8 is not None else self.rs8, RS8_CB
        src = src if src is not None else self.mm_o.y
        rs1, rs2 = rs8.rs1, rs8.rs2
        k1, k2 = {}, {}
        for rank in range(self.n):
            r, c = divmod(rank, self.C)
            k1[rank] = rs1._build(src, r, c, ready=ready, red_signals=rs8.signals(c), fab_at=fab1)[0]
            k2[rank] = rs2._build(rs1.y, r, c, ready=True, fab_at=fab2)[0]
        names1 = ("rs_send", "rs_read", "red_read", "rs_reduce", "red_write")
        names2 = ("rsl_send", "rsl_read", "rsl_red_read", "rsl_reduce", "red_write")
        cores1 = (self.lanes, self.lanes, self.red1, self.red1, self.red1)
        cores2 = (self.lanes2, self.lanes2, self.red2, self.red2, self.red2)
        # CB index remap (the 7a kernels take their CBs as compile-time args at fixed positions)
        remap1 = {2: {2: I["A"], 3: I["B"]}, 3: {2: I["A"], 3: I["B"], 4: I["Y1"]}, 4: {2: I["Y1"]}}
        remap2 = {2: {2: I["T0"]}, 3: {2: I["T0"], 3: I["Y2"]}, 4: {2: I["Y2"]}}

        def modes(idx: list):
            cc = ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
            )
            m = [ttnn.UnpackToDestMode.Default] * 64
            for i in idx:
                m[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
            cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(m)
            return cc

        def part(
            ks: dict,
            i: int,
            name: str,
            cores: list,
            srcs: dict,
            remap: dict,
            gated: bool,
            tag: str,
            compute=None,
            ins: dict | None = None,
        ):
            rt = {rank: _rt_dict(ks[rank][i].runtime_args, cores) for rank in range(self.n)}
            if (
                ins
            ):  # args inserted before the fab_at slot (the last arg), per core: (W7) rsl_send's release-back target
                rt = {rank: {c_: v[:-1] + list(ins[c_]) + v[-1:] for c_, v in d.items()} for rank, d in rt.items()}
            lens = {len(v) for d in rt.values() for v in d.values()}
            assert len(lens) == 1, (tag, lens)  # equal args on the active cores: the gate sits after them
            width = lens.pop()
            ct = list(ks[0][i].compile_time_args)
            for pos, idx in remap.get(i, {}).items():
                ct[pos] = idx
            text = srcs[name + ".cpp"]
            if gated:
                text = _gated_text(text, width, tag)
                rt = {rank: {c_: v + [1] for c_, v in d.items()} for rank, d in rt.items()}
            defs = [tuple(d) for d in ks[0][i].defines]
            # noc_words: every NoC target of the RS8 parts is a FabricStream sync word (R1RDY / FS_RED_SIG / ARR) or
            # a landing buffer, never a semaphore (pmk_layer.check_fabric_words)
            out = dict(
                name="rs8", src=_src_file(tag, text), ct=ct, defines=defs, rt_by_rank=rt, idle_ok=gated, noc_words=True
            )
            # the arg layouts of tt/fabric_stream.py (RSR._build / RSL._build): NoC (x, y) pairs, non-zero addresses and
            # the fab_at slot, per active core (homelab-1 01:22Z: an unset target = a write to NoC (0, 0) address 0)
            dec = {rank: {} for rank in rt}
            for rank, d in rt.items():
                for c_, v in d.items():
                    n = len(v) - (1 if gated else 0)
                    if name == "rs_send":  # [sync, mx, my, p, q, nl, h, up, dn, r, y, nbs, nrb] + bs + rb + [fab_at]
                        dec[rank][c_] = dict(noc_xy=(1,), nonzero=(0, 9, 10), fab=n - 1)
                    elif name == "rs_read":  # [sync, q, nl, h, pt, r, ready, nbs, nrb] + ...
                        dec[rank][c_] = dict(noc_xy=(), nonzero=(0, 4, 5), fab=None)
                    elif (
                        name == "red_read"
                    ):  # [sync, sync, per_lane, pt, r, ready, n] + n x [jo, jl, lx, ly, word, idx]
                        dec[rank][c_] = dict(
                            noc_xy=tuple(7 + 6 * i + 2 for i in range(v[6])), nonzero=(0, 1, 3, 4), fab=None
                        )
                    elif name == "red_write" and stage == 1:  # [y, n] + per + n x (x, y, addr)  (FS_RED_SIG)
                        nb_ = v[1]
                        dec[rank][c_] = dict(
                            noc_xy=tuple(2 + nb_ + 3 * i for i in range(nb_)),
                            nonzero=(0,) + tuple(2 + nb_ + 3 * i + 2 for i in range(nb_)),
                            fab=None,
                        )
                    elif name == "red_write":  # stage 2: [y, 1, b]
                        dec[rank][c_] = dict(noc_xy=(), nonzero=(0,), fab=None)
                    elif name == "rsl_send":  # [sync, lx, ly, me, nc, r2, nrows] + 3 x nrows (+ [rel x, y]) + [fab_at]
                        dec[rank][c_] = dict(noc_xy=(1, n - 3) if ins else (1,), nonzero=(0, 5), fab=n - 1)
                    elif name == "rsl_read":  # [sync, ready, r1, nrows] + rows
                        dec[rank][c_] = dict(noc_xy=(), nonzero=(0, 2), fab=None)
                    elif name == "rsl_red_read":  # [sync, sync, per, me, r1, r2, b2, ready, 1, b, lx, ly]
                        dec[rank][c_] = dict(noc_xy=(10,), nonzero=(0, 1, 4, 5), fab=None)
                    else:  # the compute kernels: no args
                        dec[rank][c_] = dict(noc_xy=(), nonzero=(), fab=None)
            d0 = dec[0]
            out["noc_xy"] = {c_: v["noc_xy"] for c_, v in d0.items()}
            out["nonzero"] = {c_: v["nonzero"] for c_, v in d0.items()}
            fabs = {v["fab"] for d in dec.values() for v in d.values()}
            assert len(fabs) == 1, (tag, fabs)
            if fabs != {None}:
                out["fab_at_arg"] = fabs.pop()  # audit: the slot holds the composite's width (where connect() appends)
            for rank, d in dec.items():  # the layout is rank-invariant (the audit reads rank 0's declarations)
                assert {c_: (v["noc_xy"], v["nonzero"]) for c_, v in d.items()} == {
                    c_: (v["noc_xy"], v["nonzero"]) for c_, v in d0.items()
                }, tag
            if compute is not None:
                out["compute"] = compute
                # the part regime: the RS reducers' compute (copy + SFPU add, <= 4 fp32 DST tiles per acquire) runs
                # under a full-sync composite (the kda scan's on the MM-only cores of a fused A3 + A4)
                out["dst_any"], out["fid_mode"] = True, "any"
            else:
                out["config"] = ks[0][i].config
            return out

        p = {}
        p[("lane", "dm0")] = part(k1, 0, names1[0], cores1[0], rs1._srcs, remap1, False, "rs8_send" + sfx)
        p[("lane", "dm0")]["defines"] = p[("lane", "dm0")]["defines"] + [("FS_HDR_CB", str(CB_HDR))]
        p[("lane", "dm1")] = part(k1, 1, names1[1], cores1[1], rs1._srcs, remap1, False, "rs8_read" + sfx)
        rdy = (
            [("FS_RDY_MULT", str(rdy_mult))] if rdy_mult != 1 else []
        )  # (P15 streamed: h_split combine units per block)
        p[("lane", "dm1")]["defines"] = p[("lane", "dm1")]["defines"] + rdy
        hb = getattr(self, "w7_tp8_handback", False) if handback is None else handback  # W7: link h back to C1X lane h
        ins2 = {l2: self.noc(l1) for l1, l2 in zip(self.lanes, self.lanes2)} if hb else None
        p[("tail", "dm0")] = [
            part(k2, 0, names2[0], cores2[0], rs2._srcs, remap2, True, "rs8_rsl_send" + sfx, ins=ins2)
        ]
        if hb:
            p[("tail", "dm0")][0]["defines"] = p[("tail", "dm0")][0]["defines"] + [
                ("FS_RSL_REL_SEM", str(SEM_LINK_ACQ))
            ]
        p[("tail", "dm0")][0]["defines"] = p[("tail", "dm0")][0]["defines"] + [("FS_HDR_CB", str(CB_HDR))]
        if self.link_handover:  # opens after C1X lane h released link h (SEM_LINK_REL)
            p[("tail", "dm0")][0]["defines"] = (
                p[("tail", "dm0")][0]["defines"]
                + [("FS_RSL_AFTER_SEM", str(SEM_LINK_REL))]
                + ([("FS_RSL_AFTER_N", str(after_n))] if after_n > 1 else [])
            )  # (W7 P15: C1X's 2nd release, after P11)
        p[("tail", "dm1")] = [part(k2, 1, names2[1], cores2[1], rs2._srcs, remap2, True, "rs8_rsl_read" + sfx)]
        # the reducer kernels, per hosting role (a stage's 8 reducers may span MM-only / tail / root cores): one gated
        # part per (role, RISC) with that role's cores only
        red = [(k1, 2, names1[2], rs1._srcs, remap1, "rs8_red_read", "dm1", None, 1),
               (k2, 2, names2[2], rs2._srcs, remap2, "rs8_rsl_red_read", "dm1", None, 2),
               (k1, 3, names1[3], rs1._srcs, remap1, "rs8_reduce", "c", modes([I["A"], I["B"]]), 1),
               (k2, 3, names2[3], rs2._srcs, remap2, "rs8_rsl_reduce", "c", modes([I["T0"] + j for j in range(4)]), 2),
               (k1, 4, names1[4], rs1._srcs, remap1, "rs8_red_write1", "dm0", None, 1),
               (k2, 4, names2[4], rs2._srcs, remap2, "rs8_red_write2", "dm0", None, 2)]  # fmt: skip
        for ks, i, name, srcs, remap, tag, risc, compute, stage in red:
            allc = self.red1 if stage == 1 else self.red2
            for role in self.rs8_roles[stage]:
                cores = [c_ for c_ in allc if self.rs8_role[c_] == role]
                rm_ = (role_remap or {}).get((role, stage), remap)  # (W7 P15 at TP8: the MM-only reclaim indices)
                cmp_ = (
                    compute
                    if rm_ is remap or compute is None
                    else modes([rm_[3][2], rm_[3][3]] if stage == 1 else [rm_[3][2] + j for j in range(4)])
                )
                p.setdefault((role, risc), []).append(
                    part(ks, i, name, cores, srcs, rm_, True, tag + sfx, compute=cmp_)
                )
                if name == "red_read" and stage == 1:
                    p[(role, risc)][-1]["defines"] = p[(role, risc)][-1]["defines"] + rdy
        if self.f_p8p10 and h6:  # M-C, H6 at TP8: each stage-2 reducer +1 root 0's FS_REDUCED after its red_write2 (the
            # pmk_sig barriers its writes first); root 0's NCRISC aggregates (in the two-pass widths: the tail BRISC
            # composite's fab_at counts this part too)
            root0, rc = self.roots[0], self.engine.place.role_cores()
            for role in self.rs8_roles[2]:
                mine = [c_ for c_ in self.red2 if self.rs8_role[c_] == role]
                cores = [tuple(c_) for c_ in rc[role]]
                sig = self._sig({c_: [self.noc(root0)] for c_ in mine}, PMK.SEM_FS_REDUCED, cores, "h6sig")
                sig["idle_ok"] = True
                p.setdefault((role, "dm0"), []).append(sig)
            p.setdefault(("root", "dm1"), []).append(self._h6_agg())
        for key, v in p.items():  # distinct sub-part names inside one composite (the namespaces ph8_<name>)
            vs = v if isinstance(v, list) else [v]
            for j, sub in enumerate(vs):
                sub["name"] = f"s{j + 1}" if len(vs) > 1 else "rs8"
        return p

    def _h6_agg(self) -> dict:
        """M-C, H6 at TP8: root 0's NCRISC waits the 8 stage-2 reducers' +1 on its FS_REDUCED, then SETs every
        member's copy (one NoC-1 multicast over the 8 x 8 rectangle) and the other roots' (unicast) to the H6 wait
        value (pmk_agg.cpp; its own copy is set first: root 0's P10 waits it too)"""
        h6 = self.engine.handoffs["H6"]
        root0 = self.roots[0]
        lx, ly = self.noc(self.members[0])
        hx, hy = self.noc(self.members[-1])
        extra = [v for r_ in self.roots[1:] for v in self.noc(r_)]
        n_ = 11 + len(extra)
        rt = {r_: [0] * n_ for r_ in self.roots}
        rt[root0] = [1, PMK.SEM_FS_REDUCED, len(self.red2), int(h6.wait_value), lx, ly, hx, hy, len(self.members), 1,
                     len(extra) // 2] + extra  # fmt: skip
        return dict(name="h6agg", src=os.path.join(ASM_KDIR, "pmk_agg.cpp"), ct=[], rt=rt, defines=[],
                    noc_xy={root0: (4, 6) + tuple(11 + 2 * i for i in range(len(extra) // 2))})  # fmt: skip

    def _h3_barrier_parts(self) -> tuple:
        """(member dm0 sig part, root dm1 agg part): the in-program H3 of the A3 merge -- every member +1 root 0's
        SEM_QKV after its P3 stores landed; root 0 waits 64 and SETs the members' SEM_QKV to NCH x NPROD (one
        multicast over the 8 x 8 rectangle), which the items' per-chunk wait_min(>= (c + 1) x NPROD) then pass at once
        """
        sem, nch = INT.KDA.SEM_QKV, self.kphase.geo.n_chunks
        root0 = self.roots[0]
        sig = self._sig({m: [self.noc(root0)] for m in self.members}, sem, self.members, "h3sig")
        lx, ly = self.noc(self.members[0])
        hx, hy = self.noc(self.members[-1])
        rt = {r_: [0] * 11 for r_ in self.roots}
        rt[root0] = [1, sem, len(self.members), nch * self.kphase.nprod, lx, ly, hx, hy, len(self.members), 1, 0]
        agg = dict(name="h3agg", src=os.path.join(ASM_KDIR, "pmk_agg.cpp"), ct=[], rt=rt, defines=[],
                   noc_xy={root0: (4, 6)})  # fmt: skip
        return sig, agg

    def _rs_parts(self, fab_at: int) -> dict:
        rsr = self.rsr
        send, read = {}, {}
        for rank in range(self.n):
            r, c = divmod(rank, self.C)
            ks, _, _, _ = rsr._build(self.mm_o.y, r, c, ready=True, fab_at=fab_at)
            send[rank] = _rt_dict(ks[0].runtime_args, self.lanes)
            read[rank] = _rt_dict(ks[1].runtime_args, self.lanes)
            send_k, read_k, red = ks[0], ks[1], ks[2]
        snd = dict(name="rs", src=_src_file("rs_send", rsr._srcs["rs_send.cpp"]), ct=list(send_k.compile_time_args),
                   defines=[("FS_HDR_CB", str(CB_HDR))], rt_by_rank=send, config=send_k.config,
                   fab_at_arg=len(next(iter(send[0].values()))) - 1, noc_xy={l: (1,) for l in self.lanes},
                   nonzero={l: (0, 9, 10) for l in self.lanes})  # fmt: skip
        rd = dict(name="rs", src=_src_file("rs_read", rsr._srcs["rs_read.cpp"]), ct=list(read_k.compile_time_args),
                  defines=[], rt_by_rank=read, config=read_k.config)  # fmt: skip
        cp = dict(name="rs", src=_src_file("rs_reduce", rsr._srcs["rs_reduce.cpp"]), ct=list(red.compile_time_args),
                  defines=[], rt={l: [] for l in self.lanes}, compute=red.config, idle_ok=True)  # fmt: skip
        return dict(lane_dm0=snd, lane_dm1=rd, lane_c=cp)

    def _pcoef_part(self, sub: int) -> dict:
        """the mhc POST's K / L coefficient tiles rebuilt from DRAM posts[sub] / combs[sub] (member NCRISC, before the
        mhc sync part of the program that runs the post)"""
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        p = self.mp
        ct = [MHC.CB["PCOEF"], MHC.CB["PSTAGE"]] + acc(self.posts[sub]) + acc(self.combs[sub])
        rt = {m: [self.posts[sub].buffer_address(), self.combs[sub].buffer_address(), G // 8, p.step, p.nrb]
              for G, m in enumerate(self.members)}  # fmt: skip
        return dict(name="pcoef", src=os.path.join(ASM_KDIR, "pmk_pcoef_dram.cpp"), ct=ct, rt=rt, defines=[],
                    nonzero={m: (0, 1) for m in self.members})  # fmt: skip

    def _zero_part(self, cores: list, entries: list, name: str) -> dict:
        """the mhc's once-per-program init state for a later part: entries = [(cb, pages, bytes)] reserved + zeroed"""
        rt = {c_: [len(entries)] + [v for e in entries for v in e] for c_ in cores}
        return dict(name=name, src=os.path.join(ASM_KDIR, "pmk_cb_zero.cpp"), ct=[], rt=rt, defines=[])

    def _x_part(self, mode: int) -> dict:
        """TP8 (resident residual): load (1) / store (2) the member's block at a program boundary (member BRISC)"""
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        ct = [mode, MHC.CB["X"], MHC.CB["X32"]] + acc(self.x)
        rt = {m: [self.x.buffer_address(), G // 8, (G % 8) * MHC.CPC] for G, m in enumerate(self.members)}
        return dict(name={1: "xload", 2: "xstore", 3: "xattach"}[mode], src=os.path.join(ASM_KDIR, "pmk_x_io.cpp"), ct=ct, rt=rt,
                    defines=[], nonzero={m: (0,) for m in self.members})  # fmt: skip

    # ------------------------------------------------------------------------------------------------ register
    def _register(self):
        e, g, lay = self.engine, self.geom, self.layer
        members = self.members
        width = lambda part: max(
            len(v)
            for rt in (part["rt_by_rank"].values() if "rt_by_rank" in part else [part["rt"]])
            for v in rt.values()
        )  # noqa: E731
        # ---- mhc P1 / P10
        bodies = INT.mhc_bodies(self.mhc, self.x, self.hns, self.ys, self.posts, self.combs, flags=self.mhc_flags,
                                layer=0, fn_gate=None, taps=self.taps)  # fmt: skip
        for pid in ("P1", "P10"):
            body = bodies[pid]
            if (
                self.mhc_flags & MHC.F_XH_DRAM
            ):  # DRAM x-hat mode: mhc_sync's push targets are never read (259 -> ~211 args)
                sp_ = body.parts[("member", "dm1")]
                sp_["rt"] = {k_: list(v)[:5] for k_, v in sp_["rt"].items()}
                sp_["noc_xy"] = {k_: (0,) for k_ in sp_["rt"]}
                sp_["nonzero"] = {k_: () for k_ in sp_["rt"]}
        p1, p10 = bodies["P1"], bodies["P10"]
        # the roots' FINAL rendezvous (GLM53_PMK_ROOT_FINAL): the member -> root targets from the mhc sync args
        # [root_x, root_y, slot, flags, round0] (NoC coordinates), counted per root
        msync0 = p1.parts[("member", "dm1")]
        self.root_of = {m_: tuple(int(v) for v in msync0["rt"][m_][:2]) for m_ in members}
        noc_root = {self.noc(r_): r_ for r_ in self.roots}
        self.root_final_n = {r_: sum(1 for v in self.root_of.values() if noc_root.get(v) == r_) for r_ in self.roots}
        assert sum(self.root_final_n.values()) == len(members), (
            "every member's root is a root core",
            self.root_final_n,
        )
        if self.root_final:
            p1.parts[("member", "dm1")] = [dict(msync0, name="mhc"), self._fin_sig()]
            p1.parts[("root", "dm1")] = [dict(p1.parts[("root", "dm1")], name="mhc"), self._fin_wait()]
        h1 = {r: self._h1_targets(r) for r in range(self.n)}
        if (
            self.f_p1p2
        ):  # H1 in-program: P1's x-hat rows (DRAM) -> the holders' SEM_HN, 8 writer signals per holder (P2's
            # c1_stub_prod waits for them); the chain needed none (A2 started after A1's DRAM stores)
            sig1 = self._sig({}, SEM_HN, members, "h1sig")
            sig1["rt_by_rank"] = {r: self._sig(h1[r], SEM_HN, members, "h1sig")["rt"] for r in range(self.n)}
            sig1.pop("rt")
            sig1["noc_xy"] = {m: tuple(3 + 2 * i for i in range(8)) for m in members}
            p1.parts[("member", "dm0")] = [dict(p1.parts[("member", "dm0")], name="mhc"), sig1]
        lay.register_phase("P1", p1)
        # P10: [PCOEF from DRAM] [TP8: X load] mhc part [TP8: X store] [H7 fan-out: the x-hat_f rows are in DRAM]
        sig = self._sig({}, SEM_HN, members, "h7sig")
        sig["rt_by_rank"] = {r: self._sig(h1[r], SEM_HN, members, "h7sig")["rt"] for r in range(self.n)}
        sig.pop("rt")
        sig["noc_xy"] = {m: tuple(3 + 2 * i for i in range(8)) for m in members}
        dm0 = [dict(p10.parts[("member", "dm0")], name="mhc")]
        if self.mp.resident:  # TP8: the block pinned since A1 is ATTACHED (3), else reloaded (1); stored after (2)
            dm0 = [self._x_part(3 if self.tp8_resident else 1)] + dm0 + [self._x_part(2)]
        p10.parts[("member", "dm0")] = dm0 + [sig]
        # A5 runs the P10 part in a FRESH program: (a) the mhc's cumulative round counters idx = round0 + sub x NRB + k
        # + 1 (sub = 1 for the FFN pre) would wait G1 >= (NRB + 1) x P on semaphores that start at 0 -> round0 = -NRB
        # (uint32) folds them back to k + 1 (member M1 / M2, root G1 / G2); (b) the init() state of part P1 (F_INIT):
        # the members' zeroed PRE / INV, the root's zeroed PLANE / SCR / PCOEF, re-created here. The root's SB tiles are
        # NOT a prelude any more (mhc rev 6, README mhc §9f): CB_SB is PER SUBLAYER (hc_attn's and hc_ffn's scale / bias
        # rows differ; the compute pops the 3 pages after each sublayer's rounds), the mhc P10 root stream part loads
        # subs[1].sb from its runtime arg 2 (PmkMhc.parts: [sb(0), flags, sb(1)]) -- exactly ONE 3-page push per
        # sublayer (a prelude push on top of it, or F_SB_PRELUDE without one, blocks the 3-page CB forever)
        nrb = self.mp.nrb
        r0 = (-nrb) & 0xFFFFFFFF
        msync = dict(p10.parts[("member", "dm1")], name="mhc")
        rsync = dict(p10.parts[("root", "dm1")], name="mhc")
        if not self.f_p1p10:  # A15: P1 and P10 share the launch -> the counters continue from P1 (natural round0)
            msync["rt"] = {k_: list(v[:4]) + [r0] for k_, v in msync["rt"].items()}
            rsync["rt"] = {k_: list(v[:3]) + [r0] + list(v[4:]) for k_, v in rsync["rt"].items()}
        T32 = PMK.TILE_BYTES["fp32"]
        p10.parts[("member", "dm1")] = ([self._zero_part(members, [(MHC.CB["PRE"], 4, 4 * T32), (MHC.CB["INV"], 1, T32)], "init")]
                                        # A15: P1 pushed the post's NRB x 8 coefficient pages into the SHARED PCOEF 22 (not
                                        # re-laid out) and P10's post pops them: a rebuild would reserve a full ring
                                        + ([] if self.f_p1p10 else [self._pcoef_part(0)]) + [msync])  # fmt: skip
        p10.parts[("root", "dm1")] = [self._zero_part(self.roots, [(MHC.CB["PLANE"], 1, T32), (MHC.CB["SCR"], 2, 2 * T32),
                                                                   (MHC.CB["PCOEF"], nrb * 8, nrb * 8 * T32)], "init"), rsync]  # fmt: skip
        if self.root_final:  # the member's FINAL +1 is the last thing its sync RISC sends; the root waits all of them
            p10.parts[("member", "dm1")] = p10.parts[("member", "dm1")] + [self._fin_sig()]
            p10.parts[("root", "dm1")] = p10.parts[("root", "dm1")] + [self._fin_wait(uses=2 if self.f_p1p10 else 1)]
        p10.parts[("root", "dm0")] = [dict(p10.parts[("root", "dm0")], name="mhc")]  # loads sb(1): rev 6
        assert not (self.mhc_flags & MHC.F_SB_PRELUDE), "A5: the mhc P10 stream part loads sublayer 1's SB (no prelude)"
        p10.io = list(p10.io) + [self.x]
        lay.register_phase("P10", p10)
        # ---- P2 / P11: the AG of hn[0] / hn[1] (each the only fabric phase of its program: fab_at = its own width)
        rel = self.link_handover
        m2, s2, r2 = self._c1_parts(1, self.hns[0], 0, release=rel)
        w_c1 = width(s2)
        m2, s2, r2 = self._c1_parts(1, self.hns[0], w_c1, release=rel)
        w_c11 = width(
            self._c1_parts(1, self.hns[1], 0, release=self.n == 8 and (self.b_p11 or self.a15_p11) and rel)[1]
        )  # (alone, P11 is its program's only fabric phase; in the fused W7 TP8 program it hands link h back and forth)
        # W7 step 1: P11 before B's reclaim window -> the lane-read ack (the reclaim drain's H8 then covers the local
        # blocks' lane reads too, not only the peer packets)
        w7_tp8 = (
            self.n == 8 and (self.b_p11 or self.a15_p11) and rel and self.attn
        )  # P11 after P8's stage 2 on the same links
        g15_tp8 = (
            self.n == 8 and rel and not self.attn and self.b_p15
        )  # G v2b: P11 then P15's stage 2 (release, no handback)
        m11, s11, r11 = self._c1_parts(1, self.hns[1], w_c11, lane_ack=self.n == 2 and (self.b_p11 or self.a15_p11),
                                       release=w7_tp8 or g15_tp8, c1x_use=2 if w7_tp8 else 1, acquire=w7_tp8)  # fmt: skip
        self.w7_tp8_handback = w7_tp8
        if (
            self.b_p11 or self.a15_p11
        ):  # W7 step 1: P11's C1 send = P2's (identical args, the same connections): ONE arg block on the lane
            # BRISC composite (with two it carries 277 > 256 runtime args, mock 14:4xZ); active only when P2 is in the program
            s11 = dict(s11, rt_alias="P2")
        c1_uses = (("P2", (m2, s2, r2), self.hns[0]), ("P11", (m11, s11, r11), self.hns[1]))
        for pid, (mem, snd, rd), hn in c1_uses if self.attn else c1_uses[1:]:
            lay.register_phase(pid, PMK.PhaseBody(parts={("member", "dm1"): mem, ("lane", "dm0"): snd, ("lane", "dm1"): rd},
                                                  io=[hn, self.c1.sync]))  # fmt: skip
        if self.attn:
            self._register_attn(lay, width)
        if self.b_p12:
            self._register_p12(lay)
        if self.b_p13:
            self._register_p13(lay)
        if (
            self.b_p15 and self.n == 2
        ):  # (TP8: after program B -- its RS8 CBs on the MM-only cores join B's reclaim window)
            self._register_p15(lay)
        # ---- the fabric connections after each program's descriptor is built (the lane BRISC composite's index);
        # ONE fabric phase per program (tt-fabric: one worker connection per EDM sender channel per link at a time)
        for (
            sc
        ) in (
            self.scopes
        ):  # stage 2: P2 then P8 on the lanes in sequence (serial open / close), fab_at patched by _connect
            fab = {"P2", "P8", "P11", "P15"} & set(sc)
            assert (
                len(fab) <= 1
                or (
                    fab == {"P2", "P8"}
                    and set(sc)
                    in (set(A2F_SCOPE), set(A15_SCOPE), set(P1A2F_SCOPE), set(A2FP10_SCOPE), set(A15B_SCOPE))
                )
                or (
                    fab == {"P2", "P8", "P11"}
                    and set(sc) in (set(A15B11_SCOPE), set(A15P11_SCOPE), set(A15B13_SCOPE), set(A15B12_SCOPE))
                )
                or (fab == {"P2", "P8", "P11", "P15"} and set(sc) == set(A15B15_SCOPE))
                or (fab == {"P11", "P15"} and set(sc) == set(DSA_G15_SCOPE))
            ), f"two fabric phases in one program: {sc}"
        e.post_build = [self._connect]
        self.report["wait_schedule"] = e.wait_schedule()
        self.report["fabric_headers"] = self._fabric_header_audit()

    def _o_rect_kw(self) -> dict:
        """PMK_O_RECTS: o_proj's 4 row-group rectangles and their root streamers (2 per rectangle, in its rows)"""
        if not self.o_rects:
            return {}
        per = len(self.members) // self.o_rects
        rects = [self.members[per * h : per * (h + 1)] for h in range(self.o_rects)]
        st = []
        for rect in rects:
            ys = sorted({y for _, y in rect})
            pair = [r_ for r_ in self.roots if r_[1] in ys]
            assert len(pair) == 2, ("each o_proj rectangle: the 2 roots beside its rows", rect, pair)
            st += pair
        return dict(rects=rects, streamers=st)

    def _o_wait_rt(self) -> dict:
        """PMK_O_RECTS: member of rectangle h waits SEM_ON >= (h + 1) x the chunks per rectangle"""
        nch = int(self.kphase.geo.n_chunks)
        per = len(self.members) // self.o_rects
        assert nch == len(self.members), ("one chunk per member row tile", nch, len(self.members))
        return {m_: (i // per + 1) * per for i, m_ in enumerate(self.members)}

    def _fin_sig(self) -> dict:
        """the member's FINAL +1 on its mhc root (pmk_sig.cpp: after this RISC's write / read barriers)"""
        return self._sig({m_: [self.root_of[m_]] for m_ in self.members}, SEM_ROOT_FINAL, self.members, "fin")

    def _fin_wait(self, uses: int = 1) -> dict:
        """the root's FINAL wait: every member that sends to this root has sent its last word (pmk_wait.cpp); uses = the
        rendezvous so far in this launch (A15: P1's and P10's on the one counter)"""
        rt = {r_: [1, SEM_ROOT_FINAL, int(self.root_final_n[r_]) * int(uses)] for r_ in self.roots}
        return dict(name="fin", src=os.path.join(ASM_KDIR, "pmk_wait.cpp"), ct=[], rt=rt, defines=[])

    def _register_attn(self, lay, width):
        """the attention programs' phases A2..A4 (P3 projections, P4 / P6 kda, P7 o_proj, P8 RS)"""
        # ---- P3: the four projection phases (QBASE / sync slot continue; the K = 512 phases on their own PARK)
        qbase, pts, used = 0, [], []
        pushed = (
            {}
        )  # PARK CB -> the blocks pushed through it so far in this program (its ring position at a phase start)
        names3 = ("qkv", "lowrank", "fb", "gb")
        xs = [None, None, self.mm_lr.y, self.mm_lr.y]
        for i, (mm, x_, nm) in enumerate(zip(self.mms3, xs, names3)):
            if nm not in self.p3_phases:
                continue
            qbase = self._qbase(qbase, mm, pushed, nm)
            kw = dict(qbase=qbase, slot=i)  # sync slot i: TOKEN / GEN / PARKADDR / VALIDADDR / READY are per phase
            if x_ is not None:
                kw.update(x=x_, x_fill=True)
            pts.append(mm.parts(mm.y, **kw))
            used.append((mm, nm, i))
            qbase += mm.blocks
        # no two stream phases of one program may share a sync slot (16's check, 20:5xZ); the o_proj phase is slot 4
        slots = [i for _, _, i in used] + [4]
        assert len(set(slots)) == len(slots) and max(slots) < 5, slots
        parts3 = {}
        for pt, (mm, nm, _) in zip(pts, used):
            for key, part in self._pfs_parts(pt, mm, nm).items():
                parts3.setdefault(key, []).append(part)
        if (
            self.h3_inprog
        ):  # the H3 barrier after the last projection's stores (member BRISC = the PfStream writer RISC)
            sig3, agg3 = self._h3_barrier_parts()
            parts3[("member", "dm0")].append(sig3)
            parts3.setdefault(("root", "dm1"), []).append(agg3)
        io3 = (
            [t for mm in self.mms3 for t in (mm.weight, mm.y)]
            + [self.sync]
            + [mm.mprobe for mm in self.mms3 if mm.probe]
        )
        lay.register_phase("P3", PMK.PhaseBody(parts=parts3, io=io3))
        # ---- P4 / P6: the kda items / scan (H3 pre-raised: the stores are DRAM, written by the previous program)
        kb = INT.kda_bodies(self.kphase, self.scr, self.state.S, self.state.C, self.valid_end)
        lay.register_phase("P4", kb["P4"])
        lay.register_phase("P6", kb["P6"])
        # ---- P7 / P8: o_proj (x = the scan's `on` rows, DRAM) with the H5 signals; the pair's RS (lanes reduce)
        # (P7's PARK ring: the P3 phases' pushes count only when they are in the same program -- a launch resets the CBs)
        # stage 2: P3 and P7 share the program but not the window, so o_proj's PARK is re-placed in its own window and
        # starts on slot 0 again (a fresh ring: no pushes counted from P3)
        same_ring = any("P3" in sc and "P7" in sc for sc in self.scopes) and not self.a2_fuse
        qbase = self._qbase(qbase, self.mm_o, pushed if same_ring else {}, "o_proj")
        if self.rsr is not None or self.rs8 is not None:
            if self.rsr is not None:
                rs0 = self._rs_parts(0)
                w_rs = width(rs0["lane_dm0"])
                rs = self._rs_parts(w_rs)
                parts8 = {("lane", "dm0"): rs["lane_dm0"], ("lane", "dm1"): rs["lane_dm1"], ("lane", "c"): rs["lane_c"]}
                if (
                    self.f_p8p10
                ):  # H6 in-program: each lane +1 FS_REDUCED on every member and root after its y rows landed
                    tg = [self.noc(c_) for c_ in list(self.members) + list(self.roots)]
                    parts8[("lane", "dm0")] = [rs["lane_dm0"], self._sig({l_: tg for l_ in self.lanes}, PMK.SEM_FS_REDUCED,
                                                                         self.lanes, "h6sig")]  # fmt: skip
                io8 = [self.mm_o.y, self.rsr.r, self.rsr.y, self.rsr.sync]
                sig = {r: self._rs_signals(r) for r in range(self.n)}
            else:  # TP8: the fabric args of the two send kernels land after their composites' own args (two-pass)
                # homelab-1 01:22Z (watched, c4eb61ea7b): the tail BRISC composite carries TWO sub-parts (rsl_send +
                # the stage-2 red_write of the reducers on tail cores) and connect() appends the fabric args after the
                # WHOLE composite -> fab_at = the sum of its sub-part widths, not the first part's (rsl_send then read
                # red_write2's zero slot as its connection: open() wrote 4 B to NoC (0, 0) DRAM[0])
                cw = lambda v: sum(width(x) for x in (v if isinstance(v, list) else [v]))  # noqa: E731
                rs0 = self._rs8_parts(0, 0)
                fab1, fab2 = cw(rs0[("lane", "dm0")]), cw(rs0[("tail", "dm0")])
                parts8 = self._rs8_parts(fab1, fab2)
                assert (cw(parts8[("lane", "dm0")]), cw(parts8[("tail", "dm0")])) == (fab1, fab2)
                rs1, rs2 = self.rs8.rs1, self.rs8.rs2
                io8 = [self.mm_o.y, rs1.r, rs1.y, rs1.sync, rs2.r2, rs2.y, rs2.sync]
                sig = {r: self._rs8_signals(r) for r in range(self.n)}
            owait = dict(wait=(INT.KDA.SEM_ON, 0), wait_rt=self._o_wait_rt()) if self.o_rects else {}
            pt_o = {r: self.mm_o.parts(self.mm_o.y, x=self.scr["on"], x_fill=True, qbase=qbase, slot=4,
                                       signals=sig[r], **owait) for r in range(self.n)}  # fmt: skip
            parts7 = self._pfs_parts(pt_o[0], self.mm_o, "o", rt_by_rank={r: pt_o[r]["rt"] for r in range(self.n)})
            lay.register_phase("P8", PMK.PhaseBody(parts=parts8, io=io8))
        else:
            owait = dict(wait=(INT.KDA.SEM_ON, 0), wait_rt=self._o_wait_rt()) if self.o_rects else {}
            pt_o = self.mm_o.parts(self.mm_o.y, x=self.scr["on"], x_fill=True, qbase=qbase, slot=4, **owait)
            parts7 = self._pfs_parts(pt_o, self.mm_o, "o")
        io7 = [self.mm_o.weight, self.mm_o.y, self.scr["on"]] + ([self.mm_o.mprobe] if self.mm_o.probe else [])
        lay.register_phase("P7", PMK.PhaseBody(parts=parts7, io=io7))
        self.qbase_end = qbase + self.mm_o.blocks  # the next stream phase of the launch continues from here (P13)

    def _register_p12(self, lay):
        """W7 step 3: P12 = the router logits PfStream phase (sync slot 7; its own member window, a fresh PARK ring; the
        block base continuing after o_proj) + the router tail on every member over its 32 rows (reader on the member
        BRISC after the PfStream writer that stored the logits: program order + a write barrier; compute on the TRISC;
        writer on the NCRISC -> B's ids / weights rows). Waits H8 (the x-hat_f block in XS); signals nothing: B reads the
        ids / weights after its rendezvous (every member's tail finished)."""

        pushed = {}
        q = self._qbase(self.qbase_end, self.rt_mm, pushed, "p12_logits")
        pt = self.rt_mm.parts(self.rt_mm.y, qbase=q, slot=P12_SLOT)
        self.qbase_end = q + self.rt_mm.blocks  # P13 continues after the router's blocks
        parts12 = {}
        for key, part in self._pfs_parts(pt, self.rt_mm, "rt_mm").items():
            parts12.setdefault(key, []).append(part)
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        cbd = [("RTAIL_CB_MAP", "1")] + [(f"RTAIL_CB_{k_}", str(v_)) for k_, v_ in P12_TAIL_CB.items()]
        tdefs = cbd + [("RTAIL_ROWTILES", "1")]
        arg = {m_: (32 * i) | (31 << 16) for i, m_ in enumerate(self.members)}  # member i = rows 32 i .. 32 i + 31
        logits, bias, ids, wt = self.rt_mm.y, self.rt_bias, self.rt_ids, self.rt_wt
        f32b = lambda v: struct.unpack("<I", struct.pack("<f", float(v)))[0]  # noqa: E731
        rd = dict(name="rtail", src=_src_file("rtail_reader", _rtail_text("reader.cpp")),
                  ct=acc(logits) + acc(bias), defines=tdefs + [("RTAIL_PREFILL", "1")],
                  rt={m_: [logits.buffer_address(), bias.buffer_address(), arg[m_]] for m_ in self.members})  # fmt: skip
        cc = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        )
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[P12_TAIL_CB["LOG"]] = modes[P12_TAIL_CB["BIAS"]] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        cp = dict(name="rtail", src=_src_file("rtail_compute", _rtail_text("compute.cpp")), ct=[],
                  defines=tdefs, compute=cc, dst_any=True, rt={m_: [arg[m_]] for m_ in self.members})  # fmt: skip
        # GLM53_PMK_ROW_MASK=1 (coordinator 12:10:42Z): rows >= the chunk's valid_end -> the sentinel expert, weight 0
        # (_rtail_writer_row_mask; W7's B and G both read these rows); its address at rt 4, the plan signals after it
        self.row_mask = os.environ.get("GLM53_PMK_ROW_MASK", "0") == "1"
        w_txt = _rtail_text("writer.cpp")
        w_defs = tdefs + [("RTAIL_WEIGHTS", "1"), ("RTAIL_W_RT", "3"), ("RTAIL_RM_OUT", "1")]
        w_rt = {m_: [ids.buffer_address(), ids.buffer_address(), arg[m_], wt.buffer_address()] for m_ in self.members}
        if self.row_mask:
            assert self.valid_end is not None, "GLM53_PMK_ROW_MASK: the program needs the chunk's valid_end tensor"
            w_txt = _rtail_writer_row_mask(w_txt)
            w_defs = w_defs + [("RTAIL_VE_RT", "4")]
            w_rt = {m_: v_ + [self.valid_end.buffer_address()] for m_, v_ in w_rt.items()}
        sig_rt = 5 if self.row_mask else 4
        wr = dict(name="rtail", src=_src_file("rtail_writer_vm" if self.row_mask else "rtail_writer", w_txt),
                  ct=acc(ids) + acc(ids) + acc(wt) + [f32b(self.rt_scaling), f32b(self.rt_eps)],
                  defines=w_defs, rt=w_rt)  # fmt: skip
        pe_b = (self.b_window or {}).get("b") if isinstance(getattr(self, "b_window", None), dict) else None
        pe = getattr(pe_b, "plan_early", None) if pe_b is not None else None
        if pe:  # Stage 14 B2.3 step (2): every member's writer +1s the early-plan cores after its ids rows landed
            sig_xy = [v for c_ in pe_b.cores[: pe_b.plan_p] for v in pe_b.noc_of[tuple(c_)]]
            wr["defines"] = list(wr["defines"]) + [
                ("RTAIL_PLAN_SIG_SEM", str(int(pe["sig"]))),
                ("RTAIL_PLAN_SIG_RT", str(sig_rt)),
            ]
            wr["rt"] = {m_: list(v_) + [pe_b.plan_p] + sig_xy for m_, v_ in wr["rt"].items()}
            assert int(pe["sig_n"]) == len(self.members), (
                "the plan waits one +1 per member writer",
                pe["sig_n"],
                len(self.members),
            )
        if _knob("PMK_P12_TAIL", 1):  # (=0: the logits phase alone, the cost split A/B; B must then read staged routes)
            parts12[("member", "dm0")].append(rd)
            parts12[("member", "c")].append(cp)
            parts12[("member", "dm1")].append(wr)
        if self.xext_on:  # P11x after the tail on each RISC: DM1 re-pushes XS, the TRISC untilizes, DM0 writes the rows
            kt_x = self.geom.hidden // PMK.TILE
            ct_x = [PMK.CB_XS_LAND, CB_XEXT, kt_x, XEXT_BW]
            xsrc = lambda n_: os.path.join(PMK.KDIR, "pmk", n_)  # noqa: E731
            # a bf16 copy is exact in either DEST width: no fp32 request of its own (the composite's union decides)
            cc_x = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                fp32_dest_acc_en=False)  # fmt: skip
            # member i holds token rows [32 i, 32 i + 32): only members with rows run (x_ext has C + 32 rows); the
            # three parts take the same run flag (XS push / pop stay balanced)
            run_x = {m_: int(TILE * i < self.c) for i, m_ in enumerate(self.members)}
            parts12[("member", "dm1")].append(dict(name="xext", src=xsrc("xext_dm.cpp"), ct=list(ct_x),
                                                   defines=[("XEXT_PUSH", "1")], rt={m_: [run_x[m_]] for m_ in self.members}))  # fmt: skip
            # dst_any: <= 4 fp32 DST tiles per acquire (BW 4), either sync mode; fid "any": a copy (no fidelity term)
            parts12[("member", "c")].append(dict(name="xext", src=xsrc("xext_c.cpp"), ct=list(ct_x), defines=[],
                                                 compute=cc_x, dst_any=True, fid_mode="any",
                                                 rt={m_: [run_x[m_]] for m_ in self.members}))  # fmt: skip
            parts12[("member", "dm0")].append(dict(name="xext", src=xsrc("xext_dm.cpp"), ct=list(ct_x) + acc(self.xext),
                                                   defines=[("XEXT_WRITE", "1")],
                                                   rt={m_: [run_x[m_], self.xext.buffer_address(), TILE * i] for i, m_ in enumerate(self.members)}))  # fmt: skip
        p12 = next(p for p in self.engine.phases if p.pid == "P12")
        assert tuple(p12.waits) == ("H8",), p12.waits
        p12.signals = ()
        io12 = (
            [self.rt_mm.weight, logits, bias, ids, wt, self.sync]
            + ([self.xext] if self.xext_on else [])
            + ([self.valid_end] if self.row_mask else [])
        )
        lay.register_phase("P12", PMK.PhaseBody(parts=parts12, io=io12))

    def _p15_h9(self, b_y) -> dict:
        """PMK_P15_STREAM at TP2 (the RSR on A's lanes): {rank: dict(targets=[(lane NoC x, y, sync base)], table=[(lane
        index, word)] per 32-row block of B's y)} -- block j is SENT by lane h at step i (word LREADY + i) or REDUCED at
        step i (OWNRDY + i), read off P15's own rs_read lists (ready mode)"""
        rsr = self.rsr15
        nblk = self.c // PMK.TILE
        out = {}
        for rank in range(self.n):
            r, c = divmod(rank, self.C)
            ks, _, _, _ = rsr._build(b_y, r, c, ready=True, fab_at=0)
            read = _rt_dict(ks[1].runtime_args, self.lanes)
            targets, table, sends = [], [None] * nblk, []
            for li, l_ in enumerate(self.lanes):
                v = read[l_]
                sync, ns, nr = int(v[0]), int(v[7]), int(v[8])
                send_j, red = v[9 : 9 + ns], v[9 + ns : 9 + ns + 2 * nr]
                x_, y_ = self.noc(l_)
                targets.append((x_, y_, sync))
                for i_, j_ in enumerate(send_j):
                    table[int(j_)] = (li, PMK.FS_WORDS["LREADY"] + i_)
                    sends.append(int(j_))
                for i_ in range(nr):
                    table[int(red[2 * i_])] = (li, PMK.FS_WORDS["OWNRDY"] + i_)
            assert all(t_ is not None for t_ in table), ("P15 stream: a row block with no lane", rank, table)
            # the combine's start on this chip: the first block it SENDS (PMK_COMB_STRIDE: that half lands in round 1)
            h_split = self._p15_ffn()[1]
            out[rank] = dict(targets=targets, table=table, uoff=h_split * (min(sends) if sends else 0))
        return out

    def _p15_ffn(self):
        """(the FFN partial P15 reduces, its combine's h_split, the combine's cores): program B's (W7) or G's (G v2b)"""
        b = self.b_window["b"] if self.b_window is not None else self.g_window["g"]
        return b.scratch(self.c)["y"], int(b.geometry(self.c).h_split), [tuple(x) for x in b.cores]

    def _p15_comb_cores(self) -> list:
        """the cores whose combine writes the FFN partial P15 reads: program B's units are dealt over ALL of b.cores (B7g
        at TP8: 104 = roots + MM-only + members) and B ends without a barrier, so the non-stream rendezvous needs every
        one of them (PMK_P15_ALL_SIGS=1, coordinator 01:56:05Z); =0: the members only (the rendezvous before 01:54Z,
        blind to an MM-only / root combine core's late rows -- the poisoned-y arm's control)"""
        if not _knob("PMK_P15_ALL_SIGS", 1):
            return list(self.members)
        return self._p15_ffn()[2]

    def _p15_sig_parts(self, targets: list, sem: int, name: str) -> dict:
        """{(role, dm0 / dm1): pmk_sig part} -- every P15 combine core (_p15_comb_cores) +1 `sem` on each target from both
        DM RISCs after its P14 (pmk_sig: after this RISC's own write barriers); the role's other cores idle"""
        comb = set(self._p15_comb_cores())
        rc = self.engine.place.role_cores()
        out = {}
        for role in ("member", "mm_only", "root"):
            cores = [tuple(c_) for c_ in rc.get(role, [])]
            mine = [c_ for c_ in cores if c_ in comb]
            if not mine:
                continue
            for risc, sfx in (("dm0", "a"), ("dm1", "b")):
                out[(role, risc)] = dict(self._sig({c_: targets for c_ in mine}, sem, cores, f"{name}{sfx}"), noc_sem=sem,
                                         idle_ok=True)  # fmt: skip
        assert sum(sum(1 for v in p["rt"].values() if v[0]) for k, p in out.items() if k[1] == "dm0") == len(
            comb
        ), "P15 signals"
        return out

    def _p15_h9_rs8(self, b_y) -> dict:
        """PMK_P15_STREAM at TP8 (P15 = the second RS8): {rank: dict(targets, table, uoff)} -- targets = the stage-1 lanes
        then reducers1 (their shared sync shard address), table[j] = the stage-1 readiness word of B's row block j, the
        map P8's H5 uses (_rs8_signals): mesh row j // bl == mine -> reducer b % nr, OWNRDY + b // nr (b = j % bl), else
        lane b % nl, LREADY + b // nl (sent to the board partner); uoff = h_split x the first block this chip SENDS"""
        rs1 = self.rs8_15.rs1
        sync = rs1.sync.buffer_address()
        nl, nr, bl = len(rs1.lanes), len(rs1.reducers), rs1.bl
        targets = [(*self.noc(l_), sync) for l_ in rs1.lanes] + [(*self.noc(c_), sync) for c_ in rs1.reducers]
        h_split = self._p15_ffn()[1]
        nblk = self.c // PMK.TILE
        assert nblk == self.R * bl, ("P15 RS8 table: B's row blocks must be RS8's (2 x its bl)", nblk, self.R, bl)
        out = {}
        for rank in range(self.n):
            r = rank // self.C
            table, sends = [], []
            for j in range(nblk):
                rr, b_ = divmod(j, bl)
                if rr == r:
                    table.append((nl + b_ % nr, PMK.FS_WORDS["OWNRDY"] + b_ // nr))
                else:
                    table.append((b_ % nl, PMK.FS_WORDS["LREADY"] + b_ // nl))
                    sends.append(j)
            out[rank] = dict(targets=targets, table=table, uoff=h_split * (min(sends) if sends else 0))
        return out

    def _register_p15(self, lay):
        """W7 step 4: P15 = the RS of program B's y partial ([C, H] bf16 TILE, DRAM) to the row owners: the second RSR
        (send / read / reduce on A's lanes, P8's connections via fab_at -> y15 [C / 2, H] fp32 per chip). Ordering:
        every member +1 lane semaphore SEM_P15 from BOTH data-movement RISCs after its P14 (pmk_sig: after its own write
        barriers -- B's y rows of that core are in DRAM); the lanes wait 2 x members before reading B's y (ready=False)
        """
        b_y = self._p15_ffn()[0]  # program B's y (W7) or G's (G v2b)
        rsr = self.rsr15
        stream = bool(
            getattr(self, "p15_stream", False)
        )  # PMK_P15_STREAM: per-block readiness from B's combine (H9 table)
        rel = {}
        for rank in range(self.n):
            r, c = divmod(rank, self.C)
            ks, _, _, _ = rsr._build(b_y, r, c, ready=stream, fab_at=0)
            rel[rank] = ks
        ks0 = rel[0]
        send = {rk: _rt_dict(rel[rk][0].runtime_args, self.lanes) for rk in rel}
        read = {rk: _rt_dict(rel[rk][1].runtime_args, self.lanes) for rk in rel}
        # FS_RS_ARITH: the send's two row lists (lane h: blocks h, h + nl, ..) computed in the kernel -> args 0..12 + fab_at
        nl_, bl_ = len(self.lanes), self.bl
        for rk in send:
            for h_, l_ in enumerate(self.lanes):
                v = send[rk][l_]
                assert v[5] == nl_ and v[6] == h_ and v[11] == v[12] == bl_ // nl_, ("RSR send header", v[:13])
                assert v[13 : 13 + 2 * (bl_ // nl_)] == [b for b in range(bl_) if b % nl_ == h_] * 2, "arithmetic rows"
                send[rk][l_] = v[:13] + [13]
        n_snd = len(next(iter(send[0].values())))
        snd = dict(name="rs15", src=_src_file("rs_send", rsr._srcs["rs_send.cpp"]), ct=list(ks0[0].compile_time_args),
                   defines=[("FS_HDR_CB", str(CB_HDR)), ("FS_RS_ARITH", "1")] + ([("FS_SEND_FIRST", "1")] if stream else []),
                   rt_by_rank=send, config=ks0[0].config,
                   fab_at_arg=n_snd - 1,
                   noc_xy={l: (1,) for l in self.lanes}, nonzero={l: (0, 9, 10) for l in self.lanes})  # fmt: skip
        h_split = self._p15_ffn()[1]
        rd = dict(name="rs15", src=_src_file("rs_read", rsr._srcs["rs_read.cpp"]), ct=list(ks0[1].compile_time_args),
                  defines=[("FS_RDY_MULT", str(h_split)), ("FS_SEND_FIRST", "1")] if stream else [], rt_by_rank=read,
                  config=ks0[1].config)  # fmt: skip
        cp = dict(name="rs15", src=_src_file("rs_reduce", rsr._srcs["rs_reduce.cpp"]), ct=list(ks0[2].compile_time_args),
                  defines=[], rt={l: [] for l in self.lanes}, compute=ks0[2].config, idle_ok=True)  # fmt: skip
        n_sig = 2 * len(self._p15_comb_cores())
        wt = lambda nm: dict(name=nm, src=os.path.join(ASM_KDIR, "pmk_wait.cpp"), ct=[],  # noqa: E731
                             rt={l: [1, SEM_P15, n_sig] for l in self.lanes}, defines=[])  # fmt: skip
        lanes_xy = [self.noc(l) for l in self.lanes]
        if stream:  # no rendezvous: each block waits its own readiness word (h_split combine units, the H9 table)
            parts15 = {("lane", "dm0"): [snd], ("lane", "dm1"): [rd], ("lane", "c"): [cp]}
        else:  # every combine core signals (the lanes hold connection words 10..: the audit checks id 5 is not one)
            parts15 = {("lane", "dm0"): [wt("w15a"), snd], ("lane", "dm1"): [wt("w15b"), rd], ("lane", "c"): [cp]}
            for key, part in self._p15_sig_parts(lanes_xy, SEM_P15, "s15").items():
                parts15[key] = [part]
        p15 = next(p for p in self.engine.phases if p.pid == "P15")
        p15.roles = tuple(
            dict.fromkeys(
                tuple(p15.roles) + (() if stream else tuple(dict.fromkeys(k[0] for k in parts15 if k[0] != "lane")))
            )
        )
        if stream:
            p15.roles = tuple(r_ for r_ in p15.roles if r_ not in ("member", "mm_only", "root"))
        p15.waits, p15.signals = (), ()  # (the ordering is the explicit signal / wait pair above, or the H9 table)
        if not stream:
            self.engine._sem(SEM_P15, "lane")
        lay.register_phase("P15", PMK.PhaseBody(parts=parts15, io=[b_y, rsr.r, rsr.y, rsr.sync]))

    def _i1_patch_xs_base(self):
        """W7 (I1): B's lever-2 gather reads the holders' XS at its absolute L1 address = CB_BASE + the shared bytes + XS's
        offset in the window before the reclaim (the reclaim keeps those bytes: reclaim_protect); patch B's NCRISC args
        """
        e = self.engine
        old = e.scope
        e.scope = set(self.scopes[0])
        try:
            wl = e._win_layout("member")
        finally:
            e.scope = old
        assert wl and wl.get("reclaim"), "W7 (I1): the member's reclaim window"
        prot = (getattr(e, "reclaim_protected", None) or {}).get(("member", PMK.CB_XS_LAND))
        assert prot is not None, "W7 (I1): XS was not protected in the reclaim layout"
        xs_base = PMK.CB_BASE + prot[0]
        p14 = next(q for q in e.phases if q.pid == "P14").body
        n_hit = 0
        for key, sub in (p14.parts or {}).items():
            for s_ in sub if isinstance(sub, list) else [sub]:
                for d in s_["rt_by_rank"].values() if "rt_by_rank" in s_ else [s_["rt"]]:
                    for v in d.values():
                        for j, a_ in enumerate(v):
                            if a_ == XS_SENTINEL:
                                v[j] = xs_base
                                n_hit += 1
        assert n_hit >= len(self.members), ("W7 (I1): B's xs_base sentinel patched on every receiver", n_hit)
        self.report["i1"] = dict(xs_base=xs_base, xs_bytes=prot[1], patched=n_hit)

    def _p15_reclaim_cbs(self) -> dict:
        """W7 P15 at TP8: the second RS8's reducer CBs on a WINDOWED reducer role (MM-only) live in B's reclaim window
        (P15 runs after P14 there): indices B's reclaim allocations do not take -- RS8's own Y1 / T0..T3 / Y2 when free
        of B, two fresh (else B-free bf16) indices for A / B -- appended to the role's reclaim allocations (+ fresh
        declarations). Returns {(role, stage): remap} for _rs8_parts (the non-windowed roles keep RS8_CB in place)"""
        e, I, out = self.engine, RS8_CB, {}
        for role in sorted(set(self.rs8_roles[1]) | set(self.rs8_roles[2])):
            if role not in e.window_roles:
                continue
            rec_b = (e.win_reclaim or {}).get(role)
            assert rec_b is not None, (
                f"P15 at TP8: reducer role {role} is windowed but has no reclaim window",
                e.window_roles,
            )
            b_idx = {i for ents, _ in rec_b["allocs"] for i, _ in ents}
            if (
                getattr(self, "b7g", False) or self.g_window is not None
            ):  # B7g / G v2b: P15's own (second) reclaim window
                e.win_reclaim2 = dict(getattr(e, "win_reclaim2", None) or {})
                rec = e.win_reclaim2.setdefault(role, dict(pid="P15", allocs=[], fresh=[]))
            else:
                rec = rec_b
            used = PBW._used(e, role)
            d = dict(e.cbs.get(role, {}))
            free = [i for i in range(63, -1, -1) if i not in used and i not in b_idx]
            reserved = set(free[:4])  # the window barriers / pad (pmk_layer._win_layout's free[:4])
            fresh_pool = [i for i in sorted(free) if i not in reserved]
            take = {}
            # RS8's own Y1 / T0..T3 / Y2 when the first reclaim (B's / G's) leaves them, else free ones (G v2b: G's reclaim
            # takes 0..18 on MM-only): T0..T3 stay CONSECUTIVE (rsl_red_read / rsl_reduce address T0 + j)
            t0 = I["T0"]
            if any(t0 + j in b_idx for j in range(4)):
                t0 = next((i for i in fresh_pool if all(i + j in fresh_pool for j in range(4))), None)
                assert t0 is not None, (
                    f"P15 at TP8: no 4 consecutive free indices for RS8 T0..T3 on {role}",
                    fresh_pool,
                )
            for j in range(4):
                take[f"T{j}"] = t0 + j
            for nm in ("Y1", "Y2"):
                i = I[nm]
                if i in b_idx or i in take.values():
                    i = next((x for x in fresh_pool if x not in take.values()), None)
                    assert i is not None, (f"P15 at TP8: no free index for RS8 {nm} on {role}", fresh_pool, take)
                take[nm] = i
            for nm in ("A", "B"):
                cand = [i for i in fresh_pool if i not in take.values()] or [
                    i
                    for i in sorted(d)
                    if i not in b_idx and i not in take.values() and d[i].fmt == "bf16" and i not in reserved
                ]
                assert cand, (f"P15 at TP8: no index for RS8 {nm} on {role}", sorted(b_idx), sorted(used))
                take[nm] = cand[0]
            sizes = dict(A=("bf16", 2048, 32), B=("bf16", 2048, 32), Y1=("fp32", 4096, 32),
                         T0=("fp32", 4096, 16), T1=("fp32", 4096, 16), T2=("fp32", 4096, 16), T3=("fp32", 4096, 16),
                         Y2=("fp32", 4096, 16))  # fmt: skip
            for nm, (fmt, page, pages) in sizes.items():
                i = take[nm]
                rec["allocs"].append(([(i, page)], page * pages))
                if i not in used:
                    rec.setdefault("fresh", []).append(PMK.CBUse(index=i, fmt=fmt, pages=1, live=("P15",), page=page,
                                                                 note=f"P15 RS8 {nm}", users=("P15",)))  # fmt: skip
            t0 = take["T0"]
            assert [take[f"T{j}"] for j in range(4)] == [t0 + j for j in range(4)], take
            out[(role, 1)] = {2: {2: take["A"], 3: take["B"]}, 3: {2: take["A"], 3: take["B"], 4: take["Y1"]},
                              4: {2: take["Y1"]}}  # fmt: skip
            out[(role, 2)] = {2: {2: t0}, 3: {2: t0, 3: take["Y2"]}, 4: {2: take["Y2"]}}
            self.report.setdefault("p15_reclaim", {})[role] = dict(take)
        return out

    def _register_p15_tp8(self, lay, b_y):
        """W7 step 4 at TP8: P15 = a second RS8 (stage 1 axis 0 on the lanes + reducers1, stage 2 axis 1 on the tails +
        reducers2) over program B's y partial; its connections = P8's (lanes: fab_at -> P8's stage-1 slot; tails: both
        stage-2 slots marked at connect time, the composite holding two fabric parts); stage 2 opens after C1X lane h's
        SECOND release (P11's). Ordering: every member +1 SEM_P15_TP8 on the lanes and the stage-1 reducers from both DM
        RISCs after its P14; the lanes' rs_read and the reducers' red_read wait 2 x members before reading B's y"""
        e, sem, n_sig = self.engine, SEM_P15_TP8, 2 * len(self._p15_comb_cores())
        role_remap = self._p15_reclaim_cbs()
        width = lambda v: sum(
            max(
                len(x)
                for rt in (q["rt_by_rank"].values() if "rt_by_rank" in q else [q["rt"]])  # noqa: E731
                for x in rt.values()
            )
            for q in (v if isinstance(v, list) else [v])
        )
        stream = bool(
            getattr(self, "p15_stream", False)
        )  # PMK_P15_STREAM: stage 1 waits B's combine per block (H9 table)
        h_split = self._p15_ffn()[1]
        n_c1x = sum(1 for p_ in ("P2", "P11") if any(p_ in s_ for s_ in self.scopes))  # C1X lane h's releases before it
        kw = dict(
            rs8=self.rs8_15,
            src=b_y,
            ready=stream,
            sfx="15",
            after_n=n_c1x,
            h6=False,
            handback=False,
            role_remap=role_remap,
            rdy_mult=h_split if stream else 1,
        )
        p0 = self._rs8_parts(0, 0, **kw)
        parts = self._rs8_parts(width(p0[("lane", "dm0")]), width(p0[("tail", "dm0")]), **kw)
        targets = [self.noc(l_) for l_ in self.lanes] + [self.noc(c_) for c_ in self.red1]
        rc = e.place.role_cores()

        def wt(nm, role, active):
            cores = [tuple(c_) for c_ in rc[role]] if role != "lane" else list(self.lanes)
            return dict(name=nm, src=os.path.join(ASM_KDIR, "pmk_wait.cpp"), ct=[], defines=[], idle_ok=True,
                        rt={c_: [1 if c_ in active else 0, sem, n_sig] for c_ in cores})  # fmt: skip

        def pre(key, w_):  # the wait part first in the (role, risc) list
            v = parts.get(key)
            parts[key] = [w_] + (v if isinstance(v, list) else [v] if v is not None else [])

        if not stream:  # the rendezvous (streamed: each block waits its own readiness word, raised by B's combine)
            pre(("lane", "dm1"), wt("w15l", "lane", set(self.lanes)))
            for role in self.rs8_roles[1]:
                pre((role, "dm1"), wt(f"w15r_{role}", role, {c_ for c_ in self.red1 if self.rs8_role[c_] == role}))
                e._sem(sem, role)
            e._sem(sem, "lane")
            for key, part in self._p15_sig_parts(
                targets, sem, "s15"
            ).items():  # every combine core; FIRST on its RISC (a
                pre(key, part)  # combine core that is also a stage-1 reducer must not wait for its own signal)
        for key, v in parts.items():  # distinct sub-part names in the ph15_* namespaces
            vs = v if isinstance(v, list) else [v]
            for j, sub in enumerate(vs):
                sub["name"] = f"t{j + 1}" if len(vs) > 1 else "rs15"
        p15 = next(p for p in e.phases if p.pid == "P15")
        sig_roles = () if stream else tuple(dict.fromkeys(k[0] for k in parts if k[0] in ("member", "mm_only", "root")))
        p15.roles = tuple(
            dict.fromkeys(
                tuple(p15.roles) + sig_roles + ("tail",) + tuple(self.rs8_roles[1]) + tuple(self.rs8_roles[2])
            )
        )
        p15.waits, p15.signals = (), ()
        rs1, rs2 = self.rs8_15.rs1, self.rs8_15.rs2
        print("PMK P15 TP8 reclaim fresh per role: " + str({r_: sum(int(u_.pages) * int(u_.page_bytes) for u_ in (rec_ or {}).get("fresh", [])
                                                            if str(u_.note).startswith("P15")) for r_, rec_ in (e.win_reclaim or {}).items()}), flush=True)  # fmt: skip
        lay.register_phase("P15", PMK.PhaseBody(parts=parts, io=[b_y, rs1.r, rs1.y, rs1.sync, rs2.r2, rs2.y, rs2.sync]))

    def _register_p13(self, lay):
        """W7 step 2: P13 = the shared expert's two PfStream phases (sync slots 5 / 6) in P13's own window: a fresh
        member PARK ring (the window re-lays PARK 29 / PARK2 47 out), the launch-global block base continuing after
        o_proj. Waits H8 (the family's: FS_ARRIVE, every slice of the holder's x-hat_f block landed); signals nothing yet
        (H9 feeds P15, not in the program: B's combine reads y_s after B's rendezvous, i.e. after every member's P13)"""
        pushed = {}
        q_gu = self._qbase(self.qbase_end, self.sh_gu, pushed, "p13_gate_up")
        q_dn = self._qbase(q_gu + self.sh_gu.blocks, self.sh_dn, pushed, "p13_down")
        pts = [(self.sh_gu.parts(self.sh_gu.y, qbase=q_gu, slot=P13_SLOTS[0]), self.sh_gu, "sh_gu"),
               (self.sh_dn.parts(self.sh_dn.y, qbase=q_dn, slot=P13_SLOTS[1]), self.sh_dn, "sh_dn")]  # fmt: skip
        parts13 = {}
        for pt, mm, nm in pts:
            for key, part in self._pfs_parts(pt, mm, nm).items():
                parts13.setdefault(key, []).append(part)
        p13 = next(p for p in self.engine.phases if p.pid == "P13")
        assert tuple(p13.waits) == ("H8",), p13.waits
        p13.signals = ()
        io13 = [self.sh_gu.weight, self.sh_gu.y, self.sh_dn.weight, self.sh_dn.y, self.sync]
        lay.register_phase("P13", PMK.PhaseBody(parts=parts13, io=io13))

    def _fabric_header_audit(self) -> dict:
        """per rank and fabric send part: the ACTIVE directions (one connection each), the PacketHeaderPool headers the
        kernel still takes (the credit header per direction) against the 12-header per-RISC pool, and the private
        data-header bytes (NHDR x directions x 144) against the CB_HDR page; raises on either overflow (the mock runs no
        kernel: this is the only check before silicon)"""
        hdr_cb_bytes = PMK.TILE_BYTES["u32"]
        parts = {}
        for rank in range(self.n):
            r, c = divmod(rank, self.C)
            d = {}
            if self.n == 8:
                d["c1x_send(P2/P11)"] = sum(1 for v in self.c1._ranges(r, c) if v)
                if self.rs8 is not None:
                    d["rs8_send(P8 axis 0)"] = 1
                    d["rs8_rsl_send(P8 axis 1)"] = sum(1 for v in (c, self.rs8.rs2.nc - 1 - c) if v)
                if self.rs8_15 is not None:  # (review 02:57Z: G v2b's P15 RS8 has no P8 to stand in for it)
                    d["rs8_send(P15 axis 0)"] = 1
                    d["rs8_rsl_send(P15 axis 1)"] = sum(1 for v in (c, self.rs8_15.rs2.nc - 1 - c) if v)
            else:
                d["c1_send(P2/P11)"] = 1
                if self.rsr is not None:
                    d["rs_send(P8)"] = 1
                if self.rsr15 is not None:
                    d["rs_send(P15)"] = 1
            parts[rank] = d
        worst = max(v for d in parts.values() for v in d.values())
        out = dict(pool_per_risc=FS_POOL_HEADERS, nhdr=FS_NHDR, hdr_bytes=FS_HDR_BYTES, hdr_cb_bytes=hdr_cb_bytes,
                   directions=parts, pool_headers_max=worst, private_bytes_max=worst * FS_NHDR * FS_HDR_BYTES,
                   pool_headers_without_fix=worst * (FS_NHDR + 1))  # fmt: skip
        assert out["pool_headers_max"] <= FS_POOL_HEADERS, out
        assert out["private_bytes_max"] <= hdr_cb_bytes, out
        return out

    def _connect(self, prog, rank: int, kidx: dict):
        """the program's fabric connections, under THE EDM CHANNEL AUDIT: tt-fabric serves one worker per EDM sender
        channel (chip, link, direction) at a time, so two cores of one program on the same (link, peer) must be a declared
        hand-over (SEM_LINK_REL: C1X lane h -> RS8 stage-2 lane h); one core reusing a channel later on its own RISC
        is ordered by program order. Raises before any launch (the mock runs no kernel: fuse8 hung on this, 02:36Z)."""
        FS.CONNECT_LOG = []
        try:
            self._connect_body(prog, rank, kidx)
            log = list(FS.CONNECT_LOG)
        finally:
            FS.CONNECT_LOG = None
        handover = {(tuple(a_), tuple(b_)) for a_, b_ in zip(self.lanes, self.lanes2)} if self.link_handover else set()
        chans = {}
        for core, link, peer in log:
            cs = chans.setdefault((link, peer), [])
            if not cs or cs[-1] != core:
                cs.append(core)
        bad = {k: v for k, v in chans.items() if any((a_, b_) not in handover for a_, b_ in zip(v, v[1:]))}
        rep = self.report.setdefault("edm_channels", {})
        rep[rank] = {f"link {k[0]} -> {k[1]}": [list(c_) for c_ in v] for k, v in sorted(chans.items())}
        assert not bad, (f"rank {rank} scope {sorted(self.engine.scope or ())}: two workers on one EDM sender channel "
                         f"without a hand-over: {bad}")  # fmt: skip

    def _connect_body(self, prog, rank: int, kidx: dict):
        r, c = divmod(rank, self.C)
        sc = self.engine.scope or set()
        if {
            "P2",
            "P8",
        } <= sc:  # stage 2: the C1 then the RS connections on the same lane BRISC composite (+ P11: C1 again)
            self._connect_two(prog, r, c, kidx, p11="P11" in sc)
            return
        if {"P11", "P15"} <= sc:  # G v2b: P11's C1 / C1X then P15's RSR / RS8 stage 1 on the lane BRISC composite
            self._connect_p11_p15(prog, r, c, kidx)
            return
        if {"P2", "P11"} & sc:
            self.c1.connect(prog, r, c, kidx[("lane", "dm0")])
        elif "P8" in sc and self.rsr is not None:
            self.rsr.connect(prog, r, c, kidx[("lane", "dm0")])
        elif "P8" in sc and self.rs8 is not None:
            self.rs8.rs1.connect(prog, r, c, kidx[("lane", "dm0")])  # stage 1: axis 0 (the board pair), family lanes
            self.rs8.rs2.connect(prog, r, c, kidx[("tail", "dm0")])  # stage 2: axis 1 (the row), the tail lanes
        assert "P15" not in sc, ("P15 needs P8's (W7) or P11's (G v2b) connection block in its program", sorted(sc))

    def _connect_two(self, prog, r: int, c: int, kidx: dict, p11: bool = False):
        """stage 2 (P2 + P8 in one program): each send part's fab_at slot (the last arg of its part, registered for its own
        program) is re-pointed to where ITS connection args land: C1's at the composite width, the RS's after C1's"""
        lay = self.engine.layouts.get(("lane", "dm0"))
        assert lay, "stage 2: the lane BRISC composite layout"
        slot = {}
        for key, _ct0, rt0, part in lay:
            if part.get("fab_at_arg") is not None:
                slot[key.split("_")[0]] = rt0 + int(part["fab_at_arg"])
        assert set(slot) >= ({"P2", "P8", "P11"} if p11 else {"P2", "P8"}), (
            "stage 2: the fab_at slots on the lane BRISC",
            slot,
            [(k_, rt0_, pt_.get("fab_at_arg"), pt_.get("name")) for k_, _c, rt0_, pt_ in lay],
        )
        p15 = "P15" in (self.engine.scope or set())
        ki = kidx[("lane", "dm0")]
        rta = prog.kernels[ki].runtime_args

        def mark(pid):
            for lx, ly in self.lanes:
                v = rta[lx][ly]
                v[slot[pid]] = len(v)

        mark("P2")
        self.c1.connect(prog, r, c, ki)
        mark("P8")
        if self.rsr is not None:
            self.rsr.connect(prog, r, c, ki)
        else:  # TP8: the lanes carry C1X (<= 3 connections) then RS8 stage 1 (axis 0); 4 connections x 2 worker words
            # exceed the 6 reserved ids 10..15, so stage 1 REUSES the words of C1X's first connection on each lane
            # (C1X has drain-closed every connection before rs_send opens: open_finish zeroes the teardown word,
            # open_start re-reads the EDM's cursor into the buffer-index word); the tails carry stage 2 alone
            self._connect_reuse_words(prog, lambda: self.rs8.rs1.connect(prog, r, c, ki))
            if "P15" in (self.engine.scope or set()):  # W7 P15 at TP8: two stage-2 sends on the tail composite -> mark
                # both slots at the connection args' position (the audit leaves a two-fabric-part composite to this hook)
                tlay = self.engine.layouts.get(("tail", "dm0"))
                tk = kidx[("tail", "dm0")]
                trta = prog.kernels[tk].runtime_args
                tslots = [rt0 + int(pt_["fab_at_arg"]) for key_, _c0, rt0, pt_ in tlay if pt_.get("fab_at_arg") is not None
                          and key_.split("_")[0] in ("P8", "P15")]  # fmt: skip
                assert len(tslots) == 2, ("P15 at TP8: the two stage-2 fab_at slots on the tail BRISC", tslots)
                for tx, ty in self.lanes2:
                    v = trta[tx][ty]
                    for sl in tslots:
                        v[sl] = len(v)
            self.rs8.rs2.connect(prog, r, c, kidx[("tail", "dm0")])
        if p11:  # W7 step 1: P11 = C1 again after the RS closed its connections on these lanes: the SAME connections
            # (same lanes, links, peers, worker words) -> P11's fab_at points at P2's connection args (no second block:
            # the lane BRISC composite carries P2 + P8 + P11 send args, 277 > 256 with a duplicate block, mock 14:4xZ);
            # open_start re-reads the EDM's cursor, so a closed connection reopens from the same args
            for lx, ly in self.lanes:
                v = rta[lx][ly]
                v[slot["P11"]] = v[slot["P2"]]
            # (the EDM channel audit is unchanged: P11 adds no connection, its sends ride P2's on the same lane cores)
        if p15:  # W7 step 4: P15 = the RSR again on P8's connections (same lanes / links / peers / worker words)
            assert "P15" in slot, ("P15's fab_at slot on the lane BRISC", slot)
            for lx, ly in self.lanes:
                v = rta[lx][ly]
                v[slot["P15"]] = v[slot["P8"]]

    def _connect_p11_p15(self, prog, r: int, c: int, kidx: dict):
        """G v2b: the lane BRISC composite carries P11's C1 send then P15's RS send -- each send part's fab_at slot points
        at its own connection block (C1's at the composite width, the RS's after it); TP8: RS8 stage 1 reuses C1X's
        worker words (C1X drain-closed first), stage 2 opens on the tails after C1X lane h's release (SEM_LINK_REL)"""
        lay = self.engine.layouts.get(("lane", "dm0"))
        assert lay, "G v2b: the lane BRISC composite layout"
        slot = {}
        for key, _ct0, rt0, part in lay:
            if part.get("fab_at_arg") is not None:
                slot[key.split("_")[0]] = rt0 + int(part["fab_at_arg"])
        assert {"P11", "P15"} <= set(slot), ("G v2b: the fab_at slots on the lane BRISC", slot)
        ki = kidx[("lane", "dm0")]
        rta = prog.kernels[ki].runtime_args

        def mark(pid):
            for lx, ly in self.lanes:
                v = rta[lx][ly]
                v[slot[pid]] = len(v)

        mark("P11")
        self.c1.connect(prog, r, c, ki)
        mark("P15")
        if self.rsr15 is not None:
            self.rsr15.connect(prog, r, c, ki)
            return
        self._connect_reuse_words(prog, lambda: self.rs8_15.rs1.connect(prog, r, c, ki))
        tlay = self.engine.layouts.get(("tail", "dm0"))
        tk = kidx[("tail", "dm0")]
        trta = prog.kernels[tk].runtime_args
        tslots = [rt0 + int(pt_["fab_at_arg"]) for key_, _c0, rt0, pt_ in tlay if pt_.get("fab_at_arg") is not None
                  and key_.split("_")[0] == "P15"]  # fmt: skip
        assert len(tslots) == 1, ("G v2b at TP8: P15's stage-2 fab_at slot on the tail BRISC", tslots)
        for tx, ty in self.lanes2:
            v = trta[tx][ty]
            v[tslots[0]] = len(v)
        self.rs8_15.rs2.connect(prog, r, c, tk)

    def _connect_reuse_words(self, prog, fn):
        """run connect `fn` with the lane cores' fabric words (ids >= FAB_SEM_BASE) hidden, so setup_fabric_connection
        hands out the lowest reserved ids again; then drop the duplicate descriptors (the words exist already)"""
        lanes = {tuple(l_) for l_ in self.lanes}
        # BY VALUE: list(prog.semaphores) holds references into the descriptor's C++ vector, which the assignments below
        # free (2x4 mock 02:00Z: SystemError listobject.c:155 reading a dangling descriptor's core ranges)
        clone = lambda s_: ttnn.SemaphoreDescriptor(id=int(s_.id), core_type=s_.core_type,  # noqa: E731
                                                    core_ranges=FS._crs(sorted(FS._sem_cores(s_))),
                                                    initial_value=int(s_.initial_value))  # fmt: skip
        sems = [clone(s_) for s_ in prog.semaphores]
        on_lane = lambda s_: int(s_.id) >= FS.FAB_SEM_BASE and bool(set(FS._sem_cores(s_)) & lanes)  # noqa: E731
        keep = [s_ for s_ in sems if not on_lane(s_)]
        prog.semaphores = keep
        fn()
        new = [clone(s_) for s_ in list(prog.semaphores)[len(keep) :]]
        have = {(int(s_.id), c_) for s_ in sems for c_ in FS._sem_cores(s_)}
        extra = []
        for s_ in new:
            cs = [c_ for c_ in FS._sem_cores(s_) if (int(s_.id), c_) not in have]
            assert not cs or not (set(cs) & lanes), ("stage 2 TP8: a lane connection word beyond C1X's", int(s_.id), cs)
            if cs:
                extra.append(s_)
        prog.semaphores = sems + extra

    # ------------------------------------------------------------------------------------------------ launch
    def __call__(self, x, ci, **_):
        assert x is self.x, "program A's residual address is baked into its args: pass the tensor it was built on"
        self.counters["calls"] += 1
        return self.layer(x, ci)

    def plan(self) -> dict:
        d = self.layer.plan()
        d.update(impl=self.impl, assembly=self.report, scopes=[list(s) for s in self.scopes],
                 launch_census=getattr(self.layer, "launch_census", {}))  # fmt: skip
        return d

    def probes(self) -> dict:
        out = {}
        if self.probe:
            for nm, mm, slot in zip(
                ("qkv", "lowrank", "f_b", "g_b", "o_proj"),
                self.mms3 + ([self.mm_o] if self.mm_o is not None else []),
                range(5),
            ):
                out[nm] = mm.probes(slot=slot)
        try:
            out["kda"] = self.kphase.probe_rows(self.scr)
        except Exception as ex:  # noqa: BLE001
            out["kda"] = repr(ex)
        return out

    def index_of(self, pid: str) -> int:
        """the index (launch order) of the scoped program that carries phase `pid`"""
        return next(i for i, s_ in enumerate(self.scopes) if pid in s_)

    def release(self):
        for mm in self.mms3 + ([self.mm_o] if self.mm_o is not None else []):
            mm.release()
        if self.reuse is not None:  # the sync / c1 / RS objects are the owner's (released with it)
            return
        if self.sync is not None:
            ttnn.deallocate(self.sync)
        self.c1.release()
        if self.rsr is not None:
            self.rsr.release()
        if self.rs8 is not None:
            self.rs8.release()


# ------------------------------------------------------------------------------------- the A -> B DRAM contract
class ProgramAHandoff:
    """What program A leaves in DRAM for program B (ruling 13:50Z), as code: every tensor with its shape / dtype /
    layout / per-chip meaning, and the mapping onto the moe2 planner's standalone entry
    (tt/pmk_moe2.ProgramB.parts(x_ext, ids_rm, wt, out_rm, y, c, x_tile=...), README moe2 sec 12):

      xhat_f_local   hn[1]   [1, 1, SP, 4096] bf16 TILE  this chip's x-hat_f rows (the mHC FFN pre of P10), DRAM
      xhat_f_store   XS 35   the gathered [C, 4096] bf16 blocks in the 64 holders' L1 after P11 (block G on member G,
                             the same address on every chip) -- not a DRAM tensor: B's lever-2 fill (x_tile) needs the
                             TILE gather in DRAM (an AG into DRAM: 7a FabricStreamAG), B's TP2 DRAM-gather entry needs
                             x_ext ROW_MAJOR (an untilize) -- OPEN
      residual       x       [1, 1, SP, 16384] fp32 DRAM, updated in place by P10 (TP2 streamed; TP8 stored by this
                             assembly's x_store part after the P10 mhc part)
      post / comb    posts[1], combs[1]  [1, 1, SP, 4] / [1, 1, SP, 16] fp32 TILE: the FFN sublayer's coefficient pages
                             (F_PC_DRAM); B's P16 post rebuilds its K / L tiles with kernels/pmk/asm/pmk_pcoef_dram.cpp
                             (the part this assembly runs before P10's post)
      y_ffn          ys[1]   [1, 1, SP, 4096] fp32: B's RS lands the FFN output here (A never writes it)
      kda state      state.S / state.C: written by P6 (the stack's prefill -> decode hand-off), not B's input"""

    def __init__(self, a: PmkProgramA):
        self.a = a
        g = a.geom
        self.sp_rows, self.c_rows, self.hidden = g.sp_rows, g.c_rows, g.hidden
        self.xhat_f_local = a.hns[1]
        self.residual = a.x
        self.post, self.comb = a.posts[1], a.combs[1]
        self.y_ffn = a.ys[1]
        self.gather_tile = None  # [1, 1, C, 4096] bf16 TILE DRAM when an AG-into-DRAM phase is attached

    def table(self) -> list:
        rows = []
        for name, t in (
            ("xhat_f_local", self.xhat_f_local),
            ("residual", self.residual),
            ("post", self.post),
            ("comb", self.comb),
            ("y_ffn", self.y_ffn),
        ):
            rows.append(dict(name=name, shape=tuple(int(v) for v in t.shape), dtype=str(t.dtype).split(".")[-1],
                             layout=str(t.layout).split(".")[-1], address=int(t.buffer_address())))  # fmt: skip
        return rows

    def for_program_b(self, c: int) -> dict:
        """the kwargs of pmk_moe2.ProgramB.parts / __call__ that A provides; the ones B must still produce or that
        need a layout phase are listed under `missing`"""
        return dict(c=c, x_tile=self.gather_tile, x_ext=None, ids_rm=None, wt=None,
                    missing=["x_tile: the TILE gather of xhat_f (an AG into DRAM, 7a FabricStreamAG) -- not launched",
                             "x_ext [C + 32, H] bf16 ROW_MAJOR: B's TP2 DRAM-gather input needs an untilize of the gather",
                             "ids_rm / wt: the router (P12) output -- B's standalone entry takes them as inputs"],
                    residual=self.residual, post=self.post, comb=self.comb, y_ffn=self.y_ffn,
                    pcoef_part="kernels/pmk/asm/pmk_pcoef_dram.cpp (ct [22, 23] + acc(post) + acc(comb); rt [post, comb, rb0, step, nrb])")  # fmt: skip


def build_program_a(pf, li: int, c: int, sp: bool, x, state, valid_end, probe: bool = False, **kw) -> PmkProgramA:
    return PmkProgramA(pf, li, c, sp, x, state, valid_end, probe=probe, **kw)
