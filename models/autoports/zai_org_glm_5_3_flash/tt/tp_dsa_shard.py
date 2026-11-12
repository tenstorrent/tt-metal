# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The sequence-sharded DSA + split-K attention half on the TP pair (1x2 / 2x1) and the TP8 2x4 mesh (tt/dsa_shard.py,
kernels/mla/dsa_shard.hpp; doc/ttnn/14-production-tp8-megakernel/dsa/shard_tp8_design.md): TPMla with
MkMla(dsa_shard=True) + in-program fabric collectives per launch (tt/fabric_stream.py, Stage 7a's kernels).
The pair (TP2, one round each):
  AG  all-gather of x_ag [2 T chunks: per token this chip's q_lat tile row | its candidate records] -> y_ag on every
      chip (the peer's q for pass 0; the coordinator's global merge); the lane waits for T (chunk cores + the
      coordinator) + 1 (the latent core's q_lat copy) producer increments, its waiter signals the index core (SEM_AGC);
  RS  the pair's reduce-scatter exchange of x_rs [2 blocks x T x 3 chunks: the partial (O, stats) for the peer's heads]
      -> y_rs (the peer's partial of this chip's heads); the lane waits for T root increments, its waiter signals the
      owners (SEM_RSC).
TP8 on the 2x4 mesh (rank = 4 row + col; 4 stream instances, the design's link table):
  AG1 (axis 0, link 0) x_ag [T chunks: the compact q piece + records] -> y_ag block ag_blk = 2 col + row, local copy on;
      its waiter signals AG2's lane;
  AG2 (axis 1, link 0) the column's 2-block bundle of y_ag multicast along the row (x = y = y_ag); its waiter signals
      the index core (SEM_AGC): y_ag holds all 8 blocks;
  RS1 (axis 0, link 1) the pass-0 partial of the other row's tile -> the column peer (the pair exchange per column);
      its waiter signals the owners (SEM_RSC, 1st);
  RS2 (axis 1, link 1) FabricStreamScatter: x_rs2 block q (the 8-row slice of row peer q of the column-combined
      partial) -> peer q's y_rs2 block p; its waiter signals the owners (SEM_RSC, 2nd).
  The o_proj AR lanes run FAR_LATE_OPEN: lane 0 releases AG1 + AG2 (link 0), lane 1 RS1 + RS2 (link 1) at launch start.
Token lanes (rev 60, W: t > 4 = native B=32 and widths 8 / 16; shard_tp8_design.md section 7): MkMla's att_lanes(t) = L
lanes, one index core per lane (reserved here AFTER the stream lanes: the t <= 4 placement is unchanged): the AG waits
for t (C / L + 1) + 1 producer increments and signals every lane index core (count-prefixed), the RS signals every lane
root; streams per t of stream_ts (ts + stream_ts + the stack's native widths GLM53_STACK_NATIVE_WIDTHS when 32 is in ts:
StackBackend's width calls t = 8 / 16 run on the 32-row FabricPhase, tp_mla.fabric, but on their own streams).
One MeshProgramDescriptor per call: the captured MkMla program + Stage 14's o_proj all-reduce lanes (TPMla) + the stream
lanes, with the chip's rank / pair-peer position patched into the shard runtime args per mesh coordinate; the link /
target / packet-header audits (tt/shard_plan.py) run on every build and raise. The sharded state (per-chip latent
pool, local page table, local pooled-key store) is the caller's (tests/s14/jobs/dsa_shard.py: tt/dsa_shard.py
ShardMap). GLM53_DSA_SHARD=1 selects this class in a stack (tt/stack_model.py: pending)."""

from __future__ import annotations

import os

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import dsa_shard as DS
from models.autoports.zai_org_glm_5_3_flash.tt import fabric_stream as FS
from models.autoports.zai_org_glm_5_3_flash.tt import mk_mla as MK
from models.autoports.zai_org_glm_5_3_flash.tt import shard_plan as SP
from models.autoports.zai_org_glm_5_3_flash.tt import tp_mla as TPM
from models.autoports.zai_org_glm_5_3_flash.tt.shard_plan import audit_links  # noqa: F401  (re-export)
from models.autoports.zai_org_glm_5_3_flash.tt.tp8_layer import _Capture, _rt_dict, _src_is
from models.autoports.zai_org_glm_5_3_flash.tt.tp_mla import _with_rt

DRAM = ttnn.DRAM_MEMORY_CONFIG
HIDDEN = DS.KV_LORA  # the pair's AG tensors / q_asm width (512 bf16 = 16 pages per 32-row chunk)
W_RS = 32 * MK.SHARD_RS_PPC  # rev 59: x_rs / y_rs width (384 bf16 = 12 pages per chunk: 3 chunks = the 36-page message)
W8 = 32 * MK.SHARD_AG8_PPC  # TP8: the compact AG / RS2 tensors' width (320 bf16 = 10 pages per chunk)
assert MK.SHARD_RS2_PPC == MK.SHARD_AG8_PPC


class TPMlaShard(TPM.TPMla):
    """TPMla on the pair / the 2x4 mesh with the split-K DSA phase: same call / outputs (o_lat: this chip's heads,
    merged over every chip's selected rows; s1 identical on every chip after the o_proj all-reduce)."""

    def __init__(self, mesh, mhc_args: dict, attn, *, ts=(1, 2, 3, 4), lanes_extra=None, stream_ts=None,
                 stream_share=None, **mk_kwargs):  # fmt: skip
        """stream_ts (rev 60): extra row counts with their own streams (no FabricPhase of their own: tp_mla.fabric
        serves t > 4 from the 32-row phase); the stack's native widths are added when 32 is in ts"""
        assert mk_kwargs.get("dsa") is not None, "TPMlaShard: dsa=DsaIndexer"
        super().__init__(mesh, mhc_args, attn, ts=ts, dsa_shard=True, **mk_kwargs)
        self.geo = SP.Geometry(self.R, self.C)  # raises unless 1x2 / 2x1 / 2x4 / 8x4
        self.n, self.L, self.P = self.geo.n, self.geo.L, self.geo.P
        wts = set(ts) | set(int(v) for v in (stream_ts or ()))
        if 32 in ts:  # StackBackend's native widths (t = w calls on this object, the 32-row FabricPhase)
            nspec = os.environ.get("GLM53_STACK_NATIVE_WIDTHS", "none")
            wts |= (
                {int(v) for v in nspec.split(",") if v.strip() and 4 < int(v) < 32}
                if nspec not in ("", "none")
                else set()
            )
        # GLM53_DSA_SHARD_STREAM_TS ("1,8,16,32"; default all): build the streams of these row counts only. Each t holds
        # 4 x 1.5 KB of L1 sync words per sparse_dsa layer = 67.6 KB per bank over the 11 layers; the served native-width
        # stack calls t in {1 (narrow), 8, 16 (native), 32}, never t = 2..4 (the 4-row chunk path), and the n29h7 env's
        # prefill MoE program needs that L1 (2026-10-23 07:29Z: its static CBs end 114.5 KB above the lowest L1 buffer;
        # dsa/shard/README.md section 6r). A call at a t without streams raises (__call__: 'no streams for t').
        only = os.environ.get("GLM53_DSA_SHARD_STREAM_TS", "")
        if only.strip():
            keep = {int(v) for v in only.split(",") if v.strip()}
            wts = {t for t in wts if t in keep}
            if not wts:
                raise ValueError(f"GLM53_DSA_SHARD_STREAM_TS={only!r} leaves no stream row count of {sorted(set(ts))}")
        self.stream_ts = tuple(sorted(wts))
        assert max(self.stream_ts) <= MK.SHARD_T_MAX, self.stream_ts
        assert self.mk.groups == 1, ("one head group per chip", self.mk.groups)
        self.sm = DS.ShardMap(self.n)
        grid = mesh.compute_with_storage_grid_size()
        used = set(self.mk.used) | set(tuple(v) for v in self.lanes)
        free = [(x, y) for y in range(int(grid.y)) for x in range(int(grid.x)) if (x, y) not in used]
        self.names = ["ag1", "rs1"] + (["ag2", "rs2"] if self.L > 1 else [])
        ex = list(lanes_extra or free[: len(self.names)])
        assert len(ex) == len(self.names), f"{len(self.names)} more free cores for the stream lanes ({ex})"
        self.lane_core = dict(zip(self.names, ex))
        self.ag_lane, self.rs_lane = self.lane_core["ag1"], self.lane_core["rs1"]
        # rev 60: the token lanes' index cores (t > 4), after the stream lanes (t <= 4 builds: no lane cores, the same
        # placement as rev 59); every lane holds the 2 LINE merge helpers
        self.ix_lane_cores = []
        # GLM53_DSA_SHARD_LANES_T4=1 (2026-11-05): lane cores for the t <= 4 streams too, so att_lanes' LANES_T4 rule
        # applies on the shard (1 M probes, layers 0-7: B = 8 native 10.45 ms vs B = 4 16.16 -- the one-lane t <= 4
        # build runs the users' select / merge / tree chains in series)
        # default ON (2026-11-05 e2e, block 45 layers B = 4 ISL 4 K: 36.1 -> 22.8 ms per step, ids identical)
        lanes_t4 = os.environ.get("GLM53_DSA_SHARD_LANES_T4", "1") == "1" and max(self.stream_ts) > 1
        if max(self.stream_ts) > MK.DSA_T_MAX or lanes_t4:
            rest = [v for v in free if tuple(v) not in {tuple(e) for e in ex}]
            # NCHIPS merge helpers per lane; rev 64 (GLM53_DSA_SHARD_KPH, the 8x4): NCHIPS / KPH -- KPH 4 (or "auto")
            # reserves up to 8 lane cores (GLM53_MLA_DSA_LANES, default 8; a cap by the free cores prints MK_SHARD_LANES)
            self.ix_lane_cores = self.mk.reserve_lane_cores(rest, self.n // MK.shard_kph_reserve(self.n, self.P))
        for t in self.stream_ts:
            self.mk.shard_lanes_check(t, self.L, self.n, self.kph(t))
        self.link_audit, self.target_audit, self.header_audit, self.census, self.censuses = {}, {}, {}, {}, {}
        # GLM53_DSA_SHARD_SHARE_SYNC=1 (stream_share = the first sparse_dsa layer's TPMlaShard): every layer's stream of a
        # row count t uses that layer's sync tensor -- the counters are monotonic across launches (fs_common.hpp: launch g
        # waits >= g x per-launch count), every layer's stream at t has the same lane cores and per-launch counts, and the
        # layers launch one after another in the same order on every chip: one stream launched 11 x per step. Frees
        # 10 x 4 x 1.5 KB of L1 per t (the n29h7 1M env: 246 KB per bank at t = 1 / 8 / 16 / 32).
        self._sync_from = None
        # default ON (2026-11-05): the 45-layer stack sits at the L1 floor; the per-layer syncs put the shard 20 KB over
        if stream_share is not None and os.environ.get("GLM53_DSA_SHARD_SHARE_SYNC", "1") == "1":
            assert stream_share.lane_core == self.lane_core, (stream_share.lane_core, self.lane_core)
            self._sync_from = stream_share
        self.streams = {t: self._new_streams(t) for t in self.stream_ts}

    def kph(self, t: int) -> int:
        """rev 64: chips per merge helper of the t-row build (GLM53_DSA_SHARD_KPH on the 8x4: 1 default, an int, or
        "auto" = 32 lanes / 64; 1 on the pair / the 2x4) -- the stream dict's "kph" (MkMla's MLA_DSA_SHARD_KPH)"""
        return MK.shard_kph(self.n, self.mk.chunks, self.mk.att_lanes(t), self.P)

    def _pos(self, r: int, c: int) -> int:
        """this chip's rank (the head shard / owner index: row-major mesh order)"""
        return self.geo.rank(r, c)

    @property
    def valid_noc(self) -> set:
        """the NoC (x, y) of every worker core of the grid (a semaphore / atomic target must be one of them)"""
        grid = self.mesh.compute_with_storage_grid_size()
        return {
            (int(w.x), int(w.y))
            for w in (
                self.mesh.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
                for y in range(int(grid.y))
                for x in range(int(grid.x))
            )
        }

    def _noc(self, core) -> tuple:
        w = self.mesh.worker_core_from_logical_core(ttnn.CoreCoord(*core))
        return (int(w.x), int(w.y))

    def _new_streams(self, t: int) -> dict:
        mesh, geo = self.mesh, self.geo
        z = lambda rows, w: ttnn.zeros(
            (1, 1, rows, w), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM
        )  # noqa: E731  # fmt: skip  (the same shape / address on every chip)
        nl = self.mk.att_lanes(t)  # rev 60: token lanes (1 at t <= 4)
        cl = self.mk.chunks // nl  # chunk cores per lane
        ix = [tuple(v) for v in self.mk._ix_nocs(t)]  # the index core / every lane's index core (the AG consumers)
        assert len(ix) == nl, (t, nl, ix)
        rs_rows = t * MK.SHARD_RS_CHUNKS * 32
        # t x (the token's lane chunk cores + its lane coordinator) + the latent core's q_lat copy
        ag_wait = (MK.SEM_FSW, t * (cl + 1) + 1)
        assert ag_wait[1] == SP.ag_producers(t, self.mk.chunks, nl), (t, nl, ag_wait)  # (the plan's per-lane count)
        kph = self.kph(t)  # rev 64: chips per merge helper (NCHIPS / kph helpers per lane)
        why = SP.lanes_reasons(t, nl, self.mk.chunks, self.L, self.mk.groups, nchips=self.n, kph=kph)
        assert not why, why
        rs_wait = (MK.SEM_FSW, t * self.mk.groups)  # one root increment per token (its lane's root)
        lane = lambda name: [self.lane_core[name]]  # noqa: E731
        share = self._sync_from.streams.get(t, {}).get("objs", {}) if self._sync_from is not None else {}
        sy = lambda name: dict(sync=share[name].sync) if name in share else {}  # noqa: E731
        # rev 59 (review 7): the RS consumers are the roots only (chunk core 0 of each head group; the other owners
        # never wait on SEM_RSC); rev 60: chunk core 0 of every (group, lane)
        roots = [tuple(self.mk.owner_noc[g * self.mk.chunks + l_ * cl]) for g in range(self.mk.groups)
                 for l_ in range(nl)]  # fmt: skip
        if self.L == 1:  # the pair: the validated one-round form (link 0: AG, link 1: RS; both gated by the AR lanes)
            x_ag = z(2 * t * 32, HIDDEN)
            x_rs = z(2 * rs_rows, W_RS)
            ag = FS.FabricStreamAG(mesh, 2 * t * 32, HIDDEN, lanes=lane("ag1"), **sy("ag1"), local_copy=True, wait=ag_wait,
                                   cons=(MK.SEM_AGC, ix), link_base=0, after=MK.SEM_FSA)  # fmt: skip
            rs = FS.FabricStreamRS2(mesh, rs_rows, W_RS, lanes=lane("rs1"), **sy("rs1"), wait=rs_wait, cons=(MK.SEM_RSC, roots),
                                    link_base=1, after=MK.SEM_FSA)  # fmt: skip
            return dict(C=self.n, line=1, x_ag=x_ag, y_ag=ag.y, x_rs=x_rs, y_rs=rs.y, ag=ag, rs=rs,
                        ag_lane=ag.lane_noc()[0], rs_lane=rs.lane_noc()[0], t=t, lanes=nl, ag_wait=ag_wait[1],
                        objs=dict(ag1=ag, rs1=rs), x=dict(ag1=x_ag, rs1=x_rs))  # fmt: skip
        if self.P > 2:  # 8x4 (doc/perf/shard_8x4_design.md): AG1 (the column of 8) -> AG2 (the row, 8-block bundles)
            # -> merge; RS1 = the column SCATTER of 8-row slices (the 2x4 round-2 format, x_rs1 block q = destination
            # position q) -> combine -> RS2 = the row scatter of 2-row slices (x_rs2b). The kernel's x_rs2 / y_rs2
            # accessors carry round 1 here; x_rs / y_rs (the pair message) are unused: a one-chunk x_rs stands in for
            # both (the kernel never reaches that path when PAIR > 2, only the accessor's address is passed).
            W32 = 32 * SP.AG32_PPC
            W2B = 32 * SP.RS2B_PPC
            # rev 68 (GLM53_DSA_SHARD_AGK, dsa_shard.hpp AGK): y_ag in the key-packed format (2 pages per token: the q
            # page + one page of record keys) -- AG1 packs x_ag's AG32 chunks on its lane (fs_read.cpp FS_KEYPACK), AG2
            # multicasts the packed bundles: a third of the bytes of both rounds
            agk = SP.shard_agk(self.P)
            # GLM53_DSA_SHARD_AGB (on AGK): y_ag as ONE 32-row chunk per block (2 t pages: the same page ids as AGK's
            # t chunks of 2 pages, so no consumer changes) -- the AG is per-chunk latency bound (AGK: 3x fewer bytes,
            # the window -21 %), and the chunks drop from 16 + 128 to 1 + 8 per source (one signal per block, AG2 in
            # 8-page batches: 32 instead of 128 at t 16; its CB 3 x 8 pages = rev 65's 48 KB)
            agb = SP.shard_agb(self.P)
            # the AG streams' rows / width / keypack / batches: shard_plan.ag_stream_args -- the plan's
            # Geometry.streams(t, agk, agb) takes the same numbers (the block audit below compares them per chip)
            aga = SP.ag_stream_args(t, agk=agk, agb=agb, P=self.P)
            WY, rows1, chunks2 = aga["ag1"]["width"], aga["ag1"]["rows"], aga["ag2"]["rows"]
            kp, nb2 = aga["ag1"]["extra"], aga["ag2"]["extra"]
            # GLM53_DSA_SHARD_RSB: both scatters in BLOCK CHUNKS -- x / y [n x 32, W t], one 32-row chunk per destination
            # block holding its t tokens' slices at the knob-off page ids (the roots' writes / landing reads unchanged):
            # fs_send.cpp's chunk-end packet (a fused write + flushed atomic inc at the receiver) then goes out once per
            # destination instead of once per token (RS1 7 t -> 7, RS2 3 t -> 3 per source) and the waiters wait for one
            # per source; RS2 in 8-page batches; the staging CBs keep their bytes; no kernel change
            # (shard_plan.rs_stream_args: Geometry.streams(t, rsb=) takes the same numbers for the block audit below)
            rsb = SP.shard_rsb(self.P)
            # GLM53_DSA_SHARD_RSNP (on RSB): the scatter lanes stage / send only the slices' data pages (9 of 10 / 3 of 4),
            # paired across slices (fabric_stream.py sub=, fs_read / fs_send FS_SUBPC / FS_SUBD): every data page lands
            # at its RSB page with the same bytes, the pad pages (never written by the roots, never read) stay home;
            # x / y / the roots / the signals unchanged (shard_plan.rs_stream_args / fs_scatter_packets)
            rsnp = SP.shard_rsnp(self.P)
            if rsnp:
                SP.rsnp_header_check()
            rsa = SP.rs_stream_args(t, rsb=rsb, P=self.P, rsnp=rsnp)
            assert (rsa["rs1"]["width"], rsa["rs2"]["width"]) == ((W8 * t, W2B * t) if rsb else (W8, W2B)), rsa
            x_ag = z(t * 32, W32)
            x_rs = z(32, W_RS)
            x_rs1 = z(self.P * rsa["rs1"]["rows"], rsa["rs1"]["width"])
            x_rs2b = z(self.L * rsa["rs2"]["rows"], rsa["rs2"]["width"])
            q_asm = z(2 * t * 32, HIDDEN)
            ag2_noc = self._noc(self.lane_core["ag2"])
            ag1 = FS.FabricStreamAG(mesh, rows1, WY, axis=geo.pair_axis, lanes=lane("ag1"), **sy("ag1"), local_copy=True,
                                    wait=ag_wait, cons=(MK.SEM_FSW, [ag2_noc]), link_base=0, after=MK.SEM_FSA, **kp)  # fmt: skip
            ag2 = FS.FabricStreamAG(mesh, chunks2, WY, axis=geo.line_axis, lanes=lane("ag2"), **sy("ag2"), bundle=True,
                                    y=ag1.y, wait=(MK.SEM_FSW, 1), cons=(MK.SEM_AGC, ix), link_base=0, after=MK.SEM_FSA,
                                    **nb2)  # fmt: skip
            # rev 65 (GLM53_DSA_SHARD_RS1_EARLY, dsa_shard.hpp RS1 early): the roots signal after each pass; the lane
            # streams the other half's destinations (pass 0's slices) once t pass-0 signals landed, the own half's at t
            # pass-1 signals -- rev 67: in two fields of the count (pass 1 adds 1 << 16; shard_plan.rs1_early_gates).
            # GLM53_DSA_SHARD_COMB: the fold combine on the roots (the kernels' define only; same streams)
            comb, early = SP.shard_comb(self.P), SP.shard_rs1_early(self.P)
            # rev 70 (GLM53_DSA_SHARD_R8): the local select / merge in 8-bit radix digits -- the kernels' MLA_DSA_R8 and
            # the mailbox's layout (MkMla), no stream change
            r8 = SP.shard_r8(self.P)
            # rev 72: GLM53_DSA_SHARD_LOCAL_ALL (the <= 512-pool skip) / GLM53_DSA_SHARD_ROUND_MARKS (the coordinator's
            # round marks): kernel defines only
            local_all, rmarks = SP.shard_local_all(self.P), SP.shard_round_marks(self.P, r8)
            split = dict(first=lambda p: SP.rs1_early_first(p, self.P), wait2=rs_wait[1]) if early else {}
            rs1 = FS.FabricStreamScatter(mesh, rsa["rs1"]["rows"], rsa["rs1"]["width"], axis=geo.pair_axis, lanes=lane("rs1"),
                                         **sy("rs1"), wait=rs_wait, cons=(MK.SEM_RSC, roots), link_base=1, after=MK.SEM_FSA,
                                         **split, **rsa["rs1"]["extra"])  # fmt: skip
            # both scatters signal the roots' SEM_RSC and the root reads it ordinally (wait_min 1 = round 1 landed,
            # 2 = round 2 landed): sound because rs2's waiter is its own lane BRISC, which signals after that lane's
            # sends, and those sends wait on rs2_sem, which the root raises only past its round-1 wait -- the second
            # +1 cannot precede the first wait. A second rs2 lane core or a waiter off the sending RISC breaks this:
            # give rs2 its own consumer semaphore then.
            rs2 = FS.FabricStreamScatter(mesh, rsa["rs2"]["rows"], rsa["rs2"]["width"], axis=geo.line_axis, lanes=lane("rs2"),
                                         **sy("rs2"), wait=rs_wait, cons=(MK.SEM_RSC, roots), link_base=1, after=MK.SEM_FSA,
                                         **rsa["rs2"]["extra"])  # fmt: skip
            assert int(ag1.y.shape[-2]) == geo.n * rows1
            pages = lambda ten: int(ten.shape[-2]) * int(ten.shape[-1]) // 1024  # noqa: E731
            # the scatters' x / y hold the knob-off pages (RSB: the same page ids in block chunks)
            assert pages(x_rs1) == pages(rs1.y) == self.P * t * SP.RS2_PPC, (tuple(x_rs1.shape), tuple(rs1.y.shape))
            assert pages(x_rs2b) == pages(rs2.y) == self.L * t * SP.RS2B_PPC, (tuple(x_rs2b.shape), tuple(rs2.y.shape))
            assert (rs1.sub is not None, rs2.sub is not None) == (rsnp, rsnp), (rs1.sub, rs2.sub)
            ypg = (SP.AGK_PPC if agk else SP.AG32_PPC) * t  # y_ag pages per block: AGB keeps AGK's page ids
            assert int(ag1.y.shape[-2]) * int(ag1.y.shape[-1]) == geo.n * ypg * 1024, (tuple(ag1.y.shape), ypg)
            return dict(C=self.n, line=self.L, pair=self.P, kph=kph, comb=comb, rs1_early=early, agk=agk, agb=agb, rsb=rsb, r8=r8,
                        rsnp=rsnp, local_all=local_all, round_marks=rmarks, x_ag=x_ag, y_ag=ag1.y,
                        x_rs=x_rs, y_rs=x_rs,
                        x_rs2=x_rs1, y_rs2=rs1.y, x_rs2b=x_rs2b, y_rs2b=rs2.y, q_asm=q_asm,
                        ag=ag1, rs=rs1, ag2=ag2, rs2=rs2, ag_lane=ag1.lane_noc()[0], rs_lane=rs1.lane_noc()[0],
                        ag2_lane=ag2_noc, rs2_lane=rs2.lane_noc()[0], t=t, lanes=nl, ag_wait=ag_wait[1],
                        objs=dict(ag1=ag1, ag2=ag2, rs1=rs1, rs2=rs2),
                        x=dict(ag1=x_ag, ag2=ag1.y, rs1=x_rs1, rs2=x_rs2b))  # fmt: skip
        # TP8: AG1 (axis 0) -> AG2 (axis 1, the bundles) -> merge; RS1 (axis 0) -> combine -> RS2 (axis 1 scatter)
        x_ag = z(t * 32, W8)
        x_rs = z(2 * rs_rows, W_RS)
        x_rs2 = z(self.L * t * 32, W8)
        q_asm = z(2 * t * 32, HIDDEN)  # rev 59: the two head tiles per token, assembled by the index core (ix_dm1.cpp)
        ag2_noc = self._noc(self.lane_core["ag2"])
        ag1 = FS.FabricStreamAG(mesh, t * 32, W8, axis=geo.pair_axis, lanes=lane("ag1"), **sy("ag1"), local_copy=True, wait=ag_wait,
                                cons=(MK.SEM_FSW, [ag2_noc]), link_base=0, after=MK.SEM_FSA)  # fmt: skip
        ag2 = FS.FabricStreamAG(mesh, 2 * t * 32, W8, axis=geo.line_axis, lanes=lane("ag2"), **sy("ag2"), bundle=True, y=ag1.y,
                                wait=(MK.SEM_FSW, 1), cons=(MK.SEM_AGC, ix), link_base=0, after=MK.SEM_FSA)  # fmt: skip
        rs1 = FS.FabricStreamRS2(mesh, rs_rows, W_RS, axis=geo.pair_axis, lanes=lane("rs1"), **sy("rs1"), wait=rs_wait,
                                 cons=(MK.SEM_RSC, roots), link_base=1, after=MK.SEM_FSA)  # fmt: skip
        rs2 = FS.FabricStreamScatter(mesh, t * 32, W8, axis=geo.line_axis, lanes=lane("rs2"), **sy("rs2"), wait=rs_wait,
                                     cons=(MK.SEM_RSC, roots), link_base=1, after=MK.SEM_FSA)  # fmt: skip
        assert int(ag1.y.shape[-2]) == geo.n * t * 32 and int(rs2.y.shape[-2]) == self.L * t * 32
        return dict(C=self.n, line=self.L, x_ag=x_ag, y_ag=ag1.y, x_rs=x_rs, y_rs=rs1.y, x_rs2=x_rs2, y_rs2=rs2.y,
                    q_asm=q_asm,
                    ag=ag1, rs=rs1, ag2=ag2, rs2=rs2, ag_lane=ag1.lane_noc()[0], rs_lane=rs1.lane_noc()[0],
                    ag2_lane=ag2_noc, rs2_lane=rs2.lane_noc()[0], t=t, lanes=nl, ag_wait=ag_wait[1],
                    objs=dict(ag1=ag1, ag2=ag2, rs1=rs1, rs2=rs2),
                    x=dict(ag1=x_ag, ag2=ag1.y, rs1=x_rs, rs2=x_rs2))  # fmt: skip

    def _audits(self, r: int, c: int, st: dict, fab, entries: list, late: dict) -> None:
        """the link-holder, NoC-target / rank-word and packet-header audits of chip (r, c)'s program: raise on any
        violation (every build, the mock included); the results go to self.*_audit[(r, c)] (recorded by the unit)"""
        geo, rank = self.geo, self.geo.rank(r, c)
        objs = st["objs"]
        # link holders from the objects themselves, then cross-checked against the plan (tt/shard_plan.py Geometry)
        holders = []
        for name, o in objs.items():
            assert o.after is not None, (name, "every stream is gated (FS_LATE_OPEN)")
            for h in range(len(o.lanes)):
                for d in o.directions(r, c):
                    holders.append((name, o.link_base + h, d, "gated", f"ar_lane{o.link_base + h}"))
        ar_dirs = geo.ar_directions(r, c)
        for h in (0, 1):
            for d in ar_dirs:
                holders.append((f"ar_lane{h}", h, d, "late", None))
        # rev 59 (review 5): every stream's block / destination offsets = the plan's (shard_plan.Geometry.block)
        for name, o in objs.items():
            p_, line = o._pos(r, c)
            src, dst, local, up, dn = o._block(p_, line)
            want = geo.block(name, st["t"], r, c, rs1_early=bool(st.get("rs1_early")), agk=bool(st.get("agk")),
                             agb=bool(st.get("agb")), rsb=bool(st.get("rsb")))  # fmt: skip
            got = dict(p=p_, src=src if not o.scatter else None, dst=dst, local=bool(local), up=up, dn=dn,
                       dsts=list(o._dsts(p_)))  # fmt: skip
            if got != {k: want[k] for k in got}:
                raise ValueError(f"chip {(r, c)} stream {name}: blocks {got} differ from the plan {want}")
        plan = geo.holders(r, c, st["t"])
        if sorted(holders) != sorted(plan):
            raise ValueError(f"chip {(r, c)}: link holders differ from the plan: {sorted(holders)} vs {sorted(plan)}")
        rel = {h: sorted(self._noc_names(late[h][1], st)) for h in late}
        for name, o in objs.items():  # every gated stream is released by the AR lane of its link
            if name not in rel[o.link_base]:
                raise ValueError(f"chip {(r, c)}: {name} on link {o.link_base} is not released by ar_lane{o.link_base}")
        self.link_audit[f"{r},{c}"] = SP.audit_links(holders)
        # NoC targets: the patched shard blocks + every stream's consumer / lane list + the AR release lists
        extra = [(f"{n}.cons", tuple(xy)) for n, o in objs.items() for xy in (o.cons[1] if o.cons else [])]
        extra += [(f"{n}.lane", tuple(xy)) for n, o in objs.items() for xy in o.lane_noc()]
        extra += [(f"ar_lane{h}.release", tuple(xy)) for h in late for xy in late[h][1]]
        self.target_audit[f"{r},{c}"] = SP.audit_targets(entries, rank, (1 - geo.pair_pos(rank)) if self.P == 2 else 0,
                                                          self.valid_noc, extra)  # fmt: skip
        # fabric packet headers per (core, RISC): fs_send NHDR per direction, lane_send 2 + 1 flag per direction
        budget = {(n, "brisc"): o.headers(r, c) for n, o in objs.items()}
        budget.update({(f"ar_lane{h}", "brisc"): SP.FAR_NHDR * len(ar_dirs) for h in (0, 1)})
        if budget != geo.header_budget(r, c, st["t"]):
            raise ValueError(f"chip {(r, c)}: header budget differs from the plan: {budget}")
        self.header_audit[f"{r},{c}"] = SP.audit_headers(budget)

    def _cb_census(self, prog, st: dict) -> dict:
        """static CB bytes per core of chip (0, 0)'s program (every chip has the same CB layout) by role: the owners'
        maximum (the L1 floor binder), the index / latent cores, the AR / stream lane cores, and the overall maximum"""
        per = {}
        for cb in prog.cbs:
            for rg in cb.core_ranges.ranges():
                for x in range(rg.start.x, rg.end.x + 1):
                    for y in range(rg.start.y, rg.end.y + 1):
                        per[(x, y)] = per.get((x, y), 0) + int(cb.total_size)
        xy = lambda c: (int(c.x), int(c.y))  # noqa: E731
        owners = [xy(c) for c in self.mk.place["k"]]
        top = max(per.items(), key=lambda kv: kv[1])
        return dict(t=st["t"], lanes=st.get("lanes", 1), kph=st.get("kph", 1), max_cb_bytes=top[1],
                    max_cb_core=list(top[0]),
                    owner_max=max(per.get(o, 0) for o in owners),
                    ix_core=per.get(xy(self.mk.ix_core), 0), lat_core=per.get(xy(self.mk.lat_core), 0),
                    ix_lane_cores={f"{c.x},{c.y}": per.get(xy(c), 0) for c in self.mk._ix_cores(st["t"])}
                    if st.get("lanes", 1) > 1 else {},
                    ar_lanes=[per.get(tuple(v), 0) for v in self.lanes],
                    stream_lanes={n: per.get(tuple(self.lane_core[n]), 0) for n in st["objs"]})  # fmt: skip

    def _cb_report(self, prog, st: dict, top: int = 12) -> None:
        """GLM53_MK_CB_REPORT: chip (0, 0)'s static CB bytes per buffer index on the `top` cores with the most CB bytes
        (+ their role) and the L1 allocator's view, printed (MK_CB_REPORT lines: diff two builds' reports)"""
        per = {}
        for cb in prog.cbs:
            idx = tuple(int(f.buffer_index) for f in cb.format_descriptors)
            for rg in cb.core_ranges.ranges():
                for x in range(rg.start.x, rg.end.x + 1):
                    for y in range(rg.start.y, rg.end.y + 1):
                        d = per.setdefault((x, y), {})
                        d[idx] = d.get(idx, 0) + int(cb.total_size)
        xy = lambda c: (int(c.x), int(c.y))  # noqa: E731
        role = {xy(c): "owner" for c in self.mk.place["k"]}
        role[xy(self.mk.place["root"])] = "root"
        role[xy(self.mk.lat_core)] = "latent"
        role.update({xy(c): "qnorm" for c in self.mk.qn_cores})
        role[xy(self.mk.ix_core)] = "index"
        role.update({xy(c): "ix_lane" for c in self.mk.co_cores})
        role.update({tuple(v): "ar_lane" for v in self.lanes})
        role.update({tuple(v): f"stream_{n}" for n, v in self.lane_core.items()})
        for c_, _, _ in self.mk.last_readers.assign:
            role.setdefault(xy(c_), "reader")
        try:
            mv = ttnn.get_memory_view(self.mesh, ttnn.BufferType.L1)
            l1 = dict(alloc_per_bank=int(mv.total_bytes_allocated_per_bank),
                      largest_free_per_bank=int(mv.largest_contiguous_bytes_free_per_bank))  # fmt: skip
        except Exception as e:  # noqa: BLE001
            l1 = dict(error=str(e)[:120])
        print(f"MK_CB_REPORT shard t={st['t']} lanes={st.get('lanes', 1)} l1={l1}", flush=True)
        if (
            os.environ.get("GLM53_MK_CB_REPORT") == "2"
        ):  # + every live L1 buffer (largest per bank first), with its tensor
            self._l1_census()
        for core, d in sorted(per.items(), key=lambda kv: -sum(kv[1].values()))[:top]:
            print(f"MK_CB_REPORT shard t={st['t']} core={core} role={role.get(core, '?')} total={sum(d.values())} "
                  + " ".join(f"{'/'.join(map(str, k))}:{v}" for k, v in sorted(d.items())), flush=True)  # fmt: skip

    def _l1_census(self, top: int = 60) -> None:
        """GLM53_MK_CB_REPORT=2: the live L1 buffers of the mesh (ttnn reports.get_buffers: address, bytes per bank,
        layout) joined with the live ttnn tensors that own them (gc: shape, dtype, memory layout), largest first"""
        import gc

        try:
            bufs = ttnn._ttnn.reports.get_buffers(self.mesh)
        except Exception as e:  # noqa: BLE001
            print(f"MK_L1_CENSUS error {str(e)[:160]}", flush=True)
            return
        own = {}

        def where(o):  # the attribute / container names that hold tensor o (one level up)
            out = []
            for r in gc.get_referrers(o):
                if isinstance(r, dict):
                    keys = [str(k) for k, v in r.items() if v is o][:2]
                    holders = [type(h).__name__ for h in gc.get_referrers(r)
                               if getattr(h, "__dict__", None) is r][:1]  # fmt: skip
                    out.append(f"{holders[0] if holders else 'dict'}.{'/'.join(keys)}")
                elif isinstance(r, (list, tuple)):
                    out.append(type(r).__name__ + f"[{len(r)}]")
            return ",".join(out[:3])

        for o in gc.get_objects():
            if type(o).__name__ != "Tensor":
                continue
            try:
                a = int(o.buffer_address())
                mc = o.memory_config()
                if "L1" not in str(mc.buffer_type):
                    continue
                own.setdefault(a, f"{list(o.shape)} {o.dtype} {mc.memory_layout} held_by={where(o)}")
            except Exception:  # noqa: BLE001
                continue
        rows = {}
        for b_ in bufs:
            if "L1" not in str(b_.buffer_type) or int(b_.device_id) != 0:
                continue
            rows[int(b_.address)] = (int(b_.max_size_per_bank), str(b_.buffer_layout))
        tot = sum(v[0] for v in rows.values())
        print(f"MK_L1_CENSUS device 0: {len(rows)} L1 buffers, {tot} B per bank (sum of max sizes)", flush=True)
        for addr, (sz, lay) in sorted(rows.items(), key=lambda kv: -kv[1][0])[:top]:
            print(f"MK_L1_CENSUS addr={addr} per_bank={sz} layout={lay} tensor={own.get(addr, '?')}", flush=True)

    def _noc_names(self, nocs: list, st: dict) -> list:
        by = {tuple(o.lane_noc()[0]): n for n, o in st["objs"].items()}
        return [by.get(tuple(xy), str(xy)) for xy in nocs]

    def _hooked_shard(self, io, desc, t: int, st: dict):
        """TPMla._hooked (the o_proj all-reduce lanes) + the stream lanes + the per-chip rank / peer-position patch."""
        fab = self.fabric(t)
        o_bit = self.mk.phase_order.index("o")
        n_readers = len(self.mk.last_readers.assign)
        slots = dict(self.mk.shard_slots)
        assert {"owner_dm0.cpp", "owner_dm1.cpp", "ix_dm0.cpp", "ix_dm1.cpp", "lat_dm1.cpp"} <= set(slots), slots
        objs = st["objs"]
        order = list(objs)  # kernel order of the stream lanes after the AR lanes
        pa = self.mk.probe.buffer_address() if self.mk.probe is not None else None  # rev 59: the lanes' probe rows
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(self.R):
            for c in range(self.C):
                p = self._pos(r, c)
                # the pair peer's position: x_rs block of pass 0 / its y_ag block (TP2); the 8x4 has no pair peer (0)
                peer = (1 - self.geo.pair_pos(p)) if self.P == 2 else 0
                kernels, seen, audit = [], {"ra": 0, "own": 0}, []
                n_cores, patched = {}, {}  # per slot source: the kernels' cores / the cores patched (review 4)
                for kd in desc.kernels:  # copies: patched per chip
                    if _src_is(kd, "mla/ra.cpp"):
                        rt = max(len(v) for v in _rt_dict(kd).values())
                        defs = list(kd.defines) + [("MK_FAB_MASK", str(1 << o_bit)), ("MK_FAB_BROWS", str(fab.brows)),
                                                   ("MK_FAB_RT", str(rt))]  # fmt: skip
                        kd = _with_rt(kd, fab.producer_args(), defs)[0]
                        seen["ra"] += 1
                    elif _src_is(kd, "mla/owner_dm0.cpp"):
                        rt = max(len(v) for v in _rt_dict(kd).values())
                        defs = [d for d in kd.defines if d[0] != "MK_POST_WAIT_COUNT"]
                        defs += [("MK_POST_WAIT_COUNT", "2"), ("MK_POST_Y_FABRIC", "1"), ("MK_POST_FAB_RT", str(rt))]
                        kd = _with_rt(kd, fab.consumer_args(), defs)[0]
                        seen["own"] += 1
                    name = os.path.basename(str(kd.kernel_source).replace("\\", "/"))
                    if name in slots:  # this chip's rank (and the peer's position) into the shard arg slots
                        n_cores[name] = n_cores.get(name, 0) + len(_rt_dict(kd))
                        for x, y in _rt_dict(kd):
                            for ri, pi, cores, nocs in slots[name]:  # the entry of this core's kernel group
                                if (x, y) not in cores:
                                    continue  # (lat_dm1.cpp also runs on the q-norm cores, without the shard block)
                                kd.runtime_args[x][y][ri] = p
                                if pi is not None:
                                    kd.runtime_args[x][y][pi] = peer
                                patched.setdefault(name, set()).add((x, y))
                                rt = list(kd.runtime_args[x][y])
                                audit.append(
                                    (
                                        name,
                                        (x, y),
                                        rt[ri],
                                        None if pi is None else rt[pi],
                                        [(rt[ri + dx], rt[ri + dy]) for dx, dy in nocs],
                                    )
                                )
                    kernels.append(kd)
                assert seen["ra"] >= 1 and seen["own"] >= 1, seen
                # rev 59 (review 4): every core of a slot source is patched (lat_dm1.cpp: exactly its slot cores = the
                # latent core; its q-norm cores run the same source without the shard block) -- a silently skipped core
                # was the pair's second hang (a zeroed NoC target)
                for name in slots:
                    want = set().union(*(e[2] for e in slots[name])) if name == "lat_dm1.cpp" else None
                    got = patched.get(name, set())
                    if (want is not None and got != want) or (want is None and len(got) != n_cores.get(name, -1)):
                        raise ValueError(
                            f"chip {(r, c)}: {name} patched {len(got)} cores of "
                            f"{n_cores.get(name)} (expected {len(want) if want is not None else 'all'})"
                        )
                # the AR lanes hold their link only around their own round (FAR_LATE_OPEN): at launch start lane h
                # closes and signals every stream lane borrowing link h (pair: AG / RS; TP8: AG1 + AG2 / RS1 + RS2)
                late = {h: (MK.SEM_FSA, [tuple(o.lane_noc()[0]) for o in objs.values() if o.link_base == h])
                        for h in (0, 1)}  # fmt: skip
                assert all(len(v[1]) == len(late[0][1]) for v in late.values()), late  # (same rt widths per lane)
                lk, lcbs = fab.lane_parts(r, c, 1, n_readers, self.mk.owner_noc, cons_sem=MK.SEM_A, late_open=late)
                sk, scbs, sems = [], [], list(desc.semaphores)
                idx = {}
                for nm in order:
                    probe = (pa, SP.PROBE_LANE_ROW[nm]) if pa is not None else None
                    assert probe is None or SP.PROBE_LANE_ROW[nm] < MK.PROBE_ROWS, (nm, MK.PROBE_ROWS)
                    k_, cb_, _, _ = objs[nm].parts(st["x"][nm], r, c, probe=probe)
                    idx[nm] = len(kernels) + len(lk) + len(sk)
                    sk += k_
                    scbs += cb_
                    sems += objs[nm].semaphores()
                prog = ttnn.ProgramDescriptor(kernels=kernels + lk + sk, semaphores=sems,
                                              cbs=list(desc.cbs) + lcbs + scbs)  # fmt: skip
                fab.add_connections(prog, r, c, send_kernel_index=len(kernels))
                for nm in order:
                    objs[nm].connect(prog, r, c, idx[nm])
                self._audits(r, c, st, fab, audit, late)
                if (r, c) == (0, 0):
                    self.census = self._cb_census(prog, st)
                    self.censuses[t] = self.census
                    if os.environ.get("GLM53_MK_CB_REPORT"):  # static CB bytes per CB index on the top cores
                        self._cb_report(prog, st)
                m = ttnn.MeshCoordinate(r, c)
                mpd[ttnn.MeshCoordinateRange(m, m)] = prog
        extra = [objs[nm].sync for nm in order]
        return list(io) + fab.io() + extra, mpd

    def __call__(self, x, cache, page_table, cur_pos, *, t: int, dsa_state: dict | None = None, keep=None,
                 stack: bool = False, s1=None):  # fmt: skip
        """MkMla's call on every chip with the split-K phase: cache / page_table / dsa_state are the per-chip SHARDED
        tensors (tt/dsa_shard.py ShardMap: local blocks, local page table, local pooled keys; the ring replicated)."""
        self.calls += 1
        st = self.streams.get(t)
        if st is None:
            raise ValueError(f"TPMlaShard: no streams for t = {t} (built for {sorted(self.streams)}: ts / stream_ts)")
        with _Capture() as cap:
            out = self.mk(x, cache, page_table, cur_pos, t=t, dsa_state=dsa_state, stack=stack, shard=st, s1=s1)
        ((io, desc),) = cap.calls
        io2, mpd = self._hooked_shard(io, desc, t, st)
        ttnn.generic_op(io2, mpd)
        if keep is not None:
            for k in [k for k in out if k not in keep]:
                v = out.pop(k)
                if v is not None:
                    ttnn.deallocate(v)
        return out

    def release_streams(self):
        for st in self.streams.values():
            for o in st["objs"].values():
                o.release()  # (AG2 lands in AG1's y: only AG1 frees it)
            for k in ("x_ag", "x_rs", "x_rs2", "x_rs2b", "q_asm"):
                if st.get(k) is not None:
                    ttnn.deallocate(st[k])
        self.streams = {}

    def shard_probe(self, mhz: float = 1350.0) -> dict:
        """the last replay's shard phase marks per chip (MkMla(probe=True)): {"r,c": tt/shard_plan.py decode_probe}"""
        import torch

        if self.mk.probe is None:
            return {}
        ar = self.fabric(max(self.streams)).lane_marks() if self.fabrics else {}
        out = {}
        for d, ten in enumerate(ttnn.get_device_tensors(self.mk.probe)):
            p = ttnn.to_torch(ten).to(torch.int64).reshape(-1, 16) & 0xFFFFFFFF
            r, c = self.geo.coord(d)
            out[f"{r},{c}"] = SP.decode_probe(p, line=self.L, mhz=mhz, ar=ar.get(d))
        return out
