# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 14 (PRODUCTION_PATH.md M2): the in-program fabric all-reduce PHASE — all-reduce of a [1,1,B,4096] bf16 partial
over the opened mesh (1x2 pair or the 2x4 mesh) from inside ONE generic_op program per chip: no dispatched CCL op, no
host work inside the step. Designed as a phase of the TP8 layer megakernel (the o_proj / down partial produced on the
16 DS-reader cores, the reduced result consumed by mhc_post) and runnable standalone for validation.

Roles per chip (kernels/fabric_ar/):
  producers (16 cores, 8 tiles each, width-sharded): pack the VALID rows of their tiles (B <= 16: 64 B per row per
      tile; B > 16: whole tiles) into their lane's landing slot, then read the reduced rows back (prod.cpp);
  2 lane cores: lane h serves producers 8h..8h+7 (half of the 4096 columns) and owns ethernet link h in every
      direction (one worker connection per router). BRISC sends (lane_send.cpp), NCRISC turns per-source packet
      counters into CB pushes (lane_recv.cpp), TRISC reduces with fp32 DST accumulation (lane_reduce.cpp).
Schedule (2 rounds = the hierarchical order of ttnn.all_reduce on this mesh, axis 0 then axis 1):
  round 1, mesh axis 0 (the on-board pair): each chip line-multicasts its packed partial into slot p1 of the other
      chip(s) of its column; r1 = slot0 + slot1 (fp32) -> bf16;
  round 2, mesh axis 1 (the 4 chips over cables): each chip multicasts r1 up / down its row into slot p2 of every row
      peer; out = ((r1_0 + r1_1) + (r1_2 + r1_3)) (fp32 DST accumulation) -> bf16.
  Same order on every chip -> every replica is bitwise identical (a replicated residual must not drift between chips).
Synchronisation: data packets are fused write + atomic increment (per-source counters on the receiving lane); every
counter is a monotonic word of a persistent L1 tensor (never reset: no lost increments across back-to-back trace
replays); an end-of-iteration increment to every chip written to (BARRIER) guarantees a peer's landing slots are free
before they are written again. Landing buffers are persistent L1 tensors (same address on every chip).
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.fabric_stream import fabric_connect  # reserved fabric semaphore ids

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "fabric_ar")
TB = 2048
SYNC_WORDS = 56  # 2x4 "cb" mode: 9 lane CB addresses at words 40..48
WORD = 16
CB_L1, CB_L2, CB_OWN2, CB_SEND, CB_RES, CB_OWN = 0, 8, 16, 17, 18, 19
LANE_TILES = 64  # column tiles of the [B, 4096] partial per lane (2 lanes)
# sync word indices (kernels/fabric_ar/far_common.hpp)
W_OWN_READY, W_ARR1, W_ARR2, W_BARRIER, W_RES_DONE, W_GEN_B, W_GEN_N, W_PROBE, W_PROBE_NC = (
    0,
    1,
    9,
    17,
    18,
    19,
    20,
    24,
    32,
)
W_RESULT_READY, W_GEN_P, W_PROD_PROBE, W_CB_ADDR = 0, 1, 24, 40


def kernel_text(name: str) -> str:
    """Kernel source with the local far_common.hpp inlined: SOURCE_CODE kernels are hashed by their text, so an edited
    kernel recompiles even inside a long-lived process (a FILE_PATH kernel is cached by its path)."""
    with open(os.path.join(KDIR, name)) as f:
        src = f.read()
    with open(os.path.join(KDIR, "far_common.hpp")) as f:
        common = f.read().replace("#pragma once", "")
    return src.replace('#include "far_common.hpp"', common)


def packed_bytes_per_tile(b: int) -> int:
    return 64 * b if b <= 16 else TB


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


class FabricPhase:
    """The persistent part of the fabric all-reduce phase on a mesh: lane-core buffers + monotonic counters (allocated
    ONCE, same L1 address on every chip, shared by every program / layer that runs the phase) and the lane kernels.
    Producers (16 cores, 8 column tiles of the [B, 4096] partial each, lane = tile // 64) write their packed valid rows
    into `own` and increment the lane's OWN_READY word; consumers get +1 on a hand-off word when the result is in `res`
    (packed: tile t of lane h at res + (t % 64) * v, face-0 rows then face-1 rows for B <= 16, whole tiles above)."""

    def __init__(self, mesh, b: int, *, lanes=None, pkt_tiles: int = 2, grp: int = 8, mode: str = "tensor"):
        """mode "tensor": own / landing / result buffers are L1 tensors (standalone M2; simple, but an L1 tensor
        reserves its address range on EVERY core); "cb": they are circular buffers of the programs on the lane cores
        only (megakernel: the other cores' large CB regions are untouched), the same CB list in every program and on
        every chip, base addresses read once by `probe()`."""
        assert mode in ("tensor", "cb")
        self.mode = mode
        self.mesh, self.b = mesh, b
        self.R, self.C = int(mesh.shape[0]), int(mesh.shape[1])
        self.lanes = lanes or [(0, 2), (1, 2)]
        assert len(self.lanes) == 2
        self.brows = b if b <= 16 else 32
        # Stage 14c: cb mode, both rounds, B > 16: SEND / OWN2 alias the own-index round-2 slot (_cb_roles)
        self.alias = mode == "cb" and self.R > 1 and self.C > 1 and b > 16
        self.v = packed_bytes_per_tile(b)
        self.ch = 32 * self.brows  # bytes of one face's valid rows (B <= 16)
        self.pt = LANE_TILES * self.v // TB  # pseudo-tiles per lane
        self.pkt_tiles = min(pkt_tiles, self.pt)
        self.grp = grp
        self.n1, self.n2 = self.R, self.C
        rep = ttnn.ReplicateTensorToMesh(mesh)
        lane_crs = _crs(self.lanes)

        def lane_buf():  # pt tiles per lane core, same address on every chip
            mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                                   ttnn.ShardSpec(lane_crs, [32 * self.pt, 32], ttnn.ShardOrientation.ROW_MAJOR))  # fmt: skip
            return ttnn.from_torch(torch.zeros(1, 1, 64 * self.pt, 32, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                                   layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=mc, mesh_mapper=rep)  # fmt: skip

        assert self.n1 in (1, 2), "mesh axis 0 = the on-board pair (1 or 2 chips)"
        # L1 per lane core (B = 32: 128 KiB per buffer): own (the producers' partial) + round-1 landing (a 2-chip axis
        # has ONE peer, so one buffer at one address: the peer always writes there) + n2 round-2 landing slots (slot q =
        # sender position q, the same address on every receiver of a line multicast; a chip's own index slot is never
        # written by a peer and holds this chip's round-1 result = CB_SEND / CB_OWN2) + res
        if mode == "tensor":
            self.own = lane_buf()
            self.land1 = [lane_buf()] if self.n1 == 2 else []
            self.land2 = [lane_buf() for _ in range(self.n2)] if self.n2 > 1 else []
            self.res = lane_buf()
            self.addr = dict(own=self.own.buffer_address(), res=self.res.buffer_address(),
                             land1=[t.buffer_address() for t in self.land1],
                             land2=[t.buffer_address() for t in self.land2])  # fmt: skip
        else:
            self.own, self.res, self.land1, self.land2 = None, None, [], []
            self.addr = None  # probe()
        self.sync_lane = self.sync_buf(self.lanes)
        self._srcs = {n: kernel_text(n) for n in ("prod.cpp", "lane_send.cpp", "lane_recv.cpp", "lane_reduce.cpp",
                                                  "far_probe.cpp")}  # fmt: skip

    def sync_buf(self, cores):
        """monotonic counters + probes, one SYNC_WORDS x 16 B shard per core, zeroed once"""
        words = SYNC_WORDS * WORD // 4
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                               ttnn.ShardSpec(_crs(cores), [1, words], ttnn.ShardOrientation.ROW_MAJOR))  # fmt: skip
        return ttnn.from_torch(torch.zeros(len(cores), words, dtype=torch.int32), dtype=ttnn.uint32,
                               layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh, memory_config=mc,
                               mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))  # fmt: skip

    def io(self):
        bufs = [self.own, self.res] + self.land1 + self.land2 if self.mode == "tensor" else []
        return [self.sync_lane] + bufs

    # ------------------------------------------------------------------ "cb" mode
    def _cb_roles(self, r: int, c: int) -> list:
        """(role, CB index) of the lane CBs in creation order, the SAME sizes / order on every chip and in every
        program (so the base addresses agree): own, round-1 landing, the n2 round-2 slots (all, incl. the own index),
        result, then (both rounds) the round-1 result for BRISC (SEND) and for TRISC (OWN2). Stage 14c, B > 16 with both
        rounds (`self.alias`): SEND / OWN2 are ALIASES of this chip's own-index round-2 slot instead (one CB allocation
        with three indices, as "tensor" mode binds all three to land2[p2]; no peer writes the own index): 7 instead of
        9 x 128 KiB on the 2x4 mesh, which otherwise end above the layer's L1 tensors at B = 32."""
        roles = [("own", CB_OWN)]
        if self.n1 > 1:
            roles.append(("land1", CB_L1 + (1 - r)))
        roles += [(f"land2_{q}", CB_L2 + q) for q in range(self.n2 if self.n2 > 1 else 0)]
        roles.append(("res", CB_RES))
        if self.n1 > 1 and self.n2 > 1 and not self.alias:
            roles += [("send", CB_SEND), ("own2", CB_OWN2)]
        return roles

    def _cb_descs(self, r: int, c: int) -> list:
        lanes = _crs(self.lanes)
        fmt = lambda i: ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=TB)  # noqa: E731
        return [ttnn.CBDescriptor(total_size=self.pt * TB, core_ranges=lanes,
                                  format_descriptors=[fmt(i)] + ([fmt(CB_SEND), fmt(CB_OWN2)]
                                                                 if self.alias and i == CB_L2 + c else []))
                for _, i in self._cb_roles(r, c)]  # fmt: skip

    def probe(self):
        """ "cb" mode: one launch of a program with the lane CB list and far_probe.cpp on every chip -> the CB base
        addresses (checked equal on every chip and both lanes)."""
        assert self.mode == "cb"
        sync = self.sync_lane.buffer_address()
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(self.R):
            for c in range(self.C):
                rt = ttnn.RuntimeArgs()
                ids = [i for _, i in self._cb_roles(r, c)]
                for lane in self.lanes:
                    rt[lane[0]][lane[1]] = [sync, len(ids)] + ids
                k = ttnn.KernelDescriptor(kernel_source=self._srcs["far_probe.cpp"],
                                          source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                          core_ranges=_crs(self.lanes), compile_time_args=[], runtime_args=rt,
                                          config=ttnn.ReaderConfigDescriptor())  # fmt: skip
                m = ttnn.MeshCoordinate(r, c)
                mpd[ttnn.MeshCoordinateRange(m, m)] = ttnn.ProgramDescriptor(kernels=[k], semaphores=[],
                                                                             cbs=self._cb_descs(r, c))  # fmt: skip
        ttnn.generic_op([self.sync_lane, self.sync_lane], mpd)  # generic_op wants >= 1 input + 1 output
        ttnn.synchronize_device(self.mesh)
        seen = set()
        names = [n for n, _ in self._cb_roles(0, 0)]
        for t in ttnn.get_device_tensors(self.sync_lane):
            w = ttnn.to_torch(t).to(torch.int64).reshape(-1, SYNC_WORDS, 4)[:, :, 0] & 0xFFFFFFFF
            for lane in range(w.shape[0]):
                seen.add(tuple(int(w[lane, W_CB_ADDR + i]) for i in range(len(names))))
        if len(seen) != 1:
            raise RuntimeError(f"fabric lane CB addresses differ across chips / lanes: {seen}")
        a = dict(zip(names, next(iter(seen))))
        self.addr = dict(own=a["own"], res=a["res"], land1=[a["land1"]] if "land1" in a else [],
                         land2=[a[f"land2_{q}"] for q in range(self.n2)] if self.n2 > 1 else [])  # fmt: skip
        return a

    def lane_marks(self) -> dict:
        """Per chip, per lane: the raw low-32-bit cycle marks of the lane BRISC (PROBE 0..7: launch start, inputs ready,
        round 1 sent, round 2 sent, result done, launch end, 2nd all-reduce inputs ready / result done) and NCRISC
        (PROBE_NC 0..4) -> {chip: [[brisc x 8], [ncrisc x 5]] x lanes} (0 = not written)."""
        out = {}
        for d, t in enumerate(ttnn.get_device_tensors(self.sync_lane)):
            w = ttnn.to_torch(t).to(torch.int64).reshape(-1, SYNC_WORDS, 4)[:, :, 0] & 0xFFFFFFFF
            out[d] = [[[int(w[lane, W_PROBE + i]) for i in range(8)], [int(w[lane, W_PROBE_NC + i]) for i in range(5)]]
                      for lane in range(w.shape[0])]  # fmt: skip
        return out

    def lane_noc(self):
        return [self.mesh.worker_core_from_logical_core(ttnn.CoreCoord(*lane)) for lane in self.lanes]

    def producer_args(self) -> list:
        """The producer side's runtime block (reader_a.cpp MK_FAB_*): lane 0 / 1 NoC (x, y), own, lane sync."""
        v = self.lane_noc()
        return [v[0].x, v[0].y, v[1].x, v[1].y, self.addr["own"], self.sync_lane.buffer_address()]

    def consumer_args(self) -> list:
        """The consumer side's runtime block (mhc_post reader_p.cpp MK_POST_Y_FABRIC): lane NoC (x, y) x 2, res,
        packed bytes per tile, bytes of one face's valid rows."""
        v = self.lane_noc()
        return [v[0].x, v[0].y, v[1].x, v[1].y, self.addr["res"], self.v, self.ch]

    def lane_parts(
        self,
        r: int,
        c: int,
        iters: int,
        n_prod: int,
        consumers: list,
        cons_addr: int = 0,
        cons_sem: int | None = None,
        cons_sem2: int | None = None,
        late_open: dict | None = None,
    ):
        """(kernels, cbs) of the lane cores for mesh coordinate (r, c); `iters` all-reduces per launch; consumers =
        NoC (x, y) of the cores signalled per lane (+1 at cons_addr, or on program semaphore cons_sem). Call
        `add_connections(prog, r, c)` on the ProgramDescriptor that holds these kernels."""
        mesh = self.mesh
        p1, p2 = r, c
        up1, dn1 = self.n1 - 1 - p1, p1
        up2, dn2 = self.n2 - 1 - p2, p2
        sync = self.sync_lane.buffer_address()
        assert self.addr is not None, "cb mode: probe() first"
        own = self.addr["own"]
        land1 = self.addr["land1"][0] if self.n1 > 1 else 0
        land2 = self.addr["land2"][p2] if self.n2 > 1 else 0
        ct_send = [self.pt, self.pkt_tiles, self.n1, self.n2, CB_SEND, self.grp]
        ct_recv = [self.pt, self.pkt_tiles, self.n1, self.n2, CB_L1, CB_L2, CB_RES, self.grp, CB_OWN]
        ct_red = [
            self.pt,
            self.n1,
            self.n2,
            CB_L1,
            CB_L2,
            CB_OWN2,
            CB_SEND,
            CB_RES,
            CB_OWN,
            int(self.mode == "tensor" or self.alias),
        ]  # ALIAS: CB_OWN2 shares CB_SEND's L1
        rt_send, rt_recv, rt_red = (ttnn.RuntimeArgs() for _ in range(3))
        for h, (lane, v) in enumerate(zip(self.lanes, self.lane_noc())):
            cons = consumers[h] if isinstance(consumers, dict) else consumers  # per lane, or one list for both
            cxy = [int(u) for xy in cons for u in xy]
            send_rt = [sync, own, land1, land2, p1, up1, dn1, p2, up2, dn2, v.x, v.y, n_prod, iters]
            if (
                late_open is not None
            ):  # kernels/fabric_ar/lane_send.cpp FAR_LATE_OPEN (the DSA shard phase, tt/tp_dsa_shard.py):
                # the link is held only around this lane's sends; at launch start (after the deferred credit) it closes
                # and increments semaphore `sem` on `cores` (the stream lane that borrows the link in between)
                sem, cores = late_open.get(h, (0, []))
                send_rt += [len(cores), int(sem)] + [int(u) for xy in cores for u in xy]
            rt_send[lane[0]][lane[1]] = send_rt
            c_arg = cons_sem if cons_sem is not None else cons_addr  # semaphore id (megakernel) or L1 address
            rt_recv[lane[0]][lane[1]] = [sync, p1, p2, n_prod, iters, c_arg, len(cons)] + cxy
            rt_red[lane[0]][lane[1]] = [p1, p2, iters]
        lanes = _crs(self.lanes)
        kd = lambda name, ct, rt, cfg, defs=(): ttnn.KernelDescriptor(
            kernel_source=self._srcs[name], source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=lanes, compile_time_args=ct, runtime_args=rt, config=cfg, defines=list(defs))  # fmt: skip
        cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
                                          fp32_dest_acc_en=True)  # fmt: skip
        rdefs = [("FAR_CONS_SEM_RT", "1")] if cons_sem is not None else []
        if cons_sem is not None and cons_sem2 is not None:
            rdefs.append(("FAR_CONS_SEM2", str(cons_sem2)))
        # FAR_CREDIT_NEXT (Stage 14, 2026-10-20 soak race; DEFAULT since the 1200-s TP2 soak: 378,151 replays bitwise vs
        # 3 bad in 175,945 without it): a launch's last end-of-iteration inc is sent at the start of the next launch
        # (kernels/fabric_ar/lane_send.cpp). GLM53_MK_FAR_CREDIT_NEXT=0 = the old end-of-launch credit (diagnostics
        # only). Every program sharing this phase's counters must use the same setting (one FabricPhase per layer object)
        sdefs = [] if os.environ.get("GLM53_MK_FAR_CREDIT_NEXT") == "0" else [("FAR_CREDIT_NEXT", "1")]
        if late_open is not None:
            sdefs.append(("FAR_LATE_OPEN", "1"))
        kernels = [
            kd("lane_send.cpp", ct_send, rt_send, ttnn.WriterConfigDescriptor(), sdefs),
            kd("lane_recv.cpp", ct_recv, rt_recv, ttnn.ReaderConfigDescriptor(), rdefs),
            kd("lane_reduce.cpp", ct_red, rt_red, cc),
        ]
        if self.mode == "cb":
            return kernels, self._cb_descs(r, c)
        cbs = [ttnn.cb_descriptor_from_sharded_tensor(CB_OWN, self.own)]
        if self.n1 > 1:  # the partner's round-1 data
            cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_L1 + (1 - p1), self.land1[0]))
        cbs += [ttnn.cb_descriptor_from_sharded_tensor(CB_L2 + q, t) for q, t in enumerate(self.land2) if q != p2]
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_RES, self.res))
        if self.n1 > 1 and self.n2 > 1:
            # the round-1 result, packed twice by TRISC (identical bytes, same offsets) into the chip's own-index
            # round-2 slot: CB_SEND (BRISC sends it) and CB_OWN2 (TRISC's round-2 operand)
            cbs += [
                ttnn.cb_descriptor_from_sharded_tensor(CB_OWN2, self.land2[p2]),
                ttnn.cb_descriptor_from_sharded_tensor(CB_SEND, self.land2[p2]),
            ]
        return kernels, cbs

    def add_connections(self, prog, r: int, c: int, send_kernel_index: int):
        """Fabric connections of lane_send (kernel `send_kernel_index` of `prog`), in the kernel's order: axis-0 up,
        axis-0 down, axis-1 up, axis-1 down; lane h uses link h."""
        mesh = self.mesh
        up1, dn1 = self.n1 - 1 - r, r
        up2, dn2 = self.n2 - 1 - c, c
        me = mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
        nbrs = [((r + 1, c), up1), ((r - 1, c), dn1), ((r, c + 1), up2), ((r, c - 1), dn2)]
        for h, lane in enumerate(self.lanes):
            core = ttnn.CoreCoord(*lane)
            for (nr, nc), rng in nbrs:
                if rng:
                    fab = fabric_connect(me, mesh.get_fabric_node_id(ttnn.MeshCoordinate(nr, nc)), h, prog, core)
                    prog.kernels[send_kernel_index].runtime_args[lane[0]][lane[1]].extend(fab)

    def release(self):
        for t in self.io():
            ttnn.deallocate(t)
        self.land1, self.land2 = [], []


class FabricAllReduce(FabricPhase):
    """M2 standalone: all-reduce of x [1,1,B,4096] bf16 (TILE, L1 width-sharded over `producers`, 8 tiles each) ->
    y (same spec); the producers (kernels/fabric_ar/prod.cpp) pack, then read the reduced rows back."""

    def __init__(self, mesh, b: int, *, producers=None, lanes=None, pkt_tiles: int = 2, iters: int = 1, grp: int = 8,
                 mode: str = "tensor"):  # fmt: skip
        super().__init__(mesh, b, lanes=lanes, pkt_tiles=pkt_tiles, grp=grp, mode=mode)
        if mode == "cb":
            self.probe()
        self.iters = iters
        self.producers = producers or [(x, y) for y in range(2) for x in range(8)]
        assert len(self.producers) == 16
        self.sync_prod = self.sync_buf(self.producers)
        self.calls = 0

    # ------------------------------------------------------------------ tensors
    def memory_config(self):
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
                                 ttnn.ShardSpec(_crs(self.producers), [32, 256], ttnn.ShardOrientation.ROW_MAJOR))  # fmt: skip

    def input_from_torch(self, x):
        """x [ndev, 1, B, 4096] (device d = mesh coordinate row-major) -> per-chip partials."""
        return ttnn.from_torch(x.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                               memory_config=self.memory_config(), mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0))  # fmt: skip

    def output(self):
        return ttnn.from_torch(torch.zeros(self.R * self.C, 1, self.b, 4096, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=self.memory_config(),
                               mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0))  # fmt: skip

    # ------------------------------------------------------------------ program
    def _program(self, r: int, c: int, x, y, iters: int):
        mesh = self.mesh
        psync = self.sync_prod.buffer_address()
        noc = lambda xy: mesh.worker_core_from_logical_core(ttnn.CoreCoord(*xy))  # noqa: E731
        prog_cores = [noc(pc) for pc in self.producers]
        rt_prod = ttnn.RuntimeArgs()
        lv = self.lane_noc()
        for h in range(2):
            for k, pc in enumerate(self.producers[8 * h : 8 * h + 8]):
                rt_prod[pc[0]][pc[1]] = [psync, x.buffer_address(), y.buffer_address(), lv[h].x, lv[h].y,
                                         self.addr["own"], self.addr["res"], 8 * k,
                                         self.sync_lane.buffer_address(), iters]  # fmt: skip
        prod = ttnn.KernelDescriptor(kernel_source=self._srcs["prod.cpp"],
                                     source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                     core_ranges=_crs(self.producers), compile_time_args=[self.brows, 8],
                                     runtime_args=rt_prod, config=ttnn.ReaderConfigDescriptor())  # fmt: skip
        # each lane signals (RESULT_READY) its own 8 producers, which then read their rows back
        cons = {h: [(prog_cores[k].x, prog_cores[k].y) for k in range(8 * h, 8 * h + 8)] for h in range(2)}
        kernels, cbs = self.lane_parts(r, c, iters, 8, cons, cons_addr=psync + W_RESULT_READY * WORD)
        prog = ttnn.ProgramDescriptor(kernels=[prod] + kernels, semaphores=[], cbs=cbs)
        self.add_connections(prog, r, c, send_kernel_index=1)
        return prog

    def __call__(self, x, y, iters: int | None = None):
        """One launch = `iters` back-to-back all-reduces of x into y (default: the constructor's)."""
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(self.R):
            for c in range(self.C):
                m = ttnn.MeshCoordinate(r, c)
                mpd[ttnn.MeshCoordinateRange(m, m)] = self._program(r, c, x, y, iters or self.iters)
        ttnn.generic_op([x, y, self.sync_prod] + self.io(), mpd)
        self.calls += 1
        return y

    def release(self):
        super().release()
        ttnn.deallocate(self.sync_prod)

    # ------------------------------------------------------------------ host side checks
    def probes(self, mhz: float = 1350.0):
        """Per chip: lane-0 BRISC / NCRISC marks and the producers' marks of iteration 1 (us after the lane-0 BRISC
        start), the whole-launch end marks, and the persistent iteration counters."""
        out = {}
        lanes = ttnn.get_device_tensors(self.sync_lane)
        prods = ttnn.get_device_tensors(self.sync_prod)
        for d in range(len(lanes)):
            wl = ttnn.to_torch(lanes[d]).to(torch.int64).reshape(-1, SYNC_WORDS, 4)[:, :, 0] & 0xFFFFFFFF
            wp = ttnn.to_torch(prods[d]).to(torch.int64).reshape(-1, SYNC_WORDS, 4)[:, :, 0] & 0xFFFFFFFF
            base = int(wl[0, W_PROBE])

            def us(v):  # signed 32-bit cycle difference (the probes keep the low word)
                dd = (int(v) - base) & 0xFFFFFFFF
                return round((dd - (1 << 32) if dd >= (1 << 31) else dd) / mhz, 2)

            prod = [[us(wp[k, W_PROD_PROBE + i]) for i in range(5)] for k in range(wp.shape[0])]
            out[d] = dict(lane0_brisc=[us(wl[0, W_PROBE + i]) for i in range(6)],
                          lane1_brisc=[us(wl[1, W_PROBE + i]) for i in range(6)],
                          lane0_ncrisc=[us(wl[0, W_PROBE_NC + i]) for i in range(5)],
                          prod_start_min=min(p[0] for p in prod), prod_packed_max=max(p[1] for p in prod),
                          prod_result_seen_max=max(p[2] for p in prod), prod_done_max=max(p[3] for p in prod),
                          prod_end_max=max(p[4] for p in prod),
                          gens=[int(wl[0, W_GEN_B]), int(wl[0, W_GEN_N]), int(wp[0, W_GEN_P])])  # fmt: skip
        return out

    def reference(self, x):
        """Host emulation of the device order: x [ndev, 1, B, 4096] -> [1, 1, B, 4096]: fp32 accumulation, rounded to
        bf16 after each round the way the packer does it (round half AWAY from zero, measured: every mismatch against
        torch's round-to-nearest-even was an exact tie)."""
        xs = x.float().reshape(self.R, self.C, *x.shape[1:])
        r1 = xs[0].clone()
        for q in range(1, self.R):
            r1 = r1 + xs[q]
        r1 = bf16_rha(r1) if self.R > 1 else r1
        acc = None
        for q in range(0, self.C, 2):
            s = r1[q] + r1[q + 1] if self.C > 1 else r1[q]
            acc = s if acc is None else acc + s
        return bf16_rha(acc).unsqueeze(0)


def bf16_rha(t):
    """fp32 -> bf16 values (as fp32) with round half away from zero (the Tensix packer's fp32 -> bf16 rounding)."""
    b = t.float().contiguous().view(torch.int32)
    return ((b + 0x8000) & -65536).view(torch.float32)
