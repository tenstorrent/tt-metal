# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Megakernel AR phase for the engine build (AR_INTERFACE.md): ArPhase hands the integrated engine everything it needs
for the real all-reduce on one device of the ring:

  engine_parts(coord, ...) -> dict
    sender_kernels   the AR_SEND reader / writer KernelDescriptors on the 4 sender cores (AR-only role), or, for a
                     combined AR_SEND+GDN role, sender_defines / sender_rd_args / sender_wr_args to host
                     mkar::SendReader / SendWriter (mk_ar_send.hpp, mk_ar_sendw.hpp) in another binary
    sender_cbs       RS / AG staging + packet headers (ids >= 50) on the sender cores
    shared_cbs       the 160 KB RS slot region (CB_RED, owners + senders: the writers address it remotely)
    owner_cbs        AR output (= residual stream on the owning device), residual, Sum(x^2) CBs on the owners
    recv_defines     MKAR_* defines for the owner DM (mkar::Recv, MKAR_ALL_RT=1)
    recv_args[core]  [common block] + [role block] runtime words to append (pass their offset to Recv::init)
    comp_defines     MKAR_* defines for the owner compute (mkar::Reduce)
    comp_args[core]  [has_work, ring, chunk tiles]
    ready_targets[core]  (noc xy, READY word address) the producer (MLP / wo owner DM) increments per release
    semaphores       the fabric-connection program semaphores of the sender cores (add them to the engine program)

Configuration is explicit (no env): rows R (0 = 32), stats, packed delivery, external x / flag / CB_RSTD targets (the
engine's K-row buffers). The global semaphores (arrival + EPOCH base, AG_READY, DONE, READY words) and the slot / output
tensors are owned here and live as long as the object. CB ids are remapped to >= 50 (spec: GDN / mka2 use < 50)."""

import os
from contextlib import contextmanager

import ttnn

from models.demos.blackhole.qwen36.tt.megakernel.oneshot_ar import MKDIR, OneShotARP, OneShotARSK

# recv kernel compile-arg index -> MKAR macro (mk_ps_receiver.cpp layout)
_RECV_CT = ["MKAR_CB_RED", "MKAR_CB_OUT", "MKAR_FUSE_RES", "MKAR_CB_RES", "MKAR_RES_TILES", "MKAR_CHUNK_TILES",
            "MKAR_IN_PAGE", "MKAR_RING", "MKAR_SEM_AG_READY", "MKAR_ITERS", "MKAR_SEM_DONE", "MKAR_PROG",
            "MKAR_SEM_READY", "MKAR_ROWS", "MKAR_DELIVER", "MKAR_STATS", "MKAR_SRED", "MKAR_AGD", "MKAR_INPK",
            "MKAR_CB_PART", "MKAR_CB_SCALER", "MKAR_CB_STAGE", "MKAR_SEM_BASE"]
# compute compile args (mk_ps_reduction_h.cpp layout)
_COMP_CT = {0: "MKAR_CB_RED", 1: "MKAR_CB_OUT", 2: "MKAR_FUSE_RES", 3: "MKAR_CB_RES", 4: "MKAR_ITERS", 5: "MKAR_STATS",
            7: "MKAR_CB_SQ", 8: "MKAR_CB_PART", 9: "MKAR_CB_SCALER"}


@contextmanager
def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    os.environ.update({k: str(v) for k, v in kv.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class ArPhase(OneShotARP):
    CB_BASE = 50

    def __init__(self, mesh_device, n_ar, rows=0, stats=True, deliver=None, x_targets=None, release_delta_ns=1, own_cb=None,
                 cb_ids=None, deliver_ext=False):
        """n_ar: all-reduces per pass (2 per layer: attention + MLP). rows: R (0 = full 32 rows; 4 / 8 / 16 packed).
        deliver: 1 = tile layout into CB_X, 2 = packed stream (default: 2 if rows else 1). x_targets: dict(xk=, xpk=,
        xflag=[slot0, slot1], rstd=) engine buffer / global-semaphore addresses (default: this object's own test buffers).
        release_delta_ns > 0: progressive release (the engine's producers release per chunk; per-position READY words)."""
        deliver = deliver if deliver is not None else (2 if rows else 1)
        with _env(MK_PS_GSEM=1, MK_PS_EPOCH=1, MK_PS_READYP=1, MK_PS_STATS=int(stats), MK_PS_DELIVER=deliver,
                  MK_PS_SEND="hdr", MK_PS_RED="hdr", MK_PS_RECV="new", MK_PS_INPK=0, MK_PS_EXTX=int(x_targets is not None)):
            super().__init__(mesh_device, n_ar, delta_ns=release_delta_ns, rows=rows)
        b = self.CB_BASE
        # own_cb (R = 32): the engine CB holding this device's partial (v12 CB_VXO); CB_RED then has no own slot
        # (ring - 1 slots, -20 KB on the owners) and the compute reads the own partial in place
        self.own_cb = own_cb if not rows else None
        self.noown = self.own_cb is not None
        # CB ids (RS, RED, OUT, HDR, RES, AG, OUTQ, stage, sq, part, scaler). The local CB-config block of EVERY core of
        # the program is sized by the highest CB index used anywhere (dispatch.cpp finalize_cbs: 16 B per index), so ids
        # >= 45 cost the STREAM cores too (ids 50-55 put the u8 STREAM program 192 B over the kernel config buffer).
        # cb_ids=None: the CB_BASE block (50..60, standalone tests); the engine passes ids that are free on the owner /
        # sender cores and <= its own max (bench_mka2.py: the v12 STREAM-only ids).
        ids = list(cb_ids) if cb_ids is not None else [b + i for i in range(11)]
        assert len(ids) == 11 and len(set(ids)) == 11
        self.cbids = tuple(ids[:7])  # RS, RED, OUT, HDR, RES, AG, OUTQ
        self.cb_stage, self.cb_sq, self.cb_part, self.cb_scaler = ids[7:]
        self.n_ar = n_ar
        # deliver_ext (engine, MKAR_DLV_SIGNAL): another RISC delivers x, so the receiver's role block drops its
        # delivery tail (14 words; the runtime-arg region is sized by the largest core of the program)
        self.deliver_ext = deliver_ext
        self._assemble = os.environ.get("MK_PS_ENGINE", "0") == "1"  # test: build the program from the engine parts
        if x_targets is not None and self.deliver_mode:   # (MK_PS_EXTX: the base set no test deliver)
            self._xt = x_targets
            w = mesh_device.worker_core_from_logical_core
            T = ttnn.CoreCoord
            phys_x = [w(T(i, 0)).x for i in range(10)]
            split = next((i for i in range(1, 10) if phys_x[i] != phys_x[i - 1] + 1), 10)

            def prect(lx0, lx1, y):
                if lx1 < lx0:
                    return [0, 0, 0, 0, 0]
                a, b_ = w(T(lx0, y)), w(T(lx1, y))
                return [a.x, a.y, b_.x, b_.y, lx1 - lx0 + 1]

            xb = (lambda c: x_targets["xpk"] + (c % 2) * 10240) if (rows and deliver == 2) else \
                (lambda c: x_targets["xk"] + (c % 2) * 10 * 2048)
            self.deliver = lambda c: ([xb(c)] + prect(0, split - 1, c // 2) + prect(split, 9, c // 2)
                                      + [x_targets["xflag"][c % 2]]
                                      + [x_targets.get("rstd", 0) + 128 * (c % 2), x_targets.get("rstd", 0) + 2048 + 16 * (c % 2)])

    # ---------------------------------------------------------------- engine parts
    def engine_parts(self, coord, inp, buf, out, residual, sem):
        """inp: the producer's CB_XO tensor (owners, 10 tiles each); buf: the RS slot region (owners, ring x PT tiles);
        out: AR output / residual-stream tensor (owners, PT tiles); residual: the residual input tensor (the same buffer
        as out in the engine); sem: the arrival semaphore (its EPOCH base word is allocated here on first use)."""
        sem_addr = ttnn.get_global_semaphore_address(sem)
        prog = self._program(coord, inp, buf, out, residual, sem_addr)
        P = self._parts
        ks = list(prog.kernels)
        recv, comp, rd, wr = ks[P["recv_k"]], ks[P["comp_k"]], ks[P["rd_k"]], ks[P["wr_k"]]
        cbs = list(prog.cbs)
        rct, cct = list(recv.compile_time_args), list(comp.compile_time_args)
        recv_defines = [(n, str(v)) for n, v in zip(_RECV_CT, rct)] + list(recv.defines) + [("MKAR_ALL_RT", "1")]
        comp_defines = [(_COMP_CT[i], str(v)) for i, v in enumerate(cct) if i in _COMP_CT]
        if self.noown:
            comp_defines += [("MKAR_NOOWN", "1"), ("MKAR_CB_OWN", str(self.own_cb))]
        common = list(recv.common_runtime_args)
        recv_args, comp_args, ready = {}, {}, {}
        ntail = len(self.deliver(0)) if (self.deliver_ext and self.deliver and not self.stats) else 0
        for c in ttnn.corerange_to_cores(self.owners_grid(), row_wise=True):
            ra = list(recv.runtime_args[c.x][c.y])
            recv_args[(c.x, c.y)] = common + (ra[:len(ra) - ntail] if ntail else ra)
            comp_args[(c.x, c.y)] = list(comp.runtime_args[c.x][c.y])
        prod = ks[4] if len(ks) > 4 else None  # synthetic producer: its args are the engine producer's READY targets
        if prod is not None:
            for c in ttnn.corerange_to_cores(self.owners_grid(), row_wise=True):
                a = list(prod.runtime_args[c.x][c.y])
                # READY word 0 = the shared READY global semaphore (the owner's own chunk)
                rs = ttnn.get_global_semaphore_address(self.gsem[3])
                ready[(c.x, c.y)] = [(a[2 + 2 * i], a[3 + 2 * i] or rs) for i in range(a[1])]
        return dict(sender_kernels=[rd, wr], sender_cbs=cbs[:P["n_snd_cbs"]], shared_cbs=[cbs[P["red_cb"]]],
                    owner_cbs=cbs[P["red_cb"] + 1:], recv_defines=recv_defines, recv_ncommon=len(common),
                    recv_args=recv_args, comp_defines=comp_defines, comp_args=comp_args, ready_targets=ready,
                    semaphores=list(prog.semaphores), program=prog)

    # ---------------------------------------------------------------- per-core L1 arg table (spec v1.25)
    # words per owner; [ar_on, n_ready, (xy, READY) x n | comp @16..18 | γ table DRAM address @19 | recv @20]
    BLK_NW, BLK_COMP, BLK_GAMMA, BLK_RECV = 64, 16, 19, 20

    def arg_table(self, parts_by_coord, gamma_addr=0):
        """The owners' AR args as a per-core L1 table (uint32, width-sharded over the owners, one row of BLK_NW words per
        core; per-device contents). parts_by_coord: {mesh column: engine_parts(...)}. Returns the tensor; its L1 address
        is uniform (MKAR_ARGS_L1 / the engine's MK_AR_BLK define). The kernels read it at init: no runtime args."""
        import torch

        own = ttnn.corerange_to_cores(self.owners_grid(), row_wise=True)
        rows = []
        for c in range(len(parts_by_coord)):
            P = parts_by_coord[c]
            dev = []
            for core in own:
                k = (core.x, core.y)
                w = [0] * self.BLK_NW
                rt = P["ready_targets"].get(k, [])
                assert 2 + 2 * len(rt) <= self.BLK_COMP
                w[0], w[1] = 1, len(rt)
                for i, (xy, ra) in enumerate(rt):
                    w[2 + 2 * i], w[3 + 2 * i] = xy, ra
                w[self.BLK_COMP:self.BLK_COMP + 3] = P["comp_args"][k]
                w[self.BLK_GAMMA] = gamma_addr  # runtime data (no recompile when the table moves)
                ra = P["recv_args"][k]
                assert self.BLK_RECV + len(ra) <= self.BLK_NW, len(ra)
                w[self.BLK_RECV:self.BLK_RECV + len(ra)] = ra
                dev += w
            rows.append(dev)
        t = torch.tensor(rows, dtype=torch.int64).to(torch.int32).reshape(len(rows), 1, 1, -1)
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
                               ttnn.ShardSpec(self.owners_grid(), [1, self.BLK_NW], ttnn.ShardOrientation.ROW_MAJOR))
        return ttnn.from_torch(t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh, memory_config=mc,
                               mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0))

    def owners_grid(self):
        return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(11, 7))])

    # ---------------------------------------------------------------- test path: program assembled from the parts
    def _program(self, coord, inp, buf, out, residual, sem_addr):
        prog = super()._program(coord, inp, buf, out, residual, sem_addr)
        if not getattr(self, "_assemble", False):
            return prog
        P = self._parts
        ks = list(prog.kernels)
        recv, comp = ks[P["recv_k"]], ks[P["comp_k"]]
        rct, cct = list(recv.compile_time_args), list(comp.compile_time_args)
        rdef = [(n, str(v)) for n, v in zip(_RECV_CT, rct)] + list(recv.defines) + [("MKAR_ALL_RT", "1")]
        ncommon = len(recv.common_runtime_args)
        rdef += [("MKAR_NCOMMON", str(ncommon))]
        rt = ttnn.RuntimeArgs()
        for c in ttnn.corerange_to_cores(self.owners_grid(), row_wise=True):
            rt[c.x][c.y] = list(recv.common_runtime_args) + list(recv.runtime_args[c.x][c.y])
        ks[P["recv_k"]] = ttnn.KernelDescriptor(kernel_source=os.path.join(MKDIR, "mk_ar_recv_engine.cpp"),
                                                core_ranges=recv.core_ranges, defines=rdef, runtime_args=rt,
                                                config=recv.config)
        cdef = [(_COMP_CT[i], str(v)) for i, v in enumerate(cct) if i in _COMP_CT]
        ks[P["comp_k"]] = ttnn.KernelDescriptor(kernel_source=os.path.join(MKDIR, "mk_ar_reduce_engine.cpp"),
                                                core_ranges=comp.core_ranges, defines=cdef,
                                                runtime_args=comp.runtime_args, config=comp.config)
        prog.kernels = ks
        return prog
