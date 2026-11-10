# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Megakernel AR, pivot step 1: a Python port of all_reduce_async's one-shot program
(ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/all_reduce_async_program_factory.cpp,
create_one_shot_descriptor) as a generic_op, using the same one_shot_{receiver,reader,writer}.cpp + reduction.cpp
kernels and argument layout:
  * one sender per (link, direction), each holding only that direction's fabric connection (1D routers have a single
    worker sender channel per direction);
  * reduce-scatter: every remote-owned output chunk is unicast (fused write + semaphore increment) into slot
    `device index` of its owner core's persistent buffer;
  * owners (output cores of slice = this device) copy their own chunk, wait for ring - 1 increments, reduce with the
    default reduction kernel (+ residual), then tell their link's senders;
  * all-gather: the senders line-multicast every reduced chunk of their link in their direction into the same core's
    output shard on every peer.
Being Python, the core / buffer layout can be changed for the megakernel (step 2: RMSNorm on the owners + x into CB_X;
step 3: a phase with fabric connections opened once per pass). Step 1 reproduces the production layout (RES40 in/out)
to check the port against all_reduce_async (bit-identical) and its per-call time (22.7 us traced).
"""

import os

import ttnn

KROOT = os.path.join(
    os.environ.get("TT_METAL_HOME", "/home/container_app_user/tt-metal"),
    "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels",
)
RS_BATCH_CHUNKS = 4  # kOneShotRsBatchChunks
ROLE_NONE, ROLE_GATHER, ROLE_OWNER = 0, 1, 2
CB_RS, CB_RED, CB_OUT, CB_HDR, CB_RES, CB_AG = 0, 1, 2, 3, 4, 5
CB_OUTQ = 8  # ag_bfp8: BFP8 copy of the reduced chunk (all-gather payload, CB_X source)
SENDER_CORES = ((10, 0), (10, 1), (10, 2), (10, 3))  # (link, dir) -> SENDER_CORES[link * 2 + dir]


def _pack_xy(p):
    return (p.x << 16) | p.y


class OneShotAR:
    _sem_bases = {}  # MK_PS_EPOCH: arrival-semaphore address -> its persistent base-word global semaphore
    def __init__(self, mesh_device, cluster_axis=1, num_links=2, sender_cores=SENDER_CORES):
        self.mesh = mesh_device
        self.axis = cluster_axis
        self.ring = list(mesh_device.shape)[cluster_axis]
        self.links = num_links
        self.sender_cores = [ttnn.CoreCoord(*c) for c in sender_cores]
        self.coords = [ttnn.MeshCoordinate(r, c) for r in range(mesh_device.shape[0]) for c in range(mesh_device.shape[1])]
        self._progs = {}
        self.recv_src = f"{KROOT}/dataflow/one_shot_receiver.cpp"
        self.comp_src = f"{KROOT}/compute/reduction.cpp"
        self.recv_proc = (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1)
        self.rd_src = f"{KROOT}/dataflow/one_shot_reader.cpp"
        self.wr_src = f"{KROOT}/dataflow/one_shot_writer.cpp"
        self.ps_iters = 0  # > 0: persistent kernels (OneShotARP), that many all-reduces per program
        # CB ids (RS, RED, OUT, HDR, RES, AG, OUTQ); the engine remaps them to >= 50 (ArPhase)
        self.cbids = (CB_RS, CB_RED, CB_OUT, CB_HDR, CB_RES, CB_AG, CB_OUTQ)
        self.ps_delta = 0  # > 0 (cycles): progressive producer releases owner chunk c at c x delta (OneShotARP)
        # BFP8 knobs (default off; numerics -> output-length gate): rs_bfp8 = partials / RS slots in BFP8 (reduced into a
        # bf16 output); ag_bfp8 = the all-gather carries a BFP8 copy of each reduced chunk (CB 8 on the output cores,
        # buffer self.xq) while the bf16 result stays only on the chunk's owner device
        self.rs_bfp8 = False
        self.ag_bfp8 = False
        self.xq = None
        self.rows = 0  # 1..16: row-compact AR (only rows 0..rows-1 move; chunks are packed pseudo-tiles), OneShotARP
        self.deliver = None  # OneShotARP: callable(c, core) -> receiver delivery args (x into the K-row CB_X)
        self.stats = False  # OneShotARP (with deliver): row statistic per chunk -> CB_RSTD slot on the K-row
        self.sred = False  # OneShotARP: reduce on arrival (per-source RS flags, fp32 accumulators)
        self.agd = False  # OneShotARP: owner writes its reduced chunk straight into the senders' AG staging
        self.rflags = self.agstage = None
        self.inpk = False  # OneShotARP rows: the input shard already holds the packed stream (v12 CB_XO at R < 32)
        self.cb_part, self.cb_scaler, self.cb_stage, self.cb_sq = 25, 30, 23, 24  # Sum(x^2) CBs (engine: >= 50)
        self.progw = None  # OneShotARP MK_PS_PROGW=1: progress-word buffer
        self.readyp = None  # OneShotARP MK_PS_READYP=1 (with GSEM): 8 per-position READY words on the senders
        self.gsem = None  # OneShotARP MK_PS_GSEM=1: [ag_ready, done, go, ready] global semaphores (engine: no program sems)
        self.noown = False  # ArPhase (R = 32): CB_RED without the owner's own slot (ring - 1 slots; MKAR_NOOWN)

    # hooks for subclasses (norm tail)
    def _comp_ct(self, base_ct, chunk_tiles):
        return base_ct

    def _comp_rt(self, c, owner, chunk_tiles):
        return [1, self.ring - int(self.noown), chunk_tiles] if owner else [ROLE_NONE, 0, 0]

    def _comp_config(self):
        return ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False)

    def _extend(self, program, coord, out, chunk_tiles):
        return program

    def _io_extra(self):
        return []

    def _dist(self, idx):
        fwd = -(-(self.ring - 1) // 2)
        bwd = self.ring - 1 - fwd
        if idx % 2 == 0:  # static alternate
            fwd, bwd = bwd, fwd
        return fwd, bwd

    def _neighbor(self, coord, step):
        r, c = coord[0], coord[1]
        if self.axis == 1:
            c = (c + step) % self.ring
        else:
            r = (r + step) % self.ring
        return ttnn.MeshCoordinate(r, c)

    def _program(self, coord, inp, buf, out, residual, sem_addr):
        CB_RS, CB_RED, CB_OUT, CB_HDR, CB_RES, CB_AG, CB_OUTQ = self.cbids  # local (engine remap)
        mesh = self.mesh
        ring, links = self.ring, self.links
        idx = coord[self.axis]
        n_fwd, n_bwd = self._dist(idx)
        assert n_fwd + n_bwd == ring - 1
        page = TILE_BYTES = 32 * 32 * 2  # bf16 in / out
        BFP8_TILE = 1088
        in_page = BFP8_TILE if self.rs_bfp8 else page
        in_dt = ttnn.bfloat8_b if self.rs_bfp8 else ttnn.bfloat16
        ag_page = BFP8_TILE if self.ag_bfp8 else page
        ag_dt = ttnn.bfloat8_b if self.ag_bfp8 else ttnn.bfloat16
        ag_addr = self.xq.buffer_address() if self.ag_bfp8 else out.buffer_address()
        in_spec, out_spec = inp.memory_config().shard_spec, out.memory_config().shard_spec
        in_shard_tiles = in_spec.shape[1] // 32
        chunk_tiles = out_spec.shape[1] // 32
        in_cores = ttnn.corerange_to_cores(in_spec.grid, row_wise=True)
        out_cores = ttnn.corerange_to_cores(out_spec.grid, row_wise=True)
        nout = len(out_cores)
        assert nout % ring == 0 and len(in_cores) * in_shard_tiles >= nout * chunk_tiles
        cps = nout // ring
        owner_of = lambda c: c // cps
        link_of = lambda c: ((c % cps) + owner_of(c)) % links
        packet = int(ttnn.get_tt_fabric_max_payload_size_bytes())
        ppp = packet // TILE_BYTES
        in_ppp, ag_ppp = packet // in_page, packet // ag_page
        assert ppp > 0
        w = mesh.worker_core_from_logical_core
        dirs = [d for d, n in ((0, n_fwd), (1, n_bwd)) if n > 0]
        nd = len(dirs)
        senders = self.sender_cores[: links * nd]
        out_grid = out_spec.grid
        snd_range = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in senders])
        all_cores = out_grid.merge(snd_range)

        def segs(c):
            if self.rows:  # row-compact: chunk c = the whole input shard of core c (packed by the reader / receiver)
                return [1, _pack_xy(w(in_cores[c])), 0, in_shard_tiles]
            s = [0]
            t, end = c * chunk_tiles, c * chunk_tiles + chunk_tiles
            while t < end:
                ic, off = t // in_shard_tiles, t % in_shard_tiles
                n = min(in_shard_tiles - off, end - t)
                s += [_pack_xy(w(in_cores[ic])), off, n]
                s[0] += 1
                t += n
            return s

        def route_to(s):
            f = (s + ring - idx) % ring
            if f <= n_fwd:
                return 0, f
            b = ring - f
            assert b <= n_bwd
            return 1, b

        # progressive release order (MK_PS_ORDER): "index" = chunk c at rank c on every device; "rot" = per device, the
        # farthest-owned chunks first and this device's own chunks last (they need no reduce-scatter hop)
        def hops(c):
            if owner_of(c) == idx:
                return 0
            return route_to(owner_of(c))[1]
        if os.environ.get("MK_PS_ORDER", "index") == "rot":
            order = sorted(range(nout), key=lambda c: (-hops(c), c))
        else:
            order = list(range(nout))
        rank = {c: r for r, c in enumerate(order)}
        rs_chunks = [[] for _ in range(links)]
        ag_chunks = [[] for _ in range(links)]
        for link in range(links):
            per_dir = [[], []]
            for c in range(nout):
                if link_of(c) != link:
                    continue
                if owner_of(c) == idx:
                    ag_chunks[link].append(c)
                else:
                    per_dir[route_to(owner_of(c))[0]].append(c)
            hsel = int(os.environ.get("MK_PS_HOPSEL", "0"))  # debug: RS only the chunks of hop distance hsel (99: none)
            if hsel:
                per_dir = [[c for c in v if route_to(owner_of(c))[1] == hsel] for v in per_dir]
            for v in per_dir:
                if self.ps_delta:
                    v.sort(key=lambda c: rank[c])  # progressive producer: release order
                else:
                    v.sort(key=lambda a: -route_to(owner_of(a))[1])  # stable: farthest owner first
            for i in range(max(len(per_dir[0]), len(per_dir[1]))):
                for v in per_dir:
                    if i < len(v):
                        rs_chunks[link].append(v[i])
            if self.ps_delta:
                rs_chunks[link].sort(key=lambda c: rank[c])
        max_ag = max(1, max(len(a) for a in ag_chunks))

        def fmt(i, dt, ps):
            return [ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=ps)]

        hdr = int(ttnn.get_tt_fabric_packet_header_size_bytes())
        cbs = [ttnn.CBDescriptor(total_size=4 * hdr, core_ranges=snd_range, format_descriptors=fmt(CB_HDR, ttnn.uint32, hdr))]
        n_snd_cbs0 = 0  # sender-only CBs are cbs[:n_snd_cbs] (recorded below)
        if not self.ps_iters:
            cbs += [
                ttnn.CBDescriptor(total_size=2 * RS_BATCH_CHUNKS * chunk_tiles * in_page, core_ranges=snd_range,
                                  format_descriptors=fmt(CB_RS, in_dt, in_page)),
                ttnn.CBDescriptor(total_size=max_ag * chunk_tiles * ag_page, core_ranges=snd_range,
                                  format_descriptors=fmt(CB_AG, ag_dt, ag_page)),
            ]
        else:
            # persistent: a CB pointer only wraps when it lands exactly on the CB end, so each sender's RS / AG CB is
            # sized to exactly its own per-iteration traffic (the single-pass sizes would drift past the end from the
            # second iteration on). RS: the whole per-iteration list (<= 2 reader batches).
            for link in range(links):
                for d, dr in enumerate(dirs):
                    core = senders[link * nd + d]
                    one = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
                    n_rs = len([c for c in rs_chunks[link] if route_to(owner_of(c))[0] == dr])
                    n_ag = len(ag_chunks[link])
                    # the reader pushes batches of min(RS_BATCH_CHUNKS, left): with <= 2 batches they end exactly at
                    # n_rs chunks, the CB end
                    assert n_rs <= 2 * RS_BATCH_CHUNKS, f"persistent RS CB: {n_rs} chunks per iteration"
                    cbs.append(ttnn.CBDescriptor(total_size=max(1, n_rs) * chunk_tiles * in_page, core_ranges=one,
                                                 format_descriptors=fmt(CB_RS, in_dt, in_page)))
                    cbs.append(ttnn.CBDescriptor(total_size=max(1, n_ag) * chunk_tiles * ag_page, core_ranges=one,
                                                 format_descriptors=fmt(CB_AG, ag_dt, ag_page)))
        red = ttnn.cb_descriptor_from_sharded_tensor(CB_RED, buf)
        red.core_ranges = all_cores
        red.total_size = chunk_tiles * (ring - int(self.noown)) * in_page
        red.format_descriptors = fmt(CB_RED, in_dt, in_page)
        o = ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out)
        o.total_size = chunk_tiles * page
        o.format_descriptors = fmt(CB_OUT, ttnn.bfloat16, page)
        n_snd_cbs = len(cbs)
        cbs += [red, o]
        if self.ag_bfp8:
            q = ttnn.cb_descriptor_from_sharded_tensor(CB_OUTQ, self.xq)
            q.total_size = chunk_tiles * ag_page
            q.format_descriptors = fmt(CB_OUTQ, ag_dt, ag_page)
            cbs.append(q)
        fuse = residual is not None
        if fuse:
            r = ttnn.cb_descriptor_from_sharded_tensor(CB_RES, residual)
            r.total_size = chunk_tiles * page
            r.format_descriptors = fmt(CB_RES, ttnn.bfloat16, page)
            cbs.append(r)
        sems = [ttnn.SemaphoreDescriptor(0, ttnn.CoreType.WORKER, all_cores, 0)]  # ag_ready
        ps = self.ps_iters
        if ps:
            sems.append(ttnn.SemaphoreDescriptor(1, ttnn.CoreType.WORKER, all_cores, 0))  # DONE (output cores -> senders)
            sems.append(ttnn.SemaphoreDescriptor(2, ttnn.CoreType.WORKER, all_cores, 0))  # GO (master sender -> owners)
            sems.append(ttnn.SemaphoreDescriptor(3, ttnn.CoreType.WORKER, all_cores, 0))  # READY (producer -> consumer)
        # GSEM (persistent only, MK_PS_GSEM=1): no program semaphores; ids 0..3 become global-semaphore addresses
        gsem = bool(ps) and self.gsem is not None
        SID = (lambda i: ttnn.get_global_semaphore_address(self.gsem[i])) if gsem else (lambda i: i)
        GDEF = [("MK_PS_GSEM", "1")] if gsem else []
        if self.noown:
            assert gsem and not self.rows and not self.sred, "MKAR_NOOWN: engine form, R = 32, no SRED"
            GDEF = GDEF + [("MKAR_NOOWN", "1")]
        # MK_PS_EPOCH=1 (v0.9.14, with GSEM): the RS / AG arrival semaphore is never reset; a persistent per-core base word
        # (one per arrival semaphore, shared by every program that uses it) carries the consumed count across programs
        epoch_ct = []
        if gsem and os.environ.get("MK_PS_EPOCH", "0") == "1":
            if sem_addr not in OneShotAR._sem_bases:
                # MK_PS_EPOCH_SEED (test): start the base (and the test's arrival semaphore) near 2^32 to cross the wrap
                OneShotAR._sem_bases[sem_addr] = ttnn.create_global_semaphore(
                    mesh, out_grid, int(os.environ.get("MK_PS_EPOCH_SEED", "0"), 0))
            epoch_ct = [ttnn.get_global_semaphore_address(OneShotAR._sem_bases[sem_addr])]
            GDEF = GDEF + [("MK_PS_EPOCH", "1")]
        if gsem:
            sems = []
        # MK_PS_PROGW=1 (hang hunt): per-RISC progress words on the AR cores (slot 0 writer, 1 reader, 2 receiver, 3 producer)
        if ps and self.progw is not None:
            GDEF = GDEF + [("MK_PS_PROGW", str(self.progw.buffer_address()))]
        prog = 1 if (ps and self.ps_delta) else 0
        readyp = bool(prog) and gsem and self.readyp is not None  # per-position READY words (engine form)
        DM = ttnn.DataMovementConfigDescriptor
        recv_rt, comp_rt, rd_rt, wr_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for c, core in enumerate(out_cores):
            if owner_of(c) == idx:
                hsel = int(os.environ.get("MK_PS_HOPSEL", "0"))
                nwait = ring - 1
                if hsel:
                    def hops_from(src):  # src's route to this owner device (src's own fwd / bwd split)
                        fw = -(-(ring - 1) // 2)
                        bw = ring - 1 - fw
                        if src % 2 == 0:
                            fw, bw = bw, fw
                        f = (idx - src) % ring
                        return f if f <= fw else ring - f
                    nwait = sum(1 for src in range(ring) if src != idx and hops_from(src) == hsel)
                a = [ROLE_OWNER, nwait, idx, nd]
                for d in range(nd):
                    a.append(_pack_xy(w(senders[link_of(c) * nd + d])))
                order = sorted(range(ring), key=lambda sc: (0 if sc == idx else 1, min((idx - sc) % ring, (sc - idx) % ring),
                                                            (idx - sc) % ring))
                sred_args = [self.rflags.buffer_address()] + order if (ps and self.sred) else []
                recv_rt[core.x][core.y] = a + segs(c) + sred_args + (self.deliver(c) if self.deliver else [])
                comp_rt[core.x][core.y] = self._comp_rt(c, True, chunk_tiles) + (order if (ps and self.sred) else [])
            else:
                recv_rt[core.x][core.y] = [ROLE_GATHER, 1] + (self.deliver(c) if self.deliver else [])
                comp_rt[core.x][core.y] = self._comp_rt(c, False, chunk_tiles)
        src = mesh.get_fabric_node_id(coord)
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=self.recv_src, core_ranges=out_grid, defines=GDEF if self.ps_iters else [],
                compile_time_args=[CB_RED, CB_OUTQ if self.ag_bfp8 else CB_OUT, int(fuse), CB_RES, chunk_tiles if fuse else 0,
                                   chunk_tiles, in_page, ring, SID(0)]
                + ([ps, SID(1), prog, SID(3), self.rows, self.deliver_mode if self.deliver else 0, int(self.stats), int(self.sred),
                    int(self.agd), int(self.inpk), self.cb_part, self.cb_scaler, self.cb_stage] + epoch_ct if ps else []),
                runtime_args=recv_rt,
                common_runtime_args=[sem_addr, inp.buffer_address()]
                + ([len(senders)] + [_pack_xy(w(c)) for c in senders] if ps else [])
                + ([self.agstage.buffer_address()] if (ps and self.agd) else []),
                config=DM(processor=self.recv_proc[0], noc=self.recv_proc[1]),
            ),
            ttnn.KernelDescriptor(
                kernel_source=self.comp_src, core_ranges=out_grid,
                compile_time_args=self._comp_ct([CB_RED, CB_OUT, int(fuse), CB_RES], chunk_tiles)
                + ([ps, int(self.stats), int(self.sred), self.cb_sq, self.cb_part, self.cb_scaler] if ps else []),
                runtime_args=comp_rt,
                config=self._comp_config(),
            ),
        ]
        rd_k = ttnn.KernelDescriptor(
            kernel_source=self.rd_src, core_ranges=snd_range, defines=GDEF if self.ps_iters else [],
            compile_time_args=[CB_RS, CB_AG, in_page, ag_page, chunk_tiles, 1 if prog else RS_BATCH_CHUNKS, SID(0)]
            + ([ps, SID(1), nout, prog, SID(3), SID(2), self.rows, int(self.agd), int(self.inpk)] if ps else [])
            + ([int(readyp), ] + [ttnn.get_global_semaphore_address(g) for g in self.readyp] if (ps and readyp) else
               ([0] + [0] * 8 if ps else [])),
            runtime_args=rd_rt,
            common_runtime_args=[inp.buffer_address(), ag_addr]
            + ([nout] + [_pack_xy(w(c)) for c in out_cores] if ps else []),
            config=DM(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1),
        )
        wr_k = ttnn.KernelDescriptor(
            kernel_source=self.wr_src, core_ranges=snd_range, defines=GDEF if self.ps_iters else [],
            compile_time_args=[idx, CB_HDR, CB_RS, CB_AG, in_page, ag_page, chunk_tiles, in_ppp, ag_ppp, CB_RED] + ([ps, prog, int(os.environ.get("MK_PS_HDRROT", "0")), int(self.sred), int(self.agd)] if ps else []),
            runtime_args=wr_rt,
            common_runtime_args=[sem_addr, ag_addr]
            + ([self.rflags.buffer_address() if self.sred else 0, self.agstage.buffer_address() if self.agd else 0]
               if ps else []),
            config=DM(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
        )
        program = ttnn.ProgramDescriptor(kernels=kernels + [rd_k, wr_k], semaphores=sems, cbs=cbs)
        # engine parts (ArPhase.engine_parts): CB split and kernel indices of this device's program
        self._parts = dict(n_snd_cbs=n_snd_cbs, red_cb=n_snd_cbs, recv_k=0, comp_k=1, rd_k=2, wr_k=3)
        wr_index = len(program.kernels) - 1
        rd_index = wr_index - 1
        for link in range(links):
            for d, dr in enumerate(dirs):
                core = senders[link * nd + d]
                my_rs = [c for c in rs_chunks[link] if route_to(owner_of(c))[0] == dr]
                ra = [len(my_rs), len(ag_chunks[link])] + ([1 if (link == 0 and d == 0) else 0] if ps else [])
                wa = [len(my_rs), len(ag_chunks[link]), n_fwd if dr == 0 else 0, n_bwd if dr == 1 else 0]
                for c in my_rs:
                    ra += segs(c)
                    # NOOWN: the destination owner's slot for this chip (it has none for itself) in bits 16+
                    dst = owner_of(c)
                    wa += [_pack_xy(w(out_cores[c])), (dr << 8) | route_to(dst)[1] | (((idx - (1 if idx > dst else 0)) << 16) if self.noown else 0)]
                for c in ag_chunks[link]:
                    xy = _pack_xy(w(out_cores[c]))
                    ra.append(xy)
                    wa.append(xy)
                program.kernels[rd_index].runtime_args[core.x][core.y] = ra
                wa.append(1 if dr == 0 else 0)
                if dr == 0:
                    wa += ttnn.setup_fabric_connection(src_fabric_node_id=src,
                                                       dst_fabric_node_id=mesh.get_fabric_node_id(self._neighbor(coord, 1)),
                                                       link_idx=link, program_descriptor=program, worker_core=core)
                wa.append(1 if dr == 1 else 0)
                if dr == 1:
                    wa += ttnn.setup_fabric_connection(src_fabric_node_id=src,
                                                       dst_fabric_node_id=mesh.get_fabric_node_id(self._neighbor(coord, -1)),
                                                       link_idx=link, program_descriptor=program, worker_core=core)
                program.kernels[wr_index].runtime_args[core.x][core.y] = wa
        if prog:
            assert max(len(a) for a in ag_chunks) <= 1, "progressive AG staging assumes <= 1 own chunk per link"
            prt = ttnn.RuntimeArgs()
            for c, core in enumerate(out_cores):
                if owner_of(c) == idx:
                    tgt = [(core, 0)]
                else:
                    dr, _ = route_to(owner_of(c))
                    l_ = link_of(c)
                    ra = 0
                    if readyp:
                        my_rs = [c2 for c2 in rs_chunks[l_] if route_to(owner_of(c2))[0] == dr]
                        ra = ttnn.get_global_semaphore_address(self.readyp[my_rs.index(c)])
                    tgt = [(senders[l_ * nd + dirs.index(dr)], ra)]
                prt[core.x][core.y] = [rank[c], len(tgt)] + [v for t, ra in tgt for v in (_pack_xy(w(t)), ra)]
            program.kernels = list(program.kernels) + [ttnn.KernelDescriptor(
                kernel_source=os.path.join(MKDIR, "mk_ps_producer.cpp"), core_ranges=out_grid,
                compile_time_args=[ps, SID(2), SID(3), self.ps_delta], runtime_args=prt, defines=GDEF,
                config=DM(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0))]
        return self._extend(program, coord, out, chunk_tiles)

    def __call__(self, inp, buf, out, sem, residual=None):
        """inp / out / residual: width-sharded [32, H] (out = the slice-owner layout, e.g. RES40); buf: the persistent
        [32, H * ring] buffer on the output cores; sem: a global semaphore on (output cores + sender cores)."""
        sem_addr = ttnn.get_global_semaphore_address(sem)
        key = (inp.buffer_address(), buf.buffer_address(), out.buffer_address(),
               residual.buffer_address() if residual is not None else 0, sem_addr)
        mp = self._progs.get(key)
        if mp is None:
            mp = ttnn.MeshProgramDescriptor()
            for coord in self.coords:
                mp[ttnn.MeshCoordinateRange(coord, coord)] = self._program(coord, inp, buf, out, residual, sem_addr)
            self._progs[key] = mp
        io = [inp, buf] + ([residual] if residual is not None else []) + self._io_extra() + ([self.xq] if self.ag_bfp8 else []) + [out]
        return ttnn.generic_op(io, mp)


MKDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
NORM_SCALER_BF16 = 0x3980  # 2^-12 (exact in bf16); output x = RMSNorm(r) / NORM_FIX, NORM_FIX folded into the consumer
NORM_FIX = (5120 / 4096.0) ** 0.5


class OneShotARNorm(OneShotAR):
    """Step 2: the one-shot AR on the megakernel layout (owners = 16 non-STREAM AR cores (0,8)-(7,9), 10 tiles each;
    input = the 80 STREAM cores' CB_OUT) + RMSNorm on the owners (leader stats over the 16) + 4-root x delivery into
    CB_X (the full [32, 5120] on every STREAM core). The new residual stays in the owners' output shards."""

    def __init__(self, mesh_device, eps=1e-6, cluster_axis=1, num_links=2):
        super().__init__(mesh_device, cluster_axis, num_links)
        self.recv_src = os.path.join(MKDIR, "mk_os_receiver.cpp")
        self.comp_src = os.path.join(MKDIR, "mk_os_compute.cpp")
        self.eps = eps
        T = ttnn.CoreCoord
        self.owners = ttnn.CoreRangeSet([ttnn.CoreRange(T(0, 8), T(7, 9))])
        self.stream = ttnn.CoreRangeSet([ttnn.CoreRange(T(0, 0), T(9, 7))])
        self.own_cores = ttnn.corerange_to_cores(self.owners, row_wise=True)
        self.nstat = len(self.own_cores)  # 16
        self.per_root = self.nstat // 4
        W = lambda grid, tiles, dt=ttnn.bfloat16: ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, (32, tiles * 32), ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.in_mem, self.out_mem = W(self.stream, 2), W(self.owners, 10)
        self.buf_mem = W(self.owners, 10 * self.ring)
        E = lambda shape, mem, dt=ttnn.bfloat16: ttnn.empty(shape, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                                            memory_config=mem)
        self.stats = E([1, 1, 32, self.nstat * self.nstat * 32], W(self.owners, self.nstat), ttnn.float32)
        self.rstd_in = E([1, 1, 32, self.nstat * 32], W(self.owners, 1), ttnn.float32)
        self.gather = E([1, 1, 32, self.nstat * self.per_root * 10 * 32], W(self.owners, self.per_root * 10))
        self.xbuf = E([1, 1, 32 * 80, 5120], ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
            ttnn.ShardSpec(self.stream, (32, 5120), ttnn.ShardOrientation.ROW_MAJOR)))
        import torch

        self.eps_t = ttnn.from_torch(torch.full((1, 1, 32, self.nstat * 32), eps * 5120 / 4096.0), dtype=ttnn.float32,
                                     layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=W(self.owners, 1),
                                     mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
        gs = lambda crs: ttnn.create_global_semaphore(mesh_device, crs, 0)
        self.stat_cnt, self.rstd_flag, self.root_cnt = gs(self.owners), gs(self.owners), gs(self.owners)
        self.xflags = [gs(self.stream.merge(self.owners)) for _ in range(4)]
        w = mesh_device.worker_core_from_logical_core
        phys_x = [w(T(i, 0)).x for i in range(10)]
        split = next((i for i in range(1, 10) if phys_x[i] != phys_x[i - 1] + 1), 10)
        self.split = split

        def prect(lx0, lx1, y0, y1):
            if lx1 < lx0:
                return [0, 0, 0, 0, 0]
            a, b = w(T(lx0, y0)), w(T(lx1, y1))
            return [a.x, a.y, b.x, b.y, (lx1 - lx0 + 1) * (y1 - y0 + 1)]

        self.srects = prect(0, split - 1, 0, 7) + prect(split, 9, 0, 7)
        self.orects = prect(0, min(split, 8) - 1, 8, 9) + prect(min(split, 8), 7, 8, 9)
        self.onoc = [w(c) for c in self.own_cores]

    def _comp_ct(self, base_ct, chunk_tiles):
        return base_ct + [chunk_tiles, self.nstat, 0]

    def _comp_rt(self, c, owner, chunk_tiles):
        lead = 1 if c == 0 else 0
        return ([1, self.ring, chunk_tiles] if owner else [0, 0, 0]) + [lead]

    def _comp_config(self):
        return ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                            fp32_dest_acc_en=True, dst_full_sync_en=True)

    def _io_extra(self):
        return [self.stats, self.rstd_in, self.gather, self.eps_t, self.xbuf]

    def _extend(self, program, coord, out, chunk_tiles):
        tb, tf = 32 * 32 * 2, 32 * 32 * 4

        def plain(idx, dt, pages, page):
            return ttnn.CBDescriptor(total_size=pages * page, core_ranges=self.owners,
                                     format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)])

        cb = ttnn.cb_descriptor_from_sharded_tensor
        program.cbs = list(program.cbs) + [
            plain(24, ttnn.bfloat16, chunk_tiles, tb), plain(25, ttnn.float32, 1, tf), cb(26, self.stats),
            plain(27, ttnn.float32, 1, tf), cb(28, self.eps_t), plain(29, ttnn.bfloat16, chunk_tiles, tb),
            plain(30, ttnn.bfloat16, 1, tb), cb(31, self.rstd_in),
        ]
        G = ttnn.get_global_semaphore_address
        rt = ttnn.RuntimeArgs()
        lead = self.onoc[0]
        for j, c in enumerate(self.own_cores):
            k = j // self.per_root
            root = self.onoc[k * self.per_root]
            rt[c.x][c.y] = (
                [j, self.xbuf.buffer_address(), self.stats.buffer_address(), self.gather.buffer_address(),
                 self.rstd_in.buffer_address()] + self.srects + self.orects
                + [0 if c.x < self.split else 1, lead.x, lead.y, root.x, root.y, G(self.stat_cnt), G(self.rstd_flag),
                   G(self.root_cnt), G(self.xflags[k]), 1 if j == 0 else 0, k if j == k * self.per_root else 0xFF]
            )
        tail = ttnn.KernelDescriptor(
            kernel_source=os.path.join(MKDIR, "mk_os_tail.cpp"), core_ranges=self.owners,
            compile_time_args=[chunk_tiles, chunk_tiles * tb, tf, self.nstat, self.per_root, NORM_SCALER_BF16, 0],
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
        )
        program.kernels = list(program.kernels) + [tail]
        return program


class OneShotARX(OneShotARNorm):
    """Spec v0.4 candidate: the one-shot AR on the megakernel layout with NO norm (CB_X carries the un-normed residual;
    consumers fold rstd and gamma), and x delivery overlapped with the all-gather: each of the 16 owners multicasts its
    10-tile chunk into CB_X on the 80 STREAM cores as soon as it lands, with one flag per chunk (16 flags)."""

    def __init__(self, mesh_device, cluster_axis=1, num_links=2):
        super().__init__(mesh_device, 1e-6, cluster_axis, num_links)
        # owners: the one-shot reduction (+ residual), then every owner the row partial sum(x^2) x 2^-12 (PART_ONLY)
        self.comp_src = os.path.join(MKDIR, "mk_os_compute.cpp")
        # CB_RSTD on the STREAM cores: [16 owners][32 rows] fp32 partials at 0, tag word of owner j at 2048 + 16 j
        self.rbuf = ttnn.empty([1, 1, 32 * 80, 32], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                               memory_config=ttnn.MemoryConfig(
                                   ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                                   ttnn.ShardSpec(self.stream, (32, 32), ttnn.ShardOrientation.ROW_MAJOR)))
        gs = lambda crs: ttnn.create_global_semaphore(mesh_device, crs, 0)
        self.xflags16 = [gs(self.stream.merge(self.owners)) for _ in range(self.nstat)]
        self.xtok = gs(self.owners)
        self.xgroups = int(os.environ.get("MK_X_GROUPS", "16"))
        self.root4 = int(os.environ.get("MK_X_ROOT4", "0"))
        # x multicast on NOC1 (-x, -y): from the owner rows (y 8-9) it enters the STREAM rect (y 0-7) at its bottom
        # edge instead of wrapping around the torus on NOC0; the receiver moves to RISCV_0/NOC0. MK_X_NOC=0 reverts.
        self.xnoc = int(os.environ.get("MK_X_NOC", "1"))
        if self.xnoc == 1:
            self.recv_proc = (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0)

    def _comp_ct(self, base_ct, chunk_tiles):
        return base_ct + [chunk_tiles, self.nstat, 1]

    def _io_extra(self):
        return [self.xbuf, self.rbuf, self.gather]

    def _extend(self, program, coord, out, chunk_tiles):
        tb, tf = 32 * 32 * 2, 32 * 32 * 4

        def plain(idx, dt, pages, page):
            return ttnn.CBDescriptor(total_size=pages * page, core_ranges=self.owners,
                                     format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)])

        program.cbs = list(program.cbs) + [
            plain(23, ttnn.float32, 1, 256), plain(24, ttnn.bfloat16, chunk_tiles, tb), plain(25, ttnn.float32, 1, tf),
            plain(30, ttnn.bfloat16, 1, tb),
        ]
        G = ttnn.get_global_semaphore_address
        rt = ttnn.RuntimeArgs()
        for j, c in enumerate(self.own_cores):
            nj = j + self.xgroups
            root = self.onoc[(j // self.per_root) * self.per_root]
            nxt = self.onoc[nj] if nj < self.nstat else None
            rt[c.x][c.y] = ([j, self.xbuf.buffer_address(), self.rbuf.buffer_address()] + self.srects
                            + [G(self.xflags16[j]), G(self.xtok), 1 if j >= self.xgroups else 0,
                               nxt.x if nxt else 0, nxt.y if nxt else 0xFFFF]
                            + [self.gather.buffer_address(), root.x, root.y, G(self.root_cnt),
                               1 if j % self.per_root == 0 else 0])
        tail = ttnn.KernelDescriptor(
            kernel_source=os.path.join(MKDIR, "mk_os_xtail.cpp"), core_ranges=self.owners,
            compile_time_args=[CB_OUT, chunk_tiles, chunk_tiles * tb, 0, int(os.environ.get("MK_X_ONLY", "16")),
                               NORM_SCALER_BF16, self.root4,
                               int(os.environ.get("MK_X_PART", "1"))], runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1 if self.xnoc else ttnn.DataMovementProcessor.RISCV_0,
                noc=ttnn.NOC.NOC_1 if self.xnoc else ttnn.NOC.NOC_0),
        )
        program.kernels = list(program.kernels) + [tail]
        return program


class OneShotARSK(OneShotAR):
    """Spec v0.6 (§12) split-K AR: 16 owners in-row at logical (10 + j % 2, j // 2) (owner j holds chunk j, 10 tiles),
    input = this device's partial of each chunk already on its owner (the split-K down stage lands there), one-shot
    reduce-scatter + all-gather (+ residual) on the owners, then each owner multicasts its chunk west on NOC1 into CB_X
    slot j % 2 of K-row j // 2 only (40 KB per STREAM core) + x flag (slot), and its sum(x^2) x 2^-12 partial (+ tag)
    to the 10 N-column roots (logical row MK_SK_ROOT_ROW, default 0). Senders move to (10..11, 8..9)."""

    SK_SENDERS = ((10, 8), (11, 8), (10, 9), (11, 9))

    def __init__(self, mesh_device, cluster_axis=1, num_links=2):
        super().__init__(mesh_device, cluster_axis, num_links, sender_cores=self.SK_SENDERS)
        T = ttnn.CoreCoord
        self.recv_src = os.path.join(MKDIR, "mk_os_receiver.cpp")
        self.comp_src = os.path.join(MKDIR, "mk_os_compute.cpp")
        self.recv_proc = (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0)
        self.owners = ttnn.CoreRangeSet([ttnn.CoreRange(T(10, 0), T(11, 7))])
        self.own_cores = ttnn.corerange_to_cores(self.owners, row_wise=True)  # (10,0), (11,0), (10,1), ... = j
        self.nstat = len(self.own_cores)
        self.stream = ttnn.CoreRangeSet([ttnn.CoreRange(T(0, 0), T(9, 7))])
        self.root_row = int(os.environ.get("MK_SK_ROOT_ROW", "0"))
        self.roots = ttnn.CoreRangeSet([ttnn.CoreRange(T(0, self.root_row), T(9, self.root_row))])
        W = lambda grid, tiles: ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, (32, tiles * 32), ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.in_mem = self.out_mem = W(self.owners, 10)
        self.buf_mem = W(self.owners, 10 * self.ring)
        H = lambda grid, shape, dt: ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                                                      ttnn.ShardSpec(grid, shape, ttnn.ShardOrientation.ROW_MAJOR))
        E = lambda shape, mem, dt: ttnn.empty(shape, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=mem)
        self.rs_bfp8 = os.environ.get("MK_AR_RS_BFP8", "0") == "1"
        self.ag_bfp8 = os.environ.get("MK_AR_AG_BFP8", "0") == "1"
        if self.rs_bfp8:
            self.in_dtype = ttnn.bfloat8_b
        else:
            self.in_dtype = ttnn.bfloat16
        xdt = ttnn.bfloat8_b if self.ag_bfp8 else ttnn.bfloat16
        if self.ag_bfp8:
            self.xq = ttnn.empty([1, 1, 32, 5120], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                 memory_config=self.out_mem)
        # CB_X (split-K): 2 chunks (slots j % 2) = 20 tiles = 40 KB (bf16) / 21.8 KB (BFP8) per STREAM core
        self.xbuf = E([1, 1, 32 * 80, 640], H(self.stream, (32, 640), None), xdt)
        # CB_RSTD on the 10 roots: [16][32] fp32 partials at 0, tag of owner j at 2048 + 16 j
        self.prow = os.environ.get("MK_SK_PDST", "root") == "row"
        self.rbuf = (E([1, 1, 32 * 80, 32], H(self.stream, (32, 32), None), ttnn.float32) if self.prow
                     else E([1, 1, 32 * 10, 32], H(self.roots, (32, 32), None), ttnn.float32))
        gs = lambda crs: ttnn.create_global_semaphore(mesh_device, crs, 0)
        self.xflags = [gs(self.stream.merge(self.owners)) for _ in range(2)]  # slot 0 / slot 1
        self.flush = int(os.environ.get("MK_SK_FLUSH", "1"))
        self.part = int(os.environ.get("MK_SK_PART", "1"))
        if not self.part:
            self.comp_src = f"{KROOT}/compute/reduction.cpp"
        w = mesh_device.worker_core_from_logical_core
        phys_x = [w(T(i, 0)).x for i in range(10)]
        self.split = next((i for i in range(1, 10) if phys_x[i] != phys_x[i - 1] + 1), 10)

        def prect(lx0, lx1, y):
            if lx1 < lx0:
                return [0, 0, 0, 0, 0]
            a, b = w(T(lx0, y)), w(T(lx1, y))
            return [a.x, a.y, b.x, b.y, lx1 - lx0 + 1]

        self.row_rects = lambda y: prect(0, self.split - 1, y) + prect(self.split, 9, y)

    def _comp_ct(self, base_ct, chunk_tiles):
        if not self.part:
            return base_ct
        return base_ct + [chunk_tiles, self.nstat, 1, int(self.ag_bfp8), CB_OUTQ, int(self.rs_bfp8)]  # PART_ONLY

    def _comp_rt(self, c, owner, chunk_tiles):
        if not self.part:
            return [1, self.ring, chunk_tiles] if owner else [ROLE_NONE, 0, 0]
        return ([1, self.ring, chunk_tiles] if owner else [0, 0, 0]) + [0]

    def _comp_config(self):
        if not self.part:
            return ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False)
        fp32 = os.environ.get("MK_SK_FP32", "1") == "1"
        return ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                            fp32_dest_acc_en=fp32, dst_full_sync_en=fp32)

    def _io_extra(self):
        return [self.xbuf, self.rbuf]

    def _extend(self, program, coord, out, chunk_tiles):
        tb, tf = 32 * 32 * 2, 32 * 32 * 4

        def plain(idx, dt, pages, page):
            return ttnn.CBDescriptor(total_size=pages * page, core_ranges=self.owners,
                                     format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)])

        program.cbs = list(program.cbs) + [
            plain(23, ttnn.float32, 1, 256), plain(24, ttnn.bfloat16, chunk_tiles, tb), plain(25, ttnn.float32, 1, tf),
            plain(30, ttnn.bfloat16, 1, tb),
        ]
        G = ttnn.get_global_semaphore_address
        rt = ttnn.RuntimeArgs()
        rrect = self.row_rects(self.root_row)
        for j, c in enumerate(self.own_cores):
            xb = chunk_tiles * (1088 if self.ag_bfp8 else tb)
            rt[c.x][c.y] = ([j, self.xbuf.buffer_address() + (j % 2) * xb] + self.row_rects(j // 2)
                            + [G(self.xflags[j % 2]), self.rbuf.buffer_address()]
                            + (self.row_rects(j // 2) if self.prow else rrect) + [j % 2 if self.prow else j])
        tail = ttnn.KernelDescriptor(
            kernel_source=os.path.join(MKDIR, "mk_sk_xtail.cpp"), core_ranges=self.owners,
            compile_time_args=[CB_OUTQ if self.ag_bfp8 else CB_OUT, chunk_tiles,
                               chunk_tiles * (1088 if self.ag_bfp8 else tb), 0, NORM_SCALER_BF16, self.flush, self.part,
                               int(os.environ.get("MK_SK_PSEND", "1"))],
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1),
        )
        program.kernels = list(program.kernels) + [tail]
        return program


class OneShotARP(OneShotAR):
    def _comp_rt(self, c, owner, chunk_tiles):
        if self.stats and not owner:
            return [0, 0, chunk_tiles]
        return super()._comp_rt(c, owner, chunk_tiles)

    def _comp_config(self):
        if self.stats or self.sred:
            return ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                fp32_dest_acc_en=True)
        return super()._comp_config()

    def _extend(self, program, coord, out, chunk_tiles):
        tb, tf = 2048, 4096
        owners = out.memory_config().shard_spec.grid
        if self.sred:
            program.cbs = list(program.cbs) + [
                ttnn.CBDescriptor(total_size=chunk_tiles * tf, core_ranges=owners,
                                  format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32,
                                                                              page_size=tf)]) for i in (26, 27)]
        if not self.stats:
            return program

        def plain(idx, dt, pages, page):
            return ttnn.CBDescriptor(total_size=pages * page, core_ranges=owners,
                                     format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)])

        program.cbs = list(program.cbs) + [plain(self.cb_stage, ttnn.float32, 1, 256),
                                           plain(self.cb_sq, ttnn.bfloat16, chunk_tiles, tb),
                                           plain(self.cb_part, ttnn.float32, 1, tf), plain(self.cb_scaler, ttnn.bfloat16, 1, tb)]
        return program

    """Step 3 (persistent fabric connections): `iters` back-to-back all-reduces in ONE program on the v0.6 in-row owner
    layout (owners (10..11, 0..7), senders (10..11, 8..9)), fabric connections opened once. Iteration it + 1's
    reduce-scatter waits until every output core of the device holds iteration it's chunk (as the next layer's input
    would). Per-AR cost = (program time - launch) / iters, against the same program with iters = 1."""

    def __init__(self, mesh_device, iters, cluster_axis=1, num_links=2, delta_ns=0, rows=0):
        super().__init__(mesh_device, cluster_axis, num_links, sender_cores=OneShotARSK.SK_SENDERS)
        self.ps_iters = iters
        assert 0 <= rows <= 16
        self.rows = rows
        self.pt = -(-10 * 2 * rows * 32 // 2048) if rows else 10  # pseudo-tiles per packed chunk
        self.sred = os.environ.get("MK_PS_SRED", "0") == "1"
        self.inpk = bool(rows) and os.environ.get("MK_PS_INPK", "0") == "1"
        if os.environ.get("MK_PS_PROGW", "0") == "1":
            Tq = ttnn.CoreCoord
            allc_p = ttnn.CoreRangeSet([ttnn.CoreRange(Tq(10, 0), Tq(11, 7))] + [ttnn.CoreRange(Tq(*c), Tq(*c)) for c in OneShotARSK.SK_SENDERS])
            self.progw = ttnn.empty([1, 1, 32 * 20, 32], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                    memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                                                                    ttnn.ShardSpec(allc_p, (32, 32), ttnn.ShardOrientation.ROW_MAJOR)))
            print(f"[hunt] progress words at L1 0x{self.progw.buffer_address():x} (+16 x slot: 0 writer 1 reader 2 receiver 3 producer)", flush=True)
        if os.environ.get("MK_PS_GSEM", "0") == "1":
            Tq = ttnn.CoreCoord
            allc = ttnn.CoreRangeSet([ttnn.CoreRange(Tq(10, 0), Tq(11, 7))] + [ttnn.CoreRange(Tq(*c), Tq(*c)) for c in OneShotARSK.SK_SENDERS])
            self.gsem = [ttnn.create_global_semaphore(mesh_device, allc, 0) for _ in range(4)]
            if os.environ.get("MK_PS_READYP", "0") == "1":
                sndc = ttnn.CoreRangeSet([ttnn.CoreRange(Tq(*c), Tq(*c)) for c in OneShotARSK.SK_SENDERS])
                self.readyp = [ttnn.create_global_semaphore(mesh_device, sndc, 0) for _ in range(8)]
            print("[hunt] gsem ag_ready/done/go/ready at " + " ".join(f"0x{ttnn.get_global_semaphore_address(g):x}" for g in self.gsem), flush=True)
        self.agd = os.environ.get("MK_PS_AGD", "0") == "1"
        T = ttnn.CoreCoord
        own = ttnn.CoreRangeSet([ttnn.CoreRange(T(10, 0), T(11, 7))])
        if self.sred:
            import torch

            self.rflags = ttnn.from_torch(torch.zeros(1, 1, 32, 16 * 32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                          device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                                          memory_config=ttnn.MemoryConfig(
                                              ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
                                              ttnn.ShardSpec(own, (32, 32), ttnn.ShardOrientation.ROW_MAJOR)))
        if self.agd:
            snd = ttnn.CoreRangeSet([ttnn.CoreRange(T(*c), T(*c)) for c in OneShotARSK.SK_SENDERS])
            self.agstage = ttnn.empty([1, 1, 32, 4 * self.pt * 32], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                      device=mesh_device, memory_config=ttnn.MemoryConfig(
                                          ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
                                          ttnn.ShardSpec(snd, (32, self.pt * 32), ttnn.ShardOrientation.ROW_MAJOR)))
        self.deliver_mode = int(os.environ.get("MK_PS_DELIVER", "0"))
        # MK_PS_EXTX=1 (ArPhase with engine x targets): no test x / flag / CB_RSTD buffers (they would cost ~64 KB of
        # lockstep L1 on every core); ArPhase supplies the delivery args
        if self.deliver_mode and os.environ.get("MK_PS_EXTX", "0") == "1":
            self.stats = os.environ.get("MK_PS_STATS", "0") == "1"
        elif self.deliver_mode:
            T = ttnn.CoreCoord
            self.stream = ttnn.CoreRangeSet([ttnn.CoreRange(T(0, 0), T(9, 7))])
            owners = ttnn.CoreRangeSet([ttnn.CoreRange(T(10, 0), T(11, 7))])
            self.xbuf = ttnn.empty([1, 1, 32 * 80, 640], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                   memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                                                                   ttnn.ShardSpec(self.stream, (32, 640),
                                                                                  ttnn.ShardOrientation.ROW_MAJOR)))
            self.xflags = [ttnn.create_global_semaphore(mesh_device, self.stream.merge(owners), 0) for _ in range(2)]
            w = mesh_device.worker_core_from_logical_core
            phys_x = [w(T(i, 0)).x for i in range(10)]
            split = next((i for i in range(1, 10) if phys_x[i] != phys_x[i - 1] + 1), 10)

            def prect(lx0, lx1, y):
                if lx1 < lx0:
                    return [0, 0, 0, 0, 0]
                a, b = w(T(lx0, y)), w(T(lx1, y))
                return [a.x, a.y, b.x, b.y, lx1 - lx0 + 1]

            G = ttnn.get_global_semaphore_address
            self.stats = os.environ.get("MK_PS_STATS", "0") == "1"
            # CB_RSTD on the STREAM cores: slot j % 2 (32 fp32 tile-row partials) at 128 x slot, tag at 2048 + 16 x slot
            self.rbuf = ttnn.empty([1, 1, 32 * 80, 32], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                   memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                                                                   ttnn.ShardSpec(self.stream, (32, 32),
                                                                                  ttnn.ShardOrientation.ROW_MAJOR)))
            # v12 layout: R = 32 -> tile-layout CB_X slot (xbuf + slot x 20 KB); R < 32 -> packed stream in the separate
            # xpk buffer at slot x XPK_SLOT_B (10 KB = 10 tiles x 2 faces x 16 rows x 32 B), pk_off(t, f) = (2t + f) R 32
            XPK_SLOT_B = 10 * 2 * 16 * 32
            self.xpk = ttnn.empty([1, 1, 32 * 80, 320], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                  memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                                                                  ttnn.ShardSpec(self.stream, (32, 320),
                                                                                 ttnn.ShardOrientation.ROW_MAJOR)))
            xdst = (lambda c: self.xpk.buffer_address() + (c % 2) * XPK_SLOT_B) if (self.rows and self.deliver_mode == 2) \
                else (lambda c: self.xbuf.buffer_address() + (c % 2) * 10 * 2048)
            self.deliver = lambda c: ([xdst(c)] + prect(0, split - 1, c // 2)
                                      + prect(split, 9, c // 2) + [G(self.xflags[c % 2])]
                                      + [self.rbuf.buffer_address() + 128 * (c % 2),
                                         self.rbuf.buffer_address() + 2048 + 16 * (c % 2)])
        self.ps_delta = int(delta_ns * 1.35)  # cycles at 1.35 GHz
        self.recv_src = os.path.join(MKDIR, "mk_ps_receiver_orig.cpp" if os.environ.get("MK_PS_RECV") == "orig"
                                     else "mk_ps_receiver.cpp")
        self.comp_src = os.path.join(MKDIR, "mk_ps_reduction_h.cpp" if os.environ.get("MK_PS_RED") == "hdr"
                                     else "mk_ps_reduction.cpp")
        hs = os.environ.get("MK_PS_SEND") == "hdr"  # engine-callable sender phases (mk_ar_send.hpp / mk_ar_sendw.hpp)
        self.rd_src = os.path.join(MKDIR, "mk_ps_reader_h.cpp" if hs else "mk_ps_reader.cpp")
        self.wr_src = os.path.join(MKDIR, "mk_ps_writer_h.cpp" if hs else "mk_ps_writer.cpp")
