# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The verify pass's MoE as one program per device (MOE_FUSED in the kernels): the post-attention boundary, the router,
the union gate|up and the union down run on disjoint cores and hand over through semaphores instead of op boundaries,
and down starts on union slot i while gate|up is still computing the later slots.

    boundary core B   reader: all-reduce + residual + RMSNorm (attn site) -> x; increments semaphore 0 on R and G
                      sender (BRISC): the moe-site fabric send of the down partial once every D core incremented
                      semaphore 2
    router core R     waits for x (semaphore 0), writes the union + weights, zeroes the per-slot counters (in its
                      L1), increments semaphore 1 on every G and D core
    gate|up cores G   read x after semaphore 0 and the union after semaphore 1 (their weight readers too); after slot i
                      is written, increment counter i on R
    down cores D      weight readers start at semaphore 1 (streaming while gate|up runs); the writers poll counter i
                      until it equals |G|, then hand slot i to compute; at the end they increment semaphore 2 on B

Separate ops would serialize router -> gate|up -> down, each with its own launch and DRAM pipeline ramp."""

import torch

import ttnn

from ..fused_decode.stream import EXPERT_BANK_ROT, row_placement
from ..fused_decode.union import ExpertDownUnion, ExpertGateUpUnion

MOE = [("MOE_FUSED", "1")]
EARLY_WSEM = 12  # the router releases gate|up / down on the union, the routing weights follow (this semaphore)
PRED_P = 3  # predicted experts per layer prefetched during the boundary + router (the router keeps 8 per layer)


def _wait_n(b):
    """Consumer x-ready count with the multi-core boundary's direct notify (one increment per helper)."""
    return [("BND_WAIT_N", str(b.H))]


def _noc(mesh_device, cores):
    return [mesh_device.worker_core_from_logical_core(c) for c in ttnn.corerange_to_cores(cores, row_wise=True)]


class MoEFused:
    def __init__(
        self,
        mesh_device,
        boundary,
        router,
        hidden,
        inter_pad,
        swiglu_limit,
        alpha,
        rows,
        real_inter,
        down_bal_plan,
        gate_up_readers=3,
        down_readers=3,
    ):
        """router: the verify pass's router LinearStream (out_mode 4, one core). real_inter: the unpadded intermediate
        size per device. down_bal_plan: (bank_cols, seg_inter) of the balanced down layout (experts/weights.py
        load_decode_down_bal). Placement (fused_decode/stream.py row_placement): per DRAM bank its gate_up_readers
        nearest row cores to gate|up (those whose pairs are all padding dropped), the rest of the row's cores to down,
        as many as its columns need (4 per core)."""
        # the balanced down layout was planned for these reader counts (fused_decode/stream.py down_bal_plan defaults
        # gate_up_readers = 3, cores_per_bank = 6)
        assert gate_up_readers == 3 and gate_up_readers + down_readers == 6, "down_bal_plan's reader counts"
        self.mesh = mesh_device
        self.boundary, self.router = boundary, router
        assert len(router.assignments) == 1
        rc = router.assignments[0][0]
        assert boundary.mc, "the multi-core (helpers) boundary"
        taken = [(rc.x, rc.y), (boundary.core.x, boundary.core.y)] + [(h.x, h.y) for h in boundary.helpers]
        rp = row_placement(mesh_device, gate_up_readers + down_readers, exclude=taken)
        gp = [(rp[b][r], b, r) for b in sorted(rp) for r in range(gate_up_readers)]
        self.gate_up = ExpertGateUpUnion(
            mesh_device,
            hidden,
            inter_pad,
            swiglu_limit,
            alpha,
            readers=gate_up_readers,
            rows=rows,
            real_inter=real_inter,
            rot=EXPERT_BANK_ROT,
            placed=gp,
        )
        g_used = {(c.x, c.y) for c, *_ in self.gate_up.assignments}
        dp = [(c, b, r) for b in sorted(rp) for r, c in enumerate([c for c in rp[b] if (c.x, c.y) not in g_used])]
        self.down = ExpertDownUnion(
            mesh_device,
            hidden,
            inter_pad,
            readers=down_readers,
            rows=rows,
            rot=EXPERT_BANK_ROT,
            placed=dp,
            bank_cols=down_bal_plan[0],
            seg_inter=down_bal_plan[1],
        )
        self.G, self.D = self.gate_up.cores, self.down.cores
        self.R = router.cores
        # the router's per-row top-k split over BRISC and NCRISC (kernels/stream_linear_{writer,reader}.cpp)
        self.split_defines = [
            ("MOE_SPLIT_ROWS", "1"),
            ("MOE_SPLIT_SEM", "3"),
            ("MOE_SPLIT_CB_OUT", "16"),
            ("MOE_SPLIT_CB_SCR", "3"),
            ("MOE_SPLIT_TOPK", str(router.heads[0])),
        ]
        self.router_noc = mesh_device.worker_core_from_logical_core(rc)
        # Per-slot counters: 32 x 16 B in the router core's L1 (one shard).
        shard = ttnn.ShardSpec(self.R, [32, 4], ttnn.ShardOrientation.ROW_MAJOR)
        self.counters = ttnn.from_torch(
            torch.zeros(32, 4, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        # Expert prediction (MOE_PRED): per layer, the previous round's 8 most-routed experts (32 B per layer in the
        # router core's L1); gate|up / down prefetch the first PRED_P during the boundary + router and process them
        # first if routed again.
        self.pred = ttnn.from_torch(
            torch.zeros(32, 8, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(self.R, [32, 8], ttnn.ShardOrientation.ROW_MAJOR),
            ),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def __call__(self, pending, router_ws, union, weights, gate_up_ws, act, down_ws, partial, layer=0):
        """pending: the attn-site PendingBoundary (its partial sent by o_proj); runs the boundary, router, gate|up and
        down and sends the down partial to the moe site. Sets pending.residual_out / pending.x."""
        # the router releases gate|up / down on the union, the softmax weights follow (semaphore EARLY_WSEM;
        # kernels/stream_linear_writer.cpp MOE_EARLY_UNION)
        MOE_ = MOE + [("MOE_WSEM", str(EARLY_WSEM))]
        b, mesh = self.boundary, self.mesh
        ctr = self.counters.buffer_address()
        rx, ry = self.router_noc.x, self.router_noc.y
        gd = _noc(mesh, self.G) + _noc(mesh, self.D)
        router_rt = [ctr, len(gd)] + [v for p in gd for v in (p.x, p.y)]
        P = PRED_P
        pred_addr = self.pred.buffer_address() + 32 * layer
        pdef = [("MOE_PRED", str(P)), ("MOE_PRED_SEM", "4"), ("MOE_PRED_CB", "6")]
        router_rt += [pred_addr, 2 * (self.G.num_cores() + self.D.num_cores())]
        rprog, rio = self.router(
            pending,
            router_ws,
            (union, weights),
            build_only=True,
            consumer_cores=self.R.merge(self.G),
            extra_defines=MOE_ + self.split_defines + pdef + [("MOE_EARLY_UNION", "1")],
            extra_writer_rt=router_rt,
        )
        x = pending.x
        n_g = self.G.num_cores()
        # semaphore 0: x ready (R, G, B) (+ the multi-core boundary's 6)
        assert {sd.id for sd in rprog.semaphores} <= {0, 6}
        GD = self.G.merge(self.D)
        sems = list(rprog.semaphores) + [
            ttnn.SemaphoreDescriptor(1, ttnn.CoreType.WORKER, GD, 0),
            ttnn.SemaphoreDescriptor(2, ttnn.CoreType.WORKER, self.D.merge(b.cores), 0),
            ttnn.SemaphoreDescriptor(3, ttnn.CoreType.WORKER, self.R, 0),
            ttnn.SemaphoreDescriptor(EARLY_WSEM, ttnn.CoreType.WORKER, GD, 0),
            ttnn.SemaphoreDescriptor(4, ttnn.CoreType.WORKER, self.R, 0),
        ]
        gprog, gio = self.gate_up(
            x,
            union,
            weights,
            gate_up_ws,
            act,
            extra_defines=MOE_ + pdef + _wait_n(b),
            extra_writer_rt=[ctr, rx, ry, pred_addr],
            extra_reader_rt=[rx, ry, pred_addr],
            pred_p=P,
        )
        dprog, dio = self.down(
            act,
            union,
            weights,
            down_ws,
            partial,
            extra_defines=MOE_ + pdef,
            extra_writer_rt=[ctr, rx, ry, n_g, pred_addr],
            notify_args=[1, b.noc_xy[0], b.noc_xy[1], 2],
            extra_reader_rt=[rx, ry, pred_addr],
            pred_p=P,
        )
        mesh_program = ttnn.MeshProgramDescriptor()
        for coord in b.coords:
            program = ttnn.ProgramDescriptor(
                kernels=list(rprog.kernels) + list(gprog.kernels) + list(dprog.kernels),
                semaphores=list(sems),
                cbs=list(rprog.cbs) + list(gprog.cbs) + list(dprog.cbs),
            )
            links = min(b.links, b.chunks)  # one sender per fabric link (BRISC / NCRISC)
            for link in range(links):
                sender = b._sender_kernel(
                    coord, partial, "moe", producers=self.D.num_cores(), sem_id=2, link=link, links=links
                )
                b._add_sender(coord, program, sender, link)
            mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = program
        io = []
        for t in [*rio, *gio, *dio, self.counters, self.pred]:
            if all(t is not u for u in io):
                io.append(t)
        ttnn.generic_op(io, mesh_program)


class DenseMLPFused:
    """The drafter's dense SwiGLU MLP as one program per device: the post-attention boundary (B), gate|up (G) and down
    (D) on disjoint cores. The MLP is split into `slices` fixed "experts" of I_pad / slices columns (union = all of
    them, weight 1), so down starts on slice i once every G core wrote it; with no router both weight streams start
    at launch, during the boundary. Slice counters: program semaphores 3 .. 3 + slices - 1 of B (reset per launch)."""

    CTR_SEM = 3

    def __init__(
        self,
        mesh_device,
        boundary,
        hidden,
        slice_pad,
        slices,
        rows,
        weight_dtype,
        rot,
        real_inter,
        real_hidden,
        gate_up_readers=4,
        down_readers=4,
    ):
        """rot: the weights' bank rotation; real_inter (per slice) / real_hidden: drop the gate|up / down cores whose
        columns are all padding (fused_decode/union.py). Placement (fused_decode/stream.py row_placement): per DRAM bank
        the gate_up_readers nearest row cores to gate|up, the next down_readers to down."""
        self.mesh, self.boundary, self.slices = mesh_device, boundary, slices
        assert slices <= 2, "slice counters on semaphores 3, 4: clear of the boundary's 6 (XCHG_SEM)"
        assert boundary.mc, "the multi-core (helpers) boundary"
        taken = [(boundary.core.x, boundary.core.y)] + [(h.x, h.y) for h in boundary.helpers]
        rp = row_placement(mesh_device, gate_up_readers + down_readers, exclude=taken)
        gp = [(rp[b][r], b, r) for b in sorted(rp) for r in range(gate_up_readers)]
        dp = [(rp[b][gate_up_readers + r], b, r) for b in sorted(rp) for r in range(down_readers)]
        self.gate_up = ExpertGateUpUnion(
            mesh_device,
            hidden,
            slice_pad,
            0.0,
            1.0,
            readers=gate_up_readers,
            rows=rows,
            dense_silu=True,
            max_experts=slices,
            weight_dtype=weight_dtype,
            real_inter=real_inter,
            rot=rot,
            placed=gp,
        )
        self.down = ExpertDownUnion(
            mesh_device,
            hidden,
            slice_pad,
            readers=down_readers,
            rows=rows,
            max_experts=slices,
            weight_dtype=weight_dtype,
            num_buffers=2,
            rot=rot,
            skip_pad=real_hidden,
            placed=dp,
        )
        self.G, self.D = self.gate_up.cores, self.down.cores
        self.defines = MOE + [("MOE_NO_ROUTER", "1"), ("MOE_CTR_SEM", str(self.CTR_SEM))]

    def __call__(self, pending, union, weights, gate_up_ws, act, down_ws, partial):
        b = self.boundary
        hx, hy = b.noc_xy
        bk, bcbs, bsems, bio = b.consumer_parts(pending, self.G)
        x = pending.x
        gprog, gio = self.gate_up(
            x,
            union,
            weights,
            gate_up_ws,
            act,
            extra_defines=self.defines + _wait_n(b),
            extra_writer_rt=[0, hx, hy],
        )
        dprog, dio = self.down(
            act,
            union,
            weights,
            down_ws,
            partial,
            extra_defines=self.defines,
            extra_writer_rt=[0, hx, hy, self.G.num_cores()],
            notify_args=[1, hx, hy, 2],
        )
        sems = list(bsems) + [ttnn.SemaphoreDescriptor(2, ttnn.CoreType.WORKER, self.D.merge(b.cores), 0)]
        sems += [
            ttnn.SemaphoreDescriptor(self.CTR_SEM + s, ttnn.CoreType.WORKER, b.cores, 0) for s in range(self.slices)
        ]
        mesh_program = ttnn.MeshProgramDescriptor()
        n_d = self.D.num_cores()
        for coord in b.coords:
            program = ttnn.ProgramDescriptor(
                kernels=list(bk) + list(gprog.kernels) + list(dprog.kernels),
                semaphores=list(sems),
                cbs=list(bcbs) + list(gprog.cbs) + list(dprog.cbs),
            )
            links = min(b.links, b.chunks)  # one sender per fabric link (BRISC / NCRISC)
            for link in range(links):
                b._add_sender(
                    coord,
                    program,
                    b._sender_kernel(coord, partial, "moe", producers=n_d, sem_id=2, link=link, links=links),
                    link,
                )
            mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = program
        io = []
        for t in [*bio, *gio, *dio]:
            if all(t is not u for u in io):
                io.append(t)
        ttnn.generic_op(io, mesh_program)
