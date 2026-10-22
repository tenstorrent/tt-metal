# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""routing_stream - the fused head's routing exchange + combine planner (ttnn.bringup.combine_rf.routing_a2a, folded
into the head program): only the routing entries that point at a chip's experts cross chips, the planner builds
combine_plan's per-chip tables (counts, regions, token_index, table, plan) from local data.

Where it runs in the head:
  planner cores  the router cores (M = their number), on the BRISC once every router core's joint rows are written
                 (a barrier among them): routing_a2a's planner, its L1 a region of the head's arena, its exchange
                 semaphores global ones (re-armed), each destination's packet slices written into the L1 buffer of the
                 port that serves it.
  same-row ports the gather's movers (direction X+ / X-, link d % L): a polled stream of their sender (between their
                 other packets), once the M slices are in - routing_a2a's sender: one packet a (source, destination)
                 pair, empty ones too, multi-hop fabric unicast into the destination's DRAM slot (page parity N +
                 source), the last chunk a fused increment of its arrival counter (planner core 0).
  other rows     first hop Y (the fabric's route order: Y, then X): Y+ / Y- senders on the watcher core's BRISC.
  own chip       on the Y sender if there is one, else on a same-row port: a local write + increment.
"""

from ttnn.bringup.combine_rf.routing_a2a import _PLAN, PLAN_WORDS, _al16, header_bytes, routing_a2a_sizes, slot_bytes

import ttnn

DIR_OF = {"fwd": 0, "bwd": 1}  # (the gather's TP directions on X+ / X-: checked against the chips' coordinates)


def _plan_fn(subs):
    """routing_a2a's planner kernel as a function in namespace ra: ra::plan(a0, B) - its CB base B (an arena region)
    and every semaphore by address (args), CT accessors at RA_CT0."""
    src = _PLAN
    head, body = src.split("void kernel_main() {", 1)
    head = head.replace('#include <cstdint>\n#include "api/dataflow/dataflow_api.h"\n', "")
    head = head.replace("TensorAccessorArgs<0>()", "TensorAccessorArgs<__CT0__>()")
    body = body.replace("size_t a = 0;", "size_t a = a0;")
    body = body.replace("const uint32_t B = get_write_ptr(0);", "")
    body = body.replace(
        "const uint32_t go_sem = get_semaphore(0), parts_sem = get_semaphore(1), ex_sem = get_semaphore(2);",
        "const uint32_t go_sem = get_arg_val<uint32_t>(a++), parts_sem = get_arg_val<uint32_t>(a++), "
        "ex_sem = get_arg_val<uint32_t>(a++);",
    )
    # (the port buffers: absolute addresses, not offsets from the planner's base)
    body = body.replace(
        "const uint32_t pb = B + get_arg_val<uint32_t>(dst_arg + 3 * d + 2);",
        "const uint32_t pb = get_arg_val<uint32_t>(dst_arg + 3 * d + 2);",
    )
    # (global semaphores: exact waits, then re-armed for the next call - every increment of this call has landed)
    body = body.replace(
        "noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ex_sem), M - 1);",
        "noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ex_sem), M - 1);\n"
        "    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ex_sem) = 0;",
    )
    body = body.replace(
        "noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_sem), 1);",
        "noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_sem), 1);\n"
        "        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_sem) = 0;",
    )
    body = body.replace(
        "noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(parts_sem), M - 1);",
        "noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(parts_sem), M - 1);\n"
        "        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(parts_sem) = 0;",
    )
    assert body.count("get_arg_val<uint32_t>(a++), parts_sem") == 1 and body.count("= 0;") >= 3
    # (zones: the packet wait, the counting / exchange before it)
    body = body.replace(
        "    // (5) every packet of the call: core 0 waits, then wakes the others\n    if (q == 0) {",
        '    // (5) every packet of the call: core 0 waits, then wakes the others\n    {\n    DeviceZoneScopedN("SE-RPWAIT");\n    if (q == 0) {',
    )
    body = body.replace("    ts[4] = get_timestamp_32b();", "    }\n    ts[4] = get_timestamp_32b();")
    fn = "namespace ra {\n" + head + "FORCE_INLINE void plan(uint32_t a0, uint32_t B) {" + body + "}  // namespace ra\n"
    for k, v in subs.items():
        fn = fn.replace(f"__{k}__", str(int(v)))
    assert "__" not in fn.replace("__attribute__", ""), "unfilled planner placeholder"
    return fn


# a routing_a2a sender as a polled stream (a same-row port's sender, or the Y sender kernel): once the planner cores'
# M slices are in (ready >= M), the headers' totals, then one packet a destination (max-payload chunks, multi-hop
# unicast routed to its chip, the last a fused increment of its arrival counter); my own chip's (SELF): a local write
# + increment. args at r0: nd (0: no role), nr (remote), self, my buffer, PCAP, slot tensor address, c0, c1, planner
# core 0 (x, y), calls (global: this core's call count), ready (global), mesh, then the destinations' chip ids [nd]
RA_SEND = r"""
struct RASend {
    uint32_t r0, nd, nr, self_, B, PCAP, sl, c0, c1, px, py, calls, ready, mesh, done;
    void init(uint32_t a) {
        r0 = a; nd = get_arg_val<uint32_t>(a);
        if (!nd) { done = 1; return; }
        nr = get_arg_val<uint32_t>(a + 1); self_ = get_arg_val<uint32_t>(a + 2); B = get_arg_val<uint32_t>(a + 3);
        PCAP = get_arg_val<uint32_t>(a + 4); sl = get_arg_val<uint32_t>(a + 5); c0 = get_arg_val<uint32_t>(a + 6);
        c1 = get_arg_val<uint32_t>(a + 7); px = get_arg_val<uint32_t>(a + 8); py = get_arg_val<uint32_t>(a + 9);
        calls = get_arg_val<uint32_t>(a + 10); ready = get_arg_val<uint32_t>(a + 11); mesh = get_arg_val<uint32_t>(a + 12);
        done = 0;
    }
    static uint32_t len(uint32_t a) { const uint32_t n = get_arg_val<uint32_t>(a); return n ? 13 + n : 1; }
    template <typename C, typename H>
    void poll(C& conn, H hdr, bool blocking, uint32_t RA_N, uint32_t RA_M, uint32_t RA_EPC, uint32_t RA_ME, uint32_t RA_HBP, uint32_t RA_BB, uint32_t RA_BBP, uint32_t MAXP) {
        if (done) return;
        volatile tt_l1_ptr uint32_t* rd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready);
        invalidate_l1_cache();
        if (*rd < RA_M) {
            if (!blocking) return;
            noc_semaphore_wait_min(rd, RA_M);
        }
        DeviceZoneScopedN("SE-RASEND");
        noc_semaphore_inc(get_noc_addr(ready), 0u - RA_M);
        noc_async_atomic_barrier();
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* cl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(calls);
        const uint32_t par = cl[0] & 1;  // (every port core and planner core counts its calls: they stay in step)
        cl[0] = cl[0] + 1;
        const uint64_t cnt_noc = get_noc_addr(px, py, par ? c1 : c0);
        const InterleavedAddrGen<true> slg = {.bank_base_address = sl, .page_size = PCAP};
        const uint64_t my_slot = get_noc_addr(par * RA_N + RA_ME, slg);  // (the same page on every chip: my slot there)
        for (uint32_t j = 0; j < nd; ++j) {
            const uint32_t pb = B + j * PCAP;
            const uint32_t* nq = reinterpret_cast<const uint32_t*>(pb);
            uint32_t n = 0;
            for (uint32_t q = 0; q < RA_M; ++q) n += nq[q];
            uint16_t* tot = reinterpret_cast<uint16_t*>(pb + 64);
            for (uint32_t e = 0; e < RA_EPC; ++e) tot[e] = 0;
            for (uint32_t q = 0; q < RA_M; ++q) {
                const uint16_t* pre = reinterpret_cast<const uint16_t*>(pb + 64 + RA_BB + q * RA_BBP);
                for (uint32_t e = 0; e < RA_EPC; ++e) tot[e] += pre[e + 1] - pre[e];
            }
            const uint32_t bytes = (RA_HBP + 4 * n + 63) / 64 * 64;
            if (j < nr) {
                for (uint32_t off = 0; off < bytes; off += MAXP) {
                    const uint32_t sz = bytes - off < MAXP ? bytes - off : MAXP;
                    noc_async_writes_flushed();  // (the header's previous packet has left)
                    (void)fabric_set_unicast_route(hdr, static_cast<uint16_t>(get_arg_val<uint32_t>(r0 + 13 + j)), static_cast<uint16_t>(mesh));
                    if (off + sz >= bytes) {
                        hdr->to_noc_fused_unicast_write_atomic_inc(NocUnicastAtomicIncFusedCommandHeader{my_slot + off, cnt_noc, 1, true}, sz);
                    } else {
                        hdr->to_noc_unicast_write(NocUnicastCommandHeader{my_slot + off}, sz);
                    }
                    conn.wait_for_empty_write_slot();
                    conn.send_current_slot_non_blocking(pb + off, sz, reinterpret_cast<uint32_t>(hdr));
                }
            } else {  // (SELF: my own chip's packet)
                noc_async_write(pb, my_slot, bytes);
                noc_async_write_barrier();
                noc_semaphore_inc(cnt_noc, 1);
            }
        }
        noc_async_writes_flushed();
        done = 1;
    }
};
"""

# the Y senders (watcher core BRISC): routing_a2a's sender alone, its own fabric connection toward the Y neighbour
_Y_SEND = (
    r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;
#include "tools/profiler/kernel_profiler.hpp"
"""
    + RA_SEND
    + r"""
void kernel_main() {
    RASend rs;
    rs.init(0);
    size_t a = RASend::len(0);
    const uint32_t remote = get_arg_val<uint32_t>(a++);
    if (remote) {
        auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
        conn.open();
        rs.poll(conn, hdr, true, RA_N_, RA_M_, RA_EPC_, RA_ME_, RA_HBP_, RA_BB_, RA_BBP_, RA_MAXP_);
        conn.close();
    } else {
        struct NoConn { void wait_for_empty_write_slot() {} void send_current_slot_non_blocking(uint32_t, uint32_t, uint32_t) {} } nc;
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = nullptr;
        rs.poll(nc, hdr, true, RA_N_, RA_M_, RA_EPC_, RA_ME_, RA_HBP_, RA_BB_, RA_BBP_, RA_MAXP_);
    }
    noc_async_atomic_barrier();
}
"""
)


class RoutingStream:
    """The routing exchange + planner in the head (module docstring). router: the head's RouterStream (its cores are the
    planner cores; its out tensor the own joint rows); xg: the gather (its ports); watcher: the watcher core."""

    def __init__(self, mesh_device, xg, router, *, S, K, EPC, A, L, luts, nps, rows_cap, ycores, arena):
        """ycores: the logical cores the Y senders run on (BRISC; one a needed Y direction); arena: (base, bytes free
        from offset off0 on the router cores: the planner's region; the ports' and Y cores' buffers in the arena's
        range there)."""
        self.mesh, self.xg, self.router = mesh_device, xg, router
        R, C = tuple(mesh_device.shape)
        self.R, self.C, self.N = R, C, R * C
        self.S, self.K, self.EPC, self.A, self.L = S, K, EPC, A, L
        self.luts, self.nps, self.rows_cap = luts, nps, rows_cap
        self.M = len(router.cores)
        M = self.M
        assert self.N * EPC <= 1024 and S <= 2048 and EPC <= 64 and S % 32 == 0 and 1 <= M <= min(EPC, A, 16)
        sz = routing_a2a_sizes(N=self.N, S=S, K=K, EPC=EPC, A=A, rows_cap=rows_cap, M=M)
        assert sz["fits"], sz["why"]
        self.plan_l1 = sz["plan_l1"]
        self.HBP = header_bytes(EPC, M)
        self.BB, self.BBP = _al16(2 * EPC), _al16(2 * (EPC + 1))
        self.PCAP = slot_bytes(S, K, EPC, M)
        self.MAXP = int(ttnn.get_tt_fabric_max_payload_size_bytes()) // 64 * 64
        self.ycores = ycores
        self.wrap_x = bool(xg.ring)
        self.wrap_y = False  # (the LoudBox's SP axis: a line of 2; the Galaxy's SP ring: torus Y)
        cfg = ttnn.get_fabric_config()
        if cfg in (ttnn.FabricConfig.FABRIC_2D_TORUS_XY, ttnn.FabricConfig.FABRIC_2D_TORUS_Y) and R > 2:
            self.wrap_y = True
        base, abytes, off0 = arena
        assert off0 + self.plan_l1 <= abytes, "router cores: no room for the planner's region"
        self.pbase = base + off0  # (the planner's B on the router cores)
        self.bufbase = base + off0  # (the ports' / Y cores' packet buffers: the same range, on their cores)
        import torch

        sh = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(0, 1))

        def dev(t):
            return ttnn.from_torch(
                t,
                device=mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=sh,
            )

        z = lambda *s: torch.zeros(R, C, *s, dtype=torch.int32)  # noqa: E731
        NG = self.N * EPC
        self.outs = dict(
            counts=dev(z(1, NG)),
            regions=dev(z(1, NG)),
            token_index=dev(z(1, rows_cap)),
            table=dev(z(rows_cap // 32, 64)),
            plan=dev(z(1, PLAN_WORDS)),
        )
        self.slots = dev(z(2 * self.N, self.PCAP // 4))
        pc = router.crs
        c0 = ttnn.CoreRangeSet([ttnn.CoreRange(router.cores[0], router.cores[0])])
        ports = sorted({(c.x, c.y) for ch in xg.chips.values() for c in ch["ports"].values()})
        senders = ports + [(c.x, c.y) for c in ycores]
        scs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in senders])
        gs = lambda crs: ttnn.create_global_semaphore(mesh_device, crs, 0)  # noqa: E731
        self.sems = dict(
            c0=gs(c0),
            c1=gs(c0),
            calls_p=gs(pc),
            go=gs(pc),
            parts=gs(pc),
            ex=gs(pc),
            rows=gs(pc),
            calls_s=gs(scs),
            ready=gs(scs),
        )
        self.addr = {k: int(ttnn.get_global_semaphore_address(v)) for k, v in self.sems.items()}

    # ---- the plan of one chip ----
    def _first_hop(self, s, d):
        R, C = self.R, self.C
        sr, sc, dr, dc = s // C, s % C, d // C, d % C
        if sr != dr:
            if self.wrap_y:
                return 2 if 2 * ((dr - sr) % R) < R else 3
            return 2 if dr > sr else 3
        if self.wrap_x:
            return 0 if 2 * ((dc - sc) % C) <= C else 1
        return 0 if dc > sc else 1

    def serve(self, coord):
        """{sender key: [destinations]} of chip coord: ("x", tp direction, link) the gather's ports, ("y", 2 | 3) the Y
        senders; my own chip last on the Y sender if any, else on the first port."""
        me = coord[0] * self.C + coord[1]
        ch = self.xg.chips[coord]
        rg = ch["rings"][0]
        xdir = {}  # (X+ / X- -> the gather's direction toward it)
        for d in ("fwd", "bwd"):
            peer = rg["next"] if d == "fwd" else rg["prev"]
            if peer is not None and rg.get(f"{d}_links"):
                xdir[0 if (peer[1] - coord[1]) % self.C == 1 else 1] = d
        out = {}
        for dd in range(self.N):
            if dd == me:
                continue
            h = self._first_hop(me, dd)
            if h < 2:
                assert h in xdir, f"chip {coord}: no X port toward {dd}"
                key = ("x", xdir[h], dd % self.L if (xdir[h], dd % self.L) in self._ports(coord) else 0)
            else:
                key = ("y", h)
            out.setdefault(key, []).append(dd)
        ykeys = sorted(k for k in out if k[0] == "y")
        own = ykeys[0] if ykeys else sorted(out)[0]
        out.setdefault(own, []).append(me)
        assert len(ykeys) <= 2, f"chip {coord}: {len(ykeys)} Y directions"
        return out

    def _ports(self, coord):
        return {(d, l): c for (_, d, l), c in self.xg.chips[coord]["ports"].items()}

    def _core_of(self, coord, key):
        if key[0] == "x":
            return self._ports(coord)[(key[1], key[2])]
        return self.ycores[0]  # (the watcher core: Y senders on its BRISC and, after the watch, its NCRISC)

    def planner_args(self, coord):
        """The planner's runtime args (routing_a2a's head + the planner core q appended by the caller per core)."""
        self.serve_cache = getattr(self, "serve_cache", {})
        self.serve_cache[coord] = sv = self.serve(coord)
        virt = self.mesh.worker_core_from_logical_core
        where = {}
        for key, dests in sv.items():
            core = self._core_of(coord, key)
            for j, d in enumerate(dests):
                where[d] = (core, j)
        o = self.outs
        head = [
            int(self.router_out.buffer_address()),
            int(self.slots.buffer_address()),
            self.addr["c0"],
            self.addr["c1"],
            self.addr["calls_p"],
            int(o["counts"].buffer_address()),
            int(o["regions"].buffer_address()),
            int(o["token_index"].buffer_address()),
            int(o["table"].buffer_address()),
            int(o["plan"].buffer_address()),
        ] + list(self.luts[coord])
        pv = [virt(c) for c in self.router.cores]
        pxy = [w for v in pv for w in (v.x, v.y)]
        dst = []
        for d in range(self.N):
            core, j = where[d]
            v = virt(core)
            dst += [v.x, v.y, self.bufbase + j * self.PCAP]
        senders = sorted(sv)
        port_args = [w for k in senders for v in [virt(self._core_of(coord, k))] for w in (v.x, v.y)]
        tail = [0, self.addr["ready"]]
        sems = [self.addr["go"], self.addr["parts"], self.addr["ex"]]
        return lambda q: head + [q, 0, 0] + pxy + dst + port_args + tail + sems

    def kernel_subs(self, coord, ct0):
        return dict(
            S=self.S,
            K=self.K,
            EPC=self.EPC,
            R=self.R,
            C=self.C,
            ME=coord[0] * self.C + coord[1],
            A=self.A,
            L=self.L,
            P=self.nps[coord],
            TPP=32,
            M=self.M,
            HBP=self.HBP,
            BB=self.BB,
            BBP=self.BBP,
            PCAP=self.PCAP,
            NPORT=len(self.serve_cache[coord]),
            ROWS_CAP=self.rows_cap,
            ENT_CAP=self.rows_cap,
            WX=int(self.wrap_x),
            WY=int(self.wrap_y),
            DCACHE=1,
            PROBE_NOSTORE=0,
            CT0=ct0,
        )

    def send_block(self, coord, key):
        """A sender's RASend args (key None or not serving: [0])."""
        sv = self.serve_cache.get(coord) or self.serve(coord)
        if key not in sv:
            return [0]
        dests = sv[key]
        me = coord[0] * self.C + coord[1]
        nr = len(dests) - int(dests[-1] == me)
        node = lambda i: self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(i // self.C, i % self.C))  # noqa: E731
        virt = self.mesh.worker_core_from_logical_core
        p0 = virt(self.router.cores[0])
        return [
            len(dests),
            nr,
            int(dests[-1] == me),
            self.bufbase,
            self.PCAP,
            int(self.slots.buffer_address()),
            self.addr["c0"],
            self.addr["c1"],
            p0.x,
            p0.y,
            self.addr["calls_s"],
            self.addr["ready"],
            int(node(me).mesh_id),
        ] + [int(node(d).chip_id) for d in dests]

    def defines(self, coord):
        me = coord[0] * self.C + coord[1]
        return "".join(
            f"#define {k} {v}\n"
            for k, v in dict(
                RA_N_=self.N,
                RA_M_=self.M,
                RA_EPC_=self.EPC,
                RA_ME_=me,
                RA_HBP_=self.HBP,
                RA_BB_=self.BB,
                RA_BBP_=self.BBP,
                RA_MAXP_=self.MAXP,
            ).items()
        )

    def _y_args(self, coord, key, prog, core):
        node = lambda c: self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(*c))  # noqa: E731
        sv = self.serve_cache[coord]
        nr = len(sv[key]) - int(sv[key][-1] == coord[0] * self.C + coord[1])
        args = self.send_block(coord, key) + [int(nr > 0)]
        if nr:
            dy = 1 if key[1] == 2 else -1
            peer = ((coord[0] + dy) % self.R, coord[1])
            link = ttnn.get_forwarding_link_indices(node(coord), node(peer))[0]
            args += list(ttnn.setup_fabric_connection(node(coord), node(peer), link, prog, core))
        return args

    def watcher_tail(self, coord, prog):
        """The second Y direction's sender (Galaxy's Y ring), run by the watcher's NCRISC after its watch: (defines,
        args) to append to the watcher's, or ("", [])."""
        ykeys = sorted(k for k in self.serve_cache[coord] if k[0] == "y")
        if len(ykeys) < 2:
            return "", []
        return self.defines(coord), self._y_args(coord, ykeys[1], prog, self.ycores[0])

    def y_kernels(self, coord, prog):
        """The first Y direction's sender of chip coord (a watcher-core BRISC kernel), with its fabric connection."""
        sv = self.serve_cache[coord]
        ks = []
        for key in sorted(k for k in sv if k[0] == "y")[:1]:
            core = self._core_of(coord, key)
            args = self._y_args(coord, key, prog, core)
            rt = ttnn.RuntimeArgs()
            rt[core.x][core.y] = args
            ks.append(
                ttnn.KernelDescriptor(
                    kernel_source=self.defines(coord) + _Y_SEND,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
                    compile_time_args=[],
                    runtime_args=rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                    ),
                )
            )
        return ks
