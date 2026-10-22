# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""x_gather3 - the MoE head's x all-gather over TP, ROW-CHUNK-MAJOR, straight into se_engine3's x-own L1.

Every chip p of a mesh row holds x [rows, kl] (TILE, DRAM). Chunk by chunk (CR row tiles), store-and-forward over
fabric unicasts (tp_gather's ports, line / ring schedule and call fence): a port per (direction, link) sends the
chunk's rows of its own shard (read from DRAM x), then each of the chunk's shards the upstream port landed here (read
from the owners' L1). A packet: up to RP tiles of one row tile within one K step, written at the step's owner core
(the engine's x families' owner tables) and slot. Row tile R (chunk-local lr) goes over link (lr / 2) % links. After each chunk a port
increments the downstream chip's watcher ((direction, link) counter); the watcher raises XREADY on every engine core
once all of a chunk's streams arrived (XREADY = m + 1: chunk m is in; the engine loads its own shard itself). x-own is
never reused within a call, so no credits: the call fence only."""

from ttnn.bringup.combine_rf.fabric_all_gather_py import _OPP, NOC0, NOC1, plan

import ttnn

INC_EVERY = 8
CB = 0

_CT = r"""
constexpr uint32_t KB = get_compile_time_arg_val(0), MT = get_compile_time_arg_val(1), SPS = get_compile_time_arg_val(2);
constexpr uint32_t RP = get_compile_time_arg_val(3), CR = get_compile_time_arg_val(4), NCH = get_compile_time_arg_val(5);
constexpr uint32_t NV = get_compile_time_arg_val(6), KLT = get_compile_time_arg_val(7), GROUP = get_compile_time_arg_val(8);
constexpr uint32_t INC = get_compile_time_arg_val(9), GX = get_compile_time_arg_val(10), GY = get_compile_time_arg_val(11);
constexpr uint32_t S0 = get_compile_time_arg_val(12), S1 = get_compile_time_arg_val(13);
"""

_WALK = r"""
constexpr uint32_t PPS = (KB + RP - 1) / RP, RPR = SPS * PPS;  // (packets a K step, a row tile)
// chunk m of shard q, link v: row tile R of the chunk (link (R / 2) % NV; part 1 / 2: even / odd R), step j, first
// tile kk0 within the step, n tiles; idx: the run's index in the link's full (part 0) walk of the chunk's shard
template <typename F>
FORCE_INLINE void walk(uint32_t m, uint32_t q, uint32_t part, uint32_t v, F&& f) {
    uint32_t cnt = 0;
    for (uint32_t R = m * CR; R < (m + 1) * CR; ++R) {
        const uint32_t lr = R - m * CR;  // (link and ring half by the chunk-local row: the host counts packets so)
        if ((lr / 2) % NV != v) continue;
        if (part == 0 || (lr & 1) == part - 1)
            for (uint32_t st = 0; st < SPS; ++st)
                for (uint32_t kk0 = 0, i = 0; kk0 < KB; kk0 += RP, ++i)
                    f(R, q * SPS + st, st * KB + kk0, kk0, KB - kk0 < RP ? KB - kk0 : RP, cnt * RPR + st * PPS + i);
        ++cnt;
    }
}
// x step j of chunk m, row tile R: its owner core (NoC) and the x-own address there (se_engine3's layout)
// args at T: fam0[S0] cols, fam1[S1] cols, per column: nown, family-1 base slot; column NoC x [GX]; row NoC y [GY]
struct Dst { uint32_t x, y, addr; };
FORCE_INLINE Dst dst_of(uint32_t T, uint32_t xown, uint32_t m, uint32_t R, uint32_t j, uint32_t kk0) {
    const uint32_t f = j & 1, h = j / 2;
    const uint32_t col = f ? get_arg_val<uint32_t>(T + S0 + h % S1) : get_arg_val<uint32_t>(T + h % S0);
    const uint32_t t = T + S0 + S1;
    const uint32_t nown = get_arg_val<uint32_t>(t + 2 * col), b1 = get_arg_val<uint32_t>(t + 2 * col + 1);
    const uint32_t slot = f ? b1 + h / S1 : h / S0;
    const uint32_t lr = R - m * CR;
    const uint32_t addr = xown + (((m * nown + slot) * MT + lr % MT) * KB + kk0) * 2048;
    return {get_arg_val<uint32_t>(t + 2 * GX + col), get_arg_val<uint32_t>(t + 3 * GX + lr / MT), addr};
}
"""

_READER = (
    '#include <cstdint>\n#include "api/dataflow/dataflow_api.h"\n'
    + _CT
    + "constexpr auto x_args = TensorAccessorArgs<14>();\n"
    + _WALK
    + r"""
void kernel_main() {
    size_t a = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(a++), xown = get_arg_val<uint32_t>(a++), v = get_arg_val<uint32_t>(a++);
    const uint32_t arrival = get_arg_val<uint32_t>(a++), n = get_arg_val<uint32_t>(a++), up_ipc = get_arg_val<uint32_t>(a++);
    const uint32_t ent = a, T = ent + 4 * n;  // entries (q, part, up_base within a chunk, packets a chunk); tables
    const auto x = TensorAccessor(x_args, x_addr, 2048);
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival);
    uint32_t batch = 0, wptr = 0;
    for (uint32_t m = 0; m < NCH; ++m)
        for (uint32_t k = 0; k < n; ++k) {
            const uint32_t q = get_arg_val<uint32_t>(ent + 4 * k), part = get_arg_val<uint32_t>(ent + 4 * k + 1);
            const uint32_t up_base = get_arg_val<uint32_t>(ent + 4 * k + 2), npk = get_arg_val<uint32_t>(ent + 4 * k + 3);
            uint32_t done = 0;
            walk(m, q, part, v, [&](uint32_t R, uint32_t j, uint32_t kj, uint32_t kk0, uint32_t np, uint32_t idx) {
                if (batch == 0) {
                    cb_reserve_back(0, RP * GROUP);
                    wptr = get_write_ptr(0);
                }
                const uint32_t dst = wptr + batch * RP * 2048;
                if (k == 0) {  // my own shard, from DRAM x [rows, KLT]
                    for (uint32_t t = 0; t < np; ++t) noc_async_read(x.get_noc_addr(R * KLT + kj + t), dst + t * 2048, 2048);
                } else {  // relay: upstream's packet idx of (chunk m, its entry k - 1) landed here
                    noc_semaphore_wait_min(arrived, m * up_ipc + up_base + idx / INC + 1);
                    const Dst d = dst_of(T, xown, m, R, j, kk0);
                    noc_async_read(get_noc_addr(d.x, d.y, d.addr), dst, np * 2048);
                }
                ++batch;
                ++done;
                if (batch == GROUP || done == npk) {
                    noc_async_read_barrier();
                    cb_push_back(0, RP * GROUP);
                    batch = 0;
                }
            });
        }
}
"""
)

_SENDER = (
    r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;
"""
    + _CT
    + _WALK
    + r"""
void kernel_main() {
    size_t a = 0;
    const uint32_t xown = get_arg_val<uint32_t>(a++), v = get_arg_val<uint32_t>(a++);
    const uint32_t arrival = get_arg_val<uint32_t>(a++), expect_in = get_arg_val<uint32_t>(a++);  // (my counter)
    const uint32_t ready = get_arg_val<uint32_t>(a++), wait_ready = get_arg_val<uint32_t>(a++);
    const uint32_t send_ready = get_arg_val<uint32_t>(a++), rx = get_arg_val<uint32_t>(a++), ry = get_arg_val<uint32_t>(a++);
    const uint32_t px = get_arg_val<uint32_t>(a++), py = get_arg_val<uint32_t>(a++);  // downstream port (NoC)
    const uint16_t mesh = get_arg_val<uint32_t>(a++), chip = get_arg_val<uint32_t>(a++);  // the neighbour
    const uint32_t wx = get_arg_val<uint32_t>(a++), wy = get_arg_val<uint32_t>(a++), watch = get_arg_val<uint32_t>(a++);
    const uint32_t n = get_arg_val<uint32_t>(a++);
    const uint32_t ent = a, T = ent + 3 * n;  // entries (q, part, packets a chunk); tables
    a = T + S0 + S1 + 3 * GX + GY;  // (past the tables: the fabric connection)
    if (n > 0 || send_ready) {
        auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[GROUP + 1];
        for (uint32_t h = 0; h <= GROUP; ++h) {
            hdrs[h] = PacketHeaderPool::allocate_header();
            (void)fabric_set_unicast_route(hdrs[h], chip, mesh);
        }
        conn.open();
        if (send_ready) {  // fence: I started this call - the neighbour's port facing me may write into my L1
            hdrs[GROUP]->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{get_noc_addr(rx, ry, ready), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdrs[GROUP]), sizeof(PACKET_HEADER_TYPE));
        }
        if (wait_ready) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready), 1);
            noc_semaphore_inc(get_noc_addr(ready), 0u - 1u);
            noc_async_atomic_barrier();
        }
        const uint64_t arr_noc = get_noc_addr(px, py, arrival);
        uint32_t h = 0, unflushed = 0;
        for (uint32_t m = 0; m < NCH && n > 0; ++m) {
            for (uint32_t k = 0; k < n; ++k) {
                const uint32_t q = get_arg_val<uint32_t>(ent + 3 * k), part = get_arg_val<uint32_t>(ent + 3 * k + 1);
                const uint32_t npk = get_arg_val<uint32_t>(ent + 3 * k + 2);
                uint32_t sent = 0;
                walk(m, q, part, v, [&](uint32_t R, uint32_t j, uint32_t, uint32_t kk0, uint32_t np, uint32_t) {
                    cb_wait_front(0, RP * (unflushed + 1));
                    const uint32_t src = get_read_ptr(0) + unflushed * RP * 2048;
                    const Dst d = dst_of(T, xown, m, R, j, kk0);
                    const uint64_t dst = get_noc_addr(d.x, d.y, d.addr);
                    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
                    h = h + 1 == GROUP ? 0 : h + 1;
                    if ((++sent % INC) == 0 || sent == npk) {
                        hdr->to_noc_fused_unicast_write_atomic_inc(NocUnicastAtomicIncFusedCommandHeader{dst, arr_noc, 1, true}, np * 2048);
                    } else {
                        hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, np * 2048);
                    }
                    conn.wait_for_empty_write_slot();
                    conn.send_current_slot_non_blocking(src, np * 2048, reinterpret_cast<uint32_t>(hdr));
                    if (++unflushed == GROUP || sent == npk) {
                        noc_async_writes_flushed();
                        cb_pop_front(0, RP * GROUP);
                        unflushed = 0;
                    }
                });
            }
            // chunk m all sent: the neighbour's watcher (behind the chunk's packets on this link)
            hdrs[GROUP]->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{get_noc_addr(wx, wy, watch), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdrs[GROUP]), sizeof(PACKET_HEADER_TYPE));
        }
        conn.close();
    }
    if (expect_in > 0) {  // everything from upstream landed; re-arm for the next call
        volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival);
        noc_semaphore_wait_min(arrived, expect_in);
        noc_semaphore_inc(get_noc_addr(arrival), 0u - expect_in);
        noc_async_atomic_barrier();
    }
}
"""
)

_WATCHER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
// chunk m is in once every incoming stream counted it: XREADY + 1 on every engine core. args: chunks, the XREADY id,
// streams (counter address) x ns, engine cores (NoC x, y) x nc
void kernel_main() {
    size_t a = 0;
    const uint32_t nch = get_arg_val<uint32_t>(a++), sem = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t ns = get_arg_val<uint32_t>(a++), nc = get_arg_val<uint32_t>(a++);
    const uint32_t s0 = a, c0 = a + ns;
    for (uint32_t m = 0; m < nch; ++m) {
        for (uint32_t i = 0; i < ns; ++i)
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(s0 + i)), m + 1);
        for (uint32_t c = 0; c < nc; ++c)
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(c0 + 2 * c), get_arg_val<uint32_t>(c0 + 2 * c + 1), sem), 1);
    }
    for (uint32_t i = 0; i < ns; ++i) noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(s0 + i)), 0u - nch);  // (re-arm)
    noc_async_atomic_barrier();
}
"""


def _incs(n):
    return n // INC_EVERY + (1 if n % INC_EVERY else 0)


def chunk_packets(CR, nv, v, part, sps, kb, rp):
    """Packets of one chunk's shard entry on link v (the kernels' walk: chunk-local row lr over link (lr / 2) % nv,
    part 1 / 2 its even / odd rows; a row tile's K steps in pieces of rp tiles)."""
    rows = sum(1 for lr in range(CR) if (lr // 2) % nv == v and (part == 0 or (lr & 1) == part - 1))
    return rows * sps * -(-kb // rp)


def port_plan(chips, coord, d, v, *, CR, nv, sps, kb, rp, nch):
    """One port's stream (pure: the device build and the CPU protocol model share it). Returns dict(peer, up,
    mine=[(shard, part, upstream base within a chunk, packets a chunk)], up_ipc (the upstream's increments a chunk),
    expect_in (a call), has_up)."""
    rg = chips[coord]["rings"][0]
    peer = rg["next"] if d == "fwd" else rg["prev"]
    up = rg["prev"] if d == "fwd" else rg["next"]
    mine = rg[d] if peer is not None else []
    ups = chips[up]["rings"][0][d] if up is not None else []
    up_base, acc = [], 0
    for _, part in ups:
        up_base.append(acc)
        acc += _incs(chunk_packets(CR, nv, v, part, sps, kb, rp))
    ent = [
        (q, part, up_base[k - 1] if k > 0 else 0, chunk_packets(CR, nv, v, part, sps, kb, rp))
        for k, (q, part) in enumerate(mine)
    ]
    return dict(peer=peer, up=up, mine=ent, up_ipc=acc, expect_in=nch * acc if ups else 0, has_up=bool(ups))


class XGather3:
    """The row-chunk-major x all-gather into an SeEngine3's x-own (module docstring); movers on sub_core_grid."""

    def __init__(self, mesh_device, engine, *, num_links=2, group=4, ring=None, sub_core_grid=None, placement="auto"):
        self.mesh, self.eng = mesh_device, engine
        e = engine
        G = self.G = tuple(mesh_device.shape)[1]
        assert e.KT % G == 0 and (e.KT // G) % e.KB == 0, "a shard must be whole K steps"
        self.klt = e.KT // G
        self.sps = self.klt // e.KB
        self.rp = max(1, min(e.KB, ttnn.get_tt_fabric_max_payload_size_bytes() // 2048))
        assert group + 1 <= 12, "the sender takes group + 1 packet headers (12 a RISC)"
        self.group, self.nv = group, num_links
        if ring is None:
            cfg = ttnn.get_fabric_config()
            ring = cfg in (ttnn.FabricConfig.FABRIC_2D_TORUS_X, ttnn.FabricConfig.FABRIC_2D_TORUS_XY) and G > 2
        self.ring = ring
        topo = ttnn.Topology.Ring if ring else ttnn.Topology.Linear
        self.chips, _ = plan(
            mesh_device,
            cluster_axis=1,
            topology=topo,
            num_links=num_links,
            balance=ring,
            sub_core_grid=sub_core_grid,
            placement=placement,
        )
        for ch in self.chips.values():
            for d in ("fwd", "bwd"):
                assert all(part == 0 for _, part in ch["rings"][0][d][:-1]), "a relay's upstream entry must be whole"
        cores = sorted({(c.x, c.y) for ch in self.chips.values() for c in list(ch["ports"].values()) + ch["copy"][:1]})
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in cores])
        self.sems = [ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(2 + 2 * num_links)]
        addrs = [int(ttnn.get_global_semaphore_address(s)) for s in self.sems]
        self.arrival, self.ready = addrs[0], addrs[1]
        self.watch = {
            (d, v): addrs[2 + i * num_links + v] for i, d in enumerate(("fwd", "bwd")) for v in range(num_links)
        }

    def _pk(self, v, part):
        """Packets of a chunk's shard entry on link v."""
        e = self.eng
        rows = sum(1 for R in range(e.CR) if (R // 2) % self.nv == v and (part == 0 or (R & 1) == part - 1))
        return rows * self.sps * -(-e.KB // self.rp)

    def watcher_core(self, coord):
        return self.chips[coord]["copy"][0]

    def tables(self):
        """The owner tables (dst_of's): family 0 / 1 columns, per column (nown, family-1 base slot), NoC x / y."""
        e, mesh = self.eng, self.mesh
        nf = e.NSG // 2
        per = []
        for c in range(e.GX):
            n0 = nf // e.S0 if c in e.fam0 else 0
            n1 = nf // e.S1 if c in e.fam1 else 0
            per += [n0 + n1, n0]
        vx = [mesh.worker_core_from_logical_core(ttnn.CoreCoord(e.cx0 + c, 0)).x for c in range(e.GX)]
        vy = [mesh.worker_core_from_logical_core(ttnn.CoreCoord(e.cx0, cy)).y for cy in range(e.GY)]
        return list(e.fam0) + list(e.fam1) + per + vx + vy

    def pieces(self, coord, prog, x, xready_sem):
        """(CBs, kernels) of chip `coord` (fabric connections set up on prog); x: this chip's [rows, kl] in DRAM."""
        mesh, chips, e = self.mesh, self.chips, self.eng
        virt = lambda c: mesh.worker_core_from_logical_core(c)  # noqa: E731
        node = lambda c: mesh.get_fabric_node_id(ttnn.MeshCoordinate(*c))  # noqa: E731
        dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(
            processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc
        )  # noqa: E731
        st = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
        cs = lambda cores: ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])  # noqa: E731
        ch = chips[coord]
        rg = ch["rings"][0]
        ports = {(d, l): core for (_, d, l), core in ch["ports"].items()}
        xown = e.base + e.off["xown"]
        T = self.tables()
        cbs = [
            ttnn.CBDescriptor(
                total_size=2 * self.group * self.rp * 2048,
                core_ranges=cs(list(ports.values())),
                format_descriptors=[ttnn.CBFormatDescriptor(CB, ttnn.bfloat16, 2048)],
            )
        ]
        rd_rt, sd_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        streams_in = []
        for (d, v), core in ports.items():
            pl = port_plan(chips, coord, d, v, CR=e.CR, nv=self.nv, sps=self.sps, kb=e.KB, rp=self.rp, nch=e.NCH)
            peer, mine, acc = pl["peer"], pl["mine"], pl["up_ipc"]
            if pl["has_up"]:
                streams_in.append(self.watch[(d, v)])
            ent_r = [x_ for q, part, ub, npk in mine for x_ in (q, part, ub, npk)]
            ent_s = [x_ for q, part, ub, npk in mine for x_ in (q, part, npk)]
            rd_rt[core.x][core.y] = [int(x.buffer_address()), xown, v, self.arrival, len(mine), acc] + ent_r + T
            args = [xown, v, self.arrival, pl["expect_in"], self.ready, int(bool(mine))]
            if peer is not None and rg.get(f"{d}_links"):
                pp = {(dd, ll): cc for (_, dd, ll), cc in chips[peer]["ports"].items()}
                rc, pc, wc = virt(pp[(_OPP[d], v)]), virt(pp[(d, v)]), virt(self.watcher_core(peer))
                pn = node(peer)
                args += (
                    [
                        int(bool(chips[peer]["rings"][0][_OPP[d]])),
                        rc.x,
                        rc.y,
                        pc.x,
                        pc.y,
                        int(pn.mesh_id),
                        int(pn.chip_id),
                        wc.x,
                        wc.y,
                        self.watch[(d, v)],
                        len(mine),
                    ]
                    + ent_s
                    + T
                )
                args += list(ttnn.setup_fabric_connection(node(coord), pn, rg[f"{d}_links"][v], prog, core))
            else:
                args += [0] * 10 + [0] + T
            sd_rt[core.x][core.y] = args
        ct = [
            e.KB,
            e.MT,
            self.sps,
            self.rp,
            e.CR,
            e.NCH,
            self.nv,
            self.klt,
            self.group,
            INC_EVERY,
            e.GX,
            e.GY,
            e.S0,
            e.S1,
        ]
        x_ct = list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_READER,
                source_type=st,
                core_ranges=cs(list(ports.values())),
                compile_time_args=ct + x_ct,
                runtime_args=rd_rt,
                config=dm("RISCV_1", NOC0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_SENDER,
                source_type=st,
                core_ranges=cs(list(ports.values())),
                compile_time_args=ct,
                runtime_args=sd_rt,
                config=dm("RISCV_0", NOC1),
            ),
        ]
        ecores = [virt(ttnn.CoreCoord(e.cx0 + c, cy)) for c in range(e.GX) for cy in range(e.GY)]
        wt = ttnn.RuntimeArgs()
        wc = self.watcher_core(coord)
        wt[wc.x][wc.y] = (
            [e.NCH, xready_sem, len(streams_in), len(ecores)] + streams_in + [v_ for c in ecores for v_ in (c.x, c.y)]
        )
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=_WATCHER,
                source_type=st,
                core_ranges=cs([wc]),
                compile_time_args=[],
                runtime_args=wt,
                config=dm("RISCV_1", NOC1),
            )
        )
        return cbs, kernels
