# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""x_gather - the MoE head's x all-gather over TP straight into the shared expert engine's L1 (tt/se_engine.py).

Every chip p of a mesh row holds x [rows, kl] (TILE, DRAM); the engine wants x_full [rows, G kl] as K steps of KB tiles
in its owners' L1: row tile R, k tile kt (step s = kt / KB) on core (s % GX, R / MT) at
xown + ((s / GX) MT + R % MT) KB + kt % KB tiles - each source chip's shard a block of K steps, no width concat.
Store-and-forward over fabric unicasts (tp_gather's ports, schedule and fence): a port per (direction, link) sends its
own shard (read from DRAM x), then each shard the upstream port landed here (read from the owners' L1 - never DRAM).
A packet: up to RP tiles of one row tile within one K step (contiguous in the owner's L1). Row tile R goes over link
(R / 2) % links: the ring's even / odd row halves and the link split stay orthogonal, so a relayed row arrives on the
same link it leaves on. After each shard a port sends one increment to the downstream chip's watcher ((direction,
link) counter); the watcher raises XREADY on every engine core once a shard is complete, in the engine's step order."""

from ttnn.bringup.combine_rf.fabric_all_gather_py import _OPP, NOC0, NOC1, plan

import ttnn

INC_EVERY = 8
CB = 0

_WALK = r"""
constexpr uint32_t PPS = (KB + RP - 1) / RP, RPR = SPS * PPS;  // (packets a K step, a row tile)
// shard q's runs for link v: row tile R (link (R / 2) % NV; part 1 / 2: even / odd R), step s, first tile kk0 within
// the step, n tiles; idx: the run's index in the link's full (part 0) walk of the shard
template <typename F>
FORCE_INLINE void walk(uint32_t q, uint32_t part, uint32_t v, F&& f) {
    uint32_t cnt = 0;
    for (uint32_t R = 0; R < RT; ++R) {
        if ((R / 2) % NV != v) continue;
        if (part == 0 || (R & 1) == part - 1)
            for (uint32_t st = 0; st < SPS; ++st)
                for (uint32_t kk0 = 0, i = 0; kk0 < KB; kk0 += RP, ++i)
                    f(R, q * SPS + st, st * KB + kk0, kk0, KB - kk0 < RP ? KB - kk0 : RP, cnt * RPR + st * PPS + i);
        ++cnt;
    }
}
FORCE_INLINE uint32_t own_addr(uint32_t xown, uint32_t R, uint32_t s, uint32_t kk0) {
    return xown + (((s / GX) * MT + R % MT) * KB + kk0) * 2048;
}
"""

_CT = r"""
constexpr uint32_t KB = get_compile_time_arg_val(0), GX = get_compile_time_arg_val(1), MT = get_compile_time_arg_val(2);
constexpr uint32_t SPS = get_compile_time_arg_val(3), RP = get_compile_time_arg_val(4), RT = get_compile_time_arg_val(5);
constexpr uint32_t NV = get_compile_time_arg_val(6), KLT = get_compile_time_arg_val(7), GROUP = get_compile_time_arg_val(8);
constexpr uint32_t INC = get_compile_time_arg_val(9);
"""

_READER = (
    r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
"""
    + _CT
    + r"""
constexpr auto x_args = TensorAccessorArgs<10>();
"""
    + _WALK
    + r"""
void kernel_main() {
    size_t a = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(a++), xown = get_arg_val<uint32_t>(a++), v = get_arg_val<uint32_t>(a++);
    const uint32_t arrival = get_arg_val<uint32_t>(a++), n = get_arg_val<uint32_t>(a++);
    const uint32_t ent = a, vxs = ent + 4 * n, vys = vxs + GX;  // entries (q, part, up_base, packets); NoC x / y
    const auto x = TensorAccessor(x_args, x_addr, 2048);
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival);
    uint32_t batch = 0, wptr = 0;
    for (uint32_t k = 0; k < n; ++k) {
        const uint32_t q = get_arg_val<uint32_t>(ent + 4 * k), part = get_arg_val<uint32_t>(ent + 4 * k + 1);
        const uint32_t up_base = get_arg_val<uint32_t>(ent + 4 * k + 2), npk = get_arg_val<uint32_t>(ent + 4 * k + 3);
        uint32_t done = 0;
        walk(q, part, v, [&](uint32_t R, uint32_t s, uint32_t kj, uint32_t kk0, uint32_t np, uint32_t idx) {
            if (batch == 0) {
                cb_reserve_back(0, RP * GROUP);
                wptr = get_write_ptr(0);
            }
            const uint32_t dst = wptr + batch * RP * 2048;
            if (k == 0) {  // my own shard, from DRAM x [rows, KLT]
                for (uint32_t t = 0; t < np; ++t) noc_async_read(x.get_noc_addr(R * KLT + kj + t), dst + t * 2048, 2048);
            } else {  // relay: upstream's packet up_base + idx (its increments: every INC-th packet and a shard's last)
                noc_semaphore_wait_min(arrived, up_base + idx / INC + 1);
                const uint64_t src = get_noc_addr(get_arg_val<uint32_t>(vxs + s % GX), get_arg_val<uint32_t>(vys + R / MT),
                                                  own_addr(xown, R, s, kk0));
                noc_async_read(src, dst, np * 2048);
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
    const uint32_t ent = a, vxs = ent + 3 * n, vys = vxs + GX;  // entries (q, part, packets); NoC x / y
    a = vys + (RT + MT - 1) / MT;
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
        for (uint32_t k = 0; k < n; ++k) {
            const uint32_t q = get_arg_val<uint32_t>(ent + 3 * k), part = get_arg_val<uint32_t>(ent + 3 * k + 1);
            const uint32_t npk = get_arg_val<uint32_t>(ent + 3 * k + 2);
            uint32_t sent = 0;
            walk(q, part, v, [&](uint32_t R, uint32_t s, uint32_t, uint32_t kk0, uint32_t np, uint32_t) {
                cb_wait_front(0, RP * (unflushed + 1));
                const uint32_t src = get_read_ptr(0) + unflushed * RP * 2048;
                const uint64_t dst = get_noc_addr(get_arg_val<uint32_t>(vxs + s % GX), get_arg_val<uint32_t>(vys + R / MT),
                                                  own_addr(xown, R, s, kk0));
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
            // shard k is all sent: the neighbour's watcher (behind the shard's packets on this link)
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
// shard j (in the engine's order, after my own) is in L1 once each of its streams counted it: then XREADY + 1 on every
// engine core. args: n shards, per shard: nreq, (counter address, count) x nreq; the counters' final values; the cores
void kernel_main() {
    size_t a = 0;
    const uint32_t ns = get_arg_val<uint32_t>(a++), sem = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t ncnt = get_arg_val<uint32_t>(a++), ncore = get_arg_val<uint32_t>(a++);
    const uint32_t cnt0 = a;  // (counter address, final) x ncnt
    a += 2 * ncnt;
    const uint32_t core0 = a;
    a += 2 * ncore;
    for (uint32_t j = 0; j < ns; ++j) {
        const uint32_t nreq = get_arg_val<uint32_t>(a++);
        for (uint32_t i = 0; i < nreq; ++i, a += 2)
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(a)), get_arg_val<uint32_t>(a + 1));
        for (uint32_t c = 0; c < ncore; ++c)
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(core0 + 2 * c), get_arg_val<uint32_t>(core0 + 2 * c + 1), sem), 1);
    }
    for (uint32_t i = 0; i < ncnt; ++i) {  // (re-arm)
        const uint32_t c = get_arg_val<uint32_t>(cnt0 + 2 * i), f = get_arg_val<uint32_t>(cnt0 + 2 * i + 1);
        if (f) noc_semaphore_inc(get_noc_addr(c), 0u - f);
    }
    noc_async_atomic_barrier();
}
"""


def _incs(n):
    return n // INC_EVERY + (1 if n % INC_EVERY else 0)


class XGather:
    """The x all-gather into an SeEngine's L1 (module docstring). The movers (ports, watcher) on sub_core_grid."""

    def __init__(self, mesh_device, engine, *, num_links=2, group=4, ring=None, sub_core_grid=None, placement="auto"):
        self.mesh, self.eng = mesh_device, engine
        G = self.G = tuple(mesh_device.shape)[1]
        e = engine
        assert e.KT % G == 0 and (e.KT // G) % e.KB == 0, "a shard must be whole K steps"
        self.klt = e.KT // G
        self.sps = self.klt // e.KB
        self.rt = e.rows // 32
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
        # arrival, ready (ports); watcher counters per (incoming direction, link)
        self.sems = [ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(2 + 2 * num_links)]
        addrs = [int(ttnn.get_global_semaphore_address(s)) for s in self.sems]
        self.arrival, self.ready = addrs[0], addrs[1]
        self.watch = {
            (d, v): addrs[2 + i * num_links + v] for i, d in enumerate(("fwd", "bwd")) for v in range(num_links)
        }

    def _rows(self, v, part):
        return sum(1 for R in range(self.rt) if (R // 2) % self.nv == v and (part == 0 or (R & 1) == part - 1))

    def _pk(self, v, part):
        return self._rows(v, part) * self.sps * -(-self.eng.KB // self.rp)

    def watcher_core(self, coord):
        return self.chips[coord]["copy"][0]

    def order(self, coord):
        """(shards in the engine's order (own first), per shard the (incoming direction, link, entry) that carry it)."""
        rg = self.chips[coord]["rings"][0]
        p = rg["p"]
        carry = {}
        for d in ("fwd", "bwd"):  # data flowing d arrives from the d-upstream chip
            up = rg["prev"] if d == "fwd" else rg["next"]
            if up is None:
                continue
            for j, (q, part) in enumerate(self.chips[up]["rings"][0][d]):
                for v in range(self.nv):
                    if self._pk(v, part):
                        carry.setdefault(q, []).append((d, v, j))
        rest = sorted(carry, key=lambda q: (max(j for _, _, j in carry[q]), q))
        return [p] + rest, carry

    def step_order(self, coord):
        """(the engine's gate / up step order, need[i]: XREADY count before step i)."""
        shards, _ = self.order(coord)
        order, need = [], []
        for pos, q in enumerate(shards):
            order += [q * self.sps + st for st in range(self.sps)]
            need += [pos] * self.sps
        return order, need

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
        vxs = [virt(ttnn.CoreCoord(e.cx0 + c, 0)).x for c in range(e.GX)]
        vys = [virt(ttnn.CoreCoord(e.cx0, cy)).y for cy in range(e.GY)]
        cbs = [
            ttnn.CBDescriptor(
                total_size=2 * self.group * self.rp * 2048,
                core_ranges=cs(list(ports.values())),
                format_descriptors=[ttnn.CBFormatDescriptor(CB, ttnn.bfloat16, 2048)],
            )
        ]
        rd_rt, sd_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for (d, v), core in ports.items():
            peer = rg["next"] if d == "fwd" else rg["prev"]
            up = rg["prev"] if d == "fwd" else rg["next"]
            mine = rg[d] if peer is not None else []
            ups = chips[up]["rings"][0][d] if up is not None else []
            up_base, acc = [], 0
            for _, part in ups:
                up_base.append(acc)
                acc += _incs(self._pk(v, part))
            ent_r, ent_s = [], []
            for k, (q, part) in enumerate(mine):
                npk = self._pk(v, part)
                ent_r += [q, part, up_base[k - 1] if k > 0 else 0, npk]
                ent_s += [q, part, npk]
            rd_rt[core.x][core.y] = [int(x.buffer_address()), xown, v, self.arrival, len(mine)] + ent_r + vxs + vys
            args = [xown, v, self.arrival, acc if ups else 0, self.ready, int(bool(mine))]
            if peer is not None and rg.get(f"{d}_links"):
                rc = virt(ports_of(chips, peer)[(_OPP[d], v)])
                pc = virt(ports_of(chips, peer)[(d, v)])
                wc = virt(self.watcher_core(peer))
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
                    + vxs
                    + vys
                )
                args += list(ttnn.setup_fabric_connection(node(coord), pn, rg[f"{d}_links"][v], prog, core))
            else:
                args += [0] * 10 + [0] + vxs + vys
            sd_rt[core.x][core.y] = args
        ct = [e.KB, e.GX, e.MT, self.sps, self.rp, self.rt, self.nv, self.klt, self.group, INC_EVERY]
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
        # the watcher: per shard after my own, its streams' counts; then XREADY on every engine core
        shards, carry = self.order(coord)
        finals = {}
        wargs = []
        for q in shards[1:]:
            reqs = [(self.watch[(d, v)], j + 1) for d, v, j in carry[q]]
            wargs += [len(reqs)] + [x_ for r in reqs for x_ in r]
            for d, v, j in carry[q]:
                finals[self.watch[(d, v)]] = max(finals.get(self.watch[(d, v)], 0), j + 1)
        ecores = [virt(ttnn.CoreCoord(e.cx0 + c, cy)) for c in range(e.GX) for cy in range(e.GY)]
        wt = ttnn.RuntimeArgs()
        wc = self.watcher_core(coord)
        # (every incoming stream counts all its entries, even ones carrying nothing for this link: re-arm by entries)
        for d in ("fwd", "bwd"):
            up = rg["prev"] if d == "fwd" else rg["next"]
            if up is not None:
                for v in range(self.nv):
                    finals[self.watch[(d, v)]] = len(self.chips[up]["rings"][0][d])
        wt[wc.x][wc.y] = (
            [len(shards) - 1, xready_sem, len(finals), len(ecores)]
            + [x_ for k, f in finals.items() for x_ in (k, f)]
            + [x_ for c in ecores for x_ in (c.x, c.y)]
            + wargs
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


def ports_of(chips, coord):
    return {(d, l): core for (_, d, l), core in chips[coord]["ports"].items()}
