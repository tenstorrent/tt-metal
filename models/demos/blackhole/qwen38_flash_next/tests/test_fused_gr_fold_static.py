# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The fused GR read's line transport without a device: the registry entry and the model hook, the kernel's argument
contracts against the Python side, the page map of the in-program all-gather against ``ttnn.all_gather(dim=3)``'s
tile order, and the three facts the semaphore protocol's proof stands on (the fold design note, a development document that is not shipped): exactly N-1 arrivals
per counter per call, the counter reset program-ordered before the release that gates every send, the data wait for
every peer's every tile; plus the fixed phase semaphores and the chain's rounding knobs."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gr_fold, gr_read
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp

PHASE = (fp.REPO_ROOT / fp.KERNEL_ROOT / "gr_fold" / "kernels" / "transport_phase.h").read_text()  # the protocol body
KERNEL = (fp.REPO_ROOT / gr_fold.TRANSPORT_KERNEL).read_text()  # one phase per program
KERNEL2 = (fp.REPO_ROOT / gr_fold.TRANSPORT2_KERNEL).read_text()  # two phases on one connection
SOURCE = PHASE + KERNEL + KERNEL2
PROBE = (fp.REPO_ROOT / gr_fold.SEM_PROBE).read_text()
STATS_NORM_SRC = (fp.REPO_ROOT / gr_fold.STATS_NORM).read_text()
WRITER2_SRC = (fp.REPO_ROOT / gr_fold.MCAST_WRITER2).read_text()
GR_KERNELS = fp.REPO_ROOT / fp.KERNEL_ROOT / "gr_read" / "kernels"
STATS_SRC = (GR_KERNELS / "stats_compute.cpp").read_text()
NORM_SRC = (GR_KERNELS / "norm_compute.cpp").read_text()
MCAST_PHASE_SRC = (GR_KERNELS / "mcast_phase.h").read_text()
READER_SRC = (GR_KERNELS / "reader.cpp").read_text()


def flat(text: str) -> str:
    return re.sub(r"\s+", "", text)


BUILDER = flat(inspect.getsource(gr_fold.transport_mesh_program))
TRANSPORT_SPEC = flat(inspect.getsource(gr_fold.Transport))
STATS = flat(inspect.getsource(gr_fold.stats_gather))
PARTIALS = flat(inspect.getsource(gr_fold.normalize_down_gather))
FRONT = flat(inspect.getsource(gr_fold.stats_normalize_down_gather))
NORMALIZE_DOWN = flat(inspect.getsource(gr_read.normalize_down))
GR_SOURCE = (Path(__file__).resolve().parents[1] / "ttnn" / "gr.py").read_text()


def test_registered_opt_in_bitwise_over_gr_read():
    entry = fused.kernel("gr_fold")
    assert entry.tolerance == fused.BITWISE and entry.default_on  # on by default since the 2026-09-25 landing
    assert entry.fused is gr_fold.gr_read_fold and entry.composed is gr_read.gr_read_fused
    assert fused.resolve("gr_fold", {}) is gr_fold.gr_read_fold
    assert (
        fused.resolve("gr_fold", {fused.OFF_ENV: "gr_fold"}) is gr_read.gr_read_fused
    )  # the merged three-program form
    assert fused.resolve("gr_fold", {fused.ENV: "gr_fold"}) is gr_fold.gr_read_fold
    hook = GR_SOURCE[GR_SOURCE.index('if fused_kernels.enabled("gr_read"):') :]
    assert 'name = "gr_fold" if fused_kernels.enabled("gr_fold") else "gr_read"' in hook
    assert "fused_kernels.gr_fold.line_semaphores(mesh_device)" in hook  # before any trace capture
    fold = flat(inspect.getsource(gr_fold.gr_read_fold))
    assert 'scaler_mode="chain",matmul="chain",read_front=read_front' in fold
    front = flat(inspect.getsource(gr_fold.read_front))
    assert (
        "gathered_stats,normalized,gathered_partials=stats_normalize_down_gather(residual,gamma_rows,down_inject)"
        in front
    )
    assert "ttnn.deallocate(gathered_stats)" in front and "returnnormalized,gathered_partials" in front
    read = inspect.getsource(gr_read.gr_read_fused)
    assert (
        "if stats_gather is None:" in read and "gathered_stats = stats_gather(residual)" in read
    )  # stages b, c stay callable
    assert (
        "normalized, gathered_partials = partial_gather(residual, gathered_stats, gamma_rows, module.weights.down_inject)"
        in read
    )
    assert "normalized, gathered_partials = read_front(residual, gamma_rows, module.weights.down_inject)" in read
    assert (
        "partial_gather / read_front fold the merged normalize_down form" in read
    )  # the split forms keep the collectives
    # the fold's read: this program + low_rank_gate; the launch goes through fp.run_program so its meta is recorded
    assert FRONT.count("fp.run_program(") == 1 and "ttnn.generic_op(" not in FRONT


def test_geometry():
    assert (gr_fold.TP_SIZE, gr_fold.TP_AXIS, gr_fold.PHASES) == (4, 1, ("stats", "partials"))
    assert (gr_fold.SEM_GO, gr_fold.SEM_SCRATCH, gr_fold.SEM_DONE) == (0, 1, 2)
    assert (gr_fold.SOURCE_TENSOR, gr_fold.SOURCE_PRODUCERS) == (0, 1)
    assert gr_fold.SCRATCH_CB < fp.CB_COUNT and gr_fold.STATS_SCRATCH_CB not in (0, 1, 2, 16)  # beside the stats CBs
    assert [(c.x, c.y) for c in gr_fold.TRANSPORT["stats"]] == [(4, 0), (5, 0)]  # one per link, off the stats cores
    assert [(c.x, c.y) for c in gr_fold.TRANSPORT["partials"]] == [
        (6, 0),
        (7, 0),
    ]  # off normalize_down's 6x2 workers + row 2
    assert (
        gr_fold.PARTIALS_SEMAPHORES == (1, 2, 3)
        and gr_fold.PARTIALS_SCRATCH_CB not in range(0, 11)
        and gr_fold.PARTIALS_SCRATCH_CB not in (16, 17)
    )
    assert not hasattr(gr_fold.LineSemaphores, "cycle")  # one pair per phase, never cycled (the design note's proof)
    # stage (d): both phases on the partials transport pair (one open sender per link per direction); the stats
    # phase's scratch semaphore beside the partials' (go, scratch, done) = 1-3, the gate beside the in0 multicast's 0
    # on the norm cores; its CBs beside normalize_down's 0-10, 16, 17 and the partials scratch
    used = {0, *gr_fold.PARTIALS_SEMAPHORES}
    assert gr_fold.FRONT_STATS_SCRATCH_SEM == 4 and gr_fold.FRONT_STATS_READY == 5
    assert (
        not {gr_fold.FRONT_STATS_SCRATCH_SEM, gr_fold.FRONT_STATS_READY} & used and max(used | {5}) < 16
    )  # NUM_SEMAPHORES
    taken = set(range(0, 11)) | {16, 17, gr_fold.PARTIALS_SCRATCH_CB}
    assert (
        len(set(gr_fold.FRONT_STATS_CBS)) == 4
        and not set(gr_fold.FRONT_STATS_CBS) & taken
        and max(gr_fold.FRONT_STATS_CBS) < fp.CB_COUNT
    )
    assert gr_fold.PROBED_CORES == gr_fold.TRANSPORT["stats"] + gr_fold.TRANSPORT["partials"]


def test_compile_time_arg_layout_matches_the_python_side():
    pattern = r"constexpr uint32_t ([A-Z_0-9]+) = get_compile_time_arg_val\((\d+)\);"
    assert dict(re.findall(pattern, KERNEL)) == {
        "SCRATCH_CB": "0",
        "TILES": "1",
        "SEM_GO": "2",
        "RING": "3",
        "SOURCE": "4",
        "SEM_SCRATCH": "5",
        "SEM_DONE": "6",
        "TILE_FIRST": "7",
        "TILE_STEP": "8",
        "PAGE_TILE_STRIDE": "9",
        "PAGE_RANK_STRIDE": "10",
    }
    assert "constexpr uint32_t ACCESSOR_BASE = 11;" in KERNEL and KERNEL.count("next_compile_time_args_offset()") == 1
    assert (
        "args=[first.scratch_cb,first.tiles,go,TP_SIZE,first.source,first.semaphore_ids[1],done,i,links,*first.page_strides,]"
        in BUILDER
    )
    two = dict(re.findall(pattern, KERNEL2))
    assert [two[k] for k in ("SEM_GO", "RING", "SEM_DONE", "TILE_FIRST", "TILE_STEP")] == ["0", "1", "2", "3", "4"]
    for phase, base in (("A", 5), ("B", 11)):
        assert [
            two[f"{phase}_{k}"]
            for k in ("SCRATCH_CB", "TILES", "SOURCE", "SEM_SCRATCH", "PAGE_TILE_STRIDE", "PAGE_RANK_STRIDE")
        ] == [str(base + j) for j in range(6)]
    assert "constexpr uint32_t ACCESSOR_BASE = 17;" in KERNEL2 and KERNEL2.count("next_compile_time_args_offset()") == 3
    assert (
        "args=[go,TP_SIZE,done,i,links]" in BUILDER
        and "args+=[t.scratch_cb,t.tiles,t.source,t.semaphore_ids[1],*t.page_strides]" in BUILDER
    )
    assert "fortintransports:args+=fp.accessor_args(t.out)+fp.accessor_args(t.local)" in BUILDER
    assert "kernel=TRANSPORT_KERNELiflen(transports)==1elseTRANSPORT2_KERNEL" in BUILDER
    assert (
        "semaphore_ids:tuple=(SEM_GO,SEM_SCRATCH,SEM_DONE)" in TRANSPORT_SPEC
        and "page_strides:tuple=(TP_SIZE,1)" in TRANSPORT_SPEC
        and "consumers:tuple=()" in TRANSPORT_SPEC
    )
    # the phase body takes every layout item as a template parameter; both kernels pass their constants in that order
    params = "SCRATCH_CB,TILES,TILE_FIRST,TILE_STEP,PAGE_TILE_STRIDE,PAGE_RANK_STRIDE,SOURCE,RING,SEM_GO,SEM_SCRATCH,SEM_DONE,true"
    assert "transport_phase<" + params + ">(line,out_args,local_args,PHASE_RT,delay_after_reset)" in flat(KERNEL)
    for phase, close in (("A", "false"), ("B", "true")):
        p = phase
        assert (
            f"transport_phase<{p}_SCRATCH_CB,{p}_TILES,TILE_FIRST,TILE_STEP,{p}_PAGE_TILE_STRIDE,{p}_PAGE_RANK_STRIDE,{p}_SOURCE,RING,SEM_GO,{p}_SEM_SCRATCH,SEM_DONE,{close}>"
            in flat(KERNEL2)
        )
    assert (
        "static_assert(A_SOURCE == 1 && B_SOURCE == 1" in KERNEL2
        and "static_assert(A_SEM_SCRATCH != B_SEM_SCRATCH" in KERNEL2
    )
    assert "constexpr uint32_t WORDS = get_compile_time_arg_val(0);" in PROBE and "TensorAccessorArgs<1>()" in PROBE


def test_runtime_arg_layout_matches_the_python_side():
    # common: 0 rank, 1 delay before the arrive, 2 delay after the reset; then the phase blocks; then the connection
    assert [int(i) for i in re.findall(r"get_arg_val<uint32_t>\((\d)\)", KERNEL + KERNEL2)] == [0, 1, 2, 0, 1, 2]
    assert (
        "constexpr uint32_t PHASE_RT_ARGS = 6;" in PHASE
        and "constexpr uint32_t PHASE_RT = 3;" in KERNEL
        and "constexpr uint32_t A_RT = 3;" in KERNEL2
    )
    assert [int(i) for i in re.findall(r"get_arg_val<uint32_t>\(rt \+ (\d)\)", PHASE)] == [0, 1, 2, 3, 4, 5]
    assert "size_t arg_idx = PHASE_RT + PHASE_RT_ARGS + 2 * get_arg_val<uint32_t>(PHASE_RT + 4);" in KERNEL
    assert "const uint32_t b_rt = A_RT + PHASE_RT_ARGS + 2 * get_arg_val<uint32_t>(A_RT + 4);" in KERNEL2
    assert "size_t arg_idx = b_rt + PHASE_RT_ARGS + 2 * get_arg_val<uint32_t>(b_rt + 4);" in KERNEL2
    assert (
        "ready.up(noc, get_arg_val<uint32_t>(rt + PHASE_RT_ARGS + 2 * c), get_arg_val<uint32_t>(rt + PHASE_RT_ARGS + 1 + 2 * c), 1);"
        in PHASE
    )
    assert "base=[rank,before,after]" in BUILDER
    assert (
        "base+=[t.out.buffer_address(),t.local.buffer_address(),barrier,data,len(t.consumers),t.consumer_sem]"
        in BUILDER
    )
    assert "base+=[vforxyint.consumersforvinxy]" in BUILDER
    assert "ifrank>0else[]" in BUILDER and "ifrank+1<TP_SIZEelse[]" in BUILDER
    # BRISC forward (ranks above) takes the forward connection, NCRISC backward (ranks below) the backward one; the
    # program's own semaphores are in the seed descriptor before the connections' (backward, then forward)
    assert "transport(fp.reader_kernel,backward)+transport(fp.writer_kernel,forward)" in BUILDER
    assert BUILDER.index("semaphores=semaphores,)") < BUILDER.index("backward=[") < BUILDER.index("forward=[")
    assert "semaphores=list(seed.semaphores)" in BUILDER  # the connections' semaphores travel with the program
    assert "range = RING - 1 - rank;" in PHASE and "range = rank;" in PHASE
    assert "line.arrive(get_arg_val<uint32_t>(PHASE_RT + 2));" in KERNEL
    assert (
        "line.arrive(get_arg_val<uint32_t>(A_RT + 2));" in KERNEL2
        and "line.arrive(get_arg_val<uint32_t>(b_rt + 2));" in KERNEL2
    )
    assert PROBE.count("get_arg_val<uint32_t>(2 + i)") == 1 and "[(core,[out.buffer_address(),slot]+addresses)" in flat(
        inspect.getsource(gr_fold.read_semaphores)
    )
    # a program's transports: the same cores, shared go/done, distinct phases and scratch semaphores, at most two
    for text in (
        "thetransportsofoneprogramrunonthesamecores",
        "thetransportsofoneprogramsharethegoanddonesemaphores",
        "onetransportperphaseperprogram,eachwithitsownscratchsemaphore",
        "aprogramcarriesoneortwotransports",
    ):
        assert text in BUILDER, text


def test_page_map_is_all_gathers_tile_order():
    """Device r's tile t lands at page t * 4 + r: the tile order of the gathered [1, B, rows, 128] tensor; with two
    links, core i carries tiles i, i + 2, ..."""

    assert "{.page_id = (TILE_FIRST + i * TILE_STEP) * PAGE_TILE_STRIDE + rank * PAGE_RANK_STRIDE}" in PHASE
    assert "out.get_noc_addr((TILE_FIRST + i * TILE_STEP) * PAGE_TILE_STRIDE + rank * PAGE_RANK_STRIDE, 0, 0)" in PHASE
    assert "MY_TILES = TILE_FIRST < TILES ? (TILES - TILE_FIRST + TILE_STEP - 1) / TILE_STEP : 0" in PHASE
    assert "const uint32_t rank = line.rank;" in PHASE
    for links in (1, 2):
        carried = sorted(t for i in range(links) for t in range(i, 4, links))
        assert carried == [0, 1, 2, 3]
    # the stats phase is tile-major (all_gather dim 3: page 4t + d), the partials phase device-major (all_gather_async
    # dim 0: page 12d + t); the two strides reproduce both
    for strides, expect in (((4, 1), lambda t, d: 4 * t + d), ((1, 12), lambda t, d: 12 * d + t)):
        for t in range(12 if strides[0] == 1 else 4):
            for d in range(4):
                assert t * strides[0] + d * strides[1] == expect(t, d)
    assert 'phase,strides="stats",(TP_SIZE,1)' in re.sub(r"\s+", "", inspect.getsource(gr_fold.gather_line))
    assert 'phase,strides="partials",(1,tiles)' in re.sub(r"\s+", "", inspect.getsource(gr_fold.gather_line))
    assert (
        'PT,SOURCE_PRODUCERS,"partials",transports,semaphore_ids=PARTIALS_SEMAPHORES,page_strides=(1,PT),)' in PARTIALS
    )
    assert (
        "(w//links,(x,y,x,y),(PT*rank+w,1),(0,0,1,1))" in PARTIALS
    )  # worker w: own page 12 * rank + w, scratch slot w // links
    assert "src_cb=17,dst_cb=PARTIALS_SCRATCH_CB,tiles=1,tiles_tensor=gathered,sem=scratch" in PARTIALS
    # stats_gather: stats core b (unit w.start = b) writes its own page 4b + rank and transport core (b % links)'s slot
    assert "(w.start//links,(x,y,x,y),(w.start*TP_SIZE+rank,1),(0,0,1,1))" in STATS
    assert "transports[w.start%links]" in STATS
    assert "src_cb=16,dst_cb=STATS_SCRATCH_CB,tiles=1,tiles_tensor=out,sem=SEM_SCRATCH" in STATS
    assert (
        'STATS_SCRATCH_CB,gr_read.BRANCHES,SOURCE_PRODUCERS,"stats",TRANSPORT["stats"])' in STATS
        and "[reader,compute,writer]" in STATS
    )
    # the front keeps both page maps: norm core b's own stats page 4b + rank and slot b // links, worker w's own
    # partial page 12 * rank + w and slot w // links
    assert "phase_a=[b//links,x,y,x,y,gathered_stats.buffer_address(),b*TP_SIZE+rank,1,0,0,0,1,1]" in FRONT
    assert "(w//links,(x,y,x,y),(PT*rank+w,1),(0,0,1,1))" in FRONT
    assert "src_cb=17,dst_cb=PARTIALS_SCRATCH_CB,tiles=1,tiles_tensor=gathered,sem=p_scratch" in FRONT
    assert (
        'Transport(gathered_stats,gathered_stats,c_sscratch,gr_read.BRANCHES,SOURCE_PRODUCERS,"stats",transports,'
        "semaphore_ids=(p_go,s_scratch,p_done),consumers=tuple(noc[(c.x,c.y)]forcinproducers),consumer_sem=FRONT_STATS_READY,)"
    ) in FRONT
    assert (
        'Transport(gathered,gathered,PARTIALS_SCRATCH_CB,PT,SOURCE_PRODUCERS,"partials",transports,semaphore_ids=PARTIALS_SEMAPHORES,page_strides=(1,PT),)'
        in FRONT
    )
    assert (
        'transports=TRANSPORT["partials"]' in FRONT and "s_scratch=FRONT_STATS_SCRATCH_SEM" in FRONT
    )  # both phases on the partials pair


def test_fact_1_exactly_n_minus_1_arrivals_per_counter_per_call():
    """Every device raises every peer's counter exactly once: the forward multicast covers ranks r+1..3, the backward
    one ranks 0..r-1, one increment per packet, one packet per direction per call."""

    assert (
        PHASE.count("fabric_multicast_noc_unicast_atomic_inc(") == 1
        and KERNEL.count("line.arrive(") == 1
        and KERNEL2.count("line.arrive(") == 2
    )
    assert PHASE.index("noc_async_writes_flushed();") < PHASE.index(
        "fabric_multicast_noc_unicast_atomic_inc("
    )  # the shared header is rewritten only after a flush
    ring = gr_fold.TP_SIZE
    for receiver in range(ring):
        arrivals = 0
        for sender in range(ring):
            forward = range(sender + 1, sender + 1 + (ring - 1 - sender))  # start 1, range RING - 1 - rank
            backward = range(sender - sender, sender)  # start 1, range rank, going down
            arrivals += int(receiver in forward) + int(receiver in backward)
        assert arrivals == ring - 1, receiver
    assert "noc_semaphore_wait_min(barrier, RING - 1);" in PHASE and PHASE.count("noc_semaphore_wait_min(barrier") == 1


def test_fact_2_reset_is_program_ordered_before_the_release_that_gates_every_send():
    brisc = PHASE[PHASE.index("#if defined(COMPILE_FOR_BRISC)\n    noc_semaphore_wait_min(barrier") :]
    assert (
        brisc.index("noc_semaphore_set(barrier, 0);")
        < brisc.index("go.set(1);")
        < brisc.index("fabric_multicast_noc_fused_unicast_with_atomic_inc(")
    )
    ncrisc = PHASE[PHASE.index("#else\n    uint32_t scratch_addr;") :]
    assert ncrisc.index("go.wait(1);") < ncrisc.index("fabric_multicast_noc_fused_unicast_with_atomic_inc(")
    assert "riscv_wait(delay_after_reset)" in brisc and brisc.index("noc_semaphore_set(barrier, 0);") < brisc.index(
        "riscv_wait(delay_after_reset)"
    ) < brisc.index("go.set(1);")


def test_fact_3_data_wait_is_every_peers_every_tile_and_every_counter_is_reset_by_its_owner():
    assert "noc_semaphore_wait_min(data, (RING - 1) * MY_TILES);" in PHASE and "noc_semaphore_set(data, 0);" in PHASE
    assert PHASE.count("fabric_multicast_noc_fused_unicast_with_atomic_inc(") == 1  # every data tile: write + inc
    body = PHASE[PHASE.index("for (uint32_t i = 0; i < MY_TILES; ++i) {\n            noc_async_writes_flushed();") :]
    assert body.index("noc_async_writes_flushed();") < body.index("fabric_multicast_noc_fused_unicast_with_atomic_inc(")
    assert "go.set(0);" in PHASE and PHASE.count("scratch_ready.wait_min(MY_TILES);") == 2
    assert (
        "done.set(1);" in PHASE
        and "done.wait(1);" in PHASE
        and "done.set(0);" in PHASE
        and "scratch_ready.set(0);" in PHASE
    )
    assert "connection.close();" in PHASE and "if constexpr (CLOSE) {\n        line.close();" in PHASE
    for kernel in (KERNEL, KERNEL2):
        assert (
            kernel.rstrip().endswith("noc_async_full_barrier();\n}")
            and kernel.count("line.open<RING>(rank, arg_idx);") == 1
        )
    assert "MulticastRoutingCommandHeader" not in PHASE  # the linear API sets the route from (start 1, range)
    # the consumer signal follows the data wait and its reset
    assert (
        PHASE.index("noc_semaphore_wait_min(data, (RING - 1) * MY_TILES);")
        < PHASE.index("noc_semaphore_set(data, 0);")
        < PHASE.index("ready.up(noc, get_arg_val<uint32_t>(rt + PHASE_RT_ARGS + 2 * c)")
    )


def test_two_phases_on_one_connection_reuse_go_and_done_in_order():
    """transport2: one connection per RISC opened once and closed after phase B's sends (CLOSE only there), both
    arrives at kernel start on distinct counters, phase B after phase A.  The go/done reuse is ordered: BRISC's
    phase-B go.set(1) follows its phase-A done.wait(1) (program order), which follows NCRISC's phase-A done.set(1),
    which follows NCRISC's phase-A go.set(0) in NCRISC program order; a phase needs SOURCE 1 for that handshake."""

    two = flat(KERNEL2)  # the phase calls sit in FUSED_ZONE blocks and wrap over lines; the order is what matters
    a = two.index("transport_phase<A_SCRATCH_CB")
    b = two.index("transport_phase<B_SCRATCH_CB")
    assert (
        two.index("line.open<RING>(rank,arg_idx);")
        < two.index("line.arrive(get_arg_val<uint32_t>(A_RT+2));")
        < two.index("line.arrive(get_arg_val<uint32_t>(b_rt+2));")
        < a
        < b
    )
    assert "SEM_DONE,false>" in two[a:b] and "SEM_DONE,true>" in two[b:]
    # within one phase body BRISC's go.set(1) precedes its done.wait(1); the body runs twice, so phase B's release
    # follows phase A's done wait in program order
    assert (
        PHASE.index("go.set(1);") < PHASE.index("done.wait(1);")
        and PHASE.count("go.set(1);") == 1
        and PHASE.count("done.wait(1);") == 1
    )
    ncrisc = PHASE[PHASE.index("#else\n    uint32_t scratch_addr;") :]
    assert (
        ncrisc.index("go.set(0);")
        < ncrisc.index("fabric_multicast_noc_fused_unicast_with_atomic_inc(")
        < ncrisc.index("done.set(1);")
    )
    assert "static_assert(A_SOURCE == 1 && B_SOURCE == 1" in KERNEL2
    # the Python side: the two specs run on the same cores with the same go/done and distinct scratch ids and phases
    assert "semaphore_ids=(p_go,s_scratch,p_done)" in FRONT and "semaphore_ids=PARTIALS_SEMAPHORES" in FRONT
    assert gr_fold.FRONT_STATS_SCRATCH_SEM != gr_fold.PARTIALS_SEMAPHORES[1]


def test_line_checks_neighbours_degree_and_links():
    source = inspect.getsource(gr_fold.Line.__init__)
    assert "ttnn.get_eth_forwarding_direction(self.nodes[r], self.nodes[r + 1])" in source
    assert "Counter(neighbours.values()) != Counter({1: 2, 2: 2})" in source
    assert "ttnn.get_forwarding_link_indices(self.nodes[a], self.nodes[b])" in source
    assert "links must be 1..{min(len(t.cores), geometry.links, t.tiles)}" in inspect.getsource(
        gr_fold.transport_mesh_program
    )
    # the link budget: one open sender per link per direction per program (fabric.cpp: sender channel 0; the open
    # handshake does not queue), hence one core per link carrying every phase of the program
    assert "ift.cores[:links]!=cores:" in BUILDER


def test_normalize_down_gather_mirrors_normalize_down():
    """The fold's normalize_down keeps normalize_down's kernels: the same reader streams and constants, the same norm
    and down compute arguments (spill = DOWN_SPILL: the chain's K-block rounding), the same in0 multicast; only the
    workers' writer changes (the tile into the gathered page and the transport scratch) and the transport joins."""

    for text in (
        "_stream(gathered_stats,1,ST,b*ST,1,0,ST)",
        "_stream(residual,1,HT,b*HT,1,0,4)",
        "_stream(norm_scale,1,HT,b*HT,1,0,4)",
        "[scaler_bits,gr_read._bits(gr_read.EPS)]",
        "gr_read.NORM,p_set,[HT,ST,4,16],fp32_dest=True,unpack_to_dest_fp32=(4,7)",
        "gr_read.DOWN,w_set,[FT,8,8,9,17,gr_read.DOWN_SPILL,10],fp32_dest=True",
        "src_cb=16,dst_cb=8,tiles=HT,tiles_tensor=normalized",
        "recv_cb=8,recv_tiles=FT,senders=gr_read.BRANCHES",
        'gr_read.avg_scaler("chain")',
    ):
        assert text in PARTIALS, text
    for text in (
        "_stream(gathered_stats,1,STATS_TILES,b*STATS_TILES,1,0,STATS_TILES)",
        "_stream(residual,1,HIDDEN_TILES,b*HIDDEN_TILES,1,0,4)",
        "NORM,p_set,[HIDDEN_TILES,STATS_TILES,4,16],fp32_dest=True,unpack_to_dest_fp32=(4,7)",
        'DOWN,w_set,[FLAT_TILES,8,8,9,17,DOWN_SPILLifmatmul=="chain"else0,10],fp32_dest=True',
        "src_cb=16,dst_cb=8,tiles=HIDDEN_TILES,tiles_tensor=normalized",
        "recv_cb=8,recv_tiles=FLAT_TILES,senders=BRANCHES",
    ):
        assert text in NORMALIZE_DOWN, text
    assert "fp.allocate((TP_SIZE,1,rows,gr_read.PARTIAL_WIDTH),gr_read.FP32,ttnn.TILE_LAYOUT,mesh)" in PARTIALS
    assert "fp.semaphore_descriptor(0,all_set)" in PARTIALS  # normalize_down's in0 multicast keeps id 0


def test_front_keeps_normalize_downs_kernels_and_streams():
    """Stage (d) keeps normalize_down's arithmetic: the same worker receiver, down compute (spill = DOWN_SPILL) and in0
    multicast (phase B of the two-phase writer = normalize_down's sender), the same constants; the norm cores'
    reader streams the residual for the stats phase, gamma, the residual again for the norm phase, then the
    gathered stats behind the gate."""

    for text in (
        "[(residual,0),(norm_scale,4),(residual,0),(gathered_stats,1)]",
        "[(gr_read.CONST_SCALER,c_sscaler),(gr_read.CONST_SCALER,2),(gr_read.CONST_COL_SCALAR,3)]",
        "[gr_read._bits(1.0),scaler_bits,gr_read._bits(gr_read.EPS)]",
        "_stream(norm_scale,1,HT,b*HT,1,0,4)",
        "_stream(gathered_stats,1,ST,b*ST,1,0,ST)",
        "gate=(3,FRONT_STATS_READY,links+1)",
        "fp.compute_kernel(STATS_NORM,p_set,[HT,ST,4,16,c_sscaler,c_x2,c_sout],fp32_dest=True,unpack_to_dest_fp32=(4,7))",
        "gr_read.DOWN,w_set,[FT,8,8,9,17,gr_read.DOWN_SPILL,10],fp32_dest=True",
        "recv_cb=8,recv_tiles=FT,senders=gr_read.BRANCHES",
        'gr_read.avg_scaler("chain")',
        "fp.semaphore_descriptor(0,pw_set)",
        "phase_b=[b*HT,*rect,normalized.buffer_address(),b*HT,1,0,0,0,1,1]",
        "[c_sout,c_sscratch,1,1,gr_read.NONE_CB,s_scratch,16,8,HT,1,gr_read.NONE_CB,0,FRONT_STATS_READY]+accessors",
        "phase_a+phase_b+list(noc[(core.x,core.y)])",
        "fp.accessor_args(gathered_stats)+fp.accessor_args(gathered_stats)+fp.accessor_args(normalized)+fp.accessor_args(normalized)",
        "c_sscaler,c_x2,c_sout,c_sscratch=FRONT_STATS_CBS",
        "fp.cb_descriptor(c_sscaler,gr_read.FP32,T_FP32,1,p_set)",
        "fp.cb_descriptor(c_x2,gr_read.FP32,T_FP32,HT,p_set)",
        "fp.cb_descriptor(c_sout,gr_read.BF16,T_BF16,1,p_set)",
        "fp.cb_descriptor(c_sscratch,gr_read.BF16,T_BF16,gr_read.BRANCHES,ps_set)",
        "[reader,compute,writer2,receiver,down,writer]",
    ):
        assert text in FRONT, text
    assert FRONT.count("_stream(residual,1,HT,b*HT,1,0,4)") == 2
    # stats' CBs 1 (fp32 scaler, 1 tile) and 2 (fp32 x^2, Wt tiles) and normalize's 2 (scaler, mode dtype): the
    # front's stats scaler is fp32 like stats', the norm scaler stays the chain-mode CB 2
    assert "fp.cb_descriptor(1,gr_read.FP32,gr_read.TILE_FP32,1,stats_cores)" in STATS
    assert "fp.cb_descriptor(2,scaler_dtype,fp.TILE_BYTES[scaler_dtype],1,p_set)" in FRONT


def test_front_compute_kernel_is_both_bodies_verbatim():
    """stats_norm_compute.cpp = stats_compute.cpp's body (its scaler and output CBs renamed to the compile-time ones)
    then norm_compute.cpp's body, unchanged text; between them the unpack/pack formats norm's hardware startup sets."""

    def body(source: str, start: str, end: str) -> str:
        return flat(source[source.index(start) : source.index(end) + len(end)])

    stats_body = (
        body(STATS_SRC, "DataflowBuffer res(c_res);", "scaler.pop_front(1);")
        .replace("c_scaler", "c_sscaler")
        .replace("c_out", "c_sout")
    )
    norm_body = body(NORM_SRC, "DataflowBuffer res(c_res);", "eps.pop_front(1);")
    fused = flat(STATS_NORM_SRC)
    assert stats_body in fused and norm_body in fused and fused.index(stats_body) < fused.index(norm_body)
    assert (
        "compute_kernel_hw_startup(c_res, c_scaler, c_x2);" in STATS_SRC
        and "compute_kernel_hw_startup(c_res, c_res, c_var);" in NORM_SRC
    )
    assert (
        STATS_NORM_SRC.count("compute_kernel_hw_startup(") == 1
        and "compute_kernel_hw_startup(c_res, c_sscaler, c_x2);" in STATS_NORM_SRC
    )
    between = STATS_NORM_SRC[
        STATS_NORM_SRC.index("scaler.pop_front(1);") : STATS_NORM_SRC.index(
            "DataflowBuffer res(c_res);", STATS_NORM_SRC.index("scaler.pop_front(1);")
        )
    ]
    assert "reconfig_data_format(c_res, c_res);" in between and "pack_reconfig_data_format(c_var);" in between
    for name, index in (
        ("res", 0),
        ("stats", 1),
        ("scaler", 2),
        ("eps", 3),
        ("gamma", 4),
        ("var", 5),
        ("recip", 6),
        ("unit", 7),
    ):
        assert f"c_{name} = {index};" in STATS_NORM_SRC
    for name, index in (("out", 3), ("sscaler", 4), ("x2", 5), ("sout", 6)):
        assert f"c_{name} = get_compile_time_arg_val({index});" in STATS_NORM_SRC
    assert (
        "Wt = get_compile_time_arg_val(0)" in STATS_NORM_SRC
        and "S = get_compile_time_arg_val(1)" in STATS_NORM_SRC
        and "blk = get_compile_time_arg_val(2)" in STATS_NORM_SRC
    )


def test_front_gate_orders_every_gathered_stats_page_before_the_norm_cores_read_it():
    """The norm core's reader streams the gathered stats only when its gate semaphore counts links + 1: each stats
    transport core raises it once after its data wait (every peer's tile of that core landed) and reset, and the
    core's own writer raises it after phase A, whose own-page write completes at a write barrier before the phase
    returns (mcast_phase raises the transport's scratch semaphore BEFORE that write, so nothing else orders it)."""

    # reader: ct 12-14, wait for the count then reset, before stream GATE_STREAM
    assert (
        "GATE_STREAM = get_compile_time_arg_val(12);" in READER_SRC
        and "GATE_SEM = get_compile_time_arg_val(13);" in READER_SRC
    )
    assert (
        "GATE_COUNT = get_compile_time_arg_val(14);" in READER_SRC
        and "constexpr uint32_t ACCESSOR_BASE = 15;" in READER_SRC
    )
    assert READER_SRC.index("ready.wait(GATE_COUNT);") < READER_SRC.index("ready.set(0);")
    for i in range(4):
        assert READER_SRC.index(f"gate({i});") < READER_SRC.index(f"read_stream(args{i},")
    assert "compile_args+=list(gate)ifgateisnotNoneelse[NONE_CB,0,0]" in flat(inspect.getsource(gr_read._reader))
    # transport: the consumer signal after the data wait and its reset (test_fact_3)
    # the two-phase writer: phase A, the own gate increment, phase B; ct 12 the gate id, accessors from 13, own NoC at rt 26-27
    assert (
        "constexpr uint32_t GATE_SEM = get_compile_time_arg_val(12);" in WRITER2_SRC
        and "constexpr uint32_t ACCESSOR_BASE = 13;" in WRITER2_SRC
    )
    assert (
        WRITER2_SRC.index("(a_tiles, a_extra, 0);")
        < WRITER2_SRC.index("gate.up(noc, get_arg_val<uint32_t>(26), get_arg_val<uint32_t>(27), 1);")
        < WRITER2_SRC.index("(b_tiles, b_extra, 13);")
    )
    assert "noc.async_atomic_barrier();" in WRITER2_SRC[WRITER2_SRC.index("gate.up(") :]
    # mcast_phase: rt layout 0..12 relative to `rt`; the scratch signal precedes the own-page write, whose barrier
    # precedes the pop (so the phase returns with the page written)
    assert [int(i) for i in re.findall(r"get_arg_val<uint32_t>\(rt \+ (\d+)\)", MCAST_PHASE_SRC)] == list(range(13))
    signal = MCAST_PHASE_SRC.index("sem.up(noc, x, y, 1);")
    write = MCAST_PHASE_SRC.index("noc.async_write(src, tiles, tile_bytes")
    assert (
        signal
        < write
        < MCAST_PHASE_SRC.index("noc.async_write_barrier();", write)
        < MCAST_PHASE_SRC.index("src.pop_front(NUM_TILES);")
    )
    # the gate semaphore lives on the norm cores only; the stats transport's (go, scratch, done) on it and the norm cores
    assert "fp.semaphore_descriptor(s_scratch,ps_set),fp.semaphore_descriptor(FRONT_STATS_READY,p_set)" in FRONT
    assert "[fp.semaphore_descriptor(sem,wt_set)forseminPARTIALS_SEMAPHORES]" in FRONT
