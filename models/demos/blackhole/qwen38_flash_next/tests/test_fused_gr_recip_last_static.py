# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The GR read's rsqrt-last front (``gr_recip_last``) without a device: the registry entry (COMPONENT, opt-in, never
default without a proof) and the model hook, the kernel argument contracts against the Python side, the stats and
recip bodies kept verbatim from gr_read's kernels (the statistics and the rsqrt stay the chain's bits), the worker's
per-stream dest index, the three-phase writer's order with the gate signal between phase A and B, the reader's gate on
the gathered-stats stream, and the transports and semaphores shared with gr_fold."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gr_fold, gr_read
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gr_recip_last as gl
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp

SRC = {name: (fp.REPO_ROOT / path).read_text() for name, path in gl.KERNELS.items()}
GR_KERNELS = fp.REPO_ROOT / fp.KERNEL_ROOT / "gr_read" / "kernels"
STATS_SRC = (GR_KERNELS / "stats_compute.cpp").read_text()
NORM_SRC = (GR_KERNELS / "norm_compute.cpp").read_text()
DOWN_SRC = (GR_KERNELS / "down_compute.cpp").read_text()
LOWRANK_SRC = (GR_KERNELS / "lowrank_compute.cpp").read_text()
MCAST_READER_SRC = (GR_KERNELS / "mcast_reader.cpp").read_text()
GR_SOURCE = (Path(__file__).resolve().parents[1] / "ttnn" / "gr.py").read_text()


def flat(text: str) -> str:
    return re.sub(r"\s+", "", text)


FRONT = flat(inspect.getsource(gl.stats_recip_down_gather))


def test_registered_component_default_with_its_proof():
    entry = fused.kernel("gr_recip_last")
    assert entry.tolerance == fused.COMPONENT and entry.default_on and entry.component_proof
    assert (
        "replica" in entry.component_proof
        and "oracle" in entry.component_proof
        and "43-prompt" in entry.component_proof
    )
    assert entry.fused is gl.gr_read_recip_last and entry.composed is gr_fold.gr_read_fold
    assert fused.resolve("gr_recip_last", {}) is gl.gr_read_recip_last
    assert fused.resolve("gr_recip_last", {fused.OFF_ENV: "gr_recip_last"}) is gr_fold.gr_read_fold  # the fold's front
    assert fused.resolve("gr_recip_last", {fused.ENV: "gr_recip_last"}) is gl.gr_read_recip_last
    assert (
        fused.resolve("gr_recip_last", {fused.ENV: "gr_recip_last", fused.OFF_ENV: "gr_recip_last"})
        is gr_fold.gr_read_fold
    )
    assert "gr_recip_last" in fused.default_names()  # the registry's guard admits it: COMPONENT with its proof
    read = flat(inspect.getsource(gl.gr_read_recip_last))
    assert 'scaler_mode="chain",matmul="chain",read_front=read_front' in read


def test_model_hook_takes_the_fold_place_when_enabled_and_creates_the_line_semaphores():
    hook = GR_SOURCE[GR_SOURCE.index('if fused_kernels.enabled("gr_read"):') :]
    assert 'name = "gr_fold" if fused_kernels.enabled("gr_fold") else "gr_read"' in hook
    assert hook.index('if fused_kernels.enabled("gr_recip_last"):') < hook.index('name = "gr_recip_last"')
    assert hook.index('name = "gr_recip_last"') < hook.index('if name in ("gr_fold", "gr_recip_last"):')
    assert hook.index('if name in ("gr_fold", "gr_recip_last"):') < hook.index(
        "fused_kernels.gr_fold.line_semaphores(mesh_device)"
    )


def test_norm_core_compute_layout_and_bodies():
    src = SRC["stats_u_recip_compute"]
    for name, index in (
        ("Wt", 0),
        ("S", 1),
        ("blk", 2),
        ("c_u", 3),
        ("c_sscaler", 4),
        ("c_x2", 5),
        ("c_sout", 6),
        ("c_recip_out", 7),
        ("c_nout", 8),
    ):
        assert f"{name} = get_compile_time_arg_val({index});" in src
    for name, index in (
        ("res", 0),
        ("stats", 1),
        ("scaler", 2),
        ("eps", 3),
        ("gamma", 4),
        ("var", 5),
        ("recip", 6),
        ("ukeep", 7),
    ):
        assert f"c_{name} = {index};" in src
    assert (
        "[HT,ST,U_BLOCK,CB_U,CB_SSCALER,CB_X2,CB_SOUT,CB_RECIP_OUT,CB_NOUT]" in FRONT
        and (gl.CB_U, gl.CB_NOUT, gl.CB_RECIP_OUT) == (16, 18, 19)
        and gl.U_BLOCK == 4
        and (gl.CB_SSCALER, gl.CB_X2, gl.CB_SOUT, gl.CB_SSCRATCH) == (20, 21, 22, 23)
    )
    # the stats body: stats_compute.cpp's, its scaler and output CBs renamed, WITHOUT the residual pop (u reads it)
    stats_body = flat(STATS_SRC[STATS_SRC.index("DataflowBuffer res(c_res);") : STATS_SRC.index("res.pop_front(Wt);")])
    stats_body = stats_body.replace("c_scaler", "c_sscaler").replace("c_out", "c_sout")
    fused_src = flat(src)
    assert stats_body in fused_src
    stats_zone = fused_src.index('FUSED_ZONE("fz_gl_c_stats")')
    u_zone = fused_src.index('FUSED_ZONE("fz_gl_c_u")')
    assert stats_zone < fused_src.index(stats_body) < u_zone
    assert "res.pop_front(Wt);" not in flat(src)[stats_zone:u_zone]  # kept for the u phase
    # the u phase: the FPU product of the residual and the gamma rows, packed to both the multicast and the kept CB
    u_phase = fused_src[u_zone : fused_src.index('FUSED_ZONE("fz_gl_c_recip")')]
    assert "mul_init(c_res,c_gamma);" in u_phase and "mul_tiles(c_res,c_gamma,wt+i,wt+i,i);" in u_phase
    assert "pack_tile(i,c_u);" in u_phase and "pack_tile(i,c_ukeep);" in u_phase
    assert "res.pop_front(Wt);gamma.pop_front(Wt);" in u_phase
    # the recip phase: norm_compute.cpp's reduce, + eps, rsqrt verbatim (the rsqrt tile packed to both recip CBs)
    recip_body = flat(
        NORM_SRC[NORM_SRC.index("compute_kernel_lib::reduce<PoolType::AVG") : NORM_SRC.index("pack_tile(0, c_recip);")]
    )
    recip_phase = fused_src[
        fused_src.index('FUSED_ZONE("fz_gl_c_recip")') : fused_src.index('FUSED_ZONE("fz_gl_c_nout")')
    ]
    assert (
        recip_body.replace("recip.reserve_back(1);", "recip.reserve_back(1);recip_out.reserve_back(1);") in recip_phase
    )
    assert "pack_tile(0,c_recip);pack_tile(0,c_recip_out);" in recip_phase
    assert (
        "reconfig_data_format(c_res,c_res);pack_reconfig_data_format(c_var);"
        in fused_src[: fused_src.index('FUSED_ZONE("fz_gl_c_recip")')]
    )
    # the normalized' phase: norm_compute's column-broadcast multiply on the kept u
    nout_phase = fused_src[fused_src.index('FUSED_ZONE("fz_gl_c_nout")') :]
    assert (
        "mul_bcast_cols_init(c_ukeep,c_recip);" in nout_phase
        and "mul_tiles_bcast_cols(c_ukeep,c_recip,wtr,0,wtr);" in nout_phase
    )
    assert "mul_tiles_bcast_cols(c_res, c_recip, wtr, 0, wtr);" in NORM_SRC  # the instruction it mirrors


def test_worker_compute_keeps_the_streams_apart_then_scales_and_sums_in_stream_order():
    src = SRC["down_streams_compute"]
    for name, index in (
        ("Kt", 0),
        ("blk", 1),
        ("Kb", 2),
        ("c_in0", 3),
        ("c_in1", 4),
        ("c_interm", 5),
        ("c_recip", 6),
        ("c_prod", 7),
        ("c_zero", 8),
        ("c_out", 9),
        ("B", 10),
    ):
        assert f"{name} = get_compile_time_arg_val({index});" in src
    assert "[FT,WEIGHT_BATCH,HT,CB_IN0,CB_W,CB_INTERM,CB_RECIP_W,CB_PROD,CB_ZERO,CB_OUT,B]" in FRONT
    assert (gl.CB_IN0, gl.CB_W, gl.CB_INTERM, gl.CB_RECIP_W, gl.CB_PROD, gl.CB_ZERO, gl.CB_OUT) == (
        8,
        9,
        10,
        12,
        13,
        14,
        17,
    )
    assert gr_read.FLAT_TILES == gr_read.BRANCHES * gr_read.HIDDEN_TILES  # Kt = B * Kb
    s = flat(src)
    assert "matmul_block(c_in0,c_in1,k+kk,kk,(k+kk)/Kb,0,1,1,1);" in s  # dest tile = the stream
    assert "matmul_block(c_in0, c_in1, k + kk, kk, 0, 0, 1, 1, 1);" in DOWN_SRC  # the chain form it comes from
    assert (
        "spill" not in src[src.index("void kernel_main()") :]
    )  # no spill/reload: the class does not reproduce that rounding
    assert (
        s.index('FUSED_ZONE("fz_gl_d_matmul")')
        < s.index('FUSED_ZONE("fz_gl_d_scale")')
        < s.index('FUSED_ZONE("fz_gl_d_sum")')
    )
    assert "mul_bcast_cols_init(c_interm,c_recip);" in s and "mul_tiles_bcast_cols(c_interm,c_recip,b,b,b);" in s
    assert (
        "add_init(c_prod,c_zero,true);reconfig_data_format(c_prod,c_zero);" in s
        and "add_tiles(c_prod,c_zero,b,0,0);" in s
    )
    assert "add_init(c_pg, c_zero, true);" in LOWRANK_SRC and "add_tiles(c_pg, c_zero, t * D + d, 0, 0);" in LOWRANK_SRC
    assert "compute_kernel_hw_startup<SrcOrder::Reverse>(c_in0, c_in1, c_interm);" in src


def test_worker_reader_receives_twice_and_the_python_side_matches():
    src = SRC["mcast_reader2"]
    for name, index in (
        ("RECV_CB", 0),
        ("RECV_TILES", 1),
        ("SENDERS", 2),
        ("SEM_ID", 3),
        ("RECV2_CB", 4),
        ("RECV2_TILES", 5),
        ("SENDERS2", 6),
        ("SEM2_ID", 7),
        ("NUM_STREAMS", 8),
        ("ZERO_CB", 11),
    ):
        assert f"{name} = get_compile_time_arg_val({index});" in src
    assert "constexpr uint32_t ACCESSOR_BASE = 12;" in src
    s = flat(src)
    assert (
        s.index("recv.reserve_back(RECV_TILES);")
        < s.index("recv2.reserve_back(RECV2_TILES);")
        < s.index("read_stream(args0")
    )
    assert s.index("sem.wait(SENDERS);recv.push_back(RECV_TILES);") < s.index(
        "sem2.wait(SENDERS2);recv2.push_back(RECV2_TILES);"
    )
    body = flat(MCAST_READER_SRC)
    assert (
        flat("template <typename Args> FORCE_INLINE void read_stream") in body
        and flat("template <typename Args> FORCE_INLINE void read_stream") in s
    )
    reader2 = flat(inspect.getsource(gl._mcast_reader2))
    assert "compile_args=[*recv,*recv2,len(streams),*[cbfor_,cbinstreams],*[0]*(2-len(streams)),zero_cb]" in reader2
    assert "recv=(CB_IN0,FT,B,0),recv2=(CB_RECIP_W,B,B,RECIP_READY),zero_cb=CB_ZERO" in FRONT
    assert gl.RECIP_READY == 6 and gl.RECIP_READY not in (
        0,
        *gr_fold.PARTIALS_SEMAPHORES,
        gr_fold.FRONT_STATS_SCRATCH_SEM,
        gr_fold.FRONT_STATS_READY,
    )


def test_three_phase_writer_orders_stats_gate_u_recip_and_the_python_side_matches():
    src = SRC["mcast_writer3"]
    assert (
        "constexpr uint32_t GATE_SEM = get_compile_time_arg_val(18);" in src
        and "constexpr uint32_t ACCESSOR_BASE = 19;" in src
    )
    assert src.count("next_compile_time_args_offset()") == 5  # six accessor sets
    a = src.index("(a_tiles, a_extra, 0);")
    gate = src.index("gate.up(noc, get_arg_val<uint32_t>(39), get_arg_val<uint32_t>(40), 1);")
    b = src.index("(b_tiles, b_extra, 13);")
    c = src.index("(c_tiles, c_extra, 26);")
    assert a < gate < b < c and "noc.async_atomic_barrier();" in src[gate:b]
    # phase A = gr_fold's (stats tile -> transport scratch + own page), B = the u row (no tensor write), C = the recip
    # tile with the normalized' row as the extra stream; the gate id is FRONT_STATS_READY
    assert (
        "[CB_SOUT,CB_SSCRATCH,1,1,gr_read.NONE_CB,s_scratch]+[CB_U,CB_IN0,HT,0,gr_read.NONE_CB,0]+[CB_RECIP_OUT,CB_RECIP_W,1,0,CB_NOUT,RECIP_READY]+[gr_fold.FRONT_STATS_READY]+accessors"
        in FRONT
    )
    assert "phase_a=[b//links,x,y,x,y,gathered_stats.buffer_address(),b*TP_SIZE+rank,1,0,0,0,1,1]" in FRONT
    assert "phase_b=[b*HT,*rect,0,0,0,0,0,0,1,1]" in FRONT
    assert "phase_c=[b,*rect,0,0,0,normalized.buffer_address(),HT,b*HT,1,4]" in FRONT
    assert "runtime.append((core,phase_a+phase_b+phase_c+list(noc[(core.x,core.y)])))" in FRONT
    assert FRONT.count("fp.accessor_args(normalized)") == 4 and FRONT.count("fp.accessor_args(gathered_stats)") == 2


def test_reader_gate_transports_and_semaphores_are_gr_folds():
    # the norm core's reader: residual, gamma, then the gathered stats behind the gate (stream 2, links + 1)
    assert (
        "[(residual,0),(norm_scale,4),(gathered_stats,1)]" in FRONT
        and "gate=(2,gr_fold.FRONT_STATS_READY,links+1)" in FRONT
    )
    # the two transports: gr_fold's stats phase (consumers = the norm cores' gate) and partials phase, on its pair
    assert 'transports=gr_fold.TRANSPORT["partials"]' in FRONT
    assert (
        "semaphore_ids=(p_go,s_scratch,p_done),consumers=tuple(noc[(c.x,c.y)]forcinproducers),consumer_sem=gr_fold.FRONT_STATS_READY"
        in FRONT
    )
    assert (
        'gr_fold.PARTIALS_SCRATCH_CB,PT,gr_fold.SOURCE_PRODUCERS,"partials",transports,semaphore_ids=gr_fold.PARTIALS_SEMAPHORES,page_strides=(1,PT)'
        in FRONT
    )
    assert "fp.semaphore_descriptor(RECIP_READY,pw_set)" in FRONT and "fp.semaphore_descriptor(0,pw_set)" in FRONT
    # the same io order and output placements as gr_fold's front (low_rank_gate consumes them unchanged)
    assert "[residual,norm_scale,down_inject,gathered_stats,normalized,gathered]" in FRONT
    assert "outputs=((gathered_stats,None),(normalized,3),(gathered,None))" in FRONT
    assert "outputs=((gathered_stats,None),(normalized,3),(gathered,None))" in flat(
        inspect.getsource(gr_fold.stats_normalize_down_gather)
    )


def test_cb_indices_are_distinct_per_core_set():
    indices = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        gl.CB_U,
        gl.CB_NOUT,
        gl.CB_RECIP_OUT,
        gl.CB_SSCALER,
        gl.CB_X2,
        gl.CB_SOUT,
        gl.CB_SSCRATCH,
        gl.CB_IN0,
        gl.CB_RECIP_W,
    ]
    assert len(set(indices)) == len(indices) and max(indices) < fp.CB_COUNT  # the norm cores
    workers = [
        gl.CB_IN0,
        gl.CB_W,
        gl.CB_INTERM,
        gl.CB_RECIP_W,
        gl.CB_PROD,
        gl.CB_ZERO,
        gl.CB_OUT,
        gr_fold.PARTIALS_SCRATCH_CB,
    ]
    assert len(set(workers)) == len(workers)
