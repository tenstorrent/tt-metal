#!/usr/bin/env python3
"""Apply the SDPA zone instrumentation to the scratch branch kernels (idempotent: refuses if already applied).

Zone index map (see bh/zone_decomposition.md for the file:line of every site after patching):
compute (TRISC0/1/2, same indices, per-thread records):
  0 STEP  1 K_WAIT  2 Q_WAIT  3 RESERVE_QKT  4 RECONFIG  5 SUBEXP  6 EXP  7 QK_MM  8 MASK  9 REDUCE
 10 OUT_RESERVE 11 QKTIM_WAIT 12 V_WAIT 13 PV_MM 14 PACK_DONE 15 SALAD_EXP 16 SALAD_CORR 17 NORM
 18 PUSH_HOLD 19 POPS 20 MASK_DIAG 21 EXP_INIT 22 PUSHES (second pass, see note at end)
reader (one DM RISC): 0 R_K_READ 1 R_V_READ 2 R_Q_READ 3 R_RESERVE 4 R_BARRIER 5 R_KCHUNK
writer (other DM RISC): 0 W_WAIT 1 W_DRAIN
raw per-q-chunk zones: QCHUNK (compute), R_QCHUNK (reader), W_QCHUNK (writer)
"""
import os
import re
import sys
from pathlib import Path

# Two layouts (PORTABLE_CONTRACT.md): this file in $WORK/handoff/revamp/data/bh_zones, or copied into
# $TTM/analysis/campaigns of a checkout, in which case that checkout is the TTM whose kernels it patches.
_SD = Path(__file__).resolve().parent
_REPO = next((p for p in (_SD, *_SD.parents) if (p / "ttnn").is_dir() and (p / "tt_metal").is_dir()), None)
DD = Path(os.environ.get("DD") or (Path.cwd() if _REPO else _SD))
WORK = Path(os.environ.get("SDPA_WORK", _REPO.parent if _REPO else _SD.parents[3]))
TTM = Path(os.environ.get("TTM", _REPO or WORK / "tt-metal"))
K = TTM / "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels"
CS = K / "compute/compute_streaming.hpp"
SD = K / "compute/sdpa.cpp"
RD = K / "dataflow/reader_interleaved.cpp"
WR = K / "dataflow/writer_interleaved.cpp"
DF = K / "dataflow/dataflow_common.hpp"


def sub1(s, old, new, count=1, flags=0):
    n = len(re.findall(old, s, flags))
    if n != count:
        raise SystemExit(f"anchor count {n} != {count}: {old[:80]!r}")
    return re.sub(old, new, s, flags=flags)


def wrap_line(s, line_re, idx, count=1):
    """Wrap every match of an indented single line in `{ SDPA_ZACC(idx); <line> }`."""

    def rep(m):
        ws = m.group(1)
        return f"{ws}{{\n{ws}    SDPA_ZACC({idx});\n{ws}    {m.group(2)}\n{ws}}}\n"

    return sub1(s, r"^([ ]+)(" + line_re + r")\n", rep, count, re.M)


def main():
    cs = CS.read_text()
    if "SDPA_ZACC" in cs:
        raise SystemExit("already patched")
    E = re.escape

    # --- compute_streaming.hpp ---
    cs = sub1(
        cs,
        E('#include "tools/profiler/kernel_profiler.hpp"\n'),
        '#include "tools/profiler/kernel_profiler.hpp"\n#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/sdpa_zones.hpp"\n',
    )
    # 6 EXP + A6 stub
    cs = sub1(
        cs,
        E('        MaybeDeviceZoneScopedN(profiling_enabled, "EXP");\n'),
        '        MaybeDeviceZoneScopedN(profiling_enabled, "EXP");\n        SDPA_ZACC(6);\n',
    )
    cs = sub1(
        cs,
        E(
            "                exp_packthread_tile<true, false, InputClamping::None, iterations>(dst_index++, vector_mode_exp);\n"
        ),
        "#if !SDPA_ABL_EXP_STUB\n                exp_packthread_tile<true, false, InputClamping::None, iterations>(dst_index++, vector_mode_exp);\n#else\n                dst_index++;\n#endif\n",
    )
    # 0 STEP
    cs = sub1(
        cs,
        E("    const uint32_t kt_num_full_subblocks = active_Sk / actual_sbw;\n"),
        "    const uint32_t kt_num_full_subblocks = active_Sk / actual_sbw;\n    SDPA_ZACC(0);\n",
    )
    # 3 RESERVE_QKT
    cs = sub1(
        cs,
        E(
            "    CircularBuffer(cb_qkt_im).reserve_back(Sq_chunk_t * KT_stride);\n\n    CircularBuffer(cur.sum).reserve_back(Sq_chunk_t);\n    if (save_max_cb != INVALID_CB) {\n        CircularBuffer(save_max_cb).reserve_back(Sq_chunk_t);\n    }\n"
        ),
        "    {\n        SDPA_ZACC(3);\n        CircularBuffer(cb_qkt_im).reserve_back(Sq_chunk_t * KT_stride);\n        CircularBuffer(cur.sum).reserve_back(Sq_chunk_t);\n        if (save_max_cb != INVALID_CB) {\n            CircularBuffer(save_max_cb).reserve_back(Sq_chunk_t);\n        }\n    }\n",
    )
    # 1 K_WAIT, 2 Q_WAIT
    cs = wrap_line(cs, E("CircularBuffer(cb_kt_in).wait_front(DHt * KT_stride);"), 1)
    cs = wrap_line(cs, E("CircularBuffer(cb_q_in).wait_front(q_wait_tiles);"), 2)
    # 4 RECONFIG phase-1 head (pack reconfig .. configure_row_pack_width)
    cs = sub1(
        cs,
        E("        sdpa_maybe_pack_reconfig_data_format<cb_normalized_out, cb_qkt_im>();\n"),
        "        {\n        SDPA_ZACC(4);\n        sdpa_maybe_pack_reconfig_data_format<cb_normalized_out, cb_qkt_im>();\n",
    )
    cs = sub1(
        cs,
        E("        configure_row_pack_width(cb_qkt_im, actual_sbw);\n"),
        "        configure_row_pack_width(cb_qkt_im, actual_sbw);\n        }\n",
    )
    # 4/5 in the kt loop (q_subblock > 0)
    cs = sub1(
        cs,
        E(
            "                uint32_t prev_q_subblock = q_subblock - 1;\n                sdpa_maybe_reconfig_data_format<cb_kt_in, cb_qkt_im, cb_q_in, cb_qkt_im>();\n                sub_exp_block_bcast_cols<profiling_enabled, scale_fp32>(\n"
        ),
        "                uint32_t prev_q_subblock = q_subblock - 1;\n                {\n                    SDPA_ZACC(4);\n                    sdpa_maybe_reconfig_data_format<cb_kt_in, cb_qkt_im, cb_q_in, cb_qkt_im>();\n                }\n                {\n                SDPA_ZACC(5);\n                sub_exp_block_bcast_cols<profiling_enabled, scale_fp32>(\n",
    )
    cs = sub1(
        cs,
        E(
            "                    /*skip_pack_configure=*/true);\n                sdpa_maybe_pack_reconfig_data_format<cb_recip_scratch, cb_qkt_im>();\n                sdpa_maybe_reconfig_data_format<cb_qkt_im, cb_kt_in, cb_qkt_im, cb_q_in>();\n                mm_no_mop_reinit_short(cb_q_in, cb_kt_in, true, actual_sbw, qkt_subblock_h, in0_block_w);\n"
        ),
        "                    /*skip_pack_configure=*/true);\n                }\n                {\n                    SDPA_ZACC(4);\n                    sdpa_maybe_pack_reconfig_data_format<cb_recip_scratch, cb_qkt_im>();\n                    sdpa_maybe_reconfig_data_format<cb_qkt_im, cb_kt_in, cb_qkt_im, cb_q_in>();\n                    mm_no_mop_reinit_short(cb_q_in, cb_kt_in, true, actual_sbw, qkt_subblock_h, in0_block_w);\n                }\n",
    )
    # 7 QK_MM
    cs = sub1(
        cs,
        E('                MaybeDeviceZoneScopedN(profiling_enabled, "Q@KT MM+Pack");\n'),
        '                MaybeDeviceZoneScopedN(profiling_enabled, "Q@KT MM+Pack");\n                SDPA_ZACC(7);\n',
    )
    # 4 restore reconfig after Q@KT
    cs = sub1(
        cs,
        E(
            "        // Restore float16b for mask/reduce after Q@KT.\n        sdpa_maybe_reconfig_data_format<cb_kt_in, cb_qkt_im, cb_q_in, cb_qkt_im>();\n"
        ),
        "        // Restore float16b for mask/reduce after Q@KT.\n        {\n            SDPA_ZACC(4);\n            sdpa_maybe_reconfig_data_format<cb_kt_in, cb_qkt_im, cb_q_in, cb_qkt_im>();\n        }\n",
    )
    # 8/20 MASK (+ A2)
    cs = sub1(
        cs,
        E("        const bool should_apply_lightweight_mask = sdpa_lightweight_mask_stamped("),
        "        const bool should_apply_lightweight_mask = (SDPA_ABL_MASK_OFF == 0) && sdpa_lightweight_mask_stamped(",
    )
    cs = sub1(
        cs,
        E("            if (should_apply_lightweight_mask) {\n"),
        "            if (should_apply_lightweight_mask) {\n                SDPA_ZACC(is_last_iter ? 20u : 8u);\n",
    )
    # 18 PUSH_HOLD
    cs = wrap_line(cs, E("cb_push_back_hold_wr_ptr(cb_qkt_im, row_tiles);"), 18)
    # 9 REDUCE
    cs = sub1(
        cs,
        E('            MaybeDeviceZoneScopedN(profiling_enabled, "Reduce max");\n'),
        '            MaybeDeviceZoneScopedN(profiling_enabled, "Reduce max");\n            SDPA_ZACC(9);\n',
    )
    # 19 POPS
    cs = wrap_line(cs, E("CircularBuffer(cb_kt_in).pop_front(DHt * KT_stride);"), 19)
    cs = sub1(
        cs,
        E(
            "        if (is_last_iter) {\n            sdpa_cb_pop_front_out_of_line(cb_q_in, Sq_chunk_t * DHt);\n        }\n"
        ),
        "        if (is_last_iter) {\n            SDPA_ZACC(19);\n            sdpa_cb_pop_front_out_of_line(cb_q_in, Sq_chunk_t * DHt);\n        }\n",
    )
    cs = sub1(
        cs,
        E(
            "        CircularBuffer(cb_v_in).pop_front(KT_stride * v_cb_physical_width_t);\n        CircularBuffer(cb_qkt_im).pop_front(Sq_chunk_t * KT_stride);\n"
        ),
        "        {\n            SDPA_ZACC(19);\n            CircularBuffer(cb_v_in).pop_front(KT_stride * v_cb_physical_width_t);\n            CircularBuffer(cb_qkt_im).pop_front(Sq_chunk_t * KT_stride);\n        }\n",
    )
    # 10 OUT_RESERVE
    cs = wrap_line(cs, E("CircularBuffer(out_cb).reserve_back(qktv_output_num_tiles);"), 10)
    # 5 SUBEXP phase-2 group-0 drain (materialized path; the in-place branch is not compiled here)
    cs = sub1(
        cs,
        E(
            "                for (uint32_t kt_sub = 0; kt_sub < kt_num_full_subblocks; ++kt_sub) {\n                    sub_exp_block_bcast_cols<profiling_enabled, scale_fp32>(\n                        cb_qkt_im,\n                        cur.max,\n                        cur.sum,\n                        KT_stride,\n                        q_num_subblocks - 1,\n                        kt_sub * actual_sbw,\n                        qkt_subblock_h,\n                        actual_sbw);\n                    if constexpr (q_num_subblocks == 1) {\n"
        ),
        "                for (uint32_t kt_sub = 0; kt_sub < kt_num_full_subblocks; ++kt_sub) {\n                    {\n                    SDPA_ZACC(5);\n                    sub_exp_block_bcast_cols<profiling_enabled, scale_fp32>(\n                        cb_qkt_im,\n                        cur.max,\n                        cur.sum,\n                        KT_stride,\n                        q_num_subblocks - 1,\n                        kt_sub * actual_sbw,\n                        qkt_subblock_h,\n                        actual_sbw);\n                    }\n                    if constexpr (q_num_subblocks == 1) {\n",
    )
    # 11 QKTIM_WAIT, 12 V_WAIT (all sites; the ring/in-place duplicates are not compiled)
    cs = wrap_line(cs, E("CircularBuffer(cb_qkt_im).wait_front(qktv_in0_wait_tiles);"), 11, count=3)
    cs = wrap_line(cs, E("CircularBuffer(cb_v_in).wait_front(Sk_chunk_t * v_cb_physical_width_t);"), 12, count=3)
    cs = sub1(
        cs,
        E(
            "            if (is_remainder_iter) {\n                CircularBuffer(cb_qkt_im).wait_front(Sq_chunk_t * KT_stride);\n            } else {\n"
        ),
        "            if (is_remainder_iter) {\n                SDPA_ZACC(11);\n                CircularBuffer(cb_qkt_im).wait_front(Sq_chunk_t * KT_stride);\n            } else {\n",
    )
    # 4 RECONFIG in the V-matmul blocks (3 sites each; in-place duplicate not compiled)
    cs = sub1(
        cs,
        r"^([ ]+)sdpa_maybe_reconfig_data_format<cb_normalized_out, cb_v_in, cb_normalized_out, cb_qkt_im>\(\n[ ]+out_cb, out_cb\);\n",
        lambda m: f"{m.group(1)}{{\n{m.group(1)}    SDPA_ZACC(4);\n{m.group(1)}    sdpa_maybe_reconfig_data_format<cb_normalized_out, cb_v_in, cb_normalized_out, cb_qkt_im>(out_cb, out_cb);\n{m.group(1)}}}\n",
        count=3,
        flags=re.M,
    )
    cs = wrap_line(cs, E("sdpa_maybe_reconfig_data_format<cb_v_in, cb_qkt_im, cb_qkt_im, cb_qkt_im>();"), 4, count=3)
    cs = sub1(
        cs,
        E(
            "                        mm_no_mop_reinit_short(cb_qkt_im, cb_v_in, false, qktv_subblock_w, qktv_h, KT_stride);\n                        configure_row_pack_width(out_cb, qktv_subblock_w);\n"
        ),
        "                        {\n                            SDPA_ZACC(4);\n                            mm_no_mop_reinit_short(cb_qkt_im, cb_v_in, false, qktv_subblock_w, qktv_h, KT_stride);\n                            configure_row_pack_width(out_cb, qktv_subblock_w);\n                        }\n",
    )
    cs = sub1(
        cs,
        E(
            "                mm_no_mop_reinit_short(cb_qkt_im, cb_v_in, false, qktv_subblock_w, cur_h, KT_stride);\n                // Configure once before v_subblock loop; skip inside.\n                configure_row_pack_width(out_cb, qktv_subblock_w);\n"
        ),
        "                {\n                    SDPA_ZACC(4);\n                    mm_no_mop_reinit_short(cb_qkt_im, cb_v_in, false, qktv_subblock_w, cur_h, KT_stride);\n                    configure_row_pack_width(out_cb, qktv_subblock_w);\n                }\n",
    )
    # 13 PV_MM: wrap both v_subblock loops
    cs = sub1(
        cs,
        r"^(?P<i>[ ]+)for \(uint32_t v_subblock = 0; v_subblock < qktv_v_num_subblocks; \+\+v_subblock\) \{\n(?P<body>.*?)\n(?P=i)\}\n",
        lambda m: f"{m.group('i')}{{\n{m.group('i')}SDPA_ZACC(13);\n{m.group('i')}for (uint32_t v_subblock = 0; v_subblock < qktv_v_num_subblocks; ++v_subblock) {{\n{m.group('body')}\n{m.group('i')}}}\n{m.group('i')}}}\n",
        count=2,
        flags=re.M | re.S,
    )
    # 14 PACK_DONE (the per-k-chunk handshake at 8-space indentation)
    cs = sub1(
        cs,
        E(
            "        PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));\n        UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));\n        UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));\n"
        ),
        "        {\n            SDPA_ZACC(14);\n            PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));\n            UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));\n            UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));\n        }\n",
    )
    # 17 NORM, 16 SALAD_CORR
    cs = sub1(
        cs,
        E('            MaybeDeviceZoneScopedN(profiling_enabled, "ROW_NORM");\n'),
        '            MaybeDeviceZoneScopedN(profiling_enabled, "ROW_NORM");\n            SDPA_ZACC(17);\n',
    )
    cs = sub1(
        cs,
        E("        auto salad_correct_row = [&](uint32_t salad_row, uint32_t w_salad, uint32_t sbh) {\n"),
        "        auto salad_correct_row = [&](uint32_t salad_row, uint32_t w_salad, uint32_t sbh) {\n            SDPA_ZACC(16);\n",
    )
    # 15 SALAD_EXP
    cs = sub1(
        cs,
        E("                CircularBuffer(cb_exp_max_diff).reserve_back(qktv_h);\n"),
        "                SDPA_ZACC(15);\n                CircularBuffer(cb_exp_max_diff).reserve_back(qktv_h);\n",
    )
    cs = sub1(
        cs,
        r"^([ ]+)CircularBuffer\(cb_exp_max_diff\)\.reserve_back\(drain_h\);\n(.*?)\n([ ]+)CircularBuffer\(cb_exp_max_diff\)\.push_back\(drain_h\);\n",
        lambda m: f"{m.group(1)}{{\n{m.group(1)}SDPA_ZACC(15);\n{m.group(1)}CircularBuffer(cb_exp_max_diff).reserve_back(drain_h);\n{m.group(2)}\n{m.group(3)}CircularBuffer(cb_exp_max_diff).push_back(drain_h);\n{m.group(1)}}}\n",
        count=2,
        flags=re.M | re.S,
    )
    # QCHUNK raw zone (first occurrence = sdpa_standard_v2)
    cs = sub1(
        cs,
        E(
            "    for (uint32_t q = 0; q < q_chunks_per_core; q++) {\n        AccumulatorHalf prev = {cb_sum_A, cb_max_A, cb_out_im_A};\n"
        ),
        '    for (uint32_t q = 0; q < q_chunks_per_core; q++) {\n        SDPA_ZRAW("QCHUNK");\n        AccumulatorHalf prev = {cb_sum_A, cb_max_A, cb_out_im_A};\n',
    )
    CS.write_text(cs)

    # --- sdpa.cpp: flush after sdpa_standard_v2 ---
    sd = SD.read_text()
    names = [
        "STEP",
        "K_WAIT",
        "Q_WAIT",
        "RESERVE_QKT",
        "RECONFIG",
        "SUBEXP",
        "EXP",
        "QK_MM",
        "MASK",
        "REDUCE",
        "OUT_RESERVE",
        "QKTIM_WAIT",
        "V_WAIT",
        "PV_MM",
        "PACK_DONE",
        "SALAD_EXP",
        "SALAD_CORR",
        "NORM",
        "PUSH_HOLD",
        "POPS",
        "MASK_DIAG",
    ]
    flush = "".join(f'        SDPA_ZFLUSH({i}, "{n}");\n' for i, n in enumerate(names))
    sd = sub1(
        sd,
        E("            use_zigzag_balancing);\n    } else {\n        // Standard SDPA path"),
        "            use_zigzag_balancing);\n" + flush + "    } else {\n        // Standard SDPA path",
    )
    SD.write_text(sd)

    # --- dataflow_common.hpp ---
    df = DF.read_text()
    df = sub1(
        df,
        E('#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/sliding_window_geometry.hpp"\n'),
        '#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/sliding_window_geometry.hpp"\n#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/sdpa_zones.hpp"\n',
    )
    df = wrap_line(df, E("cb.reserve_back(num_tiles);"), 3, count=3)
    df = wrap_line(df, E("cb.reserve_back(sb_tiles);"), 3)
    df = wrap_line(df, E("noc.async_read_barrier();"), 4, count=11)
    df = wrap_line(df, E("cb.wait_front(tiles_this_group);"), 0, count=2)
    DF.write_text(df)

    # --- reader ---
    rd = RD.read_text()
    rd = sub1(
        rd,
        E("    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_cores>();\n"),
        "#if SDPA_ABL_BARRIER_THR > 0\n    constexpr uint32_t barrier_threshold = SDPA_ABL_BARRIER_THR;\n#else\n    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_cores>();\n#endif\n",
    )
    rd = sub1(
        rd,
        E("        for (uint32_t global_q_iter = 0; global_q_iter < global_q_count; ++global_q_iter) {\n"),
        '        for (uint32_t global_q_iter = 0; global_q_iter < global_q_count; ++global_q_iter) {\n            SDPA_ZRAW("R_QCHUNK");\n',
    )
    rd = sub1(
        rd,
        E("            for (uint32_t k_chunk = k_loop_start; (k_chunk * Sk_chunk_t) < q_high_idx; ++k_chunk) {\n"),
        "            for (uint32_t k_chunk = k_loop_start; (k_chunk * Sk_chunk_t) < q_high_idx; ++k_chunk) {\n                SDPA_ZACC(5);\n",
    )
    rd = sub1(
        rd,
        r"^([ ]+)read_chunk_with_padding<k_tile_bytes>\((.*?)\);\n",
        lambda m: f"{m.group(1)}SDPA_ZACC(0);\n#if SDPA_ABL_READER_STUB\n{m.group(1)}cb_k.reserve_back(k_chunk_tiles);\n{m.group(1)}cb_k.push_back(k_chunk_tiles);\n#else\n{m.group(1)}read_chunk_with_padding<k_tile_bytes>({m.group(2)});\n#endif\n",
        flags=re.M | re.S,
    )
    rd = sub1(
        rd,
        r"^([ ]+)read_chunk_with_padding<v_tile_bytes>\((.*?)\);\n",
        lambda m: f"{m.group(1)}SDPA_ZACC(1);\n#if SDPA_ABL_READER_STUB\n{m.group(1)}cb_v.reserve_back(v_chunk_tiles);\n{m.group(1)}cb_v.push_back(v_chunk_tiles);\n#else\n{m.group(1)}read_chunk_with_padding<v_tile_bytes>({m.group(2)});\n#endif\n",
        flags=re.M | re.S,
    )
    rd = sub1(
        rd,
        E("                    if (k_chunk == k_loop_start) {\n"),
        "                    if (k_chunk == k_loop_start) {\n                        SDPA_ZACC(2);\n",
    )
    rflush = "".join(
        f'    SDPA_ZFLUSH({i}, "{n}");\n'
        for i, n in enumerate(["R_K_READ", "R_V_READ", "R_Q_READ", "R_RESERVE", "R_BARRIER", "R_KCHUNK"])
    )
    assert rd.endswith("    }  // close phase\n}\n")
    rd = rd[:-2] + rflush + "}\n"
    RD.write_text(rd)

    # --- writer ---
    wr = WR.read_text()
    wr = sub1(
        wr,
        E("        for (uint32_t global_q_iter = 0; global_q_iter < global_q_count; ++global_q_iter) {\n"),
        '        for (uint32_t global_q_iter = 0; global_q_iter < global_q_count; ++global_q_iter) {\n            SDPA_ZRAW("W_QCHUNK");\n',
    )
    wr = sub1(
        wr,
        E("            if constexpr (use_streaming_compute) {\n"),
        "            if constexpr (use_streaming_compute) {\n                SDPA_ZACC(1);\n",
    )
    wflush = "".join(f'    SDPA_ZFLUSH({i}, "{n}");\n' for i, n in enumerate(["W_WAIT", "W_DRAIN"]))
    assert wr.endswith("    }  // close phase\n}\n")
    wr = wr[:-2] + wflush + "}\n"
    WR.write_text(wr)
    print("patched OK")


if __name__ == "__main__":
    main()

# NOTE: a second pass (applied 2026-09-11 22:25Z after the first smoke run showed ~550k un-zoned PACK cycles)
# added slot 21 EXP_INIT around both exp_packthread_tile_init calls and slot 22 PUSHES around the seven
# cur.sum/out_cb push_back pairs outside normalize_row; flush names appended in sdpa.cpp. The full effective
# patch is bh/zone_patch.diff.
