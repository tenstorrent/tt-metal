#!/usr/bin/env python3
"""Port of the accumulate zones to the non-streaming compute path (compute_common.hpp sdpa_inner_loop,
selected when fp32_dest_acc_en=True, the production Llama setting). Same slot numbers and names as the
streaming port where the phase corresponds; differences: the softmax exp runs on MATH here (exp_tile inside
sub_exp_block_bcast_cols_inplace), EXP (slot 6) = the acquire..commit window (sub + exp) per DEST batch;
MASK (slot 8) only fires on the diagonal chunk on this path (apply_mask, compute_common.hpp:1748);
K_WAIT / V_WAIT are explicit full-chunk waits inserted before the matmul_blocks calls (matmul_blocks
waits for the same tile count internally, so the wait is not changed, only made visible); Q_WAIT is
not zoned (matmul_blocks waits for Q per subblock). NORM (17) = matmul_reduce plus the recip and output
multiply per q chunk. No EXP_INIT, PUSHES, PUSH_HOLD, RESERVE_QKT, OUT_RESERVE, PACK_DONE, QKTIM_WAIT on
this path (they do not exist as separate statements).
"""
import os
import re
from pathlib import Path

# Two layouts (PORTABLE_CONTRACT.md): this file in $WORK/handoff/revamp/data/bh_zones, or copied into
# $TTM/analysis/campaigns of a checkout, in which case that checkout is the TTM whose kernels it patches.
_SD = Path(__file__).resolve().parent
_REPO = next((p for p in (_SD, *_SD.parents) if (p / "ttnn").is_dir() and (p / "tt_metal").is_dir()), None)
DD = Path(os.environ.get("DD") or (Path.cwd() if _REPO else _SD))
WORK = Path(os.environ.get("SDPA_WORK", _REPO.parent if _REPO else _SD.parents[3]))
TTM = Path(os.environ.get("TTM", _REPO or WORK / "tt-metal"))
K = TTM / "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels"
CC = K / "compute/compute_common.hpp"
SD = K / "compute/sdpa.cpp"


def sub1(s, old, new, count=1, flags=0):
    n = len(re.findall(old, s, flags))
    if n != count:
        raise SystemExit(f"anchor count {n} != {count}: {old[:90]!r}")
    return re.sub(old, new, s, flags=flags)


def main():
    E = re.escape
    s = CC.read_text()
    if "SDPA_ZACC" in s:
        raise SystemExit("compute_common already patched")
    s = sub1(
        s,
        E('#include "api/compute/tile_move_copy.h"\n'),
        '#include "api/compute/tile_move_copy.h"\n#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/sdpa_zones.hpp"\n',
    )
    # QCHUNK raw
    s = sub1(
        s,
        E(
            "    for (uint32_t q_iter = iter_q_start; q_iter < iter_q_end; ++q_iter) {\n        uint32_t q_start_tile = 0;"
        ),
        '    for (uint32_t q_iter = iter_q_start; q_iter < iter_q_end; ++q_iter) {\n        SDPA_ZRAW("QCHUNK");\n        uint32_t q_start_tile = 0;',
    )
    # STEP
    s = sub1(
        s,
        E("            KV_chunks_processed_in_iter++;\n"),
        "            KV_chunks_processed_in_iter++;\n            SDPA_ZACC(0);\n",
    )
    # K_WAIT + RECONFIG before QK
    s = sub1(
        s,
        E(
            "            reconfig_data_format(cb_k_in, cb_q_in);\n            pack_reconfig_data_format(cb_qk_im);\n            matmul_blocks(\n                cb_q_in,\n"
        ),
        "            {\n                SDPA_ZACC(1);\n                cb_k_in_obj.wait_front(k_chunk_tiles);\n            }\n            {\n                SDPA_ZACC(4);\n                reconfig_data_format(cb_k_in, cb_q_in);\n                pack_reconfig_data_format(cb_qk_im);\n            }\n            {\n            SDPA_ZACC(7);\n            matmul_blocks(\n                cb_q_in,\n",
    )
    s = sub1(
        s,
        E("                qk_subblock_w,\n                true /*transpose*/);\n"),
        "                qk_subblock_w,\n                true /*transpose*/);\n            }\n",
    )
    # MASK (diagonal chunk only on this path)
    s = sub1(
        s,
        E("            if (apply_mask) {\n                /* QK += MASK */\n"),
        "            if (apply_mask) {\n                SDPA_ZACC(8);\n                /* QK += MASK */\n",
    )
    # REDUCE
    s = sub1(
        s,
        E(
            "            reconfig_data_format(cb_qk_im, cb_identity_scale_in);\n            reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t>(\n                alias_cur_max, alias_prev_max, Sk_chunk_t, processed_k_chunks > 0);\n"
        ),
        "            {\n                SDPA_ZACC(4);\n                reconfig_data_format(cb_qk_im, cb_identity_scale_in);\n            }\n            {\n                SDPA_ZACC(9);\n                reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t>(\n                    alias_cur_max, alias_prev_max, Sk_chunk_t, processed_k_chunks > 0);\n            }\n",
    )
    # SUBEXP
    s = sub1(
        s,
        E(
            "            sub_exp_block_bcast_cols_inplace<cb_qk_im, Sq_chunk_t, scale_fp32, true>(\n                alias_cur_max, alias_cur_sum, Sk_chunk_t);\n"
        ),
        "            {\n                SDPA_ZACC(5);\n                sub_exp_block_bcast_cols_inplace<cb_qk_im, Sq_chunk_t, scale_fp32, true>(\n                    alias_cur_max, alias_cur_sum, Sk_chunk_t);\n            }\n",
    )
    # V_WAIT + RECONFIG + PV_MM
    s = sub1(
        s,
        E(
            "            reconfig_data_format(cb_v_in, cb_qk_im);\n            pack_reconfig_data_format(alias_mm2_cur_out);\n\n            /* OUT_IM = QK @ V_CHUNK */\n            matmul_blocks(\n"
        ),
        "            {\n                SDPA_ZACC(12);\n                cb_v_in_obj.wait_front(v_chunk_tiles);\n            }\n            {\n                SDPA_ZACC(4);\n                reconfig_data_format(cb_v_in, cb_qk_im);\n                pack_reconfig_data_format(alias_mm2_cur_out);\n            }\n\n            /* OUT_IM = QK @ V_CHUNK */\n            {\n            SDPA_ZACC(13);\n            matmul_blocks(\n",
    )
    s = sub1(
        s,
        E(
            "                out_subblock_w,\n                false /*transpose*/);\n\n            cb_qk_im_obj.pop_front(qk_chunk_tiles);\n            reconfig_data_format(alias_prev_max, alias_cur_max);\n"
        ),
        "                out_subblock_w,\n                false /*transpose*/);\n            }\n\n            {\n                SDPA_ZACC(19);\n                cb_qk_im_obj.pop_front(qk_chunk_tiles);\n            }\n            {\n                SDPA_ZACC(4);\n                reconfig_data_format(alias_prev_max, alias_cur_max);\n            }\n",
    )
    # SALAD_EXP and SALAD_CORR inside processed_k_chunks > 0
    s = sub1(
        s,
        E(
            "                sub_exp_block<scale_fp32>(alias_prev_max, alias_cur_max, cb_exp_max_diff, Sq_chunk_t);\n                CircularBuffer(alias_prev_max).pop_front(Sq_chunk_t);\n\n                /**\n                 * cb_prev_sum *= cb_exp_max_diff\n"
        ),
        "                {\n                    SDPA_ZACC(15);\n                    sub_exp_block<scale_fp32>(alias_prev_max, alias_cur_max, cb_exp_max_diff, Sq_chunk_t);\n                    CircularBuffer(alias_prev_max).pop_front(Sq_chunk_t);\n                }\n                SDPA_ZACC(16);\n\n                /**\n                 * cb_prev_sum *= cb_exp_max_diff\n",
    )
    # NORM: matmul_reduce and the standard normalization branch
    s = sub1(
        s,
        E("        matmul_reduce<Sq_chunk_t>(cb_col_identity, alias_prev_sum);\n"),
        "        {\n            SDPA_ZACC(17);\n            matmul_reduce<Sq_chunk_t>(cb_col_identity, alias_prev_sum);\n        }\n",
    )
    s = sub1(
        s,
        E(
            "        } else {\n            /* cb_cur_sum = 1.0 / cb_cur_sum */\n            recip_block_inplace(alias_prev_sum, Sq_chunk_t);\n\n"
        ),
        "        } else {\n            SDPA_ZACC(17);\n            /* cb_cur_sum = 1.0 / cb_cur_sum */\n            recip_block_inplace(alias_prev_sum, Sq_chunk_t);\n\n",
    )
    # EXP = acquire..commit window of the in-place sub_exp (MATH: sub + exp per DEST batch)
    s = sub1(
        s,
        E(
            "            tile_regs_acquire();\n            for (uint32_t j = 0; j < dst_tiles; ++j) {\n                sub_tiles_bcast_cols(in0_cb, in1_cb, j, i, j);\n"
        ),
        "            tile_regs_acquire();\n            {\n            SDPA_ZACC(6);\n            for (uint32_t j = 0; j < dst_tiles; ++j) {\n                sub_tiles_bcast_cols(in0_cb, in1_cb, j, i, j);\n",
    )
    s = sub1(
        s,
        E(
            "                exp_tile<true /* approx */, false /* scale_en */, InputClamping::None, iterations>(j, vector_mode_exp);\n            }\n            tile_regs_commit();\n"
        ),
        "                exp_tile<true /* approx */, false /* scale_en */, InputClamping::None, iterations>(j, vector_mode_exp);\n            }\n            }\n            tile_regs_commit();\n",
    )
    CC.write_text(s)
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
    flush = "".join(
        f'            SDPA_ZFLUSH({i}, "{n}");\n'
        for i, n in enumerate(names)
        if n not in ("Q_WAIT", "RESERVE_QKT", "OUT_RESERVE", "QKTIM_WAIT", "PACK_DONE", "PUSH_HOLD", "MASK_DIAG")
    )
    # second use_zigzag_balancing); is the sdpa_standard call in the non-streaming branch
    parts = sd.split("                use_zigzag_balancing);\n")
    assert len(parts) == 2, len(parts)
    sd = parts[0] + "                use_zigzag_balancing);\n" + flush + parts[1]
    SD.write_text(sd)
    print("compute_common port applied")


if __name__ == "__main__":
    main()
