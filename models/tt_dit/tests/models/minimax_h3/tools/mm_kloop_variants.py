# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""K-loop engine isolation for the single-device minimal_matmul compute kernel (the AGMM's matmul_blocks, 2x2 subblock only).

Edits ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/compute.cpp in place (exact-match, asserts if the
kernel text moved on), adds a per-output-block KLOOP device zone around the k_block loop, and selects one variant:

    0  full loop + KLOOP zone
    1  mathmock    no MVMUL: MATH only waits for and clears the four src dvalids per K tile (tt_metal/tt-llk/tests/helpers/include/perf.h scheme)
    2  unpackmock  no UNPACR: UNPACK only waits for bank clear and sets the four dvalids; MATH runs the real matmul
    3  nopack      no PACR in matmul_blocks (tile_regs handshake kept)
    4  mathmock + nopack: the bare unpack stream

    python models/tt_dit/tests/models/minimax_h3/tools/mm_kloop_variants.py apply <0..4> [legacy]
    python -m tracy -r -p models/tt_dit/tests/models/minimax_h3/tools/transformer_op_single_device_bench.py --op ff1 --no-fusion \
        --fidelity HiFi2 --cases "8,7,10,2,2,1" --iters 3
    python models/tt_dit/tests/models/minimax_h3/tools/mm_kloop_variants.py parse generated/profiler/reports/<ts>/profile_log_device.csv
    python models/tt_dit/tests/models/minimax_h3/tools/mm_kloop_variants.py revert      # restores the pristine copy

apply keeps a pristine copy of the kernel next to it (compute.cpp.kloop_variants.orig) on first use and works from that copy, so
uncommitted kernel edits survive; `legacy` builds the variant on the per-tile `matmul_block` loop (standard reuse-A order every K
tile) instead of `matmul_block_kloop` (alternating orders), for before/after isolation tables.

parse prints, per run host ID (one per op call), the mean TRISC_1 kernel time and the KLOOP sum per core divided by the
63,504 K-tile steps a 54-row core performs at M 13664 / K 5376 / N 7168 with the (8,7,10) 2x2 blocking (one step = one K tile
of the matmul_block_kloop call = 4 unpacks + 4 tile-MACs). Output is garbage in variants 1-4; only the time counts. Not a test.
"""

import os
import shutil
import sys

K = "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/compute.cpp"
ORIG = K + ".kloop_variants.orig"  # pristine copy of the kernel, taken on the first apply, restored by revert


def restore():
    if os.path.exists(ORIG):
        shutil.copyfile(ORIG, K)
    else:
        shutil.copyfile(K, ORIG)


def apply(v, legacy=False):
    restore()
    h = {"s": open(K).read()}

    def rep(old, new):
        assert h["s"].count(old) == 1, old[:60]
        h["s"] = h["s"].replace(old, new, 1)

    rep(
        '#include "api/compute/matmul.h"\n',
        '#include "api/compute/matmul.h"\n#include "tools/profiler/kernel_profiler.hpp"\n'
        f"#define KLOOP_VARIANT {v}\n"
        "// 2x2 subblock, reuse_a (ct >= rt): per K tile the unpacker delivers B0,B1 -> SrcB (2 banks) then A0,A1 -> SrcA.\n"
        "// Math mock: consume the four dvalids in the same order the real MOP scheme releases them (tt-llk tests/helpers/include/perf.h).\n"
        "inline void kloop_math_mock_2x2(uint32_t kt) {\n"
        "    for (uint32_t k = 0; k < kt; k++) {\n"
        "    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD); TTI_CLEARDVALID(p_setrwc::CLR_A, 0);\n"
        "    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD); TTI_CLEARDVALID(p_setrwc::CLR_A, 0);\n"
        "    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCB_VLD); TTI_CLEARDVALID(p_setrwc::CLR_B, 0);\n"
        "    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCB_VLD); TTI_CLEARDVALID(p_setrwc::CLR_B, 0);\n"
        "    }\n"
        "}\n"
        "inline void kloop_unpack_mock_2x2(uint32_t kt) {\n"
        "    for (uint32_t k = 0; k < kt; k++) {\n"
        "    TTI_STALLWAIT(p_stall::STALL_TDMA, p_stall::SRCB_CLR); TTI_SETDVALID(p_setrwc::CLR_B);\n"
        "    TTI_STALLWAIT(p_stall::STALL_TDMA, p_stall::SRCB_CLR); TTI_SETDVALID(p_setrwc::CLR_B);\n"
        "    TTI_STALLWAIT(p_stall::STALL_TDMA, p_stall::SRCA_CLR); TTI_SETDVALID(p_setrwc::CLR_A);\n"
        "    TTI_STALLWAIT(p_stall::STALL_TDMA, p_stall::SRCA_CLR); TTI_SETDVALID(p_setrwc::CLR_A);\n"
        "    }\n"
        "}\n",
    )
    old_call = """            matmul_block_kloop(
                in0_cb,
                in1_cb,
                in0_index,
                in1_index,
                dst_index,
                false /*transpose*/,
                subblock_w,
                subblock_h,
                K_block_tiles,
                full_N_block_tiles /*in1_kt_stride*/);"""
    if legacy:
        # the per-tile matmul_block loop the kernel ran before matmul_block_kloop (standard reuse-A order every K tile)
        new_call = """#if KLOOP_VARIANT == 1 || KLOOP_VARIANT == 4
            for (uint32_t k = 0; k < K_block_tiles; k++) {
                UNPACK((llk_unpack_AB_matmul(in0_cb, in1_cb, in0_index + k, in1_index + k * full_N_block_tiles, subblock_w, subblock_h, K_block_tiles)));
            }
            MATH((kloop_math_mock_2x2(K_block_tiles)));
#elif KLOOP_VARIANT == 2
            UNPACK((kloop_unpack_mock_2x2(K_block_tiles)));
            for (uint32_t k = 0; k < K_block_tiles; k++) {
                MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(dst_index, subblock_w, subblock_h)));
            }
#else
            for (uint32_t k = 0; k < K_block_tiles; k++) {
                matmul_block(
                    in0_cb,
                    in1_cb,
                    in0_index + k,
                    in1_index + k * full_N_block_tiles,
                    dst_index,
                    false /*transpose*/,
                    subblock_w,
                    subblock_h,
                    K_block_tiles);
            }
#endif"""
        rep("            matmul_block_kloop_init(\n", "            matmul_block_init(\n")
    else:
        new_call = """#if KLOOP_VARIANT == 1 || KLOOP_VARIANT == 4
            UNPACK((llk_unpack_AB_matmul_kloop(in0_cb, in1_cb, in0_index, in1_index, subblock_w, subblock_h, K_block_tiles, full_N_block_tiles)));
            MATH((kloop_math_mock_2x2(K_block_tiles)));
#elif KLOOP_VARIANT == 2
            UNPACK((kloop_unpack_mock_2x2(K_block_tiles)));
            MATH((llk_math_matmul_kloop<MATH_FIDELITY>(dst_index, subblock_w, subblock_h, K_block_tiles)));
#else
            matmul_block_kloop(
                in0_cb,
                in1_cb,
                in0_index,
                in1_index,
                dst_index,
                false /*transpose*/,
                subblock_w,
                subblock_h,
                K_block_tiles,
                full_N_block_tiles /*in1_kt_stride*/);
#endif"""
    rep(old_call, new_call)
    rep(
        """                    pack_tile<true>(write_dst_index, out_cb, out_tile_id);
""",
        """#if KLOOP_VARIANT != 3 && KLOOP_VARIANT != 4
                    pack_tile<true>(write_dst_index, out_cb, out_tile_id);
#endif
""",
    )
    rep(
        """            // Accumulation buffer
            cb_intermediate.reserve_back(out_block_num_tiles);
            for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {""",
        """#if KLOOP_VARIANT == 1 || KLOOP_VARIANT == 4
            // the math mock clears the dvalids itself; matmul_block_kloop_init disabled the auto-clears for the reuse scheme
            MATH((_llk_math_matmul_uninit_()));
#endif
            // Accumulation buffer
            cb_intermediate.reserve_back(out_block_num_tiles);
            {
            DeviceZoneScopedN("KLOOP");
            for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {""",
    )
    rep(
        """            cb_intermediate.push_back(out_block_num_tiles);
            PACK((llk_pack_reconfig_l1_acc(0)));
""",
        """            }
            cb_intermediate.push_back(out_block_num_tiles);
            PACK((llk_pack_reconfig_l1_acc(0)));
""",
    )
    open(K, "w").write(h["s"])
    print(f"applied variant {v}{' (legacy per-tile matmul_block)' if legacy else ''} to {K}")


STEPS = 63504  # K-tile steps per core: 378 subblocks per K block x 168 K tiles


def parse(path):
    import collections
    import csv
    import statistics as st

    rows = list(csv.DictReader(open(path).read().split("\n", 1)[1].splitlines()))
    starts = {}
    kl = collections.defaultdict(lambda: collections.defaultdict(int))
    kern = collections.defaultdict(dict)
    for r in rows:
        if r[" RISC processor type"].strip() != "TRISC_1":
            continue
        z, t = r[" zone name"].strip(), r[" type"].strip()
        rid, core, tm = r[" run host ID"].strip(), (r[" core_x"], r[" core_y"]), int(r[" time[cycles since reset]"])
        key = (rid, core, z)
        if t == "ZONE_START":
            starts[key] = tm
        elif t == "ZONE_END" and key in starts:
            d = tm - starts.pop(key)
            if z == "KLOOP":
                kl[rid][core] += d
            elif "KERNEL" in z:
                kern[rid][core] = d
    for rid in sorted(kern, key=int):
        k = list(kern[rid].values())
        line = f"run {rid:>5s}: kernel mean {st.mean(k) / 1e6:7.3f} ms ({len(k)} cores)"
        if rid in kl:
            kd = list(kl[rid].values())
            line += (
                f" | KLOOP mean {st.mean(kd) / 1e6:7.3f} ms = {st.mean(kd) / STEPS:6.1f} cycles/step"
                f"  (min {min(kd) / STEPS:.1f} max {max(kd) / STEPS:.1f})"
            )
        print(line)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "revert":
        restore()
        os.remove(ORIG)
        print("reverted")
    elif cmd == "apply":
        apply(int(sys.argv[2]), legacy=(len(sys.argv) > 3 and sys.argv[3] == "legacy"))
    elif cmd == "parse":
        parse(sys.argv[2])
    else:
        print(__doc__)
