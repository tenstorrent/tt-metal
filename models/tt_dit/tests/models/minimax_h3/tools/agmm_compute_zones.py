# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Temporary Tracy device zones for the AGMM compute kernel, and a parser for the resulting profile_log_device.csv.

    python models/tt_dit/tests/models/minimax_h3/tools/agmm_compute_zones.py apply block     # KLOOP + the epilogue per output block
    python models/tt_dit/tests/models/minimax_h3/tools/agmm_compute_zones.py apply sampled   # ACQ/MAC/PWAIT/PACK on one subblock of every 30th K iteration
    python models/tt_dit/tests/models/minimax_h3/tools/agmm_compute_zones.py apply opwait    # block zones + OPWAIT (in0/in1 wait_front) per K iteration, first ~100 iterations
    MM_SWEEP_EXPLICIT_COMBOS='[[8,7,10,2,2]]' MM_SWEEP_PROFILER_DUMP_EVERY=100000 \\
      python -m pytest models/tt_dit/utils/sweep_mm_block_sizes.py::test_mm_sweep \\
      -k "13664_5376_7168_8x8_agmm_ff1_swiglu and wh_4x8_ring" -s --timeout 7200
    python models/tt_dit/tests/models/minimax_h3/tools/agmm_compute_zones.py parse \\
      generated/profiler/mm_sweep_wh_4x8_ring_13664_5376_7168_8x8_agmm_ff1_swiglu/reports/<ts>/profile_log_device.csv
    python models/tt_dit/tests/models/minimax_h3/tools/agmm_compute_zones.py revert          # git checkout of the kernel

The compute kernel is shared by every AGMM of the block (to_qkv, to_out, ff1); `help` prints the sweep-harness `-k`
id and shipped blocking of each (from `minimax_h3_ops.py`). `apply block` zones the K loop (`KLOOP`) and whichever
epilogue the op compiles: `SWIGLU` (ff1), `EPILOGUE_ADDCMUL` (to_out), `EPILOGUE_COPY` (to_qkv as the model runs it,
bias=None) or `EPILOGUE_BIAS` (to_qkv through the sweep harness, which always passes a bias). `apply sampled` is
inside the K loop and epilogue-independent.

`apply` edits `all_gather_minimal_matmul_async/device/kernels/compute.cpp` in place with exact-match replacements
(it asserts if the kernel text has moved on); `revert` restores it from git (discarding ANY uncommitted edit to that
file). The zones are on purpose not in the tree.

Reading the sampled mode: the compute kernel is compiled once per TRISC, so the same zone is timed on all three
threads. ACQ on MATH = waiting for a free DST half; MAC on MATH = MAC issue, on UNPACK = unpack busy; PWAIT on PACK =
waiting for math to commit; PACK on PACK = pack busy. Nominal MAC time for a subblock is
subblock_h * subblock_w * K_block_tiles * 32 cycles at HiFi2 (16 LoFi, 64 HiFi4).
"""

from __future__ import annotations

import collections
import csv
import os
import subprocess
import sys

KERNEL = "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/kernels/compute.cpp"
INCLUDE = (
    '#include "api/dataflow/circular_buffer.h"\n',
    '#include "api/dataflow/circular_buffer.h"\n#include "tools/profiler/kernel_profiler.hpp"\n',
)

BLOCK_EDITS = [
    INCLUDE,
    (
        """            intermediate_cb.reserve_back(out_block_num_tiles);
            for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {""",
        """            intermediate_cb.reserve_back(out_block_num_tiles);
            {
            DeviceZoneScopedN("KLOOP");
            for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {""",
    ),
    (
        """                if (k_block == 0) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
            }

            intermediate_cb.push_back(out_block_num_tiles);""",
        """                if (k_block == 0) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
            }
            }

            intermediate_cb.push_back(out_block_num_tiles);""",
    ),
    (
        """            swiglu_block(
                intermediate_cb.get_cb_id(), in2_cb.get_cb_id(), out_cb.get_cb_id(), M_block_tiles, N_block_tiles);""",
        """            {
                DeviceZoneScopedN("SWIGLU");
                swiglu_block(
                    intermediate_cb.get_cb_id(), in2_cb.get_cb_id(), out_cb.get_cb_id(), M_block_tiles, N_block_tiles);
            }""",
    ),
    # The other epilogues (one is compiled per op: FUSE_SWIGLU -> ff1, FUSE_TERNARY -> to_out, neither -> to_qkv,
    # which is a plain copy when the op has no bias and add_bias_block when it has one -- the sweep harness always
    # passes a bias, the model and the mesh bench never do).
    (
        """            copy_block(intermediate_cb, out_cb, M_block_tiles, N_block_tiles);
""",
        """            {
                DeviceZoneScopedN("EPILOGUE_COPY");
                copy_block(intermediate_cb, out_cb, M_block_tiles, N_block_tiles);
            }
""",
    ),
    (
        """            add_bias_block(intermediate_cb, in2_cb, out_cb, M_block_tiles, N_block_tiles);
""",
        """            {
                DeviceZoneScopedN("EPILOGUE_BIAS");
                add_bias_block(intermediate_cb, in2_cb, out_cb, M_block_tiles, N_block_tiles);
            }
""",
    ),
    (
        """            add_bias_and_addcmul_block(
                intermediate_cb,
                in2_cb,
                ternary_a_cb,
                ternary_b_cb,
                fused_ternary_scalar_uint,
                out_cb,
                M_block_tiles,
                N_block_tiles,
                broadcast_ternary_b);
""",
        """            {
                DeviceZoneScopedN("EPILOGUE_ADDCMUL");
                add_bias_and_addcmul_block(
                    intermediate_cb,
                    in2_cb,
                    ternary_a_cb,
                    ternary_b_cb,
                    fused_ternary_scalar_uint,
                    out_cb,
                    M_block_tiles,
                    N_block_tiles,
                    broadcast_ternary_b);
            }
""",
    ),
]

# `apply opwait`: the block zones plus the operand wait at the top of every K iteration (`in0/in1 wait_front`). One
# zone per iteration overflows the ~120-event per-RISC profiler buffer, so only the first ~100 iterations of each
# core are recorded -- enough to see whether the loop ever waits on the relay.
OPWAIT_EDITS = BLOCK_EDITS + [
    (
        """                in0_cb.wait_front(in0_block_num_tiles);
                in1_cb.wait_front(in1_block_num_tiles);
""",
        """                {
                    DeviceZoneScopedN("OPWAIT");
                    in0_cb.wait_front(in0_block_num_tiles);
                    in1_cb.wait_front(in1_block_num_tiles);
                }
""",
    ),
]

SAMPLED_EDITS = [
    INCLUDE,
    (
        """// Slightly modified from compute_common.hpp
void matmul_blocks(""",
        """static uint32_t g_mm_iter = 0;  // matmul_blocks calls so far (== K iterations), for zone sampling

// Slightly modified from compute_common.hpp
void matmul_blocks(""",
    ),
    (
        """    uint32_t in0_index_offset = 0;

    for (uint32_t M_start = 0; M_start < M_block_tiles; M_start += subblock_h) {
        uint32_t in1_index_offset = 0;
        for (uint32_t N_start = 0; N_start < N_block_tiles; N_start += subblock_w) {
            tile_regs_acquire();

            uint32_t dst_index = 0;
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;

            for (uint32_t inner_dim = 0; inner_dim < K_block_tiles; inner_dim++) {
                matmul_block(
                    in0_cb.get_cb_id(),
                    in1_cb.get_cb_id(),
                    in0_index,
                    in1_index,
                    dst_index,
                    false /*transpose*/,
                    subblock_w,
                    subblock_h,
                    K_block_tiles);
                in0_index++;
                in1_index += full_N_block_tiles;
            }
            tile_regs_commit();
            tile_regs_wait();
            uint32_t write_dst_index = 0;
            for (uint32_t h = 0; h < subblock_h; h++) {
                uint32_t h_tile_id = M_start + h;
                for (uint32_t w = 0; w < subblock_w; w++) {
                    uint32_t w_tile_id = N_start + w;
                    uint32_t out_tile_id = h_tile_id * full_N_block_tiles + w_tile_id;
                    pack_tile<true>(write_dst_index, out_cb.get_cb_id(), out_tile_id);
                    write_dst_index++;
                    dst_index++;
                }
            }
            tile_regs_release();
""",
        """    uint32_t in0_index_offset = 0;
    const bool sample_iter = (g_mm_iter % 30) == 0;
    g_mm_iter++;

    auto mac = [&](uint32_t in0_index, uint32_t in1_index) {
        uint32_t dst_index = 0;
        for (uint32_t inner_dim = 0; inner_dim < K_block_tiles; inner_dim++) {
            matmul_block(
                in0_cb.get_cb_id(),
                in1_cb.get_cb_id(),
                in0_index,
                in1_index,
                dst_index,
                false /*transpose*/,
                subblock_w,
                subblock_h,
                K_block_tiles);
            in0_index++;
            in1_index += full_N_block_tiles;
        }
    };
    auto pack = [&](uint32_t M_start, uint32_t N_start) {
        uint32_t write_dst_index = 0;
        for (uint32_t h = 0; h < subblock_h; h++) {
            uint32_t h_tile_id = M_start + h;
            for (uint32_t w = 0; w < subblock_w; w++) {
                uint32_t w_tile_id = N_start + w;
                uint32_t out_tile_id = h_tile_id * full_N_block_tiles + w_tile_id;
                pack_tile<true>(write_dst_index, out_cb.get_cb_id(), out_tile_id);
                write_dst_index++;
            }
        }
    };

    for (uint32_t M_start = 0; M_start < M_block_tiles; M_start += subblock_h) {
        uint32_t in1_index_offset = 0;
        for (uint32_t N_start = 0; N_start < N_block_tiles; N_start += subblock_w) {
            const bool sample = sample_iter && (M_start == 0) && (N_start == subblock_w);
            if (sample) {
                {
                    DeviceZoneScopedN("ACQ");
                    tile_regs_acquire();
                }
                {
                    DeviceZoneScopedN("MAC");
                    mac(in0_index_offset, in1_index_offset);
                }
                tile_regs_commit();
                {
                    DeviceZoneScopedN("PWAIT");
                    tile_regs_wait();
                }
                {
                    DeviceZoneScopedN("PACK");
                    pack(M_start, N_start);
                }
                tile_regs_release();
            } else {
                tile_regs_acquire();
                mac(in0_index_offset, in1_index_offset);
                tile_regs_commit();
                tile_regs_wait();
                pack(M_start, N_start);
                tile_regs_release();
            }
""",
    ),
]


def apply(edits) -> None:
    s = open(KERNEL).read()
    for old, new in edits:
        assert s.count(old) == 1, f"kernel text changed; expected exactly one match for:\n{old[:120]}"
        s = s.replace(old, new)
    open(KERNEL, "w").write(s)
    print(f"zones applied to {KERNEL}")


def parse(path: str) -> None:
    rows = list(csv.DictReader(open(path).read().split("\n", 1)[1].splitlines()))
    last = collections.defaultdict(int)  # largest run host ID per PCIe slot = the warm run
    for r in rows:
        last[r["PCIe slot"]] = max(last[r["PCIe slot"]], int(r[" run host ID"]))
    agg = collections.defaultdict(list)
    open_start = {}
    cores = collections.defaultdict(
        set
    )  # (risc, zone) -> cores that recorded it (mux-row BRISCs never see compute zones)
    for r in rows:
        if int(r[" run host ID"]) != last[r["PCIe slot"]]:
            continue
        risc, zone, typ = r[" RISC processor type"].strip(), r[" zone name"].strip(), r[" type"].strip()
        key = (r["PCIe slot"], r[" core_x"], r[" core_y"], risc, zone)
        cores[(risc, zone)].add(key[:3])
        t = int(r[" time[cycles since reset]"])
        if typ == "ZONE_START":
            open_start[key] = t
        elif typ == "ZONE_END" and key in open_start:
            agg[(risc, zone)].append(t - open_start.pop(key))
    names = {"TRISC_0": "UNPACK", "TRISC_1": "MATH", "TRISC_2": "PACK", "BRISC": "BRISC", "NCRISC": "NCRISC"}
    print("cycles at 1 GHz = ns; sum/core averages over the cores that recorded the zone")
    print(
        f"{'thread':7s} {'zone':14s} {'cores':>5s} {'n':>6s} {'mean cyc':>9s} {'min':>8s} {'max':>8s} {'sum/core us':>12s}"
    )
    for (risc, zone), d in sorted(agg.items()):
        n_cores = max(len(cores[(risc, zone)]), 1)
        print(
            f"{names.get(risc, risc):7s} {zone:14s} {n_cores:5d} {len(d):6d} {sum(d) / len(d):9.0f} {min(d):8d} {max(d):8d} "
            f"{sum(d) / n_cores / 1000:12.0f}"
        )


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "apply":
        apply({"block": BLOCK_EDITS, "sampled": SAMPLED_EDITS, "opwait": OPWAIT_EDITS}[sys.argv[2]])
    elif cmd == "revert":
        subprocess.check_call(["git", "checkout", "--", KERNEL])
        print(f"reverted {KERNEL}")
    elif cmd == "parse":
        parse(sys.argv[2])
    else:
        print(__doc__)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from minimax_h3_ops import AGMM_OPS

        print('sweep-harness ids of the block\'s AGMMs at 15 s / 768P / 16:9 (use with -k "<id> and wh_4x8_ring"):')
        for spec in AGMM_OPS:
            print(
                f"  {spec.name:7s} {spec.sweep_id():45s} shipped blocking {spec.blocks_str()}   epilogue {spec.fusion}"
            )


if __name__ == "__main__":
    main()
