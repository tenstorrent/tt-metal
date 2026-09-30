# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Patched copies of the SDPA kernels for ablations (NEGATIVE_RESULTS §65): one tree per variant under <out>/<variant>/,
used as TT_METAL_KERNEL_PATH=<out>/<variant>.

Run the op from a directory with no ttnn/ tree under it: kernel lookup tries the cwd before TT_METAL_KERNEL_PATH, so
from the repo root every variant silently compiles the repo's kernels. Give each variant its own TT_METAL_CACHE.

    python sdpa_kernel_variants.py <out> [variant ...]      (default: all)

  base     unmodified copy (the control, compiled the same way as the others)
  conly    compute only: the reader / writer skip every NoC read and write, CB handshakes kept
  dmonly   data movement only: compute stubbed to the CB protocol (reuse_kv keep / pop included)
  dmread   dmonly with writes removed;  dmwrite: dmonly with reads removed
  zones    control with device-profiler zones on the compute kernel's phases, per work unit ("unit", "P1 QK+exp
           rowgroup", "P1 reduce max", "P2 exp rg1 + PV rg0", "P2 PV rg1", "P2 normalize"); ~9 zones per unit, so
           keep to ~13 units per core (the profiler holds 250 markers per RISC); bs8 on 12x8 is ~11
  zconly   zones + conly
  sumpack  the row sums on the pack thread (SDPA_SUM_ON_PACK, the path before §65)
  nosum    sumpack without the row-sum packs (wrong output; times their cost)
  noexp    sub_exp without the exp (wrong output; times its cost)
"""

import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
K = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels"

DM_STUB = r"""
#ifdef SDPA_ABL_DM_ONLY
        {
#ifdef REUSE_KV
            constexpr uint32_t grp = NQH / NKH;
#else
            constexpr uint32_t grp = 0;
#endif
            CircularBuffer cq(cb_q_in), ck(cb_k_in), cv(cb_v_in), co(cb_out);
            for (uint32_t q = 0; q < global_q_count; ++q) {
                const uint32_t ql = global_q_start + q;
                const bool keep = grp != 0 && q + 1 < global_q_count &&
                                  ql / (q_num_chunks * grp) == (ql + 1) / (q_num_chunks * grp);
                for (uint32_t kc = 0; kc < k_num_chunks; ++kc) {
                    const bool last = kc == k_num_chunks - 1;
                    ck.wait_front(k_chunk_tiles);
                    if (kc == 0) {
                        cq.wait_front(q_chunk_tiles);
                    }
                    cv.wait_front(v_chunk_tiles);
                    if (!(keep && last)) {
                        ck.pop_front(k_chunk_tiles);
                        cv.pop_front(v_chunk_tiles);
                    }
                }
                cq.pop_front(q_chunk_tiles);
                for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
                    co.reserve_back(vDHt);
                    co.push_back(vDHt);
                }
            }
            return;
        }
#endif
"""

# Zones with line-unique locals, so they can share a scope with the kernel's own MaybeDeviceZoneScopedN zones.
ZZ = """#define ZZ_CAT2(a, b) a##b
#define ZZ_CAT(a, b) ZZ_CAT2(a, b)
#define ZZ(name)                                                                                     \\
    DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                                                     \\
    auto constexpr ZZ_CAT(zz_hash_, __LINE__) = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); \\
    MaybeProfileScope<true, ZZ_CAT(zz_hash_, __LINE__)> ZZ_CAT(zz_zone_, __LINE__);
#else
#define ZZ(name)
"""


def rep(s, old, new, count=1):
    assert s.count(old) == count, (old[:80], s.count(old))
    return s.replace(old, new)


def patch_dm(d, read, write):
    p = d / "dataflow/dataflow_common.hpp"
    s = p.read_text()
    s, nr = re.subn(r"noc\.async_read\(reader,", "if (SDPA_ABL_READ) noc.async_read(reader,", s)
    s, nw = re.subn(r"noc\.async_write\(cb, out_writer,", "if (SDPA_ABL_WRITE) noc.async_write(cb, out_writer,", s)
    assert nr == 4 and nw == 2, (nr, nw)
    s = s.replace("#pragma once", f"#pragma once\n#define SDPA_ABL_READ {read}\n#define SDPA_ABL_WRITE {write}", 1)
    p.write_text(s)


def stub_compute(d):
    p = d / "compute/sdpa.cpp"
    s = p.read_text()
    anchor = "        cb_identity_scale_in_obj.wait_front(1);\n"
    s = rep(s, anchor, anchor + DM_STUB)
    s = s.replace("#include <cstdint>", "#include <cstdint>\n#define SDPA_ABL_DM_ONLY", 1)
    p.write_text(s)


def streaming(d, fn):
    p = d / "compute/compute_streaming.hpp"
    p.write_text(fn(p.read_text()))


def add_zones(s):
    s = rep(
        s, "    MaybeProfileScope<ENABLED, hash> zone;\n#else\n", "    MaybeProfileScope<ENABLED, hash> zone;\n" + ZZ
    )
    s = rep(
        s,
        "#define MaybeDeviceZoneScopedN(ENABLED, name)\n#elif defined(PROFILE_KERNEL)",
        "#define MaybeDeviceZoneScopedN(ENABLED, name)\n#define ZZ(name)\n#elif defined(PROFILE_KERNEL)",
    )
    for old, name in (
        ("Softmax(Q@KT)", "P1 QK+exp rowgroup"),
        ("Reduce max", "P1 reduce max"),
        ("ROW_NORM", "P2 normalize"),
    ):
        s = rep(s, f'MaybeDeviceZoneScopedN(profiling_enabled, "{old}");', f'ZZ("{name}")')
    a = "        // q_subblock 0: drain last row's sub_exp in-place + first QKT@V matmul\n        {\n"
    s = rep(s, a, a + '            ZZ("P2 exp rg1 + PV rg0")\n')
    a = "                // See the q_subblock-0 V matmul above: active_Sk"
    s = rep(s, a, '                ZZ("P2 PV rg1")\n' + a)
    a = "    for (uint32_t q = 0; q < q_chunks_per_core; q++) {\n        AccumulatorHalf prev = {cb_sum_A, cb_max_A, cb_out_im_A};"
    i = s.find("void sdpa_standard_v2(")
    j = s.find(a, i)
    assert j > 0
    return s[:j] + a + '\n        ZZ("unit")' + s[j + len(a) :]


def sum_on_pack(s):
    return s.replace("#pragma once", "#pragma once\n#define SDPA_SUM_ON_PACK", 1)


def sub_exp_body(s, fn):
    i = s.find("void sub_exp_block_bcast_cols(")
    j = s.find("\n}\n", i)
    return s[:i] + fn(s[i:j]) + s[j:]


def drop_sum(body):
    m = re.search(r"        if \(!skip_row_sum\) \{\n.*?\n(    tile_regs_release\(\);)", body, re.S)
    assert m and "reduce_cb, max_row_base + i" in m.group(0)
    return body[: m.start()] + "    }\n" + body[m.start(1) :]


def drop_exp(body):
    old = "exp_packthread_tile<true, false, InputClamping::None, iterations>(dst_index++, vector_mode_exp);"
    return rep(body, old, "dst_index++; (void)vector_mode_exp;")


VARIANTS = {
    "base": lambda d: None,
    "conly": lambda d: patch_dm(d, 0, 0),
    "dmonly": stub_compute,
    "dmread": lambda d: (stub_compute(d), patch_dm(d, 1, 0)),
    "dmwrite": lambda d: (stub_compute(d), patch_dm(d, 0, 1)),
    "zones": lambda d: streaming(d, add_zones),
    "zconly": lambda d: (streaming(d, add_zones), patch_dm(d, 0, 0)),
    "sumpack": lambda d: streaming(d, sum_on_pack),
    "nosum": lambda d: streaming(d, lambda s: sub_exp_body(sum_on_pack(s), drop_sum)),
    "noexp": lambda d: streaming(d, lambda s: sub_exp_body(s, drop_exp)),
}

if __name__ == "__main__":
    out = Path(sys.argv[1]).resolve()
    for v in sys.argv[2:] or VARIANTS:
        d = out / v / K
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(REPO / K, d)
        VARIANTS[v](d)
        print(f"{v}: {out / v}")
