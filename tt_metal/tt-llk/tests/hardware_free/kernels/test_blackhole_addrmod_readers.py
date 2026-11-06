# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Hardware free: no LLK op may rely on an address mod that main's matmul never writes without writing it itself.

Main's matmul writes ADDR_MOD_0, 1, 2, 4 and 5 (and 6 under a static throttle); the head's forms also write 3 and 6 (rows form, half sync end fold, row MOP, row end fold).
An op whose mods all lie in {3, 6, 7} could skip its init after a matmul on main, which breaks with those forms. The set of such files is pinned below (every one of them
is an SFPU or custom kernel that sets the mod in its own init); a new file that reads only 3, 6 or 7 fails this test until it is added with the production kernels that run
it next to a matmul checked (addrmod_audit.md). usage: python3 test_blackhole_addrmod_readers.py (or pytest).
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "tt_llk_blackhole"
PINNED = {
    "common/inc/sfpu/experimental/ckernel_sfpu_deepseek_top32_rm.h",
    "common/inc/sfpu/experimental/ckernel_sfpu_generalized_moe_gate_topk_single_face.h",
    "llk_lib/llk_math_eltwise_ternary_sfpu.h",
}


def readers_outside_main_set():
    found = set()
    for f in (
        list(ROOT.glob("llk_lib/**/*.h"))
        + list(ROOT.glob("common/inc/sfpu/**/*.h"))
        + list(ROOT.glob("common/inc/*.h"))
    ):
        rel = str(f.relative_to(ROOT))
        if (
            "matmul" in rel and "custom" not in rel and "face_compressed" not in rel
        ) or rel.endswith("ckernel_addrmod.h"):
            continue
        used = set(re.findall(r"ADDR_MOD_([0-7])\b", f.read_text(errors="ignore")))
        if used and used <= {"3", "6", "7"} and used & {"3", "6"}:
            found.add(rel)
    return found


def test_addrmod_readers_pinned():
    found = readers_outside_main_set()
    assert (
        found == PINNED
    ), f"new: {sorted(found - PINNED)} gone: {sorted(PINNED - found)}"


if __name__ == "__main__":
    test_addrmod_readers_pinned()
    print("addr mod readers: %d files, as pinned" % len(PINNED))
