# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Hardware free: no LLK op may rely on an address mod that main's matmul never writes without writing it itself.

Main's matmul writes ADDR_MOD_0, 1, 2, 4 and 5 in its init (6 only inside a throttled or row MOP call, restored to dest += 2); the head's forms also write 3 in the
init (rows form, half sync end fold, row MOP) and 6 inside the call (row MOP, row end fold). An op whose mods all lie in {3, 6, 7} could skip its init after a matmul on main,
which breaks with those forms where it reads 3. The set of such files is pinned below (an SFPU or custom kernel that sets 3 in its own init or reads 6 / 7 at their
hw_configure invariants); a new file that reads only 3, 6 or 7 fails this test until it is added with the production kernels that run it next to a matmul checked
(addrmod_audit.md). usage: python3 test_blackhole_addrmod_readers.py (or pytest).
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "tt_llk_blackhole"
PINNED = {
    "common/inc/sfpu/experimental/ckernel_sfpu_deepseek_top32_rm.h",
    "common/inc/sfpu/experimental/ckernel_sfpu_generalized_moe_gate_topk_single_face.h",
    "common/inc/sfpu/experimental/ckernel_sfpu_topk_pack_indices.h",
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
