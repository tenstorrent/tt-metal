#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diff two SFPU_EQUIV dumps bit-for-bit. Exit non-zero if any output bit differs.

Usage: compare_vif_equiv.py <dump_dir> <a_tag> <b_tag>

Dumps come from test_vif_equiv_sweep.py run with SFPU_EQUIV_OUT set; see that
module's docstring for the baseline-vs-candidate workflow. The two runs must
have wiped the ELF cache in between, which is what the elf-fingerprint check
below enforces.
"""
import glob
import os
import sys

import numpy as np


def _inventory(base_dir, tag):
    return {
        os.path.basename(p).split("__", 1)[1][:-4]
        for p in glob.glob(f"{base_dir}/{tag}__*.npz")
    }


def main(argv):
    if len(argv) != 4:
        print(__doc__.strip())
        return 2
    base_dir, a_tag, b_tag = argv[1], argv[2], argv[3]

    if a_tag == b_tag:
        print(f"a_tag and b_tag are both {a_tag!r}: that compares a run with itself")
        return 2

    inv_a, inv_b = _inventory(base_dir, a_tag), _inventory(base_dir, b_tag)
    if not inv_a:
        print(f"no dumps for tag {a_tag} in {base_dir}")
        return 2
    if not inv_b:
        print(f"no dumps for tag {b_tag} in {base_dir}")
        return 2

    # Both inventories, not just a_tag's: a case captured only under b_tag would
    # otherwise be skipped silently and a partial baseline run would still report
    # every case identical.
    only_a, only_b = sorted(inv_a - inv_b), sorted(inv_b - inv_a)
    if only_a or only_b:
        print(f"dump sets differ between {a_tag} and {b_tag}:")
        for n in only_a:
            print(f"  only in {a_tag}: {n}")
        for n in only_b:
            print(f"  only in {b_tag}: {n}")
        print("re-run both captures over the same case set before comparing")
        return 2

    names = sorted(inv_a)
    bad = 0
    shared_elf = []
    print(f"{'case':44s} {'points':>8s}  verdict")
    print("-" * 72)
    for n in names:
        A = np.load(f"{base_dir}/{a_tag}__{n}.npz")
        B = np.load(f"{base_dir}/{b_tag}__{n}.npz")

        # A stale $RUNNER_TEMP/tt-llk-build makes the second run replay the first
        # run's binaries: variant_id hashes include paths, not header content, so
        # the two tags would compare identical no matter what changed in between.
        # The stimuli match trivially in that case and cannot catch it.
        ea, eb = A.get("elf"), B.get("elf")
        if ea is not None and eb is not None and str(ea) == str(eb):
            shared_elf.append(n)

        if not np.array_equal(A["x"], B["x"]):
            print(f"{n:44s} {'-':>8s}  STIMULUS MISMATCH")
            bad += 1
            continue
        ya, yb = A["y"], B["y"]
        if ya.shape != yb.shape:
            print(f"{n:44s} {'-':>8s}  SHAPE {ya.shape} vs {yb.shape}")
            bad += 1
            continue
        d = ya != yb
        ndiff = int(d.sum())
        if ndiff:
            idx = np.flatnonzero(d)[:5]
            ex = ", ".join(f"x={A['x'][i]:#x} {ya[i]:#x}->{yb[i]:#x}" for i in idx)
            print(f"{n:44s} {ya.size:8d}  DIFF {ndiff} ({ex})")
            bad += 1
        else:
            print(f"{n:44s} {ya.size:8d}  identical")
    print("-" * 72)

    if shared_elf:
        print(
            f"{len(shared_elf)}/{len(names)} cases ran the SAME ELF under both tags "
            f"(e.g. {shared_elf[0]}) -- the build cache was not wiped between runs, "
            f"so this comparison proves nothing. Remove $RUNNER_TEMP/tt-llk-build "
            f"(or /tmp/tt-llk-build) and re-capture."
        )
        return 2

    print(f"{len(names)-bad}/{len(names)} cases bit-identical")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
