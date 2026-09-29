#!/usr/bin/env python3
"""Shard-arithmetic self-test for galaxy_shard.sh + galaxy_combine.py.

Pure arithmetic and process plumbing: NO device, NO pytest harness, NO ELF
toolchain.  It runs the REAL galaxy_shard.sh against a stub streamer and a stub
`elf_text_sha.py`, captures the exact `--start-bit` / `--total` / `--chip`
arguments every slice would receive, and asserts:

  * the union of the NPAR slice intervals tiles [0, SPACE) exactly -- no gap,
    no overlap, no duplicate chip;
  * every slice's own band enumeration (the same `min(band, start+total-s)`
    recurrence the streamers use) covers its interval exactly;
  * degenerate geometries are REFUSED rather than trivially "covered";
  * a slice that exits non-zero is never counted as covered, including when a
    stale slice verdict from an earlier run is sitting in its directory.

Run: python3 selftest_galaxy_shard.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARD = HERE / "galaxy_shard.sh"

sys.path.insert(0, str(HERE))
import galaxy_combine  # noqa: E402

TWO32 = 1 << 32

STUB_ELF_SHA = """\
import sys, hashlib
print(hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest())
"""

# Stub streamer: records its argv, reproduces the streamers' band recurrence,
# and writes the same VERDICT line shape they write.
STUB_STREAMER = '''\
import argparse, json, os, pathlib, sys

ap = argparse.ArgumentParser()
for a in ("--op", "--sem-node", "--hand-node", "--farm", "--venv", "--llk-home",
          "--runner-temp", "--band-bits", "--chip", "--out", "--start-bit",
          "--total", "--idmap", "--idmap-source", "--golden", "--tile-dim"):
    ap.add_argument(a)
ns = ap.parse_args()
with open(os.environ["SHARD_RECORD"], "a") as fh:
    fh.write(json.dumps(vars(ns)) + "\\n")

if ns.chip in os.environ.get("DEAD_CHIPS", "").split(","):
    sys.exit(3)

out = pathlib.Path(ns.out)
out.mkdir(parents=True, exist_ok=True)
start, total = int(ns.start_bit, 0), int(ns.total, 0)
band = 1 << int(ns.band_bits)
n_bands = (total + band - 1) // band
covered = 0
for k in range(n_bands):
    s = start + k * band
    covered += min(band, start + total - s)
assert covered == total, "stub band recurrence lost coverage"
(out / (ns.op + "-VERDICT.txt")).write_text(
    "OP=%s VERDICT=BIT-EXACT-ALL-INPUTS start=%d total=%d bands=%d covered=%d "
    "(full 2^32=%s) wall_s=0.0 witness_bands=[]\\n"
    % (ns.op, start, total, n_bands, covered, covered == (1 << 32))
)
if ns.golden:
    (out / (ns.op + "-CORRECTNESS-VERDICT.txt")).write_text(
        "OP=%s NUMERIC_GATE=PASS ULP_ADMISSION=PASS\\n" % ns.op
    )
'''

PYSHIM = '#!/bin/sh\nexec "%s" "$@"\n' % sys.executable


class Farm:
    """A throwaway FARM_ROOT holding stub tools and two distinct fake ELFs."""

    def __init__(self, root: Path, sweep: str):
        self.root = root
        tools = root / "farm/tests/corpus/tools"
        tools.mkdir(parents=True)
        (root / "farm/tests/python_tests").mkdir(parents=True)
        shutil.copy(HERE / "galaxy_combine.py", tools / "galaxy_combine.py")
        (tools / "elf_text_sha.py").write_text(STUB_ELF_SHA)
        for name in ("binary_stream_sweep.py", "fp32_stream_sweep.py"):
            (tools / name).write_text(STUB_STREAMER)
        src = "sfpu_binary_test.cpp" if sweep == "binary" else "eltwise_unary_sfpu_test.cpp"
        for variant, body in (("AAA", b"sem-text\n"), ("BBB", b"hand-text\n")):
            d = root / "farm/build/tt-llk-build/sources" / src / variant / "elf"
            d.mkdir(parents=True)
            (d / "math.elf").write_bytes(body)
        self.venv = root / "pyshim"
        self.venv.write_text(PYSHIM)
        self.venv.chmod(0o755)
        self.farm_root = root / "farm"


def run_shard(tmp: Path, tag: str, sweep: str = "binary", dead: str = "", **env):
    """Run the real galaxy_shard.sh; return (rc, last_line, [slice argv dicts])."""
    work = tmp / tag
    work.mkdir()
    farm = Farm(work, sweep)
    out = work / "ev"
    out.mkdir()
    record = work / "calls.jsonl"
    record.write_text("")
    e = dict(os.environ)
    e.update(
        SHARD_RECORD=str(record),
        DEAD_CHIPS=dead,
        OP="myop",
        SWEEP=sweep,
        STAGGER="0",
        SEM="sem-node",
        HAND="hand-node",
        SEM_VARIANT="AAA",
        HAND_VARIANT="BBB",
        FARM_ROOT=str(farm.farm_root),
        VENV=str(farm.venv),
        OUT=str(out),
    )
    e.update({k: str(v) for k, v in env.items()})
    run = subprocess.run(
        ["bash", str(SHARD)], env=e, capture_output=True, text=True, timeout=600
    )
    lines = [ln for ln in (run.stdout + run.stderr).strip().splitlines() if ln.strip()]
    calls = [json.loads(ln) for ln in record.read_text().splitlines() if ln.strip()]
    return run.returncode, (lines[-1] if lines else ""), calls, out


def assert_exact_cover(calls, space: int, npar: int) -> None:
    assert len(calls) == npar, f"expected {npar} slices, got {len(calls)}"
    chips = sorted(int(c["chip"]) for c in calls)
    assert chips == list(range(npar)), f"chip ids not a 0..{npar-1} permutation: {chips}"
    intervals = sorted(
        (int(c["start_bit"], 0), int(c["total"], 0)) for c in calls
    )
    cursor = 0
    for start, total in intervals:
        assert total > 0, f"empty slice at {start}"
        assert start == cursor, (
            f"coverage break: slice starts at {start}, previous ended at {cursor} "
            f"({'gap' if start > cursor else 'overlap'} of {abs(start - cursor)})"
        )
        cursor = start + total
    assert cursor == space, f"union ends at {cursor}, expected {space}"


def test_exact_partition(tmp: Path) -> None:
    """The 32-way shard of 2^32 tiles the space exactly, in both sweeps."""
    for sweep, band_bits in (("binary", 23), ("fp32", 23)):
        rc, last, calls, _ = run_shard(
            tmp, f"part-{sweep}", sweep=sweep, NPAR=32, BAND_BITS=band_bits, SPACE=TWO32
        )
        assert rc == 0, f"{sweep}: rc={rc} :: {last}"
        assert_exact_cover(calls, TWO32, 32)
        assert "VERDICT=BIT-EXACT-ALL-INPUTS" in last, last
        assert f"covered={TWO32}" in last, last
    print("PASS 32-way shard of 2^32 tiles the space exactly (binary + fp32)")


def test_partition_matrix(tmp: Path) -> None:
    """Every divisible geometry partitions; the band recurrence never loses inputs."""
    cases = [
        (TWO32, 32, 23),
        (TWO32, 32, 27),  # band == slice
        (TWO32, 32, 30),  # band > slice: the min() clamp must still cover
        (TWO32, 16, 24),
        (TWO32, 1, 28),
        (1 << 20, 32, 10),
        (1 << 20, 32, 3),  # many small bands
        (96, 32, 1),  # slice == 3, band == 2: ragged last band
    ]
    for i, (space, npar, band_bits) in enumerate(cases):
        rc, last, calls, _ = run_shard(
            tmp, f"matrix-{i}", NPAR=npar, BAND_BITS=band_bits, SPACE=space
        )
        assert rc == 0, f"space={space} npar={npar} bb={band_bits}: rc={rc} :: {last}"
        assert_exact_cover(calls, space, npar)
    print(f"PASS {len(cases)} shard geometries partition exactly (incl. ragged bands)")


def test_degenerate_geometries_refuse(tmp: Path) -> None:
    """A geometry that cannot cover anything must refuse, never 'succeed'."""
    cases = [
        ("zero-space", dict(SPACE=0, NPAR=32, BAND_BITS=23)),
        ("space-below-npar", dict(SPACE=16, NPAR=32, BAND_BITS=4)),
        ("non-divisible", dict(SPACE=100, NPAR=32, BAND_BITS=4)),
        ("non-numeric-space", dict(SPACE="abc", NPAR=32, BAND_BITS=23)),
        ("negative-npar", dict(SPACE=TWO32, NPAR=-4, BAND_BITS=23)),
        ("zero-npar", dict(SPACE=TWO32, NPAR=0, BAND_BITS=23)),
        ("non-numeric-npar", dict(SPACE=TWO32, NPAR="32x", BAND_BITS=23)),
        ("zero-band-bits", dict(SPACE=TWO32, NPAR=32, BAND_BITS=0)),
        ("band-bits-too-wide", dict(SPACE=TWO32, NPAR=32, BAND_BITS=64)),
        ("bad-golden", dict(SPACE=TWO32, NPAR=32, BAND_BITS=23, GOLDEN=2)),
        ("bad-sweep", dict(SPACE=TWO32, NPAR=32, BAND_BITS=23, SWEEP_OVERRIDE=1)),
    ]
    for tag, env in cases:
        sweep = "binary"
        if env.pop("SWEEP_OVERRIDE", None):
            sweep = "quantum"
        rc, last, calls, out = run_shard(tmp, f"deg-{tag}", sweep=sweep, **env)
        assert rc != 0, f"{tag}: expected refusal, got rc=0 :: {last}"
        assert not calls, f"{tag}: refused geometry still launched {len(calls)} slices"
        assert "BIT-EXACT-ALL-INPUTS" not in last, f"{tag}: refusal claimed a proof :: {last}"
        verdict = out / "myop-VERDICT.txt"
        if verdict.exists():
            assert "BIT-EXACT-ALL-INPUTS" not in verdict.read_text(), tag
    print(f"PASS {len(cases)} degenerate geometries refuse without claiming coverage")


def test_dead_slice_is_not_covered(tmp: Path) -> None:
    """A slice whose process dies must never be counted as covered."""
    rc, last, calls, out = run_shard(
        tmp, "dead", dead="7", NPAR=32, BAND_BITS=23, SPACE=TWO32
    )
    assert rc != 0, f"dead slice reported success :: {last}"
    assert "BIT-EXACT-ALL-INPUTS" not in last, last
    assert "invalid=[7]" in last, last
    print("PASS a dead slice is named invalid, not covered")


def test_stale_verdict_cannot_stand_in_for_a_dead_slice(tmp: Path) -> None:
    """The regression that matters: a stale slice verdict + a dead chip.

    An earlier run's slice-7 verdict has the right op, start, total and covered
    for this geometry.  If the driver leaves it in place and the chip then dies,
    the combiner sees a full, well-formed cover and only one boolean stands
    between that and a certified proof.
    """
    work = tmp / "stale"
    work.mkdir()
    farm = Farm(work, "binary")
    out = work / "ev"
    (out / "slice-7").mkdir(parents=True)
    slice_size = TWO32 // 32
    (out / "slice-7/myop-VERDICT.txt").write_text(
        "OP=myop VERDICT=BIT-EXACT-ALL-INPUTS start=%d total=%d bands=16 "
        "covered=%d (full 2^32=False) wall_s=1.0 witness_bands=[]\n"
        % (7 * slice_size, slice_size, slice_size)
    )
    (out / "slice-7/myop-CORRECTNESS-VERDICT.txt").write_text(
        "OP=myop NUMERIC_GATE=PASS ULP_ADMISSION=PASS\n"
    )
    record = work / "calls.jsonl"
    record.write_text("")
    e = dict(os.environ)
    e.update(
        SHARD_RECORD=str(record), DEAD_CHIPS="7", OP="myop", SWEEP="binary",
        STAGGER="0", SEM="sem-node", HAND="hand-node", SEM_VARIANT="AAA",
        HAND_VARIANT="BBB", FARM_ROOT=str(farm.farm_root), VENV=str(farm.venv),
        OUT=str(out), NPAR="32", BAND_BITS="23", SPACE=str(TWO32),
    )
    run = subprocess.run(
        ["bash", str(SHARD)], env=e, capture_output=True, text=True, timeout=600
    )
    last = (run.stdout + run.stderr).strip().splitlines()[-1]
    assert run.returncode != 0, f"stale verdict laundered a dead slice :: {last}"
    assert "BIT-EXACT-ALL-INPUTS" not in last, last
    assert "invalid=[7]" in last, last
    assert not (out / "slice-7/myop-VERDICT.txt").exists(), (
        "the stale slice verdict survived the run"
    )
    print("PASS a stale slice verdict cannot stand in for a dead slice")


def test_combiner_refuses_empty_space() -> None:
    """Directly: zero-sized slices must not combine into a proof."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for chip in range(4):
            d = out / f"slice-{chip}"
            d.mkdir()
            (d / "op-VERDICT.txt").write_text(
                "OP=op VERDICT=BIT-EXACT-ALL-INPUTS start=0 total=0 covered=0 "
                "witness_bands=[]\n"
            )
            (d / "op-CORRECTNESS-VERDICT.txt").write_text(
                "OP=op NUMERIC_GATE=PASS ULP_ADMISSION=PASS\n"
            )
        summary, passed = galaxy_combine.combine(out, 4, 0, "op", True, False)
        assert not passed, summary
        assert "VERDICT=INCOMPLETE" in summary, summary
    print("PASS combiner refuses a zero-sized input space")


def test_combiner_failed_chips_override_present_verdicts() -> None:
    """A chip named in --failed-chips is invalid whatever its verdict file says."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for chip in range(2):
            d = out / f"slice-{chip}"
            d.mkdir()
            (d / "op-VERDICT.txt").write_text(
                "OP=op VERDICT=BIT-EXACT-ALL-INPUTS start=%d total=5 covered=5 "
                "witness_bands=[]\n" % (chip * 5)
            )
            (d / "op-CORRECTNESS-VERDICT.txt").write_text(
                "OP=op NUMERIC_GATE=PASS ULP_ADMISSION=PASS\n"
            )
        summary, passed = galaxy_combine.combine(out, 2, 10, "op", True, False)
        assert passed, summary
        summary, passed = galaxy_combine.combine(out, 2, 10, "op", True, False, {1})
        assert not passed and "invalid=[1]" in summary, summary
    print("PASS combiner honours --failed-chips over a present verdict file")


def main() -> int:
    if not SHARD.is_file():
        print(f"FAIL: no galaxy_shard.sh at {SHARD}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        test_exact_partition(root)
        test_partition_matrix(root)
        test_degenerate_geometries_refuse(root)
        test_dead_slice_is_not_covered(root)
        test_stale_verdict_cannot_stand_in_for_a_dead_slice(root)
    test_combiner_refuses_empty_space()
    test_combiner_failed_chips_override_present_verdicts()
    print("\nSELFTEST galaxy shard arithmetic: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
