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
  * a reduced SPACE reports BIT-EXACT-PARTIAL, never ALL-INPUTS -- tiling the
    range you asked for is not exhausting the op's space;
  * a slice that DIED is never counted as covered, including when a stale slice
    verdict from an earlier run is sitting in its directory -- while a slice that
    exited non-zero because its comparison DIVERGED is reported as a divergence
    rather than as a dead chip.

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
          "--total", "--idmap", "--idmap-source", "--golden", "--tile-dim",
          "--selected-sem-node", "--baseline-sem-node", "--baseline-hand-node",
          "--selected-flags", "--baseline-flags", "--selected-runner-temp",
          "--baseline-sem-runner-temp", "--baseline-hand-runner-temp"):
    ap.add_argument(a)
ns = ap.parse_args()
with open(os.environ["SHARD_RECORD"], "a") as fh:
    record = vars(ns)
    record["compiler_options"] = os.environ.get("TT_LLK_EXTRA_COMPILER_OPTIONS", "")
    fh.write(json.dumps(record) + "\\n")

if ns.chip in os.environ.get("DEAD_CHIPS", "").split(","):
    sys.exit(3)

# A diverging slice writes its verdict and THEN exits non-zero, exactly as the
# real streamers do (`return 0 if all_equal and numeric_ok else 1`).
diverge = ns.chip in os.environ.get("DIVERGENT_CHIPS", "").split(",")
compiler_diverge = ns.chip in os.environ.get("COMPILER_DIVERGENT_CHIPS", "").split(",")
tri = ns.selected_sem_node is not None

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
# Label exactly as the real streamers do: a slice covers SPACE/NPAR, so in a
# real galaxy run every slice says PARTIAL and only the combiner may say
# ALL-INPUTS.  A stub that always said ALL-INPUTS could not catch a combiner
# that accepted the wrong label.
verdict = (
    "BIT-EXACT-ALL-INPUTS"
    if covered == (1 << 32)
    else "BIT-EXACT-PARTIAL-%d-OF-2^32" % covered
)
witness = "[]"
if diverge:
    verdict, witness = "DIVERGENT", "[(0, %d, %d)]" % (start, min(band, total))
(out / (ns.op + "-VERDICT.txt")).write_text(
    "OP=%s VERDICT=%s start=%d total=%d bands=%d covered=%d "
    "(full 2^32=%s) wall_s=0.0 witness_bands=%s\\n"
    % (ns.op, verdict, start, total, n_bands, covered, covered == (1 << 32), witness)
)
if tri:
    compiler_verdict = (
        "DIVERGENT" if compiler_diverge else
        ("BIT-EXACT-ALL-INPUTS" if covered == (1 << 32)
         else "BIT-EXACT-PARTIAL-%d-OF-2^32" % covered)
    )
    (out / (ns.op + "-COMPILER-VERDICT.txt")).write_text(
        "OP=%s VERDICT=%s start=%d total=%d bands=%d covered=%d "
        "witness_bands=%s\\n"
        % (ns.op, compiler_verdict, start, total, n_bands, covered,
           "[(0, %d, %d)]" % (start, total) if compiler_diverge else "[]")
    )
if ns.golden:
    (out / (ns.op + "-CORRECTNESS-VERDICT.txt")).write_text(
        "OP=%s LOCAL_SEM_ABSOLUTE=PASS LOCAL_HAND_ORACLE_COMPLETE=PASS "
        "LOCAL_HAND_ABSOLUTE=PASS "
        "LOCAL_ULP_COMPARISON=PASS CAMPAIGN_ADMISSION=DEFERRED_GLOBAL\\n" % ns.op
    )
    bands = out / "bands"
    bands.mkdir()
    classes = "in_domain_finite_normal:%d:0" % total
    for leg in ("sem", "hand"):
        (bands / ("b%d-%s.txt.corr" % (start, leg))).write_text(
            "SFPU_CORRECTNESS,op=%s,leg=%s,patterns=%d,n_out_of_tol=0,"
            "max_bf16_ulp=0,within_contract=True,class_ulp=%s\\n"
            % (ns.op, leg, total, classes)
        )
if diverge or compiler_diverge:
    sys.exit(1)
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
        shutil.copy(HERE / "galaxy_numeric_admission.py", tools / "galaxy_numeric_admission.py")
        shutil.copy(HERE / "ulp_admission.py", tools / "ulp_admission.py")
        (tools / "elf_text_sha.py").write_text(STUB_ELF_SHA)
        for name in ("binary_stream_sweep.py", "fp32_stream_sweep.py"):
            (tools / name).write_text(STUB_STREAMER)
        src = "sfpu_binary_test.cpp" if sweep == "binary" else "eltwise_unary_sfpu_test.cpp"
        for variant, body in (("AAA", b"selected-text\n"), ("BBB", b"baseline-sem-text\n"),
                              ("CCC", b"baseline-hand-text\n")):
            d = root / "farm/build/tt-llk-build/sources" / src / variant / "elf"
            d.mkdir(parents=True)
            (d / "math.elf").write_bytes(body)
        self.venv = root / "pyshim"
        self.venv.write_text(PYSHIM)
        self.venv.chmod(0o755)
        self.farm_root = root / "farm"


def run_shard(
    tmp: Path, tag: str, sweep: str = "binary", dead: str = "", diverge: str = "",
    compiler_diverge: str = "", tri: bool = False, **env
):
    """Run the real galaxy_shard.sh; return (rc, last_line, [slice argv dicts])."""
    work = tmp / tag
    work.mkdir()
    farm = Farm(work, sweep)
    out = work / "ev"
    out.mkdir()
    record = work / "calls.jsonl"
    record.write_text("")
    flags_value = env.pop("FLAGS_VALUE", None)
    tri_selected_flags = env.pop("TRI_SELECTED_FLAGS", "-selected")
    flags_tsv = work / "flags.tsv"
    if flags_value is not None:
        flags_tsv.write_text(f"myop\t{flags_value}\n")
    e = dict(os.environ)
    e.update(
        SHARD_RECORD=str(record),
        DEAD_CHIPS=dead,
        DIVERGENT_CHIPS=diverge,
        COMPILER_DIVERGENT_CHIPS=compiler_diverge,
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
    if tri:
        profiles = work / "tri-profiles.tsv"
        profiles.write_text(
            "op\tcategory\tfull_space\tstate\ta_selected_sem_node\tselected_flags\t"
            "b_baseline_sem_node\tbaseline_flags\tc_baseline_hand_node\tc_baseline_flags\n"
            "myop\tunary_32_exhaustive\t4294967296\tGAP\tsemantic-node\t"
            f"{tri_selected_flags}\tsemantic-node\t-baseline\thand-node\t-baseline\n"
        )
        import hashlib
        src = "sfpu_binary_test.cpp" if sweep == "binary" else "eltwise_unary_sfpu_test.cpp"
        tri_build = farm.farm_root / "build/tri-arms/myop"
        collision_variant = "c" * 64
        for arm, variant, body in (
            ("a", collision_variant, b"selected-same-variant\n"),
            ("b", collision_variant, b"baseline-same-variant\n"),
            ("c", "d" * 64, b"baseline-hand\n"),
        ):
            elf = tri_build / arm / "tt-llk-build/sources" / src / variant / "elf/math.elf"
            elf.parent.mkdir(parents=True)
            elf.write_bytes(body)
        def digest(arm, variant):
            path = tri_build / arm / "tt-llk-build/sources" / src / variant / "elf/math.elf"
            return hashlib.sha256(path.read_bytes()).hexdigest()
        identity = work / "tri-idmap.tsv"
        identity.write_text(
            f"myop\t{collision_variant}\t{digest('a', collision_variant)}\t"
            f"{collision_variant}\t{digest('b', collision_variant)}\t"
            f"{'d' * 64}\t{digest('c', 'd' * 64)}\n"
        )
        e["TRI_PROFILES"] = str(profiles)
        e["TRI_IDMAP"] = str(identity)
        e.pop("SEM", None); e.pop("HAND", None)
        e.pop("SEM_VARIANT", None); e.pop("HAND_VARIANT", None)
    if flags_value is not None:
        e["FLAGS_TSV"] = str(flags_tsv)
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
        if space == TWO32:
            assert rc == 0, f"space={space} npar={npar} bb={band_bits}: rc={rc} :: {last}"
        else:
            assert rc != 0, "partial golden evidence was admitted as a campaign"
        assert_exact_cover(calls, space, npar)
        # Tiling the requested space must never be reported as exhausting 2^32.
        expect = (
            "BIT-EXACT-ALL-INPUTS"
            if space == TWO32
            else f"BIT-EXACT-PARTIAL-{space}-OF-2^32"
        )
        assert f"VERDICT={expect}" in last, f"space={space}: expected {expect} :: {last}"
    print(f"PASS {len(cases)} shard geometries partition exactly (incl. ragged bands)")


def test_reduced_space_cannot_certify(tmp: Path) -> None:
    """The defect this guards: a reduced SPACE tiles itself exactly.

    Every slice verdict is well-formed, the partition is perfect and rc is 0 --
    and the run still covered 0.02% of the space.  `covered == space` is true of
    ANY requested range, so the combiner must label against 2^32, exactly as the
    streamers do per slice.  Before this, `SPACE=2^18 galaxy_shard.sh` printed
    VERDICT=BIT-EXACT-ALL-INPUTS.
    """
    for space in (1 << 18, 1 << 20, TWO32 // 2):
        rc, last, calls, _ = run_shard(
            tmp, f"reduced-{space}", NPAR=32, BAND_BITS=10, SPACE=space
        )
        assert_exact_cover(calls, space, 32)
        assert "BIT-EXACT-ALL-INPUTS" not in last, (
            f"a {space}-input sweep certified itself as exhaustive :: {last}"
        )
        assert f"VERDICT=BIT-EXACT-PARTIAL-{space}-OF-2^32" in last, last
        # A partial run is useful evidence but not a numerical admission.
        assert rc != 0, f"partial space={space} was admitted :: {last}"
    print("PASS a reduced SPACE reports PARTIAL, never ALL-INPUTS")


def test_per_op_compiler_flags_reach_every_slice(tmp: Path) -> None:
    flags = "-mtt-tensix-optimize-foo -mno-tt-tensix-optimize-bar"
    rc, last, calls, _ = run_shard(
        tmp,
        "per-op-flags",
        NPAR=4,
        BAND_BITS=10,
        SPACE=1 << 12,
        FULL_SPACE=1 << 12,
        FLAGS_VALUE=flags,
    )
    assert rc == 0, last
    assert len(calls) == 4
    assert {call["compiler_options"] for call in calls} == {flags}
    print("PASS exact per-op compiler flags reach every slice")


def test_divergence_is_reported_as_divergence(tmp: Path) -> None:
    """A diverging slice is not a dead chip.

    The streamers `return 0 if all_equal and numeric_ok else 1`, so every slice
    that finds a difference exits non-zero.  The driver used to read any non-zero
    exit as a dead chip, which put each diverging slice into --failed-chips and
    made the combiner's `not all_equal and not invalid_list` branch unreachable:
    a real, fully-covered divergence came back INCOMPLETE with the whole space
    covered.  Observed on silicon for geluappx-fresh (18 of 32 slices diverging
    on every band, covered=2^32, verdict=INCOMPLETE).
    """
    for tag, diverge in (("one", "7"), ("many", "0,1,2,3,16,17")):
        rc, last, calls, _ = run_shard(
            tmp, f"div-{tag}", diverge=diverge, NPAR=32, BAND_BITS=23, SPACE=TWO32
        )
        assert_exact_cover(calls, TWO32, 32)
        assert "VERDICT=DIVERGENT" in last, f"{tag}: divergence lost :: {last}"
        assert "invalid=[]" in last, f"{tag}: a diverging slice was called dead :: {last}"
        assert f"covered={TWO32}" in last, last
        # Semantic uplift may diverge from hand and still be admitted by the
        # absolute oracle plus global per-class ULP rule.
        assert rc == 0, f"{tag}: oracle-clean uplift was refused :: {last}"
        assert "numeric_admission=PASS" in last, last
    # And a chip that really dies is still a dead chip, even alongside divergence.
    rc, last, calls, _ = run_shard(
        tmp, "div-and-dead", diverge="7", dead="9", NPAR=32, BAND_BITS=23, SPACE=TWO32
    )
    assert "invalid=[9]" in last, last
    assert "VERDICT=INCOMPLETE" in last, last
    assert rc != 0, last
    print("PASS a diverging slice reports DIVERGENT, not a dead chip")


def test_tri_arm_deployment_gate(tmp: Path) -> None:
    """A/B is strict even when B/C has a valid numerical admission."""
    geometry = dict(NPAR=4, BAND_BITS=10, SPACE=1 << 12, FULL_SPACE=1 << 12)

    rc, last, calls, out = run_shard(
        tmp, "tri-pass", tri=True, TRI_SELECTED_FLAGS="", **geometry
    )
    assert rc == 0, last
    deployment = (out / "myop-DEPLOYMENT-VERDICT.txt").read_text()
    assert "VERDICT=PASS" in deployment, deployment
    assert "compiler_gate=BIT-EXACT-ALL-INPUTS" in deployment, deployment
    assert "numeric_admission=PASS" in deployment, deployment
    assert {call["selected_flags"] for call in calls} == {""}
    assert {call["baseline_flags"] for call in calls} == {"-baseline"}
    for call in calls:
        assert call["selected_runner_temp"] != call["baseline_sem_runner_temp"]
        assert call["baseline_sem_runner_temp"] != call["baseline_hand_runner_temp"]
    identity = (tmp / "tri-pass" / "tri-idmap.tsv").read_text().strip().split("\t")
    assert identity[1] == identity[3] == "c" * 64
    assert identity[2] != identity[4], "same variant name lost distinct A/B text"

    rc, last, _, out = run_shard(
        tmp, "tri-compiler-wrong", tri=True, compiler_diverge="1", **geometry
    )
    assert rc != 0, "A/B wrong-code was admitted"
    numeric = (out / "myop-NUMERIC-ADMISSION.tsv").read_text()
    assert "\tPASS\t" in numeric or "\tPASS\n" in numeric, numeric
    deployment = (out / "myop-DEPLOYMENT-VERDICT.txt").read_text()
    assert "VERDICT=FAIL" in deployment, deployment
    assert "compiler_gate=DIVERGENT" in deployment, deployment
    assert "numeric_admission=PASS" in deployment, deployment

    rc, last, _, out = run_shard(
        tmp, "tri-uplift", tri=True, diverge="0,2", **geometry
    )
    assert rc == 0, last
    deployment = (out / "myop-DEPLOYMENT-VERDICT.txt").read_text()
    assert "VERDICT=PASS" in deployment, deployment
    assert "semantic_equivalence=DIVERGENT" in deployment, deployment
    print("PASS tri-arm deployment isolates colliding A/B variants and independently admits B/C uplift")


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
        "OP=myop LOCAL_SEM_ABSOLUTE=PASS LOCAL_HAND_ORACLE_COMPLETE=PASS "
        "CAMPAIGN_ADMISSION=DEFERRED_GLOBAL\n"
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
                "OP=op LOCAL_SEM_ABSOLUTE=PASS LOCAL_HAND_ORACLE_COMPLETE=PASS "
                "CAMPAIGN_ADMISSION=DEFERRED_GLOBAL\n"
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
                "OP=op LOCAL_SEM_ABSOLUTE=PASS LOCAL_HAND_ORACLE_COMPLETE=PASS "
                "CAMPAIGN_ADMISSION=DEFERRED_GLOBAL\n"
            )
        summary, passed = galaxy_combine.combine(out, 2, 10, "op", True, False)
        assert passed, summary
        assert "VERDICT=BIT-EXACT-PARTIAL-10-OF-2^32" in summary, summary
        summary, passed = galaxy_combine.combine(out, 2, 10, "op", True, False, {1})
        assert not passed and "invalid=[1]" in summary, summary
        summary, passed = galaxy_combine.combine(
            out, 2, 10, "op", False, False,
            full_space=1 << 32, require_full_space=True,
        )
        assert not passed and "VERDICT=BIT-EXACT-PARTIAL" in summary, summary
    print("PASS combiner honours --failed-chips over a present verdict file")


def main() -> int:
    if not SHARD.is_file():
        print(f"FAIL: no galaxy_shard.sh at {SHARD}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        test_exact_partition(root)
        test_partition_matrix(root)
        test_reduced_space_cannot_certify(root)
        test_per_op_compiler_flags_reach_every_slice(root)
        test_divergence_is_reported_as_divergence(root)
        test_tri_arm_deployment_gate(root)
        test_degenerate_geometries_refuse(root)
        test_dead_slice_is_not_covered(root)
        test_stale_verdict_cannot_stand_in_for_a_dead_slice(root)
    test_combiner_refuses_empty_space()
    test_combiner_failed_chips_override_present_verdicts()
    print("\nSELFTEST galaxy shard arithmetic: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
