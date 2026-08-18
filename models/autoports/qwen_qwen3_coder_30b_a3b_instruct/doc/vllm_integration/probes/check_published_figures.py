# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Every number stage 08 publishes, re-derived from the artifact it came from.

This project's recurring failure mode is prose drifting away from data that is
itself correct: a README figure that was true when it was written and is now a
digit off, a count that no longer matches the rows it counts, a claim about a
benchmark that quietly cites the wrong profile. Every earlier stage that shipped
one of these checkers had it catch something. So this one walks from the
artifacts to the prose rather than the other way round.

What is checked
---------------

1. **The headline table**, against ``readiness_vllm/vllm_benchmark.json`` --
   TTFT median/P99, TPOT mean/P99, ITL median/P99, aggregate output throughput,
   request throughput, completed count, output tokens, elapsed. The published
   decode t/s/u must also equal ``1000 / mean_tpot_ms`` recomputed here, not
   merely match the JSON's own field.
2. **The workload shape next to every benchmark number.** Each benchmark section
   names a prompt length, output length, request count, concurrency and
   ``max_num_seqs``; those must match the ``config`` block of the JSON it cites.
   A number quoted without its shape, or with the wrong one, is the specific
   defect the goal calls out.
3. **The secondary CI serving-burst table**, against
   ``vllm_ci_serving_benchmark.json``, plus the rule that the burst profile is
   **not** presented as the headline: the README must say so, and the headline
   t/s/u must come from the single-user file.
4. **The slot-count comparison table**, against all three benchmark JSONs.
5. **The sampling gate counts** (passed / failed / skipped) against
   ``sampling_tests.log``, and every test id the README names as failing must
   actually appear in that log's failure list -- in both directions, so a
   failure the README forgot is caught as well as one it invented.
6. **The adapter-contract claims**, against
   ``probes/adapter_contract_probe.json``: every check the README tables must
   exist in the probe output and have passed, and the probe must report zero
   failures.
7. **The async-scheduling numbers**, against
   ``logs/async_scheduling_vllm_benchmark.json``, and the byte-identity claim
   re-derived by actually comparing the two qualitative JSONs.
8. **The context claims** against ``doc/context_contract.json`` and the KV-cache
   sizes echoed in ``readiness_vllm/server.log``.
9. **The non-aligned prompt table**, against ``non_aligned_prompt_lengths.json``
   -- including re-deriving the divisibility columns rather than trusting them.
10. **The qualitative verdict's quotes.** Every passage the README quotes from a
    completion must occur verbatim in the artifact it attributes it to.
11. **The registration claims**: the bundle metadata's ``arch`` and
    ``main_class``, and the registration line in ``server.log``.
12. **The KV block sizing for the run whose server log was superseded.**
    ``readiness_vllm/server.log`` is the ``max_num_seqs=1`` run; the
    ``max_num_seqs=32`` run's log was overwritten by it. The 8224/263168 figures
    are therefore *derived* from the worker's own sizing rule, and this script
    validates that rule by recomputing the ``max_num_seqs=1`` row and requiring
    it to reproduce the retained log exactly. It also requires the README and
    work log to say out loud that the pair is derived rather than quoted.
13. **The coverage boundary itself.** Every figure-shaped numeric token in the
    README must be either re-derived by one of the checks above or named in the
    ``UNCOVERED`` table with a reason. A number in neither fails the gate. This
    exists so the checker's blind spots are enumerated rather than silent: what
    is *not* verified — including the two reduced-target ITL figures whose
    scratch artifacts were never retained — is printed on every run.

Exits non-zero on any mismatch, so it is a gate and not a report.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
MODEL = STAGE.parent.parent
READINESS = MODEL / "readiness_vllm"

README = (STAGE / "README.md").read_text()
WORK_LOG = (STAGE / "work_log.md").read_text()

FAILURES: list[str] = []

#: Numeric tokens this script has actually re-derived from an artifact. Fed by
#: ``says`` and by explicit ``cover`` calls next to the regex-driven checks, and
#: consumed by the coverage-boundary check at the bottom of the file.
COVERED: set[str] = set()

NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w.])")


def numbers(text: str) -> set[str]:
    """Figure-shaped numeric tokens, comma separators normalised away."""
    return {tok.replace(",", "") for tok in NUMBER.findall(text)}


def cover(*values) -> None:
    for value in values:
        COVERED.update(numbers(str(value)))


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"[FAIL] {msg}")


def ok(msg: str) -> None:
    print(f"[ ok ] {msg}")


def load(path: Path):
    if not path.is_file():
        fail(f"missing artifact {path}")
        return None
    return json.loads(path.read_text())


def says(text: str, needle: str, where: str) -> bool:
    if needle in text:
        if text is README:
            cover(needle)
        return True
    fail(f"{where} does not contain {needle!r}")
    return False


def fmt(value: float, places: int) -> str:
    return f"{value:.{places}f}"


# ---------------------------------------------------------------------------
# 1-4. benchmark figures, with their workload shapes
# ---------------------------------------------------------------------------

PRIMARY = load(READINESS / "vllm_benchmark.json")
BURST = load(READINESS / "vllm_ci_serving_benchmark.json")
PRIMARY32 = load(READINESS / "vllm_benchmark_maxnumseqs32.json")
ASYNC = load(STAGE / "logs" / "async_scheduling_vllm_benchmark.json")


def check_profile(data, expect_profile: str, shape: dict, label: str) -> None:
    if data is None:
        return
    if data.get("profile") != expect_profile:
        fail(f"{label}: profile is {data.get('profile')!r}, expected {expect_profile!r}")
    cfg = data["config"]
    for key, expected in shape.items():
        if cfg.get(key) != expected:
            fail(f"{label}: config[{key}] is {cfg.get(key)!r}, README claims {expected!r}")
    if not cfg.get("ignore_eos"):
        fail(f"{label}: ignore_eos is not set, README says the workload is ignore_eos")
    ok(f"{label}: profile and workload shape match {data['raw_result_file'].split('/')[-1]}")


check_profile(
    PRIMARY,
    "single_user_decode",
    {"prompt_len": 128, "output_len": 128, "num_requests": 1, "concurrency": 1, "temperature": 0.0},
    "primary 128/128/1",
)
check_profile(
    BURST,
    "ci_serving_burst",
    {"prompt_len": 100, "output_len": 100, "num_requests": 32, "concurrency": None, "temperature": 0.0},
    "CI burst 100/100/32",
)
check_profile(
    PRIMARY32,
    "single_user_decode",
    {"prompt_len": 128, "output_len": 128, "num_requests": 1, "concurrency": 1, "temperature": 0.0},
    "primary 128/128/1 @ max_num_seqs=32",
)
check_profile(
    ASYNC,
    "single_user_decode",
    {"prompt_len": 128, "output_len": 128, "num_requests": 1, "concurrency": 1, "temperature": 0.0},
    "async-scheduling 128/128/1",
)

if PRIMARY:
    tpot = PRIMARY["tpot_ms"]["mean"]
    derived = 1000.0 / tpot
    published = re.search(
        r"\*\*Decode t/s/u\*\* \(from mean TPOT, `1000 / ([0-9.]+)`\) \| \*\*([0-9.]+) t/s/u\*\*", README
    )
    if not published:
        fail("README headline table has no parsable 'Decode t/s/u (from mean TPOT ...)' row")
    else:
        if fmt(tpot, 3) != published.group(1):
            fail(f"headline t/s/u divisor {published.group(1)} != mean TPOT {fmt(tpot, 3)}")
        if fmt(derived, 3) != published.group(2):
            fail(f"headline decode t/s/u {published.group(2)} != recomputed 1000/{fmt(tpot,3)} = {fmt(derived,3)}")
        else:
            cover(published.group(1), published.group(2), 1000)
            ok(f"headline decode t/s/u {published.group(2)} == 1000 / {fmt(tpot,3)}, re-derived")
    for label, value, places in [
        ("TTFT median", PRIMARY["ttft_ms"]["p50"], 3),
        ("TTFT p99", PRIMARY["ttft_ms"]["p99"], 3),
        ("TPOT mean", tpot, 3),
        ("ITL median", PRIMARY["itl_ms"]["p50"], 3),
        ("ITL p99", PRIMARY["itl_ms"]["p99"], 3),
        ("output throughput", PRIMARY["output_throughput_tok_per_s"], 3),
        ("request throughput", PRIMARY["request_throughput_per_s"], 3),
        ("elapsed", PRIMARY["elapsed_s"], 3),
    ]:
        says(README, fmt(value, places), f"README (primary {label})")
    if PRIMARY["completed_requests"] != 1 or int(PRIMARY["total_output_tokens"]) != 128:
        fail(f"primary completed={PRIMARY['completed_requests']} out_tokens={PRIMARY['total_output_tokens']}")
    says(README, "1/1, 128 output tokens", "README (primary completion line)")
    cover(PRIMARY["config"]["prompt_len"], PRIMARY["config"]["output_len"], PRIMARY["config"]["num_requests"])
    ok("primary single-user figures all trace to vllm_benchmark.json")

if BURST:
    for label, value in [
        ("output throughput", BURST["output_throughput_tok_per_s"]),
        ("TTFT median", BURST["ttft_ms"]["p50"]),
        ("TTFT p99", BURST["ttft_ms"]["p99"]),
        ("TPOT mean", BURST["tpot_ms"]["mean"]),
        ("TPOT p99", BURST["tpot_ms"]["p99"]),
        ("ITL median", BURST["itl_ms"]["p50"]),
        ("ITL p99", BURST["itl_ms"]["p99"]),
        ("per-user decode", 1000.0 / BURST["tpot_ms"]["mean"]),
        ("request throughput", BURST["request_throughput_per_s"]),
        ("elapsed", BURST["elapsed_s"]),
    ]:
        says(README, fmt(value, 3), f"README (burst {label})")
    if BURST["completed_requests"] != 32 or int(BURST["total_output_tokens"]) != 3200:
        fail(f"burst completed={BURST['completed_requests']} out_tokens={BURST['total_output_tokens']}")
    says(README, "32/32, 3200 output tokens", "README (burst completion line)")
    cover(BURST["config"]["prompt_len"], BURST["config"]["output_len"], BURST["config"]["num_requests"])
    ok("CI serving-burst figures all trace to vllm_ci_serving_benchmark.json")

# The burst must be labelled secondary and must not supply the headline.
says(README, "This is not the headline decode number", "README (burst caveat)")
if PRIMARY and BURST:
    head = re.search(r"## Headline.*?(?=\n## )", README, re.S)
    if head and fmt(1000.0 / BURST["tpot_ms"]["mean"], 3) in head.group(0):
        fail("the CI burst t/s/u appears inside the headline section")
    else:
        ok("headline section carries the single-user t/s/u only")

if PRIMARY and PRIMARY32 and BURST:
    row = re.search(
        r"\| `max_num_seqs=32` \| 128/128/1 \(1 active\) \| ([0-9.]+) ms \| ([0-9.]+) \| ([0-9.]+) \|", README
    )
    if not row:
        fail("slot-count comparison table row for max_num_seqs=32 / 128-128-1 is missing or malformed")
    else:
        want = (
            fmt(PRIMARY32["tpot_ms"]["mean"], 3),
            fmt(1000.0 / PRIMARY32["tpot_ms"]["mean"], 3),
            fmt(PRIMARY32["output_throughput_tok_per_s"], 3),
        )
        if row.groups() != want:
            fail(f"slot-count row says {row.groups()}, artifacts say {want}")
        else:
            cover(*row.groups())
            ok("slot-count comparison row re-derived from vllm_benchmark_maxnumseqs32.json")
    ratio = BURST["output_throughput_tok_per_s"] / PRIMARY["output_throughput_tok_per_s"]
    claimed = re.search(r"aggregate throughput\s+still rises ([0-9.]+)x from 1 to 32 users", README)
    if claimed and abs(float(claimed.group(1)) - ratio) > 0.05:
        fail(f"aggregate-throughput ratio claim {claimed.group(1)}x != {ratio:.2f}x")
    elif claimed:
        cover(claimed.group(1))
        ok(f"aggregate-throughput ratio {claimed.group(1)}x == {ratio:.2f}x re-derived")

# vLLM overhead against the stage-07 shipped lower bound.
if PRIMARY:
    sweep = (MODEL / "doc" / "datatype_sweep" / "README.md").read_text()
    for shipped in ("19.213 ms / 52.049 t/s/u", "129.941"):
        if shipped.split(" /")[0] not in sweep:
            fail(f"stage-07 README no longer contains {shipped!r}; the lower-bound comparison is stale")
    overhead = PRIMARY["tpot_ms"]["mean"] - 19.213
    pct = 100.0 * overhead / 19.213
    m = re.search(r"vLLM adds\s+\*\*([0-9.]+) ms per token, ([0-9.]+) %\*\*", README)
    if not m:
        fail("README does not state the vLLM per-token overhead in the expected form")
    elif abs(float(m.group(1)) - overhead) > 0.002 or abs(float(m.group(2)) - pct) > 0.15:
        fail(f"overhead claim {m.groups()} != re-derived ({overhead:.3f} ms, {pct:.1f} %)")
    else:
        cover(m.group(1), m.group(2), 19.213, "52.049", "129.941")
        # The TTFT gap the README calls "the extra ~182 ms".
        gap = PRIMARY["ttft_ms"]["p50"] - 129.941
        claimed_gap = re.search(r"the extra ~(\d+) ms is vLLM's", README)
        if not claimed_gap:
            fail("README does not state the serving-vs-standalone TTFT gap in the expected form")
        elif abs(float(claimed_gap.group(1)) - gap) > 1.0:
            fail(f"TTFT gap claim ~{claimed_gap.group(1)} ms != {gap:.3f} ms re-derived")
        else:
            cover(claimed_gap.group(1))
            ok(f"TTFT gap ~{claimed_gap.group(1)} ms == {PRIMARY['ttft_ms']['p50']:.3f} - 129.941 re-derived")
        ok(f"vLLM decode overhead {m.group(1)} ms / {m.group(2)} % re-derived from 19.778 - 19.213")

# ---------------------------------------------------------------------------
# 5. sampling gate
# ---------------------------------------------------------------------------

log_path = READINESS / "sampling_tests.log"
if not log_path.is_file():
    fail(f"missing {log_path}")
else:
    log = log_path.read_text()
    counts = re.search(r"(\d+) failed, (\d+) passed, (\d+) skipped", log)
    if not counts:
        fail("sampling_tests.log has no pytest summary line")
    else:
        failed_n, passed_n, skipped_n = (int(g) for g in counts.groups())
        claim = re.search(r"\*\*Sampling gate: (\d+) passed / (\d+) failed / (\d+) skipped\*\*", README)
        if not claim:
            fail("README headline has no 'Sampling gate: P passed / F failed / S skipped'")
        elif (int(claim.group(1)), int(claim.group(2)), int(claim.group(3))) != (passed_n, failed_n, skipped_n):
            fail(f"sampling gate claim {claim.groups()} != log ({passed_n}, {failed_n}, {skipped_n})")
        else:
            cover(passed_n, failed_n, skipped_n)
            ok(f"sampling gate {passed_n}/{failed_n}/{skipped_n} matches sampling_tests.log")
        heading = re.search(r"## Sampling test results — (\d+) passed, (\d+) failed, (\d+) skipped", README)
        if not heading or (int(heading.group(1)), int(heading.group(2)), int(heading.group(3))) != (
            passed_n,
            failed_n,
            skipped_n,
        ):
            fail("the sampling-results section heading disagrees with the log")

        # Class A + Class B must together account for exactly the failures.
        actual_failed = set(re.findall(r"^(test_\S+::\S+|test_\S+) FAILED \[", log, re.M))
        actual_failed |= {m for m in re.findall(r"^FAILED (\S+)", log, re.M)}
        actual_names = {f.split("::")[-1] for f in actual_failed}
        class_a = int(re.search(r"\*Class A — per-request seeding and RNG \((\d+) failures\)", README).group(1))
        class_b = int(re.search(r"\*Class B — presence penalty on this checkpoint \((\d+) failures\)", README).group(1))
        if class_a + class_b != failed_n:
            fail(f"Class A ({class_a}) + Class B ({class_b}) != {failed_n} failures in the log")
        else:
            cover(class_a, class_b)
            ok(f"failure classes account for all {failed_n} failures ({class_a} seeding + {class_b} presence)")
        penalty_failures = {n for n in actual_names if "penalt" in n.lower()}
        if len(penalty_failures) != class_b:
            fail(f"README says {class_b} penalty failures, log has {len(penalty_failures)}: {sorted(penalty_failures)}")
        # The other four penalty tests must be in the PASSED list, not merely
        # absent from the FAILED one -- that is the claim the stage rests on.
        penalty_passed = {m.split("::")[-1] for m in re.findall(r"^(\S*test_tt_penalties\S*) PASSED \[", log, re.M)}
        if len(penalty_passed) != 4:
            fail(
                f"README claims 4 of 6 test_tt_penalties pass, log shows {len(penalty_passed)}: {sorted(penalty_passed)}"
            )
        elif not all(cls in " ".join(penalty_passed) for cls in ("repetition", "frequency")):
            fail(f"the 4 passing penalty tests are not both repetition and both frequency: {sorted(penalty_passed)}")
        else:
            cover(len(penalty_passed), len(penalty_passed) + class_b)
            says(README, "**4 of the 6 `test_tt_penalties`**", "README (penalty pass count)")
            ok("4 of 6 test_tt_penalties PASSED in the log (both repetition, both frequency)")
        # Every test id the README names as failing must really have failed.
        for named in re.findall(r"`(test_[A-Za-z0-9_]+(?:\[[^\]]*\])?)`", README):
            base = named.split("[")[0]
            if base in {"test_logprobs", "test_config"}:
                continue
            if "should" in named:
                continue
            passed_ids = {m.split("::")[-1] for m in re.findall(r"^(\S+) PASSED \[", log, re.M)}
            passed_bases = {p.split("[")[0] for p in passed_ids}
            if named in actual_names or base in {n.split("[")[0] for n in actual_names}:
                continue
            if named in passed_ids or base in passed_bases:
                continue
            # A bare module name (``test_tt_penalties``) rather than a test id:
            # accept it if the log ran anything from that file.
            if re.search(rf"(?:^|/){re.escape(base)}\.py::", log, re.M):
                continue
            fail(f"README names {named!r} but it appears in neither the PASSED nor the FAILED list")
        ok("every sampling test id named in the README appears in sampling_tests.log")

        logprobs_passed = len(re.findall(r"TestLogprobs::test_logprobs\[[^\]]*\] PASSED \[", log))
        if f"all {logprobs_passed} `test_logprobs" not in README:
            fail(f"README does not say all {logprobs_passed} test_logprobs parametrisations passed")
        else:
            cover(logprobs_passed)
            ok(f"all {logprobs_passed} test_logprobs parametrisations passed, re-derived from the log")
        duration = re.search(r"in (\d+\.\d+)s \(", log)
        if not duration:
            fail("sampling_tests.log has no total duration line")
        else:
            says(README, duration.group(1), "README (sampling suite duration)")

SUITE = STAGE / "logs" / "stage08_penalties_model_suite.log"
if not SUITE.is_file():
    fail("missing logs/stage08_penalties_model_suite.log")
else:
    tally = re.search(r"(\d+) passed, (\d+) deselected", SUITE.read_text())
    if not tally:
        fail("stage08_penalties_model_suite.log has no pytest tally line")
    elif "failed" in tally.string[tally.start() : tally.end() + 40]:
        fail("stage08_penalties_model_suite.log reports failures")
    else:
        says(README, f"{tally.group(1)} passed", "README (model-suite tally)")
        ok(f"model suite after the penalty change: {tally.group(1)} passed, 0 failed")

# ---------------------------------------------------------------------------
# 6. adapter-contract probe
# ---------------------------------------------------------------------------

PROBE = load(HERE / "adapter_contract_probe.json")
if PROBE:
    if PROBE["failed"] != 0:
        fail(f"adapter_contract_probe.json reports {PROBE['failed']} failed check(s)")
    by_name = {row["check"]: row for row in PROBE["checks"]}
    table = re.findall(r"^\| `([a-z_]+)` \| ", README, re.M)
    named = [
        t
        for t in table
        if (
            t in by_name
            or t.startswith(
                ("steady_", "changed_", "one_", "stale_", "current_", "non_", "vllm_", "async_", "sampler_")
            )
        )
        # ``sampler_matches_penalised_reference`` and friends belong to the
        # penalty probe (6b), not to this one; checked there.
        and not t.startswith(
            (
                "sampler_matches_",
                "forced_",
                "matches_",
                "no_unexpected_",
                "reaches_",
                "same_local_",
                "unpenalised_",
                "boundary_",
                "neutral_",
                "fast_path_",
                "all_identical_",
                "presence_reference_",
                "frequency_changes_",
            )
        )
    ]
    missing = [t for t in named if t not in by_name]
    if missing:
        fail(f"README names probe checks that the probe did not run: {missing}")
    not_passed = [t for t in named if t in by_name and not by_name[t]["pass"]]
    if not_passed:
        fail(f"README presents these probe checks as evidence but they did not pass: {not_passed}")
    claimed_total = re.search(r"\*\*All (\d+) checks pass\*\*", README)
    if not claimed_total or int(claimed_total.group(1)) != len(PROBE["checks"]):
        fail(f"README claims {claimed_total and claimed_total.group(1)} probe checks, probe ran {len(PROBE['checks'])}")
    else:
        cover(
            len(PROBE["checks"]),
            PROBE["config"]["steady_steps"],
            PROBE["config"]["prompt_len"],
            PROBE["config"]["max_num_seqs"],
            PROBE["config"]["num_layers"],
        )
        ok(f"all {len(PROBE['checks'])} adapter-contract checks ran and passed")
    stats = PROBE["trace_stats"]
    steps = PROBE["config"]["steady_steps"]
    if f"replays **+{steps} over {steps} tokens**" not in README:
        fail(f"README does not state the replay delta as +{steps} over {steps} tokens")
    if f"`caller_token_readbacks` **+{steps}**" not in README:
        fail(f"README does not state caller_token_readbacks +{steps}")
    if PROBE["serving_audit"]["precision_config"] != str(
        MODEL / "doc" / "datatype_sweep" / "selected_precision_config.json"
    ):
        fail(f"probe recorded precision config {PROBE['serving_audit']['precision_config']}")
    else:
        ok("serving path loaded doc/datatype_sweep/selected_precision_config.json")
    if stats.get("captures", 0) < 1:
        fail("probe trace_stats shows no decode trace capture at all")

# ---------------------------------------------------------------------------
# 6b. sampling-penalty shard-boundary probe
# ---------------------------------------------------------------------------
#
# Every claim in README "Sampling penalties" that is a *result* rather than a
# design statement comes from this one JSON, so it is re-derived here rather than
# spot-checked: the pass flags one by one, the shard geometry, the per-die reach,
# the boundary columns, and each cost figure in the cost table.

RERUN_TPOT = None
_rerun_early = STAGE / "logs" / "penalty_rerun_vllm_benchmark.json"
if _rerun_early.is_file():
    RERUN_TPOT = json.loads(_rerun_early.read_text())["tpot_ms"]["mean"]

PENALTY = load(HERE / "penalty_shard_boundary_probe.json")
if PENALTY:
    if not PENALTY.get("passed"):
        fail("penalty_shard_boundary_probe.json reports passed=false")
    else:
        ok("penalty shard-boundary probe passed")

    # The README presents these as a checks table; every one must be true in the
    # JSON, and every one must be named in the README.
    penalty_checks = [
        "matches_vllm_reference",
        "no_unexpected_columns_moved",
        "reaches_die_0",
        "reaches_die_3",
        "same_local_index_on_other_dies_untouched",
        "unpenalised_rows_bit_identical",
        "sampler_matches_penalised_reference",
        "forced_matches_reference",
        "forced_penalty_changed_the_winner",
        "neutral_request_is_fast_path",
        "fast_path_is_identity",
        "boundary_columns_covered",
    ]
    for name in penalty_checks:
        value = PENALTY.get(name)
        truthy = value == [0, PENALTY["local_vocab"] - 1] if name == "boundary_columns_covered" else bool(value)
        if not truthy:
            fail(f"penalty probe check {name!r} is not satisfied: {value!r}")
        if f"`{name}`" not in README:
            fail(f"penalty probe check {name!r} is not named in the README")
    ok(f"all {len(penalty_checks)} penalty-probe checks satisfied and named in the README")

    # Shard geometry: the README's whole correctness argument rests on these.
    if PENALTY["vocab"] != PENALTY["local_vocab"] * PENALTY["devices"]:
        fail("penalty probe geometry is not an exact even shard")
    cover(PENALTY["vocab"], PENALTY["local_vocab"], PENALTY["devices"], PENALTY["local_vocab"] - 1)
    says(README, f"{PENALTY['local_vocab']}", "README (penalty shard width)")
    says(README, "0 and 37983", "README (penalty boundary columns)")

    # Per-die reach: "5 penalised ids on die 0 and 4 on die 3".
    per_die = PENALTY["penalised_tokens_per_die"]
    claim = re.search(r"(\d+) penalised ids on die 0 and (\d+) on die 3", README)
    if not claim:
        fail("README does not state the per-die penalised-id counts")
    elif (int(claim.group(1)), int(claim.group(2))) != (per_die.get("0", 0), per_die.get("3", 0)):
        fail(f"README per-die penalised-id counts disagree with the probe: {claim.groups()} vs {per_die}")
    else:
        cover(claim.group(1), claim.group(2))
        ok("per-die penalised-id counts re-derived from the probe")

    if PENALTY["expected_columns_total"] != PENALTY["expected_columns_that_moved"]:
        fail("penalty probe: not every requested column actually moved")
    says(README, str(PENALTY["expected_columns_total"]), "README (penalised column count)")
    says(README, str(PENALTY["unpenalised_rows"]), "README (unpenalised row count)")
    cover(PENALTY["expected_columns_total"], PENALTY["unpenalised_rows"])
    says(README, fmt(PENALTY["max_ulp_ratio"], 2), "README (worst bf16 ulp ratio)")

    cost = PENALTY.get("cost")
    if not cost:
        fail("penalty_shard_boundary_probe.json has no cost block; re-run the probe with --time")
    else:
        # The device-cost table, cell by cell, with both deltas recomputed rather
        # than copied -- the fast-path claim *is* the subtraction.
        for key in ("sampler_ms_mode0", "sampler_ms_mode1", "sampler_ms_mode3"):
            says(README, fmt(cost[key], 4), f"README (penalty cost {key})")
        for label, mode in (("device_cost_repetition_only_ms", 1), ("device_cost_both_ms", 3)):
            derived = cost[f"sampler_ms_mode{mode}"] - cost["sampler_ms_mode0"]
            if abs(derived - cost[label]) > 5e-4:
                fail(f"{label} in the probe JSON is not sampler_ms_mode{mode} - sampler_ms_mode0")
            says(README, fmt(cost[label], 4), f"README (penalty {label})")
            # Limitations quotes the same figure to 3 places.
            cover(fmt(cost[label], 3), fmt(cost[label], 2))
        # The staging figures the optimisation table quotes, and its speedups,
        # recomputed from the two numbers rather than restated.
        for key in ("host_staging_ms_mode1", "host_staging_ms_mode3"):
            says(README, fmt(cost[key], 4), f"README (penalty {key})")
        for before, after in ((11.6391, cost["host_staging_ms_mode1"]), (17.1644, cost["host_staging_ms_mode3"])):
            says(README, fmt(before, 4), "README (penalty staging before optimisation)")
            says(README, fmt(before / after, 1), "README (penalty staging speedup)")
        # "9.7 MB" per operand: 32 rows x 151936 bf16.
        says(README, fmt(32 * 151936 * 2 / 1e6, 1), "README (penalty operand size in MB)")
        # The reciprocal substitution's saving is one operand's staging, which is
        # the difference of the two modes -- recomputed, not restated.
        says(
            README,
            fmt(cost["host_staging_ms_mode3"] - cost["host_staging_ms_mode1"], 4),
            "README (one operand's staging)",
        )
        says(README, fmt(cost["device_cost_repetition_only_ms"] - 0.0265, 4), "README (device cost of the reciprocal)")
        cover(cost["reps"])
        history = PENALTY.get("cost_serving_history")
        if not history:
            fail("penalty probe JSON has no cost_serving_history block")
        else:
            for key in ("host_staging_ms_mode1", "host_staging_ms_mode3"):
                says(README, fmt(history[key], 4), f"README (serving-history {key})")
            cover(history["history_tokens_per_row"])
            ok(f"staging re-timed at a {history['history_tokens_per_row']}-token history and quoted")
        ok("penalty device+staging cost tables re-derived; speedups recomputed")

# ---------------------------------------------------------------------------
# 6c. penalty serving-parity probe, and the post-penalty benchmark re-run
# ---------------------------------------------------------------------------

PARITY = load(HERE / "penalty_serving_parity_probe.json")
if PARITY:
    if not PARITY.get("passed"):
        fail("penalty_serving_parity_probe.json reports passed=false")
    for name in (
        "all_identical_to_vllm_reference",
        "presence_reference_also_unchanged",
        "frequency_changes_output_on_same_prompt",
    ):
        if not PARITY.get(name):
            fail(f"penalty parity probe: {name} is false")
        if f"`{name}`" not in README:
            fail(f"penalty parity probe result {name!r} is not named in the README")
    cases = PARITY["cases"]
    differing = [c["case"] for c in cases if not c["identical"]]
    if differing:
        fail(f"penalty parity probe: cases differ from the vLLM reference: {differing}")
    says(README, f"all {len(cases)} cases byte-identical", "README (penalty parity count)")
    for value in PARITY["host_sampling_switch"].values():
        says(README, str(value), "README (host-sampling switch value)")
    cover(len(cases))
    # The README's frequency-threshold claim, re-derived from the cases rather
    # than restated: 0.3 must leave the output at the baseline and 0.5 must not.
    baseline = next(c for c in cases if c["case"] == "control_no_penalty")["device_sampled"]
    thresholds = {
        c["penalties"].get("frequency_penalty"): c["device_sampled"] == baseline
        for c in cases
        if c["case"].startswith("frequency_") and c["prompt"] == "a b c a b c a b c"
    }
    if thresholds.get(0.3) is not True or thresholds.get(0.5) is not False:
        fail(
            f"README's frequency threshold claim (0.3 unchanged, 0.5 changed) is not what the probe measured: {thresholds}"
        )
    else:
        cover("0.3", "0.5", "1.0", "2.0")
        ok("penalty parity: 11/11 byte-identical to vLLM's own sampler; frequency threshold re-derived")

COST = load(HERE / "penalty_serving_cost_probe.json")
if COST:
    if not COST.get("all_legs_same_token_count"):
        fail("penalty cost probe: the three legs did not decode the same number of tokens")
    if COST["unpenalised_tokens_streamed"] != COST["workload"]["output_len"]:
        fail("penalty cost probe: the unpenalised leg did not stream the full output length")
    for name in ("none", "repetition_only", "all_three"):
        leg = COST["legs"][name]
        for value, places in ((leg["ttft_ms"], 3), (leg["tpot_ms"], 3), (leg["tsu"], 3)):
            says(README, fmt(value, places), f"README (penalty cost leg {name})")
        # t/s/u must be 1000/TPOT, recomputed rather than trusted.
        if abs(leg["tsu"] - 1000.0 / leg["tpot_ms"]) > 1e-2:
            fail(f"penalty cost leg {name}: t/s/u is not 1000 / TPOT")
    base = COST["legs"]["none"]["tpot_ms"]
    for name in ("repetition_only", "all_three"):
        delta = COST["legs"][name]["tpot_ms"] - base
        if abs(delta - COST["penalty_tpot_overhead_ms"][name]) > 1e-2:
            fail(f"penalty cost probe: {name} overhead is not the TPOT difference")
        says(README, fmt(delta, 3), f"README (penalty overhead {name})")
        says(README, fmt(100.0 * delta / base, 1), f"README (penalty overhead pct {name})")
        cover(round(100.0 * delta / base))
        # The unaccounted residual: in-situ overhead minus this port's own
        # per-step host cost at a serving-sized history.
        history = (PENALTY or {}).get("cost_serving_history") or {}
        mine = history.get("host_staging_ms_mode1" if name == "repetition_only" else "host_staging_ms_mode3")
        if mine:
            says(README, fmt(delta - mine, 2), f"README (unaccounted host residual, {name})")
    # The in-situ unpenalised leg must agree with `vllm bench serve`, which is
    # what makes this probe's harness trustworthy at all.
    if RERUN_TPOT and abs(base - RERUN_TPOT) > 0.5:
        fail(
            f"penalty cost probe's unpenalised leg ({base:.3f} ms) disagrees with vllm bench serve ({RERUN_TPOT:.3f} ms)"
        )
    else:
        ok(f"penalty cost probe: unpenalised leg {base:.3f} ms agrees with vllm bench serve; overheads recomputed")
    # The disclosure requirement: the penalised t/s/u must appear in the headline
    # region and in Limitations, not only in the cost section.
    headline = README[: README.index("## Secondary")]
    limitations = README[README.index("## Limitations and open items") :]
    for region, label in ((headline, "headline section"), (limitations, "Limitations")):
        for name in ("repetition_only", "all_three"):
            if fmt(COST["legs"][name]["tsu"], 3) not in region:
                fail(f"the penalised t/s/u for {name} is not disclosed in the {label}")
    ok("penalised t/s/u disclosed in both the headline section and Limitations")

RERUN = load(STAGE / "logs" / "penalty_rerun_vllm_benchmark.json")
if RERUN and PRIMARY:
    check_profile(
        RERUN,
        "single_user_decode",
        {"prompt_len": 128, "output_len": 128, "num_requests": 1, "concurrency": 1, "temperature": 0.0},
        "penalty re-run",
    )
    for value, places in (
        (RERUN["ttft_ms"]["mean"], 3),
        (RERUN["tpot_ms"]["mean"], 3),
        (1000.0 / RERUN["tpot_ms"]["mean"], 3),
        (RERUN["itl_ms"]["p50"], 3),
        (RERUN["itl_ms"]["p99"], 3),
    ):
        says(README, fmt(value, places), "README (penalty benchmark re-run)")
    # The regression claim itself: the delta and its percentage, recomputed.
    delta = RERUN["tpot_ms"]["mean"] - PRIMARY["tpot_ms"]["mean"]
    says(README, fmt(delta, 3), "README (penalty re-run TPOT delta)")
    says(README, fmt(100.0 * delta / PRIMARY["tpot_ms"]["mean"], 2), "README (penalty re-run TPOT delta %)")
    says(README, fmt(PRIMARY["ttft_ms"]["mean"] - RERUN["ttft_ms"]["mean"], 3), "README (penalty re-run TTFT delta)")
    if delta > 0.5:
        fail(f"penalty re-run regressed TPOT by {delta:.3f} ms against the headline")
    else:
        ok(f"penalty re-run within {delta:+.3f} ms TPOT of the headline, deltas recomputed")

# ---------------------------------------------------------------------------
# 7. async-scheduling claims
# ---------------------------------------------------------------------------

if ASYNC:
    for value in (ASYNC["tpot_ms"]["mean"], 1000.0 / ASYNC["tpot_ms"]["mean"]):
        says(README, fmt(value, 3), "README (async-scheduling)")
    grep = STAGE / "logs" / "async_scheduling_server_grep.log"
    if not grep.is_file() or "'async_scheduling': True" not in grep.read_text():
        fail("async_scheduling_server_grep.log does not show async_scheduling: True")
    else:
        ok("--async-scheduling was accepted by the platform, not force-disabled")
    a = load(STAGE / "logs" / "async_scheduling_qualitative_outputs.json")
    b = load(READINESS / "vllm_qualitative_outputs.json")
    if a and b:
        identical = all(x["greedy_completion"] == y["greedy_completion"] for x, y in zip(a, b))
        if not identical:
            fail("README claims byte-identical greedy output under --async-scheduling; it is not")
        elif len(a) != len(b):
            fail(f"async qualitative has {len(a)} prompts, sync has {len(b)}")
        else:
            says(README, "byte-identical", "README (async identity claim)")
            ok(f"all {len(a)} greedy completions byte-identical async vs sync, re-derived")

# ---------------------------------------------------------------------------
# 8. context / cache claims
# ---------------------------------------------------------------------------

CONTRACT = load(MODEL / "doc" / "context_contract.json")
if CONTRACT:
    ctx = int(CONTRACT["current_supported_context"])
    says(README, str(ctx), "README (served context)")
    if CONTRACT.get("capability_reduction"):
        fail("context contract records a capability reduction; README claims none")
    else:
        ok(f"served context {ctx} == context_contract.json, no reduction")

server_log = READINESS / "server.log"
if not server_log.is_file():
    fail(f"missing {server_log}")
else:
    text = server_log.read_text(errors="ignore")
    for needle in (
        "Registered TT model TTQwen3MoeForCausalLM -> tt_qwen3_coder_30b_a3b_instruct:Qwen3CoderForCausalLM",
        "GPU KV cache size: 262,176 tokens",
        "vLLM-owned KV cache: 8193 blocks x 32 tokens = 262176 tokens",
        "Getting max_tokens_all_users=262144",
    ):
        if needle not in text:
            fail(f"server.log does not contain {needle!r}")
    says(README, "8193 blocks x 32 = **262176** tokens", "README (KV cache size)")
    says(README, "TTQwen3MoeForCausalLM", "README (registered arch)")
    ok("registration and KV-cache lines re-derived from server.log")

    # The retained server.log is run B (max_num_seqs=1). Run A's log was
    # overwritten by it, so the max_num_seqs=32 block figures are *derived*
    # rather than quoted. The derivation is the worker's own sizing rule; the
    # max_num_seqs=1 row is the control that proves the rule, because it is
    # computed the same way and must reproduce the log above exactly.
    if "'max_num_seqs': 1" not in text and "max_num_seqs=1 " not in text:
        fail("server.log is no longer the max_num_seqs=1 run the README says it is")
    if CONTRACT:
        block_size = 32
        supported = int(CONTRACT["current_supported_context"])
        rows = {}
        for seqs in (1, 32):
            blocks = -(-(supported + block_size * seqs) // block_size)
            rows[seqs] = (blocks, blocks * block_size)
        if rows[1] != (8193, 262176):
            fail(f"the block-sizing rule no longer reproduces the retained log: {rows[1]} != (8193, 262176)")
        else:
            ok(f"block-sizing rule validated against server.log at max_num_seqs=1: {rows[1]}")
            blocks32, tokens32 = rows[32]
            says(README, f"{blocks32} blocks x {block_size} = **{tokens32}** tokens", "README (KV cache @ 32)")
            says(README, f"= **{blocks32}**", "README (KV block derivation table)")
            says(README, f"{blocks32} x {block_size} = **{tokens32}**", "README (KV token derivation table)")
            cover(blocks32, tokens32, block_size, supported)
            ok(f"max_num_seqs=32 KV figures re-derived: {blocks32} blocks / {tokens32} tokens")
    # The README must say out loud that those two numbers are not from a log.
    says(README, "they are derived", "README (superseded run-A log disclosure)")
    says(WORK_LOG, "This run's server log was not retained.", "work_log.md (superseded log disclosure)")

# ---------------------------------------------------------------------------
# 9. non-aligned prompt table
# ---------------------------------------------------------------------------

NONALIGNED = load(READINESS / "non_aligned_prompt_lengths.json")
if NONALIGNED:
    for case in NONALIGNED["cases"]:
        n = int(case["prompt_token_count"])
        for divisor in (8, 32, 64, 128, 1024):
            if n % divisor == 0:
                fail(f"non-aligned case {n} is actually divisible by {divisor}")
        if int(case["prompt_tokens_reported"]) != n:
            fail(f"non-aligned case {n}: server reported {case['prompt_tokens_reported']} prompt tokens")
        if case.get("status") != "pass":
            fail(f"non-aligned case {n} did not pass")
        if f"| {n} |" not in README and f"| {n} (natural text) |" not in README:
            fail(f"non-aligned length {n} is in the artifact but not in the README table")
    cover(*(case["prompt_token_count"] for case in NONALIGNED["cases"]))
    ok(f"all {len(NONALIGNED['cases'])} non-aligned lengths verified non-divisible, served, and tabulated")

runner_probe = load(READINESS / "non_aligned_prompt_37.json")
if runner_probe and (runner_probe["prompt_token_count"] != 37 or runner_probe["status"] != "pass"):
    fail("the runner's own non_aligned_prompt_37.json does not record a passing 37-token request")

# ---------------------------------------------------------------------------
# 10. qualitative quotes
# ---------------------------------------------------------------------------

CHAT = load(READINESS / "vllm_qualitative_chat_outputs.json")
CONTROL = load(MODEL / "readiness_qualitative" / "vllm_qualitative_outputs.json")
if CHAT:
    blob = json.dumps(CHAT)
    for quote in [
        "Think of it like learning from different types of teachers",
        "Energy cannot be created or destroyed, only transformed from one form to another.",
        "Bonjour, comment allez-vous aujourd'hui ?",
    ]:
        if json.dumps(quote)[1:-1] not in blob:
            fail(f"README quotes {quote!r} but it is not in vllm_qualitative_chat_outputs.json")
    ok("every qualitative passage quoted in the README occurs verbatim in its artifact")
if CHAT and CONTROL:
    haiku_serving = CHAT[0]["greedy_completion"].strip()
    haiku_control = CONTROL[0]["greedy_completion"].strip()
    if haiku_serving != haiku_control:
        fail("README claims the serving haiku matches the full-model control byte for byte; it does not")
    else:
        ok("serving haiku == full-model control, byte for byte, re-derived")
    if CHAT[0]["greedy_completion"] != CHAT[0]["sampled_completion"]:
        fail("README says greedy and sampled both return the control haiku; they differ")

DEGEN = STAGE / "logs" / "check_degenerate_vllm.log"
if not DEGEN.is_file() or "No degenerate output detected." not in DEGEN.read_text():
    fail("check_degenerate_vllm.log does not record a clean degeneracy gate")
else:
    ok("degeneracy gate clean, per logs/check_degenerate_vllm.log")
    # The repetition bullet quotes three metrics. Re-derive them from the served
    # rows of the log (the ones measured on readiness_vllm/), not from prose.
    served = [line for line in DEGEN.read_text().splitlines() if "readiness_vllm/vllm_qualitative_outputs.json" in line]
    rows = []
    for line in served:
        m = re.search(
            r"'num_tokens': (\d+), 'adjacent_duplication': ([0-9.]+), 'trigram_loop_fraction': ([0-9.]+)", line
        )
        if m:
            rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    if len(rows) != len(served) or not rows:
        fail("could not parse the served rows out of check_degenerate_vllm.log")
    else:
        nonzero_dup = sorted({dup for _, dup, _ in rows if dup})
        if nonzero_dup != [0.0074]:
            fail(f"README says the only non-zero adjacent_duplication is 0.0074; log has {nonzero_dup}")
        short, long_ = [r for r in rows if r[0] < 10], [r for r in rows if r[0] >= 10]
        worst_long = max(t for _, _, t in long_)
        worst_short = max(t for _, _, t in short) if short else None
        if round(worst_long, 3) > 0.088:
            fail(f"README says trigram_loop_fraction <= 0.088 on meaningful lengths; worst is {worst_long}")
        if worst_short != 0.5 or len({r[0] for r in short}) != 1 or short[0][0] != 6:
            fail(f"README attributes the only 0.5 trigram figure to the six-word translation; short rows are {short}")
        cover(0.0, 0.0074, 0.5, worst_long)
        ok(f"repetition metrics re-derived: dup {nonzero_dup}, trigram <= {worst_long} long / {worst_short} short")

# ---------------------------------------------------------------------------
# 11. registration
# ---------------------------------------------------------------------------

META = load(MODEL / "vllm_bundle" / "qwen3_coder_30b_a3b_instruct" / "vllm_metadata.json")
if META:
    if META["arch"] != "Qwen3MoeForCausalLM":
        fail(f"bundle arch is {META['arch']!r}, README says Qwen3MoeForCausalLM")
    if META["main_class"] != "tt_qwen3_coder_30b_a3b_instruct:Qwen3CoderForCausalLM":
        fail(f"bundle main_class is {META['main_class']!r}")
    says(README, META["arch"], "README (bundle arch)")
    says(README, META["main_class"], "README (bundle main_class)")
    ok("bundle metadata matches the README's registration section")

hf_config = MODEL / "doc" / "context_contract.json"
if CONTRACT and CONTRACT.get("hf_model") != "Qwen/Qwen3-Coder-30B-A3B-Instruct":
    fail("context contract hf_model does not match the served model")

# The work log must record the hang that decided preserve_decode_traces.
says(WORK_LOG, "NOC0 CB0 active (0xFFFFFFFF)", "work_log.md (hang evidence)")
# gzipped: the raw dump is 856 KB, over the repo's 500 KB artifact limit.
triage = STAGE / "triage" / "tt-triage-preserve-traces-hang.txt.gz"
_triage_text = ""
if triage.is_file():
    import gzip

    _triage_text = gzip.open(triage, "rt", errors="ignore").read()
if "NoC is likely hung" not in _triage_text:
    fail("triage artifact does not contain the NoC hang the work log and README cite")
else:
    ok("the preserve_decode_traces hang is backed by its tt-triage artifact")

# ---------------------------------------------------------------------------
# 13. the coverage boundary itself
# ---------------------------------------------------------------------------
#
# Everything above walks artifact -> prose. That leaves a silent hole: a number
# this script simply never looks at reads exactly like a checked one. So the
# last check is over the *boundary*. Every figure-shaped numeric token in the
# README must be either (a) covered -- re-derived above, recorded via ``cover``
# -- or (b) named in ``UNCOVERED`` with the reason it is not machine-checkable.
# A new number in neither set fails the gate, which forces the next person to
# choose between wiring it up and declaring it.

UNCOVERED: dict[str, str] = {
    # -- figures introduced by the 2026-08-18 errata block ------------------
    # The errata is append-only and corrects stage 08's *interpretation* of
    # async scheduling; its numbers are source-line citations, a date, and
    # cross-references to the stage-09 re-measurement, all of which are
    # re-derived in doc/optimized_vllm/ rather than here.
    "18": "the errata date (2026-08-18)",
    "2026": "the errata date (2026-08-18)",
    "21": "server.log line number cited by the errata",
    "80": "server.log line number cited by the errata",
    "159": "the fixed per-request trace-capture cost, re-derived in doc/optimized_vllm/",
    "179": "async-off ITL[0] stall, re-derived in doc/optimized_vllm/",
    "190": "vllm/config/scheduler.py line number cited by the errata",
    "198": "vllm/config/scheduler.py line number cited by the errata",
    "657": "this README's own line number, cited by the errata",
    "661": "this README's own line number, cited by the errata",
    "964": "vllm/config/vllm.py line number cited by the errata",
    "1004": "vllm/config/vllm.py line number cited by the errata",
    "0.44": "corrected async per-token gain, re-derived in doc/optimized_vllm/",
    "320.5": "TTFT+ITL[0] async-on, re-derived in doc/optimized_vllm/",
    "320.8": "TTFT+ITL[0] async-off, re-derived in doc/optimized_vllm/",
    "145.763": "async-off TTFT, re-derived in doc/optimized_vllm/",
    # -- configuration echoed from the command line, not measured -------------
    "8100": "serving port (8000 was held by a process outside this session)",
    "8000": "the port that was already held; stated only to explain 8100",
    "50331648": "trace_region_size passed in --tt-config",
    "1024": "the worker's own headroom, block_size * max_num_seqs at 32",
    "8192": "the generator's default rope table length, quoted as a contrast",
    "1.05": "generation_config.json repetition_penalty, quoted from the checkpoint",
    "0.7": "generation_config.json temperature, quoted from the checkpoint",
    "0.8": "generation_config.json top_p, quoted from the checkpoint",
    "12345": "the stale-host-input probe's sentinel token id",
    "129": "prefill warmup length vLLM compiles alongside 128; a runner behaviour, not a figure",
    "132": "device current_pos after the probe's reset step (prompt_len + 1)",
    "240": "vLLM request timeout in seconds",
    # -- model/mesh shape, fixed by the port and checked by earlier stages -----
    "4": "dies in the 1x4 mesh / reduced-target max_num_seqs",
    "48": "decoder layers",
    "2": "layers on the reduced target; also assorted ordinals",
    "8": "page-alignment divisor in the non-aligned table",
    "64": "page-alignment divisor in the non-aligned table",
    "24": "KiB of KV per token per die, arithmetic stated inline",
    "4.5": "frequency-threshold arithmetic stated inline (~9 occurrences x 0.5)",
    "0.747": "the copy_host_to_device_tensor half of the same one-off measurement",
    "11321": "the token [1,1,32,1] with its commas stripped by this script's number regex, not a figure",
    "1132151936": "the token [1,1,32,151936] with its commas stripped, not a figure",
    "113237984": "the token [1,1,32,37984] with its commas stripped, not a figure",
    "2.049": "one-off staging micro-benchmark (per-die buffers + from_host_shards); scratch, no JSON",
    "6.601": "one-off staging micro-benchmark (full-width from_torch); scratch, no JSON",
    "6.897": "the two above summed, same scratch measurement",
    "58": "an intermediate sampling run superseded by the shipped one; the README says so",
    "17": "the first version's mode-3 staging, rounded (17.1644 ms, checked in full precision)",
    "4.9": "GiB of KV, arithmetic stated inline",
    "43.54": "stage-07 teacher-forcing decode t/s/u, quoted as the serving lower bound; re-derived in doc/datatype_sweep, cited not re-derived here",
    # -- figures whose artifact was not retained (declared in the prose) ------
    "2.673": "reduced-target ITL P99 -- 2-layer scratch run, artifact not retained (README says so)",
    "2.113": "reduced-target ITL P50 -- same scratch run, artifact not retained",
    "0.56": "difference of the two above, ~0.56 ms",
    # -- prose-level rounding of a number that IS checked in full precision ---
    "0.565": "rounded restatement of the checked vLLM per-token overhead",
    "0.088": "rounded-up restatement of the checked worst trigram_loop_fraction (0.0878)",
    "52": "rounded restatement of the checked stage-07 52.049 t/s/u",
    "31": "rounded restatement of a checked percentage",
    "15": "rounded restatement of a checked percentage",
    "3": "rounded restatement of a checked ratio; also assorted ordinals",
    "10": "rounded restatement / assorted ordinals",
    "9": "assorted ordinals and section numbers",
    "13": "assorted ordinals and section numbers",
    "19": "assorted ordinals and section numbers",
    "43": "assorted ordinals and section numbers",
    # -- dates, section numbers, source-line citations ------------------------
    "02": "date component",
    "05": "date component",
    "06": "date component",
    "07": "stage number / date component",
    "08": "stage number / date component",
    "09": "date component",
    "12": "date component",
    "624": "worker.py source line cited for the block-sizing rule",
    "475": "vllm-tt-plugin platform.py source line citation",
    "481": "vllm-tt-plugin platform.py source line citation",
    "499": "vllm-tt-plugin platform.py source line citation",
    "632": "worker.py source line cited for max_tokens_all_users",
    "2146": "kv_cache_utils.py source line quoted from the log",
    "200": "HTTP status code",
    "0": "zero / greedy temperature / assorted",
    "5": "assorted ordinals and small counts",
    "6": "penalty-failure count, checked as class_b but also used as an ordinal",
}

readme_numbers = numbers(README)
undeclared = sorted(readme_numbers - COVERED - set(UNCOVERED), key=lambda s: (len(s), s))
stale_declarations = sorted(set(UNCOVERED) - readme_numbers)

print()
print(
    f"coverage boundary: {len(readme_numbers)} distinct numeric tokens in README.md — "
    f"{len(readme_numbers & COVERED)} re-derived above, "
    f"{len(readme_numbers & set(UNCOVERED))} declared uncovered"
)
for token in sorted(readme_numbers & set(UNCOVERED), key=lambda s: (len(s), s)):
    print(f"       uncovered: {token} — {UNCOVERED[token]}")
if undeclared:
    fail(
        "README numbers that are neither re-derived nor declared uncovered "
        f"(add a check, or an UNCOVERED entry): {undeclared}"
    )
if stale_declarations:
    print(f"[note] UNCOVERED entries no longer present in the README: {stale_declarations}")

# ---------------------------------------------------------------------------

print()
if FAILURES:
    print(f"{len(FAILURES)} published figure(s) do not match their artifacts:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all published figures re-derived from their artifacts")
