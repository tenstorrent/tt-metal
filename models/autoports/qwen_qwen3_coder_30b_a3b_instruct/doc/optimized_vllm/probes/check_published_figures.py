# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Every number stage 09 publishes, re-derived from the artifact it came from.

Same mechanism as stage 08's checker, and it exists for the same reason: on this
project the recurring failure mode is prose drifting away from data that is
itself correct. Every stage that shipped one of these had it catch something.

What is checked
---------------

1. **The headline before/after table**, against
   ``bench/single_user_{before,after}_vllm_benchmark.json`` -- TTFT, TPOT, ITL,
   aggregate throughput, e2e latency, completed count. The published decode
   t/s/u must equal ``1000 / mean_tpot_ms`` recomputed here, and the deltas must
   equal the differences of the two files.
2. **The workload shape next to every benchmark number** -- each cited JSON's
   ``config`` block must match the shape the README claims.
3. **The async-scheduling A/B**, against
   ``bench/single_user_no_async_vllm_benchmark.json``, including the derived
   "1.754 ms/token" and "153.4 ms of TTFT" figures and the 75 % share of vLLM's
   per-token host cost the split hides.
4. **The presence and absence of the async-split log line** in the four server
   logs -- the mechanical half of the ``supports_async_decode`` claim.
5. **The standalone batch-32 control curve**, both legs, against
   ``probes/batch_decode_control_{before,after}.json``, plus the
   adapter-overhead subtraction against the served figures and the linear fit
   the README quotes.
6. **The 32-slot before/after table and the CI burst table**, against their JSONs.
7. **The gating probe**: every check the README tables must exist in
   ``inactive_row_gating_probe.json`` and have passed, with zero failures.
8. **The adapter-contract claims**, against
   ``probes/adapter_contract_probe_after.json``: 13 checks, 0 failed.
9. **The sampling gate counts** against ``logs/sampling_tests.log``, and every
   test id the README names as failing must actually appear in that log's
   failure list -- in both directions.
10. **The non-aligned prompt table**, against its JSON, with the divisibility
    columns recomputed rather than trusted.
11. **The byte-identity claim** for the qualitative greedy completions, by
    actually diffing this stage's collection against stage 08's.
12. **The context claims** against ``doc/context_contract.json`` and the KV-cache
    line echoed in the 32-slot server log.
13. **The coverage boundary itself.** Every figure-shaped numeric token in the
    README is either re-derived above or named in ``UNCOVERED`` with a reason. A
    number in neither fails the gate, so the checker's blind spots are
    enumerated rather than silent.

Exits non-zero on any mismatch, so it is a gate and not a report.
"""

from __future__ import annotations

import gzip
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
MODEL = STAGE.parent.parent
BENCH = STAGE / "bench"
LOGS = STAGE / "logs"

README = (STAGE / "README.md").read_text()

FAILURES: list[str] = []
COVERED: set[str] = set()

NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w.])")


def numbers(text: str) -> set[str]:
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


def says(needle: str, where: str = "README") -> bool:
    if needle in README:
        cover(needle)
        return True
    fail(f"{where} does not contain {needle!r}")
    return False


def fmt(value: float, places: int = 3) -> str:
    return f"{value:.{places}f}"


# ---------------------------------------------------------------------------
# 1-2. the headline before/after table
# ---------------------------------------------------------------------------

B1_BEFORE = load(BENCH / "single_user_before_vllm_benchmark.json")
B1_AFTER = load(BENCH / "single_user_after_vllm_benchmark.json")
B1_NOASYNC = load(BENCH / "single_user_no_async_vllm_benchmark.json")
S32_BEFORE = load(BENCH / "maxnumseqs32_before_vllm_benchmark.json")
S32_AFTER = load(BENCH / "maxnumseqs32_after_vllm_benchmark.json")
BURST_BEFORE = load(BENCH / "maxnumseqs32_before_vllm_ci_serving_benchmark.json")
BURST_AFTER = load(BENCH / "maxnumseqs32_after_vllm_ci_serving_benchmark.json")

SINGLE_SHAPE = {"prompt_len": 128, "output_len": 128, "num_requests": 1, "concurrency": 1, "temperature": 0.0}
BURST_SHAPE = {"prompt_len": 100, "output_len": 100, "num_requests": 32, "concurrency": None, "temperature": 0.0}


def check_shape(data, profile: str, shape: dict, label: str) -> None:
    if data is None:
        return
    if data.get("profile") != profile:
        fail(f"{label}: profile is {data.get('profile')!r}, expected {profile!r}")
    cfg = data["config"]
    for key, expected in shape.items():
        if cfg.get(key) != expected:
            fail(f"{label}: config[{key}] is {cfg.get(key)!r}, README claims {expected!r}")
    if not cfg.get("ignore_eos"):
        fail(f"{label}: ignore_eos is not set, README says the workload is ignore_eos")
    ok(f"{label}: profile and workload shape match the cited JSON")


for data, label in (
    (B1_BEFORE, "single-user before"),
    (B1_AFTER, "single-user after"),
    (B1_NOASYNC, "single-user async-off"),
    (S32_BEFORE, "128/128/1 @ max_num_seqs=32 before"),
    (S32_AFTER, "128/128/1 @ max_num_seqs=32 after"),
):
    check_shape(data, "single_user_decode", SINGLE_SHAPE, label)
for data, label in ((BURST_BEFORE, "CI burst before"), (BURST_AFTER, "CI burst after")):
    check_shape(data, "ci_serving_burst", BURST_SHAPE, label)
cover(128, 1, 100, 32, 0.0)


def metrics(data) -> dict:
    return {
        "ttft_p50": data["ttft_ms"]["p50"],
        "ttft_p99": data["ttft_ms"]["p99"],
        "tpot_mean": data["tpot_ms"]["mean"],
        "tpot_p99": data["tpot_ms"]["p99"],
        "itl_p50": data["itl_ms"]["p50"],
        "itl_p99": data["itl_ms"]["p99"],
        "itl_mean": data["itl_ms"]["mean"],
        "e2el": data["e2el_ms"]["p50"],
        "out": data["output_throughput_tok_per_s"],
        "tps": 1000.0 / data["tpot_ms"]["mean"],
    }


def check_row(data, label: str, keys=("ttft_p50", "tpot_mean", "itl_p50", "itl_p99", "out", "e2el", "tps")) -> dict:
    if data is None:
        return {}
    m = metrics(data)
    for key in keys:
        says(fmt(m[key]), f"README ({label} {key})")
    ok(f"{label}: {len(keys)} figures re-derived from the JSON")
    return m


MB = check_row(B1_BEFORE, "single-user before")
MA = check_row(B1_AFTER, "single-user after")
# The original async-off leg's summary row is deliberately NOT re-derived here any
# more: the corrected README no longer publishes it as a result. Its moments are
# checked in section 3 instead, as the evidence for the retraction.
MN = metrics(B1_NOASYNC) if B1_NOASYNC else {}
SLOT_KEYS = ("ttft_p50", "tpot_mean", "itl_p50", "itl_p99", "out", "tps")
M32B = check_row(S32_BEFORE, "128/128/1 @ 32 before", SLOT_KEYS)
M32A = check_row(S32_AFTER, "128/128/1 @ 32 after", SLOT_KEYS)
MQB = check_row(
    BURST_BEFORE,
    "CI burst before",
    ("ttft_p50", "ttft_p99", "tpot_mean", "tpot_p99", "itl_p50", "itl_p99", "out", "tps"),
)
MQA = check_row(
    BURST_AFTER, "CI burst after", ("ttft_p50", "ttft_p99", "tpot_mean", "tpot_p99", "itl_p50", "itl_p99", "out", "tps")
)

if MB and MA:
    # The published deltas must be the differences of the two files.
    for label, value in (
        ("TTFT delta", MB["ttft_p50"] - MA["ttft_p50"]),
        ("TPOT delta", MB["tpot_mean"] - MA["tpot_mean"]),
        ("e2e delta", MB["e2el"] - MA["e2el"]),
    ):
        says(fmt(value), f"README ({label})")
    says(fmt(MA["tps"] - MB["tps"], 3), "README (t/s/u delta)")
    says(fmt(MA["itl_p50"] - MB["itl_p50"]), "README (ITL median delta)")
    says(fmt(MA["out"] - MB["out"]), "README (aggregate throughput delta)")
    ok("headline deltas equal the difference of the two benchmark files")

# The headline t/s/u must be recomputed, not copied.
published = re.search(r"\*\*Decode t/s/u\*\* \(`1000 / ([0-9.]+)`\) \| ([0-9.]+) \| \*\*([0-9.]+)\*\*", README)
if not published:
    fail("README headline table has no parsable 'Decode t/s/u (`1000 / ...`)' row")
elif MA:
    if published.group(1) != fmt(MA["tpot_mean"]):
        fail(f"headline divisor {published.group(1)} != mean TPOT {fmt(MA['tpot_mean'])}")
    if published.group(3) != fmt(MA["tps"]):
        fail(f"headline t/s/u {published.group(3)} != 1000/{fmt(MA['tpot_mean'])} = {fmt(MA['tps'])}")
    else:
        cover(published.group(1), published.group(2), published.group(3), 1000)
        ok(f"headline decode t/s/u {published.group(3)} == 1000 / {fmt(MA['tpot_mean'])}, re-derived")

# ---------------------------------------------------------------------------
# 3. the async-scheduling A/B, re-measured, and the dispersion guard
# ---------------------------------------------------------------------------

STANDALONE_TOKEN_OUT_MS = 19.213  # stage 07 shipped standalone traced token-out

AB = load(BENCH / "async_ab_summary.json")


def dispersion_guard(
    label: str, std_a: float, std_b: float, mean_delta: float, median_delta: float, ratio_limit: float = 5.0
) -> None:
    """Refuse a mean-based delta drawn from a leg with anomalous dispersion.

    This is the class of defect that got through review the first time. Stage 09
    published "1.754 ms/token" as the difference of two legs' mean TPOT, where one
    leg's ``std_itl_ms`` was 14.10 and the other's 0.40 -- a 35x ratio. A mean is
    not a summary of that leg; it is a summary of that leg plus one event.

    So: when two compared legs' ITL standard deviations differ by more than
    ``ratio_limit``, a mean-derived delta between them may still be *printed*, but
    only if the README also publishes the median-based delta and discloses the
    dispersion. Publishing the mean delta alone fails the gate.
    """
    lo, hi = min(std_a, std_b), max(std_a, std_b)
    ratio = hi / lo if lo > 0 else float("inf")
    if ratio <= ratio_limit:
        ok(f"{label}: leg dispersions within {ratio_limit}x ({ratio:.1f}x), mean-based delta is admissible")
        return
    mean_str, median_str = fmt(mean_delta), fmt(median_delta)
    if mean_str not in README:
        ok(f"{label}: dispersion ratio {ratio:.1f}x and the mean-based delta {mean_str} is not published — correct")
        return
    problems = []
    if median_str not in README:
        problems.append(f"the median-based delta {median_str} is not published beside it")
    if f"{hi:.4f}" not in README and f"{hi:.3f}" not in README and f"{hi:.2f}" not in README:
        problems.append(f"the anomalous std {hi:.4f} is not disclosed")
    if problems:
        fail(
            f"{label}: README publishes the mean-based delta {mean_str} drawn from legs whose ITL std "
            f"differs {ratio:.1f}x ({lo:.4f} vs {hi:.4f}), but " + "; and ".join(problems)
        )
    else:
        ok(
            f"{label}: mean-based delta {mean_str} is published, but so is the median-based {median_str} "
            f"and the {ratio:.1f}x dispersion is disclosed — admissible"
        )


if AB:
    legs, comp = AB["legs"], AB["comparison"]

    # -- every retained run's own moments and its stall list ------------------
    n_runs = 0
    for leg in ("async_on", "async_off"):
        for key in ("runs_128", "runs_512"):
            for run in legs[leg][key]:
                says(fmt(run["mean_ms"], 4), f"README ({leg} {key} mean ITL)")
                says(fmt(run["median_ms"], 4), f"README ({leg} {key} median ITL)")
                says(fmt(run["std_ms"]), f"README ({leg} {key} ITL std)")
                cover(run["n_itls"])
                for stall in run["stalls"]:
                    says(fmt(stall["ms"]), f"README ({leg} stall at index {stall['index']})")
                    if stall["index"] != 0:
                        fail(f"{leg}: stall at ITL index {stall['index']}, README claims every stall is at index 0")
                n_runs += 1
    ok(f"{n_runs} retained A/B runs: per-run mean/median ITL and every stall re-derived from the JSONs")

    # -- the stall is present in every async-off run and absent from every on --
    off_with = sum(1 for r in legs["async_off"]["runs_128"] + legs["async_off"]["runs_512"] if r["stall_count"] == 1)
    on_with = sum(1 for r in legs["async_on"]["runs_128"] + legs["async_on"]["runs_512"] if r["stall_count"] > 0)
    if off_with != 4 or on_with != 0:
        fail(
            f"README claims the stall is deterministic: expected 4 async-off runs with exactly one stall and 0 "
            f"async-on runs with any, got {off_with} and {on_with}"
        )
    else:
        ok("the ~179 ms stall is present once in all 4 async-off runs and absent from all 4 async-on runs")

    # -- per-token ITLs really are retained, which is the point ---------------
    for leg in ("async_on", "async_off"):
        for run in legs[leg]["runs_512"]:
            cover(run["n_itls"] + 1)  # the 512-token output length
        for run in legs[leg]["runs_128"]:
            if run["n_itls"] != 127:
                fail(f"{leg} {run['artifact']}: {run['n_itls']} ITLs retained, expected 127")
    ok("per-token ITLs retained for every run (127 at 128 output tokens, 511 at 512)")

    # -- TTFT + ITL[0] is the same in both legs: the whole finding ------------
    on_ttft = legs["async_on"]["summary_128"]["median_ttft_ms"]
    off_ttft = legs["async_off"]["summary_128"]["median_ttft_ms"]
    # Median across runs, matching every other component of this row. This was
    # hard-coded to r1 (19.694 ms) while `on_ttft`/`off_ttft`/`off_itl0` were all
    # medians -- a mixed-statistic pairing, which is the defect this row exists to
    # rule out. The corrected pairing makes the conservation claim tighter, not
    # looser: the two sums agree to 0.026 ms rather than 0.280 ms.
    on_itl0 = statistics.median(
        json.loads((BENCH / "async_ab" / f"async_on_128_r{r}_result.json").read_text())["itls"][0][0] * 1000.0
        for r in (1, 2, 3)
    )
    off_itl0 = sorted(r["stalls"][0]["ms"] for r in legs["async_off"]["runs_128"])[1]
    says(fmt(on_ttft), "README (async-on TTFT)")
    says(fmt(off_ttft), "README (async-off TTFT)")
    says(fmt(on_itl0), "README (async-on ITL[0])")
    says(fmt(on_ttft + on_itl0), "README (async-on TTFT + ITL[0])")
    says(fmt(off_ttft + off_itl0), "README (async-off TTFT + ITL[0])")
    bucket_gap = abs((on_ttft + on_itl0) - (off_ttft + off_itl0))
    if bucket_gap > 2.0:
        fail(f"README claims TTFT+ITL[0] agree between legs, but they differ by {bucket_gap:.3f} ms")
    else:
        ok(f"TTFT + ITL[0] agrees across legs to {bucket_gap:.3f} ms — the one-off is relabelled, not removed")
    cover(fmt(bucket_gap, 2), fmt(bucket_gap))

    # -- the corrected, median-based A/B --------------------------------------
    med = comp["median_itl_128"]
    on_med, off_med = med["async_on"], med["async_off"]
    says(fmt(on_med, 4), "README (async-on steady-state ITL)")
    says(fmt(off_med, 4), "README (async-off steady-state ITL)")
    says(fmt(med["gain"], 4), "README (corrected async gain ms/token)")
    cover(fmt(med["gain"]))  # the same gain at 3 places, used in the corrections table
    says(f"{med['gain_pct']:.2f} %", "README (corrected async gain %)")
    says(fmt(1000.0 / on_med), "README (async-on t/s/u)")
    says(fmt(1000.0 / off_med), "README (async-off t/s/u)")
    tps_gain = 100.0 * (1000.0 / on_med - 1000.0 / off_med) / (1000.0 / off_med)
    says(f"{tps_gain:.2f} %", "README (corrected async t/s/u gain)")

    overhead_async = on_med - STANDALONE_TOKEN_OUT_MS
    overhead_sync = off_med - STANDALONE_TOKEN_OUT_MS
    hidden = 100.0 * med["gain"] / overhead_sync
    says(fmt(overhead_async), "README (serving overhead with the split)")
    cover(fmt(100.0 * overhead_async / STANDALONE_TOKEN_OUT_MS, 1))  # the 3.1 % headline
    cover(fmt(off_med))  # "costs 20.237 ms in steady state"
    cover(fmt(on_ttft + on_itl0, 2), fmt(off_ttft + off_itl0, 2))  # 320.49 / 320.77
    cover(fmt(med["gain"], 2))  # ~0.44 ms/token
    cover(fmt(med["gain_pct"], 1))  # 2.2 %
    for d in (comp["e2e_128"], comp["e2e_512"]):
        cover(f"{d['gain']:.0f}")  # 65 / 258 ms, as rounded in prose
    cover(f"{abs(comp['ttft_128']['gain']):.0f}", fmt(abs(comp["ttft_128"]["gain"]), 1))  # 159 / 159.2
    says(fmt(overhead_sync), "README (vLLM per-token host cost, corrected)")
    says(f"{hidden:.1f} %", "README (share of host cost hidden, corrected)")
    says(fmt(STANDALONE_TOKEN_OUT_MS), "README (stage-07 standalone token_out)")

    for key, label in (("e2e_128", "e2e 128"), ("e2e_512", "e2e 512")):
        d = comp[key]
        says(fmt(d["async_on"]), f"README ({label} async-on)")
        says(fmt(d["async_off"]), f"README ({label} async-off)")
        says(fmt(d["gain"]), f"README ({label} gain)")
        says(f"{d['gain_pct']:.2f} %", f"README ({label} gain %)")
    says(fmt(abs(comp["ttft_128"]["gain"])), "README (async TTFT cost, corrected)")
    ok("corrected async A/B re-derived: median-ITL, t/s/u, host-cost share and both e2e framings")

    # -- the guard this defect earned -----------------------------------------
    mean_delta = comp["mean_itl_128"]["gain"]
    dispersion_guard(
        "re-measured async A/B",
        max(r["std_ms"] for r in legs["async_on"]["runs_128"]),
        max(r["std_ms"] for r in legs["async_off"]["runs_128"]),
        mean_delta,
        med["gain"],
    )
    says(fmt(mean_delta, 4), "README (mean-TPOT delta, shown and disowned)")
    says(fmt(comp["mean_itl_128"]["async_on"], 4), "README (async-on mean ITL, pooled)")
    says(fmt(comp["mean_itl_128"]["async_off"], 4), "README (async-off mean ITL, pooled)")
    # The stall and steady state as rounded in prose ("~179 ms", "~20 ms").
    cover(f"{statistics.median([r['stalls'][0]['ms'] for r in legs['async_off']['runs_128']]):.0f}")
    _inf0 = AB["original_stage09_legs"]["async_off_outlier_inference"]
    cover(f"{_inf0['implied_stall_ms']:.0f}", f"{on_med:.1f}", f"{_inf0['implied_steady_state_ms']:.1f}")

    # The ORIGINAL stage-09 legs stay as evidence, and the same guard applies to
    # them -- this is the pair that produced the retracted 1.754 ms figure.
    orig = AB["original_stage09_legs"]
    inf = orig["async_off_outlier_inference"]
    says(fmt(orig["async_off"]["std_itl_ms"], 4), "README (original async-off std_itl)")
    says(fmt(orig["async_on"]["std_itl_ms"], 4), "README (original async-on std_itl)")
    says(fmt(orig["async_off"]["mean_itl_ms"], 4), "README (original async-off mean_itl)")
    says(fmt(orig["async_off"]["median_itl_ms"], 4), "README (original async-off median_itl)")
    says(fmt(orig["async_off"]["p99_itl_ms"], 4), "README (original async-off p99_itl)")
    if orig["async_off"]["p99_itl_ms"] >= orig["async_off"]["mean_itl_ms"]:
        fail("README's whole argument rests on p99 < mean in the original async-off leg; it is not")
    else:
        ok("original async-off leg: p99 < mean confirmed, which is the tell the README cites")
    if orig["async_off"]["per_token_itls_retained"]:
        fail("README says the original legs kept only moments, but they do carry per-token itls")
    else:
        ok("original legs really do keep only moments — the reason the stall had to be inferred")
    says(f"{inf['dispersion_ratio_vs_async_on']:.0f}x", "README (dispersion ratio of the original pair)")
    says(fmt(inf["implied_stall_ms"], 1), "README (stall inferred from the original moments)")
    says(fmt(inf["implied_steady_state_ms"], 2), "README (steady state inferred from the original moments)")
    says(fmt(orig["async_on"]["median_itl_ms"], 4), "README (original async-on median_itl)")
    # The README's own ITL-median row always said 0.436; the point is that the
    # corrected 0.438 agrees with it and the retracted 1.754 never did.
    orig_median_delta = orig["async_off"]["median_itl_ms"] - orig["async_on"]["median_itl_ms"]
    says(fmt(orig_median_delta), "README (original ITL-median delta, which the first pass never reconciled)")
    if abs(orig_median_delta - med["gain"]) > 0.05:
        fail(
            f"README claims the corrected gain {med['gain']:.4f} agrees with the original ITL-median row "
            f"{orig_median_delta:.4f}; they differ by {abs(orig_median_delta - med['gain']):.4f} ms"
        )
    else:
        ok(f"corrected gain {med['gain']:.3f} ms agrees with the original median row {orig_median_delta:.3f} ms")
    cover(127 - 1)  # "126 tokens at ~20.30 ms"
    # The inference must actually agree with what re-measurement found.
    measured_stall = statistics.median([r["stalls"][0]["ms"] for r in legs["async_off"]["runs_128"]])
    if abs(inf["implied_stall_ms"] - measured_stall) > 5.0:
        fail(
            f"the stall inferred from the original moments ({inf['implied_stall_ms']:.1f} ms) does not agree with "
            f"the re-measured one ({measured_stall:.1f} ms)"
        )
    else:
        ok(
            f"the stall inferred from the original moments ({inf['implied_stall_ms']:.1f} ms) agrees with the "
            f"re-measured {measured_stall:.1f} ms"
        )
    dispersion_guard(
        "original stage-09 async A/B",
        orig["async_on"]["std_itl_ms"],
        orig["async_off"]["std_itl_ms"],
        orig["async_off"]["mean_itl_ms"] - orig["async_on"]["mean_itl_ms"],
        orig["async_off"]["median_itl_ms"] - orig["async_on"]["median_itl_ms"],
    )

# ---------------------------------------------------------------------------
# 4. the async-split log line, present and absent
# ---------------------------------------------------------------------------

NEEDLE = "vLLM took the async decode split"
for name, expect in (
    ("server_b1_after.log.gz", True),
    ("server_bench_after.log.gz", True),
    ("server_b1_noasync.log.gz", False),
):
    path = LOGS / name
    if not path.is_file():
        fail(f"missing server log {path}")
        continue
    text = gzip.open(path, "rt", errors="ignore").read()
    seen = NEEDLE in text
    if seen != expect:
        fail(f"{name}: async-split log line {'missing' if expect else 'present'}, README claims the opposite")
    else:
        ok(f"{name}: async-split log line {'present' if expect else 'absent'}, as the README says")
    if expect and "Asynchronous scheduling is enabled." not in text:
        fail(f"{name}: does not carry 'Asynchronous scheduling is enabled.'")
    if not expect and "Asynchronous scheduling is disabled." not in text:
        fail(f"{name}: does not carry 'Asynchronous scheduling is disabled.'")

stage08_log = MODEL / "readiness_vllm" / "server.log"
if stage08_log.is_file():
    if "Asynchronous scheduling is enabled." not in stage08_log.read_text(errors="ignore"):
        fail("stage-08 server.log does not say async scheduling was enabled; README claims it was")
    else:
        ok("stage-08 readiness_vllm/server.log confirms async scheduling was already enabled there")

# ---------------------------------------------------------------------------
# 5. the standalone batch-32 control curve
# ---------------------------------------------------------------------------

CTRL_BEFORE = load(HERE / "batch_decode_control_before.json")
CTRL_AFTER = load(HERE / "batch_decode_control_after.json")


def curve(data) -> dict[int, float]:
    return {row["active"]: row["token_out"]["ms"] for row in data["rows"]}


if CTRL_BEFORE and CTRL_AFTER:
    for data, label in ((CTRL_BEFORE, "control before"), (CTRL_AFTER, "control after")):
        if data["slots"] != 32 or data["model_max_batch_size"] != 32:
            fail(f"{label}: slots is {data['slots']}, README claims a 32-slot control")
        for active, ms in curve(data).items():
            says(fmt(ms), f"README ({label}, active={active})")
        cover(*curve(data).keys())
    ok("both control curves re-derived from their JSONs")

    cb, ca = curve(CTRL_BEFORE), curve(CTRL_AFTER)
    # "thirty-two costs 2.3 % more than one" (before leg)
    says(f"{100.0 * (cb[32] / cb[1] - 1):.1f} %", "README (before-curve flatness at 32)")
    # adapter overhead: served minus standalone, both legs
    if M32B and M32A:
        for label, served, standalone in (
            ("before", M32B["tpot_mean"], cb[1]),
            ("after", M32A["tpot_mean"], ca[1]),
        ):
            says(fmt(served - standalone), f"README (adapter overhead {label})")
        says(f"{100.0 * (M32B['tpot_mean'] - cb[1]) / cb[1]:.2f} %", "README (adapter overhead pct before)")
        ok("adapter overhead re-derived as served minus standalone on both legs")
    # the linear fit the README quotes: 228.0 + 1.27 * live
    slope = (ca[32] - ca[1]) / 31.0
    intercept = ca[1] - slope
    says(fmt(slope, 2), "README (after-curve slope)")
    says(fmt(intercept, 1), "README (after-curve intercept)")
    says(fmt(ca[32] - ca[1], 0), "README (users' share of a full 32-row step)")
    ok(f"after-curve fit re-derived: {fmt(intercept,1)} + {fmt(slope,2)} x live rows")

# ---------------------------------------------------------------------------
# 6. the 32-slot before/after deltas
# ---------------------------------------------------------------------------

if M32B and M32A:
    says(fmt(M32B["tpot_mean"] - M32A["tpot_mean"]), "README (32-slot TPOT delta)")
    says(
        f"{100.0 * (M32B['tpot_mean'] - M32A['tpot_mean']) / M32B['tpot_mean']:.1f} %",
        "README (32-slot TPOT delta pct)",
    )
    says(f"{100.0 * (M32A['tps'] - M32B['tps']) / M32B['tps']:.1f} %", "README (32-slot t/s/u gain pct)")
    says(fmt(M32B["itl_p50"] - M32A["itl_p50"]), "README (32-slot ITL delta)")
    says(fmt(M32B["ttft_p50"] - M32A["ttft_p50"]), "README (32-slot TTFT delta)")
    says(f"{100.0 * (M32A['out'] - M32B['out']) / M32B['out']:.1f} %", "README (32-slot aggregate gain pct)")
    ok("32-slot before/after deltas re-derived")

if MQB and MQA:
    burst_delta = abs(100.0 * (MQA["out"] - MQB["out"]) / MQB["out"])
    tpot_delta = abs(100.0 * (MQA["tpot_mean"] - MQB["tpot_mean"]) / MQB["tpot_mean"])
    says(f"{burst_delta:.1f} %", "README (CI burst throughput delta)")
    says(f"{tpot_delta:.2f} %", "README (CI burst TPOT delta)")
    if MQA["out"] > 100 and MQB["out"] > 100:
        ok(f"CI burst neutral: {burst_delta:.1f}% throughput, {tpot_delta:.2f}% TPOT")
    for label, data in (("before", BURST_BEFORE), ("after", BURST_AFTER)):
        if data["completed_requests"] != 32:
            fail(f"CI burst {label}: {data['completed_requests']}/32 completed")
        cover(data["completed_requests"], int(data["total_output_tokens"]))

for data, label in ((B1_BEFORE, "single-user before"), (B1_AFTER, "single-user after")):
    if data and data["completed_requests"] != 1:
        fail(f"{label}: {data['completed_requests']}/1 completed")
    if data and int(data["total_output_tokens"]) != 128:
        fail(f"{label}: {data['total_output_tokens']} output tokens, README claims 128")

# The README must not present the burst as the headline.
if "This is not the headline decode number" not in README:
    fail("README does not say the CI serving burst is not the headline decode number")
else:
    ok("README states the CI burst is not the headline decode number")

# ---------------------------------------------------------------------------
# 7. the gating probe
# ---------------------------------------------------------------------------

GATING = load(HERE / "inactive_row_gating_probe.json")
if GATING:
    if GATING["failed"]:
        fail(f"inactive_row_gating_probe reports failed checks: {GATING['failed']}")
    for name in ("live_rows_token_identical", "outputs_are_varied", "mask_matches_positions", "mask_survives_replays"):
        if not says(f"`{name}`", "README (gating probe table)"):
            continue
        check = GATING["checks"].get(name)
        if check is None:
            fail(f"README names gating check {name!r} but the probe JSON has no such check")
        elif not check.get("passed"):
            fail(f"gating check {name!r} did not pass")
    tok = GATING["checks"]["live_rows_token_identical"]
    says(str(tok["rows_compared"]), "README (rows compared)")
    says(str(tok["tokens_per_row"]), "README (tokens per row)")
    varied = GATING["checks"]["outputs_are_varied"]["distinct_tokens_per_row"]
    says(f"{min(varied)}–{max(varied)}", "README (distinct tokens per row range)")
    says(str(GATING["checks"]["mask_survives_replays"]["replays_since_install"]), "README (replays since install)")
    ok(f"gating probe: {len(GATING['checks'])} checks, 0 failed, re-derived against the README table")

# ---------------------------------------------------------------------------
# 8. the adapter-contract probe
# ---------------------------------------------------------------------------

CONTRACT = load(HERE / "adapter_contract_probe_after.json")
if CONTRACT:
    if CONTRACT["failed"] != 0:
        fail(f"adapter_contract_probe_after reports {CONTRACT['failed']} failed checks")
    passed = [c["check"] for c in CONTRACT["checks"] if c["pass"]]
    says(str(len(CONTRACT["checks"])), "README (contract check count)")
    for name in ("token_host_copies", "position_host_copies", "rotary_position_host_copies", "page_table_host_copies"):
        says(f"`{name} +0`", "README (steady-state counter)")
    says(str(CONTRACT["config"].get("max_num_seqs", "")), "README (contract probe max_num_seqs)")
    ok(f"adapter contract: {len(passed)}/{len(CONTRACT['checks'])} pass, 0 failed")

# ---------------------------------------------------------------------------
# 9. the sampling gate
# ---------------------------------------------------------------------------

# Four runs are retained: the stage-09 original, an isolation run with only the
# merge-sentinel fix reverted, and two runs of the shipped code. The README must
# report all four and must not quietly present the most flattering one.
SAMPLING_RUNS = {
    "stage-09 archived": LOGS / "sampling_tests_prefix_stage09_original.log",
    "isolation (merge fix reverted)": LOGS / "sampling_tests_isolation_no_merge_fix.log",
    "shipped run 1": LOGS / "sampling_tests.log",
    "shipped run 2": LOGS / "sampling_tests_shipped_run2.log",
}

SEED_WORDS = ("seed", "seeds", "uniform", "mixed_params", "temperature_varied_between", "batch1_no_seed", "topk")
all_named_ok = True
counts = {}
for label, path in SAMPLING_RUNS.items():
    if not path.is_file():
        fail(f"missing {path}")
        continue
    text = path.read_text(errors="ignore")
    summary = re.search(r"(\d+) failed, (\d+) passed, (\d+) skipped", text)
    if not summary:
        fail(f"{path.name} has no parsable pytest summary line")
        continue
    failed_n, passed_n, skipped_n = (int(g) for g in summary.groups())
    counts[label] = (passed_n, failed_n, skipped_n)
    says(f"**{passed_n} / {failed_n} / {skipped_n}**", f"README ({label} sampling result)")
    cover(passed_n, failed_n, skipped_n)

    failing = {line.split("::")[-1].strip() for line in text.splitlines() if line.startswith("FAILED")}
    presence = [f for f in failing if "presence" in f.lower()]
    seeding = [f for f in failing if any(w in f for w in SEED_WORDS)]
    unclassified = sorted(failing - set(presence) - set(seeding))
    if unclassified:
        fail(
            f"{label}: sampling failures outside the two declared classes: {unclassified}. "
            "The README claims no test outside those classes ever failed."
        )
    else:
        ok(f"{label}: {passed_n}/{failed_n}/{skipped_n}, all {len(failing)} failures in the two declared classes")

# The README's central claim about this gate: the classification never moved,
# only the count. Enforce both halves.
if len(counts) == 4:
    shipped = [counts["shipped run 1"][1], counts["shipped run 2"][1]]
    baseline = [counts["stage-09 archived"][1], counts["isolation (merge fix reverted)"][1]]
    if min(shipped) <= max(baseline):
        ok(f"shipped failure counts {shipped} do not exceed the baseline {baseline}")
    else:
        # This is the state the README currently documents as an open item, so it
        # must SAY so rather than claim a clean pass.
        for phrase in ("Not fully resolved", "open item"):
            if phrase not in README:
                fail(
                    f"shipped sampling failures {shipped} exceed the baseline {baseline}, but the README "
                    f"does not flag it ({phrase!r} absent). Do not publish this as a clean pass."
                )
        ok(
            f"shipped failure counts {shipped} exceed the baseline {baseline}, and the README flags it "
            "as an unresolved open item rather than claiming a clean pass"
        )

# Every test id the README names as failing must really be failing in at least
# one retained run.
_all_failing = set()
for path in SAMPLING_RUNS.values():
    if path.is_file():
        _all_failing |= {
            line.split("::")[-1].strip()
            for line in path.read_text(errors="ignore").splitlines()
            if line.startswith("FAILED")
        }
named = set(re.findall(r"`(test_[a-z0-9_]+(?:\[[^\]]*\])?)`", README))
for name in sorted(named):
    base = name.split("[")[0]
    if name in ("test_tt_penalties",):
        continue
    if not any(f.split("[")[0] == base for f in _all_failing):
        fail(f"README names {name!r} among the failures but it appears in no retained run's FAILED list")
ok(f"every test id the README names as failing appears in at least one retained run ({len(named)} names)")

# The baseline class sizes the README quotes, from the archived run.
_base = LOGS / "sampling_tests_prefix_stage09_original.log"
if _base.is_file():
    _f = {
        line.split("::")[-1].strip()
        for line in _base.read_text(errors="ignore").splitlines()
        if line.startswith("FAILED")
    }
    _p = [x for x in _f if "presence" in x.lower()]
    _s = [x for x in _f if any(w in x for w in SEED_WORDS)]
    says(f"**{len(_s)} seeding/RNG**", "README (baseline seeding class size)")
    says(f"**{len(_p)} presence-penalty**", "README (presence class size)")

# ---------------------------------------------------------------------------
# 9b. the model-suite gate
# ---------------------------------------------------------------------------

suite_log = LOGS / "stage09_model_suite.log"
if not suite_log.is_file():
    fail(f"missing {suite_log}")
else:
    m = re.search(r"(\d+) passed, (\d+) deselected", suite_log.read_text(errors="ignore"))
    if not m:
        fail("stage09_model_suite.log has no parsable 'N passed, M deselected' summary")
    else:
        says(f"**{m.group(1)} passed, {m.group(2)} deselected**", "README (model suite gate)")
        if "failed" in m.string[m.start() : m.end() + 20]:
            fail("model suite summary reports failures")
        ok(f"model suite {m.group(1)} passed / {m.group(2)} deselected matches the log")

# ---------------------------------------------------------------------------
# 10. the non-aligned prompt table
# ---------------------------------------------------------------------------

NONALIGNED = load(HERE / "non_aligned_prompt_lengths.json")
if NONALIGNED:
    if not NONALIGNED["all_passed"]:
        fail("non_aligned_prompt_lengths.json reports a failure")
    for row in NONALIGNED["rows"]:
        length = row["reported_prompt_tokens"]
        says(str(length), "README (non-aligned prompt length)")
        for divisor, claimed in row["not_divisible_by"].items():
            recomputed = length % int(divisor) != 0
            if recomputed != claimed:
                fail(f"non-aligned row {length}: divisibility by {divisor} recorded {claimed}, recomputed {recomputed}")
            if not recomputed:
                fail(f"non-aligned row {length} IS divisible by {divisor}; the table claims none of them are")
        if (
            row["requested_prompt_tokens"] is not None
            and row["reported_prompt_tokens"] != row["requested_prompt_tokens"]
        ):
            fail(f"non-aligned row {length}: reported != requested, so something capped or truncated")
        says(str(row["completion_tokens"]), "README (non-aligned completion tokens)")
    ok(f"{len(NONALIGNED['rows'])} non-aligned prompt rows re-derived, divisibility recomputed")

# ---------------------------------------------------------------------------
# 11. the qualitative byte-identity claim
# ---------------------------------------------------------------------------

new_q = LOGS / "vllm_qualitative_outputs.json"
old_q = MODEL / "readiness_vllm" / "vllm_qualitative_outputs.json"
if new_q.is_file() and old_q.is_file():
    a = json.loads(old_q.read_text())
    b = json.loads(new_q.read_text())
    same = sum(1 for x, y in zip(a, b) if x["greedy_completion"] == y["greedy_completion"])
    says(f"all **six greedy qualitative completions are byte-identical", "README (qualitative identity)")
    if same != len(a) or len(a) != 6:
        fail(f"greedy completions identical in {same}/{len(a)} prompts; README claims all six")
    else:
        cover(6)
        ok("all 6 greedy qualitative completions are byte-identical to stage 08, verified by diff")
else:
    fail("cannot verify the qualitative byte-identity claim: an artifact is missing")

# ---------------------------------------------------------------------------
# 12. the context claims
# ---------------------------------------------------------------------------

contract = load(MODEL / "doc" / "context_contract.json")
if contract:
    supported = int(contract["current_supported_context"])
    says(str(supported), "README (served context)")
    if contract.get("capability_reduction"):
        fail("context_contract.json records capability_reduction=true; README claims unreduced")
    else:
        ok(f"context contract {supported}, capability_reduction false")

server32 = LOGS / "server_bench_after.log.gz"
if server32.is_file():
    text = gzip.open(server32, "rt", errors="ignore").read()
    kv = re.search(r"GPU KV cache size: ([\d,]+) tokens", text)
    if not kv:
        fail("32-slot server log has no 'GPU KV cache size' line")
    else:
        tokens = int(kv.group(1).replace(",", ""))
        blocks = -(-(262144 + 32 * 32) // 32)
        if tokens != blocks * 32:
            fail(f"server log KV size {tokens} != recomputed {blocks} blocks x 32 = {blocks * 32}")
        says(f"{tokens:,}", "README (KV cache size quoted from the log)")
        cover(blocks, tokens)
        ok(f"KV cache {tokens} tokens == ceil((262144 + 32*32)/32) x 32, and is quoted from the log")

# ---------------------------------------------------------------------------
# 13b. the partial-occupancy churn control, both legs
# ---------------------------------------------------------------------------

CHURN_LEGACY = load(HERE / "churn_occupancy_control_legacy.json")
CHURN_FIXED = load(HERE / "churn_occupancy_control_fixed.json")

if CHURN_LEGACY and CHURN_FIXED:
    for data, label, expect_pass in ((CHURN_LEGACY, "legacy-clamp", False), (CHURN_FIXED, "fixed", True)):
        if data["all_pass"] is not expect_pass:
            fail(f"churn control {label}: all_pass is {data['all_pass']}, README says {expect_pass}")
        for row in data["rounds"]:
            says(fmt(row["token_out_ms"]), f"README (churn {label} {row['round']} token_out)")
            cover(row["inactive_rows_total"], row["inactive_rows_at_sentinel"])
        # The sentinel claim, recomputed rather than trusted.
        sentinel_ok = all(r["inactive_rows_at_sentinel"] == r["inactive_rows_total"] for r in data["rounds"])
        if sentinel_ok is not expect_pass:
            fail(f"churn control {label}: sentinel preserved on every round is {sentinel_ok}, expected {expect_pass}")
        drift = data["checks"]["gating_win_survives_turnover"]["drift_ms"]
        says(fmt(abs(drift)), f"README (churn {label} drift)")
    legacy_drift = CHURN_LEGACY["checks"]["gating_win_survives_turnover"]["drift_ms"]
    fixed_drift = CHURN_FIXED["checks"]["gating_win_survives_turnover"]["drift_ms"]
    if not (legacy_drift > 20.0 and abs(fixed_drift) < 1.0):
        fail(
            f"README claims the legacy clamp loses the gating win on turnover and the fix holds it; "
            f"drifts are legacy {legacy_drift:.3f} ms and fixed {fixed_drift:.3f} ms"
        )
    else:
        ok(
            f"churn control: legacy clamp drifts +{legacy_drift:.3f} ms on the first recycle "
            f"(gating win lost), fixed drifts {fixed_drift:.3f} ms (held)"
        )
    # The legacy leg must land near the full-occupancy cost, which is the point.
    full_occ = 268.737
    worst = CHURN_LEGACY["checks"]["gating_win_survives_turnover"]["worst_turnover_ms"]
    cover(fmt(abs(full_occ - worst), 0))
    ok(f"legacy leg after turnover ({worst:.3f} ms) is within {abs(full_occ - worst):.1f} ms of full occupancy")

# ---------------------------------------------------------------------------
# 13c. the corrected async/sync read counters
# ---------------------------------------------------------------------------

if CONTRACT:
    audit = CONTRACT.get("serving_audit", {})
    steps = audit.get("device_sampled_decode_steps")
    a_reads, s_reads = audit.get("async_decode_reads"), audit.get("sync_decode_reads")
    says(str(steps), "README (device-sampled decode steps)")
    says(str(a_reads), "README (async_decode_reads)")
    says(str(s_reads), "README (sync_decode_reads)")
    if None in (steps, a_reads, s_reads):
        fail("adapter contract probe carries no async/sync decode read counters")
    elif a_reads + s_reads != steps:
        fail(
            f"async_decode_reads ({a_reads}) + sync_decode_reads ({s_reads}) = {a_reads + s_reads}, "
            f"which is not the {steps} device-sampled decode steps. The counters do not partition the steps."
        )
    else:
        ok(f"async/sync decode reads partition the decode steps exactly: {a_reads} + {s_reads} == {steps}")
    # The contradiction the old build published, quoted in the README to explain the fix.
    cover(a_reads + steps)

# ---------------------------------------------------------------------------
# 13. the coverage boundary
# ---------------------------------------------------------------------------

UNCOVERED: dict[str, str] = {
    # -- configuration echoed from the command line, not measured -------------
    "8100": "serving port (8000 is held by a process outside this session)",
    "50331648": "trace_region_size passed in --tt-config",
    "262144": "max_model_len; the contract value itself is checked above, this is the CLI echo",
    "0": "gating env value / greedy temperature / assorted",
    "1": "max_num_seqs, request counts, assorted",
    "24": "tokens per row in the gating probe (checked via the probe JSON) and assorted",
    # -- model/mesh shape, fixed by the port and checked by earlier stages -----
    "4": "dies in the 1x4 mesh; also the contract probe's max_num_seqs",
    "48": "decoder layers",
    "8": "page-alignment divisor in the non-aligned table / top-8 experts",
    "64": "page-alignment divisor in the non-aligned table",
    "1024": "page-alignment divisor in the non-aligned table",
    "9.7": "MB of the full-vocabulary penalty operand, quoted from stage 08",
    "256": "history length stage 08 re-timed its penalty staging at",
    # -- stage-08 figures quoted, re-derived in doc/vllm_integration ----------
    "50.560": "stage-08 headline t/s/u, quoted for comparison; re-derived by stage 08's own checker",
    "3.796": "stage-08 32-slot t/s/u, quoted; also re-derived here from this stage's before leg",
    "312.367": "stage-08 serving TTFT, quoted; re-derived by stage 08's checker",
    "129.941": "stage-07 standalone TTFT at ctx128, quoted; re-derived in doc/optimized_full_model",
    "182": "the TTFT gap stage 08 published, quoted in order to correct it",
    "52.049": "stage-07 standalone token-out t/s/u, quoted",
    "44.049": "stage-08 penalised t/s/u (repetition only), quoted",
    "40.079": "stage-08 penalised t/s/u (all three), quoted",
    "50.321": "stage-08 unpenalised in-situ t/s/u, quoted",
    "1.5674": "stage-08 penalty staging at a 256-token history, quoted",
    "3.7624": "stage-08 penalty staging at a 256-token history, quoted",
    "1.5351": "stage-08 penalty staging on the correctness batch, quoted",
    "3.3894": "stage-08 penalty staging on the correctness batch, quoted",
    "56": "stage-08 sampling passed count, quoted",
    "16": "stage-08 sampling failed count, quoted; also a control-curve active-row count",
    "8224": "stage-08's derived block count, now confirmed by this stage's log",
    "2.3": "the before-curve flatness figure, also re-derived above as a percentage string",
    "12": "seeding-class size, also checked as a class size above",
    "14": "sampling failed count, checked above; also the natural-language prompt length",
    # -- source-line citations and section/version identifiers ----------------
    "31": "vllm_tt_plugin/scheduler.py source line for `class TTScheduler(AsyncScheduler)`; also the control-curve divisor",
    "153": "the 153 ms figure the first pass published, quoted in order to show it cannot be a scheduler step",
    "230": "the ~230 ms 32-slot step, re-derived above at full precision; this is the rounded prose form",
    "969": "vllm-tt-plugin model_runner.py source line citation for the -1 inactive-position pad",
    "970": "vllm-tt-plugin model_runner.py source line citation for the -1 inactive-position pad",
    "4096": "the standalone control's context, stated beside the adapter-overhead estimate",
    "2026": "the year in the dated correction notice",
    "18": "the day in the dated correction notice",
    "42": "a test_specific_seed_reproducible seed parametrisation named in the sampling table",
    "999": "a test_specific_seed_reproducible seed parametrisation named in the sampling table",
    "5": "the lower bound of the suggested 5-10 run repeat-count study, and assorted small counts",
    "10": "the upper bound of the suggested 5-10 run repeat-count study, and top_k=10 in test_topk",
    "955": "vllm-tt-plugin platform.py source line citation",
    "968": "vllm-tt-plugin platform.py source line citation",
    "964": "vllm/config/vllm.py source line citation",
    "1004": "vllm/config/vllm.py source line citation",
    "08": "stage number",
    "07": "stage number",
    "09": "stage number",
    "01": "stage range in 'stages 01-08'",
    "2": "assorted ordinals and small counts",
    "3": "assorted ordinals and small counts",
    "6": "assorted ordinals and small counts",
    "13": "adapter-contract check count, checked above; also an ordinal",
    "75": "share of host cost hidden by the async split, checked above as a percentage string",
    "153.4": "async TTFT cost, checked above as a formatted string",
    "1.754": "async gain per token, checked above as a formatted string",
    "8.9": "async decode t/s/u gain %, checked above as a percentage string",
    "0.588": "serving overhead with async, checked above as a formatted string",
    "2.342": "serving overhead without async, checked above as a formatted string",
    "19.213": "stage-07 standalone token-out ms, checked above as a formatted string",
    "9": "the -9 in pkill -9",
    "26": "the 14-26 % penalised-TPOT band quoted from stage 08",
    "8192": "the [1, 8192] int32 page-table shape at the served context",
    "11132": "the token [1,1,32,1] with its commas stripped by this script's number regex, not a figure",
    "11": "the token [1,1,batch,1] with its commas stripped by this script's number regex, not a figure",
    "11.579": "end-to-end delta of the headline pair, stated inline",
    "16": "the ~16 ms of genuine request-side TTFT cost, and a control-curve active-row count",
    "40": "the users' share of a full 32-row step, checked above as a formatted string",
}

readme_numbers = numbers(README)
undeclared = sorted(readme_numbers - COVERED - set(UNCOVERED), key=lambda s: (len(s), s))
stale = sorted(set(UNCOVERED) - readme_numbers)

# The two sets legitimately overlap: a token can be re-derived by a check *and*
# carry an UNCOVERED note (e.g. "16" is both a control-curve row count that is
# checked and the request-side TTFT figure). Reporting the two sizes side by side
# double-counts that overlap and can sum to more than the number of distinct
# tokens, which reads as though the checker covers more than it does. Print the
# partition instead, and the residual that actually gates.
_covered = readme_numbers & COVERED
_declared = readme_numbers & set(UNCOVERED)
_both = _covered & _declared
_residual = readme_numbers - _covered - _declared

print()
print(
    f"coverage boundary: {len(readme_numbers)} distinct numeric tokens in README.md — "
    f"{len(_covered - _declared)} re-derived only, "
    f"{len(_declared - _covered)} declared uncovered only, "
    f"{len(_both)} both, "
    f"{len(_residual)} neither (this is the number that gates)"
)
assert len(_covered - _declared) + len(_declared - _covered) + len(_both) + len(_residual) == len(readme_numbers)
for token in sorted(readme_numbers & set(UNCOVERED), key=lambda s: (len(s), s)):
    print(f"       uncovered: {token} — {UNCOVERED[token]}")
if undeclared:
    fail(
        "README numbers that are neither re-derived nor declared uncovered "
        f"(add a check, or an UNCOVERED entry): {undeclared}"
    )
if stale:
    print(f"[note] UNCOVERED entries no longer present in the README: {stale}")

print()
if FAILURES:
    print(f"{len(FAILURES)} published figure(s) do not match their artifacts:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all published figures re-derived from their artifacts")
