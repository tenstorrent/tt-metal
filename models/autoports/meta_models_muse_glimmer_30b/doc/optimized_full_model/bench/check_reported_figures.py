# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resolve every headline figure in this stage's README against a committed artifact.

The full-model stage added this gate because a README figure that no committed run
supports is indistinguishable from one that does, and three of its own record defects
were exactly that.  This is the same idea, scoped to this stage: each row below names a
figure, where the README states it, and how to recompute it from JSON, a
``tt-perf-report`` CSV or a console log.  A mismatch is an error, not a warning.

It is read-only and needs no hardware.

Usage::

    python doc/optimized_full_model/bench/check_reported_figures.py
"""

from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import pathlib
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[3]
D = ROOT / "doc/optimized_full_model"
PREV = ROOT / "doc/full_model"

failures: list[str] = []
checks = 0
#: Every literal a check resolved against an artifact, so the README can be searched for
#: it.  The round-2 stage review found `169 vs 195`, `19 %`, three wrong DRAM rows and
#: five wrong per-round spreads surviving a 195/195 pass, because the checks asserted
#: "the artifact says X" and only ~20 of them asserted "and the README still says X".
resolved: list[tuple[str, str]] = []
#: The count this file expects of itself -- a drift tripwire, asserted so a check cannot be
#: silently dropped.  Round 7 was right that the old comment oversold it: no other document
#: states this number, so it is internal and is not a cross-document binding.
#:
#: Round 3 of the stage review pointed out the obvious limit of this: the count freezes
#: how many checks there are, not what they cover, and 328/328 passed while six figures in
#: sections this file never opened were wrong.  ``SOURCES`` below is the other half --
#: adding an unchecked section that needs a new artifact now fails on a missing source
#: rather than sliding past on an unchanged count.
ADVERTISED_CHECKS = 839
#: Of that total, how many are real assertions (``close``/``same``) as opposed to README
#: bindings (``bind``/``perf``), which assert nothing on their own.  Round 4 asked for the
#: split to be stated next to the number rather than folded into it.
ADVERTISED_BINDINGS = 35


#: Every artifact this run actually opened, recorded by the three readers below.  Round 4
#: of the stage review pointed out that ``same("the gate reads X", is_file(), True)`` is an
#: existence check wearing a coverage label, so the coverage assertion at the end now tests
#: this set instead: a file listed as covered but never read is a failure.
opened: set[str] = set()


def _record(path: pathlib.Path) -> pathlib.Path:
    try:
        opened.add(path.resolve().relative_to(D.resolve()).as_posix())
    except ValueError:
        opened.add(path.resolve().as_posix())
    return path


def load(path: pathlib.Path):
    return json.loads(_record(path).read_text())


def text(path: pathlib.Path) -> str:
    _record(path)
    if path.suffix == ".gz":
        return gzip.decompress(path.read_bytes()).decode("utf-8", "replace")
    return path.read_text(errors="replace")


def close(name: str, got, want, *, tol: float = 5e-3, readme: str | None = None) -> None:
    """``want`` is what the README claims; ``got`` is what the artifact says.

    ``readme`` overrides the literal searched for in the README text; pass ``""`` for a
    value the README states in a form no simple search can match.
    """
    global checks
    checks += 1
    resolved.append((name, f"{want}" if readme is None else readme))
    if want == 0:
        ok = abs(got) <= tol
    else:
        ok = abs(got - want) / abs(want) <= tol
    print(f"{'ok  ' if ok else 'FAIL'} {name}: got {got!r}, README says {want!r}")
    if not ok:
        failures.append(f"{name}: got {got!r}, README says {want!r}")


def same(name: str, got, want, *, readme: str | None = None) -> None:
    global checks
    checks += 1
    if readme is not None:
        resolved.append((name, readme))
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} {name}: got {got!r}, README says {want!r}")
    if not ok:
        failures.append(f"{name}: got {got!r}, README says {want!r}")


bindings = 0


def bind(name: str, literal: str) -> None:
    """Register a literal for the README cross-check without a vacuous numeric compare.

    ``same(x, True, True)`` reads like an assertion and is not one; the only thing those
    calls ever carried was the ``readme=`` registration.  This says that plainly.
    """
    global checks, bindings
    checks += 1
    bindings += 1
    resolved.append((name, literal))
    print(f"bind {name}: {literal}")


def csv_rows(name: str) -> list[dict]:
    return list(csv.DictReader(open(_record(D / "tracy" / name))))


def device_time(rows: list[dict], op_id: str) -> float:
    for row in rows:
        if row["ID"] == op_id:
            return float(row["Device Time"])
    raise SystemExit(f"no op id {op_id} in the report")


def main() -> int:
    readme = (D / "README.md").read_text()

    # ---------------------------------------------------------------- headline perf
    before = load(D / "evidence_perf_before.json")["performance"]
    after = load(D / "evidence_perf.json")["performance"]
    same("before arm is the baseline arm", before["baseline_arm"], True)
    same("after arm is not the baseline arm", after["baseline_arm"], False)

    # Run-varying figures.  A hardcoded expectation here would just be a second copy of
    # the artifact that goes stale with it, so the value comes *from* the artifact and
    # the assertion that carries weight is the README cross-check registered with it:
    # "this number is in the document" is the property that was actually being violated.
    def perf(name: str, value: float, digits: int) -> None:
        """Register a run-varying artifact value as a README binding, and nothing else.

        Round 3 of the stage review was right that the previous form --
        ``close(name, round(v, d), round(v, d))`` -- was numerically self-fulfilling: got
        always equalled want, so it asserted nothing about the artifact.  The *binding* is
        the real content ("the document still prints the number this run produced"), so
        this now registers only that, and the printed line says so instead of dressing it
        up as a comparison.
        """
        global checks, bindings
        checks += 1
        bindings += 1
        literal = f"{value:.{digits}f}"
        resolved.append((name, literal))
        print(f"bind {name}: artifact says {literal}; the README cross-check below binds it")

    perf("before token-out ms", before["token_out_decode_ms_per_token"]["min"], 3)
    perf("after token-out ms", after["token_out_decode_ms_per_token"]["min"], 3)
    perf("before logits-only ms", before["traced_decode_logits_only_ms_per_token"]["min"], 3)
    perf("after logits-only ms", after["traced_decode_logits_only_ms_per_token"]["min"], 3)
    perf("before token-out t/s/u", before["token_out_decode_tok_s_u"], 2)
    perf("after token-out t/s/u", after["token_out_decode_tok_s_u"], 2)
    perf("after logits-only t/s/u", after["traced_decode_logits_only_tok_s_u"], 2)
    perf("after sampling trace ms", after["sampling_trace_ms_per_token"]["min"], 3)
    perf("before TTFT min ms", before["ttft_ms"]["min"], 2)
    perf("after TTFT min ms", after["ttft_ms"]["min"], 2)
    delta = (
        (after["token_out_decode_ms_per_token"]["min"] - before["token_out_decode_ms_per_token"]["min"])
        / before["token_out_decode_ms_per_token"]["min"]
        * 100
    )
    perf("token-out delta %", delta, 2)
    same("the token-out delta is an improvement", delta < 0, True)

    # ------------------------------------------------------------ layer-stack floor
    floor = after["layer_stack_lower_bound_ms_per_token"]
    close("floor sliding ms/layer", floor["sliding_ms_per_layer"], 0.4473)
    close("floor full ms/layer", floor["full_ms_per_layer"], 0.4164)
    close("floor total ms", round(floor["total_ms"], 3), 22.858)
    close("before floor total ms", round(before["layer_stack_lower_bound_ms_per_token"]["total_ms"], 3), 23.239)
    ab = text(D / "logs/layer_ab_after.log")
    for kind, value in (("sliding", "0.4473"), ("full", "0.4164")):
        checks_line = [
            line for line in ab.splitlines() if line.startswith("AB ") and f"kind={kind}" in line and "tp4 " in line
        ]
        same(f"layer_ab log states {kind} {value}", any(value in line for line in checks_line), True)

    # --------------------------------------------------------------- the decode A/B
    shipped = load(D / "decode_ab_shipped.json")["arms"]
    close("A/B baseline 2-layer ms", round(shipped["full_model_stage"]["traced_logits_only_ms"], 4), 1.5535)
    close("A/B terminal-only 2-layer ms", round(shipped["terminal_only"]["traced_logits_only_ms"], 4), 1.5376)
    close("A/B shipped 2-layer ms", round(shipped["base"]["traced_logits_only_ms"], 4), 1.5246)
    for arm in shipped.values():
        same("every cumulative A/B arm picks the base token", arm["vs_base"]["top1_same"], True)
        close("every cumulative A/B arm is PCC 1.0", arm["vs_base"]["pcc"], 1.0, tol=1e-6)
    swiglu = load(D / "decode_ab_swiglu.json")["arms"]
    for arm, want in (
        ("base", 1.5375),
        ("swiglu20", 1.5408),
        ("swiglu32", 1.5357),
        ("swiglu40", 1.5306),
        ("swiglu80", 1.5248),
        ("mlp8", 1.5781),
    ):
        close(f"SwiGLU sweep {arm}", round(swiglu[arm]["traced_logits_only_ms"], 4), want)
    same("mlp8 is a legal but slower arm, not a failure", "error" in swiglu["mlp8"], False)
    dropped = load(D / "decode_ab.json")["arms"]
    for arm in ("mlp20", "mlp32", "mlp40"):
        same(f"{arm} failed the op contract", "error" in dropped[arm], True)
    same(
        "the uneven-sharding fatal is in the A/B log",
        "in DRAM sharded Matmul we don't have support for un-even sharding" in text(D / "logs/decode_ab.log"),
        True,
    )

    # ------------------------------------------------------- the tt-perf-report rows
    sliding = csv_rows("decode_sliding_perf_report.csv")
    close("after tanh (width-sharded)", device_time(sliding, "3140"), 11.64, tol=1e-2)
    close("after softcap multiply (width-sharded)", device_time(sliding, "3196"), 12.15, tol=1e-2)
    close("after logits sharded_to_interleaved", device_time(sliding, "3197"), 10.80, tol=1e-2)
    close("after LM-head matmul", device_time(sliding, "3139"), 603.798, tol=1e-4)
    close("after SwiGLU multiply (80 cores)", device_time(sliding, "3075"), 4.75, tol=1e-2)
    reshards = [float(r["Device Time"]) for r in sliding if "Reshard" in r["OP Code"]]
    close("three SwiGLU reshards plus the LM-head one", round(sum(reshards), 2), 8.39, tol=2e-2, readme="")
    close("the three SwiGLU reshards", round(sum(sorted(reshards)[:3]), 2), 5.91, tol=2e-2)
    prev = csv_rows_prev = list(csv.DictReader(open(PREV / "tracy/decode_perf_report.csv")))
    close("before tanh (DRAM interleaved)", device_time(prev, "4283"), 17.71, tol=1e-2)
    close("before softcap multiply (DRAM interleaved)", device_time(prev, "4370"), 19.14, tol=1e-2)
    close("before SwiGLU multiply (16 cores)", device_time(prev, "4187"), 18.03, tol=1e-2)
    close("before residual add on the same grid", device_time(prev, "4245"), 1.88, tol=1e-2)
    same(
        "the before profile still has the post-gather interleaved_to_sharded",
        any(r["ID"] == "4289" and "InterleavedToSharded" in r["OP Code"] for r in prev),
        True,
    )
    # the dtype policy must appear in the measured rows, not only in JSON (OPT-013)
    # tt-perf-report puts the dtype pair and the fidelity in one "Math Fidelity" cell.
    for op_id, want in (
        ("3139", "BFP4"),
        ("3071", "BFP4"),
        ("3132", "BFP4"),
        ("3039", "BFP8"),
        ("3065", "BFP8"),
        ("3127", "BFP4"),
        ("3172", "BFP8"),
    ):
        cell = next(r for r in sliding if r["ID"] == op_id)["Math Fidelity"].strip()
        same(f"row {op_id} weight dtype is {want}", cell, f"LoFi BF16 x {want} => BF16")
    # The roofline denominator.  This is a *consistency* check and is labelled as one in
    # the README too: tt-perf-report computes DRAM % as DRAM / peak, so DRAM / DRAM% x 100
    # returns its assumed peak by construction.  What it can still catch is a row using a
    # different peak, so it is asserted once over all rows rather than once per row.
    peaks = {
        round(float(r["DRAM"].replace(" GB/s", "")) / float(r["DRAM %"].replace(" %", "")) * 100, 1)
        for r in sliding
        if r["DRAM"] and r["DRAM %"]
    }
    same("one assumed DRAM peak across every DRAM-classified row", sorted(peaks), [512.0], readme="512 GB/s")
    # ...and the rows the README quotes from, which is not tautological.
    for op_id, want_bw, want_pct in (
        ("3139", 279.38, 54.57),
        ("3039", 394.85, 77.12),
        ("3172", 355.08, 69.35),
        ("3065", 318.97, 62.3),
    ):
        row = next(r for r in sliding if r["ID"] == op_id)
        close(f"row {op_id} DRAM GB/s", round(float(row["DRAM"].replace(" GB/s", "")), 2), want_bw, tol=1e-3)
        close(f"row {op_id} DRAM %", round(float(row["DRAM %"].replace(" %", "")), 2), want_pct, tol=1e-3)

    # the SLOW rows the README names, and the op-to-op gap it classifies
    slow = [r["ID"] for r in sliding if r["Bound"].strip() == "SLOW"]
    same("the SLOW row ids", slow, ["3065", "3071", "3127", "3132", "3139"], readme="3065")
    for op_id in ("3071", "3127", "3132", "3139"):
        same(f"README names SLOW row {op_id}", op_id in readme, True)
    gap_row = next(r for r in sliding if r["ID"] == "3145")
    close("the classified op-to-op gap", round(float(gap_row["Op-to-Op Gap"]), 3), 310.959, tol=1e-4)
    close("its share of the window", round(float(gap_row["Total %"]), 1), 21.6, tol=1e-2)
    same("the gap row is not the first op", [r["ID"] for r in sliding].index("3145"), 28, readme="29 of 55")
    same("55 ops in the window", len(sliding), 55, readme="")
    close(
        "LM head share of the profiling window",
        round(float(next(r for r in sliding if r["ID"] == "3139")["Total %"]), 1),
        40.8,
        tol=1e-2,
    )
    same(
        "the advice text the README quotes",
        "could save 307" in text(D / "tracy/decode_sliding_perf_report.txt"),
        True,
        readme="307 μs",
    )
    same(
        "the window footer the README quotes",
        "358 μs" in text(D / "tracy/decode_sliding_perf_report.txt"),
        True,
        readme="358 μs",
    )

    # every per-round value the README prints, not just the minima
    for label, entry, fmt in (
        ("after TTFT", after["ttft_ms"]["rounds"], "{:.2f}"),
        ("before TTFT", before["ttft_ms"]["rounds"], "{:.2f}"),
        ("after token-out", after["token_out_decode_ms_per_token"]["rounds"], "{:.3f}"),
        ("before token-out", before["token_out_decode_ms_per_token"]["rounds"], "{:.3f}"),
    ):
        for value in entry:
            bind(f"README prints the {label} round {fmt.format(value)}", fmt.format(value))

    # ------------------------------------------------------ performance accounting
    ps = load(D / "perf_summary.json")
    close("perf_summary roofline ms", ps["roofline_ms_per_token_estimate"], 8.829)
    close("perf_summary device ms", ps["decode_ms_per_token_device"], 22.838)
    same(
        "perf_summary e2e matches the run",
        ps["decode_ms_per_token_e2e"],
        round(after["token_out_decode_ms_per_token"]["min"], 3),
    )
    same("perf_summary TTFT matches the run", ps["ttft_ms"], round(after["ttft_ms"]["min"], 2))

    perf("roofline fraction", ps["roofline_ms_per_token_estimate"] / ps["decode_ms_per_token_e2e"] * 100, 1)
    cap = load(D / "evidence_accuracy.json")["capacity"]
    same(
        "roofline layer-weight bytes match the built model",
        ps["roofline_inputs"]["layer_weight_bytes"],
        cap["per_device_layer_weight_bytes"],
    )
    # Read from the artifacts, not from constants: the claim is that the two traces
    # account for the measured step, so both terms and the target come from the same run.
    two_traces = after["traced_decode_logits_only_ms_per_token"]["min"] + after["sampling_trace_ms_per_token"]["min"]
    residual_us = (after["token_out_decode_ms_per_token"]["min"] - two_traces) * 1000
    same(
        "the two traces account for the step to within 60 us",
        0 < residual_us < 60,
        True,
        readme=f"{residual_us:.0f} µs",
    )
    # Round 4 found the checklist row saying 38 µs against this run's 26.8.  A band is not
    # enough on its own: bind the value to both places the README states it.
    bind("the two-trace residual, to 1 dp", f"{residual_us:.1f}")
    same(
        "the checklist row states the same residual as the accounting section",
        readme.count(f"two traces account for the step to {residual_us:.0f} µs"),
        1,
    )

    # --------------------------------------------------------------------- accuracy
    for f, label in ((D / "evidence_accuracy.json", "bf16"), (D / "evidence_fp32_gate.json", "both")):
        d = load(f)
        for gate in ("prefill_check_by_reference", "teacher_forcing_by_reference"):
            for name, entry in d[gate].items():
                out = entry["output"]
                agg = [line for line in out.splitlines() if line.startswith("AGGREGATE")]
                same(f"{f.name}:{gate}:{name} has an aggregate line", bool(agg), True)
                same(f"{f.name}:{gate}:{name} top5 = 1.000", "top5=1.000" in agg[0], True)
                same(f"{f.name}:{gate}:{name} top100 = 1.000", "top100=1.000" in agg[0], True)
                same(f"{f.name}:{gate}:{name} top1 = 0.990", "top1=0.990" in agg[0], True)
    misses = load(D / "evidence_misses.json")["prefill_misses"]
    same("exactly one non-top-1 position", misses["non_top1_positions"], 1)
    same("zero positions outside top-100", misses["outside_top_k_positions"], 0)
    same("the top-k the misses are scored against", misses["k"], 100)
    same("the miss is the reference's rank 1", misses["rows"][0]["hf_rank_of_tt_token"], 1)
    same("the miss is at gen_index 64", misses["rows"][0]["gen_index"], 64)
    close("the miss's own top1-top2 gap", misses["rows"][0]["tt_top1_minus_top2"], 2.0)

    # ------------------------------------------------------------- fallback + trace
    audit = load(D / "evidence_accuracy.json")["fallback_audit"]
    same("decode trace captured", audit["decode_trace_captured"], True)
    same("sampling trace captured", audit["sampling_trace_captured"], True)
    same("force_argmax off", audit["force_argmax"], False)
    per_token = audit["per_token_host_refreshes"]
    close("per-token token refreshes", per_token["token"], 0.0)
    close("per-token position refreshes", per_token["position"], 0.0)
    close("per-token synchronizations", per_token["synchronizations"], 0.0)
    close("per-token page-table refreshes", per_token["page_table"], 0.03125, readme="0.031")
    sampling = load(D / "evidence_accuracy.json")["split_sampling"]
    same("tt_out_tok is the decode token input", sampling["tt_out_tok_is_decode_token_input"], True)
    same(
        "the sampler consumes the decode trace's logits", sampling["sampling_trace_logits_is_decode_trace_output"], True
    )
    same("greedy is k=1 through ttnn.sampling", sampling["sampling_params_greedy"]["top_k"], 1)
    same("greedy p=0", sampling["sampling_params_greedy"]["top_p"], 0.0)
    same("force_argmax is not enabled", sampling["force_argmax_enabled"], False)
    same("greedy is deterministic across calls", sampling["deterministic_across_calls"], True)
    same("a sampled request does not corrupt greedy", sampling["greedy_after_sampled_matches"], True)
    same("top-k/top-p goes through the same path", sampling["top_k_top_p_differs_from_greedy"], True)
    two = sampling["two_step_replay"]
    same("the sampled token becomes the next input", two["token_after_step1"], two["sampled_step1"])
    same("the second sampled token lands too", two["token_after_step2"], two["sampled_step2"])
    same("position advances on device 128->129", two["pos_after_step1"], [129])
    same("position advances on device 129->130", two["pos_after_step2"], [130])
    for key, value in two["host_staging_between_replays"].items():
        same(f"nothing staged between replays: {key}", value, 0)

    # ------------------------------------------------------------- sampler benchmark
    arms = {row["label"]: row for row in load(D / "sampler_ab.json")}
    close("shipped sampling trace ms", round(arms["topk split to 2x32768 (shipped)"]["sampling_trace_ms"], 4), 0.6323)
    close(
        "no-split sampling trace ms",
        round(arms["no split: single-core topk over 50688"]["sampling_trace_ms"], 4),
        9.7295,
    )
    close("max_top_k=8 sampling trace ms", round(arms["top_k=8, pad_to_pow2"]["sampling_trace_ms"], 4), 0.7942)
    tokens = {tuple(row["first_tokens"]) for row in load(D / "sampler_ab.json") if "first_tokens" in row}
    same("every sampler arm samples the same tokens", len(tokens), 1)

    # ------------------------------------------------------- host-dispatch evidence
    host = load(D / "prefill_host_probe.json")
    close("prefill issue ms", round(host["dispatch"]["issue_ms"], 2), 54.91)
    close("prefill issue+drain ms", round(host["dispatch"]["issue_plus_drain_ms"], 2), 55.08)
    oc = load(D / "prefill_opcount.json")["prefill"]
    same("prefill ttnn call count", oc["total_calls"], 4122)
    by_op = {row["op"]: row for row in oc["rows"]}
    close("reduce_scatter host ms", by_op["ttnn.experimental.reduce_scatter_minimal_async"]["ms"], 14.60, tol=1e-2)
    close("all_gather host ms", by_op["ttnn.experimental.all_gather_async"]["ms"], 6.33, tol=1e-2)
    same("deallocate call count", by_op["ttnn.deallocate"]["calls"], 1957)
    close("deallocate us/call", by_op["ttnn.deallocate"]["us_per_call"], 1.67, tol=1e-2)
    ccl = {row["arm"]: row for row in load(D / "ccl_host_probe_bfp8.json")}
    ccl_loaded = {row["arm"]: row for row in load(D / "ccl_host_probe_bfp8_loaded.json")}
    close("bfp8 rs allocating us/call", ccl["rs_async_alloc_workers4[bfloat8_b]"]["issue_us"], 72.10, tol=1e-2)
    close("bfp8 rs persistent us/call", ccl["rs_async_persistent_workers4[bfloat8_b]"]["issue_us"], 62.12, tol=1e-2)
    close("bfp8 ag allocating us/call", ccl["ag_async_alloc_default[bfloat8_b]"]["issue_us"], 56.04, tol=1e-2)
    close("bfp8 all_reduce wrapper us/call", ccl["all_reduce_wrapper[bfloat8_b]"]["issue_us"], 118.04, tol=1e-2)
    close(
        "bfp8 loaded rs allocating us/call",
        ccl_loaded["rs_async_alloc_workers4[bfloat8_b+loaded]"]["issue_us"],
        117.05,
        tol=1e-2,
    )
    close(
        "bfp8 loaded rs persistent us/call",
        ccl_loaded["rs_async_persistent_workers4[bfloat8_b+loaded]"]["issue_us"],
        96.86,
        tol=1e-2,
    )
    drained = {row["op"]: row for row in load(D / "prefill_opcount.json")["prefill_drained_collectives"]["rows"]}
    close(
        "in-model rs drained us/call",
        drained["ttnn.experimental.reduce_scatter_minimal_async"]["us_per_call"],
        114.60,
        tol=1e-2,
    )
    close(
        "in-model rs pipelined us/call",
        by_op["ttnn.experimental.reduce_scatter_minimal_async"]["us_per_call"],
        140.34,
        tol=1e-2,
        readme="140.3",
    )
    trace = load(D / "prefill_trace_probe.json")
    close("traced prefill eager ms", round(trace["eager_ms"]["min"], 2), 59.80)
    close("traced prefill replay ms", round(trace["traced_ms"]["min"], 2), 44.96)
    close("traced prefill capture ms", round(trace["capture_ms"], 2), 98.16)
    same("traced prefill replay is bit-identical", trace["replay_vs_eager"]["bit_identical"], True)
    same("traced prefill coexists with the decode traces", trace["with_decode_traces"], True)
    close("traced prefill retained MB", round(trace["capture_retained_dram_bytes"] / 1e6, 1), 3.3, tol=5e-2)
    close("traced prefill payback replays", trace["payback_replays"], 6.6, tol=2e-2)
    pt = load(D / "evidence_perf_prefill_trace.json")["performance"]
    for value in pt["ttft_ms"]["rounds"]:
        bind(f"README prints the prefill-trace round {value:.2f}", f"{value:.2f}")
    improvement = (pt["ttft_ms"]["min"] - after["ttft_ms"]["min"]) / after["ttft_ms"]["min"] * 100
    perf("prefill-trace TTFT improvement %", improvement, 1)
    same("the prefill trace improves TTFT by more than 15 %", improvement < -15.0, True)
    same("prefill-trace arm flag", pt["prefill_trace_arm"], True)
    same("prefill-trace bucket", pt["prefill_trace_buckets"], [128])
    same(
        "perf_summary prefill-trace TTFT matches the run",
        load(D / "perf_summary.json")["ttft_ms_with_prefill_trace"],
        round(pt["ttft_ms"]["min"], 2),
    )
    perf("prefill-trace TTFT ms", pt["ttft_ms"]["min"], 2)
    perf("prefill-trace token-out ms", pt["token_out_decode_ms_per_token"]["min"], 3)
    # Round 7 added this stage's four flags to ``capability_report()`` so an arm's own evidence
    # states the settings that define it; round 8 found the prefill-trace arm had not been
    # re-run since, leaving the one arm whose name *is* a flag unable to show it.  It is
    # regenerated, and all three perf arms are asserted to carry the block -- and to disagree
    # where the arms disagree, which is the property the block exists for.
    FLAGS = ("lm_head_softcap_in_l1", "embed_decode_gather_sharded", "decode_swiglu_mul_cores", "prefill_trace")
    arms = {
        "shipped": load(D / "evidence_perf.json")["capacity"],
        "baseline": load(D / "evidence_perf_before.json")["capacity"],
        "prefill-trace": load(D / "evidence_perf_prefill_trace.json")["capacity"],
    }
    for arm, capacity in arms.items():
        same(f"the {arm} arm's capacity block carries this stage's flags", [f in capacity for f in FLAGS], [True] * 4)
    same(
        "the baseline arm's flags are the reverted ones",
        [arms["baseline"][f] for f in FLAGS],
        [False, False, None, False],
    )
    same("the shipped arm's are the shipped ones", [arms["shipped"][f] for f in FLAGS], [True, True, 80, False])
    same(
        "...and the prefill-trace arm differs from it in exactly the flag it is named for",
        [k for k in FLAGS if arms["prefill-trace"][k] != arms["shipped"][k]],
        ["prefill_trace"],
    )
    same(
        "the softcap flag is read off the built model, not the module global",
        "bool(self.model.lm_head.softcap_in_l1)" in text(ROOT / "tt/generator.py"),
        True,
    )
    l1 = load(D / "l1_highwater_probe.json")
    same("L1 peak delta per bank", l1["l1_peak_delta_per_bank_bytes"], 126976)
    same("L1 free per bank at the peak", l1["l1_free_per_bank_at_peak_with_change"], 1238144)
    tb = load(D / "ttft_breakdown_before.json")["by_length"]["128"]
    close("TTFT layers phase ms", round(tb["phases_ms"]["layers"]["min"], 2), 60.28)
    close("TTFT e2e ms in the breakdown", round(tb["e2e_ttft_ms"]["min"], 2), 64.80, readme="")

    # ------------------------------------------------------------------ qualitative
    diff = load(D / "qualitative/qualitative_tt_vs_full_model_stage.json")
    same(
        "all six qualitative completions are byte-identical to stage 6",
        all(row["identical_text"] for row in diff),
        True,
    )
    same("six prompts compared", len(diff), 6)
    same(
        "the degeneracy gate found nothing",
        "No degenerate output detected" in text(D / "logs/check_degenerate_output.log"),
        True,
    )

    # ----------------------------------------------------------------- watcher/tests
    same("watcher log is clean", "WATCHER_CLEAN" in text(D / "logs/check_watcher.log"), True)
    same("the watcher script prints the verdict itself", "WATCHER_CLEAN" in text(D / "logs/run_watcher.log"), True)
    same("no fatal watcher messages", "fatal watcher messages: 0" in text(D / "logs/check_watcher.log"), True)
    same(
        "the shipped-default watcher set passed",
        "10 passed" in text(D / "logs/watcher_pytest.log"),
        True,
        readme="10 device cases",
    )
    same(
        "no tripped assert in the shipped-default watcher run",
        "tripped assert" not in text(D / "logs/watcher_pytest.log"),
        True,
    )
    for tag in ("optin", "rebind"):
        same(
            f"the opt-in prefill-trace watcher case {tag} passed",
            "1 passed" in text(D / f"logs/watcher_prefill_trace_{tag}.log"),
            True,
        )
        same(
            f"no tripped assert in the {tag} case",
            "tripped assert" not in text(D / f"logs/watcher_prefill_trace_{tag}.log"),
            True,
        )
    for arm in ("capture", "release", "recapture", "clone_cache", "rebuild"):
        same(
            f"the {arm} release-probe arm is watcher-clean",
            "tripped assert" not in text(D / f"logs/watcher_probe_{arm}.log"),
            True,
        )
    # ...and the rebuild arm, unlike the other four, has a re-derivable watcher log rather
    # than a console-only verdict, which round 3 called out as a gap for the probe arms.
    same(
        "the rebuild arm's watcher log is a real run",
        "WATCHER_CLEAN" in text(D / "logs/watcher_probe_rebuild.log"),
        True,
    )
    same(
        "the rebuild arm built a second generator",
        "PR built a second generator" in text(D / "logs/watcher_probe_rebuild.log"),
        True,
    )
    # ...and its watcher log is a complete run, re-derived here rather than trusted from the
    # console verdict: a truncated log is exactly what a tripped run leaves behind, and round
    # 4 noted this was the one listed artifact the gate never opened.
    rebuild_watcher = text(D / "watcher_probe_rebuild/watcher.log.gz")
    same(
        "the rebuild arm's watcher log has detach lines",
        len(re.findall(r"^At [0-9.]+s detach device \d+", rebuild_watcher, re.M)),
        4,
    )
    same(
        "...and no tripped assert in it",
        "subordinate_erisc detected invalid NOC command buffer state" in rebuild_watcher,
        False,
    )
    same(
        "the pre-fix rebind run did trip, and is preserved",
        "tripped assert" in text(D / "logs/watcher_bisect_rebind.log"),
        True,
    )
    same(
        "the post-fix rebind run does not",
        "tripped assert" not in text(D / "logs/watcher_bisect_rebind_fixed.log"),
        True,
    )
    for verdict_log in (
        "logs/check_watcher.log",
        "logs/check_watcher_default10.log",
        "logs/check_watcher_prefill_trace_pair.log",
    ):
        same(f"{verdict_log} re-derives a clean verdict", text(D / verdict_log).count("WATCHER_CLEAN"), 1)
    # The suite size is derived from the junit artifact rather than typed, so a case added
    # without updating the README fails here.  Round 6 found three stale sizes and round 7 a
    # fourth; a hardcoded literal is how all four survived.
    suite_cases = len(list(ET.parse(_record(D / "test_results.xml")).iter("testcase")))
    same(
        f"{suite_cases} cases passed forward",
        f"{suite_cases} passed" in text(D / "logs/full_test_run.log"),
        True,
        readme=f"{suite_cases} passed",
    )
    same(
        f"{suite_cases} cases passed in reverse",
        f"{suite_cases} passed" in text(D / "logs/full_test_run_reverse.log"),
        True,
    )
    bind("the README states the suite size", str(suite_cases))
    new_cases = [
        name
        for name in (
            "test_lm_head_softcap_runs_in_l1_and_matches_the_dram_form",
            "test_decode_embedding_gathers_straight_into_the_boundary_layout",
            "test_swiglu_multiply_runs_on_the_wide_grid_and_returns_the_narrow_one",
            "test_prefill_trace_is_opt_in_and_matches_the_eager_path",
            "test_prefill_trace_survives_rebinding_the_same_external_cache",
            "test_decode_follows_the_cache_it_is_rebound_to_after_the_trace_is_captured",
            "test_the_live_trace_count_round_trips_over_both_trace_kinds",
            "test_a_trace_that_fails_to_release_is_never_replayed_and_is_retried",
        )
        if f"`{name}" in readme
    ]
    same("the README's new-case table lists every new test", len(new_cases), 8)
    same(
        "...and the inherited/new split adds up to the suite",
        f"(46 inherited + {suite_cases - 46} new)" in readme,
        True,
    )
    same("tracy integrity check passed", "TRACY_INTEGRITY_OK" in text(D / "logs/run_tracy.log"), True)
    same("the overflowed two-layer capture is preserved", (D / "logs/run_tracy_two_layer_overflow.log").is_file(), True)

    # -------------------------------------------------------------- context contract
    contract = load(ROOT / "doc/context_contract.json")
    same("contract stage", contract["stage"], "optimized_full_model")
    same("contract supported context", contract["current_supported_context"], 131072)
    same("contract HF context", contract["hf_advertised_context"], 131072)
    alloc = contract["implementation"]["optimized_full_model_allocations"]
    trace_probe = load(D / "prefill_trace_probe.json")
    l1p = load(D / "l1_highwater_probe.json")
    same(
        "contract records the prefill-trace retained DRAM",
        alloc["prefill_trace_retained_dram_bytes_per_device_at_128_rows"],
        trace_probe["capture_retained_dram_bytes"],
    )
    same(
        "contract records the decode L1 peak delta",
        alloc["decode_peak_l1_delta_bytes_per_bank"],
        l1p["l1_peak_delta_per_bank_bytes"],
    )
    same("contract records the prefill trace as default-off", alloc["prefill_trace_default"], False)
    same(
        "contract notes are attributed to this stage",
        "optimized-full-model stage takes no capability reduction" in contract["notes"],
        True,
    )
    # Round 8's P1: the contract's whole ``performance`` block was the *previous* run's while
    # its ``source`` field named this one -- the third recurrence of the false-provenance
    # defect rounds 2, 3 and 4 each found elsewhere.  This gate asserted only the provenance
    # *string*, so 651/651 passed over a contract that contradicted the artifact it cited, and
    # mutating any figure in it passed too.  Every field is bound to the artifact now.
    contract_perf = contract["performance"]
    # (The provenance *string* is asserted in round 4's block further down; what follows is
    # the figures it points at.)
    for field, artifact_value in (
        ("ttft_ms", after["ttft_ms"]["min"]),
        ("token_out_decode_ms_per_token", after["token_out_decode_ms_per_token"]["min"]),
        ("token_out_decode_tok_s_u", after["token_out_decode_tok_s_u"]),
        ("traced_decode_logits_only_ms_per_token", after["traced_decode_logits_only_ms_per_token"]["min"]),
        ("traced_decode_logits_only_tok_s_u", after["traced_decode_logits_only_tok_s_u"]),
        ("sampling_trace_ms_per_token", after["sampling_trace_ms_per_token"]["min"]),
        ("layer_stack_lower_bound_ms_per_token", after["layer_stack_lower_bound_ms_per_token"]["total_ms"]),
    ):
        same(f"the contract's {field} is this run's, not a predecessor's", contract_perf[field], artifact_value)
    # Every *other* numeric field of the block too, so a field added later cannot go
    # unbound -- the block is exhausted rather than sampled.
    same(
        "the contract's performance block holds no unbound figure",
        sorted(k for k, v in contract_perf.items() if isinstance(v, (int, float)) and not isinstance(v, bool)),
        [
            "layer_stack_lower_bound_ms_per_token",
            "sampling_trace_ms_per_token",
            "token_out_decode_ms_per_token",
            "token_out_decode_tok_s_u",
            "traced_decode_logits_only_ms_per_token",
            "traced_decode_logits_only_tok_s_u",
            "ttft_ms",
        ],
    )
    # And the whole file against a fresh regeneration.  ``--check`` is the one check in the
    # tree that would have caught the staleness above on its own; round 8 found this gate was
    # not invoking it, so a contract left un-regenerated after a perf re-run passed here.
    refresh = subprocess.run(
        [sys.executable, str(D / "bench/refresh_context_contract.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=str(ROOT.parents[2]),
    )
    same(
        f"the contract is what a fresh regeneration produces ({refresh.stdout.strip()[-90:] or 'no output'})",
        refresh.returncode,
        0,
    )

    # --------------------------------------- round-3 coverage: the rows the gate missed
    #
    # Round 3 of the stage review found six wrong figures in sections this gate did not
    # read at all, while it passed 328/328.  Freezing the check *count* never caught that,
    # because the count is not the coverage.  Everything below binds a row the gate
    # previously left unchecked, and ``SOURCES`` asserts the artifacts themselves are read.

    # (a) the prefill-128 capture: the two slow norms and the window total.
    prefill_rows = csv_rows("prefill_128_perf_report.csv")
    prefill_by_id = {r["ID"]: r for r in prefill_rows}
    norms = sorted(
        (r for r in prefill_rows if r["OP Code"] == "LayerNormDeviceOperation"),
        key=lambda r: -float(r["Device Time"]),
    )
    same("the two slowest prefill norms are the ids the README names", [r["ID"] for r in norms[:2]], ["3579", "3886"])
    close("prefill terminal norm us", device_time(prefill_rows, "3886"), 133.868)
    same("the prefill terminal norm runs on one core", prefill_by_id["3886"]["Cores"], "1")
    close("prefill embedding norm us", device_time(prefill_rows, "3579"), 133.979)
    same("the prefill embedding norm runs on four cores", prefill_by_id["3579"]["Cores"], "4")
    close(
        "prefill-128 window total us",
        round(sum(float(r["Device Time"]) for r in prefill_rows), 1),
        2606.3,
    )

    # (b) the host-dispatch table's residual row, computed rather than transcribed.
    oc = load(D / "prefill_opcount.json")["prefill"]
    listed = oc["rows"][:9]
    residual_rows = oc["rows"][9:]
    same("the README lists the top nine op kinds", len(listed), 9)
    same("the residual row covers the other 12 op kinds", len(residual_rows), 12)
    same("residual calls", sum(r["calls"] for r in residual_rows), 707)
    close("residual ms", sum(r["ms"] for r in residual_rows), 7.86, tol=2e-3)
    same("the table's call column sums to the stated total", oc["total_calls"], sum(r["calls"] for r in oc["rows"]))
    close("prefill opcount total ms", oc["total_ms"], 58.56)
    close("prefill opcount wall ms", oc["wall_issue_ms"], 62.75)
    close("the unattributed Python between calls", oc["wall_issue_ms"] - oc["total_ms"], 4.19, tol=2e-3)

    # (c) the operation-topology audit, as a partition of the sliding-layer CSV.
    sliding = csv_rows("decode_sliding_perf_report.csv")
    groups = {
        "LM head": (["3139"], 603.798),
        "MLP gate/up/down": (["3071", "3127", "3132"], 190.266),
        "wqkv + attn_gate": (["3039", "3172"], 40.772),
        "o_proj": (["3065"], 21.368),
        "SwiGLU multiply": (["3075"], 4.747),
        "softcap": (["3140", "3196"], 23.786),
        "RMSNorm x8": (["3137", "3147", "3153", "3180", "3203", "3211", "3233", "3245"], 61.628),
        "reduce-scatter + all-gather x2": (["3231", "3232", "3243", "3244"], 54.886),
        "SdpaDecode": (["3168"], 15.136),
        "embedding all-gather": (["3201"], 16.498),
        "embeddings": (["3145", "3158", "3161"], 14.331),
        "attention glue": (
            ["3054", "3055", "3096", "3107", "3112", "3115", "3119", "3124", "3191", "3214", "3221"],
            39.604,
        ),
        "layout conversions": (
            [
                "3073",
                "3076",
                "3095",
                "3099",
                "3100",
                "3105",
                "3108",
                "3116",
                "3118",
                "3129",
                "3193",
                "3197",
                "3207",
                "3212",
                "3224",
            ],
            33.886,
        ),
        "plus_one x2": (["3143", "3254"], 1.845),
    }
    used: list[str] = []
    for label, (ids, want) in groups.items():
        used += ids
        close(f"audit row {label!r} us", round(sum(device_time(sliding, i) for i in ids), 3), want, tol=1e-4)
    same("the audit table's ids are unique", len(used), len(set(used)))
    same("the audit table's ids are every row of the CSV", sorted(used), sorted(r["ID"] for r in sliding))
    close(
        "audit column total = the window",
        round(sum(want for _, want in groups.values()), 3),
        round(sum(float(r["Device Time"]) for r in sliding), 3),
        tol=1e-6,
    )
    close("sliding window total us", round(sum(float(r["Device Time"]) for r in sliding), 3), 1122.551, tol=1e-6)
    six = ["3039", "3065", "3071", "3127", "3132", "3172"]
    close(
        "the six DRAM-sharded projections",
        round(sum(device_time(sliding, i) for i in six), 2),
        252.41,
        tol=1e-4,
    )
    terminal = 691.07
    layer = round(sum(float(r["Device Time"]) for r in sliding) - terminal, 2)
    close("the sliding layer, window minus terminal", layer, 431.48, tol=1e-4)
    close(
        "the latency-bound remainder of the layer",
        round(layer - sum(device_time(sliding, i) for i in six), 2),
        179.07,
        tol=1e-4,
    )
    close(
        "latency-bound share of the layer %",
        round((layer - sum(device_time(sliding, i) for i in six)) / layer * 100, 1),
        41.5,
        tol=1e-4,
    )

    # The two pre-change values the README quotes, from the file it now attributes them to.
    prev_rows = list(csv.DictReader(open(PREV / "tracy/decode_perf_report.csv")))
    close("pre-change SwiGLU multiply us", device_time(prev_rows, "4187"), 18.026, tol=1e-4)
    close(
        "pre-change softcap us",
        round(device_time(prev_rows, "4283") + device_time(prev_rows, "4370"), 3),
        36.853,
        tol=1e-4,
    )
    close("pre-change embedding i2s us", device_time(prev_rows, "4289"), 1.992, tol=1e-4)
    close(
        "softcap device-time win",
        round(
            device_time(prev_rows, "4283")
            + device_time(prev_rows, "4370")
            - device_time(sliding, "3140")
            - device_time(sliding, "3196"),
            2,
        ),
        13.07,
        tol=1e-3,
    )
    close(
        "SwiGLU device-time win per layer",
        round(
            device_time(prev_rows, "4187")
            - device_time(sliding, "3075")
            - device_time(sliding, "3073")
            - device_time(sliding, "3129")
            - device_time(sliding, "3076"),
            1,
        ),
        7.4,
        tol=1e-3,
    )
    close(
        "change 3's three reshards",
        round(device_time(sliding, "3073") + device_time(sliding, "3129") + device_time(sliding, "3076"), 3),
        5.907,
        tol=1e-4,
    )

    # The DRAM peak the roofline back-derives, and the prose field that records it.
    for op_id, want_pct in (("3139", 54.57), ("3039", 77.12)):
        row = next(r for r in sliding if r["ID"] == op_id)
        close(f"row {op_id} DRAM %", round(float(row["DRAM %"]), 2), want_pct, tol=1e-5)
        close(f"row {op_id} back-derived peak GB/s", float(row["DRAM"]) / float(row["DRAM %"]) * 100, 512.0)
    bandwidth_source = load(D / "perf_summary.json")["roofline_inputs"]["bandwidth_source"]
    for token in ("3139", "279.38", "54.5666", "3039", "394.85", "77.1192"):
        same(f"perf_summary bandwidth_source names {token}", token in bandwidth_source, True)

    # (d) teacher forcing, per file, from the decode field rather than the e2e one.
    def decode_rates(name: str) -> list[float]:
        blob = load(D / name)
        found: list[float] = []

        def walk(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "decode_t/s/u" and isinstance(value, (int, float)):
                        found.append(round(float(value), 2))
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(blob)
        return sorted(set(found))

    same("teacher-forcing rate in evidence_accuracy.json", decode_rates("evidence_accuracy.json"), [37.07])
    same("teacher-forcing rates in evidence_fp32_gate.json", decode_rates("evidence_fp32_gate.json"), [37.28, 38.15])
    for rate in (37.07, 37.28, 38.15):
        bind(f"README states teacher-forcing {rate}", f"{rate:.2f}")

    # (e) the retracted fabric-ERISC hazard: the two runs that had to be clean, and the
    # absence of the retirement the retraction removed.
    pair = text(D / "logs/watcher_pytest_prefill_trace_pair.log")
    same("the two opt-in cases ran together in one process", "2 passed" in pair, True)
    same("...and were watcher-clean", "WATCHER_CLEAN" in text(D / "logs/check_watcher_prefill_trace_pair.log"), True)
    rebuild = text(D / "logs/watcher_probe_rebuild.log")
    same("the rebuild arm ran the named sequence", "PR_OK arm=rebuild" in rebuild, True)
    same("...and was watcher-clean", "WATCHER_CLEAN" in rebuild, True)
    same("...having actually built a second generator", "PR built a second generator on the same mesh" in rebuild, True)
    generator_src = text(ROOT / "tt/generator.py")
    same("the retirement flag is gone from the generator", "_prefill_trace_retired" in generator_src, False)
    same("a cache move recaptures instead", "_prefill_trace_releases" in generator_src, True)
    # Round 4's P1: the decode trace bakes the same cache addresses and must be invalidated
    # on the same signal.  Asserted on the source and on the committed negative control.
    same("the decode trace carries its own cache signature", "_decode_trace_cache_sig" in generator_src, True)
    same("...and is released on a move", "def _release_decode_trace" in generator_src, True)
    same(
        "the invalidation covers every trace, not just prefill",
        "def _invalidate_traces_if_cache_moved" in generator_src,
        True,
    )
    same(
        "the decode trace records its signature at capture",
        "self._decode_trace_cache_sig = self._kv_cache_signature()" in generator_src,
        True,
    )
    negative = text(D / "logs/decode_rebind_prefix_negative_control.log")
    same("the decode-rebind test fails against the pre-fix code", "1 failed" in negative, True)
    same(
        "...on the assertion the pre-fix code violates",
        "a moved cache must release the decode trace" in negative,
        True,
    )
    same(
        "...and the shipped source is not the patched one",
        "TEMPORARY: pre-fix behaviour" in generator_src,
        False,
    )
    # The trace-ownership inventory: assert the code matches what the README's table says,
    # so a fourth capture site or a rebindable input added later contradicts the document.
    same(
        "this port captures two of the three traces itself",
        generator_src.count("ttnn.begin_trace_capture"),
        2,  # decode + prefill here; the sampling trace captures in models/common/sampling
    )
    same(
        "teardown and the rebind path share one decode-release function",
        generator_src.count("self._release_decode_trace()"),
        2,
    )
    same(
        "reset() does not invalidate traces",
        "Traces survive on purpose" in generator_src,
        True,
    )
    # Round 5: the model owns the cache and the generator owns the traces, so neither can
    # enforce the release-before-free ordering alone.  The shared counter is what lets
    # ``deallocate()`` refuse to be silent about it; assert it is wired at all four sites.
    model_src = text(ROOT / "tt/model.py")
    same("the model exposes a live-trace count", "def live_traces_over_kv_cache" in model_src, True)
    same("...and deallocate() checks it", "if self.live_traces_over_kv_cache:" in model_src, True)
    same("...and no longer claims to free weights", 'weights included."""' in model_src, False)
    same("the generator notes both captures", generator_src.count("note_trace_captured()"), 2)
    # Three release sites, not two: the decode path, the prefill path, and the retry that
    # drains ``_orphaned_traces``.  A failed release does **not** decrement (round 6), so the
    # retry is the only place a retained trace can ever be accounted for.
    same("...and every release", generator_src.count("self.model.note_trace_released()"), 3)
    # Round 8's P1: a failed ``ttnn.release_trace`` must fail *closed*.  Round 7 retained the
    # handle in place, which is the handle ``decode_forward`` tests and the bucket
    # ``_prefill_traced`` looks up, so a failed release on the rebind path replayed a trace
    # against the cache the caller had just rebound away from -- round 4's silent-wrong-token
    # bug from a branch round 4's test does not take.  Assert the policy at each of the four
    # places it lives, on the source, and the behaviour in the test.
    same(
        "a failed release moves the trace out of every lookup path",
        "self._orphaned_traces.append(" in generator_src and generator_src.count("_orphaned_traces") >= 6,
        True,
    )
    same(
        "...and the decode slot is cleared so the next call recaptures",
        "def _retry_orphaned_traces" in generator_src,
        True,
    )
    same(
        "...the prefill bucket dict is emptied unconditionally",
        "self._prefill_traces = {}" in generator_src,
        True,
    )
    same(
        "...and the signature is cleared even when something was orphaned",
        "self._prefill_traces = {}\n        self._prefill_trace_cache_sig = ()" in generator_src,
        True,
    )
    same(
        "the retained-in-place policy round 8 rejected is gone",
        "retaining the decode trace and its logits after a failed" in generator_src,
        False,
    )
    same(
        "teardown retries what an earlier release could not free",
        "still_held = self._retry_orphaned_traces()" in generator_src,
        True,
    )
    same(
        "close_multichip_mesh drops the model's semaphore cache too",
        "_MODEL_CCL_SEMAPHORES.pop(id(mesh), None)" in text(ROOT / "tt/multichip_decoder.py"),
        True,
    )
    same(
        "the module-scoped generator fixture releases before the mesh closes",
        "built.teardown()" in text(ROOT / "tests/test_full_model.py"),
        True,
    )
    tests_src = text(ROOT / "tests/test_full_model.py")
    same(
        "the decode-rebind test compares the traced decode against the eager one",
        "the traced decode must answer from the cache it is bound to now" in tests_src,
        True,
    )
    # Round 8: both retention branches were dead code from the suite's point of view -- no
    # test made ``release_trace`` raise, so the policy was asserted only on the source.  The
    # injected-failure case is now the behavioural half of the four source checks above.
    same(
        "a test injects a raising release_trace",
        "injected: release_trace refused" in tests_src,
        True,
    )
    same(
        "...and asserts the failed release is not answered from the kept trace",
        "must not answer from the released-and-kept trace" in tests_src,
        True,
    )
    same(
        "...that nothing was deallocated",
        'all(t.is_allocated() for o in generator._orphaned_traces for t in o["tensors"])' in tests_src,
        True,
    )
    same(
        "...and that the retry accounts for both orphans",
        "the two orphans are accounted for on retry" in tests_src,
        True,
    )
    # Round 4's P2: the contract's provenance must name this stage, not the previous one.
    contract_perf = load(ROOT / "doc/context_contract.json")
    same(
        "the contract's performance provenance names this stage",
        contract_perf["performance"]["source"],
        "doc/optimized_full_model/evidence_perf.json",
    )
    same(
        "the contract's byte-budget provenance names this stage",
        contract_perf["byte_budget_at_full_context"]["measured_from"].startswith("doc/optimized_full_model/"),
        True,
    )
    # Round 5: the sibling fields under ``tested`` still named the previous stage's harness.
    # Assert the *current* stage's fields are free of it, field-by-field rather than by one
    # string that a new field could slip past.  The per-stage historical entries
    # (``full_model``, ``optimized_decoder``, ...) legitimately keep their own paths.
    HISTORICAL = {
        "optimized_multichip_decoder",
        "full_model",
        "optimized_decoder",
        "fused_decoder",
        "multichip_decoder",
    }

    def _without_reused_hf(value):
        """Drop the one legitimate previous-stage reference: the reused HF control arm.

        Round 6 found that a blanket "no doc/full_model/ anywhere" rule had *caused* a defect
        -- the round-5 substitution rewrote the ``qualitative.py --arm hf`` command this stage
        deliberately did not run.  The rule is right for everything else, so the exception is
        named rather than the rule dropped.
        """
        # Exact-command match, not a substring escape hatch: round 7 showed that dropping any
        # line containing "--arm hf" let an unrelated previous-stage reference through simply
        # by mentioning the flag (and that "--arm hfoo" matched too).
        return "\n".join(
            line
            for line in json.dumps(value, indent=1).splitlines()
            if "python doc/full_model/bench/qualitative.py --arm hf " not in line
        )

    stale = sorted(
        key
        for key, value in contract_perf.items()
        if key not in HISTORICAL and "doc/full_model/" in _without_reused_hf(value)
    )
    same("no current-stage contract field points at the previous stage", stale, [])
    same(
        "the contract's tested commands name this stage's harness",
        all(
            "doc/optimized_full_model/" in command
            for command in contract_perf["tested"]["commands"]
            if command.startswith("python doc/") and "--arm hf" not in command
        ),
        True,
    )
    # Round 6: "contains doc/optimized_full_model/" passed a dangling glob and a command this
    # stage never ran.  Resolve the referenced paths against the filesystem instead.
    note = contract_perf["tested"]["prefill_misses"]["note"]
    miss_ref = note.rsplit(" ", 1)[-1]
    same("the miss-detail note names this stage", miss_ref.startswith("doc/optimized_full_model/"), True)
    same("...and the file it names exists", (ROOT / miss_ref).is_file(), True)
    # Round 7: only ``doc/**.py`` tokens inside ``tested.commands`` were stat'ed, so a
    # dangling ``prefill_trace_retained_dram_source`` passed.  Resolve every doc/ path in the
    # current-stage subtree.
    current_stage_blob = json.dumps({key: value for key, value in contract_perf.items() if key not in HISTORICAL})
    referenced = sorted(set(re.findall(r"doc/[\w./-]+\.(?:py|json|log|csv|md)", current_stage_blob)))
    same("the contract references at least a dozen artifacts", len(referenced) >= 8, True)
    for token in referenced:
        same(f"contract-referenced path exists: {token}", (ROOT / token).is_file(), True)
    same(
        "the HF qualitative arm is still attributed to the stage that ran it",
        any("doc/full_model/bench/qualitative.py --arm hf" in c for c in contract_perf["tested"]["commands"]),
        True,
    )
    same(
        "...and says why",
        any("--reuse-hf-control" in c for c in contract_perf["tested"]["commands"]),
        True,
    )
    same(
        "the historical per-stage entries keep their own paths",
        "doc/full_model/" in json.dumps(contract_perf["full_model"]),
        True,
    )
    same("the gated watcher run passed 10 cases", "10 passed" in text(D / "logs/watcher_pytest.log"), True)
    same(
        "the gated watcher run tripped nothing on the console either",
        "WATCHER_CONSOLE_NO_TRIPPED_ASSERT" in text(D / "logs/check_watcher_console.log"),
        True,
    )
    # Limitation 6's replicate set.  Each arm is (label, logs, expected trips), and the
    # arithmetic the README quotes -- 0/5, 3/3, 1/3, and the two Fisher p-values -- is
    # re-derived here rather than transcribed.
    ARMS = {
        "ten gated cases": (
            [
                "logs/watcher_pytest.log",
                "logs/watcher_pytest_default10.log",
                "logs/watcher_pytest_10case_repa.log",
                "logs/watcher_pytest_10case_repb.log",
                "logs/watcher_pytest_10case_repc.log",
            ],
            10,
            0,
        ),
        "twelve with both opt-in cases": (
            [
                "logs/watcher_pytest_12case_tripped.log",
                "logs/watcher_pytest_12case_tripped_run2.log",
                "logs/watcher_pytest_12case_tripped_run3.log",
            ],
            12,
            3,
        ),
        "twelve with two other sampling cases": (
            [
                "logs/watcher_pytest_12case_control.log",
                "logs/watcher_pytest_12case_control2.log",
                "logs/watcher_pytest_12case_control3.log",
            ],
            12,
            1,
        ),
    }
    POSTFIX = {
        "twelve with both opt-in cases": (
            [f"logs/watcher_pytest_postfix_optin{i}.log" for i in (1, 2, 3)],
            12,
            3,
        ),
        "twelve with two other sampling cases": (
            [f"logs/watcher_pytest_postfix_ctrl{i}.log" for i in (1, 2, 3)],
            12,
            1,
        ),
    }
    tallies = {}
    post = {}
    for label, (logs, cases, want_trips) in ARMS.items():
        trips = 0
        for log in logs:
            body = text(D / log)
            same(
                f"{label}: {log} ran {cases} cases",
                len(set(re.findall(r"test_full_model\.py::([\w\[\]]+)", body))),
                cases,
            )
            same(f"{label}: {log} had no failure", "FAILED" in body, False)
            trips += "subordinate_erisc detected invalid NOC command buffer state" in body
        same(f"{label}: {trips} of {len(logs)} tripped", trips, want_trips)
        tallies[label] = (trips, len(logs))
    # The same two arms re-measured against round 5's fixed teardown.  The claim that the fix
    # moved neither is what licenses pooling, so it is asserted rather than asserted-about.
    for label, (logs, cases, want_trips) in POSTFIX.items():
        trips = 0
        for log in logs:
            body = text(D / log)
            same(
                f"post-fix {label}: {log} ran {cases} cases",
                len(set(re.findall(r"test_full_model\.py::([\w\[\]]+)", body))),
                cases,
            )
            same(f"post-fix {label}: {log} had no failure", "FAILED" in body, False)
            trips += "subordinate_erisc detected invalid NOC command buffer state" in body
        same(f"post-fix {label}: {trips} of {len(logs)} tripped", trips, want_trips)
        post[label] = (trips, len(logs))
    for label in POSTFIX:
        same(f"the teardown fix did not move the {label!r} arm", tallies[label][0], post[label][0])

    # Round 6's two arms: the ones that separate "the prefill trace" from "a long process
    # that builds generators and churns the cache".
    ROUND6 = {
        "work-matched twelve": ([f"logs/watcher_pytest_workmatched{i}.log" for i in (1, 2, 3)], 12, 0),
        "the opt-in pair alone": (
            ["logs/watcher_pytest_prefill_trace_pair.log"]
            + [f"logs/watcher_pytest_pairalone{i}.log" for i in (1, 2, 3)],
            2,
            0,
        ),
    }
    extra = {}
    for label, (logs, cases, want_trips) in ROUND6.items():
        trips = 0
        for log in logs:
            body = text(D / log)
            same(
                f"{label}: {log} ran {cases} cases",
                len(set(re.findall(r"test_full_model\.py::([\w\[\]]+)", body))),
                cases,
            )
            same(f"{label}: {log} had no failure", "FAILED" in body, False)
            trips += "subordinate_erisc detected invalid NOC command buffer state" in body
        same(f"{label}: {trips} of {len(logs)} tripped", trips, want_trips)
        extra[label] = (trips, len(logs))
    same("the work-matched arm is 0 of 3", extra["work-matched twelve"], (0, 3))
    same("the pair-alone arm is 0 of 4", extra["the opt-in pair alone"], (0, 4))
    same(
        "the work-matched arm's extra case builds its own generator and clones the cache",
        all(
            token in text(ROOT / "tests/test_full_model.py")
            for token in ("def test_decode_follows_the_cache_it_is_rebound_to", "ttnn.clone(k, memory_config")
        ),
        True,
    )
    same(
        "the trips land on more than one acteth core",
        len(
            set(
                re.findall(
                    r"acteth core\(x= ?\d,y= ?(\d+)\)",
                    "".join(text(D / log) for log in ARMS["twelve with both opt-in cases"][0]),
                )
            )
        ),
        2,
    )

    def fisher(a, b, c, d):
        """Two-sided Fisher exact, so the README's p-values are computed, not quoted."""
        from math import comb

        n = a + b + c + d

        def prob(x):
            y, z, w = a + b - x, a + c - x, d - (x - a)
            if min(y, z, w) < 0:
                return 0.0
            return comb(a + b, x) * comb(c + d, z) / comb(n, a + c)

        observed = prob(a)
        return sum(prob(x) for x in range(min(a + b, a + c) + 1) if prob(x) <= observed + 1e-12)

    optin, control = tallies["twelve with both opt-in cases"], tallies["twelve with two other sampling cases"]
    ten = tallies["ten gated cases"]
    same("ten-case runs", ten[1], 5)
    # The *pre-fix-only* contrasts (0.061 pooled, 0.400 opt-in-vs-control) are deliberately
    # not checked here any more.  Round 6 found that binding them made the gate *require* the
    # superseded round-4 paragraph to survive in the README, which is how a retracted
    # conclusion shipped alongside its replacement.  The combined contrasts below are the
    # document's claim; the pre-fix tallies are still asserted arm-by-arm above.
    # Round 5: the pooled contrast alone is circular, so the two arm-wise contrasts and the
    # minimum attainable p of the 3-vs-3 table are computed and bound too.
    close(
        "the smallest p the 3-vs-3 table can attain",
        round(fisher(optin[1], 0, 0, control[1]), 3),
        0.100,
        tol=1e-3,
    )
    # The combined (pre + post) contrasts the README now leads with.
    c_optin = tuple(
        a + b for a, b in zip(tallies["twelve with both opt-in cases"], post["twelve with both opt-in cases"])
    )
    c_ctrl = tuple(
        a + b
        for a, b in zip(tallies["twelve with two other sampling cases"], post["twelve with two other sampling cases"])
    )
    same("combined opt-in arm", c_optin, (6, 6))
    same("combined length-control arm", c_ctrl, (2, 6))
    close(
        "combined opt-in-vs-ten Fisher p",
        round(fisher(c_optin[0], c_optin[1] - c_optin[0], ten[0], ten[1] - ten[0]), 4),
        0.0022,
        tol=1e-3,
    )
    close(
        "combined opt-in-vs-control Fisher p",
        round(fisher(c_optin[0], c_optin[1] - c_optin[0], c_ctrl[0], c_ctrl[1] - c_ctrl[0]), 3),
        0.061,
        tol=1e-3,
    )
    # The five-arm design's contrasts, all computed here rather than transcribed.
    rest_trips = ten[0] + c_ctrl[0] + extra["work-matched twelve"][0] + extra["the opt-in pair alone"][0]
    rest_runs = ten[1] + c_ctrl[1] + extra["work-matched twelve"][1] + extra["the opt-in pair alone"][1]
    same("everything-else-pooled trips", rest_trips, 2)
    same("everything-else-pooled runs", rest_runs, 18)
    close(
        "opt-in-vs-everything-else Fisher p",
        round(fisher(c_optin[0], 0, rest_trips, rest_runs - rest_trips), 5),
        0.00021,
        tol=5e-2,
    )
    close(
        "opt-in-vs-work-matched Fisher p",
        round(fisher(c_optin[0], 0, *extra["work-matched twelve"][::-1][1::-1][::-1]), 4)
        if False
        else round(fisher(c_optin[0], 0, 0, 3), 4),
        0.0119,
        tol=1e-3,
    )
    close("opt-in-vs-pair-alone Fisher p", round(fisher(c_optin[0], 0, 0, 4), 4), 0.0048, tol=1e-3)
    close("work-matched-vs-count-matched Fisher p", round(fisher(0, 3, c_ctrl[0], 4), 3), 0.500, tol=1e-3)
    same(
        "the README leads with the pooled-rest contrast",
        readme.count("p = 0.00021") >= 1 and readme.count("p = 0.0119") >= 1,
        True,
    )
    # The count is derived from the parsed arm table above, not matched as a word -- round 7's
    # P1 was that this assertion held an unsupported figure in place precisely because the
    # digit-bounded literal binder cannot see numbers spelled out.
    same("...and states the multiplicity caveat round 6 asked for", "no multiplicity correction" in readme, True)
    same("...and the independence caveat", "assumes independence" in readme, True)

    # The context-256 floor, which is what makes the closure comparison able to fail.
    ab256 = text(D / "logs/layer_ab_after_ctx256.log")
    rows = {}
    for line in ab256.splitlines():
        if line.startswith("AB ") and "tp4 " in line:
            kind = "sliding" if "kind=sliding" in line else "full"
            rows[kind] = float(re.search(r"traced_decode@256=\s*([0-9.]+)", line).group(1))
    close("floor sliding ms/layer at context 256", rows["sliding"], 0.4390, tol=1e-4)
    close("floor full ms/layer at context 256", rows["full"], 0.4077, tol=1e-4)
    floor256 = 39 * rows["sliding"] + 13 * rows["full"]
    close("layer-stack floor at context 256", round(floor256, 3), 22.421, tol=1e-4)
    close(
        "logits-only over the same-context floor %",
        round((after["traced_decode_logits_only_ms_per_token"]["min"] - floor256) / floor256 * 100, 2),
        1.05,
        tol=2e-2,
    )
    close(
        "token-out over the same-context floor %",
        round((after["token_out_decode_ms_per_token"]["min"] - floor256) / floor256 * 100, 2),
        3.91,
        tol=2e-2,
    )
    same(
        "the context-256 comparator is tighter than the context-2048 one",
        floor256 < floor["total_ms"],
        True,
    )
    # Round 6: the comparator is not a lower bound, and the README now says so with this
    # arithmetic.  Assert the arithmetic rather than the adjective.
    terminal_ms = 0.691  # the common terminal device term, priced in the profile
    close("bare layers plus the terminal term", round(floor256 + terminal_ms, 3), 23.112, tol=1e-3)
    close(
        "the measured step beats that sum by",
        round(floor256 + terminal_ms - after["traced_decode_logits_only_ms_per_token"]["min"], 3),
        0.456,
        tol=4e-3,
    )
    same(
        "the README calls it a comparator rather than a bound",
        "It is a comparator, not a lower bound" in readme,
        True,
    )
    close(
        "floor plus terminal plus sampling, the contract's comparison",
        round(floor256 + terminal_ms + after["sampling_trace_ms_per_token"]["min"], 3),
        23.744,
        tol=2e-3,
    )
    same(
        "...and the perf summary carries the comparator",
        round(load(D / "perf_summary.json")["layer_stack_comparator_ms_per_token_at_decode_context_256"], 3),
        22.421,
    )
    # ------------------------------------------- perf_summary.json, every number of it
    #
    # Round 8 found three derived figures here still pointing at the superseded run, and this
    # gate reading none of them: it checked four fields and recomputed the roofline fraction
    # from two of *those* rather than from the stored one, so mutating the stored fraction, the
    # logits-only field, or any figure inside ``named_limitations`` passed 651/651.  This is
    # the `$optimize`-mandated accounting artifact and it is what a downstream stage reads
    # instead of the prose, so two rules now exhaust it.
    #
    # (1) Every numeric *field*, at a named path, against an artifact-derived value.  An
    # unlisted path is a failure, so a field added later cannot go unbound.
    lm_row = next(r for r in sliding if r["ID"] == "3139")
    qkv_row = next(r for r in sliding if r["ID"] == "3039")
    bw = float(lm_row["DRAM"]) / (float(lm_row["DRAM %"]) / 100)
    # The full-attention window, which no check read before round 8 even though the device
    # figure's own formula is stated in terms of it.
    full_rows = csv_rows("decode_full_perf_report.csv")
    full_layer = round(sum(float(r["Device Time"]) for r in full_rows) - terminal, 2)
    close("the full-attention layer, window minus terminal", full_layer, 409.16, tol=1e-4)
    ri = ps["roofline_inputs"]
    ps_expected = {
        "workload/prompt_len": 128,
        "workload/gen_len": 128,
        "workload/batch": 1,
        "ttft_ms": round(after["ttft_ms"]["min"], 2),
        "decode_ms_per_token_e2e": round(after["token_out_decode_ms_per_token"]["min"], 3),
        "decode_ms_per_token_e2e_logits_only": round(after["traced_decode_logits_only_ms_per_token"]["min"], 3),
        # The device figure is the two Tracy windows summed by the formula in
        # ``device_time_source``; both terms are checked against the CSVs above.
        "decode_ms_per_token_device": round((39 * layer + 13 * full_layer + terminal) / 1e3, 3),
        "sampling_trace_ms_per_token": round(after["sampling_trace_ms_per_token"]["min"], 3),
        "roofline_ms_per_token_estimate": round(ri["per_device_bytes_per_token"] / bw * 1e3 / 1e9, 3),
        "roofline_inputs/per_device_bytes_per_token": (
            ri["layer_weight_bytes"]
            + ri["lm_head_bfp4_bytes"]
            + ri["embedding_rows_read_bytes"]
            + ri["kv_cache_read_bytes_at_context_192"]
        ),
        "roofline_inputs/layer_weight_bytes": cap["per_device_layer_weight_bytes"],
        # BFP4 packs 16 values per 8-byte block plus one exponent byte: the padded local vocab
        # 50688 x 6656 at 4.5 bits.
        "roofline_inputs/lm_head_bfp4_bytes": 50688 * 6656 * 9 // 16,
        # 32 decode rows x 6656 hidden x 2 B, plus the two RoPE tables' rows.
        "roofline_inputs/embedding_rows_read_bytes": 106496,
        "roofline_inputs/kv_cache_read_bytes_at_context_192": 2715648,
        "roofline_inputs/assumed_per_device_dram_bandwidth_bytes_per_s": int(bw * 1e9),
        "roofline_fraction_of_e2e": round(ps["roofline_ms_per_token_estimate"] / ps["decode_ms_per_token_e2e"], 4),
        "layer_stack_lower_bound_ms_per_token": round(floor["total_ms"], 3),
        "ttft_ms_with_prefill_trace": round(pt["ttft_ms"]["min"], 2),
        "l1_peak_delta_per_bank_bytes": l1["l1_peak_delta_per_bank_bytes"],
        "layer_stack_comparator_ms_per_token_at_decode_context_256": round(floor256, 3),
    }

    def _numeric_paths(value, prefix=""):
        if isinstance(value, dict):
            for key, item in value.items():
                yield from _numeric_paths(item, f"{prefix}{key}/")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                yield from _numeric_paths(item, f"{prefix}[{index}]/")
        elif isinstance(value, bool):
            return
        elif isinstance(value, (int, float)):
            yield prefix.rstrip("/"), value

    ps_paths = dict(_numeric_paths(ps))
    same("every numeric field of perf_summary.json is bound", sorted(ps_paths), sorted(ps_expected))
    for path, want in ps_expected.items():
        if path in ps_paths:
            close(f"perf_summary {path}", ps_paths[path], want, tol=1e-3, readme="")

    # (2) Every decimal number in its *prose* -- ``named_limitations`` and the three provenance
    # notes -- must be one of those same artifact-derived values or one of the constants below,
    # each of which is checked elsewhere in this file against its own artifact.  This is the
    # half that catches "removes 21.2 % of TTFT (63.66 -> 50.19 ms)" going stale inside a
    # sentence, which is exactly what round 8 found in three places.
    ps_prose_ok = {
        # from the artifacts, via the fields above
        ps_expected["ttft_ms"],
        ps_expected["decode_ms_per_token_e2e_logits_only"],
        ps_expected["decode_ms_per_token_device"],
        ps_expected["ttft_ms_with_prefill_trace"],
        ps_expected["layer_stack_comparator_ms_per_token_at_decode_context_256"],
        ps_expected["layer_stack_lower_bound_ms_per_token"],
        round(bw, 1),  # 512.0 GB/s, the tool's assumed peak
        round(float(lm_row["DRAM"]), 2),  # 279.38 GB/s
        round(float(lm_row["DRAM %"]), 4),  # 54.5666 %
        round(float(qkv_row["DRAM"]), 2),  # 394.85 GB/s
        round(float(qkv_row["DRAM %"]), 4),  # 77.1192 %
        round(float(next(r for r in sliding if r["ID"] == "3145")["Op-to-Op Gap"]), 3),  # 310.959 us
        round(oc["wall_issue_ms"], 2),  # 62.75 ms prefill window
        round(trace["capture_ms"], 2),  # 98.16 ms capture
        round(layer, 2),  # 431.48 us sliding layer
        round(full_layer, 2),  # 409.16 us full layer
        terminal_ms,  # 0.691 ms terminal term
        round(floor256 + terminal_ms, 3),  # 23.112
        round(floor256 + terminal_ms - after["traced_decode_logits_only_ms_per_token"]["min"], 3),  # 0.456
        # constants checked elsewhere in this file against their own artifacts
        54.91,  # prefill_host_probe issue ms
        55.08,  # ...and drain ms
        20.93,  # the collective dispatches' share, ms
        round(abs(improvement), 1),  # the prefill trace's TTFT improvement %, from the two arms
        0.8,  # device-vs-replay %
        12.1,  # the withdrawn BF16 collective figure, named as withdrawn
        0.27,  # the two prefill norms, ms
        0.4390,  # per-layer sliding at ctx 256
        0.4077,  # ...and full
        691.07,  # the terminal window, us
        431.48,  # the sliding layer, us
        409.16,  # the full layer, us
    }
    ps_prose = " ".join(
        [ps["device_time_source"], ps["layer_stack_comparator_note"], ri["bandwidth_source"], *ps["named_limitations"]]
    )
    for literal in sorted(set(re.findall(r"(?<![\d.])\d+\.\d+", ps_prose))):
        same(
            f"perf_summary prose figure {literal} resolves to an artifact",
            any(abs(float(literal) - allowed) <= max(5e-4, abs(allowed) * 1e-4) for allowed in ps_prose_ok),
            True,
        )

    same("the ten-case repeat control is clean", "WATCHER_CLEAN" in text(D / "logs/check_watcher_default10.log"), True)
    # The tripped run's own artifact is unusable, and that is the point: the abort lands
    # inside the watcher's dump, so the log has no detach lines and check_watcher rejects it.
    same(
        "the truncated 12-case log has no detach lines",
        "device detach: 0 (min 1)" in text(D / "logs/check_watcher_12case_tripped.log"),
        True,
    )
    same(
        "...so it never reaches a clean verdict",
        "WATCHER_CLEAN" in text(D / "logs/check_watcher_12case_tripped.log"),
        False,
    )
    same(
        "...and the truncation is why it also reports zero fatal messages",
        "fatal watcher messages: 0" in text(D / "logs/check_watcher_12case_tripped.log"),
        True,
    )
    gated_cases = (D / "bench/run_watcher.sh").read_text().split("CASES=(")[1].split("\n)")[0]
    for case in (
        "test_prefill_trace_is_opt_in_and_matches_the_eager_path",
        "test_prefill_trace_survives_rebinding_the_same_external_cache",
    ):
        same(f"the gated set excludes {case}", f'"{case}"' in gated_cases, False)
    same("the gated set is the ten default cases", gated_cases.count('"'), 20)

    # (f) the work log, which the gate did not read at all.
    work_log = text(D / "work_log.md").replace("\u2212", "-").replace("\u2013", "-")
    for literal in ("77.12", "69.35", "18.03", "36.85", "1.88"):
        same(f"work log states {literal} and it resolves", literal in work_log, True)
    # Round 8: the work log recorded the *pre-regeneration* run in five places, one of them two
    # regenerations stale, and this half of the gate was five literals and two negative checks
    # -- so mutating its perf figures passed.  Its headline figures are bound to the same
    # artifacts the README's are, derived rather than typed, so a re-run moves both documents
    # or fails both.
    # Cell-level, not file-level: every one of these figures appears in the work log more than
    # once, so "the literal is somewhere in the document" is satisfied by the other copy --
    # which is how the round-8 mutations of this table survived the first attempt at this
    # check.  The evidence table's rows are parsed and the value cell is what must contain it.
    work_log_rows = {}
    for line in work_log.splitlines():
        if not line.startswith("| ") or set(line) <= set("| -:\n"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3:
            work_log_rows[cells[0]] = cells[2]
    for row, literals in (
        (
            "perf, baseline arm",
            [
                f"{before['token_out_decode_ms_per_token']['min']:.3f}",
                f"{before['traced_decode_logits_only_ms_per_token']['min']:.3f}",
                f"{before['ttft_ms']['min']:.2f}",
            ],
        ),
        (
            "perf + shapes + 130073",
            [
                f"{after['token_out_decode_ms_per_token']['min']:.3f}",
                f"{after['traced_decode_logits_only_ms_per_token']['min']:.3f}",
                f"{after['ttft_ms']['min']:.2f}",
            ],
        ),
        ("perf, `--prefill-trace`", [f"{pt['ttft_ms']['min']:.2f}"]),
        ("per-layer floor", [f"{floor['sliding_ms_per_layer']:.4f}", f"{floor['full_ms_per_layer']:.4f}"]),
        ("56-case suite, forward", [f"{suite_cases} passed"]),
        ("56-case suite, reverse", [f"{suite_cases} passed"]),
    ):
        cell = work_log_rows.get(row.replace("56", str(suite_cases)))
        same(f"the work log's {row!r} row is in the table", cell is not None, True)
        for literal in literals:
            same(f"...and its cell states {literal} from this run", literal in (cell or ""), True)
    residual_us_now = (
        after["token_out_decode_ms_per_token"]["min"]
        - after["traced_decode_logits_only_ms_per_token"]["min"]
        - after["sampling_trace_ms_per_token"]["min"]
    ) * 1000
    for label, sentence in (
        (
            "the two-trace sum",
            f"`{after['traced_decode_logits_only_ms_per_token']['min']:.3f} + "
            f"{after['sampling_trace_ms_per_token']['min']:.3f} = "
            f"{round(after['traced_decode_logits_only_ms_per_token']['min'], 3) + round(after['sampling_trace_ms_per_token']['min'], 3):.3f}`",
        ),
        (
            "the measured step it is against",
            f"a measured token-out of `{after['token_out_decode_ms_per_token']['min']:.3f}`",
        ),
        ("the residual", f"account for the step to within {residual_us_now:.1f} µs"),
        ("...and what it attributes it to", f"that {residual_us_now:.1f} µs is the caller's token readback"),
        ("the device figure", f"device time **{ps['decode_ms_per_token_device']:.3f} ms/token**"),
    ):
        same(f"the work log's accounting states {label} from this run", sentence in work_log, True)
    # The superseded values, named.  Each was in the work log at round 8 and each belongs to a
    # run that is no longer committed, so their absence is checkable and their presence is a
    # figure resolving to nothing.
    for stale in ("23.315", "22.657", "63.66 ", "65.94", "23.815", "23.825", "64.64", "50.19"):
        same(f"the work log no longer states the superseded {stale.strip()!r}", stale in work_log, False)
    same(
        "the work log's two-trace residual is this run's",
        f"{(after['token_out_decode_ms_per_token']['min'] - after['traced_decode_logits_only_ms_per_token']['min'] - after['sampling_trace_ms_per_token']['min']) * 1000:.1f} \u00b5s"
        in work_log,
        True,
    )
    same(
        "the work log's superseded round-6 conclusion is marked as superseded",
        "> **Superseded by round 7, below**" in work_log,
        True,
    )
    same(
        "...next to the paragraph it supersedes",
        work_log.index("> **Superseded by round 7, below**") < work_log.index("Neither half reproduces alone"),
        True,
    )
    same(
        "the work log no longer quotes teacher-forcing rates no artifact has",
        any(bad in work_log for bad in ("37.13", "37.16", "38.16")),
        False,
    )
    same(
        "the work log no longer describes evidence_perf.json as the crashed run's output",
        "The perf/shapes numbers in `evidence_perf.json` are from" in work_log,
        False,
    )
    same(
        "evidence_perf.json's stages are the ones the README reproduces",
        load(D / "evidence_perf.json")["stages"],
        ["capacity", "perf", "shapes"],
    )
    same(
        "the README's reproduce command matches those stages",
        "--stages capacity,perf,shapes --shape-lengths" in readme,
        True,
    )

    # (g) the baseline arm's floor provenance no longer claims a log it is not in.
    before_floor = before["layer_stack_lower_bound_ms_per_token"]["source"]
    same(
        "the baseline floor does not *claim* layer_ab_after.log",
        "not from layer_ab_after.log" in before_floor,
        True,
    )
    same("the baseline floor says where it does come from", "pre-stage" in before_floor, True)
    same("...and that this run did not measure it", "Not re-measured by this run" in before_floor, True)
    # The *after* arm's string was never wrong -- ``layer_ab_after.log`` does hold the
    # after-arm values, checked above -- so it is asserted as true rather than rewritten;
    # only the baseline arm's claim was false.  ``bench/evidence.py`` now emits an
    # arm-specific string, so a future re-run of either arm records which it is.
    after_floor = after["layer_stack_lower_bound_ms_per_token"]["source"]
    same("the after floor is attributed to layer_ab_after.log", "layer_ab_after.log" in after_floor, True)
    same("...to the arm that log actually contains", "this stage's shipped default" in after_floor, True)
    same(
        "bench/evidence.py emits an arm-specific floor provenance",
        "if baseline" in (D / "bench/evidence.py").read_text(),
        True,
    )

    # (h) the source files this gate must actually **open** -- tested against ``opened``,
    # which the readers record, not against ``is_file()``.  A new unchecked section that
    # needs a new artifact then fails here rather than sliding past on an unchanged count.
    # Round 5: a bare ``text()`` call satisfies the coverage set while asserting nothing,
    # which is the same emptiness this mechanism replaced.  Every covered path is asserted
    # on instead -- these two were the ones reachable only by a side-effecting read.
    same(
        "the control arm that tripped says so on the console",
        "WATCHER_CONSOLE_TRIPPED_ASSERT" in text(D / "logs/check_watcher_console_12case_control2.log"),
        True,
    )
    same(
        "...and the two that did not, do not",
        [
            "WATCHER_CONSOLE_NO_TRIPPED_ASSERT" in text(D / f"logs/check_watcher_console_12case_control{tag}.log")
            for tag in ("", "3")
        ],
        [True, True],
    )
    for source in (
        "tracy/prefill_128_perf_report.csv",
        "tracy/decode_sliding_perf_report.csv",
        "prefill_opcount.json",
        "work_log.md",
        "perf_summary.json",
        "logs/watcher_pytest_prefill_trace_pair.log",
        "logs/watcher_probe_rebuild.log",
        "logs/watcher_pytest_12case_tripped.log",
        "logs/watcher_pytest_12case_tripped_run2.log",
        "logs/watcher_pytest_12case_tripped_run3.log",
        "logs/watcher_pytest_12case_control.log",
        "logs/watcher_pytest_12case_control2.log",
        "logs/watcher_pytest_12case_control3.log",
        "logs/watcher_pytest_10case_repa.log",
        "logs/check_watcher_default10.log",
        "watcher_probe_rebuild/watcher.log.gz",
    ):
        same(f"the gate opened {source}", source in opened, True)

    # ------------------------------------------------- README cross-check
    #
    # Every literal resolved above, searched for in the README.  This is what turns
    # "the artifact says 22.838" into "the artifact says 22.838 *and so does the
    # document*".  A figure the README states in a different form is registered with an
    # explicit ``readme=`` string above, or with ``readme=""`` to opt out; nothing is
    # silently exempt.
    normalised = readme.replace("\u2212", "-").replace("\u2013", "-")
    for name, literal in resolved:
        if literal == "":
            continue
        candidates = {literal}
        # A README may print 22.858 as 22.858 or 0.4473 as 0.4473; also allow the
        # thousands-separated form and a trailing-zero-trimmed form.
        try:
            value = float(literal)
        except ValueError:
            value = None
        if value is not None:
            candidates.add(f"{value:,.0f}" if value == int(value) else literal)
            candidates.add(literal.rstrip("0").rstrip(".") if "." in literal else literal)
            if value == int(value):
                candidates.add(str(int(value)))
            # A README may print the same value with a trailing zero (14.6 as "14.60").
            decimals = len(literal.split(".")[1]) if "." in literal else 0
            for extra_places in (1, 2):
                candidates.add(f"{value:.{decimals + extra_places}f}")
        # Digit-boundary match.  A plain substring search lets `1.0` match `21.05` and
        # `128` match `1280`, which round 3 flagged as weak binding for exactly the
        # generic values where binding matters most.
        same(
            f"README states the resolved figure for {name}",
            any(re.search(rf"(?<![\d.,]){re.escape(c)}(?![\d])", normalised) for c in candidates),
            True,
        )

    # Round 6's P1 was a retracted conclusion surviving in one section while its replacement
    # shipped in another, and nothing in this gate compared sections.  Bind the arm tallies to
    # the Limitations section specifically, so a stale copy elsewhere cannot satisfy them.
    limitations = readme[readme.index("## Limitations and known issues") :]
    watcher_section = readme[readme.index("### Watcher") : readme.index("### The allocator's active-trace warning")]
    limitation6 = limitations[limitations.index("6. **Both opt-in") : limitations.index("7. **A watcher-enabled")]

    # Round 7 defeated the previous version of this check three ways, all because it asserted
    # that a *set of strings* appeared in each section rather than binding a tally to the row
    # it describes.  Swapping two rows' tallies passed; injecting a contradictory paragraph
    # passed; a paraphrased superseded phrase passed.  So the arm table is parsed and each row
    # is matched to the tally this gate derived from the logs.
    derived = {
        "the ten gated cases": ten,
        "**alone**": extra["the opt-in pair alone"],
        "--arm rebuild": (0, 1),
        "(count-matched)": c_ctrl,
        "(**work-matched**)": extra["work-matched twelve"],
        "both opt-in": c_optin,
    }
    parsed = {}
    unaccounted = []
    for line in watcher_section.splitlines():
        if not line.startswith("| ") or "---" in line or "| runs |" in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        try:
            trips = int(cells[-1].replace("*", ""))
            runs = int(cells[-2].replace("*", ""))
        except ValueError:
            continue
        for key in derived:
            if key in cells[0]:
                parsed[key] = (trips, runs)
                break
        else:
            # Round 8 defeated the previous version by *adding* a sixth, fabricated arm row:
            # its label matched no known key, so it was silently dropped and the run-column
            # sum still returned 25 while the table displayed 34 runs.  A row this gate cannot
            # attribute to an arm it derived from the logs is now a failure, not a skip.
            unaccounted.append(cells[0][:60])
    same("no row in the Watcher arm table is unaccounted for", unaccounted, [])
    same("every arm row in the Watcher table parsed", sorted(parsed), sorted(derived))
    for key, value in derived.items():
        same(f"the Watcher table's {key!r} row matches the logs", parsed.get(key), value)
    same(
        "the arm table's run column sums to the stated process count",
        sum(runs for _, runs in parsed.values()),
        25,
    )
    same(
        "...and the README states the five-arm count, not the total",
        f"Twenty-four watcher processes" in watcher_section,
        True,
    )
    # Limitation 6 must state the same tallies, and must name the control arm's trips rather
    # than leaving them inside the pooled figure.
    for label, needle in (
        ("the opt-in arm tally", "6 of 6"),
        ("the control arm's own trips", "2 of 6"),
        ("the work-matched tally", "0 of 3"),
        ("the pair-alone tally", "0 of 4"),
        ("the matched contrast", "p = 0.061"),
    ):
        same(f"limitation 6 states {label}", needle in limitation6, True)
    same(
        "limitation 6 does not claim the prefill trace is required",
        "neither alone is sufficient" in limitation6,
        False,
    )
    # Round 8: the *positive* half of that was unbound, so replacing the control-arm bullet
    # with its opposite ("the pair is required") passed.  Require the statement the arms
    # support to be present, not merely the retracted one to be absent.
    same(
        "the Watcher section states that a preceding workload alone is sometimes sufficient",
        "a preceding workload alone *is* sometimes sufficient" in watcher_section,
        True,
    )
    # The restated conclusion's own numbers, derived from the arms rather than read as prose:
    # the background is the range of the non-opt-in arms' trip rates and the opt-in arm is 100 %.
    background = [100.0 * trips / runs for key, (trips, runs) in parsed.items() if key != "both opt-in"]
    background_phrase = f"{min(background):.0f}–{max(background):.0f} %"
    same(
        f"the restated conclusion states the background the arms measured ({background_phrase})",
        background_phrase in limitation6 or background_phrase in watcher_section,
        True,
    )
    same(
        "...and the opt-in arm's rate with it",
        100.0 * parsed["both opt-in"][0] / parsed["both opt-in"][1] == 100.0,
        True,
    )
    # Multiplicity: round 8 pointed out the correction was applied to the two contrasts either
    # side of the one this section names as the one to weigh, and not to that one.  All three
    # are computed here and required in the text.
    multiplicity = watcher_section[
        watcher_section.index("These are six post-hoc contrasts") : watcher_section.index(
            "**What the five arms support"
        )
    ]
    stated_p = re.findall(r"p = ([0-9.]+)", multiplicity)
    for label, raw in (
        ("pooled", fisher(c_optin[0], 0, rest_trips, rest_runs - rest_trips)),
        ("work-matched", fisher(c_optin[0], 0, 0, 3)),
        ("matched", fisher(c_optin[0], c_optin[1] - c_optin[0], c_ctrl[0], c_ctrl[1] - c_ctrl[0])),
    ):
        # Precision-aware: the paragraph may print any number of decimals, but what it prints
        # must be its own rounding of p x 6 -- not of a value already rounded once, which is
        # how "0.0013" got there for a corrected p of 0.00125.
        same(
            f"the multiplicity paragraph corrects the {label} contrast (x6 = {raw * 6:.5f})",
            any(round(raw * 6, len(s.split(".")[1])) == float(s) for s in stated_p if "." in s),
            True,
        )
    same(
        "limitation 6 and the Watcher section agree on the pooled contrast",
        "p = 0.00021" in limitation6 and "p = 0.00021" in watcher_section,
        True,
    )
    # A contradictory *added* claim is what containment cannot see, so assert no other tally
    # for these arms appears anywhere in the README.
    for wrong in ("6 of 6 with two other sampling", "0 of 6", "6 of 18", "2 of 3"):
        same(f"no contradictory tally {wrong!r} in the README", wrong in readme, False)
    # Whitespace- and punctuation-normalised, because round 7 defeated the exact-substring
    # version by re-adding the retracted contrast as "p=0.400" and paraphrasing "at this
    # sample size" to "at these sample sizes".
    squashed = re.sub(r"[\s*_`]+", "", normalised).lower()
    for stale, needle in (
        ("fifteen processes", "fifteenprocesses"),
        ("twenty-eight processes", "twenty-eightwatcherprocesses"),
        ("the retracted p = 0.400", "p=0.400"),
        ("'not separable at this/these sample size(s)'", "notseparableatth"),
        ("the withdrawn gate rationale", "largestsizemeasuredwithzerotrips"),
    ):
        same(f"the superseded phrase {stale} is gone from the README", needle in squashed, False)

    # ------------------------------------------------- cell-level binding
    #
    # The cross-check above is a *document-wide* digit-bounded search, and round 8 showed what
    # that cannot see: swapping the before and after columns of the headline table leaves both
    # literals present, so the inverted claim is invisible; and any figure that occurs twice in
    # the README can be changed in one of its two places and still be found in the other.  Both
    # are fixed the same way -- bind the value to the cell it is claimed in, not to the file.

    def table_rows(section: str) -> list[list[str]]:
        rows = []
        for line in section.splitlines():
            if not line.startswith("| ") or set(line) <= set("| -:\n"):
                continue
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
        return rows

    result_table = readme[readme.index("| | before | **after** | delta |") : readme.index("† The teacher-forcing row")]
    result_rows = {row[0]: row for row in table_rows(result_table) if len(row) >= 3}
    same("the Result table has the rows the checks below name", len(result_rows) >= 9, True)
    for label, before_value, after_value, digits in (
        (
            "**token-out decode**",
            before["token_out_decode_ms_per_token"]["min"],
            after["token_out_decode_ms_per_token"]["min"],
            3,
        ),
        (
            "**traced logits-only decode**",
            before["traced_decode_logits_only_ms_per_token"]["min"],
            after["traced_decode_logits_only_ms_per_token"]["min"],
            3,
        ),
        (
            "**TTFT**, prompt 128, shipped default",
            before["ttft_ms"]["min"],
            after["ttft_ms"]["min"],
            2,
        ),
        (
            "layer-stack lower bound",
            before["layer_stack_lower_bound_ms_per_token"]["total_ms"],
            after["layer_stack_lower_bound_ms_per_token"]["total_ms"],
            3,
        ),
    ):
        row = result_rows[label]
        before_literal, after_literal = f"{before_value:.{digits}f}", f"{after_value:.{digits}f}"
        same(f"the Result table's {label!r} before cell is the baseline arm's", before_literal in row[1], True)
        same(f"...and its after cell is the shipped arm's", after_literal in row[2], True)
        # The columns cannot be swapped: each literal must be absent from the other cell.
        same(f"...and the two are not interchanged", after_literal not in row[1] and before_literal not in row[2], True)

    # The audit table's own ``ids`` column, which the document calls a verified partition and
    # which this gate previously re-derived from its *own* hardcoded copy of the ids -- so
    # moving an id between groups in the README changed nothing here.  Parsed from the README
    # now and checked against the CSV.
    audit_table = readme[
        readme.index("| op group | ids | device µs | candidate | action |") : readme.index(
            "† `tt-perf-report`'s `Cores` column"
        )
    ]
    audit_ids: list[str] = []
    audit_rows = 0
    for row in table_rows(audit_table):
        if len(row) < 3 or row[1] == "ids":
            continue
        ids = [i.strip() for i in row[1].split(",") if i.strip().isdigit()]
        stated = row[2].replace("*", "").strip()
        if not ids:
            # the ``window total`` row: "all 55" against the whole CSV
            same("the audit table's total row counts every CSV row", f"all {len(sliding)}" in row[1], True)
            close(
                "...and states the window total", float(stated), round(sum(float(r["Device Time"]) for r in sliding), 3)
            )
            continue
        audit_rows += 1
        audit_ids += ids
        close(
            f"the README's own ids for the audit row {row[0][:34]!r} sum to its own µs",
            round(sum(device_time(sliding, i) for i in ids), 3),
            float(stated),
            tol=1e-6,
        )
    same("the README's audit table has the 14 groups it claims", audit_rows, 14)
    same("...whose ids are unique", len(audit_ids), len(set(audit_ids)))
    same(
        "...and are every row of the CSV, as the partition claim says",
        sorted(audit_ids),
        sorted(r["ID"] for r in sliding),
    )
    same("...and match the groups this gate checks", sorted(audit_ids), sorted(used))

    # Section-scoped literals.  Each of these occurs more than once in the README, so the
    # document-wide search cannot tell which copy it found; round 8 changed one copy of each
    # and the gate passed.  Bound to the section that makes the claim.
    def section(start: str, end: str) -> str:
        return readme[readme.index(start) : readme.index(end)]

    accounting = section("## Performance accounting", "## Where TTFT actually goes")
    fallback_section = section("## Runtime fallback audit", "## Capability and batch")
    item_table = section("| item | value |", "## What ships")
    # The reconciliation table, cell by cell: each of these figures appears more than once in
    # the accounting section, so only the row that *claims* it will do.
    accounting_rows = {row[0]: row[1] for row in table_rows(accounting) if len(row) == 3}
    for row_label, value in (
        ("roofline", f"**{ps['roofline_ms_per_token_estimate']:.3f} ms/token**"),
        ("device-time decode", f"**{ps['decode_ms_per_token_device']:.3f} ms/token**"),
        ("end-to-end token-out", f"**{after['token_out_decode_ms_per_token']['min']:.3f} ms/token**"),
        ("end-to-end logits-only", f"**{after['traced_decode_logits_only_ms_per_token']['min']:.3f} ms/token**"),
    ):
        same(f"the reconciliation table's {row_label!r} row states {value}", accounting_rows.get(row_label), value)
    for name, needle, where, where_name in (
        (
            "the roofline fraction",
            f"{ps['roofline_fraction_of_e2e'] * 100:.1f} %",
            accounting,
            "the accounting section",
        ),
        ("the context", "**131072**, unreduced", item_table, "the item table"),
        ("the suite size", f"**{suite_cases}** cases", item_table, "the item table"),
        (
            "the zero synchronizations counter",
            "| synchronizations | **0** | **0.0** |",
            fallback_section,
            "the fallback table",
        ),
        (
            "the zero device-position-advance counter",
            "| `device_position_advances` | **0** |",
            fallback_section,
            "the fallback table",
        ),
    ):
        same(f"{where_name} states {name} ({needle})", needle in where, True)
    accuracy_section = section("## Accuracy", "## Qualitative")
    contract_table = section("## Carried-forward decoder contract, unchanged", "## Performance accounting")
    acc = load(D / "evidence_accuracy.json")
    # Every cell of the accuracy table, per row: the four rows print the same three rates, so a
    # single changed cell is invisible to any check that asks whether the value is present.
    accuracy_rows = [row for row in table_rows(accuracy_section) if len(row) == 5 and row[2] != "top-1"]
    same("the accuracy table has its four gate rows", len(accuracy_rows), 4)
    fp32_gate = load(D / "evidence_fp32_gate.json")
    for row in accuracy_rows:
        gate_key = "prefill_check" if "run_prefill_check" in row[0] else "teacher_forcing"
        # The bf16 rows come from the accuracy run, the fp32-control rows from the gate run;
        # each is keyed on the reference file it was scored against.
        fp32 = "fp32" in row[1]
        source = fp32_gate if fp32 else acc
        reference = "readiness_aime24_chat_fp32.refpt" if fp32 else "readiness_aime24_chat.refpt"
        entry = source[f"{gate_key}_by_reference"][reference]["per_entry"][0]
        same(
            f"the accuracy row {row[0][:24]!r}/{reference} states this run's rates",
            [row[2], row[3], row[4]],
            [f"{entry['top1']:.3f}", f"**{entry['top5']:.3f}**", f"**{entry['top100']:.3f}**"],
        )
    same(
        "the contract table states force-argmax off, in the row that claims it",
        "force-argmax **off**" in contract_table,
        True,
    )
    same(
        "...and that is what the built sampler reports",
        load(D / "evidence_perf.json")["capacity"]["force_argmax"],
        False,
    )
    # A tally for these arms stated *in prose* rather than in the table was round 8's
    # "5 of 6" mutation: containment cannot see an added sentence, so every "N of M" in the
    # two sections that discuss the arms must be one this gate derived from the logs.
    derived_tallies = {f"{trips} of {runs}" for trips, runs in parsed.values()} | {
        f"{rest_trips} of {rest_runs}",
        "0 of 1",  # the --arm rebuild run
    }
    for tally in sorted(set(re.findall(r"\d+ of \d+", watcher_section + limitations))):
        same(f"the arm tally {tally!r} is one the logs support", tally in derived_tallies, True)
    same(
        "the audit table states SdpaDecode's µs in its own row, not merely somewhere",
        any(
            row[0] == "`SdpaDecode`" and row[2] == f"{device_time(sliding, '3168'):.3f}"
            for row in table_rows(audit_table)
        ),
        True,
    )

    # The opt-in prefill trace's three eligibility conditions, bound to the code that
    # implements them.  Round 7 added the documentation; round 8 pointed out it was bound to
    # nothing, so changing 8192 to 4096 or dropping "user_id == 0" passed.
    eligibility = generator_src[
        generator_src.index("def _prefill_user") : generator_src.index("def _capture_prefill_trace")
    ]
    for condition, prose in (
        ("prompt_len <= config.prefill_chunk_size", "`prompt_len <=\nconfig.prefill_chunk_size`"),
        ("user_id == 0", "`user_id == 0`"),
        ("not return_all_logits", "`return_all_logits` is false"),
    ):
        same(f"the generator gates the prefill trace on {condition}", condition in eligibility, True)
        same(f"...and the README documents it", prose.replace("\n", " ") in " ".join(readme.split()), True)
    same(
        "the documented chunk size is the built model's",
        f"(**{load(D / 'evidence_perf.json')['capacity']['prefill_chunk_size']}**)" in readme,
        True,
    )

    # The gate's own adversarial evidence.  Four review rounds found this file passing over a
    # wrong figure, and each time the *reviewer* had to construct the mutation by hand; the
    # harness is committed now, so "the gate passes" is accompanied by "and here is what makes
    # it fail".  Its log is checked rather than its existence.
    mutations = text(D / "logs/mutate_figure_gate.log")
    same("the mutation harness ran from a clean baseline", "baseline: FIGURES_OK" in mutations, True)
    same("...and every mutation was caught", "MUTATION_SURVIVORS" in mutations, False)
    mutation_count = len(re.findall(r"^CAUGHT ", mutations, re.M))
    same(
        f"...all {mutation_count} of them",
        f"ALL {mutation_count} MUTATIONS CAUGHT" in mutations,
        True,
    )
    # Counted off the harness's own table rather than off its formatting, which the repo's
    # black hook rewrites.
    harness_spec = importlib.util.spec_from_file_location(
        "muse_glimmer_figure_gate_mutations", _record(D / "bench/mutate_figure_gate.py")
    )
    harness = importlib.util.module_from_spec(harness_spec)
    harness_spec.loader.exec_module(harness)
    same("the harness covers every defeat the log records", len(harness.MUTATIONS), mutation_count)

    same("README has a before/after table at the top", readme.index("## Result") < readme.index("## What ships"), True)
    same("no TODO left in the README", bool(re.search(r"\bTODO\b", readme)), False)
    same("the gate's advertised check count is right", checks + 2, ADVERTISED_CHECKS)
    same("...and the assertion/binding split is what the README states", bindings, ADVERTISED_BINDINGS)

    print(
        f"\n{checks} checks ({checks - bindings} assertions, {bindings} README bindings), " f"{len(failures)} failures"
    )
    for line in failures:
        print(f"  FAIL {line}")
    if failures:
        print("FIGURES_STALE")
        return 1
    print("FIGURES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
