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
import json
import pathlib
import re

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
#: The count this file advertises in the README and the work log.  Asserted, so the
#: number cannot drift the way the previous one did.
#:
#: Round 3 of the stage review pointed out the obvious limit of this: the count freezes
#: how many checks there are, not what they cover, and 328/328 passed while six figures in
#: sections this file never opened were wrong.  ``SOURCES`` below is the other half --
#: adding an unchecked section that needs a new artifact now fails on a missing source
#: rather than sliding past on an unchanged count.
ADVERTISED_CHECKS = 479


def load(path: pathlib.Path):
    return json.loads(path.read_text())


def text(path: pathlib.Path) -> str:
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


def bind(name: str, literal: str) -> None:
    """Register a literal for the README cross-check without a vacuous numeric compare.

    ``same(x, True, True)`` reads like an assertion and is not one; the only thing those
    calls ever carried was the ``readme=`` registration.  This says that plainly.
    """
    global checks
    checks += 1
    resolved.append((name, literal))
    print(f"bind {name}: {literal}")


def csv_rows(name: str) -> list[dict]:
    return list(csv.DictReader(open(D / "tracy" / name)))


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
        global checks
        checks += 1
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
    same("53 cases passed forward", "53 passed" in text(D / "logs/full_test_run.log"), True, readme="53 passed")
    same("53 cases passed in reverse", "53 passed" in text(D / "logs/full_test_run_reverse.log"), True)
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
    generator_src = (ROOT / "tt/generator.py").read_text()
    same("the retirement flag is gone from the generator", "_prefill_trace_retired" in generator_src, False)
    same("a cache move recaptures instead", "_prefill_trace_releases" in generator_src, True)
    same("the gated watcher run passed 10 cases", "10 passed" in text(D / "logs/watcher_pytest.log"), True)
    same(
        "the gated watcher run tripped nothing on the console either",
        "WATCHER_CONSOLE_NO_TRIPPED_ASSERT" in text(D / "logs/check_watcher_console.log"),
        True,
    )
    # The positive control for limitation 6: all twelve pass, and *then* it trips.  Both runs.
    trip1 = text(D / "logs/watcher_pytest_12case_tripped.log")
    trip2 = text(D / "logs/watcher_pytest_12case_tripped_run2.log")
    for tag, trip in (("run 1", trip1), ("run 2", trip2)):
        same(
            f"the 12-case {tag} ran all twelve cases",
            len(set(re.findall(r"test_full_model\.py::([\w\[\]]+)", trip))),
            12,
        )
        same(f"the 12-case {tag} passed all twelve", trip.count("PASSED"), 12)
        same(
            f"the 12-case {tag} then tripped the assert",
            "subordinate_erisc detected invalid NOC command buffer state" in trip,
            True,
        )
    same(
        "the two 12-case runs trip on different links",
        len(set(re.findall(r"Device (\d) acteth core", trip1 + trip2))),
        2,
    )
    same("the ten-case control is clean", "WATCHER_CLEAN" in text(D / "logs/check_watcher_default10.log"), True)
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
    work_log = (D / "work_log.md").read_text().replace("\u2212", "-").replace("\u2013", "-")
    for literal in ("77.12", "69.35", "18.03", "36.85", "1.88"):
        same(f"work log states {literal} and it resolves", literal in work_log, True)
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

    # (h) the source files this gate must actually open, so a new unchecked section is
    # visible as a missing source rather than as an unchanged count.
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
        "logs/check_watcher_default10.log",
        "watcher_probe_rebuild/watcher.log.gz",
    ):
        same(f"the gate reads {source}", (D / source).is_file(), True)

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
            for extra in (1, 2):
                candidates.add(f"{value:.{decimals + extra}f}")
        # Digit-boundary match.  A plain substring search lets `1.0` match `21.05` and
        # `128` match `1280`, which round 3 flagged as weak binding for exactly the
        # generic values where binding matters most.
        same(
            f"README states the resolved figure for {name}",
            any(re.search(rf"(?<![\d.,]){re.escape(c)}(?![\d])", normalised) for c in candidates),
            True,
        )

    same("README has a before/after table at the top", readme.index("## Result") < readme.index("## What ships"), True)
    same("no TODO left in the README", bool(re.search(r"\bTODO\b", readme)), False)
    same("the gate's advertised check count is right", checks + 1, ADVERTISED_CHECKS)

    print(f"\n{checks} checks, {len(failures)} failures")
    for line in failures:
        print(f"  FAIL {line}")
    if failures:
        print("FIGURES_STALE")
        return 1
    print("FIGURES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
