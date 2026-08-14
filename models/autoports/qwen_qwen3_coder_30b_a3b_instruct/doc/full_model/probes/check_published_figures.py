# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Every number in the stage-05 documents, re-derived from the artifact it cites.

Stage 04 left this pattern behind because prose/artifact drift has failed a
review in every stage of this project, and it left the rule with it: **a checker
that reads the document it is checking checks nothing.** This is its stage-05
continuation, and the stage-05 review found the rule had been let slip -- 15 of
the then-78 assertions restated constants defined a few lines above them inside
this file, and 10 of those were worthless. Specifically: the layer-stack lower
bound asserted `48 * 0.4286` against its own literal while 0.4286 sat in
`../../optimized_multichip_decoder/perf_decode.csv`; the rope figures were
hardcoded while the rope log was open in the same function; and the three
headline ratios were computed and then **discarded into the failure-detail
string**, with the actual assertion being an unanchored `"1.79" in README`
substring search over 478 lines.

Every figure below is now parsed out of an artifact -- a CSV row, a JSON field,
a probe log line -- and the README/work_log string is compared against the
*computed* rounding of it. Where a document quotes a ratio, the ratio is
recomputed from its two operands and the quoted string must equal the rounding
of the result, not merely appear somewhere in the file.

Runs on the host in under a second; no device.

    python .../probes/check_published_figures.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parents[1]
PROBES = DOC / "probes"
MODEL_DIR = DOC.parents[1]

README = (DOC / "README.md").read_text(encoding="utf-8")
WORK_LOG = (DOC / "work_log.md").read_text(encoding="utf-8")
CONTRACT = json.loads((MODEL_DIR / "doc" / "context_contract.json").read_text(encoding="utf-8"))
PERF = json.loads((PROBES / "perf_full_model.json").read_text(encoding="utf-8"))
FOOTPRINT = json.loads((PROBES / "footprint_262144.json").read_text(encoding="utf-8"))

failures: list[str] = []
checks = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global checks
    checks += 1
    if not ok:
        failures.append(f"{name}: {detail}")
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail and not ok else ''}")


def in_readme(text: str) -> bool:
    return text in README


def quotes(document: str, value: float, spec: str, *, name: str) -> tuple[bool, str]:
    """``value`` formatted by ``spec`` must appear in ``document``.

    The formatting is done here from the artifact's number, so the assertion
    fails when the document drifts, not when this file drifts.
    """
    text = format(value, spec)
    return text in document, f"{name}: computed {text}"


def ratio_is_quoted(document: str, numerator: float, denominator: float, spec: str, *, name: str):
    """A ratio quoted in prose must equal the rounding of its two operands.

    Both operands come from artifacts. This is the check the previous revision
    dropped: it computed the ratio and then asserted only that some literal
    string occurred somewhere in the README.
    """
    value = numerator / denominator
    text = format(value, spec)
    return text in document, f"{name}: {numerator} / {denominator} = {value!r} -> {text}"


# --- performance ------------------------------------------------------------

perf_rows = {}
with (PROBES / "perf_full_model.csv").open() as handle:
    for row in csv.DictReader(handle):
        perf_rows[row["metric"]] = float(row["ms"])

for metric, key in (
    ("ttft", "ttft_ms"),
    ("model_trace", "model_trace_ms"),
    ("token_out", "token_out_ms"),
    ("token_out_readback", "token_out_readback_ms"),
    ("sampler_split", "sampler_split_ms"),
    ("sampler_force_argmax", "sampler_force_argmax_ms"),
):
    check(
        f"csv and json agree on {metric}",
        abs(perf_rows[metric] - PERF[key]) < 1e-9,
        f"{perf_rows[metric]} vs {PERF[key]}",
    )

for label, value in (
    ("TTFT", PERF["ttft_ms"]),
    ("cold TTFT", PERF["ttft_cold_ms"]),
    ("token-out", PERF["token_out_ms"]),
    ("token-out readback", PERF["token_out_readback_ms"]),
    ("model trace", PERF["model_trace_ms"]),
):
    check(f"README quotes {label} to 2dp", in_readme(f"{value:.2f}"), f"{value:.2f}")

for label, value in (
    ("token-out t/s/u", PERF["token_out_tps_user"]),
    ("token-out readback t/s/u", PERF["token_out_readback_tps_user"]),
    ("logits-only t/s/u", PERF["model_trace_tps_user"]),
):
    check(f"README quotes {label} to 2dp", in_readme(f"{value:.2f}"), f"{value:.2f}")

for label, value in (("split sampler", PERF["sampler_split_ms"]), ("force argmax", PERF["sampler_force_argmax_ms"])):
    check(f"README quotes {label} to 3dp", in_readme(f"{value:.3f}"), f"{value:.3f}")

check(
    "both sampler strategies returned the same token",
    PERF["sampler_split_token"] == PERF["sampler_force_argmax_token"] == 16,
    f"{PERF['sampler_split_token']} vs {PERF['sampler_force_argmax_token']}",
)

# --- the layer-stack lower bound --------------------------------------------

# The per-layer figure is a row of the stage-04 artifact, not a literal here.
with (MODEL_DIR / "doc" / "optimized_multichip_decoder" / "perf_decode.csv").open() as handle:
    stage04_decode = {int(row["context_len"]): float(row["median_ms"]) for row in csv.DictReader(handle)}
per_layer_ms = stage04_decode[128]
check(
    "README quotes the stage-04 per-layer decode figure it multiplies",
    in_readme(f"{per_layer_ms:g}"),
    f"perf_decode.csv ctx128 median_ms = {per_layer_ms:g}",
)
lower_bound = PERF["layers"] * per_layer_ms
check(
    "README quotes the layer-stack lower bound to 2dp",
    *quotes(README, lower_bound, ".2f", name=f"{PERF['layers']} x {per_layer_ms:g}"),
)
check(
    "context contract carries the same lower bound",
    abs(CONTRACT["full_model_performance"]["layer_stack_lower_bound_ms"] - round(lower_bound, 3)) < 1e-9,
    f"{CONTRACT['full_model_performance']['layer_stack_lower_bound_ms']} vs {round(lower_bound, 3)}",
)
check(
    "the model trace really is at or under the lower bound",
    PERF["model_trace_ms"] <= lower_bound,
    f"{PERF['model_trace_ms']} vs {lower_bound}",
)
overhead = PERF["token_out_ms"] - PERF["model_trace_ms"]
check("README quotes the full-model-only cost to 2dp", in_readme(f"{overhead:.2f}"), f"{overhead:.2f}")
check(
    "README quotes the full-model-only share to 1dp",
    in_readme(f"{overhead / PERF['token_out_ms'] * 100:.1f}%"),
    f"{overhead / PERF['token_out_ms'] * 100:.1f}%",
)
check(
    "README quotes the readback cost to 2dp",
    in_readme(f"{PERF['token_out_readback_ms'] - PERF['token_out_ms']:.2f}"),
    f"{PERF['token_out_readback_ms'] - PERF['token_out_ms']:.2f}",
)

# --- footprint --------------------------------------------------------------

stages = FOOTPRINT["stages_gb_per_die"]
total = FOOTPRINT["total_gb_per_die"]
summed = sum(stages.values())
check(
    "the footprint rows sum to the total EXACTLY, with no residual",
    summed == total,
    f"raw rows {summed!r} vs raw total {total!r}",
)
# The 0.001 visible in the README's table is introduced by rounding each row to
# 3dp for display, which is what the document must now say. Asserting the
# rounded-row residual pins the explanation to arithmetic rather than to a story
# about the allocator.
rounded_residual = round(sum(round(v, 3) for v in stages.values()) - round(total, 3), 3)
check(
    "the residual the README's table shows is exactly the display rounding",
    abs(rounded_residual - 0.001) < 1e-9,
    f"rows rounded to 3dp minus total rounded to 3dp = {rounded_residual}, expected 0.001",
)
check(
    "the README attributes it to display rounding, not to an omitted term",
    "display rounding" in README or ("this table's own rounding" in README and "display" in README),
    "",
)
check(
    "the contract's sum_note records the rows as bit-identical to the total",
    "BIT-IDENTICAL" in CONTRACT["full_model_measured"]["sum_note"],
    CONTRACT["full_model_measured"]["sum_note"][:80],
)
check(
    "no document still asserts the allocator caused the 0.001",
    "the\nallocator's own rounding" not in README
    and "allocator's own rounding between" not in README
    and "the 0.001 is the allocator" not in WORK_LOG,
    "the superseded attribution is still asserted",
)
for label, value in (
    ("weights", stages["weights_embed_lm_head_rope"]),
    ("kv cache", stages["kv_cache"]),
    ("traces", stages["traces_and_persistent_buffers"]),
    ("total", total),
    ("headroom", FOOTPRINT["headroom_gb_per_die"]),
    ("dram per die", FOOTPRINT["dram_per_die_gb"]),
):
    check(f"README quotes footprint {label} to 3dp", in_readme(f"{value:.3f}"), f"{value:.3f}")

check(
    "context contract carries the same total",
    abs(CONTRACT["full_model_measured"]["total_gb_per_die"] - round(total, 3)) < 1e-9,
    f"{CONTRACT['full_model_measured']['total_gb_per_die']} vs {round(total, 3)}",
)
check(
    "context contract carries the same headroom",
    abs(CONTRACT["full_model_measured"]["headroom_gb_per_die"] - round(FOOTPRINT["headroom_gb_per_die"], 3)) < 1e-9,
    "",
)
check("no capability reduction is claimed", CONTRACT["capability_reduction"] is False, "")
check(
    "advertised and supported context still agree",
    CONTRACT["current_supported_context"] == CONTRACT["hf_advertised_context"] == 262144,
    "",
)
check(
    "context contract performance matches the perf json",
    CONTRACT["full_model_performance"]["decode_token_out_ms"] == round(PERF["token_out_ms"], 3),
    "",
)

# --- accuracy ---------------------------------------------------------------


def read_topk(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"AGGREGATE\s+top1=([\d.]+).*?top5=([\d.]+).*?top100=([\d.]+)", text)
    if not match:
        return {}
    return {"top1": float(match.group(1)), "top5": float(match.group(2)), "top100": float(match.group(3))}


for name, log in (("prefill", "run_prefill_check.log"), ("decode", "run_teacher_forcing.log")):
    stats = read_topk(DOC / log)
    check(f"{log} has an AGGREGATE line", bool(stats), str(stats))
    if not stats:
        continue
    check(f"{name} meets the top-5 bar", stats["top5"] >= 0.98, str(stats))
    check(f"{name} meets the top-100 bar", stats["top100"] == 1.0, str(stats))
    contract_key = "prefill" if name == "prefill" else "decode_teacher_forced"
    for field in ("top1", "top5", "top100"):
        check(
            f"context contract {name}.{field} matches {log}",
            CONTRACT["full_model_accuracy"][contract_key][field] == stats[field],
            f"{CONTRACT['full_model_accuracy'][contract_key][field]} vs {stats[field]}",
        )
    check(f"README quotes {name} top-1 as {stats['top1']:.3f}", in_readme(f"{stats['top1']:.3f}"), "")

# --- degeneracy, divergence and completions ----------------------------------

meta_path = MODEL_DIR / "readiness_autoregressive" / "autoregressive_meta.json"
if meta_path.is_file():
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    hf_ids, tt_ids = meta["hf"]["token_ids"], meta["tt"]["token_ids"]
    check("HF and TT both produced 128 tokens", meta["hf"]["num_tokens"] == meta["tt"]["num_tokens"] == 128, "")
    matching = sum(1 for a, b in zip(hf_ids, tt_ids) if a == b)
    check(
        f"README's '{matching} of 128 tokens match' is the recomputed count",
        in_readme(f"{matching} of 128 tokens match"),
        f"recomputed {matching}",
    )
    matching_indices = [i for i, (a, b) in enumerate(zip(hf_ids, tt_ids)) if a == b]
    check(
        "README lists the matching indices the metadata actually contains",
        all(str(i) in README for i in matching_indices),
        f"recomputed {matching_indices}",
    )
    # Where the two completions diverge is the length of their common prefix,
    # not the first index at which they happen to agree again.
    prefix = 0
    for a, b in zip(hf_ids, tt_ids):
        if a != b:
            break
        prefix += 1
    check(
        "the common prefix is what the README claims it is",
        prefix == 0 and "first generated token" in README,
        f"recomputed common prefix = {prefix} tokens; hf[0]={hf_ids[0]} tt[0]={tt_ids[0]}",
    )
    check(
        "no document still says it diverges at token 3",
        "diverges from HF at token 3" not in README and "diverges from HF at token 3" not in WORK_LOG,
        "the superseded divergence point is still present",
    )
    for label, ids in (("HF", hf_ids), ("TT", tt_ids)):
        check(f"README quotes {label}'s first generated token id", str(ids[0]) in README, f"{ids[0]}")
else:
    check("autoregressive metadata exists", False, str(meta_path))

# The degeneracy verdict must be a file, not a quotation. The stage gate is
# runner-side, and an earlier revision quoted output that existed nowhere but in
# README.md.
degeneracy_log = DOC / "check_degenerate_output.log"
check("the degeneracy check output is archived", degeneracy_log.is_file(), str(degeneracy_log))
if degeneracy_log.is_file():
    degeneracy = degeneracy_log.read_text(encoding="utf-8", errors="ignore")
    check("the archived degeneracy run found nothing", "No degenerate output detected" in degeneracy, "")
    measured = re.search(
        r"'num_tokens': (\d+).*?'adjacent_duplication': ([\d.]+).*?" r"'trigram_loop_fraction': ([\d.]+)",
        degeneracy,
        re.S,
    )
    check("the archived degeneracy run reports its metrics", bool(measured), "")
    if measured:
        for value in measured.groups():
            check(f"README quotes the degeneracy metric {value}", in_readme(value), "")

# --- the qualitative free-running suite --------------------------------------

qualitative = DOC / "qualitative_check.log"
check("the qualitative prompt suite is archived", qualitative.is_file(), str(qualitative))
if qualitative.is_file():
    text = qualitative.read_text(encoding="utf-8", errors="ignore")
    prompts = re.findall(r"^=== prompt (\d+)/(\d+):", text, re.M)
    check(
        "the qualitative suite ran several prompts, not one",
        len(prompts) >= 3,
        f"{len(prompts)} prompts in {qualitative.name}",
    )
    check("README states the qualitative prompt count", in_readme(f"{len(prompts)} prompts"), f"{len(prompts)}")
    # Two legs per prompt: greedy and sampled.
    check(
        "every qualitative prompt produced both a greedy and a sampled completion",
        text.count("--- completion") == 2 * len(prompts),
        f"{text.count('--- completion')} completions for {len(prompts)} prompts",
    )

# --- test counts ------------------------------------------------------------


def log_text(name: str) -> str:
    path = DOC / name
    if not path.is_file():
        return ""
    if path.suffix == ".gz":
        import gzip

        return gzip.open(path, "rt", errors="ignore").read()
    return path.read_text(encoding="utf-8", errors="ignore")


def passed_count(text: str) -> int | None:
    """The pytest tally, read out of the log rather than asserted against."""
    match = re.findall(r"(\d+) passed", text)
    return int(match[-1]) if match else None


for label, log in (
    ("stage-04 regression", "pytest_stage04_regression.log.gz"),
    ("stage-04 under the watcher", "pytest_stage04_watcher.log.gz"),
    ("full model, 2 layers", "pytest_full_model_2layer.log"),
    ("full model, 48 layers", "pytest_full_model_48layer.log"),
):
    text = log_text(log)
    count = passed_count(text)
    check(f"{label}: {log} has a pytest tally", count is not None, f"none found in {log}")
    if count is None:
        continue
    check(f"{label}: nothing failed", " failed" not in text.split("=====")[-1], "")
    check(f"README quotes '{count} passed' for {label}", in_readme(f"{count} passed"), f"{count}")

two_layer, all_layer = passed_count(log_text("pytest_full_model_2layer.log")), passed_count(
    log_text("pytest_full_model_48layer.log")
)
check(
    "both full-model tiers ran the same test count",
    two_layer == all_layer,
    f"2-layer {two_layer} vs 48-layer {all_layer}",
)

# --- rope probe ---------------------------------------------------------------

rope_log = (PROBES / "rope_hf_probe.log").read_text(encoding="utf-8", errors="ignore")
rope_cases = [line for line in rope_log.splitlines() if line.strip().startswith(("pos ", "batch "))]
check("the rope probe produced cases", bool(rope_cases), str(len(rope_cases)))
check(
    f"README quotes the rope case count ({len(rope_cases)})",
    in_readme(f"all {len(rope_cases)} cases"),
    f"{len(rope_cases)} cases in rope_hf_probe.log",
)
check(
    "every rope case is bit-identical at max|diff| 0.000e+00",
    all("max|diff|=0.000e+00" in line for line in rope_cases),
    "",
)
check("every rope case is PCC 1.0", all(line.rstrip().endswith("1.0") for line in rope_cases), "")

# Per-user distinct positions are the capability the swap buys; assert the probe
# actually exercised them rather than running batch 1 four times.
per_user = re.findall(r"batch\s+(\d+) \(\s*(\d+) distinct positions, max (\d+)\)", rope_log)
check("the rope probe swept several batch sizes", len({b for b, _, _ in per_user}) >= 3, str(per_user))
check(
    "at least one rope case used more than one distinct position",
    any(int(d) > 1 for _, d, _ in per_user),
    str(per_user),
)
highest = max((int(m) for _, _, m in per_user), default=0)
highest = max(highest, *(int(m.group(1)) for m in re.finditer(r"pos\s+(\d+):", rope_log)))
check(
    "the rope probe reached the advertised context",
    highest == CONTRACT["hf_advertised_context"] - 1,
    f"highest position tested {highest} against context {CONTRACT['hf_advertised_context']}",
)

# Trace slopes, parsed rather than hardcoded.
rope_slopes = dict(re.findall(r"trace slope ([^:]+): ([\d.]+) us", rope_log))
for name, value in rope_slopes.items():
    check(f"README quotes the rope slope for {name} ({value})", in_readme(value), value)
gathers = dict(re.findall(r"rope_cache_len (\d+): ([\d.]+) us", rope_log))
check("the gather slope is reported at more than one table length", len(gathers) >= 2, str(gathers))
for capacity, value in gathers.items():
    check(f"README quotes the gather slope at rope_cache_len {capacity} ({value})", in_readme(value), value)

# --- sampler sweep ------------------------------------------------------------

sampler_log = (PROBES / "sampler_probe.log").read_text(encoding="utf-8", errors="ignore")
sampler_ms = {}
for row in sampler_log.splitlines():
    match = re.match(r"(\S+)\s+\{.*'ms': ([\d.]+)", row)
    if match:
        sampler_ms[match.group(1)] = float(match.group(2))
check("the sampler log carries all four swept legs", len(sampler_ms) == 4, str(sorted(sampler_ms)))
for leg, value in sampler_ms.items():
    check(
        f"README or work log quotes {leg} to 3dp",
        f"{value:.3f}" in README or f"{value:.3f}" in WORK_LOG,
        f"{value:.3f}",
    )

# Every ratio the documents quote is recomputed from its two artifact operands,
# and the quoted string must be the rounding of the result.
check(
    "the 1.79x pad speedup is the ratio of the two swept legs",
    *ratio_is_quoted(
        README, sampler_ms["split_k32_padded"], sampler_ms["split_k32_unpadded"], ".2f", name="pad speedup"
    ),
)
check(
    "the 5.5x argmax speedup is the ratio of the two in-model rows",
    *ratio_is_quoted(README, PERF["sampler_split_ms"], PERF["sampler_force_argmax_ms"], ".1f", name="argmax speedup"),
)
check(
    "shrinking max_top_k really is worse, as the README claims",
    sampler_ms["split_k16_unpadded"] > sampler_ms["split_k32_unpadded"],
    f"k16 {sampler_ms['split_k16_unpadded']} vs k32 {sampler_ms['split_k32_unpadded']}",
)

# The pre-fix token-out figure has no archived artifact -- see the README. The
# check that can be made is that the documents say so rather than implying a
# measurement is on disk.
PRE_FIX_TOKEN_OUT = "31.826"
check(
    "the pre-fix token-out figure is labelled unarchived wherever it is quoted",
    (PRE_FIX_TOKEN_OUT not in README and PRE_FIX_TOKEN_OUT not in WORK_LOG)
    or ("unarchived" in README and "unarchived" in WORK_LOG),
    "31.826 is quoted without being marked unarchived",
)


def section_containing(document: str, needle: str) -> str:
    """The ``## `` section of ``document`` in which ``needle`` first appears."""
    index = document.find(needle)
    if index < 0:
        return ""
    start = document.rfind("\n## ", 0, index)
    end = document.find("\n## ", index)
    return document[max(start, 0) : end if end > 0 else len(document)]


# Two published ratios divide by that unarchived figure. Both are checked the
# way every other ratio in this file is -- recomputed from its operands, with
# the quoted string required to be the rounding of the result. What no check
# can establish is the operand itself, so what is asserted instead is that the
# caveat travels *with* the figure: in the same section, not merely somewhere
# in the file.
#
# The previous revision of this block asserted `"1.40" not in README or
# "unarchived" in README`, whose second clause the assertion just above already
# proves. It could not fail. A check that cannot fail is worse than no check,
# because it reads like cover.
stated = re.search(r"from \*\*([\d.]+) ms to ([\d.]+) ms\*\*,\s*([\d.]+)x", README.replace("×", "x"))
check("the README states both operands of the pre-fix ratio", stated is not None, "")
if stated:
    pre_fix, post_fix, quoted = float(stated.group(1)), float(stated.group(2)), stated.group(3)
    check(
        "the 1.40x token-out gain is the ratio of the two figures the README states",
        format(pre_fix / post_fix, ".2f") == quoted,
        f"{pre_fix} / {post_fix} = {pre_fix / post_fix!r} -> {format(pre_fix / post_fix, '.2f')}, quoted {quoted}",
    )
    check(
        "the 1.44x token-out gain divides the same pre-fix figure into the measured row",
        *ratio_is_quoted(README, pre_fix, PERF["token_out_ms"], ".2f", name="post-workaround gain"),
    )
    for value in (quoted, format(pre_fix / PERF["token_out_ms"], ".2f")):
        check(
            f"the {value}x gain carries the unarchived caveat in its own section",
            "unarchived" in section_containing(README, f"{value}×"),
            f"{value} is quoted in a section that does not say 'unarchived'",
        )

# --- the watcher A/B ----------------------------------------------------------

watcher_ab = DOC / "watcher_ab.log"
check("the watcher A/B matrix is archived", watcher_ab.is_file(), str(watcher_ab))
if watcher_ab.is_file():
    matrix = {}
    for row in watcher_ab.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = row.split()
        if len(parts) == 3 and parts[2] in {"clean", "TRIPPED"}:
            matrix[parts[1]] = parts[2]
    check("the A/B ran both barrier legs", {"argmax_nobarrier", "argmax_barrier"} <= set(matrix), str(sorted(matrix)))
    check(
        "the barrier semaphore is not the difference, as the README now says",
        matrix.get("argmax_nobarrier") == matrix.get("argmax_barrier") == "TRIPPED",
        f"nobarrier={matrix.get('argmax_nobarrier')} barrier={matrix.get('argmax_barrier')}",
    )
    check("the split sampling path is clean", matrix.get("split_k32") == "clean", str(matrix.get("split_k32")))
    check(
        "the minimal trigger is Linear + num_workers_per_link=1",
        matrix.get("linear_workers1") == "TRIPPED"
        and matrix.get("sampler_shape_workers1") == "clean"
        and matrix.get("linear_default_knobs_nobarrier") == "clean",
        f"linear_workers1={matrix.get('linear_workers1')} "
        f"ring_workers1={matrix.get('sampler_shape_workers1')} "
        f"linear_default={matrix.get('linear_default_knobs_nobarrier')}",
    )
    check(
        "the README no longer blames the missing barrier semaphore",
        "no barrier\n  semaphore" not in README and "only one that passes **no barrier semaphore**" not in WORK_LOG,
        "",
    )
    # The workaround the model actually ships must be in the matrix, and clean.
    check(
        "the shipped sampler class is exercised in the A/B",
        "argmax_shipped" in matrix,
        str(sorted(matrix)),
    )
    check(
        "the shipped sampler class is watcher-clean",
        matrix.get("argmax_shipped") == "clean",
        f"argmax_shipped={matrix.get('argmax_shipped')}",
    )

# --- the whole tree under the watcher ----------------------------------------
#
# The claim "stage 05 is watcher-clean" is a published figure like any other:
# a tally and an assert count, both read out of the archived run.

watcher_run = log_text("pytest_watcher_clean.log.gz")
check("the clean watcher run is archived", bool(watcher_run), "pytest_watcher_clean.log.gz")

# `watcher.log.gz` is the *final dump* of the clean run, not of the earlier
# failing one, and not the whole watcher log. The README said both things at
# once for a while -- the artifact table had it right and a sentence in the
# workaround section had it wrong -- because nothing here read the file. Its
# provenance is checkable from its own contents, so it is checked: a dump of
# the clean run trips nothing, and it begins near the end of a ~565 s run
# rather than at t=0.
watcher_dump = log_text("watcher.log.gz")
check("the watcher dump is archived", bool(watcher_dump), "watcher.log.gz")
if watcher_dump:
    dump_tripped = watcher_dump.count("tripped an assert")
    check(
        "the watcher dump is from the clean run, not the failing one",
        dump_tripped == 0,
        f"{dump_tripped} tripped asserts in watcher.log.gz",
    )
    first = re.search(r"At (\d+(?:\.\d+)?)s", watcher_dump)
    check("the watcher dump records where in the run it starts", first is not None, "")
    if first:
        check(
            "the watcher dump really is a tail, as the artifact table says",
            float(first.group(1)) > 300.0,
            f"starts at {first.group(1)}s",
        )
        check(
            "README quotes the timestamp the dump actually starts at",
            in_readme(f"{float(first.group(1)):.1f} s"),
            f"{float(first.group(1)):.1f} s",
        )
if watcher_run:
    tripped = watcher_run.count("tripped an assert")
    check("the watcher run tripped no asserts", tripped == 0, f"{tripped} tripped asserts")
    watcher_passed = passed_count(watcher_run)
    check("the watcher run has a pytest tally", watcher_passed is not None, "")
    if watcher_passed is not None:
        check(
            f"README quotes the watcher tally ({watcher_passed} passed)",
            in_readme(f"**{watcher_passed} passed, zero tripped asserts**"),
            f"{watcher_passed}",
        )
        check(
            "the watcher run covered the same tests as the plain run",
            watcher_passed == passed_count(log_text("pytest_stage04_regression.log.gz")) + two_layer,
            f"watcher {watcher_passed} vs stage04 + 2-layer",
        )
    check(
        "the contract records the watcher as clean",
        "clean" in CONTRACT["full_model_performance"]["watcher"],
        "",
    )

# --- the watcher workaround's cost --------------------------------------------
#
# The A/B claim is that dropping the pinned num_workers_per_link is not merely
# safe but cheaper. The "after" side is the artifact; the "before" side is the
# superseded measurement, quoted as a constant because its CSV was overwritten.
BEFORE_FORCE_ARGMAX_MS = 1.8592814992492397
BEFORE_TOKEN_OUT_MS = 22.678480541799217
check(
    "README quotes the post-workaround force-argmax figure from the artifact",
    *quotes(README, PERF["sampler_force_argmax_ms"], ".3f", name="force argmax after"),
)
check(
    "README quotes the post-workaround token-out figure from the artifact",
    *quotes(README, PERF["token_out_ms"], ".3f", name="token out after"),
)
check(
    "the stated sampler speed-up is the recomputed ratio",
    *ratio_is_quoted(
        README, BEFORE_FORCE_ARGMAX_MS, PERF["sampler_force_argmax_ms"], ".2f", name="workaround sampler speed-up"
    ),
)
check(
    "the stated token-out saving is the recomputed difference",
    *quotes(README, BEFORE_TOKEN_OUT_MS - PERF["token_out_ms"], ".3f", name="token-out saving"),
)
check(
    "the split path really did not move, as the README's control row claims",
    abs(PERF["sampler_split_ms"] - 6.155204998018841) < 0.01,
    f"{PERF['sampler_split_ms']}",
)
check(
    "both sampler strategies still return the same token after the workaround",
    PERF["sampler_split_token"] == PERF["sampler_force_argmax_token"] == 16,
    f"{PERF['sampler_split_token']} vs {PERF['sampler_force_argmax_token']}",
)

# --- this file's own accounting -----------------------------------------------
#
# The README describes this checker by a count, and that count is a published
# figure like any other. It was "76" against a checker reporting 78.

claimed = re.search(r"re-derives all (\d+) figures", README)
check("README states how many figures this checker re-derives", bool(claimed), "")
if claimed:
    # +1 for this check itself, which has already been counted by ``check``.
    check(
        f"README's figure count matches what this run reports ({checks + 1})",
        int(claimed.group(1)) == checks + 1,
        f"README says {claimed.group(1)}, this run reports {checks + 1}",
    )

print()
if failures:
    print(f"{len(failures)} of {checks} checks FAILED:")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)
print(f"all {checks} published figures match their artifacts")
