# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Every number in the stage-06 documents, re-derived from the artifact it cites.

Stage 04 left this pattern behind because prose/artifact drift has failed a
review in every stage of this project, and it left the rule with it: **a checker
that reads the document it is checking checks nothing.** Stage 05's continuation
carried the rule forward and its own review found the rule had been let slip --
15 of 78 assertions restated constants defined a few lines above them, and one
survived even that pass with a second clause the assertion above it had already
proven, i.e. it could not fail. A check that cannot fail is worse than no check,
because it reads like cover.

So the stage-06 rules are the stage-05 rules plus one:

1. every figure is parsed out of an artifact -- a CSV row, a JSON field, a probe
   log line -- and the document string is compared against the **computed**
   rounding of it;
2. every ratio quoted in prose is recomputed from its two artifact operands and
   the quoted string must equal the rounding of the result, not merely appear
   somewhere in the file;
3. **every assertion here was mutation-tested**: the document was edited to make
   it false and this file was confirmed to report a failure. Nothing below is a
   tautology over its own literals, and nothing below is an unanchored substring
   search for a string this file also defines.

Runs on the host in under a second; no device.

    python .../probes/check_published_figures.py
"""

from __future__ import annotations

import csv
import gzip
import json
import re
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parents[1]
PROBES = DOC / "probes"
LOGS = DOC / "logs"
MODEL_DIR = DOC.parents[1]
STAGE05 = MODEL_DIR / "doc" / "full_model"

README = (DOC / "README.md").read_text(encoding="utf-8")
WORK_LOG = (DOC / "work_log.md").read_text(encoding="utf-8")
DOCS = README + "\n" + WORK_LOG
CONTRACT = json.loads((MODEL_DIR / "doc" / "context_contract.json").read_text(encoding="utf-8"))


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


BASELINE = load(STAGE05 / "probes" / "perf_full_model.json")
PERF = {
    (prompt, leg): load(PROBES / f"perf_full_model_p{prompt}_{leg}.json")
    for prompt in (128, 1024, 4096)
    for leg in ("before", "after", "argmaxrows")
}
SHIPPED = PERF[(128, "argmaxrows")]
DECODE = load(PROBES / "profile_summary_decode.json")
PREFILL = load(PROBES / "profile_summary_prefill.json")
MOE = load(PROBES / "moe_skew_analysis_final.json")
AUDIT = load(PROBES / "runtime_fallback_audit.json")
FOOTPRINT = load(PROBES / "footprint_262144.json")
ARGMAX = load(PROBES / "argmax_outer_dim_probe_b.json")
ARGMAX_A = load(PROBES / "argmax_outer_dim_probe.json")
SDPA_DEPTH = load(PROBES / "sdpa_depth_probe.json")
SDPA_SWEEP = load(PROBES / "sdpa_sweep_confirm_bf16.json")
SDPA_PREFILL = load(PROBES / "sdpa_prefill_confirm.json")
SDPA_PCC = load(PROBES / "sdpa_hf_pcc_at_depth.json")

failures: list[str] = []
checks = 0
_seen_at_line: dict[int, int] = {}

#: The documents this file exists to check. A search over one of these is a
#: *whole-document* search and is what ``_lint_needle`` below polices; searches
#: over an artifact, a log or a source file are a different thing and are not
#: guarded, because there the string being looked for is the artifact's own.
_DOCUMENTS: dict[int, str] = {id(README): "README", id(WORK_LOG): "work_log", id(DOCS): "README+work_log"}

#: Whole-document searches this file made that the lint rejected. Reported and
#: made fatal at the end of the run, next to the assertion failures.
vacuous: list[str] = []

#: The reason string for ``appears(..., restated=)``. Rule 2 below fires on any
#: needle the document carries more than once, because a figure restated at
#: several sites is held up by whichever site is still correct while the others
#: rot -- corrupt one and the check stays green. Where the *authoritative* site
#: is anchored by an assertion of its own and the remaining sites are the same
#: distinctive figure restated (a decimal a table and a paragraph both quote),
#: that residual weakness is accepted deliberately and named here rather than
#: left implicit. It is NOT an escape hatch for rule 1: a needle short enough to
#: be produced by unrelated text has to be anchored, and cannot be opted out.
RESTATED = (
    "restated at more than one site; the needle is a distinctive figure that no "
    "unrelated text in these documents can produce"
)

_NUMERIC = re.compile(r"[\d.]+")


def _caller() -> int:
    """The line of the first frame outside this file's search helpers."""
    frame = sys._getframe(1)
    while frame.f_code.co_name in ("_caller", "_lint_needle", "appears", "quotes", "ratio_is_quoted"):
        frame = frame.f_back
    return frame.f_lineno


def _lint_needle(document: str, text: str, occurrences: int, restated: str | None) -> None:
    """Reject whole-document searches that cannot distinguish what they find.

    Three rounds of review each found the same defect and each fixed only the
    instances it happened to notice: an assertion that a figure "appears
    somewhere in the README" is satisfied by any text that happens to contain
    the same characters. ``DECODE["composite_gather_rows"] == 16`` was satisfied
    by the word ``bfloat16``; ``chi2_df == 5`` by a markdown list marker;
    ``num_tokens == 109`` by the README's own prose *about* the checker. Each
    one read like a check and was not one.

    It is a mechanical property, so it is a mechanical rule rather than a fourth
    round of reading:

    1. **a numeric needle under four characters** may not be searched over a
       whole document at all. There is no opt-out: short numbers collide with
       unrelated text and no promise about the current document keeps them from
       colliding tomorrow. Anchor it -- build the phrase that is *supposed* to
       carry the figure out of the parsed artifact value and look for that;
    2. **a needle the document carries more than once** must either be anchored
       the same way or be declared with ``restated=RESTATED``, which says the
       author looked and decided the recurrence is the same figure quoted twice.

    Anchoring is ``quotes(..., context="...{}...")`` or ``phrase()`` directly.
    """
    label = _DOCUMENTS.get(id(document))
    if label is None:
        return
    if len(text) < 4 and _NUMERIC.fullmatch(text):
        vacuous.append(
            f"L{_caller():04d}  {text!r} is a {len(text)}-character numeric needle searched over the whole "
            f"{label} ({occurrences} occurrence(s)). Anchor it: pass context= to quotes(), or use phrase()."
        )
    elif occurrences > 1 and restated is None:
        vacuous.append(
            f"L{_caller():04d}  {text!r} occurs {occurrences} times in {label}, so corrupting any one site "
            f"leaves this check green. Anchor the authoritative site, or pass restated=RESTATED."
        )


def check(name: str, ok: bool, detail: str = "") -> None:
    """Record one assertion, under a **stable identity** as well as a name.

    The name is for a human and is deliberately specific -- many of them embed
    the artifact value being checked, e.g. ``README quotes the degeneracy metric
    109``. That makes the name a poor key: the stage-06 review found that
    ``mutation_test_checker.py`` was crediting mutation coverage by name, so a
    mutation that changed the *artifact* changed the name too, the mutated run's
    ``FAIL`` line did not match any name from the clean run, and the failure was
    silently discarded. Every such assertion was scored as never-failed on the
    artifact side and the tester still reported full coverage.

    So each assertion also carries an id that cannot vary with any artifact
    value: the source line of the ``check()`` call, plus an ordinal for calls
    that come from the same line in a loop. The mutation tester keys on the id
    and prints the name.
    """
    global checks
    checks += 1
    line = sys._getframe(1).f_lineno
    ordinal = _seen_at_line.get(line, 0)
    _seen_at_line[line] = ordinal + 1
    identity = f"L{line:04d}.{ordinal}"
    if not ok:
        failures.append(f"{name}: {detail}")
    print(f"{'PASS' if ok else 'FAIL'}  [{identity}] {name}{('  -- ' + detail) if detail and not ok else ''}")


def occurrences(document: str, text: str) -> int:
    """How many times ``document`` carries ``text`` as a whole number.

    Without the boundary, "2.7" is satisfied by "12.75" and "40" by "3400" -- so
    a short figure could be "quoted" by a document that never mentions it. The
    boundary is "no digit either side, and no decimal point that would make it
    part of a longer number" -- a sentence-ending full stop is not a decimal
    point, so "94." still counts as 94.
    """
    return len(re.findall(rf"(?<!\d)(?<!\d\.){re.escape(text)}(?!\d)(?!\.\d)", document))


_FLAT: dict[int, str] = {}


def flat(document: str) -> str:
    """``document`` with its line wrapping flattened, so a phrase can span lines."""
    if id(document) not in _FLAT:
        _FLAT[id(document)] = " ".join(document.split())
    return _FLAT[id(document)]


def phrase(document: str, text: str) -> bool:
    """The anchored form: ``text`` must appear as a phrase, wrapping ignored.

    This is what replaces a whole-document ``appears()`` wherever the figure is
    short enough that unrelated text can produce it. The phrase is built from
    the *parsed artifact value* by the caller, so it still fails when the
    document drifts rather than when this file does; what it adds is that the
    document has to carry the figure **in the sentence that is supposed to
    carry it**, not merely somewhere in 55 KB.
    """
    return " ".join(text.split()) in flat(document)


def appears(document: str, text: str, *, restated: str | None = None) -> bool:
    """``text`` must appear in ``document`` as a whole number, not as a substring.

    Over one of the published documents this is a *whole-document* search and is
    linted -- see ``_lint_needle``. Over an artifact, a log or a source file it
    is unguarded, because there the haystack is the thing being measured.
    """
    found = occurrences(document, text)
    _lint_needle(document, text, found, restated)
    return found > 0


def quotes(
    document: str,
    value: float,
    spec: str,
    *,
    name: str,
    context: str | None = None,
    restated: str | None = None,
) -> tuple[bool, str]:
    """``value`` formatted by ``spec`` must appear in ``document``.

    The formatting happens here, from the artifact's number, so the assertion
    fails when the document drifts -- not when this file drifts.

    ``context`` is the anchored form: a phrase with one ``{}`` where the figure
    belongs, e.g. ``context="a spread of {} us"``. The phrase is filled in from
    the computed value and matched against the whitespace-flattened document, so
    the assertion is tied to the sentence that carries the figure. Every short
    figure must use it; ``restated`` declares a deliberate multi-site quote.
    """
    text = format(value, spec)
    if context is not None:
        wanted = context.format(text)
        return phrase(document, wanted), f"{name}: computed {text}; the document does not carry {wanted!r}"
    return appears(document, text, restated=restated), f"{name}: computed {text}, absent from the document"


def ratio_is_quoted(
    document: str,
    numerator: float,
    denominator: float,
    spec: str,
    *,
    name: str,
    context: str | None = None,
    restated: str | None = None,
):
    """A ratio quoted in prose must equal the rounding of its two operands."""
    value = numerator / denominator
    text = format(value, spec)
    if context is not None:
        wanted = context.format(text)
        return (
            phrase(document, wanted),
            f"{name}: {numerator} / {denominator} -> {text}; the document does not carry {wanted!r}",
        )
    return (
        appears(document, text, restated=restated),
        f"{name}: {numerator} / {denominator} = {value!r} -> {text}, absent",
    )


NUMBER_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    40: "forty",
    56: "fifty-six",
}


def spelled(document: str, value: int, *, name: str) -> tuple[bool, str]:
    """A small count the documents write out in words must match the computed one.

    A bare integer is a weak assertion -- "13" is satisfied by "13-wide grid"
    somewhere else in the file. Where the document spells the count, the word is
    what is checked, and the word for the *computed* count is what is looked for.
    """
    word = NUMBER_WORDS.get(value)
    if word is None:
        return False, f"{name}: no word form for {value}"
    return (
        re.search(rf"\b{word}\b", document, re.I) is not None,
        f"{name}: computed {value} -> '{word}', absent from the document",
    )


def log_text(path: Path) -> str:
    if not path.is_file():
        return ""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="ignore").read()
    return path.read_text(encoding="utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# 1. the headline: baseline against shipped
# ---------------------------------------------------------------------------

for label, value, spec in (
    ("stage-05 TTFT", BASELINE["ttft_ms"], ".2f"),
    ("stage-05 token-out", BASELINE["token_out_ms"], ".3f"),
    ("stage-05 token-out t/s/u", BASELINE["token_out_tps_user"], ".2f"),
    ("stage-05 logits-only", BASELINE["model_trace_ms"], ".3f"),
    ("stage-05 logits-only t/s/u", BASELINE["model_trace_tps_user"], ".2f"),
    ("stage-05 readback", BASELINE["token_out_readback_ms"], ".3f"),
    ("stage-05 readback t/s/u", BASELINE["token_out_readback_tps_user"], ".2f"),
    ("stage-05 greedy sampler", BASELINE["sampler_force_argmax_ms"], ".3f"),
    ("stage-05 split sampler", BASELINE["sampler_split_ms"], ".3f"),
    ("stage-05 cold TTFT", BASELINE["ttft_cold_ms"], ".2f"),
):
    check(
        f"README quotes the {label} from the stage-05 artifact",
        *quotes(README, value, spec, name=label, restated=RESTATED),
    )

for label, value, spec in (
    ("shipped TTFT", SHIPPED["ttft_ms"], ".2f"),
    ("shipped token-out", SHIPPED["token_out_ms"], ".3f"),
    ("shipped token-out t/s/u", SHIPPED["token_out_tps_user"], ".2f"),
    ("shipped logits-only", SHIPPED["model_trace_ms"], ".3f"),
    ("shipped logits-only t/s/u", SHIPPED["model_trace_tps_user"], ".2f"),
    ("shipped readback", SHIPPED["token_out_readback_ms"], ".3f"),
    ("shipped readback t/s/u", SHIPPED["token_out_readback_tps_user"], ".2f"),
    ("shipped greedy sampler", SHIPPED["sampler_force_argmax_ms"], ".3f"),
    ("shipped cold TTFT", SHIPPED["ttft_cold_ms"], ".2f"),
):
    check(
        f"README quotes the {label} from the shipped artifact",
        *quotes(README, value, spec, name=label, restated=RESTATED),
    )

check(
    "the headline 1.12x token-out gain is the ratio of the two artifacts",
    *ratio_is_quoted(
        README, BASELINE["token_out_ms"], SHIPPED["token_out_ms"], ".2f", name="token-out gain", restated=RESTATED
    ),
)
check(
    "the logits-only gain is the ratio of the two artifacts",
    *ratio_is_quoted(README, BASELINE["model_trace_ms"], SHIPPED["model_trace_ms"], ".2f", name="logits-only gain"),
)
check(
    "the readback gain is the ratio of the two artifacts",
    *ratio_is_quoted(
        README, BASELINE["token_out_readback_ms"], SHIPPED["token_out_readback_ms"], ".2f", name="readback gain"
    ),
)
check(
    "the greedy-sampler gain is the ratio of the two artifacts",
    *ratio_is_quoted(
        README,
        BASELINE["sampler_force_argmax_ms"],
        SHIPPED["sampler_force_argmax_ms"],
        ".2f",
        name="greedy sampler gain",
    ),
)
check(
    "the TTFT saving is the recomputed difference",
    *quotes(README, BASELINE["ttft_ms"] - SHIPPED["ttft_ms"], ".2f", name="TTFT delta"),
)
check(
    "the split sampler really is the control the README calls it",
    abs(BASELINE["sampler_split_ms"] - SHIPPED["sampler_split_ms"]) < 0.005,
    f"{BASELINE['sampler_split_ms']} vs {SHIPPED['sampler_split_ms']}",
)
check(
    "both sampler strategies still return the same token",
    SHIPPED["sampler_split_token"] == SHIPPED["sampler_force_argmax_token"],
    f"{SHIPPED['sampler_split_token']} vs {SHIPPED['sampler_force_argmax_token']}",
)
check(
    "the shipped run really is faster than the baseline on token-out",
    SHIPPED["token_out_ms"] < BASELINE["token_out_ms"],
    f"{SHIPPED['token_out_ms']} vs {BASELINE['token_out_ms']}",
)

# ---------------------------------------------------------------------------
# 2. the context sweep, and the flatness claim
# ---------------------------------------------------------------------------

for prompt in (128, 1024, 4096):
    for leg in ("before", "after", "argmaxrows"):
        row = PERF[(prompt, leg)]
        check(
            f"README quotes token-out at prompt {prompt}, leg {leg}",
            *quotes(README, row["token_out_ms"], ".4f", name=f"p{prompt} {leg}", restated=RESTATED),
        )
    check(
        f"README quotes the shipped t/s/u at prompt {prompt}",
        *quotes(
            README,
            PERF[(prompt, "argmaxrows")]["token_out_tps_user"],
            ".2f",
            name=f"p{prompt} t/s/u",
            restated=RESTATED,
        ),
    )
    check(
        f"the three legs at prompt {prompt} are monotonically faster",
        PERF[(prompt, "argmaxrows")]["token_out_ms"]
        < PERF[(prompt, "after")]["token_out_ms"]
        < PERF[(prompt, "before")]["token_out_ms"],
        f"{[PERF[(prompt, l)]['token_out_ms'] for l in ('before', 'after', 'argmaxrows')]}",
    )
    check(
        f"README quotes TTFT at prompt {prompt} for both the before and shipped legs",
        format(PERF[(prompt, "before")]["ttft_ms"], ".2f") in README
        and format(PERF[(prompt, "argmaxrows")]["ttft_ms"], ".2f") in README,
        f"{PERF[(prompt, 'before')]['ttft_ms']:.2f} / {PERF[(prompt, 'argmaxrows')]['ttft_ms']:.2f}",
    )

check(
    "the 'before' context blow-up is the recomputed 4096/128 ratio",
    *ratio_is_quoted(
        README,
        PERF[(4096, "before")]["token_out_ms"],
        PERF[(128, "before")]["token_out_ms"],
        ".2f",
        name="before 4096/128",
        restated=RESTATED,
    ),
)
check(
    "the shipped context flatness is the recomputed 4096/128 ratio",
    *ratio_is_quoted(
        README,
        PERF[(4096, "argmaxrows")]["token_out_ms"],
        PERF[(128, "argmaxrows")]["token_out_ms"],
        ".2f",
        name="shipped 4096/128",
        restated=RESTATED,
    ),
)
check(
    "decode really is nearly flat in context on the shipped tree, as claimed",
    PERF[(4096, "argmaxrows")]["token_out_ms"] / PERF[(128, "argmaxrows")]["token_out_ms"] < 1.10,
    f"{PERF[(4096, 'argmaxrows')]['token_out_ms'] / PERF[(128, 'argmaxrows')]['token_out_ms']}",
)
check(
    "TTFT is unchanged by the decode-only levers, as claimed",
    all(
        abs(PERF[(p, "before")]["ttft_ms"] - PERF[(p, "argmaxrows")]["ttft_ms"]) / PERF[(p, "before")]["ttft_ms"] < 0.01
        for p in (128, 1024, 4096)
    ),
    "",
)

# The allocated-cache-depth caveat rests on a probe, not on an argument.
# One leg per distinct allocated depth, at the first row the probe recorded
# for it -- the probe re-runs 65536 as its own control and the second reading
# is not a separate depth.
depth_first: dict[int, float] = {}
for row in SDPA_DEPTH:
    if row["cur_pos"] == 128:
        depth_first.setdefault(row["allocated_context"], row["ms"] * 1000.0)
depth_at_128 = list(depth_first.values())
check("the depth probe swept at least four allocated depths", len(depth_at_128) >= 4, str(sorted(depth_first)))
for context, value in depth_first.items():
    check(
        f"README quotes the depth-probe leg at allocated context {context}",
        *quotes(README, value, ".2f", name=f"depth {context}"),
    )
check(
    "SDPA-decode cost really is independent of allocated depth, as the caveat claims",
    (max(depth_at_128) - min(depth_at_128)) / min(depth_at_128) < 0.15,
    f"{min(depth_at_128)} .. {max(depth_at_128)} us",
)
check(
    "the baseline run and the shipped run really do differ in allocated context",
    BASELINE["context"] != SHIPPED["context"],
    f"{BASELINE['context']} vs {SHIPPED['context']}",
)
check(
    "README states both allocated contexts it is comparing across",
    str(BASELINE["context"]) in README and str(SHIPPED["context"]) in README,
    f"{BASELINE['context']} / {SHIPPED['context']}",
)

# ---------------------------------------------------------------------------
# 3. the decode profile and its verified window
# ---------------------------------------------------------------------------

check(
    "README quotes the window row count",
    *quotes(README, DECODE["window_rows"], "d", name="window rows", restated=RESTATED),
)
per_device_ops = set(DECODE["ops_per_device"].values())
check("every device holds the same op count", len(per_device_ops) == 1, str(DECODE["ops_per_device"]))
check(
    "README quotes the per-device op count",
    *quotes(DOCS, per_device_ops.pop(), "d", name="ops per device", restated=RESTATED),
)
check("the window covers four devices", DECODE["devices"] == 4, str(DECODE["devices"]))

for device, value in DECODE["device_kernel_us"].items():
    check(
        f"README quotes device {device}'s kernel total",
        *quotes(README, value, ".1f", name=f"device {device}", restated=RESTATED),
    )
check(
    "README quotes the device spread in us",
    # work_log restates it as "2.7 us of spread"; the README sentence that makes
    # the window argument is the authoritative one and is the one anchored.
    *quotes(README, DECODE["device_spread_us"], ".1f", name="spread us", context="a spread of {} us"),
)
check(
    "README quotes the device spread as a percentage",
    *quotes(README, DECODE["device_spread_percent"], ".3f", name="spread %"),
)
check(
    "the device spread really is negligible, as the window argument requires",
    DECODE["device_spread_percent"] < 0.1,
    f"{DECODE['device_spread_percent']}",
)

for region, value in DECODE["regions_us"].items():
    check(f"README quotes the {region} region total", *quotes(README, value, ".1f", name=region, restated=RESTATED))
    check(
        f"README quotes the {region} region share",
        *quotes(README, DECODE["regions_percent"][region], ".2f", name=f"{region} %", restated=RESTATED),
    )
check(
    "the three regions sum to the iteration total, so nothing is unaccounted for",
    abs(sum(DECODE["regions_us"].values()) - DECODE["iteration_us"]) < 1e-6,
    f"{sum(DECODE['regions_us'].values())} vs {DECODE['iteration_us']}",
)
check(
    "README quotes the in-model per-layer cost",
    *quotes(README, DECODE["per_layer_us"], ".3f", name="per layer", restated=RESTATED),
)
for device, value in DECODE["per_layer_us_all_devices"].items():
    check(
        f"README quotes the per-layer cost on device {device}",
        *quotes(README, value, ".3f", name=f"per layer d{device}", restated=RESTATED),
    )
check(
    "README quotes the LM head kernel time",
    *quotes(README, DECODE["lm_head_us"], ".3f", name="lm head", restated=RESTATED),
)
check(
    "the LM head figure the tables quote is the reported device's row in the per-device split",
    DECODE["lm_head_us"] == DECODE["lm_head_us_all_devices"]["0"],
    f"{DECODE['lm_head_us']} vs {DECODE['lm_head_us_all_devices']}",
)
check(
    "the per-device LM head split covers every device in the window",
    sorted(DECODE["lm_head_us_all_devices"]) == sorted(DECODE["ops_per_device"]),
    str(sorted(DECODE["lm_head_us_all_devices"])),
)
check("README quotes the sampler kernel time", *quotes(README, DECODE["sampler_us"], ".3f", name="sampler"))
check(
    "README quotes the composite 4-wide gather total",
    *quotes(README, DECODE["composite_gather_us"], ".2f", name="composite gather", restated=RESTATED),
)
check(
    "README quotes the number of profiler rows the two composite gathers span",
    # ``appears(README, "16")`` was satisfied by the word ``bfloat16`` -- 17
    # occurrences in this document, none of them this figure. The terminal-block
    # table row is where the row count is stated as a count, so that is the
    # sentence it is anchored to; the prose "at 41.06 us over 16 rows" one
    # section down restates it and is left to the 41.06 assertion above.
    *quotes(
        README,
        DECODE["composite_gather_rows"],
        "d",
        name="composite gather rows",
        context="summed across their {} rows",
    ),
)
check(
    "the composite gather's row count is the two gathers' structural decomposition",
    DECODE["composite_gather_rows"] == 2 * (1 + 4 + 1 + 1 + 1),
    f"{DECODE['composite_gather_rows']} rows: two runs of AllBroadcast + 4 x UntilizeWithUnpadding "
    "+ Concat + Permute + TilizeWithValPadding is 16",
)
check(
    "README quotes the composite gather's share of the iteration",
    *quotes(
        README,
        100.0 * DECODE["composite_gather_us"] / DECODE["iteration_us"],
        ".2f",
        name="composite %",
        restated=RESTATED,
    ),
)
check(
    "README quotes the composite gather's share of the sampler",
    *quotes(
        README,
        100.0 * DECODE["composite_gather_us"] / DECODE["sampler_us"],
        ".0f",
        name="composite/sampler %",
        context="and {}% of the sampler",
    ),
)

# The per-layer ranking table the README publishes, top rows, from the JSON.
rank_by_op = {row["op"]: row for row in DECODE["per_layer_ranking"]}
# Rows the README prints on their own line, with their own share.
for op in (
    "SparseMatmul 1x1x32x2048 @ 1x32x2048x1536",
    "ReduceScatterMinimalAsync 1x1x32x2048",
    "SparseMatmul 1x32x32x768 @ 1x32x768x2048",
    "TopK 1x1x32x128",
    "AllGatherAsync 1x1x32x512",
    "Unary 1x32x32x1536",
    "LayerNorm 1x1x32x2048",
    "SdpaDecode 1x1x32x128",
):
    check(f"the ranking still contains {op}", op in rank_by_op, str(op))
    if op in rank_by_op:
        check(
            f"README quotes the us/layer for {op}",
            *quotes(README, rank_by_op[op]["us_per_layer"], ".3f", name=op, restated=RESTATED),
        )
        check(
            f"README quotes the %iter for {op}",
            *quotes(README, rank_by_op[op]["percent"], ".2f", name=f"{op} %"),
        )
# Rows the README groups onto one line: the three expert-tail reshapes and the
# three projection matmuls. Each member's us/layer is quoted individually and
# the share is quoted once, for the group -- so that is what is checked.
for group, members in (
    (
        "the three expert-tail ReshapeViews",
        ("ReshapeView 1x32x32x2048", "ReshapeView 1x32x32x1536", "ReshapeView 1x1x32x768"),
    ),
    (
        "the QKV / o_proj / router matmuls",
        (
            "Matmul 1x1x32x2048 @ 1x1x2048x1280",
            "Matmul 1x1x32x1024 @ 1x1x1024x2048",
            "Matmul 1x1x32x2048 @ 1x1x2048x128",
        ),
    ),
):
    for op in members:
        check(f"the ranking still contains {op}", op in rank_by_op, str(op))
    if all(op in rank_by_op for op in members):
        for op in members:
            check(
                f"README quotes the us/layer for {op}",
                *quotes(README, rank_by_op[op]["us_per_layer"], ".3f", name=op),
            )
        check(
            f"README quotes the combined share of {group}",
            *quotes(README, sum(rank_by_op[op]["percent"] for op in members), ".2f", name=group),
        )

if {"SparseMatmul 1x1x32x2048 @ 1x32x2048x1536", "SdpaDecode 1x1x32x128"} <= set(rank_by_op):
    check(
        "the expert SparseMatmul pair is still the largest per-layer item, as the README says",
        rank_by_op["SparseMatmul 1x1x32x2048 @ 1x32x2048x1536"]["us_per_layer"]
        > rank_by_op["SdpaDecode 1x1x32x128"]["us_per_layer"],
        "",
    )
ranking_ops = [row["op"] for row in DECODE["per_layer_ranking"]]
check(
    "SdpaDecode is no longer in the top five per-layer ops, as the rank-change table claims",
    "SdpaDecode 1x1x32x128" in ranking_ops and ranking_ops.index("SdpaDecode 1x1x32x128") >= 5,
    str(ranking_ops[:6]),
)
check(
    "the LM head is the largest item in terminal-post, as the README says",
    max(DECODE["terminal_post_ranking"], key=lambda row: row["us"])["op"].startswith(
        "Matmul 1x1x32x2048 @ 1x1x2048x37984"
    ),
    max(DECODE["terminal_post_ranking"], key=lambda row: row["us"])["op"],
)
check(
    "README quotes the LM head's share of terminal-post",
    *quotes(README, 100.0 * DECODE["lm_head_us"] / DECODE["regions_us"]["terminal_post"], ".1f", name="lm/post %"),
)

post_by_op = {row["op"]: row for row in DECODE["terminal_post_ranking"]}
for op in ("Untilize 1x1x32x37984", "ArgMax 1x1x1x37984", "LayerNorm 1x1x32x2048"):
    check(f"terminal-post still contains {op}", op in post_by_op, "")
    if op in post_by_op:
        check(
            f"README quotes the {op} figure", *quotes(README, post_by_op[op]["us"], ".3f", name=op, restated=RESTATED)
        )
ARGMAX_KEY = "ArgMax 1x1x1x37984"

# -- the window's own verification, read out of the windower's log ------------

window_log = log_text(LOGS / "window_full_model_48_final.log")
check("the decode window's boundary check is archived", bool(window_log), "logs/window_full_model_48_final.log")
tallies = re.findall(r"boundary check\s+device (\d+)\s+(\S+)\s+(\d+) / (\d+)\s+(\S+)", window_log)
check("the archived boundary check ran tallies", bool(tallies), str(len(tallies)))
check(
    "every archived decode tally is exact",
    all(got == want and status == "ok" for _, _, got, want, status in tallies),
    str([t for t in tallies if t[4] != "ok"]),
)
check(
    "the decode boundary was checked on all four devices",
    len({device for device, _, _, _, _ in tallies}) == 4,
    str(sorted({d for d, _, _, _, _ in tallies})),
)
check(
    "README states how many decode tallies were checked in total, in the sentence that states it",
    # ``appears(README, "40")`` was satisfied by the *other* occurrence of 40 in
    # this document: "it is 40 assertions short of the coverage the previous
    # tree claimed", a sentence about this checker's own history. That collider
    # moves whenever an assertion is added, so the tally check was riding on a
    # self-referential figure. Anchored to the evidence table's row instead.
    *quotes(README, len(tallies), "d", name="decode tally count", context="the boundary checks, {} and "),
)
decode_ops_checked = {op for _, op, _, _, _ in tallies}
check(
    "README spells how many distinct ops the decode boundary check tallies",
    *spelled(README, len(decode_ops_checked), name="decode distinct tallies"),
)
for op, want in (
    ("ReduceScatterMinimalAsyncDeviceOperation", 96),
    ("AllGatherAsyncDeviceOperation", 96),
    ("SdpaDecodeDeviceOperation", 48),
    ("SparseMatmulDeviceOperation", 96),
    ("AllBroadcastDeviceOperation", 2),
    ("ArgMaxDeviceOperation", 1),
):
    got = {int(g) for _, o, g, _, _ in tallies if o == op}
    check(
        f"the archived decode window holds {want} {op} per device",
        got == {want},
        f"{op}: {sorted(got)}",
    )
check(
    "the composite path really did displace AllGatherAsync for the sampler's gathers",
    {int(g) for _, o, g, _, _ in tallies if o == "AllGatherAsyncDeviceOperation"} == {2 * DECODE["layers"]},
    "an extra all-gather would mean the old full-vocabulary gather is still there",
)

# -- the two cross-checks the README rests the window on ----------------------

gap_ms = SHIPPED["token_out_ms"] - DECODE["iteration_us"] / 1000.0
check("README quotes the dispatch/gap residual in ms", *quotes(README, gap_ms, ".3f", name="gap ms", restated=RESTATED))
check(
    "README quotes the dispatch/gap residual as a percentage of token-out",
    *quotes(README, 100.0 * gap_ms / SHIPPED["token_out_ms"], ".2f", name="gap %", restated=RESTATED),
)
check(
    "README quotes the per-dispatch gap",
    *quotes(README, gap_ms * 1000.0 / DECODE["ops_per_device"]["0"], ".2f", name="us per dispatch", restated=RESTATED),
)
check(
    "the window really is inside token-out, as one iteration must be",
    0 < gap_ms < 2.0,
    f"{gap_ms} ms",
)

sampler_wall_us = (SHIPPED["token_out_ms"] - SHIPPED["model_trace_ms"]) * 1000.0
check(
    "the documents quote the wall-clock cost of the sampling trace",
    *quotes(DOCS, sampler_wall_us, ".2f", name="token_out - model_trace"),
)
check(
    "the sampling trace's wall cost matches its profiled kernel time",
    abs(sampler_wall_us - DECODE["sampler_us"]) / DECODE["sampler_us"] < 0.02,
    f"{sampler_wall_us:.3f} us wall vs {DECODE['sampler_us']:.3f} us kernel",
)
check(
    "the documents quote the agreement between those two independent measurements",
    *quotes(
        DOCS,
        100.0 * abs(sampler_wall_us - DECODE["sampler_us"]) / DECODE["sampler_us"],
        ".2f",
        name="sampler agreement %",
    ),
)

# ---------------------------------------------------------------------------
# 4. the superseded pre-adoption profile, and the deltas against it
# ---------------------------------------------------------------------------

PART1 = DOC / "rank_full_model_48layer_decode_part1_preadoption.txt"
check("the superseded pre-adoption ranking is archived", PART1.is_file(), str(PART1))
part1 = log_text(PART1)
match = re.search(
    r"device 0: total\s+([\d,]+\.\d+) us = pre\s+([\d,.]+) \+ 48 layers\s+([\d,]+\.\d+) "
    r"\+ terminal\s+([\d,.]+)\s+\(per layer\s+([\d.]+) us\)",
    part1,
)
check("the superseded ranking's device-0 line is parseable", match is not None, "")
if match:
    part1_total = float(match.group(1).replace(",", ""))
    part1_post = float(match.group(4).replace(",", ""))
    part1_per_layer = float(match.group(5))
    for label, value, spec in (
        ("pre-adoption iteration total", part1_total, ".1f"),
        ("pre-adoption terminal-post", part1_post, ".1f"),
        ("pre-adoption per-layer", part1_per_layer, ".3f"),
    ):
        check(
            f"README quotes the {label} from the superseded artifact",
            *quotes(DOCS, value, spec, name=label, restated=RESTATED),
        )
    check(
        "README quotes the per-layer saving as the recomputed difference",
        *quotes(DOCS, part1_per_layer - DECODE["per_layer_us"], ".3f", name="per-layer delta", restated=RESTATED),
    )
    check(
        "README quotes the iteration saving as the recomputed difference",
        *quotes(DOCS, part1_total - DECODE["iteration_us"], ".1f", name="iteration delta"),
    )
    check(
        "README quotes the terminal-post saving as the recomputed difference",
        *quotes(
            DOCS, part1_post - DECODE["regions_us"]["terminal_post"], ".1f", name="terminal delta", restated=RESTATED
        ),
    )
    check(
        "the shipped profile really is faster than the superseded one on every headline",
        DECODE["iteration_us"] < part1_total
        and DECODE["per_layer_us"] < part1_per_layer
        and DECODE["regions_us"]["terminal_post"] < part1_post,
        "",
    )

sdpa_part1 = re.search(r"\s([\d.]+)\s+[\d.]+%\s+48\s+([\d.]+)\s+([\d.]+)\s+110\s+SdpaDecode 1x1x32x128", part1)
check("the superseded SdpaDecode row is parseable", sdpa_part1 is not None, "")
if sdpa_part1:
    part1_sdpa = float(sdpa_part1.group(3))
    check(
        "README quotes the pre-adoption SdpaDecode us/layer",
        *quotes(DOCS, part1_sdpa, ".3f", name="pre-adoption SdpaDecode", restated=RESTATED),
    )
    sdpa_delta = part1_sdpa - rank_by_op["SdpaDecode 1x1x32x128"]["us_per_layer"]
    check(
        "README quotes the SdpaDecode saving",
        *quotes(DOCS, sdpa_delta, ".3f", name="SdpaDecode delta", restated=RESTATED),
    )
    if match:
        residual = (part1_per_layer - DECODE["per_layer_us"]) - sdpa_delta
        check(
            "README quotes the residual once SdpaDecode is accounted for",
            *quotes(DOCS, residual, ".3f", name="per-layer residual", restated=RESTATED),
        )
        check(
            "README's '97% of it' claim is what the two deltas actually divide to",
            *quotes(
                DOCS,
                100.0 * sdpa_delta / (part1_per_layer - DECODE["per_layer_us"]),
                ".0f",
                name="97%",
                context="**{}% of it**",
            ),
        )

# The lever analysis's B2 table quotes the attention-side reduce-scatter's
# spread off the pre-adoption window. Nothing checked it and the stage-06 review
# found the max transcribed wrong (15.01 for 15.276), so it is re-derived from
# the CSV that table was written from.
PART1_CSV = DOC / "ops_perf_full_model_48layer_decode_part1_preadoption.csv.gz"
part1_work_log = log_text(DOC / "profile_48layer_work_log.md")
check("the pre-adoption window the lever analysis quotes is on disk", PART1_CSV.is_file(), str(PART1_CSV))
if PART1_CSV.is_file():
    with gzip.open(PART1_CSV, "rt") as handle:
        part1_rows = list(csv.DictReader(handle))
    # Two reduce-scatters per layer per device, in order: the attention-side one
    # then the MoE-side one.
    per_device: dict[str, list[float]] = {}
    for row in part1_rows:
        if row["OP CODE"] == "ReduceScatterMinimalAsyncDeviceOperation":
            per_device.setdefault(row["DEVICE ID"], []).append(int(row["DEVICE KERNEL DURATION [ns]"]) / 1000.0)
    attention_rs = [us for values in per_device.values() for us in values[0::2]]
    check(
        "the pre-adoption window carries two reduce-scatters per layer on each of four devices",
        len(per_device) == 4 and all(len(v) == 2 * 48 for v in per_device.values()),
        str({d: len(v) for d, v in sorted(per_device.items())}),
    )
    for label, value, spec in (
        ("mean", sum(attention_rs) / len(attention_rs), ".2f"),
        ("min", min(attention_rs), ".2f"),
        ("max", max(attention_rs), ".3f"),
    ):
        check(
            f"the lever analysis quotes the attention reduce-scatter's {label}",
            *quotes(part1_work_log, value, spec, name=f"attention RS {label}"),
        )

argmax_part1 = re.search(r"\s([\d.]+)\s+[\d.]+%\s+1\s+([\d.]+)\s+110\s+ArgMax 1x1x32x37984", part1)
check("the superseded ArgMax row is parseable", argmax_part1 is not None, "")
if argmax_part1:
    check(
        "README quotes the pre-adoption ArgMax cost",
        *quotes(DOCS, float(argmax_part1.group(2)), ".3f", name="pre-adoption ArgMax", restated=RESTATED),
    )
    check(
        "the shipped ArgMax really is the ~97% reduction the README claims",
        ARGMAX_KEY in post_by_op and post_by_op[ARGMAX_KEY]["us"] < 0.05 * float(argmax_part1.group(2)),
        f"{post_by_op.get(ARGMAX_KEY, {}).get('us')} vs {argmax_part1.group(2)}",
    )

# The profile's saving from the sampler lever must match the end-to-end one.
if match:
    profiled_terminal_saving = part1_post - DECODE["regions_us"]["terminal_post"]
    measured_saving_us = (PERF[(128, "after")]["token_out_ms"] - PERF[(128, "argmaxrows")]["token_out_ms"]) * 1000.0
    check(
        "README quotes the end-to-end saving from the live-row lever",
        *quotes(DOCS, measured_saving_us, ".1f", name="argmax-rows token-out saving", restated=RESTATED),
    )
    check(
        "the profiled and end-to-end savings from the sampler lever agree",
        abs(profiled_terminal_saving - measured_saving_us) / measured_saving_us < 0.05,
        f"profiled {profiled_terminal_saving:.1f} us vs measured {measured_saving_us:.1f} us",
    )

# The SDPA lever's profiled saving must be *smaller* than its end-to-end one --
# the README makes that claim explicitly and calls it conservative.
if sdpa_part1:
    profiled_sdpa_saving_ms = DECODE["layers"] * sdpa_delta / 1000.0
    measured_sdpa_saving_ms = PERF[(128, "before")]["token_out_ms"] - PERF[(128, "after")]["token_out_ms"]
    check(
        "README quotes the profiled SDPA saving over 48 layers",
        *quotes(DOCS, profiled_sdpa_saving_ms, ".3f", name="profiled SDPA saving", restated=RESTATED),
    )
    check(
        "README quotes the end-to-end SDPA saving",
        *quotes(DOCS, measured_sdpa_saving_ms, ".3f", name="measured SDPA saving", restated=RESTATED),
    )
    check(
        "the profile really does understate the SDPA lever, as the README says",
        profiled_sdpa_saving_ms < measured_sdpa_saving_ms,
        f"profiled {profiled_sdpa_saving_ms:.3f} ms vs measured {measured_sdpa_saving_ms:.3f} ms",
    )

# and the reason it understates it is in the sweep: the default is linear in
# cur_pos and the configured leg is not.
sweep = {(row["cur_pos"], row["cfg"]): row["us"] for row in SDPA_SWEEP}
wanted_sweep = [(pos, cfg) for pos in (127, 255) for cfg in ("None", "k256/c16")]
check(
    "the sweep still carries both legs at both positions the README quotes",
    all(key in sweep for key in wanted_sweep),
    str([key for key in wanted_sweep if key not in sweep]),
)
if all(key in sweep for key in wanted_sweep):
    for pos, cfg in wanted_sweep:
        check(
            f"README quotes the sweep leg {cfg} at cur_pos {pos}",
            *quotes(README, sweep[(pos, cfg)], ".2f", name=f"{cfg}@{pos}"),
        )
    check(
        "the default leg really grows with cur_pos while the configured one does not",
        sweep[(255, "None")] > sweep[(127, "None")] and sweep[(255, "k256/c16")] <= sweep[(127, "k256/c16")],
        f"None {sweep[(127, 'None')]:.2f}->{sweep[(255, 'None')]:.2f}, "
        f"k256/c16 {sweep[(127, 'k256/c16')]:.2f}->{sweep[(255, 'k256/c16')]:.2f}",
    )

# ---------------------------------------------------------------------------
# 5. prefill
# ---------------------------------------------------------------------------

check("README quotes the prefill window row count", *quotes(README, PREFILL["window_rows"], "d", name="prefill rows"))
prefill_ops = set(PREFILL["ops_per_device"].values())
check("every device holds the same prefill op count", len(prefill_ops) == 1, str(PREFILL["ops_per_device"]))
check(
    "README quotes the prefill ops per device",
    *quotes(DOCS, prefill_ops.pop(), "d", name="prefill ops/device", restated=RESTATED),
)
for device, value in PREFILL["device_kernel_us"].items():
    check(
        f"README quotes prefill device {device}'s kernel total",
        *quotes(README, value, ".1f", name=f"prefill device {device}"),
    )
check(
    "README quotes the prefill device spread as a percentage",
    *quotes(README, PREFILL["device_spread_percent"], ".3f", name="prefill spread %"),
)
check(
    "README quotes the share of TTFT that is device kernel time",
    *quotes(
        README,
        100.0 * PREFILL["iteration_us"] / 1000.0 / SHIPPED["ttft_ms"],
        ".1f",
        name="prefill kernel / TTFT",
    ),
)
check(
    "the prefill window really fits inside the measured TTFT",
    PREFILL["iteration_us"] / 1000.0 < SHIPPED["ttft_ms"],
    f"{PREFILL['iteration_us'] / 1000.0} ms vs {SHIPPED['ttft_ms']} ms",
)

prefill_rank = {row["op"]: row for row in PREFILL["ranking"]}
for op in (
    "SparseMatmul 1x1x32x2048 @ 1x32x2048x1536",
    "SparseMatmul 1x32x32x768 @ 1x32x768x2048",
    "Unary 128x1x32x128",
    "BinaryNg 1x32x32x2048",
    "TopK 1x1x128x128",
    "SDPA 1x8x128x128",
):
    check(f"the prefill ranking still contains {op}", op in prefill_rank, "")
    if op in prefill_rank:
        row = prefill_rank[op]
        check(f"README quotes the prefill total for {op}", *quotes(README, row["us"], ".1f", name=op))
        check(
            f"README quotes the prefill us/layer for {op}",
            *quotes(README, row["us_per_layer"], ".2f", name=op, restated=RESTATED),
        )
        check(
            f"README quotes the prefill share for {op}",
            *quotes(README, row["percent"], ".2f", name=f"{op} %", restated=RESTATED),
        )

sparse_total = sum(row["us"] for row in PREFILL["ranking"] if row["op"].startswith("SparseMatmul"))
check(
    "README quotes the expert matmuls' share of prefill",
    *quotes(README, 100.0 * sparse_total / PREFILL["iteration_us"], ".2f", name="prefill sparse %"),
)
check(
    "the expert matmuls really are the majority of prefill, as the README says",
    sparse_total > 0.5 * PREFILL["iteration_us"],
    f"{100.0 * sparse_total / PREFILL['iteration_us']:.2f}%",
)
ccl_total = sum(row["us"] for row in PREFILL["ranking"] if "ReduceScatter" in row["op"] or "AllGather" in row["op"])
check("README quotes the prefill collective total", *quotes(README, ccl_total, ".1f", name="prefill ccl"))
check(
    "README quotes the collectives' share of prefill",
    *quotes(README, 100.0 * ccl_total / PREFILL["iteration_us"], ".2f", name="prefill ccl %", restated=RESTATED),
)
lm_key, sdpa_key = "Matmul 1x1x32x2048 @ 1x1x2048x37984", "SDPA 1x8x128x128"
check("the prefill ranking still contains the LM head", lm_key in prefill_rank, "")
if lm_key in prefill_rank:
    check(
        "README quotes the prefill LM head total",
        *quotes(README, prefill_rank[lm_key]["us"], ".1f", name="prefill lm head"),
    )
    check(
        "README quotes the prefill LM head's share",
        *quotes(README, prefill_rank[lm_key]["percent"], ".2f", name="prefill lm %"),
    )
if sdpa_key in prefill_rank:
    check(
        "prefill SDPA really is under 1% of prefill, which is why the prefill lever was declined",
        prefill_rank[sdpa_key]["percent"] < 1.0,
        f"{prefill_rank[sdpa_key]['percent']}%",
    )

prefill_log = log_text(LOGS / "window_full_model_48_prefill.log")
check("the prefill window's boundary check is archived", bool(prefill_log), "logs/window_full_model_48_prefill.log")
repeats = re.findall(r"repeat check\s+device (\d+)\s+(\d+) ops, identical to the preceding pass\s+ok", prefill_log)
check("the prefill repeat check ran on all four devices", len(repeats) == 4, str(repeats))
check(
    "the repeat check compared the same op count the summary reports",
    all(int(count) == PREFILL["ops_per_device"]["0"] for _, count in repeats),
    str(repeats),
)
prefill_tallies = re.findall(r"boundary check\s+device (\d+)\s+(\S+)\s+(\d+) / (\d+)\s+(\S+)", prefill_log)
check(
    "every archived prefill tally is exact",
    bool(prefill_tallies) and all(g == w and s == "ok" for _, _, g, w, s in prefill_tallies),
    str([t for t in prefill_tallies if t[4] != "ok"]),
)
check(
    # Was ``quotes(README, len(prefill_tallies), "d")`` -- a bare-integer search
    # over the whole document. It was not stable: under the ``readme_digits``
    # shotgun (every digit incremented) some *other* figure in the file shifted
    # onto 56 and satisfied it, so whether this assertion could fail depended on
    # what unrelated numbers the README happened to publish that day. Anchored
    # to the sentence that states it, where no collision can reach it.
    "README states how many prefill tallies were checked, in the sentence that states it",
    f"per-device tallies, {len(prefill_tallies)} in total, all exact" in " ".join(README.split()),
    f"the prefill log carries {len(prefill_tallies)} tallies",
)
check(
    "README spells how many distinct ops the prefill boundary check tallies",
    *spelled(README, len({op for _, op, _, _, _ in prefill_tallies}), name="prefill distinct tallies"),
)
sparse_expected = {int(g) for _, o, g, _, _ in prefill_tallies if o == "SparseMatmulDeviceOperation"}
check(
    "the length-dependent SparseMatmul tally is the documented 2 * layers * ceil(S/32)",
    sparse_expected == {2 * PREFILL["layers"] * -(-128 // 32)},
    str(sorted(sparse_expected)),
)
check(
    "README quotes that SparseMatmul count beside the op it counts",
    # ``appears(README, "384")`` was satisfied by "below the S ~ 384 crossover"
    # in the prefill-SDPA rejection -- a different published figure that happens
    # to round to the same three digits.
    *quotes(
        README, 2 * PREFILL["layers"] * -(-128 // 32), "d", name="prefill sparse count", context="{} `SparseMatmul`"
    ),
)

# The prefill lever's other rejection operand: at S=128 the config is slower.
# The prefill probe carries two kinds of row: timing rows (``us``) and the
# arbitrary-S alignment rows (``pcc`` only). Only the timing rows are quoted.
prefill_sweep = {(row["seq"], row["cfg"]): row["us"] for row in SDPA_PREFILL if "us" in row}
check(
    "the prefill sweep also covers arbitrary, non-tile-aligned S",
    len({row["seq"] for row in SDPA_PREFILL if "aligned" in row and not row["aligned"]}) >= 5,
    str(sorted({row["seq"] for row in SDPA_PREFILL if "aligned" in row and not row["aligned"]})),
)
check(
    "every arbitrary-S leg still agrees with the default to five decimals",
    len({round(row["pcc"], 5) for row in SDPA_PREFILL if "pcc" in row and row["seq"] == 1}) == 1,
    "",
)
for cfg in ("None", "q128/k128"):
    check(
        f"README quotes the prefill sweep leg {cfg} at S=128",
        *quotes(README, prefill_sweep[(128, cfg)], ".2f", name=f"prefill {cfg}@128"),
    )
check(
    "the prefill config really is a loss at S=128, as the rejection says",
    prefill_sweep[(128, "q128/k128")] > prefill_sweep[(128, "None")],
    f"{prefill_sweep[(128, 'q128/k128')]} vs {prefill_sweep[(128, 'None')]}",
)

# ---------------------------------------------------------------------------
# 6. the layer-stack lower bound
# ---------------------------------------------------------------------------

with (MODEL_DIR / "doc" / "optimized_multichip_decoder" / "perf_decode.csv").open() as handle:
    stage04_decode = {int(row["context_len"]): float(row["median_ms"]) for row in csv.DictReader(handle)}
check("the stage-04 decode CSV still carries a ctx128 row", 128 in stage04_decode, str(sorted(stage04_decode)))
wall_per_layer = stage04_decode.get(128, float("nan"))
check(
    "README quotes the stage-04 wall figure the superseded bound multiplied",
    *quotes(README, wall_per_layer, "g", name="stage-04 wall/layer", restated=RESTATED),
)
check(
    "README quotes the superseded bound as the product of its own operands",
    *quotes(README, DECODE["layers"] * wall_per_layer, ".3f", name="superseded bound", restated=RESTATED),
)

window_decode = log_text(MODEL_DIR / "doc" / "optimized_multichip_decoder" / "window_decode.txt")
# The file's first line is the *previous* iteration; the published window is
# the line the file itself labels as such.
isolated = re.search(r"last-iteration window rows [\d-]+:\s*([\d,]+\.\d+) us", window_decode)
check("the stage-04 single-layer window is parseable", isolated is not None, "")
if isolated:
    isolated_us = float(isolated.group(1).replace(",", ""))
    check(
        "README quotes the stage-04 layer's isolated kernel time",
        *quotes(README, isolated_us, ".2f", name="stage-04 kernel/layer", restated=RESTATED),
    )
    check(
        "README quotes 48 x the isolated layer",
        *quotes(README, DECODE["layers"] * isolated_us / 1000.0, ".3f", name="isolated bound"),
    )
    check(
        "README quotes the in-model layer's premium over the isolated one",
        # work_log restates it unbolded; the README sentence is anchored.
        *quotes(
            README,
            100.0 * (DECODE["per_layer_us"] / isolated_us - 1.0),
            ".1f",
            name="in-model premium",
            context="The in-model layer is **+{}%** on the isolated",
        ),
    )
    check(
        "the in-model layer really is dearer than the isolated one",
        DECODE["per_layer_us"] > isolated_us,
        f"{DECODE['per_layer_us']} vs {isolated_us}",
    )

bound_ms = DECODE["regions_us"]["layer_stack"] / 1000.0
check("README quotes 48 x the optimized in-model layer", *quotes(README, bound_ms, ".3f", name="layer-stack bound"))
check(
    "the layer-stack bound really is 48 x the per-layer figure",
    abs(bound_ms * 1000.0 - DECODE["layers"] * DECODE["per_layer_us"]) < 1e-6,
    "",
)
total_bound_ms = DECODE["iteration_us"] / 1000.0
check(
    "README quotes the bound plus terminal work",
    *quotes(README, total_bound_ms, ".3f", name="bound + terminal", restated=RESTATED),
)
check(
    "the published gap percentage is the recomputed one",
    *quotes(
        README,
        100.0 * (SHIPPED["token_out_ms"] - total_bound_ms) / SHIPPED["token_out_ms"],
        ".2f",
        name="gap",
        restated=RESTATED,
    ),
)
check(
    "the gap is under the 10% the goal flags, which is what the README concludes",
    100.0 * (SHIPPED["token_out_ms"] - total_bound_ms) / SHIPPED["token_out_ms"] < 10.0,
    f"{100.0 * (SHIPPED['token_out_ms'] - total_bound_ms) / SHIPPED['token_out_ms']:.2f}%",
)
check(
    "the superseded bound really was above the measured token-out, which is the tell the README names",
    DECODE["layers"] * wall_per_layer > SHIPPED["token_out_ms"],
    f"{DECODE['layers'] * wall_per_layer} vs {SHIPPED['token_out_ms']}",
)

# ---------------------------------------------------------------------------
# 7. accuracy, readiness and degeneracy -- read out of the archived runs
# ---------------------------------------------------------------------------


def read_topk(path: Path) -> dict:
    text = log_text(path)
    found = re.search(r"AGGREGATE\s+top1=([\d.]+).*?top5=([\d.]+).*?top100=([\d.]+)", text)
    return (
        {"top1": float(found.group(1)), "top5": float(found.group(2)), "top100": float(found.group(3))} if found else {}
    )


for name, log, contract_key in (
    ("prefill", LOGS / "run_prefill_check_argmaxrows.log", "prefill"),
    ("decode", LOGS / "run_teacher_forcing_argmaxrows.log", "decode_teacher_forced"),
):
    stats = read_topk(log)
    check(f"{log.name} has an AGGREGATE line", bool(stats), str(stats))
    if not stats:
        continue
    check(f"{name} meets the top-5 bar", stats["top5"] >= 0.98, str(stats))
    check(f"{name} meets the top-100 bar", stats["top100"] == 1.0, str(stats))
    check(f"README quotes {name} top-1 as measured", in_readme := (f"{stats['top1']:.3f}" in README), "")
    baseline_stats = read_topk(
        STAGE05 / (("run_prefill_check.log") if name == "prefill" else "run_teacher_forcing.log")
    )
    check(f"the stage-05 {name} run is available for comparison", bool(baseline_stats), "")
    check(
        f"{name} accuracy did not move from stage 05, as the README claims",
        stats == baseline_stats,
        f"{stats} vs {baseline_stats}",
    )
    check(
        f"context contract {name} accuracy matches {log.name}",
        all(CONTRACT["full_model_accuracy"][contract_key][f] == stats[f] for f in ("top1", "top5", "top100")),
        f"{CONTRACT['full_model_accuracy'][contract_key]} vs {stats}",
    )


def decode_rate(path: Path) -> float | None:
    found = re.search(r"AGGREGATE.*?decode=([\d.]+) t/s/u", log_text(path))
    return float(found.group(1)) if found else None


tf_before = decode_rate(STAGE05 / "run_teacher_forcing.log")
tf_after = decode_rate(LOGS / "run_teacher_forcing_argmaxrows.log")
check("both teacher-forcing runs report a decode rate", tf_before is not None and tf_after is not None, "")
if tf_before and tf_after:
    check("README quotes the stage-05 teacher-forcing rate", *quotes(README, tf_before, ".2f", name="tf before"))
    check("README quotes the shipped teacher-forcing rate", *quotes(README, tf_after, ".2f", name="tf after"))
    check(
        "the teacher-forcing gain is the recomputed ratio",
        *ratio_is_quoted(README, tf_after, tf_before, ".2f", name="teacher-forcing gain", restated=RESTATED),
    )
    check(
        "teacher-forcing and token-out really are different numbers, as the README insists",
        abs(tf_after - SHIPPED["token_out_tps_user"]) > 1.0,
        f"{tf_after} vs {SHIPPED['token_out_tps_user']}",
    )

# The README states this tally at five sites: two readiness-table cells, the
# prose about the watcher artifact, and two rows of the evidence index. A bare
# ``appears(README, "146")`` was held up by whichever of the five was still
# right, and 146 is short enough that unrelated text could supply it too. The
# two readiness-table cells are the authoritative statement -- they are the
# gate's own result -- so each label anchors to its own cell. The other three
# restate it and are deliberately left to those two rather than becoming three
# more sites this checker has to keep in step.
for label, log, cell in (
    ("plain", LOGS / "pytest_argmax_rows.log", "**{} passed**,"),
    ("watcher", LOGS / "watcher_argmaxrows.log.gz", "**{} passed, zero tripped asserts**"),
):
    text = log_text(log)
    found = re.findall(r"(\d+) passed", text)
    check(f"the {label} pytest run is archived with a tally", bool(found), str(log))
    if found:
        check(
            f"README quotes the {label} run's tally in the readiness-table cell that reports it",
            *quotes(README, int(found[-1]), "d", name=f"{label} passed", context=cell),
        )
        check(f"nothing failed in the {label} run", " failed" not in text.split("=====")[-1], "")
watcher_text = log_text(LOGS / "watcher_argmaxrows.log.gz")
check(
    "the watcher run tripped no asserts",
    watcher_text.count("tripped an assert") == 0,
    f"{watcher_text.count('tripped an assert')} tripped",
)
check(
    "the watcher run is the whole suite, not a final dump -- it carries its own tally",
    bool(re.search(r"\d+ passed", watcher_text)) and "test session starts" in watcher_text,
    "",
)
check(
    "the plain and watcher runs covered the same number of tests",
    re.findall(r"(\d+) passed", log_text(LOGS / "pytest_argmax_rows.log"))[-1]
    == re.findall(r"(\d+) passed", watcher_text)[-1],
    "",
)

degeneracy = log_text(LOGS / "check_degenerate_argmaxrows.log")
check("the degeneracy gate output is archived", bool(degeneracy), "logs/check_degenerate_argmaxrows.log")
check("the archived degeneracy run found nothing", "No degenerate output detected" in degeneracy, "")
measured = re.search(
    r"'num_tokens': (\d+).*?'adjacent_duplication': ([\d.]+).*?'trigram_loop_fraction': ([\d.]+)",
    degeneracy,
    re.S,
)
check("the archived degeneracy run reports its metrics", measured is not None, "")
if measured:
    # This was a loop of three ``appears(README, value)`` searches, and it was
    # the worst of the vacuous ones: ``109`` is satisfied by the README's own
    # sentence *about this checker* -- "`README quotes the degeneracy metric
    # 109`" appears verbatim in the section explaining how mutation credit used
    # to be keyed by name. So the assertion that the README publishes the
    # degeneracy metric was held up by the README's prose about the assertion.
    # Corrupt the artifact and the meta-prose still satisfied it. ``0.0`` was
    # satisfied by the vllm paragraph's unrelated ``adjacent_duplication`` is
    # **0.0**. One anchored sentence, built out of all three parsed values,
    # replaces all three searches.
    tokens, duplication, trigram = measured.groups()
    sentence = (
        f"The degeneracy gate measures `num_tokens` {tokens}, `adjacent_duplication` **{duplication}**, "
        f"`trigram_loop_fraction` **{trigram}**"
    )
    check(
        "README publishes the degeneracy gate's three metrics in the sentence that reports them",
        phrase(README, sentence),
        f"the log reports {tokens} / {duplication} / {trigram}; the README does not carry that sentence",
    )

# The six-prompt qualitative suite, re-run on the shipped sampler this stage.
qual_path = PROBES / "vllm_qualitative_outputs_argmaxrows.json"
check("the stage-06 qualitative suite's completions are archived", qual_path.is_file(), str(qual_path))
# This checker reads the evidence-tree copy, which left the *committed* readiness
# artifact unchecked by anything -- the round-2 review caught that. They are the
# same run and must stay the same bytes, so assert it rather than disclose it.
readiness_qual = MODEL_DIR / "readiness_qualitative" / "vllm_qualitative_outputs.json"
check("the committed readiness qualitative artifact is on file", readiness_qual.is_file(), str(readiness_qual))
check(
    "the committed readiness completions are the same run this checker reads from the evidence tree",
    readiness_qual.is_file() and qual_path.is_file() and load(readiness_qual) == load(qual_path),
    "readiness_qualitative/vllm_qualitative_outputs.json has drifted from probes/vllm_qualitative_outputs_argmaxrows.json",
)
qual_log = log_text(LOGS / "check_degenerate_vllm_argmaxrows.log")
check(
    "the stage-06 qualitative degeneracy score is archived", bool(qual_log), "logs/check_degenerate_vllm_argmaxrows.log"
)
check("that scoring run found nothing degenerate", "No degenerate output detected" in qual_log, "")
if qual_path.is_file():
    QUAL = load(qual_path)
    # ``spelled`` alone is weak for a small count -- "six" occurs all over a
    # document like this. The word for the *computed* count must appear where
    # the README names the suite, immediately before "prompts".
    qual_word = NUMBER_WORDS.get(len(QUAL))
    check(
        "README states the re-run suite's prompt count, in words, where it names the suite",
        qual_word is not None and re.search(rf"\b{qual_word}\*{{0,2}}\s+prompts", README, re.I) is not None,
        f"computed {len(QUAL)} -> {qual_word!r}, not stated beside 'prompts'",
    )
    check(
        "every prompt has both a greedy and a sampled completion, and none is empty",
        all(row.get("greedy_completion") and row.get("sampled_completion") for row in QUAL),
        "",
    )
    check(
        "the scored artifact is the archived one -- the score covers every completion",
        qual_log.count("greedy_completion") == qual_log.count("sampled_completion") == len(QUAL),
        f"{len(QUAL)} prompts, log scores {qual_log.count('greedy_completion')} greedy legs",
    )
    zero_dup = len(re.findall(r"'adjacent_duplication': 0\.0(?!\d)", qual_log))
    check(
        "README's count of qualitative legs with zero adjacent duplication is the log's",
        NUMBER_WORDS.get(zero_dup) is not None
        and re.search(rf"\b{NUMBER_WORDS[zero_dup]}\*{{0,2}}\s+of\s+\**the twelve", README, re.I) is not None,
        f"the score log reports {zero_dup} of {2 * len(QUAL)}",
    )
    identical = sum(1 for row in QUAL if row["greedy_completion"] == row["sampled_completion"])
    # ``appears(README, str(identical)) and "collapse" in README`` was vacuous:
    # a small integer appears somewhere in a 55 KB document whatever its value,
    # so forcing all six legs to collapse (4 -> 6) still passed. The count must
    # be spelled in the sentence that makes the claim, and both operands of that
    # sentence are computed here.
    collapsed_word, suite_word = NUMBER_WORDS.get(identical), NUMBER_WORDS.get(len(QUAL))
    check(
        "README reports how many sampled legs collapsed onto their greedy leg, in the sentence that says so",
        collapsed_word is not None
        and suite_word is not None
        and re.search(
            rf"\b{collapsed_word}\s+of\s+the\s+{suite_word}\s+prompts\s+the\s+sampled\s*\**\s+"
            rf"\**\s*completion\s+is\s+byte-identical",
            README,
            re.I,
        )
        is not None,
        f"recomputed {identical} of {len(QUAL)} -> '{collapsed_word} of the {suite_word} prompts "
        "the sampled completion is byte-identical' is not in the README",
    )
    check(
        "the suite is the shared one, not a local copy of the prompts",
        "vllm_prompts.txt" in (PROBES / "qualitative_probe.py").read_text(encoding="utf-8"),
        "",
    )
    check(
        "the stage-06 probe writes its own archive rather than into the stage-05 tree",
        "vllm_qualitative_outputs_argmaxrows.json" in (PROBES / "qualitative_probe.py").read_text(encoding="utf-8")
        and "doc/full_model/qualitative_check.log" not in (PROBES / "qualitative_probe.py").read_text(encoding="utf-8"),
        "",
    )
    stage05_qual = MODEL_DIR / "doc" / "full_model" / "qualitative_check.log"
    check(
        "the stage-05 qualitative evidence is still on disk, so the two can be compared",
        stage05_qual.is_file(),
        str(stage05_qual),
    )
    check(
        "README no longer carries the inherited-qualitative-evidence limitation",
        "qualitative six-prompt suite was not re-run" not in README,
        "the limitation is closed but still stated",
    )

meta_path = MODEL_DIR / "readiness_autoregressive" / "autoregressive_meta.json"
check("the autoregressive metadata is present", meta_path.is_file(), str(meta_path))
if meta_path.is_file():
    meta = load(meta_path)
    hf_ids, tt_ids = meta["hf"]["token_ids"], meta["tt"]["token_ids"]
    check("HF and TT both produced 128 tokens", meta["hf"]["num_tokens"] == meta["tt"]["num_tokens"] == 128, "")
    matching = sum(1 for a, b in zip(hf_ids, tt_ids) if a == b)
    check(
        "README's matching-token count is the recomputed one",
        f"**{matching} of 128** tokens match" in README,
        f"recomputed {matching}",
    )
    indices = [i for i, (a, b) in enumerate(zip(hf_ids, tt_ids)) if a == b]
    # ``all(appears(README, str(i)) ...)`` was vacuous for small indices --
    # rewriting the metadata's matching positions from [1, 94] to [7, 33] still
    # passed, because "7" and "33" both occur elsewhere in the document. The
    # list has to be quoted **as a list**, in the sentence that names it, and
    # the phrase is built here from the metadata.
    listed = (
        ", ".join(str(i) for i in indices[:-1]) + " and " + str(indices[-1])
        if len(indices) > 1
        else str(indices[0])
        if indices
        else ""
    )
    check(
        "README lists the matching indices the metadata contains, as a list",
        bool(indices) and re.search(rf"at\s+indices\s+{re.escape(listed)}\b", README) is not None,
        f"recomputed 'at indices {listed}'",
    )
    prefix = 0
    for a, b in zip(hf_ids, tt_ids):
        if a != b:
            break
        prefix += 1
    check(
        "the common prefix is what the README claims it is",
        prefix == 0 and "first generated token" in README,
        f"recomputed prefix {prefix}",
    )
    # ``appears(README, str(ids[0]))`` is a bare-integer search over a 75 KB
    # document, and it did not merely look weak -- it collided. Mutating TT's
    # first token id from 264 to 265 stopped failing the moment an unrelated
    # figure published in this same file happened to be 265 (a measured shotgun
    # breadth, which moves whenever the assertion count does). The assertion
    # then had only shotgun coverage and the gate caught it. Both ids are now
    # asserted in the sentence that quotes them, where a collision elsewhere in
    # the document cannot satisfy them.
    for label, ids, template in (("HF", hf_ids, "HF's first token is `{}`"), ("TT", tt_ids, "TT's is `{}`")):
        check(
            f"README quotes {label}'s first generated token id in the sentence that states it",
            phrase(README, template.format(ids[0])),
            f"{label}'s first id is {ids[0]}; the README's sentence does not say so",
        )
    # The stage-05 figure this sentence compares against is on file in stage
    # 05's own gate log, so it is read from there rather than hardcoded. The
    # previous version parsed a regex out of the README, threw the result away
    # and compared against a literal 4 -- so the artifact side could not fail.
    stage05_degeneracy = log_text(STAGE05 / "check_degenerate_output.log")
    stage05_match = re.search(r"'matching_tokens': (\d+)", stage05_degeneracy)
    check("stage 05's own agreement figure is on file", stage05_match is not None, str(STAGE05))
    stage05_matching = int(stage05_match.group(1)) if stage05_match else -1
    check(
        "README states the stage-05 agreement it says changed, as stage 05 measured it",
        f"Stage 05 measured {stage05_matching} matching tokens" in README,
        f"stage 05's log says {stage05_matching}",
    )
    check(
        "the agreement really did change, which is what makes that sentence worth writing",
        matching != stage05_matching,
        f"{matching} vs stage 05's {stage05_matching}",
    )
    # And the reason it changed is on file: the adopted path is not bit-identical.
    adopted = [row for row in SDPA_PCC if row["leg"] == "adopted"]
    check("the in-model SDPA PCC probe has an adopted leg", bool(adopted), "")
    check(
        "every adopted-leg PCC clears the layer bar, which is why the change is acceptable",
        all(row["pcc_vs_hf"] > 0.995 for row in adopted),
        str([row["pcc_vs_hf"] for row in adopted]),
    )
    check(
        "README quotes an in-model PCC for the adopted path",
        any(f"{row['pcc_vs_hf']:.4f}" in README for row in adopted),
        str([f"{row['pcc_vs_hf']:.4f}" for row in adopted]),
    )

# ---------------------------------------------------------------------------
# 8. the rejection ledger
# ---------------------------------------------------------------------------

# -- the LM head, closed on arithmetic ---------------------------------------

report = log_text(DOC / "tt_perf_report_full_model_48layer_decode.txt.gz")
check("the shipped tt-perf-report is on disk", bool(report), "tt_perf_report_full_model_48layer_decode.txt.gz")
# ``tt-perf-report`` merges the four devices and prints the **slowest** one's
# row, so the row's device id is parsed out and its own kernel time is used.
# Pairing device 0's 226.130 us with this row's 66.4% -- which is device 3's --
# mixed two devices and was the stage-06 review's B4.
lm_row = re.search(
    r"DRAM\s+MatmulDeviceOperation 32 x 2048 x 37984\s+(\d+)\s+\d+ [^\s]+s.*?(\d+) GB/s\s+([\d.]+) %", report
)
check("the shipped report still carries the LM head's bandwidth row", lm_row is not None, "")
if lm_row:
    lm_device = lm_row.group(1)
    bandwidth, utilisation = int(lm_row.group(2)), float(lm_row.group(3))
    # Both were bare document-wide searches, and the bandwidth one was another
    # demonstrated collision: under ``readme_digits`` a shotgun-breadth figure of
    # 239 published elsewhere in this README shifts to 340 and satisfies the
    # search for "340". Because the breadths move whenever the assertion count
    # does, that made whether this assertion could fail a function of an
    # unrelated number. Anchored by its unit, where a bare integer cannot reach.
    check(
        "README quotes the LM head's measured bandwidth with its unit",
        f"{bandwidth} GB/s" in README,
        f"the report row measures {bandwidth} GB/s",
    )
    check(
        "README quotes the LM head's bandwidth utilisation where it names it",
        f"DRAM-bound at {utilisation:.1f}%" in README,
        f"the report row measures {utilisation:.1f}%",
    )
    check(
        "the report's LM-head row is a device this profile has a kernel time for",
        lm_device in DECODE["lm_head_us_all_devices"],
        f"report row is device {lm_device}, profile has {sorted(DECODE['lm_head_us_all_devices'])}",
    )
    lm_device_us = DECODE["lm_head_us_all_devices"].get(lm_device, 0.0)
    check(
        "README names the device that report row is, since it is not the device the split is quoted on",
        f"device {lm_device}" in README,
        f"the row is device {lm_device}",
    )
    check(
        "README quotes that device's own LM-head kernel time, which is what the utilisation belongs to",
        *quotes(README, lm_device_us, ".3f", name=f"lm head us on device {lm_device}", restated=RESTATED),
    )
    headroom = lm_device_us * (1.0 - utilisation / 100.0)
    check(
        "README quotes the LM head headroom", *quotes(README, headroom, ".2f", name="lm headroom us", restated=RESTATED)
    )
    # The other self-consistent reading, on the device the rest of the profile is
    # quoted on. Utilisation x duration is the same byte count on every die, so
    # the reported device's own utilisation follows from the report row's.
    reported_utilisation = utilisation * lm_device_us / DECODE["lm_head_us"]
    check(
        "README quotes the reported device's own bandwidth utilisation",
        *quotes(README, reported_utilisation, ".2f", name="lm % on the reported device"),
    )
    check(
        "README quotes the headroom that follows on the reported device",
        *quotes(README, DECODE["lm_head_us"] * (1.0 - reported_utilisation / 100.0), ".2f", name="lm headroom d0"),
    )
    check(
        "the two readings really do differ, which is why the README states which device each is",
        abs(headroom - DECODE["lm_head_us"] * (1.0 - reported_utilisation / 100.0)) > 1.0,
        "",
    )
    check(
        "README quotes that headroom as a share of token-out",
        *quotes(README, 100.0 * headroom / 1000.0 / SHIPPED["token_out_ms"], ".2f", name="lm headroom %"),
    )
    check(
        "the LM head lever really is under 1% of token-out, which is the closure argument",
        100.0 * headroom / 1000.0 / SHIPPED["token_out_ms"] < 1.0,
        "",
    )
check(
    "the report still prints the DRAM-sharded advice the README says is unexpressible",
    "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig" in report,
    "",
)
raw_local_vocab = AUDIT["audit"].get("lm_head_local_vocab", "")
check("the audit reports a numeric per-die vocabulary", str(raw_local_vocab).isdigit(), str(raw_local_vocab))
local_vocab = int(raw_local_vocab) if str(raw_local_vocab).isdigit() else 0
check(
    "README quotes the per-die vocabulary where it states the wrapper's contract",
    # 37984 legitimately recurs 13 times -- it is half of every LM-head op name
    # the profile tables print. Those are op shapes, not a statement of the
    # split; the wrapper contract line is where the README *asserts* the per-die
    # vocabulary, so that is the site anchored. The op-name occurrences are
    # covered by the report-row assertions that parse them.
    phrase(README, f"local vocab {local_vocab},"),
    str(local_vocab),
)
tiles = local_vocab // 32 if local_vocab else 0
check("README quotes the per-die vocabulary in tiles", *quotes(README, tiles, "d", name="tiles", restated=RESTATED))
check(
    "that tile count really is prime, which is the whole closure",
    tiles > 1 and all(tiles % d for d in range(2, int(tiles**0.5) + 1)),
    f"{tiles}",
)
check(
    "README quotes the per-bank column count that is not tile-aligned",
    *quotes(README, local_vocab // 8, "d", name="columns per bank", restated=RESTATED),
)
check(
    "that per-bank column count really is not a multiple of 32",
    local_vocab > 0 and (local_vocab // 8) % 32 != 0,
    f"{local_vocab // 8}",
)
check("the audit confirms the vocabulary is unpadded", AUDIT["audit"]["vocab_padding"] == "0", "")

# -- the MoE skew, closed on statistics ---------------------------------------

skew = MOE["skew_is_combinatorial"]
budget = MOE["budget"]
check(
    "README quotes the measured mean per-die maximum",
    *quotes(README, skew["measured_mean_max_k"], ".3f", name="measured max_k"),
)
check(
    "README quotes the uniform-routing expected maximum",
    *quotes(README, skew["uniform_expected_max_k"], ".3f", name="uniform max_k"),
)
check("README quotes the chi-square", *quotes(README, skew["chi2_vs_uniform"], ".2f", name="chi2"))
# The claim that used to be checked here -- "the shipped layout is already 0.69
# us/layer better than the expectation for an arbitrary partition" -- was noise
# at z = -0.8. It is withdrawn from the README and from the probe, so what is
# asserted now is that the *withdrawal* holds and that the statistics the
# rewritten argument rests on are published with it.
check(
    "no document still argues that the shipped layout beats an arbitrary partition",
    "better than the expectation for an arbitrary partition" not in DOCS
    and "a permutation is negative in expectation" not in DOCS,
    "the withdrawn expectation argument is back",
)
check(
    "the analysis publishes the chi-square's degrees of freedom beside the statistic itself",
    # ``appears(README, "5")`` -- a one-character needle with 8 hits, among them
    # "top-5", ">= 5" and a markdown list marker "5.". It could not fail.
    *quotes(README, skew["chi2_df"], "d", name="chi2 df", context=", df {}, p "),
)
check("README quotes the chi-square's p-value", *quotes(README, skew["chi2_p_value"], ".2f", name="chi2 p"))
check(
    "the chi-square pools its sparse tail rather than dropping it",
    str(skew["chi2_pooled_from_k"]) in skew["chi2_expected_pooled"]
    and all(count >= 1 for count in skew["chi2_observed_pooled"].values())
    and sum(skew["chi2_observed_pooled"].values()) == sum(n for _, n in skew["measured_k_histogram"]),
    f"pooled observations {skew['chi2_observed_pooled']} against histogram {skew['measured_k_histogram']}",
)
check(
    "README quotes the per-die marginal over the whole iteration, as the quadruple it is",
    # Four two- and three-digit needles searched separately over 55 KB. Even
    # where each happened to occur once today, nothing tied them to each other
    # or to this claim. The README prints them as one quadruple; so does this.
    phrase(
        README,
        "the four dies fire **{}** times".format(" / ".join(str(v) for v in skew["per_die_total_active_experts"])),
    ),
    str(skew["per_die_total_active_experts"]),
)
check(
    "README quotes that marginal's chi-square and p-value",
    *quotes(README, skew["per_die_chi2"], ".2f", name="per-die chi2"),
)
check(
    "...and its p-value",
    *quotes(README, skew["per_die_p_value"], ".2f", name="per-die p", restated=RESTATED),
)
check(
    "the per-die marginal really does sum to the router's top-8 over every layer",
    sum(skew["per_die_total_active_experts"]) == 48 * 8,
    str(sum(skew["per_die_total_active_experts"])),
)
check(
    "the analysis discloses that its 192 counts are not 192 independent draws",
    "not " in skew["independence_caveat"].lower()
    and "single decode token" in skew["independence_caveat"]
    and "single decode token" in README,
    "the one-token caveat is in the artifact but not in the README",
)

# -- the multi-token routing sample, which reopened this lever ----------------
#
# Every figure the README quotes about routing persistence came from these files
# and **nothing checked any of it** -- which is how the README came to be
# quoting an n=2 cross-prompt range after an n=3 run had superseded it. The
# figures are re-derived here like every other published number.

ROUTING = [load(PROBES / f"moe_routing_across_tokens{tag}.json") for tag in ("", "_prompt2", "_prompt3")]
CROSS = load(PROBES / "moe_routing_cross_prompt.json")

check(
    # ``len({run["note"][:0] or id(run) ...})`` was vacuous -- ``[:0]`` is always
    # "" and ``id(run)`` is always distinct, so the second clause could not fail
    # and the first could only crash the loader. What makes the sample three
    # *prompts* rather than one prompt three times is that the routing differs,
    # so that is what is asserted.
    "the routing sample covers three genuinely different prompts, which is the whole point of it",
    len(ROUTING) >= 3
    and len({tuple(run["per_die_counts"]["per_die_total_selections"]) for run in ROUTING}) == len(ROUTING)
    and len({round(run["per_expert_hotness"]["mean_top8_share"], 6) for run in ROUTING}) == len(ROUTING),
    str([run["per_die_counts"]["per_die_total_selections"] for run in ROUTING]),
)
for index, run in enumerate(ROUTING):
    check(
        f"routing run {index} sampled many decode tokens, not one",
        run["tokens"] >= 64 and run["router_calls"] == run["tokens"] * run["layers"],
        f"{run['tokens']} tokens x {run['layers']} layers = {run['router_calls']} router calls",
    )
    check(
        f"README quotes run {index}'s top-8 share of selections",
        *quotes(
            README,
            100.0 * run["per_expert_hotness"]["mean_top8_share"],
            ".1f",
            name=f"top8 share {index}",
            restated=RESTATED,
        ),
    )
    check(
        f"README quotes run {index}'s per-die selection counts",
        all(appears(README, str(v)) for v in run["per_die_counts"]["per_die_total_selections"]),
        str(run["per_die_counts"]["per_die_total_selections"]),
    )
    check(
        f"README quotes run {index}'s held-out within-prompt gain",
        *quotes(
            README,
            run["permutation_search"]["held_out_gain_ms_per_iteration"],
            ".3f",
            name=f"held-out {index}",
            restated=RESTATED,
        ),
    )
    check(
        f"run {index}'s held-out gain really is below its in-sample one, as the README says",
        run["permutation_search"]["fitted_mean_max_k_per_layer_held_out"]
        > run["permutation_search"]["fitted_mean_max_k_per_layer_in_sample"],
        f"in sample {run['permutation_search']['fitted_mean_max_k_per_layer_in_sample']}, "
        f"held out {run['permutation_search']['fitted_mean_max_k_per_layer_held_out']}",
    )
check(
    "the hotness really is far above what independent uniform routing would give",
    all(
        run["per_expert_hotness"]["mean_top8_share"] > 5 * run["per_expert_hotness"]["share_if_uniform_and_independent"]
        for run in ROUTING
    ),
    str([round(run["per_expert_hotness"]["mean_top8_share"], 4) for run in ROUTING]),
)
check(
    "README quotes the share independent uniform routing would give",
    # Two sites state this; the sentence in the MoE argument is the one that
    # makes the claim, the other is a parenthetical in the ledger row.
    *quotes(
        README,
        100.0 * ROUTING[0]["per_expert_hotness"]["share_if_uniform_and_independent"],
        ".1f",
        name="uniform share",
        context="Independent uniform routing would give {}%.",
    ),
)
check(
    "the cross-prompt fit used every ordered pair of the prompts sampled",
    CROSS["directions"] == len(ROUTING) * (len(ROUTING) - 1) == len(CROSS["cross_prompt"]),
    f"{CROSS['directions']} directions over {len(ROUTING)} prompts",
)
check(
    "README quotes the cross-prompt range's floor",
    *quotes(README, CROSS["gain_ms_per_iteration_min"], ".3f", name="cross min", restated=RESTATED),
)
check(
    "README quotes the cross-prompt range's ceiling",
    *quotes(README, CROSS["gain_ms_per_iteration_max"], ".3f", name="cross max", restated=RESTATED),
)
check(
    "README quotes the cross-prompt mean",
    *quotes(README, CROSS["gain_ms_per_iteration_mean"], ".3f", name="cross mean"),
)
pooled = CROSS["pooled_fit_held_out"]
check(
    "the pooled fit holds out one prompt at a time, over every prompt",
    len(pooled) == len(ROUTING) and all(row["held_out_index"] not in row["pooled_over_indices"] for row in pooled),
    str([(row["held_out_index"], row["pooled_over_indices"]) for row in pooled]),
)
for row in pooled:
    check(
        f"README quotes the pooled fit's gain on held-out prompt {row['held_out_index']}",
        *quotes(
            README,
            row["pooled_gain_ms_per_iteration"],
            ".3f",
            name=f"pooled {row['held_out_index']}",
            restated=RESTATED,
        ),
    )
pooled_mean = sum(row["pooled_gain_ms_per_iteration"] for row in pooled) / len(pooled)
check("README quotes the pooled fit's mean", *quotes(README, pooled_mean, ".3f", name="pooled mean", restated=RESTATED))
check(
    "README quotes that mean as a share of token-out",
    *quotes(README, 100.0 * pooled_mean / SHIPPED["token_out_ms"], ".2f", name="pooled % of token-out"),
)
check(
    "pooling really does transfer better than the best single-prompt fit, which is why the range is a floor",
    all(row["pooling_transfers_better"] is True for row in pooled)
    and all(
        row["pooled_gain_ms_per_iteration"] > row["best_single_prompt_fit_gain_ms_per_iteration"] for row in pooled
    ),
    str([(row["pooled_gain_ms_per_iteration"], row["best_single_prompt_fit_gain_ms_per_iteration"]) for row in pooled]),
)
# The superseded two-prompt range may still be *named* -- the documents explain
# that it was published and why it was wrong, which is the point of keeping a
# correction. What must not survive is it being quoted as the live figure, so
# what is asserted is that the ledger row and the summary table carry the
# artifact's range and not that one.
superseded = ("0.024–0.028", "0.024-0.028")
ledger_row = next((line for line in README.splitlines() if "Permuting experts across dies" in line), "")
check("the MoE ledger row is still in the table", bool(ledger_row), "")
check(
    "the ledger row quotes the pooled held-out gain, not the superseded two-prompt range",
    all(text not in ledger_row for text in superseded) and format(pooled_mean, ".3f") in ledger_row,
    ledger_row[:160],
)


def _every_occurrence(haystack: str, needle: str):
    start = haystack.find(needle)
    while start != -1:
        yield start
        start = haystack.find(needle, start + 1)


check(
    # ``DOCS.index(text)`` finds only the FIRST occurrence, so a later, bare
    # restatement of the superseded figure sailed past this -- the mutation that
    # appends one broke nothing. Every occurrence is checked now.
    "wherever a document still names the superseded range, it names it as superseded",
    all(
        "first published" in DOCS[max(0, at - 200) : at + 200] or "superseded" in DOCS[max(0, at - 200) : at + 200]
        for text in superseded
        for at in _every_occurrence(DOCS, text)
    ),
    "the two-prompt range is quoted somewhere without saying it was superseded",
)
# The range this stage publishes, asserted end to end so it cannot drift back to
# the two-prompt figure: its floor is the smallest single-prompt direction, its
# ceiling is the largest pooled held-out fit, and both ends are quoted as a share
# of token-out.
range_low = CROSS["gain_ms_per_iteration_min"]
range_high = max(row["pooled_gain_ms_per_iteration"] for row in pooled)
check(
    "the published range's ceiling is the pooled fit's, which is above every single-prompt direction",
    range_high > CROSS["gain_ms_per_iteration_max"] > range_low,
    f"{range_low:.4f} .. {CROSS['gain_ms_per_iteration_max']:.4f} single, {range_high:.4f} pooled",
)
check(
    f"README publishes the measured range as {range_low:.3f}-{range_high:.3f} ms/iteration",
    f"{range_low:.3f}–{range_high:.3f} ms/iteration" in README,
    f"{range_low:.3f}–{range_high:.3f}",
)
check(
    "README publishes that range as a share of token-out, both ends",
    f"{100.0 * range_low / SHIPPED['token_out_ms']:.2f}–{100.0 * range_high / SHIPPED['token_out_ms']:.2f}%" in README,
    f"{100.0 * range_low / SHIPPED['token_out_ms']:.2f}-{100.0 * range_high / SHIPPED['token_out_ms']:.2f}%",
)
check(
    "the rejection still stands on the top of the range, not on the bottom of it",
    100.0 * range_high / SHIPPED["token_out_ms"] < 1.0 and "declined" in README,
    f"the largest measured gain is {100.0 * range_high / SHIPPED['token_out_ms']:.2f}% of token-out",
)
# Structural corroboration of the fit, from a direction the stochastic search
# cannot bias.
structure = CROSS["shared_structure"]
check(
    "prompt pairs share far more of their hot experts than independent routing would give",
    structure["mean_top8_overlap_over_pairs"] > 3 * structure["top8_overlap_under_independent_routing"],
    f"{structure['mean_top8_overlap_over_pairs']:.3f} of 8 against "
    f"{structure['top8_overlap_under_independent_routing']:.2f} by chance",
)
check(
    "README quotes the measured top-8 hot-set overlap between prompts",
    *quotes(README, structure["mean_top8_overlap_over_pairs"], ".2f", name="top-8 overlap"),
)
check(
    "README quotes the overlap independent routing would give",
    *quotes(README, structure["top8_overlap_under_independent_routing"], ".2f", name="overlap by chance"),
)
check(
    "README quotes the per-expert rank correlation between prompts",
    *quotes(README, structure["mean_rank_correlation_over_pairs"], ".3f", name="rank correlation"),
)
check(
    "the ledger no longer claims the achievable saving is zero",
    "achievable saving **0 ms**" not in DOCS,
    "the withdrawn 0 ms claim is back",
)
check(
    "README quotes the measured skew idle per iteration",
    *quotes(README, budget["measured_idle_ms_per_iteration"], ".3f", name="measured idle"),
)
check(
    "README quotes the uniform-routing floor",
    *quotes(README, budget["uniform_routing_floor_ms_per_iteration"], ".3f", name="uniform floor"),
)
check(
    'measured idle is below the uniform-routing floor -- true, and not the same claim as "zero is achievable"',
    budget["measured_idle_ms_per_iteration"] < budget["uniform_routing_floor_ms_per_iteration"],
    f"{budget['measured_idle_ms_per_iteration']} vs {budget['uniform_routing_floor_ms_per_iteration']}",
)
check(
    "the active-expert recovery validates itself: the per-die counts sum to the router's top-8",
    MOE["expert_count_recovery"]["sums_to_top_k_in_every_layer"] is True,
    str(MOE["expert_count_recovery"]["per_layer_sum_over_dies"]),
)
rs = MOE["reduce_scatter_is_wait"]
check(
    "the MoE collective still correlates with the lag and the attention one still does not",
    rs["corr_lag_vs_moe_rs"] > 0.9 and rs["corr_lag_vs_attn_rs"] < 0.3,
    f"{rs['corr_lag_vs_moe_rs']:.3f} vs {rs['corr_lag_vs_attn_rs']:.3f}",
)
check(
    "README quotes the MoE collective's correlation with the lag",
    *quotes(DOCS, rs["corr_lag_vs_moe_rs"], ".3f", name="moe corr"),
)
check(
    "README quotes the attention collective's correlation with the lag",
    *quotes(DOCS, rs["corr_lag_vs_attn_rs"], ".3f", name="attn corr"),
)
check(
    "README quotes the regression slope",
    *quotes(DOCS, rs["slope_us_moe_rs_per_us_lag"], ".3f", name="slope"),
)
check(
    "the analysis was re-run on the shipped profile, so its chi-square differs from pass 3's",
    abs(skew["chi2_vs_uniform"] - 8.062184838677412) > 0.01,
    "the 'final' analysis reproduces pass 3's numbers exactly -- is it reading the superseded CSV?",
)

# -- the sampler ledger rows --------------------------------------------------

for key, label in (
    ("argmax_keepdim_true", "argmax, keepdim=True"),
    ("argmax_keepdim_false", "argmax, keepdim=False"),
    ("whole_reduction_32rows", "whole reduction, 32 rows"),
    ("whole_reduction_1row", "whole reduction, 1 row"),
):
    if key in ARGMAX:
        check(
            f"README or work log quotes {label}",
            *quotes(DOCS, ARGMAX[key]["ms"] * 1000.0, ".1f", name=label, restated=RESTATED),
        )
argmax_keys = {"argmax_keepdim_false", "argmax_keepdim_true", "rm_slice1_then_argmax"}
check("the argmax probe still carries the legs the ledger quotes", argmax_keys <= set(ARGMAX), str(sorted(ARGMAX)))
if argmax_keys <= set(ARGMAX):
    check(
        "keepdim=False really is faster on its own, as the ledger says",
        ARGMAX["argmax_keepdim_false"]["ms"] < ARGMAX["argmax_keepdim_true"]["ms"],
        "",
    )
check(
    "the row slice really is what wins, not keepdim -- it beats keepdim=False on the whole reduction",
    ARGMAX["rm_slice1_then_argmax"]["ms"] < ARGMAX["argmax_keepdim_false"]["ms"],
    f"{ARGMAX['rm_slice1_then_argmax']['ms']} vs {ARGMAX['argmax_keepdim_false']['ms']}",
)
# -- the rest of the sampler ledger, from the two probe runs ------------------

for key, label, spec in (
    ("topk_k32_tile_32rows", "ttnn.topk(k=32) on the TILE tensor", ".1f"),
    ("tile_slice1_plus_untilize", "the TILE-layout slice then untilize", ".1f"),
    ("untilize_37984", "the full untilize", ".1f"),
    ("full_shipped_keepdim_true", "the whole reduction over 32 rows", ".1f"),
    ("full_candidate_batch1", "the whole reduction over 1 row", ".1f"),
):
    check(f"the argmax probe still carries {label}", key in ARGMAX, str(sorted(ARGMAX)))
    if key in ARGMAX:
        check(
            f"README quotes {label}",
            *quotes(README, ARGMAX[key]["ms"] * 1000.0, spec, name=label, restated=RESTATED),
        )
if {"full_shipped_keepdim_true", "full_candidate_batch1"} <= set(ARGMAX):
    check(
        "the 2.52x whole-sampler gain is the ratio of the two measured reductions",
        *ratio_is_quoted(
            README,
            ARGMAX["full_shipped_keepdim_true"]["ms"],
            ARGMAX["full_candidate_batch1"]["ms"],
            ".2f",
            name="live-row gain",
        ),
    )
if {"tile_slice1_plus_untilize", "untilize_37984"} <= set(ARGMAX):
    check(
        "the TILE slice really saves almost nothing, as the ledger says",
        ARGMAX["untilize_37984"]["ms"] - ARGMAX["tile_slice1_plus_untilize"]["ms"] < 0.010,
        f"{(ARGMAX['untilize_37984']['ms'] - ARGMAX['tile_slice1_plus_untilize']['ms']) * 1000:.1f} us",
    )
check(
    "ttnn.topk(k=1) on the ROW_MAJOR tensor did not build, as the ledger says",
    "error" in ARGMAX.get("topk_k1_rm_32rows", {}),
    str(ARGMAX.get("topk_k1_rm_32rows")),
)
check(
    "README names the TT_FATAL that blocks it",
    "topk_device_operation.cpp:166" in README
    and "topk_device_operation.cpp:166" in ARGMAX.get("topk_k1_rm_32rows", {}).get("error", ""),
    "",
)
padding_tokens = ARGMAX.get("padding_rows_produce_token_zero", {}).get("tokens")
# The default used to be ``[1]``, sliced ``[1:]`` to the empty list, so
# ``all(...)`` was vacuously true and deleting the whole probe leg caused no
# failure at all. The leg has to be present and have padding rows to check.
check(
    "the padding-row leg is on file and has padding rows in it",
    isinstance(padding_tokens, list) and len(padding_tokens) > 1,
    str(padding_tokens),
)
check(
    "the padding rows really do reduce to token 0, which is what makes the slice exact",
    isinstance(padding_tokens, list) and len(padding_tokens) > 1 and all(token == 0 for token in padding_tokens[1:]),
    str(padding_tokens[:4] if isinstance(padding_tokens, list) else padding_tokens),
)
check(
    "the crafted tie cases are on file and still return the first-maximal index",
    bool(ARGMAX.get("ties")),
    str(ARGMAX.get("ties"))[:120],
)

sub_core = {
    int(key.rsplit("cores", 1)[1]): value["ms"] * 1000.0
    for key, value in ARGMAX_A.items()
    if key.startswith("argmax_kd_true_cores")
}
check("the sub_core_grids sweep is on file", len(sub_core) >= 3, str(sorted(sub_core)))
for cores, value in sorted(sub_core.items()):
    check(
        f"README quotes the sub_core_grids leg at {cores} cores",
        *quotes(README, value, ".1f", name=f"sub_core {cores}"),
    )
if len(sub_core) >= 3 and "argmax_keepdim_true" in ARGMAX_A:
    check(
        "fewer cores really is monotonically worse, as the ledger says",
        all(sub_core[a] > sub_core[b] for a, b in zip(sorted(sub_core), sorted(sub_core)[1:]))
        and min(sub_core.values()) > ARGMAX_A["argmax_keepdim_true"]["ms"] * 1000.0,
        str(sorted(sub_core.items())),
    )

check(
    "the sliced argmax returns the same token as the full-height one",
    ARGMAX["rm_slice1_then_argmax"]["first4"][0] == ARGMAX["argmax_keepdim_true"]["first4"][0],
    f"{ARGMAX['rm_slice1_then_argmax']['first4'][0]} vs {ARGMAX['argmax_keepdim_true']['first4'][0]}",
)

# ---------------------------------------------------------------------------
# 9. the runtime fallback audit, for the measured path
# ---------------------------------------------------------------------------

audit = AUDIT["audit"]
check("the audit was taken at 48 layers", audit["num_layers"] == "48", audit["num_layers"])
check(
    "logits never reach the host on the token-out path",
    audit["host_logit_readback_on_token_out_path"] == "False" and audit["host_argmax_on_token_out_path"] == "False",
    f"{audit['host_logit_readback_on_token_out_path']} / {audit['host_argmax_on_token_out_path']}",
)
check("README states both host-boundary fields", "host_logit_readback_on_token_out_path" in README, "")
# ``expected.split(".")[-1].lower() in README.lower()`` was far too weak: for
# ``Topology.Ring`` it was an unanchored, case-folded search for "ring", which
# matches 19 places in this README including "gathering" and "docstring", and
# for ``kv_cache_paged`` it was a search for "true". Two things fix it: the
# string the README must carry is **derived from the audit value** (so mutating
# the artifact changes what is required), and it is looked for only inside the
# README's own runtime-fallback-audit section rather than anywhere in 55 KB.
AUDIT_SECTION = README.split("## Runtime fallback audit", 1)[-1].split("\n## ", 1)[0]
check(
    "the README has a runtime-fallback-audit section to state those fields in",
    len(AUDIT_SECTION) > 500 and AUDIT_SECTION != README,
    f"{len(AUDIT_SECTION)} characters",
)


def audit_rendering(field: str, value: str) -> str:
    """How the README must render an audit value. A function of the value only."""
    if field == "kv_cache_paged":
        return "paged" if value == "True" else "not paged"
    return f"`{value}`"


for field, expected in (
    ("kv_cache_dtype", "bfloat16"),
    ("kv_cache_paged", "True"),
    ("collective_topology", "Topology.Ring"),
    ("lm_head_weight_dtype", "DataType.BFLOAT8_B"),
):
    check(f"the audit still reports {field} = {expected}", audit[field] == expected, f"{audit[field]}")
    rendered = audit_rendering(field, audit[field])
    check(
        f"the README's audit section states the audited {field}",
        rendered in AUDIT_SECTION,
        f"the audit says {audit[field]!r}, so the section must carry {rendered!r}",
    )
check(
    "the audit's sampling line describes the distributed reduction, not the old gather",
    "distributed" in audit["sampling_greedy"] and "all-gather 4 candidates" in audit["sampling_greedy"],
    audit["sampling_greedy"],
)

path = AUDIT["stage06_measured_path"]
check(
    "the measured path really takes the distributed argmax",
    path["sampler_distributed_argmax_taken"] is True,
    "",
)
check(
    "the sampler reduces exactly the model's batch, which is what the live-row lever is",
    path["sampler_dist_active_rows"] == 1,
    str(path["sampler_dist_active_rows"]),
)
check(
    "the sampler's per-die vocabulary matches the LM head's",
    path["sampler_dist_local_vocab"] == local_vocab,
    f"{path['sampler_dist_local_vocab']} vs {local_vocab}",
)
check(
    "README quotes the k_chunk_size the shipped model actually uses, in the audit paragraph",
    # ``appears(README, "256")`` was satisfied by ``SHA-256`` and ``sha256`` in
    # the manifest and provenance paragraphs, and by the position range
    # "128 to 256" -- eight hits, most of them nothing to do with SDPA.
    *quotes(
        README,
        path["sdpa_decode_k_chunk_used"],
        "d",
        name="k_chunk",
        context="`k_chunk_size` **{}** (unclamped",
    ),
)
check(
    "the depth clamp does not bind at the shipped cache depth, as the README says",
    path["sdpa_decode_k_chunk_clamped"] is False
    and path["sdpa_decode_k_chunk_used"] == path["sdpa_decode_k_chunk_tuned"],
    f"used {path['sdpa_decode_k_chunk_used']} tuned {path['sdpa_decode_k_chunk_tuned']} "
    f"depth {path['sdpa_decode_cache_depth_per_user']}",
)
check(
    "README quotes max_cores_per_head_batch in the audit paragraph",
    # Same collider as the composite gather's row count: ``16`` is inside
    # ``bfloat16``. The ledger row states it as ``max_cores_per_head_batch=16``
    # inside a constructor call; the audit paragraph states it as a measured
    # value, which is what this assertion is about.
    *quotes(
        README,
        path["sdpa_decode_max_cores_per_head_batch"],
        "d",
        name="max cores",
        context="`max_cores_per_head_batch` {},",
    ),
)
check(
    "prefill is still at the op default, which is the limitation the README names",
    path["sdpa_prefill_program_config_passed"] == "None",
    path["sdpa_prefill_program_config_passed"],
)
check(
    "the program configs really are memoised -- a handful of entries, not 48 per token",
    path["sdpa_decode_config_cache_entries"] <= 4,
    str(path["sdpa_decode_config_cache_entries"]),
)
steady = AUDIT["steady_state_two_tokens"]
check(
    "two steady-state tokens move only the replay counter",
    steady["only_replays_moved"] is True,
    str(steady["counters_that_moved"]),
)
check(
    "the replay counter moved by exactly the two tokens that were run",
    steady["counters_after"]["replays"] - steady["counters_before"]["replays"] == 2,
    str(steady["counters_that_moved"]),
)
check(
    "no token was copied from the host across those tokens",
    steady["counters_after"]["token_host_copies"] == steady["counters_before"]["token_host_copies"],
    "",
)

# ---------------------------------------------------------------------------
# 10. capacity and the context contract
# ---------------------------------------------------------------------------

stages = FOOTPRINT["stages_gb_per_die"]
total = FOOTPRINT["total_gb_per_die"]
check(
    "the footprint rows sum to the total exactly, with no residual",
    sum(stages.values()) == total,
    f"{sum(stages.values())!r} vs {total!r}",
)
for label, value in (
    ("weights", stages["weights_embed_lm_head_rope"]),
    ("kv cache", stages["kv_cache"]),
    ("traces", stages["traces_and_persistent_buffers"]),
    ("total", total),
    ("headroom", FOOTPRINT["headroom_gb_per_die"]),
):
    check(
        f"README or contract quotes footprint {label}", *quotes(DOCS + json.dumps(CONTRACT), value, ".3f", name=label)
    )
check(
    "the footprint was measured at the advertised context",
    FOOTPRINT["context"] == CONTRACT["hf_advertised_context"] == 262144,
    f"{FOOTPRINT['context']} vs {CONTRACT['hf_advertised_context']}",
)
check(
    "the model fits with room to spare at the advertised context",
    FOOTPRINT["headroom_gb_per_die"] > total,
    f"{FOOTPRINT['headroom_gb_per_die']} free vs {total} used",
)
check("no capability reduction is claimed", CONTRACT["capability_reduction"] is False, "")
check(
    "advertised and supported context still agree",
    CONTRACT["current_supported_context"] == CONTRACT["hf_advertised_context"] == 262144,
    "",
)
contract06 = CONTRACT["stage06_performance"]
check(
    "the contract's token-out matches the shipped artifact",
    contract06["decode_token_out_ms"] == round(SHIPPED["token_out_ms"], 3),
    f"{contract06['decode_token_out_ms']} vs {round(SHIPPED['token_out_ms'], 3)}",
)
check(
    "the contract's t/s/u matches the shipped artifact",
    contract06["decode_token_out_tps_user"] == round(SHIPPED["token_out_tps_user"], 3),
    f"{contract06['decode_token_out_tps_user']} vs {round(SHIPPED['token_out_tps_user'], 3)}",
)
check(
    "the contract's TTFT matches the shipped artifact",
    contract06["ttft_ms_warmed"] == round(SHIPPED["ttft_ms"], 3),
    f"{contract06['ttft_ms_warmed']} vs {round(SHIPPED['ttft_ms'], 3)}",
)
check(
    "the contract's layer-stack bound is the recomputed one, not the superseded wall figure",
    contract06["layer_stack_lower_bound_ms"] == round(DECODE["regions_us"]["layer_stack"] / 1000.0, 3),
    f"{contract06['layer_stack_lower_bound_ms']} vs {round(DECODE['regions_us']['layer_stack'] / 1000.0, 3)}",
)
for prompt in (128, 1024, 4096):
    check(
        f"the contract records the shipped decode cost at prompt {prompt}",
        CONTRACT["stage06_context_flatness"]["token_out_ms"][str(prompt)]
        == round(PERF[(prompt, "argmaxrows")]["token_out_ms"], 4),
        f"{CONTRACT['stage06_context_flatness']['token_out_ms'][str(prompt)]}",
    )
check(
    "the contract's flatness ratio is the recomputed one",
    CONTRACT["stage06_context_flatness"]["ratio_4096_over_128"]
    == round(PERF[(4096, "argmaxrows")]["token_out_ms"] / PERF[(128, "argmaxrows")]["token_out_ms"], 4),
    str(CONTRACT["stage06_context_flatness"]["ratio_4096_over_128"]),
)
check(
    "the contract says how deep decode was actually measured, and it is not the advertised context",
    CONTRACT["stage06_context_flatness"]["measured_to_context_tokens"] == 4096
    and CONTRACT["stage06_context_flatness"]["measured_to_context_tokens"] < CONTRACT["hf_advertised_context"],
    str(CONTRACT["stage06_context_flatness"]["measured_to_context_tokens"]),
)
check(
    "that measured depth is one the perf sweep actually reached",
    (CONTRACT["stage06_context_flatness"]["measured_to_context_tokens"], "argmaxrows") in PERF,
    "",
)
deep = [row["cur_pos"] for row in SDPA_SWEEP]
check(
    "the contract's op-level evidence for greater depths names a position the sweep reached",
    CONTRACT["stage06_context_flatness"]["op_level_evidence_to_cur_pos"] in deep,
    f"{CONTRACT['stage06_context_flatness']['op_level_evidence_to_cur_pos']} not in {sorted(set(deep))}",
)

# ---------------------------------------------------------------------------
# 11. the three upstream bugs, and the reproducers they claim
# ---------------------------------------------------------------------------

model_source = (MODEL_DIR / "tt" / "model.py").read_text(encoding="utf-8", errors="ignore")

for name, needle, reproducer in (
    (
        "all_gather_async Linear + num_workers_per_link=1",
        "minimal_default_writer.cpp",
        STAGE05 / "probes" / "ccl_watcher_ab.py",
    ),
    (
        # The caller bug is in shared code outside this model, so what is
        # checkable here is that the README names the file and the exact lines,
        # and that the same op-level reproducer covers it.
        "sampling_1d.py steering sub-T3K callers into it",
        "sampling_1d.py:294-346",
        STAGE05 / "probes" / "ccl_watcher_ab.py",
    ),
    (
        "argmax sub_core_grids uint32 underflow",
        "argmax_multi_core_program_factory.cpp",
        PROBES / "argmax_outer_dim_probe.py",
    ),
):
    check(f"README records the {name} bug", needle in README, needle)
    check(f"the reproducer for {name} exists on disk", reproducer.is_file(), str(reproducer))
check(
    "README records the underflow's actual arithmetic, not just its name",
    "red_dim_units_last1" in README and "4294966976" in README,
    "",
)
check(
    "the underflow figure in the README is the one uint32 wrap of the stated subtraction",
    str((608 - 928) % (1 << 32)) in README,
    f"{(608 - 928) % (1 << 32)}",
)
check(
    "the argmax probe carries the guard the README says documents it",
    "sub_core_grids" in (PROBES / "argmax_outer_dim_probe.py").read_text(encoding="utf-8", errors="ignore"),
    "",
)
check(
    "the model does not itself pass sub_core_grids, as the README claims",
    "sub_core_grids" not in (MODEL_DIR / "tt" / "model.py").read_text(encoding="utf-8", errors="ignore"),
    "",
)
# The model discusses the bug at length, so a bare substring search would pass
# on the prose. What must be absent is the *argument*.
# The file discusses the bug at length, so the prose mentions the argument
# repeatedly. Strip every backticked span -- which is all of the prose's
# mentions -- and what is left is code.
model_code = re.sub(r"``.*?``|`[^`]*`", "", model_source, flags=re.S)
# ...and drop comment lines, which also discuss it.
model_code = "\n".join(line for line in model_code.splitlines() if not line.lstrip().startswith("#"))
offender = re.search(r".*num_workers_per_link\s*=\s*\d.*", model_code)
check(
    "the shipped sampler still pins no CCL worker count, which is the local workaround",
    offender is None,
    offender.group(0) if offender else "",
)

# ---------------------------------------------------------------------------
# 12. the artifacts the documents name must exist, and the superseded ones must
#     be distinguishable from the shipped ones
# ---------------------------------------------------------------------------

for name in (
    "ops_perf_full_model_48layer_decode.csv.gz",
    "tt_perf_report_full_model_48layer_decode.txt.gz",
    "rank_full_model_48layer_decode.txt",
    "ops_perf_full_model_48layer_decode_part1_preadoption.csv.gz",
    "tt_perf_report_full_model_48layer_decode_part1_preadoption.txt.gz",
    "rank_full_model_48layer_decode_part1_preadoption.txt",
    "ops_perf_full_model_48layer_prefill_s128.csv.gz",
    "tt_perf_report_full_model_48layer_prefill_s128.txt.gz",
    "profile_48layer_work_log.md",
):
    check(f"the artifact {name} exists", (DOC / name).is_file(), str(DOC / name))
    check(f"a document names the artifact {name}", name in DOCS, name)

check(
    "the superseded ranking really does carry different figures from the shipped one",
    f"{DECODE['per_layer_us']:.3f}" not in part1,
    "the two rankings agree on per-layer -- is the superseded file actually superseded?",
)
part1_log = log_text(DOC / "profile_48layer_work_log.md")
check(
    "the pass-1 work log points at the superseded artifacts, not the shipped ones",
    "rank_full_model_48layer_decode_part1_preadoption.txt" in part1_log
    and "rank_full_model_48layer_decode.txt`" not in part1_log.replace("_part1_preadoption.txt", ""),
    "",
)
check(
    "the pass-1 work log warns the reader that its figures are superseded",
    "no\n> longer ships" in part1_log or "no longer ships" in part1_log,
    "",
)
check(
    "the MoE analysis defaults to the shipped profile, so re-running it re-derives rather than replays",
    "ops_perf_full_model_48layer_decode.csv.gz" in (PROBES / "moe_skew_analysis.py").read_text(encoding="utf-8"),
    "",
)

# ---------------------------------------------------------------------------
# 12b. the naming rule: a file whose name reads as "the result" must hold the
#      shipped result, and the docstrings that cite it must quote it
# ---------------------------------------------------------------------------

CANONICAL_PERF = load(PROBES / "perf_full_model.json")
PART1_PERF = load(PROBES / "perf_full_model_part1_preadoption.json")

check(
    "the unsuffixed perf artifact is the SHIPPED measurement",
    abs(CANONICAL_PERF["token_out_ms"] - SHIPPED["token_out_ms"]) < 1e-9,
    f"perf_full_model.json token_out {CANONICAL_PERF['token_out_ms']}, shipped {SHIPPED['token_out_ms']}",
)
check(
    "...and matches it on every published row, not only token-out",
    all(
        abs(CANONICAL_PERF[field] - SHIPPED[field]) < 1e-9
        for field in ("ttft_ms", "model_trace_ms", "sampler_split_ms", "sampler_force_argmax_ms")
    ),
    "the unsuffixed file agrees on token-out but not on the rest -- it is not the same run",
)
check(
    "the superseded part-1 measurement is kept, under a name that says it is superseded",
    (PROBES / "perf_full_model_part1_preadoption.csv").is_file()
    and (PROBES / "perf_full_model_part1_preadoption.json").is_file(),
    "the _part1_preadoption perf artifact is missing",
)
check(
    "the part-1 perf artifact really is distinguishable from the shipped one",
    abs(PART1_PERF["token_out_ms"] - SHIPPED["token_out_ms"]) > 1.0,
    "the part-1 file now matches the shipped figure -- is it actually the part-1 one?",
)
check(
    "a document names the part-1 perf artifact under its suffixed name",
    "perf_full_model_part1_preadoption" in DOCS,
    "",
)
check(
    "no document still calls the unsuffixed perf artifact the part-1 measurement",
    "unsuffixed, are the part-1 measurement" not in DOCS,
    "the superseded disclosure survived the rename",
)

# The two docstrings in tt/model.py that quote figures. Every number below is
# formatted from an artifact here, so these fail when the prose drifts.
check(
    "the sample_greedy_argmax docstring quotes the shipped greedy sampler time",
    *quotes(model_source, SHIPPED["sampler_force_argmax_ms"], ".3f", name="greedy sampler ms"),
)
check(
    "...and the shipped split sampler time it is compared against",
    *quotes(model_source, SHIPPED["sampler_split_ms"], ".3f", name="split sampler ms"),
)
check(
    "...and the ratio of the two, recomputed from them",
    *ratio_is_quoted(
        model_source,
        SHIPPED["sampler_split_ms"],
        SHIPPED["sampler_force_argmax_ms"],
        ".1f",
        name="greedy speed-up",
    ),
)
check(
    "...and the shipped whole-model token-out on that same run",
    *quotes(model_source, SHIPPED["token_out_ms"], ".3f", name="shipped token_out ms"),
)
check(
    "...and its rate",
    *quotes(model_source, SHIPPED["token_out_tps_user"], ".2f", name="shipped token_out t/s/u"),
)
check(
    "the superseded part-1 sampler figure is gone from that docstring",
    "0.901 ms against 6.155 ms" not in model_source,
    "the docstring still quotes the part-1 sampler time as the shipped one",
)
check(
    "the docstring attributes the distributed argmax to the part-1 artifact's own token-out",
    *quotes(model_source, PART1_PERF["token_out_ms"], ".3f", name="part-1 token_out ms"),
)
check(
    "...against the stage-05 baseline it is measured from",
    *quotes(model_source, BASELINE["token_out_ms"], ".3f", name="stage-05 token_out ms"),
)
check(
    "...and that pair really is like-for-like on allocated context, as the docstring claims",
    PART1_PERF["context"] == BASELINE["context"],
    f"part-1 context {PART1_PERF['context']}, stage-05 context {BASELINE['context']}",
)
check(
    "the docstring attributes the live-row slice to the paged-SDPA leg's own token-out",
    *quotes(model_source, PERF[(128, "after")]["token_out_ms"], ".3f", name="p128_after token_out ms"),
)
check(
    "...and that pair is like-for-like on allocated context too",
    PERF[(128, "after")]["context"] == SHIPPED["context"],
    f"after context {PERF[(128, 'after')]['context']}, shipped context {SHIPPED['context']}",
)

# The other docstring: the baseline sampler path, which stage 05 priced in a
# 2-layer window. What is checkable is that the docstring keeps that window's
# figures inside that window's accounting, and prices the terminal block from
# the 48-layer profile instead.
stage05_report = log_text(STAGE05 / "tt_perf_report_full_model_decode.txt")
gather_row = re.search(r"(\d+\.\d+) %\s+AllGatherAsyncDeviceOperation\s+\d+\s+889 [^\s]+s", stage05_report)
argmax_row = re.search(r"(\d+\.\d+) %\s+ArgMaxDeviceOperation\s+\d+\s+859 [^\s]+s", stage05_report)
check(
    "the 889 us / 859 us rows the _sample_argmax docstring cites are really in the stage-05 report",
    gather_row is not None and argmax_row is not None,
    "one of the two rows is not in doc/full_model/tt_perf_report_full_model_decode.txt",
)
check(
    "the docstring quotes those rows' shares of that 2-layer window, read from the report",
    gather_row is not None
    and argmax_row is not None
    and appears(model_source, gather_row.group(1))
    and appears(model_source, argmax_row.group(1)),
    f"report says {gather_row.group(1)}% / {argmax_row.group(1)}%" if gather_row and argmax_row else "unparsed",
)
check(
    "the docstring names that window as 2-layer, which is why those shares do not scale",
    "2-layer" in model_source,
    "",
)
check(
    "the docstring withdraws the old cross-accounting claim rather than repeating it",
    "the claim is withdrawn" in model_source and "essentially all of the" not in model_source,
    "the 889+859 sum is set against a token-out step again",
)
check(
    "the docstring prices the terminal block from the 48-layer profile instead",
    *quotes(model_source, DECODE["regions_us"]["terminal_post"], ".1f", name="terminal_post us"),
)
check(
    "...against the whole iteration from the same file",
    *quotes(model_source, DECODE["iteration_us"], ".1f", name="iteration us"),
)
check(
    "...as a share of it",
    *quotes(model_source, DECODE["regions_percent"]["terminal_post"], ".2f", name="terminal_post %"),
)
check(
    "...and names the sampler's own part of that block",
    *quotes(model_source, DECODE["sampler_us"], ".1f", name="sampler us"),
)

# ---------------------------------------------------------------------------
# 13. this file's own accounting
# ---------------------------------------------------------------------------

# The README quotes the size of the mutation set too, and that number drifted
# silently once. Count the mutations the tester actually defines.
sys.path.insert(0, str(PROBES))
try:
    from mutation_test_checker import MUTATIONS as _MUTATIONS

    mutation_count: int | None = len(_MUTATIONS)
except Exception:  # pragma: no cover - the mutation tester is next to this file
    mutation_count = None
check("the mutation tester is importable, so the size of its set can be counted", mutation_count is not None, "")
if mutation_count is not None:
    # ``appears(README, str(mutation_count))`` is an unanchored search for the
    # bare integer anywhere in a 55 KB file, and the round-2 review caught what
    # that costs: the README said "248 mutations" in one place and "a specific,
    # listed set of 212 perturbations" in the sentence that states the stage's
    # central QA claim, and the check stayed green because the *other* site
    # satisfied it. Every site that states the count is now asserted where it
    # stands, in the phrase that carries it.
    # Matched against the README with its line wrapping flattened, so the
    # assertion is anchored to the sentence and not to where the paragraph
    # happens to break.
    for wording in (
        f"applies **{mutation_count} mutations** one at a time",
        f"failability under a specific, listed set of {mutation_count} perturbations",
        f'the honest reading of a green run is "this set of {mutation_count} corruptions is detected"',
        f"— {mutation_count} mutations, every assertion broken by at least one targeted mutation",
    ):
        check(
            f"README states the mutation count where it says {wording[:56]!r}",
            phrase(README, wording),
            f"the tester defines {mutation_count}; the README does not carry that number in this phrase",
        )

# ...and the archived mutation run, which the README quotes but nothing checked.
mutation_log = log_text(LOGS / "mutation_test_checker.log")
check("the mutation test's own run is archived", bool(mutation_log), "logs/mutation_test_checker.log")
mutation_run = re.search(r"(\d+) assertions; (\d+) were made to fail", mutation_log)
check("the archived mutation run reports its two tallies", mutation_run is not None, "")
archived_assertions = int(mutation_run.group(1)) if mutation_run else -1
archived_broken = int(mutation_run.group(2)) if mutation_run else -1
targeted = re.search(r"(\d+) of those were made to fail by a mutation that is not one of the", mutation_log)
archived_targeted = int(targeted.group(1)) if targeted else -1
check(
    "the archived mutation run broke every assertion it found, as the README says",
    archived_assertions == archived_broken
    and "every assertion was made to fail by at least one mutation" in mutation_log,
    f"{archived_broken} of {archived_assertions}",
)
check(
    "the archived mutation run also reports how many of those a *targeted* mutation broke",
    targeted is not None,
    "the archived run predates the shotgun-only reporting",
)
check(
    "no assertion in that run was covered only by a document-wide shotgun mutation",
    archived_targeted == archived_assertions and "failed ONLY under a shotgun mutation" not in mutation_log,
    f"{archived_targeted} of {archived_assertions} have a targeted mutation",
)
# The four shotguns' *measured* breadth. The README used to say they "trip 200+
# assertions at once" as a constant; two of the four are far narrower. The
# tester measures it now and the document quotes what it measured, tied
# together here. The values move as assertions are added -- they have read
# 29/34/206/230, then 39/44/236/260, and now what the archived run says -- so
# nothing below hardcodes them; they are parsed from the log.
#
# The two *authoritative* sites are anchored to their own sentence. `work_log`
# mentions the quadruple a third time as narrative and is deliberately NOT
# checked: gating it would add a fourth self-referential figure to a fixpoint
# that took several passes to close. Stage 07 should drop it from work_log.
# An unanchored `appears(README, value)` is satisfied by whichever site is still
# correct while the other rots -- which is exactly what happened: limitation 10
# sat at "235 and 259" through a green 556/556 run because the measured-breadth
# paragraph one screen up still said 236 and 260.
shotgun_breadths = re.search(
    r"measured breadth of the \d+ declared shotguns: (\d+)-(\d+) assertions \(([^)]*)\)", mutation_log
)
check("the archived mutation run reports the measured breadth of its shotguns", shotgun_breadths is not None, "")
if shotgun_breadths:
    # The log lists the four by mutation name (``sorted(SHOTGUN)``, which is not
    # the order they run in); the documents quote them ascending by value, so
    # the parsed set is sorted here rather than taken in the order printed.
    measured = sorted(int(v) for v in re.findall(r"\b(\d+)\b", shotgun_breadths.group(3)))
    breadth_phrase = ", ".join(str(v) for v in measured[:-1]) + f" and {measured[-1]}"
    # Anchored to the sentence, not searched document-wide. An unanchored
    # `appears()` is satisfied by whichever site is still correct while the
    # other rots -- see the comment above for the instance that caused.
    #
    # TWO sites quote this set, so both are anchored to their own sentence.
    # Anchoring only one of them re-creates the same hole one level down: the
    # unanchored site is then held up by the anchored one. `readme_limitation10_
    # breadths` and `readme_shotgun_section_breadths` in the mutation tester
    # corrupt one site each, and each must break only its own assertion.
    check(
        "README's limitation 10 states the measured shotgun breadths in its own sentence",
        phrase(README, f"the four declared shotguns trip **{breadth_phrase}** assertions"),
        f"the run measured {breadth_phrase}; limitation 10 does not carry that set",
    )
    check(
        "README's shotgun-coverage section states the measured breadths in its own sentence",
        # Attribution-free on purpose: the anchor used to read "the round-2
        # review measured it", but round 2 measured 29/34/206/230 -- the set has
        # moved twice since, so naming a round in the anchor bakes in a claim
        # that goes stale every time an assertion is added.
        phrase(README, f"the shipped run reports **{breadth_phrase}**"),
        f"the run measured {breadth_phrase}; the shotgun-coverage section does not carry that set",
    )
    # There used to be a `for value in measured: appears(README, str(value))`
    # loop here as well. It is gone, and not because it was merely redundant
    # (the two anchored checks above strictly subsume it -- a value present in
    # the sentence is present in the document). It was *unstable*: these four
    # widths are themselves published in the README, so under the `readme_digits`
    # shotgun the document's own 49 shifted onto 50 and satisfied the search for
    # a measured breadth of 50. Which of the four could fail therefore depended
    # on the arithmetic relationship between this run's widths and the previous
    # run's -- and since those widths move whenever the assertion count moves,
    # the tester oscillated between two fixpoints one apart. A self-referential
    # figure cannot be checked by a document-wide search for its own digits.
# The widest mutations that are NOT declared shotguns -- limitation 10 names
# them to make its point that the targeted/shotgun split is a declared list
# rather than a measurement. These had no assertion of any kind.
widest = re.search(r"widest mutations NOT declared shotgun: ([^\n]+)", mutation_log)
check("the archived mutation run reports its widest non-shotgun mutations", widest is not None, "")
if widest:
    for name, value in re.findall(r"(\w+) (\d+)", widest.group(1))[:3]:
        check(
            f"README quotes the measured breadth of the widest non-shotgun mutation {name} ({value})",
            phrase(README, f"`{name}` ({value})"),
            f"the run measured {name} at {value}",
        )
# ...and the gap that measuring breadth exposed: assertions whose narrowest
# coverage is itself broad. This is a named limitation, so the number it names
# has to be the number the run measured.
coarse = re.search(r"(\d+) of (\d+) assertions have NO mutation narrower than (\d+) assertions", mutation_log)
check("the archived mutation run reports how coarse the narrowest coverage gets", coarse is not None, "")
if coarse:
    check(
        "README's named limitation quotes that coverage gap as the run measured it",
        phrase(
            README,
            f"**{coarse.group(1)} of {coarse.group(2)}** assertions have no mutation narrower "
            f"than {coarse.group(3)}",
        ),
        f"the run measured {coarse.group(1)} of {coarse.group(2)} at width {coarse.group(3)}",
    )
check(
    "no mutation in that run broke nothing -- every one of them is evidence about something",
    "BROKE NOTHING" not in mutation_log and "could not be applied at all" not in mutation_log,
    "the archived run contains a mutation with no effect",
)
check(
    # ``appears(README, "501")`` on its own is satisfied by any of the several
    # other places the README states that count, so the sentence that makes the
    # claim has to carry both tallies.
    "README's assertion count is the one that archived run measured, in the sentence that says so",
    f"**{archived_assertions} assertions, {archived_broken} made to" in README,
    f"the archived run counted {archived_broken} of {archived_assertions}",
)

claimed = re.search(r"re-derives (?:all )?(\d+) figures", README)
check("README states how many figures this checker re-derives", claimed is not None, "")
# ``appears(README, ...)`` above is a bare-integer search in a 55 KB document and
# is far too weak on its own -- the stage-06 review said so. The direct
# assertion is that the archived run counted *exactly* the assertions this run
# makes. All three of the checks below are stated against the total this run
# will finish on, which is the count so far plus the three of them.
run_total = checks + 3
# Nothing inspected the archived log's *provenance*. ``--bootstrap`` runs the
# tester against a clean tree that is not green -- it permits exactly the
# self-referential count checks below to fail so that a first log can exist
# after the assertion count moves -- and every measured breadth in such a run is
# inflated by up to that many assertions. The tester says so in a banner, and
# the banner was decoration: prepending it, plus a ``clean tree: N checks, 2
# failing`` line, to ``logs/mutation_test_checker.log`` passed every assertion
# above. So the archive has to name a clean tree that was green and that had
# exactly this run's assertions in it, and must carry no trace of the bootstrap
# path. This is the third self-referential check and it settles with the other
# two.
check(
    "the archived mutation run is a normal run over a green clean tree, not a --bootstrap stepping stone",
    f"clean tree: {run_total} checks, 0 failing" in mutation_log
    and "--bootstrap" not in mutation_log
    and "PERMITTED BY" not in mutation_log
    and "THIS LOG IS NOT THE ARCHIVE" not in mutation_log,
    f"the archive must open with 'clean tree: {run_total} checks, 0 failing' and carry no bootstrap banner",
)
check(
    "the archived mutation run covered exactly the assertions this run makes",
    archived_assertions == run_total,
    f"the archived run counted {archived_assertions}, this run makes {run_total}",
)
check(
    f"README's figure count matches what this run reports ({run_total})",
    claimed is not None and int(claimed.group(1)) == run_total,
    f"README says {claimed.group(1) if claimed else '(nothing)'}, this run reports {run_total}",
)

print()
if vacuous:
    # Not an assertion failure -- an assertion that cannot distinguish what it
    # found. It is reported separately and it is fatal, because the alternative
    # is a fourth round of someone noticing by hand.
    print(f"{len(vacuous)} VACUOUS whole-document searches -- each one is satisfied by any text that")
    print("happens to contain the same characters, so it is not a check. Anchor it or declare it:")
    for entry in vacuous:
        print(f"  - {entry}")
if failures:
    print(f"{len(failures)} of {checks} checks FAILED:")
    for failure in failures:
        print(f"  - {failure}")
if failures or vacuous:
    sys.exit(1)
print(f"all {checks} published figures match their artifacts")
