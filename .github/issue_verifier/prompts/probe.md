# Issue verification — measurement pass

You are running an experiment that has already been designed. Your only product
is **measurements**. A deterministic rule downstream reads them and assigns the
verdict, so you do not need to reach a conclusion and must not write one.

Working directory is a tt-metal checkout. The plan below came from a previous
stage that read the issue report.

```json
{{plan}}
```

Hardware available to you: **{{sku}}** ({{hardware_note}})

## Ground rules

Every number you emit must come from a process you actually ran. You have a
shell; use it. A value you derived by reading code and reasoning about it is not
a measurement, and recording one as though it were is the single worst thing you
can do here — it reproduces the exact defect this tool was built to detect.

If something cannot be run, record it as not-run with a reason. Gaps are fine.
Invented numbers are not.

## Gate A — re-execute the reference (always)

Write `/tmp/probe_a.py`. Define `reference(case)` from the plan's
`reference_snippet` verbatim, run it over every case, and print one JSON object.

For each case compare `reference(case)` against the report's `claimed_expected`:

- Both non-finite and equal (`inf` vs `inf`, `nan` vs `nan`) → agree.
- One finite and the other not → **disagree**. This is the high-signal outcome:
  it means the reporter's reference column was never produced by the reference.
- Both finite → agree when `math.isclose(rel_tol=1e-6)`.

Run it with `python3`. If it raises, fix the harness — not the semantics — and
retry up to three times. If it still fails, record `ran: false` with the
traceback and move on.

## Gate B — has this behavior already been decided? (always)

The current behavior may be the *result* of an earlier deliberate fix, which
makes "this is wrong" a request to revert someone's considered decision. For
each path in `cited_files`:

- `git log --oneline -15 -- <path>`
- `git log -S "<the exact expression the report quotes>" --oneline -5 -- <path>`
- For the line the report blames: `git log -L <line>,<line>:<path> --oneline -3`

Record each commit you find that touches the blamed behavior, and set
`deliberate` to `true` **only** when its subject or body shows the present
ordering, sign, associativity, or rounding was chosen on purpose — a message
containing "fix", "match torch", "order", or "associat" that is actually about
this behavior. A commit that merely introduced or renamed the code is
`deliberate: false`; still record it, since "nobody ever decided this" is itself
useful context. Quote the commit subject; do not paraphrase.

Finding "no relevant commits" is a valid result — emit an empty list rather than
a placeholder entry with null fields.

## Gate C — run the real op (only when a device is present)

Skip with a reason when the SKU is `github_hosted_cpu`.

Otherwise write `/tmp/probe_c.py` using the plan's `device_snippet`. Open the
device once, run all cases, close it, print JSON. Compare each device value
against **`reference(case)`** — the value the reference actually returns, never
the report's `claimed_expected`. Use the same comparison rules as Gate A.

Before trusting a failure here, confirm the op ran at all: a raised exception is
a Gate C error, not a mismatch.

## Gate D — would the proposed change hold up? (only when one is proposed)

Skip with a reason when `counterfactual_snippet` is null.

Otherwise evaluate `counterfactual(case)` against `reference(case)` over:

1. Every case in the plan.
2. **Mirror cases you construct yourself.** This is the part that needs your
   judgment. A reordering that rescues one overflow window almost always opens a
   symmetric one — if the change moves a multiply after a divide, the exposed
   direction flips from "large numerator" to "small denominator". Build at least
   four inputs that stress the *opposite* extreme from the report's, and check
   whether the counterfactual disagrees with the reference where current
   behavior agrees. Say in `mirror_rationale` what extreme you targeted and why.

## Output

Last thing you write must be exactly one fenced `json` block, no prose after it:

```json
{
  "gate_a": {"ran": true, "error": null,
    "rows": [{"name": "...", "claimed_expected": 1.5e38, "reference": Infinity, "agree": false}]},
  "gate_b": {"ran": true, "error": null,
    "findings": [{"path": "...", "commit": "abc1234", "subject": "...",
                  "deliberate": true, "why_relevant": "..."}]},
  "gate_c": {"ran": false, "skipped_reason": "...", "error": null, "rows": []},
  "gate_d": {"ran": true, "skipped_reason": null, "error": null,
    "rows": [], "mirror_rows": [], "mirror_rationale": "..."},
  "notes": "anything a reviewer needs that the rows do not carry"
}
```

Emit `Infinity`, `-Infinity`, and `NaN` as bare JSON tokens; the reader accepts
them. Every row must trace to a command you ran in this session.
