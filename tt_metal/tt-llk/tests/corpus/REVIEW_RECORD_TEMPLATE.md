# REVIEW_RECORD-<cc1plus-12hex> — silicon authorization for one compiler pin

<!--
Template for the sweep preflight's REVIEW RECORD gate (HANDOFF §1(4) as code).

A sweep whose phases include silicon and that passes --allow-hardware REFUSES
unless <evidence-root>/../REVIEW_RECORD-<cc1plus-12hex>.md exists for the
CURRENT pin.  Mechanics (sweep_2x2.py check_review_record):
  * the file name carries the first 12 hex of the pinned cc1plus sha256;
  * the body MUST quote the FULL 64-hex cc1plus sha256 (pin-match — a record
    minted for another build never authorizes this one);
  * the body MUST contain a non-empty "Reviewer:" line, a "## Reviewed
    commits/branches" section, and a "## Gates checked" section.
The gate makes the record's EXISTENCE and pin-binding mechanical; the
record's HONESTY stays on the reviewer.  A record that overstates its review
is worse than a refusal — say exactly what was and was not reviewed,
including whether the review was independent of the authors.  One record per
pin; a re-pin needs a new record.  Copy the checked-in record into the
evidence-root parent (default ~/sfpi-uplift/sweep-2x2/) — the preflight
looks there, beside the dated evidence dirs it authorizes.
-->

Pin: cc1plus sha256 `<full 64-hex cc1plus sha256>`
Built from: sfpi-gcc `<commit>` via sfpi `<commit>` (`scripts/build.sh` / rebuild log path)
Date: `<YYYY-MM-DD>`
Reviewer: `<who reviewed — a person or session identity; state independence from the authors explicitly>`
Independence: `<independent | NOT independent — explain>`

## Reviewed commits/branches

- `<repo> <sha>` — `<one line: what it changes>`
- ...

## Gates checked

- [ ] byte-identity of default codegen vs the previous pin (or: what changed and why that is expected)
- [ ] focused DejaGnu families green; full rvtt.exp FAIL set byte-identical to the frozen environmental set
- [ ] paired CRAQ green on the affected shape classes (which simulator sha)
- [ ] refusals byte-identical where required
- [ ] no hardcoding introduced (op names / calendars / magic words) — how verified
- [ ] known risks / carry-forwards affected by this pin, listed

## Limitations

`<what this review could NOT establish, and any self-review caveat>`
