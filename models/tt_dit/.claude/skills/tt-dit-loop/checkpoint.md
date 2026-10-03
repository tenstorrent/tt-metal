# `CAMPAIGN.md` and `lineage.jsonl`

Two files carry loop state. Neither ever grows without bound.

| File | Read | Written | Bound |
|---|---|---|---|
| `CAMPAIGN.md` | **Always**, every round, whole | **Regenerated** each round | Hard cap 200 lines |
| `lineage.jsonl` | Never whole — `jq` it | Appended, one object per round | Unbounded, but O(1) to append and query |

This split is the whole design. A single append-only markdown file cannot be
both the permanent record and the per-round context load: the H3 journal reached
4972 lines / 111 amendments and had to be compacted to stay readable, destroying
the forensic history the retraction protocol depends on. Keep prose out of the
machine state and the problem does not recur.

## `lineage.jsonl`

One object per completed round. Append only.

```jsonl
{"round":7,"ts":"2026-08-05T14:22:31Z","sha":"a1b2c3d","scope":"vae-decode","hypothesis":"fold LayerScale into out projection","metric_ms":4612.0,"delta_pct":-1.4,"quality":{"pcc":0.99942,"gate":"pass"},"bound_class":"overhead","verdict":"kept","artifacts":"artifacts/round-7/"}
```

| Field | Meaning |
|---|---|
| `round`, `ts`, `sha` | Identity. `sha` is the commit the round produced |
| `scope` | Component under work |
| `hypothesis` | One line, the same text as the commit subject |
| `metric_ms` | Warm device time at the production shape |
| `delta_pct` | Against `best`, negative is faster |
| `quality` | The gate result — a round without one is invalid |
| `bound_class` | From `../tt-dit-benchmark-profile/reading-profiles.md` |
| `verdict` | `kept` · `forensic` · `revert` · `abort` |
| `artifacts` | Path to that round's evidence |

Everything the loop needs is reconstructable from this without reading prose:

```bash
jq -s 'min_by(.metric_ms) | {round,sha,metric_ms}' lineage.jsonl        # current best
jq -s '[.[-5:][] | select(.delta_pct > -2)] | length' lineage.jsonl     # stall count
jq -r 'select(.verdict=="kept") | "\(.round) \(.delta_pct)% \(.hypothesis)"' lineage.jsonl
```

## `CAMPAIGN.md`

Regenerated each round from the lineage plus the narrative sections. **Never
appended to** — that is how the last one grew to 300KB.

```markdown
# <Model> — <mesh> — campaign

Branch `<branch>` · run root `<path>` · started <date>
Full history: `git log --follow -- CAMPAIGN.md ledgers/`

## Loop state
| Round | Best | Baseline | Δ baseline | Stall | Target | Gate status |
|---|---|---|---|---|---|---|
| 7 | 4612 ms @ r5 `a1b2c3d` | 6410 ms | −28.0% | 1/10 | 3000 ms | none fired |

## Working point
<production shape(s), mesh, parallel config, dtype per component>

## Fixed baseline (Phase 1, immutable)
<number · exact command · warm-window method · frozen-output path · SHA>

## Gates
| Gate | Command | Result |
|---|---|---|

## Pending work
1. [#12] <next action — must be unambiguous to a cold reader>
2. [#13] ...

## Pitfalls
<campaign-specific only>

## Latest amendment
<most recent only; the rest live in ledgers/amendments.md>

## Ledger index
attempts.md (rounds 1–7) · optimizations.md (3) · source-ideas.md (11) · amendments.md (4)
```

### Section bounds

Caps below were derived by measuring a real campaign's hot state, not chosen —
see "Where these numbers come from".

| Section | Cap | On overflow |
|---|---|---|
| Header | 5 | — |
| Loop state | 8 | Regenerated from lineage |
| Working point | 30 | — |
| Fixed baseline | 12 | Immutable; never grows |
| Gates | 15 | One row per gate |
| Pending work | 45 | Detail moves to `ledgers/attempts.md`; keep one line per item plus its task id |
| Pitfalls | 25 | **Graduate** general ones to `../shared/known-issues.md`, then delete here |
| Latest amendment | 12 | The one-line finding, the result, and a ledger pointer — **not** the full body |
| Ledger index | 3 | — |

**Total cap: 200 lines.**

### Where these numbers come from

The H3 campaign's compacted hot state was measured against a first draft of this
table and every section was over:

| Section | Measured | First-draft cap |
|---|---|---|
| Working point | 27 | 10 |
| Pending work | 65 | 20 |
| Pitfalls | 105 | 30 |
| Latest amendment | 58 | 10 |
| **Total** | **255** | **150** |

Two things came out of that. The caps above are the measured sizes rounded up,
except where a rule genuinely shrinks the section: `Pending work` because item
detail belongs in the ledger, and `Latest amendment` because the checkpoint needs
the finding, not the evidence tables.

And `Pitfalls` stays tight because **all six** of that campaign's pitfall
categories — machine/process, measurement scope, metrics that lie, ttnn API
traps, reference-comparison traps, test-harness traps — were general tt_dit
knowledge, not campaign-local. Graduated, they leave almost nothing behind. If
your `Pitfalls` section is near its cap, the likely fault is that graduation is
not happening, not that the cap is too small.

**Graduation is the pressure valve.** A pitfall that is true of tt_dit generally
belongs in `../shared/known-issues.md`, where every skill sees it; one true only
of this campaign stays. When the section is full, graduate before trimming —
that is how campaign knowledge compounds instead of being discarded.

## Recovery pointer

The header's `git log --follow` line is mandatory. It is what makes deletion
unnecessary: any past state is one command away, so the checkpoint never has to
carry history to preserve it.

## Regeneration rule

Loop state comes from `lineage.jsonl`. Narrative sections carry forward from the
previous `CAMPAIGN.md`, minus anything that graduated or moved to a ledger. If
the two ever disagree, **the lineage wins** — it is append-only and cannot have
been silently edited.
