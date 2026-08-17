# Page Budget — IPDPS 2027 (10 pages double-column, references unlimited)

Total: 20 columns of body (references excluded from the 10-page count).
Figures/tables are charged to the section that owns them.

| # | Section | Brief | Columns | Pages | Owns (figures/tables, charged here) |
|---|---|---|---:|---:|---|
| — | Title + authors block | — | 0.3 | 0.15 | (inside intro's allocation) |
| 1 | Abstract + Introduction | `briefs/01-intro.md` | 2.5 | 1.25 | none (title block absorbed here) |
| 2 | Background | `briefs/02-background.md` | 2.0 | 1.00 | Fig B1 (Tensix/mesh schematic, 1 col) |
| 3 | Design space — C1 negative result | `briefs/03-design-space.md` | 2.5 | 1.25 | Fig D1 (Q1/Q2/Q3 map), Tab D1 (ingredients), Fig D2 (engine shootout bars) |
| 4 | System — C2 log-tree + C3 chunk-skip | `briefs/04-system.md` | 4.0 | 2.00 | Fig S1 (operator schematic), Fig S2 (skip law), Tab S1 (WarpSelect/GVR/chunk-skip contrast) |
| 5 | Characterization — C4 | `briefs/05-characterization.md` | 2.5 | 1.25 | Tab C1 (characterization table, 1.5–2 col); optional Fig C1 (Dst layout) |
| 6 | Evaluation | `briefs/06-evaluation.md` | 4.5 | 2.25 | Tab E1 (competition, 2-col), Fig E1 (P-scaling), Fig E2 (skip A/B), Tab E2 (small-k before/after), Tab E3 (scenarios) |
| 7 | Related + Methodology + Conclusion | `briefs/07-related.md` | 2.0 | 1.00 | optional Tab M1 (gate ledger) |
| | **Total** | | **20.0** | **10.00** | 4–5 figures + 5–7 tables |

## Figure/table count sanity

Target density: 9–12 exhibits total (the style corpus averages ~20/paper at
longer formats; 10 pages supports ~10–12). Current plan: 5 figures
(B1, D1, D2, S1, S2, E1, E2 = 7 → trim to 5–6) + 6 tables (D1, S1, C1, E1,
E2, E3). First cuts if over: Fig C1 (already optional), Tab M1 (optional),
Fig D1 collapses into Tab D1's header rows, Tab E2 collapses to prose.

## Overflow rules (pre-agreed, apply in order)

1. §2.3 GPU recap is the first prose cut (hard 1-col cap already).
2. Tab E2 (small-k before/after) → three prose numbers (40×–423×, ≈89×
   cliff, 137 ns/elem).
3. Tab M1 (gate ledger) → cut; the methodology prose carries it.
4. §5's histogram subsection compresses to the table row + 2 sentences.
5. NEVER cut: the 1.35 GHz clock caveat, the blaze fairness caveats, the
   scenarios canonical-form caveat, the reopening conditions, the forecast
   no-go story. These are the paper's credibility spine.

## Reference budget

Unlimited pages, but keep to ~45–55 entries (refs.bib currently ~45).
All entries marked UNVERIFIED in refs.bib must be resolved before
camera-ready; the four pre-submission action items live at the bottom of
`paper-topk/evidence/paper/related-work.md`.
