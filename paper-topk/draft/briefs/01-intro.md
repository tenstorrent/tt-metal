# Brief 01 — Introduction (with Abstract)

**Section file:** `sections/00-abstract.tex` + `sections/01-intro.tex`
**Budget:** 2.5 columns (1.25 pages) — abstract ≈0.4 col, intro ≈2.1 cols.

## Single job

Make the reader believe one sentence: *exact top-k on a commercial NoC-mesh
dataflow chip was 10,000× off its achievable floor, and a measured design-space
study — not a clever new algorithm — closed the gap.* Everything else (the four
contributions) hangs off that sentence.

## Abstract (≈300 words, dense — the style guide's 299-word average is a target, not a cap)

Must contain, in order: (1) the problem — LLM decode-time workloads (sampling,
sparse-attention indexers, MoE gates) put exact top-k on the critical path, and
every accelerator study to date either approximates or ignores it; (2) the
platform — Tenstorrent Blackhole p150a, 130-core NoC mesh, no atomics, no
scatter; (3) the four contributions with their anchor numbers; (4) the honest
frame — a design-space study in which the incumbent absorbed the study's own
fixes.

Anchor numbers the abstract must carry (all from `evidence.md` §1.1):

| Number | Value | Evidence |
|---|---|---|
| End-to-end user-facing win | 631.5 ms → 63.4 µs = **9,956×** (`ttnn.topk`, k=2048, N=65,536) | H1, `baselines/comp3/competition_table.csv` row `2048,65536` |
| Direct-op anchor | **34.3 µs @ 32 cores**; 18,401× vs stock | H2, same row |
| 1M-context decode shape | k=512 @ N=262,144 in **23.3 µs @ 104 cores** | H3, same CSV row `512,262144` |
| Cost model | 2·⌈C/P⌉ + ⌈log₂P⌉, validated to 104 cores | C2-1, C2-3 |
| Chunk-skip win | **1.82× (−45.1%)** at rows=2, k=32, N=65,536 | C3-4, `evidence/tileskip/rowskip-implementation.md` §4 |
| Distance to roofline | **9.7–29.9×** across all 24 cells | H4, competition CSV gap columns |

## Hook (choose per the style guide's typology — §"Introduction Hooks")

Recommended type: **#5 Physical Layout and CAD Tool Reality** — the measured
cliff IS the hook. Candidate opening: the production stack's own numbers —
a vocabulary-sized top-k (k=32, N=65,536) that falls off the multi-core path
into a 137 ns/element single-core loop, 9.6 ms per sampled token
(scenarios CSV row `sampling_qwen36_tp4`: 9,596.29 µs), while 129 idle cores
watch. Alternative type: **#1 conventional-wisdom parody** — "GPU folklore
says exact top-k is a solved problem: count, partition, scatter, repeat.
Blackhole has no atomics and no scatter." Pick ONE; obey the Golden Rule —
the hook must hand off to the thesis within 2–3 sentences.

Do NOT use type #3 (biographical anecdote) here — the paper's authority is
measurement density, and the intro must reach numbers fast.

## The "why" paragraph (Patterson standard, style guide Tip #2)

One paragraph connecting the local constraint to the architecture trend:
AI accelerators traded coherent shared memory + atomics for mesh bandwidth
and software-managed SRAM; selection is the canonical data-dependent workload
that this trade punishes; nobody has measured what it actually costs
(no sorting/selection publication exists for WSE, Groq, SambaNova, Occamy,
Esperanto — related-work.md §5 "useful absence").

## Contributions (style guide grammar: noun phrase / gerund openers, NOT "We show")

Exactly four, bulleted once, each with its anchor number:

1. **C1 (negative result):** "Quantification, via single-ingredient silicon
   microbenchmarks, of why GPU radix-select economics do not transfer to a
   NoC-mesh dataflow core: exact SFPU counting is 1 bit per 2.0 cycles/vector,
   decisions cost 81 cycles, and materialization — not counting — is the
   load-bearing gap (dense emit 13.0 cyc/elem vs a 0.5 bar)."
   Evidence: C1-1..C1-7.
2. **C2 (system):** "Design and measurement of a column-parallel log-tree
   top-k operator on the 130-core Blackhole mesh — semaphore-tree rendezvous
   without atomics or global sync — with a validated cost model
   2·⌈C/P⌉+⌈log₂P⌉ and a cost-optimal rectangle embedding; 34.3 µs at k=2048,
   N=65,536 (18,401× over the stock path) and monotone scaling to 104 cores."
   Evidence: C2-1..C2-5. **Position as first MEASURED full-top-k
   (values+indices) on commercial NoC-mesh silicon** — the mesh-selection
   theory (Krizanc & Narayanan '92/'93) is cited in the intro's related-work
   sentence, not dodged.
3. **C3 (mechanism + law):** "Derivation and silicon validation of a
   distribution-free chunk-skip law P(skip) = C(cM,K)/C((c+1)M,K) ≈ e^(−K/(c+1))
   for streaming bitonic cascades, with a soundness proof, a compile-time
   K/4 gate, and a calibrated pre-build forecast that eliminated the losing
   variant; 1.82× measured on the surviving one." Evidence: C3-1..C3-6.
4. **C4 (characterization, NARROWED per related-work.md):** "First public
   characterization of the Tensix compute datapath's numeric semantics and
   sorting-relevant primitive costs: bf16 canonicalization (NaN→Inf, −0→+0,
   subnormal→+0), a native sign-magnitude total order including NaN, a
   1-in-8-sampled and 32-bin-aliased packer exponent histogram, and measured
   synchronization floors." Evidence: C4-1..C4-9. Do NOT claim "first
   Blackhole characterization" — the informal microbench report
   (`blackholemicrobench2025`) exists; the claim is the *datapath-numerics
   axis*.

## Claims owned

- H1, H2, H3 (headline anchors — stated here, defended in §6).
- The four contribution statements (C1/C2/C3/C4 verdict-safe phrasings).
- The "no selection study exists on any 2D-mesh dataflow machine" absence claim
  (related-work.md §1 Cerebras/Groq/SambaNova paragraph + §5 item 5).

## Figures/tables owned

None. (Optionally a 1-column teaser figure of the P-scaling curve or the
competition-table waterfall IF §6's budget survives; default: no intro figure.)

## Style directives

1. Sentences ≈20 words; active voice ≥65%; `×` never `x`; `≈` never `~`.
2. Contributions in noun/gerund grammar (93% corpus rule) — no "We design/We
   show" bullets; redraft table in the style guide is binding.
3. Every performance claim in the intro carries its configuration inline
   ("k=2048, N=65,536, bf16, single p150a") — no naked speedups.
4. No LLM-isms: delve, crucial, landscape, pave the way, testament.
5. Double-anonymous: no repo URLs, no commit hashes, no branch names, no
   `TOPK_LEDGER.html` mentions; artifacts are "the measurement ledger /
   canonical sweep harness (to be released)".

## Hazards

- Do not print the roofline "fits all 11 silicon points within 14.1%" claim
  (G6: derivation artifact not located). The intro may say the gap to the
  vendor roofline is 9.7–29.9× (that IS in the CSV); the "roofline itself
  infeasible" claim belongs to §6 and only if G6 resolves.
- The 9,956× is prebranch→routed for `ttnn.topk`; 18,401× is prebranch→op.
  Never mix the two chains in one sentence.
- blaze comparison (1.4×) stays out of the abstract — its fairness caveat
  (includes SDPA work, FusedProgram not an op) needs the space §6 gives it.
