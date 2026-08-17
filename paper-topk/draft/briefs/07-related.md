# Brief 07 — Related Work + Methodology + Conclusion

**Section file:** `sections/07-related.tex`
**Budget:** 2.0 columns (1.0 page) — related ≈0.9 col, methodology ≈0.7 col
(the promised "candid 0.5-page" ≈ 0.7 col at IEEE density), conclusion
≈0.4 col.

## Single job

Close three loops: (1) position each contribution against the exact prior
art the sweep found — leading with the citations a hostile reviewer would
raise; (2) disclose the agentic campaign as methodology, positioned as a
design-space-study campaign among the AccelOpt/KernelBench cousins, claiming
no first-ness; (3) conclude with the banked results and the pre-registered
reopening conditions.

## Part 1 — Related work (repositioned per related-work.md verdicts)

Organize by contribution, NOT by chronology. Most citations already appeared
inline in §2–§5; this section is the synthesis + the deltas.

- **Mesh selection theory (C2 — get ahead of the reviewer):** LEAD with
  Krizanc & Narayanan (krizanc1992optimal, krizanc1993fast — 1.45n steps on
  n×n; multipacket1992 for the N/p regime). State precisely what theory did
  NOT do: full top-k set with values+indices, real NoC costs (multicast,
  per-hop, DRAM ingress), atomics-free rendezvous on real silicon, any
  measurement. C2's claim, final form: **first measured, cost-modeled full
  top-k on commercial NoC-mesh silicon**; the cost model is the mesh-theory
  bound instantiated with measured constants.
- **GPU selection (C1):** Alabi'12 → WarpSelect → Dr.Top-k → SC'23 study →
  RadiK; TPU-KNN + two-stage as the approximation escape hatch. One sentence:
  none of this lineage was ever re-costed on a machine without atomics or
  scatter — that re-costing is §3.
- **Running-threshold pruning (C3):** the mandated contrast pair —
  **WarpSelect** (element-granular, per-lane, no law) and **GVR** (gvr2026:
  temporally *predicted* threshold needing verify/refine passes; 1.88× over
  TensorRT-LLM radix select on NVIDIA Blackwell) vs chunk-skip
  (chunk-granular, sound by construction — no verification pass,
  distribution-free law → compile-time gate, pre-build forecast). Tab S1
  (§4) is referenced, not repeated. SpAtten + Focus as hardware
  threshold-engine cousins (focus2025 must be read before submission —
  UNVERIFIED). Records-theory footing: running top-k updates =
  k·ln(n/k)+O(k), textbook (knuth1998taocp3); the chunked corollary and the
  K/4 gate have no published counterpart.
- **Accelerator characterization (C4):** the Tensix perf-paper census
  (brown2024grayskull, matmul2025tenstorrent, brown2025fft,
  stencil2026wormhole, numkernels2026wormhole, fusion2026tensix,
  amati2025nbody) + the informal Blackhole microbench
  (blackholemicrobench2025 — dead link, non-archival: make the archival
  argument once) + jia2019ipu as genre template. None covers datapath
  numerics or selection primitives. Software float-ordering prior art
  (herf2001radix, merrill2011radix, ieee754-2019) acknowledged; the delta is
  hardware-imposed order.
- **Distributed DB top-k:** Fagin TA / TPUT / KLEE — same tournament idea,
  datacenter round-trip cost regime; one sentence.
- **The absence claim:** no sorting/selection publication exists for
  Cerebras WSE, Groq, SambaNova, Occamy, or Esperanto (related-work.md §5
  item 5) — the paper fills a class-wide gap; Poplar's popops::TopK exists
  as an undocumented-in-literature production library (poplar_topk).

## Part 2 — Methodology: an agentic design-space campaign (≈0.7 col, candid)

Positioning rule (related-work.md verdict): prior agentic work optimizes
*given* kernels against benchmarks (KernelBench, AI CUDA Engineer, AccelOpt,
KForge, PEAK); this campaign ran a *design-space study with measured no-gos*
on novel silicon. Claim NO first-ness. Content:

- **Structure (M1):** research → implement → validate gates with
  pre-registered stop rules written before measurement (RBG §5.2–5.3); the
  radix track was killed BY its own pre-registered rule (CRIT-4: incumbent
  ≥2× model), §3.4's story.
- **Forecast-before-build (M3):** the calibrated host simulation as a no-go
  instrument — predicted 0.00% skip + regression for the column-parallel
  variant; the build went row-parallel and measured −45.1%. This is the
  section's most transferable idea; give it 3–4 sentences.
- **Adversarial self-audit (M2):** a swarm audit of the campaign's own
  claims — ≈55/60 branch citations exact, zero fabricated; disputed figures
  re-adjudicated on fresh silicon; a faster-and-wrong kernel class caught by
  mutation controls. Cite the AI CUDA Engineer reward-hacking episode
  (sakana2025aicuda) as the community's cautionary tale this discipline
  answers.
- **Correctness gates as agent guardrails (M4):** bit-exact-or-fail is what
  makes agent-generated measurements trustworthy; PCC thresholds would have
  let wrong-but-close kernels through.
- **Harness findings (C4-9, one sentence):** the campaign also surfaced
  perf-harness bugs (unwritten stimuli config, a semaphore leak that
  deadlocks later tests) — reported upstream.
- **Candor requirement:** name the failure modes plainly — an a-priori Dst
  derivation was wrong (calibration replaced it, §5.5); a working-note
  figure was invented and caught by the audit (M5's "~18%" example —
  generalize, don't itemize); lit-review self-corrections were logged. Two
  sentences, matter-of-fact.

## Part 3 — Conclusion (≈0.4 col)

- Restate the thesis in past tense with the three strongest numbers:
  9,956× end-to-end (631.5 ms → 63.4 µs); 23.3 µs @ 104 cores on the
  1M-context decode shape; 1.82× chunk-skip. One number per sentence.
- The design-space answer, one sentence: on a NoC mesh without scatter, the
  winning economics are parallelized comparison networks plus sound
  rank-statistic skipping — Floyd–Rivest's lesson, not radix's.
- **Reopening conditions (from RBG §7.4 — print all four as a list):**
  (1) a device-side compaction/gather primitive appears, or the composed
  packer-compressed producer+consumer bench proves ≥2× headroom at
  k=2048, N≥262,144; (2) k grows past the 2048 LLK ceiling where bitonic
  merge state stops fitting; (3) the consumer becomes a sampler (the
  contract never materializes the top-k set, deleting the materialization
  gate entirely); (4) N grows to where full-data passes dominate and
  count-guided skipping compounds. Frame as falsifiable predictions — the
  negative result comes with its own expiry conditions.
- Future work, one sentence each: Wormhole port (G1), energy (G2), composed
  producer bench (G7), end-to-end token/s after the call-site change (G12).

## Claims owned

M1–M5 (evidence.md §1.6); the four repositioned novelty statements (final
phrasings — intro states them, this section defends them); the reopening
conditions; the absence claim.

## Figures/tables owned

- **(Optional) Tab M1** (1 col): gate ledger — gate → bench → pre-registered
  rule → verdict → what was banked (evidence Tab-6 row; source RBG §5.2/§7).
  Include if the methodology subsection reads too abstract without it; cut
  first if over budget.

## Style directives

1. Related work in Ienne-grade contrast discipline: every paragraph ends
   with the one-clause delta ("...but measures nothing on silicon").
   No summary-without-contrast paragraphs.
2. Methodology voice: first-person-plural process description, zero
   defensiveness, zero grandiosity — "the forecast said no; we did not
   build it" is the register.
3. The conclusion admits no new numbers except those already defended in §6.
4. Reopening conditions as a compact enumerated list — they are the
   negative result's falsifiability contract; keep the (A)–(F) vocabulary
   from §3.
5. Anonymity stress point: the methodology subsection must not name the
   agent product/vendor in a deanonymizing way; "an LLM-agent campaign with
   human review" is sufficient; no internal tool names (afterburner,
   storm, ledger filenames).

## Hazards

- Do NOT claim "first agentic kernel work" or "first agentic design-space
  study" — position, don't claim (related-work.md verdict row 5).
- KForge may target Tenstorrent (UNVERIFIED) — check before submission; if
  it does, add one sentence distinguishing kernel synthesis from
  design-space adjudication.
- M2's "~55/60 citations exact" is an internal-audit figure; print it as
  approximate and attribute to the audit process, or omit the number — do
  not let it read as an externally verified statistic.
- The methodology subsection is 0.7 col MAX — it is a supporting
  disclosure, not a fifth contribution; if it swells, it eats the
  conclusion's reopening conditions, which matter more.
