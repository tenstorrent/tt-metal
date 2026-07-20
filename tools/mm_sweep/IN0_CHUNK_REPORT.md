# in0-ring chunk-streaming experiment (diagnostic)

Question: does the shard-sized ring bundle (C=W compute K-blocks read/forwarded/published per step) hide
useful compute overlap, or are its larger payloads needed for NoC efficiency? Test finer bundles C=4/2/1
(C=1 = K-block-granular), diagnostic-only, cache-hashed, not python-exposed, C=W byte-identical (mask 0 uses
the unchanged baseline branch). Preserves ring order / placement / PARETO / CB0 layout / K traversal /
in0-in1 pairing exactly; only the read/forward/publish granularity changes, with cumulative per-chunk
forward credits `(step-1)*chunks + chunk + 1`.

Commit (recovery): implemented under `DIAG_IN0_CHUNK4/2/1` (1<<26/27/28); W derived from the picker config
(`W = Ktl/(8·kb)`), not hardcoded.

## Correctness (bit-identity is the gate; math order is unchanged)
`ChunkCorrectness` gtest (random bf16 vs CPU f32 golden): **every chunk mode (C=4/2/1) is BIT-IDENTICAL to
C=W** and PCC≥0.99, across W=1/2/3/5, Sm=1/2/3, Ns>1, N_bpc>1, split-K + Pk=1, K/M/N tails, fresh + cached.
Public 20/20 pass (public path unchanged); 7 diag gtests pass; watcher-clean on the highest-forwarding
W=5 C=1 case.

## Performance matrix (config=None Picker v3; median kernel µs, 3 interleaved relaunches; Δ vs C=W)

| shape | W | nsb | C4 | C2 | **C1** |
|---|--|--|--|--|--|
| 256×15360×768 | 5 | 3 | +1.6% | −0.2% | **−3.1%** |
| 256×15360×1536 | 5 | 6 | +2.3% | +1.8% | +0.1% |
| 128×15360×768 | 5 | 3 | −0.6% | −0.3% | −1.4% |
| 128×15360×1536 | 5 | 3 | +0.5% | +0.4% | −0.6% |
| 32×15360×768 (bw-neg-ctl) | 5 | 3 | +0.9% | +0.2% | −0.1% |
| 256×2304×6144 | 3 | 3 | +0.1% | −0.2% | **−1.3%** |
| 32×6144×2304 | 3 | 9 | −0.1% | −0.1% | −0.0% |
| 64×4608×6144 | 3 | 8 | −0.0% | +0.1% | −0.2% |
| 32×6144×768 | 2 | 3 | −0.1% | −0.1% | −0.3% |
| 256×2048×512 (W1 ctl) | 1 | 2 | −1.1% | −1.2% | −1.0% |
| 32×2048×512 (W1 ctl) | 1 | 1 | +1.4% | −0.1% | −0.2% |
| 256×6144×4608 (W1 ctl) | 1 | 2 | −0.6% | −0.0% | −0.1% |
| 128×6144×4608 (W1 ctl) | 1 | 1 | −0.0% | +0.1% | +0.2% |

**Independent repeat (4 interleaved relaunches, fresh cache), C1 vs C=W:**
- 256×15360×768: **−2.8%** (CW 94.6–95.9 vs C1 92.2–93.2 — cleanly separated).
- 256×2304×6144: **−1.9%** (87.4–87.8 vs 85.8–86.8 — separated).
- 128×15360×768: −1.1% (64.8–65.6 vs 64.3–65.0 — overlapping, marginal).
- 256×15360×1536: −0.5% (confirmed non-winner).

## Findings
- **Only C1 (K-block-granular) helps; C2/C4 are neutral-to-worse.** So it is not "smaller is generally
  better" — only the *finest* granularity exposes the overlap, and only on the right shapes.
- The win is confined to **narrow-N (nsb3) W≥3** shapes: 256×15360×768 (−2.8%), 256×2304×6144 (−1.9%),
  128×15360×768 (−1.1%). Wide-nsb (6/8/9), W=2, the bandwidth-bound control, and all W=1 controls are neutral.
- **No regression anywhere**, including W=1 (chunk = no-op, chunks=1) and the bandwidth-bound / wide-N
  controls → the larger shard payloads are **not** needed for NoC efficiency; the finer bundles are free.
- **Answer to the question:** the shard-sized bundle *does* hide a modest amount of compute overlap
  (~2–3%) on narrow-N deep-K shapes — the finer C=1 publish lets compute start on the first K-block while the
  rest of the shard is still arriving — but the effect is small and shape-specific.

## Decision gate
Gate = stable multi-percent (≥~2%) win on ≥2 meaningful W>1 shapes, no control regression, clean
correctness/cache/watcher, no host overhead. Result: **one shape clears ≥2% solidly (256×15360×768 −2.8%),
one is borderline (256×2304×6144 −1.9%), one is marginal (128×15360×768 −1.1%)** — right at the gate
boundary, and only for C1 on a narrow subclass. Controls clean, correctness bit-identical, watcher clean.

**Decision: keep C1 as an A/B diagnostic; no production change.** The win is real and stable but modest
(one shape solidly ≥2%, one borderline, one marginal) and confined to the narrow-N nsb3 W≥3 subclass, so it
does not warrant a production default / picker trigger now. `DIAG_IN0_CHUNK4/2/1` remain committed behind the
hashed diag mask (mask 0 = the unchanged baseline ring, byte-identical); the public path, picker, placement,
ring order, compute order, reduction, and in1 reader are all untouched. A narrow-N-W≥3 picker trigger for C1
is a documented future candidate if the ~2–3% on the deep-K production shapes (e.g. 256×15360×768) becomes
worth capturing. Recovery commit: `406f13b593b`.
