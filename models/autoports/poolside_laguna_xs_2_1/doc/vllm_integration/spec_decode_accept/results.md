# Laguna-XS-2.1 ngram spec-decode: accept-rate replay on real agent trajectories

- Date: 2026-08-03
- Host-only replay (no Tenstorrent device / GPU). Model not run; RECORDED assistant
  tokens are treated as greedy target output and the exact `tt/spec_decode.py` accept
  loop is simulated per turn.
- Tokenizer: `poolside/Laguna-XS-2.1` (HF, vocab 100352).
- Trajectories: 7 real SWE-bench tool-call runs from ['swe_quick', 'swe_gate1', 'swebench_toolcall_mit'].
- Analyzed: **679 assistant turns, 176790 target tokens**.
- Metric: `mean(m+1)` = mean committed tokens per verify iteration. Because the
  on-device verify reads the full KV once per iteration regardless of K (~one decode
  step), projected decode speedup ~= `mean(m+1)`.
- Drafter fidelity: fast incremental index; `--validate` asserts it is token-for-token
  identical to the shipped `NgramProposer` (see script docstring).

## Full sweep (sorted by mean(m+1))

| min_n | max_n | K | mean(m+1) | mean_m | iters | turns | verdict |
|------:|------:|--:|----------:|-------:|------:|------:|:--------|
| 1 | 10 | 16 | 2.504 | 1.509 | 70598 | 679 | STRONG |
| 1 | 8 | 16 | 2.501 | 1.506 | 70688 | 679 | STRONG |
| 1 | 5 | 16 | 2.483 | 1.488 | 71186 | 679 | STRONG |
| 1 | 3 | 16 | 2.444 | 1.448 | 72334 | 679 | STRONG |
| 1 | 10 | 8 | 2.374 | 1.379 | 74459 | 679 | STRONG |
| 1 | 8 | 8 | 2.362 | 1.366 | 74841 | 679 | STRONG |
| 1 | 5 | 8 | 2.337 | 1.341 | 75659 | 679 | STRONG |
| 1 | 3 | 8 | 2.295 | 1.298 | 77040 | 679 | STRONG |
| 2 | 10 | 16 | 2.293 | 1.297 | 77108 | 679 | STRONG |
| 2 | 8 | 16 | 2.290 | 1.294 | 77204 | 679 | STRONG |
| 2 | 5 | 16 | 2.275 | 1.279 | 77696 | 679 | STRONG |
| 2 | 3 | 16 | 2.242 | 1.246 | 78846 | 679 | STRONG |
| 2 | 10 | 8 | 2.185 | 1.189 | 80900 | 679 | STRONG |
| 2 | 8 | 8 | 2.175 | 1.179 | 81278 | 679 | STRONG |
| 2 | 5 | 8 | 2.153 | 1.157 | 82096 | 679 | STRONG |
| 1 | 10 | 4 | 2.147 | 1.151 | 82335 | 679 | STRONG |
| 1 | 8 | 4 | 2.139 | 1.142 | 82652 | 679 | STRONG |
| 2 | 3 | 8 | 2.118 | 1.121 | 83478 | 679 | STRONG |
| 1 | 5 | 4 | 2.116 | 1.119 | 83556 | 679 | STRONG |
| 1 | 3 | 4 | 2.065 | 1.068 | 85625 | 679 | STRONG |
| 3 | 10 | 16 | 2.031 | 1.035 | 87033 | 679 | STRONG |
| 3 | 8 | 16 | 2.029 | 1.032 | 87128 | 679 | STRONG |
| 3 | 5 | 16 | 2.018 | 1.021 | 87618 | 679 | STRONG |
| 2 | 10 | 4 | 1.995 | 0.998 | 88599 | 679 | MODERATE |
| 3 | 3 | 16 | 1.991 | 0.994 | 88788 | 679 | MODERATE |
| 2 | 8 | 4 | 1.989 | 0.991 | 88906 | 679 | MODERATE |
| 2 | 5 | 4 | 1.968 | 0.971 | 89826 | 679 | MODERATE |
| 3 | 10 | 8 | 1.949 | 0.952 | 90692 | 679 | MODERATE |
| 3 | 8 | 8 | 1.942 | 0.945 | 91049 | 679 | MODERATE |
| 2 | 3 | 4 | 1.925 | 0.927 | 91853 | 679 | MODERATE |
| 3 | 5 | 8 | 1.925 | 0.927 | 91859 | 679 | MODERATE |
| 3 | 3 | 8 | 1.896 | 0.898 | 93249 | 679 | MODERATE |
| 3 | 10 | 4 | 1.805 | 0.807 | 97968 | 679 | MODERATE |
| 3 | 8 | 4 | 1.799 | 0.801 | 98272 | 679 | MODERATE |
| 3 | 5 | 4 | 1.782 | 0.785 | 99190 | 679 | MODERATE |
| 3 | 3 | 4 | 1.747 | 0.749 | 101187 | 679 | MODERATE |

## Best config: min_n=1 max_n=10 K=16

- **mean(m+1) = 2.504**  (mean accepted drafts m = 1.509)  — **STRONG**
- iterations = 70598, turns = 679

### Per-trajectory breakdown (best config)

| trajectory | mean(m+1) | mean_m | iters | turns | target_toks |
|:-----------|----------:|-------:|------:|------:|------------:|
| swe_quick/astropy__astropy-12907 | 2.412 | 1.415 | 9763 | 77 | 23552 |
| swe_quick/astropy__astropy-13033 | 2.376 | 1.380 | 14621 | 114 | 34744 |
| swe_quick/astropy__astropy-13236 | 2.266 | 1.274 | 5328 | 91 | 12072 |
| swe_quick/astropy__astropy-13398 | 2.282 | 1.288 | 7196 | 99 | 16418 |
| swe_quick/astropy__astropy-13453 | 2.126 | 1.138 | 5234 | 106 | 11129 |
| swe_gate1/astropy__astropy-13033 | 3.028 | 2.031 | 16596 | 106 | 50249 |
| swebench_toolcall_mit/astropy__astropy-12907 | 2.414 | 1.418 | 11860 | 86 | 28626 |

### Acceptance distribution (best config): m accepted drafts per iteration

| m | iterations | fraction |
|--:|-----------:|---------:|
| 0 | 43848 | 0.621 |
| 1 | 11013 | 0.156 |
| 2 | 4520 | 0.064 |
| 3 | 2633 | 0.037 |
| 4 | 1612 | 0.023 |
| 5 | 1197 | 0.017 |
| 6 | 805 | 0.011 |
| 7 | 669 | 0.009 |
| 8 | 448 | 0.006 |
| 9 | 392 | 0.006 |
| 10 | 376 | 0.005 |
| 11 | 278 | 0.004 |
| 12 | 247 | 0.003 |
| 13 | 251 | 0.004 |
| 14 | 176 | 0.002 |
| 15 | 127 | 0.002 |
| 16 | 2006 | 0.028 |

## Recommendation

**SHIP / do on-device B2.** The best config projects ~2.50x decode speedup, comfortably above the 2.0 bar.

### Reasoning
- Best of 36 configs reaches mean(m+1)=2.504; 23/36 configs clear the 2.0 STRONG bar and all clear ~1.75.
- Each verify iteration costs ~one decode step on device; a mean(m+1) of X means X target
  tokens per decode step vs 1 for plain decode, i.e. a raw ~X speedup BEFORE the
  multi-token verify's extra prefill-path cost.
- The accept loop is correctness-neutral (committed tokens are always target tokens), so
  the only question is speed; this replay answers it directly on the target workload.
