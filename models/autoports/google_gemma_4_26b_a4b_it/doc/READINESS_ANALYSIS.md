# What blocks this model, how far it is, and the gap that explains the failure

Operator analysis, 2026-08-17. Written from committed artifacts on this branch.

## Status: stages 04–10 complete, stage 11 fails on quality

Everything through optimized-vLLM passed its gate, and the serving result is the
best of the three ports run through this pipeline (28.04 t/s/u served, exactly
matching the 28.03 standalone; TTFT 201.6 ms; AIME24 top-1 98 %). Stage 11 ran
the release workflow to completion and failed on accuracy:

| mandatory eval | TT | HF control (same harness) | verdict |
|---|---:|---:|---|
| `meta_ifeval` | 82.62 % (23 correct) | 87.04 % | PASS, exactly at threshold 23 |
| `meta_gpqa_cot` | **4/10** | **10/10** | **FAIL** (9 required) |

The control is properly sourced: same snapshot `4d7ae498`, same TT lm-eval fork
commit, chat template, seed 42, deterministic generation, first 5 % subset,
`max_length=262144`. The stage refused to self-baseline. So this is a real
model-path defect, not a reference artifact.

## Is the PCC good? Yes — and that is the point

This port has the strongest correctness evidence of the three, and it is against
**real HF weights** (its source cell had only ever been checked against
synthetic weights, because Gemma's real weights were absent from the host):

| gate | sliding layer 0 | full layer 5 |
|---|---:|---:|
| HF vs TP prefill PCC | 0.9986126 | 0.9970877 |
| HF vs TP decode PCC | 0.9996529 | 0.9997856 |
| optimized TTNN vs TP | ≥ 0.995 pass/pass | ≥ 0.995 pass/pass |
| replayed decode | bit exact | bit exact |

Stage 05 moved decode PCC to 0.997778 / 0.998347 — still above bar. So **layer
PCC is healthy and does not predict the GPQA failure**. Anything that concludes
"PCC is fine, therefore the model is fine" is reasoning from the wrong evidence.

## The gap: nothing was ever checked at the length where quality is judged

| what | length |
|---|---:|
| longest **correctness-measured generation** on this branch | **100 tokens** (`--gen-len 100`) |
| prefill PCC taken at | **S=33** |
| `meta_ifeval` generation budget — **passes** | 1,280 tokens |
| `meta_gpqa_cot` generation budget — **fails** | **32,768 tokens** |

The advertised-context evidence does not close this. `position 262143` decode and
`S=262143` capacity prefill are **capacity** probes: a single decode step at a
high position with a rolled page table, and a prefill that fits. Neither is a
sustained autoregressive generation. There is no evidence on this branch for
correctness after thousands of consecutive self-fed decode steps.

So the ordering is: **passes at 100 tokens (measured), passes at 1,280 (IFEval),
fails at 32,768 (GPQA).** That is a monotone length effect, and it is consistent
with the stage's own AutoDebug conclusion, which narrowed the divergence to "the
generated autoport's long, concurrent decode path (including its numerical
policy) rather than the GPQA task definition or chat template".

## Two concrete mechanisms worth testing first

1. **Accumulated numerical drift.** The selected policy is BFP8 attention/dense/
   expert weights with LoFi on full attention, dense MLP and expert matmuls. A
   per-step error invisible at 100 steps compounds over ~32,000. Note stage 08
   found this model is unusually precision-sensitive: **BFP8 activations collapse
   it to 0 %/1 % top-1/top-5** where both dense ports ship BFP8 activations
   happily.
2. **Sliding-cache wrap.** `sliding_window=1024`, so a 32,768-token generation
   wraps the sliding cache about **32 times**. Stage 04 tested exactly one
   boundary — logical lengths 1024 vs 1025, slot 0 replaced and slots 1..1023
   preserved at PCC 0.9999. One wrap passing does not establish 32 wraps, and
   25 of 30 layers are sliding.

## The cheapest experiment that would localise it

Generate 4,096 tokens teacher-forced against the HF reference and record the
**first divergence index** plus PCC at 256/512/1024/1025/2048/4096. That
distinguishes the two mechanisms immediately: drift shows PCC decaying smoothly
with index, while a sliding-cache bug shows a step at multiples of 1024. It needs
no GPU and no new harness — the existing teacher-forcing path already reports
top-k agreement per position; it only needs to run longer than 100 tokens.

If that is clean to 4,096, escalate to 32,768 on the GPQA prompts themselves,
where the failing documents are already known (4 of 10 correct).

## How far from "fully working"

Close on everything except this. Serving, context (262,144 advertised and
served), sampling suite (72 passed / 1 skipped, no failures), non-aligned
prompts, capacity and performance are all done and gated. The single blocker is
long-generation quality, and it is a real defect with a narrowed hypothesis and a
cheap next experiment — not an infrastructure or reference problem.
