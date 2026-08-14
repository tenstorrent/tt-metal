# The datatype selection ignored TTFT, and TTFT is this model's dominant term

Operator note added after stage 08, before stage 09. The selected config passes
every gate; this is not a correctness objection. It records that the ranking
metric excluded the axis that dominates this model's latency, with the numbers
from the stage's own frontier table, so stage 09/10 can revisit it deliberately.

## What was selected, and what it cost

All rows are the stage's own S161 AIME24 chat-template measurements.

| config | top-1 | TTFT ms | traced TF t/s/u |
|---|---:|---:|---:|
| baseline optimized default | 97 % | 5153.68 | 6.91 |
| all-projection BFP8 + LoFi | 98 % | 5135.28 | 6.64 |
| all-projection BFP8 + HiFi2 | 97 % | 5139.08 | 6.26 |
| selected BFP4 + MLP HiFi2 | 93 % | 5125.51 | 6.72 |
| selected BFP4 + linear HiFi2 | 93 % | 5125.87 | 6.90 |
| **full-attention BFP4 + LoFi (selected)** | **93 %** | **5628.30** | **7.00** |

The selected row has the **highest teacher-forcing t/s/u and the worst TTFT of
the whole ~5.13 s cluster**: +502 ms (+9.2 %) against baseline, and +502 ms
against two configs with identical top-1.

## Why this is worth revisiting

1. **TTFT dominates this model.** ~5 s at S161 and 4.04 s at S128, against
   ~56 ms per decode token. For any short-output request TTFT is essentially the
   whole user-visible latency. Falcon3-7B's TTI functional target was 250 ms.
2. **The ranking metric is not the serving metric.** Selection ranked traced
   teacher forcing (7.00 t/s/u), which this stage's own README calls the
   selection metric only, and names post-selection token-out (17.90 t/s/u) as
   "the serving-comparison number". The two differ by 2.5x, and teacher forcing
   reads logits and refreshes teacher inputs on host.
3. **A strictly better trade is already measured.** `selected BFP4 + linear
   HiFi2` gives 502 ms of TTFT back at the same 93 % top-1 for 0.10 t/s/u
   (1.4 %) of teacher-forcing decode. If accuracy also matters,
   `all-projection BFP8 + LoFi` is 493 ms better on TTFT and **5 points better
   on top-1** for 5.1 % of that same non-serving metric.
4. **It also overturns a stage-04 rejection on a different oracle.** Stage 04
   rejected BFP4/LoFi full-attention projections on official-weight layer PCC
   (0.9870); stage 08 selects them on a model-level top-k gate. Both are
   defensible, but the precision decision now rests on whichever oracle the
   later stage happens to use, and the layer-PCC objection is not addressed.

## What is NOT claimed

No measurement here contradicts the stage. TTFT differences of ~500 ms on a
~5.1 s number are ~10 % and were each measured once, so they are indicative
rather than tight. The right resolution is to re-measure the two or three
candidates on the **token-out** path at B1/S128 and rank on TTFT plus token-out,
not on teacher forcing.

## How to act on it cheaply

`build_generator` / `Qwen36Model.from_pretrained` load
`doc/datatype_sweep/selected_precision_config.json` when no override is given, so
swapping candidates is a config change rather than a code change. Any switch
must re-run the top-k gate and refresh the stage-07/08 evidence chain that cites
the current policy.
