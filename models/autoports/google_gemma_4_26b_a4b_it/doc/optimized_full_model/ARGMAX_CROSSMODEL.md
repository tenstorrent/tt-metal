# force-argmax dominates decode here, and Qwen3.6-27B already solved the same problem

Operator note added after stage 07.

## What stage 07 measured

The scoped full-path profile puts **argmax at 1.393 ms of a 5.585 ms merged
device-op sum** — the single largest category, larger than all matmuls combined
(1.236 ms), ahead of async all-gather (0.790 ms) and norms (0.771 ms).

force-argmax was retained because the alternative measured far worse:

| sampler candidate | latency |
|---|---:|
| `Sampling1D` native force-argmax (selected) | 2.3434 ms |
| semantic split top-k=1 | 10.7359 ms |

Identical tokens either way, so the choice is defensible on the measurements
taken. But the stage-07 goal says that if force-argmax *dominates* token-out
decode, the LM-head/sampling contract should be fixed rather than the fast
shortcut kept — and on this profile it does dominate.

## The same defect was fixed on Qwen3.6-27B, in this same pipeline

Qwen's stage 07 hit the identical shape of problem and did not accept it:

- its prior semantic-greedy shortcut gathered the full 248,320-wide vocabulary
  and forced argmax;
- its generic split path pushed the 62,080-wide local vocabulary through a
  **single-core TopK at about 9.7 ms** — the same order as Gemma's 10.7359 ms
  split top-k=1;
- the repair was: an explicit sharded invalid-vocabulary mask, pad each local
  shard, **two 32,768-wide 65-core TopKs**, restore chunk-relative IDs, merge 64
  candidates to 32 with a device gather, all-gather only the candidate
  values/indices, then the common device sampler;
- result: **12.667 ms -> 3.451 ms, 3.67x**, with no invalid ID surviving a probe
  that deliberately made padded IDs competitive.

The two models are close enough for this to transfer: Gemma's vocabulary is
262,144 against Qwen's 248,320, both are TP4 vocabulary-sharded on the same 1x4
P300C mesh, and both were paying a slow single-core TopK on the split path.

## Why it is worth doing rather than leaving

Keeping force-argmax has two costs beyond the 1.393 ms:

1. It is *semantically* greedy only. Falcon3-7B and Qwen3.6-27B both ship split
   sampling capable of top-k/top-p on device, which is what the serving stages
   need; a greedy-only shortcut pushes non-greedy requests onto a slower or
   host path later.
2. It leaves the largest single device-op category unaddressed while the stage
   reports the assembly as optimized.

Recommended: port Qwen's chunked multi-core TopK structure
(`models/autoports/qwen_qwen3_6_27b`, stage 07 "Greedy sampler repair and
performance closure") before or during the vLLM stages, and re-measure. If it
lands near Qwen's 3.451 ms it beats force-argmax's 2.3434 ms only marginally --
so the honest case for doing it is the sampling *contract*, not raw latency, and
that should be stated rather than assumed.
