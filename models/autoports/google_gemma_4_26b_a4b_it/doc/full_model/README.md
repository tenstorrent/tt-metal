# Gemma-4 26B A4B full model

**P300X2 TP4 batch-1 readiness (128 prompt / 128 generated):** TTFT
**320.84 ms** and trace-verified token-out decode **23.76 t/s/u**. The
short-prompt 64-token diagnostic measured 215.45 ms TTFT and 2.50 t/s/u. The
canonical 100-token AIME24 traced
teacher-forcing run measured **2,253.36 ms TTFT** and **25.40 t/s/u**. These are
full 30-layer measurements including final norm, vocabulary-sharded LM head,
split Sampling1D force-argmax, device token feedback, and host token-out
readback; see `performance.json`.

## Result

`tt/model.py` implements the complete HF text path on a hard-required 1x4
P300C mesh: hidden-sharded BF16 embedding, one model-entry all-gather, all 30
optimized `MultichipDecoder` layers, final RMS norm, tied vocabulary-sharded
BF16 LM head, and logit softcapping. Inter-layer residuals remain replicated
BF16 tile-layout DRAM. Five sequential full-attention layers share one
three-buffer persistent async-all-reduce L1 pool; this preserves the optimized
CCL algorithm while avoiding redundant per-layer L1 residency.

`tt/generator.py` implements the Metal readiness `Generator` contract. It owns
padding and logical slicing for nonaligned prompts, mixed prompt lengths,
per-layer page tables for mixed cache geometry, 32 fixed decode slots,
inactive `current_pos=-1` rows, and explicit cache/position/active state.
Optimized greedy decode is a model trace followed by a sampling trace;
Sampling1D writes its argmax directly into the persistent model token input.
Current and RoPE positions advance on device and page tables remain stable.

The advertised batch-1 context stays **262,144**. State allocation enforces one
262,144-token global-cache budget across active slots; the equal batch-32
profile therefore owns 8,192 full-attention tokens plus a 1,024-token sliding
ring per slot. The worst validated conservative envelope is
22.285305300727487 GiB/device, leaving 9.714694699272513 GiB/device on each
32 GiB device. Details are in `../context_contract.json`.

## Correctness and quality

The fresh reference `readiness_aime24_chat.refpt` uses the exact canonical
Google Gemma 4 template published through `AutoProcessor` (the tokenizer
config itself has no template), AIME24 prompt 0, 100 generated tokens, and HF
top-100 predictions.

- Full-model prefill: top-1 **96%**, top-5 **100%**, top-100 **100%**.
- Traced teacher forcing: top-1 **95%**, top-5 **100%**, top-100 **100%**.
- Mixed/nonaligned probe: prompt lengths 33 and 47, batch 2, one inactive row,
  two unchanged-table traced steps with zero refreshes, then a changed physical
  mapping that recaptured exactly once and stayed stable on its next replay.
- Full all-layer fallback-raising probe: passed in 40.84 seconds.
- Full all-layer batch-32 allocator/trace probe: passed in 23.93 seconds.
- Public-generator non-aligned context probe: logical length 262,111 through
  all 30 layers plus first traced decode passed in 341.29 seconds.
- Two 100-token free-running chat comparisons are under
  `autoregressive_aime24/` and `autoregressive_explanation/`. TT and HF are
  coherent, on-topic English; no repetition collapse, wrong-language drift,
  or early semantic divergence was observed. `degeneracy_report.json` is clean.

## Sampler decision

Selected: `Sampling1D` native force-argmax. It is semantically greedy, accepts
the TP1D vocabulary shards, supports the fixed decode tile, and exposes
`tt_out_tok` for address-stable device feedback. Prefill logits are padded to
the same 32-slot sampling layout before a short-lived first-token sampling
trace.

Rejected: Sampling1D k=1 top-k. On faithful `[1,1,32,262144]` TP4 logits it is
semantically equivalent for greedy decode but measured 10.736 ms versus 2.343
ms for native force-argmax. Also rejected: the
older stateful `SamplingGenerator`, whose mutable TTTv1 state duplicates the
generator's explicit serving state. No custom sampler was written.

An explicit `sampling_mode="host"` compatibility path remains for tests that
require gathered logits/host argmax. It runs eagerly so transient gather
allocations cannot invalidate optimized trace addresses, and it is excluded
from all reported token-out measurements.

Sampled serving uses the same split path: explicit top-k/top-p/temperature and
seed tensors are retained by the sampling trace, the semantic parameter tuple
is part of the trace key, and `tt_out_tok` remains the persistent device token
feedback buffer. A hardware probe alternated greedy, sampled k=8/p=.95/t=.8,
and greedy again. Semantic keys prevented cross-mode reuse; because TT-Metal
pins allocator addresses for an active trace, each mode transition safely
released the prior trace before allocating and recapturing the next one. The
final probe log contains no active-trace allocation warning.

## Performance accounting

The inherited per-layer warmed timings imply a decoder-stack-only lower bound
of **32.329 ms/token** (25 x 1.070437 ms sliding + 5 x 1.113634 ms full), or
30.93 t/s/u. This excludes full-model trace scheduling, terminal norm/LM head,
sampling, and token-out. A signpost-scoped reduced full-model device profile
(one layer of each kind plus terminal and sampler) reports 24.830 ms of merged
four-device op time and 9.7% modeled DRAM roofline. Sparse expert work is
43.82%; sampler argmax is 5.85%, so sampling does not dominate. Autofix found
that the original batch-1 trace mistakenly ran all 32 sampler slots through
every decoder layer. Slicing model compute to logical batch 1 and padding only
at the sampling boundary improved decode from 2.63 to 23.76 t/s/u. The measured
42.08 ms/token is now 1.30x the 32.33 ms decoder-only lower bound; the remaining
9.75 ms covers terminal, sampling, trace orchestration, and scalar output.
Exact CSVs are retained as
`profile_reduced_decode.csv` and `profile_reduced_summary.csv.csv`.

## Reproduction

```bash
python -m models.common.readiness_check.generate \
  --hf-model google/gemma-4-26B-A4B-it --prompt-source aime24 \
  --chat-template --gen-len 100 --top-k 100 \
  --output models/autoports/google_gemma_4_26b_a4b_it/doc/full_model/readiness_aime24_chat.refpt

python -m models.common.readiness_check.run_prefill_check \
  --model-dir models/autoports/google_gemma_4_26b_a4b_it \
  --reference models/autoports/google_gemma_4_26b_a4b_it/doc/full_model/readiness_aime24_chat.refpt \
  --mesh-device P300X2 --fabric-config FABRIC_1D_RING

python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/google_gemma_4_26b_a4b_it \
  --reference models/autoports/google_gemma_4_26b_a4b_it/doc/full_model/readiness_aime24_chat.refpt \
  --mesh-device P300X2 --fabric-config FABRIC_1D_RING
```

Exact logs and artifacts are indexed in `work_log.md`, `trace_evidence.json`,
`performance.json`, and `runtime_fallback_audit.json`.
