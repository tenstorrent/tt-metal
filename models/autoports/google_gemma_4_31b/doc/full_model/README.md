# Gemma 4 31B Full Model

Stage: full-model (Stage 06)
Model: `google/gemma-4-31B` at revision `d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3`
Target: four Blackhole P150b devices, `MeshShape(1,4)`, TP4, `FABRIC_1D`
Implementation: `tt/model.py`, `tt/generator.py`
Status: complete; independent `$stage-review` verdict `clean-pass`.

## Headline performance

| Full 60-layer path, batch 1 | TTFT | Decode throughput | Notes |
| --- | ---: | ---: | --- |
| Teacher forcing | 1,793.14 ms | 22.79 t/s/u | 16.30 t/s/u end-to-end; ground-truth token write each step |
| Traced token-out | 693.70 ms | 24.97 t/s/u | includes first-token trace setup |
| Traced token-out, steady | — | 33.87 t/s/u | device greedy sample feeds the next replay directly |

Teacher forcing and token-out are deliberately reported separately. Teacher forcing reads one prediction and writes one ground-truth token per step. The measured token-out path uses cooperating model and sampler traces, performs no per-token token readback or Python token-feedback loop, and reads only the final sampled token.

The TTFT rows are not like-for-like cache states: the teacher-forcing readiness process includes its own cold/warmup sequence, while the token-out result is collected after persistent tensor-cache materialization in the qualitative runner. They characterize their named harnesses, not a claimed 2.6x prefill optimization.

Stage 05's final per-layer traced decode gives the required stack lower bound:

```text
50 sliding × 0.463813 ms + 10 full × 0.5166275 ms
= 28.356925 ms/token = 35.2647 t/s/u
```

Full-model steady token-out is 29.52798 ms/token. Terminal work, sampling, and trace orchestration therefore add 1.17106 ms/token (4.13% over the layer-stack bound), and delivered throughput reaches 96.04% of the decoder-stack ceiling. Source: `doc/optimized_multichip_decoder/evidence/final_latency.log` and the source-current reduced `perf/` report.

## Full-model contract

The wrapper instantiates the production `MultichipDecoder` for all 60 HF layers and preserves the optimized Stage 05 policy without fallback:

- TP4 Linear fabric, two decoder CCL links, and one mesh-scoped persistent asynchronous CCL pool;
- replicated BF16 DRAM residual at every inter-layer boundary, with no inter-layer collective or host round trip;
- attention BFP8/LoFi, gate/up and down BFP4/LoFi, and packed M=1 MLP output BFP8;
- QKV block 7 on 32 output cores, O block 8, 14-core MLP block 12, and SDPA q32/k64;
- BFP8 cache storage, BF16 decode updates, and BFP8 prefill fills;
- 50 sliding layers with physical 1,024-token circular caches and 10 full layers with physical 262,144-token caches;
- one 731,136-byte/device L1 CCL pool shared by the stack.

Terminal work remains TP-native: BF16 hidden-column-sharded embedding, final replicated RMSNorm, tied-value BF16 vocab-sharded LM head, local logit softcap, and 65,536-logit shards. The optimized path never assembles full logits on host.

The public generator owns padding, masks, logical prompt lengths, cache fill, page tables, positions, and output slicing. Arbitrary valid prompt lengths are accepted. For a partial terminal tile, the model slices the live rows before final norm; for prompts of 32 tokens or fewer, ownership is transferred directly instead of freeing a full-range slice alias. Both cases are covered by sequential short-prompt and watcher tests.

Low-level prefill/decode expose explicit cache, page-table, page-table generation, prompt-length, batch, cache-position, and RoPE-position state. Device-logit prefill iterates row/user cache ownership, preserves each non-aligned logical length, and concatenates one TP-sharded sampler-ready row per prompt on device; mixed lengths 33 and 17 pass together at batch two with no full-logit composition. Fixed physical slots and inactive rows use row-specific lengths and cache position `-1`; the inactive-row increment mask prevents position drift. External KV cache and page tables must be supplied together, and traced external state is bound to allocation identity.

## Context capacity

`doc/context_contract.json` retains the full 262,144-token HF context. Per-device accounting is:

- physical decoder, embedding, final-norm, and LM-head weights: 10,908,115,456 bytes;
- batch-1 KV cache: 2,789,212,160 bytes;
- weights plus KV: 13,697,327,616 bytes;
- replicated RoPE, sampler state, page/trace inputs, 256 MiB trace region, and retained 12 GiB allocator reserve included in total;
- accounted total: 27,672,814,984 bytes;
- usable descriptor DRAM: 34,225,520,640 bytes;
- remaining margin: 6,552,705,656 bytes/device.

The cache total retains 50 physical-1,024 sliding layers and 10 physical-262,144 full-attention layers. No context reduction is physically necessary; exact-context and non-aligned guards remain 262,144.

Production full-context accuracy and performance are batch one. The largest hardware-tested full-model batch is two at context 128, covering mixed prompt lengths and inactive fixed slots. The full-context accounting upper bound is batch three: 33,251,239,304 bytes/device, leaving 974,281,336 bytes. Batch four would require 36,040,451,464 bytes/device, a 1,814,930,824-byte shortfall; batch-32 KV alone would be 89,254,789,120 bytes/device. Thus batch 32 at full context is a hard physical impossibility, while batch three is a capacity upper bound rather than an advertised tested serving mode.

## Sampling decision and rejection ledger

Both common samplers were evaluated before custom code was accepted:

- `Sampling1D` at the required local width 65,536 took about 10.625 ms for its TopK kernel, while the reduced model trace took about 5.15 ms. It dominated token-out latency.
- Partitioned/common TopK candidates passed synthetic vocabulary boundaries but failed semantic greedy tie-breaking on real Gemma BF16 softcapped logits: equal maxima at global tokens 177 and 192 returned 192 instead of the required lowest global token 177.
- Common force-argmax also failed exact shard boundaries.
- `SamplingGenerator` rounds batch 1 to a batch-32 state and owns mutable/internal trace state that does not fit this generator's fixed-slot contract.
- Native small-TILE gathers at widths 32 and 64 asserted in `minimal_default_writer`; BF16 broadcast corrupted candidate values; a row-major broadcast/concat candidate hung during trace replay. These candidates were reverted.

The selected `Gemma4GreedyTP4Sampler` is intentionally narrow: greedy-only, TP4, BF16 tiled logits. Eight cores per device compute tile-local `(score, global_token)` winners with an explicit lower-token tie rule; a tiny two-link Linear TP all-gather exchanges pairs; the final reducer writes into the pre-existing persistent token tensor. Non-greedy compatibility continues to use the common `Sampling1D` path.

Hardware boundary evidence covers global IDs `[0, 32767, 32768, 65535, 65536, 262143]`, the 177-versus-192 equal-score tie, batch two, and three trace replays. Watcher exposed and drove fixes for the reducer's DRAM write alignment and 8-byte-versus-16-byte pair-page stride.

The final post-fix reduced profile flushes setup traffic before a four-replay signposted window. `tt-perf-report` measures the local-winner kernel at a 299.464 µs median and final reducer at 0.4205 µs; all-gather/concat is single-digit microseconds. Sampling is 9.68% of reduced device-op time and about 8.6% of the 3.484 ms steady end-to-end token time, so it does not dominate. The LM head is the dominant 56.25% of device-op time. Reduced steady token-out is 287.02 t/s/u under profiling. See `sampler_profile_summary.md` and `perf/final_report.md`.

## Trace and token-feedback evidence

Two cooperating traces run on one command queue:

1. model trace: embedding, 60 optimized decoder layers, final norm, sharded LM head, softcap, and device position updates;
2. sampler trace: exact TP4 greedy reduction into the persistent token tensor consumed by the next model replay.

The 100-token full-stack run records 99 model trace replays, zero token host refreshes, zero full-logit readbacks, two initial position/RoPE host writes, three synchronizations, and one final sampled-token readback. Unchanged page tables cause no copies. A changed identity/generation uses one distributed mesh copy, and repeating the same identity/generation performs no copy. Reset and prefill release both traces before cache or input buffer churn.

Registering the second trace region emits TT Metal's conservative warning because the model trace is already registered. The application no longer allocates sampler data in that interval: token output, local pairs, gathered pairs, parameters, and logits are persistent and preallocated; regular all-gather writes to `output_tensor=self.gathered_pairs`. The warning is therefore attributed to the second `begin_trace_capture` region allocation required by the canonical split-trace contract. Repeated replay, reset/recreate, changed tables, watcher, and source-current profile all pass; sampler tensors are explicitly released at teardown. `AUTOTRIAGE.md` records the resolved disposition.

## Correctness and qualitative results

The fresh AIME24 reference uses the exact HF revision and `GemmaTokenizer`. This base tokenizer has no chat template (`chat_template=None`), so metadata records plain exact-tokenizer completion mode rather than inventing a template.

| Gate | Top-1 | Top-5 | Top-100 |
| --- | ---: | ---: | ---: |
| Prefill, 100 reference positions | 91% | 100% | 100% |
| Teacher-forced decode, 100 positions | 91% | 100% | 100% |

The 100-token common autoregressive prompt produced coherent English story continuations from HF and TT. The first generated token matches; the streams diverge at generated token two and agree at 8/100 positions overall, but TT remains coherent, has no adjacent repetition or wrong-language drift, and ends only because of the 100-token cap.

The six-prompt qualitative suite is mechanically non-degenerate. Haiku, story, and Fibonacci continuations are coherent. Supervised-learning and thermodynamics prompts produce identical HF/TT question-list autocomplete behavior. Both HF and TT fail the requested French instruction by continuing a base-model prompt corpus; that is a checkpoint/prompt-format limitation, not TT-only drift. The detailed comparison is in `qualitative/verdict.md`.

## Runtime fallback audit

- Model: TTNN embedding, production optimized TP4 decoder stack, final norm, LM head, and softcap; no single-chip, replicated decoder, host decoder, or demo-layer branch.
- Residual/cache: generator owns standalone KV/page-table state; low-level callers may provide explicit co-owned external state; trace tables are private stable clones.
- Logits: optimized token-out never reads full logits. Readiness tests and `host_sampling_compat=True` may explicitly compose host logits.
- Sampling: measured greedy token-out uses the traced custom device sampler. Non-greedy and explicit host compatibility are not used in the optimized measurement.
- Feedback: the sampler output and next model input are the same persistent tensor; no host argmax or Python per-token feedback.
- Reset: both traces are released before cache clear, prefill allocation, sampler-mode change, or request reuse.
- Context: bounds are checked before device work and each replay; no silent truncation or aligned-length-only public contract.

## Commands and artifacts

All hardware commands used `LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH`, four P150b devices, and serialized watcher/profiler access.

```bash
pytest -q models/autoports/google_gemma_4_31b/tests/test_full_model_contract.py

python -m models.common.readiness_check.run_prefill_check \
  --model-dir models/autoports/google_gemma_4_31b \
  --reference models/autoports/google_gemma_4_31b/doc/full_model/readiness_aime24_plain.refpt \
  --mesh-device P150_X4 --fabric-config FABRIC_1D

python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/google_gemma_4_31b \
  --reference models/autoports/google_gemma_4_31b/doc/full_model/readiness_aime24_plain.refpt \
  --mesh-device P150_X4 --fabric-config FABRIC_1D

python -m models.common.readiness_check.run_autoregressive \
  --model-dir models/autoports/google_gemma_4_31b \
  --mesh-device P150_X4 --fabric-config FABRIC_1D --max-new-tokens 100

python models/autoports/google_gemma_4_31b/tests/run_full_model_qualitative.py
```

Primary compact artifacts:

- `readiness_aime24_plain.refpt` and `.metadata.json`
- `run_prefill_check.log`, `run_teacher_forcing.log`
- `autoregressive/`
- `qualitative/`
- `token_out_no_readback.json`
- `reduced_token_out_custom_greedy_perf.json`
- `reduced_token_out_final_perf.json`, `sampler_profile_summary.md`, and `perf/`
- `triage/`, repo-root `AUTOTRIAGE.md`, and the focused JUnit XML files in `doc/`
- `doc/full_model_reduced_final_watcher.xml`: source-current mixed-prefill/split-trace watcher closure

No vLLM integration was started. The `vllm_qualitative_outputs.json` filename is only the established readiness artifact schema.

## Limitations

- This base checkpoint/tokenizer exposes no chat template; instruction-like prompts can autocomplete training-corpus patterns instead of answering.
- Full-context cache provisioning makes first construction expensive; it is not a runtime single-chip or host fallback.
- Optimized production measurements are batch 1; batch two is tested at context 128, and full-context batch three is only a physical capacity upper bound.
- One TT qualitative story continuation repeats a corpus-style prompt sentence more than its HF control. The independent 100-token story is coherent and the mechanical checker passes, but the weaker style remains in the qualitative ledger.
- Stage 05 rejection policy remains binding: synchronous BFP8 CCL, persistent BF16-output decode CCL, BFP4 attention, 24-core packed MLP, Ring, fractured residual, rejected fused collective alternatives, larger prefill grids, and adapted block-sharded L1 prefill remain disabled.
