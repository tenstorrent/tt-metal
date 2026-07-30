# Llama 3.1 8B Instruct Full Model

Status: complete as of 2026-06-15 for the repo-local full-model autoport stage.
No vLLM integration work was started.

## Implementation

`tt/model.py` implements `Llama31_8B_InstructFullModel` and assembles:

- BF16 token embedding and RoPE setup from the local HF checkpoint.
- 32 `MultiChipDecoder` layers on the target T3K `1x8` Ring mesh.
- Final RMSNorm and split BF8 LM head.
- Paged KV caches owned by the full model and exposed through
  `owned_kv_cache()`.
- Canonical split-sampling args with `max_top_k=32`, padded vocab 128256,
  `sampling_all_gather_axis=1`, and `force_argmax=False`.

`tt/generator.py` implements the Metal readiness `build_generator(model_dir,
mesh_device, **kwargs)` contract plus explicit `prefill_forward`,
`decode_forward`, and `generate(..., enable_trace=True)`.

The full model preserves the optimized multichip decoder policy:
`llama31_8b_t3k_1x8_tp8_bfp4_attn_bfp4_mlp_bfp8_act_decode_v2`.
Decoder layers keep BFP8 activations and KV cache, BFP4 attention and MLP
weights, LoFi MLP math, async CCL hidden all-gathers and MLP reduce-scatter,
and width-sharded L1 residuals inside decode layers. The wrapper does not add a
single-chip, replicated-host, host-logits, or less optimized fallback.

## Readiness

Fresh AIME24 chat-template reference:

```bash
timeout 7200s python_env/bin/python -u -m models.common.readiness_check.generate \
  --hf-model meta-llama/Llama-3.1-8B-Instruct \
  --prompt-source aime24 \
  --chat-template \
  --gen-len 100 \
  --top-k 100 \
  --output models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/aime24_chat_template_100_top100.refpt
```

Reference metadata: `readiness_v1`, HF model
`meta-llama/Llama-3.1-8B-Instruct`, prompt length 184, generated length 100,
top-k 100.

Prefill:

```bash
timeout 7200s python_env/bin/python -u -m models.common.readiness_check.run_prefill_check \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --reference models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/aime24_chat_template_100_top100.refpt \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING
```

Result: top1 `90/100`, top5 `100/100`, top100 `100/100`.

Teacher forcing:

```bash
timeout 7200s python_env/bin/python -u -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --reference models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/aime24_chat_template_100_top100.refpt \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING
```

Result: top1 `92/100`, top5 `100/100`, top100 `100/100`,
TTFT `1094.60 ms`, decode `22.18 t/s/u`, e2e `17.99 t/s/u`.

Autoregressive:

```bash
timeout 7200s python_env/bin/python -u -m models.common.readiness_check.run_autoregressive \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --hf-model meta-llama/Llama-3.1-8B-Instruct \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING \
  --max-new-tokens 100 \
  --output-dir models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/autoregressive_story_100
```

HF and TT each produced 100 tokens. The TT completion is coherent English, has
no wrong-language drift, and does not collapse into repetition. It diverges
after the early shared prefix but remains a plausible story continuation. The
degeneracy check reported no findings:

```bash
python_env/bin/python models/common/readiness_check/check_degenerate_output.py \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --missing-artifacts critical \
  --scope autoregressive \
  --json models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/autoregressive_degenerate_report.json
```

Degeneracy metrics: adjacent duplication `0.0`, trigram loop fraction `0.038`,
HF/TT token agreement `29/100`.

## Trace Evidence

Artifact:
`models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/token_out_trace_evidence.json`.

Full 32-layer token-out run, story prompt, 60 prompt tokens, 100 generated
tokens:

| Metric | Value |
| --- | ---: |
| TTFT | 616.64 ms |
| Decode including first model/sampling trace capture | 49.65 t/s/u |
| Steady replay decode, excluding first trace capture | 69.21 t/s/u |
| Steady replay latency | 14.45 ms/token |

Trace counters for the 99 post-prefill decode tokens:

| Counter | Value |
| --- | ---: |
| Model trace captures / replays | 1 / 99 |
| Sampling trace invocations | 99 |
| Token input host copies | 1 |
| Position host copies | 1 |
| RoPE index host copies | 1 |
| Page-table host copies | 1 |
| Position / RoPE device increments | 99 / 99 |
| RoPE matrix refreshes / hidden refreshes | 0 / 0 |
| Sampling force argmax | false |
| Sampling max top-k | 32 |

Focused probe assertions all passed:

- sampled token is written into the persistent decode token input;
- persistent current position increments on device;
- reused trace reset copies token/position/RoPE/page table once;
- unchanged page table is not recopied;
- changed page table is recopied once and marked as a changed-only refresh;
- top-k/top-p smoke with `temperature=0.7`, `top_k=8`, `top_p=0.9` keeps
  `force_argmax=False` and writes the sampled token back to the persistent
  decode token input.

## Performance Profile

Reduced full-model profiler harness:
`doc/full_model/reduced_profile.py`.

The profiler variant uses one real layer with the same real tensor shapes,
tokenizer prompt shape, page-table/KV shape, optimized multichip decoder
policy, final norm, split LM head, and split-sampling trace behavior. The full
all-layer model was not profiled with Tracy.

Command:

```bash
python_env/bin/python -m tracy -r -p -v --check-exit-code \
  -o models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/tracy/reduced_profile/.logs \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/reduced_profile.py \
  --output-json models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/reduced_profile_summary.json
```

`tt-perf-report` commands were run over
`tracy/reduced_profile/reduced_profile_ops_perf_results.csv` with the
`PERF_REDUCED_PREFILL` / `PERF_REDUCED_PREFILL_END` and
`PERF_REDUCED_TOKEN_OUT_DECODE` / `PERF_REDUCED_TOKEN_OUT_DECODE_END`
signposts.

Reduced-profile host timings:

| Window | Host timing |
| --- | ---: |
| One-layer prefill | 86.50 ms |
| One-layer token-out decode min / avg | 1.654 / 1.764 ms |

Reduced-profile `tt-perf-report` summary:

| Window | Merged device ops | Device time sum | Op-to-op gap sum |
| --- | ---: | ---: | ---: |
| Prefill | 42 | 1643.152 us | 4909.697 us |
| Token-out decode | 61 | 1368.926 us | 8455.678 us |

Token-out decode top categories by device time:

| Category | Device time |
| --- | ---: |
| Width-sharded matmuls | 541.38 us |
| `TopKDeviceOperation` | 154.26 us |
| `AllGatherDeviceOperation` | 128.55 us |
| `SamplingDeviceOperation` | 63.89 us |

The sampler is present, but does not dominate the reduced token-out decode
window. The path uses canonical split sampling, not force argmax.

Layer-stack lower bound from the optimized multichip decoder:
`32 * 0.397455879 ms = 12.72 ms/token` min and
`32 * 0.401623140 ms = 12.85 ms/token` avg. Full-model steady token-out replay
is `14.45 ms/token`, about `1.73 ms/token` over the min layer-stack lower
bound. The residual gap is full-model terminal work: embedding boundary,
final norm, split LM head, sampling, trace orchestration, caller-visible token
readback, and remaining CCL/reshape overhead.

## Fallback Audit

- Model path: `build_generator` rejects non-`1x8` meshes and builds the full
  wrapper from `MultiChipDecoder`, not the single-chip decoder.
- Cache ownership: the full model allocates and owns per-layer paged KV caches;
  the generator exposes them through the readiness contract and does not rebuild
  caches per token.
- Host-logit boundary: host logits are read only for prefill readiness and the
  first token after prefill. Traced decode does not read full logits back to the
  host.
- Sampling: decode uses `SamplingGenerator` internal trace with `tt_out_tok`
  bound to the persistent decode token tensor. `force_argmax=False`.
- Reset behavior: reset clears generator counters and host-side page-table /
  position tracking. A reused decode trace resets token/current-position/RoPE
  index/page-table tensors once before replay; unchanged page tables are not
  recopied.
- Hot loop: token input, current position, RoPE index, masks, and page table are
  not rebuilt or copied from host per generated token in the unchanged
  page-table free-running path.

Known limitation: the first sampling trace capture emits the TTNN allocator
warning about allocating buffers while a model trace is active. The run succeeds,
trace counters are correct, and reduced/full evidence shows traced split
sampling behavior, but the warning is recorded for follow-up cleanup.

## Artifacts

- `aime24_chat_template_100_top100.refpt`
- `autoregressive_story_100/hf_completion.txt`
- `autoregressive_story_100/tt_completion.txt`
- `autoregressive_story_100/autoregressive_meta.json`
- `autoregressive_degenerate_report.json`
- `token_out_trace_evidence.py`
- `token_out_trace_evidence.json`
- `token_out_trace_evidence_stdout.txt`
- `reduced_profile.py`
- `reduced_profile_summary.json`
- `tracy/reduced_profile/reduced_profile_ops_perf_results.csv`
- `tracy/reduced_profile/reduced_profile_profile_log_device.csv`
- `tracy/reduced_profile/reduced_prefill_perf_report.{txt,csv}`
- `tracy/reduced_profile/reduced_prefill_perf_report_stacked.{csv,png}`
- `tracy/reduced_profile/reduced_token_out_decode_perf_report.{txt,csv}`
- `tracy/reduced_profile/reduced_token_out_decode_perf_report_stacked.{csv,png}`
- `tracy/reduced_profile/reduced_token_out_decode_perf_report_per_device.{txt,csv}`
- `tracy/reduced_profile/reduced_token_out_decode_perf_report_per_device_stacked.{csv,png}`
- `tracy/reduced_profile/tracy_ops_data.csv`
- `tracy/reduced_profile/tracy_ops_times.csv`
