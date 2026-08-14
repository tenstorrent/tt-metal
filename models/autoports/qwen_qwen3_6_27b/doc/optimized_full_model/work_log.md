# Optimized full-model work log

## Scope and baseline audit

Stage target: Qwen/Qwen3.6-27B optimized full model on 4x Blackhole p300c,
1x4 TP ring. Started from completed full-model commit `f0727de9616`; vLLM and a
broad datatype frontier were explicitly out of scope. `tt-smi -ls --local` and
a 1x4 `FABRIC_1D_RING` mesh smoke passed before device work.

Inherited policy and rejection ledger were audited from optimized decoder,
multichip decoder, and full-model artifacts. The selected per-layer warmed
medians give the lower bound:

```text
16 * 0.593793571 ms + 48 * 0.899718165 ms = 52.687169 ms/token
1 / 0.052687169 = 18.980 t/s/u
```

Fresh completed-path baseline command (serialized hardware run):

```bash
python -m models.autoports.qwen_qwen3_6_27b.tests.full_model_perf \
  --prompt models/autoports/qwen_qwen3_6_27b/doc/full_model/aime24_chat_prompt.txt \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_full_model/artifacts/baseline_full_force_argmax_no_readback_b1_s128_g128.json \
  --prompt-tokens 128 --decode-tokens 128 --force-argmax-greedy --feedback-overwrite-probe
```

Result: 4036.679581 ms TTFT, 17.493443 t/s/u, semantic greedy exact,
128 trace replays and zero token/position/page-table refreshes. This baseline
was numerically correct but used the now-disallowed full-vocab force-argmax
shortcut.

## Sampler experiments and AutoFix

1. Unpadded 62,080-wide local TopK: 12.667 ms reduced token-out, slow
   single-core factory (`baseline_split_unpadded_reduced.json`).
2. Pad to 65,536: 13.618 ms; refuted because the multicore factory requires a
   power-of-two width strictly below 65,535
   (`candidate_split_padded_reduced.json`).
3. AutoDebug isolated that factory constraint and the missing invalid-vocab
   mask (`AUTODEBUG.md`).
4. AutoFix implemented two 32,768 chunks, explicit chunk-base IDs, 64-to-32
   merge through positions plus device gather, and sharded invalid-vocab mask.
   It refuted use of `indices_tensor` in the single-core merge because that
   factory returns positional indices (GH #36329). Details: `AUTOFIX.md`.
5. Focused all-rank probe passed boundary IDs and invalid-ID exclusion.
   Static planner tests and Python compilation passed.

Reduced repaired result: 3.451 ms, 289.752 t/s/u over 64 replays, a 3.67x
improvement over the old split path (`candidate_split_two_stage_reduced.json`).

## Profiler and Watcher

Profiler and Watcher were run in separate processes as required.

Profiler command:

```bash
python -m tracy -r -p -m models.autoports.qwen_qwen3_6_27b.tests.full_model_perf \
  --prompt models/autoports/qwen_qwen3_6_27b/doc/full_model/aime24_chat_prompt.txt \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_full_model/tracy/split_sampler_reduced/perf.json \
  --prompt-tokens 128 --decode-tokens 1 --layer-indices 0 3 \
  --profile-only-decode --feedback-overwrite-probe
```

`tt-perf-report` was bounded by `FULL_MODEL_DECODE` and
`FULL_MODEL_DECODE_END`. Raw/provenance CSV and both machine and human reports
are under `tracy/split_sampler_reduced/`. Conclusions are recorded in README.

Watcher command:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
  python models/autoports/qwen_qwen3_6_27b/tests/full_model_mixed_slots.py \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_full_model/artifacts/mixed_slots_split_watcher.json
```

The scoped `WATCHER_DISABLE_ETH` workaround is required because Blackhole
ACTIVE_ETH Watcher firmware exceeds its local firmware size; Tensix and host
Watcher coverage remained enabled. All four devices were checked every 10 s
and the run passed `FULL_MODEL_MIXED_SLOTS_OK [198,220] [68,63]
INACTIVE_KV_EXACT RESET_REUSE_OK` with no Watcher error.

## Accuracy and qualitative refresh

```bash
python -m models.common.readiness_check.run_prefill_check \
  --model-dir models/autoports/qwen_qwen3_6_27b \
  --reference models/autoports/qwen_qwen3_6_27b/doc/full_model/readiness_aime24_chat.refpt \
  --mesh-device P300X2 --fabric-config FABRIC_1D_RING

python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/qwen_qwen3_6_27b \
  --reference models/autoports/qwen_qwen3_6_27b/doc/full_model/readiness_aime24_chat.refpt \
  --mesh-device P300X2 --fabric-config FABRIC_1D_RING

python -m models.common.readiness_check.run_autoregressive \
  --model-dir models/autoports/qwen_qwen3_6_27b \
  --hf-model /huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 \
  --prompt-file models/autoports/qwen_qwen3_6_27b/doc/full_model/aime24_chat_prompt.txt \
  --mesh-device P300X2 --fabric-config FABRIC_1D_RING \
  --output-dir models/autoports/qwen_qwen3_6_27b/doc/optimized_full_model/autoregressive_final \
  --max-new-tokens 100
```

Prefill passed 92/100 top-1 and 100/100 top-5/top-100. Teacher forcing
initially found a readiness-harness regression: the generic runner did not opt
into the optimized generator's explicit host-compatibility boundary. The
runner now enables it only when that keyword exists; four unit tests pass.
The rerun passed 97/100 top-1, 100/100 top-5/top-100, 5129.57 ms TTFT and
6.98 traced t/s/u. Autoregressive produced 100 tokens on both sides; the
degeneracy checker passed with zero adjacent duplication.

The final-code shared qualitative suite was run separately with:

```bash
python models/autoports/qwen_qwen3_6_27b/tests/full_model_qualitative.py \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_full_model/artifacts/full_model_qualitative_50.json \
  --max-new-tokens 50
```

All six matched HF/TT chat-template cases produced 50 TT tokens. Three match
HF throughout; first divergence for the others is token 1, 4, or 28. Manual
review found coherent, English, task-specific, non-repetitive continuations
with no wrong-language drift or control-token leakage. Exact command output is
`logs/full_model_qualitative_50.log`.

## Final default measurement and gap closure

```bash
python -m models.autoports.qwen_qwen3_6_27b.tests.full_model_perf \
  --prompt models/autoports/qwen_qwen3_6_27b/doc/full_model/aime24_chat_prompt.txt \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_full_model/artifacts/final_default_split_no_readback_b1_s128_g128.json \
  --prompt-tokens 128 --decode-tokens 128 --feedback-overwrite-probe
```

Default result: 4041.510380 ms TTFT, 17.637144 t/s/u, 56.698457 ms/token,
semantic greedy exact, sampler trace live, 128 model/sampler replays, zero
token/position/page-table host refreshes. Against the 52.687169 ms layer-stack
bound, terminal work is 4.011288 ms or 7.61%. No further split was required.

## Checklist and limitations

- Full-model sharding: TP4 embeddings/decoder/LM-head retained; only candidate
  logits/IDs are gathered for sampling.
- CCL: inherited decoder persistent buffers/CCL policy is preserved. Sampler
  candidate-gather buffers and compiled programs are stable allocations owned
  by the persistent captured sampler trace; cached `TT_CCL` semaphores are
  reused. Qwen does not expose the Galaxy-only keyed `line_all_gather` API.
  No rejected replicated stream is used.
- Matmuls/program configs/kernels: selected decoder configs preserved;
  profile confirms sharded BFP8 LM head and multicore split TopK.
- Layout: replicated BF16 inter-layer residual boundary preserved exactly.
- Persistent inputs: token, position/RoPE, cache, active mask, and page table
  survive replay; page table refreshes only when changed.
- Host boundaries: none in measured token-out; teacher forcing is explicitly
  separate compatibility evidence.
- Non-aligned prompts/fixed slots: refreshed S65/S63 Watcher pass.
- Capability: no reduction; `context_contract.json` remains at B1 C262144 and
  retains the established B32 physical bracket.
- Known limitation: ACTIVE_ETH Watcher firmware size requires the documented
  `TT_METAL_WATCHER_DISABLE_ETH=1` workaround; this is instrumentation-only.
- No vLLM integration and no datatype Pareto search performed.

## Review and local commits

Independent stage review and rereview completed with `clean-pass`; required
work is none (`STAGE_REVIEW.md`). Stage implementation, tests, profiler CSVs,
logs, and evidence were committed locally as `37b19aa9cc4`. This documentation
handoff is a following local-only commit so that the implementation SHA can be
recorded exactly. Nothing was pushed.
