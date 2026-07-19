# Full-model work log

Date: 2026-07-19 UTC

## Starting state

- Branch: `mvasiljevic/model/meta-llama-llama-3.1-8b-instruct`
- Initial checkout HEAD: `b0f74d915449`
- Accepted optimized multichip decoder checkpoint: `c79dc655`
- Exact HF snapshot: `0e9e39f249a16976918f6564b8830bc894c89659`
- Target: P300 1x4 ring, TP=4, two links; no vLLM work.

## Implementation

1. Added the full embedding → 32-layer TP4 decoder → final norm → split LM
   head → sampling path in `tt/model.py`.
2. Added the standard readiness generator, lazy safetensor loading, explicit
   paged-cache/page/position state, non-aligned and chunked prefill, fixed
   sampling rows, mixed/inactive-row handling, device traces, host
   compatibility, and reset in `tt/generator.py`.
3. Extended the optimized multichip decoder with shared RoPE/position state,
   logical prompt lengths, chunked cache fill, device decode positions, and
   separate TILE-prefill/row-major-decode RoPE tables.
4. Added P300 readiness mesh support, exact chat rendering, explicit HF
   attention masks, and autoregressive chat-template support.
5. Added exact-weight reduced tests, watcher repro, full evidence collection,
   public prompt-128/generate-128 performance, and bounded Tracy capture.

## Autofix ledger

- Two-link sampler watcher initially asserted in
  `minimal_default_writer.cpp`. ELF/source triage proved an unconditional
  outward fabric-connection lookup on Linear edge workers. The exact upstream
  fix from `ff8ced342517621f470946d4c2e9472e4276d829` was applied. One-link mode
  was tested but rejected because it only routed around the bug and changed the
  accepted CCL policy. Standalone and canonical two-link watcher gates pass.
- Full decode initially measured 19.236 ms/token versus a 7.894 ms decoder
  lower bound. Tracy proved two 131,072-row RoPE untilizes per layer. Separate
  shared row-major decode tables reduce model trace to 8.197 ms/token and keep
  prefill on its existing TILE tables.
- Trace lifecycle audit found cache-identity/page-shape recapture could replace
  live IDs. Traces now release before incompatible recapture, host
  compatibility, sampling-topology changes, and teardown. `reset()` zeros the
  bound cache in place and preserves compatible trace IDs across requests.
- Stochastic sampling initially reused real seeds on every replay. It now uses
  real request seeds for the first visible draw and a stable `UINT32_MAX`
  device-advance sentinel thereafter. A fixed-logit trace test proves
  reproducible variation without per-token copies.
- `ttnn.sampling(k=1)` and local top-k both failed the semantically greedy
  contract on the TP4 padded vocabulary. The canonical path now performs local
  argmax and one packed FP32 rank-candidate gather, then selects the global
  winner on device. An adversarial 32-row oracle matches host and
  full-vocabulary force-argmax exactly; the force path remains comparison-only.
- A page-boundary reset repro exposed an undersized test page table: the test
  mapped `prompt+1` while issuing two decode writes. The generator now validates
  physical-page coverage before eager/initial traced mutation, and all replay
  evidence maps its complete horizon. Missing pages fail before cache writes.

## Final gates

- Fresh AIME24 chat reference: 184 prompt tokens, 100 continuation tokens,
  top-100 captured, artifact SHA-256 `3dc948d1...f4a42`.
- Prefill: top-1 86%, top-5 100%, top-100 100%.
- Teacher forcing: top-1 86%, top-5 100%, top-100 100%, decode 101.50 t/s/u.
- Public prompt-128/generate-128: TTFT 20.51 ms, token-out 110.45 t/s/u.
- Trace pair: 111.17 t/s/u; model 8.199 ms, sampler 0.795 ms (8.84%).
- Autoregressive: 100 HF and 100 TT tokens; coherent English math, no
  degeneration or wrong-language drift; ordinary first divergence at index 6.
- Final exact-weight comprehensive reduced test: pass.
- Final exact-weight canonical watcher test: pass with two links.
- Fallback source/runtime audit: pass with fallback exceptions enabled.
- Bounded Tracy capture: pass with no dropped-marker warning; retained compact
  exact-greedy op report records local reduce/argmax, packed rank-candidate
  gather, and device global winner selection. The earlier RoPE report shows no
  full-context RoPE untilize in decode.
- Full-context hardware capacity: 2,534,929,408 bytes/device after weights,
  4,816,630,784 after the full BFP8 cache; BF16 and BFP8 plans both fit the
  complete 131,072-token context with at least 26.5 GB/device headroom.
- Real near-context public path: one exact decoder layer processed a
  non-divisible 131,071-token prompt in 64 chunks, mapped all 2,048 pages, and
  populated/hashed the final K/V page. Public `generate(..., 1)` also accepts
  the exact 131,072-token maximum prompt without allocating an extra slot.
- Shared qualitative suite: all six exact-chat HF/TT prompt pairs pass manual
  review and the degeneration checker; prompt 0 repeats bit-identically with
  the same traces across reset.
- Current exact-greedy `tt-perf-report`: decode-signpost bounded report passes,
  names local reduce/argmax plus one candidate gather, and contains no
  131,072-row untilize. The reduced one-layer sampler ratio is intentionally
  not used as the production bottleneck measure; full 32-layer share is 8.84%.
- Several successful Python hardware commands emit nanobind reference-leak
  diagnostics only after results are written and the mesh is closed. These are
  process-teardown binding warnings, not a model fallback or device failure:
  each authoritative command exits zero, UMD completes its cluster destructor,
  and the final four-device health check is recorded separately. No stage
  correctness or performance result is taken after such a warning.
- The first rereview found that the metric-bearing performance/evidence and
  reduced-suite artifacts had been produced with the fallback exception guard
  disabled despite the audit claiming otherwise. The `$autofix` repair reran
  all three with `TTNN_CONFIG_OVERRIDES` explicitly enabled; all passed, the
  JSON metrics were refreshed, and the guarded log headers are authoritative.
  The adversarial 32-row sampler oracle was also corrected to the delivered
  131,072 padded vocabulary and rerun under the same guard.

## Final command ledger

The complete copy/paste command block and exact artifact destinations are in
`README.md`. The authoritative commands executed for this stage were:

```bash
export LLAMA_SNAPSHOT=/home/mvasiljevic/hf-cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
export LLAMA_AUTOPORT=models/autoports/meta_llama_llama_3_1_8b_instruct
export TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}'

python -m models.common.readiness_check.generate --hf-model "$LLAMA_SNAPSHOT" \
  --prompt-source aime24 --aime24-prompt-index 0 --chat-template \
  --gen-len 100 --top-k 100 \
  --output "$LLAMA_AUTOPORT/doc/full_model/artifacts/aime24_chat_100.refpt"

python -m models.common.readiness_check.run_prefill_check \
  --model-dir "$LLAMA_AUTOPORT" \
  --reference "$LLAMA_AUTOPORT/doc/full_model/artifacts/aime24_chat_100.refpt" \
  --mesh-device P300 --fabric-config FABRIC_1D_RING

python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir "$LLAMA_AUTOPORT" \
  --reference "$LLAMA_AUTOPORT/doc/full_model/artifacts/aime24_chat_100.refpt" \
  --mesh-device P300 --fabric-config FABRIC_1D_RING

python -m models.common.readiness_check.run_autoregressive \
  --model-dir "$LLAMA_AUTOPORT" --hf-model "$LLAMA_SNAPSHOT" \
  --prompt-file "$LLAMA_AUTOPORT/doc/full_model/artifacts/aime24_prompt.txt" \
  --mesh-device P300 --fabric-config FABRIC_1D_RING --max-new-tokens 100 \
  --chat-template --output-dir "$LLAMA_AUTOPORT/doc/full_model/artifacts/autoregressive"

python "$LLAMA_AUTOPORT/tests/full_model_qualitative.py" \
  --model-dir "$LLAMA_AUTOPORT" --hf-model "$LLAMA_SNAPSHOT" \
  --prompts-file models/common/readiness_check/vllm_prompts.txt \
  --output-dir "$LLAMA_AUTOPORT/doc/full_model/artifacts/qualitative_suite" \
  --max-new-tokens 100

python "$LLAMA_AUTOPORT/tests/full_model_contract_coverage.py" \
  --model-path "$LLAMA_SNAPSHOT" \
  --output "$LLAMA_AUTOPORT/doc/full_model/full_model_contract_coverage.json"

python "$LLAMA_AUTOPORT/tests/full_model_perf.py" --model-dir "$LLAMA_SNAPSHOT" \
  --reference "$LLAMA_AUTOPORT/doc/full_model/artifacts/aime24_chat_100.refpt" \
  --output "$LLAMA_AUTOPORT/doc/full_model/artifacts/full_model_perf.json"

python "$LLAMA_AUTOPORT/tests/full_model_evidence.py" --model-dir "$LLAMA_SNAPSHOT" \
  --prompt-file "$LLAMA_AUTOPORT/doc/full_model/artifacts/aime24_prompt.txt" \
  --output "$LLAMA_AUTOPORT/doc/full_model/artifacts/full_model_evidence.json" \
  --replay-iterations 100 --sampler-iterations 20

pytest -q "$LLAMA_AUTOPORT/tests/test_full_model.py" -m '' --disable-warnings
python models/common/readiness_check/check_degenerate_output.py \
  "$LLAMA_AUTOPORT/doc/full_model/artifacts/qualitative_suite" \
  --scope autoregressive --missing-artifacts critical \
  --json "$LLAMA_AUTOPORT/doc/full_model/artifacts/qualitative_suite/degenerate_report.json"

python -m tracy -p -r --check-exit-code \
  -o "$LLAMA_AUTOPORT/doc/full_model/tracy_exact_greedy" \
  "$LLAMA_AUTOPORT/tests/full_model_evidence.py" \
  --model-dir "$LLAMA_SNAPSHOT" \
  --prompt-file "$LLAMA_AUTOPORT/doc/full_model/artifacts/aime24_prompt.txt" \
  --output "$LLAMA_AUTOPORT/doc/full_model/artifacts/reduced_profile_capture.json" \
  --replay-iterations 1 --sampler-iterations 1 --override-num-layers 1

tt-perf-report \
  "$LLAMA_AUTOPORT/doc/full_model/tracy_exact_greedy/reports/2026_07_19_18_40_17/ops_perf_results_2026_07_19_18_40_17.csv" \
  --start-signpost 'start FULL_MODEL_REDUCED_DECODE' \
  --end-signpost 'stop FULL_MODEL_REDUCED_DECODE' --raw-op-codes --no-color \
  --csv "$LLAMA_AUTOPORT/doc/full_model/tracy_exact_greedy/decode_perf_report.csv" \
  --summary-file "$LLAMA_AUTOPORT/doc/full_model/tracy_exact_greedy/decode_op_summary"

tt-smi -s
```

The exact Tracy capture and signpost-filtered `tt-perf-report` commands are
also retained in `README.md`; their final artifacts are under
`tracy_exact_greedy/` and the console log is
`logs/tracy_exact_greedy_final.log`.

## Commits

The final fresh `$stage-review` rereview returned `CLEAN-PASS` after the
fallback-guard evidence repair.

- Implementation, tests, guarded logs, compact profiler reports, and evidence:
  `dcbb3a32ad0` (`Add Llama 3.1 8B TP4 full model`).
- This work-log ledger is committed immediately after that implementation
  commit; both SHAs are reported in the final handoff. No push is performed.
- The three raw profiler CSV captures exceed the repository's 500 KB artifact
  hook and therefore remain repo-local ignored files. Their compact
  signpost-filtered CSV/TXT/PNG reports are committed and are the canonical
  profiler artifacts.
