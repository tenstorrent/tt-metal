# Full-model work log

Date: 2026-07-20 UTC. Starting repository HEAD: `588a3bb83ef`. Exact HF revision: `34bb99a889fe0426412da3dd2b46e6f64c8fd003`.

## Scope and implementation

Implemented the complete 40-layer autoregressive model in `tt/model.py` and the standard Metal readiness generator in `tt/generator.py`, starting from the completed optimized multichip decoder. The implementation adds TP embedding, all decoder layers, final norm, untied TP LM head, cache lifecycle, arbitrary-length chunked prefill, serving-oriented explicit state, fixed/inactive slots, separate model/sampler traces, direct device token feedback, and an explicit host-sampling compatibility mode.

The decoder was extended for 32-token page blocks, chunked prefill, independent cache/rotary positions, inactive cache rows, and rounded SDPA page validation. Canonical Sampling1D gained the exact split-greedy packet contract used by this model. Readiness mesh discovery gained the four-device `P300X2` shape. Trace-replay debug support required the Watcher run-state and all-gather writer fixes recorded in the source diff; both came from the already clean-reviewed sibling full-model implementation at commit `a0cf84aa429` and were rebuilt locally with:

```bash
ninja -C build_Release -j16 tt_metal
cmake --install build_Release
```

## Device discipline

All hardware commands were serialized with `/tmp/tt-device.lock`. Initial `tt-smi` health showed four p300c devices. A 1x4 `FABRIC_1D_RING` mesh smoke passed before model execution. All normal gates used:

```bash
export TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}'
```

Watcher and Tracy ran separately. No device hang, reset, unhealthy board, NoC corruption, assertion, or fallback occurred, so `$autofix` had no failed or unmet gate to repair.

## Reproduction commands

Common paths used below:

```bash
MODEL=models/autoports/tiiuae_falcon3_10b_base
REF=$MODEL/doc/full_model/results/aime24_plain_100.refpt
HF=/home/mvasiljevic/hf-cache/hub/models--tiiuae--Falcon3-10B-Base/snapshots/34bb99a889fe0426412da3dd2b46e6f64c8fd003
OUT=$MODEL/doc/full_model/results
CACHE=/tmp/falcon3-full-model-cache
```

Fresh reference generation first probed the requested chat rendering and confirmed that the exact tokenizer raises because `chat_template` is unset:

```bash
TRANSFORMERS_OFFLINE=1 python -m models.common.readiness_check.generate --hf-model "$HF" --prompt-source aime24 --chat-template --aime24-prompt-index 0 --gen-len 100 --top-k 100 --output "$REF"
TRANSFORMERS_OFFLINE=1 python -m models.common.readiness_check.generate --hf-model "$HF" --prompt-source aime24 --aime24-prompt-index 0 --gen-len 100 --top-k 100 --output "$REF"
```

The first command failed before writing the reference with `ValueError: tokenizer.chat_template is not set`; the second generated the committed native-completion reference. This is why the artifact is named `aime24_plain_100.refpt`.

Canonical readiness:

```bash
python -m models.common.readiness_check.run_prefill_check --model-dir "$MODEL" --reference "$REF" --mesh-device P300X2 --fabric-config FABRIC_1D_RING
python -m models.common.readiness_check.run_teacher_forcing --model-dir "$MODEL" --reference "$REF" --mesh-device P300X2 --fabric-config FABRIC_1D_RING
python -m models.common.readiness_check.run_autoregressive --model-dir "$MODEL" --hf-model "$HF" --mesh-device P300X2 --fabric-config FABRIC_1D_RING --max-new-tokens 100 --output-dir "$OUT/autoregressive"
```

Full-model evidence:

```bash
python "$MODEL/tests/full_model_contract_coverage.py" --model-dir "$MODEL" --output "$OUT/full_model_contract_coverage.json" --weight-cache-path "$CACHE"
python "$MODEL/tests/full_model_evidence.py" --model-dir "$MODEL" --reference "$REF" --output "$OUT/full_model_evidence.json" --replay-iterations 128 --sampler-iterations 20 --prompt-length 128 --weight-cache-path "$CACHE"
python "$MODEL/tests/full_model_batch32.py" --model-dir "$MODEL" --reference "$REF" --output "$OUT/full_model_batch32.json" --weight-cache-path "$CACHE"
python "$MODEL/tests/full_context_coverage.py" --model-dir "$MODEL" --output "$OUT/full_context_coverage.json" --weight-cache-path "$CACHE"
python "$MODEL/tests/full_model_capacity.py" --model-dir "$MODEL" --output "$OUT/full_model_capacity.json" --weight-cache-path "$CACHE"
```

Qualitative and control:

```bash
python "$MODEL/tests/full_model_qualitative.py" --model-dir "$MODEL" --hf-model "$HF" --output-dir "$OUT/qualitative_suite" --max-new-tokens 100 --weight-cache-path "$CACHE"
python -m models.common.readiness_check.check_degenerate_output "$OUT/qualitative_suite" --scope autoregressive --missing-artifacts critical --json "$OUT/qualitative_suite/degenerate_output.json"
python "$MODEL/tests/full_model_qualitative_control.py" --model-dir "$MODEL" --model-path "$HF" --output "$OUT/qualitative_suite/haiku_control.json" --max-new-tokens 100 --weight-cache-path "$CACHE"
python -m models.common.readiness_check.check_degenerate_output "$OUT/autoregressive" --scope autoregressive --missing-artifacts critical --json "$OUT/autoregressive/degenerate_output.json"
```

Like-for-like depth attribution:

```bash
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
python "$MODEL/tests/full_model_depth_sweep.py" \
  --model-dir "$MODEL" --reference "$REF" \
  --depths 1,2,4,8,16,40 --iterations 128 --max-context-len 32768 \
  --output "$OUT/full_model_depth_sweep.json" --weight-cache-path "$CACHE"
```

Watcher and reduced profile:

```bash
TT_METAL_WATCHER=120 TT_METAL_WATCHER_DISABLE_ETH=1 python "$MODEL/tests/full_model_sampler_watcher.py" --output "$OUT/full_model_sampler_watcher.json"
TT_METAL_WATCHER=120 TT_METAL_WATCHER_DISABLE_ETH=1 python "$MODEL/tests/full_model_evidence.py" --model-dir "$MODEL" --reference "$REF" --output "$OUT/full_model_watcher.json" --replay-iterations 128 --sampler-iterations 20 --prompt-length 128 --weight-cache-path "$CACHE"
TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r -v \
  -o "$MODEL/doc/full_model/profile/reduced" \
  "$MODEL/tests/full_model_profile.py" --model-dir "$MODEL" --reference "$REF" \
  --output "$OUT/full_model_profile.json" --weight-cache-path "$CACHE"
tt-perf-report "$MODEL/doc/full_model/profile/reduced/reports/2026_07_20_04_23_57/ops_perf_results_2026_07_20_04_23_57.csv" \
  --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END --no-color \
  --csv "$MODEL/doc/full_model/profile/reduced/prefill_perf_report.csv" \
  --summary-file "$MODEL/doc/full_model/profile/reduced/prefill_summary"
tt-perf-report "$MODEL/doc/full_model/profile/reduced/reports/2026_07_20_04_23_57/ops_perf_results_2026_07_20_04_23_57.csv" \
  --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END --no-color \
  --csv "$MODEL/doc/full_model/profile/reduced/decode_perf_report.csv" \
  --summary-file "$MODEL/doc/full_model/profile/reduced/decode_summary"
```

## Results

- Fresh AIME24 reference: exact tokenizer/revision, 155 prompt tokens, 100 generated tokens, top-k 100. The base tokenizer has no chat template; metadata records the failed probe and native completion decision.
- Prefill: top-1 91/100, top-5 99/100, top-100 100/100.
- Traced teacher forcing: top-1 91/100, top-5 99/100, top-100 100/100.
- Standard 128+128 all-40-layer workload: cold TTFT 2444.08 ms, warm TTFT 34.99 ms, device trace 52.10 t/s/u, caller-visible token-out 51.98 t/s/u, teacher-forcing trace 52.10 t/s/u.
- Sampling: exact common-sampler agreement; split greedy selected at 0.8035 ms/call. Sampling was 4.14% of full token-out time.
- Context: 32767-token public prompt padded to 32768 and executed all 40 layers; last page populated on all 320 K/V tensors.
- Capacity: batch 32 x context 32768 cache allocated with the full model and left 4.12 GB/device free.
- Batch 32: exact repeat, permutation, remap, inactive-row, and batch-1-control gates passed.
- Trace: direct token feedback, device positions, changed/unchanged/restored page tables, safe reset, and zero steady per-token host-copy deltas passed.
- Qualitative: canonical story and all six shared prompts read; no degeneracy, wrong-language drift, token leak, or trace divergence.
- Watcher: sampler and complete evidence workloads passed; no TENSIX error. Ethernet Watcher was disabled because its instrumented firmware image exceeds the watcher buffer.
- Reduced Tracy profile passed; official performance remains the all-40-layer trace measurement.
- Review remediation depth sweep passed at 1/2/4/8/16/40 real layers with full 32K cache and 128+128 workload. The model-trace fit is `0.276425 + 0.472178 * layers` ms (`R²=0.9999998875`); the 40-layer fit residual is -0.00157 ms. The inherited 0.318242 ms sequence-17 layer result was therefore a non-like-for-like provisional bound, not a stack regression. Sampling is 0.79377 ms; queued trace orchestration is about 0.0020 ms and caller observation 0.0456 ms.

## Artifacts and provenance

`results/runner_provenance.json` records source, result, log, and profile hashes. `doc/context_contract.json` contains the recomputed weight-plus-KV capacity evidence and preserves the 32768-token context. The first independent review returned `more-work-needed` solely for missing lower-bound attribution. The corrected full-context depth runner/result/log closed that finding. A fresh independent rereview returned `clean-pass` after recalculating the fit and cost decomposition, checking all recorded hashes, parsing all 142 JSON artifacts and 22 relevant Python files, and auditing the full user contract. The initial and final reports are retained in `stage_review_initial.md` and `stage_review_final.md`.

Final host-only checks passed: `python -m py_compile` on the implementation and evidence runners, Black `--check --fast` on all 15 changed Python files, `git diff --check`, and the three focused teacher-forcing readiness unit tests.

The implementation and documentation checkpoint SHAs are appended after the isolated local commits; nothing is pushed.

The pre-existing unrelated modification to `.agents/skills/forge-functional-decoder-from-ir/SKILL.md` was preserved and is excluded from this stage.
