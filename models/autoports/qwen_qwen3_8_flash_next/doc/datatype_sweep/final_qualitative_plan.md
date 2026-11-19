# Final selected-config validation and blind review plan

Prepared by CPU-only inspection on 2026-11-16. This is a plan, not a validation result or a precision-selection/P1 verdict. No TTNN import, device access, generation, cached-output mutation, or runtime edit was performed for this audit. Run the device command only after P1 and selection finish and the parent owns the four devices exclusively.

## Pinned HF controls are reusable

Use `doc/full_model/qualitative/` relative to this model directory as the control source. Its `hf_status.json` is complete; `hf_validation.json` and `prompt_validation.json` record validation. `run_controls.py` uses the original full-48-layer mmap HF loader in `tests/full_model_hf_reference.py`, with CPU checkpoint BF16, eager attention and HF cache. Reusing these outputs avoids a concurrent CPU HF run during device measurements.

The pinned checkpoint is `Qwen/Qwen3.8-Flash-Next@812d96e10a0dd82b69d749e68693698cf6c36daa`, snapshot `/home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/812d96e10a0dd82b69d749e68693698cf6c36daa`. A fresh local-tokenizer-only check during this audit established:

- The current `Qwen3_5Tokenizer` renders all six saved message lists exactly, using `apply_chat_template(add_generation_prompt=True)` with default thinking. Saved token IDs and decode roundtrips match; lengths are 60, 67, 75, 61, 66 and 62.
- Saved user messages match the currently installed `runtime/readiness_check/vllm_prompts.txt` exactly. Roles remain as recorded; the checkpoint template supplies its default system text.
- There are 66 HF rows: six greedy controls and 60 sampled controls, seeds 0–9 for each prompt. Sampled settings match temperature 1.0, top-p 0.95, top-k 20, repetition penalty 1.0 and 128 new tokens.
- EOS IDs `[248046, 248044]` match the pinned `generation_config.json`. Both sides preserve raw tokens and trim logical output at the first configured EOS, inclusive.
- `doc/full_model/qualitative/hf_outputs.json` is byte-identical to `doc/optimized_full_model/final_validation_v2/qualitative/hf_outputs.json`.

The checked source records are `prompts.json`, `qualitative_prompt_format.json`, `stopping_config.json`, `hf_outputs.json` and the snapshot tokenizer/generation configuration. The final driver copies these controls and its frozen `rubric.json`, recording each source/copy pair and byte identity. Recheck provenance if checkpoint, tokenizer, suite, template or generation settings change; prior-stage TT ratings/verdicts are not the new selected-config verdict.

## Final device run

Use the normal construction path after installing `selected_precision_config.json` as its default. The driver calls `acceptance.build_generator(..., max_seq_len=262144, max_batch_size=1)` without a precision override and records `runtime_precision`; compare that record with the selected artifact before accepting the result. Do not override precision through an experiment environment setting. This command is a validation run, not the post-selection token-out performance benchmark.

From the checkout root, with the accepted device environment and device-usage preflight already established:

```bash
export TT_MODEL_BRINGUP_ROOT=/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24
export TT_AUTODEBUG_ROOT=/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.8
BRINGUP_EXPORTS=$(python_env/bin/python "$TT_MODEL_BRINGUP_ROOT/scripts/environment.py") || exit
eval "$BRINGUP_EXPORTS"
export PYTHONPATH="$PWD:$PYTHONPATH"
export HF_HOME=/home/ttuser/qwen38-flash/hf
export TT_VISIBLE_DEVICES=0,1,2,3
export TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto
export TT_METAL_TRACE_ALLOC_TRACKING=1
export TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
set -o pipefail
timeout --signal=INT --kill-after=60 14400 python_env/bin/python -m \
  models.autoports.qwen_qwen3_8_flash_next.tests.run_optimized_full_model_validation \
  --stage datatype_sweep \
  --output models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/final_validation_v1.json \
  --accuracy --resume --qualitative --context-smoke \
  2>&1 | tee models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/final_validation_v1.log
```

Do not enable profiler or watcher instrumentation for the final measurement regime. Use a new versioned stem if this destination already exists. If a process fails or hangs, follow `$tt-device-usage` evidence/recovery rules before retrying; do not overwrite the failed run.

Expected artifacts beneath `doc/datatype_sweep/final_validation_v1/`:

- `run_prefill_check.json`, `run_teacher_forcing.json`: main `readiness_aime24_chat.refpt` checks. Explicitly check the selected full-model top-1/top-5 gate against sweep thresholds; the parent has now added top-1 ≥0.90 beside top-5 ≥0.98 and top-100 =1.0 to this harness.
- `prefill_resume_check.json`: 257-token non-aligned prompt, block size 32, installed controls and unchanged thresholds. A failure remains work.
- `context_smoke.json`: non-aligned 32,769-token prefill plus two-token decode using full 262,144 capacity, finite logits and live device-feedback trace replay. This satisfies the 32K–64K smoke option only when selected KV dtype/layout is unchanged. A changed selected KV dtype/layout still requires the separate max-context run and recomputed context contract.
- `qualitative/tt_outputs.json`: six greedy and 60 sampled outputs, exact saved chat token IDs, common device sampling, traced generation, explicit per-request sampler reset/seed. `qualitative/` also contains the copied HF controls and prompt/rubric metadata.
- The top-level JSON records precision, commands, hardware/mesh, trace counters, memory and runtime audit. Audit must report all required bodies executed with no forbidden fallback/host conversion.

The optional `--autoregressive` phase is a separate 128-token AIME continuation stress check; it is not needed to replace or supplement a missing shared chat suite. Do not describe a raw-completion stress result as chat-quality evidence.

## Blind scoring: 10 matched seeds per prompt and side

Keep the existing frozen rubric and failure margin. It scores mechanical repetition/doubled tokens, gibberish, materially off-topic/prompt-echo responses and wrong-language drift. TT exceeds the per-prompt margin only when its failure count is more than two higher than HF out of ten. No EOS before the matched cap, cap truncation, checkpoint thinking and incompleteness solely due to the cap are excluded. English reasoning before a requested French answer is allowed.

Prepare the 120-example label-hidden packet with the existing implementation, redirecting its root rather than executing its source CLI (whose default root is the original full-model directory):

```bash
python_env/bin/python - <<'PY'
import importlib.util
from pathlib import Path
model = Path('models/autoports/qwen_qwen3_8_flash_next').resolve()
source = model / 'doc/full_model/qualitative/blind_compare.py'
spec = importlib.util.spec_from_file_location('datatype_blind_compare', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.ROOT = model / 'doc/datatype_sweep/final_validation_v1/qualitative'
module.prepare(module.ROOT / 'tt_outputs.json')
PY
```

Use a fresh independent agent with no inherited conversation. Provide only `blind_packet.json`, `rubric.json`, `blind_ratings_template.json`, and an authorized output path `blind_ratings.json` in that directory. Instruct it to read no labels, other outputs, previous verdicts, or comparison code; preserve the ratings envelope/binding fields; fill every ID exactly once with a boolean failure, category and concrete snippet/reason. The frozen rubric governs statistical scoring. Have it separately note obvious control-token leakage, cross-request leakage or corrupt first tokens for follow-up rather than silently treating omitted rubric categories as passes.

Only after ratings are saved and frozen, repeat the import/root setup above and replace the last line with:

```python
module.compare(module.ROOT / 'blind_ratings.json')
```

Expected artifacts are `blind_packet.json`, `blind_packet_status.json`, `blind_ratings_template.json`, `blind_ratings.json`, the withheld `blind_labels_DO_NOT_GIVE_TO_JUDGE.json`, and the script-produced `blind_comparison.json`. Add a short human-readable review and a final `verdict.json` with concrete prompt IDs/snippets, HF/TT failure counts for every prompt, exact control-reuse paths and the selected runtime policy. Optional comparison against `doc/optimized_full_model/final_validation_v2/qualitative/tt_outputs.json` is regression context; it cannot replace fresh blind scoring. A difference in greedy text alone is not a failure.

## Actual harness limits and closing review

`report.passed=true` means execution and runtime audits passed; `qualitative.blind_verdict` intentionally remains pending until the separate compare step. The parent added an explicit top-1 ≥0.90 assertion to the driver after this audit. The driver does not itself confirm all copied metadata against the current tokenizer; the CPU provenance check above closes that gap. Its qualitative cap is 128 tokens: this suite detects comparative degeneracy/language drift, not general answer correctness, reasoning completion or long-output quality. P1's long sampled texts and their scoring remain separate required evidence. The existing frozen rubric does not exhaust every qualitative-check concern; independent review must inspect actual generated text and require controlled follow-up for concrete additional defects.

After all selected-config gates, post-selection token-out benchmark, charts and documentation are complete, invoke `$stage-review` with a fresh xhigh agent and no inherited discussion. Supply the original goal, datatype-sweep/qualitative-check/stage-review skill paths, model directory, branch/commit, live-worktree status, selected artifact, sweep results, P1 gate, context evidence, final validation and blind verdict paths. Require direct artifact and output inspection. Do not start review while P1 or other required measurements are still running. Fix findings and request fresh diff-only rereviews, at most three rounds; correctness/accuracy/missing-gate findings still block after round three. Record `clean-pass`, or permitted round-three `pass-with-carried`, in the work log before local stage-owned commit; never push.
