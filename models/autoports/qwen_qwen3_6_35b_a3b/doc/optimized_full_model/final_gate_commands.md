# Final optimized-full-model validation commands

Prepared by CPU source inspection only. These commands have not been executed by this audit. Run only after the final candidate is selected; store fresh artifacts without overwriting earlier evidence. Every device command is serial on the assigned1×4 Blackhole mesh. No vLLM integration or datatype search is involved.

## Setup and evidence regimes

Run from `/home/ttuser/qwen36-35b-a3b/tt-metal`. The initial `fuser` inspection must show no device holder before the first job; repeat if another process may have taken the mesh. Do not combine Watcher with profiler. A timeout/failure is work, not permission to advance to another device job without inspecting health/evidence.

```bash
source python_env/bin/activate
source /home/ttuser/qwen36-35b-a3b/lane-env.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_OFFLINE=1
export TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
unset TT_METAL_WATCHER TT_METAL_DEVICE_PROFILER TTNN_OP_PROFILER TT_METAL_PROFILER_TRACE_TRACKING
export TT_METAL_TRACE_ALLOC_TRACKING=1
stage_model=models/autoports/qwen_qwen3_6_35b_a3b
stage_out=$stage_model/doc/optimized_full_model/final
stage_module=models.autoports.qwen_qwen3_6_35b_a3b.tests
mkdir -p "$stage_out/logs"
timeout 30 fuser /dev/tenstorrent/*
timeout 60 tt-smi -ls --local
```

`fuser` normally returns1 for no holder. Do not interpret that expected result as a device failure. Commands below redirect their complete output into distinct logs; the parent may wrap each in the installed `scripts/wait_for` with an outer bound greater than the listed process bound.

## Gate inventory and planning estimates

Wall estimates include one model load/prepare per process; they are planning ranges, not measured final latencies. Stage6 all40 prefill took381s; its six-case suite took409s. Stage7 baseline short/tail TTFT boundary process took574s. Cached compilation and the semantic prepare-key optimization can reduce these totals; first compilation or diagnostic overhead can increase them.

| Gate | Scope | Rough wall / process bound |
|---|---|---|
| Prefill accuracy | All40,100 reference positions | 4–10min /1800s |
| Teacher accuracy | All40,100 reference positions | 4–10min /1800s |
| AIME free-running | All40,161-token exact chat prompt,100 new tokens | 3–10min /1800s |
| Shared qualitative suite | All40,six canonical prompts,128 new tokens each,one model lifetime | 4–12min /1800s |
| B32 fixed-slot smoke | All40,32 prepared slots,inactive/reordered continuation | 6–20min /3600s |
| Replay combined sections | All40,B3,nonaligned/mixed/inactive/changed pages/sampled rows | 6–25min /5400s |
| Serving command protocol | All40,B3,mixed33/47/65 prompt lengths,lease/slot/page controls | 4–15min /3600s |
| Sampler acceptance + clean timing | Real terminal geometry,32 physical rows,zero decoder layers,three correct greedy choices | 1–4min each /900s |
| All40 final clean performance | Token-out,model-only,sampler-only and separate99-step teacher logits timing | 4–12min /1800s |
| Short/tail TTFT | All40,baseline-matching context4096 and64/255/256/257/1025/2049 | 6–15min /1800s |
|32769 context smoke | All40,context65536,one nonaligned32769 prompt | 12–30min /5400s |
| Reduced native capture + postprocessing | Real0/3,complete model/generator terminal path; diagnostic attribution only | 10–35min /10800s outer bound |

The32K estimate extrapolates baseline2049-token43.16s TTFT and allows additional long-context attention cost. It is not an observed32769 measurement. Do not rerun the262144 maximum unless KV dtype/layout changed; retained contract max remains262144.

## Executable serial correctness queue

The new driver implements the recipes below, including the exact-ID AIME API call. It stops on the first failed command/timeout, checks idle device ownership before every hardware process, rejects source changes during the queue, and records commands, source hashes, complete logs and an execution summary in a fresh directory. It never resets hardware or resumes after failure. Perf/profile/TTFT/context smoke and human qualitative verdicts remain separate.

```bash
python -u "$stage_model/tests/run_stage7_final_validation.py" --output-dir "$stage_out" --dry-run
python -u "$stage_model/tests/run_stage7_final_validation.py" --output-dir "$stage_out"
```

**For the driver, do not run the earlier `mkdir` command:** its output directory must not exist. Choose another fresh directory if manual recipes already created `final`. Activate the environment and source lane-env.sh first. The driver sets tracking1/skip0 and clears profilers in every child. Only all40 replay uses Watcher10; its `generated/watcher` directory is copied to `watcher_replay` even after failure, before any later hardware process. All other jobs have Watcher unset. Its tracker positive control intentionally uses reduced0/3 only to prove rejection; every model correctness gate uses all40. The positive-control command must include `--generate`; without it, the existing CLI silently bypasses the control.

```bash
timeout 1800 python -u -m "$stage_module.run_full_model_probe" --generate --positive-control --output "$stage_out/positive_control.json" >"$stage_out/logs/positive_control.log" 2>&1
```

`execution_summary.json:passed` means this queue executed successfully; it does not declare stage acceptance or a qualitative verdict. Planned serial wall range is roughly30–100min, dominated by repeated model preparation. No new same-process ownership scheme is introduced.

## Required AIME accuracy

```bash
timeout 1800 python -u -m "$stage_module.run_full_model_readiness" --mode prefill --context 512 --output "$stage_out/prefill_accuracy.json" >"$stage_out/logs/prefill_accuracy.log" 2>&1
timeout 1800 python -u -m "$stage_module.run_full_model_readiness" --mode teacher --context 512 --output "$stage_out/teacher_accuracy.json" >"$stage_out/logs/teacher_accuracy.log" 2>&1
```

These invoke installed `run_prefill_check`/`run_teacher_forcing`, both with `readiness_aime24_chat.refpt`, and fail unless every returned row has top5≥0.98 and top100=1.0. Factory defaults use all40; wrapper provenance records40. The teacher gate uses explicit HF inputs for accuracy; it does not replace autoregressive or token-out performance evidence. Both modes own a separate generator/model lifetime: the existing runner does not support safe same-process combination.

## AIME autoregressive evidence with exact saved prompt IDs

The packaged autoregressive CLI strips prompt text before tokenization. Do not pass the saved rendered AIME chat text directly and silently lose its terminal newline. This bounded API invocation uses the exact161 saved IDs and the existing complete generator, with a fresh output directory. It reuses the pinned stage6 HF100-token control and saves both readable completions plus token IDs. It creates no source/helper changes.

```bash
timeout 1800 python -u - "$stage_model" "$stage_out/aime_autoregressive" >"$stage_out/logs/aime_autoregressive.log" 2>&1 <<'PY'
import hashlib, json, sys
from pathlib import Path
import torch
import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.run_full_model_batched_teacher import load_fixture
from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import QwenModel
from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import QwenGenerator
root, out = map(Path, sys.argv[1:])
assert not out.exists()
out.mkdir(parents=True)
entry, reference = load_fixture(root / 'readiness_aime24_chat.refpt')
ids = entry.prompt_tokens[0].tolist()
assert ids == json.loads((root / 'doc/full_model/hf_reference/aime24_prompt_tokens.json').read_text())
torch.set_num_threads(8)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1,4), trace_region_size=200000000)
gen = None
try:
    model = QwenModel(mesh, max_seq_len=512)
    assert list(model.layer_indices) == list(range(40))
    gen = QwenGenerator(model, host_sampling=False)
    generated = gen.generate(ids, 100, temperature=0.0, top_k=1, top_p=1.0, seed=0)
    hf = entry.generated_tokens[0].tolist()
    record = dict(layers=list(model.layer_indices), mesh=[1,4], reference=reference,
        prompt_mode='chat', prompt_token_ids=ids, host_sampling=gen.host_sampling,
        generation=dict(max_new_tokens=100, temperature=0.0, top_k=1, top_p=1.0, seed=0),
        tt_token_ids=generated, hf_token_ids=hf, counters=dict(gen.counters),
        trace_ids={k:int(v) for k,v in gen.traces.items()},
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [root/'tt/model.py', root/'tt/generator.py', root/'tt/multichip_decoder.py', root/'tt/prefill.py']})
    (out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    (out/'tt_completion.txt').write_text(gen.tokenizer.decode(generated,skip_special_tokens=False))
    (out/'hf_completion.txt').write_text(gen.tokenizer.decode(hf,skip_special_tokens=False))
finally:
    if gen is not None: gen.close()
    ttnn.close_mesh_device(mesh)
PY
```

Read both outputs and write a fresh descriptive verdict. Greedy EOS arrival is not a gate. A model-quality failure selected for debugging requires the qualitative skill's locked rubric and10 matched seeds before interpreting TT/HF differences. This free-running recipe includes caller token readback for text; it is not the no-host-boundary throughput measurement.

## Shared six-case suite and valid control reuse

```bash
timeout 1800 python -u "$stage_model/doc/full_model/qualitative/run_tt_controls.py" --mode suite --hf-controls "$stage_model/doc/full_model/qualitative/hf_greedy_outputs.json" --output-dir "$stage_out/qualitative" --max-new-tokens 128 --context 1024 >"$stage_out/logs/qualitative.log" 2>&1
timeout 120 python -m readiness_check.check_degenerate_output "$stage_out/qualitative" --hf-model Qwen/Qwen3.6-35B-A3B --scope autoregressive --missing-artifacts critical --json "$stage_out/qualitative/degenerate_output_check.json" >"$stage_out/logs/degenerate_output_check.log" 2>&1
```

The existing suite runner already safely shares one all40 model/generator and capture set across all six prompts, through a non-owning readiness adapter. It asserts all40, exact tokenizer class, rendered prompt text and saved IDs, the current installed six-prompt suite, pinned HF model/revision, greedy settings, and128-token control length. It records hashes and control reuse. Reusing `doc/full_model/qualitative/hf_greedy_outputs.json` is allowed because these values are unchanged; reuse the control, not the old TT verdict. The immutable revision is `345d467b117d1fd5eda9b879a5988de9ae5f47a7`. Stage6's relevant reasoning and shared q03 phrase repetition are comparison context, not automatic stage7 acceptance. Review all six fresh outputs and preserve their paired artifacts.

## Full-stack serving contracts and B32

```bash
env TT_METAL_WATCHER=10 timeout 5400 python -u -m "$stage_module.run_generator_replay_probe" --all-layers --section all --context 2048 --steps 4 --output "$stage_out/replay_all40.json" >"$stage_out/logs/replay_all40.log" 2>&1
timeout 3600 python -u -m "$stage_module.run_serving_commands_probe" --all-layers --context 1024 --output "$stage_out/serving_commands_all40.json" >"$stage_out/logs/serving_commands_all40.log" 2>&1
timeout 3600 python -u -m "$stage_module.run_full_model_probe" --all-layers --generate --batch-size 32 --batch-probe --context 1024 --output "$stage_out/b32.json" >"$stage_out/logs/b32.log" 2>&1
```

The first two now accept explicit `--all-layers`; reduced0/3 remains their default. Both assert actual model layer indices against recorded scope. Never omit the flag for these acceptance rows. The replay runner's `--section all` already combines lengths, mixed, inactive, changed-only page-table and top-k/top-p rows in one prepared lifetime; splitting sections repeats preparation. Context2048 allows its per-slot physical-page budget to cover the longest continuation. State diagnostic readbacks and forced input uploads are outside resident no-host-boundary checks. The serving-command probe additionally checks logical request/lease/reorder/page-update behavior; some detailed snapshots select the first recurrent/full layer, so do not describe them as per-layer numerical equality. The B32 probe is a fixed-slot capacity and inactive-state smoke with uniform4-token prompts; the separate B3 mixed suite supplies heterogeneous prompt coverage.

## Correct sampler choices: tracked acceptance then clean timing

```bash
timeout 900 python -u -m "$stage_module.run_sampler_contract_probe" --mode acceptance --iterations 100 --output "$stage_out/sampler_acceptance.json" >"$stage_out/logs/sampler_acceptance.log" 2>&1
env TT_METAL_TRACE_ALLOC_TRACKING=0 timeout 900 python -u -m "$stage_module.run_sampler_contract_probe" --mode timing --iterations 100 --output "$stage_out/sampler_timing.json" >"$stage_out/logs/sampler_timing.log" 2>&1
```

Both modes retain real-terminal, all-negative unique-max and cross-shard-tie correctness checks for split greedy, split greedy with power-of-two padding, and force argmax on the real4-way vocabulary geometry/32 physical rows. Timing mode enforces no Watcher/device/op profiler and no profiler trace tracking, tracking0/skip0; acceptance enforces tracking1/skip0. Its zero decoder layers are deliberate terminal-contract isolation, never an all-model result. Clean sampler comparisons justify strategy selection; full token-out performance and profiler tables establish actual end-to-end contribution. Top-k/top-p support is exercised by the replay suite.

## Final clean performance, TTFT boundaries and nonaligned32K smoke

```bash
env TT_METAL_TRACE_ALLOC_TRACKING=0 timeout 1800 python -u -m "$stage_module.run_full_model_performance" --context 1024 --output "$stage_out/performance.json" >"$stage_out/logs/performance.log" 2>&1
env TT_METAL_TRACE_ALLOC_TRACKING=0 timeout 1800 python -u -m "$stage_module.run_optimized_full_model_prefill" --context 4096 --lengths 64,255,256,257,1025,2049 --output "$stage_out/prefill_boundaries.json" >"$stage_out/logs/prefill_boundaries.log" 2>&1
env TT_METAL_TRACE_ALLOC_TRACKING=0 timeout 5400 python -u -m "$stage_module.run_optimized_full_model_prefill" --context 65536 --lengths 32769 --output "$stage_out/prefill_context32769.json" >"$stage_out/logs/prefill_context32769.log" 2>&1
```

No `--reduced` is present. Performance runs all40 and records queued token-out separately from caller-observed delivery, sampler/model controls, and99-step traced HF-input logits-only timing. Do not merge teacher logits timing with token-out sampling/feedback/readback. The runner's `device_only` guard and zero conversion/refresh/capture deltas audit the measured path; retain these raw counters. The prefill runner can combine arbitrary lengths in one all40 process, but these commands deliberately keep baseline-matching context4096 TTFT separate from context65536 smoke: changing configured cache capacity during a before/after comparison would confound it. Long length is run once after warming all physical variants, not same-length warm repetition. It asserts two traced model+sampling replays with no token/position/page refresh/readback inside the decode segment, unchanged program count/trace IDs, and device position advancement. Its long repeated AIME fixture is capacity evidence, not long-context task accuracy.

## Reduced native profiler evidence

```bash
timeout 10800 bash "$stage_model/tests/collect_full_model_components.sh" "$stage_out/profile_reduced" >"$stage_out/logs/profile_driver.log" 2>&1
```

The collector enforces an idle mesh/fresh directory, removes Watcher and allocation tracking, enables profiler/Tracy in its own process, runs real layers0/3 with `--reduced --profile`, converts native reports, and exports `tt-perf-report` CSV/text for token-out, model-only and sampler-only phases. Exact commands are in its `commands.log`; raw capture/ops CSV/provenance remain under the fresh directory. This reduced capture attributes the full embedding→decoder→terminal→sampling path; it does not replace all40 accuracy, latency, fallback, or serving gates. Review operation/kernels/configs/CCL and sampler costs, then combine representative optimized layer latencies by30 linear+10 full only as a labeled estimate; do not call cross-workload subtraction an exact full-model decomposition.

## Remaining evidence work and safe process sharing

The dominant avoidable validation expense is repeated all40 model construction/preparation, not100 decode tokens (~seconds). Supported sharing is already used by the six-prompt qualitative suite, all replay sections, the multi-length prefill boundary driver, and performance phases. The separate32K process is necessary to preserve baseline-matching context for short TTFT. Existing readiness modes and the B32/serving/sampler runners own and close separate devices/models; do not concatenate their `main()` calls on an open mesh or patch ownership to save setup without a separately reviewed harness change.

This audit added opt-in all40 scope flags and clean sampler mode only. CPU compilation passed; no new hardware pass is claimed. Finish source fallback review as well as runtime guarded evidence. Guard coverage on queued token-out must remain zero host conversions and zero per-token host operations; accuracy/text output intentionally has host observation boundaries. Refresh context contract only with supported evidence, preserve rejected precision policies, and run independent stage review after the final unchanged-source gates. Reuse prior HF controls only at their exact prompt/settings scope.

## Execution status and affected reruns

The original final correctness queue completed its first eight jobs, then the
all40 serving harness exposed an obsolete reduced-layer page-count assertion.
A second harness-only rerun exposed local variable shadowing. Both test defects
are fixed; the accepted affected rerun is
`final/serving_commands_all40_v3.json`, with its exact command and source identity
in `final/correctness_acceptance.json`. Final sampler acceptance ran separately.
The original failed execution summary and both failed result files are retained.
No inference source changed between the original queue and these reruns.

`run_final_measurements.sh` is the actual serialized follow-on command record.
Its current completed jobs are the queued layer0/3 controls, clean sampler,
all40 performance, short/nonaligned boundary TTFT and validated native conversion
plus reports. Context32769 passes; FINAL_MEASUREMENTS_DONE and driver exit0 are recorded. The run log and each artifact's
command take precedence over these recipe examples. Frozen source maps before
and after every job detect changes to the309 measured implementation/test files.
