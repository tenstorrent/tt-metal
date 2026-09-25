# Known issues

Every agent reads this before it starts. One entry per issue, in the section it belongs to. Each entry is one bullet with
the fields `Symptom:`, `Cause:`, `Fix:`, `Found:` (model and task). An agent that hits something new adds an entry under
"Proposed" at the end of its step; `python -m models.demos.common.bringup.knowledge.check` checks the format, and a
person moves the entry into its section at the next approval point.

## API behavior
- **unified_routed_expert_moe works in place.** Symptom: combine reads garbage or the device faults after freeing the expert input. Cause: for TILE input the op returns its input buffer as the output. Fix: do not deallocate the dispatched buffer before combine. Found: ernie45_d_p P3.2.
- **Expert counts are global.** Symptom: hang, cores stuck in `cb_wait_front` inside unified_routed_expert_moe. Cause: expert_token_counts and expert_region_offsets are indexed by global expert id; a local [1, 16] array makes the kernel read garbage counts. Fix: pass [1, num_experts] per chip. Found: ernie45_d_p P3.1.
- **Chunked SDPA arguments.** Symptom: nanobind TypeError calling `chunked_scaled_dot_product_attention`. Cause: positional arguments combined with `scale=`. Fix: pass every argument by keyword. Found: ernie45_d_p P2.6.
- **scatter has no fp32 TILE path.** Symptom: `ttnn.scatter` rejects fp32 TILE input. Cause: unsupported dtype/layout. Fix: scatter in bf16 or use a selector matmul. Found: ernie45_d_p P2.8.
- **32-aligned widths.** Symptom: `mesh_partition` / `slice` fail on a 16-wide column split. Cause: widths must be tile aligned. Fix: build the broadcast with a selector matmul (`routing @ M_j`). Found: ernie45_d_p P2.9.
- **DeepSeek dispatch/combine on a 1-device dispatch axis.** Symptom: TT_FATAL "No neighbors found"; offset_cumsum all_gather rejects a 1-device axis. Cause: the program factories always wire fabric on the dispatch axis. Fix: host-side patches on `dnijemcevic/ernie45_prefill` (commits 2727352de41, e30d505c4ae) that skip fabric when the axis has one device; needs `./build_metal.sh`. Found: ernie45_d_p P3.1-P3.2.

## Accuracy
- **Fused expert kernel packs to bfp8.** Symptom: top-1 agreement fell from 97.7% to 95.7% and final hidden PCC from 0.998 to 0.996. Cause: unified_routed_expert_moe packs activations to bfp8 internally, even for bf16 input. Fix: accepted (top-5 stayed 100%); budget for it in thresholds. Found: ernie45_d_p P3.2.
- **Router selection vs weights.** Symptom: routed experts differ from HF on a few tokens. Cause: ERNIE-style routers add a correction bias for top-k selection only; weights are the unbiased probabilities renormalized. Fix: keep selection and weighting separate, softmax in fp32. Found: ernie45_d_p P1.2.

## Performance
- **fp32 accumulate disables streaming SDPA on Blackhole.** Symptom: SDPA 85% of a 50k->55k chunk, about 15 TFLOPS per chip. Cause: `fp32_dest_acc_en=True` turns off the streaming kernel (`sdpa_program_factory.cpp:75`); exact exp since #57180. Fix: HiFi2, fp32 acc off, exp approx, q256/k512 (6.5x faster SDPA, accuracy unchanged in practice). Found: ernie45_d_p P3.4.
- **One SDPA program per chunk offset.** Symptom: first 55k run read 22 s instead of 4.4 s. Cause: the scalar chunk_start is part of the compiled program. Fix: measure warm; for serving use `chunk_start_idx_tensor`. Found: ernie45_d_p P3.4.
- **Dense-EP experts.** Symptom: MoE about 2.4 s per 5k chunk. Cause: every local expert over every token. Fix: dispatch + unified_routed_expert_moe + combine (55k TTFT 35.9 s -> 10.9 s). Found: ernie45_d_p P3.2.

## Infrastructure
- **comp_pcc stub.** Symptom: a PCC of exactly 0.999999 in a real pass. Cause: run_safe_pytest's precompile pass rebinds comp_pcc to a stub that can leak into lazily imported call sites. Fix: tests use `core.metrics.pcc`; tests that go through the producer run with `--no-precompile`. Found: ernie45_d_p P2.16.
- **Ambient PYTHONPATH.** Symptom: modules resolve from another checkout (`../tt-metal`). Cause: the shell exports PYTHONPATH. Fix: the gate pins PYTHONPATH to the checkout; set `PYTHONPATH=$PWD` by hand. Found: ernie45_d_p P2.16.
- **Tracy host capture crashes.** Symptom: "tracy-capture exited with code 1" under `--profile`. Cause: unknown, this box. Fix: device profiler with `TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1` and `testing/profiler.py`. Found: ernie45_d_p P3.3.
- **Pre-commit rewrites files at commit.** Symptom: a hash taken before `git commit` never matches again. Cause: black and the EOF fixer run inside the commit. Fix: format with pre-commit before hashing or testing (freeze and `gate --commit` do). Found: bringup_framework F1.
- **autoflake removes imports at commit.** Symptom: NameError after committing. Cause: an import unused at commit time is stripped. Fix: commit the import together with its use. Found: ernie45_d_p P2.9.
- **Edits by others during an agent step.** Symptom: an agent step failed with "changed files outside the allowed paths" listing files the agent never touched. Cause: the path check diffs the whole tree; a person's commit and a dashboard re-export happened during the step. Fix: files that end the step clean are ignored and the dashboard page is excluded; still avoid editing the tree while an agent runs. Found: gemma4_a4b_d_p R.2.
- **Templated YAML fails check-yaml.** Symptom: a gate commit silently did not land; the pre-commit check-yaml hook failed. Cause: a template with `$placeholders` in a `.yaml` file is not valid YAML. Fix: name templates `*.tmpl`. Found: bringup_framework F6.
- **Concurrent gates.** Symptom: verdicts overwritten, unrelated files in a gate commit. Cause: unlocked state writes; `git add -A` of the tree. Fix: the ledger lock and scoped commits (framework does both). Found: ernie45_d_p P2.x.

## Serving contract
- **K RoPE order.** Symptom: producer read-back PCC low for K only. Cause: the producer permutes HF (rotate-half) K to Meta order; a model whose K is already interleaved must skip it. Fix: set the model's golden K layout (`golden_k_rope_layout = "interleaved"`). Found: ernie45_d_p P2.16.
- **Acks before KV is on device.** Symptom: none in a synchronous test; a migration reads stale KV in serving. Cause: the runtime calls the layer-completion sink before the device finished the layer. Fix: sync (or an event) before `sink()`. The contract test's ack-timing check catches it. Found: tt-d-gen audit of ernie45_d_p.
- **Engine input.** Symptom: runtime crashes or embeds pad tokens. Cause: the engine passes a uint32 ROW_MAJOR device tensor [sp, 1, chunk/sp] with a padded tail, not a torch tensor. Fix: accept the device tensor and mask positions >= actual_end. Found: tt-d-gen audit of ernie45_d_p.

## Proposed
- **Multimodal HF wrapper hides the decoder layers.** Symptom: check_hf fails with `AttributeError: 'Gemma4Model' object has no attribute 'layers'`. Cause: AutoModelForCausalLM loads Gemma4ForConditionalGeneration; the text layers are at `model.model.language_model.layers`. Fix: define `hf_layers(model)` in the model's hooks. Found: gemma4_a4b_d_p R.2.
- **Gemma-4 global layers: V is not K.** Symptom: wrong global-layer value state if V is copied from the cached K. Cause: with `attention_k_eq_v` there is no v_proj, but V = unscaled RMS norm of the raw k_proj output, while K = RoPE(k_norm(k_proj)). Fix: store key and value separately (both derived from one projection); Gemma-4 RMSNorm is `x * w` (not `1 + w`), attention scale is 1.0. Found: gemma4_a4b_d_p R.2.
