# Optional aligned replay model-state comparison

`tests/vllm_replay_numerics_probe.py --compare-model-state` adds three diagnostic snapshots after consuming exactly the same 151 tokens: uninterrupted decode before sampling output23, host replay prefill, and device replay prefill before sampling. The earlier sampler checkpoint after22 outputs corresponds to only150 consumed tokens and is deliberately not used for model-state comparison. Existing defaults do not collect these snapshots or create their tensor artifact.

The diagnostic reads each TP rank separately. It extracts active and inactive convolution/recurrent rows with the actual pooled layout, plus active K/V in logical page order through token150. Global mappings receive no nominal per-slot page offset; private mappings do. It reads individual physical pages rather than the whole context-sized KV pool. Inactive K/V coverage is limited to one mapped sentinel page per inactive slot. No diagnostic tensor or readback value feeds generation.

The JSON's `model_state_comparison` records metadata, checks and per-tensor errors; `<output-stem>.model_state.pt` contains the CPU snapshots. Prefix length, observed position/end, page bounds/non-aliasing, equal mapping, finite values and matching shapes are checked. Host/device replay states and inactive snapshots must match exactly. Decode-versus-prefill differences are reported as exactness, maximum/mean/RMS error, relative L2 error and worst index, with no invented tolerance. The snapshots use the existing layers0/3, B3 fixture and do not establish B32/full-model correctness or rule out a systematic kernel defect.

CPU validation: eight tests pass, including comparison with the actual extracted decoder `read_linear_state`, pooled/private layouts, global/private page offsets, reordered pages and partial-page trimming, bounded collector read sizes, off-by-one rejection, mismatch/nonfinite detection, and opt-in guards. Ruff and Python compilation pass. Evidence: `replay_model_state_host_validation.log`. No hardware or server was accessed during implementation.

Root hardware follow-up, after current device work has finished and devices are released:

```bash
cd /home/ttuser/ornith-9b/tt-metal
source python_env/bin/activate
source /home/ttuser/ornith-9b/lane-env.sh
export TT_METAL_TRACE_ALLOC_TRACKING=1
export TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
unset TT_METAL_DEVICE_PROFILER TT_METAL_DEVICE_PROFILER_DISPATCH
timeout 1800 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.vllm_replay_numerics_probe \
  --context 262144 --greedy --follow-outputs 23 --compare-model-state \
  --served-reference "$MODEL_DIR/doc/vllm_integration/diagnostic9_pages/preemption.json" \
  --output "$MODEL_DIR/doc/vllm_integration/replay_model_state151.json" \
  > "$MODEL_DIR/doc/vllm_integration/replay_model_state151.log" 2>&1
```

This command is prepared, not executed. It reuses the minimal output23 decision rather than repeating the50-output follow-up. Diagnostic9 is historical served evidence; the current model includes the later prefill correction, so token-prefix comparisons retain that code-version distinction. Subsequent state/oracle interpretation remains necessary even if structural checks pass.


The first execution failed on late diagnostic KV-slice program-cache allocations before comparison. Its original log, JSON and probe source are retained under `replay_model_state151_before_slice_warmup/`. The probe now prepares the exact optional snapshot read path twice before capture and requires stable program-cache counts on the second pass; tracking and program-cache checks remain enabled. Reuse the command above for the targeted hardware rerun. Inspect `model_state_read_warmup` and require a complete comparison artifact; CPU validation alone does not establish a trace-safe device result. See `replay_model_state_slice_warmup_host.log` for10 passing CPU tests and lint/compile results.


Completed rerun: `replay_model_state151.json` passes all structural/reconstruction checks; program counts847→875→875, completion and normal device closure are recorded. See [interpretation](replay_model_state151_interpretation.md). Strict forced-preemption trajectory parity remains failed; this is not an exact decode/prefill state-equivalence result.
