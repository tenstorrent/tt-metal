# Batched prefill RoPE candidate

CPU-only implementation prepared for serial device evaluation. No hardware execution, correctness claim, or latency claim was made by this investigation. Source: `tt/prefill_rope_candidate.py`. The frozen single-chip decoder and TP4 default are unchanged.

The current `_rope` loops over the logical batch because native prefill RoPE requires `[1,H,S,64]` input with `[1,1,S,64]` trig. At B32, Q and K together issue 64 native RoPE calls. The candidate folds users into the sequence axis and issues two calls, one for Q and one for K. The parent profile recorded 274.143 us for the 64 calls in full-attention B32S33 prefill; that is an upper bound on this candidate's possible kernel savings, before added layout costs, not an expected end-to-end gain.

## Layout contract and source evidence

For Q or K `[B,H,S,256]`, only the first 64 channels rotate. Let `T=ceil(S/32)*32`:

1. Extract `[B,H,S,64]` and explicitly pad each user's sequence to T.
2. Permute to `[H,B,T,64]`, then reshape to `[1,H,B*T,64]`.
3. Each per-user cos/sin `[B,1,S,64]` is padded to T and reshaped to `[1,1,B*T,64]`.
4. Apply one native `experimental.rotary_embedding_hf` with the unchanged compute kernel config.
5. Invert the reshape/permutation, crop each user's sequence back to S, and concatenate the untouched 192-channel suffix.

All intermediates use interleaved DRAM. The operation remains rank-local: no collective, residual-layout change, host tensor read, or inter-layer conversion. Explicit padding is owned by the helper, so public non-aligned S remains valid. B1 and S1 invoke the original method; the existing direct compact decode path is untouched. Different per-user positions are preserved because trig is flattened in the same B,T order as input, rather than broadcasting user zero. The candidate supports the arbitrary batch extents that the parent decoder supports; it does not require a power-of-two B.

Native source evidence:

- `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/rotary_embedding_hf_device_operation.cpp`: prefill validates trig batch dimensions 1, matching width, and trig sequence length at least input sequence length. Width 64 is already tile aligned and avoids the partial-width midpoint hazard.
- `.../device/kernels/dataflow/reader_rotary_embedding_hf_interleaved.cpp`: `cos_sin_curr_id` resets after Ht tile rows. Thus the folded B*T trig sequence repeats for each head exactly as needed. RoPE is pointwise in sequence and cannot mix padded rows into valid rows.
- `tt/optimized_decoder.py::_rope` supplies the same primitive/config per original row; `tt/multichip_decoder.py::_qkv` invokes it separately for Q and K.

No claim is made that TTNN's complete permutation/reshape execution is validated by the CPU check. Explicitly aligned intermediate shapes avoid relying on a reshape that silently strips per-user tile padding. If TTNN rejects one of these operations, adapt the permutation/layout path and retry before rejecting this material 64-to-2-op change.

## CPU evidence

Executed from repository root, using Python without importing TTNN:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tt.prefill_rope_candidate --cpu-check
```

Result: `{"pass": true, "kind": "cpu_transform_only", "cases": 48}`. Coverage B=2,3,17,32; local heads=1,6; S=2,31,32,33,65,513. Each user had a distinct 977-token position offset. All retained rotary outputs were bitwise equal to independent per-user PyTorch pointwise rotation; unrotated channels were identical. NaN-poisoned dead padding never contaminated a retained output. This establishes index semantics only. Formatting checked with `black --target-version py312`.

## Integration and serial device commands

Programmatic instance hook, before warmup/capture:

```python
from models.autoports.qwen_qwen3_6_27b.tt.prefill_rope_candidate import install
install(model)  # full_attention only; installation twice is rejected
```

The optional module CLI wraps the existing full-layer harness, installs on the newly constructed TP4 instance, and records both a selected-policy marker and the candidate source hash. It refuses `--baseline`; use the existing matching baseline fixtures. All existing runner arguments, PCC/state/per-user gates, trace checks, and timing behavior are preserved. Run only when the parent's device scheduler is free:

```bash
source python_env/bin/activate
source /home/ttuser/qwen36-27b/lane-env.sh
D=models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder
R=models/autoports/qwen_qwen3_6_27b/doc/multichip_decoder
python -c 'import json,pathlib; d=pathlib.Path("models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder"); p=next(j["policy"] for j in json.loads((d/"profile_integrated2.json").read_text()) if j["layer"]==3); (d/"rope_candidate_policy.json").write_text(json.dumps(p))'
unset TT_METAL_WATCHER TT_METAL_DEVICE_PROFILER
TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 MULTICHIP_JOB_TIMEOUT=600 bash "$D/run_job.sh" rope_fold_l3_b32_s33 \
  python -m models.autoports.qwen_qwen3_6_27b.tt.prefill_rope_candidate \
  --layer 3 --batch 32 --length 33 --trace --repeats 100 \
  --policy-file "$D/rope_candidate_policy.json" \
  --reference "$R/delivery_perf_ref_l3_b32_s33.pt" \
  --output "$D/rope_fold_l3_b32_s33.json"
```

Compare against a fresh unhooked `tests.run_multichip_decoder` invocation with the same policy and fixture. Then measure B32S513 and B1S512 by changing all shape/name/reference/output occurrences consistently. B1 is a regression control with no selected operation change. A separate watcher run and a native operation probe with distinct per-user trig/positions are still required before adoption; the existing full-layer harness uses identical prefill positions across users and does not cover that risk by itself. The CPU check covers distinct positions only mathematically.

For profiler attribution, invoke the same module through Tracy using `--policy-file` (not a shell-joined JSON argument), retain CSV/provenance, and compare combined RoPE plus reshape/permute/pad/slice time and whole prefill latency. A reduced primitive count by itself is not a performance result. Large folded B*T adds transient DRAM footprint; this path does not change KV cache or persistent context capacity. Non-aligned S incurs T/S processing expansion in the 64-channel prefix and trig only. Keep/reject requires the complete boundary cost, PCC, trace, and watcher evidence.

## Native distinct-user probe added for parent execution

The module now accepts `--probe`. This entrypoint opens the actual 1x4 mesh and compares the folded native operation against independent per-user native RoPE for B/S pairs 2/3, 3/33, 17/31, 32/33, and 3/513, each with local heads1 and6. Rank, user, and replay step all have different positions and activations. It uploads NaN-poisoned sequence-padding lanes, retains the public logical S, validates eager output, then validates two changed-input trace replays against saved eager references. Every rank/user must be finite and PCC>=.995; bitwise equality is also recorded. The probe reports correctness only and has no timings to misinterpret under watcher.

Prepared, syntax-checked, and CPU-check rerun by the CPU agent; **not executed on hardware by that agent**. Parent invocation:

```bash
source python_env/bin/activate
source /home/ttuser/qwen36-27b/lane-env.sh
D=models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder
unset TT_METAL_DEVICE_PROFILER
TT_METAL_WATCHER=10 TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 MULTICHIP_JOB_TIMEOUT=600 bash "$D/run_job.sh" rope_native_distinct_watcher \
  python -m models.autoports.qwen_qwen3_6_27b.tt.prefill_rope_candidate --probe \
  --output "$D/rope_native_distinct_watcher.json"
```

For a bounded first check, add `--cases 3x33 --heads 6`. Successful compact mode does not replace the full default case coverage. This correctness probe deliberately leaves trace-allocation tracking to the stage's separate allocation audit; it does not use a capture exemption. Adding the probe changes the module file hash but leaves `batched_prefill_rope` and its installed computation unchanged from the parent's initial passing full-layer run.
