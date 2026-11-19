# Optional FP32-softplus and fast-zero convolution/gate integration

Prepared only. Selection remains conditional on root's cumulative and macro-fixed direct evidence. No production, existing candidate, live kernel, or existing probe was edited; no hardware was used.

`optional_after_base.patch` applies after `../base_integration_5cdc.patch`. The pair was applied in an isolated temporary directory and passed `git apply --check --whitespace=error`. Complete proposed production file contents are under `proposed/`; `validation.json` binds their hashes to the candidate source hashes. Python syntax checks passed, and all seven copied C++/header files are byte-identical to the current candidate sources.

If selected, root can review/apply in this order:

```sh
git apply models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/final_integration_audit/base_integration_5cdc.patch
git apply models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/final_integration_audit/optional_conv_gates/optional_after_base.patch
git apply models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/final_integration_audit/optional_conv_gates/manifest_helpers.patch
```

Do not reapply the base patch if root has already integrated it. Recheck patches against any intervening production changes. These commands are instructions, not actions performed by this audit.

## Production changes

The decoder's ordered, complete pooled-linear decode method calls `tt.conv_gates.conv_gates` and `tt.flat_ordered_conv_gates.flat_ordered_conv_gates`. It passes the original packed3136-wide tensor directly as the recurrent gate source. All pool eligibility/ordering checks remain outside this method, as before. Native pooled B1 behavior and chunk defaults come from the base integration patch. Private/subset/mixed-state dispatch, full attention, prefill, persistent state dtype/shape, and context arithmetic are unchanged by this optional patch.

The helper preserves the candidate's `INP_FLOAT32=1` compute define and64-entry unpack-mode assignment. CB10–13 use full-FP32 unpack; convolution tap CB4–7 retain production's existing unpack mode. BF16 beta uses its explicit BF16 sigmoid numerical branch; softplus uses the FP32 algorithm and the candidate's FP32 intermediate boundaries. The recurrent reader retains NoC fast-zero initialization, the packed beta/a-derived gate tile layout, and packed z offsets. Ordered recurrence, postnorm, FP32 state, exact HiFi4, writer and CB geometry are copied without arithmetic changes.

The candidate fixes wave104; the integrated method retains `self.decode_recurrent_wave` as the original API did, whose default is104. The new direct probe additionally validates wave7. No test imports or test-kernel paths remain in proposed production files. Existing production `tiled_conv` and `flat_ordered_post_loop` files stay available as independent controls; no duplicate replacement of their arithmetic is necessary.

## Final production direct probe

New `tests/multichip_conv_gates_final_production_probe.py` imports the actual integrated production helpers, and asserts that the decoder's bound functions are those same helper functions. It intentionally cannot run before integration. New root-runner queue: `doc/optimized_multichip_decoder/final_conv_gates_production_direct_jobs.json`.

```sh
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_APPEND=1 \
python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_conv_gates_final_production_probe \
  --output models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/final_conv_gates_production_direct_watcher_all_batches.json \
  --batches 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32 \
  --waves 104,7 --steps 8
```

Root must serialize this command through its existing hardware runner. Full success produces2048rank records (32batches ×2waves ×8steps ×4chips). Inputs alternate DRAM/L1; histories use DRAM on wave104 and L1 on wave7. Independent persistent candidate, separate-gate control, and identical-gate control states are retained. Signed/random gamma and z gates, full-FP32/BF16-widened bias/taps, sigmoid/softplus threshold cases, and zero/tiny QK are inherited from the measured matched probe. Expanding the batch list changes later random fixtures relative to the original sparse list, intentionally adding coverage; prior fixture evidence remains archived.

The probe retains bitwise convolution/history equality against production tiled convolution, bitwise state equality with identical gate tensors, and now explicitly requires bitwise output for that identical-gate control too (the measured704-record matched probe already achieved it). Separate TTNN gate comparisons and separate-gate state/output global/per-head PCC remain >=.995. Original Torch and separate-TTNN convolution checks remain. `score`, `check` and device-conversion functions are AST-identical to the source probe. The original legacy failed result and its independent production-substitution experiment remain separate evidence; this probe does not rewrite them.

This is direct kernel coverage, not final decoder performance or correctness evidence. Rerun final default decoder PCC, long-answer, trace/pool contracts, Watcher and warmed performance/profile gates after selection and integration.

## Manifest additions

`manifest_additions.json` enumerates the exact nine new production paths:

* `tt/conv_gates.py`
* `tt/flat_ordered_conv_gates.py`
* `tt/kernels/conv_gates/compute.cpp`
* `tt/kernels/conv_gates/reader.cpp`
* `tt/kernels/conv_gates/writer.cpp`
* `tt/kernels/flat_ordered_conv_gates/compute.cpp`
* `tt/kernels/flat_ordered_conv_gates/norm_stages.hpp`
* `tt/kernels/flat_ordered_conv_gates/reader.cpp`
* `tt/kernels/flat_ordered_conv_gates/writer.cpp`

The changed decoder is already in the manifest. `prepare_stage5_manifest.py` recursively collects `tt/kernels`, so the small optional `manifest_helpers.patch` adds only the two Python helper paths to its explicit source list. Root's job runners already hash all `tt/*.py` and kernel source/header files. Add the new direct job to the final kernel-job list that the manifest reads (currently `final_v1_kernel_jobs.json`), or make the manifest consume the new queue explicitly; otherwise the new direct result will not appear in `direct_kernel_artifacts`. Record the new result as `final_conv_gates_production_direct_watcher_all_batches.json` only after it passes.
