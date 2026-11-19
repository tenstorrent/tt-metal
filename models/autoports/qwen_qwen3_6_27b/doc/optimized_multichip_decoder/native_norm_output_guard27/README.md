# Unapplied persistent-output admission repair

Diagnosis: `../native_norm_source_audit27.md`. A caller-supplied output bypasses
normal output-spec construction, but the original native validator checks
only the requested memory config. The factory trusts the actual supplied
tensor when constructing output CBs/routes. The selected model owner supplies
correct tensors; the proposed change closes the broader native API gap.

`output_validation.patch` changes only:

1. `rms_allgather_device_operation.cpp`: check actual preallocated output
   device allocation, mesh identity, TILE layout, requested dtype, exact
   requested memory/shard configuration, standard tiles, logical shape and
   expected physical capacity. The existing legitimate output-reshard width
   calculation is retained. No factory/kernel/compute math changes.
2. `tests/probe_fused_minimal_norm_validation.py`: add11 output negatives to
   the existing12 stats/link negatives, retaining all original healthy
   B1/B32 fresh-binding/gamma tests after the rejections. Expected native
   messages and unchanged program-cache counts remain mandatory. Generic
   construction/API errors fail; they never count as admission evidence.

The patch is **unapplied**, with no production/helper/native edits or device
work. `prepare_patch.py` uses only stdlib, checks Python syntax and native
message anchors, pins before/after source hashes, validates128 owner logical
capacity/column interval cases, and runs `git apply --check`. `cpu_proof.json`
records those source-only checks. C++ compilation and hardware behavior remain
unverified.

The cross-device negative uses a1×1 submesh of the already-owned1×4 mesh,
allocates a tensor on that submesh, and requires native same-mesh rejection.
It does not open an unrelated physical device or reset fabric. The padded
capacity negative uses a64-row-backed logical B1 view with a requested64-row
shard. Both fixtures must construct successfully before the native call. If
the framework rejects construction or intercepts the call earlier, preserve
the failure and adapt the fixture; do not mark the guard as hardware-tested.
Submesh cleanup precedes the normal parent-mesh close.

Parent-owned serialized steps after the active queue reaches a checkpoint:

```bash
git apply --check models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/native_norm_output_guard27/output_validation.patch
git apply models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/native_norm_output_guard27/output_validation.patch
cmake --build build_Release --target ttnn -j 12
cmake --install build_Release --component ttnn-runtime
cmake --install build_Release --component tt_pybinds
timeout 600s env TT_METAL_WATCHER=10 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_fused_minimal_norm_validation \
  --layer 0 --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/fused_minimal_norm_validation27.json
```

Record a fresh canonical source/binary receipt after install. The prior
build25 receipt remains historical evidence and cannot attest the new guard.
Then rerun the already-defined `probe_fused_decode_norm_owner` commands for
layers0/3 with fresh output stems and the same explicit selected policies:
both families, B1/3/17/32, queued50, strict allocation positive control and
watcher. The patch preserves all valid owner specs and memory, but that claim
must be checked against the rebuilt binary before final stage acceptance.
