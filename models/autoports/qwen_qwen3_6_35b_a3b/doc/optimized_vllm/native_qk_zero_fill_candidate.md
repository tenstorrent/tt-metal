# Native Q/K local zero-fill follow-up

Prepared source-only on 2026-11-17 while the parent runs matched full40 timing. The adjacent `native_qk_zero_fill_candidate.patch` is **unapplied**. No production edit, hardware call, compilation, profiling, or timing was performed. This is a bounded candidate if the current floor still fails, with no claimed benefit.

The patch changes only the native Q/K reader's initialization of CB1 and CB2: replace the loop issuing 4,096 volatile uint32 stores to each buffer with two `noc.async_write_zeros(..., 4*4096)` calls followed by `noc.write_zeros_l1_barrier()`. Keep the current placement after all input reads and their read barrier; deliberately do not overlap zeroing with input reads in this first candidate. Keep the subsequent 128-word Q and 128-word K scatters unchanged, including FP32 payload bits. The transformed fallback, native metadata guards, all compute and writer code, native ACKs, activity-controlled state commit, CB publication, and persistent input ownership are unchanged.

## Source contract and ordering

- `tt/mesh_state_slab_recurrence.py` reserves four pages of each independent CB1/CB2 before initialization. Each descriptor allocates exactly four 4,096-byte FP32 tile pages (16 KiB per buffer). Existing assertions require both write pointers to be 64-byte aligned; the zero API requires only 16-byte alignment on Wormhole/Blackhole. The byte ranges exactly match the former loops and cannot touch the separate CB28 source scratch.
- `tt_metal/hw/inc/api/dataflow/noc.h` documents `DataflowBuffer` as a supported local destination and resolves its current write pointer. Destination bytes must not be read or written until `write_zeros_l1_barrier()` returns. The patch obeys that rule: constructing the pointer values before zeroing does not dereference them, and the first scatter store is after the barrier.
- `tt_metal/hw/inc/internal/tt-1xx/noc_zero_l1.inl` implements the operation as loopback reads from `MEM_ZEROS_BASE`, with a read-completion barrier. Wormhole and Blackhole memory maps set `MEM_ZEROS_SIZE=512`, so each 16 KiB buffer requires 32 loopback requests, 64 per worker. Two consecutive zero calls followed by one barrier wait for both buffers. No NoC read follows these calls, so their read-command state setup cannot invalidate a later state-dependent reader sequence in this kernel.
- All bytes become zero, exactly the previous uint32 zero stores. The unchanged scatter then overwrites only column0, preserving arbitrary source FP32 bits, including signed zero and nonfinite payloads. Remaining lanes retain positive zero. No floating-point arithmetic, narrowing, or transport canonicalization is introduced.
- Compute cannot consume either input before the existing final `push_back(4)`. The zero barrier and scatter both precede those pushes. The reader still owns both buffers exclusively during initialization; no compute or writer synchronization is removed.
- The Quasar implementation explicitly supports consecutive zero calls before one barrier; its barrier restores the borrowed write command buffer. The patch issues no other NoC write in that interval. This source observation is not a Quasar hardware qualification; the current candidate is for the existing serving device configuration.
- The changed embedded reader source provides a different generic-op kernel identity. Tensor metadata, runtime argument arity, CB descriptors and persistent input addresses do not change. Fresh-process tests must still check stable program-cache counts and trace reuse; source identity is not a substitute for those gates.

## Cost and exact discriminator

The former loop executes 8,192 volatile 32-bit stores per worker, writing 32 KiB. The candidate still writes exactly 32 KiB, now through 64 local NoC loopback requests on Wormhole/Blackhole plus a completion barrier. Across 32 workers this is 1 MiB of local zero traffic per layer. Instruction removal can be offset by loopback issue, contention or barrier cost; no source-derived speedup or floor remedy is asserted.

Apply only after the current frozen-source run ends. Preserve the current micro and full40 artifacts. In a fresh process rerun `tests.run_native_qk_reader_probe` with new output paths for rounded0/temporary and rounded1/persistent, retaining all cases, actual-upload transport checks, arbitrary FP32 edge words, device-poisoned source padding, all four chips, changed/unchanged activity and inputs, exact complete output/state comparisons, trace releases and stable cache counts. Compare the same saved full40 control and rerun the tracked mixed-bucket discriminator before matched native/adapter timing. A micro correctness pass is not performance evidence; retain only on measured benefit with the unchanged workload/configuration and continued serving gates.

Existing probe command form (from checkout root; choose new output paths):

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 timeout 1800 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_native_qk_reader_probe --rounded 0 --state-output temporary --output models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/optimized_native_qk_zero_fill/rounded0_temporary.json
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 timeout 1800 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_native_qk_reader_probe --rounded 1 --state-output persistent --output models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/optimized_native_qk_zero_fill/rounded1_persistent.json
```

Preparation checks: candidate Python AST parsing and `git apply --check` passed. No native compilation or device test was run. Source SHA256 at patch preparation: `1362bb7378f36de7a9d21b851821d0d05544ab23e43da70e236eae3d672581be`; patched-source SHA256: `2856aca4abdd9ca18df4f25b698e37ead9da793cdacf3da4592f0ece5785a3f8`.
