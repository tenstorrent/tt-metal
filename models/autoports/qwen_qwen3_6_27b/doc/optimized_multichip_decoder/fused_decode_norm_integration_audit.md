# Bounded decode norm integration candidate

CPU-only preparation, 2026-11-13. This is an **unapplied opt-in patch**, not a hardware result or a selected default. No TTNN import, device access, build, main-source edit, or native-source edit was performed. `fused_decode_norm_integration_manifest.json` pins the seven before/after files. `fused_decode_norm_integration.patch` adds `tt/fused_decode_norm.py` from the adjacent runtime candidate, integrates setup/decode routing, extends 64-owner resource receipts/checks, and adds conservative context accounting. Default `distributed_decode_norm="off"` preserves the current path. Explicit `"dit"` and `"minimal"` choices never fall back to each other.

## Ownership and trace contract

One actual decoder layer owns one fixed allocation set. Constructor records its real `layer_idx`; resource clones assign their real index and install a fresh owner before execution. Owner identity rejects shallow-copy reuse. Setup occurs after packed projection setup, so input/post canonical layouts may differ and packed GDN correction uses the actual joint projection key. `_norm` dispatches here only for decode. Input/post calls must be paired, ordered, and use the same logical B. Both phases use the actual corresponding FP32 `shard_norms` gamma, including the `(1+weight)` convention at `tt/multichip_decoder.py:153`. Minimal converts that FP32 tensor to RM `[1,1,40,32]` during setup without host conversion or BF16 gamma rounding.

Input norm is phase A, post norm phase B, making complete eager/queued/captured decoders execute A,B,A,B. A trace contains both phases; replay uses static addresses without a Python phase counter advancing. Do not capture/replay an isolated one-phase norm. CQ0 complete forwards only; no concurrent invocations on the same owner. A partial failed forward latches an invalid phase and requires rebuilding the owner after safe device recovery, not resetting a potentially live semaphore. State restore verifies the idle boundary but does not write norm semaphores or stats.

DiT needs this alternation: `models/tt_dit/tests/unit/test_distributed_rmsnorm_fused.py:598` explicitly warns that consecutive calls reusing a semaphore/stats pair can race terminal increments/reset. `dit_fused_norm_common/kernels/dataflow/dit_fused_norm_forwarder.cpp:173` waits for arrival and later resets output/arrival state before fabric closure; the DiT worker writer resets its local completion semaphore at `dit_rmsnorm_fused_worker_writer.cpp:238`. Separate phase sets preserve the tested native pattern. Minimal also receives two independent stats/output/semaphore sets; its `rms_writer.cpp:163` performs fabric writes, arrival wait/reset, multicast drain and connection closure. This is a source-based protocol argument, not a measured integrated trace guarantee.

Normalization and subsequent activation AG never share semaphores. The helper creates exactly four AG semaphores and two barriers, plus four DiT or two minimal norm semaphores. `models/demos/gpt_oss/tt/ccl.py:36` otherwise allocates a full 12-semaphore manager; reusing its rotating pool for both operations would mix unrelated protocols. The decoder's existing CCL manager remains separate. TP4/BF16/flat decode/DRAM-sharded projection/packed MLP are explicit prerequisites.

All device storage is allocated at setup. Two physical B32 AG outputs cover logical B1..32; minimal similarly owns two B32 native outputs. Setup precreates logical views with explicit padded shape using `ttnn.reshape`, and asserts identical memory config and buffer addresses on all four devices. Native physical tile height stays 32 for every B. `reshape_view/reshape.cpp:754` supports unchanged-width, tile-aligned padded-volume views; Python's experimental `view` binding does not expose the equivalent padded-shape overload. Shape/address assertions deliberately fail if this contract changes. There is one device-storage signature, not 32 cached allocations. Warmed-batch metadata is bounded to 32 entries. Normal transient forward allocations remain inside trace capture, as in the current decoder.

Rank-local residual stays `[1,B,1280]` BF16 at the layer boundary. Global statistics communication and normalized-activation AG are inside each layer; the canonical gathered activation feeds the projection and is not a restored replicated residual. Both families include input conversion, statistics exchange, local normalization, activation AG, and final projection-input conversion in the candidate boundary.

## Minimal offset placement source audit

On the target 11x10 worker grid, minimal uses eight cores `(0..7, layer_idx % 10)` and stats on `(0, layer_idx % 10)`. Source paths below are under `ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/`:

- `rms_allgather_program_factory.cpp:57` derives relative compute dimensions plus the absolute shard-grid offset; `:192` retains the actual shard core set.
- `:400` offsets sender/all-to-all groups; `:425` subtracts actual absolute core sets for remaining workers, avoiding phantom rectangular cores.
- `:873` builds physical NoC x/y lists using the offset; `:891` enumerates actual worker cores while width indices remain relative. Gamma shard offsets depend on width index, not absolute y.
- `:940` derives multicast corners from the absolute start core; `:956` and `:975` pass relative sender/receiver indices alongside the offset physical-coordinate lists.
- `:1018` derives stats reads and drain semaphore coordinates from the actual stats shard grid; `:1130` forms the other multicast rectangles with offsets. `:1295` recomputes the same layout on cache hits and patches stats/gamma/semaphore runtime addresses.
- `kernels/dataflow/rms_sender_reader.cpp:50` and `rms_receiver_reader.cpp:44` consume those coordinate arrays; they do not substitute row0. `rms_writer.cpp:118` consumes supplied stats coordinates, and `:155` uses the supplied drain semaphore physical coordinates. Factory line362's temporary `{0,0}` is transformed before runtime args are emitted.

This supports a bounded row-offset experiment; it does not prove the native implementation has no hardware-only offset issue. Rows0,1,9 with watcher, followed by mixed 64-owner stress at B1/B32, remain mandatory before keeping minimal.

## Persistent capacity and the reported L1 collision

Per minimal layer, each phase has 81,920 bytes of BF16 native output over eight cores and 16,384 bytes of FP32 stats on the sender. Two phases therefore cost 53,248 bytes on the sender, 20,480 on each other worker, and 196,608 total tensor bytes per device. Both phase gammas add 10,240 DRAM bytes; two DRAM AG outputs add 655,360. Values below are payload only, excluding alignment, semaphores, native program CBs, existing tensors, and transient canonical/projection inputs.

| Placement of 64 independent owners | Owners on a row | Sender-core persistent tensor bytes | Each other worker's bytes |
|---|---:|---:|---:|
| All fixed at row0 | 64 | 3,407,872 | 1,310,720 |
| Real indices, rows0..3 | 7 each | 372,736 | 143,360 |
| Real indices, rows4..9 | 6 each | 319,488 | 122,880 |

Spreading is necessary; the table does **not** establish that live native CBs fit. DiT has no new persistent L1 tensor when AG is DRAM; native CBs and transient projection L1 still matter. DiT stats use an enforced per-page upper bound of 65,536 bytes, conservatively eight DRAM banks per tensor. All64 DRAM additions are bounded by 109,051,904 bytes for DiT and 42,598,400 for minimal with DRAM AG. Existing capacity logic still reserves the legacy CCL inventory too, conservatively overcounting bypassed norm AG slots. The patch adds these bytes and leaves the accepted context value unchanged; neither an estimate nor a probe allocation failure authorizes a context reduction.

Parent reported all32 minimal focused cases passing and 18 DiT cases before `dit_norm25_l3_b1_post_quality.log` failed during setup: static CB end1,016,320 exceeded live-buffer start1,016,064 by256 bytes. No PCC failure occurred. The focused probe retained six canonical L1 verification outputs and six projected L1 verification outputs. This helper retains neither category: canonical projection input and projection output follow ordinary transient decoder lifetimes. Parent is moving only retained verification copies to DRAM, preserving the actual canonical-to-projection L1 path for both arms. That rerun can distinguish probe retention from a production conflict; it cannot replace whole-layer and 64-owner live-L1 measurement. The integration itself remains entirely unrun.

## Receipts, CPU checks, and required parent validation

The patch extends resource receipts with family, real layer index, phase keys, projection keys, bounded batch metadata, semaphore and persistent tensor addresses, all four rank addresses, and actual warmed batches. Semaphores remain unique across existing CCL/conv owners; new tensor owners remain unique across phases/layers and each rank, with legacy state/CCL cross-checks retained. The original 912 legacy semaphore and 128 independent state checks are unchanged. Final validation snapshots already include all `tt/*.py`; the standalone resource report additionally pins the new helper and relevant native factories. Native loaded-library/build provenance still comes from the existing validation runner. Pure CPU validator cases cover off/dit/minimal plus nine deliberate failures, including a remote-rank alias.

Executed CPU-only: compile six proposed Python files in memory; check 128 logical/padded geometry combinations; check capacity and 64-owner placement; exercise three positive and nine negative resource-validator fixtures; check forward has no recursive `_norm`, persistent `empty`, clone or semaphore creation; `git apply --check` succeeds. Results are in `fused_decode_norm_integration_cpu.json`. The initial CPU geometry fixture omitted tile height and was corrected; runtime code was unchanged by that fixture correction.

After parent review and a closed queue, inspect manifest before hashes, then:

```bash
git apply --check models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/fused_decode_norm_integration.patch
# Parent applies only after deciding the proposed seven-file patch is appropriate.
```

Merge `distributed_decode_norm: dit` or `minimal` into the otherwise unchanged candidate policy for each kind; never substitute a sparse policy for the selected controls. Use the existing `run_job.sh` lease/environment/provenance wrapper. Representative module arguments (reference and policy paths must point to the existing matching assets):

```text
-m models.autoports.qwen_qwen3_6_27b.tests.run_multichip_decoder --layer 0 --batch 3 --length 33 --trace --stress 50 --repeats 100 --policy-file POLICY --reference MATCHING_REFERENCE --output NEW_JSON
-m models.autoports.qwen_qwen3_6_27b.tests.check_multichip_stack_resources --batch 1 --prefill-batch 1 --prefill-length 65 --replays 50 --policy-file MIXED_KIND_POLICY --output NEW_JSON
-m models.autoports.qwen_qwen3_6_27b.tests.check_multichip_stack_resources --batch 32 --prefill-batch 1 --prefill-length 65 --replays 50 --policy-file MIXED_KIND_POLICY --output NEW_JSON
```

First verify offset rows0/1/9 in the focused minimal probe. Then test complete input/post pairs with different actual gammas, both layer kinds, B1/3/17/32 and all32 batch metadata views. Include changed inputs, rebinding, queued50 pairs, multiple traces with different B and no intervening external live allocation, allocation-tracker positive control, and watcher separately from timings. The existing decoder runner covers full decoder state/output gates; a targeted owner sequence must additionally establish all32 view sharing and no phase-cross contamination. Preserve all aggregate/per-row/group PCC thresholds and HF references. Re-run boundary instrumentation to establish no new inter-layer collective/conversion. Compare complete warmed/traced decoder paths with both families and off under matched native builds. Validate64 independent owners, final context capacity, and B32 512-step stress before selecting defaults. Neither standalone norm savings nor prior focused passing cases establish integrated speed, correctness, trace safety, or maximum context.

## Review amendment: minimal SwiGLU decode destination

The reviewed constructor now rejects `minimal_swiglu is True` before allocating any candidate resources. `multichip_decoder.py:559` sends that decode branch through DRAM into `experimental.minimal_matmul`; returning the packed gate/up canonical L1 input would introduce an unnecessary L1 conversion followed immediately by DRAM restoration and misdescribe the measured consumer contract. The integration deliberately supports the selected `minimal_swiglu="prefill"` policy (and False/absent), whose decode consumer remains `mlp.gate_up.weight`. This is an explicit unsupported combination, not a runtime fallback or a rejection of the separate minimal SwiGLU family. Supporting that family later requires choosing DRAM as the post-norm destination and measuring the complete matching consumer boundary.

CPU checks execute the isolated constructor guard for absent, False, "prefill", and True policies and verify its placement before storage allocation. All four cases behave as intended. Parent separately reports offset-row1/9 B1/B32 focused probes passing; this CPU-only integration remains unrun and actual64-owner validation remains pending. Patch and manifest were regenerated and `git apply --check` repeated after this amendment.
