**Implementation plan: Kimi-K3 device-side zero-state initialization (#59988), validated on LoudBox**

Reviewed on 2026-10-09 against local `main` at `48468ff598a`. The LB implementation and bonus are complete; actual test results, measurements, and qualification limits are recorded in [issue-59988-validation.md](issue-59988-validation.md).

Independently reviewed by Claude Code and Codex Astra. The amendments below reconcile both reviews; the findings and their dispositions are recorded in [issue-59988-reviews.md](issue-59988-reviews.md).

Current user-approved scope: implement and validate the KDA-layer change on the local eight-device Blackhole P150b LoudBox (LB). Full-model, Galaxy, and SC4 qualification are follow-up work. Extending KDA-focused transformer tests to LB is a bonus milestone, separate from the core completion gate.

**Objective and scope**

Implement the preferred device-side initialization approach from [issue #59988](https://github.com/tenstorrent/tt-metal/issues/59988): a Kimi-K3 chunk starting at absolute position zero consumes zero recurrent state and zero request history without clearing the persistent tensors. Subsequent chunks consume the previous committed state. Remove the retained zero-state copies and the request-start reset pass after every supported execution path implements this behavior.

The issue estimates approximately 30 MiB saved per chip for an 18-KDA-layer stage, or about 3.7 GiB across SC4. These provide deployment context, not LB acceptance targets. Measure actual per-layer allocated bytes locally, including convolution padding; any projection to larger stages must be labelled as an estimate. [Issue and follow-up comment](https://github.com/tenstorrent/tt-metal/issues/59988).

Keep persistent carry addresses and in-trace commit/export operations. Device-side slot selection belongs to [issue #59977](https://github.com/tenstorrent/tt-metal/issues/59977); this change can land independently and must retain the current traced multi-slot guard until that issue is resolved.

**Goals and evidence for completion**

| Goal | Observable pass condition |
| --- | --- |
| G1. Correct request initialization and continuation | On LB SP1xTP8, SP2xTP4, and SP4xTP2, dirty incoming recurrent/convolution carries are ignored at absolute start zero; positive starts consume the previous state. Synthetic inputs/weights and independent CPU references verify outputs and both carries, including padding and predecessor history. |
| G2. Reuse one trace across requests | A single captured KDA/K3 attention-adapter trace, including persistent-state commit/export, passes `0 -> positive -> 0 -> positive` with device-only bounds, no production host reset, no recapture, and unchanged carry addresses. Full serving-runtime replay is not required for this layer-level gate. |
| G3. Preserve compatibility and bounded scope | Default-off generic callers retain explicit-state behavior. K3 construction enables the policy; policy-enabled SP1 direct execution is rejected before device work. Existing generic direct tests, host-bound validation, eager slot isolation, and program-cache checks pass; the traced multi-slot guard remains. |
| G4. Preserve local state-export correctness | On LB, committed recurrent/convolution slabs match live carries after first chunks, continuation, and slot reuse; import preserves continuation. Host-only tests cover completion gating and missing/partial/timed-out acknowledgements, including rejection when no completion proof covers trailing KDA exports. Remote migration and multi-host ordering are deferred. |
| G5. Remove retained zeros and reset overhead | `_zeros` and production request-start reset calls are gone. LB allocation measurements account for the removed per-layer buffers, and profiles show no request-start reset pass. Native reader inspection and dirty/NaN seed tests establish that first-chunk external seed reads are skipped; the operation-level profile does not measure individual DRAM transactions. Record local first-chunk and continuation latency and pass applicable LB performance gates using synthetic weights. |
| G6. Build and validate reproducibly | `./build_metal.sh --release` succeeds. All required LB cases pass through `scripts/run_safe_pytest.sh` without external checkpoints or golden traces; record commands, meshes, pass/fail/skip counts, and measurement artifacts. A skipped required LB case is a remaining gap. Deferred larger-system cases do not block this milestone. |

These goals cover the K3 grouped paths exercised locally. Direct-policy support, device-side multi-slot trace selection, and a general migration redesign are outside this change. Reset removal still requires G4's local commit/export and host-contract checks; it does not require a remote deployment to be available.

**Required LB validation matrix**

| Configuration | Purpose |
| --- | --- |
| Single device | Native op contracts, generic direct-path regression, invalid configurations, and program-cache checks. |
| SP1xTP8, mesh `(1, 8)` | Grouped local recurrence and external convolution-history initialization. |
| SP2xTP4, mesh `(2, 4)` | Distributed chain, production TP4 head/state shapes, trace reuse, and local slab round trips. |
| SP4xTP2, mesh `(4, 2)` or transposed axes on `(2, 4)` | More chronological ranks, rank rotation, split boundaries, padding, and predecessor history. |

Use the existing `FABRIC_1D` LB profiles and axis-orientation cases. The distributed-chain component suite uses the supported `FABRIC_2D` profile for LB 2x4, so its Galaxy topology markers do not cause LB cases to be skipped. Include small fast fixtures and synthetic production KDA dimensions (`K = V = 128`). No required case may depend on `KIMI_K3_CKPT`, `KIMI_K3_HF_MODEL`, or `KIMI_K3_GOLDEN_TRACE`; none is configured locally and the default golden-trace locations are absent.

For paths below, `model/` means `models/demos/deepseek_v3_d_p/`, and `ops/` means `ttnn/cpp/ttnn/operations/experimental/kda/`.

**Baseline findings before implementation**

| Area | Baseline behavior and implementation implication |
| --- | --- |
| [`KdaStateCache`](models/demos/deepseek_v3_d_p/tt/kimi_k3/kda_state.py) | Allocates persistent carries per slot/layer and an additional `_zeros` pair per layer. `reset()` copies zeros into the selected carry and migration slab. `commit()` overwrites both and frees temporary returned state. |
| [`TtK3KdaAttention.forward`](models/demos/deepseek_v3_d_p/tt/kimi_k3/attention.py) | Already passes device `actual_start` and `actual_end` to `ttKDA` in eager and traced execution. No extra metadata tensor or host readback is needed. |
| [`KDARecurrence`](models/demos/deepseek_v3_d_p/tt/kda/recurrence.py) | Dispatches to distributed, single-rank grouped, or single-rank direct execution. Production K3 always selects grouped execution through `kimi_k3_program_config`; generic direct execution remains unchanged and rejects the new policy until separately supported. |
| [`chain_affine_transforms` reader](ttnn/cpp/ttnn/operations/experimental/kda/chain_affine_transforms/device/kernels/dataflow/reader_writer_chain_affine_transforms.cpp) | Currently starts reading the incoming state before reading `actual_start`, sharing a barrier. The new policy must inspect the scalar before issuing the state read. |
| [`qkv_causal_conv1d_silu` reader](ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/dataflow/reader_qkv_causal_conv1d_silu.cpp) | Already distinguishes request history from predecessor history. Only the request-history branch may become zero; history generated by earlier segments of this chunk remains necessary. |
| [`Noc::async_write_zeros`](tt_metal/hw/inc/internal/tt-1xx/noc_zero_l1.inl) | Already fills local buffers with chunked loopback reads from `MEM_ZEROS_BASE`. KDA scan kernels use it with `write_zeros_l1_barrier()`, so no new DRAM zero-fill operation is required. |
| Existing generic KDA tests | `tests/kda/components/test_chain_affine_transforms.py` uses a random initial state even at start zero. `tests/kda/layer/test_stateful.py` continues a stream while passing zero for chronology on each call. Unconditionally changing generic zero-offset semantics would break these callers. |

**1. Define a compatible state-initialization contract**

- Add an immutable `KDAProgramConfig` field, provisionally `zero_initial_state_on_start=False`, and enable it only at the K3 `build_attention` construction site, using a replaced configuration. Store/thread the policy through `ttKDA` and `KDARecurrence` construction. Keep `ttKDA.forward`'s signature unchanged; all eager and traced calls of that K3 layer then share the same policy.
- Keep both `KDAProgramConfig` and the shared `kimi_k3_program_config()` helper default-off: generic tests also use that helper and must preserve explicit-state behavior. Synthetic K3 adapter tests must explicitly construct a policy-enabled layer; ordinary generic tests remain default-off.
- Reject only policy-enabled **SP1 direct execution**, before expensive weight loading or device setup. SP > 1 always selects grouped execution even when `local_scan_strategy` says `direct`, so that configuration must remain accepted. Add a hardware-free construction-contract test for this distinction.
- With this policy enabled, compute `fresh_request = (actual_start == 0)` on device on every execution. Use the absolute scalar, not its modulo position, `first_rank`, local row offset, slot ID, or the value observed during capture. A later chunk can wrap back to physical rank zero without starting a new request.
- Thread the static policy through Python recurrence orchestration and the external-seed native operations' C++ public functions, nanobind signatures/documentation, operation parameters, and program factories. Include it in the program-cache attributes and reader compile-time arguments. Scalar contents must remain runtime data, so changing from zero to nonzero does not compile another program or require another trace.
- Pass the policy only where the tensor represents the external request carry. Leave it disabled at downstream operations whose inputs are already-computed rank/group entry states.
- Keep input tensors borrowed and immutable inside `ttKDA.forward`. A fresh request ignores their contents; `KdaStateCache.commit()` remains responsible for replacing the persistent state in place.

This static opt-in is a compatibility choice based on the local callers, not an additional device-side request flag. Document the distinction between generic zero-offset chronology and K3 request-start semantics.

**2. Initialize the recurrent seed in every supported K3 graph variant**

| Execution path | External seed consumer | Required routing |
| --- | --- | --- |
| SP > 1 | `chain_affine_transforms` | Enable the policy here. All SP ranks start their replicated transform chain from zero for a fresh request. Later rank entry states still include preceding ranks' transforms. |
| SP = 1, grouped | `affine_exclusive_scan` | Enable the policy for the external state supplied by `_ordinary_group_scan`. Keep applying the group transforms, so later groups get their computed entry states. |
| SP = 1, direct | `recurrent_chunk_scan` | Outside this issue's production K3 scope. Reject policy-enabled direct construction before device work; keep default-off generic direct behavior unchanged. Supporting this combination later requires a separate reader/API change. |
| Scan after a prefix | `affine_exclusive_scan` / `recurrent_chunk_scan` | Disable the affine policy and leave the recurrent scan API unchanged: rank/group entry states may already be nonzero within the first request chunk. |

Implementation details:

- In `ops/chain_affine_transforms/device/kernels/dataflow/reader_writer_chain_affine_transforms.cpp`, retain the raw `actual_start` before the chronology buffer is overwritten with derived topology. When opted in, decide between a carry read and a local zero fill before issuing either. Preserve the existing overlapped read schedule for the default policy if practical.
- Fill the reserved initial-state DFB completely in its FP32 representation. Complete the zero-fill barrier before publishing it or using it to write the first rank's entry state. Maintain existing reserve/push/pop counts and output writes.
- In `ops/affine_exclusive_scan/device/kernels/dataflow/reader_writer_affine_exclusive_scan.cpp`, apply the same decision to the external seed for the single-rank grouped path. Every active group worker reads its own copy of this seed: zero all those copies and execute each worker's zero-fill barrier before `push_back`, including workers outside the existing `reset_worker` branch. Preserve inactive-worker early exits.
- Add native validation rejecting an enabled policy for `affine_exclusive_scan` on SP > 1, where its input is a computed rank entry state. In SP = 1, chronology never creates a local split, so the aliased external `tail_entry_states` passed by `_ordinary_group_scan` is unused; document and test that invariant.
- Leave `recurrent_chunk_scan` and summary mode unchanged. Preserve padded early exits, value-column slicing, intra-chunk tail restarts, and the writer's existing last-valid-group carry placement. Add a construction-time rejection test for policy-enabled SP1 direct execution instead of expanding its native API in this issue.
- Update the corresponding operation wrappers, parameter structures, and `*_program_factory.cpp` files. No compute arithmetic change should be needed when readers publish the same buffer format and number of entries.

**3. Initialize convolution history at the request boundary**

Modify `ops/qkv_causal_conv1d_silu/device/kernels/dataflow/reader_qkv_causal_conv1d_silu.cpp` and its API/factory plumbing:

- Preserve the raw scalar while deriving topology and retain a `fresh_request` boolean.
- In the existing `source_row < row_floor` handling, preserve `read_history(predecessor_carry)` when `initial_from_predecessor || row_floor != 0`.
- Only where the reader would otherwise use the external `history` tensor, substitute a local zero fill when the policy is enabled and the request starts at zero. For other calls, retain the current history read.
- At zero absolute start there is no local split. For `fresh_request && !initial_from_predecessor && mt == 0`, zero the contiguous first `history_rows * block_row_bytes` of the scratch window in one call, complete `write_zeros_l1_barrier()`, and skip the corresponding three history-row reads. Retain the current alignment assertions. Save the scalar-derived boolean first: the scalar initially lands in that same window and will be overwritten.
- Complete the zero-fill barrier before any subsequent NoC write or publication, including `write_state(initial, ...)` in the chain reader. Keep projected token reads, tap assembly, and collective participation unchanged.
- Keep the outgoing convolution carry derived from valid projected tokens. Validate padded first chunks as well as full chunks; committed history must stop at `actual_end`.

A blanket zero fill on every SP rank would erase valid predecessor history. Tests must specifically detect this case.

**4. Validate local slab ownership and the migration completion contract**

The existing [`KDA state migration contract`](models/demos/deepseek_v3_d_p/tt/kda/KDA_STATE_MIGRATION.md) says reset also clears the slab. Replace that promise with an explicit lifecycle:

- A newly assigned or reused slot has no valid KDA slab state for its new request until that request's layer state has been committed and exported. Old bytes may remain until then.
- Preserve the existing in-trace order: compute replacement state, copy into the persistent carry, export both slab regions, then expose completion to a migration reader. Retain the applicable layer/chunk acknowledgement ordering.
- Repair or establish the completion guarantee for the in-repo [`migration_driver.py`](models/demos/common/prefill/runners/migration_driver.py). Its positive-`real_len` resident filter is useful but not proof of execution completion: `run_schedule` records submitted work. The driver's current ack drain uses `NUM_LAYERS * pushes`, whereas K3 emits only MLA acks, and ignores the returned count. Missing channels and timeouts can proceed to migration. `H2DStreamService.barrier()` only waits for input delivery into backing DRAM, not KDA compute/export completion.
- Require completion covering KDA exports before reading/migrating them. Where layer acks fence every requested export, use the model-aware `_ack_layers_per_chunk(kv_table)` count and reject missing/incomplete completion. An earlier MLA ack cannot fence trailing KDA layers or a KDA-only slice; reject migration when no suitable completion proof exists. Do not treat zero expected MLA acks as proof of KDA completion. Supporting additional distributed completion protocols is follow-up work.
- Test the consumer's host logic locally with deterministic completion-channel/client doubles: no migration call on missing, insufficient, or timed-out completion; correct counts and progression when completion is established. Separately exercise real KDA compute, in-trace commit/export, synchronization, and slab readback on LB. Host doubles do not establish remote device ordering.
- Keep this scoped to local state correctness and narrow host completion checks. The ack weakness predates this issue; retain its documented production qualification requirement. No new general per-slot validity protocol is required. Do not mark a replayed commit valid solely from Python `commit()`: it runs during capture, not replay.
- Keep migration import at nonzero `actual_start` consuming the imported carry. Verify both recurrent and convolution slabs after the first commit and after continuation.

Document that external consumers must not read a newly assigned slot before its first completed export. Verification against those consumers and real cross-host transfers is deferred to production qualification, not a blocker for LB layer validation. Keep the existing #59977 guard.

**5. Remove zero-state storage and reset callers after kernel coverage passes**

- Delete `_zeros`, its allocation/deallocation, and `KdaStateCache.reset()`. Retain `_states`, `commit()`, slab binding/export/import, and their stable addresses. Initial allocation can remain zero-initialized; removing allocation-time initialization is unnecessary for this issue.
- Remove the `actual_start == 0` reset branch from `TtKimiK3Runtime.prefill_chunk`. **Preserve `validate_kda_bounds` and validation-before-dispatch behavior.** The override currently provides both validation and resetting, so deleting the entire override would lose a separate correctness check. Remove only imports made unused by the final implementation.
- Remove or deliberately retire `TtKimiK3Transformer.reset_streams()` and update every in-repo caller together. Do not retain a misleading reset method that silently does nothing.
- Update these known reset-dependent tests and harnesses:
  - `model/tests/kimi_k3/test_kda_padding.py`.
  - `model/tests/kimi_k3/test_chunked_prefill.py`.
  - `model/tests/kimi_k3/test_prefill_perf.py`.
  - `model/tests/test_prefill_transformer_chunked.py`, including `_reset_kda_carries` after capture and between iterations.
- Update these call sites even when their full-model hardware tests are deferred, so removing the reset API leaves no dangling references. Validate shared state helpers using the local synthetic fixtures; running their existing Galaxy/checkpoint suites is outside the LB gate.
- Rewrite `test_kda_padding.py`'s reference semantics, not only its reset call: at `start == 0`, build a zero `KDAReferenceState`; otherwise reconstruct the incoming carry. Its native comparison must use a policy-enabled layer too. Leave persistent device state dirty after warmup/capture and between requests, and retain its stable-address and eager/native/trace comparisons.
- In `test_runtime_contract.py`, verify that K3 construction enables the policy while the generic config helper stays default-off. Keep exact forward-kwargs checks, since the flag is constructor-owned. Rename the `before_reset_or_replay` test and remove the vacuous assertion on a deleted `Mock.reset` attribute; assert invalid bounds prevent parent dispatch and valid bounds delegate without resetting KDA state.
- For request-start runs, restart each iteration through device position zero and execute the same graph. Remove the request-start resets in `test_chunked_prefill.py` and `test_prefill_perf.py`; also cover the single-chunk perf call where host `actual_start=None` maps to device zero.
- Preserve the shared harness's existing synthetic `preload_isl > 0` benchmark semantics with a harness-local helper: allocate one temporary zero state per layer/slot in turn, install it via the existing in-place `KdaStateCache.commit()` path, and free it through that commit. Run this after capture and before repeated measurements, outside timed regions. It also restores bound slabs without retaining `_zeros` or adding a production reset API. These zeros are a deterministic synthetic seed, not the true state of a prefilled prefix; any positive-offset correctness claim must instead restore matching reference prefix state or reject that unsupported test case.
- Update docstrings in the cache, attention, runtime, transformer, KDA layer, and migration contract to explain device initialization and slab validity. Re-scan for obsolete reset references.

**6. Validate KDA-layer behavior, traces, memory, and latency on LB**

Add focused coverage to the existing suites; use their reference functions and numerical thresholds.

| Test | Required evidence |
| --- | --- |
| Dirty incoming state, first chunk | With the K3 policy enabled, deliberately different nonzero recurrent and convolution carries produce the same output and returned state as an explicit-zero baseline at `actual_start == 0`. Include NaN poison where the test utilities support it to detect accidental reads or multiply-by-zero masking. |
| Continuation | At positive starts, the result matches the explicit carried-state reference and differs from a reset baseline. Include a start equal to a complete physical SP cycle, where rank zero is first again. |
| Supported recurrence variants | Cover SP1 grouped and distributed SP; include production K/V=128 and a smaller fast case. Preserve default-off direct regressions and verify that policy-enabled SP1 direct construction is rejected. |
| Convolution boundary | Verify zero external history at the request head while later SP ranks still use predecessor tokens. Cover rotated/split continuation and both mesh-axis orientations. |
| One trace, multiple requests | Capture the K3 attention plus in-place commit, then replay `0 -> positive -> 0 -> positive` with different request inputs. No host carry resets, buffer replacement, or recapture. Compare each request with its reference and verify persistent addresses remain unchanged. Also exercise capture starting at a positive offset. |
| Device-only bounds and host contracts | Exercise the two-request sequence through real `TtK3KdaAttention` with persistent device start/end scalars and absent host bounds. Keep host-only runtime tests for validation/delegation and construction policy. Full `TtKimiK3Runtime` serving integration is deferred. |
| Warmup and capture | Leave dirty state from warmup/capture, then run the first real request at zero and verify independence from those earlier executions. |
| Padding | Include the shortest supported nonempty aligned interval, full chunks, partial final chunks, and inactive SP ranks/groups. Compare valid output rows and both returned carries. |
| Slot reuse and isolation | Alternate two slots eagerly; restarting one must leave the other's state intact. Keep traced tests single-slot until #59977 is fixed. |
| Local slabs and completion contract | On LB compare real exported/imported state with the live carry, including reused slots and nonzero-start continuation after import. Use host-only tests to verify rejection of missing/incomplete completion and configurations where an earlier MLA ack cannot cover later KDA exports. No remote migration client is required. |
| Compatibility and caching | Default generic operations still honor nonzero initial state at zero offset. Test both policy values and verify that scalar value changes reuse the compiled program; policy values must not collide in the program cache. |
| Host validation | Supplied invalid or unaligned host bounds still fail before dispatch/replay after reset removal. Device-only scalar contents remain the caller's aligned, nonempty-interval contract; preserving host validation does not validate them. |

Primary test locations:

- Native readers: `tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_chain_affine_transforms.py`, `test_affine_exclusive_scan.py`, and `test_qkv_causal_conv1d_silu.py`; extend eligible `test_padding_prefix.py` cases for both policy values. Retain `test_recurrent_chunk_scan.py` and `test_padding_recurrence.py` as downstream/default-policy regression coverage.
- Distributed seed and orchestration: `model/tests/kda/components/test_chain_affine_transforms.py`, `test_recurrence.py`, and `test_convolution.py`.
- Layer behavior: `model/tests/kda/layer/test_stateful.py`, `test_dynamic_trace.py`, `test_actual_start.py`, and `test_padding_early_exit.py`.
- K3 layer integration: `model/tests/kimi_k3/test_runtime_contract.py` and `test_kda_padding.py`. Bind local `KdaStates` slabs to the synthetic adapter/cache fixture so the request-restart trace covers commit/export as well as the layer output.
- Local slabs: the `2x4` cases of `model/tests/kda/test_state_adapter_device.py`, plus host-only `model/tests/kimi_k3/test_kda_migration_stages.py` and completion-consumer failure tests. New host-contract tests should live alongside the existing host-only coverage.

From the project root, build the modified code and TTNN bindings with the required release command. Complete the build successfully before testing:

```bash
./build_metal.sh --release
```

Run all tests through [`scripts/run_safe_pytest.sh`](scripts/run_safe_pytest.sh), including newly added tests, local integration checks, and performance tests. It accepts a test path followed by normal pytest arguments, serializes hardware access, and handles device hangs. Select LB cases explicitly in files that also contain Galaxy parametrizations. Representative focused commands are:

```bash
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kimi_k3/test_runtime_contract.py -q
scripts/run_safe_pytest.sh tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_chain_affine_transforms.py tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_affine_exclusive_scan.py tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_qkv_causal_conv1d_silu.py tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_padding_prefix.py -q
scripts/run_safe_pytest.sh tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_recurrent_chunk_scan.py tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_padding_recurrence.py -q
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kda/components/test_chain_affine_transforms.py -k SP2xTP4 -q
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kda/components/test_recurrence.py -k 'not distributed' -q
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kda/components/test_recurrence.py -k distributed -q
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kda/components/test_convolution.py -q
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kda/layer/test_stateful.py -q
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kda/layer/test_dynamic_trace.py models/demos/deepseek_v3_d_p/tests/kimi_k3/test_kda_padding.py -q
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kda/layer/test_actual_start.py models/demos/deepseek_v3_d_p/tests/kda/layer/test_padding_early_exit.py -q
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kda/test_state_adapter_device.py -k 2x4 -q
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kimi_k3/test_kda_migration_stages.py -q
KDA_PERF_SKU=bh_loudbox scripts/run_safe_pytest.sh 'models/demos/deepseek_v3_d_p/tests/kda/perf/test_layer_perf.py::test_synthetic_kimi_k3_perf[blackhole-SP2xTP4-fabric-1d]' -q
```

These commands target existing test names; add the new policy, request-restart, memory, and host completion-contract cases to the appropriate local suites during implementation. A skipped required LB case is not validation. The synthetic performance node must be extended to exercise the enabled policy as described below. If using the wrapper's optional `--profile` mode, first pass correctness in normal mode: profiling can mask pytest's failure exit status, so a profiling-only PASS is not correctness evidence.

Measure before/after on the same geometry and revision setup:

- Persistent DRAM allocation after a synthetic LB cache is constructed, accounting separately for carries, removed zero copies, and retained migration slabs. Confirm per-layer savings at TP4 using one-layer and a small multi-layer cache; `_zeros` is per layer, not per slot. No full transformer or 18-layer stage allocation is required.
- KDA-layer/attention-adapter first-chunk wall time including the old reset cost, and steady-state continuation latency separately. The chain reader now needs metadata before selecting its source, so check the extra dependency's cost. This measures local layer latency, not full-model throughput.
- Device program profiles showing no request-start reset copies or zero-slab exports; retain ordinary commit/export traffic. Verify the absence of first-chunk external carry reads through native source branches and dirty/NaN seed tests. The available operation profile does not resolve individual DRAM transactions, so do not claim a hardware bus-counter measurement.
- Extend `test_synthetic_kimi_k3_perf` in `model/tests/kda/perf/test_layer_perf.py` for enabled-policy zero-start and positive-start measurements on LB; retain a default-off control and the applicable LB baseline. Its unchanged generic calls measure only the compatibility path. No checkpoint-backed layer or transformer benchmark is required for this milestone. Quantify any regression instead of assuming local zero fills are free.

**Bonus: KDA-focused transformer tests on LB**

- Extend the transformer test infrastructure around `model/tests/kimi_k3/test_transformer_depth.py` and `test_chunked_prefill.py` with an explicitly named LB case. Begin with a one-layer KDA-only prefix, a small supported chunk length, and an eight-device placement such as `(2, 4)`; add SP1 coverage if the fixture supports it without expanding scope.
- Provide a small synthetic model configuration and deterministic synthetic weights/reference for this case. Exercise the real transformer orchestration, K3 attention adapter, and KDA kernels; do not mock KDA computation. The existing `PLACEMENTS` list alone is insufficient because its full-model tests also require checkpoints/golden traces and production model dimensions.
- Compare one-shot and chunked outputs/carries against an independent reference, test a second request after dirty state, and retain slab and address-stability checks. Use existing synthetic adapter/layer reference utilities where possible, without making the implementation its own oracle.
- Keep the existing Galaxy/checkpoint cases intact. Select the new case explicitly through the safe runner, for example an `LB` parametrization ID once introduced. Its additional configuration/weight fixture work is a separate bonus deliverable and does not block G1–G6.

The implemented bonus fixture is `model/tests/kimi_k3/test_transformer_kda_loudbox.py`, using `LB-SP2xTP4`. It runs a one-layer transformer with real embedding, RMSNorm, KDA and dense FFN, deterministic synthetic weights, and the explicitly selected plain residual arm. This isolates KDA orchestration; full AttnRes/checkpoint qualification remains in the existing depth tests. Run it with:

```bash
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kimi_k3/test_transformer_kda_loudbox.py -q
```

**Deferred production qualification**

- SP8xTP4 on a 32-device Galaxy, production fabric/topology behavior, and multi-host SC4 pipelines.
- Full-model checkpoint/golden correctness and the existing Galaxy-only transformer depth, chunked-prefill, and prefill-performance runs.
- Real cross-host migration and verification of external consumers' completion/empty-prefix contracts.
- Measured SC4-wide memory savings and end-to-end production throughput. Local extrapolations are estimates only.

These are explicit follow-up tasks, not skipped requirements for LB completion. Report the finished work as **KDA-layer implementation validated on LB**, without claiming full production qualification.

**Suggested implementation sequence and completion criteria**

1. Land the opt-in native reader policies, orchestration, and component tests with host reset still available.
2. Enable the policy at K3 construction; prove dirty-state starts, continuation, and request restart in one real KDA/adapter trace across the LB matrix, including device-only bounds and local slab exports.
3. Pass local state and host completion-contract checks, remove `_zeros` and reset APIs/callers together, and collect LB correctness, allocation, and performance evidence. Update deferred harness call sites without requiring their remote execution.
4. As a separate bonus, add the synthetic KDA-focused transformer LB case. Track production qualification separately.

Core LB completion checklist (all items require recorded local evidence):

- [x] G1: request-start, continuation, convolution, and padding references pass on the required LB grouped-path matrix.
- [x] G2: one KDA/attention-adapter trace handles two requests from device bounds alone, including commit/export and stable carry addresses.
- [x] G3: default-off compatibility, unsupported-path rejection, validation, slot isolation, and caching checks pass.
- [x] G4: local slab round trips and host completion/failure checks pass before slab reset removal; no remote migration is required.
- [x] G5: retained zero buffers and production reset passes are removed; LB allocation/traffic measurements and layer-latency gates confirm the result.
- [x] G6: the release build and required synthetic safe-runner tests pass; no checkpoint or larger-system dependency remains in the core gate.

Bonus checklist:

- [x] A synthetic KDA-only transformer case runs and passes on LB through the safe runner; existing Galaxy cases remain available for later qualification.
