# Batched-fill production integration proposal — not applied

Prepared `batched_cache_fill_integration.patch` (new ownership module + main decoder + flat prefill) and a separate `batched_cache_fill_runner.patch`. Both pass `git apply --check` against the inspected workspace; no production file was modified by this preparation. Python syntax checks and lease alias tests passed in `batched_cache_fill_integration_cpu.json` without device imports.

Parent evidence in `batched_fill_perf.json` supports selecting B>1 only: B32S33 eager region1052.871→56.438 us and trace287.159→16.822 us; B32S513 eager1394.363→100.141 us. B1 eager regressed27.302→33.975 us at S33 and38.644→44.515 us at S513. These are focused-region measurements, not whole-layer gains.

## Why a permanent boolean is insufficient

`state.page_table` is a mutable TTNN tensor. The existing runner calls `ttnn.copy_host_to_device_tensor(..., state.page_table)` at line321; a caller can also retain that tensor and write it later. Its buffer address/shape remain unchanged. No exposed content-mutation version was found in the Tensor public header/bindings, so address identity cannot certify current physical-page uniqueness. Requiring callers to remember an invalidation function while leaving arbitrary raw aliases eligible would silently weaken the existing mutable-table contract.

The proposal instead revokes the certificate **when the raw tensor escapes**:

- With policy off or B1, `create_paged_state` returns the original `PagedState` and behavior is unchanged.
- With default-off `batched_prefill_cache` enabled and B>1, setup uploads a private table, certifies the actual CPU table after allocator padding, preallocates replicated identity indices for B, and returns a `CertifiedPagedState` subclass of the original dataclass.
- Public `state.page_table` reads return the same live tensor callers expect, but permanently revoke that table lease before returning it. Public assignment revokes the old lease and installs an unknown, uncertified one. Thus existing raw in-place writes and replacements remain valid, and subsequent prefill uses the original ordered loop. Even a read-only public read is conservatively treated as possible future mutation.
- The lease object is shared by shallow state copies. Exporting through one copy invalidates all aliases of that old table. A copied layer contains no closure bound to another layer; immutable identity indices may be shared on the same mesh, while table leases belong to request states.
- Internal attention code uses a private read-only accessor. Frozen helpers receive a temporary ordinary `PagedState` view inside trusted internal calls. No frozen source is edited. Safe public shape/address metadata properties and `repr` do not export the tensor.
- `replace_paged_table_from_host` is an explicit **setup-only** operation: it checks shape/cache bounds, uploads a fresh private tensor, and certifies that new allocation. It never recertifies an already-exposed tensor. Old retained aliases still point to the old revoked lease and cannot mutate the fresh table. Existing traces must be released/recaptured before changing addresses; the method is not a runtime-forward operation.

Global page uniqueness is conservative: -1 skip entries are ignored, but any repeated nonnegative physical page disables batching, including shared read-only prefixes. Such states retain ordered fills. After the allocator performs copy-on-write and supplies a completely disjoint fresh CPU table, setup can establish a new eligible lease. Unknown ordinary/external states always use the old loop. More selective per-write-window certification can be considered separately; this patch does not infer that shared pages are safe to overwrite.

As with the original mutable state contract, callers must serialize table mutations with forward execution. Private fields/internal read accessors are not public mutation APIs.

## Captured-prefill trap and deliberate boundary

A Python certificate check cannot revoke an already-captured bulk program. If callers export/mutate the table after capture, replay bypasses Python and could race aliased writes. The proposal therefore enables bulk fill only for **eager flat prefill**. During capture it selects the ordered loop, which preserves the public mutable-table replay contract. The source-audited check is `ttnn.is_trace_capture_active(mesh)`: exposed in `ttnn/ttnn/device.py:77`, re-exported in `ttnn/ttnn/__init__.py:350`, and implemented in `ttnn/cpp/ttnn-nanobind/device.cpp:740` by inspecting active trace IDs on all command queues. This is a metadata query, not a device readback/synchronization.

The current production harness times eager prefill and traces decode, so the measured optimization target remains eligible. Focused bulk trace timings demonstrate native performance but are **not** claimed for this conservative production capture path. A future immutable-table execution context with trace lifetime ownership could enable captured bulk separately.

For a prefill-trace test, warm the ordered path before capture as well as the bulk path; temporarily disable the bulk policy during setup warmup, then restore it before capture. Otherwise the capture-dependent fallback can introduce an uncompiled program. The manifest explicitly includes this prerequisite. No tracker exemption is proposed.

## Main/flat patch scope

The default-off policy routes only selected B>1 flat prefill through `_prefill_cache_fill`. B1 and legacy tails use the inherited ordered implementation. The helper chooses the two-call native implementation only for a private certified lease, preallocated B indices and eager execution; otherwise it spells out the original table slice and per-user K/V loop. Cache dtype/layout, upstream casts/padding, QKVG packing and batched RoPE remain unchanged.

Internal page-table reads in compact decode, ordinary decode and batch discovery use the private accessor. Wrappers around inherited `_attention_chunk`, `_decode_attention_chunk` and `_attention_prefill` pass internal views, so routine scheduler reads and one-token tails do not accidentally revoke a normal lease. This is necessary: merely changing the two fill loops would lose certification immediately in `_attention_chunk`.

State construction and fresh-table replacement perform all host certification/uploads. Forward performs no host tensor conversion and allocates no identity tensor. Bulk/ordered dispatch counters are plain per-instance integers, allowing evidence that the requested path actually ran. Immutable identity-index buffers add B*4 logical bytes per prepared batch shape, plus allocator padding; include their actual allocation in resource accounting.

## Test harness consequence

The existing full-layer runner legitimately exposes and mutates tables during trace-case preparation. That **must** revoke the lease. If used unchanged, it would subsequently time the ordered fallback and incorrectly suggest batching had no effect. The separate runner patch uploads a fresh certified table from its already-saved `physical_table` CPU fixture **after** those tests, before timed eager prefill and before decode capture. It records counter deltas and fails an eligible candidate if timed prefill never selected bulk. All existing PCC, state, trace, remapping and allocation-tracking gates remain unchanged.

Resource harness introspection should use `page_table_shape`/`page_table_address` for certified states, with ordinary property access for legacy states. Its current raw `.page_table` reads would safely force fallback. Do not hide this by re-enabling a stale boolean. Extend the static runtime auditor to traverse the new private accessors, wrappers, `page_table_certified`, `_prefill_cache_fill` and native helper; honestly exclude CPU setup certificate/upload functions.

## Concrete validation plan

`batched_cache_fill_test_plan.json` contains nine exact whole-layer argv entries in control/candidate/control order for L3 B32S33, B32S513 and B1S512. Both policy files retain the profile4 attention policy; their only difference is `batched_prefill_cache`. Commands are conditional on applying both proposed patches; nothing was executed here. Existing fixture paths, output paths and .995/output/state/per-user/contribution gates are explicit. Candidate B>1 must report nonzero timed bulk counter; B1 and controls must report zero.

The manifest also specifies ten required ownership/public-boundary controls: shared prefixes, external states, retained raw aliases, public replacements, shallow state copies, fresh COW maps, capture-time fallback followed by mutation, metadata-only inspection, B1, and unaligned public starts crossing into bulk. Current `run_multichip_contract.py` has no policy argument and cannot by itself certify this candidate; its harness needs a setup-only policy wrapper plus the listed B>1 cases. Mark those controls incomplete until executed. Compare aliased-table semantics to the original ordered implementation; do not impose an HF interpretation on deliberately conflicting writable aliases.

Before adoption: focused remap/skip controls, proposed ownership tests, whole-layer paired gates/timings, resource ownership/capacity, runtime host-call audit, and a matched profile showing two native fills per certified eager bulk chunk. The full public context/start/length contract remains governed by the existing scheduler and its token-wise unaligned fallback.
