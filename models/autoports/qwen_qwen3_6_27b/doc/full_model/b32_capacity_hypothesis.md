# B32 preparation allocation hypothesis

Independent source-only AutoFix investigation, 2026-11-14. No devices were opened, no implementation was changed, and no capacity reduction is justified by this note. Preserve the selected TP4 decoder, datatype/fidelity/CCL policies, physical paged-cache split, residual ownership, and all prepared signatures.

## Failure and what it establishes

`slots32_all64_pool.log:3188` records the all64/B32/context262144/P8224 preparation failure. The completed single-row helper warmed all512 lengths. The uniform helper printed160/512, then failed in `BoundedPrefill.prepare -> Model.terminal -> ttnn.matmul`. Preparation precedes the first generator trace capture.

| Allocator quantity | Bytes per bank | Bytes per device, eight banks |
| --- | ---: | ---: |
| General DRAM capacity | 4259840384 | 34078723072 |
| Already allocated | 4081165120 | 32649320960 |
| Free | 178675264 | 1429402112 |
| Largest free block | 51515008 | 412120064 equivalent |
| Failed output request | 95354880 | 762839040 |

The immediate allocation is fragmented: each bank has more free bytes than the request but no sufficiently large contiguous block. The aggregate-equivalent largest block is an accounting convenience, not one cross-bank contiguous allocation.

The request is exactly `32 * 192 * 62080 * 2`, the BF16 local LM-head result when logical length161 pads to192. The native compile arguments immediately before the failure also contain161. At512, the same head requires2034237440 bytes/device, exceeding even the current aggregate free space. Therefore "only fragmentation" is not established for the complete preparation loop; subsequent program growth and transient lifetimes still matter. This failure is not evidence that the per-request context262144 must shrink.

Reproduction recorded by the parent:

```text
python -u -m models.autoports.qwen_qwen3_6_27b.tests.probe_full_slots --batch 32 --all-layers --prefill-length 33 --context 262144 --physical-pages-per-slot 257 --full-prepare --output models/autoports/qwen_qwen3_6_27b/doc/full_model/slots32_all64_pool.json
```

The physical pool is8224 pages, not32 independent8192-page pools. Profiler was disabled; allocation tracking and its program-cache checks were enabled. Preserve the failed log and command record.

## Persistent CCL: real cost, but not a growing B32 long-shape explanation

`tt/multichip_decoder.py:125` selects `flat_ccl=True` for both layer kinds. Both selected policies enable persistent CCL and use DRAM for decode CCL. `_collective:459–469` first reshapes to `[1,1,B*S,width]`, then tests `x.shape[-2] <= 32`. Thus persistent allocation is governed by the **packed** row count, not logical sequence length alone.

- Single-row preparation can retain separate logical M=1..32 signatures even when their padded tile shape is the same. The key includes the logical shape, reduction flag, dtype, and memory config.
- Uniform B32 calls with S>1 have packed M>32 and cannot add persistent buffers for those large shapes through this predicate. The single-token head helper has M32 and reuses its fixed signature; a different statistics signature can be additional, but it does not scale with every long sequence length.
- `BoundedPrefill._prepare_layer_buffers:279` makes private equivalents for matching layers only when a signature is absent. It does not alias mutable buffers across layers or duplicate an existing signature.
- The copy pass is **after the entire length loop** at `bounded_prefill.py:359`. The completed B1 pass has replicated its short signatures; the failing uniform pass has not reached its own replication pass. Attributing the failure to per-layer replication of all B32 lengths is inconsistent with this control flow.
- Layer0 has final-norm weights in its weight inventory and may form a separate representative signature from the other GDN layers. Do not assume exactly two representatives without enumerating `_signature` results.

For scale, one short BF16 hidden all-gather output plus one RS intermediate/output consumes327680+655360+81920=1064960 bytes/device/layer. Thirty-two signatures across64 layers would consume2181038080 bytes. This is an illustrative allocation model, not an exact count: statistics gathers, dtype variants, final-norm ownership, and actual signatures require an inventory. The existing `tests/full_model_evidence.py::ccl_inventory` can report actual live owned CCL buffers without reading device tensor values. Add the signature/logical-M breakdown when diagnosing growth.

## Definite preparation-only overlap

At the first uniform length greater than1, `bounded_prefill.py:352` allocates512 distinct clones of `one`, and the local `concat_inputs` retains them throughout every later terminal warmup. Each B32 one-token hidden tensor has logical shape `[32,1,1280]`, padded shape `[32,32,1280]`, and BF16 consumption2621440 bytes/device. The pool therefore consumes **1342177280 bytes/device**. It is deliberately distinct storage because concat hashes the input alias partition; replacing it with `[one] * N` would not warm the runtime signature.

The same loop replaces its real full-attention result with the concat result, then warms both all-position and final-position terminal outputs while retaining that entire pool. Other locals (`value`, `rope`, `result`, `row`, `one_rope`, `one`) can also remain live until rebinding or function return. Individual hidden activations are much smaller than the vocabulary-wide logits, but their liveness should be counted at the failing call rather than after exception unwinding.

A minimal candidate repair is to put distinct-input concat preparation in its own helper-local phase, preserve all512 concat arities and the actual distinct-input alias partition, and release that pool before all-position terminal warmup. The existing real `_layer_chunk` result has the required `[B,S,1280]` terminal shape. Preserve the output memory/layout/dtype signature as well as shape when implementing that separation. This changes preparation lifetime only; it must not skip either runtime branch or any terminal shape. Runtime unaligned attention may still need its actual head-output list, but that helper list can die when concat returns, before the caller's terminal operation.

Releasing the pool would add1342177280 bytes to current free space, leaving737341952 bytes beyond the512-position head allocation **at the current snapshot only**. That arithmetic is not a success prediction: new cached programs, intermediate tensors, and fragmented placement can consume it.

## Accounting residual and native program storage

The measured loaded-model baseline is13857710080 bytes/device. Source accounting in `capacity_accounting.md` gives2362195968 for P8224 K/V plus deduplicated reset zero,2012768768 for listed B32 fixed categories, and87457792 for five wide common-sampler penalty arrays. Adding the definite concat pool and the illustrative CCL scale yields21843347968 bytes, approximately10805972992 below the observed allocation.

That residual is **unclassified**, not a measurement of a leak or program binaries. Small controls and live operation scratch are omitted, the CCL number is illustrative, and identities must be checked against the measured baseline before treating this as a complete additive inventory.

`tt_metal/impl/program/program.cpp:3223` provides a concrete other owner: each cached program/device allocates `program_transfer_info.binary_data.size()*sizeof(uint32_t)` in DRAM and retains it in `kernels_buffer_`. Binaries are allocated top-down specifically to reduce interference with bottom-up tensor allocations. Thousands of shape-specialized programs can therefore consume DRAM despite no Python tensor owner. Log-end JIT build-cache hit/build counters are not the live TTNN program count and must not be used as that inventory.

The fused prefill MMRS mesh resource cache is a different category: its source describes one per-mesh mailbox/semaphore resource set, not a distinct gigabyte DRAM allocation for every length. Do not infer per-length mailbox leakage from the number of MMRS compile lines. Likewise, this failure occurs before trace capture, so live trace-owned intermediates are not its explanation.

## Minimal discriminating instrumentation

Keep instrumentation in a diagnostic test wrapper, outside production model/generator behavior. No device profiling or tensor-to-host value transfer is needed for this census.

1. Record allocator allocated/free/largest-block counters and `mesh.num_program_cache_entries()` after model load, generator construction, completed B1 preparation, and uniform lengths1,2,32,128,160,161 and512. Wrap `model.terminal` so the length161 pre-call snapshot survives a failure; an after-exception snapshot alone loses the preparation frame's temporaries.
2. At those points record `_ccl_buffers` identities, padded/logical shapes, dtype/layout/memory config, and modeled bytes, grouped by layer and representative signature. A snapshot before/after `_prepare_layer_buffers` distinguishes replication from shape discovery. Existing buffer addresses identify simultaneous aliases; do not compare address equality across time as allocation identity because the allocator can reuse an address.
3. Use the bound native API `ttnn._ttnn.reports.get_buffers(mesh)`. `ttnn/core/reports.cpp:43` walks the allocator's live buffer set, including native-owned buffers. The Python binding exposes `device_id`, `address`, `max_size_per_bank`, `buffer_type`, and `buffer_layout`. Save DRAM records and reconcile addresses with known model/cache/CCL/helper/sampler tensors. The reported size is page-derived; allocator alignment can require rounding, so compare its sum against allocator totals rather than treating it as an infallible total. The API does not label a buffer's owner: unassigned records are candidates, not proof of program ownership.
4. Record the preparation frame's local tensor metadata at terminal entry: in particular all512 concat inputs, current result/value/head tensors, and normalized terminal input. This requires no logits readback. A diagnostic wrapper can inspect the caller's locals and avoid retaining the tensors after emitting metadata. Avoid a wrapper that itself keeps tensors alive in a history list.
5. First use real layers0/3 with B32, the same virtual/physical mapping, and exact preparation operations. A short `[1,2,160,161,512]` diagnostic distinguishes the immediate concat/head peak; it **cannot** reproduce cumulative program storage for all1..160 lengths. If that succeeds with ample memory while the full-prefix run grows, run the complete prefix to measure the program/signature slope. Reduced weights make this a hypothesis discriminator, never a capacity acceptance result.
6. Compare fresh-process phase-separated preparation against the original with the same complete signatures, counters, and tensor inventory. A lower peak with the same program-cache entries/signatures supports the lifetime repair. Continued growth in unmatched native buffers correlated with program count requires binary/allocation attribution before another policy-preserving repair.

Do not clear the program cache, disable persistent CCL, share layer-private scratch, omit shapes, lower the physical pool/context, or change dtype as an acceptance workaround. The final all64 B32 preparation, serving checks, allocation-tracked replay, and capacity evidence remain pending until the parent executes and verifies the repaired contract.

## Source-only checks performed

Read selected decoder defaults and `_collective`, bounded representative/copy/concat lifetimes, generator prepare/capture order, the failing raw log, capacity accounting, the native program binary allocator, and native buffer-report implementation/bindings. All byte arithmetic above was recomputed on the host. No hardware pass or root-cause confirmation is claimed.
