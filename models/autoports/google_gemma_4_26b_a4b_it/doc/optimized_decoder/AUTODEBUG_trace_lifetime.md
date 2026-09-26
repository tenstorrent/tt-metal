# AutoDebug: request-reuse trace allocation lifetimes

Source-only investigation, 2026-09-26. No hardware calls, TTNN imports, or implementation edits were performed. Scope is the two trace-allocation failures; the separate tight-cache K-chunk issue is excluded.

## Verdict

1. **Verified unnecessary lifetime:** sliding prefill retains two final K/V tail clones after their last use. Clear the tail at the end of `OptimizedDecoder.prefill_forward`, or avoid creating it for the final chunk. Decode and prefix continuation use the supplied paged KV cache, so this tail is not required between public calls.
2. **Verified missing initialization coverage:** the request-reuse harness captures after length 31, then allows previously unseen prefill programs to be initialized while that trace remains active. Length 33 introduces 53 surviving `program_cache:` allocations on layer 5.
3. **Not established by the existing logs:** whether any of those 53 allocations actually overlaps an address written by the trace. The tracker checks allocation time and survival, not address intersections. Its error text is stronger than its test. Kernel-binary DRAM backing buffers are the source-supported explanation for these allocations, but the logs do not include allocation-site C++ stacks, addresses, or binary-buffer IDs to prove every individual buffer's identity.
4. **Top-down allocation is not a proof of safety.** Binaries and ordinary tensors share the DRAM free list. A binary buffer created after capture can occupy a freed trace temporary's address. Cached program hits do not restore its bytes. Do not blanket-disable program-cache tracking or mark those buffers corruptible on the strength of their context labels.

**Chosen resolution, pending tests:** the parent selected final-chunk tail-allocation avoidance plus exact-catalog initialization before capture, with program-cache misses forbidden while the trace lives. The proposed harness patch below implements that initialization contract. Tracking must remain fully enabled and report zero survivors for all nine requests on both layers. This eliminates the observed unsafe-lifetime condition structurally, without asserting that the old buffers actually overlapped. The decoder does not own trace creation: callers must initialize new prefill signatures before capture or release/rebuild their traces before admitting them. Logical context capacity is unchanged.

The optional range experiment below is needed only if an exact historical overlap/false-positive classification is desired, or if remaining survivors need attribution. It is not required to validate the chosen zero-survivor repair. Full prewarm proves the initialized-signature contract; it does not establish unrestricted cold-signature admission while a trace lives.

## Evidence examined

- `trace_alloc_v5_commands.json`: both commands use `TT_METAL_TRACE_ALLOC_TRACKING=1` and `TT_METAL_TRACE_ALLOC_TRACEBACKS=1`, call `tests.run_optimized_contract --contract request_reuse`, and return 1.
- `trace_alloc_v5_layer0.log`: length 31 passes; the next replay fails with IDs 1804 and 1809, both allocated by `ttnn.clone` at `optimized_decoder.py:1382`, referenced as `layer.layer.self_attn.tail[0/1]`, shape `[1, 8, 32, 256]`.
- `trace_alloc_v5_layer5.log`: lengths 31 and 32 pass; replay after length 33 fails with 53 `program_cache:` IDs, with zero Python tensor referrers found. A generated context/ID inventory is appended below.
- `validated_v5_request_reuse_layer0.json` and `validated_v5_request_reuse_layer5.json`: each records all 9 request rows passing, runtime SHA256 `169c0d97d7d0e9f35d97633f133305f1088b987ef625693d3faa100d25c3e67b`. These runs demonstrate numerical success for this allocation history; they do not prove allocation safety.
- `tests/request_reuse.py`, `tests/run_optimized_contract.py`, `tt/optimized_decoder.py`, its inherited attention implementations, TTNN launch/cache code, mesh workload binary initialization, allocator tracking and graph/report APIs.

Line references for optimized decoder findings refer to the v5 source represented by the logs. The parent is concurrently making independent changes to this untracked runtime file; use the named methods and code snippets if line numbers move. This investigation did not modify that file.

## A. Final sliding tail has no reader

The causality is direct:

1. `OptimizedDecoder.prefill_forward`, `optimized_decoder.py:681-682`, clears the old tail before a fresh request.
2. `OptimizedAttention.prefill`, lines 1358-1382, reads a previous chunk's tail, computes attention, then unconditionally clones the last `min(window, physical_chunk_length)` rows for the next chunk.
3. The public prefill loop finishes at lines 716-717 without clearing the last clone pair.
4. `FusedAttention.decode`, `fused_decoder.py:609-643`, updates and reads `kwargs["kv_cache"]`; it never reads `tail`.
5. Prefix continuation uses `decode_forward` per token (`optimized_decoder.py:658-677`). The next fresh prefill clears the tail before any read.
6. `_release_sliding_prefill_tail`, `decode_attention.py:37-38`, simply drops the tuple reference. No device contents are required to decide when to release it.

The clones are distinct buffers, so releasing them does not alias the output or paged cache. Calls remain ordered on the same command queue. Releasing only at the end of the outer loop preserves inter-chunk history.

### Minimal proposed patch, unapplied

```diff
             outputs.append(out[:, :, :valid, :])
+        attention._release_sliding_prefill_tail(clear_persistent=True)
         return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)
```

This is the smallest verified lifecycle fix. A slightly larger valid alternative avoids the final clone allocation:

```diff
                 chunk_page_table=chunk_table,
+                retain_prefill_tail=start + valid < length,
             )
```

and in the sliding branch of `OptimizedAttention.prefill`:

```python
if kwargs.get("retain_prefill_tail", True):
    tail_length = min(cfg.sliding_window, k.shape[-2])
    self.tail = tuple(ttnn.clone(t[:, :, -tail_length:, :]) for t in (k, v))
else:
    self.tail = None
```

Use an explicit call argument if choosing this alternative; an object-level mutable flag adds avoidable state and can persist across exceptions or direct helper use. The default `True` preserves direct calls that rely on retaining chunk history. The flag propagates through `FusedDecoder._forward`'s existing attention kwargs. The second option is a performance hypothesis only in the sense that it removes work; no measured latency benefit is claimed.

### Focused prediction

Rerunning the original layer-0 tracking command after only this fix should eliminate IDs attributed to `self.tail`. Later prompt sizes may then expose additional program-cache allocations; that does not refute the tail fix. Check both single-chunk and multi-chunk requests, including 1025, 2049, and the decreasing-length sequence, to preserve the history contract.

## B. Meaning and lifetime of `program_cache:` allocations

`ttnn/api/ttnn/device_operation.hpp:512` creates output tensors **before** entering the cache-miss allocation context. The cache-miss wrapper at lines 371-385 tags the entire factory/create/cache/enqueue path as `program_cache: <op> <attributes>`. Consequently that prefix is provenance, not a memory type and not a blanket promise that the allocation can be overwritten.

The shared binary path explains the surviving buffers:

- `create_and_cache_mesh_workload`, lines 330-355, stores the workload in the device program cache and enqueues it.
- `tt_metal/distributed/mesh_workload.cpp:131-150`: first enqueue sizes the combined kernel binary image across the workload.
- Lines 153-180: it allocates **one replicated DRAM mesh buffer**, `bottom_up=false`, page size `HostMemDeviceCommand::PROGRAM_PAGE_SIZE`, and retains it as `kernel_bin_buf_`. On this 1x1 mesh, that corresponds to one owned device allocation. `mesh_workload_impl.hpp:61-62` holds binary status and the shared mesh buffer.
- Lines 181-206: per-program views alias that backing allocation, and the program's binary data is written once.
- Lines 136-143: later enqueues only require `ProgramBinaryStatus::Committed`; they do not upload the binary again.
- `device_operation.hpp:271-291`: cache hits update runtime arguments or apply the descriptor, then enqueue the cached workload.
- `tt_metal/impl/program/dispatch.cpp:2547-2572`: binary dispatch commands use the cached kernel buffer, including its base address.

The 53 reported contexts span ordinary slicing, casting, normalization, projection, SDPA, routing, and movement operations. Source searches across their operation directories found output tensor allocation in `create_output_tensors`, not retained tensor allocations in the corresponding program factories. For representative potentially confusing cases:

- `reduction/topk/device/topk_device_operation.cpp:404-405` allocates the values and indices as returned tensors; `topk_device_operation.hpp:26-40` constructs program artifacts/descriptors. Its cache-tagged buffer is not evidence of retained routing indices.
- `transformer/sdpa/device/sdpa_device_operation.hpp:24-29` builds a program descriptor; output allocation is in `sdpa_device_operation.cpp:545`. Its CB scratch is a program resource, not a new Python tensor retained after the prefill call.
- `experimental/paged_cache/device/fill_cache/paged_fill_cache_program_factory.hpp:16-49` constructs descriptors and patches runtime arguments on cache hits; the paged KV cache itself was allocated before capture by the harness.

These are strong structural reasons to identify the reported persistent allocations as kernel binary backing buffers. Still, the prefix also covers the entire cache-miss factory, so **do not claim an exact ID-to-binary proof from the Python traceback alone**. A graph allocation event nested under each program's initialization, or C++ buffer inventory tied to `kernel_bin_buf_`, closes that final attribution gap.

### Why a corrupted binary matters after the current replay

The existing decode trace does not use new prefill programs introduced after its capture. It can therefore produce a correct decode while overwriting an unrelated new prefill binary. When that prefill signature is reused, the program cache can reuse the corrupted binary. The nine successful requests reduce the likelihood of an overlap for this run; they do not make such overlaps impossible or justify acknowledging the binary as intentionally corruptible.

## C. What the tracker actually proves

`tt_metal/impl/allocator/trace_allocation_tracker.cpp`:

- Lines 76-83 register a trace; `mesh_device.cpp:1421-1424` does this on exit from `end_mesh_trace`, not at the start of the first capture.
- Lines 117-135 record **every** subsequently allocated non-TRACE buffer for each active trace, except explicit suppression scopes or the optional program-cache skip. No address, size, storage bank, or trace write footprint is compared.
- Lines 138-174 remove IDs that are no longer allocated and report remaining IDs.

Thus the exact verified assertion is: “these allocations were created after this trace and are still alive.” `ttnn/ttnn/unsafe_allocation_tracker.py:89-92` says “These will be corrupted,” but no intersection check supports that certainty. Zero Python referrers is expected for a C++ cached workload and does not imply a leak or fabricated tracker ID.

The nontracking warning is already qualified: `allocator.cpp:126-130` says the buffers **may** be corrupted.

## D. Allocation direction does not partition DRAM

- Ordinary `Buffer` defaults to `bottom_up_(bottom_up.value_or(this->is_dram()))` (`tt_metal/impl/buffers/buffer.cpp:426`), so normal DRAM tensors grow from the bottom.
- Kernel binaries explicitly request top-down DRAM (`mesh_workload.cpp:163-166`).
- Both reach `dram_manager_->allocate_buffer(...)` in `allocator.cpp:176-179`.
- `bank_manager.cpp:433-438` sets `address_limit=0` for DRAM; there is no lower bound tied to an active trace's earlier addresses.
- `bank_manager.cpp:112-113` uses `FreeListOpt::SearchPolicy::FIRST`. `free_list_opt.cpp:99-150` searches free blocks and lines 161-175 place the allocation at the requested end of the selected block. Top-down controls placement; it does not create a separately reserved binary region.
- `allocator.cpp:223-249` returns freed capture temporaries to the normal free list. The old trace still contains their recorded addresses.

In this harness, `trace_region_size=0`. `mesh_trace.cpp:69-72` allocates the trace command buffer top-down in DRAM, and lines 104-150 validate that **that trace-storage buffer** lies above the capture allocation/deallocation high-water marks. This does not protect later kernel buffers: they can be allocated below the trace command buffer, and their ordinary DRAM allocation path does not repeat this check. A nonzero trace region likewise protects command storage, not every freed activation address.

Therefore a safe result needs actual range evidence or an initialization/lifetime contract. Source alone cannot classify this particular 53-buffer instance as overlapping or disjoint.

## E. Smallest discriminating address experiment

No C++ edit is needed to obtain conservative allocation intervals. Existing NORMAL graph capture records real allocator activity:

- `ttnn/core/graph/graph_processor.cpp:256-285`: `buffer_allocate` includes `address`, `max_size_per_bank` from `aligned_size_per_bank()`, `buffer_type`, layout, and device ID.
- Lines 308-334: `buffer_deallocate` records the corresponding range and buffer-node connection.
- Lines 724-747: each unique Buffer has a graph node; allocation/deallocation events connect to that node. The global allocator unique ID is used internally but is not exported in node params.
- `ttnn/core/graph/graph_nanobind.cpp:111-167` exposes `begin_graph_capture(RunMode.NORMAL)` and returns JSON from `end_graph_capture()`.

Suggested harness-only diagnostic:

```python
trace = ttnn.begin_trace_capture(mesh, cq_id=0)
ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
traced = decode()
capture_allocations = ttnn.graph.end_graph_capture()
ttnn.end_trace_capture(mesh, trace, cq_id=0)
```

For each later request, start a separate NORMAL graph immediately before the eager prefill input allocation. End it after output consumption, explicit `out`/`xt` deallocation, and garbage collection, but **before replay**. Also snapshot the tracker mapping with `ttnn._ttnn.operations.trace.get_unsafe_tracked_ids(mesh, trace)`. Keep tracking enabled; no replay bypass is needed to collect evidence before its deliberate failure.

Produce an artifact with one row per surviving post-capture allocation:

```text
request_length, enclosing_op, graph_buffer_node, device_id, buffer_type,
address, aligned_size_per_bank, end_address, overlap_capture_nodes
```

Track liveness through graph buffer-node connections, not addresses alone, because addresses can be reused. Compare half-open intervals `[address, address + max_size_per_bank)` on the same device and storage type. Use the union of allocations **and deallocations** during traced decode as a conservative footprint; include retained preexisting input/output ranges if inspecting unexpected aliases. For interleaved DRAM, the maximum aligned per-bank size provides a sufficient conservative disjointness proof. L1 sharded/per-core ranges need their actual core/bank geometry if any survive.

Interpretation:

- **Every surviving program buffer disjoint:** establishes that this trace cannot write those buffers via the recorded allocation footprint in this run. Save the actual ranges and classification with the result. Repeat as signatures are introduced; a first-three-request result does not cover all later sizes.
- **A range intersection:** establishes a potential alias; combine the graph's producing/writing op with binary-buffer ownership to identify whether replay actually writes that region. Avoid executing known-overlapping program binaries just to test whether they hang.
- **Unmatched survivors / different counts:** first resolve lifetime/ownership attribution. Program-cache prefix alone must not suppress them.

`ttnn._ttnn.reports.get_buffers(mesh)` can also snapshot live buffer addresses, types, layouts and `max_size_per_bank` (`reports.cpp:43-101`, nanobind `reports.cpp:29-36`). It does not export Buffer unique IDs; `get_buffer_pages` currently filters to L1 (`reports.cpp:110`), so it is not a DRAM page inventory API. The graph path uses aligned size and is preferred for trace intervals. An exact global-ID ownership map would require targeted C++ instrumentation or an inspector extension; it is unnecessary for a conservative all-live-buffer disjointness proof.

## F. Contained remedy if address evidence is unavailable or overlaps

For a fixed supported request catalog, preinitialize **all exact eager prefill signatures** before the first decode capture, then prefill the real first request again, warm the exact decode signature, and capture. Include original logical lengths and per-chunk offsets, not only rounded physical lengths: slicing, padding, concat, scalar attributes, and internal routing shapes can specialize programs. Warming only the largest length does not cover every smaller signature.

The current catalog is `[31, 32, 33, 1023, 1024, 1025, 2049, 33, 2047]`. Reuse the same layer/cache/table/RoPE setup and dtype/memory/config policies. Discard warm outputs, clear final sliding tails, and restore request data/page-table ownership before the measured/validated sequence. Then `mesh.set_program_cache_misses_allowed(False)` can make this initialization contract executable; it is exposed in `ttnn/core/distributed/distributed_nanobind.cpp:358-359`, and checked at `device_operation.hpp:419-421`. Record program-cache entry counts before/after the catalog and assert no growth during the traced request sequence.

This remedy needs no allocator exception. It is a claim about initialized signatures, **not** proof that arbitrary unseen prompt shapes are safe while a trace lives. For unrestricted serving, initialize declared buckets up front or explicitly release and recapture affected traces before admitting a new signature. Do not silently narrow the model contract based on a harness warmup pass.

## Required verification after integration

1. Tail-only patch: original layer-0 tracker command loses both `tail` reports; multi-chunk PCC and output shapes remain correct.
2. Chosen catalog initialization: all nine real requests pass with tracking enabled, cache misses forbidden, zero survivors, and stable program-cache count. Preserve input-fixture windows, random page ownership, and the original single-trace reuse.
3. If any survivors remain, use the address diagnostic to resolve their lifetime and provenance. The optional historical diagnostic must retain full tracking and save actual intervals before labeling old reports false positives.
4. Rerun ordinary request reuse for both layers after the final runtime SHA is frozen. Record any residual restriction on unseen signatures rather than declaring general trace safety from the finite catalog.

No hardware experiments were run by this investigation; all proposed hardware outcomes remain predictions. No broad tracker/runtime change is recommended on the existing evidence.

## Layer-5 reported buffer inventory

These are log IDs and provenance contexts, not measured address ranges. All 53 need the address experiment above for an instance-specific safety verdict.

| Context | Count | Buffer IDs |
| --- | ---: | --- |
| `BinaryNgDeviceOperation` | 1 | 2332 |
| `ConcatDeviceOperation` | 4 | 1951, 2011, 2047, 2284 |
| `FillPadDeviceOperation` | 1 | 1899 |
| `LayerNormDeviceOperation` | 5 | 1930, 1973, 1987, 2318, 2326 |
| `MatmulDeviceOperation` | 3 | 2131, 2290, 2312 |
| `MinimalMatmulDeviceOperation` | 2 | 1938, 2101 |
| `NLPConcatHeadsDeviceOperation` | 1 | 2095 |
| `NlpCreateHeadsDeviceOperation` | 1 | 1961 |
| `PagedFillCacheDeviceOperation` | 1 | 2081 |
| `SDPAOperation` | 1 | 2089 |
| `ScatterDeviceOperation` | 1 | 2198 |
| `SliceDeviceOperation` | 16 | 1907, 1917, 1944, 1995, 2004, 2031, 2040, 2145, 2152, 2212, 2219, 2248, 2255, 2296, 2303, 2338 |
| `SoftmaxDeviceOperation` | 1 | 2159 |
| `TilizeDeviceOperation` | 1 | 2204 |
| `TopKDeviceOperation` | 1 | 2139 |
| `TypecastDeviceOperation` | 10 | 1924, 1967, 1981, 2017, 2063, 2069, 2077, 2107, 2166, 2174 |
| `UntilizeCodegenDeviceOperation` | 1 | 2180 |
| `UntilizeWithUnpaddingDeviceOperation` | 2 | 2186, 2192 |

## Unapplied harness patch

`AUTODEBUG_request_reuse_prewarm.patch` adds the exact nine prefill calls before capture, records program-cache counts, and forbids cache misses for the entire live-trace portion of validation. It preserves real fixture windows and uses zeros only for signature warmup when no fixture was supplied, so warmup does not consume the random-number stream used by the original requests. The original measured request loop still refreshes random page ownership and overwrites valid KV positions. The final-tail runtime fix remains independently required.

Patch base SHA256 for `tests/request_reuse.py`: `3fd1c7f31b0acf8947c0644f7c1f76e0d36ef25edceb48b8735745058b2c2292`. Python syntax was checked with `ast.parse` on the proposed text without importing TTNN, and `git apply --check` passed. The patch was not applied and no device test was run. Parent reports that the final-tail keyword-argument alternative is now integrated independently; this report did not make that implementation edit. Final acceptance remains pending the parent's tracked hardware runs.
