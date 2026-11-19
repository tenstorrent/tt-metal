# Explicit persistent broadcast sampler: rejected native geometry

The isolated persistent-output broadcast candidate is **rejected for correctness on this checkout**, not for being a small performance component. No production or native source was changed. The selected split sampler remains correct; decoder persistent workspace/CCL optimizations remain enabled. This finding does not establish that every possible persistent sampler implementation is unsupported.

## Reproduction and measured evidence

- [Watched v2 log](logs/persistent_sampler_l1_watched_v2.log): queued greedy candidate emits `127615` for every row, versus distinct correct baseline tokens. `127615 = 65535 + 62080`, matching the poisoned UINT16 index plus one vocabulary-shard offset.
- [Watched v3 log](logs/persistent_sampler_l1_watched_v3.log) and [raw-gather diagnostics](persistent_sampler_l1_watched_v3.diagnostics.json): fail before sampler execution, at the first BF16 gather. Every rank has its own 32 columns correct and the other 96 columns unchanged. Exactly **3,072 of 4,096 elements mismatch on every rank**. The nonfinite BF16 poison reads back as infinity, rather than NaN, on this path. Device closure was clean; this was a wrong-result failure, not a hang.
- Exact input: TP4 mesh `(1,4)`, local logical/physical TILE `[1,1,32,32]`, BF16, interleaved DRAM. Each local tile is **2,048 bytes**. Output is persistent interleaved DRAM `[1,1,32,128]`, **8,192 bytes per rank**, gather dimension 3, physical mesh axis 1, one link, `use_broadcast=True`.
- The independent raw input encodes rank, row and column, replacing any dependence on the sampler's top-k ordering or trace lifetimes. The failure occurs eagerly, before capture, and is therefore sufficient to reject this candidate without attributing it to trace allocation or ownership.

Reproduce from checkout root under the existing serialized device procedure:

```bash
TT_METAL_WATCHER=1 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 PYTHONPATH=. python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_stage7_sampler_persistent_probe --mode acceptance --num-links 1 --output models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_full_model/persistent_sampler_repro.json
```

The runner writes `persistent_sampler_repro.diagnostics.json` before asserting. The v3 diagnostics include SHA256 hashes for the Python runner/helper, shared sampler, native factory and kernels. Retain those existing hashes as the provenance of the observed failure.

## Exact root cause

Source locations below are relative to checkout root.

1. `tt_metal/fabric/erisc_datamover_builder.hpp:365,387` gives the default 1D fabric payload as **4,352 bytes**. `tt_metal/fabric/fabric_context.cpp:132–134` adds the packet header to form channel-buffer size. The experiment uses default `FABRIC_1D`, without a router override.
2. `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_via_broadcast_factory.cpp:91–99,141` rounds that channel-buffer size down to a power of two, giving an exact **4,096-byte transport packet**, and passes it as writer compile argument 3. Input/output pages remain 2,048 bytes; the CB page is 16,384 bytes.
3. `.../device/kernels/broadcast_rm_writer.cpp:41–44` therefore chooses the scatter writer with **two output pages per packet**.
4. `FabricScatterWriter::send`, lines 118–136, increments `scatter_header.chunk_count` and emits a fabric packet **only when the count equals two**. This rank owns only one page, so the only call leaves count one. There is **no partial-packet flush** before `cb0.pop_front(1)` at line 233 or before final ready signaling at line 236.
5. The local `noc_obj.async_write` at lines 223–229 executes for that same page regardless. Thus each rank writes its own tile locally, sends no remote payload, and still signals completion. That predicts the exact observed own-32-correct/remote-96-poison pattern.

The unsupported condition is a positive per-worker output-page count not divisible by `packet_size / output_page_size` on the scatter branch. This is not restricted to persistent output tensors. The existing compile-time assertion checks CB-page divisibility, which does **not** establish that the final worker segment fills a packet. The final semaphore protocol orders the operations that were issued; it cannot compensate for a payload that was never sent.

## Existing configuration and padding alternatives

There is **no per-operation packet-size argument** in either `all_gather_async` Python overload (`all_gather_async_nanobind.cpp:265–301`). `num_links`, workers and buffer counts do not alter this factory's packet size. Two links assign the one input page to one worker and zero pages to the other; the one-page tail remains.

There is a legal **global** configuration API:

```python
router = ttnn.FabricRouterConfig()
router.max_packet_payload_size_bytes = 2048
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, router_config=router)
```

This is exposed in `ttnn/cpp/ttnn-nanobind/fabric.cpp:122–142`; `FabricContext::validate_and_apply_packet_size` accepts a positive aligned 2,048-byte payload. It would select a one-page unicast packet for an isolated sampler process. It is **not a legal drop-in full-model optimization under the selected policy**: it changes fabric configuration for every collective. Selected linear-attention output partials remain FP32 (`tt/multichip_decoder.py:705–711`, reduced by `_reduce` at 1299). The all-reduce factory computes `num_pages_per_packet = channel_buffer_size / input_page_size` at `all_reduce_async_program_factory.cpp:270–272`. A 2,048-byte payload plus its small header cannot hold a 4,096-byte FP32 tile, yielding zero pages per packet; the writer loop consumes that value at `device/kernels/dataflow/worker_writer.cpp:120`. Do not test this global workaround on the full model or silently change FP32 reduction policy to accommodate it.

Padding each local top-k from width 32 to 64 at **one link** would supply the required two tiles. The output then becomes `[1,1,32,256]`, with an inserted 32-column pad after each rank's 32 valid columns. Preserving rank order and logical indices requires removing all four pad chunks before sampling. For both values and indices, a straightforward implementation adds two pads, eight slices and two concatenations, doubles gather payload/storage to 32,768 output bytes per rank, and still needs the ownership hook. A custom compaction could reduce launches but would be another kernel and validation surface. No speedup was measured; this is an untried workaround, not an accepted optimization. Width-64 padding with two links is insufficient because each worker receives only one tile; width 128 would be needed to give each two tiles. Padding the batch dimension instead violates the native contiguous-output-slice validation for gather dimension 3.

## Narrow native fix proposal — not applied

The smallest specialization that repairs exactly this sampler geometry is to select a one-tile packet in the broadcast factory when each rank contributes one 2,048-byte tile. It preserves the existing larger-packet behavior for every other geometry and leaves arithmetic, barriers, output ownership and layouts unchanged:

```diff
 // After computing num_output_pages, before writer_compile_args:
+const uint32_t writer_packet_size =
+    (num_output_pages == 1 && output_page_size == 2048 && MAX_PACKET_SIZE_BYTES > output_page_size)
+        ? output_page_size
+        : MAX_PACKET_SIZE_BYTES;
 ...
-    MAX_PACKET_SIZE_BYTES,  // packet_size,
+    writer_packet_size,    // one-tile ranks must not enter an unflushed scatter tail
```

This uses the already implemented unicast writer and requires a native rebuild. It is a **proposal**, not a tested fix, and does not repair all other possible partial tails. A general repair would flush residual pages before releasing the source CB, using unicast for a singleton (scatter requires at least two chunks). That broader change is unnecessary for documenting the present exact blocker and is deferred.

Before promotion of any future repair, require raw BF16 **and UINT16** gathered bytes on all ranks, input-address/cache-hit changes, poisoned eager sampler, single trace replay, all existing queued changing-input cases, Watcher, and clean timing versus the selected baseline. The current v3 failure stops at BF16; UINT16 is predicted to share the same page geometry but has not independently passed the raw gate.

The accepted full-model candidate's approximately 21.424 ms token-out time and approximately 0.488 ms selected sampler put the sampler near 2.3% of decode. Those measurements bound the practical upside; the **reason for rejecting this explicit persistent-broadcast candidate is the exact native data-loss defect above**, not that percentage. Full-model source remains frozen for the parent's final validation.
