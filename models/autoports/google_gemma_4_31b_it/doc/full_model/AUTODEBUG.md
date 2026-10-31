# Stage06 AutoDebug: prepare the whole generator before retaining decode traces

Inspection date: 2026-10-31. Checkout HEAD: `bdb236e49bdd887c9719e7a13b35af8bdc659964`; the inspected Stage06 generator, model, probes and full-model documentation are untracked additions. Findings refer to these working-tree files, not necessarily HEAD. Read `bringup/run_contract.json` and `bringup/agentic_evals/inputs.json`. No device execution, implementation edits, external publication, removed implementation inspection, or AutoDebug relaunch was performed. This report is the only authored artifact. Three isolated xhigh source audits assisted the investigation.

## Leading finding: decode-only preparation leaves persistent programs unprepared

**High confidence source diagnosis; observed failure is a pre-dispatch safety rejection, not observed device corruption.** `Gemma4Generator.prepare_decode()` warms decode and sampling only (`tt/generator.py:50-71`). `generate()` then runs reset and prefill while both traces are retained (`:150-153`). Those operations introduce persistent program-cache allocations. The first subsequent model replay correctly rejects them.

Direct observations in `doc/full_model/probe_trace3.log`:

- Lines 24-31 show layers 0 and 5, successful capture, and three replays producing token 27 with positions 33, 34, 35.
- Lines 44-59 identify `generate([2]*33,4)` and rejection in `verify_before_replay`, before executing the model trace.
- All 54 `Buffer ... [op: ...]` entries explicitly say `program_cache`. Buffer 4179 is reset's in-place cache MUL (`:60-73`). Buffers 4204 and 4223 are prefill chunk-page slices. Remaining entries include embedding, CCL, norm, projections, cache fill, SDPA, and terminal operations. A static count found 18 operation classes, one reset buffer and 53 prefill buffers; this is not one leaked Python tensor.
- Lines 1195-1200 report no matching Python references for these 54 buffers. Python GC is already attempted and the authoritative allocator state rechecked (`ttnn/ttnn/tools/trace_allocation_tracker.py:138-153`).

For this exact prompt, `start=0`, `valid=33`, `physical=64`, chunk table width=2 pages. Decode warmup did not exercise this prefill signature or reset. The log's source locations match that current call chain. This explains both the passing decode-only sequence and the failing transition through generation, without assuming a decoder numerical defect.

The allocator tracks allocations made while traces are active, including program-owned allocations (`tt_metal/impl/allocator/trace_allocation_tracker.cpp:183-200`). Tracking is conservative: the log does not establish an actual address collision or corruption. Disabling program-cache tracking, acknowledging all prefill allocations as corruptible, or relying on `del`/GC would suppress the evidence rather than establish safe persistent storage.

### Lifecycle and completion ledger

| Phase | State and consequence |
|---|---|
| Construction | Weights, KV, page tables, decode tokens/positions, sampler state are created before capture. |
| `prepare_decode` eager warmup | `_decode()` writes the actual KV cache, increments positions, and sampling overwrites tokens (`generator.py:44-57`; `model.py:148-157`; `multichip_decoder.py:455-463`). This is destructive setup. |
| Split capture | Model output is explicitly acknowledged because model replay rewrites it before sampling (`generator.py:58-70`). Capture recording must not be confused with ordinary execution; command-queue bypass recording appends trace nodes and returns without dispatch at `tt_metal/distributed/fd_mesh_command_queue.cpp:439-452`. |
| Decode-only replay | Model then sampler enqueue on CQ0; token readback is the completion boundary (`generator.py:126-133`). Position readbacks in the probe additionally observe completion. |
| `generate` after capture | Reset emits a new MUL program; eager prefill emits previously unseen programs. Host logits readback completes the prefill, but does not free persistent program-cache storage. |
| Next replay | Python verifies allocations before submitting replay (`ttnn/ttnn/__init__.py:141`). The exception is synchronous and precedes this replay, so teardown is not its cause. |

**Do not simply move real prompt prefill before `prepare_decode()`.** Its eager warmup mutates position-zero cache and feedback state. Prepare using disposable setup contents, then reset and perform the real prompt after all signatures are prepared and capture is complete. Releasing both traces and repreparing is a diagnostic fallback, not a bounded steady-state design; any such fallback must preserve or rebuild the real request state after destructive warmup.

## Separate capacity blocker: the Cartesian maximum does not fit the current allocator

Bounded preparation does not make B16/B32 simultaneous262144-token caches fit. This run's own `doc/optimized_multichip_decoder/memory_plan.json:2-24` gives 50 sliding layers,10 full layers, BFP8 KV payloads2176/1088 bytes per token per device, and a2048-token sliding pool. The current `model.py:121-134` allocates that full capacity for every batch slot and every layer. KV alone is:

`B * (50*2048*2176 + 10*262144*1088) = B * 3,074,949,120 bytes/device`.

That is49,199,185,920 bytes at B16 and98,398,371,840 at B32, already beyond the artifact's measured available34,178,731,008 bytes/device (`memory_plan.json:67-82`), without weights, programs, traces or activations. These are this run's analytical payloads and recorded allocator evidence, not an old Gemma implementation. Total estimates at B8/B16/B32 are33.04/57.64/106.84 decimal GB (`:41-64`); B8 is tight and requires fresh full-stack reconciliation.

This is a separate admission/allocation limitation, not the cause of the two-layer context256 trace failure. The existing check validates nominal batch/context ranges only. Preserve advertised per-request262144 and required batch signatures, but distinguish them from their simultaneous Cartesian product. Supporting B16/B32 at maximum length concurrently requires a separately validated memory policy; otherwise document evidence-backed concurrency limits. The report does not authorize a lower advertised context or claim all these allocations succeed.

## Why warming one prefill prompt is insufficient

Current specialization axes are explicit:

| Axis | Current source | Consequence |
|---|---|---|
| Physical tail length | `generator.py:91-101` | 32 lengths: 32,64,...,1024; logical lengths within each bucket must not leak into downstream op shapes. |
| Slot and chunk-page bounds | `generator.py:89,99` | Static slice coordinates depend on slot and absolute chunk start. |
| Full-attention offset | `optimized_decoder.py:587-600` | Scalar `chunk_start_idx` specializes SDPA by offset, independent of constant Q shape. |
| Sliding attention view | `optimized_decoder.py:607-621` | Rebasing bounds and table width change with start and tail. |
| Terminal mode | `model.py:136-146`; `generator.py:102-105` | Terminal head runs for every row of every chunk; host transfer selects the final valid row afterward. |
| Cache geometry and decode batch | `model.py:121-134` | Context, batch and layer type determine persistent cache shapes and reset/decode signatures. |

At maximum context, starts are `1024*j`, `j=0..255`. For each start all 32 tail buckets can occur across admitted requests. A scalar approach potentially requires 8192 start/shape combinations before accounting for static slices, layers and terminal variants. It is finite but a poor preparation contract. Warming just lengths 32,33,1024 at start zero does not cover it.

## Recommended bounded repair, retaining `MultichipDecoder`

Implement a generator-wide `prepare()` with an explicit signature plan and a READY state. Keep the selected TP4 projection, residual, precision and collective choices. The changes belong primarily at generator orchestration and the inherited prefill attention/terminal boundary, not in a replacement decoder.

### 1. Preserve 32 physical tail buckets and serialize prefill rows

Keep `S=32*ceil(min(1024,L-start)/32)` and the existing per-slot B=1 prefill loop. This accepts every length 1..262144, while decode supports configured B=1,8,16,32. It avoids expanding prefill work to B×1024 and changing the decoder's L1 strategy (`optimized_decoder.py:571-575`). B1 sequential prefill is a supported scheduling design, not a claim of simultaneous batched prefill throughput.

For every supported S, allocate persistent token `[1,S]` and absolute RoPE-position `[1,S]` buffers before capture. Refresh their contents with host-to-device copies, following `_copy` (`generator.py:35-38`). Feed physical S through the decoder stack; carry logical `valid` separately for output selection and accounting. The current generator deliberately passes `valid_tokens=physical`; do not change this to arbitrary `valid` at every layer and accidentally introduce logical-shape specialization.

Retain the configured context boundary for admission and the existing rounded physical cache capacity. A fixed1024 alternative reduces shape families further, but needs additional allocation/admission handling for short configured contexts such as the probe's 256, and performs unnecessary tail work. It is not necessary for this minimal repair.

### 2. Replace request-dependent page-table slices with prepared tables

Retain a host mirror of each cache's ownership map. Before capture allocate:

- Full-attention prefill row `[1,full_blocks]`, where `full_blocks=4*ceil(context/128)`; refresh its values for the selected slot.
- Chunk-fill table `[1,S/32]` for each S; copy the selected slot's absolute page IDs into it. `paged_fill_cache` writes using those physical page IDs, so the per-row operation still uses local batch index zero.
- Sliding-attention row `[1,64]`, with valid owned page IDs throughout. For the first chunk map the initial logical range and use attention offset 0. For later chunks map `[start-1024,start+1024)` through the slot's 64-page circular pool and use attention offset 1024. The chunk-fill table still uses absolute cache-page ownership; RoPE positions stay absolute.

This removes `generator.py:89,99` and `_prefill_attention_view`'s variable-coordinate slicing from the steady-state generator path. It does not change the public decoder's ordinary scalar API: add an optional prepared attention view/offset route, preserving existing callers.

Sliding capacity is exactly 2048 tokens per slot (`model.py:61,128-132`). At start2048, for example, attention table maps prior pages32..63 then current pages0..31; current writes cannot overwrite the preceding1024 tokens needed by the first query. At a short tail only S/32 current pages are written. Future entries must still map to owned valid pages and be excluded by causal masking. Preserve request-exclusive ownership.

The actual page size is **32**, not 128 (`optimized_decoder.py:212-218`). At262144, full table width is8192; a B1 prefill row is32KiB. The SDPA factory holds one such row in a per-reader-core L1 circular buffer (`sdpa_program_factory.cpp:163-165,1191-1192`). A full B32 decode table is1MiB per layer before mesh replication. Include this overhead in capacity testing; no full-model B32×262144 fit is established by this report.

### 3. Use existing generic tensor-offset SDPA

No new generic SDPA implementation is required just to make the offset dynamic. This checkout already exposes:

`ttnn.transformer.chunked_scaled_dot_product_attention(..., chunk_start_idx_tensor=start_tensor, ...)`

Use a persistent device INT32 tensor of shape `[1]`, omit `chunk_start_idx`, and refresh the contents before enqueueing attention. Evidence:

- Python dispatch: `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa_nanobind.cpp:265-310,516-571`.
- Validation and descriptor identity: `device/sdpa_device_operation.cpp:216-251`.
- Capacity-based factory geometry: `device/sdpa_program_factory.cpp:593-621`.
- Reader loads and broadcasts the offset through CBs: `device/kernels/dataflow/reader_interleaved.cpp:342-365`; runtime prefix bound: `:400-408`. Compute and writer consume the dynamic offset as well.
- Generic tests: `tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_chunked.py:242-312` checks dynamic offsets/trace; `:315-378` expects two scalar-offset cache entries; `:595-682` covers scalar/tensor sliding-window cases.

Keep host `chunk_start` for validation and planning. Do not pass a tensor blindly through existing Python `%` and comparison expressions (`optimized_decoder.py:567-570,598-599,610`). Add a separate runtime offset argument and prepared attention-table argument. Fixed starts0 or multiples1024 (and sliding offset1024) preserve the selected q/k chunk alignment. The tensor contains one offset shared by all rows, not a vector of per-user offsets; sequential B1 prefill avoids that mismatch.

The flexible factory uses `Sk = Sq + page_table_capacity`, an upper bound, while runtime bounds limit actual causal reads. Do not interpret the larger compile-time Sk as permission to access beyond the table: planner admission must ensure `attention_start+S` fits its supplied table. Full width8192, endpoint262144, S=1024 requires explicit boundary validation on hardware.

### 4. Split hidden-state prefill from terminal and select the last row on device

Separate `prefill_hidden_chunk` from terminal projection. Generation needs no terminal head on non-final chunks. On the final chunk, select row `valid-1` **before** norm/head/softcap, since those are rowwise (`model.py:109-119`). Keep a separate all-logits mode for validation/scoring and prewarm it if it can run while traces are retained.

A concrete generic last-row route exists without scalar row specialization:

1. Keep final hidden tensor `[1,1,S,H]` TILE, interleaved DRAM. Let `r=valid-1`, `a=32*floor(r/32)`.
2. Refresh prepared UINT32 RM start/end tensors `[0,0,a,0]`, `[1,1,a+32,H]`. Call tensor-argument `ttnn.slice(hidden, starts, ends, slice_dim=2, num_devices=S//32)` to select a fixed32-row tile.
3. Untilize that tile and reshape to `[32,H]`. Use a prepared UINT32 RM index `[[r%32]]` with `ttnn.embedding(..., layout=TILE)` to select one row. Reshape to `[1,1,1,H]`, then call terminal.

The slice path is device-only for TILE inputs (`data_movement/slice/slice.cpp:509-576`); output extent is input extent divided by `num_devices` (`device/slice_device_operation.cpp:249-276`). Its hash includes bounds-tensor metadata rather than values (`:395-450`). Do not use this dynamic-slice path on RM hidden: it falls back to host tensor conversion (`slice.cpp:577-579`). Here `num_devices` is the slice partition count, not TP mesh size.

Embedding requires BF16 interleaved RM weights (`embedding/device/embedding_device_operation.cpp:33-47`). The public wrapper accepts the one-element index: it chooses the non-fused RM primitive then converts the output to TILE (`embedding/embedding.cpp:45-75`), so the fused primitive's index-width multiple32 requirement does not invalidate this route. Prewarm that exact conversion chain.

For B>1 first-token device sampling, assemble selected rows or per-row vocabulary shards into a prepared B-row buffer before sampling. Slot insertion must use a bounded prepared operation family (at most the configured B slots), and the sampling input shape must match the warmed B-specific sampler. Do not assume independently allocated per-row logits alias the captured model-output address. Either prepare a separate first-token sampling route, or explicitly copy into the exact persistent sampling input using a prewarmed copy route and verify lifetimes. The existing host first-token argmax can remain an explicitly identified interim behavior, but is not device-sampling completion.

### 5. Warm reset and all reachable branches before either capture

Preparation order:

1. Construct all persistent inputs, page views, selection metadata, samplers and workspaces.
2. Execute reset for every actual cache shape/dtype. `output_tensor=tensor` avoids a new cache tensor but does not avoid compiling/allocating the reset program.
3. Warm all reachable physical S signatures with prepared full/sliding views and tensor offsets, plus last-row selection/terminal, optional all-logits terminal, slot assembly, first-token sampling and copy paths.
4. Warm configured-batch decode and sampling. Include any CCL semaphore cycling variants actually selected by the runtime. Repeat the bounded plan until program-cache count and persistent allocation inventory stabilize; equal counts alone are not a correctness proof.
5. Free temporary outputs, synchronize, capture split model/sampling traces, then perform the already-warmed reset and real prefill. Keep live prefill temporaries out of replay; do not retain unused last-chunk device logits in an outer frame.
6. For steady state, require zero new persistent program allocations after preparation. Handle external cache rebinding or new configuration as an explicit trace-invalidating lifecycle, never silently after capture.

A smaller lifecycle-only alternative is available if retaining the same trace across every request is not required: split one-time destructive eager warmup from capture-only construction; synchronize and release both traces before each unseen eager prefill; run the actual prefill while no trace is active; then capture only the already-warmed decode/sampling operations without eager decode or position-zero copies. Capture records rather than dispatches device work, so this can preserve the newly built prompt cache. It incurs recapture and retains growing scalar-specialized programs unless the bounded inputs above are also adopted. Treat it as an explicit fallback/control, not equivalent performance evidence for permanent traces.

Publish both trace IDs atomically after successful capture, and clean partial captures on exceptions. Currently `prepare_decode` tests only `model_trace` at line51 but assigns `sample_trace` at line70, so failure in sampling capture can leave a falsely prepared object.

This is at most32 prefill shape families per fixed configuration, plus finite layer/configuration/terminal/decode families—not 32 total programs. No dimension scales with the number of distinct prompt lengths, absolute chunk offsets or number of completed requests. If one instance must serve all four batches, construct/warm all four batch-specific cache/trace contexts before retaining traces, or explicitly release/reprepare between configurations; allocating a new generator on the same device with old traces active is not covered by warming only one instance.

## Required runtime experiments (not performed)

1. **Reproduce and partition the exact transition.** Retain layers(0,5), context256, B1 and allocation tracking with program-cache inclusion. Record allocation IDs/cache counts after prepare, reset, prefill33, terminal and replay. Baseline should match the observed sequence, not necessarily identical buffer IDs. Synchronize at each diagnostic boundary.
2. **Controlled warmup contrast.** Warm exact reset+prefill64 before capture, then rerun `[2]*33,4`. This isolates lifecycle from decoder math. Next request a previously unwarmed tail/start to show why one-probe warmup is insufficient. Do not count this narrow workaround as final acceptance.
3. **Dynamic-offset equivalence.** Compare scalar and tensor SDPA with identical cache/table/Q data at starts0,1024,2048,261120. Check program-count invariance, output agreement and trace replay after offset refresh. Use page32, actual local head dimensions, selected q/k geometry and full table8192; also test persistent sliding64 with offset0/1024 and circular page permutations. Generic tests' B1/16K coverage is not the full target matrix.
4. **Length/slot matrix.** For B1,8,16,32, exercise lengths1,31,32,33,63,64,65,127,128,129,1023,1024,1025,2047,2048,2049,4095,4096,4097,262111,262112,262113,262143,262144, and mixed lengths/slots. Test all32 tail buckets and row-index residues cheaply with synthetic hidden data. At the maximum prompt length test prefill only; generation output must fit remaining context.
5. **Terminal selection oracle.** Compare device last-row logits/tokens against the same eager all-logits implementation and then pinned HF. Check rows0,31,32,63 and S-1, tensor index refresh without new programs, B-specific sampling assembly, softcap and ties. Full-stack accuracy gates remain separate from two-layer smoke tests.
6. **Repeated request/reset correctness.** Alternate short/long prompts and slot permutations on one prepared generator; compare against freshly reset eager results. Include sliding wrap at2048 and subsequent wraps, padded tail followed by decode at logical L, and all-logits calls between generations. Keep allocation tracking enabled; assert no post-prepare persistent growth and synchronized completion, not merely successful enqueue.
7. **Capacity/performance evidence.** Measure full-stack peak DRAM/L1, prepared program storage, table CBs, sampler workspaces and trace region at all required batches. Establish the advertised context independently from concurrent working-set targets; do not lower262144 or reuse old implementation settings as evidence. Benchmark required workload only after correctness, safe repeated generation and capacity pass.

## Other discrepancies and unproven alternatives

- `generate()` explicitly rejects B>1 (`generator.py:144-145`); low-level prefill loops rows and decode has B-shaped state, but this is not a completed batched generation API. Add a separate batch API or contract-compatible orchestrator and validate prompt row counts, slot capacity and token lengths. `prefill_forward` currently checks only length-count agreement and per-length context (`:83-88`), not all those conditions.
- First token is always host argmax and prefill transfers all physical rows' logits before slicing (`:105,154`). This is a concrete device-sampling/performance gap, not the explanation for the54 buffers by itself. Host all-logits mode can remain intentional for accuracy evaluation.
- Arithmetic `cache*0` is not a general NaN sanitizer. There is no observed nonfinite-cache evidence here; do not headline this as the failure. Removing cache clear entirely requires separate proof that all readable entries are initialized/masked, including warmup, rounding and reset across requests.
- Padded dummy rows are not automatically a correctness defect. Real rows precede them under causal masking; final selection uses `valid-1`, and decode resumes at logical L and overwrites its own position. Sliding-ring ownership and rounded reads still need the boundary tests above.
- No evidence here establishes a need to replace the optimized decoder, change precision, disable allocation tracking, reduce context, or rewrite generic SDPA. Reduced traces do not prove full-stack correctness, memory fit, or throughput.

## Claim review

Rechecked each headline against the reached current code and log: decode-only preparation; reset and prefill after capture;54 program-owned buffers; GC already attempted; rejection before replay; scalar offsets and static slice coordinates; existing dynamic SDPA and dynamic TILE slice support. Challenged the tempting reorder fix and rejected it because eager warmup writes real cache/feedback state. Challenged one-element embedding validation against the wrapper and confirmed its RM-then-tilize fallback. Padding, NaN reset, batch completeness and full-capacity concerns remain separate from the observed causal claim. The bounded plan is a concrete repair design, not an implementation or a hardware-validated result.
