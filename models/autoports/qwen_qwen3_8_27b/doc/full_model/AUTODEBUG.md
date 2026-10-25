# AutoDebug: Qwen3.8-27B Stage06 preparation and partial-prefill state

Inspection date: 2026-10-24. This investigation read source, existing evidence, and lower TTNN implementations, and ran CPU-only static/arithmetic checks. It did not launch AutoDebug, execute device code, reset hardware, or edit implementation files. Three independent xhigh investigators reviewed state, trace/cache lifetime, and wrapper contracts. This report is the only file written by this investigation.

## Findings and priority

1. **P0 — The generator still lacks preparation over its advertised request domain.** The current `prepare_decode(prefill_lengths=())` warms an explicit debug list and then captures decode/sampling. The empty default still warms no prefill. Exact-length uploads, decoder slices, convolution history extraction, and page-table suffixes make an unlisted request a new program signature. This explains the existing trace-allocation rejection after preparation. Warming a few representative prompt lengths is insufficient.
2. **P0 — Simply padding the request to a finite bucket would change linear-attention state.** The current decoder correctly distinguishes logical tokens from tile padding inside `_delta`. Padding before that boundary loses the distinction: invalid convolution rows contain history, decay is nonzero even for zero projections, and convolution history would be taken from the physical tail. Add explicit validity and history indices; do not merely pass the bucket length as the logical length.
3. **P0 — Bucket preparation must include logical-row selection and continuation table handling.** Moving the last-token slice before the head reduces memory, but `hidden[:, -1:, :]` still specializes by input extent. `page_table[:, chunk_start // 32:]` specializes by position. Use runtime row indices and a separately prepared fill-table window. The existing tensor-valued SDPA start is already the correct mechanism and should remain.
4. **P1 — The public factory returns a generator that standard readiness generation cannot use.** `build_generator()` does not prepare it, while `generate()` requires existing traces. The installed readiness runners construct and call `generate()` without a model-specific preparation call.
5. **P1 for B>1 — High-level singleton APIs do not match the configured batch.** Both high-level prefill and generation construct one prompt row while state/traces are allocated for `self.batch_size`; generation later reshapes a single vocabulary row into B rows. Either reject these calls for B>1 explicitly or map/pad the singleton into the configured slots.

The smallest conservative source change uses **128 tile-rounded physical sizes, 32 through 4096**, plus the existing one-token specialization if its recurrence path must be preserved. A smaller **16-size outer bucket set, 256 through 4096**, is also feasible, but use a separately tile-rounded GDN extent to avoid extra all-padding state updates. Both support arbitrary logical lengths through 262144 without changing the accepted TP4 precision, residual, or CCL policies. Details follow.

## Evidence boundary and changing source

Git HEAD is `cdcf1091d55e793f68fd46acdbe80372337e1c34`. The Stage06 `model.py` and `generator.py` are untracked working files, not part of that commit. They changed externally during inspection; no investigator edited them. Line references below describe this reviewed snapshot:

| File under `models/autoports/qwen_qwen3_8_27b/tt/` | SHA256 |
|---|---|
| `generator.py` | `af071c780411aa77850b9f8bf9f58bcaa1beb589f6a5ba7ce5495f7255e53fe3` |
| `model.py` | `5a39d32177529ce64686f1efdbd95e9ff68165b874c99ae3e152bc3a97b68e31` |
| `multichip_decoder.py` | `d999166db24800aec9c16001201f26baad683c76150f5401db4834e4c937e82e` |
| `optimized_decoder.py` | `d355691a69232c06f4c072bdd417352b081f5bea3f64d603646c74071fb9f2ac` |

Direct observations from existing artifacts, not reproductions by this investigation:

- `doc/full_model/probe_prefill_01.log:50` reports rejection of the original row-major in-place multiply reset. `probe_prefill_02.log:34` subsequently records finite `[1,33,248320]` logits.
- `probe_trace_01.log:63` reports the exact sampling `k.shape == [num_users]` assertion.
- `probe_trace_02.log:41–55` reaches `PREPARED`, then reports **71 device buffers still alive before replay**. Listed allocations include prefill convolution at sequence 64, a history slice `[30:33]`, and GDN with two 32-token chunks (`:69,73,84–85`). This is a concrete match to the previously unwarmed 33-token prefill path.
- `probe_trace_03.log:43–45` reaches `PREPARED` and prints tokens/completion. This is evidence for the bounded probe after intervening fixes, not for every logical length or the complete 64-layer model.

These logs do not contain a complete source-hash mapping for every run. Do not combine all their failures into one current execution. The live fixes listed below explain why some earlier failures no longer apply. The headline preparation gap remains visible in current source independently of those logs.

## 1. Causal chain: preparation ends before the signature domain is closed

[Generator preparation](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/generator.py:69) now performs decode/sampling warmup, optional explicit prefill lengths in both logits modes, a final decode/sampling warmup, reset, and capture. This is a useful ordering for those signatures. Its comment correctly calls it debug preparation. It does not define or enforce a finite supported set of physical request signatures.

The request path introduces these specializations:

| Source | Runtime-dependent value | Consequence |
|---|---|---|
| `generator.py:123–129` | `count=min(4096, length-start)` controls token and position upload shapes | Up to 4096 logical tail shapes, including distinct one-token behavior |
| `optimized_decoder.py:549–563` | `n` controls hidden/RoPE slices, padding, and cropping | Normalization, projection, movement, and residual operations still see logical-length variants |
| `multichip_decoder.py:575–584` | convolution crop and history `[length-3:length]` | Even equal tile capacity does not imply equal slice programs |
| `model.py:91–95` | no-head, all-head, and last-hidden selection | Last-row mode has a length-dependent selector; all-row mode has a different terminal shape |
| `multichip_decoder.py:724` | full page-table suffix beginning at `chunk_start/32` | Different slice coordinates and output widths at later 4096-token chunks |

This is not merely Python object churn. [Slice program hashing](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_device_operation.cpp:395) includes start/end/step and input/output logical and physical specifications. Negative indices are normalized before lowering, so fixed `logits_to_keep=1` still means start `length-1` ([slice wrapper](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/data_movement/slice/slice.cpp:218)).

On a cache miss, [device-operation dispatch](/home/mvasiljev/tt-metal/ttnn/api/ttnn/device_operation.hpp:349) constructs/caches the workload under a program-cache allocation context. [Program compilation](/home/mvasiljev/tt-metal/tt_metal/impl/program/program.cpp:3087) allocates DRAM kernel binaries retained in `kernels_buffer_`. Freeing eager activation tensors does not free these cached program buffers.

With allocation tracking enabled and program-cache allocations included, [the allocator](/home/mvasiljev/tt-metal/tt_metal/impl/allocator/trace_allocation_tracker.cpp:183) records nonexempt allocations after capture against live traces. [Replay validation](/home/mvasiljev/tt-metal/ttnn/ttnn/tools/trace_allocation_tracker.py:130) rejects surviving buffers. The tracker is conservative: it does not establish an actual address intersection. The observed rejection is supported; actual memory corruption without the guard would require an overlap, which this inspection did not measure.

The inherited [mixed-prefill/decode trace test](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tests/probe_mixed_prefill_decode_traces.py:87) warms every tested context before any capture, then replays those contexts. It supports the shared-workspace lifecycle. It does not cover novel eager prefill signatures after capture or the generator embedding/head/sampler boundary.

## 2. Exact partial-chunk state

### What is correct today

The active TP4 `_delta` is defined in [multichip_decoder.py](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/multichip_decoder.py:558), with analogous single-device code at [optimized_decoder.py:377](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/optimized_decoder.py:377). It currently receives logical `length`, pads only the convolution working input, crops convolution outputs back to that logical extent, selects the last three actual projected rows, and pads **q, k, v, log-decay, and beta after their production** to a multiple of 32. That order is essential and is not a current partial-state bug.

If the wrapper instead passes a padded physical P as `length`:

- Convolution outputs immediately after the real tail still contain preceding real projected rows. Zero input padding does not make these outputs zero.
- Even in the favorable case of exactly zero projections, `g = -exp(A_log)*softplus(dt_bias)` is negative, and `beta=sigmoid(0)=0.5` (`multichip_decoder.py:591–593`). A dummy row can decay or update the real recurrent state.
- History selected at P contains padded rows, rather than the actual last three projected inputs. Even if logits for real rows remain correct by causality, the next decode starts from the wrong state.

The actual sequential recurrence is

```text
Sd = exp(g) * S
u  = beta * (v - k @ Sd)
S' = Sd + outer(k, u)
```

See [the FP32 kernel](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/kernels/fp32_recurrence.cpp:199). Therefore invalid rows need **g=0 (log-domain), beta=0, q=k=v=0**, applied after convolution and gate nonlinearities. Setting a projection to zero, or multiplying the final output by a validity mask, is insufficient. For example, with `A_log=dt_bias=0`, zero k/v, and nonzero S, each unmasked dummy row halves S.

### Minimal history selector with existing TTNN operations

For TP4 each rank's history width is 2560. Preserve the incoming history until both convolution and history extraction have consumed it:

```text
H = concat(old_conv[B,3,2560], projected_qkv_RM[B,P,2560], dim=1)
H = reshape(H, [B*(P+3), 2560])       # row-major, BF16, interleaved
idx[b,j] = b*(P+3) + n[b] + j        # j=0,1,2; UINT32 device data
next_conv = embedding(idx[B,3], H, layout=ROW_MAJOR)
copy(next_conv, state.conv)
```

For n=1 this selects old history rows 1/2 and new row 0; for n=2, old row 2 and new rows 0/1; for n≥3, new rows n−3 through n−1. It also expresses n=0 as unchanged history if inactive-slot support is added later. Update the index **values**, not Python slice bounds. Concat, conversion, lookup, and copy shapes depend only on prepared B/P.

This is compatible with the existing [embedding validator](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/embedding/device/embedding_device_operation.cpp:33): UINT32 indices and interleaved row-major BF16 table; [the wrapper](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/embedding/embedding.cpp:35) returns `[B,3,2560]`. Explicit row-major output avoids the fused tiled-index multiple-of-32 constraint. Mesh replication of indices with rank-local table contents preserves TP ownership. No new convolution exchange/semaphore protocol is needed.

Use the same mechanism to select final hidden rows, with a `[B*P,Hlocal]` row-major table and indices `b*P+n[b]-1`, before final norm/head. Keep all-row outputs physical through device execution; crop logical rows only after host readback when `return_all_logits=True`.

**Do not replace these selectors with arbitrary tensor-start slices without inspecting their contract.** The current tensor-argument slice uses a tile-oriented device path; row-major falls back to `to_vector()` host reads ([slice.cpp:512–579](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/data_movement/slice/slice.cpp:512)). It is not a general device-only nonaligned row selector.

### Physical buckets and the meaning of exact state

The minimum conservative Python-level plan is `P=32*ceil(n/32)`, n≤4096: **128 finite physical prefill sizes**. Mask invalid rows after convolution/nonlinear gates and select logical history as above. This keeps GDN's time extent identical to the current logical-n path's internal rounding; it does not add wholly padded 32-token chunks. The flat GDN path explicitly requires a multiple of its chunk size ([validation](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/chunk_gated_delta_rule.cpp:302)).

For lower setup cost in the expensive projection/attention body, use **16 outer P sizes**, `P=256*ceil(n/256)`, but execute GDN at `R=32*ceil(n/32)`. Those R values are another finite set of 128 signatures. Tile-aligned slices to R depend on R, never the exact logical n; mask values handle the partial last tile. Pad GDN's head-major output back to P before the existing gated norm, or run that norm at R and pad its output. The [gated-norm validator](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/kda/sigmoid_gated_rms_norm/device/sigmoid_gated_rms_norm_device_operation.cpp:67) requires its projected gate and recurrence sequence extents to agree. Do not feed R output alongside P projected gates directly.

Why not mask and run GDN through all of coarse P? That is an identity in real arithmetic, but every wholly padded chunk still executes `(I) @ S` in [GDN state update](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/compute/chunk_gdn_math.hpp:1143). Matrix-engine FP32 inputs are not a guarantee of full FP32 arithmetic; [local architecture documentation](/home/mvasiljev/tt-metal/docs/source/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.rst:168) describes TF32 accuracy. Extra identity matmuls can lose state bits. Do not claim bitwise FP32 carry from algebra alone. Avoiding extra chunks is smaller than adding valid-length handling across the GDN prep/scan readers, compute, and writers. Such a kernel change would have to drain all produced work correctly, not just shorten one loop.

The existing n=1 path uses sequential FP32 recurrence, while n>1 uses chunk GDN (`multichip_decoder.py:594,617–632`). Retain a prepared one-token specialization if preserving that algorithmic choice is required. Running n=1 through a masked P≥32 GDN is a numerical-path change and needs explicit evidence. Also, optimized and multichip modules define their own `fp32_recurrence` wrappers; changing only `tt/fp32_recurrence.py` would miss these paths.

Here, **exact state means state after exactly the valid logical tokens, with no padded-token transition**. Bitwise identity of complete model outputs across different physical matmul geometries is a separate, unproven numerical claim. The conservative extent plan removes the concrete extra-transition problem; hardware correctness comparisons remain necessary.

## 3. Continuation, page capacity, and finite signatures

Keep the generator's 4096-token outer stride. Every continuation start is then divisible by the inherited 128-token boundary and by the SDPA q/k chunks. Do not call prefill at an arbitrary last-token remainder: [the inherited validator](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/optimized_decoder.py:543) requires a 128-aligned start, and `chunk_positions[start//128]` otherwise cannot express the requested position.

For 16-size outer buckets:

```text
start = 4096 * floor((logical_length-1)/4096)
n     = logical_length - start
P     = 256 * ceil(n/256)
mapped_capacity = 256 * ceil(configured_context/256)
start + P <= mapped_capacity
```

Examples:

| Logical length | Final start | Logical tail n | P at 256 rounding | GDN R at 32 rounding |
|---:|---:|---:|---:|---:|
| 1 | 0 | 1 | 256 | 32, or preserved n=1 path |
| 33 | 0 | 33 | 256 | 64 |
| 257 | 0 | 257 | 512 | 288 |
| 4095 | 0 | 4095 | 4096 | 4096 |
| 4097 | 4096 | 1 | 256 | 32, or preserved n=1 path |
| 262143 | 258048 | 4095 | 4096 | 4096 |
| 262144 | 258048 | 4096 | 4096 | 4096 |

This agrees with [current state allocation](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/multichip_decoder.py:481): 32-token pages, eight-page row alignment. Naive power-of-two tails do not always fit: configured context 4640 has mapped capacity 4864; its 544-token final tail rounded to 1024 ends at 5120, outside that pool. Rounding to 768 ends at 4864. With 32-rounded body buckets, round the fill-table window width up to eight page IDs while keeping actual data extent P.

Replace the suffix at [multichip attention:724](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/multichip_decoder.py:724) with a prepared fill table:

- For 256-rounded P: fixed `[B,P/32]` INT32 window populated from actual page IDs for `[start,start+P)`.
- For 32-rounded P: fixed `[B,8*ceil(P/256)]` window, with every extra ID backed by the caller's mapped pool.
- Preserve a separate full bound page table for SDPA/decode. Never fabricate ownership of padding pages or read alignment bytes as IDs.
- With an authoritative host page map, refresh the prepared window from host. If the caller supplies only a device map, use a separately prepared device selection path; do not silently assume identity mappings or read it back during replay.

[Paged-fill validation](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/paged_cache/device/fill_cache/paged_fill_cache_device_operation.cpp:123) needs sufficient mapped capacity for the physical input. Physical tail K/V writes are acceptable only within mapped capacity: causal attention for real rows cannot see later padded keys, and decode must resume at logical `start+n`, overwriting the next padded position before consuming it. Neither RoPE positions nor decode positions may advance by P.

The existing [flexible SDPA call](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/multichip_decoder.py:741) supplies `chunk_start_idx_tensor` and keeps scalar `chunk_start_idx=None`. The factory [binds that tensor and the page table](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp:709); tensor contents/addresses do not become a per-position program hash. Keep it. There are at most 64 outer origins through 262144, but they need not imply 64 SDPA programs per bucket. For P multiples of 256, current q/k chunks settle at 128/256. For P multiples of 32, the finite q variants are 32/64/128 and are part of preparation.

## 4. Smallest concrete intervention by file

| File / owner | Required change |
|---|---|
| `generator.py` | Introduce one whole-generator preparation API with a frozen configured B/context/bucket set. Allocate masks, row indices, fill-table buffers, persistent inputs, and explicit output targets before capture; retain outputs produced by their own capture. Route requests to physical buckets; refresh values in those buffers. Integrate preparation into the public construction/generation lifecycle. |
| `model.py` | Carry physical length separately from logical validity. Retain physical hidden shapes across every layer. Replace length-indexed final selection with the prepared row selector; retain the existing no-head/all-head modes. |
| `optimized_decoder.py` | Add an optional prepared physical-chunk path/metadata to `prefill_forward`, preserving the existing logical API. Avoid logical-n slices/pads/crops between layers. Thread validity and fill-table metadata into the mixer calls. Keep the start-boundary check. |
| `multichip_decoder.py` | Extend active `_delta` to consume validity/history metadata, and `_attention` to consume the prepared fill table. Preserve n=1 specialization if required. Keep all selected TP4 policy and residual dispatch. Apply analogous optional arguments to the single-device overrides only where the shared base API requires it. |
| `decode_workspace.py`, `fused_prefill.py` | No fundamental redesign is required. Retain shared owner lifetime and complete out/down order. The existing native prefill projection is guarded for B=1/P=4096; smaller buckets and other batches retain existing fallback paths. |

Warmup must cover a **signature closure**, not merely a list of prompt lengths:

1. **Allocate/bind first:** model weights, all layer state, shared CCL workspaces, sampler buffers, token/position/table targets, metadata and explicit output targets for supported shapes. An output can instead be created during its own capture and retained, as current `self.logits` is; this is distinct from an unrelated persistent allocation after capture. Outputs of a later trace also need lifetime review relative to earlier live traces. Cache-binding changes after capture require an explicit setup lifecycle outside requests.
2. **Phase A — warm all execution variants before any capture:** every supported physical prefill bucket and required R/n=1 branch, no/last/all-terminal paths, runtime selectors/masks/conversions, table-window preparation, reset, prefill-to-sampler copy, full decode, and sampling. Include every configured batch/layout rather than assuming B=1 covers B≥8. Release transient outputs; preserve intentional persistent ownership. Synchronize and repeat the relevant signatures until cache entries stop growing; use cache-miss rejection if available to verify closure.
3. **Phase B — capture only after closure:** reset to deterministic empty state, stabilize output ownership, capture model then sampler against stable addresses, synchronize before publishing readiness, and reset again without allocating new persistent variants. The existing final decode warmup is useful but not a substitute for Phase A.
4. **Requests:** select a prepared bucket, refresh data, execute prefill, copy final logits into the persistent sampler input, release eager temporaries, and replay the existing sampler/decode traces. For this recommended eager-capture lifecycle, keep capture out of requests as well. The essential requirements are no request-time trace retirement, lazy persistent workspaces, or new program-cache entries. A deferred first capture is a different possible design only if the signature was fully prepared and its ownership is safe relative to every live trace. Reject unsupported batch/cache/layout configurations before mutating request state.
5. **Teardown only:** synchronize outstanding device work, then release traces and their retained state/workspace owners. Host-side `finish()` merely validates balanced enqueue order; it is not a device completion fence.

The numerical/communication contract to retain is explicit in [TP4 policy setup](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/multichip_decoder.py:24): inherited BFP4 projection weights and LoFi matrix policy; BF8 selected prefill activation/CCL payloads; BF8 KV cache; FP32 recurrent state and HiFi4 state math; sharded prefill residual; replicated small-batch decode and native sharded decode from B=8; shared alternating out/down scratch. Do not force all paths to BF16, replicated residuals, or a different collective merely to reduce warmup variants.

One subtle distinction: tuple keys in `_gather` intentionally preserve BF16 for model embedding/terminal boundaries ([line 274](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/multichip_decoder.py:274)). Those wrapper calls are valid and are not evidence of a precision mismatch.

## 5. Immediate wrapper bugs and resolved findings

**Still open: readiness lifecycle.** [The factory](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/generator.py:201) returns without preparation, and [generate](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/generator.py:163) raises. The actual [teacher-forcing runner](/home/mvasiljev/tt-metal/.codex/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.21/runtime/readiness_check/run_teacher_forcing.py:224) and [autoregressive runner](/home/mvasiljev/tt-metal/.codex/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.21/runtime/readiness_check/run_autoregressive.py:156) construct then generate. Have the factory prepare the complete configured domain, or perform the same one-time preparation at the entry to generation **before** request prefill. The latter must not become per-request expansion/recapture.

**Still open: configured batch versus singleton input.** [High-level methods](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/generator.py:136) always supply `torch.tensor([prompt_token_ids])` and one prompt length. For B>1, state copies may fail before the impossible logits reshape at line 172. An explicit B=1 high-level restriction is a small truthful interim fix; complete B>1 support needs fixed-slot input/mask mapping and correct output selection. Mixed-length prefill is explicitly rejected and is not solved merely by having buckets.

**Resolved during inspection; do not headline as current failures:**

- Sampling k/p/temp changed from `[32]` to `[batch_size]` (`generator.py:34–36`). [The sampler validator](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/reduction/sampling/device/sampling_device_operation.cpp:155) requires the exact user count; the original B=1 failure was real.
- Reset changed from `multiply(value,0,output_tensor=value)` to zero-fill plus copy (`:106`). [Binary validation](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp:788) rejects the old preallocated row-major scalar operation used for convolution state. The current zero-fill/copy also clears NaNs instead of multiplying them by zero.
- Eager prefill logits are now copied into `self.logits` and deleted before sampler replay (`:170–176`). The original live local would be tracked as a late allocation. Keep this lifetime boundary; warm the copy path too.
- Final-only generation now skips intermediate heads and selects hidden before the head (`model.py:91–95`). This removes needless large logits allocations but leaves the selector-signature issue described above.

**False positives excluded:** ignoring Sampling1D's return does not drop token ownership: [its output creation](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/reduction/sampling/device/sampling_device_operation.cpp:183) returns the supplied `self.tokens`. `self.logits` is retained across captures. `plus_one` accepts both INT32 and UINT32 and updates its input. Embedding/decode residual widths match the inherited phase policy. The page allocation's factor of eight is alignment, not accidental eightfold page indexing. Singleton first-token, position advancement, and teacher-forcing feedback order are internally consistent.

## Other potential issues / scope limits

- The low-level output contract remains incomplete: `decode_forward` returns TP-sharded rank-four device logits, whereas [the contract](/home/mvasiljev/tt-metal/.codex/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.21/runtime/readiness_check/contract.py:163) describes caller-owned cache and `[B,V]` logits or `[B]` sampled tokens. The latest constructor now accepts a bound `kv_cache` and host `page_table`; the earlier missing-binding complaint is resolved at that entry point. Bind only appropriate empty/setup state because preparation and reset mutate it. Address output formatting at the public adapter boundary, preserving internal device-only helpers. This is not the cause of the singleton debug trace-allocation rejection.
- Lazy CCL dictionaries can allocate in `_gather`/`_reduce` (`multichip_decoder.py:282,345`) when persistent policies select those branches. Do not blame them unconditionally: the default shared workspace and native decode/pre-fill setup already allocate their retained pools before capture. Verify the effective policy for each supported batch.
- `host_sampling=True` still uses device sampling for the initial prefill token. With current greedy settings it may agree, but it is not a fully independent host-sampling comparison, especially for ties. No observed wrong token is attributed to this here.
- Very long all-logits requests necessarily have a large host result; finite device buckets do not prove memory feasibility for every batch/context combination. Existing context admission and memory planning must bound configurations separately.

## Validation performed and required follow-up

Performed here, without importing TTNN or using hardware:

- AST parsing of all four focus modules.
- Exhaustive integer verification of the 16-bucket plan for every logical length 1..262144, including continuation alignment and compatibility with the current 256-token rounded pool.
- Exhaustive index arithmetic for the flattened history selector at B=1/3/32 and every n=1..4096; independent investigator checked additional physical buckets and a concrete convolution-history example.
- Lowered contracts checked for slice hashing, embedding row selection, paged fill, flexible SDPA, GDN shape/state update, sampling output/parameter rules, reset restrictions, and trace allocation tracking.
- Revisited headline claims against the current source; moved fixed wrapper failures and unproven precision/corruption interpretations out of current-cause claims.

Required device evidence after implementation, not claimed by this report:

1. Prepare once, record trace IDs/persistent addresses/cache count, and exercise different logical n within the same physical bucket and across buckets. Use lengths 1/2/3, 31/32/33, 127/128/129, 255/256/257, 4095/4096/4097, and near 262144; vary continuation origins and nonidentity page mappings. Keep allocation tracking on with program-cache tracking included. Assert no new persistent allocation/program variant, no recapture, and unchanged trace IDs.
2. For each partial tail, compare valid logits, exact convolution-history rows, recurrent state, and several subsequent decode tokens against the existing logical-length implementation/HF. Include nonzero initial history/state, n=1/2, and a coarse P containing at least one wholly padded GDN chunk to challenge the precision concern. Compare both layer types and then the full stack.
3. Exercise last/all/no-head paths and reset between requests. Check selected B values on both sides of the B=8 residual switch; test the high-level batch contract explicitly. Do not claim mixed-length/slot support from equal-length tests.
4. Run standard readiness construction followed directly by generation. Confirm all promised checks complete at a device synchronization/readback boundary, not merely at Python enqueue return.

No report claim establishes hardware accuracy, latency, full-context capacity at every batch, or successful 64-layer completion. The code and existing artifacts do establish the preparation-domain mismatch and the state semantics that a safe finite-bucket repair must preserve.
