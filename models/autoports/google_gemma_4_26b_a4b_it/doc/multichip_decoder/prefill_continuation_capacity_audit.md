# Bounded prefix-continuation output assembly

Status: source-checked proposal; CPU control-flow simulation passed. The parent subsequently applied the patch to its AGMM runtime and owns device validation. This audit used no hardware. The change covers only `MultichipDecoder` continuation output assembly. Existing per-token attention, active-expert execution, cache updates, precision, and collectives remain unchanged.

Patch: [prefill_continuation_capacity.patch](prefill_continuation_capacity.patch). Base runtime SHA256: `e1994ff8634fa6e7d3518726bb3b5d53bb55fc7c7d56f45545f712f89821a420`. Candidate SHA256: `79c810865b4aa6e2e495ee3bec24452577d12e1cc578932bfa05c027940eda97`. The base includes the fresh-prefill bounded assembly and concurrent AGMM source changes present when the patch was generated. The patch adds one helper and a continuation dispatch branch; it does not modify those other features.

## Source finding

At the recorded base, `tt/multichip_decoder.py:559` delegates every nonzero `start_pos` to `OptimizedDecoder.prefill_forward`. `tt/optimized_decoder.py:676–701` performs one `decode_forward` for every logical continuation token and retains every result in `outputs` until a final concat. Each BF16 result is logically `[1,1,1,H]`, physically tile-padded to `[1,1,32,H]`. At `H=2816`, 262,143 retained results consume **47,244,460,032 bytes = 43.9998 GiB per device**, before input, weights, KV cache, or concat temporaries. The fresh-prefill maximum-context evidence does not exercise this branch.

The per-token calls are required for the current implementation's prefix contract: a continuation can start inside an occupied cache page. The existing code selects `page_table[user_id:user_id+1,:]`, uses absolute position `start_pos+offset`, and issues a one-token paged update. The proposed helper copies that call sequence exactly. It neither adds padded decode calls nor invokes a dense all-expert path.

## Proposed assembly and layout

Each 32 logical one-token outputs is immediately concatenated into a logical and physical 32-row tile. Thirty-two tiles become a 1024-row chunk; thirty-two chunks become a 32768-row group. The final concat consumes at most eight groups at the 262144 context limit. Every other concat consumes at most 32 tensors.

For the last partial token block, the patch repeats the last real output tensor reference until the list contains 32 rows. These extra rows pass only through concat and are removed by the final logical slice. Their values never reach attention, cache writes, expert routing, normalization, or a subsequent layer. This padding requires no fill allocation, host tensor operation, or logical-volume-expanding reshape. A length-one continuation returns its existing single output directly.

`merge()` clears the child list after constructing its result. Its single-input case returns the same tensor, so the patch deliberately does not force deallocation. There is no remaining local reference to every emitted token or previous child group. Width is inherited from the output tensors, so assembly also preserves the existing local width 704 for the optional hidden-sharded residual layout; it does not gather, reshape, or change mesh ownership.

Relevant TTNN source:

- `ttnn/cpp/ttnn/operations/data_movement/concat/concat.cpp:93–135` chooses untilize/unpad → row-major concat → tilize when the concat dimension has implicit tile padding. Only each bounded 32-token group triggers that path. Every higher level has tile-aligned logical row counts.
- `ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_device_operation.cpp:240–317` calculates the interleaved concat limit as 47 inputs and recursively batches larger lists. The patch remains below that limit at every level.
- `ttnn/cpp/ttnn/operations/data_movement/slice/slice.cpp:247–253,331–385` selects the TILE path when slice starts are tile aligned, rounds the ends physically, and restores the requested logical shape with a view. The final slice starts at zero, so its nonaligned end does not require full-result row-major conversion.
- `ttnn/cpp/ttnn/operations/data_movement/pad/pad.cpp:357–376` adds TILE padding to the existing padded shape and requires aligned physical ends. Repeating a logical row avoids relying on padding a logical partial tile with `32-logical_rows` or expanding its logical volume by reshape.

## Capacity accounting

Let `A = round_up(length,32) * H * 2`. At maximum context and replicated width, `A = 1,476,395,008 bytes = 1.375 GiB`.

| Live payload | Conservative bound per device |
| --- | ---: |
| Caller input | `A` |
| Retained assembled output rows | approximately `A`, plus bounded pending token-tile overhead |
| Final concat output, while its groups remain live | `A` |
| Final logical slice | At most another `A`, after child groups have been released |
| Largest intermediate rollup output | 184,549,376 bytes, 32768 rows |
| At most 32 separate one-token output tiles | 5,767,168 bytes, 5.5 MiB |

The large-buffer envelope remains **`3A = 4,429,185,024 bytes = 4.125 GiB`**, with rollup workspaces and decode temporaries covered separately by the existing 2 GiB reserve. These are calculated payload bounds, not allocator or device measurements. The prior full-stack peak estimates must continue to include both the 4.125 GiB envelope and the reserve; this proposal does not justify reducing either. Allocation granularity, live kernel workspace, and complete continuation execution still require hardware validation.

The inherited input expression `hidden_states[:,:,offset:offset+1,:]` is also unchanged. At a nonaligned start, native slice can temporarily untilize the full continuation input; that is another `A` during slicing, alongside caller input and retained output, within the same large-buffer envelope. It introduces substantial repeated data movement for long continuations. The patch fixes output retention; it makes no continuation-latency claim.

## Checks completed without TTNN or hardware

The candidate parses with Python AST. `git apply --check` passed against the live source at generation time. The changed methods passed Black with the repository's line length 120; the whole temporary candidate had an unrelated pre-existing AGMM formatting difference. A stdlib-only harness extracted the actual helper AST and supplied abstract tensor/position/concat objects. It checked output order, exact decode-call count, selected request ownership, absolute cache/current positions, output width, and concat fan-in. It ran 28 valid cases: lengths `1,2,31,32,33,1023,1024,1025,32767,32768,32769,65537,262113,262143`, each at widths 2816 and 704. Starts were 31, except length 262143 used start 1. Three invalid cases (zero length, negative start, context overflow) raised before decode.

For `length=1025,start_pos=31`, there were exactly 1025 decode calls, 33 bounded unaligned concats producing 32 rows each, one 1024-row merge, and one two-input merge producing 1056 physical/logical assembly rows. The final slice returned exactly 1025 rows in order. At both maximum-context cases, the final concat had eight inputs and every concat had at most 32 inputs. This simulation validates Python orchestration, not TTNN kernel behavior, numerical accuracy, or real memory allocation.

## Bounded device test plan; not run

1. First isolate native assembly with BF16 outputs at widths 2816 and 704. Cover 32, 33, 1024, and 1025 logical rows, including repeated tensor references in the final block. Compare the returned logical rows exactly with a setup-created reference. Run under the existing `device_only` guard; record physical/logical shapes and allocation peaks.
2. For each actual layer kind (layer 0 sliding and layer 5 full), use the existing real fixture and a selected request with a 31-token populated prefix. Continue with **1025 tokens at start 31**, crossing both the 32-row collapse and 1024-row rollup boundaries. Allocate page size 32 with read extent rounded to 1152 tokens. Compare every continuation output with the unchanged per-token reference or TP1, requiring PCC ≥0.995; no dummy padding row may be returned.
3. Snapshot every rank's K/V before and after. Preserve prefix `[0,31)`, all other request pages, and selected-request rows `[1056,1152)` exactly. The last continuation write is absolute position 1055. Replay the next decode at absolute position 1056, then an advancing step at 1057, comparing outputs and caches. Use the existing independent cache fixtures for reference and candidate.
4. Retain nonaligned start and output length in later capacity probes. Use staged continuation lengths 32769 and then `262113` at start 31 only after the bounded test and measured allocation evidence pass. Reuse the full-stack reservation, maximum-context RoPE, rounded page/cache coverage, and complete process-close checks. Fresh-prefill maximum-context runs alone cannot close this continuation contract.

No runtime/test source was edited by this audit, and no device was opened by this audit. The parent's subsequent checks are separate evidence.
