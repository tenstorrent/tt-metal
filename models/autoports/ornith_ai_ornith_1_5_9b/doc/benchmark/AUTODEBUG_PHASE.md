# Phase-collector diagnosis from the preserved setup pilot

The collector correctly rejected an unobservable prefill/decode boundary in the invalid-length text-transport pilot. Do not replace this rejection with prefill Python-return time or inferred request latency. Correcting transport to exact 4096 token IDs should avoid the unsupported boundary under the current Stage10 4096-token, prefill-only scheduling steps. That prediction still requires the fresh setup measurement.

## Evidence

Snapshot: `servers/b32-1794637295287193086/timing.json`, read without touching the running process or hardware. It contains 793 completed event records, 141 prefill calls, 652 decode calls, and zero pending decode handles. Of the prefill calls, 128 have a terminal completion boundary and 13 are all-discarded. Their original prompt lengths are all greater than 4096 (range among these records: 4101 through 4684). The observed lengths exceed the intended 4096 input contract.

The first unsupported transition is:

| Event | Kind | Host start | Host return/completion | Input position | Original length |
|---|---|---:|---:|---|---:|
| 3 | All-discarded prefill | 3602327.079314044 | 3602327.240461162 (return only) | 0 → 4096 | 4612 |
| 4 | Decode of an older request | 3602327.241691929 | 3602327.654907342 | 4098 | — |
| 5 | Decode of an older request | 3602327.655830953 | 3602327.673632273 | 4099 | — |
| 6 | Mixed prefill | 3602327.674383646 | 3602328.823066528 (terminal read) | 0 → 3456 and 4096 → 4612 | 4612 / 4612 |

The 13 all-discarded event IDs are 3, 45, 99, 233, 281, 393, 398, 407, 449, 560, 566, 575, 617. They all contain a single row. Other mixed prefill events have `(discarded=True, discarded=False)` in execution order. No captured mixed event ends with an asynchronously discarded row.

## Source-supported causal chain

1. `tt/prefill_runtime.py:529–531` returns `None` immediately after issuing layer/cache/history operations when `discard_logits=True`; it does not execute `_head` or a host read.
2. `tt/prefill_contract.py:433–439` replaces that discarded output with a CPU `torch.zeros` tensor. Its later `torch.cat` is a CPU operation. This does not prove prior device completion.
3. A terminal host-logit path executes `_head`, whose `model.host_logits` performs `ttnn.to_torch`. Device sampling completes through the existing sampler/readback path. Those establish a real completion boundary; a discarded-only return does not.
4. The canonical generator's check for uncompleted decode handles prevents prefill from entering ahead of an unfinished decode. It does **not** provide the reverse guarantee: decode may enqueue behind an asynchronously discarded prefill.
5. All work is ordered on CQ0, so the subsequent decode completion proves both the preceding discarded prefill and that decode have completed. It does not locate the boundary between them. Charging the combined interval entirely to either phase, or subtracting a neighboring decode duration, would invent phase measurements.

## Does a valid 4096 input necessarily split?

The TT scheduler explicitly separates prefill and decode. In `vllm_tt_plugin/scheduler.py:957–981`, `_schedule_prefill_only` temporarily hides every running decode request before invoking the base scheduler. Thus active decodes do not deduct one token apiece from the 4096 prefill token budget. Responsive 1:2 alternates whole scheduler modes; it does not share this token budget with decode.

In an idle workload containing only exact-4096-token prompts, the first prompt consumes the full 4096 budget and completes. Inductively there is no partial remainder to steal the next prompt's budget. At 32 × (4096 + 128), the aggregate workload also remains below the selected 262144-token cache pool, so ordinary capacity-driven preemption is not expected. A smaller effective scheduled-token cap, preemption/replay, or mixed-length concurrent traffic could break these premises and must remain observable.

Some captured exact-4096 requests were split in the invalid pilot, but each occurred amid longer prompts and was paired with a preceding partial continuation. Examples: event18 `[0→1920, 1920→4096]`, event78 `[0→1024, 1024→4096]`. These do not show that a homogeneous exact-4096 workload must split.

## Minimal supported repair path

First repair token transport and rerun the existing setup performance collection. Preserve current incomplete evidence. Require actual server prompt lengths all4096, terminal completion for every measured prefill interval, exact request/output counts, and successful phase collection before accuracy. No timing-source change is necessary if these conditions hold.

There is no supported offline reclassification that recovers independent phase times from this invalid pilot. If a valid benchmark still requires discarded-only prefill followed by decode, extend completion observation rather than use Python enqueue/return timestamps. The native public `tt::tt_metal::distributed::EventQuery(const MeshEvent&)` exists (`tt_metal/api/tt-metalium/distributed.hpp:108`) and is explicitly nonblocking. The current Python events bindings expose record/wait/synchronize, but not query. A possible future minimal implementation is a nonblocking prefill-end CQ0 event plus an observer that queries it and records the actual host-observed completion frontier, without any event wait/device synchronize and without blocking serving dispatch. This requires a reviewed binding/integration change and live validation; it is a proposal, not an available or measured repair. Polling observation uncertainty and event overhead would need reporting.

A separate generic recorder limitation is that mixed prefill completion is currently marked true if any row is terminal. This is valid for this snapshot's observed `(discarded, terminal)` ordering, but would be invalid for `(terminal, discarded)`. A future general chunked collector must identify the final executed active row's completion (and earlier terminal frontiers), rather than use `any terminal` as a universal guarantee. Exact4096 single-row profiles avoid this limitation.

No source changes, server calls, device commands, or process operations were made for this diagnosis.
