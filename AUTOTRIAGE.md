# AUTOTRIAGE: Gemma 4 full-model changed-page-table hang

## Incident and live evidence

`test_reduced_full_model_prefill_split_greedy_and_trace` once hung after unchanged-table replay when alternate page-table buffers were supplied. Live evidence is in `models/autoports/google_gemma_4_31b/doc/full_model/triage/`.

ARC heartbeats, all Ethernet links, and DDR were healthy. Device operation state instead split across independent INT32 page-table copy operations. Source inspection matched the evidence: the old refresh path unwrapped the distributed table and submitted one physical-device `ttnn.copy` per device immediately before mesh trace replay.

## Root cause and repair

The per-device copies broke distributed dispatch ordering. They were replaced with one distributed `ttnn.copy(source, target)` per changed logical table. Private stable trace tables plus allocation identity and explicit generation deduplication now provide these invariants:

- unchanged identity/generation performs zero copies;
- one changed logical table performs one distributed copy;
- repeating the same identity/generation performs zero copies;
- external KV cache and page tables must be supplied together;
- reset releases both traces before buffers or cache state are changed.

The source-current reduced mixed-prefill/trace test passes all changed/unchanged cases.

## Split-trace allocator-warning disposition

The original triage also treated an active-trace allocation warning as unresolved because the old sampler created output and parameter tensors after the model trace was registered. That application-level defect is fixed:

- sampling parameters are allocated before model capture;
- token output, local candidate pairs, gathered pairs, and model logits are persistent;
- regular all-gather writes to the preallocated `gathered_pairs` output;
- prewarm creates exact-shape programs/resources before capture;
- sampler and model traces are released together, and custom persistent tensors are explicitly released at teardown.

One conservative warning remains when `begin_trace_capture` registers the second trace region while the model trace is already registered. This is the framework trace-region allocation required by the `$tt-enable-tracing` canonical split-sampling contract, not a data tensor allocated inside the serving loop. The post-fix controls are: repeated token feedback replay, reset/recapture, changed tables, batch-two mixed prefill, watcher, and a source-current profile with no profiler-buffer drops. All pass with zero per-token host token refreshes and zero full-logit readbacks.

## Resolution

Resolved. The hang has a proven source/evidence repair; the unsafe application allocation has been eliminated; the remaining second-trace registration warning has explicit lifetime controls and passing source-current evidence.
