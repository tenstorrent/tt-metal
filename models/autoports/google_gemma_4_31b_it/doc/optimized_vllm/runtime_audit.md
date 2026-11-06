# Serving runtime audit

The adapter is `tt/generator_vllm.py:Gemma4ForCausalLM`. Current full60 server
registration records this path and the selected `head_bfp4` policy. The plugin
runtime matches pinned stock commit `ca7a18246e4111629ed6cb4a7d1192002749fccf`;
local commit `90f33d2e392eb8bfda6948df9dabe712a724cb1a` contains the previously
authorized test-only changes. Stage10 changes no plugin/vLLM interface or runtime.

Decode keeps the existing persistent token, position, RoPE, page-table, cache
and sampler inputs. The adapter forwards `read_from_device=False`; model and
sampler replays use `ttnn.execute_trace(..., blocking=False)`. Deferred readback
returns sampled token IDs under the plugin split-read contract. Current
`adapter_state_current.json` verifies stale host tokens/positions do not replace
resident device state, page-only reload preserves it, and deferred reads and
shared cache storage behave correctly. Fresh full served changed-page/row-order
sync/async comparison passes68requests, as detailed below.

The LM head and split sampler are unchanged from the selected full-model path:
physical localTopK32, candidate-only gather, semantic greedyk1, and device token
feedback. The adapter's `torch.aminmax` inspects already-read IDs solely to reject
out-of-vocabulary values; it neither selects a token nor adds a device read.
There is no host greedy/top1 selection or full-vocabulary readback in the measured
device-sampling path. Explicit shared-plugin host compatibility for requested
features and intermediate-containing prefill remains a documented boundary.

Short start-zero groups use persistent input buffers prepared before capture.
Token/padding, last-row indices and row-owned page mappings are staged for each
new prefill admission. Padding stays in each request's final page; causal
attention never lets valid queries read padded keys, and decode overwrites new
positions before reading them. Runtime row extraction uses embedding indices
into untilized hidden rows, avoiding logical-length/row-offset-specific slice
programs. Output slices are bounded by prepared group size and warmed. All
selected logits temporaries are released before sampling trace replay.

The paired serial/candidate benchmark lifetime audits each record7 device
prefill steps and259 device decode steps; these include benchmark warmups and
are not claimed as exact timed-window counts. The candidate additionally records
10 actual batched groups,9 page-only refreshes,7 full reloads and243 unchanged
steps. Final benchmark/mixed-traffic mode counts are below; cleanup.json confirms
all completed server/EngineCore and stress workers exited.

No Tracy, device profiler, adapter profiler, tt-perf-report or ReadDeviceProfiler
collection was used. CPU process counters in the completed host-contention check
are external workload measurements, not device or adapter profiling.

Final default B32 `stage10_final_full/performance/runtime_audit.json` is bounded
to the runner performance interval: 7 device-prefill and259 device-decode steps,
10 batched groups, no host-compatible sampling. This includes runner warmups
and the128/128/C1 diagnostic,100/100/32 CI-shaped profile, and128/32/C32 cold
burst. It is separate from the server lifetime audit, which includes requested
host-only sampling features in the full test suite.

The final async mixed test (`stage10_final_full/mixed_async.json`) passes all68
requests and matches the fresh tracked synchronous control. It records93
page-only reloads,48 full input reloads and2773 unchanged-state decode steps.
Sampling is34 device-prefill,2914 device-decode and12 accepted host-compatible
intermediate-prefill steps. All four batched physical sizes32/64/96/128 appear;
nonaligned and long prompts, continuation chunks, concurrent decode, changed
allocator mappings, retirement/reordering and sampling switches pass. Three
prepared signatures retain their IDs with one capture each and no workload
retirement. The tracked sync process used allocation tracking1 with program
cache exclusion0. The separate tracked positive-control probe proved the guard
rejects a deliberately live unsafe buffer and accepts replay after release.

The initial final server closed after a client invocation error before growing
traffic sent HTTP requests. Its completed checks remain in their original
paths; growing/load evidence uses a fresh server with identical launch source
hashes. No runtime source changed between these servers. No device owner
remained before launching the follow-up server.

The final exact4096/128/B1/C1 headline has1accepted host-compatible intermediate
prefill,1device final-prefill and127device decode steps. Growing160requests and
corrected98-request load diagnostics pass. The first load client was discarded
for repeated tokenizer-length work; its corrected cached-length version passes
and cleans all16stress workers. No runtime source changed.
