# Runtime boundaries and completed audit

The measured autoregressive path is `Generator.generate` -> eager prepared
`prefill_forward` -> split sampler -> `_generate_buffered_tail`. Each queued decode
step calls the model trace and the buffered sampler trace with `blocking=False`.
`_decode` consumes persistent `tokens`, `positions`, `rope_indices`, cache and page
table allocations, writes sampler-ready logits, and advances position/RoPE on the
device. `_sample_buffered` passes `tt_out_tok=self.tokens` to canonical Sampling1D,
updates seeds, writes the collector with `indexed_fill`, and advances its index.
No Torch operation, CPU logits transfer, host argmax, input refresh or readback is
inside that per-token queue loop. Python submits trace IDs only.

The first-token read defines TTFT. A read after each bounded output window collects
caller-visible output; it never reconstructs feedback. The fixed4096/128 target
uses one127-step tail window, so its entire decode interval has no intermediate
host refresh/readback. EOS-enabled requests use16-step windows to bound excess
queued work; requests exceeding a window reset its cursor at the boundary.
`steady_state_counters`, `inner_loop_host_refreshes`, `inner_loop_readbacks`, trace
IDs and capture counters are recorded in benchmark JSON and tracker-enabled probes.

Request setup legitimately uses Torch for prompt encoding/padding, metadata and
host-to-device copies. Changed-only page-table refresh remains explicit. Checkpoint
loading and static RoPE table creation occur during setup. Public scheduler helpers,
accuracy logits export, teacher-forcing callbacks and explicit host-sampling
compatibility are separate paths; their host work is not presented as device-only
autoregressive throughput. The streaming comparison mode intentionally reads tokens
per step and is separately labeled.

Benchmark runners set `ttnn.CONFIG.throw_exception_on_fallback=True`; reduced
collector probes do the same. Runtime profile rows verify device matmul,
collective and sampler choices. The selected sampler is semantic greedy k1/p0/temp1
through the split32-candidate path, not force-argmax. Full logits remain vocabulary
sharded; only small candidate values/indices are gathered. The optional argmax mode
is retained solely as the measured comparison control.

Timing logs can emit the allocator's generic "potentially unsafe" warning after
capture because tracking is intentionally disabled for measurement. Eager prefill
creates temporary operation tensors; persistent request/bucket/CCL buffers already
exist, and the terminal result is copied into persistent logits before replay.
Tracker-enabled final all64 B1 validation and reduced B32 full/mixed/full probes
exercise these lifetimes with program-cache allocations included and no broad
acknowledgment scopes. The warning is not itself a tracker finding. Separate watcher
and tracker probes pass in watcher_summary.json.

Legacy small-candidate gather controls also emit the current API's deprecated-args
notice. The same compiled build executes them successfully, and the measured async
replacement is slower at B1/B8/B16. B32 selects the faster explicit async route.
This is an API migration notice, not a CPU fallback or a failed collective.

The B32 profiler warns that captured trace1 has no replay/device data. That is the
masked model trace prepared for request reuse; this profile executes all-active
model trace0 and buffered sampler trace3. Capture/preparation rows are outside the
signposted measurement windows. The collector requires nonempty device-cycle data
for every device row in those windows. Masked-trace correctness/reuse is established
separately by tracker-enabled transitions and full-stack mixed_trace_validation,
not by claiming an unexecuted masked performance profile.
