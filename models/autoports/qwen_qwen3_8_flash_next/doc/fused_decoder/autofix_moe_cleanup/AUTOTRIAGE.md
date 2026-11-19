# AUTOTRIAGE

## Diagnosis

The prototype's `zero_dtype` rewrite changes a device SFPU fill into a host-to-device upload during trace capture. For FP32 logits, `zeros_like(logits, dtype=bfloat16)` cannot use the same-dtype `full_like_impl` fast path. Its fallback `full_impl` also declines device fill for this small BF16 buffer. The upload raises the observed `Writes are not supported during trace capture. trace id: 6` exception. The prototype then releases the trace without ending capture; this leaves command-queue capture state and bypass mode active. That lifecycle defect plausibly explains the subsequent teardown wait, but the provided capture contains no host stack proving the exact teardown stop site.

Scope: source diagnosis of this single hypothesis only. No TTNN imports, device operations, tests, resets, process termination, or implementation edits were performed. Runtime `tt/fused_decoder.py` is unchanged by this investigation.

## Triage Evidence

- Primary evidence: `tt-triage.txt` and `triage_capture.log` in this directory. `dump_op_mesh.py` reports device 0 idle; running-operation and circular-buffer sections identify no active model kernel or blocked model CB.
- Dispatch cores wait for new work: `cq_prefetch` at `fetch_q_get_cmds`, waypoint HQW; `cq_dispatch` at `CBReader::acquire_pages`, waypoint UAPW; subordinate dispatch at `cb_acquire_pages_dispatch_s`, waypoint DAPW. These are compatible with idle queues and do not identify an MoE producer failure.
- The lightweight-assert section explicitly says `ASSERT() not found!`; its TRISC PCs map to firmware's wait-for-GO loop (`tt_metal/hw/firmware/src/tt-1xx/trisc.cc:135,151`). These entries are not evidence of an LLK assertion.
- Raw triage reports an Ethernet erisc1 NoC counter mismatch on physical core 15-1. It also reports cores no longer halted after triage halted them, failed dispatch-symbol reads because BRISC was not halted, and unavailable watcher-ringbuffer members. Retain these limitations; they do not establish an Ethernet root cause for the earlier host write-contract exception.
- `triage-summary.txt` labels every script pass, including sections with raw failures/warnings. Treat that as script-completion metadata, not an all-clear hardware verdict.
- `../logs/moe_cleanup_layer0_watcher_v1.log` records five count=1 variants with exact eager and replay output, then count=2 `current` with exact output, followed at 13:06:28.442 UTC by the trace-id-6 write exception. Loop order identifies count=2 `zero_dtype` as the next capture. There is no Python traceback or successful result row for that capture.
- `../moe_cleanup_layer0_watcher_v1_provenance.json` records `timeout 600 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_fused_moe_cleanup`, layer 0, BF16 experts, Watcher enabled, device 0, and trace allocation tracking. Prototype and frozen `../sources/next_prototype_plan_v1/probe_fused_moe_cleanup.py.txt` both hash to `daf21b374876bca9a058f595ca7aab0f0eb685f346cd350e1bff143954b73f79`, matching provenance.

## Source Evidence

Paths below are relative to the repository root.

1. `models/autoports/qwen_qwen3_8_flash_next/tt/fused_decoder.py:404-425`: router logits are explicitly FP32. For one token, `compact_experts=True` selects the compact route and skips the dense zero buffer entirely. Thus passing count=1 variants do not exercise the rewrite. Count=2 is the first affected case.
2. `models/autoports/qwen_qwen3_8_flash_next/tests/probe_fused_moe_cleanup.py:23-27` replaces `zeros_like(typecast(logits, bfloat16))` with `zeros_like(logits, dtype=bfloat16)`.
3. `ttnn/cpp/ttnn/operations/creation/creation.cpp:282-313`: same-dtype tiled device tensors take `ttnn::fill`; dtype mismatch instead calls `full_impl`. The original cast makes input/output BF16 and takes the device fill; the rewrite's FP32/BF16 mismatch does not.
4. `creation.cpp:206-248`: the fallback is not universally host-only. It has its own device fill optimization, but BF16 requires a packed buffer at least `128 * 1024` bytes (`:26`). The checkpoint config specifies 512 experts. For logical logits `[1,1,2,512]`, tile padding yields `[1,1,32,512]`, whose BF16 payload is 32,768 bytes. All tested counts through 32 remain below 131,072 bytes. Hence this fallback constructs a host BF16 tensor and uploads it (`creation.cpp:67-88`).
5. `tt_metal/distributed/fd_mesh_command_queue.cpp:856-876`: `write_shard_to_device` rejects writes when `trace_id_` is set, with the exact message and trace-id suffix from the failure log. The check occurs before the device-buffer write.

Trace lifecycle ledger:

| Boundary | Source behavior | Required pairing |
| --- | --- | --- |
| Begin | `FDMeshCommandQueue::record_begin`, lines 1626-1643, saves host dispatch state, sets trace id/context, and enables sysmem bypass mode | End capture must restore this state |
| Body failure | Prototype line 112 raises during `model(x)` | Prototype line 113 is skipped |
| Release | Prototype lines 129-130 call `release_trace`; `MeshDeviceImpl::release_mesh_trace`, lines 1686-1703, erases trace ownership and unregisters it | Release does not call `record_end` or disable bypass |
| End | `MeshDeviceImpl::end_mesh_trace`, lines 1752-1790, calls `record_end`; `record_end`, lines 1886-1907, clears capture id/context, restores host dispatch state, disables bypass | Must precede release/ordinary close after a body exception |
| Close | Prototype line 131 calls close; mesh close initiates profiler/queue teardown | Capture state is still active in the failing prototype |

`FDMeshCommandQueue::finish_nolock` (lines 801-821) records a host event and waits for outstanding reads. Capture bypass and capture-relative dispatch state are incompatible with ordinary completion/teardown expectations. This is a source explanation of the observed idle device plus live host process, not proof of the host's exact blocked frame.

## Downstream Effects

The initiating violation is a forbidden host write during capture, not an observed model-kernel CB, NoC, or semaphore imbalance. Dispatch waits and the later teardown stall should not be repaired by changing worker kernels or hardware synchronization. The current `finally` also writes the JSON report only after `close_mesh_device`; a stuck close therefore prevents durable failure reporting.

## Proposed Fix

Prefer a prototype-local helper that explicitly allocates BF16 device storage and fills that storage with the native unary op:

```python
def device_bf16_zeros_like(logits):
    output = ttnn.empty_like(logits, dtype=ttnn.bfloat16)
    return ttnn.fill(output, 0.0, output_tensor=output)
```

Replace the `zero_dtype` candidate expression with this helper and provide the helper in the candidate's execution namespace. This removes the redundant cast while preserving dtype, shape, layout, placement, and topology. `creation.cpp:442-462` implements `empty_like` by `create_device_tensor`, including a dtype override and inherited topology. `unary.cpp:271-291` maps `fill` to the native unary FILL operation; its Python binding accepts `output_tensor` (`unary_nanobind.cpp:455-464`). The exact in-place empty-plus-fill pattern is already used by C++ `full_impl` at `creation.cpp:226-232`, but the explicit helper avoids its size threshold. FILL lowers to `fill_tile_init` plus `fill_tile_bitcast` (`unary_op_utils.cpp:144-177`), assigning the scalar rather than computing a value from uninitialized input. No FP32-to-BF16 cast is necessary for zero.

Fix exception lifecycle independently: track whether capture successfully began and whether its end completed. If the model body throws, end that partial capture on the same CQ before releasing it; never replay the partial trace. A narrow `try/finally` around the capture body ensures `end_trace_capture` runs before the existing outer release/close. Keep the original model exception and any cleanup exception separately visible, and persist `passed=false` plus completed rows and failure detail before potentially blocking teardown. Ordinary release must not be treated as an abort API.

If `end_trace_capture` itself fails, do not blindly repeat it: C++ may already have cleared CQ capture state before a later population/allocation failure. The simple body-finally handles this reported body-write failure; fully general recovery from an end failure needs explicit state-aware handling and must not be advertised as proven by this report.

Suggested discriminating checks for the parent, after preserving live evidence: smallest device smoke comparing old and new BF16 zeros for `[1,1,2,512]`, then capture/replay and exact output comparison; run the repaired prototype's count=2 case followed by its existing token-count matrix; a controlled exception inside a capture body should exit with a report and complete cleanup. Preserve Watcher and provenance evidence. No performance claim follows from source inspection alone.

## Uncertainty

- The failing log plus exact message strongly support the upload path, but contain no Python/C++ backtrace for the initiating call. The matching source snapshot, loop order, shapes, and C++ branch conditions supply the attribution.
- No host teardown stack was supplied; exact blocked frame and sufficiency of capture-end cleanup remain to be tested.
- The source includes an additional size-gated device-fill fast path; do not generalize this diagnosis to all dtype-overriding `zeros_like` calls.
- Raw triage warnings prevent interpreting summary passes as proof of complete hardware health. No hardware repair is inferred from them.
- Proposed helper and exception cleanup are unimplemented and untested here. This report records a supported diagnosis and minimal repair proposal, not a verified AutoFix result.
