# Tiled native-math batched prefill convolution adaptation

New opt-in prototype only. No main/native/shared file was edited. This task imported neither torch nor TTNN and ran no device code. Device compilation, API compatibility, numerical accuracy, and speed remain unverified.

The completed `batched_prefill_conv_perf.json` gives the reason to adapt the schedule rather than reject batching:

| B / S | Native user loop (ms) | Per-token batched SFPU (ms) | Interpretation |
|---|---:|---:|---|
| 1 / 512 | 0.060287 | 0.302312 | 5.0x regression |
| 3 / 33 | 0.099604 | 0.139623 | 1.4x regression |
| 32 / 33 | 0.880540 | 0.709351 | 19.4% improvement |
| 32 / 512 | 2.526479 | 7.863622 | 3.1x regression |

These are measured warmed traced kernel-family times from that artifact, not results of this adaptation. The first candidate rereads four raw rows and all four taps for each time/channel work item and performs untilize/tilize. The long-sequence regression is consistent with that schedule's traffic and per-token work; the measurements alone do not establish the exact bottleneck.

## Source audit and selected adaptation

The native operation's validation, history/exchange shapes, chronology binding, and factory are batch-one. Changing only the validator is insufficient. Its factory uses the Metal 2 named `KernelSpec`/tensor/DFB/scratchpad binding API, whereas Python `generic_op` exposes `KernelDescriptor`/`CBDescriptor`. Directly injecting its `TT_KERNEL` sources into a generic descriptor would leave generated `tensor::`, `dfb::`, `scratch::`, and semaphore bindings undefined.

The useful native schedule is 32 time rows per channel block, with tap diagonals built once and reused across time tiles. Native `compute/qkv_causal_conv1d_silu.cpp` multiplies each of four shifted input tiles by a diagonal tap tile and applies pack-thread SiLU. The new `tt/tiled_prefill_conv_candidate.py` preserves that compute body, adapting only entrypoint/compile arguments and named DFB bindings to numbered circular buffers. No FP32 arithmetic, tap order, matmul operation, or native SiLU code is rewritten. The native source hash and generated-compute hash are in `tiled_prefill_conv_plan.json`.

The new bounded reader/writer target **TP4 with SP-axis size one**, as used by this decoder. No fabric chronology or exchange is needed between TP ranks: each rank retains its local channel history. This is not a sequence-parallel implementation.

- Work order is `(user, channel_block, time_tile)`. A core's contiguous work range reuses tap tiles/diagonals until its block changes. `block_ct=8` matches 256-channel native blocks; the bounded alternatives are 4 and 2 (1 is available for diagnosis).
- Input stays TILE. Both `[B,S,W]` and compact `[1,B*S,W]` are accepted, with W>=2560; a full `[1,B*S,4160]` packed projection is legal without a QKV slice. `column_offset` is tile aligned and defaults to zero.
- The reader interprets compact user starts as `user*S`; standard per-user input uses `user*ceil(S/32)*32`. It loads at most three complete source tiles for a 35-row causal window, using 64-byte-aligned DRAM reads. Those tiles can contain other users' rows physically, but only this user's valid row ranges are copied into the window.
- Two local 35-row strips per channel tile contain three halo rows and 32 current/dead rows. Negative time is copied from that user's stable raw history. Face-row local copies are coalesced until a face boundary; dead logical rows remain zero. Four shifted activation tiles are produced from these strips using the native face-copy layout.
- The reader writes the final three **raw logical** token rows to a separate carry allocation. For S=1/2 this correctly retains older history rows. The input history is never mutated while other cores read it. One post-kernel copy preserves the stable state allocation.
- The writer produces Q/K/V directly into `[B,ceil(S/32)*32,512/512/1536]` TILE outputs. There is no input untilize, output tilize, or Q/K/V channel split. No whole packed reshape is introduced.

`block_ct=8` uses 288,768 bytes/core of explicit CB/scratch allocations; 4 uses 188,416 and 2 uses 98,304. These are descriptor estimates, not measured peak/free L1. B32/S512 has 5,120 time-tile work items at block8, versus 49,152 per-token/channel-chunk tasks in the prior candidate. Unlike the most optimized native reader, this bounded version can reread a previous source tile rather than retaining its halo across time tiles. It still reuses each source tile across all four taps and computes 32 time rows together; persistent halo reuse is a justified next adaptation if measured data movement remains material.

The constructor API was checked against `ttnn-nanobind/program_descriptors.cpp`: `ComputeConfigDescriptor` has no `packer_l1_acc` argument, so the prototype does not pass one. Compute uses HiFi4, FP32 destination, precise math, half destination sync and native pack-SiLU. Numbered circular-buffer reservations and pack/wait/pop ordering follow the native compute body. The generated reader uses generic `TensorAccessorArgs`, not unresolved Metal 2 bindings.

## CPU checks performed

```bash
python -m models.autoports.qwen_qwen3_6_27b.tt.tiled_prefill_conv_candidate --cpu-check
python -m py_compile models/autoports/qwen_qwen3_6_27b/tt/tiled_prefill_conv_candidate.py models/autoports/qwen_qwen3_6_27b/tests/probe_multichip_tiled_prefill_conv.py
```

Eighty-eight address/source cases pass: B=1,3,17,32; S=1,2,3,17,31,32,33,65,257,512,513; both compact and per-user layouts. Checks cover the maximum three-tile window, history/valid/dead row classification, source face offsets, final logical carry indices, source transformation, group divisibility and explicit CB-size bounds. `tiled_prefill_conv_cpu.log` and `tiled_prefill_conv_plan.json` record these results. They do not prove hardware execution.

## Bounded parent device sequence

Start with one short shape to resolve compilation/addressing before expensive sweeps:

```bash
D=models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder
TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 MULTICHIP_JOB_TIMEOUT=900 bash "$D/run_job.sh" tiled_prefill_conv_first \
  python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_tiled_prefill_conv \
  --cases 3x33 --block-ct 8 --output "$D/tiled_prefill_conv_first.json"
```

The new probe retains the native reference, eager/changed-input trace comparisons, exact history, exact split/resume, and first/last replay checks. It additionally uploads **compact full packed width 4160**, with NaNs in the unused Z/BA columns, and requires exact retained-output equality to the same candidate on standard per-user input. Thus the compact family is tested without a restore-to-old-contract inside its convolution. Timing remains an apples-to-apples native versus candidate comparison on standard user input; full-layer compact-family timing follows separately.

This candidate intentionally restores native prefill arithmetic, unlike the previous explicit-decode SFPU candidate. Its probe therefore reports/gates PCC>=.995 against the explicit-decode arithmetic oracle rather than requiring bit identity to that different arithmetic. Native-prefill PCC>=.995, raw-history equality, exact compact/regular equivalence, exact split/resume, trace/eager bit equality, and all existing full-layer accepted PCC gates remain mandatory. Native bit mismatches are also recorded for diagnosis.

After the first short pass, repeat the measured matrix with fresh artifacts:

```bash
TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 MULTICHIP_JOB_TIMEOUT=1200 bash "$D/run_job.sh" tiled_prefill_conv_perf \
  python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_tiled_prefill_conv \
  --cases 1x512 3x33 32x33 32x512 --block-ct 8 --output "$D/tiled_prefill_conv_perf.json"
```

Try `--block-ct 4` and then 2 for regressions, preserving each report. First TTNN/API/compiler failure is an implementation defect to repair, not performance evidence against this material family. Inspect native source/config and adapt the binding/scratch/face addressing; use the stage's autofix process if needed.

Run watcher separately, profiler off, on `1x1 3x2 3x33 17x31 32x33 3x513`, optionally `--cores 1` to stress ring-buffer reuse across many work items. Do not use inherited `--poison-dead-padding`: it applies to the obsolete half-channel SFPU scratch. The tiled probe always poisons unused packed columns and the reader owns zeroed time-window padding.

## Compact full-layer integration proposal

In the optional `flat_packed_gdn` consumer, replace its QKV-slice/regular-convolution invocation with:

```python
q, k, v = tiled_prefill_conv(
    packed, state.conv, layer.taps,
    batch=batch, seq=seq, block_ct=8,
)
```

This directly consumes `[1,B*S,4160]`, preserving compact Z/beta/g and the previously proposed compact readout/gate/out-projection family. It removes both the whole packed reshape and the extra compact QKV slice. The parent owns this optional integration; no shared helper was modified by this task. Test the complete family against current default and the prior per-token flat family on identical final policies, fixtures and accepted gates. Avoid selecting by convolution-only time if the whole-layer result regresses. Context/resource/watcher/stress gates and final-default reproduction remain necessary before keeping it.
