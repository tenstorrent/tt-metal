# Optional batched GDN prefill convolution

CPU prototype only. No device run, correctness result, or speedup is claimed here. The final/default decoder is unchanged.

The current native `qkv_causal_conv1d_silu_with_carry` operation is batch-one only: `device/qkv_causal_conv1d_silu_device_operation.cpp` validates `[1,T,W]` (lines 115–119), `[1,3,W]` token-paged history, and a positive tile-aligned sequence. Its native batching cannot be enabled by passing `[B,S,W]`. Flattening users into sequence would mix each user's three-token causal history and is invalid. The exchange-buffer shape and history routing also assume one sequence.

The current `_gdn` TP4 caller launches this operation once per user and concatenates three outputs and carries. Parent-provided `perf/interim_profile4_l0_b32_s33` evidence attributes about 605 us to 32 native convolution calls; the four reshape kernels cost about 742 us. These are the motivating measurements, not candidate measurements.

`tt/prefill_conv_candidate.py` implements an optional generic kernel using the existing validated decode convolution's compute source without changing it. Work items are `(user, logical_time, channel_chunk)` with 1024/1024/512 channels. Each reads four chronological raw BF16 input rows, selecting negative time from that same user's three-row history. It performs FP32 products and ordered FP32 adds, rounds to BF16, then applies native BF16 SiLU. The unused half of the last channel chunk cannot reach a retained lane; the probe can poison it with NaNs.

History is immutable during the kernel. Only the final-time task for each channel chunk writes three carry rows into a distinct output allocation. A subsequent TTNN copy updates the existing state allocation after all cores finish. This avoids a cross-core race where a late-time task overwrites history while early-time tasks still read it. For logical lengths 1 and 2, untouched older history rows remain in the carry. Internal sequence padding is added only after convolution; padded tokens never enter carry. Q/K/V outputs are tiled `[B,ceil(S/32)*32,512/512/1536]`, matching the current downstream chunk-GDN consumer. No residual transpose is introduced.

Native prefill computes diagonal matmuls followed by pack-thread SiLU (`device/kernels/compute/qkv_causal_conv1d_silu.cpp`); that is not source-identical to the decode helper's explicit BF16 rounding before SiLU. The focused probe therefore requires exact agreement with explicit decode arithmetic, native-prefill PCC >=0.995, exact raw-history preservation, and exact split/resume equivalence. All full-layer accepted PCC gates remain mandatory before selection. If rounding changes fail those gates, adapt the candidate arithmetic or native kernel; do not weaken gates.

CPU validation performed:

```bash
python -m models.autoports.qwen_qwen3_6_27b.tt.prefill_conv_candidate --cpu-check
python -m py_compile models/autoports/qwen_qwen3_6_27b/tt/prefill_conv_candidate.py models/autoports/qwen_qwen3_6_27b/tests/probe_multichip_prefill_conv.py
```

The source-transform and ownership/address model checks pass 50 cases: B=1,2,3,17,32 and S=1,2,3,17,31,32,33,65,257,513. They verify every output has one owner, reads remain within that user, final three carry rows are exact, and compute source is unchanged, for normal and poisoned source variants. These checks do not compile device kernels or validate TTNN layouts.

Parent-serialized first device command (no reset; use the existing stage-owned bounded runner):

```bash
D=models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder
TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 MULTICHIP_JOB_TIMEOUT=900 bash "$D/run_job.sh" batched_prefill_conv_first \
  python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_prefill_conv \
  --cases 3x2 3x33 32x33 --output "$D/batched_prefill_conv_first.json"
```

The probe uses real TP4 `[1,4]` tensor-sharded synthetic inputs/taps with distinct ranks/users and nonzero histories. It compares the entire native user-loop/slices/concats/history-copy path with the entire candidate untilize/kernel/carry-copy/tilize/pad/split path. It tests eager changed-input resumes; explicit decode arithmetic; all-at-once vs short split/resume; and separately captured traces with changed inputs and exact expected history for each replay. No TTNN output is read inside capture or timing. Warmed trace timings use five endpoint-synchronized samples of ten replays and include necessary conversions. Captures run serially and are released before capturing the other branch, avoiding newly retained allocation ambiguity.

Then run a separate watcher/poison test, with profiler disabled and no timing:

```bash
TT_METAL_WATCHER=120 TT_METAL_DEVICE_PROFILER=0 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 MULTICHIP_JOB_TIMEOUT=1200 bash "$D/run_job.sh" batched_prefill_conv_watcher \
  python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_prefill_conv \
  --cases 1x1 3x2 17x31 32x33 3x257 3x513 --poison-dead-padding \
  --output "$D/batched_prefill_conv_watcher.json"
```

Use a fresh output name per attempt; the probe refuses to overwrite evidence. Optional `--cores 1` exercises the two-slot producer/consumer ring across long work-item sequences. All source hashes and exceptions are saved in the report.

## Minimal parent integration after focused acceptance

Add exactly one optional hook in `_gdn`, ahead of its existing `if use_batched_conv` branch (keep its existing calculation and decode behavior):

```python
if s > 1 and hasattr(self, "_prefill_batched_conv"):
    q, k, v = self._prefill_batched_conv(qkv, state.conv)
elif use_batched_conv:
    q = k = v = self._local_batched_conv(qkv, state.conv)
else:
    # existing per-user native prefill path
```

Call `prefill_conv_candidate.install(model)` after `from_state_dict`. It fails closed if that explicit hook is absent. No runtime output stash or cross-operation cache is used. Main-source changes and instance installation are parent-owned, and are not implemented by this CPU task. Before default selection, run the existing layer-0 TP4 fixture harness at B32/S33 and B1, plus nonaligned/list/resume contract cases and watcher; profile the whole layer and retain/reject using end-to-end warmed prefill time. All decode gates remain unchanged. Added transient row-major buffers must be included in context/resource testing if selected.

## Movement and prospective follow-up

This bounded prototype removes B native launches but rereads four input rows and four tap rows per time/chunk, and adds untilize/tilize plus three channel slices. It may lose on long S. A material regression should motivate row-block/tap reuse or adapting the native diagonal-matmul worker to explicit batch indices, rather than treating one configuration as a final rejection.

The parent profile shows the whole packed projection reshape `[1,B*S,4160] -> [B,S,4160]` materializes per-user tile padding (1056 ->2048 rows at B32/S33). This prototype deliberately starts from the existing `[B,S,2560]` QKV slice. A prospective coherent flat producer/consumer path would slice first 2560 columns directly from flat packed output, pass logical B/S separately to this reader's `user*S+time` indexing, and retain flat Z/beta/g until each actual consumer requires padding. That could avoid the whole packed reshape and some channel copies. It is not implemented or tested here, and merely restoring immediately to the old padded packed contract would not measure its movement benefit.
