# Coherent compact packed GDN prefill candidate

CPU implementation only; no torch/TTNN import or device execution occurred while preparing it. `tt/flat_gdn_prefill_candidate.py` is new and opt-in; main decoder, existing helpers, and accepted artifacts are unchanged. It depends on the batched prefill convolution candidate passing its native and full-layer correctness gates.

The current `_flat_chunk` projects compact `[1,B*S,5120]` rows and combines the primary/correction result into `[1,B*S,4160]`. It then reshapes the entire packed tensor to `[B,S,4160]`, materializing per-user tile padding. At B32/S33, this expands 1056 compact rows to 2048 physical rows. The packed-reshape payload alone is 8,785,920 input bytes plus 17,039,360 output bytes/device (BF16); this is an address-volume estimate, not measured NoC traffic. Parent profile `perf/interim_profile4_l0_b32_s33` reports about 742 us across four reshape kernels; that total must not be attributed solely to this one reshape.

The candidate is a complete consumer family:

| Value | Candidate contract | Padding boundary |
|---|---|---|
| Corrected packed projection | `[1,B*S,4160]` BF16 | Only global row tile padding, owned by projection |
| Raw QKV projection | `[1,B*S,2560]` BF16 | None by user before convolution |
| Convolution input | Row-major compact `[1,B*S,2560]` | Reader maps `user*S+time`, negative time to that user's history |
| Q/K/V convolution output | `[B,ceil(S/32)*32,512/512/1536]` BF16 | Added after raw carry extraction, only for temporal recurrence |
| Z gate | `[1,B*S,1536]` BF16 | Stays compact through SiLU/multiply |
| beta/g | `[B,ceil(S/32)*32,12]` BF16/FP32 | Sigmoid/decay operate compact; only these small tensors acquire user padding |
| GDN state/readout | Original FP32 chunk recurrence and head-major normalization | Existing masks and public state publication unchanged |
| Normalized readout/out projection | `[1,B*S,1536]` | User padding removed before gate; following projection/RS stays compact |

The convolution launcher uses exactly the regular batched candidate's generic kernels, configs, CB sizes, and task schedule. Its only launcher difference is the input shape check: physical `[1,B*S,2560]` is interpreted using explicit logical B/S. The row-major input still stores each user's contiguous tokens, with no dummy rows between users; the existing reader's `user*S+time` is correct. The result allocation alone is `[B,S,2560]`, because the temporal chunk-GDN consumer needs user axes. A distinct carry allocation prevents early-task old-history reads racing with late-task final-history writes.

The full 4160-column result is never restored to the old padded contract. QKV channels alone are padded at their temporal consumer. Z stays compact from projection through SiLU and multiply. The candidate retains `layer._decay`, exact beta/g column ranges, chunk size 32, all recurrence masks, FP32 live recurrent state, original normalization weight/epsilon/config, and the original BF16 SiLU then multiply gating order. The readout flatten occurs before elementwise gating; it does not change mathematical dependencies.

The shared scheduler retains chunk length, logical start, list dispatch, single-token tails, and final BF16 state publication. The helper asserts FP32 live recurrence and never rounds state between chunks. Single-user or single-token calls stay on the existing `_legacy_chunk` path. Each user's raw three-token history and beta/g sequence axes remain separate; neither user boundaries nor padding can contribute to state. No inter-layer collective or residual-layout contract changes.

## Minimal parent-owned integration

In `flat_prefill.py::_flat_chunk`, replace only the existing two lines after packed projection:

```python
packed = ttnn.reshape(packed, (batch, seq, packed.shape[-1]))
attention = layer._gdn(h, state, packed=packed, flat_output=True)
```

with this optional branch:

```python
if hasattr(layer, "_flat_packed_gdn"):
    attention = layer._flat_packed_gdn(packed, state, batch=batch, seq=seq)
else:
    packed = ttnn.reshape(packed, (batch, seq, packed.shape[-1]))
    attention = layer._gdn(h, state, packed=packed, flat_output=True)
```

No change is needed in the packed-correction helper: with physical batch 1 its `project_prefill` already returns `[1,B*S,4160]`, and current `_prefill_packed_gdn` retains exact quantized weights and FP32 correction combine. The existing projection row cap/alignment remains owned by that producer. `install(layer)` fails closed until the new optional hook exists. No new weights, cross-operation output stash, or runtime policy switch is introduced. The instance closure is for candidate testing; if selected, use an owner-explicit class method for production, matching the current packed-correction owner pattern, before the 64-owner lifetime/resource gate.

The module CLI installs the helper on the TP4 decoder instance, forwards all existing accepted harness gates, and records sources, invocation count, and failures. It refuses baseline mode and output overwrite. The hook must be exercised, so the first full-layer command uses B32/S33:

```bash
D=models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder
TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 MULTICHIP_JOB_TIMEOUT=900 bash "$D/run_job.sh" flat_packed_gdn_first \
  python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tt.flat_gdn_prefill_candidate \
  --output "$D/flat_packed_gdn_first.json" -- \
  --trace --repeats 20 --layer 0 --batch 32 --length 33 \
  --reference models/autoports/qwen_qwen3_6_27b/doc/multichip_decoder/delivery_perf_ref_l0_b32_s33.pt
```

For attribution, compare three coherent full-layer paths with identical final policy/fixtures: current default; regular batched convolution after whole packed reshape; this flat packed consumer. A kernel-only comparison that first reshapes flat input back to `[B,S,4160]` does not measure this family. Profile the complete layer after passing PCC and compare warmed prefill; traced decode should remain unchanged because this helper is never used there. The source dependency's native convolution arithmetic distinction remains material: native prefill uses diagonal matmul/pack-SiLU while the candidate uses validated decode arithmetic. Accepted full-layer PCC gates must decide equivalence; do not relax them.

Required follow-up after first pass: B2/S2, B3/S33, B17/S31, B32/S257, B3/S513; repeated chunk/resume with nonzero distinct histories; unchanged list and B1 behavior; watcher/trace-allocation runs separate from profiler; final resource/context/stress gates if selected. The parent owns serial device execution and fresh references for any shape without an accepted fixture. The helper keeps logical lengths valid and owns padding; no aligned public-input requirement is introduced.

## CPU validation and remaining limitations

```bash
python -m models.autoports.qwen_qwen3_6_27b.tt.flat_gdn_prefill_candidate --cpu-check
python -m py_compile models/autoports/qwen_qwen3_6_27b/tt/flat_gdn_prefill_candidate.py
```

CPU checks pass the shared 50 convolution address/source cases and 32 compact/padded user-index cases. An AST comparison proves everything after the input shape checks in the flat convolution launcher equals the regular batched launcher, including kernel/config/CB construction. No torch or TTNN is imported by these checks. They do not establish physical TTNN reshape/slicing semantics, kernel compilation, native numerical accuracy, or performance.

The prototype still has a compact QKV channel slice and untilize, output tilize/slices, two small beta/g reshapes, and one necessary head-major readout flatten. Those costs are included in full-layer acceptance. Transient memory may change; measure final context/resource behavior before selection. If the regular batching kernel's per-time tap rereads dominate, adapt it to reuse tap/halo blocks while keeping this compact input contract; do not reject the compact family merely because a first kernel schedule loses.
