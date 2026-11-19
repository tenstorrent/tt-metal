# Packed-output SwiGLU epilogue prototype

Added [packed_swiglu_candidate.py](../../tt/packed_swiglu_candidate.py) and [probe_packed_swiglu.py](../../tests/probe_packed_swiglu.py). Neither is wired into the decoder. No device operation or native compilation was performed by this CPU task. The [CPU artifact](packed_swiglu_cpu.json) records350 tile-address/worker-coverage cases and runtime-source checks; this is not device equivalence or performance evidence.

The candidate consumes the **existing contiguous** BF16 packed gate/up output `[1,B,8704]` and emits BF16 `[1,B,4352]`. It retains exactly the same projection weights, packing and matmul. Its reader fetches gate tile `row*input_stride+col` and up tile `row*input_stride+136+col`; one compute kernel evaluates native SiLU then native multiply and one writer emits tile `row*136+col`. There are no TTNN slices, layout conversions or intermediate device tensors in the candidate. Inputs must already be interleaved DRAM BF16 TILE. Unsupported boundaries fail explicitly instead of choosing a fallback.

## Source graph and native alternatives

| Current operation | GDN / round_silu=True | Full attention / round_silu=False | Movement |
|---|---|---|---|
| Gate/up projection | Existing `_linear` | Existing `_linear` | Existing sharded matmul followed by DRAM result/crop |
| Two halves | Two4352-wide slices | Two4352-wide slices | Read packed input, write separate gate/up tensors |
| Gate SiLU | Standalone `ttnn.silu` | Input activation inside multiply | GDN writes/reads a BF16 intermediate; full path uses a local BF16 CB |
| Gate×up | `ttnn.multiply` | Same binary op with input activation | Two inputs read, one BF16 result written |
| Down projection | Existing `_linear` | Existing `_linear` | Unchanged consumer |

Repository search found `ttnn.swiglu` is a split/SiLU/multiply composite, and activates its **second** half (`up * silu(second)`), whereas this model requires `silu(first gate) * second up`. It is not a dedicated replacement for this unchanged packed layout. The minimal-matmul fused SwiGLU changes the projection path and uses paired gate/up tiles; that family has already been measured by the parent. DeepSeek/GPT-OSS/Kimi fused FFN kernels own different matmuls, routing, clamps or scaled activation contracts. This prototype therefore fuses just the packed result epilogue and uses the existing native arithmetic APIs.

## Rounding finding: both BF16 source chains round SiLU

The Python `round_silu` flag names different graphs, but it does **not** imply an unrounded FP32 SiLU in the full-attention BF16 path on this checkout:

- `unary/unary.cpp` chooses BF16 DST for BF16 `silu` input/output.
- Blackhole `ckernel_sfpu_silu.h::calculate_silu<false>` uses the native BF16 sigmoid and explicitly converts the SiLU result to BF16 using nearest rounding.
- Default binary multiply selects the SFPU implementation (`fast_and_approximate_mode.value_or(false)`); BF16 inputs/output leave FP32 destination accumulation disabled.
- The multiply input-activation path uses that same SiLU and packs it into a **BF16 intermediate CB** before multiplication (`binary_ng_program_factory.cpp`, `eltwise_utils_sfpu.hpp`).
- Native `calculate_sfpu_binary_mul<...,false>` performs its software BF16 RNE and native zero handling. A handwritten `gate*sigmoid(gate)*up`, a different sigmoid approximation, or an FP32-DST multiplication is not an equivalent replacement.

The candidate calls `silu_tile<false>(0)` followed by `mul_binary_tile<false>(0,1,0)`, with `fp32_dest_acc_en=False` and precise math. It does not implement its own approximation. `round_silu` is an explicit required bool identifying the source chain for evidence; both values use the same BF16 native math on the inspected tree.

Two implementations are available:

- `register`: keeps the already-BF16-rounded SiLU in DST, then calls native multiplication. It removes the idempotent BF16 intermediate pack/reload.
- `roundtrip`: packs SiLU into a local BF16 CB and reloads it before native multiplication, reproducing the source intermediate-storage boundary as a diagnostic control. It still performs one device dispatch and eliminates both global slices.

If register differs from the unchanged chain, inspect bit mismatches against roundtrip before rejecting the family. The device probe requires **bitwise equality**, including signed-zero bits, for both graph contracts. The native source files and helper/probe are hashed in the artifact so changed upstream arithmetic is visible.

## Shape, ownership and capacity

The reader uses the tensor's actual padded input width for row stride and the fixed mathematical4352-column half boundary. CPU checks cover all B1..32, extra compact row counts33/65/513, physical input widths8704/9216, and worker counts1/8/32/64/110. Each output tile has exactly one worker and both input halves remain within the intended row. Multiple M tiles work in the mapping; only decode is proposed for initial parent integration.

The function accepts optional `output=` owned by the caller. It validates shape, BF16/TILE/interleaved-DRAM placement, canonical padding, device, and non-aliasing with the input. It retains no model, tensor, closure or mutable cache. With no output supplied it allocates one output, as the current elementwise chain does. Persistent output must be separately owned by each executing layer/signature, and excluded from input aliasing; sharing an immutable helper does not authorize sharing mutable output buffers between resource clones.

For B1..32, H=136 BF16 tiles=278,528 bytes/device:

| Source-requested DRAM payload | Bytes/device |
|---|---:|
| Candidate: read both packed halves + write output | 835,584 |
| Current GDN: slices + standalone SiLU + multiply | 2,506,752 |
| Current full: slices + multiply with local input activation | 1,949,696 |
| GDN payload reduction | 1,671,168 |
| Full payload reduction | 1,114,112 |

These are graph-level logical tile payloads, not bus transaction measurements. They assume the inspected DRAM boundaries and count no weight traffic. The mathematical minimum remains reading the two operands and writing the result; redundant old movement must not inflate the roofline objective. Wider M scales these numbers by `ceil(rows/32)`. Register CBs use12,288 bytes/worker and roundtrip16,384 bytes/worker, all L1. Optional persistent output adds H per decode signature/device. Weight/cache/state capacity and the context contract are unchanged.

## Probe behavior and commands

The device probe compares literal unchanged TTNN source chains, on every mesh rank, against both candidate implementations. Inputs contain different random values per rank plus signed zeros, small magnitudes and large finite gate/up values. It checks eager output and original/changed-input trace replays using raw BF16 bits, reports PCC/max error for diagnosis, and verifies the captured function returns the supplied persistent output object. Baseline and candidate both capture their actual graphs. Timing uses100 asynchronous trace replays with synchronization at endpoints, three alternating-order rounds; profiler/watcher/tracking environments mark timing ineligible. Results remain pointwise-only and provisional.

The probe patches TTNN host upload/readback APIs during candidate/reference calls, trace capture and replay, with a positive control that proves rejection. Data preparation/readback occurs outside the guard. The candidate's separate CPU AST audit finds no Torch call, host tensor conversion or implicit layout fallback. This bounded guard does not replace the stage's reachable runtime audit and authoritative whole-decoder sentinel.

CPU-only rerun:

```bash
python -m models.autoports.qwen_qwen3_6_27b.tests.probe_packed_swiglu --cpu-check \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/packed_swiglu_cpu.json
```

Parent-serialized short device probe, through the existing command/provenance wrapper and environment:

```bash
D=models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder
bash "$D/run_job.sh" packed_swiglu_smoke \
  python -m models.autoports.qwen_qwen3_6_27b.tests.probe_packed_swiglu \
  --device --chips 4 --rows 1,32 --implementation both --replays 4 --repeats 100 \
  --output "$D/packed_swiglu_smoke.json"
```

After a legal matching smoke, use `--rows all` for B1..32; optional multiple-M checks use `--rows 33,65,513`. `--cores` permits worker-count adaptation without changing math or layout. Watcher and allocation-tracking probes use separate names and are excluded from timing conclusions. Do not interpret the absence of a device artifact as a pass.

After device equivalence and speed justify it, the parent can replace only the packed-MLP epilogue inside `_finish` with `packed_swiglu(packed,round_silu=self.policy['round_silu'])`, leaving the minimal-matmul prefill branch and separate-weight branch untouched. Full-decoder contribution/state PCC, changed-input tracing, final source/profiler evidence and owner isolation still apply. No such integration is included here.
