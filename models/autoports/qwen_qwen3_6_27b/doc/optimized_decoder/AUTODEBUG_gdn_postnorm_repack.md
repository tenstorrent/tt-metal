# GDN post-RMSNorm layout and gate candidate

Source-only AutoFix investigation over frozen `candidates/delivery118.py`. Main decoder and recurrence kernels were not edited; no hardware was run here. The new helper requires an isolated same-input device probe before model/performance selection.

## Attribution from the measured path

`perf/final118_profile_profile_l0_b32_s33/ops.csv.gz` identifies global call1324032 as **BF16 `[1,1536,128]` → BF16 `[1,32,6144]`**, after native RMSNorm and its standalone affine multiply. It is not the recurrence's initial metadata reshape.

`decode_perf_report.csv`:

| Global call | Operation | Device time |
|---:|---|---:|
| 1319936 | recurrence generic op | 532.153 us |
| 1320960 | FP32 recurrence output → BF16 | 4.278 us |
| 1321984 | native RMSNorm, no affine | 4.976 us |
| 1323008 | BF16 norm-weight multiply | 5.694 us |
| 1324032 | head-major → flat tiled reshape | 29.444 us |
| 1325056 | packed Z gate slice | 2.559 us |
| 1326080 | gate → FP32 | 3.944 us |
| 1327104 | weighted norm → FP32 | 4.173 us |
| 1328128 | FP32 SiLU gate/multiply → BF16 | 9.517 us |

The six operations from affine multiply through gated output total **55.331 us**. This is the candidate's replacement budget, not a claimed saving. Recurrence, its BF16 boundary, and native RMSNorm remain unchanged.

A plain view cannot remove this reshape: head-major tiles contain32 heads ×32 channels, whereas flat tiles contain32 users ×32 channels of one head. Reinterpreting the same tiled buffer changes user/head indexing. Making recurrence write flat output and calling ordinary RMSNorm on6144 channels would also change the norm groups; reshaping it back to128-wide head rows would reintroduce the movement. Fusing the post-norm pointwise work with the required conversion avoids changing reduction arithmetic.

## New immutable artifacts

- `candidates/gdn_affine_gate_flat.py`: one generic op taking normalized BF16 `[1,B*48,128]`, real BF16 gamma `[1,1,128]`, and the packed BF16 projection `[1,B,N]`; returns flat BF16 `[1,B,6144]`.
- `candidates/gdn_affine_gate_decode.py`: frozen delivery118 plus an opt-in `gdn_fused_affine_gate:true` branch after native no-affine RMSNorm.
- `candidates/make_gdn_affine_gate_decode.py` and `.generation.json`: deterministic generator, input/output hashes; refuses to replace an output.
- `tests/run_gdn_affine_gate_probe.py`: real-gamma, same-input TTNN comparison, authored only.
- `tests/test_gdn_affine_gate_mapping.py` and `gdn_affine_gate_cpu_checks.json`: executed CPU mapping/source checks.

The decoder branch requires explicit decode phase plus the existing `hf_gated_norm` path. It is disabled by default. Prefill, including compact S1 tails, remains byte-for-byte on the existing branch. Projection selection, recurrence/state ownership and all full-attention paths are unchanged. `gdn_affine_gate_cores` optionally caps helper workers; the default uses up to110 available cores.

## Layout and arithmetic

The reader partitions192 output column tiles. For tile t, head=`t//4` and channel tile=`t%4`. User u's normalized source row is `u*48+head`; its tile is `((u*48+head)//32)*4+(t%4)`. The reader gathers the corresponding BF16 face rows into a flat user tile and broadcasts the appropriate real gamma vector. Each32-byte BF16 face row is obtained from a64-byte-aligned pair, then the correct half is selected. It reads the gate tile directly from packed projection tile`320+t`, avoiding the gate slice.

The compute kernel preserves these boundaries:

1. BF16 normalized value × BF16 gamma in FP32, then explicit BF16 round-to-nearest-even.
2. BF16 gate converted exactly on unpack to FP32, native `silu_tile<true>`, FP32 multiplication with the rounded affine value, then explicit BF16 round-to-nearest-even.

Native RMSNorm's reduction tree and BF16 output are unchanged. Native FP32 SiLU dispatch is documented in `tt_metal/hw/inc/api/compute/compute_kernel_api.h:194`; Blackhole's FP32 calculation is in `ckernel_sfpu_silu.h:14`. The existing binary factory enables FP32 destination and direct Float32 unpack when either operand is FP32 (`binary_ng_program_factory.cpp:1138,1383-1399`), matching the source path's gate precision. These source facts support the arithmetic hypothesis; they do **not** replace the strict same-input device comparison.

The writer emits complete2048-byte BF16 tiles directly in the target flat layout. Padded users are explicitly zeroed, including poisoned padding in CPU tests. There is no tilize/untilize, reshard, host conversion or recurrence-state access in the helper. Scratch and CB storage total16KiB per worker; no extra resident weights are introduced. Output allocation matches the existing flat BF16 physical shape `(1,32,6144)` for all B1..32. Optional supplied output supports stable buffer reuse and trace capture.

This trades the baseline's whole-tensor reshape and intermediate writes for many small normalized-row reads. The source budget is favorable, but NoC/read overhead and SFPU execution must be measured; no latency claim is made.

## Validation and parent commands

Executed CPU checks passed64 layout cases: every B1..32 at packed widths16512 and17408. They verify every normalized value, gamma broadcast, packed gate position and deterministic zero padding. Five worker partitions cover output tiles exactly once. Source tests verify only `_gdn_compact` changed, the disabled/prefill branch restores delivery118's exact AST, and both BF16 rounding boundaries are explicit. Python compilation passed.

The isolated runner loads only pinned real `linear_attn.norm.weight`, prepares a shared native RMSNorm output, and compares the helper against the actual TTNN chain. It covers B1/B3/B31/B32, both packed widths, two random draws with saturated/near-zero gate cases, repeated execution and a reused output across changing inputs. The probe intentionally requires **bit equality**, rather than accepting aggregate PCC; failure must be localized before model selection.

Run under the existing serial device wrapper:

```bash
python -m models.autoports.qwen_qwen3_6_27b.tests.run_gdn_affine_gate_probe \
  --candidate models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/gdn_affine_gate_flat.py \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/gdn_affine_gate_probe.json
```

After that probe passes, compare the immutable decoder candidate with policy `{"gdn_fused_affine_gate":true}` against delivery118 using real-weight B1/B32 correctness, the unchanged512-step traced sequence gate, watcher and end-to-end traced latency. Record the replacement generic-op row and verify removal of the six original rows. A first API/compiler failure is not rejection evidence; the concrete mapping, dtype boundary or CB failure should be diagnosed and repaired in a new frozen version.
