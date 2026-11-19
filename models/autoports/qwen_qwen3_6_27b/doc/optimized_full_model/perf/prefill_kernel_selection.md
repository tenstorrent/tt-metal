# Prefill kernel selection by chunk size

Scope: batch 1 TP4 selected full-model generator, frozen decoder policy. This
is an inspection and existing-evidence inventory, not a new kernel sweep.
No TT device was opened. Times below are instrumented individual projection
kernel times, never TTFT or claims of global optimality.

## Logical requests and physical work

`tt/bounded_prefill.py:20,56,403` splits at 512-token boundaries and accepts
logical lengths 1..512 within each chunk. Thus requests 64,128,256,512 use one
chunk from start0; a logical 1024 request uses512+512. There is **no selected
single 1024-row batch 1 prefill projection**. Nonzero starts and nonaligned tails
use their own bounded intervals; logical lengths are not rounded at the API.
The nine measured projection rows below have M=128 for decoder work, except
the final-token LM head, whose logical M=1 is physically padded to 32.

`tt/multichip_decoder.py:526` selects fused matmul/reduce-scatter only when
`256 < rows <= 32768`. In particular, **256 remains the separate matmul +
reduce-scatter path**, while512 uses the fused family. Packed gate/up uses
`minimal_matmul` plus fused SwiGLU for 32 < rows <=512, config M2/K8/N26,
subblock1×2, grid 11×8 (`tt/multichip_decoder.py:704`). B1 retains the inherited
path; `tt/flat_prefill.py:14` permits flattened multirow routing only for B>1.

GDN packed projection retains its owner-safe joint primary/correction family;
its physical N7168 includes packing/padding, not a change to model width.
Full-attention packed QKVG is N3584. Runtime 2D geometry is derived from M,
selected grid and role N in `tt/optimized_decoder.py:499`; M=128 configs must
not be extrapolated as measured selections at other M.

## Measured stage7 chunk 128 projections

Source: `../tracy/reports/2026_11_14_06_17_56/ops_perf_results_2026_11_14_06_17_56.csv`,
`PERF_PREFILL`, rank0. Raw source line numbers and full config/kernel paths
are preserved in `runtime_matmul_policy.csv`. Inputs and outputs are DRAM
interleaved in these measured prefill rows; the primary input projection has
FP32 output, all others BF16. Activations are BF16 and weights BFP8, except
the BF16 head. These are raw observed active-core counts; an allowed grid can
contain more cores than the specific M×N execution uses.

| Role | Physical M×K×N (logical M if different) | Weight / fidelity | Raw cores | Kernel/config | Kernel us | Raw line |
| --- | --- | --- | ---: | --- | ---: | ---: |
| GDN packed primary/correction | 128[128]×5120[5120]×7168[7168] | BFLOAT8_B / HiFi4 | 32 | 2D grid8×8, Kblock8, Mcore1/Ncore28, subblock1×4 | 298.051 | 7498 |
| GDN output | 128[128]×1536[1536]×5120[5120] | BFLOAT8_B / HiFi4 | 44 | 2D grid 11×10, Kblock8, Mcore1/Ncore15, subblock1×1 | 56.167 | 7518 |
| GDN gate/up + SwiGLU | 128[128]×5120[5120]×8704[8704] | BFLOAT8_B / HiFi2 | 88 | minimal M2/K8/N26, subblock1×2, grid 11×8, fused SwiGLU | 264.152 | 7525 |
| GDN down | 128[128]×4352[4352]×5120[5120] | BFLOAT8_B / HiFi2 | 44 | 2D grid 11×10, Kblock8, Mcore1/Ncore15, subblock1×1 | 95.107 | 7526 |
| FA packed QKVG | 128[128]×5120[5120]×3584[3584] | BFLOAT8_B / LoFi | 44 | 2D grid 11×10, Kblock8, Mcore1/Ncore11, subblock1×1 | 81.823 | 7531 |
| FA output | 128[128]×1536[1536]×5120[5120] | BFLOAT8_B / LoFi | 44 | 2D grid 11×10, Kblock8, Mcore1/Ncore15, subblock1×1 | 39.493 | 7552 |
| FA gate/up + SwiGLU | 128[128]×5120[5120]×8704[8704] | BFLOAT8_B / LoFi | 88 | minimal M2/K8/N26, subblock1×2, grid 11×8, fused SwiGLU | 270.961 | 7557 |
| FA down | 128[128]×4352[4352]×5120[5120] | BFLOAT8_B / LoFi | 44 | 2D grid 11×10, Kblock8, Mcore1/Ncore15, subblock1×1 | 96.396 | 7558 |
| LM head | 32[1]×5120[5120]×62080[62080] | BFLOAT16 / HiFi4 | 110 | 1D grid 11×10, Kblock2, Mcore1/Ncore18, subblock1×3 | 1566.099 | 7569 |

The 2D and 1D compute source is
`ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp`;
the 1D head additionally reports `mcast_tail_reservation.cpp`. Minimal uses
`ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/compute_metal2.cpp`,
with `dm_in0_sender_metal2.cpp` and `dm_in1_sender_out_metal2.cpp`. Full dataflow
source paths and `program_config`/`config` strings are in the cited CSV.

## Per-chunk, per-role family inventory

`NM7` means **not measured in stage7; retained selected family; alternatives
not tried under recorded Stop Rule2**. `—` means no applicable measured
candidate at that size in this inventory, never a measured rejection.
The interleaved-copy column denotes an alternative weight/layout family:
selected2D and minimal rows already use their prepared interleaved weights;
no additional copy-only latency or alternate-copy win is invented.

`H5` denotes historical final-default stage5 B1/S512 profile kernel time,
not a stage7 full-path run or a current candidate comparison. Exact artifacts:
`../../optimized_multichip_decoder/perf/final29_profile_l0_b1_s512/accounting.json`
and `final29_profile_l3_b1_s512/accounting.json`. Those records preserve the
same physical M/K/N and selected fidelity per role. Their timing statistic is
merged per-device profile accounting, whereas stage7 above shows rank0; do
not subtract these two timing regimes as a before/after optimization result.

| Logical request / physical chunk | Role | 1D multicast candidate | 2D multicast | Minimal/SwiGLU | Fused matmul+RS | Interleaved-copy alternative | Retained choice and evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 64 /64 | GDN packed primary/correction | NM7 | NM7 | — | — | NM7 | 2D packed projection; no RS at this boundary; NM7 |
| 64 /64 | GDN output | NM7 | NM7 | — | — | NM7 | 2D + separate RS for row-parallel output; NM7 |
| 64 /64 | GDN gate/up + SwiGLU | NM7 | — | NM7 | — | NM7 | minimal fused SwiGLU; NM7 |
| 64 /64 | GDN down | NM7 | NM7 | — | — | NM7 | 2D + separate RS for row-parallel output; NM7 |
| 64 /64 | FA packed QKVG | NM7 | NM7 | — | — | NM7 | 2D packed projection; no RS at this boundary; NM7 |
| 64 /64 | FA output | NM7 | NM7 | — | — | NM7 | 2D + separate RS for row-parallel output; NM7 |
| 64 /64 | FA gate/up + SwiGLU | NM7 | — | NM7 | — | NM7 | minimal fused SwiGLU; NM7 |
| 64 /64 | FA down | NM7 | NM7 | — | — | NM7 | 2D + separate RS for row-parallel output; NM7 |
| 128 /128 | GDN packed primary/correction | NM7 | 298.051us (stage7) | — | — | NM7 | 2D packed projection; no RS at this boundary; measured row above |
| 128 /128 | GDN output | NM7 | 56.167us (stage7) | — | — | NM7 | 2D + separate RS for row-parallel output; measured row above |
| 128 /128 | GDN gate/up + SwiGLU | NM7 | — | 264.152us (stage7) | — | NM7 | minimal fused SwiGLU; measured row above |
| 128 /128 | GDN down | NM7 | 95.107us (stage7) | — | — | NM7 | 2D + separate RS for row-parallel output; measured row above |
| 128 /128 | FA packed QKVG | NM7 | 81.823us (stage7) | — | — | NM7 | 2D packed projection; no RS at this boundary; measured row above |
| 128 /128 | FA output | NM7 | 39.493us (stage7) | — | — | NM7 | 2D + separate RS for row-parallel output; measured row above |
| 128 /128 | FA gate/up + SwiGLU | NM7 | — | 270.961us (stage7) | — | NM7 | minimal fused SwiGLU; measured row above |
| 128 /128 | FA down | NM7 | 96.396us (stage7) | — | — | NM7 | 2D + separate RS for row-parallel output; measured row above |
| 256 /256 | GDN packed primary/correction | NM7 | NM7 | — | — | NM7 | 2D packed projection; no RS at this boundary; NM7 |
| 256 /256 | GDN output | NM7 | NM7 | — | — | NM7 | 2D + separate RS for row-parallel output; NM7 |
| 256 /256 | GDN gate/up + SwiGLU | NM7 | — | NM7 | — | NM7 | minimal fused SwiGLU; NM7 |
| 256 /256 | GDN down | NM7 | NM7 | — | — | NM7 | 2D + separate RS for row-parallel output; NM7 |
| 256 /256 | FA packed QKVG | NM7 | NM7 | — | — | NM7 | 2D packed projection; no RS at this boundary; NM7 |
| 256 /256 | FA output | NM7 | NM7 | — | — | NM7 | 2D + separate RS for row-parallel output; NM7 |
| 256 /256 | FA gate/up + SwiGLU | NM7 | — | NM7 | — | NM7 | minimal fused SwiGLU; NM7 |
| 256 /256 | FA down | NM7 | NM7 | — | — | NM7 | 2D + separate RS for row-parallel output; NM7 |
| 512 /512 (largest) | GDN packed primary/correction | NM7 | 613.991us H5 | — | — | NM7 | 2D packed projection; no RS at this boundary; H5 component evidence only |
| 512 /512 (largest) | GDN output | NM7 | — | — | 123.329us H5 | NM7 | fused matmul+RS; H5 component evidence only |
| 512 /512 (largest) | GDN gate/up + SwiGLU | NM7 | — | 288.540us H5 | — | NM7 | minimal fused SwiGLU; H5 component evidence only |
| 512 /512 (largest) | GDN down | NM7 | — | — | 184.883us H5 | NM7 | fused matmul+RS; H5 component evidence only |
| 512 /512 (largest) | FA packed QKVG | NM7 | 99.384us H5 | — | — | NM7 | 2D packed projection; no RS at this boundary; H5 component evidence only |
| 512 /512 (largest) | FA output | NM7 | — | — | 111.090us H5 | NM7 | fused matmul+RS; H5 component evidence only |
| 512 /512 (largest) | FA gate/up + SwiGLU | NM7 | — | 291.953us H5 | — | NM7 | minimal fused SwiGLU; H5 component evidence only |
| 512 /512 (largest) | FA down | NM7 | — | — | 180.482us H5 | NM7 | fused matmul+RS; H5 component evidence only |
| 1024 /512+512 | GDN packed primary/correction | NM7 | NM7;512 family only | — | — | NM7 | 2D packed projection; no RS at this boundary; NM7 |
| 1024 /512+512 | GDN output | NM7 | — | — | NM7;512 family only | NM7 | fused matmul+RS; NM7 |
| 1024 /512+512 | GDN gate/up + SwiGLU | NM7 | — | NM7;512 family only | — | NM7 | minimal fused SwiGLU; NM7 |
| 1024 /512+512 | GDN down | NM7 | — | — | NM7;512 family only | NM7 | fused matmul+RS; NM7 |
| 1024 /512+512 | FA packed QKVG | NM7 | NM7;512 family only | — | — | NM7 | 2D packed projection; no RS at this boundary; NM7 |
| 1024 /512+512 | FA output | NM7 | — | — | NM7;512 family only | NM7 | fused matmul+RS; NM7 |
| 1024 /512+512 | FA gate/up + SwiGLU | NM7 | — | NM7;512 family only | — | NM7 | minimal fused SwiGLU; NM7 |
| 1024 /512+512 | FA down | NM7 | — | — | NM7;512 family only | NM7 | fused matmul+RS; NM7 |

For 1024, no operation time is asserted: two 512 projection invocations cannot
be assigned a measured summed time from the historical 512 row. Prefix/cache
state, chunk orchestration and whole64 dispatch differ. The historical 512
MMRS numbers include reduction/communication, unlike the 128 separate matmul
rows; a numerical comparison must add their matching separate collectives.

The selected final-token head is a distinct terminal role, not an M=chunk
projection. For each request64/128/256/512/1024, it runs once after the last
chunk with logical M=1, padded M=32, K=5120/N=62080, BF16/HiFi4 and the 1D config
above. Only 128 has a current stage7 op measurement: 1566.099us rank0. Other
request-specific head timings are NM7. Earlier unchanged-policy adapted
DRAM-sharded split2/whole3 candidates lose complete terminal timings in
`../../full_model/perf_terminal/lm_head_decision.json` and
`../../full_model/lm_head_uninstrumented.json`; these terminal candidates are
not alternative large-M prefill kernels. No decode-only DRAM-sharded program
is proposed for M64–512.

## Material non-matmul work and candidate treatment

Stage7 chunk 128 rank0 measurements, same raw CSV window:

| Boundary | Observed kernel us / raw line | Treatment |
| --- | --- | --- |
| GDN first distributed norm post-gather + hidden gather |36.433 /7496;34.952 /7497 | Retain selected norm/collective contract. Prior fused-norm family has resource and whole-layer tradeoffs, not a new chunk 128 win. |
| GDN causal convolution |22.289 /7501 | Retain selected causal state semantics; joint combine/conv candidates previously regressed complete boundary. |
| GDN chunk prep/scan |31.884 /7509;28.823 /7510 | Temporal recurrent work remains necessary; no stage7 replacement timing. |
| GDN separate output/down reduce-scatter |32.547 /7519;33.265 /7527 | Included in complete prefill cost; cannot compare 128 matmul alone to512 fused MMRS. |
| FA norm (input/post-attention) |79.376 /7530;79.536 /7556 | Material short-prefill cost; retained current implementation. Shape-specific replacement NM7 under Stop Rule2. |
| Chunked SDPA |73.850 /7549 | Preserve BFP8 cache, padded read window and logical position semantics. |
| FA separate output/down reduce-scatter |37.979 /7553;40.276 /7559 | Selected at rows<=256; fused family enters strictly above 256. |
| Logit FillPad/Copy |24.245 /7570;21.435 /7571 | Persistent sampler input staging; removal unmeasured and must preserve buffer identity. |
| Local TopK large-indices |278.516 /7837 | Common physical top 32/semantic greedy path; no full-vocab gather or generic TopK substitution. |

Historical evidence is mapped in
`../../optimized_multichip_decoder/final29_candidate_decisions.md`:
`advice5_*.json` for L1/minimal family comparisons,
`mmrs2_summary.csv` and `family3_summary.csv` for output projection topology,
`prefill_chunks7*.json` and `prefill_alignment8*.json` for chunk/layout behavior,
`prefill_qkv26_perf.json` and `qkv_movement28_audit.md` for extraction/movement
regressions, and `final29_handoff.md` for capped untried adaptations. These
support retaining the integrated family; they do not select every M64/256
alternative or establish stage7 optimality. This inventory leaves those
unmeasured family/size crosses explicit for the stage owner's stop-rule record.
