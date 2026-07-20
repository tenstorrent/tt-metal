# Regime-A single-chip fusion — implementation report

`ttnn.experimental.regime_a_matmul` (+ new `regime_a_matmul_split`) now supports optional epilogue
fusions on a single Blackhole chip, preserving the no-fusion Regime-A architecture (Picker v3, PARETO
ring, IN1_NEAR placement, DRAM-sharded in1, pipelined reduction drain, balanced tails, in0/in1 overlap).
Branch `cglagovich/regime-a-bh`.

## 1. What landed
API (all fusion args optional; nullopt/1 => byte-identical no-fusion path):
- `regime_a_matmul(in0, in1, config=None, *, bias_tensor, fused_activation, fused_ternary_scalar,
  fused_ternary_input_a, fused_ternary_input_b, memory_config, dtype, compute_kernel_config)`.
- `regime_a_matmul_split(in0, in1, chunks, dim=-1, config=None, *, <same fusion kwargs>)` -> list of
  `chunks` equal-width [M, N/chunks] tensors, written directly (no materialize+slice). dim==-1,
  N%chunks==0, N/chunks tile-aligned.
- bias `Y=A@B+bias`; unary activation `Y=act(A@B+bias)` (UnaryWithParam; bias first);
  addcmul `Y=residual+scalar*(A@B+bias)*gate` (gate [1,N] bcast or [M,N] full, incl fp32-gate
  workaround). activation and addcmul are mutually exclusive (validated).

Kernels:
- compute.cpp: split-K is now fusion-aware — the epilogue is applied EXACTLY ONCE at the reduction
  ROOT (is_top; Pk==1 => every core). Non-root bands forward RAW partials (new copy_block_raw; and
  reduce_add_in_place sums the forwarded partial into the fp32 intermediate before the root epilogue).
  Closes the old REDUCE_K "K-par never fuses" gap.
- in0_ring_reduce_writer.cpp: on fusing (root) cores only, reads bias/residual/gate into CBs c_4/c_5/c_6
  (global-(m,n) indexed, tail-zeroed) for compute, and routes output tiles to the right chunk buffer.
  All new behavior gated by defines (FUSE_BIAS/FUSE_TERNARY/TERNARY_B_IS_FLOAT32/OUT_CHUNKS) so the
  mask-0 compile is byte-identical.

Host/device op:
- Device op returns vector<Tensor> (chunk 0 == single output; public regime_a_matmul extracts [0]).
- Fusion presence + activation op/params + scalar + chunks feed the reflection program-cache hash
  (optional-tensor presence in tensor_args is hashed), so a fused program never aliases a no-fusion one.
  override_runtime_arguments refreshes in0/in1/out-chunk + bias/residual/gate addresses on cache replay.

## 2. Correctness matrix (random BF16 vs CPU FP32, fresh + cached program; PCC>=0.999) — ALL PASS
tests/ttnn/unit_tests/operations/matmul/test_regime_a_matmul.py

| combination | Pk=1 | Pk>1 | notes |
|---|---|---|---|
| no-fusion (regression) | PASS | PASS | 8 pre-existing cases, PCC unchanged |
| bias | PASS | PASS | |
| activation (relu, gelu) | PASS | PASS | |
| bias + activation | PASS | PASS | |
| addcmul, broadcast gate, scalars 1.0/0.5/2.5 | PASS | PASS | |
| addcmul + bias | PASS | PASS | |
| addcmul, full [M,N] gate (M=64) | PASS | PASS | |
| addcmul, fp32 gate | PASS | PASS | fp32-gate workaround |
| Ns>1 + bias | PASS | - | 32x2048x2048 Ns2 |
| Sm>1 + addcmul | - | PASS | 128x6144x4608 Sm2 (fuse at root under M-split) |
| W>1 deep-K + bias+act | - | PASS | 32x15360x3072 W5 |
| chunking 2/3: none/bias/bias+act/addcmul | PASS | PASS | direct chunk writes |
| balanced tails + bias (6 non-divisible shapes) | auto | auto | Kt/Nt/Mt/sub-tile tails, finite-checked |
| config=None + auto Pk>1 + bias/act/addcmul | - | PASS | code-review regression: split-K auto-select IS fusion-aware |
| cache replay (bias+addcmul) fresh buffers | PASS | PASS | different buffer addresses |
| validation (act+addcmul, bad bias/residual/gate, dim!=-1, N%chunks!=0) | raises | raises | clear TT_FATAL |

## 3. Perf A/B (same config vs mask-0; median device-kernel us, delivered GB/s, per-RISC critical span)
per-RISC = BRISC/NCRISC (the two DM RISCs: in1 reader + in0-ring/reduce/output writer, split across NoCs)
/ TRISC (compute). All three overlap ~= total => in0/in1 overlap preserved under fusion.

### 256x2048x1024  (Sm>1, W=1; cfg [1,4,2,2,4], 64 cores)
| variant | us | GB/s | overhead | PCC | BRISC/NCRISC/TRISC us |
|---|---|---|---|---|---|
| none | 22.59 | 255.2 |  | 0.99999 | 22.6/20.9/21.4 |
| bias | 23.00 | 250.7 | +1.8% | 0.99999 | 23.0/20.5/21.9 |
| act | 25.64 | 224.9 | +13.5% | 0.99999 | 25.6/20.3/24.3 |
| addcmul | 25.80 | 223.6 | +14.2% | 0.99999 | 25.8/20.4/24.6 |

### 256x6144x768  (Pk>1, Sm>1, W=1; cfg [1,6,2,4,2], 96 cores)
| variant | us | GB/s | overhead | PCC | BRISC/NCRISC/TRISC us |
|---|---|---|---|---|---|
| none | 45.48 | 285.3 |  | 0.99999 | 45.5/44.6/44.9 |
| bias | 46.81 | 277.2 | +2.9% | 0.99999 | 46.8/44.7/46.3 |
| act | 49.71 | 261.0 | +9.3% | 0.99999 | 49.7/45.6/49.2 |
| addcmul | 51.62 | 251.4 | +13.5% | 0.99999 | 51.6/44.7/51.1 |

### 256x15360x768  (W>1 deep-K; cfg [1,6,2,2,3], 96 cores)
| variant | us | GB/s | overhead | PCC | BRISC/NCRISC/TRISC us |
|---|---|---|---|---|---|
| none | 95.62 | 333.1 |  | 0.99999 | 95.6/94.2/94.7 |
| bias | 95.96 | 331.9 | +0.4% | 0.99999 | 96.0/93.9/95.0 |
| act | 97.96 | 325.1 | +2.5% | 0.99999 | 98.0/94.0/97.1 |
| addcmul | 98.87 | 322.1 | +3.4% | 0.99999 | 98.9/94.6/97.9 |

### 32x2304x6144  (wide-N control; cfg [2,3,1,1,6], 48 cores)
| variant | us | GB/s | overhead | PCC | BRISC/NCRISC/TRISC us |
|---|---|---|---|---|---|
| none | 59.77 | 482.7 |  | 0.99999 | 59.7/59.7/59.1 |
| bias | 60.97 | 473.2 | +2.0% | 0.99999 | 60.9/60.9/60.4 |
| act | 61.10 | 472.2 | +2.2% | 0.99999 | 61.1/60.9/60.5 |
| addcmul | 62.52 | 461.5 | +4.6% | 0.99999 | 62.2/62.5/61.9 |

### 32x6144x3072  (Pk=1, no reduction; cfg [1,1,1,4,6], 8 cores)
| variant | us | GB/s | overhead | PCC | BRISC/NCRISC/TRISC us |
|---|---|---|---|---|---|
| none | 106.61 | 359.6 |  | 0.99999 | 104.8/106.6/105.9 |
| bias | 106.96 | 358.4 | +0.3% | 0.99999 | 105.2/107.0/106.3 |
| act | 107.62 | 356.2 | +1.0% | 0.99999 | 104.9/107.6/107.1 |
| addcmul | 108.32 | 354.0 | +1.6% | 0.99999 | 105.5/108.3/107.7 |

### Unfused regression (<=~1% and bit-identical)
The mask-0 (no-fusion, chunks==1) path is bit-identical by construction: every fused code path in
compute/writer is gated behind a define unset when no fusion is requested, no fused CBs (c_4/5/6) are
created, and no extra compute/writer runtime args are appended — so the compiled programs, CB layout and
runtime-arg vectors are identical to the pre-change no-fusion program. The 8 pre-existing no-fusion tests
pass with unchanged PCC. The "none" rows are the reference for the fusion overhead. (The added
RegimeAMatmulParams fields change only the cache KEY, never the compiled program, so no runtime effect.)

### Fusion overhead summary
- bias: +0.3% .. +2.9% (one extra [1,N] read + a bcast-add).
- activation: +0.9% .. +13.5% (SFPU per output tile; largest on the smallest/fastest shape).
- addcmul: +1.6% .. +14.2% (residual [M,N] read ~= 1x output size + gate + mul/mul-scalar/add). The
  residual read is inherent to the op (an input, not a re-read of our own output) — NO extra output DRAM
  round-trip; the epilogue post-processes the existing intermediate/out CBs.
- Overhead is largest on compute-light shapes (256x2048x1024, ~22us) and negligible on deep-K/large
  shapes. Overlap + RISC ownership intact: all three RISCs remain active and overlapping; extra work lands
  on the writer-side DM RISC + compute.

## 4. Unsupported combinations / planner constraints
- activation + addcmul together: rejected (matches minimal_matmul / dit addcmul).
- addcmul residual must be BFLOAT16 (shares in1's bf16 tile format / CB c_5); gate may be BF16 or FLOAT32.
- Full [M,N] gate vs [1,N] broadcast is decided from the gate's LOGICAL M (==1 => broadcast, else full),
  matching validate(). (Fix: previously decided by padded tile-row count, which silently broadcast a
  genuine per-row gate whenever M<=32 padded to a single tile row — see test_regime_a_fused_addcmul_full_gate
  M=32.) Any M>=1 per-row gate now applies correctly.
- NO planner/picker constraint added and Pk was NOT forced: every fusion coexists with the picker's chosen
  Pk/Ns/Sm/kb/nsb (Pk=1 & Pk>1, Ns>1, Sm>1, W=1 & W>1 all validated). Picker NOT retuned on fused timings.
- Chunking composes with all fusions; requires N%chunks==0 + N/chunks tile-aligned, dim=-1 only.
- Single-chip only; multi-chip in0 all-gather deferred (unchanged this turn).

## 5. Watcher status
Fused subset (addcmul Pk=1/Pk>1, Sm>1 addcmul, W>1 bias+act, bias+act chunking) under TT_METAL_WATCHER=1:
CLEAN — no watcher errors / pending-transaction / hang / assert dumps; only normal periodic checks +
"Dump completed". Writer fused reads use a read barrier before CB push; no new non-posted atomics on the
fused path.

## Raw data
tools/mm_sweep/regime_a_fusion.json (this matrix); regenerate with tools/mm_sweep/regime_a_fusion_perf.py.
