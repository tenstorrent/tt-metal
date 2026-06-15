# Comprehensive KERNEL-µs table — every run (Blackhole P150, pi05 3-stage)

DEVICE KERNEL DURATION (tracy col-20), min-of-5. tt-metal `e4500c1f`; op-support-count 20000.
`plain ttnn.linear` = TTNN default program config. `tuned ttnn.linear` = best-explicit native
config sweep. `CLAUDE optimized` = `MatmulDecodeLinear` with the adopted lever schemes
(`deep-plan_15`-patched binary). mmd no-patch cols = clean committed base. `—INV`/`—NOFIT` =
factory rejected on the committed op. `Tb` = best valid chunked-T (noted). All µs.

## All runs, per projection

| proj [M·K·N dtype] | plain ttnn.linear | tuned ttnn.linear | mmd FULL unch | mmd PART unch | mmd FULL ·Tb | mmd PART ·Tb | CLAUDE optimized |
|---|--:|--:|--:|--:|--:|--:|--:|
| SigLIP.qkv [256·1152·4608 bf16] | 55.0 | 55.44 | —INV | 66.9 | —INV | 83.7 (T64) | 45.97 |
| SigLIP.o   [256·1536·1152 bf16] | 47.7 | 20.43 | 33.1 | 27.4 | 35.5 (T64) | 32.5 (T64) | 38.58 |
| SigLIP.fc1 [256·1152·4320 bf8]  | 54.5 | 41.37 | —INV | 67.1 | —INV | 86.2 (T64) | 65.15 |
| SigLIP.fc2 [256·4320·1152 bf8]  | 173.0 | 42.09 | —NOFIT | —NOFIT | 129.4 (T64) | —NOFIT | 131.45 |
| VLM.qkv    [288·2048·2560 bf16] | 73.0 | 38.38 | 50.4 | —INV | 51.3 (T96) | 84.6 (T96) | 56.04 |
| VLM.o      [288·2048·2048 bf16] | 63.9 | 37.35 | 50.5 | —INV | 51.5 (T96) | 81.6 (T96) | 56.40 |
| VLM.gate   [288·2048·16384 bf16]| 259.4 | 254.61 | —NOFIT | —INV | —INV | 818.7 (T32) | 263.96 |
| VLM.up     [288·2048·16384 bf16]| 259.4 | 254.69 | —NOFIT | —INV | —INV | 819.0 (T32) | 264.00 |
| VLM.down   [288·16384·2048 bf8] | 415.7 | 236.72 | —INV | —INV | 415.5 (T96) | —INV | 405.76 |
| DENOISE.gate [64·1024·4096 bf16]| 24.7 | 34.64 | —INV | 23.2 | —INV | 29.7 (T32) | 9.90 |
| DENOISE.up   [64·1024·4096 bf16]| 24.7 | 34.63 | —INV | 23.2 | —INV | 29.7 (T32) | 9.83 |
| DENOISE.down [64·4096·1024 bf8] | 66.1 | 35.19 | 22.1 | 17.0 | 22.9 (T32) | 18.8 (T32) | 22.76 |

## Authoritative ratio — CLAUDE optimized ÷ tuned ttnn.linear

| proj | tuned ttnn.linear µs | CLAUDE optimized µs | ratio | verdict (ε=0.012) |
|---|--:|--:|--:|---|
| DENOISE.up   | 34.63 | 9.83 | 0.28 | **STRICT-BEAT** |
| DENOISE.gate | 34.64 | 9.90 | 0.29 | **STRICT-BEAT** |
| DENOISE.down | 35.19 | 22.76 | 0.65 | **STRICT-BEAT** |
| SigLIP.qkv   | 55.44 | 45.97 | 0.83 | **STRICT-BEAT** |
| VLM.gate     | 254.61 | 263.96 | 1.04 | LOSS |
| VLM.up       | 254.69 | 264.00 | 1.04 | LOSS |
| VLM.qkv      | 38.38 | 56.04 | 1.46 | LOSS |
| VLM.o        | 37.35 | 56.40 | 1.51 | LOSS |
| SigLIP.fc1   | 41.37 | 65.15 | 1.57 | LOSS |
| VLM.down     | 236.72 | 405.76 | 1.71 | LOSS |
| SigLIP.o     | 20.43 | 38.58 | 1.89 | LOSS |
| SigLIP.fc2   | 42.09 | 131.45 | 3.12 | LOSS |

**STRICT-BEAT 4/12** (DENOISE gate/up/down + SigLIP qkv) · **LOSS 8/12** (all other SigLIP/VLM prefill).

## Notes
- `tuned ttnn.linear` and `CLAUDE optimized` numbers were independently reproduced on the current
  binary within ±3% (native max |Δ| 0.56%, CLAUDE-optimized max |Δ| 1.20%; lever block-counts identical).
- `plain ttnn.linear` is 1.3–4× slower than `tuned ttnn.linear` on poorly-default-tiled shapes
  (SigLIP.o 47.7 vs 20.4; fc2 173.0 vs 42.1; VLM.qkv 73.0 vs 38.4) — it is NOT the baseline; the
  tuned column is.
- `no-patch mmd` (FULL/PART, un-tuned committed base) is below `CLAUDE optimized` on the rows whose
  numbers need the levers (DENOISE gate/up ×2.3; VLM gate/up ×3.1) and above it on a few
  (SigLIP.o, VLM.qkv/o) where the unchunked resident-FULL beats the adopted chunked scheme — neither
  changes the LOSS-vs-tuned-linear verdict on those prefill rows.
- The SigLIP.qkv win (0.83×) is the chunked-M=32 lever scheme; the no-patch structural cells do not
  reproduce it (unchunked-partial 66.9µs = ×1.21).
