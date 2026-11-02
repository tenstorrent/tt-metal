# a3: exact-ternary decode engine (PQ2_0, single chip)

`BONSAI_NATIVE=pq2_0` runs every ternary projection of Ternary Bonsai 2 27B at decode straight from
the PQ2_0 GGUF: 2-bit codes as `BFLOAT2_B` tiles plus the fp16 per-128-group scales (held as fp32),
applied inside the matmul. Nothing is requantized. With BONSAI_SCALE16 (the pq2_0 default) the
scale tiles are fp16 (the GGUF's own type; a bf16-declared CB carries them and TRISC unpacks them as Float16). `BONSAI_NATIVE_SITES` picks sites (default: all
of `mlp, gdn_in, gdn_out, attn_qkv, attn_o, lm_head`); decode only (M = 1). Defaults: 256 B code
tiles with a forced exponent (`BONSAI_A2B=0` for 320 B Bfp2_b) and the 32 A3 workers as two
rectangles (`BONSAI_A3_RECT=0` for the scattered set).

## Data path per projection

1. **prep** (`kernels/prep_hadamard_kernel.cpp`, ~10-40 cores): x' = H(D x) per 1024-block as
   x_b @ M_b with M_b = diag(d_b) H/32 in bfp2 (exact), M laid out so each work unit is one
   contiguous DRAM run. Each output K-tile k is written as a whole 8x32 tile to every A3 worker:
   live row (k/4) % 8, zeros elsewhere (the block-masked input).
2. **A3** (`kernels/ternary_mm_v2_kernel.cpp`, 32 workers = 4 per DRAM bank): deepseek's DRAM
   streaming skeleton (custom_mm, LoFi, fp32 DST). One custom_mm per super-group of 8 groups
   (32 K-tiles) leaves 8 exact group partials in DST rows 0-7; SFPU does ACC += P * S_G; at the end
   `sfpu_reduce` sums the rows. Scales for one output tile are unpacked to DST once (fp32).
   GU mode streams (gate, up) column pairs and writes silu(gate) * up.
3. Output: interleaved, 32x32 tiles with row 0 live (what the model expects) or 1x32 pages.

Host side: `op.weights_for_device` (bank shuffle + `scale_rows`, cached as `.tensorbin`, written
once, never rewritten), `linear.TernaryLinear` (one projection), `mlp.TernaryMLP` (fused SwiGLU),
`linear.gdn_in_rows` / `attn_qkv_rows` (rows in the order the dense model uses, from the converter's
`bonsai_conversion_report.json`: vperm_rows for GDN qkv/gate, as_is elsewhere).

## Numbers (chip 1, six-flag stack, test_decode_e2e, 5-run medians unless marked dev)

| config | ms/token | top-1/top-5 scored |
|---|---|---|
| six-flag control | 79.00 | 93.55 / 100.00 |
| + exact-ternary MLP | 59.01 | 95.31 / 100.00 |
| + GDN in/out (vs MLP-only 58.24, same session) | 49.43 | 95.70 / 100.00 |
| all sites (counted, vs control 79.09) | 44.25 | 95.51 / 100.00 |
| + A2b 256 B code tiles + rectangular workers (vs 44.25 same session) | 39.40 | 95.51 / 100.00 |
| + shared scale tails + B's GDN RECT/HMI (counted 18:33) | 38.23 | 95.51 / 100.00 |
| + B's GDN CONV_HM/TRIM/HMO (3-arm count 19:13, vs 38.32 same session) | 35.85 | 96.09 / 100.00 |
| + norm sites ffnorm, attnnorm, lmnorm (rmsprep; same count) | 34.97 | 96.09 / 100.00 |
| + glue ops removed at the GDN/attention sites (count 20:17, vs 34.76 same session) | 33.92 | 96.09 / 100.00 |
| + B's attention flags ATTN_LEAN/QGVIEW/TAIL (count 22:04, vs 33.92 same session) | 32.74 | 96.09 / 100.00 |
| + fp16 scale tiles, BONSAI_SCALE16 (count 22:26, vs 32.75 same session) -- current default | 31.94 | 96.09 / 100.00 |

Against the six-flag control: 2.48x (12.6 -> 31.3 tok/s). Q2_0 runs number for number as PQ2_0
(bit-identical codes, deduplicated scales; the cost of finer groups was not measured).

Op level (device us, real weights, PCC >= 0.99999 vs float64 cpu_ref): MLP 240 per layer (dense
~575), GDN [qkv|z] 79 (dense 259), ssm_out / attn_output 38 (91 / 102), attn qkv 71 (215),
LM head 1053 (3957). Engine at ~442 GB/s over all ternary bytes (86% of peak).

## Fused RMSNorm (rmsprep)

`kernels/rmsprep_kernel.cpp` does the decode RMSNorm, the Hadamard prep and the scatter to the A3
workers in one op (sites ffnorm / attnnorm / lmnorm, `TernaryLinear.norm`, `TernaryMLP.forward_residual`):
the cores exchange per-block partial sums of squares through a master core, and rstd is applied to the
outputs (H(D(x g rstd)) = rstd H(D(x g))), so the exchange overlaps the matmul. Options: `y_out` also writes
the normalized rows (the GDN a|b matmul reads them), `add`/`h_out` fuses the residual add in front.
15.1 us at op level (x in L1) vs 9.7 for plain prep; it replaces 3-4 ops per site.

## Traps (details in agentA/tt4x_local/TRAPS.md)

- No persistent L1 buffers: they are reserved on every core and clash with large-CB ops.
- Per-core compile-time args split a program into many kernel binaries; use runtime args.
- Scattered worker sets cost ~6 us of dispatch gap per call inside a trace; use rectangles.
- A2b: the unpacker's Force_shared_exp (exponent 127) reads 256 B code tiles from the tile base.
- generic_op caches by structure: keep runtime-arg counts fixed; don't reuse a descriptor
  after freeing a tensor its CBs were built on.
- An sfpi DST unit is 4 rows of even or odd columns.
- Stock ttnn ops reject 1x32 tiles; narrow-N stock matmuls on auto configs are ~9x slow.
- The decode tails around the 6144-wide sites are dispatch-bound: removing one 2.4 us op (a typecast)
  saved 0.5 ms/token. Count ops, not just kernel time.
