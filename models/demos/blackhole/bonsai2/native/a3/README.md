# a3: exact-ternary decode engine (PQ2_0, single chip)

`BONSAI_NATIVE=pq2_0` runs every ternary projection of Ternary Bonsai 2 27B at decode straight from
the PQ2_0 GGUF: 2-bit codes as `BFLOAT2_B` tiles plus the fp16 per-128-group scales (held as fp32),
applied inside the matmul. Nothing is requantized. `BONSAI_NATIVE_SITES` picks sites (default: all
of `mlp, gdn_in, gdn_out, attn_qkv, attn_o, lm_head`); decode only (M = 1).

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
| all sites (dev, one run) | 44.75 | 95.51 / 100.00 |

Op level (device us, real weights, PCC >= 0.99999 vs float64 cpu_ref): MLP 240 per layer (dense
~575), GDN [qkv|z] 79 (dense 259), ssm_out / attn_output 38 (91 / 102), attn qkv 71 (215),
LM head 1053 (3957). Engine at ~442 GB/s over all ternary bytes (86% of peak).

## Traps (details in agentA/tt4x_local/TRAPS.md)

- No persistent L1 buffers: they are reserved on every core and clash with large-CB ops.
- Per-core compile-time args split a program into many kernel binaries; use runtime args.
- generic_op caches by structure: keep runtime-arg counts fixed; don't reuse a descriptor
  after freeing a tensor its CBs were built on.
- An sfpi DST unit is 4 rows of even or odd columns.
- Stock ttnn ops reject 1x32 tiles; narrow-N stock matmuls on auto configs are ~9x slow.
