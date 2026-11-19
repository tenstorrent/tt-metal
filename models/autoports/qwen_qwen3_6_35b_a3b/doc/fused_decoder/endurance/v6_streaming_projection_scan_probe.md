# Streaming BF16 projection product and terminal scan

`tests/v6_streaming_projection_scan_probe.py` is a new, opt-in diagnostic that streams BF16 weights and coefficient tiles into an FP32 destination product, then runs the existing native default cumsum without materializing the product tensor. Source SHA256: `b555f5caae383fb2cf1976860e92b9af3f1f2e2e39ec560e4c43d4d05bb23354`.

The previously frozen terminal-scan and router helpers remain unchanged. Dependencies are `v6_terminal_projection_scan_probe.py` (`8c9eeb05d3cd388fdffe581c459b2178fd2c526c6117f630a779154d39736b64`) and `v6_ordered_router_fusion_probe.py` (`067595f30f1b7e29c65795748dddd9b594725d40190c8541c0863a9cf8dd5314`). No production decoder, shared/native/compiler source, integration builder, or measured prior probe was edited.

The parent-owned `v6_terminal_projection_scan_device.json` passed 32 adapter fixtures, multiple leading axes, and exact outputs after 100 trace replays. Its complete adapter traced results were:

| Adapter | Native ms | Terminal-only ms |
|---|---:|---:|
| Router | 0.090378 | 0.060586 |
| Attention output | 0.264155 | 0.151061 |
| Packed attention K512 | 0.278174 | 0.188974 |
| Packed Q/Z K512 | 0.281274 | 0.186734 |
| Packed attention K1024 | 0.250511 | 0.187181 |
| Packed Q/Z K1024 | 0.252521 | 0.189709 |

Eager latency regressed because generic descriptor construction adds host overhead. The parent is therefore evaluating decode-only terminal-scan integration separately. These measurements support investigating streaming products; they are not measurements of this new helper.

## Candidate and arithmetic boundary

`StreamingProjectionScan(ttnn, mesh)(weight, coefficients)` accepts the existing BF16 TILE interleaved L1/DRAM tensors. Weights have shape `(..., K, width)` and coefficients have exactly `(..., K, 1)`. K is 512 or 1024; width is a multiple of 32. These are internal projection-block requirements, with no public sequence-length restriction. The output is FP32 TILE L1 with shape `(..., 1, width)`, matching the original partial-sum placement.

The reader supplies one 2048-byte BF16 weight tile and one 2048-byte BF16 coefficient tile for each height step, duplicating the coefficient read across width runs. A column-broadcast FPU multiply (`mul_bcast_cols_init`, `mul_tiles_bcast_cols`, HiFi4, FP32 destination enabled) creates the product directly in destination registers. The native `cumsum_tile_init<CumsumMode::Default>` and `cumsum_tile<CumsumMode::Default>` then scan that product. FPU arithmetic is chosen because an intervening SFPU multiply would overwrite the SFPU registers carrying native cumsum totals. Native carry resets at the beginning of each run. This source-level argument requires device verification of the FPU/SFPU transition.

Only the terminal tile is packed in the measured path. The proven writer copies row31 raw FP32 words into row0 of a zero-filled tile. There is no runtime host conversion, product tensor allocation, product placement copy, or product-prefix output. Coefficient preparation/gathers/layouts remain unchanged. Each active core has 16 KiB of circular buffers; all available grid cores receive balanced contiguous runs. Input traffic includes repeated coefficient tiles, and reinitializing FPU and SFPU per tile has a cost. No speedup is assumed.

The complete selected `linear` method is copied from `v6_inline_router_silu_chunk64_output_mkl_explicit_norm_v2.py.snapshot`. Exactly three seven-statement sequences—multiply, cumsum, and two crops with their deletes—become a streaming call. Reversing those substitutions restores the complete method AST exactly. Native reduction across blocks, output casts/placement, prefill recursion, and all other branches remain intact. The candidate is always compared to the frozen native multiply/cumsum baseline, not to an approximate CPU reduction.

## Product precision checks

Finite BF16 operands have at most eight significand bits each, so normal products need at most sixteen, within FP32's twenty-four-bit significand. CPU fixture bounds prove that all nonzero products represented by the saved/random inputs and actual weights stay within normal FP32 exponent range. This establishes a mathematical representation bound for these fixtures, not device FPU equivalence or general IEEE behavior. Signed zero, subnormal operands/products, overflow, infinities, and NaNs are not covered by that argument.

The opt-in test therefore adds a `products_only=True` diagnostic. It uses the same reader, FPU broadcast multiplication, FP32 destination format, and packing, but omits cumsum and writes every product tile to DRAM for comparison against native `ttnn.multiply(... dtype=float32, fast_and_approximate_mode=False)`. This mode is outside all measured adapter calls. It requires exact product bits on every singleton case, including zeros. Reports distinguish signed-zero-only differences from nonzero differences and preserve the first sixteen differing indices and bit patterns. No exception or tolerance is applied to either category. The normal terminal mode separately requires exact native partial sums and complete FP32/BF16 outputs.

## CPU/source artifacts

```bash
PYTHONPATH=. python_env/bin/python \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_streaming_projection_scan_probe.py \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_streaming_projection_scan_cpu_setup_v2.json
```

Result: `CPU_SOURCE_SETUP_PASS 32`; Python compilation passes. TTNN was not imported and no device was opened. Fixtures and setup assignments come from the frozen generalized probe: actual layer34 checkpoint weights; captured `post_norm`, `input_norm`, and `gated_core` at positions 227, 738, and 5799; one seeded random group; zeros; and two three-row non-aligned prefill inputs assembled from the actual captures. The separate `(2,3,512,256)` case uses actual-derived BF16 weights and coefficients, checking multiple leading axes and DRAM input handling.

The v2 JSON and sibling generated-source snapshots freeze the exact reader, compute, terminal writer, product writer, complete baseline/candidate methods, product/input extraction helpers, setup assignments, fixture tensors, source/dependency hashes, scheduling proofs, and product bounds. The initial `v6_streaming_projection_scan_cpu_setup.json*` artifacts preserve the first setup before product difference classification and exponent-bound reporting. Use v2 with the delivered source hash.

## Parent-owned execution

Minimal first component run:

```bash
PYTHONPATH=. python_env/bin/python \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_streaming_projection_scan_probe.py \
  --run-device --case-filter actual_227/router \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_streaming_projection_scan_device_smoke.json
```

This also executes the leading-axis product/terminal checks. Full run:

```bash
PYTHONPATH=. python_env/bin/python \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_streaming_projection_scan_probe.py \
  --run-device --timing \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_streaming_projection_scan_device.json
```

Every selected singleton checks full products, partial sums, and FP32/BF16 outputs bit for bit. Multirow cases compare complete outputs. Actual position227 and both three-row prefill cases include three warmups, five synchronized eager calls, and ten groups of ten trace replays, checking unchanged output bits after replay. Both variants measure complete coefficient preparation, projection, block reduction, casts, and placement. The runtime audit rejects Torch arithmetic and host/device conversions inside forward. Output downloads and product-only diagnostics remain outside timing.

Compilation and device behavior have not been tested by this subagent. Parent-owned product equivalence, native carry preservation, watcher cleanliness, complete-adapter timing, and full-decoder validation are required before integration. No streaming component, full decoder, or stage completion claim is made.
