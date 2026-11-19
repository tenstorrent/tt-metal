# Generalized terminal projection scan component

`tests/v6_terminal_projection_scan_probe.py` generalizes the parent-validated terminal-only router cumsum without modifying its frozen source. New source SHA256: `8c9eeb05d3cd388fdffe581c459b2178fd2c526c6117f630a779154d39736b64`. Frozen router dependency SHA256: `067595f30f1b7e29c65795748dddd9b594725d40190c8541c0863a9cf8dd5314`.

The earlier parent-owned router component passed all 12 fixtures with bit-identical FP32 partials, FP32 outputs, and BF16 outputs (`v6_ordered_router_fusion_device.json`). Its complete adapter traced median improved from 0.090582156554 ms to 0.061141885817 ms, approximately 32.50%, with exact outputs after 100 replays (`v6_ordered_router_fusion_timing_device.json`). These measurements support this follow-up experiment; they do not establish generalized-projection or decoder speedup.

## Arithmetic and scope

`TerminalProjectionScan(ttnn, mesh)(products)` accepts FP32 TILE interleaved L1 or DRAM products, K512/K1024, tile-aligned widths, and arbitrary leading dimensions. It emits the same leading dimensions with one logical row and the original width, memory placement, and FP32 dtype. K restrictions describe the internal reduction block; no public sequence-length restriction is added.

The helper uses all available compute/storage grid cores, or fewer cores when there are fewer runs. Contiguous `(leading_batch, width_tile)` runs are balanced across cores. Each core may process several runs. The compute kernel preserves the proven native default cumsum instructions, direct FP32 unpack, FP32 destination storage, double-buffered destination synchronization, and per-tile carry sequence. `first=true` resets carry for each new run. Only the terminal tile is packed. The writer copies its row31 words into row0 of a zero-filled FP32 tile. No floating-point arithmetic occurs in this relocation.

The selected decoder snapshot remains `v6_inline_router_silu_chunk64_output_mkl_explicit_norm_v2.py.snapshot`. The complete `linear` method is cloned. Its three scan/crop sequences are replaced, covering four projection keys:

| Projection | Input width | Output width | Reduction K |
|---|---:|---:|---:|
| Router gate | 2048 | 256 | 512 |
| Linear attention output | 4096 | 2048 | 512 |
| Packed attention state projection | 2048 | 6208 | 512 and 1024 |
| Packed Q/Z output projection | 2048 | 6144 | 512 and 1024 |

Reversing the three substitutions restores the complete method AST exactly. All weight setup assignments are extracted from the same snapshot. Products, input gathers and layout changes, BF16 weight boundaries, FP32 block reduction, output casts, and output memory placement remain unchanged. The existing multirow prefill recursion still invokes K1024 projection one row at a time; three-row non-aligned input is included. No shared/native/compiler, production decoder, integration builder, or previously measured test source was edited.

Each active core needs 16 KiB of circular buffers. The helper eliminates full prefix tensor writes and two subsequent crops, retaining the existing product tensor. Generic descriptor overhead, grid scheduling, scratch zero-fill, and reader/writer synchronization remain possible costs. Performance must be measured.

## CPU/source evidence

```bash
PYTHONPATH=. python_env/bin/python \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_terminal_projection_scan_probe.py \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_terminal_projection_scan_cpu_setup_v2.json
```

Result: `CPU_SOURCE_SETUP_PASS 32`; Python compilation also passes. The setup does not import TTNN or open a device. Scheduling proofs visit every input tile exactly once for all supported projection dimensions, unequal workloads, and a `(2,3,512,256)` tensor on 8x8 and 11x10 grids. Native SFPU arithmetic is intentionally not approximated by a CPU sum model.

Fixtures use actual layer34 checkpoint weights at HF revision `345d467b117d1fd5eda9b879a5988de9ae5f47a7`, with actual `post_norm`, `input_norm`, and `gated_core` captures at positions 227, 738, and 5799. The 32 adapter cases consist of six projection/K combinations for each actual position, one seeded random input group, zeros, and two three-row prefill cases. Three-row inputs stack the saved actual normalized vectors; they are component inputs, not a claim that those distant positions form a contiguous decoder sequence. A separate actual-derived `(2,3,512,256)` product fixture validates multiple leading axes and DRAM input/output handling.

The v2 JSON records source, frozen-router, snapshot, native cumsum dependency, capture, and fixture hashes. Sibling `.reader.snapshot`, `.compute.snapshot`, `.writer.snapshot`, `.baseline.snapshot`, `.candidate.snapshot`, `.products.snapshot`, `.setup.snapshot`, and `.fixtures.pt` files freeze the exact generated sources and inputs. The initial `v6_terminal_projection_scan_cpu_setup.json*` artifacts preserve the first source-only setup before adding the separate leading-axis device fixture and eager timing. Use v2 for the delivered source hash.

## Parent-owned device run

```bash
PYTHONPATH=. python_env/bin/python \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_terminal_projection_scan_probe.py \
  --run-device --timing \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_terminal_projection_scan_device.json
```

Optional `--case-filter actual_227 stacked_` limits the initial adapter run to eight cases while retaining the separate leading-axis check. The result lists precisely which cases executed; a filtered run does not establish all-fixture coverage.

Every singleton adapter compares native and terminal FP32 partial sums bit for bit, then complete native and candidate FP32/BF16 outputs. Three-row prefill compares complete FP32/BF16 outputs. The separate leading-axis fixture compares native and generalized FP32 partials. Any bit difference fails. No tolerance is relaxed. Forward calls run under the existing runtime audit, rejecting Torch arithmetic and host/device conversion. Fixtures are uploaded before calls; outputs are downloaded afterward.

For actual position227 and both three-row prefill adapters, timing warms each full projection three times, measures five synchronized eager calls, captures the complete adapter, measures ten groups of ten trace replays, and checks output bits after those 100 replays. Both variants include coefficient preparation, products, scan, block reduction, casts, and placement. Native and candidate outputs are saved beside the result JSON.

This subagent has not executed the generalized helper on hardware or compiled its device kernels. Remaining gates are parent-owned compilation, full component equivalence, watcher cleanliness, timing, and full-decoder integration/regression. No generalized speedup, end-to-end acceptance, or optimized-stage completion is claimed.
