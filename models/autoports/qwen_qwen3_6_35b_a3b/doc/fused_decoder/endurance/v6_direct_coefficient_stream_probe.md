# Direct BF16 coefficient construction in the streaming reader

`tests/v6_direct_coefficient_stream_probe.py` removes ordered-projection coefficient preparation from the guarded decode path by reading the original TILE activation and copying its raw BF16 bits directly into coefficient column0. Frozen source SHA256: `94e8be48e0fdd447310af7a1d6974c0e51aee6d2f721acfcad10fdbe463121e7`.

The baseline is snapshot `v6_corrected_fma_full_v5_streaming_scan_norm_tail_integration.py.snapshot`, SHA256 `700c10db90b23fac1d40427c9bbcd82a6924c7704699d54768a0160c2b641e05`. Streaming compute, terminal writer, and product writer are reused unchanged from the measured helper `b555f5caae383fb2cf1976860e92b9af3f1f2e2e39ec560e4c43d4d05bb23354`. No production, shared/native, measured helper, or existing snapshot is edited.

## Mapping and scope

For each ordered projection, the existing preparation moves input to DRAM, converts TILE to ROW_MAJOR, reshapes groups of32, gathers pair-flipped columns, reshapes to `(blocks,1,K,1)`, and converts back to TILE. Weights were already pair-permuted during setup. The new reader constructs precisely that coefficient column directly from the original input:

```text
activation_tile = block * (K / 32) + height_tile
paired_lane = row ^ 1
source_BF16_offset = (paired_lane / 16) * 256 + paired_lane % 16
destination_BF16_offset = (row / 16) * 512 + (row % 16) * 16
```

The source offsets select original TILE row0 from faces0/1; destination offsets select coefficient column0 in faces0/2. All transfers are uint16 copies. No BF16 value is converted, rounded, multiplied, or otherwise interpreted during this mapping. Both coefficient CB pages are zeroed once before either is published; all32 column0 elements are overwritten on every reuse, preserving deterministic zero padding without clearing an entire tile at every step.

The input vector is `(1,1,2048)` for router/packed attention/QZ or `(1,1,4096)` for attention-out. Projection K512/K1024 determines only internal block addressing. The reader obtains input page addresses from `TensorAccessor`, preserving the original interleaved or sharded memory placement. Attention-out in the frozen decoder comes from `nlp_concat_heads_decode`: width-sharded across32 cores with `(32,128)` shards. The probe explicitly reproduces this layout. The accessor constructor statically includes shard coordinates and page shapes; the helper asserts that no common runtime accessor arguments are omitted.

The compute kernel is exactly the existing streaming BF16 column-broadcast multiply into FP32 destination registers followed by native default cumsum. The final native block reduction, dtype casts, and memory placement are unchanged. The helper adds one 2048-byte raw input tile buffer, for 18 KiB of circular buffers per active core.

The candidate copies the complete baseline `linear` method and inserts a new early branch inside each ordered singleton projection, guarded by the existing `terminal_scan` flag. That branch retains original weight/block/width setup, invokes the direct-input helper, and retains the original final reduction/return. Removing those three inserted branches restores the complete baseline method AST. Default/false public prefill, non-aligned multirow prefill, full-attention V5, and all other branches remain intact. K1024 direct-input cases are diagnostic component coverage; public prefill still selects its original path.

## CPU/source checks and fixtures

Executed without TTNN import or a device:

```bash
PYTHONPATH=. python_env/bin/python \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_direct_coefficient_stream_probe.py \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_direct_coefficient_stream_cpu_setup_v2.json
```

Result: `CPU_SOURCE_SETUP_PASS 40 65536 coefficient bit patterns`; Python compilation passes. The raw mapping proof exhausts all65,536 BF16 bit patterns, including signed zeros, subnormals, infinities, and NaN encodings, without floating-point arithmetic. Separate integer-address proofs match the native pair gather/reshape for total input widths2048/4096 and internal K512/K1024. These are CPU mapping proofs, not device support or arithmetic claims for special values.

The 40 device fixtures comprise the prior32 actual/random/zero and non-aligned prefill adapter cases, five attention-out width-sharded cases (three actual positions, random, zero), and three actual-position227 L1 interleaved cases (router, packed attention K512, packed Q/Z K512). Actual captures are positions227,738,5799 and actual layer34 checkpoint weights. JSON and sibling source/fixture artifacts freeze inputs, generated kernels, complete baseline/candidate/input/setup methods, and all recorded dependency hashes.

The initial `v6_direct_coefficient_stream_cpu_setup.json*` preserves the first CPU setup, which cleared coefficient padding on every tile. The delivered v2 clears both pages once before reuse. No hardware was run for either setup by this subagent.

## Parent-owned validation

Initial smoke:

```bash
PYTHONPATH=. python_env/bin/python \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_direct_coefficient_stream_probe.py \
  --run-device --case-filter actual_227/router \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_direct_coefficient_stream_smoke.json
```

Full correctness and timing:

```bash
PYTHONPATH=. python_env/bin/python \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_direct_coefficient_stream_probe.py \
  --run-device --timing \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_direct_coefficient_stream_device.json
```

For every singleton case, a coefficient-only diagnostic executes the same reader mapping and writes the assembled BF16 coefficient tensor. It is compared bit for bit to native preparation. Product-only diagnostics compare every FP32 product to native multiply, then normal mode compares terminal FP32 partials to the frozen streaming helper. Complete adapters compare FP32/BF16 outputs and memory placement. All gates remain exact, including signed zeros. The native baseline prepares coefficients normally; there is no functional fallback in the candidate's selected fast path.

Diagnostic coefficient/product writes occur outside measured projection calls. The runtime audit rejects Torch arithmetic or host/device conversion inside forward. Whole-adapter timing covers every position227 placement variant with three warmups, five synchronized eager calls, and ten groups of ten trace replays, checking exact outputs after100 replays. Prefill uses its preserved native path and is covered for correctness.

No device compilation, sharded-accessor execution, correctness, or latency has been measured by this subagent. Reader scalar-copy cost, duplicated input-tile reads across width runs, and descriptor overhead may offset dispatch/movement savings. Parent-owned exact device results, watcher cleanliness, and full-decoder comparison against the best correct candidate are required before selection.
