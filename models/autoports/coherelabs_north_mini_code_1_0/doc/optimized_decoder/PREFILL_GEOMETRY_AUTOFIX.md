# Review 5 dense-expert prefill AutoFix

## Verdict

The retained `117.903 ms` result is not a reproducible best-correct candidate.
It came from an uncommitted predecessor of the first optimized-decoder commit,
used the old incorrect batch-32 attention construction
(`_attention_prefill` called `_qkv_prefill(..., batch_size=1)`), has no PCC
result, and does not
record a source revision, activation dtype, router policy, prefill policy, or
warmup/iteration provenance. Commit `f77d4e00940` added that JSON and the
already-corrected code atomically, so Git cannot reconstruct the code which
produced the number.

Two controls isolate the apparent regression:

| Artifact | Relevant contract | Mean |
|---|---|---:|
| `dense_expert_chunk1024_prefill_b32.json` | stale pre-commit code; incomplete policy; 3 samples; no PCC | 117.903 ms |
| `real_validated_bfp4_expert_prefill_b32.json` | corrected path; one BFP4 expert family; BF16 intermediates | 139.965 ms |
| `review3_final_runtime/layer1_prefill_b32.json` | final mixed BFP8-sparse/BFP4-dense policy | 139.959 ms |
| `review5_single_bfp4_residency.json` | final code; BFP4 sparse/dense aliases, so no extra BFP8 copies | 140.177 ms |
| `review5_prefill/split_active_w8_6.json` | BFP4/LoFi split, explicit 64/80-core programs | 100.909 ms |
| `review5_prefill/split_88_w8_6.json` | BFP4/LoFi split, explicit 88/88-core programs | 107.555 ms |
| `review5_prefill/packed_80_w8_6.json` | BFP4/LoFi packed gate/up, explicit 80/80-core programs | **96.844 ms** |
| `review5_prefill/packed_80_w8_6_layer4.json` | selected packed candidate, layer 4 control | **96.644 ms** |

The last control refutes extra BFP8 expert residency as the cause: removing
the duplicate family changes `139.959` to `140.177 ms`, not to `117.903 ms`.
It does still save a material amount of DRAM. Each projection contains
`128 * 64 * 24 = 196,608` tiles. A BFP8 tile is 1,088 bytes and a BFP4 tile is
576 bytes, so the final mixed policy holds:

- sparse BFP8 gate/up/down: `641,728,512 B` (`612 MiB`);
- dense BFP4 gate/up/down: `339,738,624 B` (`324 MiB`);
- total expert projections: `981,467,136 B` (`936 MiB`).

The old single-BFP4 family held only 324 MiB. The extra 612 MiB is real
capacity overhead, but the final-code single-family A/B shows it is not the
latency explanation.

The selected phase-specific packed default adds a separate 216-MiB BFP4
gate/up family because decode keeps its faster split topology. Selected sparse
layer expert residency is therefore 1,152 MiB: 612 MiB sparse BFP8, 324 MiB
split dense BFP4, and 216 MiB packed-prefill BFP4. The advertised-context
capacity gate must be repeated against this final resident set.

## Previous automatic M=1024 rows

The final profile executes four 1,024-token chunks for batch 32, sequence
128. The 128-expert batched rows are BF16 x BFP4 with LoFi compute:

| Role | Shape | Program selected by TTNN | Active cores | Memory |
|---|---|---|---:|---|
| gate/up | `b={128} x 1024 x 2048 x 768` | declared 10x10 2-D multicast; `in0_block_w=1`, `per_core_M/N=4/3`, output block `4x3`, subblock `2x3` | 64 | input, weight, output DRAM interleaved |
| down | `b={128} x 1024 x 768 x 2048` | declared 10x10 2-D multicast; `in0_block_w=1`, `per_core_M/N=4/7`, output block `4x7`, subblock `1x7` | 80 | input, weight, output DRAM interleaved |

The four chunks contribute about 67.4 ms for gate/up, 22.2 ms for down,
15.6 ms for repeat, and the remaining routing, unary, binary, and reduction
work. The profile correctly advises a larger inner block for the matmuls.

For these dimensions, legal inner blocks divide the tiled K dimension:
gate/up K=64 tiles permits `1,2,4,8,16,32,64`; down K=24 tiles permits
`1,2,3,4,6,8,12,24`. The candidate subblock products below are at most eight
LoFi destination tiles and divide their output blocks.

## Legal candidate geometries

The selected default is `packed_80_w8_6`. Packing and geometry are
prefill-specific: `prefill_packed_dense_experts=True` and the explicit
`MatmulMultiCoreReuseMultiCastProgramConfig` fields below are selected by
default, while `packed_dense_experts=False` and the zero-valued legacy decode
program fields keep decode split and framework-selected.

| Candidate | Gate/up geometry | Down geometry | Packed gate/up |
|---|---|---|---|
| `split_active_w8_6` | grid 8x8, `iw=8`, core M/N `4/3`, block `4x3`, subblock `2x3` | grid 10x8, `iw=6`, core M/N `4/7`, block `4x7`, subblock `1x7` | no |
| `split_88_w8_6` | grid 11x8, transpose multicast, `iw=8`, core M/N `3/3`, block `3x3`, subblock `1x3` | grid 11x8, `iw=6`, core M/N `4/6`, block `4x6`, subblock `2x3` | no |
| `packed_80_w8_6` | grid 10x8, `iw=8`, core M/N `4/5`, block `4x5`, subblock `1x5`; N is 48 tiles | same as `split_active_w8_6` | yes |

For a direct block-width fallback, keep the active grids/blocks and change
gate/down `iw=8/6` to `2/3` (advice floor) or `16/12` (largest first
candidate). A runtime L1-CB failure rejects only that block pair; it does not
reject the geometry family.

BF16 M=1024 activation placement is capacity-constrained. The repeated input
is 512 MiB, each split gate/up output is 192 MiB (384 MiB concurrently), the
packed output is 384 MiB, and down output is 512 MiB. Even the theoretical
aggregate across 110 workers at the observed 1.573 MB/core limit is only about
165 MiB, before runtime/CB reservations. Therefore no whole-tensor L1
input/intermediate placement is legal at M=1024. DRAM-interleaved
input/intermediate/output is the only legal final-policy placement; an L1
trial would be a predetermined capacity failure, not a meaningful sweep.

## Minimal serialized hardware matrix

Set the common command once:

```bash
PERF="python models/autoports/coherelabs_north_mini_code_1_0/tests/optimized_decoder_perf.py \
  --mode prefill --batch 32 --layer 1 --sequence 128 --warmups 3 --iterations 20"
OUT=models/autoports/coherelabs_north_mini_code_1_0/doc/optimized_decoder/candidates/review5_prefill
mkdir -p "$OUT"
```

Reproduce the previous automatic split baseline:

```bash
$PERF --candidate auto_split_final \
  --no-prefill-packed-dense-experts \
  --dense-expert-prefill-gate-grid-x 0 \
  --dense-expert-prefill-gate-grid-y 0 \
  --dense-expert-prefill-down-grid-x 0 \
  --dense-expert-prefill-down-grid-y 0 \
  --json-out "$OUT/auto_split_final.json"
```

Measure the active-grid split candidate:

```bash
$PERF --candidate split_active_w8_6 \
  --no-prefill-packed-dense-experts \
  --dense-expert-prefill-gate-grid-x 8 \
  --dense-expert-prefill-gate-grid-y 8 \
  --dense-expert-prefill-gate-up-in0-block-w 8 \
  --dense-expert-prefill-gate-up-per-core-m 4 \
  --dense-expert-prefill-gate-up-per-core-n 3 \
  --dense-expert-prefill-gate-up-out-block-h 4 \
  --dense-expert-prefill-gate-up-out-block-w 3 \
  --dense-expert-prefill-gate-up-subblock-h 2 \
  --dense-expert-prefill-gate-up-subblock-w 3 \
  --dense-expert-prefill-down-grid-x 10 \
  --dense-expert-prefill-down-grid-y 8 \
  --dense-expert-prefill-down-in0-block-w 6 \
  --dense-expert-prefill-down-per-core-m 4 \
  --dense-expert-prefill-down-per-core-n 7 \
  --dense-expert-prefill-down-out-block-h 4 \
  --dense-expert-prefill-down-out-block-w 7 \
  --dense-expert-prefill-down-subblock-h 1 \
  --dense-expert-prefill-down-subblock-w 7 \
  --json-out "$OUT/split_active_w8_6.json"
```

Measure the larger 88-core split family by changing the geometry:

```bash
$PERF --candidate split_88_w8_6 \
  --no-prefill-packed-dense-experts \
  --dense-expert-prefill-gate-grid-x 11 \
  --dense-expert-prefill-gate-grid-y 8 \
  --dense-expert-prefill-gate-transpose-mcast \
  --dense-expert-prefill-gate-up-in0-block-w 8 \
  --dense-expert-prefill-gate-up-per-core-m 3 \
  --dense-expert-prefill-gate-up-per-core-n 3 \
  --dense-expert-prefill-gate-up-out-block-h 3 \
  --dense-expert-prefill-gate-up-out-block-w 3 \
  --dense-expert-prefill-gate-up-subblock-h 1 \
  --dense-expert-prefill-gate-up-subblock-w 3 \
  --dense-expert-prefill-down-grid-x 11 \
  --dense-expert-prefill-down-grid-y 8 \
  --dense-expert-prefill-down-in0-block-w 6 \
  --dense-expert-prefill-down-per-core-m 4 \
  --dense-expert-prefill-down-per-core-n 6 \
  --dense-expert-prefill-down-out-block-h 4 \
  --dense-expert-prefill-down-out-block-w 6 \
  --dense-expert-prefill-down-subblock-h 2 \
  --dense-expert-prefill-down-subblock-w 3 \
  --json-out "$OUT/split_88_w8_6.json"
```

Measure packed gate/up on its legal 80-core output tiling:

```bash
$PERF --candidate packed_80_w8_6 --prefill-packed-dense-experts \
  --dense-expert-prefill-gate-grid-x 10 \
  --dense-expert-prefill-gate-grid-y 8 \
  --dense-expert-prefill-gate-up-in0-block-w 8 \
  --dense-expert-prefill-gate-up-per-core-m 4 \
  --dense-expert-prefill-gate-up-per-core-n 5 \
  --dense-expert-prefill-gate-up-out-block-h 4 \
  --dense-expert-prefill-gate-up-out-block-w 5 \
  --dense-expert-prefill-gate-up-subblock-h 1 \
  --dense-expert-prefill-gate-up-subblock-w 5 \
  --dense-expert-prefill-down-grid-x 10 \
  --dense-expert-prefill-down-grid-y 8 \
  --dense-expert-prefill-down-in0-block-w 6 \
  --dense-expert-prefill-down-per-core-m 4 \
  --dense-expert-prefill-down-per-core-n 7 \
  --dense-expert-prefill-down-out-block-h 4 \
  --dense-expert-prefill-down-out-block-w 7 \
  --dense-expert-prefill-down-subblock-h 1 \
  --dense-expert-prefill-down-subblock-w 7 \
  --json-out "$OUT/packed_80_w8_6.json"
```

After promotion, reproduce the selected path without any policy flags:

```bash
$PERF --candidate final_default \
  --json-out "$OUT/final_default.json"
```

Run only one candidate at a time. Keep BFP4/LoFi dense-expert weights, BF16
expert activations, all other final policy fields, and DRAM placements fixed.
The packed candidate passed real-weight sequence-33 batch-32 prefill at PCC
0.99923857 for layer 1 and 0.99993403 for layer 4; the retained JUnit artifact
is `artifacts/review5_packed_prefill_authentic.xml`. It is now the default, so
the equivalent correctness command needs no policy environment override:

```bash
pytest -q -s \
  models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py \
  -k 'real_weight_moe_precision_matrix and 32-prefill'
```

`NORTH_MINI_AUTHENTIC_PREFILL_GEOMETRY` still accepts explicit candidate
overrides, including `prefill_packed_dense_experts`. The old
`packed_dense_experts` switch remains a global prefill-and-decode experiment;
it is not the selected decode policy.

## Static verification

```text
python -m py_compile optimized_decoder.py optimized_decoder_perf.py
pytest -q test_optimized_decoder_prefill_geometry.py
8 passed
```

No TT device, profiler, or watcher was opened for the geometry implementation.
