# Fused decoder work log

## Baseline and hardware

- Starting functional-decoder commit:
  `c71fc9900b0f2da46a4ea736d34fdfca6d66fe5d`.
- Four local Blackhole p300c devices were healthy in `tt-smi -ls --local`;
  all Stage-02 work used one 1x1 mesh and required no reset/recovery.
- Contract retained at 500000 tokens for batch 1 and dense batch 32. After
  eliminating duplicate packed/separate weights, sparse batch 32 has a hard
  measured p300c DRAM boundary at 496928 tokens; page 496960 fails allocation.
  BF16 paged K/V and page size 32 are unchanged.

## Exhaustive graph-fusion audit

Full measured topology and movement boundaries:

| Path | Ordered device topology | Layout/movement |
|---|---|---|
| dense prefill | RMSNorm -> packed QKV linear -> split heads -> RoPE Q/K -> paged K/V fill -> SDPA -> concat heads -> output linear -> residual add -> packed gate/up linear -> two slices -> fused SiLU-multiply -> down linear -> residual add | BF16 TILE, DRAM interleaved throughout; slices remain on device |
| dense decode | RMSNorm -> packed QKV linear -> decode head creation -> cos/sin sharding -> RoPE -> paged K/V update -> paged SDPA -> concat-input sharding -> concat heads -> output linear -> residual -> packed gate/up linear -> split -> fused SiLU-multiply -> down linear -> residual | three required interleaved-to-height-sharded boundaries for cos, sin, and decode concat; batch-1 final reshape is a view |
| sparse prefill | prefill attention -> RMSNorm -> router linear -> TopK -> sigmoid -> zeros/scatter -> reshape/repeat experts -> packed expert gate/up matmul -> two slices -> fused SiLU-multiply -> expert down matmul -> route permute/reshape -> route multiply -> expert sum -> residual | TopK/scatter routing crosses TILE/row-major; repeat output is tilized once for expert matmul; weights/results stay in device DRAM |
| sparse decode | decode attention -> RMSNorm -> router linear -> TopK -> sigmoid -> zeros/scatter -> reshape/repeat experts -> packed expert gate/up matmul -> slices -> fused SiLU-multiply -> expert down matmul -> route multiply -> expert sum -> residual | same router/expert contract boundary; no host transfer or general reshard |

| Pattern family | Assessment and result |
|---|---|
| dedicated QKV/head ops | already fused in the functional baseline |
| RoPE | already dedicated in prefill/decode |
| SDPA/softmax | already dedicated SDPA/paged SDPA; router uses sigmoid top-k semantics |
| paged cache update/fill | already dedicated and watcher-clean |
| RMSNorm | already dedicated; residual+norm would cross the layer boundary |
| TopK | already dedicated |
| shared-LHS dense gate/up | packed linear accepted for prefill/decode |
| shared-LHS expert gate/up | packed batched matmul accepted for both sparse layer kinds |
| activation consumer fusion | SiLU folded into binary multiply and accepted |
| bias/scale/transpose folding | no projection bias/scale; weights setup-transposed |
| final slice/transpose | batch-1 reshape accepted; batch-32 reshape rejected as slower |
| structural permute/reshape | remaining head transforms owned by dedicated head ops |
| sparse_matmul | tested: B1 PCC 0.999795; tile-union B32 PCC 0.938366, rejected |
| experimental moe_compute | one-token process SIGFPE (136); 32-token control passes but requires BF4 packed weights, COL dispatch, and an extra score combine, rejected |
| `moe_gate_mm` | only a DeepSeek-specific matmul into a caller-preallocated output, with mandatory `layer_id`/`column_id`; it does not replace North's top-k/sigmoid/scatter and assumes the DeepSeek gate pipeline |
| `deepseek_moe_gate` | fixed DeepSeek top-8 routing plus score normalization on height-sharded tensors; North needs unnormalized `sigmoid(topk(logits))`, so the semantics differ |
| `generalized_moe_gate` | supports top-k 4/6/8 and sigmoid, but always normalizes selected scores (linear or softmax), requires bias/index/preallocated height-sharded state, and its direct tests are skipped on Blackhole; North's unnormalized scores cannot be represented |
| grouped-topk gate | performs DeepSeek group selection and normalized/scaled routing; North uses global top-k without groups or normalization |
| hash-gate variants | routing policy is hash/group based rather than North's learned global router logits; semantic mismatch |
| `TTMoEGate` | DeepSeek module wrapper around the above preallocated, normalized gate operations and `TTMoEDecode`; it does not preserve North routing semantics or arbitrary prefill |
| TTMoEDecode | contract-refuted: multi-device CCL decode orchestration, no compute-only/prefill path, outside the single-device contract |
| reductions | sparse expert sum cannot merge with route multiply under available matching contracts |
| conv/pool/batchnorm/spatial mean | absent |
| collectives/distributed norm | absent in this single-device decoder |
| LM head/sampler | outside decoder and absent |

The final dense profiles contain no conversion, tilize/untilize, reshard, or
host fallback. Sparse TopK/scatter necessarily produces row-major routing data,
whereas expert matmul consumes TILE data; the reported conversions occur once
at that contract boundary and are not redundant round trips. This conclusion
is supported by `AUTODEBUG.md`, `autofix_sparse_matmul.md`, and
`moe_compute_probe.md`.

## Candidate record

1. SiLU followed by multiply -> multiply with SiLU input activation: correct,
   faster, retained for dense and sparse.
2. Packed dense gate/up plus `ttnn.split`: correct for decode and length 33,
   but reproducibly fails at prefill length 65 in
   `reader_tm_tile_layout_split_two_chunks.cpp` and
   `writer_tm_tile_layout_split_two_chunks.cpp` because
   `single_tile_size_bytes` is undeclared.
3. Packed dense gate/up plus two device slices: correct at lengths 33/65;
   length-65 mean 0.655171 -> 0.600521 ms, seq-128 batch-1
   0.618305 -> 0.599085 ms, batch-32 13.565878 -> 12.543409 ms.
   Retained as default prefill.
4. Packed sparse expert gate/up: nonzero-weight PCC 0.999821/0.999817.
   Layer-1 prefill batch-1 14.635811 -> 10.073406 ms, batch-32
   141.856883 -> 122.069731 ms; decode batch-1
   9.488808 -> 6.703998 ms, batch-32 11.209368 -> 8.420571 ms.
   Layer 4 reproduced the same gains. Retained as default.
5. Decode output slice/transpose -> slice/reshape: batch-1
   0.339064 -> 0.334451 ms, retained. Batch-32 6.025637 -> 6.050667
   ms, rejected; the faster transpose geometry remains.
6. `ttnn.sparse_matmul`: B1 traced PCC 0.999795 passed, but its required
   per-M-tile union mask at serving batch 32 reproduced PCC 0.938366 after the
   canonical output ordering. Rejected before latency acceptance.
7. `ttnn.experimental.moe_compute`: the exact one-token North probe reached
   the extension and died with SIGFPE/exit 136. Its 32-token control passed
   canonical goldens (PCC 0.992406/0.990744), proving shape capacity but also
   requiring padding, BF4 packed weights, COL dispatch, and a separate
   route-score combine. Rejected for Stage-02.
8. `TTMoEDecode`: rejected by API contract because it always performs
   multi-device dispatch/combine/reduce-scatter, is decode-only, and cannot
   replace the single-device arbitrary-length stage.
9. Packed-weight ownership: the initial loader retained unused separate
   gate/up device tensors. The fused loader now directly uploads only the
   selected packed family, reserves K/V first, and places largest weights first.
   Tests assert the unused keys are absent.
10. Dedicated gate family: source contracts for `moe_gate_mm`,
    `deepseek_moe_gate`, `generalized_moe_gate`, grouped-topk, hash routing,
    and `TTMoEGate` were audited. None can express North's unnormalized global
    `sigmoid(topk(router_logits))` at both arbitrary prefill and decode.
    Consequently no semantically valid hardware candidate existed to benchmark.

Exact packed-split reproduction:

```bash
python models/autoports/coherelabs_north_mini_code_1_0/tests/fused_decoder_perf.py \
  --implementation fused --mode prefill --layer 0 --batch 1 --sequence 65 \
  --warmups 0 --iterations 1 --dense-gate-up-variant packed_all
```

Failure: TTNN JIT build error, `single_tile_size_bytes` undeclared in both
split-two-chunks reader and writer kernels. The `packed_slice` adaptation is
the delivered default and removes this restriction. Exact captured output is
preserved in `candidate_dense_packed_split_failure.txt`.

## Final commands and artifacts

Correctness:

```bash
pytest -q models/autoports/coherelabs_north_mini_code_1_0/tests/test_fused_decoder.py \
  -s --junitxml=models/autoports/coherelabs_north_mini_code_1_0/doc/fused_decoder/correctness.xml
```

Wall performance (run for layers 0/1/4, batches 1/32, prefill/decode):

```bash
python models/autoports/coherelabs_north_mini_code_1_0/tests/fused_decoder_perf.py \
  --implementation fused --mode decode --layer 1 --batch 32 --sequence 128 \
  --warmups 5 --iterations 50 \
  --json-out models/autoports/coherelabs_north_mini_code_1_0/doc/fused_decoder/perf/final_decode_l1_b32.json
```

Profiler and report:

```bash
python -m tracy -r -p -o <artifact-directory> \
  models/autoports/coherelabs_north_mini_code_1_0/tests/fused_decoder_perf.py \
  --implementation fused --mode decode --layer 0 --batch 1 \
  --sequence 128 --warmups 3 --iterations 2

tt-perf-report <ops_perf_results.csv> --start-signpost PERF_DECODE \
  --end-signpost PERF_DECODE_END --no-color \
  --csv <perf_report.csv> --summary-file <perf_summary.csv>
```

Final evidence:

- `correctness.xml`: 11 passed;
- `perf/final_*.json`: all twelve 50-sample final points;
- `performance_matrix.csv`: functional/final comparison;
- `tracy/final_*`: raw op CSVs and tt-perf-report CSV/summary files;
- `AUTODEBUG.md`, `autofix_sparse_matmul.md`, and `moe_compute_probe.md`:
  dedicated sparse-op assessments and exact outcomes;
- `moe_compute_probe.junit.xml`: passing 32-token canonical control;
- `decode_layer{1,4}_context496928_batch32.json`: maximum sparse serving-batch
  finite traced decode;
- `sparse_batch32_context496960_oom.txt`: exact next-page DRAM failure;
- `watcher_correctness.xml`: 11 passed;
- `watcher_final/generated/watcher/watcher.log`: 2170 clean lines.

The older `candidate_sparse_packed_decode_l4_b32.json` and
`fused_decode_l0_b32.json` are not distinct configurations: they invoke the
same retained packed/default runtime as the final files, with fewer samples
and normal timing variation. Their means (8.401016 and 6.025637 ms) differ
from the authoritative 50-sample repeats (8.416626 and 6.038892 ms) by only
0.19% and 0.22%. Configuration selection therefore compares distinct runtime
graphs, not lucky repeats of the same graph. The retained graph beats every
correct distinct traced-decode baseline/candidate; the final 50-sample matrix
is the shipped measurement.

## Review and commits

The first independent review returned `more-work-needed`: it identified stale
decode profiling, missing layer-4 performance, an unearned sparse packed-matmul
rejection, and missing reproducible evidence for dense packed prefill. All four
findings were addressed by the final profiler set, layer-4 matrix, accepted
packed sparse path, and packed-split reproduction/adaptation above.

The second independent review returned `more-work-needed` for untested
dedicated sparse-MoE ops, incomplete movement tables, missing sparse-prefill
profiling, and the absent split-failure artifact. AutoFix then tested
`sparse_matmul` and `moe_compute`, contract-refuted `TTMoEDecode`, added the
full topology table, collected `tracy/final_sparse_prefill_b1`, and preserved
`candidate_dense_packed_split_failure.txt`.

The third independent review found duplicate packed/separate device weights and
required direct fused capacity proof. The loader now keeps only the selected
weight family. A fragmentation-minimized page-aligned sweep established 496928
as the largest feasible sparse batch-32 context on p300c; at 496960, total free
DRAM equals the largest block and is 81536 bytes per bank below the packed
weight requirement. Both sparse layer kinds pass finite traced decode at the
retained boundary, and `doc/context_contract.json` records the hard physical
limitation.

The final independent rereview returned `clean-pass` with no required work.
It confirmed the hard physical capacity boundary, identical-runtime timing
noise, dedicated gate-family contract audit, correctness/watcher evidence,
and profiler artifacts. The stage-owned local commit SHA is recorded below;
no push was performed.

Stage implementation commit: recorded by the following documentation-only
checkpoint after creation.
