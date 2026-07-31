# North-Mini-Code-1.0 fused decoder

Status: complete. Implementation, hardware gates, review remediation, and the
fresh independent `clean-pass` are recorded.

## Scope and contract

The Stage-02 implementation is
`models/autoports/coherelabs_north_mini_code_1_0/tt/fused_decoder.py`.
`FusedDecoder` preserves the functional decoder's constructor, prefill/decode
signatures, paged BF16 KV-cache representation, page-table semantics, stable
trace inputs, parallel residual order, and three layer kinds:

| Representative | Attention | RoPE | MLP |
|---|---|---|---|
| layer 0 | full | forced | dense SwiGLU, intermediate 3072 |
| layer 1 | sliding window 4096 | yes | 128 experts, sigmoid top-8 |
| layer 4 | full | no | 128 experts, sigmoid top-8 |

No cache dtype, page size, sharding, or advertised capacity changed, so
`doc/context_contract.json` remains unchanged at 500000 tokens. The fused
capacity wrapper reran the functional stage's exact allocation and trace
probes. Batch-1 prefill passed at 500000 for dense/full and at the non-aligned
499999 boundary for both MoE kinds. All three kinds passed traced batch-1
decode at position 499999, and dense batch-32 passed with its 32.768 GB cache.

## Fused topology

The delivered path performs these Stage-02 rewrites:

- packs dense gate/up weights once and replaces two projections with one
  projection plus width slices;
- packs every expert's gate/up weights once;
- folds SiLU into the consuming elementwise multiply;
- uses exact top-8 `ttnn.sparse_matmul` expert dispatch for decode token counts
  below one tile;
- uses a packed, tile-efficient all-expert graph at 32 or more tokens, selected
  because it is faster than the exact sparse alternative at serving batch 32;
- fuses routing-score multiplication and expert reduction with
  `deepseek_moe_fast_reduce_nc_fused` at the exact serving tile of 32 tokens;
- replaces two decode cache updates with
  `ttnn.experimental.paged_fused_update_cache`.

The fused cache op requires K and V inputs on disjoint core sets. One
value-side `ReshardDeviceOperation` is therefore required before it; this
replaces the second cache-update dispatch and improves dense batch-1 decode.
The fused score/reduce kernel was also tested for prefill. Although it measured
9.974 ms at sequence 128 versus 10.080 ms for the selected path, a direct
active-expert reference check produced PCC 0.408818. Logical length 33 likewise
produced PCC 0.758886, and a 1024-token chunk requested 4,882,432 L1 bytes per
bank against 1,461,504 available. Splitting sequence 128 into four known-good
32-token fused reductions restored fallback-equivalent numerics but regressed
wall latency from 10.080 to 10.410 ms. Prefill therefore keeps the selected
multiply plus fast-reduce lowering; no public alignment restriction was
introduced.

The sequence-128 candidate diagnostic used a new all-token synthetic
active-expert stress, not one of the functional stage's accepted reference
cases. The unchanged `FunctionalDecoder` and selected fused fallback both
produce the identical PCC 0.98764764 on that diagnostic; it therefore measures
candidate equivalence, not a fused regression. The established representative
MoE prefill gates remain sequence 1025 and non-aligned sequence 33, where the
delivered fused path clears 0.995.

The profiler contains no host ops. There are no runtime `torch`,
`from_torch`, `to_torch`, or functional-MLP fallbacks. MoE tilize/untilize
rows are contract-required transitions around the current top-k/scatter and
sparse-matmul operators, not redundant user-authored conversions.

The complete pattern inventory, full op sequences, applied/rejected candidates,
and structural exclusions are in `graph_fusing_audit.md`.

## Correctness

The fused suite reuses the functional stage's accepted reference tests after
replacing its constructor with `FusedDecoder`; a source audit additionally
proves that the overridden fused methods are dispatched.

| Claim | Functional PCC | Fused PCC | Result |
|---|---:|---:|---|
| dense prefill, logical 33 | 0.999737 | 0.999715 | pass |
| dense paged traced decode | 0.999768 | 0.999765 | pass |
| sliding MoE prefill, 1025 | 0.999827 | 0.999573 | pass |
| full/no-RoPE MoE prefill, 33 | 0.999763 | 0.999763 | pass |
| populated sliding-history trace | 0.999849 | 0.999533 | pass |
| sliding MoE decode, batch 1 | acceptance >=0.995 | 0.999836 | pass |
| sliding MoE decode, batch 32 | 0.998193 | 0.998193 | pass |
| full/no-RoPE MoE decode, batch 1 | 0.999823 | 0.999827 | pass |
| dense traced decode, batch 32 | acceptance >=0.995 | 0.999854 | pass |
| official layer-1 real weights | 0.999798 | 0.999751 | pass |

The material-but-accepted MoE prefill delta comes from replacing two separate
BF16 expert projections with a packed BF16 projection, which changes rounding
order but not routing or mathematical semantics. Every PCC remains above the
functional acceptance threshold of 0.995.

The 19-test suite also covers logical lengths 1/31/32/33/65, reversed physical
page placement, randomized nonzero decode positions, bitwise determinism,
sliding-window length 4097, populated history, all representative layers, and
real weights. It passed normally and again under `TT_METAL_WATCHER=10`; the
20-iteration warmed measurements exercise repeated trace replay for every
decode kind at batch 1 and batch 32.

## Performance

All prefill rows use batch 1 and logical sequence 128. Decode is complete
forward trace replay. Wall values are means of 20 samples after five warmups.
Device values are the signpost-filtered `tt-perf-report` totals. Every fused
report has zero host ops.

| Layer kind / regime | Functional wall | Fused wall | Wall gain | Functional device | Fused device | Device gain |
|---|---:|---:|---:|---:|---:|---:|
| dense prefill b1 | 0.636 ms | 0.580 ms | 8.8% | 586 us | 542 us | 7.5% |
| sliding MoE prefill b1 | 14.908 ms | 10.080 ms | 32.4% | 14644 us | 9998 us | 31.7% |
| full MoE prefill b1 | 14.655 ms | 10.067 ms | 31.3% | 14567 us | 9961 us | 31.6% |
| dense decode b1 | 0.356 ms | 0.320 ms | 10.1% | 338 us | 302 us | 10.7% |
| sliding MoE decode b1 | 9.528 ms | 2.130 ms | 77.6% | 9452 us | 2103 us | 77.8% |
| full MoE decode b1 | 9.524 ms | 2.131 ms | 77.6% | 9439 us | 2088 us | 77.9% |
| dense decode b32 | 6.652 ms | 5.698 ms | 14.3% | 6614 us | 5663 us | 14.4% |
| sliding MoE decode b32 | 11.122 ms | 8.273 ms | 25.6% | 11084 us | 8248 us | 25.6% |
| full MoE decode b32 | 11.129 ms | 8.279 ms | 25.6% | 11077 us | 8234 us | 25.7% |

Thus the final fused runtime beats the best correct traced functional baseline
for every measured layer kind and required batch regime. The filtered CSV,
human-readable table, and canonical raw ops CSV for each row live under
`tracy/<kind>/<regime>/`.

## Capacity results

| Probe | Result |
|---|---:|
| dense/full prefill b1, context 500000 | finite, 159575.45 ms |
| sliding/RoPE/MoE prefill b1, context 499999 | finite, 17107.08 ms |
| full/no-RoPE/MoE prefill b1, context 499999 | finite, 173316.15 ms |
| dense/full decode b1, position 499999 | finite traced replay, 44.32 ms |
| sliding/RoPE/MoE decode b1, position 499999 | finite traced replay, 2.50 ms |
| full/no-RoPE/MoE decode b1, position 499999 | finite traced replay, 46.11 ms |
| dense/full decode b32, position 499999 | finite traced replay, 133.72 ms |

## Reproduction

```bash
pytest -q -s \
  models/autoports/coherelabs_north_mini_code_1_0/tests/test_fused_decoder.py

TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1 \
TT_METAL_LOGS_PATH=models/autoports/coherelabs_north_mini_code_1_0/doc/fused_decoder/watcher_remediation \
  pytest -q -s \
  models/autoports/coherelabs_north_mini_code_1_0/tests/test_fused_decoder.py

python models/autoports/coherelabs_north_mini_code_1_0/tests/fused_decoder_perf.py \
  --mode decode --batch 32 --layer 1 --warmups 5 --iterations 20

python models/autoports/coherelabs_north_mini_code_1_0/tests/fused_decoder_capacity.py \
  --mode prefill --context 500000 --batch 1 --layer 0

python models/autoports/coherelabs_north_mini_code_1_0/tests/fused_decoder_capacity.py \
  --mode decode --context 500000 --batch 32 --layer 0
```

The Tracy command is the same performance command prefixed by
`python -m tracy -r -o <artifact-directory> -n <case-name>`. Reports are
filtered with `--start-signpost PERF_{PREFILL,DECODE}` and the matching
`--end-signpost`.

## Artifacts

- `pytest_full.log`, `pytest_results.xml`, and `watcher_pytest_results.xml`:
  exact normal stdout plus normal and watcher correctness results.
- `watcher_remediation/generated/watcher/watcher.log`: 20734-line watcher log; signature
  scan found no fatal, assert, illegal-NoC, timeout, hang, or stuck-core report.
- `latency_*.json`: final 20-sample wall measurements.
- `candidate_*.json` and `candidate_split_failure.txt`: measured retained and
  rejected alternatives, including PCC, latency, and operator-contract
  failures. `candidate_seq128_remediation_matrix.json` records the complete
  functional control and repair matrix for the extra sequence-128 diagnostic;
  `candidate_moe_compute_single_card.json` records the exact North-geometry
  1x1 `moe_compute` trial.
- `prefill_*context*.json` and `decode_*context*.json`: capacity evidence.
- `tracy/**/_perf_report.csv`: filtered per-op CSVs.
- `tracy/**/_perf_table.txt`: human-readable per-op tables.
- `final_tt_smi_list.log`: final inventory of four healthy Blackhole p300c
  boards.

The repeated nanobind leak diagnostic printed at Python shutdown is an existing
binding-level diagnostic also present in the functional stage; it occurs after
all tests pass and devices close normally.
