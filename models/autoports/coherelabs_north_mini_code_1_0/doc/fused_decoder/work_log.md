# Fused decoder work log

## 2026-07-31

- Read and applied the `graph-fusing` and `tt-device-usage` stage procedures.
- Confirmed four local Blackhole p300c boards with `tt-smi`; opened only a 1x1
  mesh. The bounded mesh smoke passed before work. All hardware commands were
  serialized, watcher and profiler were never combined, and no reset was
  needed.
- Audited the functional source and baseline Tracy reports for all three layer
  kinds at prefill batch 1, decode batch 1, and serving decode batch 32.
- Added `FusedDecoder`, a fused performance wrapper, a fused capacity wrapper,
  and a fused regression suite. Work stayed within Stage-02 decoder code,
  tests, and documentation.
- Packed dense and expert gate/up weights once at construction, deallocated the
  original device tensors, fused SiLU into multiply, implemented exact top-8
  sparse expert dispatch for sub-tile token counts, and implemented packed
  tile-efficient MoE for 32 or more tokens.
- Replaced two decode paged-cache updates with the fused update. Its first run
  exposed overlapping K/V core grids. A disjoint V core set plus the required
  V reshard satisfied the op contract, passed PCC at batch 1 and 32, and
  improved latency.
- `ttnn.split` on the packed 6144-wide dense output was rejected after a
  Blackhole JIT failure (`single_tile_size_bytes` undeclared). Equivalent width
  slices passed and were faster than the functional baseline.
- Corrected the sparse-down geometry after the per-token mask hit the current
  A-sparse per-expert-mask constraint at batch 32. All-expert sparse down is
  exact because inactive gate/up rows are already zero.
- Measured exact sparse batch-32 decode at 13.395 ms, slower than the 11.122 ms
  functional baseline. The packed all-expert path measured about 8.29 ms and
  was selected for 32+ tokens. Exact sparse remains selected below one tile.
- Tried `fast_reduce_nc` explicitly for batch-32 route reduction:
  8.293376 ms versus 8.293310 ms with `sum`; reverted. Current TTNN lowers the
  chosen reduction to its fast kernel when the geometry permits.
- Assessed combined Q/K RoPE, fused routers, `moe_compute`, residual/RMSNorm,
  structural movement, and every graph-fusing pattern. Exact contracts and
  decisions are recorded in `graph_fusing_audit.md`.
- The first independent stage review returned `more-work-needed`: it requested
  assessment of the fused weighted-reduction operator and cross-branch dense
  packing, a direct dense batch-32 traced PCC test, machine-readable rejection
  evidence, and a corrected topology description.
- Tested `deepseek_moe_fast_reduce_nc_fused`. It preserved serving batch-32 PCC
  0.998193 and improved final sliding/full latency to 8.273/8.279 ms, so the
  exact 32-token path retains it. It was rejected for prefill because the
  measured length-128 workload, despite improving wall latency from 10.080 to
  9.974 ms, failed the active-expert reference at PCC 0.408818. Length 33 also
  fell to PCC 0.758886, and the 1024-token chunk exceeded available per-bank L1.
- Tested a single 11264-wide QKV+gate/up projection for dense layers. PCC
  passed, but prefill, batch-1 decode, and batch-32 decode all regressed
  (0.610, 0.329, and 5.886 ms versus 0.580, 0.320, and 5.698 ms), so it was
  reverted. Raw 20-sample evidence is retained.
- Added a direct fused dense batch-32 trace-replay PCC test; it passed at
  0.99985352. Candidate summaries now record all measured decisions.
- The fresh re-review identified the missing measured-length-128 PCC for the
  prefill fused-reduction candidate. A direct hardware test reproduced the
  candidate and rejected it at PCC 0.408818; the selected fallback remains the
  final prefill path and the evidence is now machine-readable.
- Per the graph-fusing adaptation rule, also tested sequence 128 as four
  independent known-good 32-token fused reductions plus concat. It restored
  fallback-equivalent diagnostic numerics but measured 10.410 ms, 3.3% slower
  than the selected 10.080 ms path, and was rejected with a machine-readable
  result.
- Controlled the new all-token sequence-128 synthetic diagnostic against the
  unchanged `FunctionalDecoder`: functional and the selected fused fallback
  both produced exactly PCC 0.98764764. Thus this additional stress is not an
  established >=0.995 functional acceptance case, while the whole-shape fused
  candidate's PCC 0.408818 is a decisive regression and the four-tile
  adaptation is equivalent but slower. The established sequence-1025 and
  non-aligned sequence-33 MoE acceptance gates remain above 0.995.
- Also rejected a one-row exact hybrid, four exact-sparse tiles, whole-shape
  exact sparse (256 blocks exceeded 64 cores), HiFi4 FP32 accumulation, and a
  separate-projection control. The complete matrix is retained in
  `candidate_seq128_remediation_matrix.json`; all experimental code was
  reverted.
- Corrected the earlier source-only dismissal of `moe_compute`: its
  `compute_only=True` path does run on a 1x1 Blackhole mesh. The exact North
  E=128, tokens=32, top-k=8, hidden=2048, intermediate=768 shape ran, but the
  operation's fixed BFP4 expert weights produced PCC 0.992762/0.990510, below
  the required 0.995 even before the external routing-score combine. It was
  rejected with exact command and geometry in
  `candidate_moe_compute_single_card.json`.

## Validation

Normal correctness:

```text
pytest -q -s models/autoports/coherelabs_north_mini_code_1_0/tests/test_fused_decoder.py
Result: 19 passed in 66.79s
Artifacts: pytest_full.log, pytest_results.xml
```

PCC highlights:

```text
dense prefill 33                         0.99971542
dense paged trace replay                 0.99976456
sliding MoE prefill 1025                 0.99957275
full/no-RoPE MoE prefill 33              0.99976320
sliding populated-history trace          0.99953302
sliding MoE batch-1 trace                0.99983603
sliding MoE batch-32 trace               0.99819297
full/no-RoPE MoE batch-1 trace           0.99982732
dense batch-32 trace                     0.99985352
official layer-1 real-weight decode      0.99975057
```

Watcher correctness:

```text
TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1 \
TT_METAL_LOGS_PATH=<artifact>/watcher_remediation \
  pytest -q -s models/autoports/coherelabs_north_mini_code_1_0/tests/test_fused_decoder.py
Result: 19 passed in 74.36s
Artifacts: watcher_pytest_results.xml, watcher_remediation/generated/watcher/watcher.log
```

The 20734-line watcher log was scanned case-insensitively for fatal, exception,
assert, illegal/invalid NoC, timeout, hang, stuck, and mailbox-error signatures;
none matched. Final `tt-smi` listed all four boards normally.

Capacity:

```text
prefill layer 0, b1, 500000: finite, 159575.45 ms
prefill layer 1, b1, 499999: finite,  17107.08 ms
prefill layer 4, b1, 499999: finite, 173316.15 ms
decode layer 0, b1, position 499999: finite traced replay
decode layer 1, b1, position 499999: finite traced replay
decode layer 4, b1, position 499999: finite traced replay
decode layer 0, b32, position 499999: finite, 133.72 ms traced replay
```

This preserves the 500000-token BF16 paged-cache contract. No
`context_contract.json` change is needed.

Final warmed wall/device performance:

| Kind | Functional prefill | Fused prefill | Functional decode | Fused decode |
|---|---:|---:|---:|---:|
| dense b1 | 0.636 ms / 586 us | 0.580 ms / 542 us | 0.356 ms / 338 us | 0.320 ms / 302 us |
| sliding MoE b1 | 14.908 ms / 14644 us | 10.080 ms / 9998 us | 9.528 ms / 9452 us | 2.130 ms / 2103 us |
| full MoE b1 | 14.655 ms / 14567 us | 10.067 ms / 9961 us | 9.524 ms / 9439 us | 2.131 ms / 2088 us |
| dense b32 | n/a | n/a | 6.652 ms / 6614 us | 5.698 ms / 5663 us |
| sliding MoE b32 | n/a | n/a | 11.122 ms / 11084 us | 8.273 ms / 8248 us |
| full MoE b32 | n/a | n/a | 11.129 ms / 11077 us | 8.279 ms / 8234 us |

Wall values use five warmups and 20 measured iterations. Decode is trace replay,
so those same runs provide repeated-run coverage. Nine Tracy captures were
filtered between exact prefill/decode signposts with `tt-perf-report`; every
table reports zero host ops. Canonical raw ops CSVs, filtered CSVs, tables, and
stacked summaries are retained under `tracy/`. Multi-gigabyte duplicated Tracy
scratch logs and raw device-profile copies were pruned after the canonical ops
CSVs were secured.

## Independent review

- The initial review and first re-review returned `more-work-needed`; their
  findings drove the dense cross-pack, weighted-reduction, sequence-128
  control, and single-card `moe_compute` experiments recorded above.
- A fresh final reviewer inspected the fully remediated source and artifacts
  and returned `clean-pass` in `stage_clean_review.md`.

## Local checkpoint

- Stage implementation, tests, documentation, and evidence:
  `2e7e734ba3d` (`Add fused North Mini decoder stage`).
- This SHA-record update is a documentation-only follow-up. Neither commit is
  pushed.
