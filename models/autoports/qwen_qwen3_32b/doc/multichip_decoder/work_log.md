# Qwen3-32B multichip decoder work log

Date: 2026-07-17/18 UTC

Scope: `tt/multichip_decoder.py`, its tests, `doc/multichip_decoder`, and
`doc/context_contract.json`. No full-model or vLLM work was started.

## Inventory and hardware

- Read `$multichip`, `$tt-device-usage`, `$optimize`, `$graph-rewrite`,
  `$tt-enable-tracing`, `$autofix`, `$stage-review`, and `$shard-advise`.
- Preserved the unrelated pre-existing modification to
  `.agents/skills/forge-functional-decoder-from-ir/SKILL.md`; it is excluded
  from the stage commit.
- Inspected `tt/optimized_decoder.py`, its tests/evidence, compiler multichip
  provenance, and `doc/context_contract.json` before choosing the final plan.
- `tt-smi` found four Blackhole p300c devices, IDs 0-3. A 1x4 Ring open/close
  smoke and a shape-faithful BF16 AG/RS probe passed.

The pre-coding plan was fixed 1x4 TP=4 on axis 1, sharded layer boundaries,
Q/K/V head ownership 16/2/2, local intermediate width 6,400, BFP8 local cache,
BF16 Ring CCL, and two AG plus two RS per layer. Dense Qwen3 has no MoE path.
Rejected strategic alternatives and calculated shapes are in `README.md`.

## Implementation and review repair history

- Added `MultichipDecoder(OptimizedDecoder)` with real rank-local TP weights,
  not four replicated decoder wrappers.
- Kept interleaved prefill and DRAM-sharded decode weight layouts resident.
- Added sharded stack I/O, local contiguous/paged caches, one authoritative
  page table, stable replicated positions, and persistent traced collectives.
- Added a device-only 16Q/2KV Blackhole SDPA workaround.
- Added ownership-aware prefill deallocation and immediate row-parallel chunk
  reduce-scatter.
- Added independent QKV/O/gate/down core and `in0_block_w` controls. QKV now
  owns an explicit input memory config, so alternate role grids measure their
  required reshard instead of relying on the norm grid accidentally.
- Removed the ambiguous `chunk_page_table` argument.
- Added a measured compiler-provenance replicated-boundary/two-all-reduce
  candidate and rejected it in favor of the selected sharded boundary.
- Corrected capacity accounting for 64-token paged rounding and both TILE and
  ROW_MAJOR RoPE layouts.
- Added paged trace refresh, exact Watcher log retention, program/shard details,
  and role-specific profiler controls in response to the first independent
  review.

Final source hashes before documentation-only review edits:

- `tt/multichip_decoder.py`:
  `7a1e67a0215117c3a746d93de7459e135c8004e5855eb87398a1c45d1f833a83`
- `tests/test_multichip_decoder.py`:
  `d8892ea12d1bbb2689fe3d4156ee8ffc0408921d6e313c2d51f3663e9fcc4806`

## Correctness, cache, stack, and trace gates

Static fallback audit:

```bash
pytest -q \
  models/autoports/qwen_qwen3_32b/tests/test_multichip_decoder.py::test_multichip_contract_is_optimized_owned_and_host_free
```

Synthetic non-aligned and stacked-decoder gate:

```bash
TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 pytest -q \
  models/autoports/qwen_qwen3_32b/tests/test_multichip_decoder.py::test_multichip_synthetic_non_aligned_prefill_matches_hf
```

This covers logical length 31, contiguous and paged caches, page permutation,
stacked decoder handoff, positions 32/33 under one trace, and ten-replay
determinism. The final evidence values are summarized in `README.md`.

The independent optimized baseline was captured with:

```bash
QWEN3_32B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 \
QWEN3_32B_MULTICHIP_BASELINE_PATH=models/autoports/qwen_qwen3_32b/doc/multichip_decoder/artifacts/optimized_single_chip_real_layer32.pt \
TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 pytest -q \
  models/autoports/qwen_qwen3_32b/tests/test_multichip_decoder.py::test_capture_real_optimized_single_chip_baseline
```

The local ignored `.pt` is 12,521,607 bytes, SHA256
`f59eecf4b142ccf1be6f30c29a518c8a1e1294324bb0736c7fe9ba0230923bd4`.

Real layer-32 gate:

```bash
QWEN3_32B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 \
QWEN3_32B_MULTICHIP_BASELINE_PATH=models/autoports/qwen_qwen3_32b/doc/multichip_decoder/artifacts/optimized_single_chip_real_layer32.pt \
TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 pytest -q \
  models/autoports/qwen_qwen3_32b/tests/test_multichip_decoder.py::test_multichip_real_layer_matches_optimized_single_chip_baseline
```

The final Watcher run of this gate measured query 0.99993715, SDPA/concat
0.99966767, attention residual 0.99999055, prefill 0.99999995, decode
0.99997244, prefill K/V 0.99998010/0.99996648, and decode K/V
0.99993595/0.99990264.

Paged trace refresh gate:

```bash
QWEN3_32B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 \
pytest -q \
  models/autoports/qwen_qwen3_32b/tests/test_multichip_decoder.py::test_multichip_paged_trace_refresh_matches_eager
```

It prefills length 64, captures at position 64, refreshes the page table in
place, and advances to position 65. Unchanged, remapped, and advanced outputs
and K/V updates all have PCC 1.0. Artifact:
`results/paged_trace_refresh.json`.

## Topology and tuning evidence

Compiler-provenance comparison:

```bash
QWEN3_32B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 \
QWEN3_32B_MULTICHIP_RUN_TOPOLOGY=1 \
TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 pytest -q \
  models/autoports/qwen_qwen3_32b/tests/test_multichip_decoder.py::test_multichip_compiler_provenance_topology
```

Selected sharded-boundary trace: 0.629466 ms. Replicated-boundary/two-all-reduce
provenance: 1.860160 ms warmed eager. Output and cache PCC are 1.0. An earlier
attempt to trace the provenance family made no progress for over four minutes;
the process was terminated, chip 0 was reset with `tt-smi -r all`, and all four
boards passed fresh discovery. The selected family never stalled. Artifact:
`results/topology_family_benchmark.json`.

The common role-sweep command was:

```bash
QWEN3_32B_REAL_WEIGHT_DIR=<checkpoint> \
QWEN3_32B_MULTICHIP_BASELINE_PATH=<baseline.pt> \
QWEN3_32B_MULTICHIP_RUN_PERF=1 \
QWEN3_32B_MULTICHIP_{QKV|O|GATE|DOWN}_CORES=<candidate> \
QWEN3_32B_MULTICHIP_{QKV|O|GATE|DOWN}_IN0=<candidate> \
QWEN3_32B_MULTICHIP_RESULT_NAME=<artifact>.json \
pytest -q \
  models/autoports/qwen_qwen3_32b/tests/test_multichip_decoder.py::test_multichip_warmed_prefill_and_traced_decode
```

All legal candidates passed PCC. QKV10 was rerun after adding its explicit
input reshard and measured 0.636208 ms. The O8 candidate led at 0.629554 ms.
Three alternating confirmations used 200 replays/trial and measured O16
`0.631343/0.631262/0.631355` versus O8
`0.629469/0.629486/0.629413`; O8 became the default. Other role results and
HiFi2 advice candidates are retained under `results/role_*.json` and
`results/advice_*.json`.

Fused probes:

```bash
QWEN3_32B_REAL_WEIGHT_DIR=<checkpoint> \
QWEN3_32B_MULTICHIP_RUN_FUSED_PROBES=1 pytest -q \
  models/autoports/qwen_qwen3_32b/tests/test_multichip_decoder.py::test_multichip_fused_collective_candidates
```

Fused matmul+RS is correct at PCC 0.99997153 but slows the whole layer to
0.768530 ms. Fused AG+matmul is shape-valid but incorrect at PCC 0.00440685.

## Final wall performance

```bash
QWEN3_32B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 \
QWEN3_32B_MULTICHIP_BASELINE_PATH=models/autoports/qwen_qwen3_32b/doc/multichip_decoder/artifacts/optimized_single_chip_real_layer32.pt \
QWEN3_32B_MULTICHIP_RUN_PERF=1 \
QWEN3_32B_MULTICHIP_RESULT_NAME=final_selected_o8.json \
QWEN3_32B_MULTICHIP_DECODE_REPLAYS=200 \
QWEN3_32B_MULTICHIP_DECODE_TRIALS=9 \
QWEN3_32B_MULTICHIP_PREFILL_TRIALS=9 pytest -q \
  models/autoports/qwen_qwen3_32b/tests/test_multichip_decoder.py::test_multichip_warmed_prefill_and_traced_decode
```

- Single-chip: 5.502352957 ms prefill; 1.217318163 ms traced decode.
- Final 1x4: 3.127878997 ms prefill; 0.629460900 ms traced decode.
- Speedup: 1.759132x prefill; 1.933906x decode.
- Four-device efficiency: 43.9783% prefill; 48.3476% decode.

## Tracy and tt-perf-report

```bash
QWEN3_32B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 \
QWEN3_32B_MULTICHIP_RUN_PROFILE=1 \
TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
python -m tracy -r -p -v \
  -o models/autoports/qwen_qwen3_32b/doc/multichip_decoder/tracy_final_o8 \
  -m pytest -q \
  models/autoports/qwen_qwen3_32b/tests/test_multichip_decoder.py::test_profile_selected_multichip_decoder
```

Raw CSV:
`tracy_final_o8/reports/2026_07_18_02_10_04/ops_perf_results_2026_07_18_02_10_04.csv`,
SHA256 `446de16e5ee09185cd4d0824ac25a95a4f2cff9ea256bfbf0a10457924cc5714`.

Filtered CSV SHA256 values are
`32ab91a07bc282d5b2bf7be1adbe5addc7777cb2d4bdeedea33f83fe64617ec8`
for decode and
`03af5dff3bc5dafe96ec7a24199424493f583393cddd833d1f56621355d74e85`
for prefill. `perf_report.md` contains tables and advice dispositions.

## Shard-advisor attempt

```bash
cd /home/mvasiljevic/tt-mlir
source /home/mvasiljevic/tt-metal/.agents/skills/shard-advise/scripts/bootstrap.sh
ttnn-advise capture --help
```

The current pinned advisor fails before capture with an `_ttnn.so` undefined
`moe_compute` symbol. `shard_advisor_status.md` records the exact blocker and
explains how the prior optimized-decoder report was used. Building tt-mlir is
outside this stage per the skill contract.

## Full-stack capacity and context

```bash
QWEN3_32B_REAL_WEIGHT_DIR=<checkpoint> \
QWEN3_32B_MULTICHIP_RUN_CAPACITY=1 \
QWEN3_32B_MULTICHIP_CAPACITY_SEQUENCE=12352 \
QWEN3_32B_MULTICHIP_CAPACITY_EXPECT=pass pytest -q \
  models/autoports/qwen_qwen3_32b/tests/test_multichip_decoder.py::test_multichip_full_stack_capacity

QWEN3_32B_REAL_WEIGHT_DIR=<checkpoint> \
QWEN3_32B_MULTICHIP_RUN_CAPACITY=1 \
QWEN3_32B_MULTICHIP_CAPACITY_SEQUENCE=12353 \
QWEN3_32B_MULTICHIP_CAPACITY_EXPECT=fail pytest -q \
  models/autoports/qwen_qwen3_32b/tests/test_multichip_decoder.py::test_multichip_full_stack_capacity
```

Logical 12,352 fits. Logical 12,353 rounds to 12,416 physical cache positions
and fails the prefill live-set allocation. Exact byte accounting is in
`doc/context_contract.json` and the two capacity JSON artifacts.

## Watcher and health

```bash
QWEN3_32B_REAL_WEIGHT_DIR=<checkpoint> \
QWEN3_32B_MULTICHIP_BASELINE_PATH=<baseline.pt> \
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 pytest -q \
  models/autoports/qwen_qwen3_32b/tests/test_multichip_decoder.py::test_multichip_real_layer_matches_optimized_single_chip_baseline
```

The exact 2,243-line log is retained at `results/watcher_clean.log`, SHA256
`14ee2af4ec29d339c405fa59c3acf73059814f40489b0c847720c9be735d26e4`,
with no fault signatures. ETH instrumentation is disabled only because full
Watcher grows active-Ethernet firmware beyond the kernel-config buffer.

## Final review and commit

The first independent `$stage-review` returned `more-work-needed`; every
finding above was addressed. The consolidated final invocation retained at
`final_gate.xml` passed static audit, synthetic non-aligned/stacked/cache/trace,
real optimized-baseline, and paged trace-refresh gates: **4 passed in 43.25s**.
The independent rereview returned **clean-pass** with no remaining findings.
Its verification scope and corroborated evidence are recorded in
`stage_review.md`.

The stage-only implementation/evidence commit SHA is appended after the first
local commit. The final documentation commit cannot contain its own SHA, so
that SHA is reported in the handoff. No push is performed.
