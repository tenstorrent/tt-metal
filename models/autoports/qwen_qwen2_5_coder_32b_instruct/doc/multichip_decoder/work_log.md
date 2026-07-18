# Multichip decoder work log

All real-weight evidence uses `Qwen/Qwen2.5-Coder-32B-Instruct` snapshot
revision `381fc969f78efac66bc87ff7ddeadb7e73c218a7`, locally at
`/home/mvasiljevic/hf-cache/hub/models--Qwen--Qwen2.5-Coder-32B-Instruct/snapshots/381fc969f78efac66bc87ff7ddeadb7e73c218a7`.

## 2026-07-18: stage start and strategy lock

- Read `$multichip`, `$tt-device-usage`, `$optimize`, `$tt-enable-tracing`,
  `$stage-review`, `$autofix`, and `tech_reports/LLMs/llms.md` section 3.3.
- Preserved unrelated dirty file `.agents/skills/forge-functional-decoder-from-ir/SKILL.md`.
- `timeout 60 tt-smi -ls --local`: four Blackhole p300c devices visible.
- Mesh smoke command:

  ```bash
  timeout 120 python_env/bin/python - <<'PY'
  import ttnn
  ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
  mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
  print(ttnn.get_num_devices(), mesh.shape, mesh.get_num_devices(), ttnn.get_arch_name())
  ttnn.close_mesh_device(mesh)
  ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
  print("MESH_SMOKE_OK")
  PY
  ```

  Result: `4`, `(1,4)`, `4`, `blackhole`, compute/storage grid `11x10`,
  `MESH_SMOKE_OK`.
- Read compiler provenance. It authoritatively specifies 1x4 TP, Q/K/V and
  gate/up column sharding, O/down row sharding, local 10Q/2KV heads, local
  cache ownership, and Ring reductions.
- Read the completed optimized decoder, correctness tests, advisor report, and
  profiler summaries. Dense Qwen2 is the only meaningful layer kind.
- Locked the pre-implementation plan recorded in `README.md`: 1x4 TP,
  sharded layer residual, local KV heads, internal QKV/MLP padding, common async
  CCL/semaphore helpers, and BF16 initial CCL.
- Nearby Qwen3-32B TP4 stage commit `9a49fbb0bf1` is used as a proven Blackhole
  API/topology reference. Qwen2-specific head counts, QKV biases, absent Q/K
  norms, dimensions, precision baseline, and padding are independently adapted.
- First construction probe rejected QKV `1792→1920` and MLP `6912→7040` before
  running model kernels: DRAM-width sharding requires `N % (8 banks * 32) == 0`.
  Revised to the smallest jointly legal `1792→2048` and `6912→7168` padding,
  enabling exact 16-core QKV/gate/down geometry. This supersedes the initial
  arithmetic in the same pre-final strategy phase.
- The revised non-aligned prefill and direct stacked-layer contract ran before
  decode: prefill PCC `0.99968302`, key/value cache PCC
  `0.99978215/0.99978918`, and stacked handoff PCC `0.99922952`.
- Packed DRAM-sharded decode gate/up then failed its exact L1 check: static CBs
  requested `2,094,848 B` versus `1,572,864 B`. Prefill packing remains valid;
  decode is changed to two phase-specific padded DRAM-sharded projections so
  each program's output/CB width is halved. This is the required packed-vs-split
  contract experiment, not a functional fallback.

## 2026-07-18: implementation and correctness

- Added `tt/multichip_decoder.py`, using `OptimizedDecoder` as its baseline and
  owning the actual TP4 prefill/decode path. Runtime forward methods contain no
  Torch conversion or parent-decoder fallback.
- Selected load-time rank packing and zero padding: QKV `1792→2048`, MLP
  `6912→7168`, local 10Q/2KV heads, BF16 `[1,32,L,1280]` stack boundary.
- Correctness command:

  ```bash
  python_env/bin/python -m pytest \
    models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_multichip_decoder.py::test_multichip_synthetic_non_aligned_prefill_decode_and_paged_cache \
    models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_multichip_decoder.py::test_multichip_paged_trace_refresh_matches_eager -q -s
  ```

  Result: `2 passed`. Length-31 selected-policy PCC: prefill `0.99655177`,
  decode `0.99644716`, direct prefill stack handoff `0.99118541`; the
  two-instance decode handoff with shared RoPE/position/CCL workspace is
  `0.99073898`, with second-layer K/V above `0.9962`. First-layer K/V is at
  least `0.99978215`. Paged/contiguous PCC `1.0`. Ten trace replays are bitwise
  equal. Position refresh 32→33 passes. Physical-page remap plus 64→65
  position advance is eager/traced PCC `1.0` for output and K/V.
- Captured a one-device baseline in a separate process, then compared the real
  TP4 layer:

  ```bash
  QWEN2_5_CODER_32B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/hub/models--Qwen--Qwen2.5-Coder-32B-Instruct/snapshots/381fc969f78efac66bc87ff7ddeadb7e73c218a7 \
  QWEN2_5_CODER_32B_MULTICHIP_BASELINE_PATH=/tmp/qwen2_5_coder_32b_optimized_baseline.pt \
  python_env/bin/python -m pytest \
    models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_multichip_decoder.py::test_capture_real_optimized_single_chip_baseline -q -s

  QWEN2_5_CODER_32B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/hub/models--Qwen--Qwen2.5-Coder-32B-Instruct/snapshots/381fc969f78efac66bc87ff7ddeadb7e73c218a7 \
  QWEN2_5_CODER_32B_MULTICHIP_BASELINE_PATH=/tmp/qwen2_5_coder_32b_optimized_baseline.pt \
  python_env/bin/python -m pytest \
    models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_multichip_decoder.py::test_real_multichip_matches_optimized_single_chip_baseline -q -s
  ```

  Final mixed-policy real PCC: prefill `0.99339170`, decode `0.99400587`,
  prefill K/V `0.99983827/0.99966163`, decode K/V
  `0.99990991/0.99981420`. Layer 32 covers the model's only homogeneous dense
  decoder kind; this model has no MoE experts.

## 2026-07-18: optimization and topology audit

- The `ttnn-advise` import path was corrected and retried, but
  `libTTMLIRRuntime.so` failed on an undefined `moe_compute` symbol. Recorded in
  `shard_advisor_status.md`; direct TP4 measurements were used.
- Swept all-BFP8, mixed BFP8/BFP4, all-BFP4, 16/32-core programs, independent
  O/down grids, BF16/BFP8 CCL, persistent/non-persistent CCL, and fused
  matmul+reduce-scatter. After review, all ten candidates were run on
  the final physical graph (`QKV=2048`, `MLP=7168`) with the uniform protocol
  of seven prefill samples and seven groups of 100 decode replays. Every run
  retains all samples, shapes, and PCC in `results/candidate_*.json`.

  ```bash
  QWEN2_5_CODER_32B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/hub/models--Qwen--Qwen2.5-Coder-32B-Instruct/snapshots/381fc969f78efac66bc87ff7ddeadb7e73c218a7 \
  QWEN2_5_CODER_32B_MULTICHIP_BASELINE_PATH=/tmp/qwen2_5_coder_32b_optimized_baseline.pt \
  QWEN2_5_CODER_32B_MULTICHIP_RUN_PERF=multichip \
  QWEN2_5_CODER_32B_MULTICHIP_PREFILL_TRIALS=7 \
  QWEN2_5_CODER_32B_MULTICHIP_DECODE_TRIALS=7 \
  QWEN2_5_CODER_32B_MULTICHIP_DECODE_REPLAYS=100 \
  QWEN2_5_CODER_32B_MULTICHIP_RESULT_NAME=<candidate.json> \
  <candidate-specific precision/core/CCL overrides> \
  python_env/bin/python -m pytest \
    models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_multichip_decoder.py::test_real_multichip_warmed_prefill_and_traced_decode_perf -q -s
  ```

  Substitute the result name and exact overrides below; all omitted variables
  use the harness defaults.

  | Result artifact | Exact candidate overrides |
  | --- | --- |
  | `candidate_bfp8_hifi2_16c.json` | `QWEN2_5_CODER_32B_MULTICHIP_PRECISION=optimized_baseline QWEN2_5_CODER_32B_MULTICHIP_O_CORES=20` |
  | `candidate_mlp_bfp4_lofi_16c.json` | `QWEN2_5_CODER_32B_MULTICHIP_PRECISION=mlp_bfp4_lofi QWEN2_5_CODER_32B_MULTICHIP_O_CORES=20` |
  | `candidate_mlp_bfp4_lofi_o8.json` | `QWEN2_5_CODER_32B_MULTICHIP_PRECISION=mlp_bfp4_lofi QWEN2_5_CODER_32B_MULTICHIP_O_CORES=8` |
  | `candidate_mlp_bfp4_32c_o8.json` | `QWEN2_5_CODER_32B_MULTICHIP_PRECISION=mlp_bfp4_lofi QWEN2_5_CODER_32B_MULTICHIP_CORES=32 QWEN2_5_CODER_32B_MULTICHIP_DOWN_CORES=32 QWEN2_5_CODER_32B_MULTICHIP_O_CORES=8` |
  | `candidate_mlp_bfp4_o8_down32.json` | `QWEN2_5_CODER_32B_MULTICHIP_PRECISION=mlp_bfp4_lofi QWEN2_5_CODER_32B_MULTICHIP_DOWN_CORES=32 QWEN2_5_CODER_32B_MULTICHIP_O_CORES=8` |
  | `candidate_mlp_bfp4_o8_bfp8_ccl.json` | `QWEN2_5_CODER_32B_MULTICHIP_PRECISION=mlp_bfp4_lofi QWEN2_5_CODER_32B_MULTICHIP_O_CORES=8 QWEN2_5_CODER_32B_MULTICHIP_CCL_DTYPE=bfp8` |
  | `candidate_mlp_bfp4_o8_nonpersistent.json` | `QWEN2_5_CODER_32B_MULTICHIP_PRECISION=mlp_bfp4_lofi QWEN2_5_CODER_32B_MULTICHIP_O_CORES=8 QWEN2_5_CODER_32B_MULTICHIP_PERSISTENT_CCL=0` |
  | `candidate_mlp_bfp4_o8_fused_rs.json` | `QWEN2_5_CODER_32B_MULTICHIP_PRECISION=mlp_bfp4_lofi QWEN2_5_CODER_32B_MULTICHIP_O_CORES=8 QWEN2_5_CODER_32B_MULTICHIP_FUSED_RS=1` |
  | `candidate_all_bfp4_lofi_rejected.json` | `QWEN2_5_CODER_32B_MULTICHIP_PRECISION=all_bfp4_lofi QWEN2_5_CODER_32B_MULTICHIP_O_CORES=8 QWEN2_5_CODER_32B_MULTICHIP_PCC_THRESHOLD=0.98` (measurement only; rejected by the stage's 0.99 gate) |
  | `candidate_all_bfp4_attention_hifi2.json` | `QWEN2_5_CODER_32B_MULTICHIP_PRECISION=all_bfp4_attention_hifi2 QWEN2_5_CODER_32B_MULTICHIP_O_CORES=8 QWEN2_5_CODER_32B_MULTICHIP_PCC_THRESHOLD=0.98` (measurement only; rejected by the stage's 0.99 gate) |
- All-BFP4 decode was fastest (`0.766400 ms`) but rejected because real prefill
  PCC `0.98979556` misses the `0.99` gate. Mixed MLP BFP4/LoFi retains
  `0.99339170/0.99400587` PCC. O on eight cores is the best decode geometry.
- The follow-up all-BFP4 attention-HiFi2 isolation measures
  `3.280096/0.790089 ms` and improves prefill PCC to `0.98991918`, but still
  misses `0.99`; the near-threshold rejection is not an untested LoFi effect.
- Fused matmul+reduce-scatter is correct but slower (`0.911753 ms` decode versus
  `0.791939 ms`; `3.354337 ms` prefill versus `3.341834 ms`). BFP8 CCL and
  non-persistent CCL also regress decode.
- Measured compiler provenance:

  ```bash
  QWEN2_5_CODER_32B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/hub/models--Qwen--Qwen2.5-Coder-32B-Instruct/snapshots/381fc969f78efac66bc87ff7ddeadb7e73c218a7 \
  QWEN2_5_CODER_32B_MULTICHIP_RUN_TOPOLOGY=1 \
  python_env/bin/python -m pytest \
    models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_multichip_decoder.py::test_multichip_compiler_provenance_topology -q -s
  ```

  Result: output/K/V and trace/eager PCC `1.0`, with bitwise-stable traces.
  Final persistent-buffer measurements are selected/provenance eager
  `2.053337/2.037526 ms` and selected/provenance traced
  `0.791554/0.869899 ms`. The selected trace is `1.09898x` faster; provenance
  is not rejected from an eager/trace mismatch or non-persistent handicap.
- A shape-faithful fused all-gather+column-sharded-O probe first exposed the
  API's full-gather shard requirement, then executed with the legal
  `[32,640]` shard. Review expanded this to all 24 global K-rank weight
  permutations. The best was `[0,2,1,3]` at PCC `0.249461`, far below `0.99`:
  Ring gather K order is output-rank-relative, while the shared column-sharded
  weight exposes only one global K packing. It was rejected before performance
  acceptance. The executable rejection is `fused_all_gather_o_probe.json`.
- Distributed RMSNorm was rejected by dependency and measured lower bound:
  dense-K QKV and gate/up still need both hidden gathers, while the distributed
  norm adds a statistics collective to the profiled 7-us norm.

## 2026-07-18: stage-review prefill geometry sweep

- The fresh rereviewer found that the original ten-candidate campaign varied
  decode programs but left prefill fixed at 10x10 with K-block limit 10. The
  profiler marks each prefill matmul slow, so prefill grid/block controls were
  added to the constructor and performance harness. Every retained successful
  candidate uses the selected BFP4/LoFi policy and real layer-32 weights.
- Primary command (substitute grid, block, and result name):

  ```bash
  QWEN2_5_CODER_32B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/hub/models--Qwen--Qwen2.5-Coder-32B-Instruct/snapshots/381fc969f78efac66bc87ff7ddeadb7e73c218a7 \
  QWEN2_5_CODER_32B_MULTICHIP_BASELINE_PATH=/tmp/qwen2_5_coder_32b_optimized_baseline.pt \
  QWEN2_5_CODER_32B_MULTICHIP_RUN_PERF=multichip \
  QWEN2_5_CODER_32B_MULTICHIP_PREFILL_TRIALS=7 \
  QWEN2_5_CODER_32B_MULTICHIP_DECODE_TRIALS=7 \
  QWEN2_5_CODER_32B_MULTICHIP_DECODE_REPLAYS=100 \
  QWEN2_5_CODER_32B_MULTICHIP_PREFILL_GRID_X=<8-or-10> \
  QWEN2_5_CODER_32B_MULTICHIP_PREFILL_GRID_Y=10 \
  QWEN2_5_CODER_32B_MULTICHIP_PREFILL_IN0_BLOCK_W=<10-16-20-32> \
  QWEN2_5_CODER_32B_MULTICHIP_RESULT_NAME=<candidate_prefill.json> \
  python_env/bin/python -m pytest \
    models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_multichip_decoder.py::test_real_multichip_warmed_prefill_and_traced_decode_perf -q -s
  ```

- Primary medians were: 10x10 block 10/16/20 =
  `3.322598/3.288673/3.381160 ms`; 8x10 block 10/16 =
  `3.489707/3.560582 ms`. All successful PCC values exceed `0.9932`.
  10x10/block32 fails packed gate/up at `2,216,704 B` CB versus
  `1,572,864 B` L1. 8x10 block20/block32 fail at
  `1,794,816/2,667,264 B`.
- Because the primary 10x10 block-10/block-16 separation was small, both were
  repeated with 21 warmed prefill trials (decode reduced to one 10-replay
  correctness guard). Medians were `3.233757 ms` for block 10 and
  `3.295315 ms` for block 16, a `1.90%` block-16 regression. The original
  10x10/block-10 program remains selected, so the retained Tracy profile is
  still the exact accepted device path. Evidence is in
  `results/candidate_prefill_*.json` and
  `results/prefill_geometry_sweep.json`.

## 2026-07-18: final performance and profiler

- Serialized single-device command used
  `QWEN2_5_CODER_32B_MULTICHIP_RUN_PERF=single`; final TP4 used
  `QWEN2_5_CODER_32B_MULTICHIP_RUN_PERF=multichip`, seven prefill trials and
  seven groups of 100 trace replays. Both use real layer 32, batch 32, length
  17, and the independent baseline activation.
- Final single-chip optimized: prefill `9.99722 ms`, traced decode
  `1.930812 ms`. Final reproduced TP4 after the prefill sweep: prefill
  `3.343838 ms`, traced decode `0.791826 ms`, with bitwise-equal repeated trace
  output. Speedup/efficiency: prefill `2.990x/74.7%`; decode
  `2.438x/61.0%`.
- Tracy command, deliberately without Watcher:

  ```bash
  QWEN2_5_CODER_32B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/hub/models--Qwen--Qwen2.5-Coder-32B-Instruct/snapshots/381fc969f78efac66bc87ff7ddeadb7e73c218a7 \
  QWEN2_5_CODER_32B_MULTICHIP_RUN_PROFILE=1 \
  python_env/bin/python -m tracy -r -p -v \
    -o models/autoports/qwen_qwen2_5_coder_32b_instruct/doc/multichip_decoder/tracy/layer32 \
    -n selected_mixed -m pytest \
    models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_multichip_decoder.py::test_profile_selected_multichip_decoder -q -s
  ```

  Reduced with `tt-perf-report`. Retained human tables, op CSVs, summary
  CSV/PNGs, hashes, and exact provenance under `tracy/layer32/`. Decode:
  70 ops, `630 us` merged device work, `114 GB/s` modeled DRAM, `68 us` total
  Ring collectives. Prefill: 222 ops, `2006 us` device work, `61 GB/s` modeled
  DRAM. Raw 367-MB Tracy logs/databases and combined CSV were removed only
  after deterministic reduction; provenance contains the rerun command.

## 2026-07-18: capacity, Watcher, and final audits

- Exact capacity probes, including future full-model resident tables but no
  out-of-scope embedding/logit operations:

  ```bash
  QWEN2_5_CODER_32B_MULTICHIP_RUN_CAPACITY=1 \
  QWEN2_5_CODER_32B_MULTICHIP_CAPACITY_SEQUENCE=12224 \
  QWEN2_5_CODER_32B_MULTICHIP_CAPACITY_EXPECT=pass \
  python_env/bin/python -m pytest \
    models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_multichip_decoder.py::test_multichip_full_stack_capacity -q -s

  QWEN2_5_CODER_32B_MULTICHIP_RUN_CAPACITY=1 \
  QWEN2_5_CODER_32B_MULTICHIP_CAPACITY_SEQUENCE=12225 \
  QWEN2_5_CODER_32B_MULTICHIP_CAPACITY_EXPECT=fail \
  python_env/bin/python -m pytest \
    models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_multichip_decoder.py::test_multichip_full_stack_capacity -q -s
  ```

  The allocator-faithful test reserves each layer's decode projections as five
  separate width-sharded tensors and each layer's prefill projections as four
  separate interleaved tensors, plus 128 local K/V cache tensors, 128 separately
  tile-padded rank-1 norm tensors, and 128 bias buffers. Stack-global RoPE tables,
  decode position buffers, page table, and persistent CCL workspace are reserved
  once, matching the implemented sharing APIs. It also reserves
  `438,691,840 B/device` for a future TP4 BFP8 embedding, untied LM head, and
  final norm. Logical length `12224` passes at physical length `12224`: static
  residency is `24,350,738,432 B/device`, the prefill peak is
  `32,562,137,088 B/device`, and `16,593,920 B/device` remains free. Logical
  length `12225` pads to `12288` and fails its modeled prefill live set:
  `200,802,304 B/bank` is requested with `191,272,064 B/bank` free.
  `context_contract.json` therefore records `12224` as the measured largest
  feasible batch-32 logical context. At `32768`, local BFP8 KV alone is
  `36,507,222,016 B/device`, over physical DRAM.
- Full Watcher instrumentation was attempted first and failed before test
  execution because active-Ethernet firmware was `27,920 B` versus the
  `25,600 B` kernel-config buffer. Per `$tt-device-usage`, reran separately
  with Ethernet Watcher disabled:

  ```bash
  TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
  QWEN2_5_CODER_32B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/hub/models--Qwen--Qwen2.5-Coder-32B-Instruct/snapshots/381fc969f78efac66bc87ff7ddeadb7e73c218a7 \
  QWEN2_5_CODER_32B_MULTICHIP_BASELINE_PATH=/tmp/qwen2_5_coder_32b_optimized_baseline.pt \
  python_env/bin/python -m pytest \
    models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_multichip_decoder.py::test_real_multichip_matches_optimized_single_chip_baseline -q -s
  ```

  Result: pass; this is explicitly recorded as a partial Ethernet gate
  exception. Worker/dispatch coverage has no
  `error/assert/hang/stuck/timeout` matches, SHA256
  `2932c0b68f98e778b902ee446b14efa8fc0d97d13d4a42b6f8823c77995c0036`.
- Static fallback audit checks all forward-path methods for Torch conversions,
  parent calls, or host fallback. Device health was checked with `tt-smi -s`
  before final hardware runs; four p300c boards reported healthy DRAM.
- The recurring `/dev/shm` warning reports 16,777,216 bytes requested with
  about 17.5 MB available. It did not abort or change any accepted result.

## 2026-07-18: first independent stage review and remediation

- `$stage-review` returned `more-work-needed`. It identified understated
  capacity, stale pre-padding candidate artifacts, an eager/trace topology
  comparison, an incomplete fused-AG rank-order probe, a cache allocator
  override inconsistency, nonuniform provenance wording, public `forward()` not
  used inside trace tests, stale capture hashes, and overstated Watcher scope.
- Capacity now mirrors per-layer allocator granularity, shares stack-global
  RoPE/position/CCL state once, includes future embedding/LM-head/final-norm
  residency, records the live prefill peak and L1 snapshot, and has an adjacent
  measured boundary. Every candidate was rerun on the final graph.
  Topology now reports eager/eager and trace separately. Fused AG tests all 24
  legal global K permutations. `allocate_kv_cache` rejects a size inconsistent
  with constructor-owned RoPE/cache state. Public `forward()` accepts stable
  position buffers and is used by trace tests. Current source hashes are
  recorded separately from capture hashes. Watcher is labeled partial with the
  exact Ethernet instrumentation limit.
- A fresh rereview and stage-only local commit SHA(s) are appended after the
  final gate. No push.

## 2026-07-18: remediation validation

- `python -m py_compile` passes for both decoder and test; Black reports both
  files unchanged and `git diff --check` is clean.
- Default suite: `3 passed, 8 skipped`; the three active gates are the static
  fallback/ownership audit, non-aligned synthetic+stacked+paged+public-forward
  trace test, and paged-table remap trace test. Manual evidence gates are
  intentionally environment-selected.
- Real selected layer-32 gate rerun: `1 passed`, retaining prefill/decode PCC
  `0.99339170/0.99400587` and K/V PCC at least `0.99966163`.
- Review-remediation manual gates: capacity `12224` pass and `12225` expected
  OOM both pass; fair topology and 24-permutation fused-AG probes each pass.
- The fresh rereview's remaining prefill-program concern was closed by a
  device sweep over 10x10 and 8x10 grids with input block limits 10, 16, 20,
  and 32. The selected 10x10/10 path measured `3.23375687 ms` over 21 trials;
  10x10/16 measured `3.29531496 ms` (1.90% slower), 8x10 candidates were
  slower, and the larger legal-looking cases hit exact L1 circular-buffer
  limits. `prefill_geometry_sweep.json` and the raw successful candidate JSONs
  retain the protocols, per-role program configs, failures, and provenance.
- The final post-sweep default suite is `3 passed, 8 skipped`; the real
  layer-32 gate is `1 passed` with the same PCC values above. All 30 retained
  JSON documents parse, `py_compile`, Black, and `git diff --check` pass. A
  post-run `tt-smi -s` reports four healthy p300c boards, all DRAM status true,
  no corrected or uncorrected GDDR errors, and temperatures from 46.4–50.3 C.
- Final fresh `$stage-review` verdict: `clean-pass`, with no required work and
  no other concerns. The independent review reconstructed sweep medians and
  performance arithmetic, checked all 30 JSON artifacts and retained hashes,
  and verified every prior finding fixed or explicitly controlled. The full
  verdict is retained in `stage_review.md`.
- Stage-owned implementation/evidence commit: `ba8a83b14fd`. This commit was
  created locally after all repository hooks passed and was not pushed.
