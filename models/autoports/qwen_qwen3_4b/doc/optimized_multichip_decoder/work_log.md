# Optimized Multichip Decoder Work Log

## Inputs

- Model: Qwen/Qwen3-4B
- Stage source: completed `doc/multichip_decoder` decoder path.
- Target mesh: 1x4 Blackhole p150b ring.
- Scope: optimize the multichip decoder in place. No full-model or vLLM work was started.
- Base SHA before this pass: `7b46bc53bd7521dc69ff2686baaa7b5d15bcf1aa`.

## Baseline

Correctness:

```bash
pytest -q -s models/autoports/qwen_qwen3_4b/tests/test_multichip_decoder.py --tb=short
```

Result: `8 passed, 3 skipped`.

Baseline PCC:

- prefill seq16: 0.9997333805764328
- prefill seq17: 0.9997359676733848
- prefill seq64: 0.999684148069019
- paged decode prefix16: 0.9980224018654454
- paged decode prefix17: 0.9978650895682959
- trace replay: 1.0
- inter-device replication: 1.0

Baseline host perf:

```bash
QWEN3_4B_MULTICHIP_RUN_PERF=1 \
QWEN3_4B_MULTICHIP_PERF_OUT=models/autoports/qwen_qwen3_4b/doc/optimized_multichip_decoder/before/perf_host_timings_before.csv \
pytest -q -s models/autoports/qwen_qwen3_4b/tests/test_multichip_decoder.py::test_multichip_perf_signposts --tb=short
```

Result:

- warmed prefill seq16: 2.696187 ms
- traced warmed decode pos16: 0.437868 ms

## Topology Audit

The starting topology used packed QKV, local paged SDPA, row-parallel WO plus all-reduce, separate gate/up projections, row-parallel down plus all-reduce, and a replicated BF16 residual at the layer boundary. The pass audited same-input gate/up matmuls, material projection collectives, decode reshard/layout conversions, fused CCL+matmul options, residual layout alternatives, activation/CCL dtype, persistent buffers, and context contract impact.

Subagent/source work found:

- Persistent all-reduce lower overload is separate from high-level cluster-axis all-reduce; Blackhole requires non-DRAM `WIDTH_SHARDED` input.
- Qwen TP4 fused matmul+reduce-scatter and fused all-gather+matmul are not blocked by the TP8 common helper gate.
- A lower-movement residual family requires distributed RMSNorm and sharded projection contracts, not only replacing the projection collective.

## Candidate Log

1. Adapted single-chip decode DRAM-sharded matmul for multichip MLP down. The first 8-core attempt failed because local K=2432 is 76 tiles, not divisible by 8. Retried with the largest <=8 divisor, 4 cores.

2. Measured DRAM-sharded down with L1 staging also used by prefill:

```text
down_dram_candidate_perf.csv
prefill: 3.349813 ms
traced decode: 0.407778 ms
```

Rejected because it materially regressed prefill. Adapted to decode-only staging.

3. Measured decode-only DRAM-sharded down with old all-reduce:

```text
down_dram_decode_only_perf.csv
prefill: 2.487458 ms
traced decode: 0.414478 ms
```

This became the base for CCL work.

4. Tried `num_links=2` for ring all-reduce:

```text
num_links2_down_dram_perf.csv
prefill: 2.376138 ms
traced decode: 0.429568 ms
```

Rejected because traced decode regressed.

5. Tried BFLOAT8_B KV cache:

```text
bfloat8_cache_down_dram_perf.csv
prefill: 2.525658 ms
traced decode: 0.413958 ms
```

Trace replay PCC remained 1.0. Rejected because it did not improve the final decode family and would change the context contract.

6. Tried high-level async all-reduce with DRAM-sharded down:

```text
async_all_reduce_down_dram_perf.csv
prefill: 2.391728 ms
traced decode: 0.411478 ms
```

Useful, then superseded by lower persistent decode all-reduce and geometry.

7. Probed persistent/preallocated all-reduce:

- `persistent_all_reduce_minimal_probe.log`: first lower-overload call used the wrong high-level `math_op` argument.
- `persistent_all_reduce_minimal_probe_retry.log`: corrected signature but DRAM input failed on Blackhole.
- `persistent_all_reduce_l1_probe.log`: adapted L1 width-sharded input/scratch passed.
- `persistent_ar_decode_only_perf.csv`: prefill 2.523017 ms, traced decode 0.370138 ms.

Decode persistent all-reduce was kept. Prefill stayed high-level.

8. Swept decode geometry modes with persistent decode CCL fixed:

- `qkv_1d_dram_24c_i5_s2`: 2.386648 / 0.343278 ms.
- `qkv_width_l1_10c_i8_s5`: 2.427608 / 0.345198 ms.
- `qkv_width_l1_8c_i10_s6`: 2.487517 / 0.350308 ms.
- `gate_up_1d_l1_20c_i10_s4`: 2.631487 / 0.303408 ms.
- `gate_up_1d_l1_10c_i8_s8`: 2.669297 / 0.326158 ms.
- `gate_up_width_l1_10c_i8_s8`: 2.524747 / 0.326578 ms.
- `wo_1d_explicit_8c_i4_s4`: rejected with exact blocker `Number of blocks exceeds number of cores: 20 blocks > 8 cores`.
- `down_dram_2c_i38_n40`: 2.505407 / 0.369568 ms.

9. Swept combinations:

- `combo_gate_up_1d_l1_20c_i10_s4`: 2.423578 / 0.304228 ms.
- `combo_qkv_1d_dram_24c_i5_s2_plus_gate_up_1d_l1_20c_i10_s4`: 2.414958 / 0.280689 ms.
- `combo_gate_up_1d_l1_20c_i10_s4_plus_down_dram_2c_i38_n40`: 2.751126 / 0.306159 ms.
- `combo_qkv_1d_dram_24c_i5_s2_plus_gate_up_1d_l1_20c_i10_s4_plus_down_dram_2c_i38_n40`: 2.447248 / 0.282649 ms.

The QKV plus gate/up combination became the final default geometry.

10. Swept prefill-specific geometry:

- `prefill_qkv_1d_dram_24c_i5_s2`: 2.457618 / 0.369878 ms under env override.
- `prefill_gate_up_1d_dram_20c_i10_s4`: 2.440627 / 0.368888 ms under env override.
- Combined prefill QKV plus gate/up: 2.536607 / 0.370549 ms.
- `prefill_down_1d_dram_20c_i10_s4`: rejected with exact blocker `Kt must be divisible by in0_block_w`.

None beat the final default family.

11. Probed fused CCL+matmul paths:

- `fused_matmul_reduce_scatter_qwen_probe.log`: Qwen WO-like TP4 decode shape M=32, K=1024, N=2560 passed. RS PCC was about 0.99988.
- `fused_all_gather_matmul_qwen_probe.log`: first attempt hit `out_block_w (3) must be divisible by out_subblock_w (4)`, then adapted to `out_subblock_w=1` and passed. Matmul PCC was 0.999993 and all-gather PCC was 1.0.

These are viable local-output primitives, not adopted as final default because they change the projection output to local hidden shards or local output columns. Preserving the completed decoder's replicated layer boundary would require a trailing all-gather, while the lower-movement family requires distributed RMSNorm and sharded next-projection contracts.

12. Reran reduce-scatter residual contract probe:

```bash
QWEN3_4B_MULTICHIP_RUN_RS_PROBE=1 \
pytest -q -s models/autoports/qwen_qwen3_4b/tests/test_multichip_decoder.py::test_multichip_reduce_scatter_residual_contract_probe --tb=short \
  > models/autoports/qwen_qwen3_4b/doc/optimized_multichip_decoder/reduce_scatter_residual_contract_probe.log 2>&1
```

Result: reduce-scatter to local width 640 works, but the existing full-width contract fails at residual add, full-width RMSNorm gamma, and gate/up matmul with exact shape errors. This closed the naive local-residual insertion path.

13. Probed a stack-compatible sharded residual micro-path:

```bash
python - <<'PY' > models/autoports/qwen_qwen3_4b/doc/optimized_multichip_decoder/sharded_residual_stack_probe.log 2>&1
...
PY
```

Result: local residual add, distributed RMSNorm, local-K gate/up, reduce-scatter to local intermediate, local down, reduce-scatter back to local hidden, and final local residual add all completed. The final local shape was `[1, 1, 16, 640]` on every device and all outputs were finite. This proves the lower-movement family is viable as a contract rewrite, not blocked. It was not made the final default because it changes the completed decoder's inter-layer contract from replicated hidden `[1, 1, M, 2560]` to TP-local hidden `[1, 1, M, 640]`; adopting it correctly requires every adjacent decoder layer, both RMSNorms, QKV, MLP, and full-model bringup to preserve the sharded residual layout.

14. Measured the coherent sharded residual family against the current replicated residual-plus-MLP stack:

```text
sharded_residual_stack_perf.log
replicated_stack_ms_min=0.942055
replicated_stack_ms_mean=0.955941
sharded_stack_ms_min=2.410048
sharded_stack_ms_mean=2.445269
sharded_vs_replicated_min_ratio=2.558287
```

Rejected for this in-place decoder stage because the stack-compatible lower-movement family is materially slower than the replicated contract path under the same S=16, H=2560, I=9728 TP4 mesh shape.

15. Fixed trace replay correctness evidence. `trace_decode_once` can now copy a replay-specific host decode token into the captured device input before `ttnn.execute_trace`. `test_multichip_trace_replay_is_deterministic` captures with one token, replays with a different non-affine random token, and compares the replay-updated output tensor against eager decode on a separate KV cache. The test also snapshots capture output and records `capture_vs_replay_max_delta=8.84375`, so the capture-time output can no longer satisfy the check.

16. Removed the decode-down L1-to-DRAM-to-L1 bounce before persistent all-reduce. The final path keeps the DRAM-sharded down matmul output in L1 width-sharded memory and lets `_all_reduce_hidden(..., use_persistent=True)` perform the required L1-to-L1 reshard to the all-reduce layout.

17. Added and measured an adapted packed gate/up candidate after stage review found the earlier prose rejection insufficient. The `packed_gate_up_1d_l1_20c_i10_s4` mode packs each device's gate/up weights, runs one decode matmul to `2 * local_intermediate`, slices gate/up on device, then runs SiLU, multiply, DRAM-sharded down, and persistent all-reduce.

```text
packed_gate_up_trace_replay.log
PCC=1.0
capture_vs_replay_max_delta=6.65625

packed_gate_up_perf.csv
prefill: 2.413568 ms
traced decode: 0.284548 ms
```

Rejected because the primary target is traced decode and the split gate/up default remains faster at 0.279989 ms. The packed weight is env-gated and is not loaded on the final default path.

18. Searched for same-model same-stage optimized references in the current checkout:

```bash
find models/autoports/qwen_qwen3_4b \( -path '*optimized_multichip_decoder*' -o -path '*multichip_decoder*' \) -maxdepth 6 -type f
```

No conflicting optimized reference outside this stage artifact root was found. Artifact: `reference_search.log`.

## Final Validation

Correctness:

```bash
pytest -q -s models/autoports/qwen_qwen3_4b/tests/test_multichip_decoder.py --tb=short \
  > models/autoports/qwen_qwen3_4b/doc/optimized_multichip_decoder/final_correctness.log 2>&1
```

Result: `8 passed, 3 skipped in 78.02s`.

Final PCC:

- prefill seq16: 0.9997333805764328
- prefill seq17: 0.9997359676733848
- prefill seq64: 0.999684148069019
- paged decode prefix16: 0.999583534225651
- paged decode prefix17: 0.9995836610905112
- trace replay with replay-specific input update: 1.0; `capture_vs_replay_max_delta=8.84375`
- inter-device replication: 1.0

Final host perf:

```bash
rm -f models/autoports/qwen_qwen3_4b/doc/optimized_multichip_decoder/perf_host_timings.csv
QWEN3_4B_MULTICHIP_RUN_PERF=1 \
QWEN3_4B_MULTICHIP_PERF_OUT=models/autoports/qwen_qwen3_4b/doc/optimized_multichip_decoder/perf_host_timings.csv \
pytest -q -s models/autoports/qwen_qwen3_4b/tests/test_multichip_decoder.py::test_multichip_perf_signposts --tb=short
```

Result:

- warmed prefill seq16: 2.439038 ms
- traced warmed decode pos16: 0.279989 ms

Tracy/perf report:

```bash
QWEN3_4B_MULTICHIP_RUN_PERF=1 \
QWEN3_4B_MULTICHIP_PERF_OUT=models/autoports/qwen_qwen3_4b/doc/optimized_multichip_decoder/perf_host_timings_tracy.csv \
python -m tracy -r -p -v --sync-host-device --dump-device-data-mid-run --check-exit-code \
  -o models/autoports/qwen_qwen3_4b/doc/optimized_multichip_decoder/tracy/perf_capture_final \
  -n qwen3_4b_optimized_multichip_decoder_final \
  -m pytest -q -s models/autoports/qwen_qwen3_4b/tests/test_multichip_decoder.py::test_multichip_perf_signposts --tb=short
```

Result: `1 passed`. Generated compressed raw ops artifact `optimized_multichip_ops_final.csv.gz`, `tt_perf_report_prefill.*`, and `tt_perf_report_traced_decode.*`.

Tracy host CSV from that profiler run recorded prefill 2.944105 ms and traced decode 0.313698 ms with profiler overhead. The compact final non-profiler headline remains `perf_host_timings.csv` above.

Watcher:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1 \
QWEN3_4B_MULTICHIP_RUN_WATCHER_STRESS=1 \
pytest -q -s models/autoports/qwen_qwen3_4b/tests/test_multichip_decoder.py::test_multichip_watcher_single_mesh_stress --tb=short
```

Result: `1 passed in 18.28s`; watcher attached to devices 0-3 and detached cleanly. Artifact: `watcher_stress.log`.

Context:

```bash
python .agents/scripts/check_context_contract.py --model-dir models/autoports/qwen_qwen3_4b
```

Result: full 40960-token context supported. Artifact: `context_contract_check.log`.

Device health:

```bash
tt-smi -ls --local
```

Result: four Blackhole p150b devices visible and resettable. Artifact: `tt_smi_after.log`.

## Review And Repair

The first `$stage-review` returned `more-work-needed`. An `$autofix` loop used `$autodebug` to identify missing persistent CCL, fused CCL, residual-layout, geometry, and accounting evidence. The follow-up pass added:

- Persistent L1 all-reduce probes and default decode implementation.
- Decode geometry sweeps and final default geometry.
- Qwen-specific fused matmul+reduce-scatter and all-gather+matmul probes.
- Fresh residual-contract probe, stack-compatible sharded residual micro-path plus perf rejection evidence, and final correctness/perf/watcher artifacts.
- Trace replay correctness fix proving replay-updated output rather than capture output.
- Decode-down bounce removal with final correctness and perf evidence.
- Packed gate/up candidate with replay correctness and whole-decoder perf rejection evidence.
- Updated README and work log tied to the final default path.

Final `$stage-review` rereview returned `clean-pass` with no required work. Reviewer noted two controlled anomalies, not blockers: Tracy trace-replay host timestamps precede replay signpost even though rows carry trace/replay session IDs, and process-exit nanobind refcount warnings occur after passing tests with clean device close and healthy `tt-smi`.

Checkpoint commit:

- Repo: `/home/ubuntu/tt-metal`
- Branch: `agentic-research/fast-models-fast`
- Stage checkpoint SHA: `bfa8d3c1bedd3ba1832122ab10bc58f8e24283ba`

## Final State

The optimized multichip decoder keeps the replicated inter-layer residual contract, preserves non-aligned logical sequence support, preserves the 40960-token context contract, improves warmed prefill and traced warmed decode, and leaves no full-model or vLLM changes in this stage.
