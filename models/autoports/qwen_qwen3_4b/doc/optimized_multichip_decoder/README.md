# Qwen3-4B Optimized Multichip Decoder

Stage: optimized-multichip-decoder
Model: Qwen/Qwen3-4B
Implementation: `models/autoports/qwen_qwen3_4b/tt/multichip_decoder.py`
Target: 1x4 Blackhole p150b ring, TP4, `FABRIC_1D_RING`
Base revision before this pass: `7b46bc53bd7521dc69ff2686baaa7b5d15bcf1aa`

## Final Default Path

The final default path is the repo-local multichip decoder path on the 1x4 mesh. It is not a single-chip or replicated fallback.

Kept changes:

- Decode QKV uses an explicit 1D matmul geometry: `qkv_1d_dram_24c_i5_s2`.
- Decode gate/up use an explicit 1D L1-input matmul geometry: `gate_up_1d_l1_20c_i10_s4`.
- Decode MLP down uses a DRAM-sharded weight and L1 width-sharded activation/output staging. The local K is 2432, so the default decode DRAM-sharded matmul uses 4 cores because 76 K tiles is divisible by 4 but not by 8.
- Decode row-parallel projection reductions use the lower `ttnn.experimental.all_reduce_async(input, buffer, cluster_axis, mesh_device, semaphore, ...)` overload with preallocated L1 width-sharded scratch buffers and semaphores.
- Prefill remains on high-level async all-reduce and default prefill matmul geometry. Prefill-specific geometry candidates either regressed or hit exact shape blockers.

The public layer boundary remains a replicated BF16 hidden-state tensor of shape `[1, 1, logical_seq_or_batch, 2560]`. Inside a layer, QKV, attention heads, gate/up, and down are TP-local. There is no gather, reshard, or all-reduce between decoder layers. Full-model bringup should preserve this replicated inter-layer residual contract unless it also implements a distributed residual, RMSNorm, and projection stack.

The context contract is unchanged: paged KV cache remains BF16, local KV heads per device, 2560 blocks of 16 tokens, maximum context 40960. `doc/context_contract.json` did not need an update.

## Correctness

Final default command:

```bash
pytest -q -s models/autoports/qwen_qwen3_4b/tests/test_multichip_decoder.py --tb=short \
  > models/autoports/qwen_qwen3_4b/doc/optimized_multichip_decoder/final_correctness.log 2>&1
```

Result: `8 passed, 3 skipped in 78.02s`.

PCC:

| Case | PCC |
| --- | ---: |
| prefill seq16 | 0.9997333805764328 |
| prefill seq17 | 0.9997359676733848 |
| prefill seq64 | 0.999684148069019 |
| paged decode prefix16 | 0.999583534225651 |
| paged decode prefix17 | 0.9995836610905112 |
| trace replay | 1.0 |
| inter-device replication | 1.0 |

The seq17 and prefix17 cases preserve valid non-aligned logical sequence lengths.

Trace replay correctness is now proven with a replay-specific input update: the trace test captures decode with one token, copies a different non-affine random host token into the captured device input before `ttnn.execute_trace`, and compares the replay-updated output tensor against eager decode on a separate KV cache. The capture-time output is also snapshotted, and `capture_vs_replay_max_delta=8.84375` proves replay did not return the stale capture result.

## Performance

Host-timed warmed perf command:

```bash
rm -f models/autoports/qwen_qwen3_4b/doc/optimized_multichip_decoder/perf_host_timings.csv
QWEN3_4B_MULTICHIP_RUN_PERF=1 \
QWEN3_4B_MULTICHIP_PERF_OUT=models/autoports/qwen_qwen3_4b/doc/optimized_multichip_decoder/perf_host_timings.csv \
pytest -q -s models/autoports/qwen_qwen3_4b/tests/test_multichip_decoder.py::test_multichip_perf_signposts --tb=short
```

Before/after host timings:

| Mode | Before multichip ms | Final multichip ms | Change |
| --- | ---: | ---: | ---: |
| warmed prefill seq16 | 2.696187 | 2.439038 | -9.54% |
| traced warmed decode pos16 | 0.437868 | 0.279989 | -36.06% |

Final CSV: `perf_host_timings.csv`. These numbers were regenerated after the trace replay correctness fix and after removing the decode-down L1-to-DRAM-to-L1 bounce before persistent all-reduce.

Final Tracy host timings:

| Mode | Single-chip ms | Multichip ms | Speedup | TP4 efficiency |
| --- | ---: | ---: | ---: | ---: |
| warmed prefill seq16 | 1.830191 | 2.944105 | 0.621646 | 0.155411 |
| traced warmed decode pos16 | 0.539447 | 0.313698 | 1.719635 | 0.429909 |

Final Tracy CSV: `perf_host_timings_tracy.csv`.

## Perf Report

Tracy capture:

```bash
QWEN3_4B_MULTICHIP_RUN_PERF=1 \
QWEN3_4B_MULTICHIP_PERF_OUT=models/autoports/qwen_qwen3_4b/doc/optimized_multichip_decoder/perf_host_timings_tracy.csv \
python -m tracy -r -p -v --sync-host-device --dump-device-data-mid-run --check-exit-code \
  -o models/autoports/qwen_qwen3_4b/doc/optimized_multichip_decoder/tracy/perf_capture_final \
  -n qwen3_4b_optimized_multichip_decoder_final \
  -m pytest -q -s models/autoports/qwen_qwen3_4b/tests/test_multichip_decoder.py::test_multichip_perf_signposts --tb=short
```

Artifacts:

- `tracy/optimized_multichip_ops_final.csv`
- `tt_perf_report_prefill.csv`
- `tt_perf_report_prefill.txt`
- `tt_perf_report_prefill.console.log`
- `tt_perf_report_traced_decode.csv`
- `tt_perf_report_traced_decode.txt`
- `tt_perf_report_traced_decode.console.log`

Final prefill table summary:

- 32 device ops, 0 host ops.
- Dominant rows: QKV 41.476 us, WO 21.867 us, gate/up 50.356/50.439 us, down 49.519 us.
- Prefill row-parallel reductions still show reduce-scatter/all-gather pairs because prefill uses the high-level async all-reduce path.

Final traced decode table summary:

- 43 device ops, 0 host ops.
- Decode QKV row is 18.234 us with 24 cores, `in0_block_w=5`, output subblock 1x2.
- Decode gate/up rows are 18.556/18.162 us with 19 cores, L1 input, `in0_block_w=10`, output subblock 1x4.
- Decode persistent all-reduce rows are 16.123/15.675 us with L1 width-sharded input.
- Decode down row is DRAM-sharded and L1 width-sharded, 14.727 us.
- The Tracy report still shows a large QKV op-to-op gap. The final runtime number is the host trace replay timing in `perf_host_timings.csv`; the per-op table is used for operation topology and device-time evidence.

## Operation Topology Audit

Repeated same-input matmuls:

- QKV is already packed per device and now uses a better decode program config.
- Gate and up still share the same post-attention RMSNorm input and remain separate matmuls in the final default. A packed gate/up candidate was added under `packed_gate_up_1d_l1_20c_i10_s4`: it packs local gate/up weights, runs one decode matmul to `2 * local_intermediate`, slices gate/up on device, then runs SiLU, multiply, DRAM-sharded down, and persistent all-reduce. It passed trace replay with PCC 1.0 and `capture_vs_replay_max_delta=6.65625`, but whole-decoder traced decode measured 0.284548 ms versus 0.279989 ms for the split default. It is rejected for decode latency and its packed weight is loaded only when that geometry mode is selected.

Material collectives:

- Baseline decode row-parallel reductions used high-level all-reduce, which appears as reduce-scatter plus all-gather in the profiler.
- Lower-overload persistent all-reduce initially failed on Blackhole with DRAM input; `persistent_all_reduce_minimal_probe_retry.log` records the validator. The adapted L1 width-sharded version passed; `persistent_all_reduce_l1_probe.log` records `PERSISTENT_AR_L1_OK`.
- Decode-only persistent all-reduce improved traced decode from 0.414478 ms to 0.370138 ms before geometry. It is kept for both decode row-parallel reductions.
- Prefill stayed with high-level async all-reduce because the persistent work was decode-targeted and prefill geometry/CCL changes did not beat the final default.

Reshard/layout conversions:

- Decode QKV avoids extra L1 staging in the kept mode and uses a DRAM-input 1D geometry.
- Decode gate/up intentionally stage post-norm to L1 for the kept geometry.
- Decode down intentionally converts the gated activation to L1 width-sharded input and runs a DRAM-sharded matmul. The earlier L1-to-DRAM-to-L1 bounce before persistent all-reduce was removed; the final path feeds the L1 width-sharded down output into the persistent all-reduce helper, which performs only the required L1-to-L1 reshard to the all-reduce layout.
- The final path avoids new inter-layer resharding.

Fused and packed projections:

- QKV is fused/packed and kept.
- Decode down was repacked for a DRAM-sharded weight.
- Gate/up packing remains unadopted for this decoder because the adapted packed candidate is slower on traced decode after counting on-device split, SiLU, multiply, down, and collectives. Evidence: `packed_gate_up_trace_replay.log` and `packed_gate_up_perf.csv`.

Fused matmul/CCL paths:

- `fused_matmul_reduce_scatter_qwen_probe.log` proves `minimal_matmul_strided_reduce_scatter_async` works on a Qwen WO-like TP4 decode shape: M=32, K=1024, N=2560, `N_tiles=80`, Ring, with RS PCC about 0.99988. The previous GPT-OSS Blackhole gate is therefore not used as the blocker for this model.
- `fused_all_gather_matmul_qwen_probe.log` first hit an invalid subblock config, then adapted `out_subblock_w=1` and passed a WO-like TP4 local-output shape: local input 1024, gathered K=4096, local output 640, matmul PCC 0.999993 and all-gather PCC 1.0.
- These fused paths produce local hidden shards (`[1, 1, M, 640]`) or local-output columns. Adopting them while preserving the current replicated layer boundary requires a trailing all-gather; adopting them as a lower-movement family requires the sharded residual/RMSNorm/projection contract below. The final default keeps replicated residual because that is the accepted completed multichip decoder contract for this in-place optimization stage.

Lower-movement residual layout:

- `reduce_scatter_residual_contract_probe.log` reran the existing Qwen probe on hardware. Reduce-scatter to `[1, 1, 16, 640]` works.
- The old full-width residual contract cannot consume the shard: residual add fails with `Invalid subtile broadcast type`, full-width RMSNorm gamma fails with `Input and gamma padded widths must match`, and gate/up fails with `width=640 height=2560`.
- `sharded_residual_stack_probe.log` then measured the coherent family through local residual add, distributed RMSNorm, local-K gate/up, reduce-scatter to local intermediate, local down, and reduce-scatter back to local hidden. The micro-path completed with finite `[1, 1, 16, 640]` outputs on all four devices.
- This proves the lower-movement family is viable, not blocked. It is not a drop-in replacement for the completed multichip decoder because it changes the inter-layer contract from replicated `[1, 1, M, 2560]` to TP-local `[1, 1, M, 640]` and requires QKV, both RMSNorms, MLP, and every adjacent layer to consume that contract. A stack-compatible performance probe then compared the coherent sharded residual path against the current replicated residual-plus-MLP stack at S=16, H=2560, I=9728. The replicated stack measured 0.942055 ms min, while the sharded stack measured 2.410048 ms min, a 2.558287x slowdown. The final default therefore keeps the accepted replicated decoder contract; the sharded residual contract remains documented for full-model bringup only if that stage has a broader reason to preserve local residuals across layers.

Activation/CCL dtype and precision:

- Final activations and CCL payloads remain BF16.
- BFLOAT8_B KV cache passed trace replay but measured prefill 2.525658 ms and decode 0.413958 ms in its candidate run. It did not improve the final decode family and would change the context contract, so BF16 KV was kept.
- `tt-perf-report` repeatedly advised higher fidelity for accuracy, not speed. The accepted PCC baseline is preserved with the current LoFi matmul policy, so no precision reduction was applied.

Persistent buffers:

- Decode persistent all-reduce resources are cached per `(M, width, dtype)` and reused through trace capture/replay.
- Scratch buffer shard volume satisfies the lower overload's ring-size rule by allocating a replicated logical width of `width * TP`.

MoE:

- Not applicable. Qwen3-4B is dense.

## Candidate Summary

| Candidate | Prefill ms | Traced decode ms | Decision |
| --- | ---: | ---: | --- |
| Baseline before this pass | 2.696187 | 0.437868 | Replaced. |
| Decode-only DRAM-sharded down, old all-reduce | 2.487458 | 0.414478 | Superseded. |
| `num_links=2` with DRAM-sharded down | 2.376138 | 0.429568 | Rejected: decode regressed. |
| BFLOAT8_B KV cache | 2.525658 | 0.413958 | Rejected: no decode win and context contract change. |
| Persistent decode all-reduce only | 2.523017 | 0.370138 | Kept as CCL base. |
| QKV geometry only, `qkv_1d_dram_24c_i5_s2` | 2.386648 | 0.343278 | Useful, superseded by combo. |
| Gate/up geometry only, `gate_up_1d_l1_20c_i10_s4` | 2.631487 | 0.303408 | Useful, superseded by combo. |
| QKV plus gate/up geometry | 2.414958 | 0.280689 | Kept as final geometry family. |
| QKV plus gate/up plus `down_dram_2c_i38_n40` | 2.447248 | 0.282649 | Rejected: slower than final geometry. |
| Packed gate/up, `packed_gate_up_1d_l1_20c_i10_s4` | 2.413568 | 0.284548 | Rejected: decode slower than split gate/up default. |
| Stack-compatible sharded residual family | 2.410048 local stack min | n/a | Rejected for this stage: 2.558287x slower than replicated stack micro-path. |
| Final default retest after trace fix, L1-down bounce removal, and packed candidate rejection | 2.439038 | 0.279989 | Kept. |

Prefill-specific geometry:

- `prefill_qkv_1d_dram_24c_i5_s2`: 2.457618 ms prefill, 0.369878 ms decode when run alone under env override; not kept because final default geometry provides much better decode and comparable prefill.
- `prefill_gate_up_1d_dram_20c_i10_s4`: 2.440627 ms prefill, 0.368888 ms decode when run alone; not kept for the same reason.
- Combined prefill QKV plus gate/up: 2.536607 ms prefill, 0.370549 ms decode; rejected.
- `prefill_down_1d_dram_20c_i10_s4`: exact blocker `Kt must be divisible by in0_block_w`.

## Runtime Gates

Watcher:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1 \
QWEN3_4B_MULTICHIP_RUN_WATCHER_STRESS=1 \
pytest -q -s models/autoports/qwen_qwen3_4b/tests/test_multichip_decoder.py::test_multichip_watcher_single_mesh_stress --tb=short
```

Result: `1 passed in 18.28s`; watcher attached to devices 0-3 and detached cleanly. ETH watcher remains disabled for this known single-mesh fabric instrumentation issue, matching the previous multichip decoder evidence.

Artifact: `watcher_stress.log`.

Runtime fallback audit:

```bash
pytest -q -s models/autoports/qwen_qwen3_4b/tests/test_multichip_decoder.py::test_multichip_runtime_has_no_host_fallback --tb=short
```

Covered in the full final test run. The static fallback audit remains clean for `prefill_forward` and `decode_forward`. Decode persistent CCL scratch buffers are created lazily during warmup/capture and then reused; the final traced replay path does not allocate the optional packed gate/up weight unless `packed_gate_up_1d_l1_20c_i10_s4` is explicitly selected.

Reference search:

```bash
find models/autoports/qwen_qwen3_4b \( -path '*optimized_multichip_decoder*' -o -path '*multichip_decoder*' \) -maxdepth 6 -type f
```

No same-model same-stage optimized reference outside this stage artifact root was found. Artifact: `reference_search.log`.

Context contract:

```bash
python .agents/scripts/check_context_contract.py --model-dir models/autoports/qwen_qwen3_4b
```

Result: `Context contract OK for models/autoports/qwen_qwen3_4b: target=40960, supported=40960 (full HF context).`

Artifact: `context_contract_check.log`.

Post-run device visibility:

```bash
tt-smi -ls --local
```

Result: four local Blackhole p150b devices visible and resettable.

Artifact: `tt_smi_after.log`.
