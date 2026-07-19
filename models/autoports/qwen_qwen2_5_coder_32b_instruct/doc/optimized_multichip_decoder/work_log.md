# Optimized multichip decoder work log

All real-weight evidence uses `Qwen/Qwen2.5-Coder-32B-Instruct` revision `381fc969f78efac66bc87ff7ddeadb7e73c218a7` at `/home/mvasiljevic/hf-cache/hub/models--Qwen--Qwen2.5-Coder-32B-Instruct/snapshots/381fc969f78efac66bc87ff7ddeadb7e73c218a7`.

## Scope and starting state

- Date: 2026-07-19 UTC.
- Branch: `mvasiljevic/model/qwen-qwen2.5-coder-32b-instruct`.
- Starting HEAD: `860d7d688063c51fcc41a49c9d7611a1f761d535`.
- Completed multichip decoder commit: `ba8a83b14fd`.
- Preserved unrelated dirty file: `.agents/skills/forge-functional-decoder-from-ir/SKILL.md`.
- Loaded and followed `$optimize`, `$graph-rewrite`, `$shard-advise`, `$tt-device-usage`, `$autofix`, `$autodebug`, and `$stage-review`.
- This stage changed only the multichip decoder, its tests/probes, context contract, and stage-owned documentation/artifacts. No full-model or vLLM path was started.

## Hardware safety

`timeout 60 tt-smi -ls --local` reported four Blackhole p300c devices. A serialized smoke opened and closed `MeshShape(1,4)` with `FABRIC_1D_RING`, Ring topology, two links, and an 11x10 compute grid.

All hardware commands were serialized. Watcher and Tracy/profiler were run separately. Material hangs were captured, terminated, reset, and followed by a clean 1x4 mesh smoke before resuming.

## Before baseline

Command:

```bash
QWEN2_5_CODER_32B_REAL_WEIGHT_DIR=<revision-path> \
QWEN2_5_CODER_32B_MULTICHIP_BASELINE_PATH=/tmp/qwen2_5_coder_32b_optimized_baseline.pt \
QWEN2_5_CODER_32B_MULTICHIP_RUN_PERF=multichip \
QWEN2_5_CODER_32B_MULTICHIP_PREFILL_TRIALS=7 \
QWEN2_5_CODER_32B_MULTICHIP_DECODE_TRIALS=7 \
QWEN2_5_CODER_32B_MULTICHIP_DECODE_REPLAYS=100 \
QWEN2_5_CODER_32B_MULTICHIP_RESULT_NAME=optimized_multichip_before.json \
timeout 1800 python_env/bin/python -m pytest \
  models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_multichip_decoder.py::test_real_multichip_warmed_prefill_and_traced_decode_perf -q -s
```

Result: prefill PCC 0.993391703, decode PCC 0.994005871, warmed prefill median 3.577945 ms, warmed traced decode median 0.791976 ms, 7 trials and 100 trace replays/trial, bitwise stable. `results/before_default.json` SHA256 is `749a5a9479ab8e79dc658859259cfc3d30dcd5031b348026b051e288a3028d49`.

The operation-topology audit was written in `README.md` before candidate tuning. It identified the inter-layer DRAM restore, same-input gate/up matmuls, four material collectives, projection/collective conversions, head-padding movement, and prefill projection/CCL costs.

## Graph and topology work

### Residual layout and packed gate/up

- Removed the final decode L1-to-DRAM return and retained a 20-core L1 `WIDTH_SHARDED` hidden fracture across layers: prefill 3.576924 ms, decode 0.786000 ms, unchanged PCC.
- Packed gate/up into one physical `32x5120x14336` BFP4/LoFi projection. The first 16-core layout hit exact CB capacity; 32 cores passed. Full 7x100 evidence: prefill 3.642007 ms, decode 0.782612 ms, PCC 0.993392/0.993982.
- A later stack gate exposed strict `MemoryConfig` inequality caused only by reshape-created ND metadata. Physical grid/shape/orientation/L1 comparison now avoids a false `to_memory_config`. The direct two-layer prefill/decode gate passes at PCC 0.990702/0.990737.

### Collective placement

The topology family benchmark used the same recorded activation and precision on one 1x4 mesh:

- Selected fractured boundary, 2 AG + 2 RS: eager 2.251915 ms, traced 0.758101 ms.
- Compiler-provenance replicated boundary, 2 Ring all-reduces: eager 2.059678 ms, traced 0.842635 ms.
- Output, K cache, V cache, eager/trace checks: PCC 1.0.

The selected trace family is 10.03% faster and no layer-boundary collective remains.

### Fused all-gather plus matmul AutoFix

Initial fused attempts were not rejected at their first error:

1. Adapted distributed-statistics norm output from DRAM to the required one-core L1 width shard.
2. Adapted weights to rank 4 without storage duplication.
3. Tried 2-link fused QKV; it stalled. Captured triage, reset, and retried one link.
4. Tried fused packed gate/up at 8x1; its 14336-wide output exceeded CB capacity.
5. Retried a 10x1 padded geometry; captured and reset after stall.
6. Split the coherent family to one persistent fused gate AG+matmul and direct up matmul consuming the returned gathered tensor, avoiding a second AG or restoration to the old residual.

The first final adapted fused family passes focused trace and full real-layer correctness: PCC 0.993392/0.993894, prefill 4.351594 ms, decode 0.960300 ms. Independent review noticed that this run used the earlier attention HiFi2 policy, so it was not used as the final coherent rejection. The fused family was rerun at 7x100 under the final attention/MLP LoFi policy and 8x4 SDPA: PCC 0.992527/0.993664, prefill 3.657417 ms, decode 0.930669 ms. It remains 22.8% slower than the final default and was rejected. Evidence: `validation/fused_ag_autofix.md`, `results/autofix_fused_ag_full_probe.json`, `results/sweep_fused_ag_final_full.json`, and `tests/test_fused_ag_projection_probe.py`.

### BFP8 async CCL AutoFix

Focused exact AG/RS traces found that standard persistent AG output had to remain in DRAM for its requested contract. After the buffer fix, exact BFP8 AG/RS traces passed. The first full layer gave PCC 0.993243/0.993903, prefill 4.184984 ms, decode 0.805522 ms. Independent review requested a final-policy cross; its 7x100 result is PCC 0.992494/0.993622, prefill 3.708916 ms, decode 0.787956 ms. It remains 3.94% slower than final BF16 CCL and was rejected. Evidence: `validation/bfp8_ccl_autofix.md`, `results/autofix_bfp8_ccl_full_probe.json`, `results/sweep_ccl_bfp8_final_full.json`, and `tests/test_bfp8_async_ccl_probe.py`.

### Fused reduce-scatter

Adapted O/down output shapes and fused RS buffers until the full layer compiled and passed. Result: PCC 0.992527/0.993671, prefill 4.365227 ms, decode 0.885514 ms. Rejected.

## Shard advisor

The first capture failed with an undefined `moe_compute` symbol. `$autodebug` proved that the current checkout's `_ttnncpp.so` was overriding the advisor environment's vendored TTNN runtime. The capture was repaired by using the advisor Python/native libraries as one matched set.

The first repaired graph returned a tuple unsupported by the capture sink. The sink was adapted to combine live slices of QKV, O, gate/up, and down. The final report contains 16 ops, 15 choices, spill analysis, and zero spills.

Focused device comparison (`validation/advisor_projection_compare.json`):

| Projection | Current DRAM-sharded | Advisor L1/interleaved | Ratio advisor/current |
| --- | ---: | ---: | ---: |
| QKV | 46.588 us | 43.342 us | 0.930 |
| O | 32.812 us | 35.087 us | 1.069 |
| Packed gate/up | 139.139 us | 148.627 us | 1.068 |
| Down | 73.047 us | 154.139 us | 2.110 |
| Coherent family | 291.6 us | 381.2 us | 1.307 |

The current DRAM-sharded family was retained. Files: `shard_advise/result/report.json`, `report.txt`, `final_ir.mlir`, and `tests/test_advisor_projection_probe.py`.

## Precision, fidelity, activation, cache, and buffer sweeps

Short candidate probes used the real layer-32 activation/reference, one prefill timing, and 10 or 20 traced replays; retained changes were reconfirmed at 7x100. Full numbers and artifact paths are in `results/candidate_summary.csv`.

- Selected: BFP8/LoFi attention + BFP4/LoFi MLP, PCC 0.992527/0.993698, 0.759633 ms short-probe decode.
- All BFP4/LoFi: 0.758270 ms but prefill PCC 0.989796; rejected.
- All BFP4 with attention HiFi2: 0.780703 ms but prefill PCC 0.989919; rejected.
- MLP BFP4/HiFi2: 0.951910 ms; rejected.
- Gate BFP4/down BFP8: 0.799069 ms; rejected.
- Gate BFP8/down BFP4 first hit CB pressure; adapted to 64 cores and prefill block limit 5, then passed at PCC 0.996685/0.997276 and 0.812011 ms; rejected.
- BF16 KV control: 0.785161 ms; BFP8 retained.
- BFP8 matmul outputs were isolated under the final policy/topology after the old global probe: attention-only was 0.771944 ms; MLP-only was 0.759567 ms at 7x100 and PCC 0.992527/0.993653. BF16 at 0.758047 ms was retained.
- Persistent decode collectives disabled under the final policy/8x4 topology: 0.771160 ms at 7x100 versus 0.758047 ms enabled; persistent shared buffers retained.

The final-policy matrix used the same real-weight performance command as the
final default, changing only one environment variable and result name:

```bash
QWEN2_5_CODER_32B_REAL_WEIGHT_DIR=<revision-path> \
QWEN2_5_CODER_32B_MULTICHIP_BASELINE_PATH=/tmp/qwen2_5_coder_32b_optimized_baseline.pt \
QWEN2_5_CODER_32B_MULTICHIP_RUN_PERF=multichip \
QWEN2_5_CODER_32B_MULTICHIP_PREFILL_TRIALS=7 \
QWEN2_5_CODER_32B_MULTICHIP_DECODE_TRIALS=7 \
QWEN2_5_CODER_32B_MULTICHIP_DECODE_REPLAYS=100 \
<candidate-variable> QWEN2_5_CODER_32B_MULTICHIP_RESULT_NAME=<artifact> \
timeout 1800 python_env/bin/python -m pytest \
  models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_multichip_decoder.py::test_real_multichip_warmed_prefill_and_traced_decode_perf -q -s
```

Candidate variables were `QWEN2_5_CODER_32B_MULTICHIP_MLP_ACTIVATION_DTYPE=bfp8`,
`QWEN2_5_CODER_32B_MULTICHIP_CCL_DTYPE=bfp8`,
`QWEN2_5_CODER_32B_MULTICHIP_PERSISTENT_CCL=0`, and the pair
`QWEN2_5_CODER_32B_MULTICHIP_DISTRIBUTED_NORM=1`
`QWEN2_5_CODER_32B_MULTICHIP_FUSED_AG_MATMUL=1`. Attention-only BFP8 used the
same command with one trial/20 replays because its 0.771944 ms loss was clear;
MLP-only BFP8 was promoted to the full 7x100 gate because its short result was
within noise of the default.

## Geometry and block sweep

- SDPA 8x4: 0.758606 ms, retained. Group width 8: 0.772048 ms, rejected.
- QKV 32 cores: 0.767764 ms; rejected.
- O 20/40 cores: 0.761706/0.775129 ms; rejected.
- The first gate/up target-64 request measured 0.759756 ms but silently resolved to the unchanged 32-core program: logical K/N are 160/448 tiles and their greatest common divisor is 32. `sweep_geometry_gate64.json` is therefore a diagnostic/no-op screen, not 64-core rejection evidence.
- The material retry padded packed gate/up K from 5120 to 6144 with zeros. This produced a real `32x6144x14336` 8x8/64-core matmul (`in0_block_w=3`, `per_core_N=7`) and a separate largest-legal 8x7/56-core grid for the 7168-wide SiLU-times-up result. Its first attempt reached the 64-core matmul but failed because a 64-core 7168-wide elementwise shard is 112 channels, not tile aligned; the 56-core adaptation passed. Full 7x100 evidence is PCC 0.992527216/0.993759643, prefill 3.525263 ms, decode 0.792797 ms, and bitwise trace stability. It is 4.58% slower than the selected 32-core family and was rejected. Artifact SHA256: `7fd7a765a4e22ea023fa4aa9748762136bb961fff155bb2f6778aa0cf1fc54b0`.
- Down 32 cores: 0.763292 ms; rejected.
- QKV block 5/2: 0.763179/0.792033 ms; rejected.
- O block 1: 0.771079 ms; rejected.
- Gate block 1: 0.872325 ms; rejected.
- Down block 7/2: 0.762449/0.787857 ms; rejected.

## Prefill family

- Final 10x10 grid, block-limit 10 control: 3.571924 ms median over three trials.
- 8x10/block-10: 3.680131 ms; rejected.
- 10x10/block-20 exceeded CB capacity. Adapted retries at 8x10/block-20 and 8x10/block-16 also exceeded exact CB capacity. The smaller block-10 default was retained.
- Final current-source 7-trial default: 3.566172 ms.

Prefill remains valid for non-aligned logical inputs. Sequence 31 passes, and all internal tile/head/cache padding, masking, and output slicing are decoder-owned.

## Profiler and tt-perf-report

Capture command:

```bash
QWEN2_5_CODER_32B_REAL_WEIGHT_DIR=<revision-path> \
QWEN2_5_CODER_32B_MULTICHIP_RUN_PROFILE=1 \
python_env/bin/python -m tracy -r -p -v \
  -o models/autoports/qwen_qwen2_5_coder_32b_instruct/doc/optimized_multichip_decoder/tracy/layer32 \
  -n final_default -m pytest \
  models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_multichip_decoder.py::test_profile_selected_multichip_decoder -q -s
```

Signpost windows are `PERF_PREFILL..PERF_PREFILL_END` and `PERF_DECODE..PERF_DECODE_END`. Compact report hashes and raw-capture hashes are in `tracy/layer32/provenance.json`.

Decode: 72 device ops, 595 us merged device work. Dominant/important ops: packed gate+up 134 us, down 67 us, QKV 26 us, O 19 us, SDPA 19 us, AG 11/11 us, RS 23/20 us. Prefill: 222 ops, 1992 us device work; packed gate+up 356 us, down 185 us, QKV 72 us, O 57 us, AG 83/81 us, RS 87/90 us.

Every actionable report family was tried: projection core/block/subblock families, advisor sharding, packing, fused AG/RS, dtype/fidelity, SDPA padding/layout, and persistent buffers. Rejections have candidate JSON.

## Final default and correctness commands

Final 7x100 command:

```bash
QWEN2_5_CODER_32B_REAL_WEIGHT_DIR=<revision-path> \
QWEN2_5_CODER_32B_MULTICHIP_BASELINE_PATH=/tmp/qwen2_5_coder_32b_optimized_baseline.pt \
QWEN2_5_CODER_32B_MULTICHIP_RUN_PERF=multichip \
QWEN2_5_CODER_32B_MULTICHIP_PREFILL_TRIALS=7 \
QWEN2_5_CODER_32B_MULTICHIP_DECODE_TRIALS=7 \
QWEN2_5_CODER_32B_MULTICHIP_DECODE_REPLAYS=100 \
QWEN2_5_CODER_32B_MULTICHIP_RESULT_NAME=optimized_multichip_final_default.json \
timeout 1800 python_env/bin/python -m pytest \
  models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_multichip_decoder.py::test_real_multichip_warmed_prefill_and_traced_decode_perf -q -s
```

Final result: PCC 0.992527216/0.993698060, prefill 3.566172 ms, traced decode 0.758047 ms, bitwise stable. `results/final_default.json` SHA256 is `7f862083c1b3b38c0d232a54712183f9c1a2c4e1e1dc0ac15c832d97a0bb40e6`.

Other final gates:

```bash
python_env/bin/python -m pytest \
  models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_multichip_decoder.py::test_multichip_contract_is_optimized_owned_and_host_free \
  models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_multichip_decoder.py::test_multichip_synthetic_non_aligned_prefill_decode_and_paged_cache -q -s

QWEN2_5_CODER_32B_REAL_WEIGHT_DIR=<revision-path> \
QWEN2_5_CODER_32B_MULTICHIP_RUN_TOPOLOGY=1 \
QWEN2_5_CODER_32B_MULTICHIP_DECODE_TRIALS=7 \
QWEN2_5_CODER_32B_MULTICHIP_DECODE_REPLAYS=100 \
python_env/bin/python -m pytest \
  models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_multichip_decoder.py::test_multichip_compiler_provenance_topology -q -s

QWEN2_5_CODER_32B_REAL_WEIGHT_DIR=<revision-path> \
QWEN2_5_CODER_32B_MULTICHIP_BASELINE_PATH=/tmp/qwen2_5_coder_32b_optimized_baseline.pt \
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
python_env/bin/python -m pytest \
  models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_multichip_decoder.py::test_real_multichip_matches_optimized_single_chip_baseline -q -s
```

Synthetic/non-aligned/paged/stacked/trace gate passed. Runtime fallback audit passed. Topology family passed at PCC 1.0. Final Watcher worker/dispatch gate passed on all four devices with no fault-pattern matches.

Full Watcher instrumentation was attempted before the scoped retry. It fails at mesh open because the instrumented active-Ethernet Ring program is 27,920 bytes while the runtime kernel-config buffer is 25,600 bytes. ETH-only monitoring was disabled for the final gate; model worker/dispatch kernels remained monitored.

Capacity evidence was preserved at the unchanged maximum 12224, with adjacent 12225 physical allocation failure. `doc/context_contract.json` records final residual/precision/grids and points to `results/capacity_seq12224.json` and `capacity_seq12225.json`.

## Completion-audit follow-up

The current-tree completion audit mechanically parsed all result JSON, matched every candidate-ledger number to its artifact, verified compact profiler and Watcher hashes, reran the source fallback contract, collected all multichip tests, and checked that both local commits touch only this autoport. A bounded `tt-smi -ls --local` saw all four p300c devices and a serialized `MeshShape(1,4)` open/close printed `MESH_SMOKE_OK`.

The audit found one weak item: the final prefill report's generic advice to place matmul input 0 in L1 had not been directly measured. A candidate-only `QWEN2_5_CODER_32B_MULTICHIP_PREFILL_L1_INPUTS=1` switch now converts each QKV/O/gate-up/down working input to L1 interleaved while keeping the same final policy, 10x10 2D configs, TP4 residual/CCL topology, and DRAM outputs. The first three-trial screen passed but contained a compile outlier, so it was not used as rejection evidence. The cached seven-prefill-trial rerun passed PCC 0.992527216/0.993698060 and measured 3.579746 ms prefill plus 0.758915 ms traced decode; it loses to the current default and remains disabled. Artifact: `results/sweep_prefill_l1_inputs_full.json`, SHA256 `03e8b25393ef64a7fe31287fa9e3a6b6619a4a9192383eea645cbd8f669168a0`.

Independent review then caught that the earlier target-64 gate entry had silently resolved to 32 cores. The completion audit added candidate-only logical-K zero padding and a distinct legal elementwise grid, retried after the first post-matmul tile-alignment error, and produced the true 64-core full result described in the geometry section. Both candidate switches default off; neither changes the public input shape or the selected runtime path.

After both completion-audit candidate switches and the true-64 layout adaptation were present, the exact final default was rerun twice from the current source with the seven-prefill/seven-decode/100-replay command above. The last run is the authoritative artifact: PCC 0.992527216/0.993698060, 3.566172 ms prefill, 0.758047 ms traced decode, and bitwise stability. Both switches default off; the final mesh plan records `prefill_program.input_memory = DRAM INTERLEAVED`, gate/up 8x4/32 cores, and gated elementwise 8x4/32 cores. The scoped Watcher gate was then rerun from this same source and passed all four devices with no fault-pattern matches; its whitespace-normalized log SHA256 is `4f3e73777839c48f2c8a675a35eeada7d27bdc809eeb893bf43e35b4fd3c0f16`.

## Independent review and commits

Fresh `$stage-review` returned `clean-pass`; the full independent anomaly ledger is in `stage_review.md`. Review findings caused final-policy 7x100 reruns for fused AG, BFP8 CCL, persistent-off, and MLP-only BFP8 activation; an attention-only BFP8 probe; stage-local result retention; restoration of completed prior-stage artifacts; corrected validation paths; corrected allocation-neutral fidelity metadata; and a final current-source default/Watcher rerun.

- Implementation, tests, review, and evidence: `f0ab6bfcf5c` (`Optimize Qwen2.5 Coder TP4 decoder`).
- The repository commit hooks passed. Their only source changes were Black/isort formatting; the 1.07 MiB fused-AG hang trace is retained losslessly as `validation/fused_ag_hang_triage.txt.gz` to satisfy the 500 KiB artifact gate.
- This SHA-log update is a documentation-only follow-up commit; its SHA is reported in the final handoff because a commit cannot record its own hash.

No push was performed.
