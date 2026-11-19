# Fused RMS minimal: bounded TP4 repair

The installed AutoDebug launcher completed with exit0; its fresh source-only report is [AUTODEBUG.md](AUTODEBUG.md). The relevant claims were checked against current native source and compute-kernel APIs before preparing [repair.patch](repair.patch). Parent applied the patch at its serialized hardware checkpoint, then built and installed it. This report distinguishes source/CPU checks by this agent from parent-run hardware evidence.

## Failing contract and discriminating checks

1. Each of four ranks produces one stats tile. The factory's `C + R - (1 / R)` subset length is C+4 at TP4 and constructs an out-of-bounds iterator range for every stats core count C. Parenthesizing a ceiling division would still misplace rank tiles when several ranks share a stats core. The correct global tile is g=rank; destination core is g/P and local tile offset g%P for P pages/core. The CPU test checks all C/P combinations in {1,2,4} and physical capacity, including shared-core remainder positions.
2. The consumer is a sender-local dynamic CB containing all gathered stats. A multicore P1/P2 backing does not supply that local contiguous CB; it needs an explicit staging reader. The bounded repaired path admits one stats core matching the first input shard core, one full FP32 tile/rank, standard tiled WIDTH_SHARDED L1 and row-major shard orientation. This is an explicit current capability, not a rejection of future distributed-stats adaptation.
3. A second unconditional hardware-startup call used sender-only c19/c17 on ordinary workers and repeated a once-per-kernel startup transition. Remove that call; only the sender reconfigures input and output formats at its stats reduction. Subsequent multiply operations already configure their formats. This changes phase setup, not numerical arithmetic.
4. The factory creates one stats producer while multi-link receiver credits demand more work. Require one link before native program construction. Unsupported link counts cannot enter the device wait.

## Changed native files

All paths are under `ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/`:

- `device/rms_allgather_device_operation.cpp`: bounded link/stats admission, sender alignment and full local stats capacity.
- `device/rms_allgather_program_factory.cpp`: global tile-to-core/page mapping, range/capacity checks before vector indexing; no unsafe rank vector slices.
- `device/kernels/compute/rms_compute.cpp`: sender-only `INPUT_AND_OUTPUT` format reconfiguration instead of repeated global startup.
- `rms_allgather_nanobind.cpp`: document actual final-output ownership, stats shape/location and link capability.

No fidelity reduction, output precision change, arithmetic shortcut, semaphore credit relaxation or allocation exemption is part of the patch. Cache-hit validation remains the generic operation framework's fallback to miss validation; adding a specialized hit validator later must preserve these guards.

## Validation and receipts

CPU/source-only commands:

```bash
D=models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder
python3 "$D/debug_fused_rms_minimal/check_repair.py"
python3 -m py_compile models/autoports/qwen_qwen3_6_27b/tt/fused_minimal_norm_candidate.py models/autoports/qwen_qwen3_6_27b/tests/probe_fused_minimal_norm.py models/autoports/qwen_qwen3_6_27b/tests/probe_fused_minimal_norm_validation.py
```

[repair.cpu.json](repair.cpu.json) passes all9 core/page transport cases, all32 logical M values, rank counts1/2/4/8, guard source checks, one startup per pre-add branch, sender format transition and cache-hit validation fallback. The applied four source hashes equal [repair.provenance.json](repair.provenance.json). CPU checks do not execute native admission or kernels.

Parent canonical build and install passed, recorded in [fused_rms_minimal_build25.json](../fused_rms_minimal_build25.json) and its log:

```bash
cmake --build build_Release --target ttnn -j 12
cmake --install build_Release --component ttnn-runtime
cmake --install build_Release --component tt_pybinds
```

The receipt includes18 source/binary hashes and replaces earlier build3 pins for this repaired source. Device kernels are JIT compiled by hardware runs, not by the host build alone.

Parent's [fused_minimal_norm_b1_smoke25.json](../fused_minimal_norm_b1_smoke25.json) passes96 quality rows,33 replay rows, four queued changed-input pairs, six independent trace signatures and the exact allocation positive control with tracking1/skip0. Mesh closure is recorded. Per-user/rank minimum canonical/projected PCC against the current control is0.9999917724 for native and rebound owners; independent FP64 RMS minimum is0.9999984950. The actual FP32 gamma, opposite live input owners and stale old output/stats checks pass. Timing is intentionally disabled for this instrumented run. These are parent-observed device results, not this agent's device execution.

The separate `tests/probe_fused_minimal_norm_validation.py` is ready for parent execution. It checks12 exact native validation failures with unchanged cache counts before the first legal fused launch, followed by B1/B32 legal calls and fresh input/gamma/stats/semaphore/output bindings. All positive per-rank/per-user outputs must reach PCC>=.995 to independent FP64 RMS. Generic API or allocation errors fail the harness and never count as native validation. CPU plan and syntax checks pass; hardware admission results are pending when this report is written.

```bash
env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  bash "$D/run_job.sh" fused_minimal_norm_validation25 \
  python -m models.autoports.qwen_qwen3_6_27b.tests.probe_fused_minimal_norm_validation \
  --layer 0 --output "$D/fused_minimal_norm_validation25.json"
```

## Offset storage row and remaining risk

[storage_row.patch](storage_row.patch) changes only the isolated helper/probe to add optional `--storage-row` (default0). It is prepared, not applied by this agent. [storage_row.cpu.json](storage_row.cpu.json) checks all10 rows ×32 logical batches through pure geometry AST, without importing TTNN. The native factory computes the actual bounding-box offset, offsets sender/all-to-all core sets, subtracts actual workers, translates multicast vectors/corners through the device logical-to-physical map, and translates actual stats/core-drain coordinates. The one-row x=0..7 case does not enter two-stage reduction. Therefore y=1..9 requires no native source change on inspection; row1/9 B1/B32 watcher and strict trace tests remain required. `compute_with_storage_grid_size=(8,1)` represents rectangle extent, while the input shard carries its offset; validator compares extent and the factory obtains actual device bounds.

Persistent output plus stats accumulate per-layer L1 ownership. Parent identified ~53,248 bytes/sender/layer in the production allocation plan: pinning64 owners to row0 is not viable. Distribute layer ownership across available rows and report the actual persistent allocation rather than judging one isolated instance's capacity. The isolated probe has duplicate test owners and two phases; it is not production memory accounting.

Broader B3/17/32, both layer kinds and norm weights, queued50, watcher and final whole-decoder timing/quality are parent-run work. Offset storage and final production lifetime need their own evidence. No native-family performance win or stage completion is claimed by this repair report.
