# AutoFix discriminator: preserve the native rank-one DST seed

Prepared CPU-only after the independent diagnosis in `native_update_precision_audit25.md`. No hardware run, TTNN/Torch import, build, native-source edit, production edit, or existing shared helper/probe edit. Original numerical failures remain untouched. This candidate is off-default and unverified.

## Files and bounded change

`tt/prepared_native_precise_seed.py` exports `PreparedNativePreciseSeed(initial_state)` and `.step(packed_qkv, g, beta)`, matching the existing prepared owners. It subclasses the precise-readout owner and changes only the decayed-state seed path:

1. Native FPU decay computes exactly the original FP32 DST result. That result is packed to original CB10 and an additional CB17 before releasing DST.
2. Native prediction continues reading CB10 through its original FPU path. Delta, K transpose and the outer-product matmul retain their source and arithmetic.
3. The rank-one update seeds DST by copying CB17 with full-FP32 direct unpack. It then explicitly reconfigures both sources to `(kcol,u)` before the unchanged FPU matmul accumulates into DST.
4. Precise readout, output writing, FP32 persistence, BF16 RNE persistence and per-head draining retain the existing implementation.

CB17 contains sixteen FP32 tiles, adding 65,536 L1 bytes per active worker. Source recurrence CB estimates become 606,208 bytes for FP32 state and 573,440 for BF16, excluding kernels/other allocations. CB17 is compute-owned: reserve/pack/push during decay, wait/copy/pop once per head. It has no FPU consumer and is the only new direct-unpack index. Existing CB10 remains Default, avoiding the invalid experiment of marking a live FPU input direct-to-Dest. No new global tensor, preparation layout, launch, collective, public-input restriction or full-tensor restore is added.

The new helper copies the already-audited static host descriptor method. Its normalized AST differs from precise readout only by CB17's allocation and direct-unpack index. It inherits the prototype's native-source read at owner construction; it is an isolated experiment, not a production-ready runtime source policy.

## Probe and evidence contract

`tests/probe_prepared_native_precise_seed.py` keeps the six existing paths and adds seed-preserving FP32/BF16 owners. The original/readout-only controls retain .995 per-rank/user/head output and state metrics with explicit failure status. Their known numerical failures do not prevent collecting seed results. Custom and seed families must pass the same .995 gates against persistence-matched FP64 and custom references. Original versus readout-only native persistent states must remain bit-identical; seed state is intentionally allowed to change. All eight paths retain exact eager/trace replay and address-rebinding checks.

The synthetic fixture and generator order are unchanged: B1 normal uses seed `52813 + 97*B + rank`, the same BF16 initial state and exact bank0 packed QKV/g/beta as `native_readout_perf25`. A new independent fixed-input run resets every state, then applies exactly 702 bank0 updates. Checkpoints are 1, 2, 4, 16, 64, 128, 256, 512 and 702. Every path's raw output/state, all prepared records, actual inputs, initial state, host bank0 and FP64 references are saved to a fresh `.pt` artifact **before numerical metrics or assertions**. Quality failures are collected through all checkpoints and only fail the case after the remaining measurements. `--fixed-steps` can shorten a compile smoke; the report explicitly marks it as lacking full 702 coverage.

For all three FP32 native families, checkpoint diagnostics contract the saved live/persistent FP32 state with the actual prepared Q in FP64. Reports include per-head q·k, centered reference-output norm, projected state-error norm and CPU-contraction versus hardware-output norm/PCC. This diagnostic is distinct from, and cannot replace, the original trajectory gate. BF16 state is excluded from that shortcut because live readout precedes BF16 persistence rounding.

The inherited changed-state/input tests, all-path rebound ownership, separate warmed traces, allocation tracking positive control, and 50 queued changed-input steps remain. All device banks/retained outputs exist before capture; checkpoint dumps allocate host tensors only. Timing remains optional, uninstrumented, rotates path order and measures the complete raw-QKV → preparation → recurrence → compact-output/state-update boundary. The timed trajectory is independently reset and validated; checkpoints are excluded from timing.

## CPU proof

`native_seed_cpu25_v2.json` passes:

- Inverting the four intentional source edits reconstructs precise-readout source exactly.
- Four missing-anchor negative cases reject source drift.
- Descriptor AST comparison permits only CB17 allocation/direct unpack.
- Original reader/writer/preparation, decay math, prediction, residual, outer product and persistence source remain unchanged.
- No retained FPU consumer takes CB17; direct-to-FPU transition is explicit.
- Existing all-B1..32 head/compact-output geometry proof passes.
- AST confirms raw dumps precede numerical metrics.
- No TTNN/Torch import. Python AST/Black checks pass.

Earlier `native_seed_cpu25.json` is preserved as intermediate provenance. These checks do not establish pack/unpack hardware behavior or a numerical improvement.

## Parent-owned commands

Run serially under the existing device owner, with fresh names. First a bounded compile/format smoke:

```bash
timeout 600s env TT_METAL_WATCHER=10 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_prepared_native_precise_seed \
  --batches 1 --modes normal --steps 2 --fixed-steps 4 --queued 0 --skip-timing \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/native_seed_smoke25.json
```

The discriminating trajectory, retaining all nine checkpoints and queued coverage:

```bash
timeout 1200s env TT_METAL_WATCHER=10 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_prepared_native_precise_seed \
  --batches 1 --modes normal --steps 4 --fixed-steps 702 --queued 50 --skip-timing \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/native_seed_fixed702_25.json
```

If matched cost evidence is needed, this command still saves timings and all checkpoint failures before its final numerical assertion:

```bash
timeout 1200s env -u TT_METAL_WATCHER -u TT_METAL_DEVICE_PROFILER TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_prepared_native_precise_seed \
  --batches 1 --modes normal --steps 4 --fixed-steps 702 --queued 50 --warmup 2 --samples 7 --replays 100 \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/native_seed_perf25.json
```

A successful seed-only result needs later changed-input/zero-state/batch/whole-layer gates before selection. An insufficient seed repair plus complete-boundary timing can support rejecting this adapted slower family without undertaking a full native arithmetic rewrite. Neither outcome is claimed in this preparation.
