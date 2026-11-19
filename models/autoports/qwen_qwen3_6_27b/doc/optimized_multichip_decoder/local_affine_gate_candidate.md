# Implemented local affine/gate candidate

New files only: `tt/local_affine_gate_candidate.py` and `tests/probe_local_affine_gate.py`. Production methods/defaults and frozen decoder are untouched. This implements the bounded adaptation described in `local_affine_gate_next_proposal.md`; no device execution or latency claim by the preparing agent.

`local_affine_gate(normalized, weight, packed, *, output=None, cores=None)` takes BF16 TILE DRAM `[1,B*12,128]`, `[1,1,128]`, and compact `[1,B,4160]` (at least4096 columns), returning BF16 TILE DRAM `[1,B,1536]` for B1–32. It changes precisely three frozen reader anchors: the two head-stride48 expressions become12 and gate tile base320 becomes80. Frozen compute and writer are reused directly, byte-identical. The wrapper schedules48 complete output tile tasks, clamps worker count to48, and retains the original CB sizes, formats, barriers and alignment fallback. No state tensor is touched.

The compute preserves the separate BF16 norm-times-gamma rounding boundary and FP32 SiLU/multiply with final BF16 rounding. RMSNorm and recurrent computation are outside the helper and probe. The reference spells out the same production multiply, reshape, gate slice, two typecasts and final fused SiLU/multiply. Its final output is preallocated through `output_tensor`; no arithmetic primitive/config is substituted.

Optional output ownership is explicit: matching shape/padding/dtype/layout/DRAM/device, no alias with inputs. The helper stores no owner reference or tensor cache. No instance monkeypatch or shallow-bound method is installed. Parent can call the helper directly in the compact `s==1` branch after unweighted RMSNorm, then feed the unchanged output projection; the previous proposal describes the default-off policy/fallback. Legacy `[B,1,4160]` packed layouts must retain their old path; nonaligned **batch sizes** are supported with native tile padding.

## CPU evidence

The focused probe's `--cpu-check` passed without importing Torch/TTNN. It verifies the current reader/compute/writer against frozen commit8502f42bca20dcad1507fbc8cebf7032220b7d8e and reverses the three reader edits to recover the exact original. Compute SHA256 is `842e02998fe429dbe1cfda6f68a0392f7fdedafc4630751d8d9838bbf153a5f1`; writer SHA256 is `5cb5612c39acea7eb787cc19c7b00aa91f7cb044e75ccfd39f22f45c00565f5e`.

All32 batch extents pass element coverage, tiled face bounds, aligned64-byte fallback bounds, per-user scratch bounds and worker partition checks. The actual host wrapper body also executes against descriptor-only stubs for32 batches ×6 core limits; all48 output tiles have one owner, CB byte sizes match the frozen helper, and the emitted compute/writer source strings remain unchanged. This validates source/descriptor/index construction only. Syntax and formatting checks pass.

## Focused device protocol (parent only)

The default covers every B1–32 with normal inputs plus B3 cores1 wrap. `--modes` adds zero norm, zero gate and alternating±32 gate controls. Rank/user/head values differ; every step changes normalized input, gamma and packed gate. Outputs start as NaN, exposing missing live writes. Comparisons require exact BF16 bits, finite values and every rank/user/head; no PCC fallback. The probe does not yet inject NaN into physical input dead padding.

All fixture banks, active input buffers and both output buffers are allocated before capture. Eager reference temporaries die on return; both programs and every input-copy shape warm before the first capture. Capture/replay checks forbid new program-cache entries. All retained buffers outlive both traces, and traces release before buffers deallocate. No allocation-tracking acknowledgement, exemptions or skip-program-cache setting is used. Changed-input replay compares both implementations against saved eager outputs. Alternating serialized trace timings include the full reference region and the whole fused helper; both outputs are checked again after timings.

Run from repository root in the parent's established device environment, with fresh output names:

```bash
D=models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder
python -m models.autoports.qwen_qwen3_6_27b.tests.probe_local_affine_gate --cpu-check

timeout 180s env TT_METAL_TRACE_ALLOC_TRACKING=1 python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_local_affine_gate --batches 32 --low-cores 0 --output "$D/local_affine_gate_smoke.json"

timeout 600s env TT_METAL_TRACE_ALLOC_TRACKING=1 python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_local_affine_gate --output "$D/local_affine_gate_all_batches.json"

timeout 600s env TT_METAL_TRACE_ALLOC_TRACKING=1 python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_local_affine_gate --batches 1 2 3 17 31 32 --modes zero_norm zero_gate extreme_gate --output "$D/local_affine_gate_controls.json"
```

A separate watcher invocation uses the parent's watcher settings and a fresh artifact name; do not interpret watcher timings as delivery latency. If exact comparison fails, preserve the JSON and investigate the first differing head/stage rather than weakening the gate. Parent integration and whole-layer benchmarks follow only after focused validation. No full-layer wrapper is installed by this probe.
