# Per-chunk projection kernel probe

`tests/probe_multichip_prefill_kernels.py` is a CPU-prepared, parent-serialized hardware probe. No devices were used to implement it. It produces a JSON provenance log and matching CSV with actual shape, logical/physical N, dtype, program configuration, PCC, changed-input trace result, warmed median microseconds, and microseconds per row. It accepts one role per invocation, explicit row counts/families/layouts, and bounded sample/replay counts. There is no automatic exhaustive device queue or default selection.

## Role and numerical scope

| Layer kind | Roles |
|---|---|
| GDN layer0 | `packed_gdn`, `packed_gdn_residual`, `joint`, `gateup`, `down`, `attention_out` |
| Full-attention layer3 | `packed_qkvg`, `gateup`, `down`, `attention_out` |

All weights come from the actual checkpoint's TP4 setup and must be BFP8. Primary/correction/joint outputs stay FP32; all other projection outputs are BF16. Attention or MLP compute config is inherited unchanged. The joint role uses an existing joint weight, or prepares one with the existing exact-quantized-value setup helper without installing a forward hook. If final selection removes separate weights, testing the old primary/correction roles requires an explicit retained-weight policy or the recorded earlier evidence; the probe does not silently reconstruct a different reference.

Activations are captured during an untimed setup-only full TP4 prefill on recorded real text inputs (`text_l{0,3}_b32_s254.pt` by default). Normalized inputs supply packed projections and gate/up; the actual `_linear` operand supplies down/output projections. Host readback exists only in this explicit capture setup and correctness checks, never in the timed trace. This is not a model runtime-fallback audit. Each rank keeps its real local weight/activation shard; no single-chip or replicated model fallback is used.

The oracle uses the same captured inputs and decoded weights with the inherited2D configuration and an exact setup interleaved weight copy. Every candidate must be finite and PCC>=.995 on retained logical columns for every rank, both original and changed inputs. First/last original-input trace replay must be bitwise equal; the changed input must match its separately computed oracle. The reference is a TTNN arithmetic control, not HF mathematical accuracy. Whole-layer output/state/per-user/contribution gates remain mandatory before keeping a candidate.

`gateup` measures the unfused packed gate/up projection; it is not numerically the fused SwiGLU result. It informs a topology-family comparison but cannot by itself replace the current fused SwiGLU timing.

## Families and source-based adaptations

- `inherited2d`: current grid/K/subblock/per-core settings, including role-specific `prefill_projection_configs`. The accepted `matmul_row_cap` remains part of this control: M1024/M4096 use repeated512-row calls plus concat unless the selected policy has changed the cap. Actual kernel rows/configs are logged, so those results cannot be mislabeled as one4096-row native kernel.
- `tuned2d`: caller-selectable grid/K block, with output height block1 to bound CB requirements when testing a single large logical chunk.
- `1d-in0`: M fits entirely in per_core_M, N spans the chosen grid; small internal H/W blocks divide per-core extents.
- `1d-in1`: N fits entirely in per_core_N, M spans the grid; small internal blocks retain the logical N tail in the final block.
- `minimal`: native minimal_matmul with caller-selectable M/K/N blocks, BF16/BFP8 inputs and FP32 or BF16 output as required. Default M/K/N blocks are2/currentK/8 for M<=512 and4/currentK/16 for larger M.

Native `matmul_device_operation.cpp` requires one M partition for mcast_in0 and one N partition for mcast_in1. `matmul_nanobind.cpp` exposes the internal output block H/W options used here. All per-core/block/subblock divisibility constraints and FP32 subblock area<=4 are enforced in the generated configs. The requested grid is checked against the actual device grid before execution.

**Important source adaptation:** the ordinary1D mcast-in0 factory treats a non-global-CB sharded weight as locally resident (`in1_is_locally_sharded = in1_is_sharded && !use_global_cb`) and binds its buffer address into an L1 circular buffer. This is unsuitable for the model's ordinary DRAM-sharded weight without a dedicated delivery path. When `1d-in0` is requested with `dram_sharded`, the probe explicitly records that source blocker and measures the exact interleaved weight copy instead. It does not launch an unsafe DRAM-address-as-L1 trial or claim a direct DRAM-sharded measurement. Dedicated gather/prefetch delivery is a different family. Other meaningful families retain the requested weight layout.

`minimal_matmul_device_operation.cpp` supports BF16/BFP8/BFP4/FP32 input formats, rank>=2, matching K, tiled data, and generic tensor accessors. The probe retains BFP8 despite broader API support. Shape legality does not prove L1 capacity; preserve any resource failure, then retry smaller blocks or another grid/layout. A first error is not a material family rejection.

The interleaved weight copy is prepared once outside timing and verified to have identical decoded values on all four ranks. A deployed persistent copy would add resident capacity; account for it, or measure conversion within the full-layer path if it remains per call. The probe timings exclude this setup cost by design. Inputs and outputs remain interleaved DRAM for a common kernel-family comparison; role-specific input-L1 policy is not automatically credited as a measured activation-placement win here.

## Parent commands

A matching inherited2D/interleaved control is always included in the table, even when only one candidate family is requested. Start with a bounded family/role rather than all combinations:

```bash
source python_env/bin/activate
source /home/ttuser/qwen36-27b/lane-env.sh
D=models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder
unset TT_METAL_WATCHER TT_METAL_DEVICE_PROFILER
TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 MULTICHIP_JOB_TIMEOUT=900 bash "$D/run_job.sh" prefill_kernel_l0_primary_m128 \
  python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_prefill_kernels \
  --layer 0 --role packed_gdn --rows 128 \
  --family inherited2d tuned2d 1d-in0 1d-in1 minimal --weight-layout interleaved \
  --grid 8 4 --samples 5 --replays 10 \
  --output "$D/prefill_kernel_l0_primary_m128.json"
```

Then cover the requested chunk table rows64,128,256,1024,4096 in deliberate serial jobs; choose applicable roles from the table above. Compare `--weight-layout dram_sharded interleaved` for2D/minimal/1d-in1. Use `--k-block 4` or `--minimal-blocks 2 4 8` for an explicit adaptation, preserving a new artifact name. `--policy-file` may supply an explicit candidate policy; it is recorded and does not turn the microbenchmark into a final-default measurement.

Watcher runs use the same command with `TT_METAL_WATCHER=10` and no profiler. The probe intentionally omits timing under watcher but keeps eager, changed-input trace and first/last gates. Never combine watcher and profiler: the script rejects that environment. Under profiler it emits `PERF_PREFILL_KERNEL` / `PERF_PREFILL_KERNEL_END` Tracy signposts carrying role/shape/family/layout metadata and drains profiler data before closing the mesh. Timed CSV entries are synchronized warmed trace wall latency, **not** device-op microseconds; collect/render the corresponding Tracy CSV for device-kernel attribution.

Existing output paths are rejected to preserve failed and accepted evidence. The first failure saves partial JSON/CSV and propagates the error. Run each adapted retry under a new name; follow device-use triage before restarting after a hang/watcher error. The probe never resets devices.

## CPU preparation evidence

Syntax and Black checks pass. Imported the module without importing torch or TTNN and checked175 combinations: seven actual K/N geometries, five requested row counts, and five families. Per-core/block/subblock divisibility, FP32 destination area,1D partition count and core count constraints all passed. This does not validate native numerical results, L1 capacity, synchronization, or performance. No hardware results exist from this preparation task. Select any apparent winner only after a coherent whole-layer PCC/perf comparison using the same policy and final-default reproduction.
