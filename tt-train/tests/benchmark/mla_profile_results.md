# TTML MLA Section Profile

This note records initial section-level timings for the TTML DeepSeek
`MultiHeadLatentAttention` implementation. The goal is to identify which parts
of MLA are worth optimizing before changing kernels or model code.

## Environment

- Branch: `mdragula/mla_benchmark`
- Build: clean Release build in `build_mla_profile`
- Device: Wormhole N300 class machine, single TTML device context
- Benchmark: `tt-train/tests/benchmark/mla_profile_benchmark.py`
- Grad mode: disabled
- Warmup / iterations: `--warmup 1 --iterations 3`
- Estimate mode:
  - memory traffic is a shape-derived lower-bound estimate for BF16 tensors
  - FLOPs use dense matmul / SDPA formulas and are approximate for RoPE/softmax

Run commands:

```bash
source python_env/bin/activate
export PYTHONPATH=/home/ubuntu/tt-metal/build_mla_profile:/home/ubuntu/tt-metal/build_mla_profile/ttml:/home/ubuntu/tt-metal/tt-train/sources/ttml:/home/ubuntu/tt-metal/tools:$PYTHONPATH
export LD_LIBRARY_PATH=/home/ubuntu/tt-metal/build_mla_profile/lib:$LD_LIBRARY_PATH

python tt-train/tests/benchmark/mla_profile_benchmark.py \
  --cases smoke,nano-s128,nano,nano-b2,tiny-short,tiny-s1024,tiny-full \
  --warmup 1 \
  --iterations 3 \
  --end-to-end \
  --csv tmp/mla_profile_sweep.csv
```

## Caveats

- This is a section-level host-timed benchmark with `ttnn.synchronize_device`
  after every section. It is useful for relative stage attribution, not a final
  end-to-end throughput number.
- It profiles the TTML training-style MLA path, which is the naive
  non-absorbed implementation in `ttml.models.deepseek.mla`. Production
  DeepSeek demo stacks use different absorbed MLA paths and should be profiled
  separately.
- Grad mode is disabled, so these are forward-only stage timings. Backward
  profiling should be added before prioritizing training-only optimizations.
- The section sum is not equal to a fully fused end-to-end MLA latency. Extra
  synchronizations intentionally expose stage costs and prevent overlap.
- `--end-to-end` times a normal `module(x, mask)` call with only one
  synchronize before and after the whole MLA forward. This is closer to the
  user-visible forward latency, while the section profile explains where time
  goes.
- First-run JIT compilation is excluded by warmup where practical, but JIT cache
  effects can still affect short runs.

## Shapes

| case | B | S | dim | heads | q_lora | kv_lora | qk_nope | qk_rope | v |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| smoke | 2 | 64 | 64 | 2 | 32 | 32 | 32 | 32 | 32 |
| nano-s128 | 1 | 128 | 512 | 8 | 256 | 128 | 64 | 32 | 64 |
| nano | 1 | 256 | 512 | 8 | 256 | 128 | 64 | 32 | 64 |
| nano-b2 | 2 | 256 | 512 | 8 | 256 | 128 | 64 | 32 | 64 |
| tiny-short | 1 | 512 | 1536 | 12 | 512 | 256 | 96 | 32 | 128 |
| tiny-s1024 | 1 | 1024 | 1536 | 12 | 512 | 256 | 96 | 32 | 128 |
| tiny-full | 1 | 2048 | 1536 | 12 | 512 | 256 | 96 | 32 | 128 |

`smoke` is a script sanity check, not an optimization target.
`nano` matches `tt-train/configs/model_configs/moe/nano_deepseek_char.yaml`;
`nano-s128` shortens sequence and `nano-b2` doubles batch. `tiny-short`,
`tiny-s1024`, and `tiny-full` use dimensions from
`tt-train/configs/model_configs/moe/tiny_deepseek_char.yaml`, with
sequence length varied to show scaling.

## Sweep Results

Average section-sum and end-to-end forward time, in microseconds:

| case | section-sum avg us | end-to-end avg us | top measured stages |
|---|---:|---:|---|
| smoke | 4263 | not recorded | qk_assemble 23.7%, sdpa 20.1%, kv_up_split 14.2% |
| nano-s128 | 3720 | 2962 | sdpa 23.2%, kv_up_split 15.9%, q_rope 9.9% |
| nano | 3756 | 3086 | sdpa 23.4%, kv_up_split 16.7%, kv_down_split 9.6% |
| nano-b2 | 4332 | 3071 | sdpa 23.3%, kv_up_split 16.3%, q_rope 9.8% |
| tiny-short | 5128 | 3250 | sdpa 23.5%, kv_up_split 22.2%, q_split 12.3% |
| tiny-s1024 | 9400 | 6624 | sdpa 32.1%, kv_up_split 22.5%, q_split 12.7% |
| tiny-full | 19100 | 17309 | sdpa 54.7%, kv_up_split 20.2%, q_split 11.0% |

Average section times, in microseconds:

| stage | nano-s128 | nano | nano-b2 | tiny-short | tiny-s1024 | tiny-full |
|---|---:|---:|---:|---:|---:|---:|
| q_projection | 364 | 338 | 401 | 338 | 634 | 547 |
| q_split | 262 | 271 | 407 | 632 | 1196 | 2109 |
| q_rope | 369 | 356 | 425 | 375 | 561 | 400 |
| kv_down_split | 367 | 362 | 384 | 396 | 490 | 368 |
| k_rope_broadcast | 326 | 334 | 373 | 354 | 532 | 396 |
| kv_up_split | 590 | 629 | 706 | 1137 | 2114 | 3864 |
| qk_assemble | 315 | 331 | 347 | 344 | 384 | 349 |
| sdpa | 864 | 879 | 1010 | 1208 | 3015 | 10453 |
| heads_fusion | 122 | 115 | 124 | 140 | 180 | 180 |
| output_projection | 140 | 142 | 154 | 205 | 295 | 434 |

## Initial Interpretation

- SDPA is the largest component, especially at longer sequence length.
  At `S=2048`, it is roughly half of the section sum.
- Outside SDPA, `kv_up_split` is the largest local target across all useful
  shapes. It includes `kv_norm`, `wkv_b`, `split_heads`, and slices into
  `k_nope` and `v`.
- `q_split` becomes significant at larger hidden/head dimensions.
- Igor's proposed `mla_qkv_assemble` targets a real part of the graph:
  head split / K/V demux / K rope broadcast / Q/K concat. The direct
  `qk_assemble` section alone is small at `S=2048`, but the broader layout
  movement around `q_split`, `kv_up_split`, and broadcast is still meaningful.
- `heads_fusion` and `output_projection` are not top targets in this TTML
  single-device profile.

## Estimate Mode Snapshot

For `tiny-full` with `--end-to-end`, the section-sum average was about
`19.1 ms`, while the normal unsplit forward averaged about `17.3 ms`.
This confirms the section profile adds synchronization overhead, but the
stage ordering remains useful.

Selected `tiny-full` estimated lower-bound traffic / achieved FLOPs:

| stage | avg us | est traffic MiB | est TFLOP/s |
|---|---:|---:|---:|
| q_projection | 547 | 17.00 | 11.79 |
| q_split | 2109 | 12.00 | 0.00 |
| kv_up_split | 3864 | 12.81 | 0.73 |
| qk_assemble | 349 | 24.00 | 0.00 |
| sdpa | 10453 | 24.00 | 2.49 |
| output_projection | 434 | 16.50 | 22.29 |

Notes:

- `q_split`, `qk_assemble`, and `heads_fusion` are memory movement / layout
  stages, so MFU is not meaningful and estimated TFLOP/s is zero.
- `kv_up_split` mixes a real matmul (`wkv_b`) with norm, split, and slices, so
  its low TFLOP/s estimate should not be interpreted as pure matmul efficiency.
  It is a strong hint that this section is more than just compute.
- `qk_assemble` itself is fast at `S=2048`, but its estimated 24 MiB of
  read/write traffic is exactly the kind of movement Igor's fused assembly work
  tries to reduce or combine with neighboring layout steps.
- SDPA dominates full sequence latency, but SDPA is owned by separate ongoing
  work. Outside SDPA, `kv_up_split` and `q_split` are the largest measured
  targets in this TTML implementation.

## Follow-Up Measurements

To make optimization decisions harder to dispute:

1. Add backward profiling sections. The current benchmark is forward-only.
2. Add device-profiler or Tracy runs for kernel/device time attribution.
3. Compare this baseline with Igor's `mla_qkv_assemble` branch to quantify
   the actual layout-fusion win.
4. Add a memory high-water-mark measurement if TTML exposes one; the current
   report only estimates per-stage traffic from shapes.
