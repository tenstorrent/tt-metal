# AutoFix: preserve recurrent state precision

Failing contract: real-weight linear attention must reach PCC>=.995 at advertised context, with FP32 persistent recurrence and traced device-only decode. Original16385 run failed prefill .992926 / decode .990660; state PCC .907568 while convolution PCC .999992. See AUTODEBUG.md and baseline_states.log.

Hypotheses and discriminating checks:

1. HF chunk scheduling corrupts state: CPU reference_control.py compared1024-token prefix calls plus tail against128-token calls. Last mixer PCC .999998, state .999984; insufficient to explain device failure.
2. BF16 versus FP32 Q/K L2: bf16_l2.log remained below acceptance and state .907493. Reverted.
3. FP32 state storage loses precision through TF32 unpack/FPU arithmetic each token: confirmed in source and repaired locally. Candidate initial blanket direct-unpack flags were rejected by static review before execution because matrix operands have incompatible requirements. Final candidate separates matrix-readable copies from direct-to-DST state/scalars.

Changed files: tt/fp32_recurrence.py, tt/kernels/fp32_recurrence.cpp, and FunctionalDecoder._delta call site. Generic-op descriptor retains upstream dataflow kernels and creates no persistent runtime tensors. Two staging CBs add131072 bytes/core. Total multi-token CB footprint671744 bytes/core. Original matrix inputs remain default-mode; recurrent state, decay and residual use FP32 direct copies/SFPU operations. HiFi4 matrix math remains approximate relative to CPU; measured PCC establishes adequacy for these cases, not bitwise equivalence.

Validation commands use the environment documented in ../work_log.md:

- `python -m models.autoports.qwen_qwen3_8_27b.tests.run_decoder --layer 0 --seq 16385 --debug-states --output .../long_linear_debug/fp32_local_v2.json`: prefill .999976456, state .999970257, decode .999977708, changed-input decode .999997556; three identical replays bitwise identical with TT_METAL_TRACE_ALLOC_TRACKING=1.
- Same runner with `--seq 262142 --debug-states`: prefill .999936640, state .999973834, decode .999996245 and .999994516 at positions262142 and262143. Artifacts ../linear_262142_decode_max.json/log.
- First local candidate probe failed before kernel execution because Shape does not accept a slice; changed `v.shape[2:]` to explicit indices. Preserved fp32_local.log.

Final validation completed: exact262144 prefill, final boundaries, B1/B8/B16/B32 tracked replay/resume,14 synthetic cases, reuse, watcher and accepted profiler rows. See ../README.md, ../PERFORMANCE.md and ../final_validation.log. Whole-stage closure is governed by ../STAGE_REVIEW.md.
