# AutoFix Report

## Starting Evidence

- Source: `AUTODEBUG.md`, TP4 split-greedy sampler findings.
- Original evidence: a 62,080-wide local `TopKDeviceOperation` took 9.697361 ms on one core; padding to 65,536 remained on the single-core factory and regressed reduced token-out replay from 12.667 to 13.618 ms/token.
- Factory verification: `topk_device_operation.cpp::select_program_factory` requires power-of-two width `< uint16_t::max()` for multicore TopK. Therefore 62,080 and 65,536 are both ineligible, while 32,768 is legal.

## Hypothesis Experiments

- Hypothesis: two independently padded 32,768-wide local TopK reductions, followed by a 64-to-32 merge, preserve exact token IDs and make invalid padded-vocabulary IDs impossible.
  Experiment: host planning test plus TP4 sampler-only traced probe with maxima at global IDs 0, 32,767, 32,768, and 248,063. Invalid IDs 248,064..248,319 were deliberately assigned zero logits above ordinary valid logits.
  Result: initial merge returned merge positions because the 64-wide single-core TopK factory hardcodes generated indices and ignores `indices_tensor` (source comment references GH #36329). A device-side gather fixed the merge. A second probe showed first-stage indices were chunk-relative; explicit +32,768 for the second chunk fixed them. Final trace replay returned `[0, 32767, 32768, 248063]` on every TP rank, with no invalid ID.
  Verdict: verified.
  Evidence artifacts: `models/autoports/qwen_qwen3_6_27b/tests/split_topk_sampler_probe.py`; console command below.
  Fix: opt-in `local_topk_num_chunks=2`; explicit sharded invalid-vocabulary additive mask; two legal 32,768 TopKs; device-side chunk-base restoration; 64-wide values merge plus `ttnn.gather` of original candidate IDs; Qwen enables the contract by default.
  Verification:
  - `python -m pytest models/common/sampling/tests/test_local_topk_plan.py -q` -> 3 passed.
  - `timeout 300 python models/autoports/qwen_qwen3_6_27b/tests/split_topk_sampler_probe.py` -> `SPLIT_TOPK_SAMPLER_OK [0, 32767, 32768, 248063]`.
  - `python -m compileall -q models/common/sampling/tt_sampling.py models/autoports/qwen_qwen3_6_27b/tt/generator.py` -> passed.

## Final Status

- Fixed for sampler-only traced TP4 correctness, including the chunk boundary, last valid vocabulary ID, invalid-vocabulary masking, and direct persistent output overwrite.
- Remaining verification: collect a sampler/reduced-token-out profile proving both 32,768 first-stage TopK calls select multicore and improve end-to-end replay; rerun fixed-seed non-greedy top-k/top-p coverage. Those performance and sampled-mode gates are intentionally not claimed by this focused correctness repair.
