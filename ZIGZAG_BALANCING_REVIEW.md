# Causal Zigzag Balancing Readiness

Current follow-up list for causal ring-joint SDPA zigzag balancing.

## Resolved

- Fixed the causal ring-joint runtime hang.
  - Even per-head Q chunk counts use pair-aligned zigzag Q distribution, preserving the fast skipped-Q path without desynchronizing K/V multicast chains.
  - Odd per-head Q chunk counts use the reader's chain-bypass fallback for skipped causal ring iterations so active cores fetch K/V directly instead of waiting on a bypassed chain hop.

- Clarified causal + joint attention.
  - The op validation rejects causal + joint attention today.
  - Regression coverage exists in `test_ring_joint_attention_rejects_causal_joint_attention`.

- Split the causal layout contract from local zigzag work balancing.
  - Causal ring-joint always enables local per-chip Q zigzag scheduling for load balance.
  - `input_is_zigzag_layout` now only describes the physical cross-chip input layout.
  - Sequential input layout skips future KV shards numerically across chips, while zigzag input layout keeps the existing cross-chip half-shard behavior.

- Fixed sequential causal finalization order.
  - Sequential layout now scans the actual alternating ring order to find the last active non-future KV iteration.
  - This prevents local shards from normalizing too early on chips that still need earlier sequential KV shards later in the ring order.

## Remaining Work

- Resolve code size back to the default worker-L1 setup.
  - Current bring-up uses `RING_JOINT_WORKER_L1_SIZE = 1457664` on Blackhole causal ring-joint tests.
  - The default kernel-config buffer was too small for at least the cold sequential `q128/k320` kernel variant.

- Clean up naming in tests/model code.
  - Python-side `is_balanced` now means physical zigzag input layout, not whether the op balances causal work.
  - Rename to `input_is_zigzag_layout` or `use_zigzag_layout` where practical.

- Keep tracking ring-joint perf as code-size work continues.
  - Existing perf coverage `test_ring_joint_attention_perf_check[mla_100k-q160-k320-ring4]` passes with zigzag physical layout.
  - MLA-like misaligned zigzag coverage now exists as `test_ring_joint_attention_perf_check[mla_100k-q160-k320-zigzag_misaligned-ring4]`.
    - This keeps the q160/k320 tiling from the MLA gate but uses local sequence length `3136`, so the physical zigzag half is not aligned to Q chunks.
  - Odd local-Q-chunk zigzag coverage now exists as `test_ring_joint_attention_perf_check[mla_100k-q160-k320-zigzag_odd_qchunks-ring4]`.
    - This keeps the q160/k320 tiling from the MLA gate and uses local sequence length `3328`, so each local shard has 21 Q chunks and 11 K chunks.
  - Sequential physical layout correctness passes, but expectedly has lower utilization because causal work is imbalanced across chips when the physical input is not cross-chip zigzagged.
  - DeepSeek MLA perf parametrization now includes sequential and zigzag layouts, but the local 4-device setup cannot run the `2x4` perf mesh directly.

## Validation Status

- Passed:
  - `./build_metal.sh --release`
  - `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/sdpa`
    - Result: 29 passed, 3 skipped.
  - `env TT_METAL_CACHE=/tmp/tt_metal_cache_zigzag_expanded_odd_<fresh> scripts/run_safe_pytest.sh 'tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_odd_num_q_chunks' -s`
  - `env TT_METAL_CACHE=/tmp/tt_metal_cache_zigzag_contract_pair_<fresh> scripts/run_safe_pytest.sh 'tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_causal_layout_contract[sequential-layout]' 'tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_causal_layout_contract[zigzag-layout]' -s`
  - `env TT_METAL_CACHE=/tmp/tt_metal_cache_zigzag_mla_smoke_<fresh> scripts/run_safe_pytest.sh 'models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_ring_joint_mla.py::test_mla_sdpa[blackhole-sequential-rpxup-2x2-line-1link-pcc_check-no_trace-single_run-1-128-1-576-128-seq100k-q_bf16_kv_bf8]' 'models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_ring_joint_mla.py::test_mla_sdpa[blackhole-zigzag-rpxup-2x2-line-1link-pcc_check-no_trace-single_run-1-128-1-576-128-seq100k-q_bf16_kv_bf8]' -s`
  - `env TT_METAL_CACHE=/tmp/tt_metal_cache_zigzag_correctness_<fresh> scripts/run_safe_pytest.sh 'tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_causal_layout_contract[sequential-layout]' 'tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_causal_layout_contract[zigzag-layout]' 'tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_odd_num_q_chunks' 'tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_rejects_causal_joint_attention' -s`
    - Result: 7 passed.
    - Sequential layout: PCC `0.9995729864655006`, RMSE `0.006618`.
    - Zigzag layout: PCC `0.9995719113461311`, RMSE `0.006626`.
  - `env SDPA_PERF_CHECKS=1 scripts/run_safe_pytest.sh 'tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_perf_check[mla_100k-q160-k320-ring4]' -s`
    - Result: passed, duration `5.165 ms`, math utilization `58.56%`, expected band `[58.11%, 58.69%]`.
  - `scripts/run_safe_pytest.sh 'tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_zigzag_odd_qchunk_perf_impl[mla_100k-local3328-q160-k320-zigzag-odd-qchunks]' -s`
    - Result: passed.
  - `env SDPA_PERF_CHECKS=1 scripts/run_safe_pytest.sh 'tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_perf_check[mla_100k-q160-k320-zigzag_odd_qchunks-ring4]' -s`
    - Result: passed, duration `6.875 ms`, math utilization `47.59%`, expected band `[47.26%, 47.74%]`.
  - `scripts/run_safe_pytest.sh 'tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_zigzag_misaligned_perf_impl[mla_100k-local3136-q160-k320-zigzag-misaligned]' -s`
    - Result: passed.
  - Checked accuracy for the same misaligned q160/k320 setup:
    - Result: passed, PCC `0.9955360699990052`, RMSE `0.021219`.
  - `env SDPA_PERF_CHECKS=1 scripts/run_safe_pytest.sh 'tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_perf_check[mla_100k-q160-k320-zigzag_misaligned-ring4]' -s`
    - Result: passed, duration `5.119 ms`, math utilization `56.75%`, expected band `[56.52%, 57.08%]`.
    - Device durations were balanced: `[5.119, 5.110, 5.069, 5.097] ms`.
  - `env SDPA_PERF_CHECKS=1 scripts/run_safe_pytest.sh 'tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_perf_check[mla_100k-q160-k320-ring4]' -s`
    - Result: passed after perf-check dispatch refactor, duration `5.167 ms`, math utilization `58.54%`.
  - One-off Tracy profile for sequential physical layout contract case:
    - Result: passed, duration `8.149 ms`, math utilization `37.12%`.

- Pending:
  - Full requested nightly SDPA sweep:
    - `scripts/run_safe_pytest.sh tests/ttnn/nightly/unit_tests/operations/sdpa/`
