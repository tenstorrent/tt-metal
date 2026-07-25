# AutoFix: Sparse Fidelity and Final-Topology Attention Precision

## Starting evidence

- `stage_review.md` P1 found that every retained gate/up/down sparse row is
  LoFi even though the stage documentation described the expert policy as
  HiFi4. Sparse matmuls account for 54.79% of profiled prefill.
- The retained profiler rows in `perf/final/ops.csv` resolve the old implicit
  sparse config exactly as
  `LoFi, math_approx_mode=0, fp32_dest_acc_en=0, packer_l1_acc=1`.
- The previous BFP4 attention measurement used the slower DRAM-sharded family,
  not the selected interleaved packed-QKV/local-O family.

## Hypothesis experiments

### Sparse fidelity was legal but not wired

- Hypothesis: `ttnn.sparse_matmul` accepts a compute-kernel config, but the EP
  call sites omitted it, so their explicit matmul program configs selected the
  generic LoFi default.
- Source experiment:
  - `matmul_nanobind.cpp:1153-1167` binds `compute_kernel_config` for
    `sparse_matmul`.
  - `matmul_device_operation.cpp:2201-2207` selects LoFi whenever an explicit
    program config is present; lines 2251-2257 establish exact math, BF16 DST,
    and L1 packer defaults.
  - The sparse factory consumes the resolved config at
    `sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp:106-107`.
- Verdict: verified. There is no TTNN/API blocker.
- Fix: `MultichipConfig.ep_sparse_math_fidelity` now exposes
  `lofi|hifi2|hifi4`, defaulting to the measured prior LoFi behavior.
  `MultichipDecoder` constructs a sparse-only config while holding approximate
  math off, FP32 DST off, and L1 packer accumulation on, then passes it to all
  four packed/split gate/up/down call sites. This isolates fidelity from the
  unrelated global decoder compute config.
- A/B control: `MULTICHIP_EP_SPARSE_MATH_FIDELITY`.
- Provenance: performance JSON now records `ep_prefill_geometry.sparse_math_fidelity`.

### BFP4 can be materialized on the selected attention topology

- Hypothesis: the selected interleaved local attention weights have legal
  BFP4 shapes and do not require the DRAM-sharded program family.
- Source experiment:
  - `_shard_to_tp` already accepts a dtype and materializes tiled DRAM-interleaved
    per-rank tensors.
  - Packed local QKV is `[2880,1280]` per rank (90 by 40 tiles); local O is
    `[1024,2880]` per rank (32 by 90 tiles). Both are tile-aligned.
  - The TTNN linear path supports BFP4 tiled weights with the existing explicit
    local QKV/O program configs; no layout conversion or padding adaptation is
    required at load time.
- Verdict: verified at the materialization/API boundary; hardware PCC/perf is
  the remaining A/B, not an implementation blocker.
- Fix: `MultichipConfig.interleaved_attention_weight_dtype` now directly
  materializes both packed local QKV and local O as `bfloat16|bfloat8_b|bfloat4_b`.
  It defaults to BF16 and is rejected when the separate DRAM-sharded attention
  family is enabled, preventing an incoherent cross-family test.
- A/B control: `MULTICHIP_INTERLEAVED_ATTENTION_WEIGHT_DTYPE`.
- Provenance: performance JSON distinguishes
  `interleaved_attention_weight_dtype` from `dram_attention_weight_dtype`.

## Verification

Static focused checks:

```text
python -m pytest -q \
  models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_multichip_runtime_contract_and_fallback_audit \
  models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_multichip_perf_candidate_parsing
```

Result: 2 passed. The runtime audit asserts the production defaults remain
LoFi/BF16 and all four sparse calls receive the sparse-only compute config.
`py_compile` and `git diff --check` also passed. `ruff` was unavailable in the
checkout environment.

Hardware commands handed to the root agent for serialized execution:

```text
MULTICHIP_TEST_CONFIG_FROM_ENV=1 \
MULTICHIP_EP_SPARSE_MATH_FIDELITY=hifi2 \
python -m pytest -q \
  models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_real_weight_prefill_decode_matches_single_chip_optimized \
  -k ep --junitxml=<hifi2-pcc-artifact>

MULTICHIP_TEST_CONFIG_FROM_ENV=1 \
MULTICHIP_EP_SPARSE_MATH_FIDELITY=hifi4 \
python -m pytest -q \
  models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_real_weight_prefill_decode_matches_single_chip_optimized \
  -k ep --junitxml=<hifi4-pcc-artifact>

MULTICHIP_TEST_CONFIG_FROM_ENV=1 \
MULTICHIP_INTERLEAVED_ATTENTION_WEIGHT_DTYPE=bfloat4_b \
python -m pytest -q \
  models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_real_weight_prefill_decode_matches_single_chip_optimized \
  -k ep --junitxml=<bfp4-attention-pcc-artifact>
```

Passing PCC candidates must then use `test_multichip_decoder_perf` with
`RUN_MULTICHIP_DECODER_PERF=1`, the same control, and a unique
`MULTICHIP_PERF_RESULT_PATH` for warmed prefill and traced decode evidence.

## Final status

The API/materialization blockers are fixed with off-by-default controls and
static evidence. Hardware A/B is intentionally delegated to the root agent so
only one owner uses the four-chip mesh at a time. Production defaults remain
the measured LoFi sparse policy and BF16 interleaved attention until those
real-path PCC/performance results justify a change.
