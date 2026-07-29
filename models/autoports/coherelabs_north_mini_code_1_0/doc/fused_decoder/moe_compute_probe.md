# `ttnn.experimental.moe_compute` North-Mini probe

## Contract

- Operation path: `compute_only=True`, single Blackhole card, 1x1 mesh.
- Device fixture: `DispatchCoreAxis.COL`, production 11x10 worker grid.
- Shape: 128 experts, top-8, hidden size 2048, intermediate size 768.
- Numeric policy accepted by the operation: BF16 input and routing scores,
  BFLOAT4_B packed rank-6 weights, SwiGLU activation, no bias.
- The probe is explicit-only and is not collected by the normal
  `test_fused_decoder.py` suite.

## Batch-1 / one-token result

Hypothesis: the operation's `compute_only=True` path can execute the smallest
North-Mini decode shape before integration work begins.

Exact command:

```bash
NORTH_MOE_PROBE_TOKENS=1 pytest -q \
  models/autoports/coherelabs_north_mini_code_1_0/tests/moe_compute_probe.py \
  -s \
  --junitxml=models/autoports/coherelabs_north_mini_code_1_0/doc/fused_decoder/moe_compute_probe.junit.xml
```

Observed on 2026-07-29:

```text
tokens_per_device: 1, total_tokens: 1
experts: 128, experts_per_device: 128
selected_experts_k: 8
hidden_size: 2048, N: 768
output_height_shard_dim: 4
output_width_shard_dim: 4
========== Running op (compute_only=True) ==========
Fatal Python error: Floating-point exception
...
tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_compute_single_card.py,
line 330, in _run_moe_compute_single_card_test
...
models/autoports/coherelabs_north_mini_code_1_0/tests/moe_compute_probe.py,
line 48, in test_north_mini_moe_compute_only_shape
```

Process exit code: 136 (`SIGFPE`). Pytest could not finalize the requested
JUnit file because the process terminated in the extension call. A post-failure
`tt-smi -s` health query completed successfully on all four p300c boards with
DRAM healthy, live heartbeats, and zero uncorrectable GDDR errors.

Verdict: the one-token candidate is refuted. This does not by itself refute a
32-token internally padded control, which is the canonical operation granularity
and is tested next.

## 32-token control

Exact command:

```bash
pytest -q \
  models/autoports/coherelabs_north_mini_code_1_0/tests/moe_compute_probe.py \
  -s \
  --junitxml=models/autoports/coherelabs_north_mini_code_1_0/doc/fused_decoder/moe_compute_probe.junit.xml
```

Result:

```text
tokens_per_device: 32, total_tokens: 32
Matmul output tensor shape: Shape([110, 2, 32, 2048])
Layer 0, Expert 126 (buffer 0): PCC=0.992406 (Passed)
Layer 0, Expert 127 (buffer 1): PCC=0.990744 (Passed)
Per Expert Total Tokens: PASSED
Expert Activation: PASSED
E-T Tensor: PASSED
Matmul Output Tensor: PASSED
1 passed in 14.14s
```

The operation call ran from 22:12:57.036 through 22:12:58.624 in the cold
probe. The complete pytest result is preserved in
`moe_compute_probe.junit.xml`.

Verdict: the arithmetic shape and packed-weight capacity are verified at the
operation's 32-token granularity. The candidate is nevertheless refuted for the
Stage-02 decoder runtime: direct serving batch 1 crashes, padding it to 32 would
compute 32 token slots, and adopting the only passing path would also change
both the weight precision and device-opening contract before adding the omitted
score combine. It is not a local replacement for the current traced batch-1
path.

## Stage-02 integration constraints

Even a passing 32-token control would not establish acceptance:

- the op requires BFLOAT4_B packed weights, while the fused decoder currently
  preserves BF16 weights;
- the decoder accepts an already-open device, while this op requires a
  separately opened COL-dispatch fixture;
- `compute_only=True` omits the score-weighted combine;
- decode batch 1 would require internal padding and still must beat the current
  traced warmed decoder after local combine and slicing.
