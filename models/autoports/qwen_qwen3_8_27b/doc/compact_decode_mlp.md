# Compact decode layouts

The compact residual path still expanded the packed gate/up projection to
`[B,1,2I]` before SwiGLU. The new `compact_decode_mlp` policy retains
`[1,1,B,2I]`, moves it to interleaved L1 for slicing, and feeds a compact
SwiGLU result to the existing down projection. Matmul weights, precision,
activation, and request-state algorithms are unchanged.

Enable with `QWEN_COMPACT_DECODE_MLP=1` when starting the model. It requires
compact residuals (`QWEN_COMPACT_DECODE_RESIDUAL=1`, currently the model
default for B>1). The MLP option defaults off pending wider batch/context
and agentic evaluation coverage; B1 and prefill retain their existing paths.

## Measurements, 2026-09-23

Full 64-layer TP4 model, B8, eight distinct synthetic prompts, greedy sampling,
32 decode steps per trial, two trials per policy and context, using the
`mvasiljevic-ttxla` Docker container. Both policies use compact residuals.
Each steady measurement excludes the first decode step (capture/first use).
Values below are the second trial for each policy.

| Prompt tokens | Existing MLP ms/step | Compact MLP ms/step | Existing aggregate tok/s | Compact aggregate tok/s |
|---|---:|---:|---:|---:|
| 128 | 48.077 | 42.712 | 166.40 | 187.30 |
| 4096 | 49.025 | 43.731 | 163.18 | 182.94 |

Every generated token matched the control in all repeats and both contexts.
These are decode measurements, not end-to-end agentic evaluation speedups or
an accuracy evaluation against HF. Long contexts, other batch sizes, and
dynamic request turnover still require validation.

Raw full-model report:
`/home/mvasiljevic/qwen38-full-rerun/compact_mlp_b8_comparison.json`.

The B8 mixed linear/full-attention component stack also passed the single-chip
reference comparison (decode PCC 0.9999951; 16-step stress output PCC
0.9999984), exact eager/trace and changed-input replay comparisons, cache-write
ownership checks, and 16 queued evolving-state replays. Reports and reference
fixture are under `generated/compact_mlp_validation/`. Four host shape tests
passed, including equality of the compact and expanded MLP arithmetic.

The Python checkout was `qwen38-concurrency` at base commit
`adc2cdba7a0ca908bc1953f95caeb5d563883079` plus the local compact residual/MLP
patches. The installed TTNN/runtime came from
`/home/mvasiljevic/qwen38-full-rerun/tt-metal` (base
`0ba1d3d46eb922cbc3e05117eedbe50a5df0e6c1`, existing local build).
Implementation SHA256 for the initial MLP-only measurement (before the
subsequent RoPE/attention additions):

- `tt/optimized_decoder.py`: `1b6970a86637177716fdb95d7984ff7ebdc1060162da3f38d4630358f4a7abb1`
- `tt/model.py`: `bf6bc43e7fc028cd9a0e0f92b5b4e1b04c96fe77e8f63098b5ee49a741d69db1`

## Reproduce

From the checkout with the configured TTNN Python environment and weights:

```bash
python -m models.autoports.qwen_qwen3_8_27b.tests.profile_concurrency \
  --batch 8 --lengths 128,4096 --steps 32 --distinct-prompts \
  --compare-compact-mlp --output compact_mlp_b8_comparison.json
```

The comparison releases and recaptures traces when switching policy, retains
the same weights and prefill implementation, and fails if any generated token
differs. For component reference/state checks, use `run_multichip_decoder`
with `--stack --batch 8 --length 128 --stress-iterations 16`, first with
`--baseline` and then with
`--policy '{"compact_decode_residual":true,"compact_decode_mlp":true}'`.
Use the same dedicated `--compare-dir` for both; the reference must include
the same stress outputs as the candidate.

## Cumulative model-only layout optimizations

Two additional opt-in policies reuse existing TTNN operations:

- `QWEN_BATCHED_DECODE_ROPE=1`: transpose request rows into the sequence
  dimension and rotate all requests in one call, instead of a Python loop
  over requests. Each row retains its own position's cos/sin values. Restore
  the logical shape after rotation because the operation exposes padded rows.
- `QWEN_COMPACT_DECODE_ATTENTION=1`: keep the packed QKV/gate projection
  and output gating compact, avoiding expansion to one tile per request.
  This policy requires compact residuals, like compact MLP.

All three new policies default off pending broader serving/eval validation.
No Metal operations, kernels, precision settings, or state algorithms changed.

Full 64-layer TP4 B8 comparison in `mvasiljevic-ttxla`, now with 64 decode
steps, eight distinct prompts, and two trials per configuration. Warm second
trials, milliseconds per decode step:

| Context | Compact residual only | + compact MLP | + batched RoPE | + compact attention |
|---|---:|---:|---:|---:|
| 128 | 48.490 | 42.676 | 40.465 | 38.639 |
| 4096 | 49.427 | 43.859 | 42.002 | 39.606 |

Combined aggregate throughput is **207.04 tok/s** at context 128 and
**201.99 tok/s** at context 4096, versus 164.98 and 161.86 for compact
residuals alone: approximately **25% higher throughput**, or **20% lower
step latency**. All tokens matched the control in all 16 trials. This does
not measure five-eval wall time; prefill, tool execution, and scheduling still
contribute to end-to-end time.

Raw report: `/home/mvasiljevic/qwen38-full-rerun/decode_layouts_b8_comparison.json`.
Reproduce using `--compare-decode-layouts --steps 64` instead of
`--compare-compact-mlp --steps 32` in the command above.

Both additions also passed the mixed-stack reference, cache ownership,
changed-input trace replay, and 16-step evolving-state checks. The standalone
`tests/check_batched_decode_rope.py` device check verifies **bitwise equality**
to the per-user path for B1/B2/B5/B8, both local head counts (1 and 6), and
different positions in each slot spanning 0 through 32767. Non-rotary channels
are also checked unchanged on all four devices.

To enable the measured combination at model startup:

```bash
export QWEN_COMPACT_DECODE_MLP=1
export QWEN_BATCHED_DECODE_ROPE=1
export QWEN_COMPACT_DECODE_ATTENTION=1
```

## Follow-up experiments and checks

The packed decode convolution now receives only the live row instead of an
explicitly padded 32-row input. Its existing helper builds the three-history
plus one-current-row windows itself. This avoids redundant model-level work
without changing that helper or any kernel. Mixed-stack state/trace checks
passed; the full 64-layer B8 rerun matched every token in the cumulative
comparison above. Warm results were 38.740 ms at context 128 and 39.591 ms at
context 4096, essentially unchanged within run-to-run variation; no additional
full-model speedup is claimed. Report:
`/home/mvasiljevic/qwen38-full-rerun/decode_layouts_unpadded_conv_b8.json`.

A further compact linear-attention projection/gate experiment also passed
correctness, but its mixed-stack median increased from 1.023 to 1.043 ms
(30 repeats). The extra layout conversions outweighed the saving. That
experiment was removed; its report remains at
`generated/compact_mlp_validation/compact_delta.json`.

Host checks: the four compact-layout tests and 21 serving-prefill trace tests
passed in Docker. Wider `*_host.py` discovery ran 74 tests successfully but
could not import `test_vllm_prefill_host` because the Python environment lacks
the `vllm` package. This is not a complete vLLM integration test. Black and
`git diff --check` passed. A full five-eval parallel rerun has not been performed
for these new policies.

## B16 local validation, 2026-09-23

The same cumulative comparison passed at B16 in `mvasiljevic-ttxla`: all
64 layers, 16 distinct synthetic prompts, 64 decode steps, and two trials
per policy at each context. Every generated token matched compact-residual-only
control in all 16 trials. Warm second-trial measurements:

| Context | Residual-only ms/step | Combined ms/step | Residual-only tok/s | Combined tok/s |
|---|---:|---:|---:|---:|
| 128 | 79.499 | 62.296 | 201.26 | 256.84 |
| 4096 | 81.140 | 63.766 | 197.19 | 250.92 |

This is 27–28% higher aggregate decode throughput, not a prediction of
end-to-end benchmark gain. B16 offers about 24% more aggregate throughput
than the measured B8 combination, not twice as much, and has higher per-user
step latency. Local full-model validation uses serial prefill and contexts up
to 4096; the C8/C16 benchmark sweep exercises the configured grouped prefill
and longer contexts separately.

Report: `/home/mvasiljevic/qwen38-full-rerun/decode_layouts_b16_comparison.json`.
Reproduce the cumulative command with `--batch 16 --lengths 128,4096 --steps 64`.
