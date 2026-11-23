# TiRex-2 forecaster (TTNN)

[TiRex-2](https://huggingface.co/NX-AI/TiRex-2) is NX-AI's xLSTM time-series forecaster. This tree runs the
`NX-AI/TiRex-2-gifteval-pretrain` checkpoint (revision `54f88ac06ccfba9fe3e727aa8856b39775481f0f`, Apache-2.0) on one
Blackhole chip with TTNN. It covers one forward pass (`tta_sign_flip=False`, `pad_context=True`), the sign-flip
average built on top of it, univariate and grouped multivariate input, and horizons up to 928 steps.

## Model

- **Tokens:** every row is left-padded with NaN to 8192 context steps plus 928 future steps, then cut into
  285 patches of 32 steps. A per-row scaler and the patch tokenizer run on the host.
- **Backbone:** a residual-MLP patch embedding and 12 blocks, d_model 512, 4 heads. Even blocks are **mLSTM** (matrix
  memory, computed in the stabilized chunkwise form, chunk 64). Odd blocks are **sLSTM** (scalar memory, a nonlinear
  recurrence over all 285 tokens with exponential gating). Each block is `x + cell(RMSNorm(x))`, then a SwiGLU FFN.
  On multivariate input each block ends with a variate mixer: attention across the rows of a group.
- **Head:** a T5 norm and a residual-MLP head give 9 quantiles x 32 steps per read-out token. Inverse scaling and
  band calibration run on the host.

`reference/dataflow.py` is the fp32 torch spec of the device decomposition, written from the math (not the upstream
code, which needs `flashrnn` and `mlstm_kernels`). `tests/golden/test_dataflow.py` checks it against the upstream
goldens, and the device modules mirror it op for op.

## Setup

All commands run from the tt-metal root, with the tt-metal build and `python_env` active.

**Weights.** The tests load the pinned snapshot from the Hugging Face cache:

```bash
hf download NX-AI/TiRex-2-gifteval-pretrain --revision 54f88ac06ccfba9fe3e727aa8856b39775481f0f \
    --include model.safetensors config.json
```

**Goldens.** The device tests compare against fp32 CPU goldens from the upstream package (`tirex-2==0.4.0`). They are
not in git (about 510 MB) and are generated once in a **separate** venv. `tirex-2` needs Python >= 3.11 and
torch >= 2.8, so it cannot be installed into `python_env`, which is Python 3.10:

```bash
uv venv --python 3.12 ~/venvs/tirex2-ref
uv pip install --python ~/venvs/tirex2-ref/bin/python -r models/experimental/tirex2_forecast/tests/golden/requirements.txt
env -u PYTHONPATH -u VIRTUAL_ENV TORCHDYNAMO_DISABLE=1 ~/venvs/tirex2-ref/bin/python \
    models/experimental/tirex2_forecast/tests/golden/make_goldens.py
```

This writes `models/experimental/tirex2_forecast/goldens/` (git-ignored) in about 20 minutes on 8 CPU threads; the
bench shapes take most of it. Set `TIREX2_GOLDEN_DIR` to keep them elsewhere. `tests/golden/MANIFEST.json` records
the goldens the test floors were set on: seeds, package versions and sha256 of every file. The device tests skip
when the goldens are missing.

## Tests

```bash
# PCC suite (device) and the CPU dataflow check
pytest models/experimental/tirex2_forecast/tests/pcc models/experimental/tirex2_forecast/tests/golden/test_dataflow.py

# Benchmark: paper / long_context / multivariate x accuracy / performance, plus the sign-flip forecast
MODEL_PERF_JSON=bench.json pytest -s models/experimental/tirex2_forecast/tests/perf/test_forward_perf.py

# Tracy profile of a reduced forward (see the test's docstring)
python -m tracy -r -p -m pytest models/experimental/tirex2_forecast/tests/perf/test_profile_forward.py
```

The suite covers:
- module PCC: embedding, each mLSTM and sLSTM block, sLSTM per-step states, FFN, norms, head;
- the fused sLSTM kernel against the op-level loop;
- full-model PCC: one series, 16 series, 64 rows of the paper bench, long context (H = 928), multivariate
  groups, ragged groups that do not divide 32 rows, and the pad-skip path against the full path;
- the sign-flip forecast in both precisions;
- trace replay against the eager forward.

Every case is scored with the metrics in `tests/pcc/metrics.py` against the floors in `tests/gates.json`. Each also
checks that a collapsed output (every quantile replaced by the quantile mean) fails those floors.

## Results

Device time on one Blackhole p150, `accuracy` precision: the median of 5 traced replays, each after a 0.3 s idle.
Measured on chip 1 of a 4-chip host, which drifts by up to 5 % under load, at main `c83644ccafa`:

| shape | input | round 0 | now | speedup | steady wall | pcc_normalized | pcc_final |
|---|---|---|---|---|---|---|---|
| paper | 1024 series x 2048 -> 64 | 3871.2 ms | 881.2 ms | 4.4x | 1299.0 ms | 0.999975 | 0.999992 |
| long_context | 32 series x 8192 -> 928 | 371 ms | 66.0 ms | 5.6x | 72.2 ms | 0.999981 | 0.999995 |
| multivariate | 128 groups x 8 rows x 2048 -> 64 | 10990 ms | 2300.1 ms | 4.8x | 2629.4 ms | 0.999939 | 0.999992 |

- Round 0 is the first full port: stock ops, the op-level sLSTM loop, default program configs.
- The sign-flip forecast of 16 series (one 32-row batch) takes 81.5 ms on device.
- The upstream CPU reference takes 413 s for the paper shape on 8 threads.
- `performance` precision: 859.4 / 64.0 / 2147.1 ms (timing only).
- `optimizations.md` lists every change behind these numbers.

## Implementation notes

- **Host / device split** (`tt/model.py`): `prepare_inputs` (scaler, tokens, row packing into 32-row blocks) and
  `postprocess` run on the host. `forward_device` runs embed, 12 blocks, the final norm and the head on a tile-aligned
  window of tokens, and is captured as a trace (`tt/trace.py`); the op-level sLSTM loop alone is tens of thousands
  of ops.
- **Constant left-pad skip:** with context <= 6144 steps, every row starts with at least 64 all-NaN pad tokens.
  The states after those tokens are the same for every row and input. The host computes them once per cut with the
  fp32 dataflow and keeps them as device constants, so the device runs only the remaining tokens. At the paper
  shape the cut is 192 of 285 tokens. Grouped (multivariate) batches keep the full path.
- **Fused sLSTM kernel** (`ops/slstm_step`): one `ttnn.generic_op` runs every step of an sLSTM block. Per step it
  computes `Wx + y @ R` (HiFi4, fp32 DST) and the gate math in one SFPI pass. The c / n / m states stay fp32 in L1;
  cores that share a head exchange their bf16 `y` tiles through L1 and semaphores.
  - It dispatches only when the recurrence starts from the zero state and `y` is fed back in bf16: long context,
    multivariate, and other full-path runs.
  - Runs with a pad cut start from a cached nonzero state, so they use the op-level loop in `tt/layers.py`. That
    includes the paper shape.
  - `TIREX2_SLSTM_KERNEL=0` forces the loop everywhere.
- **Precision** (`tt/common.py: Precision`): `accuracy` is the shipping policy, with bf16 weights and activations and
  every dense matmul at HiFi4 with fp32 accumulation. In both policies the recurrent states and exponential gating
  are fp32. `performance` (bfp8 weights, LoFi / HiFi2 per role) is timing only: it is under 3 % faster.

## Known limitations

- **The paper shape does not use the fused sLSTM kernel yet.** That needs an initial-state input on
  `ops/slstm_step`, so the pad-skip path can start the kernel from the cached state. The estimated gain is about 25 %
  at the paper shape. This is the main open follow-up.
- `tests/pcc/test_model.py::test_ragged_groups` misses one floor on this base: `pcc_final_spread` 0.998954 against
  0.999.
  - That case is 48 rows in groups of 5 and 1, packed into 64 device rows with dummy rows.
  - It passes with `TIREX2_SLSTM_KERNEL=0` (0.999104) and passed with the kernel on the earlier main base
    (0.999116), so the fused kernel's numerics on this layout moved with the base update.
  - Every bench shape and the kernel's own test still pass.
- Horizons past 928 steps (upstream rollout) and known-future covariates are not supported.
- A group (target plus past covariates) may have at most 32 rows: variate attention runs on 32 x 32 row blocks.
- Chip-to-chip and duty-cycle drift: device times move by up to 5 % on a loaded chip. The bench idles 0.3 s before
  each timed replay.

## Layout

```
tt/                 TTNN model: common (precision), weights, layers (mLSTM, sLSTM loop, FFN, variate), model, trace
ops/slstm_step/     fused sLSTM generic_op and its reader / compute / writer kernels
reference/          fp32 torch dataflow (the spec the device code mirrors)
tests/golden/       golden generator, its requirements, the manifest, and the CPU dataflow check
tests/pcc/          device PCC suite and the gate metrics
tests/perf/         benchmark and profile tests
tests/gates.json    gate floors per shape and precision
optimizations.md    every optimization candidate, kept or rejected
```
