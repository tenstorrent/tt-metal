# Kairos-50M forecaster (TTNN)

[Kairos](https://github.com/foundation-model-research/Kairos) ([paper](https://arxiv.org/abs/2509.25826)) is a
50M-parameter time-series foundation model for zero-shot quantile forecasting: a T5-style encoder-decoder with a
mixture-of-size dynamic patching tokenizer and instance-wise rotary position embeddings. This directory runs
[`mldi-lab/Kairos_50m`](https://huggingface.co/mldi-lab/Kairos_50m) (Apache-2.0, revision `b788434d`) on one
Blackhole chip, checked against the reference implementation at
[`16eb4b98`](https://github.com/foundation-model-research/Kairos/tree/16eb4b984eb547023187813056a5f289734b9cf6).

## What runs where

- **Host** (`tt/host_frontend.py`, torch fp32): the reference's `encode` up to the embedding input, op for op, so the
  routing matches exactly: mask and truncate to 2048 steps, instance norm, rfft features, the MoE router that picks a
  patch size per 128-step block (16 to 64 tokens plus a `[REG]` token), the dynamic patch division and `x_final`
  gather, and the per-series RoPE frequencies (one table set per attention module). The patching index logic in
  `tt/kairos_patch_utils.py` is vendored unchanged from the reference.
- **Router state** (`tt/router_state.py`): the gate's 5-element bias, which the reference updates on every forward.
  It lives on the host and follows the reference's sequence exactly (x half, then -x half, then the next rollout
  step).
- **Device** (`tt/model.py`): packed mixture-of-size patch embedding, 6 encoder and 6 decoder T5 blocks (decoder:
  causal self-attention and cross-attention) with instance-wise RoPE on q and k (the rotate-half matrix is folded
  into extra projection columns at load), RMSNorm, and the residual head: 2 decoder tokens x 9 quantiles x 64 steps.
  `tt/trace.py` replays one trace per batch size.
- **Generation** (`tt/generator.py`): GIFT-Eval settings run x and -x and average them. A horizon up to 128 steps
  is one forward, with both halves routed on the host and run as one device batch. Longer horizons roll out
  autoregressively, 128 steps per forward, with a host round trip per step: the device forecast becomes the next
  step's context for the host front-end.

`reference/dataflow.py` is a pure-torch fp32 copy of the device decomposition, checked against the reference to
1e-5 relative (`tests/pcc/test_dataflow.py`).

## Setup

All commands run from the tt-metal root with the tt-metal build and `python_env` active. The device tests need
nothing beyond `python_env` (torch and safetensors).

**Weights**, into the Hugging Face cache (`HF_HOME` / `HF_HUB_CACHE` are honoured):

```bash
hf download mldi-lab/Kairos_50m --revision b788434dd95f9893be002e6858bfb2afb0e3f360 \
    --include model.safetensors config.json
```

**Goldens**: CPU fp32 outputs of the reference that every test compares against (about 240 MB: `test/` with
per-block intermediates, `bench/` with the bench shapes). They are not in git. Make them in a separate reference
venv: Kairos pins `numpy==2.2.5` and `transformers~=4.56`, which would replace `python_env`'s numpy 1.26 and
transformers 4.53.

```bash
git clone https://github.com/foundation-model-research/Kairos ~/Kairos
git -C ~/Kairos checkout 16eb4b984eb547023187813056a5f289734b9cf6
uv venv --python 3.12 ~/kairos-ref
uv pip install --python ~/kairos-ref/bin/python -r models/experimental/kairos_forecast/tests/golden/requirements.txt ~/Kairos
export KAIROS_REFERENCE_SRC=~/Kairos   # the golden scripts check its HEAD and hash its sources
for s in make_goldens make_bench_goldens write_manifest; do
    ~/kairos-ref/bin/python -I models/experimental/kairos_forecast/tests/golden/$s.py
done
```

They land in `models/experimental/kairos_forecast/goldens/` (git-ignored). Set `KAIROS_GOLDEN_DIR` to keep them
elsewhere; the tests read the same variable.

## Tests

```bash
# CPU dataflow and host front-end checks, device modules, full model, the bench-case suite in both precisions,
# trace replay on fresh inputs
pytest models/experimental/kairos_forecast/tests/pcc

# Bench: traced generate() on every bench shape and precision; KAIROS_PERF_JSON keeps the records
KAIROS_PERF_JSON=bench.json pytest models/experimental/kairos_forecast/tests/perf/test_forward_perf.py

# Tracy profile of a reduced-depth forward (PROFILE_LAYERS=2+2 by default)
python -m tracy -r -p -m pytest models/experimental/kairos_forecast/tests/perf/test_profile_forward.py
```

Every suite and bench case checks `pcc_normalized` (head output in normalized space) and `pcc_final` (final
quantiles in data space) against `tests/helpers.py`'s floors, and requires the routing to match the reference
exactly (`route_mismatch_rows == 0`, router bias within 1e-6). A test that allocates device memory while a trace is
alive fails (`tests/conftest.py`).

## Results

One Blackhole p150a, device time of one `generate()` (median of 5 traced runs, each after a 0.3 s idle), accuracy
precision (bf16 weights and activations, HiFi2+ matmuls with fp32 accumulation).

| case | input (x and -x halves, GIFT settings) | round 0 (ms) | this branch (ms) | pcc_normalized | pcc_final |
|---|---|---|---|---|---|
| `paper` | 256 series, 2048 -> 64 | 691.09 | **134.77** (5.1x) | 0.999993 | 0.999999 |
| `long_horizon` | 64 series, 2048 -> 720 (6-step rollout) | 1029.58 | **240.84** (4.3x) | 0.999992 | 0.999996 |
| `ragged` | 128 series, lengths 64..2048 -> 64 | 339.77 | **69.20** (4.9x) | 0.999987 | 0.999999 |
| `single` | 1 series, 2048 -> 64 | 5.14 | **3.12** | 1.000000 | 0.999992 |
| `single_long_horizon` | 1 series, 2048 -> 720 | 43.58 | **32.34** | 0.999987 | 0.999816 |

Round 0 is the bring-up baseline; "this branch" is the bench at this code. The PCCs are the device output against the
fp32 reference: `pcc_normalized` on the head output in normalized space, `pcc_final` on the final quantiles. Routing
matched the reference exactly on every case. The paper shape's steady wall time is about 1.2 s, almost all of it host
front-end (see Limitations).

The `performance` precision (bfp8 weights, LoFi) passes its looser floors but is only about 2.5 % faster on `paper`
(131.38 ms), so accuracy is the default. Every change and its measurements are in [optimizations.md](optimizations.md).

## Limitations

- **Optimization stopped after round 2.** Round 2 gained less than 5 %, so the loop stopped; the candidates it left
  (R2-4, R2-5, R2-9, R1-6) were never run, and no final audit pass (watcher run, profiler sweep over every shape) was
  done.
- **The host dominates end-to-end time.** The front-end (instance norm, rfft, routing, patch division) runs in torch
  on the host; on the paper shape the steady wall time is several times the device time.
- **The router bias is mutable host state.** Like the reference, every forward changes it, so results depend on
  call history; the tests restore each golden's starting bias.
- Long horizons roll out with a host round trip per 128 steps.
