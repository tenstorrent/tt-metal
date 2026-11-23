# t0-gamma forecaster (TTNN)

TTNN port of [t0-gamma](https://huggingface.co/theforecastingcompany/t0-gamma) (The Forecasting Company, Apache-2.0),
a decoder-only, patch-based time-series foundation model that forecasts 21 quantiles. The network (one
`_single_pass`, horizons up to 1024 steps) runs on one Blackhole p150a chip; everything around it (the causal per-step
scaler, patching, masks, quantile post-processing and the GIFT-Eval adjustments) runs on the host in torch, as in the
reference [`tfc-t0`](https://github.com/theforecastingcompany/tfc-t0) package.

| | |
|---|---|
| checkpoint | `theforecastingcompany/t0-gamma` @ `e08aee08` (1.02 GB fp32) |
| reference | `tfc-t0` v0.6.0 (`4871c643`) |
| architecture | patch 32; 24 layers, d 1024, 8 heads, as [time, time, group] x 8; SwiGLU 2048; QK RMSNorm and xPos RoPE; monotone cumulative-softplus head over 21 quantiles |
| hardware | Blackhole p150a, one chip |

## Layout

    tt/                 TTNN model: model.py (forward), attention.py, mlp.py, residual_block.py, weights.py,
                        program_configs.py, common.py (precisions); host.py is the torch host side
    tests/pcc/          device PCC suite against CPU fp32 goldens: modules and the full model
    tests/perf/         benchmark (test_forward_perf.py) and a Tracy profile of a reduced forward
    tests/golden/       golden generator, an fp32 CPU re-implementation of the device decomposition (dataflow.py),
                        CPU checks against the reference, the reference requirements and the goldens' manifest
    optimizations.md    every optimization tried, kept or rejected

## Setup

From the tt-metal root, with the tt-metal build and `python_env` active (`ttnn` comes from that build). The device
side needs nothing beyond `python_env`.

1. Weights, into the Hugging Face cache:

       hf download theforecastingcompany/t0-gamma --revision e08aee083778f1f32da4974c0a34276776a35c52 \
           --include model.safetensors config.json

2. Goldens. The device tests compare against CPU fp32 outputs of the reference package, generated once into
   `models/experimental/t0_forecast/goldens/` (git-ignored, about 175 MB; set `T0_GOLDEN_DIR` to use another
   directory). The reference needs `huggingface-hub >= 1`, which `python_env`'s `transformers` does not accept, so it
   gets its own CPU environment:

       uv venv /path/to/t0-reference --python 3.12
       uv pip install --python /path/to/t0-reference/bin/python \
           -r models/experimental/t0_forecast/tests/golden/requirements.txt
       /path/to/t0-reference/bin/python -I models/experimental/t0_forecast/tests/golden/make_goldens.py

   All eight goldens take about 13 minutes on 8 CPU threads, most of it the two 1024-row bench shapes.
   `tests/golden/MANIFEST.json` records the run the test floors were set on (versions, seeds, sha256 per file).

## Tests

    # device PCC suite (50 tests, about 3.5 minutes on a p150a)
    pytest models/experimental/t0_forecast/tests/pcc

    # benchmark: paper, long_context and multivariate at both precisions, and predict() at the GIFT-Eval settings;
    # MODEL_PERF_JSON appends one JSON record per case
    MODEL_PERF_JSON=t0_bench.json pytest -s models/experimental/t0_forecast/tests/perf/test_forward_perf.py

    # CPU, in the reference environment: host pipeline and dataflow re-implementation against tfc-t0
    env -u VIRTUAL_ENV PYTHONPATH=$PWD /path/to/t0-reference/bin/python -m pytest -p no:cacheprovider -c /dev/null \
        --noconftest models/experimental/t0_forecast/tests/golden/test_dataflow.py

The PCC suite scores the read-out quantiles in asinh space (`pcc_normalized`, the quantile spread `pcc_spread`,
per-row PCC) and the final data-space quantiles (`pcc_final`, `pcc_final_spread`), with floors per case class in
`tests/helpers.FLOORS`. Every case also scores a collapsed output (each quantile replaced by the quantile mean) and
asserts that the floors reject it.

## Results

One p150a, `accuracy` precision. Device time is the median of 5 forwards, each after a 0.3 s idle; steady time is
the median of 5 back-to-back end-to-end calls (prepare, upload, forward, post-process). PCC is against the fp32 CPU
reference.

| shape | input | first port, device | device | steady | pcc_normalized | pcc_final |
|---|---|---|---|---|---|---|
| paper | 1024 series x 2048 -> 64 | 1056.8 ms | **511.9 ms** (2.06x) | 677.4 ms | 0.999966 | 0.999986 |
| long_context | 32 x 8192 -> 1024 | 96.1 ms | **57.6 ms** (1.67x) | 80.6 ms | 0.999897 | 0.999891 |
| multivariate | 128 x 8 variates x 2048 -> 64 | 2003.9 ms | **712.1 ms** (2.81x) | 873.9 ms | 0.999943 | 0.999982 |

For scale, the fp32 reference on 8 CPU threads takes 308 s (paper), 32 s (long_context) and 309 s (multivariate).
The largest wins were the fused SwiGLU, QKV and residual-epilogue matmuls on `minimal_matmul`, the read-out cut after
the last time attention, permute-free group attention and a fused per-head QK-norm + RoPE; `optimizations.md` has
the full list with the rejected candidates.

## Precision

`accuracy` (the default) runs bf16 weights and activations with HiFi2 matmuls and fp32 accumulation everywhere; the
patch encoder runs at HiFi4, and norm statistics, the decoder and the monotone head run in fp32 (`TtT0Precision` has
the per-role table and the evidence). `experimental_performance` (bfp8 weights, LoFi) is kept for timing only: it is
not faster here (585 against 512 ms device at the paper shape) and misses the accuracy band floors on long contexts.

## Limitations

- Horizons above 1024 steps (the reference's autoregressive rollout) and known-future covariates are not supported
  (`tt/host.py`, `MAX_HORIZON`).
- One chip, no trace capture; host preparation is part of the steady time above.
- SDPA with a user mask is capped at k-chunk 128 (`tt/attention.py`, `_MAX_MASKED_CHUNK`): at chunk 256 a masked
  group attention over 768 keys came out wrong (PCC 0.985).
