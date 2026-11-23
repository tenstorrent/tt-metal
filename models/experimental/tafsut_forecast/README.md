# Tafsut forecaster

[Tafsut-FM/tafsut-univariate-base](https://huggingface.co/Tafsut-FM/tafsut-univariate-base) (MIT, about 105M
parameters, 0.42 GB fp32; upstream package [`tafsut==0.1.0`](https://github.com/Tafsut-FM/tafsut)) in TTNN on one
Blackhole chip.

Tafsut is a univariate quantile forecaster: a bidirectional pre-norm encoder over 32-step patches of the context,
32 future tokens and one REG token. It has 14 blocks of width 768 (12 heads x 64, FFN 3072) with RoPE, and a head that
maps each future token to 9 quantiles x 32 steps, so one forward forecasts up to 1024 steps. Longer horizons append
the median and run a second forward.

What runs where (`tt/model.py`; the fp32 torch spec of the same decomposition is `reference/dataflow.py`):

- **Host** (fp32): NaN-left-padded ragged batches are bucketed to a context of 2048 ... 32768 (pad keys masked with
  a finite fill), then instance norm and asinh, patching, and the per-shape constants (time-encoding terms, future
  and REG rows, RoPE tables, key masks). After the device: sinh, rescale, and the upstream constant-series fallback.
- **Device**: the patch embedding, 14 blocks with a `[B, 1, S_pad, S_pad]` additive key mask, the read-out cut in the
  last block (queries, O and the FFN run on the 32 future rows only), and the head. Two model-local
  `ttnn.generic_op` ops in `ops/`, ported from `chronos_forecast`: `qkv_heads_rope` (head split + RoPE in one pass)
  and `add_rms_norm`.
- `tt/trace.py` captures one trace per input shape and replays it.

Precisions (`tt/common.py`):
- `accuracy` (default): bf16 weights, HiFi2 matmuls with fp32 accumulation, QKV, norms, RoPE and the embedding at
  HiFi4.
- `performance`: bfp8 QKV / FFN-up weights and LoFi encoder matmuls.

## Setup

Run everything from the tt-metal root, with the build and `python_env` active (ttnn comes from the build).

1. Weights, into the Hugging Face cache (the model reads `$HF_HOME/hub`, default `~/.cache/huggingface/hub`, at the
   pinned revision):

   ```bash
   hf download Tafsut-FM/tafsut-univariate-base --revision a17c92cbab2cccecbf033406644af88586f6731c
   ```

2. Goldens: CPU fp32 outputs of the upstream package, with their inputs (about 210 MB, not in git). Generate them in
   a separate reference venv with the pinned versions the test floors were set on:

   ```bash
   uv venv --python 3.12 /path/to/tafsut-ref
   uv pip install --python /path/to/tafsut-ref/bin/python -r models/experimental/tafsut_forecast/tests/golden/requirements.txt
   /path/to/tafsut-ref/bin/python -m models.experimental.tafsut_forecast.tests.golden.make_goldens
   ```

   This writes the test and bench goldens and their `MANIFEST.json` to `models/experimental/tafsut_forecast/goldens/`
   (git-ignored; a symlink to a shared directory works too) in a few minutes. To keep them elsewhere, pass `--out`
   and set `TAFSUT_GOLDENS` to that directory for the tests.

   `tafsut==0.1.0` would also install into `python_env` without changing anything else there (checked with
   `uv pip install --dry-run`). `make_goldens.py` still refuses to run in `python_env`, so the goldens always come
   from the pinned torch.

## Tests

`TT_METAL_TRACE_ALLOC_TRACKING=1` is needed by `tests/pcc/test_trace.py` (ttnn reads it at import), so set it for
the whole suite:

```bash
# PCC suite: full model in both precisions, block-by-block taps, the two ops, masked SDPA, trace replay
TT_METAL_TRACE_ALLOC_TRACKING=1 pytest models/experimental/tafsut_forecast/tests/pcc/

# Bench: traced forecast per <shape>-<precision>, gated on tests/perf/floors.json; MODEL_PERF_JSON=<file> keeps the records
TT_METAL_TRACE_ALLOC_TRACKING=1 pytest -s models/experimental/tafsut_forecast/tests/perf/test_forward_perf.py

# CPU only: the dataflow spec against the reference goldens
pytest models/experimental/tafsut_forecast/tests/golden/test_dataflow.py
```

`tests/perf/test_profile_forward.py` profiles a two-layer forward under Tracy
(`python -m tracy -r -p -m pytest ...`), with signposts around the embedding, a layer, the last layer and the head.

## Results

Blackhole p150a, device time of one traced forecast, accuracy precision unless noted. Round 0 is the first working
port; round 1 is where optimization stopped (see `optimizations.md`).

| shape | input -> horizon | round 0 (ms) | round 1 (ms) | round 1, performance (ms) | CPU reference (s) |
|---|---|---|---|---|---|
| paper | 1024 x 2048 -> 64 | 1699.2 | 399.8 | 358.6 | 66.8 |
| long_context | 32 x 32768 -> 64 | 604.6 | 573.2 | 560.3 | 20.9 |
| long_horizon | 512 x 2048 -> 2048 (2 forwards) | 2351.2 | 697.2 | 629.3 | 67.2 |
| gift_ragged | 32 ragged series (GIFT-Eval batch) -> 48 | 604.4 | 573.2 | 560.3 | 46.7 |
| gift_ragged_odd | 16 ragged series, bucket 8192 -> 48 | 75.0 | 48.5 | 46.3 | 3.5 |

Accuracy at round 1 (paper): `pcc_normalized` 0.999996 and `pcc_final` 0.999998 at accuracy; 0.999940 and 0.999960
at performance. The bench also gates band, row-bias and spread-ratio metrics per case (`tests/perf/floors.json`) and
checks that a collapsed forecast fails them.

## Known limitations

- **Optimization paused after round 1** (paper 1697.5 -> 399.8 ms, 4.25x). Round 2 and a final pass over every shape
  with the watcher were not run. `long_context` and `gift_ragged` gained only about 5 %; both run 1088 padded tokens
  per row, where round 2's first candidate (the SDPA chunk choice at that length, R1-6) applies. They were not
  profiled separately.
- **ttnn SDPA bug at an odd q-chunk count.** `ttnn.transformer.scaled_dot_product_attention` returns wrong values
  (masked or not) when `s_pad / q_chunk_size` is odd and greater than 1 at batch 16 or 32 (fine at 2 and 64): for
  example 160/32, 384/128, 640/128, 1088/64 and 1152/128 give worst-row PCC 0.90-0.998 against torch, while every even count is exact. The
  model avoids it with a per-bucket `(s_pad, chunk)` table (`reference/dataflow.py:PAD_TABLE`, asserted by
  `chunk_count_ok`). `tests/pcc/test_sdpa_mask.py::test_sdpa_odd_chunk_count_bug` keeps the repro and asserts the bug
  is still there; once ttnn fixes it, that test fails and the table can be revisited.
- Context is capped at 32768 steps (as upstream); horizons above 1024 run one forward per 1024 steps.
