# Code predictor precision levels

The code predictor (codebooks 1 to 15 of every frame) has three precision levels. The
default is the original numerics and is faster than before. `accurate` fixes over half of
the default's top-1 errors at about the original speed; `max` fixes a little more and runs
up to a few percent slower.

## Design

- The default is bit-identical to the original predictor, so picking up this change
  alters no output and no accuracy target. Its time to first frame is no higher than before.
- Changes that keep the numerics exactly as they were are always on; there is no switch
  for them.
- Higher precision is opt-in and is one value: the level name. Moving between levels later
  needs no code change.
- No level alters the weights: every level loads the checkpoint's values unchanged.

## Switching

```python
pipeline = Qwen3TTSPipeline(device, predictor_precision="accurate")
```

or, for anything that builds the pipeline itself (the demo, `tests/e2e/test_e2e.py`,
`tests/pcc/test_pipeline.py`):

```bash
export QWEN3_TTS_PREDICTOR_PRECISION=accurate
```

Unset, or `main`, is the original behaviour.

## Levels

| Level | Accuracy | Speed | Weights |
|---|---|---|---|
| `main` (default) | identical to the original predictor | 5–8% faster | checkpoint values, as before |
| `accurate` | removes 53–66% of main's top-1 errors | 1–3% faster | same as `main` |
| `max` | removes 60–72% of main's top-1 errors | 2–4% slower | checkpoint values, MLP at the checkpoint's own precision |

## Measured

Teacher-forced over the e2e reference frame in `tests/e2e/test_e2e.py`. Time to first frame
and frames/s are against main over 3 rounds alternating with it on each card (`test_e2e.py`
for `main`, the same warm utterance repeated for the other levels). tt-metal 416ff2dfd8.

| card, model | level | top-1 % | top-5 % | first frame against main | frames/s against main |
|---|---|---|---|---|---|
| n150, 0.6B | `main` | 84.46 | 99.58 | 0% | +6% |
| n150, 0.6B | `accurate` | 92.74 | 99.92 | +5% | +1% |
| n150, 0.6B | `max` | 94.43 | 100.00 | +4% | −4% |
| n150, 1.7B | `main` | 85.85 | 99.93 | −2% | +8% |
| n150, 1.7B | `accurate` | 94.16 | 100.00 | +2% | +3% |
| n150, 1.7B | `max` | 95.54 | 100.00 | +5% | −2% |
| p100a, 0.6B | `main` | 83.61 | 99.75 | −3% | +6% |
| p100a, 0.6B | `accurate` | 93.24 | 100.00 | +1% | +2% |
| p100a, 0.6B | `max` | 93.41 | 100.00 | +3% | −3% |
| p100a, 1.7B | `main` | 84.48 | 99.73 | 0% | +5% |
| p100a, 1.7B | `accurate` | 94.71 | 100.00 | +2% | +2% |
| p100a, 1.7B | `max` | 95.60 | 100.00 | +3% | −2% |

`main` matches the previous predictor's top-1 / top-5 exactly, since its output is
bit-identical.

## model_targets.yaml

The default level reproduces the current targets, so nothing changes unless a level is
switched. When running a model at another level, its `wh_n150` accuracy targets become:

| Model | Level | top1 | top5 |
|---|---|---|---|
| qwen3-tts-0.6b | `accurate` | 92.74 | 99.92 |
| qwen3-tts-0.6b | `max` | 94.43 | 100.00 |
| qwen3-tts-1.7b | `accurate` | 94.16 | 100.00 |
| qwen3-tts-1.7b | `max` | 95.54 | 100.00 |

The `bh_p150` entries were not measured at the other levels.

## Tests

`tests/pcc/test_predictor_precision.py`:

- `accurate` and `max` have under three quarters of `main`'s logits error against the
  reference model;
- every level gives the same output when replayed as when run directly;
- every level loads the checkpoint's MLP weights unchanged.

The existing pipeline, KV cache guard, generation-stop and e2e tests run at any level
through `QWEN3_TTS_PREDICTOR_PRECISION`.

## Limits

- `max` keeps the MLP weights at the checkpoint's own precision: the predictor's weights go
  from 113 MB to 157 MB, and its frame rate is 2–4% below main.
- Sampled output at `accurate` and `max` differs from `main` frame by frame, so utterance
  length for a given seed differs too; whether generation stops is covered by
  `tests/pcc/test_generation_stops.py` at each level.
- Blackhole P150 was not measured.
