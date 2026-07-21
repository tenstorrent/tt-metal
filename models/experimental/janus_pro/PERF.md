# Model performance and accuracy

Performance is collected from [demo/vision_demo.py](demo/vision_demo.py) and
[demo/text_demo.py](demo/text_demo.py); text accuracy is collected from
`demo/text_demo.py` against the reference in
[models/tt_transformers/tests/reference_outputs/Janus-Pro-7B.refpt](../../tt_transformers/tests/reference_outputs/Janus-Pro-7B.refpt).

Note: accuracy and perf are gathered in separate runs. Accuracy uses tracing
**off** (`-k notrace`); perf uses tracing **on** (`-k trace`).

Note: unlike Gemma-3, Janus-Pro runs a **single fixed precision** (bfloat8_b
decoder, bfloat16 vision tower) — there is no separate performance/accuracy
precision mode, so accuracy and perf share one configuration.

Note: provisional bring-up numbers only. No performance optimization has been
done for this model.

## Build type: Debug is fine for accuracy, Release is required for perf

Accuracy (Top-1 / Top-5) is **build-independent**: it measures *which* tokens the
model predicts, and the ops, dtypes, and numerics are identical between a `Debug`
and a `Release` build — Debug only runs slower, it does not change the result. So
the accuracy numbers below, collected on a `Debug` build, are valid as-is.

Perf (Speed / TTFT) is **not** valid on a `Debug` build: unoptimized host code plus
extra assertions and watcher overhead make it orders of magnitude slower than
`Release`, so Debug timings are meaningless as a performance figure. Perf must be
measured on a `Release` build — the `[janus_pro][demo] * perf benchmark` launch
configs use the `build release` task for exactly this reason.

Perf cells tagged `(debug)` below are placeholders and will be refilled from a
`Release` run.

## Text accuracy (LLaMA decode path)

Top-1/Top-5 are from the **text accuracy run** (`-k notrace`), scoring the device's
greedy predictions against the fp32 HF reference (256 tokens, teacher-forced) — a
`Debug` build is fine here (build-independent). Speed/TTFT are notrace/`Debug`
values and are **not** valid perf figures.

| Model        | Device | Build | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|--------------|--------|-------|-----------|-----------|---------------|-----------|
| Janus-Pro-7B | N150   | Debug | 97.66     | 100.00    | 1.13          | 866       |

## Vision perf (vision tower + LLaMA decode path)

This is the **perf test** — the vision perf benchmark (`-k "trace and single"`,
tracing **on**), single image. No reference-token accuracy for the multimodal
path (image-conditioned generation); functional output is verified by inspection.

| Model        | Device | Build   | Scenario | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|--------------|--------|---------|----------|-----------|-----------|---------------|-----------|
| Janus-Pro-7B | N150   | Debug   | OCR      | N/A       | N/A       | 17.73         | 1439.6    |
| Janus-Pro-7B | N150   | Release | haiku    | N/A       | N/A       | 15.12         | 554.2     |
| Janus-Pro-7B | N150   | Release | OCR      | N/A       | N/A       | 17.93         | 763.5     |

Only the **Release** rows are valid performance figures; the **Debug** row is kept
for reference (only the OCR scenario was captured for the Debug run). All rows use
tracing **on**; scenarios are the two default single-image prompts (haiku on
`dog.jpg`, OCR on `ocr_image.jpeg`).

Comparing the same scenario (OCR) across builds: **decode speed is nearly
identical** (Debug 17.73 vs Release 17.93) because the traced decode loop is
device-bound — trace replay runs on-device, so the slow Debug host path barely
matters. **TTFT roughly halves** on Release (1439.6 → 763.5) because prefill runs
**notrace** (host-dispatched) and a Release build cuts that host overhead. TTFT
includes the vision tower + prefill over the ~596-token image+prompt sequence.

Not directly comparable to Gemma-3's published vision perf, which is measured
**multi-image** (`batch1-multi-image-trace`, 8 images per prompt) in a tunable
`performance` precision mode. Janus here is **single-image** at its one fixed
precision (`-k "trace and single"`); a multi-image run is not yet validated.
