# Janus-Pro demos

Manual-run benchmarks, modeled on `models/demos/multimodal/gemma3/demo/`:

- `text_demo.py::test_demo_text` — text path only (LLaMA decoder, no vision tower).
  `notrace` scores greedy predictions against a HuggingFace fp32 reference (top-1 /
  top-5); `trace` is the perf run (TTFT + decode tok/s/user).
- `vision_demo.py::test_multimodal_demo_text` — full multimodal path (vision tower +
  decoder), image + prompt → text, with warmup and `BenchmarkProfiler`.

Both require real weights (`HF_MODEL=deepseek-community/Janus-Pro-7B`); dummy weights
produce garbage. Results are recorded in [../PERF.md](../PERF.md).

## Build type: Debug for accuracy, Release for perf

Accuracy (top-1 / top-5) is **build-independent** — same ops/dtypes/numerics, Debug
just runs slower — so a `Debug` build is fine for the `notrace` accuracy runs.

Perf (Speed / TTFT) must come from a **`Release`** build: a Debug build's unoptimized
host path makes timings meaningless. The `[janus_pro][demo] * perf benchmark` launch
configs build Release for this reason. See PERF.md for the trace/notrace and
device-bound/host-bound details.

## 1. Generate the reference (host, one-time)

The text-accuracy demo needs `models/tt_transformers/tests/reference_outputs/Janus-Pro-7B.refpt`.
It holds the ground-truth top-5 next-token predictions from the HuggingFace reference
in **float32 on CPU**. Janus is loaded as `JanusForConditionalGeneration` (via
`AutoModelForImageTextToText`) and run text-only (no `pixel_values`): that forward
path is the LLaMA decoder + LM head, so the vision tower is never exercised.

This is a **host** job (loads the 7B model on CPU, needs host RAM); it does not touch
the device.

```bash
python3 models/tt_transformers/tests/generate_reference_hf.py \
    --model deepseek-community/Janus-Pro-7B \
    --output_file models/tt_transformers/tests/reference_outputs/Janus-Pro-7B.refpt \
    --total_length 1024 --trust-remote-code
```

`--trust-remote-code` is not required for Janus (it is a native `transformers` model)
— it is harmless here and kept for models that do need it. The script prints its own
top-1/top-5 summary; sanity-check it before trusting the file.

## 2. Text accuracy (device, `notrace`)

```bash
MESH_DEVICE=N150 HF_MODEL=deepseek-community/Janus-Pro-7B \
    pytest "models/experimental/janus_pro/demo/text_demo.py::test_demo_text" -k notrace -sv
```

Prints top-1/top-5 (256 tokens, teacher-forced) plus TTFT + decode. The device runs
`bfloat8_b` against the fp32 reference, so some gap is expected — observed 97.66 /
100.00 on N150. `notrace` + a `Debug` build is fine for accuracy. The `trace` variant
(`-k trace`) is the perf run; use a Release build for it.

## 3. Vision perf (device, `trace`)

```bash
MESH_DEVICE=N150 HF_MODEL=deepseek-community/Janus-Pro-7B \
    pytest "models/experimental/janus_pro/demo/vision_demo.py::test_multimodal_demo_text" -k "trace and single" -sv
```

Parametrized over `enable_trace` (`notrace`/`trace`) and `multi_image`
(`single`/`multi`), batch1. Prints generated text + TTFT + decode tok/s/user per
scenario. Perf figures come from `trace` on a Release build.

- `single`: two single-image scenarios (mirroring gemma3's default set) — a generative
  prompt (`dog.jpg` → "Write a haiku for this image.") and an OCR prompt
  (`ocr_image.jpeg` → "What is the full text of this image? Do OCR"). **Device-validated:**
  output is correctly image-conditioned.
- `multi`: one prompt over two images (`dog.jpg` + `ocr_image.jpeg`). The Janus fusion
  path coalesces a per-image list of vision features, so images are fed as a list of
  single-image tensors. **Verified only at source level — run it and confirm the output
  before trusting it.**

Select a subset with `-k`, e.g. `-k "notrace and single"` for a quick functional check.
Set `JANUS_DEMO_IMAGE` and/or `JANUS_DEMO_PROMPT` to force one custom single-image
scenario (overrides the parametrized set).

Note: the input is built manually (the Janus HF processor does not expand its single
`<image_placeholder>` into the per-image token block the decoder needs), inserting
`mm_tokens_per_image` image-token placeholders per image — the same construction
`tests/test_e2e.py` uses.

## CI status

Manual-run only. These demos are **not** wired into any CI yaml. For contrast, the
gemma3 demos are `-k`-selected in `tests/pipeline_reorg/models_e2e_tests.yaml`
(gated by `is_ci_env`); Janus CI currently covers only `test_ci_dispatch.py` and the
PCC `test_e2e.py`.
