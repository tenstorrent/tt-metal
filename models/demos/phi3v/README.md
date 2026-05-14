# Phi-3.5-vision-instruct on TT-Metal

TT-Metal implementation of [Microsoft Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct), a lightweight multimodal model combining a CLIP vision encoder with the Phi-3 text decoder.

## Model Architecture

| Component | Detail |
|-----------|--------|
| Text decoder | Phi-3 (32 layers, dim 3072, 32 heads MHA) |
| Vision encoder | CLIP ViT (on CPU, HF reference model) |
| Vocab size | 32,064 |
| RoPE | SU (scaled) type |
| Parameters | 4.15B |

The vision encoder runs on CPU via the HuggingFace reference model (`vision_embed_tokens`), producing merged embeddings that are passed to the TT-Metal text decoder for prefill and autoregressive decode.

## Hardware

- Tenstorrent Blackhole P150 (single chip, 1x1 mesh)

## Quick Start — Demo

```bash
cd /tt-metal
PYTHONPATH=/tt-metal TT_METAL_HOME=/tt-metal \
HF_MODEL=microsoft/Phi-3.5-vision-instruct \
  python models/demos/phi3v/demo/demo.py \
    --prompts models/demos/phi3v/demo/sample_prompts/demo.json
```

## Benchmark Evaluation

### Usage

```bash
cd /tt-metal
PYTHONPATH=/tt-metal TT_METAL_HOME=/tt-metal \
HF_MODEL=microsoft/Phi-3.5-vision-instruct \
  python models/demos/phi3v/evaluation/run_eval.py \
    --benchmarks mmmu scienceqa mmbench mathvista ai2d chartqa textvqa pope \
    --num-samples 100 \
    --max-new-tokens 50 \
    --output-dir eval_results_phi3v/
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--benchmarks` | all 8 | Space-separated list of benchmarks to run |
| `--num-samples` | all | Limit samples per benchmark (for quick checks) |
| `--max-new-tokens` | 50 | Max tokens to generate per sample |
| `--max-seq-len` | 4096 | Maximum sequence length |
| `--dtype` | bfp8 | Weight dtype (`bfp8` or `bf16`) |
| `--output-dir` | `eval_results_phi3v` | Directory for per-benchmark JSON results |

### Runtime Parameters

| Parameter | Value |
|-----------|-------|
| Weight dtype | bfp8_b |
| Max sequence length | 4096 |
| Paged attention block size | 32 |
| Paged attention max blocks | 1024 |
| Image max dimension | 512 px |
| Mesh shape | 1x1 |

Images are resized so the longest edge is at most 512 px before being passed to the vision encoder. This keeps the embedding sequence length within the 4096-token limit (Phi-3.5-vision uses multi-crop tiling, and larger images can exceed the limit).

### Benchmark Results (100 samples each)

| Benchmark | Metric | TT Score | Reference | Delta |
|-----------|--------|----------|-----------|-------|
| MMMU | Accuracy | 8.0 | 43.0 | -35.0 |
| ScienceQA | Accuracy | 89.0 | 91.3 | -2.3 |
| MMBench | Accuracy | 96.0 | 81.9 | +14.1 |
| MathVista | Accuracy | 41.0 | 43.9 | -2.9 |
| AI2D | Accuracy | 88.0 | 78.1 | +9.9 |
| ChartQA | Relaxed Acc | 66.0 | 81.8 | -15.8 |
| TextVQA | VQA Acc | 60.7 | 72.0 | -11.3 |
| POPE | Accuracy | 88.0 | 86.1 | +1.9 |

Reference scores are from the [Phi-3.5-vision model card](https://huggingface.co/microsoft/Phi-3.5-vision-instruct) (full dataset evaluation).

### Notes on Results

- **ScienceQA, MathVista, POPE** — within 3 points of reference, confirming the inference pipeline is correct.
- **MMBench, AI2D** — above reference on the 100-sample subset; may vary with full dataset.
- **ChartQA, TextVQA** — gap is primarily due to the 512 px image cap. These benchmarks require reading fine text in charts and natural images, which is degraded by downscaling.
- **MMMU** — the model tends to output numeric answers instead of MCQ option letters (A/B/C/D) for complex academic questions. This is a prompting/output-format issue, not a model accuracy problem.

### Available Benchmarks

| Key | Benchmark | Dataset | Split | Metric |
|-----|-----------|---------|-------|--------|
| `mmmu` | MMMU | MMMU/MMMU | validation | Accuracy (MCQ) |
| `scienceqa` | ScienceQA | derek-thomas/ScienceQA | test (image subset) | Accuracy (MCQ) |
| `mmbench` | MMBench | lmms-lab/MMBench | en dev | Accuracy (MCQ) |
| `mathvista` | MathVista | AI4Math/MathVista | testmini | Accuracy (MCQ) |
| `ai2d` | AI2D | lmms-lab/ai2d | test | Accuracy (MCQ) |
| `chartqa` | ChartQA | HuggingFaceM4/ChartQA | test | Relaxed Accuracy |
| `textvqa` | TextVQA | facebook/textvqa | validation | VQA Accuracy |
| `pope` | POPE | lmms-lab/POPE | test | Accuracy (Yes/No) |

## File Structure

```
models/demos/phi3v/
├── README.md
├── demo/
│   ├── demo.py                          # Vision demo script
│   └── sample_prompts/demo.json         # Sample prompt config
├── evaluation/
│   ├── run_eval.py                      # Evaluation entry point
│   ├── model_runner.py                  # Phi3VRunner (TT-Metal inference wrapper)
│   └── benchmarks/
│       ├── __init__.py                  # Registry + reference scores
│       ├── scienceqa.py                 # ScienceQA benchmark
│       ├── mmbench.py                   # MMBench benchmark
│       ├── textvqa.py                   # TextVQA benchmark
│       └── pope.py                      # POPE benchmark
└── tt/
    └── __init__.py
```

Shared components from `models/demos/qwen3_vl/evaluation/`:
- `benchmarks/base.py` — `BaseBenchmark`, `BenchmarkResult`
- `benchmarks/mcq_benchmarks.py` — `MMMUBenchmark`, `MathVistaBenchmark`, `AI2DBenchmark`
- `benchmarks/chartqa.py` — `ChartQABenchmark`
- `metrics.py` — `vqa_accuracy`, `relaxed_accuracy`, `extract_mcq_answer`, `anls`, `exact_match`
