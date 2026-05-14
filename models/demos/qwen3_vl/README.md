# Qwen3-VL

## Introduction
This codebase includes the Qwen3 family of models and currently supports the model variants:
- Qwen3-VL-2B: [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- Qwen3-VL-32B: [Qwen/Qwen3-VL-32B](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)

## Model Configuration (Qwen3-VL-2B)

### Text Decoder

| Parameter | Value |
|---|---|
| Decoder layers | 28 |
| Hidden dimension | 2048 |
| Attention heads | 16 |
| KV heads (GQA) | 8 |
| Head dimension | 128 |
| Vocabulary size | 151,936 |
| Max sequence length | 8,192 (default, configurable) |
| Batch size | 1 |

### Vision Encoder

| Parameter | Value |
|---|---|
| Architecture | ViT (Vision Transformer) |
| Depth | 24 layers |
| Hidden dimension | 1,024 |
| Patch size | 14 x 14 |
| Temporal patch size | 2 |
| Spatial merge size | 2 |

### Compute Precision

| Component | Precision |
|---|---|
| Text decoder weights | BFP8 (default) or BF16 |
| Vision encoder | BF16 |
| Activations | BF16 |
| KV cache | BF16 |

BFP8 is the default weight precision for the text decoder. BF16 mode can be selected via `--dtype bf16` in the evaluation runner, though testing shows minimal accuracy difference between the two.

### Paged Attention

The text decoder uses paged attention for KV cache management:

| Parameter | Value |
|---|---|
| Block size | 32 tokens |
| Max blocks | 1,024 |
| Max addressable length | 32,768 tokens |

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- Install additional python dependencies:

```
pip install -r models/demos/qwen3_vl/requirements.txt
```

## How to Run
For a single user example:
```
MESH_DEVICE=<device_name> HF_MODEL=<model_name> pytest models/demos/qwen3_vl/demo/demo.py -k 'batch-1'
```

**Notes:**
- `<model_name>` is the HuggingFace model repo string, e.g. `Qwen/Qwen3-VL-3B-Instruct`
- `<device_name>` is the TT device string, e.g. `N150`, `N300`, `T3K`
- `-k` is the pytest filter; to run a specific test, use `-k <test_name>`; additional test names are listed in `models/demos/qwen3_vl/demo/demo.py`
- different model variants are supported on different devices:

| Model Variant      | `<model_name>` (HF_MODEL)                   | `<device_name>` (MESH_DEVICE) |
|--------------------|---------------------------------------------|-------------------------------|
| Qwen3-VL-2B      | Qwen/Qwen3-VL-2B-Instruct                 | `P150`                        |
| Qwen3-VL-32B     | Qwen/Qwen3-VL-32B-Instruct                | `T3K`                         |

Prompts are defined in JSON files under `models/demos/qwen3_vl/demo/sample_prompts/`. The demo supports both single-image and multi-image inputs.

## Benchmark Evaluation

An evaluation framework is provided under `models/demos/qwen3_vl/evaluation/` supporting 9 VQA benchmarks.

### Available Benchmarks

| Benchmark | Metric | Dataset |
|---|---|---|
| DocVQA | ANLS | HuggingFaceM4/DocumentVQA (validation) |
| InfoVQA | ANLS | LIME-DATA/infovqa (train) |
| ChartQA | Relaxed Accuracy | HuggingFaceM4/ChartQA (test) |
| MMMU | Accuracy | MMMU/MMMU (validation) |
| MMStar | Accuracy | Lin-Chen/MMStar (parquet) |
| MathVista | Accuracy | AI4Math/MathVista (testmini) |
| AI2D | Accuracy | lmms-lab/ai2d (test) |
| RealWorldQA | Accuracy | lmms-lab/RealWorldQA |
| OCRBench | Score (x1000) | echo840/OCRBench |

### Running Benchmarks

```bash
# Run a single benchmark (100 samples)
python models/demos/qwen3_vl/evaluation/run_eval.py \
    --benchmarks docvqa \
    --num-samples 100

# Run all 9 benchmarks
python models/demos/qwen3_vl/evaluation/run_eval.py \
    --num-samples 100 \
    --output-dir eval_results/

# Run with BF16 precision
python models/demos/qwen3_vl/evaluation/run_eval.py \
    --benchmarks docvqa chartqa \
    --num-samples 100 \
    --dtype bf16

# Run with increased sequence length (for high-resolution images)
python models/demos/qwen3_vl/evaluation/run_eval.py \
    --benchmarks infovqa \
    --num-samples 100 \
    --max-seq-len 16384
```

### CLI Options

| Option | Default | Description |
|---|---|---|
| `--benchmarks` | all | Space-separated list of benchmarks to run |
| `--num-samples` | all | Limit samples per benchmark |
| `--output-dir` | `eval_results` | Directory for per-sample JSON results |
| `--max-new-tokens` | 50 | Max tokens to generate per sample |
| `--max-seq-len` | 8192 | Max sequence length (increase for large images) |
| `--dtype` | bfp8 | Weight dtype: `bfp8` or `bf16` |
| `--no-tt-vision` | false | Use CPU reference vision model instead of TT device |

### Benchmark Results (BFP8, 100 samples)

| Benchmark | Metric | TT Score | Reference | Delta |
|---|---|---|---|---|
| DocVQA | ANLS | 78.2 | 93.3 | -15.1 |
| InfoVQA | ANLS | 48.9* | 72.4 | -23.5 |
| ChartQA | RelaxedAcc | 54.0 | 72.8 | -18.8 |
| MMMU | Accuracy | 14.0 | 53.4 | -39.4 |
| MMStar | Accuracy | 50.0 | 58.3 | -8.3 |
| MathVista | Accuracy | 39.0 | 61.3 | -22.3 |
| AI2D | Accuracy | 70.0 | 76.9 | -6.9 |
| RealWorldQA | Accuracy | 66.0 | 63.9 | +2.1 |
| OCRBench | Score (x1000) | 844 | 881 | -37 |

Reference scores are from the [Qwen3-VL-2B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) (full dataset evaluation).

\* InfoVQA score is with `--max-seq-len 16384`. At the default 8192, some high-resolution document images exceed the sequence length limit.

### Known Limitations

- **Batch size 1 only**: The evaluation runner processes one sample at a time.
- **Sequence length**: Images with many visual tokens (high-resolution documents) may exceed the default 8192 limit. Use `--max-seq-len 16384` for InfoVQA and similar document-heavy benchmarks.
- **MMMU multi-image**: Some MMMU samples with multiple images fail with tensor allocation errors. These are skipped gracefully.
- **Accuracy gap**: The main remaining accuracy gap vs. reference scores is attributed to vision encoder spatial precision in document-understanding tasks. The reference scores are computed on full datasets with the HuggingFace PyTorch implementation.

## Details
- On the first execution of each model, TTNN will create weight cache files for that model, to speed up future runs.
These cache files only need to be created once for each model and each weight (i.e. new finetuned weights will need to be cached) and will be stored accordingly to the machine you are running the models.
