# LLaVA on Tenstorrent Hardware

This directory contains the Tenstorrent implementation of the LLaVA vision
pipeline and the MoE-LLaVA-Qwen-1.8B-4e NPU path.

## Current Status

| Area | Status |
| --- | --- |
| CLIP ViT-L/14 vision encoder | Validated on n150 and n300, PCC 0.984 |
| SigLIP SO400M encoder | Validated on n150 and n300, PCC 0.934 |
| Projector and token interleaver | Validated, PCC 0.9999 / deterministic |
| Qwen1-MoE checkpoint conversion and integration | Complete |
| Qwen1-MoE device PCC gates | Passing on n150 and n300 |
| Full MoE-LLaVA image-to-text demo on n150 | Passing and logged |
| Full MoE-LLaVA image-to-text demo on n300 | Pending a usable checkpoint in this environment |
| LLaMA 3.1 TTNN tensor parallel path | Deprecated; CPU path remains available |

The primary supported NPU path is:

```text
Image 336x336
  -> CLIP ViT-L/14 (TTNN): [B, 576, 1024]
  -> MLP projector: [B, 576, 2048]
  -> token interleaving at <image>
  -> Qwen1-MoE 1.8B (TTNN)
```

The n150 path uses one chip. The n300 path uses attention tensor parallelism
and expert parallelism across two chips. Vision tensors are replicated across
the mesh. Multi-chip runs must enable `FABRIC_1D` before opening the mesh;
the demo and MoE tests do this automatically.

## Quick Start

### Prerequisites

1. Verify the hardware:

   ```bash
   tt-smi
   ```

2. Build tt-metal or use the pre-installed TTNN wheel. The wrapper scripts
   automatically detect source builds versus wheel mode.

3. Install the demo dependencies:

   ```bash
   pip install -r models/demos/llava/requirements.txt
   ```

4. Authenticate with Hugging Face if the model is not already cached:

   ```bash
   huggingface-cli login
   ```

### NPU Demo

Use an absolute image path when running the demo directly:

```bash
# n150, single chip
MESH_DEVICE=N150 python models/demos/llava/demo/demo_llava_full.py \
  --llm-backbone qwen-moe --use-ttnn-llm \
  --image /path/to/image.jpg \
  --question "Describe this image in one sentence." \
  --max-tokens 24

# n300, two chips
MESH_DEVICE=N300 python models/demos/llava/demo/demo_llava_full.py \
  --llm-backbone qwen-moe --use-ttnn-llm \
  --image /path/to/image.jpg \
  --question "Describe this image in one sentence." \
  --max-tokens 24
```

The wrapper can configure the environment automatically:

```bash
bash models/demos/llava/run_demo.sh \
  --image /path/to/image.jpg \
  --question "What is in this image?"
```

The first run downloads model files and compiles kernels. The MoE checkpoint
is approximately 10 GB. Do not commit downloaded model files or tensor caches.

### CPU LLM Fallback

To run the TTNN vision path with a CPU Qwen reference LLM, omit
`--use-ttnn-llm`:

```bash
MESH_DEVICE=N300 python models/demos/llava/demo/demo_llava_full.py \
  --llm-backbone qwen-moe \
  --image /path/to/image.jpg
```

The legacy LLaVA-MORE LLaMA 3.1 path also remains CPU-only:

```bash
MESH_DEVICE=N300 python models/demos/llava/demo/demo_llava_full.py \
  --image /path/to/image.jpg
```

### Vision-Only Demo

```bash
MESH_DEVICE=N300 python models/demos/llava/demo/demo_vision_pipeline.py
```

This runs CLIP, the LLaVA-MORE-sized projector, and token interleaving without
loading an LLM.

## Architecture

### Components

- `tt/vision_encoders/clip_encoder.py`: CLIP ViT-L/14 at 336 pixels.
- `tt/vision_encoders/siglip_encoder.py`: SigLIP SO400M at 384 pixels.
- `tt/projectors/mlp_projector.py`: two-layer projector.
- `tt/token_handlers/standard_interleaver.py`: host-side image-token weaving.
- `tt/qwen1_moe_checkpoint.py`: HF-to-Meta Qwen1-MoE checkpoint conversion.
- `tt/llm_wrapper.py`: TTNN transformer wrapper and Qwen1-MoE loader.
- `tt/llava_model.py`: device lifecycle and multimodal orchestration.
- `demo/demo_llava_full.py`: full CPU/NPU demo entry point.

The MoE projector is `1024 -> 2048 -> 2048`. The older LLaVA-MORE vision-only
pipeline uses a `1024 -> 4096 -> 4096` projector. SigLIP outputs 1152 features
and is therefore not wired into the MoE demo without a matching trained
projector.

### Multi-Chip Device Lifecycle

The vision and LLM models do not fit together in device DRAM. The NPU demo
therefore:

1. Opens the mesh with `FABRIC_1D` enabled.
2. Loads CLIP and the projector, then extracts vision features.
3. Closes the device to release vision memory.
4. Reopens the mesh with fabric enabled and loads the Qwen1-MoE LLM.
5. Runs chunked prefill and decode.

## Validation

Run the regression sweep from the repository root:

```bash
bash models/demos/llava/run_regression_sweep.sh
```

The sweep covers CLIP, SigLIP, projector, token interleaver, vision pipeline,
checkpoint conversion, CPU integration, and MoE device PCC tests.

The component runner is also available:

```bash
bash models/demos/llava/RUN_ALL_TESTS.sh
SKIP_E2E=1 bash models/demos/llava/RUN_ALL_TESTS.sh
bash models/demos/llava/RUN_VISION_TESTS.sh
```

For CPU-only MoE integration checks that avoid device PCC tests:

```bash
pytest -q models/demos/llava/tests/test_moe_llava.py \
  -k 'not tokenizer and not pcc'
```

### PCC Gates

| Test | Threshold | Result |
| --- | ---: | ---: |
| CLIP encoder | 0.88 | 0.984 |
| SigLIP encoder | 0.92 | 0.934 |
| MLP projector | 0.99 | 0.9999 |
| Qwen1-MoE two-layer block | 0.98 | 0.9995 |
| Qwen1-MoE 24-layer prefill | 0.91 | 0.9979 |
| Qwen1-MoE decode | 0.99 | 0.9984 |

The LLaMA 3.1 TTNN decode PCC test is intentionally skipped because that
backbone is deprecated. The Qwen1-MoE tests are the authoritative NPU LLM
coverage.

## Troubleshooting

### Device Open or Ethernet Errors

On n300, device open may fail with an Ethernet heartbeat or remote-I/O
timeout. Reset the board and retry:

```bash
tt-smi -r 0
sleep 3
```

### Wheel and Source-Kernel Mismatch

When using a pre-installed wheel, run `run_demo.sh` or the test runners. They
temporarily hide incompatible source-tree kernel directories. Do not add the
source `ttnn/` package to `PYTHONPATH` unless the source build is complete.

### Hugging Face Checkpoint

The NPU demo requires the complete `LanguageBind/MoE-LLaVA-Qwen-1.8B-4e`
checkpoint and the Qwen tokenizer. Incomplete cache blobs are not sufficient;
remove or repair the cache, then rerun with network access and a valid HF
token.

## Repository Hygiene

The following are local-only and must remain untracked:

- `MoE_models/`: downloaded Hugging Face model cache.
- `models/sample_data/test_img_mas.png`: local sample image.
- `models/tt_transformers/model_params/Qwen1-1.8B-moe/*/tensor_cache_*`: generated tensor caches.

Historical implementation logs and superseded plans are preserved under
`history_docs/`. They are reference material, not current status.

## License

The demo follows the tt-metal project license. Downloaded model weights retain
their upstream licenses.
