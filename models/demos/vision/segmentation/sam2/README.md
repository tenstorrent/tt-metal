# SAM 2 Hiera Tiny on N300

TTNN implementation of [`facebook/sam2-hiera-tiny`](https://huggingface.co/facebook/sam2-hiera-tiny) for Wormhole N300.

- Batch size 1
- Image segmentation with point, box, or mask prompts
- Single object video tracking
- Encoder on chip 0, video memory and tracking on chip 1

## Prerequisites

- A cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal)
- [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md) installed
- The tt-metal Python environment activated:

```bash
source "${PYTHON_ENV_DIR:-python_env}/bin/activate"
```

## Browser demo

```bash
python -m models.demos.vision.segmentation.sam2.demo
```

## Tests

PCC tests compare TTNN modules and complete image/video paths with the Hugging Face reference.

```bash
pytest -q models/demos/vision/segmentation/sam2/tests/pcc
```

The video performance benchmark uses 32 warmup frames and measures the FPS on next 100 frames.

```bash
pytest -q models/demos/vision/segmentation/sam2/tests/perf/test_sam2_perf.py
```
