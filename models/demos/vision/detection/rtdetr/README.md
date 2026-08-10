# RT-DETR

TTNN inference implementation of RT-DETR and RT-DETRv2 with a ResNet-50-vd backbone, hybrid encoder, and deformable-attention decoder. The implementation currently targets batch size 1 and supports the `PekingU/rtdetr_r50vd` and `PekingU/rtdetr_v2_r50vd` checkpoints.

## Demos

Run inference on the default COCO image:

```bash
python demo.py
```

Run an image or direct image URL:

```bash
python demo.py --image <path-or-url> --model_version 1
```

Run up to 20 seconds of a video or direct video URL:

```bash
python demo.py --video <path-or-url> --model_version 2
```

Use `--threshold` to set the detection threshold and `--device-id` to select a device. Annotated results are written to a unique directory under `./outputs`.

## Tests

Run the complete test suite:

```bash
pytest tests -s
```

Run tests by category:

```bash
pytest tests/pcc.py -s
pytest tests/detection.py -s
pytest tests/latency.py -s
```
