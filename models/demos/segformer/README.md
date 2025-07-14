# Segformer

### Platforms:

Wormhole N150, N300

**Note:** On N300, make sure to use `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml` with the pytest.

Or, make sure to set the following environment variable in the terminal:
```
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

To obtain the perf reports through profiler, please build with following command:
```
./build_metal.sh -p
```

### Introduction

SegFormer's architecture is adept for both classification and segmentation tasks, utilizing a hierarchical design that extracts rich, multi-scale visual features. Its robust Transformer encoder generates powerful representations, suitable for discerning object categories, while a lightweight MLP decode head precisely maps these features for accurate pixel-level segmentation.

Image classification: [source](https://huggingface.co/nvidia/mit-b0)
Semantic segmentation: [source](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)

### Details

- Entry point for the model is models/demos/segformer/tt/ttnn_segformer_model.py
- Batch Size: 1
- Support Input Resolution: 512x512 (Height, Width)

### How to run

- Use the following command to run the Segformer Encoder model (Classification):
  ```python
  pytest tests/ttnn/integration_tests/segformer/test_segformer_encoder.py
  ```


- Use the following command to run the Segformer Decoder model:
  ```python
  pytest tests/ttnn/integration_tests/segformer/test_segformer_decode_head.py
  ```


- Use the following command to run the Segformer full model (Segmentation):
  ```python
  pytest tests/ttnn/integration_tests/segformer/test_segformer_for_semantic_segmentation.py
  ```

### Segformer Semantic Segmentation Performant Demo

- Use the following command to run the demo script(Segmentation) which returns **mIoU** score for both reference, and ttnn models:
  ```python
  pytest --disable-warnings models/demos/segformer/demo/demo_for_semantic_segmentation.py
  ```
- This script downloads 30 validation images and their annotations of [ADE20K](https://www.kaggle.com/datasets/awsaf49/ade20k-dataset) Dataset.
  ```python
  models/demos/segformer/demo/data_download.sh
  ```
- For Running Demo with Custom Images and their Annotations(Masks), add them to this path:
  ```python
  models/demos/segformer/demo/validation_data_ade20k/images/image.jpg
  ```
  ```python
  models/demos/segformer/demo/validation_data_ade20k/annotations/annotation.png
  ```

### Segformer Image Classification Demo

- Use the following command to run the demo script(Classification) which returns **Accuracy** score for both reference, and ttnn models, validated with Imagenet Dataset samples:
  ```python
  pytest --disable-warnings models/demos/segformer/demo/demo_for_image_classification.py
  ```

### Performant Model with Trace+2CQ
- end-2-end perf is 84 FPS <br>

Use the following command to run Model performant running with Trace+2CQ

```
pytest models/demos/segformer/tests/test_e2e_performant.py
```
