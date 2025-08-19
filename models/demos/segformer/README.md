# Segformer

## Platforms:
Wormhole (n150, n300)

## Introduction
SegFormer's architecture is adept for both classification and segmentation tasks, utilizing a hierarchical design that extracts rich, multi-scale visual features. Its robust Transformer encoder generates powerful representations, suitable for discerning object categories, while a lightweight MLP decode head precisely maps these features for accurate pixel-level segmentation.

Image classification: [source](https://huggingface.co/nvidia/mit-b0)

Semantic segmentation: [source](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
   - To obtain the perf reports through profiler, please build with: `./build_metal.sh -p`

## How to Run
- Use the following command to run the Segformer Encoder model (Classification):
  ```python
  pytest models/demos/segformer/tests/pcc/test_segformer_encoder.py
  ```

- Use the following command to run the Segformer Decoder model:
  ```python
  pytest models/demos/segformer/tests/pcc/test_segformer_decode_head.py
  ```

- Use the following command to run the Segformer full model (Segmentation):
  ```python
  pytest models/demos/segformer/tests/pcc/test_segformer_for_semantic_segmentation.py
  ```

### Performant Model with Trace+2CQ
#### Single Device (BS=1):
- end-2-end perf is 88 FPS

Use the following command to run Model performant running with Trace+2CQ
```
pytest models/demos/segformer/tests/perf/test_e2e_performant.py::test_segformer_e2e
```

#### Multi Device (DP=2, n300):
- end-2-end perf is 171 FPS

Use the following command to run Model performant running with Trace+2CQ
```
pytest models/demos/segformer/tests/perf/test_e2e_performant.py::test_segformer_e2e_dp
```

### Segformer Semantic Segmentation Performant Demo
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

#### Single Device (BS=1):
- Use the following command to run the demo script(Segmentation) which returns **mIoU** score for both reference, and ttnn models:
  ```
  pytest models/demos/segformer/demo/demo_for_semantic_segmentation.py::test_demo_semantic_segmentation
  ```

#### Multi Device (DP=2, n300):
- Use the following command to run the demo script(Segmentation) which returns **mIoU** score for both reference, and ttnn models:
  ```
  pytest models/demos/segformer/demo/demo_for_semantic_segmentation.py::test_demo_semantic_segmentation_dp
  ```

### Segformer Image Classification Demo
#### Single Device (BS=1):
- Use the following command to run the demo script(Classification) which returns **Accuracy** score for both reference, and ttnn models, validated with Imagenet Dataset samples:
  ```
  pytest models/demos/segformer/demo/demo_for_image_classification.py::test_segformer_classification_demo
  ```

#### Multi Device (DP=2, n300):
- Use the following command to run the demo script(Classification) which returns **Accuracy** score for both reference, and ttnn models, validated with Imagenet Dataset samples:
  ```
  pytest models/demos/segformer/demo/demo_for_image_classification.py::test_segformer_classification_demo_dp
  ```

## Testing
### Performant Data Evaluation with Trace+2CQ
#### Single Device (BS=1):
- Use the following command to run the performant data evaluation with Trace+2CQs:
  ```
  pytest models/demos/segmentation_evaluation/test_segmentation_eval.py::test_segformer_eval

#### Multi Device (DP=2, n300):
- Use the following command to run the performant data evaluation with Trace+2CQs:
  ```
  pytest models/demos/segmentation_evaluation/test_segmentation_eval.py::test_segformer_eval_dp
  ```

## Details
- Entry point for the model is models/demos/segformer/tt/ttnn_segformer_model.py
- Batch Size: 1
- Support Input Resolution: 512x512 (Height, Width)
- Dataset: [Semantic Segmentation](https://www.kaggle.com/datasets/awsaf49/ade20k-dataset), [Image classification](https://huggingface.co/datasets/ILSVRC/imagenet-1k)
