# Segformer Demo

## How to run demo

- Use the following command to run the Segformer Encoder model (Classification):
  ```python
  pytest tests/ttnn/integration_tests/segformer/test_segformer_encoder.py
  ```


- Use the following command to run the Segformer Decoder module model:
  ```python
  pytest tests/ttnn/integration_tests/segformer/test_segformer_decode_head.py
  ```


- Use the following command to run the Segformer full model (Segmentation):
  ```python
  pytest tests/ttnn/integration_tests/segformer/test_segformer_for_semantic_segmentation.py
  ```

## Segformer Semantic Segmentation Demo

- Use the following command to run the demo script(Segmentation) which returns **mIoU** score for both Reference,ttnn models:
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

## Segformer Image Classification Demo

- Use the following command to run the demo script(Classification) which returns **Accuracy** score for both Reference,ttnn models and between them when validated with Imagenet Dataset samples:
  ```python
  pytest --disable-warnings models/demos/segformer/demo/demo_for_image_classification.py
  ```
