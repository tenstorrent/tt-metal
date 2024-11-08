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
