---

# Functional Bloom Model for Question Answering (QA) Demo

## Introduction

The Bloom model, exemplified by the `bigscience/bloom-560m` version, is a powerful language model renowned for its text generation capabilities. From simple completions to complex linguistic analyses, Bloom excels in tasks like cloze tests, counterfactuals, and reframed text generations. Applied downstream, it aids in information extraction, question answering, and summarization.

This demo showcases the Functional Bloom Model specifically for Question Answering task.

### Details

The entry point to the Functional Bloom model is the `bloom` function located in `ttnn_functional_bloom.py`. The `bigscience/bloom-560m` version from Hugging Face is utilized as the reference model.

## Sequence Size: 384

Sequence size determines the maximum length of input sequences processed by the model, optimizing performance and compatibility. It's recommended to set the `sequence_size` to 384

## Batch size: 8

Batch Size determines the number of input sequences processed simultaneously during training or inference, impacting computational efficiency and memory usage. It's recommended to set the `batch_size` to 8

## How to Run

To run the demo for question answering using the Bloom model, follow these instructions:

- Use the following command to run the demo with the ttnn functional bloom model:
  ```
  pytest --disable-warnings --input-path="models/demos/grayskull/functional_bloom/demo/input_data.json" models/demos/grayskull/functional_bloom/demo/demo_qa.py::test_demo[models.experimental.functional_bloom.tt.ttnn_functional_bloom]
  ```

- Alternatively, to run the demo with the ttnn optimized functional bloom model, use the following command:
  ```
  pytest --disable-warnings --input-path="models/demos/grayskull/functional_bloom/demo/input_data.json" models/demos/grayskull/functional_bloom/demo/demo_qa.py::test_demo[models.experimental.functional_bloom.tt.ttnn_optimized_functional_bloom]
  ```

- To execute the demo with an alternative input file, substitute <address_to_your_json_file.json> with the file path of your JSON file in the provided command.:
  ```
  pytest --disable-warnings --input-path=<address_to_your_json_file.json> models/demos/grayskull/functional_bloom/demo/demo_qa.py::test_demo[path_to_functional_bloom]
  ```

## Inputs

The demo receives inputs from input_data.json by default. To modify the inputs or specify a different path, adjust the input_path parameter in the command accordingly. It's recommended to avoid direct modifications to the input_data.json file.

## Results

The demo presents a comprehensive view of the Bloom model's performance in question answering tasks. It provides processing time metrics for various stages of the inference process based on a batch size of specified and showcases sample questions and their corresponding answers.

---
