---

# Functional Bloom Model Demos For Text Generation and Question Answering

## Introduction

The Bloom model, exemplified by the `bigscience/bloom-560m` version, is a powerful language model renowned for its text generation capabilities. From simple completions to complex linguistic analyses, Bloom excels in tasks like cloze tests, counterfactuals, and reframed text generations. Applied downstream, it aids in information extraction, question answering, and summarization.

The demos showcases Functional Bloom Model for Text Generation and Question Answering tasks.

### Details

The entry point to the Functional Bloom model is the `bloom` function located in `ttnn_optimized_functional_bloom.py`. The `bigscience/bloom-560m` version from Hugging Face is utilized as the reference model.

## Sequence Size: 384

Sequence size determines the maximum length of input sequences processed by the model, optimizing performance and compatibility. It's recommended to set the `sequence_size` to 384

## Batch size: 8

Batch Size determines the number of input sequences processed simultaneously during training or inference, impacting computational efficiency and memory usage. It's recommended to set the `batch_size` to 8

## How to run demo for Text Generation task

To run the demo for text generation using the Bloom model, follow these instructions:

- Use the following command to run the bloom for text generation demo with ttnn functional bloom:
  ```
  pytest --disable-warnings --input-path="models/demos/grayskull/functional_bloom/demo/input_data_causal_lm.json" models/demos/grayskull/functional_bloom/demo/demo_causal_lm.py::test_demo[models.demos.grayskull.functional_bloom.tt.ttnn_functional_bloom]
  ```

- Alternatively, use the following command to run the bloom for text generation demo with the ttnn optimized functional bloom model:
  ```
  pytest --disable-warnings --input-path="models/demos/grayskull/functional_bloom/demo/input_data_causal_lm.json" models/demos/grayskull/functional_bloom/demo/demo_causal_lm.py::test_demo[models.demos.grayskull.functional_bloom.tt.ttnn_optimized_functional_bloom]
  ```

- If you wish to run the demo with a different input file, replace <address_to_your_json_file.json> with the path of your JSON file in the following command:
  ```
  pytest --disable-warnings --input-path=<address_to_your_json_file.json> models/demos/grayskull/functional_bloom/demo/demo_causal_lm.py::test_demo[path_to_functional_bloom]
  ```

- Our second demo is designed to run HellaSwag dataset, run this with the following command:
  ```
  pytest --disable-warnings models/demos/grayskull/functional_bloom/demo/demo_causal_lm.py::test_demo_hellaswag
  ```


## How to run demo for Question Answering task

To run the demo for question answering using the Bloom model, follow these instructions:

- Use the following command to run the bloom for question answering demo with ttnn functional bloom:
  ```
  pytest --disable-warnings --input-path="models/demos/grayskull/functional_bloom/demo/input_data_qa.json" models/demos/grayskull/functional_bloom/demo/demo_qa.py::test_demo[models.demos.grayskull.functional_bloom.tt.ttnn_functional_bloom]
  ```

- Alternatively, use the following command to run the bloom for question answering demo with the ttnn optimized functional bloom model:
  ```
  pytest --disable-warnings --input-path="models/demos/grayskull/functional_bloom/demo/input_data_qa.json" models/demos/grayskull/functional_bloom/demo/demo_qa.py::test_demo[models.demos.grayskull.functional_bloom.tt.ttnn_optimized_functional_bloom]
  ```

- If you wish to run the demo with a different input file, replace <address_to_your_json_file.json> with the path of your JSON file in the following command.:
  ```
  pytest --disable-warnings --input-path=<address_to_your_json_file.json> models/demos/grayskull/functional_bloom/demo/demo_qa.py::test_demo[path_to_functional_bloom]
  ```

- Our second demo is designed to run bloom for question answering with demo SQuADv2 dataset, run this with the following command:
  ```
  pytest --disable-warnings models/demos/grayskull/functional_bloom/demo/demo_qa.py::test_demo_squadv2
  ```

## Inputs

The demo receives inputs from respective input_data.json by default. To modify the inputs or specify a different path, adjust the input_path parameter in the command accordingly. It's recommended to avoid direct modifications to the input_data.json file.

## Results

The demo presents a comprehensive view of the Bloom model's performance in text generation and question answering tasks. It provides processing time metrics for various stages of the inference process based on a batch size of specified and showcases sample questions and their corresponding answers.

---
