## Bert-Tiny Demo

## Introduction
BERT stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

# Platforms:
    WH N300, N150

## How to Run

Use `pytest --disable-warnings models/demos/wormhole/bert_tiny/demo/demo.py::test_demo[wormhole_b0-True-models/demos/wormhole/bert_tiny/demo/input_data.json-mrm8488/bert-tiny-finetuned-squadv2-128-device_params0]` to run the demo.

If you wish to run the demo with a different input use `pytest --disable-warnings models/demos/wormhole/bert_tiny/demo/demo.py::test_demo[wormhole_b0-True-<address_to_your_json_file.json>-mrm8488/bert-tiny-finetuned-squadv2-128-device_params0]`. This file is expected to have exactly 16 inputs.


Our second demo is designed to run SQuADV2 dataset, run this with `pytest --disable-warnings models/demos/wormhole/bert_tiny/demo/demo.py::test_demo_squadv2[wormhole_b0-True-1-mrm8488/bert-tiny-finetuned-squadv2-384-device_params0]`.

If you wish to run for `n_iterations` samples, use `pytest --disable-warnings models/demos/wormhole/bert_tiny/demo/demo.py::test_demo_squadv2[wormhole_b0-True-<n_iterations>-mrm8488/bert-tiny-finetuned-squadv2-384-device_params0]`


# Inputs
Inputs by default are provided from `input_data.json`. If you wish you to change the inputs, provide a different path to test_demo.

We do not recommend modifying `input_data.json` file.

# Details
The entry point to  bert model is bert_for_question_answering in `models/demos/wormhole/bert_tiny/tt/bert_tiny.py`. The model picks up certain configs and weights from huggingface pretrained model. We have used `mrm8488/bert-tiny-finetuned-squadv2` version from huggingface as our reference.
