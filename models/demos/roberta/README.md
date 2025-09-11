# Functional_roberta

## Platforms:
    Wormhole (n150, n300)

## Introduction
 Functional_roberta is a large-scale question answering model based on RoBERTa, fine-tuned on SQuAD 2.0. It is designed to extract precise answers from text by understanding the relationship between questions and context.

Read more about it at [huggingface.co/deepset/roberta-large-squad2](https://huggingface.co/deepset/roberta-large-squad2)

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run
- If you wish to run the demo for ttnn_optimized_functional_roberta, use:
```
pytest --disable-warnings models/demos/roberta/demo/demo.py::test_demo[models.demos.bert.tt.ttnn_optimized_bert-8-384-deepset/roberta-large-squad2-models/demos/roberta/demo/input_data.json]
```

- If you wish to run the demo with a different input use
```
pytest --disable-warnings models/demos/roberta/demo/demo.py::test_demo[models.demos.bert.tt.ttnn_optimized_bert-8-384-deepset/roberta-large-squad2-<address_to_your_json_file>]
```
This file is expected to have exactly 8 inputs.

- Our second demo is designed to run SQuADV2 dataset, run this with
```
pytest --disable-warnings models/demos/roberta/demo/demo.py::test_demo_squadv2[models.demos.bert.tt.ttnn_optimized_bert-8-384-3-deepset/roberta-large-squad2]
```

- If you wish to run for `n_iterations` samples, use:
```
pytest --disable-warnings models/demos/roberta/demo/demo.py::test_demo_squadv2[models.demos.bert.tt.ttnn_optimized_bert-8-384-<n_iterations>-deepset/roberta-large-squad2]
```

## Details
The entry point to  functional_roberta model is bert_for_question_answering in `models/demos/bert/tt/ttnn_bert.py` (`models/demos/bert/tt/ttnn_optimized_bert.py` for optimized version). The model picks up certain configs and weights from huggingface pretrained model. We have used `deepset/roberta-large-squad2` version from huggingface as our reference.

### Inputs
- Inputs by default are provided from `input_data.json`. If you wish you to change the inputs, provide a different path to test_demo.
- We do not recommend modifying `input_data.json` file.
