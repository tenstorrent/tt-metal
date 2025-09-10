# Bert

## Platforms
    Wormhole (n150)

## Introduction
BERT stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run
Run the demo:
 ```
 pytest --disable-warnings --input-path="models/demos/bert/demo/input_data.json" models/demos/bert/demo/demo.py::test_demo[models.demos.bert.tt.ttnn_bert-phiyodr/bert-large-finetuned-squad2]
 ```

Run the demo for ttnn_optimized_bert:
```
pytest --disable-warnings --input-path="models/demos/bert/demo/input_data.json" models/demos/bert/demo/demo.py::test_demo[models.demos.bert.tt.ttnn_optimized_bert-phiyodr/bert-large-finetuned-squad2]
```

Run the demo with a different input:
```
pytest --disable-warnings --input-path="<address_to_your_json_file.json>" models/demos/bert/demo/demo.py::test_demo[models.demos.bert.tt.ttnn_bert-phiyodr/bert-large-finetuned-squad2]
```
This file is expected to have exactly 8 inputs.

Our second demo is designed to run SQuADV2 dataset, run it with:
```
pytest --disable-warnings models/demos/bert/demo/demo.py::test_demo_squadv2[3-models.demos.bert.tt.ttnn_optimized_bert-phiyodr/bert-large-finetuned-squad2]
```

If you wish to run for `n_iterations` samples, use:
```
pytest --disable-warnings models/demos/bert/demo/demo.py::test_demo_squadv2[<n_iterations>-models.demos.bert.tt.ttnn_optimized_bert-phiyodr/bert-large-finetuned-squad2]
```

## Details
The entry point to  bert model is bert_for_question_answering in `models/demos/bert/tt/ttnn_bert.py` (`models/demos/bert/tt/ttnn_optimized_bert.py` for optimized version). The model picks up certain configs and weights from huggingface pretrained model. We have used `phiyodr/bert-large-finetuned-squad2` version from huggingface as our reference.

### Inputs
Inputs by default are provided from `input_data.json`. If you wish you to change the inputs, provide a different path to test_demo.

We do not recommend modifying `input_data.json` file.
