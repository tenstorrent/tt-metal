# RoBERTa demo

Demo showcasing Data Parallel implementation of RoBERTa running on Wormhole - n150, n300 using ttnn.

## Introduction
RoBERTa builds on BERT and modifies key hyperparameters, removing the next-sentence pretraining objective and training with much larger mini-batches and learning rates.
RoBERTa is similar to BERT but with better pretraining techniques like Dynamic Masking, Sentence Packing, Larger Batches, Byte-level BPE vocabulary. The RoBERTa model was proposed in [RoBERTa: A Robustly Optimized BERT](https://arxiv.org/abs/1907.11692) Pretraining Approach based on Google’s BERT model released in 2018.

## Details
The entry point to ttnn_optimized_roberta model is bert_for_question_answering in `models/demos/wormhole/roberta/tt/ttnn_optimized_roberta.py`. The model picks up certain configs and weights from huggingface pretrained model. We have used `deepset/roberta-large-squad2` version from huggingface as our reference.

### Sequence Size: 384
Sequence size determines the maximum length of input sequences processed by the model, optimizing performance and compatibility. It's recommended to set the sequence_size to 384

### Batch size: 16
Batch Size determines the number of input sequences processed simultaneously during training or inference, impacting computational efficiency and memory usage. On each device, the batch size will be 8, as the operations run in parallel. It's recommended to set the batch_size to 16

## How to Run

Use `pytest --disable-warnings models/demos/wormhole/roberta/demo/demo.py::test_demo[wormhole_b0-True-models.demos.wormhole.roberta.tt.ttnn_optimized_roberta-8-384-deepset/roberta-large-squad2-models/demos/wormhole/roberta/demo/input_data.json]` to run the demo.

If you wish to run the demo with a different input use `pytest --disable-warnings models/demos/wormhole/roberta/demo/demo.py::test_demo[wormhole_b0-True-models.demos.wormhole.roberta.tt.ttnn_optimized_roberta-8-384-deepset/roberta-large-squad2-<address_to_your_customized_inputs_file.json>]`. This file is expected to have exactly 16 inputs.

Our second demo is designed to run SQuADV2 dataset, run this with `pytest --disable-warnings models/demos/wormhole/roberta/demo/demo.py::test_demo_squadv2[wormhole_b0-True-models.demos.wormhole.roberta.tt.ttnn_optimized_roberta-8-384-3-deepset/roberta-large-squad2]`.

If you wish to run for `n_iterations` samples, use `pytest --disable-warnings models/demos/wormhole/roberta/demo/demo.py::test_demo_squadv2[wormhole_b0-True-models.demos.wormhole.roberta.tt.ttnn_optimized_roberta-8-384-<n_iterations>-deepset/roberta-large-squad2]`

## Inputs
The demo receives inputs from respective `input_data.json` by default. To modify the inputs or specify a different path, adjust the input_path parameter in the command accordingly. It's recommended to avoid direct modifications to the input_data.json file.
