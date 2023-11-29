# metal_BERT_large 15 Demo

## How to Run

Use `pytest --disable-warnings models/demos/metal_BERT_large_11/demo/demo.py::test_demo[models/demos/metal_BERT_large_11/demo/input_data.json-1]` to run the demo.

If you wish to run the demo with a different input use `pytest --disable-warnings models/demos/metal_BERT_large_11/demo/demo.py::test_demo[address_to_your_json_file.json-1]`. This file is expected to have exactly 8 inputs.

Our second demo is designed to run SQuADV2 dataset, run this with `pytest --disable-warnings models/demos/metal_BERT_large_11/demo/demo.py::test_demo_squadv2`.

## Inputs

Inputs by default are provided from `input_data.json`. If you wish you to change the inputs or provide a different path to `test_demo`.

We do not recommend modifying `input_data.json` file.

## Details

The entry point to metal bert model is `TtBertBatchDram` in `bert_model.py`. The model picks up certain configs and weights from huggingface pretrained model. We have used `phiyodr/bert-large-finetuned-squad2` version from huggingface as our reference.

For fast model loading, we have cached preprocessed weights for TT tensors on Weka. These weights are directly read in and loaded to device.

If your machine does not have access to Weka, during model loading it will preprocess and convert the pytorch weights from huggingface to TT tensors before placing on device.
