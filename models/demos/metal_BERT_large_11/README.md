# metal_BERT_large 11 Demo

## How to Run

Use `pytest --disable-warnings models/demos/metal_BERT_large_11/demo/demo.py::test_demo[models/demos/metal_BERT_large_11/demo/input_data.json-1]` to run the demo.

If you wish to run the demo with a different input use `pytest --disable-warnings models/demos/metal_BERT_large_11/demo/demo.py::test_demo[address_to_your_json_file.json-1]`. This file is expected to have exactly 8 inputs.

Our second demo is designed to run SQuADV2 dataset, run this with `pytest --disable-warnings models/demos/metal_BERT_large_11/demo/demo.py::test_demo_squadv2`.

Expected device perf: `~410 Inferences/Second`

To get the device performance, run `./tt_metal/tools/profiler/profile_this.py -c "pytest --disable-warnings models/demos/metal_BERT_large_11/tests/test_bert.py::test_bert[BERT_LARGE-batch_12-BFLOAT8_B-SHARDED_BATCH12]"`.
This will generate a CSV report under `<this repo dir>/generated/profiler/reports/ops/<report name>`. The report name will be shown at the end of the run.
<!-- csv_example = "images/BERT-Large-device-profile.png" -->

Expected end-to-end perf: `Ranges from 337 to 364 Inferences/Second, depending on the machine`

To get the end-to-end performance, run `pytest --disable-warnings models/demos/metal_BERT_large_11/tests/test_perf_bert11.py::test_perf_bare_metal`.

## Inputs

Inputs by default are provided from `input_data.json`. If you wish you to change the inputs or provide a different path to `test_demo`.

We do not recommend modifying `input_data.json` file.

## Details

The entry point to metal bert model is `TtBertBatchDram` in `bert_model.py`. The model picks up certain configs and weights from huggingface pretrained model. We have used `phiyodr/bert-large-finetuned-squad2` version from huggingface as our reference.

For fast model loading, we have cached preprocessed weights for TT tensors on Weka. These weights are directly read in and loaded to device.

If your machine does not have access to Weka, during model loading it will preprocess and convert the pytorch weights from huggingface to TT tensors before placing on device.
