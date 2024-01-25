# ttnn_functioanl_t5 Demo

## How to Run

Use `pytest --disable-warnings --input-path="models/experimental/functional_t5/demo/input_data.json" models/experimental/functional_t5/demo/demo.py::test_functional_t5_demo` to run the functional t5 demo.

If you wish to run the demo with a different input use `pytest --disable-warnings --input-path="[address_to_your_json_file]" models/experimental/functional_t5/demo/demo.py::test_functional_t5_demo`. This file is expected to have exactly 8 inputs.

Our second demo is designed to run SQuADV2 dataset, run this with `pytest --disable-warnings models/experimental/functional_t5/demo/demo.py::test_functional_t5_demo_squadv2`


## Inputs

Inputs by default are provided from `input_data.json`. If you wish you to change the inputs or provide a different path to `test_functional_t5_demo`.

We do not recommend modifying `input_data.json` file.

## Details

The entry point to metal bert model is `t5_for_conditional_generation` in `ttnn.functional_t5.py`. The model picks up certain configs and weights from huggingface pretrained model. We have used `t5-small` and `google/flan-t5-small` versions from huggingface as our reference.
