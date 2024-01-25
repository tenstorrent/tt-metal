# ttnn_functional_bloom Demo

## How to Run

Use `pytest --disable-warnings --input-path="models/experimental/functional_bloom/demo/input_data.json" models/experimental/functional_bloom/demo/demo.py::test_demo[models.experimental.functional_bloom.tt.ttnn_functional_bloom]` to run the ttnn functional bloom demo.

Use `pytest --disable-warnings --input-path="models/experimental/functional_bloom/demo/input_data.json" models/experimental/functional_bloom/demo/demo.py::test_demo[models.experimental.functional_bloom.tt.ttnn_optimized_functional_bloom]` to run the ttnn optimized functional bloom demo.

If you wish to run the demo with a different input use `pytest --disable-warnings --input-path='path_to_input_prompts.json' models/experimental/functional_bloom/demo/demo.py::test_demo[path_to_functional_bloom]`.This file is expected to have exactly 8 inputs.

Our second demo is designed to run HellaSwag dataset, run this with `pytest --disable-warnings models/experimental/functional_bloom/demo/demo.py::test_demo_hellaswag`

## Inputs

Inputs by default are provided from `input_data.json`. If you wish you to change the inputs or provide a different path to `test_demo`.

We do not recommend modifying `input_data.json` file.

## Details

The entry point to functional bloom model is `bloom` in `ttnn_functional_bloom.py`. We have used `bigscience/bloom-560m` version from huggingface as our reference.
