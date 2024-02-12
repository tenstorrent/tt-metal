# ttnn_functional_mistral Demo

## How to Run

Use `pytest --disable-warnings --input-path="models/experimental/functional_mistral/demo/input_data.json" models/experimental/functional_mistral/demo/demo.py::test_demo` to run the ttnn functional mistral demo.


If you wish to run the demo with a different input use `pytest --disable-warnings --input-path='path_to_input_prompts.json' models/experimental/functional_mistral/demo/demo.py::test_demo`.This file is expected to have exactly 8 inputs.

Our second demo is designed to run HellaSwag dataset, run this with `pytest --disable-warnings models/experimental/functional_mistral/demo/demo.py::test_demo_hellaswag`

If you wish to run for `n_iterations` samples, use `pytest --disable-warnings models/experimental/functional_mistral/demo/demo.py::test_demo_hellaswag[<n_iterations>]`

## Inputs

Inputs by default are provided from `input_data.json`. If you wish to change the inputs, provide a different path to `test_demo`.

We do not recommend modifying `input_data.json` file.

## Details

The entry point to functional mistral model is `mistral_transformer` in `ttnn_functional_mistral.py`. We have used `mistral-7B-v0.1` version as our reference.
