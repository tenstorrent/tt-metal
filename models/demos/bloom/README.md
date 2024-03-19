# ttnn_functional_bloom for QA Demo

## How to Run

Use `pytest --disable-warnings --input-path="models/demos/bloom/demo/input_data.json" models/demos/bloom/demo/demo_qa.py::test_demo[models.demos.bloom.tt.ttnn_functional_bloom]` to run the ttnn functional bloom for question answering demo.

Use `pytest --disable-warnings --input-path="models/demos/bloom/demo/input_data.json" models/demos/bloom/demo/demo_qa.py::test_demo[models.demos.bloom.tt.ttnn_optimized_functional_bloom]` to run the ttnn optimized functional bloom for question answering demo.

If you wish to run the demo with a different input use `pytest --disable-warnings --input-path=<address_to_your_json_file.json> models/demos/bloom/demo/demo_qa.py::test_demo[path_to_functional_bloom]`.

Our second demo is designed to run with SQuADV2 dataset.

Use `pytest --disable-warnings models/demos/bloom/demo/demo_qa.py::test_demo_squadv2[5-models.demos.bloom.tt.ttnn_functional_bloom]` to run the second demo with ttnn functional bloom for question answering demo.

Use `pytest --disable-warnings models/demos/bloom/demo/demo_qa.py::test_demo_squadv2[5-models.demos.bloom.tt.ttnn_optimized_functional_bloom]` to run the second demo with ttnn optimized functional bloom for question answering demo.

If you wish to run for `n_iterations` samples, use `pytest --disable-warnings models/demos/bloom/demo/demo_qa.py::test_demo_squadv2[<n_iterations>-models.demos.bloom.tt.ttnn_optimized_functional_bloom]`.

## Inputs

Inputs are provided from `input_data.json` by default. If you wish you to change the inputs or provide a different path to `test_demo`.

We do not recommend modifying `input_data.json` file.

## Details

The entry point to functional bloom model is `bloom` in `ttnn_functional_bloom.py`. We used `bigscience/bloom-560m` version from huggingface as reference.
