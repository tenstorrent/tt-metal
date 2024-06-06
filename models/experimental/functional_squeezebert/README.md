## functional_squeezebert Demo
## How to Run

Use `pytest --disable-warnings --input-path="models/experimental/functional_squeezebert/demo/input_data.json" models/experimental/functional_squeezebert/demo/demo.py::test_demo[models.experimental.functional_squeezebert.tt.ttnn_functional_squeezebert-squeezebert/squeezebert-uncased]` to run the demo.

If you wish to run the demo with a different input use `pytest --disable-warnings --input-path="<address_to_your_json_file.json>" models/experimental/functional_squeezebert/demo/demo.py::test_demo[models.experimental.functional_squeezebert.tt.ttnn_functional_squeezebert-squeezebert/squeezebert-uncased]`. This file is expected to have exactly 8 inputs.

Our second demo is designed to run SQuADV2 dataset, run this with `pytest --disable-warnings models/experimental/functional_squeezebert/demo/demo.py::test_demo_squadv2[3-models.experimental.functional_squeezebert.tt.ttnn_functional_squeezebert-squeezebert/squeezebert-uncased]`.

If you wish to run for `n_iterations` samples, use `pytest --disable-warnings models/experimental/functional_squeezebert/demo/demo.py::test_demo_squadv2[<n_iterations>-models.experimental.functional_squeezebert.tt.ttnn_functional_squeezebert-squeezebert/squeezebert-uncased]`


# Inputs
Inputs by default are provided from `input_data.json`. If you wish you to change the inputs, provide a different path to test_demo.

We do not recommend modifying `input_data.json` file.

# Details
The entry point to  functional_squeezebert model is squeezebert_for_question_answering in `models/experimental/functional_squeezebert/tt/ttnn_functional_squeezebert.py`. The model picks up certain configs and weights from huggingface pretrained model. We have used `squeezebert/squeezebert-uncased` version from huggingface as our reference.
