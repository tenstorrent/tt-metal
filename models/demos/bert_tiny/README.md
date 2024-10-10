## Bert-Tiny Demo
## How to Run

Use `pytest --disable-warnings --input-path="models/demos/bert_tiny/demo/input_data.json" models/demos/bert_tiny/demo/demo.py::test_demo[mrm8488/bert-tiny-finetuned-squadv2-128-8-device_params0]` to run the demo.


If you wish to run the demo with a different input use `pytest --disable-warnings --input-path="<address_to_your_json_file.json>" models/demos/bert_tiny/demo/demo.py::test_demo[mrm8488/bert-tiny-finetuned-squadv2-128-8-device_params0]`. This file is expected to have exactly 8 inputs.

Our second demo is designed to run SQuADV2 dataset, run this with `pytest --disable-warnings models/demos/bert_tiny/demo/demo.py::test_demo_squadv2[1-mrm8488/bert-tiny-finetuned-squadv2-384-8-device_params0]`.

If you wish to run for `n_iterations` samples, use `pytest --disable-warnings models/demos/bert_tiny/demo/demo.py::test_demo_squadv2[<n_iterations>-mrm8488/bert-tiny-finetuned-squadv2-384-8-device_params0]`


# Inputs
Inputs by default are provided from `input_data.json`. If you wish you to change the inputs, provide a different path to test_demo.

We do not recommend modifying `input_data.json` file.

# Details
The entry point to  bert model is bert_for_question_answering in `models/demos/bert_tiny/tt/bert_tiny.py`. The model picks up certain configs and weights from huggingface pretrained model. We have used `mrm8488/bert-tiny-finetuned-squadv2` version from huggingface as our reference.
