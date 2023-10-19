# How to Run
use 'pytest models/demos/metal_BERT_large_15/demo/demo.py::test_demo[models/demos/metal_BERT_large_15/demo/input_data.json-1]' to run the demo. If you wish to run the demo with a different input use 'pytest models/demos/metal_BERT_large_15/demo/demo.py::test_demo[address_to_your_json_file.json-1]', this file is expected to have exactly 8 inputs.
Our second demo is designed to run SQuADV2 dataset, run this with 'pytest models/demos/metal_BERT_large_15/demo/demo.py::test_demo_squadv2'

# Inputs
Inputs by default are provided from 'input_data.json'. If you wish you to change the inputs or provide a different path to 'test_demo'. We do not recommend modifying 'input_data.json' file.

# Details
The entry point to metal bert model is 'TtBertBatchDram' in 'bert_model.py'. The model picks up certain configs and weights from huggingface pretrained model. We have used 'phiyodr/bert-large-finetuned-squad2' version from huggginface as our reference.
