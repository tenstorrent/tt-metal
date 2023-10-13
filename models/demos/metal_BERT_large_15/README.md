# How to Run
use 'pytest models/demos/metal_BERT_large_15/demo/demo.py' to run the demo

# Inputs
Inputs by default are provided from 'input_data.json'. If you wish you to change the inputs, you can modify 'input_data.json' or provide a different path to 'test_demo'.

# Details
The entry point to metal bert model is 'TtBertBatchDram' in 'bert_model.py'. The model picks up certain configs and weights from huggingface pretrained model. We have used 'phiyodr/bert-large-finetuned-squad2' version from huggginface.
