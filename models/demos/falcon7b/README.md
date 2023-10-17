# How to Run
To run the demo using prewritten prompts run:
'pytest -q -s --input-path 'models/demos/falcon7b/demo/input_data.json' models/demos/falcon7b/demo/demo.py '

# Inputs
A sample of input prompts is provided in 'input_data.json' in demo directory. If you wish you to run the model using a different set of input prompts you can provide a different path, e.g.:

'pytest -q -s --input-path 'path_to_input_prompts.json' models/demos/falcon7b/demo/demo.py  '

# Details
This demo support same token-length inputs from 32 users. This model picks up certain configs and weights from huggingface pretrained model. We have used 'tiiuae/falcon-7b-instruct' version from huggginface.
