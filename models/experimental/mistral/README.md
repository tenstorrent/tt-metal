# Mistral7B Demo

## How to Run

To run the model for a single user you can use the command line input:

`pytest --disable-warnings -q -s --input-method=cli --cli-input="YOUR PROMPT GOES HERE!"  models/experimental/mistral/demo/demo.py`

To run the demo using prewritten prompts for a batch of 16 users run :

`pytest --disable-warnings -q -s --input-method=json --input-path='models/experimental/mistral/demo/input_data.json' models/experimental/mistral/demo/demo.py`

## Inputs

A sample of input prompts for 32 users is provided in `input_data.json` in demo directory. If you wish you to run the model using a different set of input prompts you can provide a different path, e.g.:

`pytest --disable-warnings -q -s --input-method=json --input-path='path_to_input_prompts.json' models/experimental/mistral/demo/demo.py`

## Details

This model loads configs and weights of pretrained model from Weka path. We have used `mistral-7B-v0.1` version of the pretrained model.
