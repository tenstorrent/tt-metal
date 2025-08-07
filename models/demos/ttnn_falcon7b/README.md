# Falcon7B

## Platforms:
    Note: This version has been deprecated

## Introduction
This code was designed for the `tiiuae/falcon-7b-instruct` model version from Huggingface.

Read more about it at [huggingface.co/tiiuae/falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct)

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run

To run the model for a single user you can use the command line input:

`pytest --disable-warnings -q -s --input-method=cli --cli-input="YOUR PROMPT GOES HERE!"  models/demos/ttnn_falcon7b/demo/demo.py`

To run the demo using prewritten prompts for a batch of 32 users run (currently only supports same token-length inputs):

`pytest --disable-warnings -q -s --input-method=json --input-path='models/demos/ttnn_falcon7b/demo/input_data.json' models/demos/ttnn_falcon7b/demo/demo.py`

### Inputs

A sample of input prompts for 32 users is provided in `input_data.json` in demo directory. If you wish you to run the model using a different set of input prompts you can provide a different path, e.g.:

`pytest --disable-warnings -q -s --input-method=json --input-path='path_to_input_prompts.json' models/demos/ttnn_falcon7b/demo/demo.py`

## Details

This model picks up certain configs and weights from huggingface pretrained model. We have used `tiiuae/falcon-7b-instruct` version from huggingface. The first time you run the model, the weights are downloaded and stored on your machine, and it might take a few minutes. The second time you run the model on your machine, the weights are being read from your machine and it will be faster.
