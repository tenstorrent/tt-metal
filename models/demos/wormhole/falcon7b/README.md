# Falcon7B Demo (Wormhole)

Falcon7b prefill uses 8x8 core grid size, so the following environment variable needs to be set on N300 setup:

`export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml`

## How to Run

To run the model for a single user you can use the command line input:

`pytest --disable-warnings -q -s --input-method=cli --cli-input="YOUR PROMPT GOES HERE!"  models/demos/wormhole/falcon7b/demo_wormhole.py::test_demo[user_input0-default_mode_1024_stochastic]`

To run the demo using prewritten prompts for a batch of 32 users run (currently only supports same token-length inputs):

`pytest --disable-warnings -q -s --input-method=json --input-path='models/demos/falcon7b/demo/input_data.json' models/demos/wormhole/falcon7b/demo_wormhole.py::test_demo[user_input0-default_mode_1024_stochastic]`

## Inputs

A sample of input prompts for 32 users is provided in `input_data.json` in demo directory. If you wish you to run the model using a different set of input prompts you can provide a different path, e.g.:

`pytest --disable-warnings -q -s --input-method=json --input-path='path_to_input_prompts.json' models/demos/wormhole/falcon7b/demo_wormhole.py::test_demo[user_input0-default_mode_1024_stochastic]`

## Details

This model picks up certain configs and weights from huggingface pretrained model. We have used `tiiuae/falcon-7b-instruct` version from huggingface. The first time you run the model, the weights are downloaded and stored on your machine, and it might take a few minutes. The second time you run the model on your machine, the weights are being read from your machine and it will be faster.
