# Falcon7B Demo (T3000)

Falcon7b prefill uses 8x8 core grid size, so the following environment variable needs to be set on N300 setup:

`export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml`

## How to Run

To run the demo using prewritten prompts for a batch of 256 users split evenly on 8 devices run (currently only supports same token-length inputs):

`pytest --disable-warnings -q -s --input-method=json --input-path='models/demos/t3000/falcon7b/input_data_t3000.json' models/demos/t3000/falcon7b/demo_t3000.py::test_demo_multichip[user_input0-8-True-default_mode_1024_stochastic]`

## Inputs

A sample of input prompts for 256 users is provided in `input_data_t3000.json` in demo directory. If you wish you to run the model using a different set of input prompts you can provide a different path, e.g.:

`pytest --disable-warnings -q -s --input-method=json --input-path='path_to_input_prompts.json' models/demos/t3000/falcon7b/demo_t3000.py::test_demo_multichip[user_input0-8-True-default_mode_1024_stochastic]`

## Running on a different number of devices

To run the demo on a different number of devices, an input file with the appropriate number of inputs must be prepared (the number of inputs should be (32 x num-devices)). Then, the command above can be modified to replace '8' with
the desired number of devices. For example, to run with 4 devices:

`pytest --disable-warnings -q -s --input-method=json --input-path='path_to_input_prompts.json' models/demos/t3000/falcon7b/demo_t3000.py::test_demo_multichip[user_input0-4-True-default_mode_1024_stochastic]`

## Details

This model picks up certain configs and weights from huggingface pretrained model. We have used `tiiuae/falcon-7b-instruct` version from huggingface. The first time you run the model, the weights are downloaded and stored on your machine, and it might take a few minutes. The second time you run the model on your machine, the weights are being read from your machine and it will be faster.
