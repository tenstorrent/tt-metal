# Falcon40B Demo

## How to Run

Falcon40b prefill uses 8x8 core grid size, so the following environment variable needs to be set on T3000 setup:

`export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml`

To run the model for a single user you can use the command line input:

`pytest --disable-warnings -q -s --input-method=cli --cli-input="YOUR PROMPT GOES HERE!"  models/demos/t3000/falcon40b/demo/demo.py`

## Inputs

A sample of input prompts for 32 users is provided in `models/demos/t3000/falcon40b/demo/input_data.json`. If you wish to run the model using a different set of input prompts you can provide a different path `--input-path`. Run using:

`pytest --disable-warnings -q -s --input-method=json --input-path='models/demos/t3000/falcon40b/demo/input_data.json' models/demos/t3000/falcon40b/demo/demo.py`

## Details

- **Weight caching**: This model picks up certain configs and weights from huggingface pretrained model. The default model weights are the `tiiuae/falcon-40b-instruct` version from huggingface. The first time you run the model, weights are downloaded, pre-processed, and stored on your machine. This might take a few hours. The second time you run the model on your machine, the weights are being read from cached files on your machine and it will be faster.
- **Max Context Length**: The maximum context/sequence length for the demo is currently limited to 128 tokens. Support for context length 2048 is in testing.
- **Batch Size**: Currently only a batch size of 32 is supported.
- **Token Generation Scheme**: The model will first run in prefill mode on the input sequences to fill the KV cache and then in decode mode to generate the output tokens.
