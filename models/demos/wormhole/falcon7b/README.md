# Falcon7B Demo (Wormhole)

Falcon7b prefill uses 8x8 core grid size, so the following environment variable needs to be set on N300 setup:

`export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml`

## How to Run

#### Token Generation (Default) Mode

To run the model for a single user you can use the command line input:

`pytest --disable-warnings -q -s --input-method=cli --cli-input="YOUR PROMPT GOES HERE!"  models/demos/wormhole/falcon7b/demo_wormhole.py::test_demo[wormhole_b0-True-user_input0-1-default_mode_1024_stochastic]`

To run the demo using prewritten prompts for a batch of 32 users run (currently only supports same token-length inputs):

`pytest --disable-warnings -q -s --input-method=json --input-path='models/demos/falcon7b_common/demo/input_data.json' models/demos/wormhole/falcon7b/demo_wormhole.py::test_demo[wormhole_b0-True-user_input0-1-default_mode_1024_stochastic]`

- **Decoding method**: The default decoding method is top-k/top-p (stochastic) sampling, however greedy decoding can also be used by replacing `stochastic` with `greedy` in the command above.

#### Performance Measurement Mode

To measure the performance of generating the `i`'th token while the KV cache is filled with `i-1` rows (where `i` is 128 in the command below):

`pytest --disable-warnings -q -s --input-method=json --input-path='models/demos/falcon7b_common/demo/input_data.json' models/demos/wormhole/falcon7b/demo_wormhole.py::test_demo[wormhole_b0-True-user_input0-1-perf_mode_128_stochastic]`

- **Supported sequence lengths**: Currently `i` can only be set to 128, 1024, or 2048 for performance measurement mode.

## Inputs

A sample of input prompts for 32 users is provided in `input_data.json` in demo directory. If you wish you to run the model using a different set of input prompts you can provide a different path, e.g.:

`pytest --disable-warnings -q -s --input-method=json --input-path='path_to_input_prompts.json' models/demos/wormhole/falcon7b/demo_wormhole.py::test_demo[wormhole_b0-True-user_input0-1-default_mode_1024_stochastic]`

## Details

- **Weight caching**: This model picks up certain configs and weights from the huggingface pretrained model. We have used the `tiiuae/falcon-7b-instruct` version from huggingface. The first time you run the model, the weights are downloaded and stored on your machine, and it might take a few minutes. The second time you run the model on your machine, the weights are being read from cached files on your machine and it will be faster.
- **Max Context Length**: The maximum context/sequence length is currently limited to 2048 tokens (the default maximum sequence length for the huggingface model).
- **Batch Size**: Currently only a batch size of 32 is supported.
- **Token Generation Scheme**: The model will first run in prefill mode on the input sequences to fill the KV cache and then in decode mode to generate the output tokens.
