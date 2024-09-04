# Falcon7B Demo (T3000)

Falcon7b prefill uses 8x8 core grid size, so the following environment variable needs to be set on a T3000 setup:

```sh
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

## How to Run

### Token Generation (Default) Mode

- To run the demo using prewritten prompts for a batch of 256 users split evenly on 8 devices (currently only supports same token-length inputs):

```sh
pytest --disable-warnings -q -s --input-method=json --input-path='models/demos/t3000/falcon7b/input_data_t3000.json' models/demos/t3000/falcon7b/demo_t3000.py::test_demo_multichip[wormhole_b0-True-user_input0-8-True-default_mode_1024_stochastic]
```

- **Decoding method**: The default decoding method is top-k/top-p (stochastic) sampling, however greedy decoding can also be used by replacing `stochastic` with `greedy` in the command above.

### Performance Measurement Mode

- To measure the performance of generating the `i`'th token while the KV cache is filled with `i-1` rows (where `i` is 128 in the command below):

```sh
pytest --disable-warnings -q -s --input-method=json --input-path='models/demos/t3000/falcon7b/input_data_t3000.json' models/demos/t3000/falcon7b/demo_t3000.py::test_demo_multichip[wormhole_b0-True-user_input0-8-True-perf_mode_128_stochastic]
```

- **Supported sequence lengths**: Currently `i` can only be set to 128, 1024, or 2048 for performance measurement mode.

## Inputs

- A sample of input prompts for 256 users is provided in `input_data_t3000.json` in demo directory. If you wish you to run the model using a different set of input prompts you can provide a different path, e.g.:

```sh
pytest --disable-warnings -q -s --input-method=json --input-path='path_to_input_prompts.json' models/demos/t3000/falcon7b/demo_t3000.py::test_demo_multichip[wormhole_b0-True-user_input0-8-True-default_mode_1024_stochastic]
```

## Running on a different number of devices

- To run the demo on a different number of devices, an input file with the appropriate number of inputs must be prepared (the number of inputs should be (32 x num-devices)). 

- Then, the command above can be modified to replace '8' with the desired number of devices. 

- For example, to run with 4 devices:

```sh
pytest --disable-warnings -q -s --input-method=json --input-path='path_to_input_prompts.json' models/demos/t3000/falcon7b/demo_t3000.py::test_demo_multichip[wormhole_b0-True-user_input0-4-True-default_mode_1024_stochastic]
```

## Details

- **Weight caching**: This model picks up certain configs and weights from the Hugging Face pretrained model. We have used the `tiiuae/falcon-7b-instruct` version from Hugging Face. The first time you run the model, the weights are downloaded and stored on your machine, and it might take a few minutes. The second time you run the model on your machine, the weights are being read from cached files on your machine and it will be faster.
- **Max Context Length**: The maximum context/sequence length is currently limited to 2048 tokens (the default maximum sequence length for the Hugging Face model).
- **Batch Size**: Currently only a batch size of 32 is supported.
- **Token Generation Scheme**: The model will first run in prefill mode on the input sequences to fill the KV cache and then in decode mode to generate the output tokens.
