# Falcon40B

## Platforms:
        LoudBox, QuietBox (WH)

## Introduction
Read more about Falcon40b at the Huggingface for [tiiuae/falcon-40b-instruct](https://huggingface.co/tiiuae/falcon-40b-instruct).

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run
- Run the model for a single prompt using the command line input:

```sh
pytest --disable-warnings -q -s --input-method=cli --cli-input="YOUR PROMPT GOES HERE!"  models/demos/t3000/falcon40b/demo/demo.py
```

### Inputs
- A sample of input prompts for 32 users is provided in `models/demos/t3000/falcon40b/demo/input_data.json`.
- If you wish to run the model using a different set of input prompts you can provide a different path `--input-path`.

```sh
pytest --disable-warnings -q -s --input-method=json --input-path='models/demos/t3000/falcon40b/demo/input_data.json' models/demos/t3000/falcon40b/demo/demo.py
```

## Details
- **Weight caching**: This model picks up certain configs and weights from huggingface pretrained model. The default model weights are the `tiiuae/falcon-40b-instruct` version from huggingface. The first time you run the model, weights are downloaded, pre-processed, and stored on your machine. This might take a few hours. The second time you run the model on your machine, the weights are being read from cached files on your machine and it will be faster.

- **Max Context Length**: The maximum context/sequence length for the demo is currently limited to 128 tokens. Support for context length 2048 is in testing.

- **Batch Size**: Currently only a batch size of 32 is supported.

- **Token Generation Scheme**: The model will first run in prefill mode on the input sequences to fill the KV cache and then in decode mode to generate the output tokens.
