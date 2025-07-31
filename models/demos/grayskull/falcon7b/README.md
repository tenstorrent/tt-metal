# Falcon7B

## Platforms:
    Grayskull

## Introduction
This code was designed for the `tiiuae/falcon-7b-instruct` model version from Huggingface.

Read more about it at [huggingface.co/tiiuae/falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct)

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run
- To run the model for a single user you can use the command line input:
```
pytest --disable-warnings -q -s --input-method=cli --cli-input="YOUR PROMPT GOES HERE!"  models/demos/grayskull/falcon7b/demo_grayskull.py::test_demo[user_input0-default_mode_stochastic]
```

- To run the demo using prewritten prompts for a batch of 32 users run (currently only supports same token-length inputs):
```
pytest --disable-warnings -q -s --input-method=json --input-path='models/demos/falcon7b_common/demo/input_data.json' models/demos/grayskull/falcon7b/demo_grayskull.py::test_demo[user_input0-default_mode_stochastic]
```

### Inputs
A sample of input prompts for 32 users is provided in `input_data.json` in demo directory. If you wish you to run the model using a different set of input prompts you can provide a different path, e.g.:
```
pytest --disable-warnings -q -s --input-method=json --input-path='path_to_input_prompts.json' models/demos/grayskull/falcon7b/demo_grayskull.py::test_demo[user_input0-default_mode_stochastic]
```

## Details
- This model picks up certain configs and weights from huggingface `tiiuae/falcon-7b-instruct` pretrained model.
- The first time you run the model, the weights are downloaded and stored on your machine, and it might take a few minutes. The second time you run the model on your machine, the weights are being read from your machine and it will be faster.
