# Mistral7B Demo

Demo showcasing Mistral-7B-instruct running on Wormhole, using ttnn.

## How to Run

To run the model for a single user you can use the command line input:

`pytest --disable-warnings -q -s --input-method=cli --cli-input="YOUR PROMPT GOES HERE!"  models/demos/mistral7b/demo/demo.py`

To run the demo using pre-written prompts for a batch of 32 users run (currently only supports same token-length inputs):

`pytest --disable-warnings -q -s --input-method=json --input-path='models/demos/mistral7b/demo/input_data_questions.json' models/demos/mistral7b/demo/demo.py`


## Inputs

A sample of input prompts for 32 users is provided in `input_data_question.json` in the demo directory. These are to be used in instruct-mode (default).
We also provide another set of generative inputs `input_data.json` for generative-mode of open-ended generation.

If you wish you to run the model using a different set of input prompts you can provide a different path, e.g.:

`pytest --disable-warnings -q -s --input-method=json --input-path='path_to_input_prompts.json' models/demos/mistral7b/demo/demo.py`

Keep in mind that for the instruct-mode, the prompts are automatically prefixed and suffixed by `[INST]` and `[/INST]`, respectively.


## Details

This model can be used with the general weights from Mistral-AI [Mistral-7B-v0.1](https://models.mistralcdn.com/mistral-7b-v0-1/mistral-7B-v0.1.tar) or the instruct weights
 [Mistral-7B-Instruct-v0.2](https://models.mistralcdn.com/mistral-7b-v0-2/Mistral-7B-v0.2-Instruct.tar).

Both these weights are consolidated into a single file `consolidated.00.pth`.
Keep in mind that the demo code expects the instruct weights to be named `consolidated_instruct.00.pth` instead, and the tokenizer to be named `tokenizer_instruct.model`.

You can provide a custom path to the folder containing the weights by adding the path argument to `TtModelArgs(model_base_path=<weights_folder>)`.

For more configuration settings, please check the file `tt/model_config.py`.

The `demo.py` code is set to run in instruct-mode by default. Change the hardcoded flag inside the code for the general weights.
The `test_mistral_model.py` is currently parametrized to choose between the general generative weights or the instruct weights.

The first time you run the model, the weights will be processed into the target data type and stored on your machine, which will take a few minutes for the full model. In future runs, the weights will be loaded from your machine and it will be faster.
