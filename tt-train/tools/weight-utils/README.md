# Weight export utilities for tt-train

This folder contains a collection of scripts for exporting the weights of
various Huggingface models into a format suitable for use in tt-train. It is
recommended to run these scripts with [`uv`](https://docs.astral.sh/uv/) which
will automatically bring in their dependencies. If you prefer to run with plain
Python, you must install the dependencies (listed below) first and replace `uv
run` with your Python interpreter in the commands given below.

# Dependencies
- ipdb
- msgpack-numpy
- numpy
- torch
- transformers

# Llama 3

The script `llama_export.py` supports exporting weights for the Llama 3 family
of models.

## Synopsis

``` shell
uv run tinyllama_export.py [-i INPUT_PATH] [-o OUTPUT_PATH] [--hf_model HF_MODEL]
                [--meta_style] [-t DUMP_TOKENIZER_PATH]
```

## Details

- `-i, --input_path=PATH`
  - Path to the dumped original weights file. Default:
    "data/tinyllama_init.msgpack"
- `-o, --output_path=PATH`
  - Path to the output weights file. Default: "data/tinyllama_exported.msgpack"
- `--hf_model=MODEL_NAME`
  - Name of the Hugging Face model to use. Default: "TinyLlama/TinyLlama_v1.1"
- `--meta_style`
  - Specifies that the model is in the Meta style (QKV projections have
    dimensions interleaved). By default it is unset so that the projects have
    their even/odd dimensions split. What this means is that for each row `x`
    of, e.g., the query vector produced by the query projection, the row is
    expected to be represented as: `(x_0, x_1, x_2, x_3, ...)` in the Meta style
    and `(x_0, x_2, x_4, ..., x_1, x_3, x_5...)` in the Huggingface style.

    Whether to set this flag depends on the particular RoPE implementation used
    by the model you're trying to export. If you're unsure what to use, try both
    or feel free to ask us in the issue tracker.
- `-t, --dump_tokenizer_path=PATH`
  - Path to the output tokenizer file. If not given, the tokenizer will not be
    exported.

# GPT2
## Synopsis

``` shell
uv run gpt2_export.py INPUT_FILE OUTPUT_FILE
```

## Details
- `INPUT_FILE`
  - Path to the input msgpack file containing the existing GPT-2 weights.
- `OUTPUT_FILE`
  - Path where the updated msgpack file will be saved.
