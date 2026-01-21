The C++ version of this example depends on the training dataset to be pre-tokenized if using byte-pair encoding (BPE); it will not work otherwise! (Character-level tokenizing works natively)

To pre-tokenize the Shakespeare dataset (as an example), use the `dataset_to_tokens.py` script found in tt-train/tools.

Example usage:
```
cd $TT_METAL_HOME/tt-train
python tools/dataset_to_tokens.py --text_file data/shakespeare.txt --hf_tokenizer gpt2
```

Arguments:

`--text_file` : the path to the dataset in text format (assumes a single corpus for autoregressive training)

`--hf_tokenizer` : name of Hugging Face pre-trained tokenizer

`--output_file` : output YAML filename, defaults to `$TT_METAL_HOME/tt-train/data/tokenized_data.yaml`

The script will use the specified pre-trained Hugging Face tokenizer to pre-tokenize the corpus. This pre-tokenized data will then be saved to a YAML file as a list of integers, along with the data length (in tokens), and the tokenizer vocab size. The YAML file will be saved to the specified `--output_file`, which can then be used in the training config.
