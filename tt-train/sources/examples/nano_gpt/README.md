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

`--output_file` : output filename (defaults to `$TT_METAL_HOME/tt-train/data/tokenized_data.csv`)

Once you have a pre-tokenized dataset, ensure that the data_path in your chosen YAML config matches the path to the tokenized dataset. If using a pre-tokenized dataset, ensure that the vocab size in your chosen model's YAML config matches the vocab size of the tokenizer you used. (This will be updated in a future PR to be more automatic)
