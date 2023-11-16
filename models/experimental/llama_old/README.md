# Installation guide

In order to use HuggingFace Llama model we need a development version of Transformers library (v4.28.0.dev0).

# Use pretrained weights

The weights used in the tests are downloaded from: https://huggingface.co/baffo32/decapoda-research-llama-7B-hf

How to use the weights:
    ```
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(hf-internal-testing/llama-tokenizer)
    model = AutoModelForCausalLM.from_pretrained("baffo32/decapoda-research-llama-7B-hf)
    ```

An issue which probably will appear is:

`ValueError: Tokenizer class LLaMATokenizer does not exist or is not currently imported`

To solve it please use this thread: https://github.com/huggingface/transformers/issues/22222#issuecomment-1477171703
Change the LLaMATokenizer in tokenizer_config.json into lowercase LlamaTokenizer and it works.
