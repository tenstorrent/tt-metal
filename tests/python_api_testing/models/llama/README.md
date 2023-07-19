# Installation guide

In order to use HuggingFace Llama model we need a development version of Transformers library (v4.28.0.dev0).

# Use pretrained weights

The weights used in the tests are downloaded from: https://huggingface.co/huggyllama/llama-7b

How to use the weights:
    ```
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b")
    ```

An issue which probably will appear is:

`ValueError: Tokenizer class LLaMATokenizer does not exist or is not currently imported`

To solve it please use this thread: https://github.com/huggingface/transformers/issues/22222#issuecomment-1477171703
Change the LLaMATokenizer in tokenizer_config.json into lowercase LlamaTokenizer and it works.

 # Stacked decoders tests

 Tests in which only Llama Decoders are stacked are needed because it has been observed that as their number increases, the PCC value drops. Please check the spreedsheet/Llama card
