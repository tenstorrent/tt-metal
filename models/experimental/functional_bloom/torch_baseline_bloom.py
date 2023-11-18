# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from transformers import BloomForCausalLM, BloomTokenizerFast


def generate_text(input_text, model, tokenizer, max_length=64):
    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate text using greedy approach
    output_ids = model.generate(input_ids, max_length=max_length)

    # Decode the generated ids to a string
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
    model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")
    model.eval()  # Set the model to evaluation mode

    input_text = "summarize: QuillBot's Summarizer wants to change how you read! Instead of reading through loads of documents, you can get a short annotated summary or bullet points with all the key information."
    expected_generated_text = generate_text(input_text, model, tokenizer)
    print(expected_generated_text)
