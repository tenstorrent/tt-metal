# gemma-4-31B-it tokenizer with add_bos_token (update 6)

The tokenizer files of google/gemma-4-31B-it @ 842da3794eaa0b77d5f08bae87a17459d91ff475 with two changes, so a /v1/completions prompt gets exactly one
leading <bos> (id 2); chat requests are unchanged (their template already writes <bos>, and the chat path encodes it
without special tokens). Both serving profiles pass this directory to vLLM as --tokenizer.

- tokenizer_config.json: "add_bos_token": true, "add_eos_token": false.
- tokenizer.json: the post_processor emits <bos> before each sequence (single "<bos>:0 $A:0", pair "<bos>:0 $A:0
  <bos>:1 $B:1"), as transformers builds it from add_bos_token.
- chat_template.jinja, config.json, generation_config.json, processor_config.json: unchanged copies.

Don't add <bos> to a completions prompt yourself: a prompt that starts with the literal text "<bos>" gets two.
