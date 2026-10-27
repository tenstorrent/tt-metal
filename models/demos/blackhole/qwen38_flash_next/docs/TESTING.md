# Testing

## Tests

Run from the repository root:

    QWEN38_CHECKPOINT=/data/Qwen3.8-Flash-Next python_env/bin/python -m pytest models/demos/blackhole/qwen38_flash_next/tests

`tests/` holds the no-device tests (protocol, chat format, sampling, prefill driver, MTP session and MTP rows /
verify / draft, batched and MTP lanes, lane scheduler, vision inputs, launcher), the component tests against the
`tt/` torch reference, and the device tests (`*_device.py`, `*_tp.py`). The checkpoint-reading tests skip unless
`QWEN38_CHECKPOINT` (or `MODEL_WEIGHTS_DIR`) names the checkpoint.

## The acceptance gate

Start the server with `--acceptance --require-json-96`. The CPU greedy records in
`tools/acceptance/greedy-prompts.jsonl` replay before the server listens (with `--prefill-slab` also the slab
record, `document`); the `json` record must match 96/96 or the server refuses to serve, and every record's first
divergence index is written to `acceptance.json` in the run directory. See [NUMERICS.md](NUMERICS.md).

## The reference corpus

`tools/reference/` freezes `Q38-REF-v1`: 36 teacher-forced items (the acceptance prompts as their records and as
the server render them, requests with tools, the opening of two books, evaluation items and two long prompts scored
in windows) with sha256s. `tools/qwen38_reference_corpus.py` has the modes:

- `hf`: the Transformers `qwen4_exp` model on the CPU (bf16 weights, fp32 LM head; needs a transformers checkout
  with the `qwen4_exp` class and several hundred GB of RAM);
- `oracle`: the same through `tt/` (bf16 or BF4-emulated experts);
- `device`: a served chain's agreement records, written by the server with
  `--agreement-reference FILE [--agreement-full-logits DIR]`;
- `score`: compares two columns (top-1 / top-5 agreement, clear-margin top-1, truncated KL over the shared top-32
  support, first divergence);
- `freeze` and `verify`: build and check the corpus files.

`hf` and `oracle` keep each scored position's full logits with `--full-logits DIR`, so two columns can be compared
over the whole vocabulary.
