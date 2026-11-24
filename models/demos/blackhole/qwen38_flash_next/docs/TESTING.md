# Testing

> Historical source notes from Samuel Jett (sjettTT, sjett@tenstorrent.com), frozen at
> `cd9a11771107ea2c27da3303a0556ff7343e4af5`. Measurements and implementation details below
> describe that source snapshot. Use the [port README](../README.md) for current commands and
> [PROVENANCE.md](../PROVENANCE.md) for dependencies and the separation of source and port evidence.

The no-device tests, the acceptance gate and the reference corpus.

## The no-device tests

Run from the repository root:

    HF_MODEL=/data/Qwen3.8-Flash-Next python_env/bin/python -m pytest models/demos/blackhole/qwen38_flash_next/tests

`tests/` holds the no-device tests: component tests against the `tt/` torch reference (the CPU oracle every test
compares against), static contract tests over the `ttnn/` device model, the protocol, sampling, prefill-driver,
reference-corpus and harness tests.  Set `HF_MODEL` for the checkpoint-reading ones (the config, chat
format and component tests).

## The acceptance gate

Start the server with `--acceptance --require-json-96` (the README's section 4).  The shipped CPU greedy records
in `tools/acceptance/greedy-prompts.jsonl` replay against the CPU before the server listens (the twelve study records;
with `--prefill-slab` also the slab record, the `document` record, 2,228 prompt tokens); the `json` record
must match 96/96 or the server refuses to serve, and every record's first divergence index lands in `acceptance.json`
in the run directory.  The pinned indices a start is compared against are the chunked-prefill table (plain decode) and the `--mtp 4` table; a
replay that leaves the CPU stream earlier than its pin is a regression.  `NUMERICS.md` has the tables and what they
measure.

## The reference corpus and the agreement tool

`tools/reference/` freezes `Q38-REF-v1`: 36 teacher-forced items (the twelve acceptance
prompts as their records render them and as the server renders a request today, four requests with tools, the first
1024 tokens of two books, four evaluation items, two long prompts scored in 32-position windows), about 13k positions,
with sha256s.  `tools/qwen38_reference_corpus.py hf` runs the Transformers `qwen4_exp` model on the CPU over it (bf16
weights, fp32 LM head; a transformers checkout with the `qwen4_exp` model class, several hundred GB of RAM, hours) and
keeps per position the top-32 ids and log-probs and the teacher's log-prob; `oracle` does the same through `tt/` (bf16
or BF4-emulated experts); `device` converts a served chain's agreement records; `score` compares two columns (top-1 /
top-5 agreement, clear-margin top-1, truncated KL over the shared top-32 support, first divergence).  The HF column is
the acceptance reference the other columns are read against.  `hf` and `oracle` keep every scored position's full
logits with `--full-logits DIR` (fp16, one `.pt` per item); the server keeps the device's full-vocabulary logits at
the same positions with `--agreement-reference FILE --agreement-full-logits DIR` (one eager gather of the LM head's
bf16 row per recorded position, beside the agreement records), so two columns can be compared over the whole
vocabulary and not only over the candidate row.
