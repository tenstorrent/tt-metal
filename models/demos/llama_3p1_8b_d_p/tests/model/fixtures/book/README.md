# Book continuation fixture

`pride_and_prejudice.txt` is a short excerpt from Jane Austen's *Pride and
Prejudice*, a public-domain work. `provenance.json` identifies the exact Project
Gutenberg source, source/body hashes and selected character/token ranges. Preserve
the excerpt's CRLF bytes and lack of a final newline: its end is a token boundary,
not a complete source line. The file is excluded from the end-of-file fixer because
normalizing it would invalidate the pinned prompt hash.

`token_ids.json` contains one Llama BOS followed by 4,095 book tokens. The 2K case
uses its first 2,048 IDs; the 4K case uses all 4,096. These are raw book-continuation
prompts, without a chat template or repeated text.

`golden_2048.json` and `golden_4096.json` record the independent Hugging Face
FP32 model's final-token top five and checkpoint/tokenizer hashes. The test checks
whether the reference's **highest-ranked token** is in TT's top five over all
128,256 vocabulary entries. The next printed word in the book is not the expected
model prediction.

Regenerate on a CPU compute host with enough memory for the 8B FP32 model:

```sh
python models/demos/llama_3p1_8b_d_p/scripts/generate_book_golden.py \
  --checkpoint "$LLAMA31_8B_CHECKPOINT" \
  --threads 8 \
  --output-dir /path/to/private-golden-run
```

The script loads the checkpoint once, computes all 32 layers and final norm,
projects only the final valid hidden row, and saves 2K before starting 4K. Copy only
the two `golden_*.json` files here. Keep `run.json`, timing/affinity information,
full-vocabulary logits and logs in the private output directory.
