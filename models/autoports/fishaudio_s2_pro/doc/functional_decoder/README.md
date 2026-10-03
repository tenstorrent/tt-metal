# Functional decoder — slow tower on TTNN (stage 1 analogue)

Hardware: Blackhole p300c chip 0 of a QuietBox 2 (`TT_METAL_VISIBLE_DEVICES=0`, mesh 1x1). tt-metal `8e18a76`, weights
`fishaudio/s2-pro@1de9996b`, slow tower weights bfp8_b (tt_transformers default accuracy policy), activations bf16.
Reference: fish-speech's own `DualARTransformer` blocks (vendored `reference/llama_ref.py`, fp32, CPU).

Test: `tests/test_slow_layer.py` (real weights, real shapes, prefill lengths 32 and 127 padded to 128).

| stage | PCC vs fish torch |
|---|---|
| host frame embedding (text + masked, scaled codebook sum) vs fish `embed` | 0.999999 |
| TT input residual (hidden-sharded bf16) vs host embedding | 1.000000 |
| 1 decoder layer, seq 32 | 0.999633 (rotate-half RoPE variant: 0.976 -> the interleaved/Meta convention is the right one) |
| 36 decoder layers, seq 32 / 127 | 0.999300 / 0.999387 (per-position 0.9996..1.0) |
| post-norm hidden state (LM-head tap) | 0.998728 |
| tied LM head logits (155776-way) | 0.999146, argmax identical |
| Generator path: prefill logits / hidden | 0.99915 / 0.99933 |
| Generator path: traced decode step (capture and replay) logits / hidden | 0.99935 / 0.99941 |

Conventions established: no `reverse_permute`/`use_hf_rope` (fish weights are already interleaved); `wqkv` split into
wq/wk/wv for tt_transformers; the head is tied (`output.weight` is `tok_embeddings.weight`); the fast decoder input is
the post-final-norm hidden state, tapped by wrapping `lm_head` (which deallocates its input, so the head gets a clone).

Not yet covered here: paged prefill/decode, batch > 1, lengths near the 32768 context, traced prefill, watcher run,
tt-perf-report tables — tracked in `work_log.md`.
