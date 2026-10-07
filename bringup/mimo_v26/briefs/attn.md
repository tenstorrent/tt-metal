# Track 1 — Attention (SWA + global), unequal head dims 192 QK / 128 V

Branch/worktree: `ssinghal/mimo-attn` / `/mnt/tt-data/ssinghal/mimo/wt/attn`. Host: .30 only. Skill: `tt-moe-correctness` (PCC section), `tt-lab-ops`.
Package: `models/demos/blackhole/mimo_v26/tt/attention*.py`, tests in `models/demos/blackhole/mimo_v26/tests/`. Must not touch `reference/` (track 0) or `tt/moe*` (track 2).

Phase A (first, 3 h box): op probe on a single device / small mesh BEFORE designing the module. Candidates (study section 4.1, unverified): `flash_multi_latent_attention_decode` / paged variant with
explicit V (head_dim_v=128, attention_sink, sliding_window_size), MLA-style prefill SDPA with explicit V, vs the fallback of zero-padding V to 192. Check: accepts shapes (Q 64 heads x 192, K 192, V 128; GQA 8 and 4 kv heads), sink and sliding
window (128) in decode and prefill, paged KV, causal, chunked prefill with cache read. Compare each against a torch fp32 reference with the sink convention (HF adds the raw sink logit after scaling, scale 192^-0.5; device sink tensors may
expect sink/scale: derive and TEST it). Measure device time of MLA-style (V=128) vs padded V=192 at the shapes that matter (decode B up to 32 users/row, prefill chunk 2048 tokens, cache 64k global / 128-window SWA).
Decide pad vs native from measurements; LOG the decision with numbers (paired, in-process). Report to the coordinator at the end of phase A.
Phase B: attention module per layer type (SWA ring/window KV; global paged KV, GA 4 KV heads replicated across 8 TP columns), TP=8 on mesh columns, partial RoPE (64 of 192), V*0.707 (cache holds scaled V), bfp8 qkv/o weights, bf16 KV.
Weights: use track 0's `load_layer` once merged to your base; until then test with torch-generated weights of the real shapes plus (when available) real layer weights from `ssinghal/mimo-ref`.
Gate: each layer type PCC >= 0.999 vs track-0 dump on real activations, several shapes (S and batch incl. extremes: decode 1 user, 32 users/row; prefill 1 chunk and 4+ chunks; position far past the window).
