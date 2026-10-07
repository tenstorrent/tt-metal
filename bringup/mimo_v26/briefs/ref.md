# Track 0 — Reference + checkpoint + dumps (CPU only)

Branch/worktree: `ssinghal/mimo-ref` / `/mnt/tt-data/ssinghal/mimo/wt/ref`. No device. CPU budget: run heavy jobs with `nice`, <= 32 threads, host RAM 566 GB (never load the whole model; stream layer by layer).
Skill: `tt-model-reference`. Output package: `models/demos/blackhole/mimo_v26/reference/` (+ `tests/`), dumps under `/mnt/tt-data/ssinghal/mimo-dumps/`.
Checkpoint: find the local/NFS copy of the HF weights (ask coordinator if none exists; do NOT download without asking). Index/config copies in `bringup_study/mimo/`.

Goal / gate (tt-moe-bringup track 0): (a) single-layer oracle per layer type (dense L0, SWA-MoE L1, global-MoE L5) matches the HF module (`modeling_mimo_v2.py`) on a real hidden state, PCC > 0.999, same router set;
short CPU chain (first 6 layers = L0..L5 covers all three types) matches HF; (b) per-layer dumps for real-text prompts (GSM8K chat-template prompts + one long book-style prompt): S=128 b16 first, then S=512 b4, S=2048 b1, later S=8192 b1;
format per `tt-model-reference` section 3 (h_in, h_out, attn_in/out, ffn_in/out, routing_idx+weights, KV after prefill, decode-step dump). Hash prompts (sha256) in LOG; (c) bf16 and fp32 variants -> noise floor.

First task (blocks everything else, do it first, report within ~2 h): resolve the qkv_proj row layout / scale layout (study risk 1: scale_inv shapes [108,32] GA / [116,32] SWA suggest per-TP4-rank
[q_r|k_r|v_r] blocking). Decide using the checkpoint files, `sgl.py` and the HF code, with a real test (e.g. dequant both ways, run layers with real weights on real tokens, compare to a
known-good signal such as sensible per-layer statistics or HF/sglang logic). Write a `decision` LOG entry, and a loader function `load_layer(i, dtype)` used by everyone (dequant fp8 128x128 block, fp4+e8m0 expert dequant:
verify nibble order and e8m0 bias against DSV4 `reference/ref_kernels.dequant_fp4_weight` AND against the fp8->bf16/known value ranges; report max abs error vs re-quantisation).
Also verify and log (as `decision` entries, status per what you actually checked): SWA mask convention (window 128 inclusive of self, from transformers `create_sliding_window_causal_mask`), sink semantics, V scale 0.707,
partial RoPE (first 64 of 192, rotate_half split-half, theta 1e7 global / 1e4 SWA), router semantics, no clamp.
Secondary: MTP-3 structure (in-checkpoint, no HF code): collect from README/sgl.py/dflash code what exists; do NOT build it now. Hosted-oracle prompts (OpenRouter) are optional: ask before using any network service.
