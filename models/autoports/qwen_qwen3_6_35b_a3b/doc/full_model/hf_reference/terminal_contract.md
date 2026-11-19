# Pinned HF text autoregressive path

Model: `Qwen/Qwen3.6-35B-A3B`, revision
`345d467b117d1fd5eda9b879a5988de9ae5f47a7`, Transformers 5.12.1.
Exact installed source locations are in `hf_source_locations.json`.

- The checkpoint declares `Qwen3_5MoeForConditionalGeneration`; the installed
  `AutoModelForCausalLM` mapping supports this config and selects the text-only
  `Qwen3_5MoeForCausalLM`. No image/video features participate in text inference.
- Embedding is a direct `nn.Embedding(248320, 2048, pad_token_id=None)` lookup,
  without embedding scaling. Checkpoint key:
  `model.language_model.embed_tokens.weight`.
- Forty layers use the unchanged residual stream: 30 linear-attention layers
  and ten full-attention layers (indices 3, 7, ..., 39). Position embeddings are
  computed once by the text model and passed to all layers. For text, the
  temporal/height/width rotary coordinates equal absolute text positions;
  masks and cache updates also use absolute text positions.
- Terminal norm key is `model.language_model.norm.weight`. Its learned vector
  is an **offset**: normalize in FP32 as
  `x * rsqrt(mean(x**2, dim=-1) + 1e-6)`, multiply by
  `(1 + weight.float())` in FP32, then cast back to the input activation dtype.
  Loading this vector directly as the multiplicative RMSNorm scale is wrong.
- `lm_head.weight` is a bias-free `[248320, 2048]` projection. The checkpoint
  has `tie_word_embeddings=false`; do not alias LM-head and embedding weights.
  The head consumes final-normalized states and returns logits in its compute
  dtype (no unconditional FP32 logit conversion). `logits_to_keep=0` means all
  positions; positive integers select the final positions before projection.
- Standard autoregressive semantics: prefill logit at position `L-1` predicts
  the first generated token at position `L`; the next decode consumes that
  sampled token at position `L`. DynamicCache tracks KV for full attention
  and recurrent convolution/SSM state for linear attention.

The reference helper preserves the packaged readiness computation and cache.
The package CLI has no revision argument and its default `main` cache ref is
absent, so the helper binds both HF loaders and cache revision resolution to
this immutable revision. It makes no TTNN import or device call. The runner's
ordinary torch CPU implementation is expected to report unavailable CUDA-only
linear-attention acceleration; this is an HF CPU fallback, not TT runtime
fallback evidence.
