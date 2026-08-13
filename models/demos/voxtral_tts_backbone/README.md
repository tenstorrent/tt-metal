# Voxtral TTS backbone — end-to-end TTNN pipeline

`/localdev/lserbedzija/hf_models/voxtral-tts-backbone` on Blackhole, single chip.

Despite the `tts-backbone` name the checkpoint registers exactly **one** task
head: causal language modelling. `architectures: [MistralForCausalLM]`,
`is_encoder_decoder: false`, no audio/codec sub-config, no vocoder — the name
refers to the checkpoint's downstream use, and what is in the file is a pure
text decoder. So the pipeline has the two stages a decoder-only LM has,
`prefill` and `decode`, and there is no vocode stage.

| | |
|---|---|
| hidden / layers / heads | 3072 / 26 / 32 Q, 8 KV, head_dim 128 |
| MLP | SwiGLU, intermediate 9216 |
| vocab | 131072, `tie_word_embeddings: true` |
| RoPE | `rope_theta` 1e6, HF half-split layout |
| stop token | `generation_config.eos_token_id` = 2 |
| topology | chips 1, tp 1, dp 1 — native single-device bodies |

## The two calls

| Call | id | task | TT entrypoint | golden |
|---|---|---|---|---|
| 1 | `text_generation` | prompt → continuation | `run_generate` | `hf.generate(...)` per-step scores |
| 2 | `causal_lm_logits` | prompt → next-token logits at every position | `run_prefill_logits` | `hf(input_ids).logits` |

Call 2 is also Call 1's prefill stage with the head read at every position, so
one `tt/pipeline.py` and one agent own both.

## Measured (this checkout, Blackhole, depth 26)

| metric | value |
|---|---|
| Call 2 PCC (logits, 10 real positions) | **0.99705** |
| Call 1 PCC (48 stacked per-step logits) | **0.99788** |
| Call 1 token-sequence equality vs HF | **48/48, no divergence** |
| Call 2 next-token / top-5 overlap | match / 4 of 5 |
| KV-cache proof: decode@10 vs re-prefill(11) | corr 0.99963, same argmax |
| G1 runtime native probe | `torch_ops=0` (1051 ttnn dispatches prefill, 56798 for the full generate) |
| G2 invocations, Call 1 | decoder_layer/attention/m_l_p 1248, r_m_s_norm 2544, rotary 48 |
| trace capture | both stages captured, replay corr 1.0 (prefill 52.0 ms, decode 33.0 ms per replay) |
| host-op observer | 0 host aten ops in the forward |

Required gate threshold is PCC ≥ 0.95; the per-component suite under
`tests/pcc/` holds its own 0.99 target.

## Layout

```
tt/pipeline.py          THE shared chain: build_pipeline, VoxtralTtsBackbonePipeline
                        (PIPELINE_STAGES, run_prefill_logits, run_generate,
                        decode_prefill/decode_step, per-stage trace hooks,
                        trace_capture_selftest, host_op_selftest) and the HF
                        reference helpers, kept strictly out of the TT path
_stubs/                 the five graduated components the chain composes
demo/demo_text_generation.py     Call 1 entrypoint
demo/demo_causal_lm_logits.py    Call 2 entrypoint
tests/e2e/              G1/G2/G3 + the everything-on-device gate
tests/pcc/              the per-component 0.99 PCC suite
selftest_device.py      opens the device for the ZERO-ARG selftest hooks only
```

The demos contain no wiring of their own — both import `build_pipeline` and a
`run_*` from `tt/pipeline.py`, so a green test is a working demo.

## Running

```bash
cd /localdev/lserbedzija/repos/tt-metal-pr46283
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD

# e2e gates (both calls, native probe, invocation counts, PCC, KV-cache proof)
./python_env/bin/python -m pytest models/demos/voxtral_tts_backbone/tests/e2e -s

# demos
./python_env/bin/python -m models.demos.voxtral_tts_backbone.demo.demo_text_generation \
    --prompt 'The quick brown fox jumps over the lazy dog.' --max-new-tokens 48
./python_env/bin/python -m models.demos.voxtral_tts_backbone.demo.demo_causal_lm_logits \
    --prompt 'The quick brown fox jumps over the lazy dog.'

# per-component PCC suite
./python_env/bin/python -m pytest models/demos/voxtral_tts_backbone/tests/pcc -s
```

`python_env/bin/python` re-publishes the three sys.path entries tt-metal's
editable install would (`<repo>`, `<repo>/ttnn`, `<repo>/tools`): this checkout
was built in place without that install, and with only the repo root on the path
`import ttnn` picks up the same-named source directory as an empty namespace
package. `tt/pipeline.py` repairs the same paths itself, so it stays importable
from a bare subprocess (which is how the trace/host-op probes reach it).

## The chain

```
prefill  tokens[1,S] -> ttnn.embedding
                     -> TtRotaryEmbedding(arange(S))              graduated
                     -> 26 x TtDecoderLayer(additive causal mask, (cos,sin),
                        resident KV, cache_fill=True)              graduated
                        -> TtAttention, TtMLP, TtRMSNorm x2        graduated
                     -> TtRMSNorm (model-level final norm)         graduated
                     -> ttnn.linear(lm_head) -> logits[1,S,131072]

decode   one token per step, reading ONLY resident device tensors:
         token[1,1] -> ttnn.embedding
                    -> TtRotaryEmbedding(resident position)        graduated
                    -> 26 x TtDecoderLayer(resident KV, resident int32 row index)
                    -> TtRMSNorm -> ttnn.linear -> ttnn.argmax
                    -> the argmax token is written back into the [1,1] input
                       buffer and both position buffers advance on device
```

All five graduated modules are inside that chain with their outputs feeding
downstream compute; there is no coverage sweep and nothing is invoked just to
move a counter. `embed_tokens` and `lm_head` are not graduated components (no
stub was emitted for them) so they are authored glue — still pure ttnn
(`ttnn.embedding`, `ttnn.linear` on the tied embedding matrix transposed).

### Why it traces

Every weight and both KV buffers are staged once in `__init__` and stay
resident. A decode step reads only the resident `[1,1]` token buffer, the
`[1,1]` rotary position, the `[1]` int32 cache row index and the per-layer KV
`[1,8,C,128]`; the generated token never round-trips through the host
(`ttnn.argmax` feeds `ttnn.embedding` directly) and both position buffers
advance with `ttnn.add`/`ttnn.copy` in place. Shapes are therefore constant
every step, `ttnn.begin_trace_capture` succeeds for both stages, and replaying
the captured decode step advances the sequence by itself — the cache write goes
through `ttnn.experimental.paged_update_cache`, which takes its row number from
a device tensor rather than a host scalar.

`C` (KV/trace capacity) is fixed at build time, default 128; the correctness
path prefills at `S = prompt_len` padded to a tile and decodes from
`prompt_len`, so the pad rows of the cache are overwritten by real tokens before
they can ever be inside an attention window.

### Decode horizon

The stop rule is the model's own `generation_config.eos_token_id` (= 2) and it
truncates **both** sides: `common_stop_length` cuts the TT sequence and the HF
golden at the first stop token on either side, so the comparison length is
model-grounded and the golden is never forced to a length the TT side invented.
The safety cap is `min(capacity - prompt_len, 48)`; 48 is chosen here for lack
of any model signal — `generation_config` carries neither `max_new_tokens` nor
`max_length`, and `max_position_embeddings` is 128000, which is not a runnable
bound. It is a cap, not the stop rule, it is passed identically to
`generate(max_new_tokens=H)` and to the TT loop, and it is overridable with
`--max-new-tokens` / `TT_E2E_MAX_NEW_TOKENS`.

The eager decode loop runs the whole cap without a host readback, because
breaking early in Python would mean copying each token to the host inside the
step and that is exactly the host op this pipeline must not have. The stop token
is applied to the compared window instead, which is where it changes the result.

### Notes on the graduated bodies

`_stubs/attention.py` and `_stubs/decoder_layer.py` grew OPTIONAL cache kwargs
(`kv_cache`, `cache_fill`, `cache_pos`, `cache_pos_tensor`, all defaulting to
None/False) so that ONE graduated attention body serves prefill and decode.
With those kwargs absent the forward is what the per-component PCC test pinned,
which is why `tests/pcc` still passes unchanged. Nothing else in the five stubs
was touched.
