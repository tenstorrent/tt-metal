# FLUX.2-klein-9B text encoder — end-to-end TTNN pipeline

The `text_encoder` of [`black-forest-labs/FLUX.2-klein-9B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B):
a **`Qwen3ForCausalLM`** — 36 decoder layers, hidden 4096, 32 query / 8 KV heads,
head_dim 128, SwiGLU intermediate 12288, vocab 151936, RoPE θ=1e6.

Brought up on **T3K (8 × wormhole_b0), TP=8, mesh 1×8** — the topology the
components graduated at.

## What runs here

| | Call 1 — `text_generation` | Call 2 — `prompt_encoding` |
|---|---|---|
| task | text → text (autoregressive causal LM) | text → prompt embedding |
| why | `config.json` `architectures == ["Qwen3ForCausalLM"]` → the AutoModel registry entry is `AutoModelForCausalLM`, whose reference behaviour is `model.generate()` | `model_index.json` declares `text_encoder: Qwen3ForCausalLM` — `Flux2Transformer2DModel` consumes this model's hidden states, not its logits |
| input | `Qwen2TokenizerFast` (the repo's own `tokenizer/` folder) → `input_ids [1, S]` | same tokenizer |
| output | generated token ids → decoded text | `prompt_embeds [1, S, 4096]` |
| golden | `model.generate(..., do_sample=False)` for the reference text, plus per-step logits scored on the contexts the TT loop actually decoded (see below) | `model.model(input_ids).last_hidden_state` |
| stages | prefill + decode | prefill |

Both calls run **one shared chain** in `tt/pipeline.py`. The demos and the e2e
tests import that same function, so a green test and a working demo are the same
fact.

## The chain

```
input_ids ──▶ token_embed ──▶ rotary_embedding ──▶ encoder_stack ─┬──▶ prompt_embeds   (Call 2)
              (TtTokenEmbed)  (TtRotaryEmbedding)  36 × decoder_layer / layer
                                                    ├── attention (+ r_m_s_norm ×2)
                                                    └── mlp / m_l_p
                                                                  └──▶ decoder_head ──▶ logits
                                                                       └──▶ ttnn.argmax ──┐  (Call 1)
                                                                         ▲                │
                                                                         └── decode loop ─┘
```

Every arrow is a TTNN tensor. No reference tensor is injected at any joint: each
decode step's input token is the previous step's own on-device `ttnn.argmax`.

## Graduated modules — all 10 routed, all 10 invoked

| module | body | placement at TP=8 |
|---|---|---|
| `token_embed` | `TtTokenEmbed` | replicated lookup table |
| `r_m_s_norm` | `TtRMSNorm` | replicated gamma; **is** the per-head q/k norm inside attention |
| `rotary_embedding` | `TtRotaryEmbedding` | replicated `inv_freq`; builds (cos, sin) **on device** |
| `attention` | `TtAttention` | q/k/v column-parallel (4 q + 1 kv head per chip), o_proj row-parallel + `all_reduce` |
| `m_l_p` / `mlp` | `TtMLP` (one body, two discovery names) | gate/up column-parallel (1536/chip), down row-parallel + `all_reduce` |
| `decoder_layer` / `layer` | `TtDecoderLayer` (one body, two discovery names) | replicated residual stream, one collective per sublayer |
| `encoder_stack` | `TtEncoderStack` | chains replicated-in/replicated-out layers, no inter-layer collective |
| `decoder_head` | `TtDecoderHead` | column-parallel over vocab + `all_gather`, 151936 → 152064 zero-pad |

`mlp`/`m_l_p` and `layer`/`decoder_layer` are two discovery passes' names for one
module of the checkpoint (their `_captured/*/manifest.json` record the same
`submodule_path`). Routing each to a private copy would create drift, not
coverage, so each pair shares one body — and the e2e test asserts that identity
so neither name can be silently skipped.

### Wiring fixes this bring-up required

Composing the stubs naively would have **wasted three graduated bodies**, so the
following additive, sharding-preserving edits were made in `_stubs/`:

* `decoder_layer` carried an **inline copy** of the SwiGLU TP scheme; it now calls
  the graduated `TtMLP`, so `m_l_p`/`mlp` are actually in the forward path.
* `attention` inlined `ttnn.rms_norm` for q/k norm; it now calls the graduated
  `TtRMSNorm` — which is exactly what `r_m_s_norm` is bound to
  (`model.layers.0.self_attn.q_norm`).
* `encoder_stack` built RoPE tables on the **host** with `torch.outer`, and
  `attention` re-uploaded them per call; both now accept device-resident
  `(cos, sin)` from the graduated `TtRotaryEmbedding`. This is also what makes the
  forward host-op-free.
* the causal mask is now SDPA's own `is_causal=True` instead of a host-built bias.
* `attention` gained an **additive** decode path (resident KV cache indexed by this
  chip's own kv heads, `paged_update_cache` + `scaled_dot_product_attention_decode`).
  Prefill is untouched.

Every per-component PCC test still passes after these edits — see below.

## Results

| gate | result |
|---|---|
| **Gate 1** — routed stubs still real ttnn, sharded bodies still sharded | PASS |
| **Gate 2** — all 10 graduated modules invoked in the real forward | PASS |
| **Gate 3** — final output PCC vs HF golden ≥ 0.95 | PASS |

Measured at TP=8, full 36 layers, `E2E_GATE_MAX_NEW_TOKENS = 40`:

| number | value |
|---|---|
| Call 2 — `prompt_embeds` vs `last_hidden_state` | **0.999797** |
| Call 1 — 40 decode steps of logits, stacked | **0.999881** |
| Call 1 — worst *single* decode step (what Gate 3 asserts) | **0.999569** |
| same-context top-1 agreement (TT argmax vs reference argmax) | 39 / 40 |
| free-running greedy agreement (TT vs `model.generate`) | 25 / 40 |

### How Call 1 is scored

The TT side is the real free-running chain: every step's input token is the
previous step's own on-device `ttnn.argmax`, nothing is injected. The **reference**
is then evaluated on the *same contexts that chain actually decoded* —
step `i` of both sides conditions on `prompt + tt_tokens[:i]`
(`model_ref.hf_reference_step_logits`, one causal forward, exact because the model
is causal).

That alignment is what makes the number a measurement of *this port*. bf16 weights
against an fp32 reference put the two greedy runs at ~0.9996 per step, which is
enough for one near-tie to flip a token — here at step 25, on `weathered` vs
`red` — after which the two runs are answering different questions and their
logits are no longer comparable at all. Both facts are reported by the test: the
last two rows above are printed every run, and neither is hidden behind the PCC.

<!-- RESULTS -->

### Per-component PCC after the wiring edits (regression check, TP=8)

| component | PCC | | component | PCC |
|---|---|---|---|---|
| `token_embed` | 1.0 | | `decoder_layer` | 0.999855 |
| `rotary_embedding` | 1.0 | | `layer` | 0.999855 |
| `r_m_s_norm` | 0.999999 | | `decoder_head` | 0.999998 |
| `m_l_p` | 0.999996 | | `encoder_stack` (36 layers) | 0.997095 |
| `mlp` | 0.999996 | | `attention` | 0.999937 |

## Running it

```bash
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=wormhole_b0

# Call 1 — text -> text
python -m models.demos.flux_2_klein_9b_text_encoder.demo.demo_text_generation \
    --prompt "A photograph of a rusted lighthouse at dawn" --compare-hf

# Call 2 — text -> prompt embedding for the diffusion transformer
python -m models.demos.flux_2_klein_9b_text_encoder.demo.demo_prompt_encoding \
    --prompt "A photograph of a rusted lighthouse at dawn" --compare-hf

# the gates
./python_env/bin/python -m pytest \
    models/demos/flux_2_klein_9b_text_encoder/tests/e2e/test_e2e_pipeline.py -s

# the trace contract + fully-on-device check
./python_env/bin/python -m pytest \
    models/demos/flux_2_klein_9b_text_encoder/tests/e2e/test_trace_contract.py -s
```

Both demos take `--layers N` to cap the decoder depth and `--tp N` to change the
mesh width.

### Decode horizon

`generation_config.json` states **no** `max_new_tokens` and no `max_length`, so the
horizon is **stop-token driven**: decoding ends when the model emits one of its own
end tokens (`eos_token_id = [151645, 151643]`). `E2E_GATE_MAX_NEW_TOKENS = 40` in
`tt/model_ref.py` is a **safety bound** on a non-terminating run, applied
identically to the TT loop and to `model.generate()`, never to one side only.

## Trace contract (perf-engine seam)

`PIPELINE_STAGES = ["prefill", "decode"]`, derived from Source A
(`Qwen3ForCausalLM`, not encoder-decoder → no `encode`; no speech output → no
`vocode`). Each stage exposes, on the object `build_pipeline` returns:

| hook | prefill | decode |
|---|---|---|
| `<stage>_trace_setup(inputs)` | pins the sequence axis to `C` (default 128), pre-uploads padded ids + (cos, sin) taken from the reference's own `rotary_emb` | seeds the resident per-layer KV cache, pins one position, pre-uploads its (cos, sin) |
| `<stage>_trace_step()` | one host-op-free forward over persistent buffers | one token step; reads the resident KV, never recomputes it, never advances the cursor |
| `<stage>_trace_inputs()` | zero-arg; assembles the same golden inputs the demo/e2e use (position convention read from `_captured/decoder_layer`) | same |
| `<stage>_trace_items()` | `batch × C` — a prefill retires the whole pinned window | `batch` — one token per call |

`trace_capture_selftest(device)` captures, executes, PCC-checks and **releases**
one trace per stage (stage traces never co-reside), shrinking `C` and printing the
fallback if the trace region overflows. `host_op_selftest()` runs the model math
under `host_op_observer.observe_host_ops()` with tokenization and the one-time
weight build outside the observed region, and reports a per-task verdict. Measured:
`prefill` and `decode` both capture and replay at PCC 1.0, and both task heads fire
**zero** host aten ops.

Both also exist as **zero-arg module-level** functions in `tt/pipeline.py`, for the
observers that run outside pytest:

```bash
python -m models.demos.flux_2_klein_9b_text_encoder.tt.pipeline   # runs both
```

`tt/` never opens a device — the pipeline runs on whatever `build_pipeline` is
handed, and under pytest that is the `mesh_device` fixture, the sole opener. The
one device owner outside pytest is `selftest.py::own_a_mesh`, which both demos and
both module-level self-tests borrow, so there is a single place where the
standalone mesh parameters are kept equal to the fixture's.

### Depth knobs

`build_pipeline(device, model=None, layers=None, prefill_layers=None, decode_layers=None, **kwargs)`
returns the pipeline **object** (it never runs it) and accepts/ignores demo kwargs.

* `layers` — default depth for every repeated block; `None` means all 36, never 0.
* `prefill_layers` / `decode_layers` — the per-stage overrides named after
  `PIPELINE_STAGES`. This model has **one** repeated stack (`model.layers`) that
  both stages run — prefill fills the KV cache decode then reads — so the two
  overrides address the same list and a genuine conflict is **refused** rather
  than silently collapsed to one value.
* `TT_PERF_LAYERS` is honoured when `layers` is not passed.

A capped build stays a model: embeddings, the RoPE table, the final norm and the
LM head are all still there, so it exercises every distinct op the full model
runs — just fewer times. The HF reference stays reachable as `pipeline.hf_model`,
which is the ground truth for how many sections the model has and how deep each is.

## Layout

```
models/demos/flux_2_klein_9b_text_encoder/
  e2e_plan.json                    the plan this package was built from
  tt/pipeline.py                   THE shared chain + trace contract + selftests
  tt/model_ref.py                  tokenizer, HF reference, goldens, PCC
  selftest.py                      the mesh every non-pytest entrypoint borrows
  demo/demo_text_generation.py     Call 1 entrypoint (__main__ + argparse)
  demo/demo_prompt_encoding.py     Call 2 entrypoint (__main__ + argparse)
  tests/e2e/test_e2e_pipeline.py   Gates 1 / 2 / 3
  tests/e2e/test_trace_contract.py Command-3 contract + host-op verdict
```

The graduated stubs themselves stay where bring-up put them, under
`models/tt_transformers/demo/flux_2_klein_9b_text_encoder/_stubs/`.
