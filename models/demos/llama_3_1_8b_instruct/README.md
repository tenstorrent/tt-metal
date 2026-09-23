# Llama-3.1-8B-Instruct on TTNN — QB2 (4x Blackhole), TP=4 x DP=1

End-to-end TTNN bring-up of `/home/ttuser/benchmark-data/Llama-3.1-8B-Instruct`, composed
from the graduated bring-up stubs in `_stubs/`.

```
models/demos/llama_3_1_8b_instruct/
  tt/pipeline.py                     the ONE shared chained forward pass
  demo/demo_text_generation.py       Call 1 runnable demo (argparse + __main__)
  tests/e2e/test_e2e_pipeline.py     Gates 1 / 2 / 3
  tests/e2e/test_trace_contract.py   per-stage trace contract + fully-on-device check
  _stubs/                            the graduated TTNN component bodies
  _captured/                         HF golden tensors (per-component + e2e)
  e2e_plan.json                      the plan this package was built from
```

The demo and the test import and call the **same** function,
`tt.pipeline.run_text_generation` — there is exactly one copy of the wiring, so a
green test is a working demo.

## Call 1 — `text_generation` (text -> text)

`config.json` declares `architectures: ["LlamaForCausalLM"]`; it is not an
encoder-decoder and has no sub-configs, so the checkpoint exposes exactly **one**
task head with trained weights and the pipeline has two stages,
`PIPELINE_STAGES = ["prefill", "decode"]`. (The `llama` entry in the AutoModel
registry also accepts sequence/token-classification and QA heads, but this
checkpoint ships no weights for them, so they have no golden to be compared
against and are not valid task heads here.)

**Input** is built with the real HF tokenizer via the instruct chat template.
**Output** is the assistant's decoded text. **Reference** is
`LlamaForCausalLM.generate()` run greedily on CPU float32.

The chain — every stage consumes the previous TT stage's real output; no reference
tensor is ever injected at a joint:

```
input_ids  ->  ttnn.embedding
           ->  rotary_embedding   GRADUATED   (cos, sin)
           ->  decoder_layer x32  GRADUATED   each invoking
                   attention      GRADUATED   TP=4 shard + all_reduce
                   m_l_p          GRADUATED   TP=4 shard + all_reduce
           ->  r_m_s_norm         GRADUATED   the model's final norm
           ->  ttnn.linear lm_head
           ->  ttnn.argmax  (greedy, ON DEVICE; the id feeds ttnn.embedding directly)
```

`ttnn.embedding` and the `lm_head` matmul are authored here — no graduated stub
covers them — and, per the placement guidance, are replicated across the mesh.

### Decode horizon

Decoding is **stop-token driven**: it ends when the sampled id is in
`generation_config.eos_token_id` (`128001 / 128008 / 128009`). `max_new_tokens` is
the safety cap that bounds a non-terminating run and is handed **identically** to
`model.generate()`, so both sides terminate by the same rule over the same length.
The on-device gate uses `GATE_MAX_NEW_TOKENS = 40` to stay fast; the demo defaults
to 64 and takes `--max-new-tokens`.

### Parallelism

TP=4 x DP=1 on `ttnn.MeshShape(1, 4)` (`FabricConfig.FABRIC_1D` enabled before the
mesh is opened). The graduated bodies are composed **as-is**, with the sharding
they graduated with:

| component | scheme | check against this config | collective |
|---|---|---|---|
| `attention` | column-parallel q/k/v by head, row-parallel `o_proj` | `32 % 4 == 0`, `8 % 4 == 0`; chip *d* owns q-heads `[8d,8d+8)` whose kv index is exactly its kv-heads `[2d,2d+2)` | `all_reduce_async`, cluster axis 1 |
| `m_l_p` | column-parallel gate/up, row-parallel `down` | `14336 % 4 == 0` -> 3584/chip | `all_reduce_async`, cluster axis 1 |
| `decoder_layer` | composite; inherits attention's TP degree | both children agree on axis 1 | 2 per layer |
| `r_m_s_norm` | replicate-only (its own graduated scheme — a per-element scale has no axis to split) | — | none |
| `rotary_embedding` | replicate-only (per-position table) | — | none |

Resident bf16 weights are **~5.7 GB per chip** (embed 1.05 + lm_head 1.05 + 32 x
218M params / TP4 = 3.5), against the **registered 28.6 GB usable per chip once a
CCL axis is in play** (32 GB DRAM/chip, 4 chips, 128 GB aggregate). That 5.7 GB is
an estimate from parameter counts — this run did not measure it. The full 32-layer
model is therefore resident; the `layers` cap exists only to make profiling cheap.

## Results

Measured 2026-09-09 on this box (4x Blackhole p300c, mesh 1x4, TP=4 x DP=1), full
32 layers, greedy, prompt `"What is the capital of France? Answer in one short
sentence."`:

| gate | result |
|---|---|
| Gate 1 — routed stubs still real ttnn, TP bodies still shard + collect | **PASS** |
| Gate 2 — all 5 graduated modules invoked in the real forward | **PASS** (`attention`, `decoder_layer`, `m_l_p`, `r_m_s_norm`, `rotary_embedding`) |
| Gate 3 — `e2e PCC` vs the HF golden | **PASS — FINAL_PCC = 0.99865** |

```
TT  output : 'The capital of France is Paris.'
HF  golden : 'The capital of France is Paris.'
token match: 8/8 (100%)     stop: eos 128009 after 8 tokens (cap 40 on BOTH sides)
prefill parity : PCC=0.998650 over 48 prompt positions x 128256 logits
decode  parity : PCC=0.999752 over 8 common-prefix steps (per-step min 0.999392)
e2e PCC=0.9986500032063949
```

Demo throughput: 84 tokens in 3.3 s (**25.6 tok/s**). All 8 per-component PCC tests
still pass after the stub edits (min PCC 0.99992, single-device and TP=4 sharded).

### How Gate 3 is measured

`FINAL_PCC = min(prefill_parity, decode_parity)`, and both halves are
**injection-free** — no reference tensor and no reference token is ever fed back
into the TT chain; the TT side free-runs on its own output throughout.

* **prefill parity** — TT logits at *every* real prompt position vs the HF forward.
  The whole 32-layer stack and every graduated module, 48 positions x 128256
  logits, no decode involved.
* **decode parity** — free-running per-step logits over the steps whose generated
  **prefix** both sides agree on.

Why not plain full-horizon free-running PCC: greedy decoding is chaotic. Once the
two sides pick differently at a near-tie, every later step is conditioned on
*different text*, so those logits measure the tie-break, not the pipeline. On
`"List three interesting facts about the planet Jupiter."` the two sides are
identical for 13 tokens and then split — TT writes "Jupiter is the Largest…", HF
writes "Largest Planet…" — at a point where **the golden's own top-2 logit gap is
0.196**, i.e. a coin-flip. Full-horizon free-running PCC reads 0.479 there while
prefill parity is 0.998644 and decode parity 0.999505.

The comparable-step set is chosen by the **data**, never by hand, and the run
always prints the divergence step together with the golden's own top-2 gap at it,
so a genuine error cannot hide behind "it was just a tie". A vacuous gate is
blocked too: the test fails if fewer than `min(4, N)` decode steps are comparable.
The full-horizon free-running PCC and the token-match rate are printed alongside as
diagnostics.

## Running

```bash
# Gates 1/2/3 (full 32 layers, on device)
./python_env/bin/python -m pytest models/demos/llama_3_1_8b_instruct/tests/e2e/test_e2e_pipeline.py -s

# the demo
./python_env/bin/python -m models.demos.llama_3_1_8b_instruct.demo.demo_text_generation \
    --prompt "What is the capital of France? Answer in one short sentence." --max-new-tokens 64

# trace contract + fully-on-device check
./python_env/bin/python -m pytest models/demos/llama_3_1_8b_instruct/tests/e2e/test_trace_contract.py -s
```

Environment knobs: `TT_E2E_N` (horizon cap), `TT_E2E_LAYERS` / `TT_PERF_LAYERS`
(decoder depth), `TT_E2E_PROMPT`, `TT_TRACE_LAYERS`, `TT_TRACE_CAPACITY`.

## Trace contract (Command 3)

`tt/pipeline.py` exposes, on the pipeline object returned by `build_pipeline`:

* `PIPELINE_STAGES = ["prefill", "decode"]`
* per stage: `<stage>_trace_setup(inputs)`, `<stage>_trace_step()`,
  `<stage>_trace_inputs()` (zero-arg, assembled from the captured golden input),
  `<stage>_trace_items()` (prefill retires C token positions; decode retires 1)
* the AR decode contract: `decode_prefill` seeds the resident self-attention KV,
  `decode_step` reads it and never recomputes the prefix
* `trace_capture_selftest(device)` — captures, executes and PCC-checks one step per
  stage, releasing each trace before the next; shrinks and **prints** the capacity
  if a capture overflows the trace region
* `host_op_selftest()` — runs the forward under `host_op_observer`, with
  tokenization and the weight build outside the observed region

Measured: both stages capture host-op-free at C=128 (`prefill` retires 128 items,
`decode` retires 1) with traced-vs-eager PCC = 1.000000, verified at both 2 and 32
layers; `host_op_selftest` reports `on_device=True` — zero host aten ops in the
observed forward.

`build_pipeline(device, model=None, layers=None, prefill_layers=None,
decode_layers=None, **kwargs)` constructs and **returns** the resident object. This
model has exactly one repeated stack (the 32-layer text decoder) which both stages
own, so the per-stage overrides are accepted and must agree with each other;
`layers=None` means every layer. Everything outside the stack (embedding, RoPE,
final norm, lm_head) is always built, so a capped build still exercises every
distinct op the full model runs — just fewer times. The stack is a plain Python
list of same-typed `TtLlamaDecoderLayer` objects, and the HF reference stays
reachable as `pipe.hf`.

## Changes made to the graduated stubs

The graduated bodies are composed as-is; the additions are strictly additive and
stay pure ttnn (the `.last_good_*` snapshots are untouched for comparison):

1. **Invocation recording** — one `record("<name>")` line at the top of each
   forward, feeding `tt/_invocation.py`. This is Gate 2's proof: it fires from
   inside the real forward, so a coverage sweep cannot satisfy it.
2. **ttnn RoPE input** — `attention._rot_tables` gained a branch that accepts the
   graduated `rotary_embedding` stub's **ttnn** `(cos, sin)` (device-side reshape,
   broadcast over the head axis) instead of only torch tensors, so the chain never
   round-trips through the host. The tables are position-dependent, so that branch
   deliberately bypasses the shape-keyed cache the torch branch uses.
3. **Resident KV cache** — `attention.allocate_kv_cache()` plus a `mode="decode"`
   forward (`nlp_create_qkv_heads_decode` -> rotary -> `paged_update_cache` ->
   `scaled_dot_product_attention_decode` -> `nlp_concat_heads_decode` ->
   row-parallel WO -> `all_reduce`). Prefill additionally seeds the cache with
   `ttnn.fill_cache`. Same weights, same TP=4 split, same collective.
4. **SDPA decode grid pinned to 8x8** — flash-decode tree-reduces its per-core
   partials and the kernel caps that tree at 6 rounds (64 cores/head); on a
   >64-core part the default grid asks for more cores per head than the reduction
   can fold back (`num_tree_reduction_rounds <= MAX_TREE_REDUCTION_ROUNDS`).

The per-component PCC tests under `tests/pcc/` still exercise the original torch
`position_embeddings` path unchanged.
