# Mistral Small 4 119B — prefill bring-up

Working notes for the shared branch `mistral-small4-bringup`. Short on purpose: it records what is
on the branch and what still needs doing, not how to do it.

## Goal

Bring up `mistralai/Mistral-Small-4-119B-2603` prefill by **reusing `ttMLA` and `ttMoE` from
`deepseek_v3_d_p`**, not by writing a new model. Mistral is in-family — DeepSeek-style MLA attention
with identical weight naming, plus GPT-OSS-style MoE routing.

Order (from Marko): chunked MLA → MoE → weight loading → transformer test = "something functional".
**Runner integration comes last** — no block test needs it.

## Environment

```bash
cd <checkout> && export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
./python_env/bin/pytest ...
```

Checkpoint (already downloaded, 113 GB) — point the env var at it, the adapter has no default path:

```bash
export MISTRAL4_HF_MODEL=/data/kmabee/models/Mistral-Small-4-119B-2603
```

Block tests run on the **galaxy** at production shapes, not the QB2.

## Shapes

```
36 layers · hidden 4096 · vocab 131072 · 32 heads · 1M context · bf16/fp8
MLA:  q_lora 1024 · kv_lora 256 · qk_nope 64 · qk_rope 64 · v_head 128 · YaRN factor 128
MoE:  128 routed · top-4 · 1 shared · moe_inter 2048 · n_group 1 · first_k_dense_replace 0
```

Checkpoint is a `Mistral3ForConditionalGeneration` wrapper: the LM sits under `language_model.`
alongside a Pixtral `vision_tower` we ignore for text-only prefill.

## What is already on this branch

Everything lives under `models/demos/deepseek_v3_d_p/` unless noted.

| file | what |
|---|---|
| `reference/mistral_small4_config.py` | dims class + `mistral4_hf_config()`; every field traced to `config.json` |
| `tt/runners/adapters/mistral_small4.py` | `MistralSmall4Adapter(MLAPrefillAdapter)` |
| `tests/test_mla.py` | `test_mistral4_mla`, plus `mistral_small4` in `test_mla_chunked_prefill`'s variant list |
| `models/demos/common/prefill/adapter.py` | `"mistral_small4"` in `ADAPTER_PATHS` *(outside the folder)* |

So the `variant` fixture resolves `mistral_small4` and the MLA tests are addressable. Both MLA tests
pass at Mistral shapes on random weights today — treat that as the regression baseline, not as
proof the model is right.

Two things to know about the adapter as written:

- `supports_pretrained = False`. The pretrained test ids skip with a message rather than failing deep
  in the dequantizer. Flip it in the same change that lands the weight converters below.
- `default_gate_mode = "GPT_DEVICE"`. Reasoning in the adapter docstring; **it is an argument, not a
  test.** See MoE below.

## What still needs doing

**Attention** — MLA is the most Mistral-divergent block.
- Broader chunked coverage: `production-50k+5k`, `deep-*`, `rot-*`, `with_determinism`, `metadata`
- KV cache layout at `kv_lora_rank = 256`
- Wire `reference_attention_cls` on the adapter. Until then `run_model`'s second reference check
  (`test_mla.py`, `run_reference_mla`) returns `None` and **silently does nothing**.

**MoE** — the largest remaining unknown.
- Gate/routing at 128 experts, top-4, no correction bias
- Dispatch + combine, routed + shared experts
- **Assert per-expert token counts.** A routing bug that collapses onto one expert still produces
  plausible output and passes a loose PCC.

**Embeddings + LM head**
- `tests/pcc/test_parallel_embedding.py` and `tests/pcc/test_lm_head.py` at Mistral vocab/emb
- LM head in both column- and row-parallel modes
- These two are the per-stage probes we bisect with when full-model PCC is wrong

**Weights**
- state_dict mapping (`language_model.` prefix; `detect_language_model_prefix` already handles it)
- fp8 dequant — Mistral is **per-tensor**, not the `[128,128]` block scheme
- Expert tensor layout — stacked with gate/up **fused**
- TTNN weight cache for the chosen mesh

**Integration**
- `test_prefill_block_chunked`, then the transformer test
- Adapter is done; runner + pipeline (`common/prefill/docs/ADDING_A_PREFILL_MODEL.md`) last

## Mistral-specific facts worth not rediscovering

Each of these was checked against the code or the checkpoint.

- **fp8 is per-tensor.** `weight_block_size` is `null`; dense weights carry a rank-0 scalar
  `*_scale_inv`, stacked expert tensors carry `[128, 1, 1]`. The shared dequantizer asserts
  `tensor.ndim == inv_scale.ndim` and a matching `block_shape` rank, so it **raises** on both — a
  loud failure, not silent corruption.
- **Experts are stacked and fused.** `mlp.experts.gate_up_proj` is `[128, 4096, 4096]`, matching
  neither `experts.{i}.*` nor `experts_stacked.*`. transformers 5.12 ships `mistral4` natively, and
  `modeling_mistral4.py` declares it `[num_experts, 2*intermediate_dim, hidden_dim]` consumed with
  `.chunk(2, dim=-1)` — so gate is the first half of the output dim, up the second. Contiguous, not
  interleaved.
- **Router is softmax affinity with no correction bias.** The grouped-topk kernel implements only
  `sigmoid` and `sqrtsoftplus`, so that path is not usable as-is. `n_group = 1` is *not* unusual —
  Kimi, GLM and V4-Flash are ungrouped too; only DeepSeek-V3 uses 8 groups. Note
  `TtMoEGatePrefill.check_cache_complete` requires an `e_score_correction_bias` cache entry that
  Mistral has no weight for.
- **rope: `rope_parameters` → `rope_scaling`, and `rope_theta` must be hoisted out of it.** The
  config builder does both. `original_max_position_embeddings` stays at the checkpoint's 8192 (the
  pre-extension length) — YaRN's frequency ramp is computed against it, so substituting `max_seq`
  changes the rope. GLM's builder does substitute, but only because GLM's `factor` is 1.0 and the
  value is inert there.
- **YaRN is live** at `factor = 128`, so the attention softmax scale is not the bare
  `qk_head_dim**-0.5`. `config.json` exposes `llama_4_scaling_beta: 0.1`, which is the same constant
  `mla.py` hardcodes — they agree today, but nothing reads the field.
- **`kv_lora_rank = 256`** is unprecedented here (family uses 512). It makes the packed-FP8 KV
  cache's rope offset 264 bytes, which is not 16-byte aligned and fails `validate_scaled()`. It does
  **not** affect the MLA tests, which pass the tiled format explicitly. It binds at serving, where 1M
  context likely wants the packed format. Smallest fix looks like 8 bytes of padding (264 → 272);
  worth taking to the MLA owners as a proposal.
- **`first_k_dense_replace = 0`** — every layer is MoE. DeepSeek-V3 and GLM have 3 dense layers,
  Kimi 1. This is the first resident with none.
- **`embed_tokens` and `lm_head` are unquantized bf16** with no scale tensors — the only large
  matmul weights that are. Norms and the router gate are unquantized too; everything else is fp8.

## Gotchas

- **A mesh mismatch is a SKIP, not a failure.** Several block tests are parametrised for meshes
  smaller than 32 and skip on a galaxy — `N skipped` with exit code 0 reads like success. Check the
  passed/skipped counts, not the exit status. `test_mistral4_mla`'s `(8,1)` id does this too: it
  needs an 8-chip carve via `TT_VISIBLE_DEVICES` and skips otherwise.
- **`--collect-only -q -k ...` before running.** `test_mla.py` collects thousands of cases; confirm
  you selected the number you meant.
- **Point cache env vars somewhere writable in `$HOME`.** The shared `/mnt/models/...` tree is
  read-only; a cache *write* fails with `errno=13` from `serialization.cpp:74`. Reads from a complete
  cache are fine.
- **The cache path is keyed on `ttnn.get_num_devices()`**, not the mesh shape
  (`{name}_{arch}_{N}dev/{sp}x{tp}`). Carving to 8 visible devices changes `N` from 32 to 8, so
  nothing cached at 32 is found.
- **After any hang, `tt-smi -r` before the next run.** A wedged ethernet core makes the *next*
  person's run fail with a misleading timeout instead of their real error.

## Open

- `(8,1)` vs `(4,2)` for a PP=4 stage on one galaxy — compute vs CCL. Block correctness is
  split-independent, so this blocks nothing here.
- KV cache dtype/layout for the prefill↔decode ABI, given `kv_lora_rank = 256`.
- Vision scope — is text-only acceptable for v1?
- No Mistral golden trace exists. `test_prefill_transformer_chunked` cannot run without one, and
  every other resident's lives under `/mnt/models/deepseek-prefill-cache/golden/`. Capturing one is
  wall-clock work that nothing else shortens.

## Contacts

MLA — Iva Potkonjak, Pavle Popovic · MoE — Danilo Djekic, Kosta Grujcic · runner — Jaksa Jovicic
