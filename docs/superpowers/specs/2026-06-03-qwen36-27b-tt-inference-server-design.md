# Design: Qwen3.6-27B (text-only) → tt-inference-server on BH Galaxy

**Date:** 2026-06-03
**Author:** ssinghal
**Status:** Draft for review
**Branch:** `ssinghal/qwen36_bhglx`
**Base tt-metal commit:** `fa021238662`

## Goal

Expose the existing `models/demos/qwen3_6_galaxy_v2/` TTNN implementation
(Qwen3.6-27B, **text-only language tower**) through `tt-inference-server`'s
OpenAI-compatible vLLM HTTP API, running on **BH Galaxy** (32× P150,
`DeviceTypes.BLACKHOLE_GALAXY`). This is the full Phase-6 integration:
generator → registration → model spec → Docker/workflow → validation → docs.

## Prerequisites (confirmed)

- Decode **and** trace work end-to-end on BH Galaxy (user-confirmed).
- Demo (`demo/demo_decode.py` / `text_demo_qwen36.py`) produces correct text.
- HF checkpoint: `Qwen/Qwen3.6-27B`. `config.architectures[0]` =
  **`Qwen3_5ForConditionalGeneration`**, `model_type` = `qwen3_5`. The LM tower
  lives under `model.language_model.*`; `load_checkpoints.py` already maps it.
- Model is fundamentally a VLM checkpoint; we serve the **text path only**
  (vision / MTP out of scope).

## Key gap discovered

`models/demos/qwen3_6_galaxy_v2/tt/generator_vllm.py` is currently a
**byte-identical stale copy** of `llama3_70b_galaxy`'s version (only an import
reorder differs). It imports and constructs the *Llama* `TtTransformer` /
`TtModelArgs` — NOT the local qwen3.6 modules. The unused
`TtQwenModelArgs` import is a leftover. The single most important change is
rewiring the generator to construct the model the same way the **demo** does.

## Architecture / device facts

- Target device: `DeviceTypes.BLACKHOLE_GALAXY`, mapped in
  `workflows/device_utils.py` from `(("tt-galaxy-bh", 32),)`.
- Model config (`qwen36_model_config.py`) hard-requires 32 devices (BH GLX
  8×4); `n_kv_heads = 4`; `num_devices_per_group = 4`.
- Parent `llama3_70b_galaxy` is already registered and served on galaxy via
  `release_model_spec.json` + `vllm-tt-metal/multihost_entrypoint.sh`; we
  follow that precedent.
- **max_context: 262144 (256k)** — user-confirmed the model supports 256k.

## Dependency decision: transformers stays at 4.53.0 (NO bump)

Investigated and rejected a transformers bump (the checkpoint's
`config.json` carries `"transformers_version": "4.57.1"`, which only records
the Qwen-internal fork that *wrote* it):

- `model_type: qwen3_5` / `Qwen3_5ForConditionalGeneration` exists in **no
  public transformers release** — verified absent in 4.53.0, **4.57.1**,
  4.57.6, and 5.10.1 (downloaded wheels; no `qwen3_5` in any auto-mapping,
  no `Qwen3_5` class anywhere). There is **no `auto_map` / remote-code** in
  the HF repo, so `trust_remote_code=True` cannot help either.
- Weight loading bypasses transformers entirely (raw safetensors).
- Tokenizer resolves by name (`"tokenizer_class": "Qwen2Tokenizer"`) — fine
  on 4.53.0.
- A bump would also collide with tt-metal's hard `transformers == 4.53.0`
  pin (`tt_metal/python_env/requirements-dev.txt:35`).

The only thing that needs `qwen3_5` is vLLM's `AutoConfig.from_pretrained`
(see Component 2). We solve that with a config registration, not a bump.

## Components (5 units; Component 2 has two parts)

### 1. `models/demos/qwen3_6_galaxy_v2/tt/generator_vllm.py` — rewrite

The file currently has a `QwenForCausalLM` class + a qwen init fn, but every
import/constructor points at parent `llama3_70b_galaxy`. Concrete changes:

- **Imports:** redirect all four parent imports to local v2:
  `qwen3_6_galaxy_v2.tt.generator.Generator`,
  `qwen3_6_galaxy_v2.tt.llama_model.TtTransformer`,
  `qwen3_6_galaxy_v2.tt.model_config.LlamaOptimizations`,
  `qwen3_6_galaxy_v2.tt.qwen36_model_config.TtQwen36ModelArgs`. Drop the
  parent `qwen_model_config.TtQwenModelArgs` import.
- **Model args:** construct `TtQwen36ModelArgs(submesh, instruct=...,
  max_batch_size=..., max_seq_len=...)`. `linear_attention_pattern`
  auto-loads from HF `config.layer_types` in `__init__`; RoPE is built and
  threaded internally by the local `TtTransformer` — **no manual wiring**
  (the demo's `_build_partial_rope_cos_sin_tt` is eager-probe scaffolding,
  not needed here).
- **Weight load (the verified risk):** do **NOT** call
  `args.load_state_dict()` — its `from_hf_url=True` branch runs
  `AutoModelForCausalLM.from_pretrained`, which raises *"Unrecognized
  configuration class"* on this VLM checkpoint (confirmed). Mirror the demo
  instead: load raw safetensors from the snapshot dir (resolve via
  `huggingface_hub.snapshot_download` / `model.safetensors.index.json`) and
  pass the raw HF `state_dict` (`model.language_model.*` keys) straight into
  `TtTransformer`. (Equivalently, set `args.from_hf_url = False` so
  `load_state_dict()` takes the `load_hf_state_dict` + qwen36-remap path —
  but the CKPT_DIR must then be a local dir, not a hub id.)
- **TtTransformer call:** `dtype=ttnn.bfloat8_b`, `use_paged_kv_cache=True`,
  `mode="prefill"`, **drop `enable_prefetcher_performance_mode=True`**
  (prefetcher permanently off in v2 → pass `False`/omit).
- `allocate_vllm_kv_cache(...)`: match qwen36 KV layout — `n_kv_heads = 8`
  (padded from 4), `head_dim = 256`; honor the `QWEN36_KV_BF8` env flag for
  dtype instead of hard-coding bf8.
- **Serving class:** named to match the HF arch
  (`Qwen3_5ForConditionalGeneration(WarmupForwardMixin, Generator)`),
  subclassing the **local** `Generator`. `initialize_vllm_model` default
  `max_seq_len=262144`. `prefill_forward`/`decode_forward`/`allocate_kv_cache`/
  `cache_path` delegate as in the existing class. Set `model_capabilities`
  to what's actually validated, not the inherited Llama defaults. Decode
  trace **enabled** (works e2e per confirmation).
- **Delete the dead Llama path** (`initialize_vllm_text_transformer`,
  `LlamaForCausalLM`, unused `input_processor_*`).
- **Test:** import + `initialize_vllm_model` construction smoke (per-device
  DRAM footprint sane; ShardTensor2dMesh weights, not replicated — per
  CLAUDE.md galaxy rule).

### 2. `tt-vllm-plugin` — register the model AND the `qwen3_5` config

**2a. Config registration (new — the real vLLM blocker).** At plugin import,
register `qwen3_5` so vLLM's `AutoConfig.from_pretrained` can parse the
checkpoint:
```python
from transformers import AutoConfig
AutoConfig.register("qwen3_5", Qwen3_5Config)  # thin config exposing the
# fields vLLM plumbing needs (num_hidden_layers, hidden_size,
# num_attention_heads, max_position_embeddings, vocab_size, …) sourced from
# the checkpoint's text_config. The TT model still reads config.json itself.
```
Open detail for the plan: register a thin flat config built from `text_config`
vs. reuse a `Qwen3VL`-style text-config extraction. Architecture name stays
`Qwen3_5ForConditionalGeneration` so the platform "TT" prefix resolves to our
class.

**2b. Model registration** in `__init__.py`:

```python
ModelRegistry.register_model(
    "TTQwen3_5ForConditionalGeneration",
    "models.demos.qwen3_6_galaxy_v2.tt.generator_vllm:Qwen3_5ForConditionalGeneration",
)
```
Wrapped in try/except + log, like the BGE / Qwen3-Embedding entries.
**No collision** with the existing `TTQwen3ForCausalLM` (embedding) registration.

### 3. `workflows/model_spec.py` + `release_model_spec.json` — add entry

- `hf_model_repo = "Qwen/Qwen3.6-27B"`, `code_path = models/demos/qwen3_6_galaxy_v2`.
- `DeviceModelSpec(device=BLACKHOLE_GALAXY, model_type=LLM, max_context=262144,
  max_concurrency=<from demo batch>, default_impl=True,
  vllm_args={"trust_remote_code": True}, has_builtin_warmup=True)`.
- `tt_metal_commit` = branch HEAD at integration time.
- `status = EXPERIMENTAL`, `supported_modalities = ["text"]`.

### 4. Docker / run workflow — verify, minimal change

- Confirm `vllm-tt-metal/multihost_entrypoint.sh` + the multihost Dockerfile
  used for the parent galaxy model also covers BH-galaxy (32-chip). Add a
  `run.py` model-id mapping entry if required. No new Dockerfile expected.

### 5. Validation & docs

- Start: `python run.py --model Qwen3.6-27B --workflow server --local-server
  --tt-device tt-galaxy-bh --skip-system-sw-validation`.
- Smoke: `curl .../v1/chat/completions` → sensible output.
- Accuracy: server ≥ demo within 2–3 pp on a matched prompt set.
- Perf/eval: run benchmark + eval workflows; document latency vs reference.
- `tt-smi -r` not needed between requests.
- Update `BRINGUP_LOG.md` (server results, V2-10) and model `README.md`
  (server start/test/benchmark commands).

## Approach decision

**Approach A (chosen): thin local wrapper mirroring the demo's construction.**
Point the generator at `qwen3_6_galaxy_v2` modules; drop llama imports.

Rejected — **Approach B:** keep importing the parent's classes and plumb an
`is_qwen36` flag through the parent generator. More parent coupling; risks the
parent config diverging from the demo's actual config, which would pass a
single-layer PCC test while being wrong at full depth (CLAUDE.md warning).

## Out of scope (YAGNI)

- Vision / multimodal path (`qwen36_mm_*`, `vision_*`).
- MTP speculative decoding.
- Any non-BH-galaxy device target.

## Risks

- **`qwen3_5` config registration must satisfy vLLM plumbing.** The thin
  config must surface the right text params (layers/heads/dim/max_pos/vocab)
  for scheduler + KV-cache sizing; getting these from `text_config` wrong
  would misconfigure paging. Validate the registered config against the
  checkpoint at server start. (The `AutoModelForCausalLM` weight-load risk is
  now *resolved* — generator uses the raw-safetensors loader; see Component 1.)
- vLLM may try to treat a `*ForConditionalGeneration` arch as multimodal.
  Mitigation: our registered text class is a plain `Generator` subclass (no
  `SupportsMultiModal`); text-only requests carry no media. Verify vLLM
  dispatch picks our class and does not require an mm processor.
- 256k context server stability is a separate bar from demo coherence;
  flagged for the validation step (stress at long context).
- Galaxy weight sharding correctness: enforce `ShardTensor2dMesh` per CLAUDE.md
  and verify per-device DRAM footprint at init with the full 64-layer load.

## Implementation note

Per project CLAUDE.md, Phase 6 is executed via the `/tt-inference-server`
skill; the implementation plan will follow that skill's steps.
