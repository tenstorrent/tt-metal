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

## Components (5 independently-testable units)

### 1. `models/demos/qwen3_6_galaxy_v2/tt/generator_vllm.py` — rewrite

- `initialize_vllm_text_transformer(...)`: build the **qwen36** model args
  object (the exact config the demo constructs, with `is_qwen36=True` /
  full-grid), then construct the **local** `qwen3_6_galaxy_v2`
  `TtTransformer`. Drop all `llama3_70b_galaxy` model/config imports.
- `allocate_vllm_kv_cache(...)`: match qwen36 KV layout — `n_kv_heads = 4`,
  bf8 default, honor the `QWEN36_KV_BF8` env flag (already in the codebase).
- Text serving class named to match the HF arch:
  `class Qwen3_5ForConditionalGeneration(WarmupForwardMixin, Generator)`
  with `model_capabilities` mirroring the parent (prefix caching /
  async-decode flags as currently supported). `prefill_forward` /
  `decode_forward` inherited from `Generator`; decode trace **enabled**
  (works e2e per confirmation).
- **Interface:** `initialize_vllm_model` / `allocate_kv_cache` classmethods
  per the tt-inference-server SKILL text-only pattern.
- **Test:** import + `initialize_vllm_model` construction smoke (per-device
  DRAM footprint sane; ShardTensor2dMesh weights, not replicated — per
  CLAUDE.md galaxy rule).

### 2. `tt-vllm-plugin/tt_vllm_plugin/__init__.py` — add registration

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
