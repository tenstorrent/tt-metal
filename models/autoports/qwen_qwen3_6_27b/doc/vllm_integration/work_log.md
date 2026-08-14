# Qwen3.6-27B vLLM integration work log

## Contract

- Model: `Qwen/Qwen3.6-27B`, pinned local revision from `tt/functional_decoder.py`.
- Hardware: four Blackhole p300c devices, TP4 `MeshShape(1, 4)`.
- Context: advertise `262144` as required by `doc/context_contract.json`.
- KV pool: serving reports `2290400` aggregate tokens so vLLM's mandatory
  32-page lookahead produces the physically bounded page-800 allocation of
  `2316000` tokens. This still admits one full-context request.
- Precision: `doc/datatype_sweep/selected_precision_config.json` through
  `Qwen36Model.from_pretrained`, including BFP4 weights/LoFi projection policy,
  BF16 activation and CCL payloads, BFP8 KV/recurrent state, and BFP8/HiFi2 LM head.

## Implementation

- Added `tt/generator_vllm.py` and registered
  `TTQwen3_5ForConditionalGeneration` in the sibling vLLM TT plugin.
- Decode delegates to `Qwen36Generator.setup_token_out_decode` and
  `token_out_decode_step`; the adapter has no sampler implementation, host
  argmax, full-logits readback, or token feedback loop.
- vLLM owns full-attention paged K/V. Constant-size linear-attention conv and
  recurrent state remain model-owned and are passed in the same per-layer cache list.
- Prefix caching is disabled. Async decode and overlap are declared because the
  canonical trace owns token and position advancement between scheduler steps.

## Commands and evidence

- Host contract gate:
  `pytest -q models/autoports/qwen_qwen3_6_27b/tests/test_full_model_public_contract.py models/autoports/qwen_qwen3_6_27b/tests/test_vllm_adapter_contract.py`
  — 10 passed.
- Device health: `timeout 60 tt-smi -ls --local` — four Blackhole p300c chips visible.
- Mesh smoke: TP4 open/close with `FABRIC_1D_RING` — `MESH_SMOKE_OK`.
- Reduced target: four real layers (linear layers 0-2 and full-attention layer 3),
  batch 2, non-aligned prompts 65/63. The pinned-source alignment message was a
  benign Metal info log, confirmed by fresh AutoDebug in `AUTODEBUG.md`.
  Subsequent isolated failures exposed and fixed a host/device page-table
  boundary, terminal-prefill-logit sampler shape, and tuple return handling.
  Final `logs/reduced_target.log` records `REDUCED_PREFILL_OK` and
  `REDUCED_DECODE_OK [12, 220]` with one trace replay and zero token, position,
  page-table refreshes or readbacks.
- Full server startup attempts fixed, in order: obsolete `--plugin-config`
  spelling (now `--additional-config`), Qwen3.5 multimodal registry metadata,
  hybrid-cache page-size rebinding (64 to 800), and vLLM's 16-attention-layer
  cache count versus the model's 64 total layers.
- First complete cache-allocation attempt proved a hard serving-specific limit:
  requested 2,522,400 tokens failed at 4,034,994,496/4,072,341,376 bytes
  allocated per bank. Fresh AutoFix found that datatype capacity used a 4 MB
  trace reservation while serving uses 200 MB per bank. At the measured tiled
  BFP8 cost of 8,704 bytes/token/device, the page-aligned physical pool is
  2,316,000 tokens (2,895 pages). A follow-up startup showed vLLM adds one page
  per sequence, so the model report is reduced by `32 * 800` to land on that
  exact allocation. `max_model_len` remains the contract value 262,144.
- Sampling capture now passes the requested final profile directly to the
  canonical `setup_token_out_decode`; steady replay alone updates common sampler
  parameters. This prevents setup from silently replacing the final profile
  with its greedy default.

## Pending

- Run full `run_vllm_server` sampling, qualitative, primary single-user, and CI
  serving-burst gates; inspect outputs and record metrics/artifacts.
- Independent stage review and local isolated commits.
