# Qwen3.6-27B vLLM integration work log

## Contract

- Model: `Qwen/Qwen3.6-27B`, pinned local revision from `tt/functional_decoder.py`.
- Hardware: four Blackhole p300c devices, TP4 `MeshShape(1, 4)`.
- Context: advertise `262144` as required by `doc/context_contract.json`.
- KV pool: the primary max-num-seqs-1 server reports `1726400` base tokens;
  vLLM's lookahead produces `1727200` allocated tokens (2,159 page-800
  blocks), or 6.58 full-context equivalents. The max-num-seqs-32 CI capacity
  profile allocates `1752000` tokens (2,190 blocks).
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
  — 14 passed, 2 warnings.
- Device health: `timeout 60 tt-smi -ls --local` — four Blackhole p300c chips visible.
- Mesh smoke: TP4 open/close with `FABRIC_1D_RING` — `MESH_SMOKE_OK`.
- Reduced target: four real layers (linear layers 0-2 and full-attention layer 3),
  batch 2, non-aligned prompts 65/63. The pinned-source alignment message was a
  benign Metal info log, confirmed by fresh AutoDebug in `AUTODEBUG.md`.
  Subsequent isolated failures exposed and fixed a host/device page-table
  boundary, terminal-prefill-logit sampler shape, and tuple return handling.
  Final `logs/reduced_target_stale_input.log` records `REDUCED_PREFILL_OK`,
  `REDUCED_DECODE_OK`, `REDUCED_STALE_INPUT_OK`, and `REDUCED_SLOT_REMAP_OK`.
  The second replay deliberately supplies stale token/current-position values
  and an unchanged page table; replay advances while all four host refresh/readback
  counters remain zero.
- Full server startup attempts fixed, in order: obsolete `--plugin-config`
  spelling (now `--additional-config`), Qwen3.5 multimodal registry metadata,
  hybrid-cache page-size rebinding (64 to 800), and vLLM's 16-attention-layer
  cache count versus the model's 64 total layers.
- An early complete cache-allocation attempt found a serving allocation limit:
  requested 2,522,400 tokens failed at 4,034,994,496/4,072,341,376 bytes
  allocated per bank. Fresh AutoFix found that datatype capacity used a 4 MB
  trace reservation while serving uses 200 MB per bank. At the measured tiled
  BFP8 cost of 8,704 bytes/token/device, this initially suggested a larger pool.
  The final large-batch prefill gate showed that inference temporaries, not KV
  alone, set the usable serving point: the restored report is 1,726,400 tokens,
  and vLLM's one-page-per-sequence lookahead yields 1,752,000 allocated tokens
  (2,190 pages). `max_model_len` remains the contract value 262,144.
- Sampling capture now passes the requested final profile directly to the
  canonical `setup_token_out_decode`; steady replay alone updates common sampler
  parameters. This prevents setup from silently replacing the final profile
  with its greedy default.

## Final serving evidence (2026-08-15 UTC)

- Full shared-plugin sampling: 72 passed, 1 skipped in 1,381.77 seconds. The
  host compatibility path originally dropped `slot_remap`; forwarding and
  applying it fixed mixed-parameter and seeded batch-order reproducibility.
- Large-batch prefill initially exhausted fragmented DRAM in the 64-token
  linear recurrent scan. Cache-pool reductions were A/B refuted, and TTNN
  concat output-tensor reuse was refuted by `concat.cpp:272`. The proven fix is
  a 32-token scan chunk plus consistent 32-token mask/selector metadata and
  sequential concat lifetime release. Full cache capacity was restored.
- Primary benchmark at max-num-seqs 1, 128 input / 128 output / 1 request /
  concurrency 1: TTFT P50/P99 4,138.6/4,138.6 ms; mean TPOT 70.7335 ms;
  ITL P50/P99 55.861/57.502 ms; output 9.7545 tok/s; TPOT-derived decode
  14.1376 t/s/u.
- CI burst, 100 input / 100 output / 32 requests: TTFT P50/P99
  165,477/165,478 ms; mean TPOT 280.1 ms; ITL P50/P99 244.0/578.1 ms;
  output throughput 16.78 tok/s. This is secondary evidence, not headline t/s/u.
- Exact 65-token non-aligned prompt completed with four output tokens.
- Direct reduced-target hardware testing passed stale-token, current-position,
  unchanged-page-table, and slot-remap checks through the adapter with trace
  replay enabled and no host refresh/readback loop.
- Six chat-template-correct prompts were run greedy and sampled. All outputs
  were coherent/on-topic with no repetition, gibberish, wrong-language drift,
  or contamination, but the 128-token cap ended during exposed reasoning before
  final answers. See `README.md` for the explicit limitation.
- The final qualitative gate passed. `vllm_chat_prompt_metadata.json` records
  rendered prompts/token IDs and links the datatype-sweep and full-model control
  evidence; `degenerate_output_check.json` and `.log` retain the checker result.
- The final primary command was
  `python -m models.common.readiness_check.run_vllm_server --stages serve,benchmark --model-dir models/autoports/qwen_qwen3_6_27b --hf-model Qwen/Qwen3.6-27B --mesh-device P300x2 --max-num-seqs 1 --max-model-len 262144 --sampling-profile full --no-benchmark-ci-serving --tt-config '{"trace_region_size": 200000000, "fabric_config": "FABRIC_1D_RING"}'`.
- A same-path capacity A/B retained the identical 128/128/1/concurrency-1
  workload, trace/sampling modes, precision, fabric, and context. Max-num-seqs
  32 measured 251.656 ms TPOT / 3.9737 t/s/u; max-num-seqs 1 measured
  70.7335 ms / 14.1376 t/s/u. Device sampling remains padded to 32 rows in
  both arms, so the 3.56x speedup isolates fixed-capacity inactive-slot model
  execution. Artifacts are under `readiness_vllm/capacity_ab/maxseq{1,32}`.
  Primary serving now exceeds the 6.96 t/s/u teacher-forcing lower bound.
- Successful benchmark server and final evidence server both terminated
  cleanly. Process audit found no live vLLM/EngineCore device owner afterward.

## Review and local commits

- Independent `$stage-review` final verdict: `clean-pass`.
- tt-metal stage commit: `6f597ee054c133fe242ea7983a08c5a373d6b902`.
- sibling vLLM plugin commit: `ed7a409b9c56f276acaa9a764c409703e41c8ef0`.
- Commits are local only; nothing was pushed.
