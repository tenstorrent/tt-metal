# DiffusionGemma vLLM integration — work log (#47466 / #47488)

Stage: SERVING (dg-09). Branch: `diffusion-gemma-function`. Device: QB2 / `bh-qbge-06`
(4× Blackhole, chips 0–3, TP=4, `(1,4)` mesh). RUN-first (degenerate output OK until #48291).

## Summary of what was built

| Artifact | Path | Purpose |
|---|---|---|
| vLLM adapter | `tt/generator_vllm.py` | `DiffusionGemmaForCausalLM(HybridAttentionForCausalLM)` — block-granular bridge |
| Block-emission core (vLLM-free) | `tt/serving.py` | `BlockDiffusionServingSession` — prefill + per-block decode state machine |
| Reduced-surface driver | `demo/serving_smoke.py` | drives prefill→N blocks on device without vLLM; per-block metrics |
| Block-contract tests | `tests/test_serving_block_contract.py` | CPU scaffolding + device block-emission gate |
| Context contract | `doc/context_contract.json` | served max_model_len, datatype policy, block granularity |
| Plugin registration | `doc/vllm_integration/plugin_registration.patch` | `register_tt_models()` diff for the tenstorrent/vllm fork |

## Architecture: how the adapter delegates

The whole denoise loop lives inside the tt-metal model's forward. The adapter is a thin
vLLM interface over the existing `tt/generate.py` engine:

```
prefill_forward(tokens, prompt_lens, ...)         # per request row
  → BlockDiffusionServingSession.prefill()
      → tt.generate.prefill_prompt_tokens()        # write prompt K/V (causal)
      → make_generation_logits_fn_builder_from_checkpoint_state()  # stateful DenoiseLogitsAdapter
  → BlockDiffusionServingSession.decode_block()     # emit block 0 (mirrors AR "first token")
      → tt.generate.denoise_and_commit_block()
          → tt.denoise_loop.denoise_block()          # ≤48 on-device denoise steps:
              → tt.sampling.gumbel_max / token_entropy
              → tt.denoise_loop.entropy_budget_accept (sort/cumsum/scatter) / renoise
          → tt.generate.commit_canvas_tokens()       # append committed clean argmax K/V

decode_forward(...)                                  # per request row, per step
  → BlockDiffusionServingSession.decode_block()      # emit block N (256 tokens)
```

No separate host sampling path, no full-logits readback: only per-step `[B,L]` decision
tensors (argmax/entropy/sampled/accept) are read back for the data-dependent halt; the
`[B,L,vocab]` logits stay on device. This is the reuse of the on-device canvas sampling path
required by the goal.

Low-level `tt/generate.py` methods delegated to: `prefill_prompt_tokens`,
`make_generation_logits_fn_builder_from_checkpoint_state`, `denoise_and_commit_block`
(→ `denoise_block` + `commit_canvas_tokens`), `decode_generation`, `tokenize_prompt`, and the
seeded canvas/noise/Gumbel hooks (`make_seeded_host_canvas_init_fn`,
`make_seeded_host_noise_tokens_fn`, `make_seeded_{host,chunked,}_gumbel_noise_fn`).

## Block-granular emission contract (explicit)

- The adapter emits a **256-token block per decode step** (`canvas_length`), not one token.
- The async-decode contract is redefined **per-BLOCK**: `supports_async_decode`, stale-input
  refresh, and page-table update are once per emitted 256-token block, not per token. We
  declare `supports_async_decode=False` (the per-block async path is unproven without the
  #47488 runner; never advertise async without proof).
- Position advances by `canvas_length` per decode step (`next_pos += 256`).

## The #47488 upstream runner/scheduler change (scoped precisely)

The current tenstorrent/vllm TT runner assumes **one committed token per decode step**.
Verified against the plugin source (`/tmp/ttvllm_check/`, image `0.14.0-80180b9-7678b70`):

| What | Where | Change for a 256-block |
|---|---|---|
| Hard assert one token | `model_runner.py:2471` `assert num_out_tokens == 1` | accept `num_out_tokens == canvas_length` |
| Sampled-id shape | `model_runner.py:2378` (`view(sz,1)`), `:1878-1880` (`shape[1]==1` assert) | allow `[num_reqs, 256]` |
| Runner output build | `model_runner.py:2437,2444` (one-token lists) | emit 256 tokens/req |
| Host position advance | `_apply_sampled_tokens_to_state` `:2479-2489`, `:2508-2518` (`+1`) | advance `num_tokens` by 256 |
| Context bound | same fns, `assert end_idx <= max_model_len` | check `start_idx + 256 <= max_model_len` |
| Decode input read | `model_runner.py:931-934` (`num_tokens-1`, single last token) | correct next-block position |
| Scheduler advance | `scheduler.py` / `_update_states` (`num_computed_tokens` from scheduler) | advance by `canvas_length`/step |

Contract we already respect (verified): phase-based batching (a step is all-prefill OR
all-decode — `scheduler.py:30-100`, `model_runner.py:911-914`); no chunked prefill
(`platform.py:339-341`, re-asserted `model_runner.py:251-252`); speculative decoding
hard-blocked (`platform.py:342-344`); APC force-disabled for sliding-window models
(`platform.py:512-521`); no prompt-length divisibility requirement (`model_runner.py:924-928`).

## Cache ownership (the second half of #47488)

The diffusion denoise-read path reads the frozen prompt prefix from the model-owned
**contiguous** `tt_model.tt_kv_cache` via `ttnn.slice` (`tt/denoise_forward.py`
`read_prompt_kv_cache_by_layer` / `read_prompt_kv_cache_slice`), NOT from a vLLM paged block
pool. Serving therefore runs in the generator/standalone cache-ownership mode: the model owns
its `max_model_len` cache (`create_kv_cache=True`) and is driven with `page_table=None`;
`allocate_kv_cache[_per_layer]` returns those existing handles (no double allocation). One
contiguous cache backs one active sequence. Routing the frozen-prefix read through a vLLM
paged cache + per-request block tables (concurrent batched serving) is #47488 + batched canvas
decode #47557.

## Datatype policy served

bf16 weights + bf16 KV cache + bf16 CCL (self-conditioning softmax/soft-embedding accumulate
fp32). This is the full-model bring-up policy (`create_tt_model` default); the dg-07
datatype-sweep stage has not run, and fp32 experts do not fit QB2 DRAM, so bf16/MoE is the
fastest policy that fits. Matches the gemma4 vLLM bridge. See `doc/context_contract.json`.

## Served context

max_model_len = 262144 (HF `text_config.max_position_embeddings`, = 256 × 1024). No reduction
below the advertised context. Physical-limit note: full-depth 256K weights+KV fit
(29.704 GiB/chip used, 2.163 GiB free); full-vocab Gumbel materialization OOMs at 256K, so
serving uses the chunked / argmax sampler (`DG_VLLM_GUMBEL_MODE`) which fits. See context
contract.

## Serving environment reality — SUPERSEDED 2026-07-03 (fresh host vLLM now live)

The earlier blocker ("vLLM + plugin only in a stale ghcr image; runner asserts one token") is no
longer the state. A fresh, project-matching vLLM was built on host and the #47488 runner change was
applied and verified LIVE. See **§ Live serving verification (fresh vLLM) — 2026-07-03** below. The
stale image (`0.14.0-80180b9-7678b70`) is NOT used; `models/common/readiness_check/run_vllm_server`
still does not exist in this checkout, so the live serve is driven directly against
`vllm.entrypoints.openai.api_server` (the fork's documented server path).

## Device runs (reduced-surface serving driver)

Command:
```
TT_LOGGER_LEVEL=ERROR DG_CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it \
python -m models.experimental.diffusion_gemma.demo.serving_smoke \
  --mesh P150x4 --num-layers 1 --max-seq-len 1024 \
  --num-blocks 2 --canvas-length 256 --max-denoising-steps 2 \
  --gumbel-mode argmax --local-files-only \
  --metrics-json doc/vllm_integration/serving_smoke_reduced.json
```

### Reduced-surface run (1 layer) — inner-loop contract proof (2026-07-03, QB2)

`DG_VLLM_SERVING_SMOKE_SUCCESS prompt_len=24 prompt_aligned_256=False cache_len=32 blocks=2
tokens=512 canvas=256 ttft_s=8.238 mean_block_latency_s=6.894 tokens_per_block_per_s=37.14
final_next_pos=544 text_chars=4652`

- Non-256-aligned prompt (24) served; padded to cache_len=32; **2 × 256-token blocks emitted**;
  position advanced 32 → 544 (cache_len + 2×256). Per-block metrics captured (see
  `serving_smoke_reduced.json`). DRAM: baseline 0.0 → post-build 1.202 → post-prefill+block0
  2.616 GiB/chip.
- Output is degenerate `<unused*>` tokens — **expected for the 1-layer reduced target**; this is
  the block-emission *contract* proof (prefill → committed blocks → detok), not the model's
  output. Reduced target is an inner-loop tool only (per skill), not final accuracy/perf.

### Full-depth run (30 layers) — final block-emission + qualitative evidence (2026-07-03, QB2)

Command: same driver, no `--num-layers` (full 30), prompt `"Hello, how are you?"`,
`--max-seq-len 1024 --num-blocks 2 --max-denoising-steps 4 --gumbel-mode argmax`.
post-build DRAM 13.268 GiB/chip (matches plan's ~13.24 backbone footprint). prompt_len=19
(non-aligned).

`DG_VLLM_SERVING_SMOKE_SUCCESS prompt_len=19 prompt_aligned_256=False cache_len=32 blocks=1
tokens=256 canvas=256 ttft_s=65.980 mean_block_latency_s=64.450 tokens_per_block_per_s=3.97
final_next_pos=288 text_chars=0`

- Full 30-layer block emission works: **256-token block committed**, position advanced 32 → 288.
  Requested 2 blocks but block 0 committed an EOS token → the session's per-block stop check
  halted generation after 1 block (correct serving behavior). Per-block metrics: **TTFT 65.98 s,
  block latency 64.45 s, 3.97 tokens/block/s** at 4 denoise steps (argmax). (Slower than the
  1-layer proof because the per-step 262144-vocab logits + 30-layer forward dominate; this is the
  block-diffusion cost profile, reported per-block, never as `1000/mean_tpot_ms`.)
- `text_chars=0`: the committed block is EOS/special-heavy, so `skip_special_tokens` yields empty
  text — the **expected RUN-first #48291 degeneracy** (bf16/MoE/TP=4 argmax ≈50% vs HF; diffusion
  commits the clean argmax with no temperature cushion), not a serving regression. See the
  visible-text run below and the qualitative verdict.
- Artifacts: `serving_smoke_fulldepth.json`, `/tmp/dg_serving_smoke_fulldepth.log`.

### Full-depth visible-text run (qualitative control comparison)

Same driver, full 30 layers, `--num-blocks 1 --max-denoising-steps 16 --gumbel-mode argmax
--disable-eos-stop` (EOS-stop off so non-EOS positions surface, matching the recorded
`text_demo` visible-dialogue control that produced coherent text at more steps).

`DG_VLLM_SERVING_SMOKE_SUCCESS prompt_len=19 prompt_aligned_256=False cache_len=32 blocks=1
tokens=256 canvas=256 ttft_s=111.611 mean_block_latency_s=110.195 tokens_per_block_per_s=2.32
final_next_pos=288 text_chars=66`

- **Coherent visible output through the serving path:**
  `你好！I'm doing well, thank you for asking. How can I help you today?`
- This matches the recorded `text_demo` RUN control almost exactly (plan.md 2026-07-02:
  `你好！I'm doing well, thank you for asking! How can I help you today?`). ⇒ the serving
  block-emission path is **not a serving regression** — it faithfully reproduces the model's RUN
  output. The denoise loop early-halted at step 11/16 (`halted=True`) via the stable+confident
  stopping criterion, exercising the data-dependent halt through the serving session.
- Metrics: TTFT 111.6 s, block latency 110.2 s, 2.32 tokens/block/s (16-step cap; block-diffusion
  cost profile — reported per-block, never `1000/mean_tpot_ms`).
- Artifact: `serving_smoke_fulldepth_visible.json`, `/tmp/dg_serving_smoke_visible.log`.

### Qualitative verdict (HF-vs-TT control)

- **Verdict: PASS (RUN-first).** The serving path reproduces the model's RUN-path output. With
  sufficient denoise steps (16, halted 11) it emits coherent text identical to the recorded RUN
  control; with the fast RUN config (4 steps / EOS-stop) it emits the expected EOS-heavy
  degenerate block (#48291, not a serving bug).
- **Control:** the recorded `text_demo` visible-dialogue RUN output (same prompt, same on-device
  sampling + KV path) and the R0.5 HF-vs-TT committed-argmax replay (`demo/replay_hf_tt.py`,
  `plan.md` R0.5 — HF itself commits mostly EOS on this bf16/MoE/TP=4 path; the residual non-EOS
  miss is at the logits level, deferred with #48291). The serving output is **not materially
  worse** than the prompt-correct control → no serving regression.
- The shared `models/common/readiness_check/check_degenerate_output.py` degeneracy gate is
  unavailable in this checkout (the whole `readiness_check` harness is absent — see below); the
  visible-text vs RUN-control comparison serves the same purpose here.

### gemma4 isolation gate

`git status --porcelain -- models/demos/gemma4/` is empty — this stage edits **zero** gemma4
files (all changes under `models/experimental/diffusion_gemma/`). The `git diff main --
models/demos/gemma4/` delta is pre-existing branch state (main advanced: #47817/#47556/#47172;
plus the deferred F1/R-new decode-isolation item), not introduced by this stage.

## Process / device hygiene

- Device audited free before runs: 4× Blackhole (chips 0–3), no `vllm`/`EngineCore` processes.
- The driver opens/closes its own mesh (`_open_mesh_device`/`_close_mesh_device`, fabric
  disabled on close). No profiler/Tracy in the serving stage (per skill). No leftover
  processes after runs.

## Stage review

`stage-review` (fresh independent xhigh subagent, 2026-07-03): **clean-pass**, no required work.
Re-derived the delegation chain (adapter → `BlockDiffusionServingSession` →
`denoise_and_commit_block` → `denoise_block` on-device sampling + `commit_canvas_tokens`), the
block-granular contract, the #47488 scope vs the runner code, the plugin patch, context contract,
per-block metrics, qualitative control, and the gemma4 stage-gate. Two P3 concerns (disclosed
residual risk, not blockers) were then tightened:

- Unified both KV allocators to return model-owned handles via `_model_owned_kv_handles`
  (fixed the legacy `allocate_kv_cache` "no new DRAM" docstring/behavior mismatch; dropped the now
  unused `allocate_vllm_kv_cache` import).
- Documented the adapter-class execution-coverage gap in README limitations (adapter methods
  import `vllm.*`; proven by inspection + the device-tested session core; full coverage lands with
  #47488 + installed plugin).

Anomalies were classified as controlled: the 4-step degenerate/empty block and the reduced
`<unused*>` output are the RUN-first #48291 fidelity limitation (the 16-step full-depth run
reproduces the coherent RUN control), and the 1-block early stop at 4 steps is correct EOS-stop
behavior (`stop` distinct from denoise `halted`).

## Live serving verification (fresh vLLM) — 2026-07-03

Goal met: stand up a CURRENT-project vLLM (not the stale ghcr image) and do the LIVE serving
verification. Host install succeeded, so no container was built.

### Environment (host install into the current DG venv)

- venv `/home/zni/venvs/tt-diffusion-gemma`: Python 3.12.12, `ttnn` editable from
  `/home/zni/tt-metal` (current checkout), transformers 5.12.1, torch 2.11.0+cpu, `uv` present.
- fork `/home/zni/tt-vllm` (branch `dev`, head `6b4a3a7`) = full vLLM source + TT plugin
  (`plugins/vllm-tt-plugin`, a pip package).
- Install (per `plugins/vllm-tt-plugin/docs/install-vllm-tt.sh`):
  `VLLM_TARGET_DEVICE=empty uv pip install -e . --extra-index-url .../cpu --index-strategy unsafe-best-match`
  then `uv pip install -e plugins/vllm-tt-plugin`.
  - empty target → `ext_modules=[]` (no CUDA/kernel compile; gRPC codegen only). Runtime deps read
    from `requirements/common.txt` (`_no_device()` branch of `get_requirements`), which pins **no**
    torch → the venv's torch 2.11 / transformers 5.12.1 / ttnn are preserved. Build isolation
    installs `torch==2.10.0` (build-system.requires) only in a throwaway env.
- Verified: `vllm 0.1.dev1+g6b4a3a7b4.empty`, `vllm-tt-plugin 0.0.0`,
  `platform_plugin() -> vllm_tt_plugin.platform.TTPlatform`, ttnn/transformers/torch versions
  unchanged, DG adapter imports.

### Applied patches (saved as files here; fork is a separate repo)

- `plugin_47488_registration.patch` — `register_tt_models()` in `platform.py`: HF arch
  `DiffusionGemmaForBlockDiffusion` + `DiffusionGemmaForCausalLM` + `TT*` aliases →
  `models.experimental.diffusion_gemma.tt.generator_vllm:DiffusionGemmaForCausalLM`, inserted after
  the Gemma4 block.
- `plugin_47488_model_runner.patch` — the #47488 runner change (`model_runner.py`), verified live.
  Line numbers here are for `6b4a3a7` (the stale-image numbers in the older scope table differ):

  | What | Where (6b4a3a7) | Change |
  |---|---|---|
  | Collapse device sample to 1 token | `_get_output_tokens` :2760 `reshape(sz)` | `reshape(sz, -1)` → keep `[sz, num_out]`; guard 1-token logprobs path |
  | DP-pack 1-token assert | `sample_tokens` :2252 `shape[1]==1` | accept `shape[1] >= 1` |
  | apply-to-state 1-token assert + writes | `_apply_sampled_tokens_to_state` :2887 | drop `num_out_tokens==1`; write `[start:start+N]` block, advance `num_tokens` by N, extend output by N (sync + captured paths) |
  | build-output 1-token list | `_build_runner_output` :2839 `view(num_reqs)` | `reshape(num_reqs, num_out)`, emit N tokens/req |

  Design: `num_out_tokens = sampled_token_ids.shape[1]`. N=1 keeps the autoregressive path
  byte-identical; N=`canvas_length`(256) for DiffusionGemma. vLLM v1 core already supports a
  per-request token *list* (`update_from_output` → `_update_request_with_output` appends all and
  trims at the stop point), so a single 256-token block satisfying `max_tokens <= 256` needs no
  scheduler change; multi-block (`max_tokens > 256`) needs the scheduler
  `num_computed_tokens += canvas_length` half of #47488 (recorded, not yet applied).

### Live server + request

- Launch: `python -m vllm.entrypoints.openai.api_server --model $DG_CKPT --max-model-len 1024
  --max-num-seqs 1 --block-size 64 --additional-config '{"tt":{"sample_on_device_mode":"all",
  "enable_model_warmup":false}}'`, env `MESH_DEVICE=P150x4 DG_CKPT=... DG_VLLM_GUMBEL_MODE=argmax
  VLLM_ENABLE_V1_MULTIPROCESSING=0 TT_METAL_HOME=/home/zni/tt-metal`.
  - `--block-size 64` is required (else `get_num_available_blocks_tt` hits
    `block_size(None) * max_batch` → TypeError at KV init). Launch flag, not a model bug.
  - `--max-model-len 1024` = inner-loop bring-up value (fast KV/build); context contract's 262144
    unchanged and is the value for final headline evidence.
  - Startup: `/health` 200, `/v1/models` OK, arch → `DiffusionGemmaForBlockDiffusion`,
    30-layer 26B builds on `(1,4)`, `GPU KV cache size: 21,824 tokens`, warmup skipped.
- Live #47488 blocker (pre-patch), reproduced on the running server:
  `_get_output_tokens :2757 next_token_ids = _take(tt_out).reshape(sz)` →
  `RuntimeError: shape '[1]' is invalid for input of size 256` → `EngineDeadError`, HTTP 500 after a
  253 s prefill+denoise. Confirms #47488 exactly (256-token block vs one-token runner).
- Device recovery: a hard-killed EngineCore left ethernet cores un-reset
  (`TT_THROW ... assert_active_ethernet_cores_to_reset`); `tt-smi -r` + `(1,4)` mesh-smoke recovered
  (per tt-device-usage). Prefer graceful SIGTERM shutdown to avoid this.
- Live request after patch (`POST /v1/completions`, prompt "Hello, how are you?", max_tokens 32,
  temperature 0):
  `HTTP 200` (wall 254 s), `{"choices":[{"text":"","finish_reason":"stop"}],
  "usage":{"prompt_tokens":6,"completion_tokens":1}}`.
  Server per-block: `prefill row=0 prompt_len=6 cache_len=32 block0 next_pos=288 steps=35
  latency=252.259s` — a 256-token block emitted through vLLM, position 32→288, 35 denoise steps
  (early-halt), full 30-layer 26B, argmax, bf16, TP=4. Empty text/1 token = EOS committed at
  canvas position 0 → the known RUN-first #48291 fidelity behavior, not a serving regression.
- Remaining #47488 **scheduler half**, captured live: a second request with `ignore_eos:true`
  (`max_tokens:48`, so >1 token must survive per step) → HTTP 500 at
  `vllm/v1/core/sched/async_scheduler.py:53 assert request.num_output_placeholders >= 0`. The async
  scheduler reserves 1 output placeholder/step (`_update_after_schedule` `+= 1 + spec`) while a
  block-diffusion step emits up to `canvas_length=256` → underflow when >1 token survives the
  stop-trim (the first request passed only because it EOS-stopped at token 0). This is beyond the
  scoped `model_runner.py` change (done); generalizing placeholder reservation + the
  `num_computed_tokens += canvas_length` advance (prefill-block + decode-block) is the scheduler
  half — left as the precise live blocker rather than hacked (a wrong num-computed advance can
  silently corrupt generation). Batch-1 single-block serving that stops within block 0 (the DG
  default-stop path) works live end-to-end.

### Result vs task goal

Fresh CURRENT-project vLLM (host install, not the stale image) + live serving verification: DONE.
Live server serves DiffusionGemma on QB2; a real `/v1/completions` returns HTTP 200 with a
256-token block; per-block metrics captured; the #47488 runner blocker was reproduced live and
fixed live; the remaining scheduler-half blocker is reproduced + documented live (task step 6).



- `faebfbcc358` — feat(diffusion_gemma): block-granular vLLM serving adapter (#47466/#47488).
  Pushed to `diffusion-gemma-function` (194dbd432db..faebfbcc358). All pre-commit hooks passed.
  Adds `tt/serving.py`, `tt/generator_vllm.py`, `demo/serving_smoke.py`,
  `tests/test_serving_block_contract.py`, `doc/context_contract.json`, `doc/vllm_integration/*`.
- `4d320be2615` — refactor(diffusion_gemma): tighten vLLM adapter KV-cache ownership post-review.
  Post-clean-pass touch-ups (unify KV allocators via `_model_owned_kv_handles`, docstring fix,
  drop unused import, README coverage-gap note, work-log review record). Pushed
  (faebfbcc358..4d320be2615). All pre-commit hooks passed.
