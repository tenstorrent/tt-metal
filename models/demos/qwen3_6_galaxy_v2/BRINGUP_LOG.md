# Qwen3.6-27B Galaxy v2 — Bringup Log

Live tracker. Append-only. Mirrors the format in
`models/demos/olmo_galaxy/BRINGUP_LOG.md`.

## 2026-06-04 (cont.2) — full prefill forward runs end-to-end (eager, batch-1); device wedged on final confirm

Drove the request prefill all the way through the model on the 32-chip mesh
(eager, batch-1). Each blocker below was diagnosed against the WORKING demo
(`text_demo_qwen36.py::_run_prefill`) and fixed by matching it. Fixes committed
(tt-metal model + tt-inference-server catalog/plugin):

1. **delta-attn user_id** ttnn-tensor → int row 0 (batch-1) — `qwen36_delta_attention.py`.
2. **GDN seq-prefill L1 clash** → `QWEN36_GDN_SEQ_PREFILL=0` (ttnn chunk path).
3. **prefill trace-capture** GDN clash → eager prefill (`_disable_prefill_tracing`,
   generator_vllm) + worker `trace_mode=false`.
4. **partial-RoPE subtile broadcast** → `get_or_create_prefill_rot_mats` now
   builds the demo's partial cos/sin (rope_dim=64, `build_mrope_cos_sin`) and the
   qwen36 attention slices to seq T (`llama_model.py`, `llama_attention.py`).
5. **paged_fill_cache** ttnn-tensor user_id → `batch_idx_tensor=` (`llama_attention.py`).
6. **distributed-norm all_gather hang** → `fabric_config: FABRIC_1D_RING`
   (match the demo's `set_fabric_config`).
7. **on-device sampling %32 / batch-32 warmup** → skip built-in warmup
   (`prefill_warmup_completed=True`) + host sampling; `max_concurrency=1`
   (max_batch_size=32 L1-OOMs the per-batch activation buffers at 256k).
8. **ModelRunnerOutput** dropped `spec_token_ids` kwarg (vLLM API drift).

**Result:** on the run before fix #8, the **full prefill forward completed on
device and produced sampled tokens** — the only error was the #8 output kwarg,
now fixed. The server is **code-complete for batch-1**; a single clean run
should return the completion.

**BLOCKER (hardware, not code):** the repeated `tt-smi -r` between dirty engine
deaths wedged the galaxy — `tt-smi -glx_reset` now reports `ARC Status: 0/1
initialized` and `Read 0xffffffff over PCIe ID 0: the board should be reset`.
A board dropped off PCIe; software reset can't recover it. **Needs a host
power-cycle / IPMI reset.** After recovery: relaunch (command in the prior
entry) → curl should return "Paris" → then ISL-256k benchmarks.

## 2026-06-04 (cont.) — server reaches the model prefill forward; blocked on V2 prefill-kernel bugs

After clearing ALL vLLM-plugin drift (below), the server now loads + schedules +
enters the **actual model prefill forward**. The remaining blockers are
**model-kernel bugs in the Generator prefill path** (the BLOCKED V2-decode/V2-9
area), NOT server integration. Two hit so far, in order:

1. **DeltaNet GDN seq-prefill L1/CB clash** — `gdn_chunk_ops_seq.py:
   chunk_gated_delta_rule_seq` throws "Statically allocated circular buffers …
   clash with L1 buffers" at seq_len 128 (cores (0,0)-(0,5), 128-byte overlap),
   in BOTH trace and eager prefill. **Worked around** by
   `QWEN36_GDN_SEQ_PREFILL=0` (env, in catalog) → uses `chunk_gated_delta_rule_ttnn`
   instead of the C++ seq parallel-scan kernel. The seq kernel's L1 clash is a
   real bug to fix for the optimized path.
2. **Full-attention partial-RoPE subtile broadcast** — `llama_rope.py:
   partial_rope_apply` → `ttnn.multiply/addcmul(x_rot, cos_tt)` →
   "Invalid subtile broadcast type" (binary_ng) at seq_len-128 prefill. cos/sin
   table shape doesn't broadcast against the rotated activation in the Generator
   prefill path. **OPEN** — needs model-side fix (cos_tt construction vs x_rot
   shape for partial RoPE rope_dim=64).

These were never exercised before because the demo/PCC tests use a different
prefill harness; the server is the first to drive `Generator.prefill_forward_text`
(warmup sweep + ttnn_prefill_forward) end-to-end. Prefill trace-capture is
disabled (`override_tt_config.trace_mode=false` + generator
`_disable_prefill_tracing=True`) and decode is eager.

**Server-side fixes also committed today:** delta-attn `user_id` int-coercion
(ttnn-tensor on the Generator path → batch-1 row 0); see commit log. Device
runs use `tt-smi -r` between attempts (dirty engine death hangs the ethernet).

**Status:** server integration COMPLETE; batch-1 generation blocked on the V2
prefill-kernel bugs above (#2 open). Next: fix partial-RoPE broadcast (and the
GDN seq-kernel L1 clash for the optimized path), then accuracy + ISL-256k bench.

## 2026-06-04 — local vLLM server: live on BH galaxy through prefill; vLLM-version drift fixes in tt-vllm-plugin

Stood up the **local** vLLM server (`run.py --workflow server --local-server
--tt-device blackhole_galaxy`) for batch-1. The server now **starts, opens the
32-chip mesh (grid 8×4), loads the 27B model on device (64/64 layers),
registers our class, schedules, and runs prefill.** Remaining: a cascade of
vLLM-API-drift fixes in the request hot-path before a clean batch-1 generation,
then ISL-256k benchmarks. Device validation is **paused** (not yet a clean
end-to-end completion). All fixes committed in the `tt-inference-server` repo
(separate git repo nested under tt-metal).

### Environment (one-time, into `python_env`)

- vLLM clone at `tt-metal/vllm`, checked out to **`8f36910`** (Llama70B-galaxy
  pin; the plugin pins `vllm==0.10.1.1` and `f4b029385`/main is too new — see
  drift list). Installed editable: `VLLM_TARGET_DEVICE=empty uv pip install
  --no-deps -e .`.
- `tt-inference-server/tt-vllm-plugin` installed editable (`uv pip install
  --no-deps -e .`) — provides the `tt` platform + `tt_model_registry` entry
  points vLLM discovers.
- **transformers pinned `>=4.56,<5`** → 4.57.6 (vLLM needs ≥4.56; 5.x breaks
  config promotion / "too new for tenstorrent"; hf-hub dropped to 0.36.2).
- See memory `qwen36-vllm-local-server-setup` for the full recipe.

### Launch (offline, batch-1)

    cd tt-inference-server
    SNAP=~/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
    export TT_METAL_HOME=$(cd .. && pwd) HF_HOME=~/.cache/huggingface \
      HF_TOKEN=hf_placeholder HF_HUB_OFFLINE=1 MODEL_WEIGHTS_DIR=$SNAP HF_MODEL=$SNAP
    MODEL_SPECS_ENV=dev python run.py --model Qwen3.6-27B --workflow server \
      --local-server --tt-device blackhole_galaxy --skip-system-sw-validation \
      --no-auth --tt-metal-python-venv-dir $(cd .. && pwd)/python_env

Launcher gates cleared: short model name `Qwen3.6-27B`; `HF_TOKEN` set
(placeholder, weights cached); `--no-auth` (skip JWT); catalog `version`
`>=0.11.0` floor → `0.11.1`; `MODEL_WEIGHTS_DIR`+`HF_MODEL`→snapshot for offline
weights; `mesh_grid_dict["BH-Galaxy"]=(8,4)`.

### vLLM-API-drift fixes in tt-vllm-plugin (commits in tt-inference-server repo)

The plugin targets vLLM ~0.10.1.1; the clone is newer, so each of these drifted:

| area | fix |
|---|---|
| platform.py imports | `ProcessorInputs`/`SamplingParams` deferred to TYPE_CHECKING (circular import) |
| platform.py | `getattr(envs,"VLLM_USE_V1",True)` (env removed) |
| native MM qwen3_5 vs runner check | `hf_overrides:{"architectures":["Qwen3ForCausalLM"]}` + register `TTQwen3ForCausalLM`→qwen3.6 class (shadows embedding fallback) |
| qwen3_5 config | drop `text_config`/`vision_config` sub-dicts so `get_text_config()` returns self (patch_rope_parameters) |
| worker/model_runner/scheduler imports | `cdiv`→`utils.math_utils`, `STR_DTYPE_TO_TORCH_DTYPE`→`utils.torch_utils`; `LayerBlockType` now a `Literal` (pass `"attention"`) |
| FullAttentionSpec | drop removed `use_mla` kwarg |
| MultiGroupBlockTable | add required `kernel_block_sizes` (=block_sizes) |
| AscendScheduler.__init__ | `*args,**kwargs` passthrough (`block_size` added) |
| KVCacheManager | `get_num_common_prefix_blocks(request_id)` single-arg |
| encoder cache | `get_freed_mm_hashes()` + SchedulerOutput `free_encoder_mm_hashes` (scheduler + model_runner consume) |
| CachedRequestState | `mm_features` vs `mm_kwargs`/`mm_positions` (pick via `dataclasses.fields`) |

### Remaining (when device runs resume)

1. **`tt-smi -r` before each launch** — a dirty engine death leaves the mesh
   ethernet hung (`Timed out waiting for active ethernet core`); confirmed that
   skipping the reset reproduces it. (Galaxy CPLD warns to use `-glx_reset` if
   `-r` fails.)
2. Relaunch → retry `curl /v1/chat/completions` → fix any further
   `model_runner.execute_model`/sampling drift (cascade is converging; each
   iteration ≈ 4–5 min: reset + cache-load + request).
3. Once batch-1 generates cleanly: accuracy parity vs demo, then **ISL sweep to
   256k** (`BENCHMARK_ISL_OSL_PAIRS` extended; note `isl+osl ≤ 262144`).
4. Then raise dev catalog `max_concurrency` 1→32 and re-validate.

**Root-cause note:** none of the failures are in the qwen3.6 model code — all
are tt-vllm-plugin↔vLLM-version skew. A plugin built against the clone's exact
vLLM (or pinning the clone to the plugin's 0.10.1.1) would avoid the cascade.

## 2026-06-03 — tt-inference-server integration (text-only, BH Galaxy) — code landed, device validation pending

Wired Qwen3.6-27B (text-only LM tower) into `tt-inference-server`'s vLLM
OpenAI API for `BLACKHOLE_GALAXY` (32× P150). Spec/plan:
`docs/superpowers/specs/2026-06-03-qwen36-27b-tt-inference-server-design.md`,
`docs/superpowers/plans/2026-06-03-qwen36-27b-tt-inference-server.md`.

**Key discovery — transformers / `qwen3_5`:** the checkpoint's arch is
`Qwen3_5ForConditionalGeneration` / `model_type: qwen3_5`, which exists in NO
public transformers release (verified 4.53.0 / 4.57.1 / 4.57.6 / 5.10.1; no
`auto_map` remote code). Consequences, baked into the integration:
- **No transformers bump** (tt-metal pins `transformers==4.53.0`; no version
  fixes it anyway).
- Generator loads weights from **raw safetensors**, never
  `AutoModelForCausalLM` (which raises "Unrecognized configuration class").
- vLLM's only blocker was `AutoConfig` — solved by registering a thin
  `qwen3_5` config in the tt-vllm-plugin at import time.

| stage | status | commit (repo) |
|---|---|---|
| V2-server-1 generator_vllm → local v2 model + raw-safetensors load | DONE | `e97cd06201b` (tt-metal) |
| V2-server-2 tt-vllm-plugin: qwen3_5 AutoConfig + TT model registration | DONE | `47b554b1` (tt-inference-server) |
| V2-server-3 model_spec ImplSpec + BLACKHOLE_GALAXY YAML catalog entry | DONE | `45e551c2` (tt-inference-server) |
| V2-server-4 launcher wiring verified + `DEVICE_TO_MESH_STR` BH fix | DONE | `c916a6d0` (tt-inference-server) |
| V2-server-5 server start / accuracy / 256k stress on hardware | **HAND-OFF (device)** | pending |

Registered arch name: `TTQwen3_5ForConditionalGeneration` →
`models.demos.qwen3_6_galaxy_v2.tt.generator_vllm:Qwen3_5ForConditionalGeneration`.
Catalog: `Qwen/Qwen3.6-27B`, impl `qwen3_6_galaxy`, `max_context=262144`,
`status=EXPERIMENTAL`, text-only.

**Off-device tests passing:** generator structural import test (3),
`qwen3_5` config registration + checkpoint-parse test (2), catalog-load test (1).

### Device hand-off checklist (required before EXPERIMENTAL → FUNCTIONAL)

Start command (run on the BH Galaxy host):

    cd tt-inference-server
    export TT_METAL_HOME=/home/tt-admin/ssinghal/qwen36/new/tt-metal
    MODEL_SPECS_ENV=dev python run.py --model Qwen/Qwen3.6-27B --workflow server \
        --local-server --tt-device tt-galaxy-bh --skip-system-sw-validation

Then validate: smoke `curl` → coherent "Paris"; accuracy ≥ demo within 2–3 pp;
one ~200k-token request for the 256k bar; 5 sequential requests with no
`tt-smi -r` between them.

Config still to fill in on hardware (sibling WH-galaxy values are a *starting
reference only* — measure on BH):

- `override_tt_config`: `dispatch_core_axis` (col?), `fabric_config`
  (FABRIC_1D_RING?), `worker_l1_size`, `trace_region_size` (profile),
  `sample_on_device_mode` (all?).
- `system_requirements`: BH `firmware` / `kmd` minimums (from HW team).
- `vllm_commit` for the BH vLLM image; final `tt_metal_commit` SHA.
- Container env: `ARCH_NAME=blackhole` + the BH dispatch arch yaml (the dev
  Dockerfile defaults to `wormhole_b0` — must be overridden at runtime).
- **DO NOT set `data_parallel_size`/`tt_data_parallel` > 1.** The generator
  asserts `tt_data_parallel == 1` (qwen3.6 v2 galaxy is TP-only); the sibling
  entries' `data_parallel_size: 4` must NOT be copied here.

## 2026-06-02 — BH grid widening: sub_core_grids 60 → 110 (dynamic) ✅

**Step 1 of BH_GRID_MIGRATION.** `sub_core_grids` master compute grid was
hard-coded to the inherited Wormhole 60-core band `(1,0)→(6,9)`. Widened to a
DYNAMIC full band derived from the live device grid:
`CoreRange((1,0), (grid.x-1, grid.y-1))` — col 0 reserved for DRAM. On this BH
P150 galaxy (`compute_with_storage_grid_size() == (12,10)`) that is cols 1-11 ×
10 rows = **110 cores** (was 60; +83% compute area). The "col 7 = dispatch"
comment was a WH leftover (dispatch is the last worker column, outside the band).
Not hard-coded — harvested SKUs auto-derive the right width.

Files: `tt/qwen36_model_config.py` (dynamic derivation + docstring); the only
60-core literal duplicate was the config itself (demo/test `(1,0)→(3,9)+(5,0)→(6,9)`
50-core literals are the local SAMPLING/argmax grid, independent of the master
compute grid — left untouched; ops that read `args.sub_core_grids` auto-pick the
wider grid). Updated `tests/test_ccl_buffer_keys.py` mock (grid 7→12,
`num_cores 60→110`). New contract test `tests/test_sub_core_grids_dynamic.py`
(parametrized: grid (12,10)→110, (7,10)→60).

Per-op ring/SDPA program configs (XQKV/WO/FF/LM-head 24-core rings, SDPA decode)
NOT widened — separate later migration steps with their own shape math.

**Gates (all PASS):**
- Unit: contract test 3/3 + `test_ccl_buffer_keys.py` 6/6.
- ISL-128 (`test_decode_perf_intrace::test_qwen36_64L_decode_intrace_perf`,
  greedy): coherent text (123 alpha chars, condiment-prompt continuation),
  **20.52 tok/s/user** (up from ~18.x baseline). No op broke on the wider grid.
- 128k demo (`text_demo_qwen36::test_qwen36_demo_batch1`, sampled): ISL=131072
  (103409 real tokens), COHERENT on-topic output, prefill 1406 tok/s warm,
  decode **18.52 tok/s/user** (vs ~18.3 baseline). TTFT warm 93.2 s.

No op assumed the cols 1-6 layout — the dynamic ops (CREATE_HEAD_OUTPUT_MEMCFG,
llama_ccl sub_device_crs, RoPE trans-mat) cleanly picked up the wider grid.

## 2026-06-01 — COHERENCE FIXED + sampling; clean ISL sweep 128→128k ✅

**The coherence bug (garbage output at every ISL) — ROOT CAUSE & FIX.**
The GDN/DeltaNet chunk masks (`triu_ones`/`tril_mask`/`eye`/`lower_causal`) were built
**lazily mid-forward** — `ttnn.from_torch`/`ones`/`triu` ran on the *first* call of each of
the 48 GDN layers, during prefill. Under the galaxy **TP=32** layout those mid-prefill
allocations collided with live activation DRAM and **corrupted the residual stream** → garbage
tokens at ALL ISLs (incl. 128). Why it was so hard to find: the seq op's own PCC was clean
(0.9998 out / 0.99999 state), and P150 (TP=4) was unaffected because its allocator layout
differs — so it reproduced only in the full 32-chip model, ~2 layers downstream of the op.
Took ~50 device experiments to localize (ruled out: multi-core kernel addressing, CB
single/double-buffering, input deallocs/UAF, core relocation, L1-vs-DRAM placement, the C++
kernel itself, and the wrapper preprocessing — by truncation-bisecting the adapter).
**FIX:** build the masks once at `__init__` (`TtQwen36DeltaAttention._build_seq_masks`),
never lazily mid-forward. One-line-class change; PCC/perf unchanged.

**Seq prefill scaling recap (still required):** multi-core `gated_delta_attn_seq`
(`QWEN36_SEQ_CORES_PER_HEAD=4`, value-dim split, 24 cores) + chunked-4k prefill
(`QWEN36_PREFILL_CHUNK=4096`) → no L1 CB clash through 128k. Inner kernel chunk `C=128` fixed.

**Sampling (fixes greedy repetition).** Greedy argmax degenerated into loops at long ctx
(64k/128k → "the the the"). Added on-device `TTSampling` (`models/common/sampling/tt_sampling.py`)
to the demo decode loop, replacing greedy. Qwen3.6 thinking-mode defaults: **temp=1.0, top_p=0.95,
top_k=20**. Env: `QWEN36_SAMPLE=1` (default; `=0` greedy fallback), `QWEN36_TEMP/TOP_P/TOP_K`
override. (k/p/temp passed as torch tensors of length 32 — the decode is 32-user packed and
`ttnn.sampling` asserts `k.shape==[32]`.) Verified varied/non-repetitive output at every ISL.

**Perf (`text_demo_qwen36.py`, real Qwen3.6-27B weights, BH_GLX, batch-1):**

| ISL | TTFT cold | TTFT warm | prefill tok/s (warm) | decode tok/s/u (greedy) | decode tok/s/u (sampled) | coherent |
|---|---|---|---|---|---|---|
| 128 | 4.0 s | 1.6 s | — | 18.7 | 18.78 | ✅ |
| 4k  | 5.1 s | 1.7 s | 2343 | 17.7 | 17.87 | ✅ |
| 8k  | 6.4 s | 3.4 s | 2388 | 16.9 | 17.05 | ✅ |
| 16k | 10.2 s | 7.2 s | 2289 | 15.4 | 15.55 | ✅ |
| 32k | 18.7 s | 15.5 s | 2112 | 13.1 | 13.19 | ✅ |
| 64k | 39.6 s | 36.0 s | 1820 | 10.1 | 10.18 | ✅ |
| 128k | 98.6 s | 93.6 s | 1395/1401 | 8.0 | 8.05 | ✅ |

All ISLs run error-free (no clash/OOM), coherent, and non-repetitive with sampling. 128k
greedy correctly continued the Frankenstein/Walton context; sampled outputs are varied and on-topic.


## Current Status

**Stage:** V2-9 BLOCKED on V2-decode (dirty workspace). Prefill end-to-end PCC > 0.99 verified through 64L. Decode codepath never wired in v2; eager decode crashes at the embedding-output / decoder-input layout boundary. Trace machinery in `generator.py` is already wired; gated on V2-decode landing. See `tests/test_decode_trace_parity.py` module docstring for the full blocker chain.

| stage | status | commit |
|---|---|---|
| V2-1 bulk copy from llama3_70b_galaxy | DONE | `45b2138d759` |
| V2-2 qwen36_model_config.py — is_qwen36 + full-grid | DONE | `45b2138d759` |
| V2-3 load_checkpoints.py — qwen3.6 HF key map | DONE | `27e57ca0aa1` |
| V2-5a qwen36_delta_attention.py (NEW, +545 lines) | DONE | `27e57ca0aa1` |
| V2-6 llama_ccl.py — dual-dtype CCL buffer keys | DONE | `27e57ca0aa1` |
| V2-norm distributed_norm.py — zero_centered kwarg | DONE | `27e57ca0aa1` |
| V2-rope llama_rope.py — partial RoPE | DONE | `27e57ca0aa1` |
| V2-4 llama_attention.py — is_qwen36 (QKVG + QK-norm + RoPE + gate) | DONE | `41c190106dd` |
| V2-decoder llama_decoder.py — hybrid dispatch | DONE | `41c190106dd` |
| V2-embedding llama_embedding.py — bf16 force | DONE | `41c190106dd` |
| V2-model llama_model.py — per-layer + rope_setup threading | DONE | `58ee671e46e` |
| V2-config2 model_config populated (~100 keys) | DONE | `6472cdd551f` |
| V2-device-smoke setup + 7 construction bugs fixed | DONE | `16ba2ca1fcc` `9a9b2c86439` |
| V2-7 Block-test suite (DeltaNet 0.9995, full-attn 0.9997) | DONE | `eaefe1e13b8` |
| V2-7b 1L/4L hybrid + decoder gather/scatter | DONE | `bc2b24d3074` |
| V2-7c 64L hidden + logits PCC + Paris parity | DONE | `2227b2709c0` |
| V2-decode (qwen3.6 decode end-to-end) — REQUIRED for V2-9 | **BLOCKED** | dirty (V2-9 attempt) |
| V2-9 Trace capture parity (test added as skipped sentinel) | **BLOCKED** | dirty (this session) |
| V2-10 Tracy perf sheet + PERF.md | pending | |

### V2-9 attempt (2026-05-14) — findings

- Generator trace machinery (`begin_trace_capture` / `end_trace_capture` /
  `execute_trace`, `trace_ids_decode`, `_capture_trace_text`,
  `_decode_easy_trace_text`, `release_traces` in `__del__`) is already
  wired in `tt/generator.py` — no `_TRACE_SUPPORTED=False` flag exists
  in v2. (v1's flag was a manual gate; v2 inherits 70B's hot path.)
- Eager decode is broken before trace can even be attempted. Five
  separate layout/contract mismatches between v2's inherited 70B decode
  contract (batch-32 packed in T-dim, L1-WIDTH-sharded residual via
  `DECODE_RESIDUAL_MEMCFG`, `tt_sharded_distributed_rmsnorm`) and the
  qwen3.6 attention/DeltaNet blocks (written against v1's single-user
  `[B=batch, 1, T=1, H]` DRAM-interleaved contract with
  `tt_distributed_rmsnorm`).  Full chain in
  `tests/test_decode_trace_parity.py` docstring.
- Two small, safe infrastructure fixes APPLIED (no impact on the
  passing prefill tests, validated):
    - `TtLlamaAttention.prefetch`: skip `insert_tensor(self.wqkv)` /
      `insert_tensor(self.wo)` for `is_qwen36=True` (we use `wqkvg`).
    - `_NoOpPrefetcherSetup.worker_sub_device_id` attribute added +
      synced in `setup_decode`.  `TtTransformer.forward(mode='decode')`
      reads this unconditionally for the stall-group set call.
- Test `tests/test_decode_trace_parity.py` lands as a skipped sentinel
  (flag `_DECODE_ENABLED=False`).  Flip to `True` once V2-decode lands.
- **Static review on qwen3.6 forward paths found no host-write
  blockers** for trace capture: no `from_torch(device=...)` /
  `to_torch` / `copy_host_to_device_tensor` calls in the hot path.
  All persistent buffers (DeltaNet `dn_state_buffer`,
  `conv_state_buffer`, `_conv_zero_pad`) are allocated at
  `__init__` and written in-place via `ttnn.copy` (V2-5a contract).
  The v1 PERF.md residual `to_memory_config` host-write blocker is
  in the *70B branch* of `TtLlamaAttention.forward_decode`, NOT in
  `_forward_decode_qwen36`.  So once V2-decode runs eager, the trace
  capture itself should succeed without intervention.

### Recommended V2-decode plan (predecessor)

Rather than incrementally bridging the 70B↔qwen3.6 decode boundaries
one layout converter at a time, mirror v1's decode contract directly:
add a `forward_decode_qwen36` entry point on `TtTransformer` that
takes `[B=batch, 1, T=1, H]` natively, uses DRAM-interleaved residual
throughout, and calls `tt_distributed_rmsnorm` (the prefill primitive,
which is already verified at PCC > 0.99 for 64L).  The generator's
`ttnn_decode_forward` should dispatch to this for qwen3.6.  Estimated
effort: 1-2 sessions.

After V2-decode lands and eager decode is PCC-verified vs HF reference:

- Flip `_DECODE_ENABLED=True` in `test_decode_trace_parity.py`
- Static review predicts trace capture will succeed cleanly.
- Then V2-10: tracy perf sheet, target >= 17 tok/s/user 64L decode
  (olmo precedent on the same mesh).

## Model Overview

| Param | Value |
|-------|-------|
| dim | 5120 |
| n_layers | 64 (hybrid: `[lin,lin,lin,full] × 16`) |
| Full-attn heads (Q) | 24 (6:1 GQA) |
| Full-attn KV heads | 4 |
| head_dim | 256 |
| rope_dim (partial RoPE) | 64 (rotary factor 0.25) |
| mrope_section | [11, 11, 10] (text mode collapses) |
| mrope_theta | 10,000,000 |
| MLP intermediate | 13,824 |
| vocab_size | 248,320 (padded to 248,832) |
| norm_eps | 1e-6 |
| Norm | zero-centered RMSNorm (`w' = w + 1`) |
| Linear-attn | DeltaNet — 16 K-heads, 48 V-heads, head_dim 128, conv_kernel 4 |

## Sessions

### Session 1 — 2026-05-14
- **V2-1** (commit `45b2138d759`):
  bulk-copied `llama3_70b_galaxy/` → `qwen3_6_galaxy_v2/` via
  `rsync -a --exclude '__pycache__'`. Tree intact (16 .py files in tt/).
  Removed inherited `README.md` and `PERF.md` (llama3_70b-specific);
  wrote v2-specific `README.md` + this `BRINGUP_LOG.md`.

- **V2-2** (same commit): added
  `tt/qwen36_model_config.py` with `TtQwen36ModelArgs(TtModelArgs)`.
  Subclasses `TtModelArgs` directly (not `TtQwenModelArgs` — qwen3.6 head
  layout 24×256 diverges too far from qwen3-32B 64×80). Sets:
  - `is_qwen36 = True` flag for downstream branches in `llama_*.py`
  - `use_prefetcher = False` everywhere
  - `sub_core_grids = (1,0) → (6,9)` = 60 contiguous Tensix cores (was
    50 with the 70B col-4 carve-out). +20% compute area for matmul / CCL.
  - All qwen3.6 hyperparams documented in the docstring at the top of
    the file.
  - Hybrid layer pattern populated from HF `config.layer_types` in
    `_set_qwen36_hf_params()`.

  Smoke-test: `python -c "from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs"` imports without error.

  Per-op program configs (matmul-2d, sharded norm, etc.) are NOT yet
  re-tuned for the 60-core grid — that's intentional and will be done
  per-op as V2-4 through V2-6 wire things up. Inherited 50-core configs
  will still run on the 60-core grid; extra cores simply sit idle until
  per-op tuning.

- **Wave 1** (commit `27e57ca0aa1`): five parallel sub-agents landed
  V2-3 (load_checkpoints.py qwen3.6 HF key map — 9 tests), V2-5a
  (qwen36_delta_attention.py NEW +545 lines), V2-6 (llama_ccl dual-dtype
  buffer keys — 6 tests), V2-norm (distributed_norm zero_centered — 2
  tests), V2-rope (llama_rope partial-RoPE — 3 tests). All cpu_only,
  20/20 tests pass, cross-imports clean.

  Key findings folded forward:
  - HF DeltaNet uses SPLIT linear_attn weights
    (`in_proj_qkv/z/a/b`), not combined `qkvz/ba` form the plan brief
    expected.
  - q_proj fused with output gate: `[12288, 5120]` = 2 × 24 × 256 ×
    5120. `convert_hf_qkv_to_meta_format` skipped for qwen3.6.
  - QKVG persistent CCL buffer width = 56 × 256 = 14336 (per-chip 3584
    across 4 mesh cols).

- **Wave 2** (commit `41c190106dd`): three parallel sub-agents landed
  V2-4 (llama_attention.py +763 lines — QKVG split + per-head QK-norm +
  partial RoPE + sigmoid-gate; 2 tests), V2-decoder (llama_decoder.py
  +61 lines — hybrid dispatch, all 4 scenarios verified; 4 tests),
  V2-embedding (llama_embedding.py +9 lines — bf16 force; 4 tests).

  Full Wave-1 + Wave-2 suite: **30 passed, 1 skipped (placeholder), 0
  failed**. Cross-imports across all 9 v2 modules verified.

  TODO for V2-model: must thread `rope_setup` onto `TtLlamaAttention`
  instances (qwen3.6 forward path reads `self.rope_setup`; the 70B
  path constructs its own internally).

- **Next: Wave 3** — V2-model (llama_model.py per-layer instantiation,
  rope_setup creation + threading, qwen36 weight ingestion via V2-3's
  `standardize_hf_keys_qwen36`) and V2-generator (DeltaNet state
  plumbing through trace). Both CPU-buildable with mocks. Then V2-7
  block tests on BH GLX (strictly sequential — one device session at
  a time, `tt-smi -r` between failures).

- **LM-head ring parametric/widen attempt** (`qwen36_model_config.py`):
  Made `LM_HEAD_RING_SIZE` parametric (env `QWEN36_LM_HEAD_RING_SIZE`,
  default 24) + derive the ring in/out shard grids dynamically from the
  live `sub_core_grids` instead of the old hard-coded 24-core coordinate
  lists. Widened candidate = 72 (only clean divisor of the 1944-tile
  per-col vocab that `num_to_coregrid` can realize as a rectangle: 8x9;
  81/108/54 rejected — not 8-multiples). Added shape-math unit test
  `test_lm_head_ring_size.py` (24 & 72 both PASS).
  - **Finding (measured A/B, identical build):** ring 24 vs 72 give the
    SAME decode rate — ISL-128 20.53 tok/s/user (48.71 vs 48.72 ms/step),
    128k demo 18.5 tok/s/user, both coherent (token 248068). The LM-head
    is DRAM-WEIGHT-BANDWIDTH-bound (~80 MB bf8 1280x62208 read/token over
    8 banks), NOT core-count-bound — widening the ring buys nothing.
  - **Decision:** default stays 24 (fewer cores, no contention); 72 is a
    validated-coherent opt-in. The real LM-head lever is weight bandwidth
    (e.g. bf4 weights / fewer DRAM reads), not ring width.
  - Status: DONE_WITH_CONCERNS (correct + coherent, perf-neutral).
    PCC: identical math, token 248068 at both sizes. Block hash: n/a.
