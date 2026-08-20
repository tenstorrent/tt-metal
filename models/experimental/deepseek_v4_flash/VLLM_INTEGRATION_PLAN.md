# DeepSeek-V4-Flash → Tenstorrent vLLM Integration Plan

This is a step-by-step implementation plan for integrating the ttnn
`models/experimental/deepseek_v4_flash` model into the Tenstorrent vLLM backend
(the `vllm-tt-plugin` in the `tenstorrent/vllm` `dev` branch).

It is written so a junior/simple agent can execute it task-by-task. Do the tasks
**in order**. Each task lists: what to do, which files to touch, and how to
verify before moving on. Read the "Background" and "Architectural gaps" sections
first — they explain *why* the phased approach is required.

> IMPORTANT EXPECTATION SETTING: DeepSeek-V4-Flash is **not** a standard
> tt-transformers model. It has three attention layer types, a hyper-connection
> residual stack, hash-MoE + routed-MoE layers, and it currently runs **batch=1,
> one user per forward call**, prefills by replaying decode one token at a time,
> and manages its own KV caches internally. The stock vLLM KV-cache/paging model
> does **not** map cleanly onto it. The plan therefore proceeds in phases:
> **Phase 1 = functional bringup (correct but slow, batch handled by looping over
> users). Phases 2–3 = performance/batching/paging.** Do not attempt to make it
> fast on the first pass.

---

## Background: how TT models plug into vLLM

The plugin (`tenstorrent/vllm`, `dev` branch, `plugins/vllm-tt-plugin/`) provides
all the vLLM platform/worker/scheduler machinery. **Nothing in vLLM core needs
editing.** To add a model you provide two things:

1. **A "generator" wrapper class** living in *this* repo (tt-metal), next to the
   model, e.g. `models/experimental/deepseek_v4_flash/tt/generator_vllm.py`. It
   adapts the ttnn model to the interface vLLM expects (see "The contract"
   below). This is the bulk of the work.

2. **A registration entry** in the plugin repo
   (`plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_registry.py` and
   `platform.py`) mapping the HuggingFace architecture name to a
   `TT`-prefixed vLLM model class that imports our generator wrapper.

Reference the in-repo guide `tech_reports/LLMs/vLLM_integration.md` and the
existing wrappers:
- `models/tt_transformers/tt/generator_vllm.py` (Llama/Qwen/Gemma/GPT-OSS — the
  canonical examples, all subclass `Generator`).
- `models/demos/deepseek_v3/tt/generator_vllm.py` (**closest analog**: a
  non-tt-transformers custom model that implements the same contract on top of
  its own `DeepseekGenerator` base rather than the shared `Generator`).

### The contract (methods the wrapper class must expose)

From `tech_reports/LLMs/vLLM_integration.md` and the DeepSeek V3 example:

```python
class DeepseekV4FlashForCausalLM(<base>):
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": False,   # Phase 1: host sampling only
    }

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size,
                              max_seq_len, tt_data_parallel=1, optimizations=None):
        ...  # build and return the wrapper (loads weights, builds ttnn model)

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        ...  # returns the kv cache object vLLM hands back on every forward

    def prefill_forward(self, *args, **kwargs):
        # kwargs: tokens [B, max_prompt_len] (zero-padded), page_table [B, nblocks],
        #         kv_cache, prompt_lens [B], sampling_params (or None), empty_slots
        # returns: host logits [B, S, V]  (Phase 1, host sampling)

    def decode_forward(self, *args, **kwargs):
        # kwargs: tokens [max_batch_size, 1] (zero-padded to max_batch_size),
        #         start_pos [max_batch_size], page_table [max_batch_size, nblocks],
        #         kv_cache, enable_trace, read_from_device, sampling_params (or None)
        # returns: host logits [B, 1, V]  (Phase 1, host sampling)

    # inherited/optional: warmup_model_prefill, read_decode_output, get_max_tokens_all_users
```

Key facts about the contract:
- **Batch semantics**: `max_batch_size` is the *global* batch across all
  data-parallel groups; `tt_data_parallel` is the number of KV-cache
  replicas/submeshes. Decode inputs are padded to `max_batch_size` (constant
  shape for tracing).
- **Sampling**: default vLLM path passes `sampling_params=None` and samples on
  host from returned logits. On-device sampling is opt-in and requires
  `supports_sample_on_device=True`. **Phase 1 uses host sampling** — the wrapper
  just returns logits.
- **KV cache**: `kv_cache_shape = (max_num_blocks, num_kv_heads, block_size,
  head_size)`. vLLM owns the tensors and passes them back each call. (See the
  hard problem in "Architectural gaps" — V4's compressor caches do not fit this
  shape, so Phase 1 manages caches internally.)

---

## Current state of the model (what exists today)

Files under `models/experimental/deepseek_v4_flash/`:
- `tt/model.py` — `DeepSeekV4Model`. Entry points:
  - `decode(token_id: int, pos: int, rope) -> hidden [B,1,1,D]` (B=1, eager).
  - `decode_traced(token_id: int, pos: int) -> logits [1,1,V]` (B=1, traced,
    lm_head folded in). Requires `prepare_static_decode(rope, max_seq, lm_head,
    num_sessions=..., total_tokens=..., block_size=...)`.
  - Paged multi-session API on the traced path: `open_session()`,
    `activate_session(sid)`, `reset_session(sid)`, `close_session(sid)`,
    `session_usage()` / `session_tokens_left(sid)`. Sessions share one block pool
    (`total_tokens` is the whole budget) and one set of captured traces; a step
    runs as `activate_session(sid)` then `decode_traced(...)`.
  - `reset_caches(max_seq)` — single-stream dense (non-paged) caches.
  - `decode_sampled_burst(...)` — on-device greedy multi-step (single submesh).
- `tt/attention.py` — three attention flavors, all of which page their KV when a
  `PagedLayerView` is supplied: `paged_update_cache` /
  `paged_scaled_dot_product_attention_decode` with `cache_position_modulo` to
  bound the sliding ring. The small per-token compressor window state
  (`win_kv`/`win_gate`/`prev_kv`/`prev_gate` in `_StaticLayerCache`) stays dense
  and is swapped per session on `activate_session`.
- `tt/paged_cache.py` — `PagedGroup` / `build_groups` (block geometry per layer
  type), `plan_pool_blocks`, `round_context`, `PagedKVManager` (shared block pool
  + per-session page-table rows), `PagedLayerView` (device pool + page table +
  `position_modulo` handed to attention), `PagedCacheFull`.
- `tt/weight_loader.py` — `DeepseekV4WeightLoader` (reads safetensors directly,
  not the HF `state_dict` path). `resolve_snapshot_dir` finds the checkpoint.
- `tt/quant.py`, `tt/moe.py`, `tt/hyperconnection.py`, `tt/decoder_layer.py`,
  `tt/embedding.py`, `tt/layers.py`, `tt/weight_cache.py`, `tt/common.py`.
- `configuration_deepseek_v4.py` — `DeepseekV4Config` (vocab 129280, hidden 4096,
  43 layers, head_dim 512, `layer_types`, `compress_rates`, `hc_mult=4`,
  `sliding_window=128`, `qk_rope_head_dim` derived).
- `tests/test_full_model_decode_demo.py`, `tests/test_multi_user_paged_decode_demo.py`
  — the working reference flows to imitate.

The model builds across **submeshes as a pipeline ring** (`use_submeshes=True`):
layer `li` lives on submesh `li % pipeline_stages`. This is *model/pipeline
parallelism* over the whole mesh, **not** data parallelism. Consequence for
vLLM: use `tt_data_parallel=1`, one replica spanning the full mesh.

---

## Architectural gaps (READ THIS — it shapes every task)

These are the mismatches between what the model does and what vLLM expects,
ordered by severity. Each notes the Phase-1 workaround.

1. **Batch = 1 (BIGGEST GAP).** Every `DeepSeekV4Model` entry point handles a
   single token for a single user. vLLM decode passes `tokens[max_batch_size,1]`
   and expects all users advanced in one call.
   - **Phase 1 workaround**: In the wrapper, loop over the active users and run
     `activate_session(sid)` + `decode_traced(token, pos)` once per user, stacking
     the per-user logits into `[B,1,V]`. Correct, and each step is a trace replay,
     but still O(batch) slower than a real batched kernel. Map vLLM's request slot
     index → session id (`self.slots`).

2. **KV cache ownership / shape mismatch (HARDEST PROBLEM).** vLLM's
   `allocate_kv_cache` assumes a *uniform per-layer paged KV* of shape
   `(max_num_blocks, num_kv_heads, block_size, head_size)`. V4 has **three**
   layer types:
   - `sliding_attention`: paged sliding KV — *does* map to vLLM paging.
   - `compressed_sparse_attention` / `heavily_compressed_attention`: their KV is
     paged too, but on a *combined* logical axis (sliding ring blocks followed by
     `max_seq // compress_rate` compressed-entry blocks) with its own per-group
     block size, so the per-layer geometry is not uniform. The small compressor
     **window** state is a separate dense per-session buffer, not per-token-block.
   - **Phase 1 workaround**: Do **not** try to hand V4's caches to vLLM. Let the
     model own all caches internally via `prepare_static_decode(...,
     num_sessions=max_num_seqs, total_tokens=...)`, one session per vLLM slot.
     Return a **minimal/placeholder** `kv_cache` from
     `allocate_kv_cache` (enough to satisfy the worker's bookkeeping) and ignore
     the `kv_cache`/`page_table` args vLLM passes into forward. First verify the
     plugin tolerates a model that self-manages caches — check
     `TTModelRunner.initialize_kv_cache` / `TTWorker` in the plugin; if it hard-
     requires a real paged buffer, allocate a tiny dummy of the requested shape
     and still route reads/writes through the internal caches.
   - **Phase 3 (real fix)**: Adopt the hybrid KV path
     (`HybridAttentionForCausalLM` in `models/tt_transformers/tt/generator_vllm.py`)
     for the sliding layers, and decide whether compressor caches become a second
     KV group or stay model-owned. This is a substantial design task; defer it.

3. **No real prefill.** Prefill is emulated by replaying decode one token at a
   time (`for pos in range(prompt_len): decode_traced(...)`).
   - **Phase 1 workaround**: `prefill_forward` loops per user, per prompt token,
     seeding the caches, and returns the final-token logits (or full per-position
     logits if vLLM needs them). Slow (O(prompt_len)) but correct.
   - **Phase 2**: add a batched/chunked prefill kernel. Note the plugin disables
     chunked prefill, and a TT step is prefill-only or decode-only.

4. **Sampling.** Model does host/on-device `argmax` (greedy). vLLM wants logits
   for host sampling by default.
   - **Phase 1**: return logits, set `supports_sample_on_device=False`.

5. **RoPE tables are precomputed to a fixed `max_seq`.** The wrapper must build
   the YaRN RoPE bundle once for `max_model_len` (see `_build_rope` in the demo
   tests) and slice per step. `max_seq` must be rounded up to a multiple of
   a length every group's block geometry tiles — use
   `paged_cache.round_context(max_seq_len, compress_rates, block_size)`.

6. **Weight loading is bespoke.** Uses `DeepseekV4WeightLoader` + a checkpoint
   dir (default
   `/home/ttuser/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-DSpark`)
   and an optional converted-tile cache (`DEEPSEEK_V4_CACHE_DIR`). Wire these via
   env vars in `initialize_vllm_model` (mirror how deepseek_v3 reads
   `DEEPSEEK_V3_HF_MODEL` / `DEEPSEEK_V3_CACHE`).

7. **Memory / device fit.** The full 43-layer bf4 stack does not fit one
   Blackhole; it is split across submeshes. Respect `use_submeshes=True` and the
   full mesh. `DEEPSEEK_V4_DECODE_LAYERS=N` caps layers for bringup on small
   meshes — use it while developing.

---

## Phase 0 — Environment & discovery (no code yet)

**Task 0.1 — Confirm the model runs standalone.** Run the existing demo (small
layer cap) to confirm the checkpoint, ttnn env, and device are healthy. This is
the ground truth the vLLM path must reproduce.
```bash
DEEPSEEK_V4_DECODE_LAYERS=4 DEEPSEEK_V4_MAX_NEW_TOKENS=8 \
DEEPSEEK_V4_CACHE_DIR=/path/to/cache \
pytest -s models/experimental/deepseek_v4_flash/tests/test_full_model_decode_demo.py
```
Verify: it generates tokens without error. Note the `mesh_device` / device_params
used (`fabric_config=FABRIC_2D, num_command_queues=2`).

**Task 0.2 — Confirm the multi-user paged path runs.** This is the Phase-1 bridge
target.
```bash
DEEPSEEK_V4_DECODE_LAYERS=4 DEEPSEEK_V4_CACHE_DIR=/path/to/cache \
pytest -s models/experimental/deepseek_v4_flash/tests/test_multi_user_paged_decode_demo.py
```
Verify: two users decode without clobbering each other.

**Task 0.3 — Clone and install the plugin.** Follow the README:
```bash
git clone -b dev https://github.com/tenstorrent/vllm.git
cd vllm
# tt-metal env must already be active (ttnn importable)
source plugins/vllm-tt-plugin/docs/install-vllm-tt.sh
python -c "import vllm_tt_plugin, ttnn; print('ok', vllm_tt_plugin.__file__)"
```
Verify: both imports succeed. Record the plugin path — you'll edit
`model_registry.py` and `platform.py` there.

**Task 0.4 — Find the exact HF architecture name.** Read the checkpoint's
`config.json` `architectures` field (e.g. `["DeepseekV4ForCausalLM"]`). The
plugin maps *this string* → the TT class. Also confirm `model_type ==
"deepseek_v4"` so `transformers` can build the config
(`DeepseekV4Config.from_pretrained`). Record the exact string.
```bash
python -c "import json;print(json.load(open('/home/ttuser/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-DSpark/snapshots/<hash>/config.json'))['architectures'])"
```

**Task 0.5 — Read the plugin's model_runner + worker.** Open
`plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py` and `worker.py` and
locate: `initialize_kv_cache`, `_prepare_model_inputs`,
`execute_with_model_input`. Confirm exactly what shapes/args are passed to
`prefill_forward` / `decode_forward`, and whether a model may self-manage KV
cache (Gap #2). Write findings into a short note; the following tasks depend on
them. **Do not skip this** — it validates the Phase-1 cache workaround.

---

## Phase 1 — Functional bringup (batch handled by user-loop, host sampling)

Goal: `offline_inference_tt.py` produces correct greedy text for one prompt, then
for a small batch, using the internal per-user caches. Correctness first, speed
later.

**Task 1.1 — Create the generator wrapper skeleton.**
File: `models/experimental/deepseek_v4_flash/tt/generator_vllm.py` (new).
Model the structure on `models/demos/deepseek_v3/tt/generator_vllm.py`. Start
with:
- Class `DeepseekV4FlashForCausalLM` (does NOT need to subclass tt-transformers
  `Generator` in Phase 1; it can be a thin standalone class, mirroring how
  DeepSeek V3 wraps its own generator). It must hold: the `DeepSeekV4Model`, the
  `lm_head` `Linear`, the tokenizer, the `hf_config`, the prebuilt `rope` bundle,
  `max_seq`, and a slot→user_id map.
- `model_capabilities = {"supports_prefix_caching": False,
  "supports_async_decode": False, "supports_sample_on_device": False}`.

**Task 1.2 — Implement `initialize_vllm_model`.**
```python
@classmethod
def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size,
                          max_seq_len, tt_data_parallel=1, optimizations=None):
```
Steps (copy patterns from the demo `_build_and_prefill` and the V3 wrapper):
1. Read checkpoint dir from env `DEEPSEEK_V4_HF_MODEL` (default the DSpark path);
   read `DEEPSEEK_V4_CACHE_DIR`. Build `DeepseekV4WeightLoader`, load
   `DeepseekV4Config.from_pretrained(loader.snapshot_dir)`, set
   `config._attn_implementation = "eager"`, load tokenizer.
2. Enforce `tt_data_parallel == 1` (raise a clear error otherwise — V4 is
   pipeline-parallel over the whole mesh, not DP). Set `use_submeshes=True`.
3. Compute `max_seq` = `round_context(max_seq_len, config.compress_rates.values(),
   block_size)` (see `test_multi_user_paged_decode_demo.py`).
4. Build the RoPE bundle with the demo's `_build_rope(config, max_seq)`.
5. Respect `DEEPSEEK_V4_DECODE_LAYERS` for `max_layers` (bringup).
6. Build `DeepSeekV4Model(config, loader, mesh_device, cache=WeightCache(...),
   weight_dtype=ttnn.bfloat4_b, max_layers=..., use_submeshes=True)` and the
   external `lm_head` `Linear` (as in the demo).
7. Call `model.prepare_static_decode(rope, max_seq, lm_head=lm_head,
   num_sessions=max_batch_size, total_tokens=max_batch_size * max_seq,
   block_size=...)`, then `open_session()` once per slot. Every device buffer
   (block pools, page tables, per-session state) is allocated here, before any
   trace is captured.
8. Return `cls(...)` storing all of the above.

Verify: instantiation runs to completion with `DEEPSEEK_V4_DECODE_LAYERS=4`
inside a tiny standalone script that calls `initialize_vllm_model` directly with
a real `mesh_device` (borrow the `mesh_device` fixture setup from the demo test).

**Task 1.3 — Implement `allocate_kv_cache`.**
```python
def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
```
Per Gap #2: the caches are already owned internally (Task 1.2 step 7). Based on
the Phase-0.5 findings:
- If the plugin allows self-managed caches, return a lightweight sentinel object
  (e.g. `self._internal_kv_marker`) that the forward methods recognize and
  ignore.
- If the plugin requires a real allocation of `kv_cache_shape`, allocate a
  minimal dummy ttnn tensor of that shape per layer to satisfy bookkeeping, but
  still route actual attention through the internal caches. Do NOT wire vLLM's
  page_table into the compressor layers.

Add a prominent comment explaining that V4's compressor caches cannot be
represented in vLLM's uniform paged shape, hence internal management.

Verify: `allocate_kv_cache` returns without error for the shape the worker passes.

**Task 1.4 — Implement `prefill_forward` (per-user token replay).**
```python
def prefill_forward(self, *args, **kwargs):
    tokens = kwargs["tokens"]          # [B, max_prompt_len], zero-padded
    prompt_lens = kwargs["prompt_lens] # [B]
    empty_slots = kwargs.get("empty_slots")
    sampling_params = kwargs.get("sampling_params")  # Phase 1: expect None
```
For each user `i` in the batch:
1. slot = `empty_slots[i]` if provided else `i`; map slot → session id, and
   `model.reset_session(sid)` — a recycled slot still holds the previous
   sequence's cache, and the new prefill restarts at position 0.
2. For `pos in range(prompt_lens[i])`: `model.activate_session(sid)` then
   `logits = model.decode_traced(int(tokens[i,pos]), pos)`; keep the last one.
3. `logits = ttnn.to_torch(logits).reshape(1, -1).float()` (`lm_head` is folded
   into the trace, so no separate matmul).
4. Track each user's `next_pos = prompt_lens[i]` for the decode phase.
Return host logits stacked to `[B, S, V]` (Phase 1 can return just the final
position per user, `[B, 1, V]`, if the plugin accepts it — confirm in
Phase 0.5 what shape `_prepare_model_inputs` expects; the V3 wrapper returns
`[B, S, V]` padded — mirror that if required).

Verify against ground truth: for a single prompt, the first generated token id
must equal the token the standalone demo produces for the same prompt/seed with
the same `DEEPSEEK_V4_DECODE_LAYERS`.

**Task 1.5 — Implement `decode_forward` (per-user loop).**
```python
def decode_forward(self, *args, **kwargs):
    tokens = kwargs["tokens"].squeeze(1)   # [max_batch_size]
    start_pos = kwargs["start_pos"]        # [max_batch_size]
    read_from_device = kwargs.get("read_from_device", True)
    sampling_params = kwargs.get("sampling_params")  # Phase 1: None
    enable_trace = kwargs.get("enable_trace", False)  # Phase 1: ignore / False
```
For each *active* user slot (ignore padding rows beyond the real batch):
1. `model.activate_session(self.slots[slot])`, then
   `logits = model.decode_traced(int(tokens[slot]), int(start_pos[slot]))`.
2. `out[slot, 0] = ttnn.to_torch(logits).reshape(-1).float()`.
Return `[B, 1, V]` host logits (host sampling). `enable_trace` is moot — every
step is already a trace replay; batched (B>1) tracing is Phase 2.

Verify: multi-step greedy decode of a single prompt through
`decode_forward` reproduces the standalone demo's token sequence
(argmax on the returned logits each step).

**Task 1.6 — Register the model in the plugin.**
In `plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_registry.py` add an entry
mapping the HF arch string from Task 0.4 (e.g. `"DeepseekV4ForCausalLM"`) to a
new `TTDeepseekV4FlashForCausalLM` that imports
`models.experimental.deepseek_v4_flash.tt.generator_vllm.DeepseekV4FlashForCausalLM`.
Follow the exact pattern used for `TTDeepseekV3ForCausalLM`. If `platform.py`
has a per-model validation/allow-list or feature gate, add V4 there too
(disable prefix caching, async decode, on-device sampling, TP/PP, spec decode,
LoRA — all unsupported in Phase 1).

Verify: `VLLM_PLUGINS=tt,tt_model_registry python -c "import vllm; from
vllm import ModelRegistry; print('DeepseekV4ForCausalLM' in
ModelRegistry.get_supported_archs() or 'registered')"` (or the plugin's own
registration check). At minimum, `platform_plugin()` selects `TTPlatform` when
ttnn is importable.

**Task 1.7 — Run offline inference end to end.**
```bash
MESH_DEVICE=<your mesh> DEEPSEEK_V4_DECODE_LAYERS=4 \
DEEPSEEK_V4_HF_MODEL=/path/to/checkpoint DEEPSEEK_V4_CACHE_DIR=/path/to/cache \
python plugins/vllm-tt-plugin/examples/offline_inference_tt.py \
  --model deepseek-ai/DeepSeek-V4-Flash-DSpark \
  --max_seqs_in_batch 1 --max_model_len 512
```
Verify: it generates coherent text for one prompt. Then bump
`--max_seqs_in_batch 2` and confirm two prompts both decode. This is the Phase-1
exit criterion.

**Task 1.8 — Full-depth smoke (all 43 layers) on a large enough mesh.** Drop
`DEEPSEEK_V4_DECODE_LAYERS` (use all layers) on a mesh that fits the bf4 stack.
Verify text quality is reasonable vs the HF reference for a short prompt.

---

## Phase 2 — Performance & tracing (optional, after Phase 1 is correct)

- **Task 2.1** (done in the model, not yet batched): per-user decode already runs
  on the **traced** path — one capture is shared by every session, with the active
  session selected by swapping the page-table rows and compressor window state
  (`activate_session`). What remains is folding the per-slot loop into a single
  B>1 traced step so a batch costs one replay instead of `B`.
- **Task 2.2**: Add on-device sampling. Reuse `decode_sampled_burst`'s
  on-device argmax; wire `sampling_params` (temperature/top_k/top_p). Flip
  `supports_sample_on_device=True` and support the `sample_on_device_mode` TT
  config.
- **Task 2.3**: Real batched prefill (avoid the O(prompt_len) replay) if the V4
  attention can support an S>1 prefill kernel; otherwise keep replay but batch
  users.
- **Task 2.4**: Implement `warmup_model_prefill` so the server warms up before
  reporting healthy (see the contract). Add `get_max_tokens_all_users` if the
  worker needs a KV-token budget.
- **Task 2.5**: Server smoke test — start `server_example_tt.py`, send a
  completion request (README "Running The Server Example").

---

## Phase 3 — Native vLLM paging (the hard, deferred design task)

Only attempt once Phases 1–2 are solid.
- **Task 3.1**: Move the **sliding_attention** layers onto vLLM-owned paged KV
  by adopting the `HybridAttentionForCausalLM` base
  (`models/tt_transformers/tt/generator_vllm.py`): implement `get_kv_cache_spec`
  (emit `FullAttentionSpec`/`SlidingWindowSpec` from `config.layer_types`),
  `allocate_kv_cache_per_layer`, and consume `page_tables_per_group` /
  `page_tables_per_layer` in `prefill_forward`/`decode_forward` (README
  "Hybrid Attention Models").
- **Task 3.2**: Decide the representation for CSA/HCA caches under vLLM. They are
  already paged model-side, but on a combined ring+compressed axis with a
  per-group block size, so either they become distinct KV cache groups with
  bespoke specs or stay model-owned. Document the choice; it affects memory
  accounting in the worker.
- **Task 3.3**: True batched (not looped) decode kernel for all three layer
  types. This likely requires changes in `tt/attention.py` and
  `tt/decoder_layer.py` to accept a batch dim > 1.

---

## Validation checklist (run at each phase)

1. **Parity vs standalone demo**: same prompt + same `DEEPSEEK_V4_DECODE_LAYERS`
   → identical greedy token ids between the demo test and the vLLM path.
2. **Multi-user isolation**: interleaved sessions produce the same tokens they
   would alone (mirror `test_multi_user_paged_decode_demo.py`; op-level dense-vs-
   paged equivalence lives in `test_paged_kv_equivalence.py`).
3. **Increasing seq lens**: `offline_inference_tt.py --test_increasing_seq_lens`.
4. **Server**: completion request returns coherent text.

## Files you will create or edit

- **Edit** `models/experimental/deepseek_v4_flash/tt/generator_vllm.py` (bulk of
  the work; Tasks 1.1–1.5 — a first cut of it exists and drives the paged
  sessions).
- **Edit** (in the `tenstorrent/vllm` clone)
  `plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_registry.py` and
  `platform.py` (Task 1.6).
- **Maybe edit** `tt/model.py` / `tt/attention.py` only in Phases 2–3 (batching,
  tracing, paging).

## Things that will bite you (pitfalls)

- **Slot ↔ session mapping**: vLLM reuses/condenses slots across requests. Keep a
  stable slot → session-id map and call `model.reset_session(sid)` when a slot is
  reassigned to a new request; that rewinds it to position 0 and returns its
  blocks to the shared pool.
- **`max_num_seqs` vs `num_sessions`**: the session slots are allocated at init
  and must be `>= max_batch_size`; size them from `max_batch_size` in
  `initialize_vllm_model`. `total_tokens` is a *shared* budget across sessions, so
  a session that outgrows what the pool has left raises `PagedCacheFull` even if
  its own `max_seq` is not reached.
- **Position accounting**: V4 uses *absolute* positions for RoPE and cache slots
  (`pos % sliding_window` for the ring). Feed vLLM's `start_pos` directly as the
  absolute position; do not re-derive.
- **`max_seq` rounding**: use `paged_cache.round_context`, which rounds to
  `lcm(TILE_SIZE, *(rate * block_size for rate in compress_rates))` — every
  compressor's entry count has to be a whole number of blocks. vLLM's
  `max_model_len` will not be pre-rounded; round it up inside the wrapper and
  make sure RoPE tables cover it.
- **bf4 weight conversion is slow**: always set `DEEPSEEK_V4_CACHE_DIR` so
  converted tiles are reused across runs.
- **Do not hand compressor caches to vLLM paging** — the model pages them itself
  on a combined ring+compressed axis, and the compressor *window* state is not
  per-token blocks at all. This is the single most common way this integration
  goes wrong.
- **Never allocate device buffers after a trace is captured.** Pools, page tables
  and per-session state are all created in `prepare_static_decode`; a later
  allocation can corrupt every capture.
