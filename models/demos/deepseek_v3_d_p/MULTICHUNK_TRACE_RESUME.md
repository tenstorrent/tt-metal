# Multi-chunk ttnn-trace prefill (metadata-driven) — RESUME CONTEXT

Branch: `ppopovic/trace_experiments`. Date: 2026-06-26. Status: **DONE + VERIFIED on 8×4 (L1 + L61
pass). Changes UNCOMMITTED.** Author: Pavle Popovic.

---

## Goal

Make ONE captured ttnn device trace of the Kimi/DeepSeek chunked-prefill `transformer.forward()` replay
correctly across **11 chunks**, and capture per-chunk timings. A trace freezes every host-baked arg, so
the per-chunk scalars (`slot_id`, `actual_start`, `actual_end`) cannot be host args — they must be read
on-device from a small **metadata DRAM tensor** that we update in-place between replays. The four
chunked-prefill MLA device ops were already made trace-safe (read those scalars on-device from the
metadata tensor); this task wired the metadata tensor **through the transformer/block layers** and
**through the trace capture/replay loop**.

Metadata tensor = replicated uint32 ROW_MAJOR DRAM `[slot_id, actual_start, actual_end, 0]` (the
runner's h2d_socket_sync payload; trailing 0 pads to 4 words). See [[kimi-prefill-metadata-ops]] and
`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/TRACEABLE_METADATA_PATH.md`.

---

## What was changed (4 files, Python only — NO C++/kernel rebuild needed)

### 1. `models/demos/deepseek_v3_d_p/tt/tt_prefill_transformer.py`
- `forward(...)`: added param `metadata: Optional[ttnn.Tensor] = None`.
- **CRITICAL FIX (rope gate, ~line 303):** changed `if actual_start is not None:` to
  `if actual_start is not None or metadata is not None:` — the metadata path passes `actual_start=None`,
  and without this it would wrongly fall to single-shot rope instead of `self.indexed_rope`.
- Passes `metadata=metadata` into the per-layer `layer(...)` call.

### 2. `models/demos/deepseek_v3_d_p/tt/tt_prefill_block.py`
- `forward(...)`: added `metadata: Optional[ttnn.Tensor] = None`; passes `metadata=metadata` into
  `self.mla.forward(...)`.

### 3. `models/demos/deepseek_v3_d_p/tt/mla/mla.py`
- `forward(...)` already accepted `metadata`. The gap was the **kv_only last layer**: `forward` routes
  to `_forward_kv_only(...)` when `self.kv_only`, and the test uses `kv_only_last_layer=True`, so the
  LAST layer hit a scalar-only path. Fixed:
  - `forward` now passes `metadata=metadata` into the `_forward_kv_only(...)` call.
  - `_forward_kv_only(...)`: added `metadata` param; uses metadata variants of `_apply_rope`,
    `update_padded_kv_cache`, and `zero_padded_kv_cache` (mirrors the dual paths in `_chunked_attn`).

### 4. `models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py`
- `run_chunked_transformer_no_pcc(...)`: added `use_metadata=False` param.
- New branch `if use_trace and use_metadata:` (the legacy chunk-0 scalar trace path became `elif
  use_trace:`):
  - Persistent buffers created ONCE (never realloc): `trace_input` (sharded uint32 tokens) +
    `trace_metadata` (replicated uint32 `[0,0,CHUNK,0]`, `ttnn.ReplicateTensorToMesh`).
  - Pre-built per-chunk HOST tensors `tok_host_tt[c]` / `meta_host_tt[c]` (`[0, c*CHUNK, c*CHUNK+CHUNK,
    0]`).
  - Capture once via `_forward_meta()` (actual_start=None, actual_end=None, metadata=trace_metadata).
  - Replay loop: per chunk `ttnn.copy_host_to_device_tensor(tok_host_tt[c], trace_input)` +
    `copy_host_to_device_tensor(meta_host_tt[c], trace_metadata)` (in-place, cq 0), then
    `controller.replay()` (`SubDeviceTraceController`, execute_trace cq 0 blocking → ordered after
    copies). Records `per_chunk_seconds`.
  - Perf JSON dump extended with `per_chunk_seconds` + `avg_per_chunk_seconds`.
- New test fn `test_kimi_prefill_transformer_chunked_trace_multichunk` — params: `n_chunks=[11]`,
  `num_iters=[1,2]`, `num_layers=[1,10,61]`, 8×4 mesh (l1_small_size=512, trace_region_size=256MB),
  kimi variant. Calls runner with `use_trace=True, use_metadata=True`.

---

## How to run (uses PREBUILT weights — no download)

The TEST fixtures use different env-var names than the runner. Set these (both prebuilt on the box):

```bash
export TT_KIMI_PREFILL_TTNN_CACHE=/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill
export KIMI_K2_6_HF_MODEL=/mnt/models/kimi-forge   # MUST be dot-free (see gotcha)
```

Sanity (1 layer, ~15s):
```bash
TT_KIMI_PREFILL_TTNN_CACHE=/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill \
KIMI_K2_6_HF_MODEL=/mnt/models/kimi-forge \
python_env/bin/python -m pytest \
  "models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_kimi_prefill_transformer_chunked_trace_multichunk[blackhole-kimi-mesh-8x4-L1-iters1-chunks11]" -s
```

Full timing (61 layers, ~6 min; dump timings):
```bash
TT_KIMI_PREFILL_TTNN_CACHE=/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill \
KIMI_K2_6_HF_MODEL=/mnt/models/kimi-forge \
TT_PREFILL_PERF_JSON=/tmp/multichunk_L61.json \
python_env/bin/python -m pytest \
  "models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_kimi_prefill_transformer_chunked_trace_multichunk[blackhole-kimi-mesh-8x4-L61-iters1-chunks11]" -s
```

If an eth core wedges (JIT/segfault): `tt-smi -glx_reset` (30s; re-inits 32 boards). Kill any pytest
holding the device first.

---

## Verified results (8×4 Blackhole, 2026-06-26)

- **L1**: PASS. 1 trace segment (0.09 MB) captured once, 11 replays ~0.4 ms each.
- **L61**: PASS. **119 trace segments / 51.14 MB captured ONCE**, 11 replays = **14.26 s total**, per
  chunk **1149 → 1458 ms (monotonic +27%)**, avg 1295 ms. The monotonic growth is the proof the
  metadata is read on-device per chunk (causal attention over growing KV); a frozen chunk-0 trace would
  be flat ~1149 ms. `1 passed in 377.74s`.

---

## Gotchas (cost real time)

1. **Rope gate** (`tt_prefill_transformer.py:303`) — the one real correctness trap; metadata path uses
   actual_start=None and must still select `indexed_rope`.
2. **kv_only last layer** — `_forward_kv_only` is a separate code path that the `kv_only_last_layer=True`
   transformer hits; it was initially missed and must also be metadata-aware.
3. **In-place update only** — never rebuild `trace_input`/`trace_metadata` with `from_torch(device=...)`
   inside the loop; that reallocates and the replayed trace reads a stale/freed address. Use
   `ttnn.copy_host_to_device_tensor(host_tt, device_tt)`.
4. **Dot-free HF dir** — `KIMI_K2_6_HF_MODEL=/mnt/models/moonshotai/Kimi-K2.6-dequantized` FAILS:
   `ModuleNotFoundError: No module named 'transformers_modules.Kimi-K2'` (HF trust_remote_code truncates
   the dynamic module name at the `.`). Use `/mnt/models/kimi-forge` (dot-free; has config.json +
   configuration_deepseek.py + index.json + modeling_*). Only index.json + config.json + the config .py
   modules are read at runtime; weights come from the TTNN cache (shards not loaded when
   check_cache_complete is True).
5. **Test vs runner env names** — test fixtures: `TT_KIMI_PREFILL_TTNN_CACHE` + `KIMI_K2_6_HF_MODEL`
   (conftest.py / model_variants.py KIMI_V2_6). Runner (prefill_runner.py): `PREFILL_TTNN_CACHE` +
   `PREFILL_HF_MODEL` with RunnerVariant defaults (runner_utils.py). Same assets, different names.

---

## Remaining / optional follow-ups

- **Commit** the 4-file change (currently uncommitted on `ppopovic/trace_experiments`).
- Run the other parametrizations: `L10`, and the `two_iters` variants (repeat the 11-chunk sequence).
- **KV-cache PCC under multichunk trace** (correctness, not just timing): after the 11-chunk replay the
  KV cache holds tokens 0..11*CHUNK; could add a PCC vs the golden `kv_post_transform` (golden trace at
  `VARIANT.prefill_trace_default = /mnt/models/deepseek-prefill-cache/golden/kimi-26/kimi_longbook_56320`,
  loader is the chunked_group_a_v1 layout). The per-op bit-exact metadata==scalar equivalence is already
  proven; this would close the loop at transformer scale.
- **Runner**: `prefill_runner.py run_request_loop` can now drop the `to_torch` deconstruction of
  `tt_metadata` (from `inbound_socket_service_sync`) and pass it straight into
  `transformer.forward(metadata=...)`.

---

## Key file:line anchors

- Trace branch: `test_prefill_transformer_chunked.py` — `run_chunked_transformer_no_pcc`, the
  `if use_trace and use_metadata:` block; new test fn at the bottom.
- Rope gate: `tt_prefill_transformer.py` `forward`, `if actual_start is not None or metadata is not None`.
- kv_only metadata: `mla.py` `_forward_kv_only`.
- Trace controller: `models/demos/deepseek_v3_d_p/utils/sub_device_trace.py` (`SubDeviceTraceController`,
  cq_id=0, begin/end_capture, replay).
- Metadata tensor construction reference: `tests/test_mla.py` (use_metadata_tensor variant) and
  `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::_make_ring_mla_metadata`.

Related memory: [[kimi-prefill-trace]], [[kimi-prefill-metadata-ops]], [[kimi-prefill-env-vars]],
[[ttnn-so-refresh-procedure]].
