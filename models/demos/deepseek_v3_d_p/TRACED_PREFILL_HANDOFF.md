# Traced (metadata-driven) chunked prefill — work handoff

Self-contained context to resume this work on another machine. Branch: **`ppopovic/trace_experiments`**.
Repo root assumed `/home/ppopovic/tt-metal` (adjust paths to your checkout). Hardware: 8×4 Blackhole
galaxy (32 devices).

Companion docs in this dir (read for depth): `NIGHT_REPORT.md` (chronological log of the whole effort),
`ring_mla_metadata_perf_conclusion.md` (perf analysis), `MULTICHUNK_TRACE_RESUME.md`,
`tests/perf/CHUNKED_PREFILL_PERF.md`, and
`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/TRACEABLE_METADATA_PATH.md`.

---

## 1. Goal / what this work delivers

Make ONE captured ttnn device trace of the Kimi/DeepSeek chunked-prefill `transformer.forward()` replay
correctly across all 11 chunks (collapsing the per-op host-dispatch "op2op" tax). A trace freezes every
host-baked arg, so the per-chunk scalars `[slot_id, actual_start, actual_end]` must be read **on-device**
from a small **metadata DRAM tensor** (replicated uint32 `[1,1,1,4]`) updated in place between replays —
NOT passed as host args. The four chunked-prefill MLA ops (`update_padded_kv_cache`,
`rotary_embedding_indexed`, `zero_padded_kv_cache`, `ring_mla`) read their per-chunk scalars on-device
from that tensor.

Delivered this effort:
1. **Correctness at scale** — 11-chunk metadata trace replay writes a KV cache matching the golden
   longbook trace (L10 + L61).
2. **The ring_mla per-layer cache-slot bug fix** (the headline; see §3).
3. **Productionized runner** — `prefill_runner.py` replays a captured trace (gated by `PREFILL_USE_TRACE`),
   incl. the per-layer migration ack chopped out of the trace; explicit `pipeline.capture_trace()` API.
4. **Padded (variable/partial-chunk) trace tests** for DeepSeek + Kimi, asserting traced==untraced.
5. **ring_mla metadata-vs-scalar perf characterization** (genuine fixed ~3µs/call, <1% at realistic KV).

---

## 2. Commit map (this work, on top of `1002cdf04e8`)

```
8922fe94cea  explicit pipeline.capture_trace() (after compile) — replaces lazy capture-in-prefill
fbef83114bf  test_kimi_prefill_transformer_chunked_padded_trace (Kimi padded trace twin)
c8785fd30d6  ring_mla metadata perf characterization (conclusion.md + micro-bench tooling)
648d4fb1f95  ring_mla steady-state micro-bench
f33bb21ec3d  verify DS padded trace test (L1+L10 metadata==untraced bit-exact)
d94777fcd03  ring_mla per-call perf (all eq tests) + DS padded trace test
d76184eb61d  Phase D — request-loop e2e traced prefill over H2D socket (KV PCC 0.994 + ack chopping)
5e53ede2b37  Phase E — ring_mla device kernel time metadata vs scalar (line+ring, 32 devices)
1419c4a4c14  Phase C — gated metadata-trace path in pipeline + runner (PREFILL_USE_TRACE)
ce16113e0cb  Phase B verified (L61 11-chunk KV PCC) + Phase C trace-controller ack foundation
10cea053409  *** ring_mla per-layer cache-slot FIX (the key bug fix) ***
3fb9d4573c7  WIP — 11-chunk metadata KV-PCC test scaffold + root-cause notes
```
(Pre-existing context below `1002cdf04e8`: the `ring_mla task1-5` + `update_padded_kv_cache` /
`rotary_embedding_indexed` / `zero_padded_kv_cache` metadata work + `Segmented ttnn trace`.)

Commits used `git commit --no-verify` (pre-commit reformats C++/Python); **run `pre-commit run` before
pushing / opening a PR.**

---

## 3. THE KEY BUG + FIX (commit 10cea053409) — read this first

**Symptom:** 11-chunk metadata-trace KV-cache PCC collapsed with layer depth (L0 nope ~0.9999 → L8 ~0.11)
on the METADATA path only; the scalar path was perfect. Reproduced even single-chunk + at L10 single
layer was fine.

**Root cause:** the KV cache batch dim is **(user, layer)-major**:
`cache_batch_idx = cache_user_id*num_layers + cache_layer_idx`. The SCALAR ring_mla passed that full
index as `kv_cache_batch_idx`. The METADATA ring_mla read the slot as `metadata[0]` = `slot_id` =
`cache_user_id` **only** — missing `*num_layers + cache_layer_idx`. So every layer > 0 read **layer-0's
KV** → wrong attention → compounding drift. `num_layers`/`layer_idx` existed nowhere in the ring_joint /
all-gather code. (`update_padded_kv_cache` already took `layer_idx`+`num_layers` and computed the flat
slot, so the cache was *written* correctly — only `ring_mla`'s *read* was wrong.) Single-layer test_mla
(slot 0 → 0*N+0=0) and the per-op equivalence tests (single slot) could not catch it.

**Fix:** added `kv_cache_num_layers` (default 1) + `kv_cache_layer_idx` (default 0) to `ring_mla`,
threaded through the `ring_joint_sdpa` device op + program factory + the fused all-gather helper into
BOTH readers; the slot is now `meta[0]*kv_cache_num_layers + kv_cache_layer_idx`. Defaults reduce to
`meta[0]` → all existing callers + the bit-exact equivalence tests stay identical. `mla.py`
`_chunked_attn` metadata branch passes `kv_cache_num_layers=self.layer_num,
kv_cache_layer_idx=cache_layer_idx`.

---

## 4. Architecture / where things live

### Metadata trace controller (per-layer ack chopping)
`utils/sub_device_trace.py::SubDeviceTraceController` — captures the forward as multiple trace segments,
splitting at MoE sub-device load/clear AND (NEW) at each per-layer migration ack. A host shm-bump ack
cannot live inside a trace, so at replay the controller fires the ack callback *between* the two trace
segments (after the first segment's KV writes flush). API: `set_layer_ack_callback()`, `has_layer_ack()`,
`layer_ack(layer_idx)` (capture splits w/o injecting; eager calls cb; replay injects).
`mla.py` both ack sites (`_chunked_attn`, `_forward_kv_only`) route through it when the controller carries
an ack callback, else direct sync+call (test path: controller has no ack cb → unchanged). The controller
is threaded transformer→block→MLA (and MoE) via `set_trace_controller`.

### Pipeline API (commit 8922fe94cea — the explicit flow)
`tt/tt_deepseek_prefill_pipeline.py`:
- `config.use_metadata_trace` (separate from the legacy chunk-0-pinned `config.use_trace`).
- `compile()` — build + warmup.
- **`capture_trace()`** — EXPLICIT, call after `compile()` (and, for the request loop, after
  `set_layer_ack_channel()`). Allocates persistent `_trace_input` + `_trace_metadata`, runs a metadata
  warmup forward (JITs the metadata-program variants), then captures the segmented forward. Idempotent;
  no-op unless `use_metadata_trace`.
- `prefill(input_tensor, slot_id, actual_start, actual_end)` — on the metadata-trace path: `ttnn.copy`'s
  the fresh inbound tokens into the held `_trace_input` (so inbound persistence is automatic — no
  socket-op change needed), writes `[slot,start,end,0]` into `_trace_metadata`, then `controller.replay()`.
  Asserts the trace was captured (no lazy capture).

### Runner (gated)
`tt/runners/prefill_runner.py`: `PREFILL_USE_TRACE=1` → `config.use_metadata_trace=True` +
`open_mesh_device(trace_region_size=256MB)`. `main()` calls `pipeline.capture_trace()` after `compile()`
(standalone) and after `set_layer_ack_channel()` (request loop). `runner_utils.py::open_mesh_device`
gained a `trace_region_size` param.

### Files changed (vs `1002cdf04e8`)
- **ring_mla op (C++, needs rebuild):** `ttnn/cpp/.../transformer/sdpa/sdpa.{hpp,cpp}`,
  `sdpa_nanobind.cpp`, `.../sdpa/device/ring_joint_sdpa_device_operation.{hpp,cpp}`,
  `ring_joint_sdpa_device_operation_types.hpp`, `ring_joint_sdpa_program_factory.cpp`.
- **kernels (JIT-recompiled, no rebuild):** `.../sdpa/device/kernels/dataflow/ring_joint_reader.cpp`,
  `.../experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_all_gather_reader.cpp`
  (+ that op's `..._multi_core_with_workers_program_factory.{cpp,hpp}` — host, rebuild).
- **model/runner (Python):** `tt/mla/mla.py`, `tt/tt_prefill_block.py`,
  `tt/tt_deepseek_prefill_pipeline.py`, `tt/runners/prefill_runner.py`, `tt/runners/runner_utils.py`,
  `utils/sub_device_trace.py`.
- **tests/tools (Python):** `tests/test_prefill_transformer_chunked.py` (11-chunk metadata trace_kv_pcc,
  DS+Kimi padded-trace tests, `_record_kv_cache_pcc` gained `assert_layer_depth`/`return_per_layer` +
  `KV_PE_DEBUG` diag), `tests/perf/{ring_mla_metadata_perf.py, ring_mla_eq_perf.py,
  test_ring_mla_microperf.py, ring_mla_microperf_driver.py}`.

---

## 5. Environment variables

```bash
# --- pytest fixtures (test_prefill_transformer_chunked.py, test_mla.py) — Kimi ---
export KIMI_K2_6_HF_MODEL=/data/nbabin/Kimi-K2_6-dequantized        # dot-free dir (HF trust_remote_code)
export TT_KIMI_PREFILL_TTNN_CACHE=/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill
export TT_KIMI_PREFILL_HOST_REF_CACHE=/mnt/models/kimi-prefill-cache/golden
# golden longbook trace (kv_post_transform + tokens): /mnt/models/deepseek-prefill-cache/golden/kimi-26/kimi_longbook_56320

# --- pytest fixtures — DeepSeek (for the DS padded-trace test) ---
export DEEPSEEK_V3_HF_MODEL=/mnt/models/deepseek-ai/DeepSeek-R1-0528
export TT_DS_PREFILL_TTNN_CACHE=/mnt/models/DeepSeek-R1-0528-Cache/DeepSeek-R1-0528-Cache-prefill_secure
# DS golden: /mnt/models/deepseek-prefill-cache/golden/longbook_qa_eng_prefill_56320_nopad (variant default)

# --- runner (prefill_runner.py / prefill_h2d_producer.py) ---
# PREFILL_MODEL_VARIANT=kimi_k2_6  PREFILL_TTNN_CACHE=<...prefill>  PREFILL_HF_MODEL defaults to in-tree
# PREFILL_TRACE_DIR=<golden>  PREFILL_USE_TRACE=1  PREFILL_NUM_LAYERS  PREFILL_NUM_USERS  PREFILL_MAX_SEQ_LEN
# PREFILL_STANDALONE / PREFILL_STANDALONE_{NCHUNKS,PCC,ITERS,SLOT}  PREFILL_REQUEST_LOOP_PCC
# PREFILL_STANDALONE_CHUNKED_NCHUNKS  DEEPSEEK_PREFILL_TRACE_DIR (request-mode validator; see gotcha)
```
Run python as `PYTHONPATH=. python_env/bin/python ...` from the repo root. Paths above are this box's
mounts; on a fresh machine point them at your copies of the same assets (HF config dir, TTNN weight
cache, golden trace).

---

## 6. Run commands

### (a) Transformer trace + metadata KV-cache PCC (the core correctness test)
```bash
export KIMI_K2_6_HF_MODEL=/data/nbabin/Kimi-K2_6-dequantized \
       TT_KIMI_PREFILL_TTNN_CACHE=/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill \
       TT_KIMI_PREFILL_HOST_REF_CACHE=/mnt/models/kimi-prefill-cache/golden
# 11-chunk metadata trace, KV PCC vs golden (asserts L0..GATED_LAYER_DEPTH=10, records deeper):
python_env/bin/python -m pytest "models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_kimi_prefill_transformer_chunked_trace_kv_pcc[blackhole-kimi-mesh-8x4-L10-chunks11]" -s
# also: ...-L61-chunks11  (full model)  and ...-L1-chunks1 / ...-L10-chunks1 (smoke/isolation)
```

### (b) Padded (variable/partial-chunk) trace test — traced == untraced (bit-exact)
```bash
# Kimi (DEVICE_FP32 gate, MoE):
python_env/bin/python -m pytest "models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_kimi_prefill_transformer_chunked_padded_trace[blackhole-kimi-mesh-8x4-L1-full55k]" -s   # L1/L10/L61
# DeepSeek (needs DEEPSEEK_V3_HF_MODEL + TT_DS_PREFILL_TTNN_CACHE):
python_env/bin/python -m pytest "models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_ds_prefill_transformer_chunked_padded_trace[blackhole-deepseek_v3-mesh-8x4-L10-full55k]" -s
```
PASS = `max |untraced - traced| per-layer KV PCC = 0.00e+00`.

### (c) Runner STANDALONE loop, traced + KV PCC (single process, file input)
```bash
env PREFILL_MODEL_VARIANT=kimi_k2_6 \
    PREFILL_TTNN_CACHE=/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill \
    PREFILL_TRACE_DIR=/mnt/models/deepseek-prefill-cache/golden/kimi-26/kimi_longbook_56320 \
    PREFILL_NUM_LAYERS=10 PREFILL_NUM_USERS=1 PREFILL_MAX_SEQ_LEN=56320 \
    PREFILL_STANDALONE=1 PREFILL_STANDALONE_NCHUNKS=11 PREFILL_STANDALONE_PCC=1 \
    PREFILL_STANDALONE_ITERS=2 PREFILL_USE_TRACE=1 \
    PYTHONPATH=. python_env/bin/python -m models.demos.deepseek_v3_d_p.tt.runners.prefill_runner 2>&1 | tee /tmp/kimi_standalone.log
# flow: compile() -> capture_trace() (17 seg/7.31MB @ L10) -> loop replays. ITERS=2 gives clean
# steady-state per-iter timing (capture is a separate step now). Expect KV PCC 0.994096 PASS.
```

### (d) Runner REQUEST loop (producer + socket) traced + KV PCC (two processes)
```bash
pkill -9 -f prefill_runner; pkill -9 -f prefill_h2d_producer
rm -f /dev/shm/tt_h2d_stream_service_ds_prefill.bin /dev/shm/tt_prefill_layer_acks_ds_prefill
# RUNNER (background). DEEPSEEK_PREFILL_TRACE_DIR must be the vllm SUBDIR (see gotcha):
nohup env PREFILL_MODEL_VARIANT=kimi_k2_6 \
  PREFILL_TTNN_CACHE=/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill \
  PREFILL_TRACE_DIR=/mnt/models/deepseek-prefill-cache/golden/kimi-26/kimi_longbook_56320 \
  DEEPSEEK_PREFILL_TRACE_DIR=/mnt/models/deepseek-prefill-cache/golden/kimi-26/kimi_longbook_56320/vllm-kimi-k26-b783c42e-56321tok \
  PREFILL_NUM_LAYERS=10 PREFILL_NUM_USERS=1 PREFILL_MAX_SEQ_LEN=56320 \
  PREFILL_USE_TRACE=1 PREFILL_REQUEST_LOOP_PCC=1 PREFILL_STANDALONE_CHUNKED_NCHUNKS=11 \
  PYTHONPATH=. python_env/bin/python -m models.demos.deepseek_v3_d_p.tt.runners.prefill_runner > /tmp/kimi_runner.log 2>&1 &
until grep -q "entering request loop" /tmp/kimi_runner.log; do sleep 5; done
# PRODUCER (pushes 11 longbook chunks):
env PREFILL_MODEL_VARIANT=kimi_k2_6 \
  PREFILL_TRACE_DIR=/mnt/models/deepseek-prefill-cache/golden/kimi-26/kimi_longbook_56320 \
  PREFILL_STANDALONE_NCHUNKS=11 PREFILL_NUM_USERS=1 PREFILL_STANDALONE_SLOT=0 \
  PYTHONPATH=. python_env/bin/python -m models.demos.deepseek_v3_d_p.tt.runners.prefill_h2d_producer > /tmp/kimi_producer.log 2>&1
grep -E "KV cache PCC|all .* slot.* PASSED|Shutdown complete" /tmp/kimi_runner.log   # expect PASSED 0.994096
```

### (e) ring_mla perf (metadata vs scalar)
```bash
# per-call steady-state (clean), DRAM vs L1 metadata, scaling sweep:
PYTHONPATH=. python_env/bin/python models/demos/deepseek_v3_d_p/tests/perf/ring_mla_microperf_driver.py 3
# production (real Kimi dims, 8x4, 32 devices, worst-device, line+ring):
PYTHONPATH=. python_env/bin/python models/demos/deepseek_v3_d_p/tests/perf/ring_mla_metadata_perf.py
# per-call across all 5 ring_mla metadata equivalence tests (old scalar vs new metadata):
PYTHONPATH=. python_env/bin/python models/demos/deepseek_v3_d_p/tests/perf/ring_mla_eq_perf.py
```

### Per-op bit-exact equivalence (correctness guard after any kernel/op change)
```bash
python_env/bin/python -m pytest "tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_mla_metadata_matches_scalar_rotation" "tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_mla_metadata_matches_scalar_indexed" -q
```

---

## 7. Verified results (8×4 Blackhole)
- 11-chunk metadata trace KV PCC: **L10 min 0.994096**; **L61** asserted-L0-10 min 0.9935 / full-61 min 0.967.
- Standalone runner `PREFILL_USE_TRACE=1` L10/11-chunk: KV PCC **0.994096**.
- Request-loop e2e (producer+socket, ack chopping engaged): KV PCC **0.994096**, clean shutdown.
- Padded trace (DS L1+L10, Kimi L1): traced == untraced **bit-exact** (|diff|=0.00e+00 per layer).
- ring_mla metadata vs scalar perf: **<1% at realistic KV** (chunk 5120 → +0.72%); fixed ~3µs/call;
  production worst-device mean line +0.9–1.2% / ring +0.4–0.9% (weighted by the small first chunk). See
  `ring_mla_metadata_perf_conclusion.md`.

---

## 8. Build / rebuild notes
- **Kernel `.cpp` edits** (the dataflow kernels) are JIT-recompiled at runtime — just re-run, NO cmake.
- **Host-side C++ edits** (device op `.cpp/.hpp`, program factory, sdpa.cpp, nanobind) need a rebuild:
  `cmake --build build_Release --target ttnncpp` and (for nanobind signature changes) `--target ttnn`,
  then refresh `.so`s (see `ttnn-so-refresh-procedure`): cp `build_Release/ttnn/_ttnncpp.so`→`lib/`,
  `tt_metal/libtt_metal.so`→`lib/`, `build_Release/ttnn/_ttnn.so`→`ttnn/ttnn/`. Verify `import ttnn` from
  repo root with `PYTHONPATH=.`.
- On a fresh machine the build may need llvm-20 (apt or the `~/.lldshim` ar shim).

## 9. Known gotchas
- **Request-mode validator dir:** it reads `DEEPSEEK_PREFILL_TRACE_DIR` (NOT `PREFILL_TRACE_DIR`) and needs
  the `vllm-kimi-...` SUBDIR (where `kv_cache/` + `metadata.json` live), e.g.
  `.../kimi_longbook_56320/vllm-kimi-k26-b783c42e-56321tok`. `ls` the trace dir to find the subdir.
- **Dot-free HF dir:** `KIMI_K2_6_HF_MODEL` must be dot-free (HF `trust_remote_code` truncates the dynamic
  module name at the first `.`). `/data/nbabin/Kimi-K2_6-dequantized` is fine; a `Kimi-K2.6` path is not.
- **Device wedge recovery:** a JIT-compile failure / TT_FATAL can wedge an eth core or leave a pytest
  holding `CHIP_IN_USE`. `pkill -9 -f prefill` (or the test) then `tt-smi -glx_reset` (~30s, re-inits 32
  boards). Always kill the holder first.
- **Persistent buffers:** never realloc `_trace_input`/`_trace_metadata` per chunk — replay would read a
  freed/stale address. Update in place (`ttnn.copy` / `copy_host_to_device_tensor`).
- `git commit --no-verify` was used (pre-commit reformats); run `pre-commit run` before pushing.
- `*.log` is gitignored — the perf logs (`ring_mla_*.log`) are regenerable artifacts, not committed; the
  numbers live in `ring_mla_metadata_perf_conclusion.md` / `NIGHT_REPORT.md`.

## 10. Remaining / optional follow-ups
- Push the branch + open a PR (after `pre-commit run`). Nothing is pushed yet.
- ring_mla perf is already <1% at realistic per-call KV; getting the production worst-device *mean*
  solidly <1% would need reducing the multi-core metadata reads (read-once-and-share, or a 16B
  multicast) — assessed low-ROI (see conclusion.md). Left as a documented follow-up.
- L61 request-loop + full migration WORKER (the scheduler side) not exercised; the runner fires the
  per-layer acks correctly (counter), but no worker consumes them here.
- `KV_PE_DEBUG=1` env on `_record_kv_cache_pcc` dumps candidate pe layouts + per-chunk nope (debug aid).
