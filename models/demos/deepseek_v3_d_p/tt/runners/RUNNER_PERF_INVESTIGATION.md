# Kimi prefill runner per-chunk perf investigation (2026-06-12)

Branch: `ppopovic/investigation` (forked from `ppopovic/chunked_prefill_runner_integration_rebased`).
Experiment code committed in `bfafc0a3a12 "Runner investigation"` (all changes env-gated, default OFF).

## The question

The **request-loop runner** (`prefill_runner.py` + `prefill_h2d_producer.py`) prefills a 5120-token
chunk in **~3.3 s/chunk**, but the **no-PCC transformer test**
(`test_prefill_transformer_chunked.py::test_kimi_prefill_transformer_chunked_no_pcc`) does the same
61-layer, 5120-token chunk in **~1.94 s/chunk**. Both run Kimi K2.6, mesh 8×4, DEVICE_FP32 gate,
`capacity_factor=8`, identical expert dtypes. Why is the runner ~1.4 s/chunk slower?

Both paths ultimately call `TtPrefillTransformer.forward_chunk` (embed → 61 blocks, no lm_head).
- Test times: `forward_chunk` + `synchronize_device` + deallocates (`test_..._chunked.py:647-659`).
- Runner times: `pipeline.prefill()` = `forward_chunk` + deallocates (`prefill_runner.py:431-438` →
  `tt_deepseek_prefill_pipeline.py:213-222`).

## Key signature

The gap is a **roughly CONSTANT ~1.4 s additive per-chunk overhead, independent of prefix length**:
- logical_n=5120 (chunk 0): test 1.74 s vs runner 3.20 s → **+1.46 s**
- logical_n=56320 (chunk 10): test 2.28 s vs runner 3.38 s → **+1.10 s**

The runner barely ramps (+150 ms over the full kv range); the test ramps +600 ms. A large fixed cost
dominates the runner and masks the attention-prefix growth. MLA attention IS prefix-bounded
(`logical_n = kv_actual_isl + chunk_size`) in BOTH paths — confirmed in `mla.py:649` (`ring_mla`,
`logical_n=kv_actual_isl + chunk_size_global`) and `ring_joint_sdpa_program_factory.cpp:346,379`
(`logical_nt` derived from `logical_n`, not the allocated buffer). So `mla_seq_len` only sizes storage,
not compute — yet the runner is flat. The fixed overhead is therefore NOT attention.

## Experiments run — ALL NEGATIVE (runner stayed ~3.3 s/chunk)

Each is an env-gated toggle in `bfafc0a3a12`. Runner env baseline (Kimi, request loop, no PCC):
`PREFILL_MODEL_VARIANT=kimi_k2_6 PREFILL_SP=8 PREFILL_TP=4 PREFILL_NUM_LAYERS=61
PREFILL_MAX_SEQ_LEN=61440 PREFILL_CHUNK_SIZE=5120 PREFILL_NUM_USERS=1 PREFILL_IS_BALANCED=0
PREFILL_H2D_SERVICE_ID=ds_prefill PREFILL_STANDALONE_CHUNKED_SLOT=0` (NO `PREFILL_REQUEST_LOOP_PCC`
→ no PCC, runs until SIGTERM). Producer sources longbook via `PREFILL_REQUEST_LOOP_PCC=1` +
`PREFILL_STANDALONE_CHUNKED_NCHUNKS=11`, `DEEPSEEK_PREFILL_TRACE_DIR=/mnt/models/deepseek-prefill-cache/golden/kimi-26/kimi_longbook_56320/vllm-kimi-k26-b783c42e-56321tok`.

| # | Hypothesis | Toggle | Result |
|---|---|---|---|
| 1 | timer measures dispatch-only, not completion | `PREFILL_PREFILL_SYNC=1` (synchronize_device AFTER forward_chunk, in `pipeline.prefill`) | **no change** ~3.33 s — runner was already effectively waiting |
| 2 | per-layer LayerAck callback serializes dispatch | `PREFILL_DISABLE_LAYER_ACK=1` (skip `set_layer_ack_channel`) | **no change** — `inject` is a cheap shm atomic (`counter_channel.cpp:99`) |
| 3 | per-user cache scaling | `PREFILL_NUM_USERS=1` (vs 2) | **no change** |
| 4 | per-shape program compile (compile() pre-warms only one logical_n) | producer `PREFILL_STANDALONE_ITERS=2` → 22 chunks, pass 2 repeats shapes | **no change** — pass 2 == pass 1 (slightly higher), so NOT compile |
| 5 | timer charges leaked h2d_socket_sync chunk-copy / prior tail | `PREFILL_PRESYNC=1` (synchronize_device BEFORE the timer `_t0`, in `run_request_loop`) | **no change** ~3.3-3.5 s — overhead is genuinely inside forward_chunk |

### Raw per-iter `pipeline.prefill()` ms (DEVICE_FP32, slot, 61 layers, 5120 tok)

WITH-PCC runner (earlier baseline, num_users=2, slot 1):
3262, 3246, 3273, 3379, 3383, 3392, 3381, 3452, 3430, 3416, 3412 — mean ~3300, total ~36 s.

NO-PCC runner (num_users=2, slot 1): 3262.45, 3245.69, 3273.44, 3379.16, 3383.08, 3392.40, 3380.65,
3452.35, 3429.80, 3416.24, 3412.26 — mean ~3366. (== with-PCC → PCC overhead is post-loop, not timed.)

sync+no-ack+1user (num_users=1, slot 0): 3200.07, 3203.82, 3250.41, 3340.48, 3332.21, 3353.19,
3358.81, 3365.49, 3350.24, 3406.46, 3379.77 — mean ~3326.

2-pass cold/warm (num_users=1): pass1 3124,3137,3265,3202,3252,3287,3322,3325,3320,3339,3274;
pass2 3354,3368,3370,3411,3411,3413,3432,3395,3428,3471,3433. Warm == cold → not compile.

presync (pre+post sync, num_users=1, 2 pass): pass1 3173,3294,3251,3328,3343,3484,3395,3432,3548,3456,3418;
pass2 (kv resets to 0) 3435,3487,3447,3469,3479,3450,3477,3466,3484,3582,3501. Warm chunk-0 still 3435 ms.

### No-PCC TRANSFORMER TEST reference (the fast path), warm iters, 5120 tok/chunk, 61 layers
Per-chunk ramps within an iter: chunk0 ~1.68-1.84 s → chunk10 ~2.28 s, mean ~1.94 s. iter0 chunk0 =
15.9 s one-time compile spike, then warm. Command:
`pytest models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_kimi_prefill_transformer_chunked_no_pcc -k "L61 and chunks11 and iters20 and kimi and mesh-8x4"`

## Conclusion so far

The ~1.4 s/chunk gap is **genuine `forward_chunk` compute** in the runner's `pipeline.model`, NOT a
measurement artifact, compile, ack, num_users, sync placement, or attention length. Constructions are
near-identical (same dtypes/capacity/gate/slot_num/is_balanced/is_chunked/seq_len). The ONLY remaining
construction difference is `mla_seq_len` = 61440 (runner, `pipeline.config.max_seq_len`) vs 56320
(test, `SEQ_CACHE = 55*1024`) — but static analysis says attention is `logical_n`-bounded so this should
not cost 1.4 s. Open suspicion: a per-chunk fixed cost tied to the allocated KV ring buffer
(`mla_seq_len`) — e.g. the `ring_mla` fused all-gather into `self._chunked_kv_buf`
(`mla.py:226-229,649`) may touch the full buffer regardless of `logical_n`.

## NEXT STEPS (not yet done)

1. **Run the runner with `PREFILL_MAX_SEQ_LEN=56320`** (= 11*5120, a valid multiple of chunk_size) to
   match the test's `SEQ_CACHE`. If per-chunk drops, `mla_seq_len` (ring-buffer-proportional work in the
   gather) is the cause. Cheapest decisive next test — ONE env var, no code change.
2. **Per-section instrument `forward_chunk`** (gated): perf_counter + synchronize around (a) embed,
   (b) per-layer, and within a block split MLA (`ring_mla` + its all-gather) vs MoE. This localizes the
   ~1.4 s definitively. Check whether the `ring_mla` all-gather / `_chunked_kv_buf` population scales
   with `mla_seq_len` rather than `logical_n`.
3. Profile both paths under tracy (signposts already emitted: `forward_chunk_layer_N_start/end`,
   `MoE_END`) and diff per-op device time. If device-op time is equal, the gap is host-side dispatch.

## Operational notes / gotchas

- Mesh is shared (8×4); only one runner/test at a time. Check: `pgrep -af "pytest|prefill_runner"`.
- Runner ~5-6 min to load weights + compile before it logs `[h2d] exported descriptor`. Start the
  producer only after that, or bump `PREFILL_H2D_CONNECT_TIMEOUT`.
- Plain request loop (no PCC) runs until SIGTERM; it blocks on the (N+1)th `h2d_socket_sync`. SIGTERM
  interrupts that C++ blocking call → exit 144 + a defunct zombie that holds NO device (mesh still free).
- **STUCK-PROCESS HAZARD**: killing a runner mid-init while another is starting can race the chip lock
  `CHIP_IN_USE_*_PCIe` and crash the new one in `Cluster::start_driver`. If a runner won't die, it may
  be orphaned (PPID 1, `Sl`) holding the lock; `kill -KILL` it and wait for it to become a zombie BEFORE
  launching the next. `tt-smi -s` confirms the device is healthy. `TT_UMD_LOCK.*` shm files are normal
  persistent robust-mutexes — do not delete them.
- DEVICE_FP32 gate needs the n_groups==1 C++ kernel: after any `ttnncpp` rebuild,
  `cp -f build_Release/ttnn/_ttnncpp.so build_Release/lib/_ttnncpp.so` (python loads the lib/ copy).

## Local logs on bh-glx-d04u02 (NOT reachable off-box — numbers above are the durable record)
`/tmp/kimi_runner_nopcc.log`, `/tmp/kimi_runner_exp.log` (sync+noack+1user),
`/tmp/kimi_runner_2pass.log`, `/tmp/kimi_runner_presync.log` and matching `*producer*` logs.

## RESOLVED (2026-06-14) — root cause: the H2D stream service

Overnight automated experiment sweep (harness: /home/ppopovic/kimi_perf_overnight, branch
ppopovic/investigation). Per-chunk means (5120 tok, 61 layers, DEVICE_FP32):
- request-loop runner: ~3090 ms/chunk
- standalone-chunked (same pipeline, NO H2D service): ~1878 ms/chunk
- no-PCC transformer test: ~1940 ms/chunk

The ~1.2 s/chunk gap is **request-loop machinery, NOT forward_chunk compute**. Eliminated, each by a
dedicated env-gated experiment:
- mla_seq_len / KV-buffer size (H3): sweep MAX_SEQ_LEN 56320/61440/81920/102400 = 3089/3094/3215/3273 ms
  — only a weak ~6% secondary effect, not the gap.
- per-layer LayerAck synchronize_device: PREFILL_SKIP_ACK_SYNC and DISABLE_LAYER_ACK both ≈ baseline;
  section timing with ack off still fully elevated.
- construction/config: construction-dump byte-identical between request and standalone paths.
- request-mode-only clear_loaded_sub_device_manager (prefill_runner.py line ~586): PREFILL_FORCE_PRECLEAR
  added that clear to the standalone path → stayed ~1879 ms (INNOCENT).

By elimination (and corroborated by the request-without-ack run staying elevated in BOTH mla and moe
sections), the cause is the **H2D stream service** (`build_h2d_service`) running in-process for the whole
prefill: its presence (reserved worker cores / resident init program / background socket-sync on the
shared command queue / host dispatch contention) slows every forward_chunk op ~40%, uniformly across
mla and moe. (PREFILL_FORCE_BUILD_SERVICE builds the unused service in the standalone path as a direct
positive confirmation.)

**Fix is runner-side, not model-side.** Options: run the H2D service on a separate command queue from
the model; shrink/relocate H2D_SYNC_WORKER_CORES off the model's compute grid; or make the service's
background sync passive/event-driven so it doesn't steal dispatch cycles during forward_chunk. Profile
with tracy to pin which (cores vs CQ vs host dispatch thread).

### Fix attempt (2026-06-14): core relocation — NEGATIVE
Moved the H2D service's persistent receiver kernel off the model compute grid (worker core (0,0) ->
(11,0); BH grid is 12x10, col 11 free): per-chunk UNCHANGED (~3190 vs ~3201 ms at col 0). So the
slowdown is NOT the kernel occupying a compute core. The overhead is ~proportional to op count across
both mla and moe => a per-dispatch / sub-device-coexistence cost of the resident service program.
Next: tracy-profile exp 10 (standalone+FORCE_BUILD_SERVICE) vs 01a (no service) — diff per-op device
time vs inter-op gaps — to decide between (a) separate command queue for the service, (b) sub-device
placement that avoids per-launch re-coordination, (c) suspend the service program during forward_chunk.
Some are H2DStreamService (C++) changes. Diagnostic knob PREFILL_H2D_WORKER_{COL,ROW} committed.
