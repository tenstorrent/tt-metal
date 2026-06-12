# Kimi K2.6 chunked-prefill — runner perf investigation handoff

Self-contained context to resume this work on any machine. Repo: `tt-metal`.
Box used: `bh-glx-d04u02` (Blackhole 8×4 mesh). User: ppopovic@tenstorrent.com.

## Branches
- Work branch: `ppopovic/chunked_prefill_runner_integration_rebased` — Kimi chunked prefill (tests +
  runner + DEVICE_FP32 grouped-topk gate for n_groups==1). All Kimi block/transformer tests AND
  runner+producer PASS. Committed, not pushed.
- Investigation branch: **`ppopovic/investigation`** (PUSHED to origin, off the work branch). Holds the
  env-gated perf experiment toggles + full notes. View:
  https://github.com/tenstorrent/tt-metal/tree/ppopovic/investigation
  - `bfafc0a3a12` experiment toggles · `8061e7b54fd` notes
  - Full writeup: `models/demos/deepseek_v3_d_p/tt/runners/RUNNER_PERF_INVESTIGATION.md`

## The open question
Request-loop **runner** prefills a 5120-tok chunk in **~3.3 s**; the no-PCC **transformer test** does the
same 61-layer chunk in **~1.94 s**. Both: Kimi K2.6, mesh 8×4, DEVICE_FP32 gate, capacity_factor=8,
identical expert dtypes, both call `TtPrefillTransformer.forward_chunk`. Gap = **~constant ~1.4 s
additive per-chunk, prefix-independent**.

## Status: ALL measurement-side hypotheses RULED OUT (each env-gated, all stayed ~3.3 s)
1. `PREFILL_PREFILL_SYNC=1` — synchronize_device AFTER forward_chunk → no change
2. `PREFILL_DISABLE_LAYER_ACK=1` — skip per-layer ack callback → no change (`inject` is a cheap atomic)
3. `PREFILL_NUM_USERS=1` (vs 2) → no change
4. producer `PREFILL_STANDALONE_ITERS=2` (cold vs warm, repeated shapes) → no change ⇒ NOT compile
5. `PREFILL_PRESYNC=1` — synchronize_device BEFORE the timer `_t0` → no change ⇒ NOT leaked socket/copy

⇒ The ~1.4 s is **genuine `forward_chunk` compute**, not a measurement artifact. MLA attention is
`logical_n`-bounded in both paths (`mla.py:649` ring_mla; `ring_joint_sdpa_program_factory.cpp:346,379`),
so it should ramp identically — yet the runner is flat. Only remaining construction diff:
**`mla_seq_len` = 61440 (runner = `pipeline.config.max_seq_len`) vs 56320 (test `SEQ_CACHE=55*1024`)**.
Suspect: the `ring_mla` fused all-gather into `_chunked_kv_buf` touches the full ring buffer regardless
of `logical_n` (`mla.py:226-229`).

## NEXT STEPS (not yet done — start here)
1. **Decisive, cheap:** rerun the runner with `PREFILL_MAX_SEQ_LEN=56320` (= 11*5120, valid multiple of
   chunk_size) to match the test's SEQ_CACHE. If per-chunk drops → `mla_seq_len` is the cause. ONE env var.
2. Per-section instrument `forward_chunk` (gated): perf_counter+synchronize around embed / per-layer /
   MLA(`ring_mla`+gather) vs MoE. Localizes the 1.4 s. Check if the gather scales with mla_seq_len.
3. tracy per-op diff (signposts exist: `forward_chunk_layer_N_start/end`, `MoE_END`).

## How to run the runner + producer (request loop, no PCC, longbook 11×5120)
From `/home/ppopovic/tt-metal`, `source python_env/bin/activate`. Mesh is shared — first check free:
`pgrep -af "pytest|prefill_runner"`. Runner takes ~5-6 min to compile before it logs
`[h2d] exported descriptor`; start the producer only after that.

```bash
# Terminal A — runner (Kimi, request loop, NO PCC → runs until SIGTERM, logs per-iter prefill ms).
# Add experiment toggles as needed (see list above). Example with all controls:
env -u DEEPSEEK_V3_HF_MODEL -u TT_DS_PREFILL_TTNN_CACHE -u TT_DS_PREFILL_HOST_REF_CACHE \
    -u PREFILL_REQUEST_LOOP_PCC -u PREFILL_STANDALONE -u PREFILL_STANDALONE_CHUNKED \
PREFILL_MODEL_VARIANT=kimi_k2_6 \
PREFILL_SP=8 PREFILL_TP=4 PREFILL_NUM_LAYERS=61 \
PREFILL_MAX_SEQ_LEN=61440 PREFILL_CHUNK_SIZE=5120 PREFILL_NUM_USERS=1 \
PREFILL_IS_BALANCED=0 PREFILL_H2D_SERVICE_ID=ds_prefill PREFILL_STANDALONE_CHUNKED_SLOT=0 \
  python -m models.demos.deepseek_v3_d_p.tt.runners.prefill_runner

# Terminal B — producer (sources longbook trace, pushes 11×5120 into the slot).
# PREFILL_REQUEST_LOOP_PCC=1 here ONLY selects the trace token source (independent of runner PCC).
PREFILL_REQUEST_LOOP_PCC=1 PREFILL_STANDALONE_CHUNKED_NCHUNKS=11 PREFILL_STANDALONE_CHUNKED_SLOT=0 \
DEEPSEEK_PREFILL_TRACE_DIR=/mnt/models/deepseek-prefill-cache/golden/kimi-26/kimi_longbook_56320/vllm-kimi-k26-b783c42e-56321tok \
PREFILL_SP=8 PREFILL_TP=4 PREFILL_MAX_SEQ_LEN=61440 PREFILL_CHUNK_SIZE=5120 PREFILL_NUM_USERS=1 \
PREFILL_IS_BALANCED=0 PREFILL_H2D_SERVICE_ID=ds_prefill PREFILL_H2D_CONNECT_TIMEOUT=120 \
PREFILL_STANDALONE_ITERS=1 \
  python -m models.demos.deepseek_v3_d_p.tt.runners.prefill_h2d_producer
```
Read per-iter timing from the runner log: `grep "pipeline.prefill() =" <runner.log>`.

## Fast reference: the no-PCC transformer TEST (the ~1.94 s path)
```bash
pytest models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_kimi_prefill_transformer_chunked_no_pcc \
  -k "L61 and chunks11 and iters20 and kimi and mesh-8x4"
```

## Gotchas
- Plain request loop runs until SIGTERM; it blocks on the (N+1)th `h2d_socket_sync`. SIGTERM → exit 144 +
  a defunct zombie that holds NO device (mesh still free).
- **Stuck-process hazard:** killing a runner mid-init while another is starting races the chip lock
  `CHIP_IN_USE_*_PCIe` and crashes the new one in `Cluster::start_driver`. Kill the old one and confirm
  it's a zombie BEFORE launching the next. `tt-smi -s` checks device health. Never delete `TT_UMD_LOCK.*`
  shm files (normal persistent robust-mutexes).
- DEVICE_FP32 gate needs the n_groups==1 C++ kernel: after any `ttnncpp` rebuild,
  `cp -f build_Release/ttnn/_ttnncpp.so build_Release/lib/_ttnncpp.so` (python loads the lib/ copy).
- Tests need Kimi env exports (HF model + caches); the RUNNER needs only `PREFILL_MODEL_VARIANT=kimi_k2_6`.

## Logs (LOCAL to bh-glx-d04u02 only — all numbers are durably recorded in RUNNER_PERF_INVESTIGATION.md)
`/tmp/kimi_runner_{nopcc,exp,2pass,presync}.log` + matching `*producer*` logs.
