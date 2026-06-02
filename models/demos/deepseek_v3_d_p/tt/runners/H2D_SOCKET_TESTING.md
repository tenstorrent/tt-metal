# H2D Socket + Metadata — Testing

How to exercise the cross-process H2D streaming path: an external producer pushes
token IDs + per-iter control metadata over a shared `H2DStreamService`, and the
prefill runner reads them on-device via `h2d_socket_sync`.

All commands assume the venv is active (`source python_env/bin/activate`) and the
galaxy is reset first (`tt-smi -glx_reset_auto`).

Shared env (weights/config caches — point at the pre-populated /mnt caches):

```bash
export TT_DS_PREFILL_TTNN_CACHE=/mnt/models/DeepSeek-R1-0528-Cache/DeepSeek-R1-0528-Cache-prefill_secure
export DEEPSEEK_V3_HF_MODEL=/mnt/models/deepseek-ai/DeepSeek-R1-0528
export TT_DS_PREFILL_HOST_REF_CACHE=/mnt/models/deepseek-prefill-cache/golden/
export PREFILL_GATE_FALLBACK_MODE=DEVICE_FP32
```

---

## 1. Cross-process socket test (runner reads tokens + metadata from a producer)

Two processes. The runner builds the service, exports a descriptor to
`/dev/shm/tt_h2d_stream_service_<id>.bin`, and blocks on `h2d_socket_sync`; the
producer attaches via `H2DStreamService.connect()` and pushes.

**Terminal A — runner (consumer).** Builds the model (~10 min) then exports the
descriptor and waits:

```bash
tt-smi -glx_reset_auto
PREFILL_STANDALONE=1 \
PREFILL_H2D_EXTERNAL_PRODUCER=1 \
PREFILL_STANDALONE_ITERS=3 \
python -m models.demos.deepseek_v3_d_p.tt.runners.prefill_runner
```

Wait for the log line: `[h2d] exported descriptor service_id='ds_prefill' -> /dev/shm/...`.

**Terminal B — producer (sender).** Deviceless; connects to the descriptor and
pushes `PREFILL_STANDALONE_ITERS` times:

```bash
PREFILL_STANDALONE_ITERS=3 \
python -m models.demos.deepseek_v3_d_p.tt.runners.prefill_h2d_producer
```

Notes:
- **`PREFILL_STANDALONE_ITERS` must match** on both sides (runner waits for N
  syncs; producer pushes N).
- The producer must match the runner's token packing:
  `PREFILL_SP` / `PREFILL_TP` / `PREFILL_MAX_SEQ_LEN` / `PREFILL_IS_BALANCED` /
  `PREFILL_H2D_SERVICE_ID`. Defaults already match — only override in lockstep.
- Launch the producer *after* the runner prints "exported descriptor"
  (`PREFILL_H2D_CONNECT_TIMEOUT`, default 60 s, bounds how long the producer
  waits for the descriptor file).

Expected: the runner logs, per iter,
`[standalone] iter=i metadata: actual_isl=1021 slot_id=0 dst_slot=-1` and a
`first_token=...`, then `Shutdown complete` with a clean (exit 0) teardown.

> **Generated tokens are NOT reproducible run-to-run** (prefill output is
> non-deterministic — same input gives different first tokens across processes).
> Don't use `first_token` to judge correctness; use the equivalence test below.

---

## 2. Input-equivalence test (the real "socket doesn't degrade anything" proof)

Fast, single-process, no model build. Proves the socket delivers byte-identical
model input by comparing the `h2d_socket_sync` output against the tensor
`_prepare_input_tensor` would build for the same tokens (all 32 device shards),
plus a metadata round-trip check.

```bash
tt-smi -glx_reset_auto
pytest -xvs models/demos/deepseek_v3_d_p/tests/test_h2d_input_equivalence.py
```

Expected: `1 passed`, with
`[equiv] PASS: 32 device shards byte-identical (socket == non-socket input)` and
`[equiv] PASS: metadata round-trip [actual_isl,slot_id,dst_slot]=[1021, 0, -1]`.

---

## 3. Non-socket baseline (optional, for comparison)

Same model path, tokens pushed straight through `pipeline.prefill(token_ids=...)`
with no H2DStreamService. Useful to demonstrate the run-to-run token
non-determinism is independent of the socket.

```bash
tt-smi -glx_reset_auto
PREFILL_STANDALONE=1 \
PREFILL_H2D_EXTERNAL_PRODUCER=0 \
PREFILL_STANDALONE_ITERS=3 \
python -m models.demos.deepseek_v3_d_p.tt.runners.prefill_runner
```
