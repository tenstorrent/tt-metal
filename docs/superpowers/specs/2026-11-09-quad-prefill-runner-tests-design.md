# Quad prefill-runner hardware screen: design

Date: 2026-11-09. Branch: `mbezulj/quad-prefill-runner-tests` (off `mbezulj/prefill-orchestrator-rebased`).

## Intent

- Syseng / DC staff need a quad (4 Blackhole galaxies) screen.
- No model weights download, no weka mounts.
- Answer: does the quad work, and if not, which galaxy or link.
- Today's quad tests run full models; too heavy for this.
- Also an intermediate step toward running real models on a quad.

## Goals

- Three tests, added to `gwdt` behind `--quad` (off by default):

| Step key | Body per stage | Budget | Check |
|---|---|---|---|
| `quad:passthrough-smoke` | 2 `ttnn.clone` "layers" | fixed 3 requests, fast | every stage vs known host pattern |
| `quad:block-random-smoke` | real K2.7 L0 (dense) + L1 (MoE), random weights | ~5 min loop | every stage vs iteration 0 |
| `quad:block-random-soak` | same | 15 min default, 60 via `--soak-minutes 60` | every stage vs iteration 0 |

- Real runner path: `prefill_runner.py`, H2D producer, D2D sockets, layer acks, KV table.
- Verdict names the failing galaxy (compute) or link (k-1→k), plus chip coords.
- Each test starts on fabric `2d_torus_xy`; on FAIL/HANG it reruns on mesh `2d`.

## Non-goals

- No PCC vs a golden. Determinism and exact pattern only.
- No trace (`PREFILL_USE_TRACE=0`). Eager path.
- No KV migration, no MTP, no DFlash.
- No new CI job in this change.

## Anti-goals

- Changing shared runner behaviour for real models.
- A healthy quad reporting FAIL (false alarms kill trust).
- Needing weka, HF download, or a TTNN weight cache.

## Constraints (from the code)

- Engine: `models/demos/common/prefill/runners/prefill_runner.py`, one rank per galaxy, 8x4 mesh.
- Activations: `D2DStreamService`, `Shard(2)`(SP) x `Shard(3)`(TP); chip (r,c) talks to chip (r,c) next rank.
- Last rank's `prefill_chunk` must return None (asserted, `prefill_runner.py:791`).
- Runner reads `runtime.config.is_first_rank/is_last_rank/use_trace/first_layer_idx/num_layers`.
- Every rank count needs `build_kv_chunk_table` + `kv_migration_stages` + a layer-completion sink.
- Master rank's completion ring needs a consumer, or the runner blocks on a full ring.
- Producer reads tokens from a trace dir `metadata.json` (`token_ids`).
- `tt-run` drops shell `PREFILL_*`; set them inside the `bash -lc` string.
- Before each launch: `tools/scaleout/exabox/recover.sh`, purge `/dev/shm/*prefill*` on all hosts.
- `PREFILL_MAX_SEQ_LEN` must be a multiple of 2048 (bank alignment, SP=8).
- `/data/$USER` is shared NFS; gwdt run dir lives there.

## Decisions (user)

- Option A: fake adapters inside the real runner, not a standalone test.
- Per-stage determinism check on every stage, input and output.
- Each stage = K2.7 L0 dense + L1 MoE, shapes and seeds as `block-random`.
- Durations as in the table above; passthrough as fast as possible.
- Launched from `gwdt --quad`; torus 2D XY first, mesh 2D fallback.

## Design

### Adapters (registered in `ADAPTER_PATHS`)

- File: `models/demos/deepseek_v3_d_p/tt/runners/adapters/quad_screen.py`.
- `QuadBlockRandomAdapter(KimiK27Adapter)`, name `quad_block_random`.
- `QuadPassthroughAdapter(KimiK27Adapter)`, name `quad_passthrough`.
- Both: `weight_cache_path()` returns None.
- Both: `load_hf_config()` reads the in-repo K2.7 config (`reference/kimi_k2_7`).
- Both: `prefill_trace_default = ""`; producer gets a synthetic trace dir.
- Manifests: `tt/runners/manifests/quad_block_random.json`, `quad_passthrough.json`.
- Manifest env: `PREFILL_NUM_USERS=16`, `PREFILL_MAX_SEQ_LEN=40960`, `PREFILL_USE_TRACE=0`,
  `PREFILL_LAYER_ACK_D2H=0`, `PREFILL_TP_SHARD_KV=1`.
- `PREFILL_NUM_LAYERS = 2 x ranks`, set by the launcher.

### Runtimes

- File: `models/demos/deepseek_v3_d_p/tt/runners/quad_screen_runtime.py`.
- `QuadBlockRandomRuntime`: wraps an inner `TtPrefillRuntime`.
  - Inner built as a middle stage: `is_first_rank=False`, `is_last_rank=False`, `kv_only_last_layer=False`.
  - So every stage runs L0+L1 on a hidden state; no embedding, no norm.
  - Inner `model_cfg`: K2.7 config with `NUM_DENSE_LAYERS = first_layer_idx + 1`.
    Makes global layer 2r dense and 2r+1 MoE on rank r; global ack indices stay unique.
  - Inner state dict: `create_hf_model(variant, config, 2)` seed 42 →
    `{"layers": [L0, L1]}`; same weights on every rank, same as `block-random`.
  - Wrapper `config`: inner config copy with the real `is_first_rank/is_last_rank`.
  - Every rank: received input is checked, then replaced by the
    `_embedding_like_hidden` (seed 1234) slice for that chunk, pre-uploaded at compile.
    So every stage computes on exactly the `block-random` input; no random-weight chain
    (a chain drifts in scale, may trip MoE capacity, and raise false alarms).
  - Rank 0 input is token ids (H2D path); ranks k>0 input is stage k-1's output.
  - Last rank: inner returns the hidden; wrapper checks it, frees it, returns None.
  - Everything else (`kv_migration_stages`, `build_kv_chunk_table`,
    `set_layer_completion_sink`, `layer_ack_layers`, `compile`) delegates to inner.
- `QuadPassthroughRuntime(TtPrefillRuntime)`: `_build_model` builds nothing.
  - `prefill_chunk`: first rank swaps tokens for pattern; each of its layers
    does `h = ttnn.clone(h)` and fires the layer completion; returns h / None.
  - KV cache allocated (tiny) but never written; table hooks inherited.
  - Pattern per chunk: `_embedding_like_hidden`-style, seed from chunk index.

### Stage check (`StageMonitor`, same file)

- Key: `chunk = actual_start // chunk_size`; iteration counts chunk-0 arrivals.
- Reads back input (before compute) and output (after compute), per device.
- Per device via `ttnn.get_device_tensors`; device i → mesh coord (i // TP, i % TP).
- block-random: compare per device with `torch.equal` vs iteration 0.
- passthrough: compare per device vs the host pattern shard; tokens vs iteration 0.
- Iteration 0 must be finite and non-zero, else FAIL ("vacuous").
- Also records a per-device hash (xxh/sha1 of raw bytes) of input and output, per
  `(iteration, chunk)`, to `rank{r}.hashes.jsonl`. gwdt compares stage k input hashes with
  stage k-1 output hashes: catches a link that corrupts the same way every time.
- Progress markers: `last_in = (iter, chunk)` after input readback,
  `last_out = (iter, chunk)` after output readback.
- Writes `$PREFILL_QUAD_RESULTS_DIR/rank{r}.json` after each chunk (atomic rename):
  `rank, host, mode, fabric, chunks_done, iterations, last_progress, mismatches[] (capped 20), first_bad`.

### Verdict (gwdt side)

- Reads all `rank*.json`; earliest failing stage wins.

| Stage k input | Stage k output | Verdict |
|---|---|---|
| OK | OK | stage k PASS |
| OK | differs | FAIL galaxy k (host) compute, chips (r,c) |
| differs, k-1 output OK | — | FAIL link k-1→k, chips (r,c) |
| differs, k = 0 | — | FAIL host→galaxy 0 (H2D) |
| k input hash ≠ k-1 output hash | — | FAIL link k-1→k, chips (r,c) (same-iteration check) |
| k-1 `last_out` ahead of k `last_in` | — | HANG link k-1→k |
| k `last_in` ahead of k `last_out` | — | HANG galaxy k compute |
| stage 0 never got input | — | HANG host→galaxy 0 (H2D) |

- HANG attribution: first stage (in pipeline order) matching a HANG row.
- Missing rank file after launch → FAIL "rank k never started".
- Link verdict = logical link k-1→k + destination chip coords. Not the physical cable.
- Table: one sub-row per stage: `stage k  host  STATUS  iters  note`.

### Producer

- One small change: `PREFILL_PRODUCER_DRAIN_ACKS=1` consumes the completion ring
  without PCC; fails if received ≠ expected.
- Tokens: launcher writes `<run>/tokens/metadata.json` with seeded random ids;
  `PREFILL_TRACE_DIR` points there.
- block-random: `PREFILL_PRODUCER_CHUNKS=2`, `MAX_REQUESTS` large, `DURATION_S` = budget.
- passthrough: `MAX_REQUESTS=3`.
- `PREFILL_SEND_SHUTDOWN=1` so ranks exit cleanly.

### gwdt `--quad`

- `galaxy_workload_deployment_test.py --quad --hosts h0,h1,h2,h3 [--soak-minutes N]`.
- `--hosts` order = pipeline order = physical ring order. Required.
- If `/data/scaleout_configs/bh_glx_exabox/{cabling,deployment}_descriptor.textproto` exist,
  gwdt runs `tools/scaleout/exabox/resolve_host_ring_order.py`; a different order → exit 2
  with the right order printed (no FAIL verdict). Missing descriptors → warn, trust `--hosts`.
- gwdt runs on any host of the allocation; uses `ssh` for shm purge and resets,
  `mpirun -np 1 --host <rank-0 host>` for the producer (rank 0 from the tt-run rankfile).
- `--quad` replaces the single-galaxy steps; it does not add to them.
- 1 host allowed (single-rank run, for bring-up on one galaxy).
- New module `scripts/gwdt_quad.py`: step builders, launcher, verdict.
- Steps: `quad:ring-order`, `quad:reset` (gates), then the three tests.
- Recovery: see "Open: recovery entry point".
- `quad:passthrough-smoke` blocks the rest on FAIL/HANG.
- Each test: reset → shm purge → `ttrun` runner (bg) → wait all ranks
  "setup complete" → producer via `mpirun -np 1` on rank-0 host → wait → verdict.
- MGD: 4 hosts `ci/sc4_mgd.textproto` (torus) / `pipeline_prefill_4galaxy_connected_fabric2d_...` (mesh);
  1 host `ci/sc1_mgd.textproto`.
- Fallback: `with_fallbacks` style; sub-row "fallback 1 PREFILL_FABRIC_MODE=2d".
  Step status = torus result (as `with_fallbacks` today); a mesh pass shows in the sub-row.
  Every test starts on torus. Passthrough blocks the rest only if both modes fail.
- Watchdog: no rank progress for 300 s, or setup > 45 min → HANG; kill runner.
- `TT_METAL_OPERATION_TIMEOUT_SECONDS=120` on ranks so a device hang raises.
- `status.json` keeps the per-stage rows for later tooling.

## Open: recovery entry point

- `recover.sh` says: on Exabox use the `recover-hosts` helper, not the script.
- `recover-hosts` is interactive-shell only; a gwdt subprocess cannot call it.
- Default until decided: operator runs `recover-hosts` before `gwdt --quad`;
  gwdt resets each host with `tt-smi -glx_reset_auto` over ssh before each test.

## Testing plan (no quad until tomorrow)

| What | Where |
|---|---|
| Verdict logic, rank-file parsing, command building | laptop unit tests (`test_gwdt_quad.py`) |
| Adapters + runtimes, 1 rank | one galaxy, `gwdt --quad --hosts <1 host>` |
| 4-stage passthrough plumbing | one galaxy, 4 ranks x 2x4 (`intragalaxy_4rank`), manual |
| 4 galaxies, cross-galaxy D2D | tomorrow on the quad |

## Calls I made (check these)

- Stage 0 swaps tokens for the `block-random` hidden; no embedding layer.
- No norm / LM head anywhere.
- `PREFILL_MAX_SEQ_LEN=40960` (2 chunks); `block-random` uses a 56320 cache.
- KV cache not compared; output covers compute, next chunk reads KV.
- Passthrough failure blocks the block-random tests.
- Fallback runs on any FAIL/HANG, not only bring-up failures.
- Watchdog thresholds: 300 s no progress, 45 min setup.
- Every stage computes on the seeded block-random hidden, not on the received one.
- Per-device hashes for the cross-stage link check.
- Ring order checked from descriptors when present.

## Left to the builder

- Exact readback batching; result-file write cadence.
- Log line formats; table column widths.
