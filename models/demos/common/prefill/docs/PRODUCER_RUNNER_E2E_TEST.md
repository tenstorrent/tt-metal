# `test_producer_runner_e2e` — runbook

Automated **Gate 1** of the prefill KV-migration ladder (see `PREFILL_MIGRATION_TESTING.md` for the manual
three-gate reference). One pytest, two processes, one Blackhole galaxy, no decode and no migration
endpoint.

| | |
|---|---|
| **Test** | `models/demos/common/prefill/tests/test_producer_runner_e2e.py::test_producer_runner_pcc` |
| **Parametrized by** | scenario name (5 scenarios; see [Scenarios](#scenarios)) |
| **Needs** | 1 Blackhole galaxy (32 ASICs, SP8×TP4), a built `ttnn`, weight cache + golden trace on disk |
| **Runtime** | ~4 min (shallow scenarios) to ~37 min (GLM-5.2 full depth) |

---

## 1. What question this test answers

> **Is the KV chunk address table the runner publishes correct?**

During disaggregated serving, a *migration worker* in another process is handed only two artifacts — a
serialized `KvChunkAddressTable` (protobuf) and a device map (JSON) — and must locate, read and interpret
every token block of every layer of every user slot straight out of device DRAM. If any address in that
table is wrong, the migrated KV is garbage and the decode host produces nonsense.

This test replaces the migration worker with the **producer**, which does the same `read_dram_umd` reads
over a bare UMD cluster and PCCs the result against the golden trace. Everything downstream of the table
(endpoint, worker, DCN transport, MPI) is removed, so a failure points at the table or the cache layout and
nothing else.

### Failure classes it catches

- wrong device — the `fabric_node_id → unique_id` resolution in the device map
- wrong DRAM NoC address / bank within a device
- wrong **block-cyclic rotation** — the chunk-local → natural token order un-rotation
- wrong per-layer or per-slot stride (slots are user-major: `slot = user_id * num_layers + layer_idx`)
- wrong **config** in a multi-config table, including the compacted-rank ↔ global-layer mapping
  (GLM-5.2's index cache stores only the 21 `full` layers, renumbered 0..20, while the golden trace is
  numbered by global layer — see [Scenarios](#scenarios) note 4)
- wrong dtype/layout decode (bfp8 tile vs bf16 row-major vs scaled fp8)

It also incidentally exercises the H2D socket, the per-layer LayerAck protocol, and the runner's ability
to prefill `L` layers × `N` chunks at all.

### What it does NOT answer

- **Gate 0's question — does prefill compute the correct KV in the first place?** A PCC failure here is
  *ambiguous* between "the table is wrong" and "the KV in DRAM was already wrong". Always have a Gate 0
  run at the **same** layer count and sequence length as the control. See
  [Interpreting a failure](#7-interpreting-a-failure).
- **Gate 2's question — does the real DRAM→DCN→DRAM copy work?** Nothing here touches the transport.

---

## 2. How it works

Three processes, all spawned by pytest:

```
pytest (this test)
 ├── runner   : python -m models.demos.common.prefill.runners.prefill_runner    [holds the 8x4 mesh]
 │              PREFILL_MOCK_MIGRATION=1
 │              1. builds the model, loads weights, JITs kernels
 │              2. allocates the KV cache(s) via the model adapter
 │              3. builds + serializes the KV chunk address table  -> /tmp/ci_prefill_kv_table.pb
 │              4. writes the device map                           -> /tmp/ci_prefill_kv_devmap.json
 │              5. publishes the H2D descriptor  -> /dev/shm/tt_h2d_stream_service_ci_ds_prefill.bin
 │              6. serves chunk pushes until signalled
 │
 └── producer : python -m models.demos.common.prefill.runners.prefill_producer  [DEVICE-LESS]
                1. connects to the H2D descriptor
                2. pushes N chunks of golden token ids
                3. drains NUM_LAYERS x N LayerAcks
                4. imports the table (ttnn.experimental.disaggregation.import_from_protobuf_file)
                5. reads the cache back with read_dram_umd, decodes, PCCs vs the golden trace
                6. exit(1) if any slot is below threshold
```

pytest spawns the runner, **polls for all three rendezvous files** to appear (that is the readiness
signal — there is no handshake), runs the producer under a timeout, asserts `returncode == 0`, then tears
the runner down with SIGINT → 120 s → SIGKILL.

**The read-back is device-less on purpose.** Opening a ttnn device in the producer would take the
`CHIP_IN_USE` lock the runner already holds and deadlock. That is why it goes through UMD directly and
decodes raw device bytes in Python (`_decode_kv_chunk` and friends in `prefill_producer.py`).

---

## 3. Prerequisites

**Hardware.** A whole Blackhole galaxy, exclusively. `PREFILL_SP=8` × `PREFILL_TP=4` = 32 devices, and a
single dead ASIC blocks every device open with no reduced-mesh fallback. Confirm nobody else is on the box
before starting.

**Build.** `ttnn` must be built and must expose `ttnn.experimental.disaggregation.import_from_protobuf_file`
(binding in `ttnn/cpp/ttnn-nanobind/disaggregation.cpp`). If it is missing the producer logs
`import_from_protobuf_file missing — rebuild ttnn` and silently skips the whole read-back — a rebuild is
required, not optional.

**Assets** (paths as configured on the Tenstorrent dev boxes; override via env if yours differ):

| What | Default | Override |
|---|---|---|
| Model | `PREFILL_MODEL`, else `DEFAULT_MODEL = "deepseek_v3_d_p"`. The GLM scenario pins `glm_5_2`. | `PREFILL_MODEL` |
| HF config + weights | the selected adapter's `hf_model_default` (GLM-5.2: `/mnt/models/deepseek-prefill-cache/GLM-5.2-FP8`) | `PREFILL_HF_MODEL` |
| Tilized weight cache | the adapter's `ttnn_cache_default` (GLM-5.2: `/mnt/models/deepseek-prefill-cache/glm52_ttnn_cache`) | `PREFILL_TTNN_CACHE` |
| Golden trace | the adapter's `prefill_trace_default`; **the GLM scenario pins its own** (see note 4) | `PREFILL_TRACE_DIR` |

Scenarios 1–3 carry no `env` block, so they run whatever `PREFILL_MODEL` is in your shell (default
`deepseek_v3_d_p`) against that adapter's default paths.

The weight-cache leaf is assembled as `$PREFILL_TTNN_CACHE/{model}_{arch}_{N}dev/{sp}x{tp}`, e.g.
`glm52_ttnn_cache/glm_5_2_bh_32dev/8x4`. With a cold cache the run additionally pays a full weight
tilization; with a cold Tensix kernel cache (`~/.cache/tt-metal-cache/<build-hash>/kernels`, invalidated
by any ttnn rebuild) it pays one full JIT warm-up forward pass.

**Shell.**

```bash
cd <tt-metal>
export TT_METAL_HOME="$PWD" PYTHONPATH="$PWD"
```

---

## 4. Running it

### Pre-flight (do this every time)

```bash
# 1. nothing else may hold the mesh
pgrep -af 'prefill_runner|prefill_producer|migration_' || echo "clean"

# 2. stale rendezvous files make the readiness poll pass INSTANTLY against a dead run
rm -f /dev/shm/tt_h2d_stream_service_ci_ds_prefill.bin \
      /tmp/ci_prefill_kv_table.pb /tmp/ci_prefill_kv_devmap.json
```

### One scenario

```bash
python_env/bin/python3 -m pytest \
  models/demos/common/prefill/tests/test_producer_runner_e2e.py::test_producer_runner_pcc \
  -k glm52_full_depth_kv_table \
  2>&1 | tee /tmp/e2e_run.log
```

`pytest.ini`'s `addopts` already carries `-vvs`, so no extra verbosity flags are needed.

### All scenarios

Drop the `-k`. Each scenario spins up its **own** runner, so you pay a full model load + JIT per scenario —
that isolation is deliberate (independent config, no cross-contamination, one crash doesn't block the rest).

### Following along while it runs

```bash
tail -f generated/test_reports/ci_runner_glm52_full_depth_kv_table.log     # model load, prefill, chunks
tail -f generated/test_reports/ci_producer_glm52_full_depth_kv_table.log   # pushes, acks, PCC
```

Both logs are also tail-echoed inline into the pytest output at teardown (wrapped in `::group::` under
GitHub Actions), and the full files land in `generated/test_reports/` so CI uploads them as artifacts.

---

## 5. Scenarios

| Scenario | Users | Layers | `max_seq_len` | Chunks | Purpose |
|---|---|---|---|---|---|
| `single_user_full_depth` | 1 | `$PREFILL_NUM_LAYERS` (default **2**) | 56320 | 11×5120 | deepest single-slot correctness gate |
| `round_robin_4users` | 4 | `$PREFILL_NUM_LAYERS` | 20480 | 4 each | deterministic interleave (u0c0,u1c0,u2c0,u3c0,u0c1,…) |
| `random_8users` | 8 | `$PREFILL_NUM_LAYERS` | 10240 | 1–2 each | seeded chaotic interleave + gaps/bursts + slot recycling |
| `glm52_full_depth_kv_table` | 1 | **78** (pinned) | 56320 | 11×5120 | GLM-5.2 DSA **merged two-config** table |
| `glm52_full_depth_kv_table_tp_sharded` | 1 | **78** (pinned) | 56320 | 11×5120 | same, with **SP×TP KV dedup** on (see note 5) |

Chunk size is fixed at `CHUNK_SIZE = 5120`. Constraints inherited from the runner:
`max_seq_len % chunk_size == 0` and `chunk_size % (SP*32) == 0`.

**Note — scenarios 1–3 are shallow by default.** They read the module-level
`NUM_LAYERS = int(os.environ.get("PREFILL_NUM_LAYERS", "2"))`, i.e. **2 layers** unless you export
`PREFILL_NUM_LAYERS`. They are transport/interleave gates, not depth gates. Only
`glm52_full_depth_kv_table` pins its own layer count.

**Note 4 — the GLM-5.2 scenario.** This is the gate for the merged two-config table:

- **config 0** — the MLA KVPE cache, **all 78 layers**, 576 wide (512 `nope` KV-LoRA latent + 64 `pe` rope)
- **config 1** — the DSA lightning-indexer KEY cache, `index_head_dim` = 128 wide, **only the 21 `full`
  layers**, compacted to dense ranks 0..20

GLM-5.2 reuses indexer selections across layers: `config.indexer_types` marks 21 of 78 layers `full`
(`{0,1,2,6,10,14,…,74}`); the other 57 are `shared` and own no indexer. So config 1's rank *r* corresponds
to global layer `full_layers[r]`, while the golden trace directory `dsa/indexer_k_layer_<n>` is numbered by
**global** layer. `_full_indexer_layer_indices()` in the producer does that translation.

Two hard requirements for this scenario, both already set in `SCENARIOS`:

- `PREFILL_KV_ONLY_LAST_LAYER=0` — the table describes all 78 layers, so the last layer must still write
  its KV; the runner's default headless-last-layer optimization would leave layer 77 empty.
- **All 78 layers, no truncation.** The index cache is sized from the model's whole `indexer_types` map
  (21 ranks), so a truncated run leaves the upper ranks unwritten. The producer asserts on that mismatch
  rather than PCC'ing untouched memory.
- `PREFILL_TRACE_DIR=/mnt/models/deepseek-prefill-cache/glm-traces/vllm-glm52-indexer-kcache-55k` — this
  trace has both `kv_cache/` (78 layers) and `dsa/` (21 layers). The adapter's own `prefill_trace_default`
  **omits `dsa/`**, which would leave config 1 with no golden and silently skip its PCC.

**Note 5 — the TP-sharded scenario (`..._tp_sharded`).** Identical run to note 4 with
`PREFILL_TP_SHARD_KV=1`, which turns on GLM-5.2 **KV dedup**: instead of each SP row's 640 tokens/chunk
being *replicated* across all 4 TP columns, every one of the 32 devices stores a **distinct** 160-token
sub-slice. Per-device cache drops 4× (7040 → 1760 tokens; ~633 → ~158 MB for the 78-layer KVPE block);
the payoff at 1M context is ~1 → ~6 concurrent users per Galaxy. Design:
`models/demos/deepseek_v3_d_p/docs/glm52_kv_cache_tp_sharding.md`.

It is the only end-to-end gate that the four pieces agree byte-for-byte:

| piece | TP-sharded form |
|---|---|
| allocation | `init_kvpe_cache(tp_axis=1)` → `seq_len // (sp*tp)` rows per device |
| write | `update_padded_kv_cache(tp_axis=1)` writes only this chip's `chunk_local/tp` window |
| model read | TP-inner **then** SP-outer all-gather, decoded with an effective `sp*tp` stripe count |
| migration table | per-`(row,col)` **singleton** device groups + a `col*160` offset |

Entry count and readback time are unchanged (`99 × 1760 = 174240`): the same 32-token blocks, just
attributed to 32 devices instead of 8 row-groups.

**Acceptance is a diff, not a threshold.** TP dedup is pure *storage* dedup — bit-identical by design — so
the check that means something is

> per-layer `nope` / `pe` / `index` PCC **equals** the `glm52_full_depth_kv_table` baseline, layer for layer

not "above some number". That is why this scenario lowers `PREFILL_STANDALONE_CHUNKED_PCC` to **0.85**: a
floor just under the known SP-only minimum (0.8608, `nope` @ layer 75 — §11) so it does **not** re-fail on
that pre-existing full-depth KVPE issue, while still printing every per-layer line for the diff. A genuinely
broken TP layout reads the wrong device or the wrong `1/tp` window and lands near zero, far below the floor.
Conveniently, this also means the TP work is **not** blocked on the missing Gate-0 control (§11): you are
comparing against a same-configuration SP baseline, not against golden absolutes.

Extra requirements beyond note 4:

- `PREFILL_KV_ONLY_LAST_LAYER=0` becomes **load-bearing twice** — besides leaving layer 77 unwritten, the
  kv-only path (`ttMLA._forward_kv_only`) has no TP-sharded write and asserts. The runner refuses the
  combination at import.
- **GLM-5.2 only.** `PREFILL_TP_SHARD_KV=1` on any other model is rejected at runner import via
  `PrefillModelAdapter.supports_tp_shard_kv` — GLM-5.1 is also sparse, so without that guard it would build
  a TP-sharded model against TP-replicated caches and corrupt silently instead of failing.
- **Not** compatible with `PREFILL_STANDALONE_PCC=1` (Gate 0) or `PREFILL_VALIDATE_MIGRATION=1` (Gate 2):
  both reconstruct the cache on the host through `read_slot_kv`, which keeps a single TP column. That is a
  full replica only when TP-replicated, so it asserts under TP sharding. The table-driven producer read-back
  *is* the TP-sharded read path.

Cheaper TP gates to run first (minutes, not ~35 min):
`test_kv_cache_table.py::test_glm52_tp_sharded_kv_cache_table` (synthetic writer + table readback, no
weights) and `test_prefill_block.py` parametrized `[tp_sharded]` (one block, real ops).

---

## 6. Reading the output

The producer log, in order. `L` = layers, `S` = real sequence length.

| Line | Meaning |
|---|---|
| `[producer] push slot=0 cidx=N start=… end=…` | one chunk pushed |
| `[producer] layer acks D/E` | ack drain progress; `E = NUM_LAYERS × total_pushes` |
| `[producer] DONE wall=…s pushes=… throughput=…` | **end of the push loop only** — the read-back has not started |
| `[producer] slot 0 layer NN KV PCC: nope=… pe=…` | config 0, one line per layer |
| `[producer] slot 0 KV PCC over [0,S) across L layers -> M` | config 0 summary |
| `[producer] slot 0 layer NN (index rank R) index PCC: …` | config 1, one line per `full` layer |
| `[producer] slot 0 index PCC over [0,S) across N layers -> M` | config 1 summary |
| `[producer] kv_cache_pcc_complete slots_checked=C min_pcc=M` | machine-readable verdict (plain `print`) |
| `[producer] KV cache PCC PASSED (min M >= T across C slots)` | **pass** |
| `[producer] KV cache PCC below T for (slot, real_len, pcc): […]` | **fail**, then `exit(1)` |

The test's own assertion message is
`producer scenario '<name>' failed (rc=1; PCC below threshold or error)`.

Sanity checks worth doing on the runner log: the table's reported `entries=` must equal
`(config-0 layers + config-1 layers) × (max_seq_len / NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK)`. For the GLM
scenario that is `(78 + 21) × (56320 / 32) = 99 × 1760 = 174240`.

### Thresholds

One env var, several read sites, **two different defaults**:

| Site | Default | Role |
|---|---|---|
| `prefill_producer.py:362` `_config_from_env()` | **0.93** | the Gate-1 gate this test enforces |
| `deepseek_v3_d_p/tt/runners/prefill_kv_validation.py:197` | **0.88** | Gate 0 standalone (DeepSeek / GLM path) |
| `prefill_runner.py:658` (config banner) | 0.88 | display only — why the runner log prints 0.88 while the producer enforces 0.93 |

(`minimax_m3` carries its own validator with the same 0.88.)

Override with `PREFILL_STANDALONE_CHUNKED_PCC=<float>`. Nothing in the test sets it. **Gate 1 is currently
stricter than Gate 0**, so a KV cache at 0.90 passes standalone and fails through the producer. Both
defaults predate any full-depth model; treat the split as an open item, not intent.

---

## 7. Interpreting a failure

The summary `min_pcc` is a single number over *all* configs and columns, which hides where the error is.
Read the **per-layer** lines instead — they localize it:

- **All columns of a layer bad, or whole layers bad while others are perfect** → suspect the table.
  Addressing and decode faults are not selective: `nope` and `pe` are columns of the *same* 576-wide cache
  row (one DRAM block, one `read_dram_umd` call, one decode), so a wrong device, wrong NoC address,
  mis-rotated block-cyclic index or bad decode corrupts **both**.
- **One column degrades while the other stays clean** → the table is fine; the numbers written to DRAM were
  already imperfect. That is a Gate 0 question, and you need a Gate 0 run **at the same depth and sequence
  length** to decide whether it is inherent (thresholds mis-calibrated) or a real write-path bug.
- **Config 1 clean out to its last layer while config 0 degrades** → also rules out accumulated activation
  drift, since the indexer keys and the KV latent derive from the same layer-input hidden states.

Caveat: PCC sensitivity is not equal across tensors of different width and dynamic range, so "the error
lives in tensor X" is a sound conclusion from these lines, but "it is only quantization noise" is not —
that needs the Gate 0 control.

---

## 8. Expected timings

Measured 2026-07-30, `glm52_full_depth_kv_table`, Blackhole Galaxy 6U, **warm** weight cache, **cold**
kernel cache. Use these to tell "slow" from "hung".

| Phase | Duration |
|---|---|
| model build + weight load | ~6 min |
| cold-JIT warm-up forward pass | ~5.5 min (skipped with a warm kernel cache) |
| service published (runner ready) | ~12 min in |
| chunk loop | **91–100 s per 5120-token chunk** → ~17 min for 11 chunks |
| layer-ack drain | ~1.25 s per layer-ack (78 × 1.25 ≈ 98 s per chunk, overlapped with the loop) |
| KV read-back | ~2.8 s per KVPE layer + ~0.4 s per index layer → ~4.5 min |
| **total** | **~37 min** |

The shallow scenarios (2 layers) finish in ~4 min each.

---

## 9. Traps

| Symptom | Cause | Fix |
|---|---|---|
| Test killed at exactly 300 s | `pytest.ini` sets a blanket `timeout = 300`, far below what the test itself waits for | `_scenario_params()` attaches a per-scenario `pytest.mark.timeout(ready + producer + 600)` at **collection** time (pytest-timeout reads the marker at setup, so it cannot be added in the test body). GLM gets 11400 s. If you add a scenario, it inherits this automatically — don't bypass `_scenario_params()`. |
| ~2 min of dead air after the PCC verdict | `PREFILL_SEND_SHUTDOWN` is never set by any test, so the runner stays in its request loop and teardown falls back to SIGINT → 120 s → SIGKILL | Expected. Not a hang. |
| "Hangs" for the whole `ready_timeout_s` (1 h for GLM) | The runner died of an exception but hung in UMD/Cluster teardown, so `proc.poll()` stays `None` and the readiness loop keeps waiting | **Always read `generated/test_reports/ci_runner_<scenario>.log` before assuming progress** — the traceback is usually already sitting there. Needs SIGKILL, not SIGTERM. |
| Readiness passes instantly, then the producer fails weirdly | Stale `/dev/shm` descriptor or `/tmp` table from a previous run | The pre-flight `rm -f`. `_cleanup_ipc()` also runs around each scenario. |
| `import_from_protobuf_file missing — rebuild ttnn` | ttnn built without / before the disaggregation binding | Rebuild ttnn. The read-back is skipped entirely otherwise. |
| Index PCC lines never appear | `PREFILL_TRACE_DIR` points at a trace with no `dsa/` | Use a trace with both `kv_cache/` and `dsa/`. |
| Assert on config-1 layer count | Layer count truncated below the model's full-indexer count | Run all layers for DSA models. |
| `RuntimeError: ARC startup error … GDDR training failure`, or `Read 0xffffffff over PCIe ID N` | A board has dropped off the bus. `ttnn.get_num_devices()` raises before any model code | Probe `/sys/class/tenstorrent/tenstorrent!N/tt_heartbeat`/`tt_aiclk`/`tt_serial` — a bad board reads all-ones while healthy neighbours don't (`tt-smi -ls` may hang; sysfs is reliable). Reset a 6U galaxy with **`tt-smi -glx_reset_auto`**, not the per-board `tt-smi -r`. Don't pass `--eth_train_skip`; a mesh run needs ethernet trained. |
| Stopped (`T`-state) leftovers ignore SIGTERM | Suspended process | `kill -CONT` first, then TERM, then KILL. |
| Under `mpirun`, pytest is green but the job exits non-zero | The runner's `MPI_Init` joined the launcher's PMIx session; it is torn down by signal and never calls `MPI_Finalize`, so prterun reports abnormal termination | Handled: `_launch_mode()` auto-detects `OMPI_*`/`PMIX_*`/`PRTE_*` and strips them from the child env so the children run as singletons. Pin with `PREFILL_RUNNER_LAUNCH=ci` or `=standard`. |

### Cleanup after a failed or interrupted run

```bash
pkill -f prefill_runner ; pkill -f prefill_producer
rm -f /dev/shm/tt_h2d_stream_service_ci_ds_prefill.bin \
      /tmp/ci_prefill_kv_table.pb /tmp/ci_prefill_kv_devmap.json
pgrep -af 'prefill_runner|prefill_producer' || echo "clean"
```

---

## 10. Useful env overrides

| Var | Effect |
|---|---|
| `PREFILL_NUM_LAYERS` | layer count for scenarios 1–3 (GLM pins its own) |
| `PREFILL_STANDALONE_CHUNKED_PCC` | the producer's PCC gate (default 0.93; the TP-sharded scenario pins 0.85) |
| `PREFILL_TP_SHARD_KV=1` | GLM-5.2 SP×TP KV dedup (note 5). Requires `PREFILL_KV_ONLY_LAST_LAYER=0` and `PREFILL_MODEL=glm_5_2`; both enforced at runner import |
| `PREFILL_CI_RUNNER_READY_TIMEOUT_S` | startup budget for scenarios without their own override (default 1200) |
| `PREFILL_CI_PRODUCER_TIMEOUT_S` | producer budget for scenarios without their own override (default 900) |
| `PREFILL_CI_LOG_TAIL_LINES` | inline log tail length (default 200) |
| `PREFILL_RUNNER_LAUNCH` | `ci` / `standard`, overriding launcher auto-detection |
| `PREFILL_SEND_SHUTDOWN=1` | producer sends an all-`-1` sentinel so the runner drains and exits cleanly instead of being signalled |
| `PREFILL_HF_MODEL`, `PREFILL_TTNN_CACHE`, `PREFILL_TRACE_DIR` | asset paths |

---

## 11. Current known state — 2026-07-30

First on-hardware `glm52_full_depth_kv_table` run (78 layers, 56320 tokens, 11 chunks, 1 user, SP8×TP4).
**Read this before debugging: the test currently FAILS, and the failure is understood and open.**

```
slot 0 KV PCC    over [0,56320) across 78 layers -> 0.860812   FAIL (producer gate 0.93)
slot 0 index PCC over [0,56320) across 21 layers -> 0.984348   PASS
kv_cache_pcc_complete slots_checked=1 min_pcc=0.860812
```

**The table is validated for both configs.** All 174240 entries; config 1's 21 ranks resolved to exactly
`{0,1,2,6,10,…,74}`, min 0.98435.

**The failure is confined to one column.** Across all 78 layers `nope` (512-wide latent) falls
0.99972 → 0.86081 while `pe` (64-wide rope) holds 0.99992 → 0.98978 and the index cache holds
0.99995 → 0.98435 — and every layer's minimum is its `nope`. Per §7, that rules out addressing, decode and
upstream activation drift, and points at the numerical fidelity of the compressed KV latent.

**Open items:**

1. **The missing control** — a Gate 0 standalone run at the same 78 layers / 56320 tokens. If Gate 0 also
   lands near 0.86 this is inherent to full-depth chunked DSA prefill and the thresholds are
   mis-calibrated; if Gate 0 is clean at depth, the chunked KV write path degrades the latent and it is a
   real bug. **Do this before changing anything.**
2. The 0.93 / 0.88 threshold split (§6).
3. The summary `min_pcc` hides which column failed — reporting `nope` / `pe` / `index` separately would
   make a failure self-localizing.
4. ~~TP-sharded KV has not been run on hardware.~~ **RESOLVED 2026-07-30 — see §11a.**

---

## 11a. TP-sharded KV — first on-hardware run, 2026-07-30 (PASS)

`glm52_full_depth_kv_table_tp_sharded` on `bh-glx-120-c04u02`, 78 layers / 56320 tokens / 11 chunks /
1 user / SP8×TP4, `PREFILL_TP_SHARD_KV=1`. **`1 passed in 1140.98s (0:19:00)`** — 78-layer weight load
from a warm 401 GiB cache, one JIT warm-up chunk (cold Tensix kernel cache), 11 chunks pushed through
`cidx=10 end=56320`, then the producer read-back:

```
slot 0 KV PCC    over [0,56320) across 78 layers -> 0.860911
slot 0 index PCC over [0,56320) across 21 layers -> 0.984355
KV cache PCC PASSED (min 0.860911 >= 0.85 across 1 slots)
```

**Acceptance was the diff against §11's SP baseline, not the threshold** (TP dedup is bit-identical by
design). Aggregate KV delta **9.9e-05**, index delta **7.0e-06**. Per-layer over all 156 (layer, column)
pairs: **140 identical (≤1e-4), 16 ≤1e-3, 0 changed**; largest single delta 2.4e-04 (layer 58 `nope`).
Deltas are **signed in both directions**, the signature of bf16 all-gather reassociation (the TP read adds
a TP-inner gather ahead of the SP-outer one) rather than a systematic error. ⇒ **The dedup is confirmed
correct end-to-end through the production runner.** Each of the 32 devices now holds a distinct 1/32
slice: 1760 rows/device instead of 7040, per-device KV 11.3 → 2.83 GiB.

Two side conclusions:

- **The full-depth `nope` issue is orthogonal to TP sharding.** TP tracks the SP `nope` curve layer for
  layer including the depth drift (0.99972 @ L0 → ~0.86 @ L69 → 0.92791 @ L77). §11 open item 1 (the
  missing Gate 0 control at depth) is unaffected and still the right next step.
- 0.860911 **clears** the scenario's 0.85 floor, so this scenario passes today. That floor is deliberately
  set below the known SP minimum to catch gross breakage only (a wrong device or wrong 1/tp window lands
  near zero) — it is not a quality bar. Do not raise it without re-reading note 5.

**Gate coverage, and what is still missing.** `test_glm52_tp_sharded_kv_cache_table -k 8x4` passed 8/8 in
53 s (table + writer + allocation agree byte-for-byte at sp=8/tp=4). But the tightest *direct*
device-vs-device equality tests — `test_sparse_tp_sharded_kv_matches_sp` and `_multichunk` in
`tests/sparse_mla/test_sparse_mla_cache.py` — are pinned to `mesh-2x4` and **cannot run on a Blackhole
galaxy at all** (`Blackhole only supports 32-device mesh configs (requested 8)`); they are LoudBox-only.
`test_glm_prefill_block[glm52…8x4]` was not completed either (it pulls ~12.6 GB of HF weights via the
conftest download path and needs the mesh exclusively). So the on-galaxy evidence is
byte-exact-table + e2e-equivalence, **not** an MLA-level output-equality proof.

Artifacts: `generated/test_reports/ci_{runner,producer}_glm52_full_depth_kv_table_tp_sharded.log`.
The SP baseline log (`…_glm52_full_depth_kv_table.log`, 10:08) is a *different* filename, so the TP run
does not clobber it — keep both for future diffs.

---

## 12. File map

| Path | Role |
|---|---|
| `models/demos/common/prefill/tests/test_producer_runner_e2e.py` | this test: scenarios, launch modes, timeouts |
| `models/demos/common/prefill/runners/prefill_runner.py` | the runner (holds the mesh, publishes the table) |
| `models/demos/common/prefill/runners/prefill_producer.py` | the producer: schedule, ack drain, UMD read-back, PCC |
| `models/demos/common/prefill/adapter.py` | model registry + `PrefillModelAdapter` contract |
| `models/demos/deepseek_v3_d_p/tt/runners/adapters/glm_5_2.py` | GLM-5.2 adapter: `allocate_kv_cache`, trace/cache defaults |
| `models/demos/deepseek_v3_d_p/tt/runners/kv_chunk_table.py` | builds + serializes the (merged) table |
| `models/demos/deepseek_v3_d_p/utils/kv_cache_utils.py` | cache allocation + `populate_kv_chunk_address_table_kimi` |
| `models/demos/deepseek_v3_d_p/tt/mla/indexer.py` | `indexer_layer_is_reused`, `num_full_indexer_layers`, `full_indexer_rank` |
| `models/demos/common/prefill/docs/PREFILL_MIGRATION_TESTING.md` | the manual three-gate command reference (Gates 0 and 2) |
| `models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md` | wiring a new model into the runner |
