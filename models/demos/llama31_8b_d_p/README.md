<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Llama-3.1-8B-Instruct — disaggregated-prefill bring-up

TTNN implementation of **Llama-3.1-8B-Instruct** *prefill* inference for Tenstorrent Blackhole.
Target: one Blackhole Galaxy (`(4, 8)` mesh) running **SP=4 × TP=8**, registered with the
disaggregated-prefill engine as `llama31_8b_d_p`.
Config: [`configs/Llama-3.1-8B-Instruct/config.json`](configs/Llama-3.1-8B-Instruct/config.json).

Prefill only. The product of a run is the **KV cache**; no decode loop is built here.

The full bring-up record — every decision, gate, measured number and open risk — is in
[`bringup_log/`](bringup_log/), and the recipe the bring-up followed is
[`BRINGUP_RECIPE.md`](BRINGUP_RECIPE.md).

## Why a new package, and not `models/common/`

This is the first question a reviewer asks, because `models/common/models/llama3_8b/` is already a
complete Llama-3.1-8B and `models/common/modules/` is a shared, unit-tested module library
(`MLP1D/2D`, `RMSNorm1D/2D`, `Attention1D`, `RotarySetup1D`, `Embedding1D`, `LMHead1D`, cached
`TT_CCL`). Both were evaluated in P2 (`bringup_log/02_SURVEY.md`, `BRINGUP_RECIPE.md` Appendix F.3)
and neither can carry this deployment:

- **`models/common/models/llama3_8b/` cannot run the target mesh.** `models/common/models/llama3_8b/model.py:890`
  raises `ValueError("Llama3Transformer1D only supports 1D mesh topologies.")` on a 32-device
  cluster — N150/N300/T3K only. It is also decode/generation-oriented, with no chunked-prefill
  runtime and no `models/demos/common/prefill` adapter.
- **`MLP2D`'s "2D" is 2D *tensor* parallelism, not TP × SP.** Its prefill path reduce-scatters on
  `cluster_axis=1` and closes with `all_reduce(cluster_axis=0)`
  (`models/common/modules/mlp/mlp_2d.py:461`). With SP on the row axis, that all-reduce would sum
  activations belonging to **different tokens** — silently wrong, and it would still produce
  plausible PCC on a one-row mesh. The tempting shortcut *"an MLP is token-pointwise, so SP looks
  like DP to it"* holds for the math but **not** for this module's collectives.
- **There is no `Attention2D`.**

So the templates are `models/demos/minimax_m3/tt/dense_mlp.py` for the MLP (it collectives on the TP
axis **only**, which is what makes it SP-safe) and `models/demos/gpt_oss_d_p/tt/attention/` for
attention. What this package adds over the existing Llama is: **2D TP × SP on a 32-chip Blackhole
Galaxy**, and the **disaggregated-prefill engine contract**.

## Architecture

Every row is read from `config.json`; provenance for each is in `bringup_log/00_MODEL_CARD.md` §2.

| | |
|---|---|
| Decoder layers | **32**, all identical — no layer-type schedule |
| Hidden / FFN intermediate | **4096** / **14336** |
| Attention | GQA: **32 q / 8 kv** heads, `head_dim` **128** (derived: 4096/32), group 4 |
| RoPE | **full rotary** (`rotary_dim == head_dim == 128`), θ = **500000.0**, `rope_type="llama3"` scaling (`factor=8.0`, `low_freq_factor=1.0`, `high_freq_factor=4.0`, `original_max_position_embeddings=8192`) |
| Norm | plain **RMSNorm**, `eps=1e-05`, **no `+1` weight fold** (that is Gemma, not Llama) |
| MLP | dense **SwiGLU** — `down(silu(gate(x)) * up(x))` — on **every** layer |
| Vocab | **128256** (tile-friendly: 128256/32 = 4008) |
| Max positions | 131072 |
| Biases | **none** — `attention_bias=false`, `mlp_bias=false` |
| Tied embeddings | **false** — `lm_head.weight` is a separate tensor |
| MoE / attention sinks / sliding window / QK-norm | **none of them** |
| Checkpoint dtype | `bfloat16` |

## Deployment path (Blackhole Galaxy, `(4, 8)`)

- **TP = 8** on the mesh **columns**, **SP = 4** on the mesh **rows**. **EP: n/a** — Llama is dense.
- **Attention:** GQA prefill. Two cores, selected by cache capacity, not per chunk:
  - `max_seq_len == chunk_size` → the **SP bootstrap** (all-gather Q/K/V → plain causal SDPA →
    reduce-scatter). `fp32_dest_acc_en=True`.
  - `max_seq_len > chunk_size` → the **`ring_joint` SDPA cache-read** (`tt/attention/dense_sp.py`),
    on *every* chunk including chunk 0. `fp32_dest_acc_en` must be `False` here — the op refuses
    otherwise (`DEC-084`).
- **MLP / norms:** collectives on the TP axis only, which is what keeps them SP-safe.
- **KV cache:** packed, block-cyclic, `NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32`, shard row
  `[1, 1, 32, head_dim]`, bf8_b. One **KV head per chip** across the TP columns.
- **Fabric:** `FABRIC_1D_RING` + `Topology.Ring` + `num_links=2`, which requires the **torus**
  mesh-graph descriptor (`TT_MESH_GRAPH_DESC_PATH`, below). A Ring topology on a plain `FABRIC_1D`
  fabric **hangs** rather than erroring (`DEC-081`, `DEC-108`).

### Two hard constraints

1. **`TP` must equal `num_key_value_heads` = 8.** The packed cache holds exactly one KV head per
   chip, so the model → cache mapping is *head `c` → mesh column `c`*. Any other TP either has no
   head for a column or two heads for one, and the op's own `TT_FATAL` ("cache and input num-heads
   dim must match") is what you get. This is proved bit-exactly by `G-KV-TP8`, and the package
   refuses sub-axis TP shapes at construction rather than producing a wrong cache.
2. **`max_seq_len` must be *strictly* greater than `chunk_size`.** At `max_seq_len == chunk_size`
   the per-chip cache shard leaves the ring op no room, and attention silently falls back to the
   one-shot SP bootstrap — a correct-but-different core, so anything you measure is measuring the
   wrong path (`DEC-021`). The adapter logs a warning when this happens; it is not an error, because
   the bootstrap *is* the right answer for a genuine single-chunk request.

All ttnn C++ ops and fabric mesh descriptors this model uses are consumed from `main`; this
directory is Python only. The bring-up touched exactly **two files** outside it, both in
`models/demos/common/prefill/`: one line registering the adapter (`adapter.py:291`), and two lines
adding this model to the producer's packed-GQA KV read-back branch (`DEC-105`) — without the second,
the device-less read-back falls through to the **MLA** reader and reports a PCC computed over the
wrong bytes.

## Status

Verified on a Blackhole Galaxy `(4, 8)`, SP=4 × TP=8, all **32 layers**, real checkpoint, against the
**fp32 golden KV trace** (`transformers`' own fp32 math on the checkpoint's bf16 weights upcast
exactly). Device dtypes: **bf8_b weights, bf16 activations, bf8_b KV cache**. Thresholds K ≥ 0.99 /
V ≥ 0.98. Full detail and the raw logs: `bringup_log/06_GATES.md`.

| Run | attention core | min PCC across 32 layers (K / V) | tok/s |
|---|---|---|---|
| one-shot, 1 × 512 (`G-MESH-KV`) | SP bootstrap | **0.99789** / **0.99134** | 2394 |
| chunked, 2 × 256 (`G-MESH-KV`) | ring cache-read | 0.99695 / 0.98859 | 1429 |
| chunked, 4 × 512 @ 2048 tok (`G-MESH-KV`) | ring cache-read | **0.99646** / **0.98445** | 2846 |
| chunked, 2 × 256, weights from cache (`G-MESH-KV`) | ring cache-read | 0.99695 / 0.98859 | 1438 |
| served through the engine, 4 × 512 @ 2048 tok (`G-MOCK-MIG`) | ring cache-read | **0.996456** / **0.984451** | 2413\* |

\* The 2413 tok/s is `G-REQUEST`'s compute span for the same shape and env (`E2E_CLOCK`, 848.7 ms
for 2048 tokens); `G-MOCK-MIG` adds the migration-table build and the layer-ack drain and makes no
throughput claim. Both sit below the standalone harness's 2846 tok/s by the cost of the H2D socket
and per-chunk metadata. **No throughput threshold exists anywhere in this package** — every tok/s
here is recorded, not gated.

The last row is the one that matters most, and not because it is the best number: it was read back
**device-lessly** over `read_dram_umd` by the engine's producer, in a different process and through
a completely different code path from the on-device `G-MESH-KV` row above it — and the two agree to
five decimal places. Two independent readers agreeing on the same DRAM is the evidence that the KV
addresses are right.

Other measured properties:

- **Race-free** (`G-RACE`): three prefills in **one process on one `CCLManager`** → **1 distinct
  SHA-256** over the full 32-layer KV read-back. The same digest also came out of two other
  processes, and out of the cache-only weight-load path.
- **Weight loading** (`G-WEIGHTS`): 291/291 checkpoint tensors consumed, 0 missing, 0 unused; the
  cache-only rebuild is byte-identical at `(4, 8)`.
- **The one-shot → chunked gap** (K 0.99789 → 0.99695) is the ring op's mandatory loss of the fp32
  accumulator, measured and attributed in `DEC-084`. It is not a regression.

**Decode is not part of this bring-up.**

## Run

Llama-3.1-8B-Instruct is a first-class `transformers` model (no `trust_remote_code`). Download the
checkpoint — safetensors weights + `config.json` + tokenizer — into a local directory and point
`HF_MODEL` at it. The bundled `configs/Llama-3.1-8B-Instruct/config.json` carries the **dimensions**
only, so the adapter can report the model shape without a machine-local path.

```bash
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export HF_MODEL=/path/to/Llama-3.1-8B-Instruct
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/\
single_bh_galaxy_torus_xy_graph_descriptor.textproto
# Optional, and only read by the galaxy harness and the engine adapter (never by the unit tests):
# reuse the tilized bf8_b weight cache the first galaxy run wrote, instead of re-tilizing 7.9 GB.
export LLAMA_WEIGHTS_FROM_CACHE=1
```

### Module-by-module PCC suite

Every module is gated against a torch/HF reference on random weights, so the suite runs without a
checkpoint (real-weight tests skip). The galaxy tests inside it open the `(4, 8)` mesh.

```bash
# whole package — this is the G-CLEAN-REGRESSION / G-P10-REGRESSION command:
PREFILL_TRACE_DIR=/path/to/golden/p7_s2048 pytest models/demos/llama31_8b_d_p/tests -q

# one module:
pytest models/demos/llama31_8b_d_p/tests/unit/test_mlp_vs_ref.py -x -q
```

### Galaxy KV-cache PCC vs the fp32 golden

Golden traces are produced by [`scripts/`](scripts/) and verified before use:

```bash
python3 models/demos/llama31_8b_d_p/scripts/generate_golden_kv_cache.py \
  --prompt-file prompt.txt --max-tokens 2048 --pad-to 2048 --out /path/to/golden/p7_s2048
python3 models/demos/llama31_8b_d_p/scripts/verify_golden_kv.py /path/to/golden/p7_s2048
```

Then the harness — this is `G-MESH-KV` and `G-RACE`:

```bash
# one-shot (SP bootstrap):
PREFILL_CHUNKED=0 PREFILL_TRACE_DIR=/path/to/golden/p7_s512 \
  python3 models/demos/llama31_8b_d_p/tests/galaxy_prefill_kv_pcc.py

# chunked 4 x 512 @ 2048 (the ring cache-read — the served path):
PREFILL_CHUNKED=1 PREFILL_CHUNK_SIZE=512 PREFILL_TRACE_DIR=/path/to/golden/p7_s2048 \
  python3 models/demos/llama31_8b_d_p/tests/galaxy_prefill_kv_pcc.py

# G-RACE: three runs in one process on one CCLManager, hashes compared:
PREFILL_CHUNKED=1 PREFILL_CHUNK_SIZE=256 PREFILL_RUNS=3 PREFILL_KV_HASH_ONLY=1 \
  PREFILL_TRACE_DIR=/path/to/golden/p7_s512 \
  python3 models/demos/llama31_8b_d_p/tests/galaxy_prefill_kv_pcc.py
```

The harness auto-SKIPs (exit 0) without a 32-device galaxy or without a golden trace.

### Request-mode serving — `G-REQUEST` and `G-MOCK-MIG` (two terminals)

Both processes must see an **identical** shared env: the byte layout of the cache depends on it.

```bash
# --- shared, export in BOTH terminals ---
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export HF_MODEL=/path/to/Llama-3.1-8B-Instruct
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/\
single_bh_galaxy_torus_xy_graph_descriptor.textproto
export PREFILL_MODEL=llama31_8b_d_p PREFILL_SP=4 PREFILL_TP=8
export PREFILL_NUM_LAYERS=32 PREFILL_CHUNK_SIZE=512 PREFILL_MAX_SEQ_LEN=2048
export PREFILL_NUM_USERS=1 PREFILL_H2D_SERVICE_ID=llama_prefill
export PREFILL_FABRIC_MODE=1d_ring PREFILL_TOPOLOGY=ring
export PREFILL_TRACE_DIR=/path/to/golden/p7_s2048
```

```bash
# --- terminal 1: the runner ---
export PREFILL_MANIFEST=$TT_METAL_HOME/models/demos/llama31_8b_d_p/tt/runners/manifests/llama31_8b_d_p.json
export LLAMA_WEIGHTS_FROM_CACHE=1
# for G-MOCK-MIG only (engine Gate 1) — remove the four for a plain G-REQUEST run:
rm -f /tmp/prefill_kv_chunk_table.pb /tmp/prefill_kv_device_map.json
export PREFILL_MOCK_MIGRATION=1 PREFILL_ENABLE_LAYER_ACK=1
export PREFILL_MIGRATION_TABLE_PATH=/tmp/prefill_kv_chunk_table.pb
export PREFILL_MIGRATION_DEVICE_MAP_PATH=/tmp/prefill_kv_device_map.json

python -m models.demos.common.prefill.runners.prefill_runner
```

```bash
# --- terminal 2: the producer (start after the runner prints that it is serving) ---
export PREFILL_PRODUCER_CHUNKS=4 PREFILL_PRODUCER_MAX_REQUESTS=1
export PREFILL_SEND_SHUTDOWN=1 PREFILL_H2D_CONNECT_TIMEOUT=120
export PREFILL_PRODUCER_CHECK_PCC=1      # G-MOCK-MIG only

python -m models.demos.common.prefill.runners.prefill_producer
```

`PREFILL_ENABLE_LAYER_ACK=1` is **not optional** with `PREFILL_PRODUCER_CHECK_PCC=1`: the producer
exits 1 without it, because a UMD read that does not wait on the layer acks races the runner's
prefill (an H2D push *returning* is not the layers being *done*).

## Environment variables

Two groups. The first is owned by this package — grep `os.environ` / `os.getenv` under
`models/demos/llama31_8b_d_p/` and you get exactly this list. The second is owned by the
disaggregated-prefill engine (`models/demos/common/prefill/`); it is reproduced here because a
`G-REQUEST` / `G-MOCK-MIG` run is not reproducible without it, and because two of its variables are
read by *our* code.

### Package-owned

| Variable | Default | Read at | Effect |
|---|---|---|---|
| `HF_MODEL` | — | `conftest.py`, `tests/test_factory.py`, `tt/model_config.py`, `tt/runners/adapters/llama.py`, `scripts/generate_golden_kv_cache.py` | The checkpoint directory. Without it, real-weight tests **skip** (they do not fail) and every module test runs on random weights, which is a supported mode, not a degraded one. |
| `LLAMA31_8B_TTNN_CACHE` | unset | `tt/model_config.py`, `tt/runners/adapters/llama.py` | Root of the tilized weight cache. Falls back to `TT_CACHE_PATH`, then `$HF_MODEL/ttnn_cache`. |
| `TT_CACHE_PATH` | unset | as above | Repo-wide weight-cache root; second in the fallback order. |
| `LLAMA_WEIGHTS_FROM_CACHE` | unset | `tests/galaxy_prefill_kv_pcc.py`, `tt/runners/adapters/llama.py` | `1` → build from an **empty** state dict and load the tilized cache instead. Refuses loudly if no cache path resolves. KV is byte-identical to a checkpoint-loaded run (`R-017`). |
| `LLAMA_KV_PCC_MIN` | unset | `tests/galaxy_prefill_kv_pcc.py` | Fail the harness when the min KV PCC drops below this. Unset = report only. |
| `LLAMA31_8B_DELTA_PROBE` | unset (off) | `tt/layer.py` | Any non-empty value logs L2 / mean\|x\| / signed-mean / max\|x\| of every residual delta, per layer, from device 0's shard. A **debug probe**: it is wrapped in `try/except` so it can never fail a run, and a failure is logged at WARNING. |
| `PREFILL_TRACE_DIR` | — | `tests/galaxy_prefill_kv_pcc.py`, `tt/tt_prefill_runtime.py`, three `tests/unit/` chunked tests | The golden trace directory (`metadata.json` + `kv_cache/layer_N.safetensors`). Required by the galaxy harness; the unit tests skip without it. |
| `PREFILL_CHUNKED` | `0` | `tests/galaxy_prefill_kv_pcc.py` | `1` → chunked (ring cache-read); `0` → one-shot (SP bootstrap). |
| `PREFILL_CHUNK_SIZE` | `512` | `tests/galaxy_prefill_kv_pcc.py` | Chunk size in tokens; must satisfy `chunk % (32 * sp) == 0`. |
| `PREFILL_RUNS` | `1` | `tests/galaxy_prefill_kv_pcc.py` | Prefill repetitions in **one process on one `CCLManager`**, each hashed. `G-RACE` uses 3. |
| `PREFILL_NUM_LAYERS` | all 32 | `tests/galaxy_prefill_kv_pcc.py` | Build and run only the first N decoder layers. |
| `PREFILL_TOPOLOGY` | `ring` | `tests/galaxy_prefill_kv_pcc.py`, `tt/runners/adapters/llama.py` | `ring` or `linear`. Must agree with `PREFILL_FABRIC_MODE`: `ring` on a plain `FABRIC_1D` fabric **hangs**. |
| `PREFILL_KV_HASH_ONLY` | `0` | `tests/galaxy_prefill_kv_pcc.py` | `1` → hashes only, skipping the 129 MB fp32 golden read per run. |
| `PREFILL_HF_MODEL` | unset | `tt/runners/adapters/llama.py` | Engine-side override for the directory `load_hf_config` reads `config.json` from. Unset → the config bundled with this package. Distinct from `HF_MODEL`, which names the **weights**. |
| `PREFILL_TTNN_CACHE` | unset | `tt/runners/adapters/llama.py` | Engine-side weight-cache root, tried **before** `LLAMA31_8B_TTNN_CACHE`, so a deployment can redirect it without touching the package's own variables. |
| `TT_MESH_GRAPH_DESC_PATH` | — | not read by this package's Python; consumed by `ttnn` fabric init | Must point at `single_bh_galaxy_torus_xy_graph_descriptor.textproto` for the cyclic `FABRIC_1D_RING` route. **A manifest cannot set this** — it has to be in the environment. |

### Engine-owned (`models/demos/common/prefill/`)

Set for a request-mode run; the authoritative matrix is `bringup_log/08_PREFILL_INTEGRATION.md` §3.

| Variable | Value used here | Notes |
|---|---|---|
| `PREFILL_MODEL` | `llama31_8b_d_p` | registry key (`models/demos/common/prefill/adapter.py:291`) |
| `PREFILL_MANIFEST` | `tt/runners/manifests/llama31_8b_d_p.json` | runner only; pins six values and **no** workload knob (`DEC-108`) |
| `PREFILL_SP` / `PREFILL_TP` | `4` / `8` | mesh rows / cols. `TP=8` is forced, see the hard constraints above |
| `PREFILL_NUM_LAYERS` | `32` | the runner otherwise defaults to 61 |
| `PREFILL_CHUNK_SIZE` | `512` | `DEC-110` |
| `PREFILL_MAX_SEQ_LEN` | `2048` | `= 4 × 512`, and **strictly** `> chunk_size` |
| `PREFILL_NUM_USERS` | `1` | `> 1` is untested through the serving loop |
| `PREFILL_H2D_SERVICE_ID` | `llama_prefill` | H2D descriptor name |
| `PREFILL_FABRIC_MODE` | `1d_ring` | from the manifest; the engine would otherwise pick `FABRIC_1D` at `sp ≤ 8` and Ring would hang |
| `PREFILL_MOCK_MIGRATION` | `1` | runner, `G-MOCK-MIG` only (engine Gate 1; single-rank by design) |
| `PREFILL_ENABLE_LAYER_ACK` | `1` | runner; mandatory with `PREFILL_PRODUCER_CHECK_PCC` |
| `PREFILL_MIGRATION_TABLE_PATH` | `/tmp/prefill_kv_chunk_table.pb` | runner; where our `build_kv_chunk_table` serializes |
| `PREFILL_MIGRATION_DEVICE_MAP_PATH` | `/tmp/prefill_kv_device_map.json` | runner; where device-less readers find the chips |
| `PREFILL_PRODUCER_CHUNKS` | `4` | producer |
| `PREFILL_PRODUCER_MAX_REQUESTS` | `1` | producer |
| `PREFILL_PRODUCER_CHECK_PCC` | `1` | producer; the device-less KV read-back + PCC |
| `PREFILL_SEND_SHUTDOWN` | `1` | producer; the graceful-shutdown sentinel |
| `PREFILL_H2D_CONNECT_TIMEOUT` | `120` | producer |
| `PREFILL_STANDALONE_CHUNKED_PCC` | engine default `0.93` | the producer's pass threshold; we measure 0.984451 |
| `PREFILL_KV_ONLY_LAST_LAYER` | engine default `1` | **accepted and ignored** (`DEC-104`). A DeepSeek-family optimisation this runtime does not implement. Correctness is unaffected — prefill's product is the KV cache and the LM head is never built — the cost is one MLP per chunk. Logged at INFO on every run rather than asserted, so the engine default still runs. |
| `PREFILL_DFLASH` | unset | must not attach to Llama: the drafter is a Kimi-only checkpoint. The adapter sets `supports_dflash = False`. |

## Layout

```
tt/attention/     GQA prefill: SP ring cache-read (dense_sp) + one-shot bootstrap (prefill),
                  QKV/RoPE/o_proj ops, packed block-cyclic KV cache, weight loading, configs
tt/runners/       the disaggregated-prefill adapter, the KV chunk address table, the model manifest
tt/               ccl, config (MeshConfig), embedding, layer, lm_head, mlp, model, model_config
                  (ModelArgs + LlamaHFConfig), model_dims (zero-import constants), rms_norm, rope,
                  tt_prefill_runtime
utils/            state-dict substate slicing, weight-cache filename helpers
scripts/          fp32 golden KV-cache generation + verification, and the citation verifier
configs/Llama-3.1-8B-Instruct/config.json   dimensions only; weights come from $HF_MODEL
tests/unit/       module-by-module PCC tests, plus the adapter and KV-table contract tests
tests/            galaxy harnesses (prefill KV-cache PCC / race), the fabric topology matrix,
                  and the shared TestFactory
bringup_log/      the bring-up record: model card, reference, survey, outline, CCL plan,
                  decisions, gates, risks, prefill integration — and raw/ with every gate's log
```

## What is **not** implemented

Stated so nobody has to discover it at runtime. Each item names where the reasoning lives.

- **Decode.** Prefill only; the product of a run is the KV cache. The LM head exists and is
  PCC-gated (`G-MODEL`, top-1 5/5) but is never built by the prefill runtime.
- **Performance work.** Every choice in this package is functional-first: three separate Q/K/V
  matmuls rather than a fused QKV (`DEC-014`), no fused-kernel tuning, no program-config sweep. The
  throughput numbers above are *recorded, not gated* — there is no perf threshold anywhere.
- **Trace capture and 2-CQ.** Neither is wired up.
- **Multi-rank pipelined prefill.** `kv_migration_base_address` is implemented to the documented
  contract but **never executed**, and the multi-rank *merged* KV-chunk table is not written.
  `build_kv_chunk_table` **raises** `NotImplementedError` naming `R-040` when handed a foreign
  `first_layer_idx`, a foreign `num_my_layers`, or a stage layout spanning more than one rank
  (`DEC-109`) — so the first pipelined run gets an error, not a wrong table. The D2D pipeline
  activation layout (`pipeline_activation_emb_tp_sharded = True`) is an assumption from `DEC-018`,
  not a measurement, for the same reason.
- **`num_users > 1` through the serving loop.** Multi-slot *addressing* is covered bit-exactly by
  `G-KV-TABLE` (2 users); what is not covered is the runner interleaving two live requests
  (`R-013`).
- **`kv_migration_stages`.** Deliberately absent: the engine prefers it when present, and it is the
  multi-cache/renumbering hook we have not tested. With one migratable cache pair, the
  base-address form is the documented sufficient hook.
- **`G-LOOPBACK` (the engine's Gate 2, loopback KV migration).** **Out of scope by decision**
  (`DEC-070`), not blocked — it needs binaries from the private `tt-llm-engine` repo, and what it
  verifies is the *engine's* model-agnostic byte copy. The distinction matters: "blocked" would
  imply this package has untested surface that it does not. The residual gap is `R-040`, above.
  `bringup_log/08_PREFILL_INTEGRATION.md` §4 enumerates exactly what Gate 1 proves and what Gate 2
  would have added.
