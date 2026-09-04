<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 08 — Disaggregated-prefill integration (P10)

Written in P10, which ran **before** P9 (`BRINGUP_RECIPE.md` F.12). Contract:
`models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md`. Gates: `G-ADAPTER`, `G-REQUEST`,
`G-MOCK-MIG`, `G-KV-TABLE` — all four in `06_GATES.md`. Decisions: `DEC-100` .. `DEC-111`.

---

## 1. The contract mapping

Every abstract member of `PrefillModelAdapter` (`models/demos/common/prefill/adapter.py:104`) and
every method of the §2 runtime interface, against the code that implements it.

### Adapter — `tt/runners/adapters/llama.py:76` (`LlamaPrefillAdapter`)

| Contract member | Our implementation | Notes |
|---|---|---|
| `name` | `"llama31_8b_d_p"` | also the weight-cache dir prefix and the registry key |
| `model_config` | `tt/model_dims.py:32` `Llama31_8BConfig` | new zero-import constants class (`DEC-101`) |
| `hf_model_default` | `configs/Llama-3.1-8B-Instruct` (bundled) | dimensions never depend on a machine-local path |
| `ttnn_cache_default` | `""` | ⇒ resolved from the environment; see `weight_cache_path` |
| `prefill_trace_default` | `""` | ⇒ must come from `PREFILL_TRACE_DIR` (`DEC-057`) |
| `l1_small_size` | `0` | no op in this package routes semaphores to L1_SMALL |
| `supports_dflash` | `False` | the drafter is a Kimi-only checkpoint (`adapter.py:133`) |
| `pipeline_activation_emb_tp_sharded` | `True` | `DEC-018` scheme A; **untested** — single-rank only (`R-040`) |
| `load_hf_config()` | `tt/runners/adapters/llama.py` | returns `RuntimeLlamaHFConfig` (`tt/model_config.py:179`), because the engine assigns `.max_seq_len` to it (`DEC-100`) |
| `weight_cache_path(mesh_shape)` | `tt/runners/adapters/llama.py` | must equal `ModelArgs.weight_cache_path(bfloat8_b)` (`tt/model_config.py:402`); asserted by a test |
| `allocate_kv_cache(...)` | `tt/runners/adapters/llama.py` → `tt/attention/kv_cache.py` `allocate_kv_cache` | returns `LlamaKvCaches` (`:58`), a `KvCaches` subclass holding one `LlamaKVCache` |
| `build_runtime(...)` | `tt/runners/adapters/llama.py` | builds `TtPrefillRuntime` with `owns_kv_cache=False` (`DEC-055`) |

### Runtime — `tt/tt_prefill_runtime.py` (`TtPrefillRuntime`)

| §2 member | Our implementation | Notes |
|---|---|---|
| `mesh_device`, `config` | attributes | `config` exposes `chunk_size` (property alias, `DEC-054`), `max_seq_len`, `first_layer_idx`, `is_first_rank`, `is_last_rank` |
| `compile(kv_cache)` | `tt/tt_prefill_runtime.py` | warms 1 or 2 chunks per supported size |
| `make_chunk_input(token_ids)` | `tt/tt_prefill_runtime.py` | SP-sharded `uint32` `[1,1,1,chunk/sp]`, the layout H2D delivers |
| `prefill_chunk(...)` | `tt/tt_prefill_runtime.py` | also accepts `request_id`, `d2h_service`, `record_dev` **and `metadata_msg`** — the last is undocumented but passed on every chunk (`DEC-106`) |
| `set_layer_ack_channel(channel)` | `tt/tt_prefill_runtime.py` | one `inject(1)` per layer; drained 128/128 in `G-MOCK-MIG` |
| `kv_migration_base_address(kv)` | `tt/tt_prefill_runtime.py:579` | K's DRAM base. **Implemented, never executed** — `R-040` |
| `build_kv_chunk_table(kv, path)` | `tt/tt_prefill_runtime.py:583` → `tt/runners/kv_chunk_table.py:243` | closes `R-030`. Refuses multi-rank merge arguments rather than ignoring them (`DEC-109`) |
| `kv_migration_stages(...)` | **not implemented** | deliberate: the engine prefers it when present, and it is the multi-cache/renumbering hook we have not tested. One migratable cache pair ⇒ the base-address form is the documented sufficient hook. |

### Registration and config files

| Thing | Where |
|---|---|
| Registry line | `models/demos/common/prefill/adapter.py:291` (one line; the only permitted edit outside the package besides `DEC-105`'s) |
| Model manifest | `tt/runners/manifests/llama31_8b_d_p.json` — six pinned values, no workload knobs (`DEC-108`) |
| KV chunk table | `tt/runners/kv_chunk_table.py` — 16 configs (`k_h0..k_h7`, `v_h0..v_h7`), built through the engine's shared `serialize_kv_chunk_table` (`common/prefill/runners/migration.py:220`) |

---

## 2. The producer's KV read-back — a shared-code edit (`DEC-105`)

**The trap, confirmed and then closed.** The device-less read-back that powers
`PREFILL_PRODUCER_CHECK_PCC` is *not* adapter-dispatched. It branches on `ADAPTER.name` in
`models/demos/common/prefill/runners/prefill_producer.py:511`, and any unknown name falls through to
the **MLA** reader (`:696`). The MLA reader is wrong for Llama's plain packed K/V, so without a
branch `G-MOCK-MIG` would silently report a PCC computed over the wrong bytes.

`_read_slot_kv_and_check_pcc_gpt_oss` (`:544`) *is* the plain packed-K/V block-cyclic GQA reader, and
it is correct for us because P2 committed to keeping gpt-oss's block geometry
(`NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32`, shard row `[1,1,32,head_dim]`). So the edit generalises
the name check rather than duplicating the function:

```
_PACKED_GQA_MODELS = ("gpt_oss_d_p", "llama31_8b_d_p")     # prefill_producer.py:508
...
if ADAPTER.name in _PACKED_GQA_MODELS:                     # prefill_producer.py:515
    return _read_slot_kv_and_check_pcc_gpt_oss(...)
```

plus one log f-string that said "GPT-OSS KV PCC" now saying `{ADAPTER.name} packed-GQA KV PCC`, so a
Llama transcript does not claim to be a GPT-OSS one. **Two lines changed, one constant and one
comment block added.** `gpt_oss_d_p` reaches the same function through the same branch; no other
model's dispatch moves.

The residual risk — that the two layouts diverge later and this branch silently misaligns — is closed
by a test, not a comment: `tests/unit/test_prefill_adapter.py::test_our_dram_block_geometry_still_equals_gpt_oss`
asserts `OUR_BLOCK == GPT_OSS_BLOCK == 32` and its failure message says to write a fourth reader
rather than widen the branch. A second test asserts our name is in the tuple at all.

**Independent corroboration that the branch is right.** The producer's device-less read produced
min K **0.99646** / V **0.98445** over 32 layers — identical to five decimal places to P8's
`G-MESH-KV` "chunked 4x512 @ 2048 tok" row, which was read back on device through
`TtPrefillRuntime.gather_layer` in a different process by a completely different path. Two
independent readers agreeing to 1e-5 on the same DRAM is strong evidence the addresses are right.

---

## 3. Env matrix actually used

Shared, and identical on both sides (the byte layout depends on it):

| Variable | Value | Why |
|---|---|---|
| `PREFILL_MODEL` | `llama31_8b_d_p` | registry key |
| `PREFILL_SP` / `PREFILL_TP` | `4` / `8` | mesh **rows** / **cols**. `TP=8` is forced by the packed cache (`R-027`) |
| `PREFILL_NUM_LAYERS` | `32` | the runner defaults to 61 |
| `PREFILL_CHUNK_SIZE` | `512` | `DEC-110` |
| `PREFILL_MAX_SEQ_LEN` | `2048` | `= 4 x 512`, and **strictly** `> chunk_size` so the ring path runs |
| `PREFILL_NUM_USERS` | `1` | deployment config |
| `PREFILL_H2D_SERVICE_ID` | `llama_prefill` | H2D descriptor name |
| `PREFILL_TRACE_DIR` | `/home/mstojkovic/llama31_8b_golden/p7_s2048` | tokens **and** golden KV |
| `PREFILL_FABRIC_MODE` | `1d_ring` | manifest; the engine would pick `FABRIC_1D` at `sp<=8` and `Topology.Ring` would then **hang** (`DEC-108`) |
| `PREFILL_TOPOLOGY` | `ring` | manifest |
| `TT_MESH_GRAPH_DESC_PATH` | `tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto` | the cyclic route `FABRIC_1D_RING` needs; **a manifest cannot set it** |
| `HF_MODEL` | `/home/mstojkovic/models/Llama-3.1-8B-Instruct` | the checkpoint, read by `ModelArgs` |
| `LLAMA_WEIGHTS_FROM_CACHE` | `1` (runner) | reuse P8's tilized bf8_b cache; byte-identical KV proven by `R-017` |

Runner-only: `PREFILL_MANIFEST`, and for `G-MOCK-MIG` `PREFILL_MOCK_MIGRATION=1`,
`PREFILL_ENABLE_LAYER_ACK=1`, `PREFILL_MIGRATION_TABLE_PATH`, `PREFILL_MIGRATION_DEVICE_MAP_PATH`.
Producer-only: `PREFILL_PRODUCER_CHUNKS=4`, `PREFILL_PRODUCER_MAX_REQUESTS=1`,
`PREFILL_PRODUCER_CHECK_PCC`, `PREFILL_SEND_SHUTDOWN=1`, `PREFILL_H2D_CONNECT_TIMEOUT=120`.

`PREFILL_ENABLE_LAYER_ACK=1` is **not optional** with `PREFILL_PRODUCER_CHECK_PCC=1`: the producer
exits 1 without it, because a UMD read that does not wait on the acks races the runner's prefill
(an H2D push returning is not the layers being done).

Full transcripts: `raw/G-REQUEST-runner_20260904T000335Z.log.gz`,
`raw/G-REQUEST-producer_20260904T000335Z.log`, `raw/G-MOCK-MIG-runner_20260904T000550Z.log.gz`,
`raw/G-MOCK-MIG-producer_20260904T000550Z.log`, `raw/G-ADAPTER_20260904T000808Z.log`,
`raw/G-KV-TABLE_20260904T000847Z.log`.

---

## 4. Migration coverage: what Gate 1 proves, and what Gate 2 would have added

This bring-up runs the engine's **Gate 1** (mock migration) and does **not** run **Gate 2** (loopback
migration). Gate 2 needs binaries from the private `tt-llm-engine` repo, which is not available in
this environment; the full reasoning and the exact missing products are in `DEC-070`, and the one
residual gap is `R-040`. It is recorded as **out-of-scope by decision**, not as a blocked gate — the
distinction matters, because "blocked" would imply this package has untested surface that it does not.

### Covered here

| Property | By | Measured |
|---|---|---|
| Prefill writes correct KV for a slot | `G-MOCK-MIG` — device-less `read_dram_umd` read-back, PCC vs the golden | min K 0.99646 / V 0.98445 over 32 layers |
| This package's `build_kv_chunk_table` is correct (one rank's layer slice) | `G-MOCK-MIG` (every chunk resolves through our table) **and** `G-KV-TABLE` (bit-exact, labelled pattern, 2 users, protobuf round trip, negative control) | `torch.equal` over 2 users x 2 layers x 8 heads x K/V x 512 tokens |
| The device map is published where every device-less reader expects it | `G-MOCK-MIG` | 32 chips read back |
| The adapter satisfies the engine's contract | `G-ADAPTER` | 29/29 checks |
| Request-mode serving: H2D push, chunk schedule, per-layer ack drain, graceful shutdown | `G-REQUEST` + `G-MOCK-MIG` | 4 chunks, 128/128 acks, sentinel handled |

### NOT covered, and what each omission means

| Not exercised | What it means in practice |
|---|---|
| The real DRAM → transport → DRAM copy | Whether the *engine's* byte copy lands. Model-agnostic: its default `dst-bytes` mode decodes nothing and asserts only dst == src. No Llama-specific property rides on it. |
| `MigrationLayerClient` attach, `WORKER_READY` handshake, cross-endpoint pairing | Engine/worker-side orchestration. Our runtime's only obligation is `set_layer_ack_channel`, and that **is** driven in request mode by `G-MOCK-MIG` (128/128 acks); what is untested is migration-triggered scheduling on top of it. |
| **The multi-rank merged KV-chunk table** | **The one real gap in our own code** (`R-040`). Gate 1 is single-rank only by design, so `kv_migration_base_address` is implemented to the documented contract but never executed. P10 made this *loud* rather than latent: `build_kv_chunk_table` now **raises** `NotImplementedError` naming `R-040` when handed a foreign `first_layer_idx`, a foreign `num_my_layers`, or a stage layout spanning more than one rank (`DEC-109`), so the first pipelined run gets an error instead of a wrong table. |
| The D2D pipeline activation layout (`pipeline_activation_emb_tp_sharded = True`) | Single-rank runs build no D2D socket. This is an assumption from `DEC-018`, not a measurement. Same owner as `R-040`. |
| Destination-slot read-back (`dst-bytes` / `dst-golden`) | Verification of a copy we never make. |
| `num_users > 1` **through the serving loop** | `G-MOCK-MIG` ran one user. Multi-slot *addressing* is covered bit-exactly by `G-KV-TABLE` (2 users); what is not covered is the runner interleaving two live requests. See `R-013`'s P10 note. |

### Inherited limitations that apply the moment anyone runs Gate 2

Straight from the engine doc, and true regardless of model:

* **Loopback only.** Cross-endpoint prefill → decode is skipped with a warning, because the
  destination lives in the decode galaxy's address space and looking `dst` up in *our* table would
  confidently read the wrong slot. Verify P→D on the decode side against a `--dump-src-kv` reference.
* **Cross-talk is invisible with one prompt.** If every slot replays the same trace, all sources are
  byte-identical and a copy landing in the wrong destination is indistinguishable from a correct one.
  Use `PREFILL_PRODUCER_SLOT_TRACES` to give each slot its own prompt. This applies to `dst-golden`
  equally. *(It applies to `G-MOCK-MIG` here too: with `num_users = 1` there is no cross-talk to see —
  which is why `G-KV-TABLE` labels the slot index in the data.)*
* **A layer subset makes a `PASS` a sample.** `--verify-migration-layers` restricts the read and the
  driver warns; with a subset only config 0 is checked.
* **`read_dram_umd` is host-local.** On a multi-host runner each driver process verifies only its own
  host's chips; chunks whose chips are not in that host's device map are *skipped*, and the skip count
  is printed so a `PASSED` cannot be mistaken for whole-model coverage.

### If you later need it

Get the migration component built (products and constraints: `DEC-070`), then run Gate 2 on a 2- or
4-rank binding to close `R-040`. `DEC-109`'s three refusals are the exact list of what has to start
working first.

---

## 5. Things found in the engine while integrating (not fixed — reported)

Neither is a defect in this package, and neither was worked around silently.

1. **`prefill_chunk` is called with two keywords the contract does not document.** §2 documents
   `request_id`; the engine also always passes `d2h_service` and `metadata_msg`
   (`common/prefill/runners/prefill_runner.py:364`). A runtime written strictly to the doc dies with
   a `TypeError` on its first served chunk. `gpt_oss_d_p`'s runtime does **not** accept
   `metadata_msg` and would hit this. We accept it (`DEC-106`) and added an AST test that reads the
   engine's own call site, so a future engine keyword fails a device-free unit test instead of a
   galaxy run.
2. **The mock-migration table is built twice per run.** In `_serve_request`, the block at
   `prefill_runner.py:566` (`if _mock_migration and not _migration_enabled:`) builds and serializes
   the table, and then the `elif os.environ.get("PREFILL_MOCK_MIGRATION", "0") == "1":` branch at
   `:684` does it again — both fire for a single-rank `PREFILL_MOCK_MIGRATION=1` run. Visible in
   `raw/G-MOCK-MIG-runner_20260904T000550Z.log.gz` as two identical
   `[llama31_8b_d_p-kv-table] built: configs=16 …` lines. Harmless (idempotent, ~15 ms) but
   redundant, and the second block carries a `single_rank` guard the first one lacks.
