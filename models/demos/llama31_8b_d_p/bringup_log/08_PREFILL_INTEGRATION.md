# 08 — Disaggregated-prefill integration (P10)

Integration with the model-agnostic prefill engine in `models/demos/common/prefill/`. Written in
P10; the gate numbers live in `06_GATES.md` and the judgement calls in `05_DECISIONS.md`
(`DEC-094`..`DEC-114`).

Two documents define the contract, and **neither is complete**; the engine's own source is:

- `models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md` — the adapter (§1), the runtime (§2),
  registration (§3), validation (§4), and the closing checklist.
- `models/demos/common/prefill/docs/PREFILL_MIGRATION_TESTING.md` — the three config files and the
  two migration gates.

§4 of this file lists every place they disagree with the code, measured rather than argued.

---

## 1. The contract mapping

### 1.1 Adapter — `models/demos/common/prefill/adapter.py:104` `PrefillModelAdapter`

| abstract member | our implementation | notes |
|---|---|---|
| `load_hf_config` (`adapter.py:143`) | `tt/runners/adapters/llama.py:172` | returns `LlamaHfConfig`, a **mutable** attribute view of the `config.json` dict. Not `AutoConfig`: on transformers 5.12.1 that object has no `rope_theta` (`R-005`). A `PREFILL_HF_MODEL` whose `config.json` differs from the bundled copy is **refused** (`DEC-094`). |
| `weight_cache_path` (`adapter.py:161`) | `tt/runners/adapters/llama.py:216` | `<root>/tensor_cache_bfp8_<sp>x<tp>`, mirroring `tt/model_config.py:250` — the layout P8's own populate runs wrote, **not** the engine's `{name}_{arch}_{N}dev/{sp}x{tp}` (`DEC-095`). Touches no device: it is called before the mesh is open. |
| `allocate_kv_cache` (`adapter.py:166`) | `tt/runners/adapters/llama.py:249` | forwards to `tt/attention/kv_cache.py:73`. Returns `LlamaKVCache`, which **is** a `KvCaches` (`tt/attention/kv_cache.py:56`), so no one-element-list wrapper. |
| `build_runtime` (`adapter.py:183`) | `tt/runners/adapters/llama.py:279` | builds `TtPrefillRuntimeConfig` from `params` and returns `TtPrefillRuntime`. Refuses `use_trace` and `dflash_enabled` **at build time** rather than mid-request. |
| `layer_split_boundaries` (`adapter.py:173`) | inherited (`None`) | dense model: any split is legal. Moot — multi-rank is out of scope (`R-032`). |
| `default_sparse_kv_cache_format` (`adapter.py:148`) | inherited (`None`) | no format choice; the cache is `bfloat8_b` (`DEC-021`). |
| `name`, `model_config`, `hf_model_default`, `ttnn_cache_default`, `prefill_trace_default` | `tt/runners/adapters/llama.py:136-145` | `model_config` is `Llama31_8BConfig` (`:60`), whose every constant `G-ADAPTER` checks against `config.json`. |
| `l1_small_size`, `supports_dflash`, `pipeline_activation_emb_tp_sharded` | `tt/runners/adapters/llama.py:153-163` | `0`, `False`, `True`. |

### 1.2 Runtime — `ADDING_A_PREFILL_MODEL.md:111` (a **structural** contract, not a base class)

| member the engine uses | where the engine uses it | our implementation |
|---|---|---|
| `mesh_device` | `prefill_runner.py:281` | `tt/tt_prefill_runtime.py:209` |
| `config.{chunk_size,max_seq_len,first_layer_idx,is_first_rank,is_last_rank}` | the chunk schedule | `TtPrefillRuntimeConfig` (`tt/tt_prefill_runtime.py:117`) |
| `config.use_trace` — **undocumented** | `prefill_runner.py:303`, `:745`, `:773` | pinned `False` (`tt/tt_prefill_runtime.py:156`) |
| `compile(kv_caches)` | `prefill_runner.py:501` | `tt/tt_prefill_runtime.py:338` |
| `make_chunk_input(token_ids)` | no engine call site (the H2D socket delivers the tensor) | `tt/tt_prefill_runtime.py:287` |
| `prefill_chunk(...)` — 2 positional + 6 keywords | `prefill_runner.py:286` | `tt/tt_prefill_runtime.py:381` |
| `build_kv_chunk_table(kv_caches, path, ...)` | `prefill_runner.py:570`, `:644`, `:655`, `:674`, `:699` | `tt/tt_prefill_runtime.py:593` -> `tt/runners/kv_chunk_table.py:192` |
| `kv_migration_base_address(kv_caches)` | `prefill_runner.py:616-617` (behind `hasattr`) | `tt/tt_prefill_runtime.py:652` |
| `kv_migration_stages(...)` | `prefill_runner.py:613-615` (behind `hasattr`) | **deliberately absent** — its presence selects the multi-stage merge path (`R-032`) |
| `set_layer_ack_channel(channel)` | `prefill_runner.py:768` | `tt/tt_prefill_runtime.py:533` |
| `set_layer_completion_sink(sink)` — **undocumented** | `prefill_runner.py:752` | present, **raises** (multi-rank; `R-024`) |
| `set_d2h_ack_service(service)` — **undocumented** | `prefill_runner.py:746` | present, **raises** (trace path; `R-024`) |
| `capture_trace`, `release_trace`, `trace_metadata_msg`, `warmup_ack_count` | behind `getattr` | absent, which is legal |

`G-RUNTIME` re-derives this table with an AST walk over `prefill_runner.py` at test time, so it
cannot go stale silently (`tests/unit/test_prefill_runtime_chunked.py`).

### 1.3 Registration and the producer read-back — the two edits outside the package

| file | change |
|---|---|
| `models/demos/common/prefill/adapter.py:291` | one `ADAPTER_PATHS` line: `"llama31_8b_d_p": "models.demos.llama31_8b_d_p.tt.runners.adapters.llama:LlamaPrefillAdapter"` |
| `models/demos/common/prefill/runners/prefill_producer.py:511-517` | the device-less KV read-back **branches on `ADAPTER.name`** and is *not* adapter-dispatched. `_PACKED_GQA_MODELS` (`:508`) generalises the single-model check, and `_read_slot_kv_and_check_pcc_gpt_oss` is renamed `_read_slot_kv_and_check_pcc_packed_gqa` (`:542`) with its log line naming `ADAPTER.name` (`:598`). `DEC-104`. |

Without that second edit the branch falls through to `_read_slot_kv_and_check_pcc_mla` (`:694`),
which decodes a merged MLA latent+rope row — so `G-MOCK-MIG` would have PCC'd **plausible but wrong
bytes**. The packed-GQA reader is already this model's layout exactly, which is why P5.6 kept
`NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32` and gpt-oss's packing (`tt/attention/kv_cache.py:17-30`).

---

## 2. The env matrix actually used

Shared by both processes; a mismatch on any row makes the byte layout disagree
(`PREFILL_MIGRATION_TESTING.md:552-568`).

| variable | `G-REQUEST` / `G-MOCK-MIG` | `G-REQUEST` (deployment arm) | why |
|---|---|---|---|
| `PREFILL_MODEL` | `llama31_8b_d_p` | `llama31_8b_d_p` | the registry key |
| `PREFILL_MANIFEST` | the package manifest | the package manifest | applied by `setdefault`, so the exports below win |
| `PREFILL_SP` / `PREFILL_TP` | `4` / `8` | `4` / `8` | `TP == num_key_value_heads == 8` is an equality |
| `PREFILL_NUM_LAYERS` | `32` | `32` | ack count is `layers x chunks`; a mismatch hangs the drain |
| `PREFILL_CHUNK_SIZE` | `256` | `8192` | 256 is `G-MESH-KV`'s chunked arm, so the two numbers are comparable |
| `PREFILL_MAX_SEQ_LEN` | `2816` | `131072` | `>= chunks * chunk_size`, `% chunk_size == 0`, and **strictly >** `chunk_size` |
| `PREFILL_NUM_USERS` | `1` | `1` | one slot; cross-talk is invisible with one prompt either way (§3) |
| `PREFILL_H2D_SERVICE_ID` | `llama_prefill` | `llama_prefill` | the H2D socket's name |
| `PREFILL_MOCK_MIGRATION` | `1` | `0` | publishes the table + device map, no worker |
| `PREFILL_ENABLE_MIGRATION` | `0` (arm 1) / **`1`** (arm 2) | `0` | **arm 2 is the whole point**: with it the runner takes the real stage-gather branch (`prefill_runner.py:626`, `:644`) and calls `kv_migration_base_address`, and it still needs no worker binaries. `DEC-111` |
| `PREFILL_ENABLE_LAYER_ACK` | `1` (`G-MOCK-MIG` only; arm 2 gets it free, since it defaults to `PREFILL_ENABLE_MIGRATION`) | `0` | the producer's PCC read-back requires it |
| `PREFILL_MIGRATION_TABLE_PATH` | `/tmp/llama_kv_chunk_table.pb` | — | single host, so `/tmp` is legal |
| `PREFILL_MIGRATION_DEVICE_MAP_PATH` | `/tmp/llama_kv_device_map.json` | — | must stay host-local |
| `PREFILL_TRACE_DIR` | the 1024-token golden | the 1024-token golden | tokens **and** golden KV |
| `PREFILL_PRODUCER_CHUNKS` | `11` / `4` | `2` | 4 x 256 = 1024 = the golden's length |
| `PREFILL_PRODUCER_CHECK_PCC` | `0` / `1` | `0` | `G-MOCK-MIG` is the PCC arm |
| `PREFILL_SEND_SHUTDOWN` | `1` | `1` | the request loop is unbounded; without it the runner sits in `recv` |
| `HF_MODEL` | the staged checkpoint | same | read by `ModelArgs`; the adapter only refuses it when unset |
| `TT_CACHE_PATH` | `~/.cache/llama31_8b_d_p` | same | the weight-cache root P8 populated |

**No new `PREFILL_*` variable was invented.** Every knob above is the engine's own. This package
introduces none in P10 — the topology is pinned in code (`DEC-097`) precisely so that it did not
have to become a fifth package env var.

---

## 3. What these gates prove, and what they do not

### Covered

| Property | By which gate |
|---|---|
| the engine can resolve, construct and drive this model end to end | `G-ADAPTER` + `G-REQUEST` |
| the adapter satisfies every abstract method, with no heavy import | `G-ADAPTER` (subprocess-measured, with a control) |
| every `model_config` constant equals `config.json` | `G-ADAPTER` |
| the runtime cannot `TypeError` on the engine's real call | `G-RUNTIME` (AST, extended in P10) |
| request-mode serving: every chunk accepted, served, and a clean shutdown | `G-REQUEST` |
| the **deployment** chunk/cache pair (8192 / 131072) is servable | `G-REQUEST`, deployment arm (closes half of `R-039`) |
| prefill writes correct KV, read back by a **second reader in another process** | `G-MOCK-MIG` |
| the address table: position -> address, head -> config -> chip, K/V separation | `G-KV-TABLE`, **bit-exactly** over UMD |
| the engine's **stage-gather** migration branch: `allgather_kv_stage_layouts`, `kv_migration_base_address`, the merged-table build | `G-MOCK-MIG` arm 2 (`DEC-111`) — no worker binaries needed |
| the protobuf round trip preserves addresses **and** config names | `G-KV-TABLE` |
| the per-layer LayerAck fires exactly `num_layers` times per chunk | `G-RUNTIME` + `G-MOCK-MIG`'s drain |
| the unimplemented multi-rank merge raises instead of discarding its argument | `G-RUNTIME` (P10 arm) |

### NOT covered, and what each omission means

| Not exercised | What it means in practice |
|---|---|
| **the real DRAM -> transport -> DRAM copy** (`G-LOOPBACK`, doc Gate 2) | Out of scope by `DEC-103`: it needs the tt-llm-engine binaries, and it verifies the *engine's* model-agnostic byte copy rather than this model. `R-043` is the residual gap, and **arm 2 shrank it**: what is left is `publish_serialized_table_and_wait_ready`, the two worker processes and the destination read-back. Everything the runner itself does on the real path short of the publish has now run. |
| **multi-rank / pipeline-parallel prefill** | Two galaxies, ruled out by the user (`R-032`). The merge raises; it does not guess. |
| **cross-endpoint P->D migration** | Skipped by the driver itself even when run (`PREFILL_MIGRATION_TESTING.md:298-300`): the destination lives in another address space. |
| **cross-talk between slots** | Invisible with one prompt: every slot's KV would be byte-identical (`PREFILL_MIGRATION_TESTING.md:301-304`). `PREFILL_PRODUCER_SLOT_TRACES` with two distinct traces is what would cover it, and it needs a second golden trace. `R-044`. |
| **the trace / 2CQ path** | `use_trace` is refused at build time; `capture_trace` does not exist. Explicit non-goal. |
| **`PREFILL_KV_ONLY_LAST_LAYER`** | The engine defaults it **on** and this runtime ignores it, writing every layer's KV — which is what the producer's per-layer PCC and migration both need. Warned about at build time (`DEC-096`). |
| **a multi-turn schedule** | `PREFILL_PRODUCER_MULTI_TURN_PROB` untested here, and untested upstream too (`PREFILL_MIGRATION_TESTING.md:251-255`). |
| **the golden beyond 1024 tokens** | `G-MOCK-MIG` PCCs `[0, 1024)`. The deployment arm serves 16384 tokens but scores nothing: there is no golden that deep (`R-026`). |
| **`Topology.Ring` under the engine** | This galaxy has no ring fabric at all (`R-030`/`R-031`); the adapter pins `Linear`. A torus-cabled machine is a different measurement. |

### If you later need it

- **`G-LOOPBACK`**: build `migration_endpoint` + `migration_worker` from tt-llm-engine against this
  same tt-metal tree, export `PREFILL_MIGRATION_CLIENT_DIR`, and follow
  `PREFILL_MIGRATION_TESTING.md` Gate 2 with `--verify-migration both`. Everything on this side is
  already in place: `build_kv_chunk_table` returns the path the engine publishes, and
  `kv_migration_base_address` answers the stage gather. Expect the first failure to be the
  `wait_ready` timeout described at `PREFILL_MIGRATION_TESTING.md:599-622`, which is a launcher
  problem, not a model one.
- **cross-talk coverage**: generate a second golden trace from a different prompt
  (`scripts/generate_golden_kv_cache.py`) and set `PREFILL_PRODUCER_SLOT_TRACES=<dirA>,<dirB>` with
  `PREFILL_NUM_USERS=2` and `PREFILL_PRODUCER_MAX_REQUESTS=2`.
- **multi-rank**: implement the merge in `tt/runners/kv_chunk_table.py` (the four refusals in
  `assert_single_rank_stage` are where), and expect `set_layer_completion_sink` next.

---

## 4. Where the engine's own documents are wrong, incomplete or stale

Measured against the code, in the order a reader hits them. Each of these cost time or would have.

| # | Document | What it says | What the code does |
|---|---|---|---|
| 1 | `ADDING_A_PREFILL_MODEL.md:129` | `prefill_chunk(input_tensor, kv_cache, *, slot_id, actual_start, actual_end, request_id=0)` | the engine also always passes `d2h_service` **and** `metadata_msg` (`prefill_runner.py:286-295`). A runtime written to the doc dies with a `TypeError` on its first served chunk. Found in P7 by `G-RUNTIME`; the recipe records it. |
| 2 | `ADDING_A_PREFILL_MODEL.md` §2 (whole) | lists eight runtime members | the engine calls **two more** unguarded: `set_layer_completion_sink` (`prefill_runner.py:752`) and `set_d2h_ack_service` (`:746`), and reads a third field, `config.use_trace` (`:303`, `:745`, `:773`). None appears in the doc. |
| 3 | `ADDING_A_PREFILL_MODEL.md:146` | `build_kv_chunk_table(self, kv_cache, path: str)` | called five times, with `path` **positional** at `:644`, `:655`, `:674` and keyword at `:570`, `:699`, plus `first_layer_idx` / `num_my_layers` and a `**stage_layout` splat. A keyword-only `path` breaks on three of the five. |
| 4 | `ADDING_A_PREFILL_MODEL.md:64` | "The engine sets `.max_seq_len` on the returned config" | true, and the consequence is unstated: it **mutates** the object, so a frozen dataclass raises `FrozenInstanceError` after `_print_config` has already printed a healthy table. The recipe records this one. |
| 5 | `ADDING_A_PREFILL_MODEL.md:241` | "The producer's reader knows two cache layouts — merged MLA ... and MiniMax-M3's triple cache; a third layout needs a branch" | **stale**: there are three, and the third (`_read_slot_kv_and_check_pcc_gpt_oss`, now `_packed_gqa`) is exactly this model's layout. The recipe already flags the doc as predating it. |
| 6 | `PREFILL_MIGRATION_TESTING.md:539-545` | the hook table gives Gate 1 exactly one hook, `build_kv_chunk_table` | Gate 1 **also** needs `set_layer_ack_channel`: with `PREFILL_PRODUCER_CHECK_PCC=1` the producer refuses to run without the LayerAck channel (`prefill_producer.py:1065-1071`), and the channel only exists when the runner calls that hook (`prefill_runner.py:767-768`). A runtime built to the table alone cannot pass Gate 1. |
| 7 | `PREFILL_MIGRATION_TESTING.md:421-427` | Gate 1's binding sets `PREFILL_MOCK_MIGRATION: "1"` and the producer manifest `check_pcc: true` | neither mentions `PREFILL_ENABLE_LAYER_ACK`, which defaults to `PREFILL_ENABLE_MIGRATION`, i.e. **0** on the mock path (`prefill_runner.py:552-554`). So the documented Gate 1 configuration exits 1 with "LayerAck channel missing". |
| 8 | `prefill_runner.py:567-576` vs `:691-705` | — | the mock-migration table is built and the device map serialized **twice** on the `MOCK_MIGRATION=1, ENABLE_MIGRATION=0` path: once in the block at `:567` and again in the `elif` at `:691`, which that first block does not skip. Harmless (idempotent, and the serialize is atomic), but it doubles the table build and logs every publish line twice, which reads like a retry. **Measured**, not inferred: `raw/G-MOCK-MIG-runner_20260904T173939Z.log.gz` carries both sets, from `_serve_request:573` and `:702`. `R-049`. |
| 9 | `ADDING_A_PREFILL_MODEL.md:85` | "read them instead of `os.environ`" | the reference adapter reads `PREFILL_TOPOLOGY` from `os.environ` (`models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:140`) because `PrefillRunParams` carries no topology field. Ours pins `Linear` in code instead (`DEC-097`). |
| 10 | `models/demos/common/prefill/runners/prefill_producer.py:531` | — | the shared packed-GQA reader imports `NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK` from **`models.demos.minimax_m3`** inside `_read_kv_slice`, i.e. one model's constant is used to read every model's cache. All three are 32 today, so it is latent; `tt/runners/kv_chunk_table.py:56` asserts our value against the template's so a divergence fails loudly on our side. |
| 11 | `models/demos/common/prefill/adapter.py:66` | `PrefillRunParams.num_links` | **no model in this tree reads it.** The engine computes `2 if is_blackhole() else 1` (`prefill_runner.py:489`) and both this adapter and the reference one ignore it; each runtime derives its own from `get_default_num_links` (`models/demos/gpt_oss_d_p/utils/general_utils.py:27`), which disagrees with the engine's value on Wormhole and on any single-row mesh. They agree at `(4,8)` on Blackhole, which is why nothing is wrong today. `R-050`. |
| 12 | `ADDING_A_PREFILL_MODEL.md:146`, `:158-162` | `build_kv_chunk_table(self, kv_cache, path)`; `kv_migration_stages` "per config of your merged table" | **the shape of `stage_layout` is documented nowhere.** The doc's `build_kv_chunk_table` signature omits the argument entirely, and the only hint is `kv_migration_stages`' prose. It is a **list of one dict per rank** (`migration.py:315-334`), and the engine passes `stage_layouts[0]` — stage 0's per-rank list (`prefill_runner.py:634`). A reader who guesses "one stage layout, so one dict" writes code that rejects every real migration run, which is exactly what happened (`DEC-111`). The only way to learn it is to read `allgather_kv_stage_layout`, or another model: `models/demos/deepseek_v3_d_p/tt/tt_prefill_runtime.py:929` iterates it and `models/demos/deepseek_v3_d_p/utils/kv_cache_utils.py:394` sums over it. |
| 13 | `PREFILL_MIGRATION_TESTING.md:456-461` | Gate 2 "Needs the tt-llm-engine binaries built" | true of Gate 2, and it obscures that **`PREFILL_ENABLE_MIGRATION=1` + `PREFILL_MOCK_MIGRATION=1` needs none of them**: the worker handshake is only in the `else` branch at `prefill_runner.py:673`, so that combination exercises the stage gather, `kv_migration_base_address` and the merged-table build with nothing but tt-metal. No document mentions this arm, and it is the one that catches a whole class of contract error (`DEC-111`). |

Where the **bring-up recipe** is wrong is in `05_DECISIONS.md` and in this session's report, not here.

---

## 5. Verbatim gate transcripts

Trimmed only of tt-metal's per-op `Pinned source memory start address ... must be aligned 64 B`
warnings, which account for ~5,000 lines per runner log (`R-040`). The full logs are under
`bringup_log/raw/`; the runner logs are gzipped because each exceeds the repo's 500 KB hook limit.

### `G-REQUEST`, arm 1 — gate geometry (chunk 256 / cache 2816, 11 chunks)

Runner (`raw/G-REQUEST-runner_20260904T173723Z.log.gz`):

```
prefill_runner configuration
======================================================================
  PREFILL_MODEL                       = llama31_8b_d_p
  PREFILL_HF_MODEL                    = models/demos/llama31_8b_d_p/configs/Llama-3.1-8B-Instruct
  PREFILL_TTNN_CACHE                  =
  resolved weight_cache_path          = /home/mstojkovic/.cache/llama31_8b_d_p/tensor_cache_bfp8_4x8
  PREFILL_SP                          = 4
  PREFILL_TP                          = 8
  PREFILL_NUM_LAYERS                  = 32
  PREFILL_KV_ONLY_LAST_LAYER          = True
  DFLASH_ENABLED                      = False (adapter.supports_dflash=False, DFLASH_HF_MODEL=<unset>)
  PREFILL_USE_TRACE                   = False (trace_region=0 MB)
  PREFILL_CHUNK_SIZE                  = 256
  PREFILL_MAX_SEQ_LEN                 = 2816
  PREFILL_NUM_USERS                   = 1
  PREFILL_FABRIC_MODE                 = 1d
  PREFILL_H2D_SERVICE_ID              = llama_prefill
  PREFILL_TRACE_DIR                   = /home/mstojkovic/prefill_traces/llama31_8b_d_p/s1024
  PREFILL_ENABLE_MIGRATION            = 0
  PREFILL_MOCK_MIGRATION              = 0
======================================================================
[pp rank 0/1] mesh=(4, 8) layers=[0, 32) is_first=True is_last=True chunk_size=256 max_seq_len=2816 num_users=1
Fabric config: FabricConfig.FABRIC_1D (sp=4, PREFILL_FABRIC_MODE=1d)
[llama31_8b_d_p] config.json from models/demos/llama31_8b_d_p/configs/Llama-3.1-8B-Instruct (24 keys, bundled copy asserted equal)
[llama31_8b_d_p] loading real bf16 weights from /home/mstojkovic/models/Llama-3.1-8B-Instruct (safetensors read) ...
building Llama TtPrefillRuntime: layers=32 max_seq_len=2816 chunk_sizes=(256,) users=1 mesh=(4, 8) sp=4 tp=8 topology=Topology.Linear
[llama31_8b_d_p] KV cache allocated: 1 user(s) x 32 layer(s), capacity 2816 tokens, per-chip rows 704, head_dim 128, bfloat8_b
TtPrefillRuntime.compile(): warming a 256-token chunk
TtPrefillRuntime.compile(): warming a second 256-token chunk (cache-backed)
[h2d] H2DStreamService built: global_shape=(4,1,64) uint32 ROW_MAJOR DRAM, per_chip_bytes=256
[pp rank 0] [h2d] descriptor service_id='llama_prefill' -> /dev/shm/tt_h2d_stream_service_llama_prefill.bin
[migration] LayerAck channel disabled (set PREFILL_ENABLE_LAYER_ACK=1 to enable)
[pp rank 0] setup complete, entering request loop
[pp rank 0/1] request (unbounded) loop start (is_first=True is_last=True input=h2d)
[pp rank 0] CHUNK_START c=0  compute_start=1788543483.725558 slot=0 [0,256)
[pp rank 0] CHUNK_START c=1  ... slot=0 [256,512)
...
[pp rank 0] CHUNK_START c=10 compute_start=1788543485.803478 slot=0 [2560,2816)
[pp rank 0] SHUTDOWN sentinel received after 11 chunks; exiting request loop
[pp rank 0] E2E_CLOCK first_compute_start=1788543483.725558 last_compute_end=1788543486.009879
[pp rank 0] processed 11 chunks in 16001.98 ms
[pp rank 0] shutdown complete
```

Producer (`raw/G-REQUEST-producer_20260904T173723Z.log`):

```
[producer] service_id='llama_prefill' users=1 chunks=[11,11] max_requests=1 verify=False seed=1234
[producer] attached; payload=1024B
[producer] CHECK_PCC off — skipping the KV table read and not consuming the LayerAck channel
[producer] push slot=0 cidx=0 start=0 end=256
...
[producer] push slot=0 cidx=10 start=2560 end=2816
[producer] DONE wall=0.8s pushes=11 requests=1 tokens=2816 throughput=3330 tok/s push_ms p50=0.1 p90=206.4 p99=225.8
[producer] sending SHUTDOWN sentinel (metadata=-1,-1,-1)
[producer] exiting; SHUTDOWN sentinel sent — runner will drain and shut down.
```

Driver verdict: `=== G-REQUEST producer_rc=0 runner_rc=0`.

One line above is quoted **as it ran** and no longer matches the code: `loading real bf16 weights
... (safetensors read)` was reworded to `mapping the bf16 checkpoint ... (safetensors, lazy)` after
this gate, because `DEC-098`'s measurement showed the old wording was misleading — the call returns
in 46 ms and, with the cache populated, never reads a byte. A raw log records what happened, so it
keeps the old text (`BRINGUP_RECIPE.md` §0.2 rule 4). Nothing this gate asserts depends on the
string.

### `G-REQUEST`, arm 2 — the real deployment geometry (chunk 8192 / cache 131072)

```
[pp rank 0/1] mesh=(4, 8) layers=[0, 32) is_first=True is_last=True chunk_size=8192 max_seq_len=131072 num_users=1
[llama31_8b_d_p] KV cache allocated: 1 user(s) x 32 layer(s), capacity 131072 tokens, per-chip rows 32768, head_dim 128, bfloat8_b
[pp rank 0] CHUNK_START c=0 compute_start=1788543755.262681 slot=0 [0,8192)
[pp rank 0] CHUNK_START c=1 compute_start=1788543755.526063 slot=0 [8192,16384)
[pp rank 0] SHUTDOWN sentinel received after 2 chunks; exiting request loop
[pp rank 0] processed 2 chunks in 13633.29 ms
[producer] DONE wall=0.0s pushes=2 requests=1 tokens=16384
=== G-REQUEST-DEPLOYMENT producer_rc=0 runner_rc=0
```

This is the pair `07_RISKS.md` R-039 recorded as never run.

### `G-MOCK-MIG` — the doc's Gate 1

Runner (`raw/G-MOCK-MIG-runner_20260904T173939Z.log.gz`), the two lines that matter:

```
[gpt-oss-d-p-kv-table] multi-config table built (configs=16 [k_h0, k_h1, ..., v_h6, v_h7], entries=45056, banks=8, chunk_bytes=4352)
[llama31-8b-d-p-kv-table] 16 configs (k_h0..7, v_h0..7), 45056 entries, seq_len=2816 period=256 users=1 layers=32 -> /tmp/llama_kv_chunk_table.pb
[migration] KV chunk address table serialized to /tmp/llama_kv_chunk_table.pb (configs=16, entries=45056)
[migration] device map (32 chips) serialized to /tmp/llama_kv_device_map.json
[mock-migration] ... (from _serve_request:573)
[gpt-oss-d-p-kv-table] multi-config table built (configs=16 ..., entries=45056, banks=8, chunk_bytes=4352)
[llama31-8b-d-p-kv-table] 16 configs (k_h0..7, v_h0..7), 45056 entries, seq_len=2816 period=256 users=1 layers=32 -> /tmp/llama_kv_chunk_table.pb
[migration] KV chunk address table serialized to /tmp/llama_kv_chunk_table.pb (configs=16, entries=45056)
[migration] device map (32 chips) serialized to /tmp/llama_kv_device_map.json
[mock-migration] ... (from _serve_request:702)
[migration] LayerAck channel ready at /tt_prefill_layer_acks_llama_prefill; runner emits one ack per layer
LayerAck channel registered: 32 acks per chunk (the producer drains num_layers x chunks — prefill_producer.py:1115)
```

**Everything above appears twice**, and that is the engine's doing, not a retry: on the
`PREFILL_MOCK_MIGRATION=1, PREFILL_ENABLE_MIGRATION=0` path the block at `prefill_runner.py:567`
builds and publishes the table, and the `elif` at `:691` — which belongs to `if _migration_enabled:`
and is therefore **not** skipped by that first block — builds and publishes it again. Idempotent, so
harmless, but it doubles the build and reads like a retry (`R-049`). The two `_serve_request` line
numbers in the log (`:573` and `:702`) are the proof.

`entries = 45056` = 16 configs x 32 layers x 1 slot x 88 positions (2816 / 32), i.e. every 32-token
block of the whole cache capacity, which is what the table is supposed to address.

Producer (`raw/G-MOCK-MIG-producer_20260904T173939Z.log`):

```
[producer] read KV chunk table /tmp/llama_kv_chunk_table.pb: entries=11264 num_layers=32 num_slots=1 max_seq_len=2816 chunk_n_tokens=32
[producer] connected LayerAck channel /tt_prefill_layer_acks_llama_prefill
[producer] read device map /tmp/llama_kv_device_map.json: 32 chips
[producer] push slot=0 cidx=0 start=0 end=256
[producer] push slot=0 cidx=1 start=256 end=512
[producer] push slot=0 cidx=2 start=512 end=768
[producer] push slot=0 cidx=3 start=768 end=1024
[producer] DONE wall=0.6s pushes=4 requests=1 tokens=1024
[producer] layer acks 128/128
[producer] drained 128/128 layer acks in 0.42s
  layer  0: K=0.99996 V=0.99994
  layer  1: K=0.99995 V=0.99977
  ...
  layer 22: K=0.99678 V=0.98971
  ...
  layer 28: K=0.99749 V=0.98662
  layer 31: K=0.99876 V=0.99506
[producer] slot 0 llama31_8b_d_p packed-GQA KV PCC over [0,1024) across 32/32 local layers -> K=0.99678 V=0.98662 (min 0.986623)
[producer] KV cache PCC PASSED (min 0.986623 >= 0.93 across 1 slots; per cache: k=0.996784, v=0.986623)
[producer] kv_cache_pcc_complete slots_checked=1 min_pcc=0.986623 k_pcc=0.996784 v_pcc=0.986623
```

`G-MESH-KV`'s on-device chunk-256 row, for the comparison `06_GATES.md` sets out in full:
`min_k 0.9967844268417902 (argmin 22) / min_v 0.9866232360049921 (argmin 28)`.

### `G-KV-TABLE`

```
[G-KV-TABLE] period=512: 16 configs, 1024 entries, 4352 B/chunk, 16 positions x 2 layers x 2 slots
[G-KV-TABLE] period=512: 1024 chunks bit-identical over UMD (torch.equal, rtol=atol=0)
[G-KV-TABLE] control rotated_head:  differs from the correct chunk (max|delta| = 1.0)
[G-KV-TABLE] control k_through_v:   differs from the correct chunk (max|delta| = 16.0)
[G-KV-TABLE] control next_layer:    differs from the correct chunk (max|delta| = 1.0)
[G-KV-TABLE] control next_position: differs from the correct chunk (max|delta| = 32.0)
[G-KV-TABLE] control next_slot:     differs from the correct chunk (max|delta| = 2.0)
[G-KV-TABLE] protobuf round trip: 16 configs, 1024 entries, names ['00', '01', '02', '03']... preserved
[G-KV-TABLE] period=128: 16 configs, 1024 entries, 4352 B/chunk, 16 positions x 2 layers x 2 slots
[G-KV-TABLE] period=128: 1024 chunks bit-identical over UMD (torch.equal, rtol=atol=0)
11 passed in 38.01s
```

### The failed first `G-REQUEST` — retained on purpose

`raw/G-REQUEST-runner_20260904T173012Z.log.gz`. It is `DEC-108`'s evidence, and it is what makes
`G-REQUEST`'s "every chunk served" a discriminating claim rather than a formality:

```
[pp rank 0] setup complete, entering request loop
[pp rank 0] CHUNK_START c=0 compute_start=1788543070.854658 slot=0 [0,256)
Traceback (most recent call last):
  ...
  File ".../models/demos/common/prefill/runners/prefill_runner.py", line 286, in _compute_and_send
    out = runtime.prefill_chunk(
  File ".../models/demos/llama31_8b_d_p/tt/tt_prefill_runtime.py", line 420, in prefill_chunk
    raise NotImplementedError(
NotImplementedError: metadata_msg is the engine's trace-safe metadata tensor and needs
config.use_trace plus capture_trace, neither of which this runtime implements (risk R-024).
```

The process then sat at ~125% CPU and ignored SIGTERM; it needed `kill -9` and a manual
`rm -f /dev/shm/tt_h2d_* /dev/shm/tt_socket_manifest_*` (`R-048`).

### `G-MOCK-MIG` arm 2 — the stage-gather branch (`PREFILL_ENABLE_MIGRATION=1`)

Runner (`raw/G-MOCK-MIG-STAGED-runner_20260904T180914Z.log.gz`). The two lines that distinguish this
arm from arm 1 are the branch it took and the hook it called:

```
[migration] rank 0: local device map -> /tmp/llama_kv_device_map.json
[llama31-8b-d-p-kv-table] 16 configs (k_h0..7, v_h0..7), 45056 entries, seq_len=2816 period=256 users=1 layers=32 -> /tmp/llama_kv_chunk_table.pb
[migration] KV chunk address table serialized to /tmp/llama_kv_chunk_table.pb (configs=16, entries=45056)
[mock-migration] merged KV chunk table -> /tmp/llama_kv_chunk_table.pb (no migration worker)      <- _serve_request:651
[mock-migration] rank 0: local device map -> /tmp/llama_kv_device_map.json                         <- _serve_request:652
[migration] LayerAck channel ready at /tt_prefill_layer_acks_llama_prefill; runner emits one ack per layer
```

`_serve_request:651` is the log line immediately after the `:644` build, so reaching it proves that
`kv_migration_base_address` answered at `:617`, `allgather_kv_stage_layouts` ran at `:626`, and the
gathered **list** reached `assert_single_rank_stage` and was accepted. Before `DEC-111`'s fix this
line could not have been reached: the guard raised `TypeError` on the list.

Producer (`raw/G-MOCK-MIG-STAGED-producer_20260904T180914Z.log`) — the same numbers as arm 1, to
every printed digit:

```
[producer] drained 128/128 layer acks in 0.41s
[producer] slot 0 llama31_8b_d_p packed-GQA KV PCC over [0,1024) across 32/32 local layers -> K=0.99678 V=0.98662 (min 0.986623)
[producer] KV cache PCC PASSED (min 0.986623 >= 0.93 across 1 slots; per cache: k=0.996784, v=0.986623)
[producer] kv_cache_pcc_complete slots_checked=1 min_pcc=0.986623 k_pcc=0.996784 v_pcc=0.986623
=== G-MOCK-MIG-STAGED producer_rc=0 runner_rc=0
```
