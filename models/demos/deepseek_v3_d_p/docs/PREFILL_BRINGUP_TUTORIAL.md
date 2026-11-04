# Bringing up a prefill model on the disaggregated prefill stack

A working tutorial derived from the four models that are already serving on this stack:
DeepSeek-V3 (`deepseek_v3_d_p`), Kimi-K2.6 / K2.7 (`kimi_k2_6`, `kimi_k2_7`) and GLM-5.1 / 5.2
(`glm_5_1`, `glm_5_2`).

The engine-side contract is summarised in
[`common/prefill/docs/ADDING_A_PREFILL_MODEL.md`](../../common/prefill/docs/ADDING_A_PREFILL_MODEL.md).
That document tells you *what* to implement. This one tells you *how the pieces fit*, *what the
existing models actually did*, and *in what order to do it* — including the parts that live
outside the adapter (TT modules, weight cache, golden traces, migration tables, gate/matmul
tuning) and that a new architecture usually has to touch.

---

## Table of contents

1. [The shape of the system](#1-the-shape-of-the-system)
2. [The runner engine](#2-the-runner-engine)
3. [The adapter](#3-the-adapter)
4. [The runtime](#4-the-runtime)
5. [KV caching](#5-kv-caching)
6. [The TT model: transformer, block, attention, MoE](#6-the-tt-model)
7. [Weights and the TTNN cache](#7-weights-and-the-ttnn-cache)
8. [Golden traces and PCC](#8-golden-traces-and-pcc)
9. [Migration and the KV chunk table](#9-migration-and-the-kv-chunk-table)
10. [Bring-up order for a new model](#10-bring-up-order-for-a-new-model)
11. [Where model-specific code actually lives](#11-where-model-specific-code-actually-lives)

---

## 1. The shape of the system

Disaggregated prefill splits a serving deployment into a *prefill* cluster that fills a KV cache
and a *decode* cluster that consumes it. This repo owns the prefill half. The pieces:

```
producer / scheduler                    runner (one process per rank)
─────────────────────                   ────────────────────────────────────────────────
prefill_producer.py   ──H2D socket──▶   rank 0 ── D2D ──▶ rank 1 ── D2D ──▶ … ── rank N-1
(token chunks +                          │                 │                     │
 PrefillMetadata)                        └── KV cache ─────┴─────────────────────┘
                                                    │
                                          KV chunk table (protobuf)
                                                    │
                                          migration worker ──▶ decode cluster
```

Three layers of code, with a hard boundary between the first two:

| Layer | Location | Owns |
|---|---|---|
| **Engine** | `models/demos/common/prefill/` | topology, rank split, sockets, serving loop, cache lifetime, migration handshake, shutdown |
| **Adapter** | `<your_model>/tt/runners/adapters/` | config loading, weight-cache path, KV allocation, runtime construction |
| **Runtime + TT modules** | `<your_model>/tt/` | the actual device model |

The engine **never imports a model class**. It resolves an adapter by name from `PREFILL_MODEL`
and drives everything through it. That is the whole point of the split: adding a model must not
require editing `common/`, except for one line in the registry.

### The two "modes" you will run in

* **Request mode** — the only serving path. The runner builds an H2D service and exports a
  descriptor; a producer connects by `PREFILL_H2D_SERVICE_ID` and pushes chunks. This is what
  production uses and what you validate against.
* **pytest** — the per-module PCC tests under `tests/`. These use the same adapter (via the
  `variant` fixture in `tests/conftest.py`), so the adapter is the single source of truth for
  both paths.

---

## 2. The runner engine

`models/demos/common/prefill/runners/prefill_runner.py`. Read this once end to end; it is the
only file that defines the order in which your adapter's methods get called.

### 2.1 Startup sequence

```
import time
  1. _apply_manifest_env()          # PREFILL_MANIFEST JSON → os.environ.setdefault
  2. module-level env reads         # SP/TP, NUM_LAYERS, CHUNK_SIZE, MAX_SEQ_LEN, NUM_USERS, …
  3. ADAPTER = get_adapter(PREFILL_MODEL)
  4. os.environ.setdefault("PREFILL_TTNN_CACHE", ADAPTER.ttnn_cache_default)

main()
  5. signal handlers (SIGTERM/SIGINT → _shutdown flag)
  6. ttnn.init_distributed_context(); rank, num_ranks
  7. _assert_ranks_agree_on_config()      # CRC32 fingerprint allgather — hard fail on mismatch
  8. compute_layer_split(NUM_LAYERS, num_ranks, ADAPTER.layer_split_boundaries(NUM_LAYERS))
  9. open_mesh_device(...)                # fabric config, ADAPTER.l1_small_size, trace region
 10. hf_config = ADAPTER.load_hf_config(); hf_config.max_seq_len = MAX_SEQ_LEN
 11. params = PrefillRunParams(...)       # every per-rank knob, resolved
 12. runtime  = ADAPTER.build_runtime(mesh_device=…, hf_config=…, params=params)
 13. kv_caches = ADAPTER.allocate_kv_cache(mesh_device=…, hf_config=…, params=params)
 14. runtime.compile(kv_caches)
 15. _serve_request(...)                  # sockets, migration, LayerAck, trace, request loop

shutdown
 16. runtime.release_trace()   (if traced)
 17. ttnn.set_fabric_config(DISABLED); ttnn.close_mesh_device(...)
```

Note the ordering at 12/13: `build_runtime` runs **before** `allocate_kv_cache`. The runtime must
not capture the cache — it receives it as an argument on every call.

### 2.2 Pipeline-parallel rank roles

`compute_layer_split` gives each rank a contiguous slice of layers. `PREFILL_PP_LAYER_COUNTS`
overrides the even split. `ADAPTER.layer_split_boundaries()` can constrain where a rank may start
(GLM-5.2 uses this — a rank must begin on a `full` indexer layer, because a `shared` layer has no
prior top-k to reuse).

| Rank | Input | Work | Output |
|---|---|---|---|
| first | H2D socket: token IDs + metadata | embedding + its layer slice | hidden state → D2D |
| middle | D2D: activation + metadata | its layer slice | hidden state → D2D |
| last | D2D (or H2D if `num_ranks == 1`) | its layer slice; optional KV-only last layer | nothing — the populated cache *is* the output |

`params.is_first_rank` / `is_last_rank` are what the runtime keys on. `kv_only_last_layer`
(default on, `PREFILL_KV_ONLY_LAST_LAYER=1`) makes the last layer run only the KV branch and skip
Q/SDPA/FFN/LM-head, because in a prefill-only deployment nobody reads the logits.

### 2.3 Sockets and the per-chunk loop

* **H2D (rank 0 only)** — `build_h2d_service()`, global shape `(sp, 1, chunk_size // sp)` uint32,
  plus 12 bytes of metadata: `[slot_id, actual_start, actual_end]`.
* **D2D (between ranks)** — `[1, 1, chunk_size, hidden_size]` bf16 TILE DRAM, sequence-sharded on
  the SP axis. Whether the embedding axis is TP-sharded or replicated is an adapter attribute:
  `pipeline_activation_emb_tp_sharded` (default `True`).
* **Fabric link lease/reclaim** — `_lease_reclaim()` before each chunk, `release_fabric_links()`
  after the outbound push. The engine handles this; you never touch it.
* **Shutdown** — all three metadata words `= -1`. Each rank forwards the sentinel downstream and
  exits.

The loop itself:

```python
while not _shutdown:
    _lease_reclaim(d2d_in, d2d_out)
    inp, meta, metadata_device = _socket_next(h2d_service) if is_first_rank else _d2d_recv(d2d_in)
    if _is_shutdown_sentinel(meta):
        _forward_shutdown(...); break
    _compute_and_send(runtime, kv_caches, rank, c, inp, meta, d2d_out, ...)
    c += 1
```

### 2.4 Environment variables

Every knob the engine reads, with its code default. Under `tt-run`, set these in the binding
YAML's `global_env` — a shell-exported `PREFILL_*` does **not** propagate (only the
`TT_/ARCH_/WH_/TTNN_/DEEPSEEK_/MESH_` prefixes do), which is exactly the mistake
`_assert_ranks_agree_on_config()` exists to catch.

| Var | Default | Meaning |
|---|---|---|
| `PREFILL_MANIFEST` | *(unset)* | JSON manifest applied with `setdefault` before all other reads |
| `PREFILL_MODEL` | `kimi_k2_7` | adapter registry key |
| `PREFILL_SP` / `PREFILL_TP` | `8` / `4` | mesh shape |
| `PREFILL_NUM_LAYERS` | `61` | global layer count |
| `PREFILL_CHUNK_SIZE` | `5120` | tokens per chunk |
| `PREFILL_MAX_SEQ_LEN` | `chunk * 11` | per-user cache length |
| `PREFILL_NUM_USERS` | `2` | concurrent cache slots |
| `PREFILL_CAPACITY_FACTOR` | `8` | MoE dispatch buffer capacity factor |
| `PREFILL_GATE_FALLBACK_MODE` | `ADAPTER.default_gate_mode` | MoE gate compute mode |
| `PREFILL_KV_ONLY_LAST_LAYER` | `1` | strip the tail on the last layer |
| `PREFILL_USE_TRACE` | `0` | capture/replay the per-chunk forward as a ttnn trace |
| `PREFILL_TTNN_CACHE` | `ADAPTER.ttnn_cache_default` | weight cache root |
| `PREFILL_HF_MODEL` | `ADAPTER.hf_model_default` | config/weights dir (read by the adapter) |
| `PREFILL_TRACE_DIR` | `ADAPTER.prefill_trace_default` | golden trace dir |
| `PREFILL_H2D_SERVICE_ID` | `ds_prefill` | H2D socket name — must match the producer |
| `PREFILL_ENABLE_MIGRATION` / `PREFILL_MOCK_MIGRATION` | `0` / `0` | real worker vs publish-only |
| `PREFILL_ENABLE_LAYER_ACK` | falls back to migration flag | per-layer ack channel |
| `PREFILL_FABRIC_MODE` | auto (`1D` if sp ≤ 8 else `2D`) | fabric override |
| `PREFILL_PP_LAYER_COUNTS` | even split | per-rank layer counts |
| `PREFILL_DFLASH` | `0` | attach the DFlash drafter (needs `supports_dflash` + `DFLASH_HF_MODEL`) |

### 2.5 Manifests

Rather than scattering model knobs across every rank binding, each model ships a manifest in its
own package and the binding points at it:

```json
// tt/runners/manifests/glm52.json
{
  "env": {
    "PREFILL_MODEL": "glm_5_2",
    "PREFILL_NUM_LAYERS": "78",
    "PREFILL_KV_ONLY_LAST_LAYER": "0",
    "PREFILL_GATE_FALLBACK_MODE": "DEVICE_FP32"
  }
}
```

```yaml
# binding global_env
PREFILL_MANIFEST: "models/demos/deepseek_v3_d_p/tt/runners/manifests/glm52.json"
```

Precedence: `global_env` > manifest > code default. The manifest's `env` map is applied with
`setdefault`, so a binding value still wins. This is what keeps the topology descriptors under
`common/prefill/runners/topology_configuration/` model-agnostic — the same binding runs any model
by swapping the manifest.

---

## 3. The adapter

`PrefillModelAdapter` in `common/prefill/adapter.py`. Four abstract methods plus a set of
class attributes. Everything else is inherited.

### 3.1 The class hierarchy that already exists

```
PrefillModelAdapter (ABC)
└── MLAPrefillAdapter                  adapters/mla.py — MLA + MoE over TtPrefillRuntime
    ├── DeepSeekV3Adapter              adapters/deepseek_v3.py   dense MLA, AutoConfig
    ├── KimiK26Adapter                 adapters/kimi_k2_6.py     + L1_SMALL routing, DFlash
    │   └── KimiK27Adapter             adapters/kimi_k2_7.py     checkpoint-only delta
    ├── GLM51Adapter                   adapters/glm_5_1.py       sparse DSA, dual cache
    ├── GLM52Adapter                   adapters/glm_5_2.py       + cross-layer indexer reuse
    ├── KimiK3Adapter                  adapters/kimi_k3.py       TEST-ONLY (69/93 layers are KDA)
    └── SparseMLAPrefillAdapter        adapters/sparse_mla.py    test-only DSA base
        └── DeepSeekV32Adapter         adapters/sparse_mla.py
```

**Start here**: if your model is MLA + MoE, subclass `MLAPrefillAdapter` and you may only need to
set attributes. `KimiK27Adapter` is the extreme case — it changes six attributes and nothing else:

```python
class KimiK27Adapter(KimiK26Adapter):
    name = "kimi_k2_7"
    ttnn_cache_default = "/mnt/models/moonshotai/Kimi-K2_7-Code-Cache/Kimi-K2_7-Code-Cache-prefill"
    prefill_trace_default = ".../vllm-kimi-k27-codedebug-56320"
    env_var = "KIMI_K2_7_HF_MODEL"
    default_local_path = Path("/mnt/models/moonshotai/Kimi-K2_7-Code-dequantized")
    mla_trace_defaults = ()
```

### 3.2 Class attributes

Runner-facing:

| Attribute | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | required | registry key; also the weight-cache dir prefix `{name}_{arch}_{N}dev` |
| `model_config` | `type` | required | static dimension-constants class; must expose `FABRIC_PAYLOAD_SIZE` |
| `hf_model_default` | `str` | required | config.json dir |
| `ttnn_cache_default` | `str` | required | TTNN weight-cache root |
| `prefill_trace_default` | `str` | required | golden trace dir |
| `default_gate_mode` | `str` | `"DEVICE_FP32"` | MoE gate compute mode name |
| `l1_small_size` | `int` | `0` | L1_SMALL carve-out at mesh open |
| `routing_use_l1_small_for_semaphores` | `bool` | `False` | route MoE routing all-gather semaphores to L1_SMALL |
| `pipeline_activation_emb_tp_sharded` | `bool` | `True` | D2D hidden state: TP-sharded vs replicated on the emb axis |
| `supports_dflash` | `bool` | `False` | may `PREFILL_DFLASH=1` attach the drafter |

Test-facing (only needed if you wire pytest): `hf_repo_id`, `env_var`, `default_local_path`,
`shared_path`, `num_layers_to_download`, `ttnn_cache_env`, `ref_cache_env`, `mla_ref_cache_env`,
`moe_pcc_threshold`, `mla_pcc_threshold`, `supports_pretrained`, `pretrained_mla_layer`,
`mla_trace_defaults`, `tokenizer_trust_remote_code`, `needs_flat_config_dir`, `config_builder`,
`prefill_trace_layout`, `test_prefill_trace_default`.

Note `l1_small_size` + `routing_use_l1_small_for_semaphores`: Kimi uses `768`, GLM uses `1152`,
DeepSeek-V3 uses `0`. This is not cosmetic — it is required when the model runs a single expert
group through the device gate, because the routing all-gather semaphores need a carve-out.

### 3.3 The four methods

**`load_hf_config()`** — return an HF-attribute config. Two patterns exist:

* *AutoConfig* (DeepSeek, Kimi) — the base `MLAPrefillAdapter` implementation. It also runs
  `unwrap_multimodal_config()`, which peels `text_config` off Kimi's multimodal wrapper and stubs
  `quantization_config.weight_block_size` for pre-dequantized checkpoints.
* *Hand-built* (GLM, DeepSeek-V3.2) — the model type is not `AutoConfig`-loadable, so the adapter
  calls a builder that returns a `SimpleNamespace` with the right attributes:

```python
def load_hf_config(self):
    max_seq = int(os.environ.get("PREFILL_MAX_SEQ_LEN", 8192))
    return glm_5_2_hf_config(max_seq=max_seq)
```

  Seed the builder with `max_seq` so the rope config is consistent — the runner overwrites
  `.max_seq_len` afterwards regardless.

**`weight_cache_path(mesh_shape)`** — one implementation serves everyone:

```python
env_cache = os.environ.get("PREFILL_TTNN_CACHE", self.ttnn_cache_default)
arch = "bh" if is_blackhole() else "wh"
path = Path(env_cache) / f"{self.name}_{arch}_{ttnn.get_num_devices()}dev" / f"{sp}x{tp}"
```

The important property is that it **mirrors what the cache-populate run wrote**. If they diverge,
you get a silent cache miss and random weights.

**`allocate_kv_cache(*, mesh_device, hf_config, params)`** — see §5.

**`build_runtime(*, mesh_device, hf_config, params)`** — construct the model for this rank. Read
`params`, not `os.environ`. `MLAPrefillAdapter.build_runtime` translates `PrefillRunParams` into
`TtPrefillRuntimeConfig` and returns `TtPrefillRuntime`, and every serving adapter inherits it
unchanged. You only override this if your architecture needs a different runtime class.

### 3.4 `PrefillRunParams`

The frozen dataclass the engine hands you. Read these instead of the environment:

```
mesh_shape (sp, tp)   num_layers          first_layer_idx     is_first_rank / is_last_rank
max_seq_len           chunk_size          num_users           capacity_factor
num_links             gate_mode_name      kv_only_last_layer  weight_cache_path
sp_axis / tp_axis     sparse_kv_cache_format                  use_trace
overlap_shared_expert_with_dispatch       dflash_enabled
```

`num_layers` is **this rank's** count, not the model total. `first_layer_idx` is the global index
of this rank's first layer. Both matter: the global index selects dense-vs-MoE and indexer type,
the local index addresses the KV cache slot.

### 3.5 Register it

One line in `ADAPTER_PATHS` (`common/prefill/adapter.py`), as a lazily-imported dotted string:

```python
ADAPTER_PATHS = {
    "deepseek_v3_d_p": "models.demos.deepseek_v3_d_p.tt.runners.adapters.deepseek_v3:DeepSeekV3Adapter",
    "deepseek_v32":    "models.demos.deepseek_v3_d_p.tt.runners.adapters.sparse_mla:DeepSeekV32Adapter",
    "glm_5_1":         "models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_1:GLM51Adapter",
    "glm_5_2":         "models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_2:GLM52Adapter",
    "kimi_k2_6":       "models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k2_6:KimiK26Adapter",
    "kimi_k2_7":       "models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k2_7:KimiK27Adapter",
    "minimax_m3":      "models.demos.minimax_m3.tt.runners.adapters.minimax_m3:MiniMaxM3PrefillAdapter",
    "gpt_oss_d_p":     "models.demos.gpt_oss_d_p.tt.runners.adapters.gpt_oss:GptOssPrefillAdapter",
}
```

Keep the adapter module **import-light**: no reference-modeling, device or runtime imports at
module load. The H2D producer imports adapters just to read attributes, and it has no device.

---

## 4. The runtime

Structural contract, not a base class. `TtPrefillRuntime`
(`tt/tt_prefill_runtime.py`) is the reference implementation.

```python
class PrefillRuntime:
    mesh_device: ttnn.MeshDevice
    config: "exposes chunk_size, max_seq_len, first_layer_idx, is_first_rank, is_last_rank"

    def compile(self, kv_cache) -> None: ...
    def make_chunk_input(self, token_ids: list[int]) -> ttnn.Tensor: ...
    def prefill_chunk(self, input_tensor, kv_cache, *, slot_id,
                      actual_start, actual_end, request_id=0): ...

    # optional — only if the model supports cache migration
    def build_kv_chunk_table(self, kv_cache, path: str) -> str: ...
    def kv_migration_base_address(self, kv_cache) -> int: ...
    def kv_migration_stages(self, kv_cache, first_layer_idx=None, num_my_layers=None): ...
    def set_layer_ack_channel(self, channel) -> None: ...
```

### The chunk contract

`[actual_start, actual_end)` is the absolute KV-position range of the chunk's **real** (non-pad)
tokens:

* `actual_start` is the cache write offset — the cumulative valid KV before this chunk.
* `actual_end - actual_start` is the real token count. The last chunk's tail may be pad, so
  `actual_end < actual_start + chunk_size`.
* Chunks for a slot must arrive **in order**: a chunk's KV must be written before the next chunk
  reads it.

Return this rank's output hidden state on a non-last rank, `None` on the last/single rank.

`request_id` is the engine's chunk counter. The runner always passes it, so accept it even if you
ignore it — only the pipelined layer-completion sink consumes it, to build a globally dense
`seq = request_id * num_layers + layer_idx`.

### `make_chunk_input`

```python
def make_chunk_input(self, token_ids):
    if self.config.is_first_rank:
        return prepare_prefill_input_tensor(token_ids, self.mesh_device, self.config.sp_factor,
                                            False, self.config.mesh_shape, self.config.sp_axis)
    return self.make_placeholder_activation()
```

A non-first rank does not embed — it gets a placeholder that the D2D socket overwrites with the
real activation.

### Tracing

`PREFILL_USE_TRACE=1` captures the per-chunk forward once and replays it. The per-chunk scalars
(`slot_id`, `actual_start`, `actual_end`) then have to live in *device* tensors that the runtime
updates in place, because a Python int baked into a program forces a recompile per value. This is
why `TtPrefillRuntime` keeps three persistent 1-element uint32 buffers and `update_padded_kv_cache`
has a metadata-tensor overload.

Ordering matters: `compile()` → sockets + `set_layer_ack_channel()` → `capture_trace()`. And
`release_trace()` before closing the mesh, or you segfault.

Trace is currently **dense-MLA only**. The transformer asserts against it when an indexer is
present, because the DSA ops do not yet take per-element metadata tensors.

---

## 5. KV caching

This is the part most likely to need real work for a new architecture, so it gets the most detail.

### 5.1 `KvCaches` is a marker

```python
class KvCaches(ABC):
    """Opaque handle for a model's on-device KV cache(s), returned by allocate_kv_cache."""
```

No abstract methods. Concrete subclasses are dataclasses holding device tensors. The engine
allocates it once, owns its lifetime, passes it into every runtime call that touches it, and frees
it with the mesh. It never introspects it — which is why growing or renaming a model's caches never
touches `common/`.

Existing subclasses:

```python
# deepseek_v3_d_p/tt/runners/kv_caches.py — DeepSeek / Kimi / GLM
@dataclass
class MlaKvCaches(KvCaches):
    kvpe: MlaKvCache                    # merged latent + rope
    index: Optional[ttnn.Tensor] = None # DSA lightning-indexer keys (GLM only)

# minimax_m3/tt/attention/kv_cache.py — GQA + MSA indexer
@dataclass
class MiniMaxKVCache(KvCaches):
    k: ttnn.Tensor; v: ttnn.Tensor; index_k: ttnn.Tensor
    num_users: int; num_layers: int; max_seq_len: int; sp: int
```

### 5.2 The physical substrate

Every cache in this stack is allocated by `init_kvpe_cache()`
(`utils/kv_cache_utils.py`) and shares one substrate:

```python
kvpe_cache = ttnn.allocate_tensor_on_device(
    ttnn.Shape([num_users * num_layers, 1, seq_len_local, head_dim]),
    dtype, layout, mesh_device, kv_mem_config,
)
DRAMZeroFill.op(kvpe_cache)
kvpe_cache.update_tensor_topology(ttnn.TensorTopology(dist_shape, placements, coords))
```

with

```python
kv_nd_shard_spec = ttnn.NdShardSpec(
    shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head_dim],   # 32 tokens
    grid=grid,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
)
```

Four things to internalise:

1. **Dim 0 is a user-major fold**: `batch_idx = slot_id * num_layers + layer_idx`. All users and
   all of this rank's layers live in one flat tensor.
2. **Dim 2 is SP-sharded**: `seq_len_local = max_seq_len // sp`.
3. **32 tokens per DRAM shard**, round-robin across banks. `NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32`
   is the granularity of one migration address-table entry. Everything downstream assumes it.
4. **The sequence is block-cyclic, not contiguous.** Global position `p` does not live at local row
   `p // sp`. The mapping is the inverse of the `update_padded_kv_cache` writer:

```python
def blockcyclic_positions(sp, chunk_size_global, seq_len_cache):
    """Shard row r -> chip c = r // seq_len_local, local row lr = r % seq_len_local, holding
    global position (lr // chunk_local) * chunk_size_global + c * chunk_local + (lr % chunk_local)."""
```

  With `PREFILL_CHUNK_OUTPUT_TOKENS = 5120` as the period. You need this function any time you read
  the cache back and compare against a natural-order golden.

### 5.3 The MLA merged cache

DeepSeek/Kimi/GLM store one row per token per layer: `[latent || rope]`, width
`kv_lora_rank + qk_rope_head_dim` = 512 + 64 = **576**. There is no separate K and V — MLA's
compressed latent is both.

Physical encoding is explicit, wrapped in `MlaKvCache`:

```python
class MlaKvCacheFormat(str, Enum):
    BFP8_TILE  = "bfp8_tile"   # bfloat8_b / TILE      — dense DeepSeek, Kimi
    BF16_RM    = "bf16_rm"     # bfloat16  / ROW_MAJOR — GLM-5.2 (sparse_sdpa reads it natively)
    SCALED_FP8 = "scaled_fp8"  # fp8_e4m3  / ROW_MAJOR — GLM-5.1 (512 E4M3 + 4 FP32 scales + 64 BF16 rope)

@dataclass(frozen=True)
class MlaKvCache:
    format: MlaKvCacheFormat
    storage: ttnn.Tensor
    geometry: MlaKvCacheGeometry   # latent_dim, rope_dim, packed row math
```

`MlaKvCache.pack(latent, rope)` produces the physical row; `unpack_host()` recovers logical BF16.
Keeping format and geometry next to the tensor is what lets one `sparse_sdpa` call read three
different encodings.

### 5.4 The DSA index cache

GLM (and DeepSeek-V3.2) run a lightning indexer that scores every cached position and keeps a
top-k. Its keys need their own cache — same substrate, different width:

```python
index_cache = init_kvpe_cache(
    kvpe_cache_head_dim=hf_config.index_head_dim,   # 128, not 576
    ...,
    num_kvpe_cache_layers=num_index_layers,
    dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
)
return MlaKvCaches(kvpe=kvpe_cache, index=index_cache)
```

GLM-5.2 adds cross-layer reuse: only `full` layers own an indexer and write this cache; `shared`
layers reuse a prior full layer's top-k. So the index cache is sized to the **full-layer count**,
not all layers, and slots are numbered by compacted rank:

```python
first_full       = full_indexer_rank(hf_config, params.first_layer_idx)
num_index_layers = full_indexer_rank(hf_config, params.first_layer_idx + params.num_layers) - first_full
```

That compaction is also why `GLM52Adapter` overrides `layer_split_boundaries` — a pipeline rank
must start on a `full` layer.

### 5.5 Writing a chunk

There is no `paged_fill_cache` here. The chunked path uses:

```python
ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
    cache.storage, values,
    slot_idx=cache_user_id, layer_idx=cache_layer_idx, num_layers=self.layer_num,
    kv_actual_global=kv_actual_isl,        # == actual_start
    cluster_axis=self.sp_axis,
)
```

or, under trace, the metadata-tensor overload where `slot_idx` and `kv_actual_global` are device
tensors. The op derives each chip's local write offset on-device from `kv_actual_global`, which is
what makes the block-cyclic layout work without host-side index math.

After the write, the block zeroes the pad window
(`zero_padded_kv_cache`) so a migrated prefix contains no garbage, then fires the layer ack.

Dense single-shot (non-chunked, test-only) still uses `ttnn.kv_cache.fill_cache_for_user_`.

### 5.6 What to decide for a new model

Answer these five questions before writing `allocate_kv_cache`:

1. **How many tensors?** One per semantically distinct cache. MLA: 1. GLM: 2. MiniMax-M3: 3.
2. **What is the row?** Width, and whether it is homogeneous (one dtype across the row) or mixed
   (like scaled FP8). Mixed rows need an `MlaKvCache`-style wrapper.
3. **Is every layer's cache the same length?** The `[num_users * num_layers, 1, seq_local, dim]`
   fold assumes yes. If your layer types cache different amounts (e.g. per-layer compression
   ratios), you need either separate tensors per layer class or a per-layer stride table — and the
   migration table's uniform layer stride has to follow.
4. **How does the sequence shard across SP?** If you keep the block-cyclic layout, you get
   `update_padded_kv_cache` and the existing chunk-table builder for free. Deviating means writing
   both.
5. **What is the dtype?** The read path constrains this: `sparse_sdpa` needs row-major, the ring
   MLA path wants bfp8/TILE.

---

## 6. The TT model

`TtPrefillRuntime` → `TtPrefillTransformer` → `TtPrefillBlock` → {`ttMLA`, `TtMoe` | `TtFfn`}.

### 6.1 `TtPrefillTransformer`

```
embed (first rank) → [TtPrefillBlock × num_layers] → norm + lm_head (last rank, unless kv_only)
```

Per-rank slicing:

```python
for local_idx in range(num_layers):
    layer_idx = first_layer_idx + local_idx          # GLOBAL: dense-vs-MoE, indexer type
    layer = TtPrefillBlock(..., layer_idx=layer_idx, layer_num=num_layers,
                           kv_only=kv_only_last_layer and (local_idx == num_layers - 1),
                           first_layer_idx=first_layer_idx)
```

`layer_num` is *this rank's* layer count — it is the stride in the flat KV slot
`cache_user_id * layer_num + cache_layer_idx`.

RoPE tables are built once at construction. Chunked and sparse paths need the *indexed* tables
(`get_rope_tensors_indexed`), which cover the whole cache in block-cyclic order rather than one
chunk in natural order.

### 6.2 `TtPrefillBlock`

`attn_norm → MLA → residual → ffn_norm → FFN/MoE → residual`. Both norms are
`TtDistributedRmsNorm` on the TP axis.

Dense vs MoE is one line:

```python
self.is_moe = layer_idx >= model_cfg.NUM_DENSE_LAYERS
```

`kv_only=True` builds only `attn_norm` + `ttMLA` and returns `(None, None)`.

The layer ack fires here, after the KV write and the pad-zero:

```python
ttnn.experimental.deepseek_prefill.zero_padded_kv_cache(...)
if d2h_service is not None:
    ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(d2h_service, metadata=record_dev)
else:
    on_layer_complete(self.mla.layer_idx)
```

### 6.3 `ttMLA`

Attention is bound at construction from two booleans:

```python
if self._has_indexer:
    self._attention = self._sparse_chunked_attn
else:
    self._attention = self._dense_chunked_attn if self.is_chunked else self._dense_single_attn
```

| Mode | Write | Attend |
|---|---|---|
| dense single-shot | `fill_cache_for_user_` | `ring_joint_scaled_dot_product_attention` |
| dense chunked | `update_padded_kv_cache` | `ring_mla` (SP ring SDPA over the cache) |
| sparse chunked (DSA) | `update_padded_kv_cache` ×2 | `_gather_kvpe_prefix` + `sparse_sdpa` |

`resolve_has_indexer(config)` decides, in order: an explicit override, `config.has_indexer`,
`TtIndexer.matches_config(config)` (checks `index_topk` / `index_n_heads` / `index_head_dim`), host
weights, cache completeness, else dense.

Tuned program configs live in `mla/mla_config.py`, keyed by local seq_len and gated on
`num_heads` / `q_lora_rank` / `chunked_only`. **A new head geometry needs new entries here**, or
you fall off the tuned path.

### 6.4 MoE

```
x → gate → routing setup → dispatch → routed experts → combine → reduce ─┐
x → shared expert ───────────────────────────────────────────────────────┴→ +
```

* **Gate** (`TtMoEGatePrefill`) — matmul plus a routing mode. `GateComputeMode` covers
  `DEVICE_FP32`, host fallbacks, `HASH_HOST` / `HASH_DEVICE` (DeepSeek-V4 hash routing) and
  `GPT_HOST`. Score functions and expert-group handling come from `model_cfg.SCORE_FUNC` /
  `NUM_EXPERT_GROUPS`.
* **Dispatch / combine** — all-to-all on the SP axis via fabric, into expert-centric buffers.
  Capacity factor sizes the buffer:
  `capacity_factor * dispatch_group_size * seq_len_per_chip`, plus tile alignment.
* **Routed experts** — one fused `unified_routed_expert_ffn` kernel. Activation is chosen
  explicitly by the caller from `ttnn.RoutedExpertActivation`: `Silu` (DeepSeek), `SwiGluOai`
  (clamped, MiniMax-M3 / GPT-OSS), `SituGlu` (Kimi-K3). **A new activation means a new kernel
  variant.**
* **Shared expert** — always at full `emb_dim`, optionally overlapped with dispatch on a second
  sub-device.
* **Expert weights** are sharded `(mesh_row, mesh_col)`; each chip holds `experts_per_chip`.

### 6.5 `tt_ccl.py`

A per-mesh singleton owning semaphores and every persistent CCL buffer (ring-attention K/V,
chunked-KV scratch, sparse gather buffers, high-bandwidth all-gather outputs, indexer ring K).
`per_axis_topology()` derives `(sp_topology, tp_topology)` from the opened fabric — `Ring` on a
physically wrapped axis, `Linear` otherwise — so the adapter never hardcodes it.

---

## 7. Weights and the TTNN cache

Full contract in [`tt/WEIGHTS_AND_CACHE.md`](../tt/WEIGHTS_AND_CACHE.md). Every weight-bearing
module implements the same three methods:

| Method | Kind | Purpose |
|---|---|---|
| `check_cache_complete(cache_path, ...)` | `@staticmethod` | are all `.tensorbin` files present? |
| `build_ttnn_cache(torch_weights, ..., cache_path)` | `@staticmethod` | write the cache without touching device memory |
| `_convert_and_cache_weights(torch_weights, ..., device)` | `@staticmethod` | the workhorse; `device=None` builds cache, `device=mesh` loads |

and a three-way branch in `__init__`:

```python
if torch_weights is not None:      # real weights → convert, optionally write cache
elif weight_cache_path is not None:# cache-only → pass None weights, load .tensorbin
else:                              # tests → random/dummy
```

The subtlety in the cache-only branch: `_convert_and_cache_weights(None, ...)` builds
`torch.empty()` placeholders with the **post-transform shapes** and skips the expensive host work
(`randn`, transpose, stacking). `ttnn.as_tensor` ignores the placeholder when a `.tensorbin` exists.
Get the placeholder shape wrong and you get a shape error at load, not silent corruption — which is
the point.

One deliberate deviation: `TtLatentMoeProjections` **raises** instead of falling back to random
weights, because a silently-random latent projection produces plausible-but-wrong output that no
downstream PCC check would attribute to it. Copy that pattern for any weight that is not
independently checkable.

`TtPrefillTransformer.check_cache_complete` walks embedding → each block (dense vs MoE by global
`layer_idx`) → norm + lm_head. Extend it whenever you add a weight-bearing submodule.

---

## 8. Golden traces and PCC

A golden trace is a directory:

```
<trace_dir>/
  metadata.json                     # {"token_ids": [...]}
  kv_cache/
    layer_0.safetensors             # key kv_post_transform_layer_0, shape [seq, 576]
    ...
  dsa/                              # sparse models only
    indexer_k_layer_0/rows_*.safetensors
```

Two layouts exist, selected by `adapter.prefill_trace_layout`:

* `"single_file"` — `kv_cache/layer_N.safetensors` (DeepSeek; what `generate_prompt_trace.py` writes)
* `"chunked_group_a_v1"` — `kv_cache/layer_N/rows_<start>_<end>.safetensors` (Kimi / GLM vLLM dumps)

`tt/runners/generate_prompt_trace.py` produces the first: it runs the HF reference layer by layer
and saves the post-transform KV, with the rope half de-interleaved from Meta to HF layout. It is
dense-MLA only and rejects sparse models, because it has no indexer golden.

### Validating

```bash
# terminal A — runner, publishing its KV chunk table and device map
PREFILL_MODEL=my_model PREFILL_SP=8 PREFILL_TP=4 PREFILL_H2D_SERVICE_ID=my_prefill \
PREFILL_MOCK_MIGRATION=1 \
  python -m models.demos.common.prefill.runners.prefill_runner

# terminal B — producer, pushing 11 chunks and PCC-ing the read-back KV
PREFILL_MODEL=my_model PREFILL_SP=8 PREFILL_TP=4 PREFILL_H2D_SERVICE_ID=my_prefill \
PREFILL_PRODUCER_CHUNKS=11 PREFILL_PRODUCER_CHECK_PCC=1 \
  python -m models.demos.common.prefill.runners.prefill_producer
```

The runner PCCs nothing, on any path. It publishes the table and the device map; the read-back runs
in the reader's own process, device-lessly, via `read_dram_umd`. The shared env
(`PREFILL_MODEL`, `SP`, `TP`, `CHUNK_SIZE`, `NUM_USERS`, `MAX_SEQ_LEN`, `H2D_SERVICE_ID`) must match
on both sides or the byte layout disagrees.

**One thing here is not adapter-dispatched.** The producer's
`_read_slot_kv_and_check_pcc` branches on `ADAPTER.name`:

```python
if ADAPTER.name == "minimax_m3":  return _read_slot_kv_and_check_pcc_m3(...)
if ADAPTER.name == "gpt_oss_d_p": return _read_slot_kv_and_check_pcc_gpt_oss(...)
return _read_slot_kv_and_check_pcc_mla(...)
```

A third cache layout needs a fourth branch in `common/prefill/runners/prefill_producer.py`. This is
the one place where adding a model does touch `common/`, besides the registry.

---

## 9. Migration and the KV chunk table

The migration worker copies KV from prefill DRAM to the decode cluster. It needs an address table:
for each `(layer, token_position, slot[, config_id])`, which fabric node holds the data, at what NOC
address, and how many bytes.

Two runtime hooks:

```python
def kv_migration_base_address(self, kv_cache) -> int:
    """This rank's KV base DRAM address. Enough for ONE migratable cache."""

def kv_migration_stages(self, kv_cache, first_layer_idx=None, num_my_layers=None):
    """One KvCacheStage per config of your merged table, in config order.
    Implement this instead when the model migrates SEVERAL caches, or one whose
    layer numbering is not the model's global numbering."""
```

```python
class KvCacheStage(NamedTuple):
    base_addr: int
    first_layer: int
    count: int
```

The engine all-gathers the stages across ranks and hands the merged layout to
`build_kv_chunk_table`, which serialises to protobuf. Use the shared helper for the boilerplate and
supply only your address math:

```python
def serialize_kv_chunk_table(*, table_builder, num_layers, max_seq_len, num_users,
                             chunk_n_tokens, chunk_size_bytes, path) -> str
```

Two facts to get right:

* **The table must reflect the block-cyclic layout.** A contiguous (wrong) table still works for a
  blanket copy of the *whole* cache, but any sub-cache migration — a `[0, N)` prefix, or a prompt
  shorter than `max_seq_len` — lists the wrong, scattered chunks and fails PCC. This is a
  latent bug that only surfaces under partial migration.
* **Merged tables use config IDs.** GLM's merged table is config 0 = KVPE, config 1 = index.
  MiniMax-M3 uses 2·n_kv + 1 configs. Chunk size in bytes is derived per config from dtype:
  bfp8/TILE is `(head_dim // 32) * 1088`, row-major is `32 * buffer_aligned_page_size()`.

---

## 10. Bring-up order for a new model

Ordered so that each step is independently checkable. Do not skip ahead — a PCC failure at step 8
with steps 4–7 unvalidated is very expensive to debug.

**Phase 1 — reference and constants (no device)**

1. Land a torch reference under `reference/<model>/` (usually the HF `modeling_*.py` +
   `configuration_*.py`). It is the PCC oracle for everything below.
2. Write the static dimension-constants class (`reference/<model>_config.py`): `EMB_SIZE`,
   `FABRIC_PAYLOAD_SIZE`, `NUM_LAYERS`, `NUM_DENSE_LAYERS`, `NUM_ROUTED_EXPERTS`,
   `NUM_EXPERTS_PER_TOKEN`, `MOE_INTERMEDIATE_SIZE`, `SCORE_FUNC`, `ROUTE_SCALE`, head dims, …
3. If the model type is not `AutoConfig`-loadable, write a `<model>_hf_config(max_seq=…)` builder
   returning a `SimpleNamespace` with HF attribute names. GLM's is the template.

**Phase 2 — modules, bottom up, each with its own PCC test**

4. Per-module PCC vs the reference, on a small mesh, with random weights:
   norm → embedding → FFN → gate → experts → MoE → attention → block.
   Mirror the existing tests under `tests/pcc/` and `tests/op_unit_tests/`.
5. Add tuned program configs where the geometry is new: `MLA_MATMUL_CONFIG` / `MLA_SDPA_CONFIG`
   keyed by seq_len and gated on your head count / `q_lora_rank`; `TtMoEGateConfig.mm_configs`
   keyed by `(sp, emb // tp, n_experts)`.
6. Implement `check_cache_complete` / `build_ttnn_cache` / `_convert_and_cache_weights` on every
   weight-bearing module, and extend `TtPrefillTransformer.check_cache_complete`. Test with
   `tests/cache/`.

**Phase 3 — model assembly**

7. Wire the block: dense-vs-MoE selection, attention variant, `kv_only` support.
8. Transformer-level PCC (`tests/test_prefill_transformer.py`), then the chunked variant
   (`test_prefill_transformer_chunked.py`). Chunked is where cache layout bugs surface.

**Phase 4 — serving**

9. Write the adapter. Subclass `MLAPrefillAdapter` if you can. Implement `allocate_kv_cache`;
   inherit `build_runtime` and `weight_cache_path` if the runtime fits.
10. Register in `ADAPTER_PATHS`; add a manifest JSON under `tt/runners/manifests/`.
11. Populate the TTNN weight cache. Stage a golden trace.
12. Single-rank request-mode run with the producer, no PCC. Then with
    `PREFILL_MOCK_MIGRATION=1` + `PREFILL_PRODUCER_CHECK_PCC=1`. Add a producer read-back branch if
    your cache layout is new.

**Phase 5 — scale-out**

13. Multi-rank pipeline parallel. Check `layer_split_boundaries` if your layers are not
    interchangeable. Watch the D2D activation spec if your inter-layer hidden state is not a plain
    `[1, 1, S, hidden]`.
14. Migration: `kv_migration_stages` + `build_kv_chunk_table`, then
    `PREFILL_ENABLE_MIGRATION=1` against a real worker.
15. Trace (`PREFILL_USE_TRACE=1`) if all your ops accept metadata tensors.

**Checklist**

- [ ] Adapter implements every abstract method; `name`, `model_config`, default paths set
- [ ] `build_runtime` returns a runtime matching the §4 contract (cache passed in, not stored)
- [ ] No reference-modeling / device / runtime imports at adapter module load
- [ ] Registered in `ADAPTER_PATHS`; manifest written
- [ ] Weight cache populated; golden trace staged
- [ ] Producer PCC passes with `PREFILL_PRODUCER_CHECK_PCC=1`
- [ ] Multi-rank + migration exercised if applicable

---

## 11. Where model-specific code actually lives

If you are estimating effort, this is the table to cost. Everything not listed here is generic.

| Concern | File | Selector |
|---|---|---|
| dense vs MoE layer | `tt_prefill_block.py` `__init__` | `layer_idx >= model_cfg.NUM_DENSE_LAYERS` |
| MoE dims, groups, scaling | `tt_prefill_block._build_moe` | `model_cfg.*` |
| latent MoE (Kimi-K3) | `tt_prefill_block._build_moe`, `moe/tt_moe.py` | `ROUTED_EXPERT_HIDDEN_SIZE != EMB_SIZE` |
| dense FFN width | `tt_prefill_block.__init__` | `config.intermediate_size` |
| activations | `tt_prefill_block`, `moe/tt_routed_expert.py` | `*_ACTIVATION`, `ttnn.RoutedExpertActivation` |
| sparse indexer present | `mla/indexer.resolve_has_indexer` | `index_*` config fields |
| indexer cross-layer reuse | `tt_prefill_transformer.forward` | `config.indexer_types` |
| attention variant binding | `mla/mla.__init__` | `_has_indexer`, `is_chunked` |
| NoPE / output gate | `mla/mla.py`, `mla/rope.py` | `mla_use_nope`, `mla_use_output_gate` |
| KV cache format | adapter + runtime | `sparse_kv_cache_format` |
| gate routing mode | `moe/tt_moe_gate_prefill.py` | `GateComputeMode`, `SCORE_FUNC` |
| gate matmul tuning | `moe/tt_moe_gate_prefill.py` | `(sp, emb // tp, n_experts)` |
| MLA matmul / SDPA tuning | `mla/mla_config.py` | `num_heads`, `q_lora_rank`, `chunked_only` |
| indexer key chunk | `mla/mla_config.get_indexer_key_chunk` | `DSA_INDEXER_CONFIG[index_n_heads]` |
| trace eligibility | `tt_prefill_transformer` | `assert not _has_indexer` |
| KV read-back for PCC | `common/prefill/runners/prefill_producer.py` | `ADAPTER.name` branch |
| rank start boundaries | adapter | `layer_split_boundaries` |
| L1_SMALL carve-out | adapter | `l1_small_size`, `routing_use_l1_small_for_semaphores` |
| D2D activation sharding | adapter | `pipeline_activation_emb_tp_sharded` |

**Rule of thumb from the four models that shipped:**

* *New checkpoint, same architecture* (Kimi-K2.6 → K2.7): six attributes. Hours.
* *Same family, new dims + sparse attention* (DeepSeek-V3 → GLM-5.1): a config builder, an
  `allocate_kv_cache` override, an indexer module, tuning entries. Weeks.
* *New attention mechanism or a residual structure that is not `x + f(x)`*: new TT modules, a new
  cache layout, a new chunk-table builder, a new producer read-back branch, possibly a new runtime.
  This is a quarter-scale project, not an adapter change.
