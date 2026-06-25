# Adding a new model to the prefill runner

The runner (`runner.py`) and producer (`producer.py`) are model-agnostic: they drive any model through
the `PrefillModelAdapter` seam. Adding a model is **a runtime + an adapter + one registry line** — no
core changes. DeepSeek/Kimi (`models/demos/deepseek_v3_d_p/tt/runners/ds_prefill_adapter.py`) is the
worked reference; copy it and swap the bodies.

## The seam

`adapter.py` defines two Protocols. Your model's prefill pipeline is the **runtime** the runner drives;
it satisfies `PrefillRuntime`. You write a `PrefillModelAdapter` to build and validate it.

```python
class PrefillRuntime(Protocol):
    """What your build_runtime() returns — your model's prefill pipeline. The runner drives ONLY
    these members; the KV cache is intentionally absent: it's created and owned inside your runtime,
    reached only by your adapter's own methods (kv_cache_pcc_check / build_and_serialize_kv_chunk_table)."""

    mesh_device: ttnn.MeshDevice
    # config must expose: sp_factor, sp_axis, tp_factor, chunk_size, num_users,
    #                     max_seq_len, num_layers, mesh_shape
    config: object

    def compile(self) -> None: ...                 # warm-up compile (one dummy chunk)
    def prefill(self, input_tensor, slot_id, actual_start, actual_end) -> None: ...
    #   fill ONE chunk into slot_id's KV cache. [actual_start, actual_end) = absolute KV range of
    #   the real (non-pad) tokens; positions past actual_end are PAD. No sampling — the cache is the output.
    def set_layer_ack_channel(self, layer_ack_channel) -> None: ...
    #   register a per-layer ack callback (disaggregation); inject(1) once per layer. No-op-able.


class PrefillModelAdapter(Protocol):
    # --- static knobs (no device needed) ---
    name: str                          # model name; matches weight-cache dir prefix {name}_{arch}_{N}dev
    default_gate_mode: str             # gate-mode NAME as a string; PREFILL_GATE_FALLBACK_MODE overrides.
                                       #   crosses the seam as a string so common/ never imports your enum
    uses_l1_small_semaphores: bool     # True -> runner opens the mesh with an L1_SMALL region (e.g. Kimi routing)
    fabric_payload_size: int           # max fabric packet payload (your model_cfg.FABRIC_PAYLOAD_SIZE)
    h2d_mapper_config: ttnn.MeshMapperConfig  # how a token push shards across the mesh (SP shard / TP replicate)
    supports_migration: bool           # False -> skip build_and_serialize_kv_chunk_table entirely

    # --- resource resolution ---
    def load_hf_config(self, max_seq_len: int): ...
    #   load + unwrap the HF config, set max_seq_len. Opaque to the core — only handed back to build_runtime.
    def resolve_weight_cache_path(self, mesh_shape: tuple) -> Optional[Path]: ...
    #   where the .tensorbin weight cache lives (None = none). Layout: {name}_{arch}_{N}dev/{sp}x{tp}.
    def resolve_trace_dir(self) -> Path: ...                  # golden trace dir holding metadata.json
    def load_trace_token_ids(self, trace_dir, total_len=None) -> list: ...   # token_ids from metadata.json

    # --- runtime build (OWNS KV cache creation) ---
    def build_runtime(self, *, mesh_device, hf_config, mesh_shape, num_layers, max_seq_len,
                      chunk_size, num_users, capacity_factor, gate_fallback_mode,
                      weight_cache_path, kv_only_last_layer) -> PrefillRuntime: ...
    #   construct your runtime. gate_fallback_mode arrives as a string — convert to your enum here.

    # --- token input layout (model-owned) ---
    def prepare_prefill_input_tensor(self, token_ids, mesh_device, sp_factor,
                                     is_balanced, mesh_shape, sp_axis) -> ttnn.Tensor: ...
    #   token-ids -> SP-sharded uint32 ROW_MAJOR DRAM tensor the way your runtime.prefill expects.

    # --- validation (reaches into your runtime's own cache) ---
    def kv_cache_pcc_check(self, runtime, slot_id, n_chunks, trace_dir=None) -> float: ...
    #   gather your cache, restore natural order, PCC vs the golden trace. See section below.

    # --- disaggregation (only if supports_migration) ---
    def build_and_serialize_kv_chunk_table(self, runtime, path) -> str: ...
    #   build the KV chunk address table from YOUR cache layout, serialize to `path`. See section below.
```

The registry constructs your adapter with **no args**, so registering the adapter class directly is
enough (or expose a zero-arg `make_adapter()` factory if you need custom construction). DeepSeek/Kimi
are two such classes (`DeepSeekPrefillAdapter` / `KimiPrefillAdapter`) sharing a base.

## Steps

1. **Runtime** (your model code) — your model's prefill pipeline, satisfying `PrefillRuntime`. Allocate
   the KV cache however your model needs (merged kv+pe like DeepSeek, or regular separate K/V, replicated
   or TP-head-sharded). Keep it free of any `prefill_runner` import — `prepare_prefill_input_tensor` lives
   in the adapter, so there's no back-dependency.
2. **Adapter** — implement the Protocol above (copy `ds_prefill_adapter.py`).
3. **KV chunk address table** (if `supports_migration`) — see below.
4. **Register** — one line in `registry.py` (the value is `module:zero-arg-factory`, an adapter class works):
   ```python
   "<your_model_name>": "models.demos.<your_model>...:<YourAdapterClass>",
   ```
   Out-of-tree: set `PREFILL_ADAPTER_FACTORY=module.path:YourAdapterClass` instead. Select at runtime with
   `PREFILL_MODEL_NAME=<your_model_name>`.

## KV cache PCC check

This is your main correctness signal. You need a **golden trace**: a dir with `metadata.json`
(holding `token_ids`) plus per-layer KV goldens under `kv_cache/`. `kv_cache_pcc_check` then:

1. Gathers your device cache to host (`ttnn.to_torch` with the right mesh composer for your sharding).
2. **Restores natural `[position, dim]` order** — undo whatever device layout you used (DeepSeek
  un-rotates a block-cyclic SP layout via `blockcyclic_positions`; a plain TP-sharded cache instead
   concatenates the TP shards along the head dim).
3. Loads the golden per layer (a small format-agnostic loader — see `_load_golden_kv_post`) and
  `comp_pcc`s it against the gathered cache, layer by layer, asserting `>= threshold`
   (`PREFILL_STANDALONE_CHUNKED_PCC`, default 0.88; `PREFILL_STANDALONE_CHUNKED_RECORD_ONLY=1` to log
   without failing).

Watch for **RoPE basis**: DeepSeek's golden stores the HF half-split layout, so it re-interleaves the
pe slice to the Meta-interleaved basis before comparing. Match whatever basis your reference uses.
`runner_utils.kv_cache_pcc_check` is the full worked example — start there and adapt the gather +
un-rotate to your layout.

## KV chunk address table (disaggregation)

`build_and_serialize_kv_chunk_table(runtime, path)` maps `(layer, position) -> NoC address` for your
cache and serializes it via `ttnn.experimental.disaggregation.export_to_protobuf_file`. The DeepSeek
builder (`integration_setup.py`: block-cyclic, merged-kvpe, TP-replicated) is the example. A different
layout needs its own builder — e.g. **regular K/V, TP-head-sharded** (MiniMax M3) has two tensors and
per-TP-device-distinct addresses, which the current `KvChunkAddressTable` may not express; expect to
extend that ttnn API. No disaggregation? Set `supports_migration = False` and skip this.

## Running it — C1 (standalone PCC) and C2 (request mode)

Run from the repo root with the venv active. Swap `<your_model_name>` and the mesh/chunk sizes for yours.

**C1 — standalone chunked prefill + golden KV PCC** (the key correctness check; single process):

```bash
PREFILL_MODEL_NAME=<your_model_name> \
PREFILL_STANDALONE=1 PREFILL_STANDALONE_PCC=1 \
PREFILL_SP=8 PREFILL_TP=4 \
PREFILL_NUM_LAYERS=61 PREFILL_CHUNK_SIZE=5120 PREFILL_MAX_SEQ_LEN=61440 \
PREFILL_STANDALONE_NCHUNKS=11 PREFILL_NUM_USERS=2 \
  python -m models.common.prefill_runner.runner
```

PASS: stdout `[standalone] prefill_complete ...` and no PCC assertion failure. Bring-up tip: start
small (`PREFILL_NUM_LAYERS=5 PREFILL_STANDALONE_NCHUNKS=2 PREFILL_MAX_SEQ_LEN=20480`).
`PREFILL_MAX_SEQ_LEN` must be a multiple of `PREFILL_CHUNK_SIZE`, `> CHUNK_SIZE`, and
`>= NCHUNKS*CHUNK_SIZE`.

**C2 — request mode** (H2D socket; two processes). Terminal A, the runner (request mode = no
`PREFILL_STANDALONE`):

```bash
PREFILL_MODEL_NAME=<your_model_name> \
PREFILL_SP=8 PREFILL_TP=4 \
PREFILL_NUM_LAYERS=61 PREFILL_CHUNK_SIZE=5120 PREFILL_MAX_SEQ_LEN=61440 \
PREFILL_NUM_USERS=2 PREFILL_H2D_SERVICE_ID=ds_prefill \
  python -m models.common.prefill_runner.runner
# wait for: "[h2d] exported descriptor ..." then "Setup complete, entering request loop"
```

Terminal B, the producer (match SP/TP/CHUNK_SIZE/SERVICE_ID):

```bash
PREFILL_MODEL_NAME=<your_model_name> PREFILL_H2D_SERVICE_ID=ds_prefill \
PREFILL_SP=8 PREFILL_TP=4 PREFILL_CHUNK_SIZE=5120 \
PREFILL_STANDALONE_NCHUNKS=11 PREFILL_STANDALONE_ITERS=1 \
  python -m models.common.prefill_runner.producer
```

PASS: producer logs per-push timing and exits; runner logs `[request] iter=… prefill = … ms` per
chunk. The runner then idle-waits in `h2d_socket_sync` (a blocking device call) — Ctrl-C only sets a
flag checked between pushes, so to stop it: `pkill -9 -f prefill_runner.runner` then
`rm -f /dev/shm/tt_h2d_stream_service_ds_prefill.bin /dev/shm/tt_prefill_layer_acks_ds_prefill`.

> Request mode needs the H2DStreamService nanobind bindings to include `<nanobind/stl/string.h>`
> (`export_descriptor`/`connect` take `std::string`) — rebuild ttnn if `connect` raises
> "incompatible function arguments".

## Notes / gotchas

- Golden trace = `metadata.json` (`token_ids`) + per-layer `kv_cache/` goldens; your
`kv_cache_pcc_check` must match its format.
- Weight cache dir naming is `{name}_{arch}_{N}dev/{sp}x{tp}` (see `resolve_weight_cache_path`).
- The 12-byte `PrefillMetadata` wire struct (`h2d_service.H2D_METADATA_SIZE_BYTES`) is fixed — it's the
scheduler IPC contract, shared across all models.
- `gate_fallback_mode` crosses the seam as a string; never import a model's gate enum into `common/`.
