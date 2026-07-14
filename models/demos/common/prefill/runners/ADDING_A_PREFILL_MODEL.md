# Adding a model to the prefill runner

The disaggregated prefill runner is a **model-agnostic engine** in the common
package (`models/demos/common/prefill/`). It owns everything that is the same for
every model:

- rank topology and the per-rank contiguous layer split (pipeline parallel),
- the H2D input socket (rank 0) and the D2D inter-rank activation sockets,
- the request (unbounded, production) and standalone (bounded, bring-up) loops,
- fabric-link lease/reclaim per chunk, per-layer LayerAck, and graceful shutdown
  (the producer/scheduler closes the request stream with an all -1 PrefillMetadata
  sentinel; each rank forwards it downstream and exits — SIGKILL is the hard fallback),
- KV-chunk-table publish + WORKER_READY handshake for cache migration.

The engine never imports a model class. It selects an adapter by name
(`PREFILL_MODEL` env var) and drives the model entirely through it:

```
common/prefill/runners/prefill_runner.py ──> PrefillModelAdapter ──build_runtime──> runtime
              (engine, common)                  (you implement)        (you implement)
```

Integrating a model is two pieces, both living in **your model's own package**:

1. a `PrefillModelAdapter` subclass — a thin factory + descriptor: it says where the
   model's config / weights / trace live, how to allocate its KV cache, and how to
   build its runtime;
2. the **runtime** object your adapter's `build_runtime` returns — knows how to run a
   chunk and read/write the cache, but holds no resources itself.

The engine owns the resources and the orchestration: it allocates the KV cache (via
the adapter) and passes it into every runtime call, and it owns the loop, the sockets,
and all comms (migration publish, the LayerAck channel, shutdown). The adapter and
runtime never trigger those. Then register the adapter and run. The interfaces below
are the entire contract.

---

## 1. The adapter interface

Subclass `PrefillModelAdapter` (`models/demos/common/prefill/adapter.py`) and
implement every abstract method. Set the identity / default-path class attributes;
they are read by the engine and the producers.

```python
from models.demos.common.prefill.adapter import PrefillModelAdapter, PrefillRunParams


class MyModelAdapter(PrefillModelAdapter):
    # --- identity & defaults (class attributes) ---
    name: str                  # registry key; also the weight-cache dir prefix ({name}_{arch}_{N}dev)
    model_config: type         # static model-dimension constants (must expose FABRIC_PAYLOAD_SIZE, etc.)
    hf_model_default: str      # config.json dir; PREFILL_HF_MODEL overrides
    ttnn_cache_default: str    # TTNN weight-cache root; PREFILL_TTNN_CACHE overrides
    prefill_trace_default: str # golden trace dir (token_ids + KV); PREFILL_TRACE_DIR overrides
    default_gate_mode: str = "DEVICE_FP32"  # MoE gate mode name; PREFILL_GATE_FALLBACK_MODE overrides
    l1_small_size: int = 0     # L1_SMALL carve-out at mesh-open (only if an op routes semaphores there)

    def load_hf_config(self):
        """Load and normalize the HF config from PREFILL_HF_MODEL (falling back to
        hf_model_default). The engine sets `.max_seq_len` on the returned config."""

    def weight_cache_path(self, mesh_shape: tuple):
        """The TTNN weight-cache dir for this model + mesh (or None if disabled).
        Mirror the layout the cache-populate run wrote so the runner reads the same files."""

    def allocate_kv_cache(self, *, mesh_device, hf_config, params: PrefillRunParams) -> KvCaches:
        """Allocate (and zero) this model's KV cache(s) on device and return them as a `KvCaches`
        (ordered tuple) — the single place your model's KV layout is defined. Index 0 is the primary
        KV cache; a dense model returns `KvCaches([kvpe])`, a sparse-attention (DSA) model appends its
        secondary cache (`KvCaches([kvpe, index])`) instead of implementing a second method. The
        ENGINE owns the returned caches: it allocates them once, passes them into every runtime call
        that touches them, and frees them with the mesh at shutdown. `params` carries max_seq_len /
        mesh_shape / this rank's num_layers / num_users / ... ."""

    def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams):
        """Construct the model for this rank and return the runtime handle (section 2).
        The runtime is stateless w.r.t. the KV cache — it receives the engine-owned cache
        as an argument on each call. The engine calls `.compile(kv_cache)` next. `params`
        carries the resolved per-rank knobs (mesh shape, this rank's num_layers /
        first_layer_idx / is_first_rank / is_last_rank, chunk_size, num_users, max_seq_len,
        capacity_factor, num_links, gate_mode_name, kv_only_last_layer, weight_cache_path)
        — read them instead of os.environ."""
```

That is the entire adapter contract — four methods plus the attributes. The adapter is
a factory + descriptor; it performs no device work or comms itself, and it does not
hold the cache (the engine does).

Test-only metadata (HF download coordinates, reference-model classes, PCC
thresholds) is optional and only needed if you wire pytest coverage; see the
attributes and lazy `reference_*_cls` properties on `PrefillModelAdapter`. Keep the
adapter module **import-light** — no reference-modeling / device / runtime imports at
module load (do them lazily inside the methods), so importing an adapter stays cheap
for the H2D producers.

---

## 2. The runtime interface

`build_runtime` returns a runtime handle. This is where the model actually lives: the
engine compiles it, drives it per chunk, and reads a few attributes off it. The
runtime knows how to read/write the KV cache but does NOT own it — the engine
allocates the cache (via the adapter's `allocate_kv_cache`) and passes it into every
call that touches it. The engine owns the loop, the sockets, the cache lifetime, and
all comms (migration publish, LayerAck channel lifecycle, shutdown).

```python
class PrefillRuntime:  # structural contract — not a base class you must inherit
    mesh_device: "ttnn.MeshDevice"
    """The open mesh device. The engine synchronizes and closes it."""

    config: "PrefillRunParams-like"
    """An object exposing at least: chunk_size, max_seq_len, first_layer_idx,
    is_first_rank, is_last_rank. The engine reads these to drive the chunk schedule
    and the per-rank pipeline role."""

    def compile(self, kv_cache: "ttnn.Tensor") -> None:
        """Warm up / compile the model so the per-chunk loop hits no first-run cost.
        The engine calls this once, after build_runtime, with the cache it owns."""

    def make_chunk_input(self, token_ids: list[int]) -> "ttnn.Tensor":
        """Build one chunk's device input for prefill_chunk: the chunk's token IDs on
        the first rank, or a placeholder hidden-state activation on a non-first pipeline
        rank (which receives the real activation over the D2D socket)."""

    def prefill_chunk(self, input_tensor, kv_cache, *, slot_id, actual_start, actual_end):
        """Prefill ONE chunk into user `slot_id`'s slice of `kv_cache` (the engine-owned
        cache, passed in), in order (a chunk's KV must be written before the next reads
        it). `[actual_start, actual_end)` is the absolute KV-position range of the chunk's
        real (non-pad) tokens: actual_start is the cache write offset, and the last chunk's
        tail may be pad (actual_end < actual_start + chunk_size). Return this rank's output
        hidden state on a non-last pipeline rank, or None on the last/single rank (the
        populated cache is the output)."""

    # --- OPTIONAL hooks; only if the model supports golden-trace bring-up / cache migration. The
    #     engine guards kv_cache_pcc_check with getattr, so a model that omits it just can't run
    #     PREFILL_STANDALONE_PCC=1. Production serving never calls any of these. Keep the heavy PCC /
    #     table logic in your model's own validation module (a thin forwarder on the runtime), not
    #     inline here — see deepseek_v3_d_p/tt/runners/prefill_kv_validation.py. ---
    def kv_cache_pcc_check(self, kv_cache, *, slot_id, n_chunks, trace_dir, first_layer_idx) -> float:
        """PCC `kv_cache` for `slot_id` against the golden trace; return the min per-layer
        PCC (asserting on failure). Called only when PREFILL_STANDALONE_PCC=1."""

    def build_kv_chunk_table(self, kv_cache, path: str) -> str:
        """Build + serialize the KV-chunk address table for `kv_cache` (your model's
        block-cyclic layout) to `path` and return it. The engine then PUBLISHES it to the
        migration worker — this method issues no comms. Called when PREFILL_ENABLE_MIGRATION=1.
        Use the shared `serialize_kv_chunk_table` helper (common/prefill/runners/migration.py)
        for the config-population + protobuf-serialize boilerplate; supply only your model's
        table builder + chunk geometry."""

    def set_layer_ack_channel(self, channel) -> None:
        """Register the per-layer LayerAck channel (the engine creates and owns it); the
        runtime bumps it once per layer so the scheduler can drive migration."""
```

---

## 3. Register it

Add one line to `ADAPTER_PATHS` in `models/demos/common/prefill/adapter.py`, mapping
the model name to its adapter class as a `"module.path:ClassName"` string (imported
lazily, so the common module never imports your model at load):

```python
ADAPTER_PATHS = {
    "deepseek_v3_d_p": "models.demos.deepseek_v3_d_p.tt.runners.adapters.deepseek_v3:DeepSeekV3Adapter",
    "kimi_k2_6": "models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k2_6:KimiK26Adapter",
    "my_model": "models.demos.my_model.tt.runners.adapters.my_model:MyModelAdapter",
}
```

`PREFILL_MODEL` selects it (default `deepseek_v3_d_p`). Under `tt-run` (which does not
propagate shell `PREFILL_*`), set it in the binding YAML's `global_env`. The same
registry feeds the pytest `variant` fixture (`tests/conftest.py`), so the adapter is
the single source of truth for both the runner and the tests.

### Per-model manifest (keeps bindings model-agnostic)

Rather than scattering `PREFILL_MODEL` and the model's knobs across every rank-binding,
put them in a per-model **manifest** in your package and point the binding at it:

```json
// models/demos/my_model/tt/runners/manifests/my_model.json
{ "env": { "PREFILL_MODEL": "my_model", "PREFILL_GATE_FALLBACK_MODE": "DEVICE_FP32" } }
```

```yaml
# the binding's global_env then needs only this model reference:
PREFILL_MANIFEST: "models/demos/my_model/tt/runners/manifests/my_model.json"
```

The manifest's `env` map is applied with `setdefault` before the runner reads its config, so a
binding `global_env` value still wins. Precedence under `tt-run`: `global_env` > manifest > code
default — uniform across ranks (model/run config must be identical on every rank). NOTE: a
shell-exported `PREFILL_*` is NOT an override here — `tt-run` only passes through the
`TT_/ARCH_/WH_/TTNN_/DEEPSEEK_/MESH_` prefixes, so override via `global_env`, not the shell.

This keeps the rank-binding + mesh-graph descriptors model-agnostic shared topology config (under
`models/demos/common/prefill/runners/topology_configuration/`) — the same binding runs any model by
swapping the manifest. (A manifest may also carry a migration `users[]` block for pairwise
KV-migration validation; see `_apply_manifest_env`.)

## 4. Validate

**Standalone + KV PCC** — single galaxy, golden-trace input, no external producer:

```bash
PREFILL_MODEL=my_model PREFILL_STANDALONE=1 PREFILL_STANDALONE_PCC=1 \
  python -m models.demos.common.prefill.runners.prefill_runner
```

**Request mode + producer** — production path (request mode is the default). The
runner builds the H2D service and exports its descriptor; the producer connects to it
by `PREFILL_H2D_SERVICE_ID` and pushes token chunks. Run two terminals; the shared
env (`PREFILL_MODEL`, `PREFILL_SP/TP`, `PREFILL_CHUNK_SIZE`, `PREFILL_NUM_USERS`,
`PREFILL_H2D_SERVICE_ID`) must match so the byte layout agrees:

```bash
# terminal A — runner (creates the H2D service, exports the descriptor, serves):
PREFILL_MODEL=my_model PREFILL_SP=8 PREFILL_TP=4 PREFILL_H2D_SERVICE_ID=my_prefill \
  python -m models.demos.common.prefill.runners.prefill_runner

# terminal B — producer (pushes PREFILL_STANDALONE_NCHUNKS chunks from the golden trace):
PREFILL_MODEL=my_model PREFILL_SP=8 PREFILL_TP=4 PREFILL_H2D_SERVICE_ID=my_prefill \
PREFILL_STANDALONE_NCHUNKS=11 \
  python -m models.demos.common.prefill.runners.prefill_producer
```

**Single-rank migration** — `PREFILL_ENABLE_MIGRATION=1` on the runner (requires the
migration endpoint up; see `deepseek_v3_d_p/tt/runners/kv_migration_setup.py`).

## Checklist

- [ ] Adapter implements every abstract `PrefillModelAdapter` method (incl.
      `allocate_kv_cache`); `name`, `model_config`, and the default paths are set.
- [ ] `build_runtime` returns a runtime satisfying the section-2 interface (cache
      passed in, not stored).
- [ ] No reference-modeling / heavy imports at module load (lazy inside methods).
- [ ] Registered in `ADAPTER_PATHS` (`models/demos/common/prefill/adapter.py`).
- [ ] Weight cache populated; golden trace staged.
- [ ] Standalone PCC run passes; request + (if applicable) migration paths exercised.
