# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The single-rank **chunked-prefill runtime**: the deployment path, and the engine's handle on it.

**HF anchor:** none — this is serving lifecycle, not model math. It wraps `tt/model.py`'s `Model`.
**Template:** `models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:96` (`TtPrefillRuntime`), second
opinion `models/demos/minimax_m3/tt/tt_prefill_runtime.py:96`.

Lifecycle: `build_runtime` -> `compile(kv_caches)` -> `prefill_chunk(...)` once per chunk, in order.
Written to the engine's contract now so P10 is wiring rather than rework
(`models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md` §2).

**The runtime does not own the KV cache.** The engine allocates it through the adapter and passes it
into every call that touches it (`ADDING_A_PREFILL_MODEL.md:101-107`), so there is no
`owns_kv_cache=True` branch here — the template has one for its standalone galaxy harness
(`models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:12-14`) and carrying it would give the cache two
owners (`DEC-062`).

**The signature is written to the engine's real call site, not to the doc.** The doc's
`prefill_chunk(input_tensor, kv_cache, *, slot_id, actual_start, actual_end, request_id=0)`
(`ADDING_A_PREFILL_MODEL.md:129`) is missing two parameters the engine **always** passes —
`d2h_service` and `metadata_msg`
(`models/demos/common/prefill/runners/prefill_runner.py:286-295`) — and a runtime written to the doc
dies with a `TypeError` on its first served chunk, after the mesh is open and the weights are loaded.
`G-RUNTIME` audits the call site with an AST walk (`tests/unit/test_prefill_runtime_chunked.py`)
rather than trusting either prose.

**What this runtime refuses, and why each refusal is load-bearing rather than a stub.** Recipe P7
requires the unsupported single-card configuration to fail loudly instead of silently running a
different attention core (`BRINGUP_RECIPE.md:1692-1697`):

* **`tp != num_key_value_heads`** — refused at construction. The packed cache holds exactly one KV
  head per chip (`tt/attention/kv_cache.py`), so at any smaller TP the model emits more local KV
  heads than the slot holds and `update_padded_kv_cache` aborts with
  `TT_FATAL: cache and input num-heads dim must match`
  (`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp:230`)
  — a message that names neither TP nor the mesh. So on a `(1,1)` mesh this runtime cannot be built
  at all, which is correct: a single card cannot serve this model's cache.
* **`actual_start > 0` on the dense (non-`sequence_parallel`) path** — refused per chunk. That is
  delta 3: chunk *k*'s queries must attend the prefix **read back out of the cache**, which plain
  `is_causal` SDPA cannot do (its mask assumes Q row 0 aligns with K row 0, so it is off by
  `actual_start` and silently wrong). `tt/attention/prefill.py::attention_forward` already refuses
  `cached_len > 0`; this refusal is the same one, one level up, so it names the phase that owns the
  fix (P8's `dense_sp.py`, gate `G-CHUNK-ATTN`, risk `R-023`) instead of surfacing as a
  `NotImplementedError` from inside a layer.
* **`set_d2h_ack_service` and `set_layer_completion_sink`** — the engine calls both unguarded on
  the paths its own env vars enable (measured by the `G-RUNTIME` AST walk, and neither appears in
  the contract doc), so both are present here and both **raise**, naming the risk they carry. A
  silently-ignored registration would let a trace or pipelined run report success having acked
  nothing. `build_kv_chunk_table`, `kv_migration_base_address` and `set_layer_ack_channel` were the
  other three; P10 implemented them (`G-KV-TABLE`, `G-MOCK-MIG`), and `build_kv_chunk_table` still
  refuses a **pipeline-rank layer slice** rather than discarding the argument (`R-032`).

**Deltas 1 and 2 are this file's, delta 3 is P8's** (`BRINGUP_RECIPE.md:1647-1654`):

| delta | what changes | where |
|---|---|---|
| 1 | the RoPE table and its per-chunk position offset — the **indexed** builder, not the contiguous one | `_build_indexed_rope`, one table per supported chunk size |
| 2 | the cache-write offset, `kv_actual_global` advancing per chunk | `prefill_chunk`'s `cached_len=actual_start` |
| 3 | chunk *k* attending the prefix read back out of the cache | `tt/attention/dense_sp.py` (P8) |
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

import ttnn
from models.demos.gpt_oss_d_p.utils.general_utils import get_default_num_links

from .attention.kv_cache import LlamaKVCache
from .ccl import CCLManager
from .config import MeshConfig
from .model import Model
from .rope import build_indexed_rope

# The deployment chunk size and per-user cache capacity (`DEC-061`, closing `R-004`). Both are
# derived, not picked:
#
# * `MAX_SEQ_LEN` = `config.json:max_position_embeddings` = 131072
#   (`bringup_log/00_MODEL_CARD.md` §2), i.e. the model's own context, not a number of ours.
# * `CHUNK_SIZE` = 8192, which satisfies both constraints and is the in-repo GQA template's own
#   default (`models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:63`). The other template uses 5120
#   (`models/demos/minimax_m3/tt/tt_prefill_runtime.py:49`), which its 8-row SP axis needs and this
#   model's 4-row axis does not — 5120 also fails `MAX_SEQ_LEN % CHUNK_SIZE == 0` at 131072:
#     - `CHUNK_SIZE % (SP * TILE_SIZE) == 0`: 8192 % (4 * 32) = 8192 % 128 = 0
#     - `MAX_SEQ_LEN % CHUNK_SIZE == 0`: 131072 / 8192 = 16 chunks
#   (both constraints: `models/demos/common/prefill/docs/PREFILL_MIGRATION_TESTING.md`, "Shared
#   setup", via `bringup_log/00_MODEL_CARD.md` §4.)
DEPLOYMENT_CHUNK_SIZE = 8192
DEPLOYMENT_MAX_SEQ_LEN = 128 * 1024


def resolve_chunk_sizes(chunk_size: int, additional_chunk_sizes, max_seq_len: int) -> tuple:
    """Supported chunk sizes, deduped, largest first. Every one must divide `max_seq_len`.

    The divisibility is not cosmetic: `tt/rope.py::build_indexed_rope` block-cyclic-reorders the
    whole-cache cos/sin by `chunk_size // sp`, so a chunk size that does not tile the cache produces
    a table whose rows do not line up with the positions the cache holds
    (`models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:46-48`).
    """
    sizes = tuple(sorted({int(chunk_size), *(int(c) for c in additional_chunk_sizes)}, reverse=True))
    for size in sizes:
        if size <= 0 or max_seq_len % size != 0:
            raise ValueError(
                f"every supported chunk size must divide max_seq_len ({max_seq_len}); {size} does "
                f"not (supported: {sizes}). build_indexed_rope tiles the cache by chunk_size // sp."
            )
    return sizes


@dataclass
class TtPrefillRuntimeConfig:
    """The runtime's own config. **A plain, mutable dataclass, deliberately.**

    Exposes the five names `ADDING_A_PREFILL_MODEL.md:116-117` requires
    (`chunk_size`, `max_seq_len`, `first_layer_idx`, `is_first_rank`, `is_last_rank`) plus
    `use_trace`, which the engine also reads off `runtime.config`
    (`models/demos/common/prefill/runners/prefill_runner.py:303`, `:745`, `:773`) and the doc does
    not mention — found by the `G-RUNTIME` AST walk, not by reading the prose.

    Not frozen: the engine assigns `max_seq_len` onto the object `load_hf_config` returns
    (`prefill_runner.py:477`), and a frozen dataclass raises `FrozenInstanceError` at runner startup
    (recipe P10 warning 2). That is the *adapter's* config rather than this one, but the same class
    of surprise applies to anything the engine holds, so nothing here is frozen either.
    """

    num_layers: int
    max_seq_len: int = DEPLOYMENT_MAX_SEQ_LEN
    chunk_size: int = DEPLOYMENT_CHUNK_SIZE
    additional_chunk_sizes: tuple = ()
    mesh_shape: tuple = (4, 8)
    num_users: int = 1
    sp_axis: int = 0
    tp_axis: int = 1
    topology: ttnn.Topology = ttnn.Topology.Ring
    # No `cache_dtype`: the cache is the **engine's** (`DEC-062`), allocated by the adapter's
    # `allocate_kv_cache` from `PrefillRunParams`, so a dtype field here would be read by nothing
    # and would invite a second, disagreeing answer to a question `DEC-021` already settled.
    weight_dtype: ttnn.DataType = ttnn.bfloat8_b  # `DEC-022`
    activation_dtype: ttnn.DataType = ttnn.bfloat16  # `DEC-022`
    weight_cache_path: Optional[Path] = None
    sequence_parallel: bool = True
    # Pipeline-parallel rank flags the engine reads off `runtime.config`. Single-rank => both True,
    # `first_layer_idx` 0. Multi-rank pipeline parallelism is an explicit non-goal for this
    # iteration (`BRINGUP_RECIPE.md:16`).
    is_first_rank: bool = True
    is_last_rank: bool = True
    first_layer_idx: int = 0
    # The engine's trace capture. `False` because this runtime implements no `capture_trace`, and
    # the engine's trace branches are all gated on this flag.
    use_trace: bool = False
    chunk_sizes: tuple = field(init=False, default=())

    def __post_init__(self):
        self.mesh_shape = tuple(self.mesh_shape)
        self.chunk_sizes = resolve_chunk_sizes(self.chunk_size, self.additional_chunk_sizes, self.max_seq_len)
        if self.sp_axis == self.tp_axis:
            raise ValueError(f"sp_axis and tp_axis must differ, both are {self.sp_axis}")
        sp = self.sp_factor
        for size in self.chunk_sizes:
            if size % (ttnn.TILE_SIZE * sp) != 0:
                raise ValueError(
                    f"chunk_size ({size}) must be a multiple of TILE_SIZE x sp "
                    f"({ttnn.TILE_SIZE * sp}); the per-chip chunk must be tile-aligned"
                )
        if self.max_seq_len % (ttnn.TILE_SIZE * sp) != 0:
            raise ValueError(
                f"max_seq_len ({self.max_seq_len}) must be a multiple of TILE_SIZE x sp "
                f"({ttnn.TILE_SIZE * sp}); the per-chip cache rows must be tile-aligned"
            )
        if self.num_users < 1:
            raise ValueError(f"num_users must be at least 1, got {self.num_users}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {self.num_layers}")

    @property
    def sp_factor(self) -> int:
        return self.mesh_shape[self.sp_axis]

    @property
    def tp_factor(self) -> int:
        return self.mesh_shape[self.tp_axis]

    @property
    def max_chunk_size(self) -> int:
        return self.chunk_sizes[0]


class TtPrefillRuntime:
    """`Model` + the per-chunk lifecycle the engine drives. Holds no KV cache."""

    def __init__(self, mesh_device, hf, state_dict, config: TtPrefillRuntimeConfig, *, ccl_manager=None):
        """
        Args:
            mesh_device: the open mesh. Its shape must equal `config.mesh_shape`.
            hf: the normalised config **dict** (recipe P1 trap 2) — `ModelArgs.hf_config`.
            state_dict: the whole HF checkpoint dict, keys unchanged; `{}` for a cache-only build,
                which requires `config.weight_cache_path`.
            config: `TtPrefillRuntimeConfig`.
            ccl_manager: an existing `CCLManager`, or `None` to build one. Passed in only by tests
                that need to inspect the semaphores; the engine path builds it here so there is
                exactly one per mesh (`bringup_log/04_CCL_PLAN.md` §2, `G-SEMAPHORE`).
        """
        self.mesh_device = mesh_device
        self.hf = hf
        self.config = config

        mesh_shape = tuple(mesh_device.shape)
        if mesh_shape != config.mesh_shape:
            raise ValueError(
                f"mesh_device shape {mesh_shape} != config.mesh_shape {config.mesh_shape}; every "
                f"cache, rope table and weight shard is built for one shape and is wrong at another"
            )
        num_kv_heads = hf["num_key_value_heads"]
        tp = mesh_shape[config.tp_axis]
        if tp != num_kv_heads:
            # The equality, not a bound. See the module docstring.
            raise ValueError(
                f"chunked prefill needs tp == num_key_value_heads: mesh axis {config.tp_axis} of "
                f"{mesh_shape} gives tp={tp} but the model has {num_kv_heads} KV heads. The packed "
                f"cache holds exactly one KV head per chip, so at tp={tp} the model emits "
                f"{num_kv_heads // tp if tp else 0} local KV heads per slot and "
                f"update_padded_kv_cache aborts with a message naming neither tp nor the mesh. A "
                f"single card cannot serve this model's KV cache "
                f"(bringup_log/00_MODEL_CARD.md section 4.1)."
            )

        self.mesh_config = MeshConfig(mesh_shape, tp=tp, tp_axis=config.tp_axis)
        self.ccl_manager = ccl_manager or CCLManager(
            mesh_device, num_links=get_default_num_links(mesh_device), topology=config.topology
        )

        logger.info(
            f"building Llama TtPrefillRuntime: layers={config.num_layers} max_seq_len={config.max_seq_len} "
            f"chunk_sizes={config.chunk_sizes} users={config.num_users} mesh={mesh_shape} "
            f"sp={self.mesh_config.sp} tp={tp} topology={config.topology}"
        )
        self.model = Model(
            mesh_device,
            hf,
            state_dict,
            ccl_manager=self.ccl_manager,
            mesh_config=self.mesh_config,
            weight_dtype=config.weight_dtype,
            activation_dtype=config.activation_dtype,
            tensor_cache_path=config.weight_cache_path,
            max_seq_len=config.max_seq_len,
            n_layers=config.num_layers,
            # The populated cache is prefill's output; logits are not (`ADDING_A_PREFILL_MODEL.md:134-136`).
            with_lm_head=False,
            sequence_parallel=config.sequence_parallel,
        )
        self.rope_indexed = self._build_indexed_rope()
        self.compiled = False
        self._on_layer_complete = None

    # ------------------------------------------------------------------------------------------
    # delta 1: the indexed RoPE, built once per supported chunk size
    # ------------------------------------------------------------------------------------------
    def _build_indexed_rope(self) -> dict:
        """`{chunk_size: [cos, sin]}` — whole-cache, block-cyclic, SP-sharded tables.

        One table per supported chunk size, because the block-cyclic period is `chunk_size // sp`
        and therefore size-specific (`models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:172-175`).
        Persistent for the runtime's life: `prefill_chunk` must not deallocate them.
        """
        return {
            size: build_indexed_rope(
                self.mesh_device,
                self.hf,
                max_seq_len=self.config.max_seq_len,
                chunk_size=size,
                sp_axis=self.config.sp_axis,
                dtype=self.config.activation_dtype,
            )
            for size in self.config.chunk_sizes
        }

    # ------------------------------------------------------------------------------------------
    # the engine-facing surface
    # ------------------------------------------------------------------------------------------
    def make_chunk_input(self, token_ids, chunk_size: Optional[int] = None) -> ttnn.Tensor:
        """One chunk's device input: SP-sharded uint32 ROW_MAJOR token ids, `[1, 1, 1, chunk_local]`.

        The same layout the engine's request-mode H2D socket delivers, so both paths feed one code
        path and `prefill_chunk` embeds on device
        (`models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:204-217`). The engine itself never calls
        this — the AST walk over `prefill_runner.py` finds no call site — but
        `ADDING_A_PREFILL_MODEL.md:124` requires it and the package's own harnesses use it, so it is
        part of the contract either way.

        On a non-first pipeline rank the real input arrives over D2D already embedded, so this
        returns a correctly-specced placeholder activation for compile warm-up.
        """
        chunk_size = self.config.chunk_size if chunk_size is None else chunk_size
        self._require_supported_chunk_size(chunk_size)
        sp = self.config.sp_factor
        chunk_local = chunk_size // sp

        if not self.config.is_first_rank:
            return ttnn.from_torch(
                torch.zeros(1, 1, chunk_local, self.hf["hidden_size"]),
                device=self.mesh_device,
                dtype=self.config.activation_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        if len(token_ids) != chunk_size:
            raise ValueError(
                f"a chunk input must be exactly chunk_size={chunk_size} token ids (pad the tail and "
                f"pass the real range as [actual_start, actual_end)), got {len(token_ids)}"
            )
        # `[sp, 1, 1, chunk_local]` sharded on the SP axis -> per-chip `[1, 1, 1, chunk_local]`, the
        # shape `tt/embedding.py::Embedding.__call__` documents. The template's per-chip tensor is
        # 3D (`models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:231`) and relies on
        # `unsqueeze_to_4D` downstream; `bringup_log/03_OUTLINE.md` §2.12 pins 4D, so 4D it is.
        tokens = torch.tensor(list(token_ids), dtype=torch.int32).reshape(sp, 1, 1, chunk_local)
        shard_dims = [None, None]
        shard_dims[self.config.sp_axis] = 0
        return ttnn.from_torch(
            tokens,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, mesh_shape=self.config.mesh_shape, dims=tuple(shard_dims)
            ),
        )

    def compile(self, kv_caches) -> None:
        """Warm the kernels by running zero-token chunks, so no served chunk pays a JIT cost.

        The engine calls this once, after `build_runtime`, with the cache it owns
        (`models/demos/common/prefill/runners/prefill_runner.py:501`). Each supported chunk size is
        warmed separately — a size whose kernels are not warmed here JITs inside the first *served*
        chunk instead (`models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:264-275`).

        A multi-chunk config additionally warms a **second** chunk at `actual_start = chunk`, whose
        cache-growth runtime arguments differ from chunk 0's. That call is what makes this method
        surface the delta-3 refusal at *compile* time rather than mid-request — see
        `prefill_chunk`.
        """
        if not self.config.is_first_rank:
            raise NotImplementedError(
                "compile on a non-first pipeline rank needs the D2D activation spec, and pipeline "
                "parallelism is out of scope for this iteration (BRINGUP_RECIPE.md line 16). "
                "Run single-rank."
            )
        for chunk_size in self.config.chunk_sizes:
            logger.info(f"TtPrefillRuntime.compile(): warming a {chunk_size}-token chunk")
            # `prefill_chunk` consumes its input, so build a fresh one per call.
            self.prefill_chunk(
                self.make_chunk_input([0] * chunk_size, chunk_size),
                kv_caches,
                slot_id=0,
                actual_start=0,
                actual_end=chunk_size,
                chunk_size=chunk_size,
            )
            if self.config.max_seq_len > chunk_size:
                logger.info(f"TtPrefillRuntime.compile(): warming a second {chunk_size}-token chunk (cache-backed)")
                self.prefill_chunk(
                    self.make_chunk_input([0] * chunk_size, chunk_size),
                    kv_caches,
                    slot_id=0,
                    actual_start=chunk_size,
                    actual_end=2 * chunk_size,
                    chunk_size=chunk_size,
                )
        ttnn.synchronize_device(self.mesh_device)
        self.compiled = True

    def prefill_chunk(
        self,
        input_tensor,
        kv_caches,
        *,
        slot_id: int,
        actual_start: int,
        actual_end: int,
        request_id: int = 0,
        d2h_service=None,
        metadata_msg=None,
        chunk_size: Optional[int] = None,
    ):
        """Prefill ONE chunk into user `slot_id`'s slice of the engine-owned `kv_caches`.

        `[actual_start, actual_end)` is the chunk's absolute KV-position range: `actual_start` is
        the cache write offset (the valid prefix already cached) **and** the indexed RoPE's
        `kv_actual_global`, which is the whole of deltas 1 and 2. The last chunk's tail may be pad,
        so `actual_end < actual_start + chunk_size` is legal.

        Returns `None` on the last/single rank — the populated cache is the output. There is no
        `skip_lm_head` or `get_last_token` knob: this runtime builds its `Model` with
        `with_lm_head=False`, so logits do not exist to be returned or sliced, and a parameter that
        cannot change the answer is a knob a caller can believe in wrongly.

        `request_id`, `d2h_service` and `metadata_msg` are the engine's, and all three are always
        passed (`prefill_runner.py:286-295`). Two are **accepted and ignored**, one is refused, and
        which is which is not guessable from the doc — it took a served chunk to find (`DEC-108`):

        * `request_id` — the engine's chunk counter. Only the pipelined layer-completion sink
          consumes it, to build a globally-dense `seq` (`ADDING_A_PREFILL_MODEL.md:139-142`), and
          this runtime is single-rank.
        * `metadata_msg` — **the H2D socket's own metadata tensor**, not a trace artefact. The engine
          reads it out of the socket (`prefill_runner.py:156-159`), decodes it into the `meta` dict
          it then passes as `slot_id` / `actual_start` / `actual_end` (`:142-144`), and hands the
          raw tensor down as well so a *pipeline* rank can forward it verbatim over D2D
          (`:304-312`). It is therefore **never `None` in request mode**, and a runtime that refuses
          a non-`None` value dies on its first served chunk — which is exactly what this one did
          until `DEC-108`. The engine owns it (it deallocates it on the shutdown sentinel, `:359`),
          so it is not touched here.
        * `d2h_service` — genuinely refused. It is `None` unless `PREFILL_LAYER_ACK_D2H=1`
          (`prefill_runner.py:707`, `:730-736`), and this runtime acks per layer through the host
          callback `set_layer_ack_channel` registers, not through device D2H records.
        """
        if d2h_service is not None:
            raise NotImplementedError(
                "this runtime emits no D2H layer-ack records: it acks per layer through the host "
                "callback set_layer_ack_channel registers, not through the device D2H record path. "
                "Run with PREFILL_LAYER_ACK_D2H=0 (the default), or wire the D2H ack path (risk R-024)."
            )
        chunk_size = self.config.chunk_size if chunk_size is None else chunk_size
        self._require_supported_chunk_size(chunk_size)

        # Every argument check runs BEFORE the cache is resolved, so a bad chunk range is refused on
        # its own terms rather than behind a cache-type error.
        if not 0 <= slot_id < self.config.num_users:
            raise ValueError(f"slot_id {slot_id} out of range [0, {self.config.num_users})")
        if not actual_start < actual_end <= actual_start + chunk_size:
            raise ValueError(
                f"[actual_start={actual_start}, actual_end={actual_end}) is not a non-empty range "
                f"inside one chunk of {chunk_size}"
            )
        if actual_start + chunk_size > self.config.max_seq_len:
            raise ValueError(
                f"a chunk at actual_start={actual_start} (+{chunk_size}) runs past the per-user "
                f"cache capacity {self.config.max_seq_len}"
            )
        if actual_start % ttnn.TILE_SIZE != 0:
            raise ValueError(
                f"actual_start ({actual_start}) must be tile-aligned: it is both the cache write "
                f"offset and the indexed RoPE's kv_actual_global, and both divide it by "
                f"{ttnn.TILE_SIZE}"
            )
        if actual_start > 0 and not self.config.sequence_parallel:
            # Delta 3. Refuse loudly instead of running a causal mask that is off by `actual_start`.
            raise NotImplementedError(
                f"a chunk at actual_start={actual_start} must attend the prefix read back out of "
                f"the cache, which the dense path cannot do: plain is_causal SDPA assumes Q row 0 "
                f"aligns with K row 0. The ring-joint path over the block-cyclic cache "
                f"(tt/attention/dense_sp.py) is P8's, gated by G-CHUNK-ATTN; risk R-023. "
                f"Chunk 0 alone is servable on this configuration."
            )

        kv_cache = self._resolve_kv(kv_caches)
        hidden_states = self.model.embedding(input_tensor) if self.config.is_first_rank else input_tensor
        if self.config.is_first_rank:
            ttnn.deallocate(input_tensor)

        out = self.model.prefill_forward(
            hidden_states,
            self.rope_indexed[chunk_size],  # persistent; never deallocated per chunk
            user_id=slot_id,
            kv_cache=kv_cache,
            skip_lm_head=True,
            on_layer_complete=self._on_layer_complete,
            cached_len=actual_start,  # delta 2 (cache write offset) and delta 1 (rope offset)
            indexed_rope=True,
        )
        # Both accepted for the engine contract and unused here; see this method's docstring.
        # `metadata_msg` is the engine's tensor and is deliberately NOT deallocated.
        del request_id, metadata_msg
        if not self.config.is_last_rank:
            # A non-last pipeline rank hands its activation to the engine's D2D socket.
            return out
        if out is not None:
            out.deallocate(True)
        return None

    # ------------------------------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------------------------------
    def _require_supported_chunk_size(self, chunk_size: int) -> None:
        if chunk_size not in self.rope_indexed:
            raise ValueError(
                f"chunk_size={chunk_size} has no indexed RoPE table; supported sizes are "
                f"{tuple(self.rope_indexed)}. Add it to config.additional_chunk_sizes before "
                f"building the runtime — a table cannot be built per chunk."
            )

    def _resolve_kv(self, kv_caches) -> LlamaKVCache:
        """The engine's `KvCaches` handle -> this model's `LlamaKVCache`.

        The engine treats the cache as opaque (`ADDING_A_PREFILL_MODEL.md:71-79`), so it hands back
        whatever `allocate_kv_cache` returned — here a `LlamaKVCache`, which is itself a
        `models.demos.common.prefill.adapter.KvCaches`. A one-element sequence is accepted too,
        because the template's own resolver takes `kv_caches[0]`
        (`models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:194-203`) and a harness written against
        it would otherwise break.

        `None` is refused: this runtime owns no cache (`DEC-062`), so there is nothing to fall back
        to — the template's `owns_kv_cache` fallback would silently write a *different* cache from
        the one the engine believes it allocated.
        """
        if kv_caches is None:
            raise ValueError(
                "prefill_chunk needs the engine-owned KV cache: this runtime owns none (DEC-062). "
                "Pass the LlamaKVCache that the adapter's allocate_kv_cache returned."
            )
        if isinstance(kv_caches, LlamaKVCache):
            return kv_caches
        if isinstance(kv_caches, (list, tuple)) and kv_caches and isinstance(kv_caches[0], LlamaKVCache):
            return kv_caches[0]
        raise TypeError(
            f"kv_caches must be a LlamaKVCache (or a one-element sequence holding one), got "
            f"{type(kv_caches).__name__}"
        )

    # ------------------------------------------------------------------------------------------
    # Optional engine hooks. Every one of these is called UNGUARDED by `prefill_runner.py` on the
    # path its env var enables (`G-RUNTIME`). The three P10 needs are implemented; the two that
    # belong to the trace and multi-rank paths refuse rather than silently doing nothing (`R-024`).
    # ------------------------------------------------------------------------------------------
    def set_layer_ack_channel(self, layer_ack_channel) -> None:
        """Register the engine's per-layer ack channel; every layer then bumps it once.

        The engine creates and owns the channel and calls this once, after `compile`, on the
        single-rank path its `PREFILL_ENABLE_LAYER_ACK` enables
        (`models/demos/common/prefill/runners/prefill_runner.py:767-768`). The seam is
        `tt/model.py::prefill_forward`'s `on_layer_complete` (`DEC-050`), so the whole wiring is the
        callback below (`models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:359-368`).

        **The count the reader expects is `num_layers * chunks`** — the producer drains exactly
        that (`prefill_producer.py:1115`) and its own PCC read-back is gated on the drain, because
        an H2D push returning is not the same as the layers being written
        (`prefill_producer.py:1065-1071`). So this must fire once per layer per chunk, on every
        chunk, which is what `prefill_chunk` passing `self._on_layer_complete` unconditionally does.

        Registered **after** `compile()` on purpose: `compile` runs throwaway chunks, and acking
        those would put `num_layers * chunk_sizes` phantom acks into the channel before the first
        real chunk, which the drain would silently absorb and then hang one chunk short at the end.
        The engine's own ordering already guarantees it (`prefill_runner.py:501` then `:768`); the
        assertion states it so a reordering fails here rather than as a hang.
        """
        assert self.compiled, (
            "call compile() before set_layer_ack_channel(): compile() runs throwaway chunks, and "
            "acking those would inject num_layers phantom acks per warmed chunk size before the "
            "first served chunk. The engine orders it correctly (prefill_runner.py:501 then :768)."
        )

        def on_layer_complete(layer_idx: int) -> None:
            layer_ack_channel.inject(1)

        self._on_layer_complete = on_layer_complete
        logger.info(
            f"LayerAck channel registered: {self.config.num_layers} acks per chunk "
            f"(the producer drains num_layers x chunks — prefill_producer.py:1115)"
        )

    def set_layer_completion_sink(self, sink) -> None:
        """Register the multi-rank layer-completion sink. **Not in this iteration.**

        Called unguarded by the engine's multi-rank migration path
        (`models/demos/common/prefill/runners/prefill_runner.py:752`), which the doc's §2 does not
        list at all. Pipeline parallelism is an explicit non-goal (`BRINGUP_RECIPE.md:16`).
        """
        raise NotImplementedError(
            "set_layer_completion_sink is the multi-rank pipeline path; this runtime is single-rank "
            "and multi-galaxy pipeline parallel is a non-goal for this iteration "
            "(BRINGUP_RECIPE.md line 16). Risk R-024."
        )

    def set_d2h_ack_service(self, d2h_service) -> None:
        """Register the D2H ack service. **Trace path; not implemented.**

        Reached only when `config.use_trace` is true (`prefill_runner.py:745`), and this runtime
        pins `use_trace=False` because it implements no `capture_trace`.
        """
        raise NotImplementedError(
            "set_d2h_ack_service belongs to the trace path: this runtime implements no "
            "capture_trace and pins config.use_trace=False. Run with PREFILL_USE_TRACE=0. Risk R-024."
        )

    def build_kv_chunk_table(self, kv_caches, path: str, *, first_layer_idx=0, num_my_layers=None, stage_layout=None):
        """Build + serialize the KV-chunk address table for `kv_caches` to `path`. Returns `path`.

        Issues **no comms** — the engine publishes the file to the migration worker and owns the
        `WORKER_READY` handshake (`ADDING_A_PREFILL_MODEL.md:146-150`). The heavy address arithmetic
        lives in `tt/runners/kv_chunk_table.py`, as the doc asks ("keep the heavy table logic in
        your model's own module ... not inline here", `:143-145`); this is the forwarder.

        The engine calls it unguarded on five sites (`prefill_runner.py:570`, `:644`, `:655`,
        `:674`, `:699`) with `path` **positional on three** and keyword on two, so the parameter is
        positional-or-keyword — measured by `G-RUNTIME`'s AST walk, and stated in neither doc.

        `first_layer_idx` / `num_my_layers` / `stage_layout` describe a pipeline rank's layer slice.
        They are **checked, not discarded**: anything but "this rank owns the whole model" raises,
        naming `R-032` (recipe P10 step 4; the template `del`s them at
        `models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:388`).

        `seq_len` is the cache's whole capacity, not the served prefix: the table addresses every
        position a slot could hold, and the reader looks up only the ones it wants
        (`prefill_producer.py:559-561` rounds its read up to a 32-token block).
        """
        from .runners.kv_chunk_table import build_and_serialize_kv_chunk_table

        kv_cache = self._resolve_kv(kv_caches)
        config = self.config
        if len(config.chunk_sizes) != 1:
            # The table describes ONE block-cyclic period, and the cache's rows were laid out by
            # whatever `prefill_chunk` actually ran. With two supported sizes no single period
            # describes the cache, and a table built for `config.chunk_size` would hand out
            # addresses that resolve, decode, and are wrong — the one silent-wrongness this file
            # can produce (`DEC-112`). The engine never passes a second size (it does not pass
            # `chunk_size` to `prefill_chunk` at all, `prefill_runner.py:287-296`), so this can only
            # be reached by a caller that set `additional_chunk_sizes`.
            raise NotImplementedError(
                f"this runtime supports {config.chunk_sizes} chunk sizes, and a KV chunk table "
                f"describes exactly one block-cyclic period. A cache written at more than one "
                f"period needs a table per period, which is unimplemented; build the runtime with a "
                f"single chunk_size (the engine always does)."
            )
        return build_and_serialize_kv_chunk_table(
            mesh_device=self.mesh_device,
            kv_cache=kv_cache,
            seq_len=config.max_seq_len,
            num_layers=config.num_layers,
            mesh_shape=config.mesh_shape,
            sp_axis=config.sp_axis,
            num_users=config.num_users,
            # The block-cyclic period is the DEPLOYED chunk size, not `max_chunk_size`: the cache's
            # rows were laid out by whatever `prefill_chunk` actually ran, and `config.chunk_size`
            # is the size the engine passed in `PrefillRunParams.chunk_size`.
            chunk_size=config.chunk_size,
            num_kv_heads=self.hf["num_key_value_heads"],
            head_dim=kv_cache.k.shape[-1],
            path=path,
            first_layer_idx=first_layer_idx,
            num_my_layers=num_my_layers,
            stage_layout=stage_layout,
        )

    def kv_migration_base_address(self, kv_caches) -> int:
        """This rank's KV base DRAM address — the anchor the engine all-gathers to merge stages.

        Reached only behind `hasattr` (`prefill_runner.py:616`), and only on the real-migration
        path; the engine wraps it in one `KvCacheStage` (`:617`) whose gathered layout comes back
        into `build_kv_chunk_table` as `stage_layout`.

        **K's base, and the asymmetry is deliberate.** This model migrates two tensors (K and V) but
        `kv_migration_stages` — the multi-cache hook — is *not* implemented, because implementing it
        would put this runtime on the engine's multi-stage merge path
        (`prefill_runner.py:613-615`, `_layout_kwarg` at `:634`), which is the unimplemented
        multi-rank merge in a different disguise (`R-032`). The table itself does not need the
        anchor: every config takes its own tensor's `buffer_address()`
        (`models/demos/gpt_oss_d_p/tt/runners/kv_chunk_table.py:138`). So what this returns is what
        the engine gathers for its stage bookkeeping, and V's base reaches the table directly.
        """
        return int(self._resolve_kv(kv_caches).k.buffer_address())
