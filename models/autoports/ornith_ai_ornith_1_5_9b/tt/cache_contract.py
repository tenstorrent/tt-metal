# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only Ornith cache specifications, measured budgets and ownership.

No vLLM or TTNN imports. Specs are metadata for a later adapter to translate.
Allocation remains model.allocate_cache(batch=..., context=..., global_pages=True,
num_blocks=...), executed during setup BEFORE traces. Register and bind that exact
returned object; release_request then targets it rather than an unused standalone
cache. A standalone cache retained alongside it still consumes measured DRAM.
Do not credit its bytes as free until traces, pending outputs and runtime bindings
have released it and a NEW memory measurement confirms the reclaimed storage.

A free-memory arithmetic bound is not proof of allocation/peak-runtime feasibility.
The public token budget requires an independently validated physical block count.
"""

from dataclasses import dataclass, replace
from math import ceil, prod
from numbers import Integral
from types import MappingProxyType

FULL_LAYERS = tuple(range(3, 32, 4))
BLOCK_SIZE = 32
PERSISTENT_CATEGORIES = frozenset(("weights", "existing_cache", "trace", "staging", "sampler", "ccl", "recurrent"))


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def dtype_name(dtype):
    value = str(getattr(dtype, "name", dtype)).lower().split(".")[-1]
    aliases = {"bf8": "bfloat8_b", "bfp8": "bfloat8_b", "bf16": "bfloat16", "fp32": "float32"}
    value = aliases.get(value, value)
    if value not in ("bfloat8_b", "bfloat16", "float32"):
        raise ValueError(f"Unsupported cache storage dtype: {dtype}")
    return value


def storage_bytes(shape, dtype, *, tiled=True):
    """Per-device bytes, including BF8 exponents and physical tile padding."""
    shape = tuple(_integer(n, "shape dimension", 1) for n in shape)
    if len(shape) < 2:
        raise ValueError("Storage geometry must have at least two dimensions")
    dtype = dtype_name(dtype)
    if tiled:
        tiles = prod(shape[:-2]) * ceil(shape[-2] / 32) * ceil(shape[-1] / 32)
        return tiles * {"bfloat8_b": 1088, "bfloat16": 2048, "float32": 4096}[dtype]
    if dtype == "bfloat8_b":
        raise ValueError("BF8 cache storage requires tiled layout")
    return prod(shape) * {"bfloat16": 2, "float32": 4}[dtype]


@dataclass(frozen=True)
class LayerCacheSpec:
    layer_index: int
    group_index: int = 0
    block_size: int = BLOCK_SIZE
    local_kv_heads: int = 1
    total_kv_heads: int = 4
    head_size: int = 256
    key_dtype: str = "bfloat8_b"
    value_dtype: str = "bfloat8_b"
    prefill_input_dtype: str = "bfloat16"

    @property
    def name(self):
        return f"model.layers.{self.layer_index}.self_attn"

    @property
    def key_bytes_per_block(self):
        return storage_bytes((1, self.local_kv_heads, self.block_size, self.head_size), self.key_dtype)

    @property
    def value_bytes_per_block(self):
        return storage_bytes((1, self.local_kv_heads, self.block_size, self.head_size), self.value_dtype)

    @property
    def bytes_per_block(self):
        return self.key_bytes_per_block + self.value_bytes_per_block


def _validate_model_layout(model, *, allow_reduced=False):
    indices = tuple(model.layer_indices)
    if not indices or len(indices) != len(model.layers) or tuple(sorted(set(indices))) != indices:
        raise ValueError("Model layer indices must be ordered, unique and match loaded layers")
    if any(index not in range(32) for index in indices) or (not allow_reduced and indices != tuple(range(32))):
        raise ValueError("Serving specs require the complete 32-layer model")
    if tuple(model.device.shape) != (1, 4):
        raise ValueError("Serving caches require the reviewed 1x4 mesh")
    for index, layer in zip(indices, model.layers):
        expected = "full_attention" if index in FULL_LAYERS else "linear_attention"
        if layer.kind != expected:
            raise ValueError("Unexpected layer topology")
    return indices


def get_kv_cache_spec(model=None, *, cache=None):
    """Eight full-attention layers in one group; recurrence is separately owned.

    With a model, reject reduced/changed architectures. With cache, inspect actual
    K and V dtype separately; do not derive one tensor's bytes from its peer.
    This is not a vLLM KVCacheSpec and makes no allocation or capability claim.
    """
    if model is None:
        if cache is not None:
            raise ValueError("Actual cache inspection requires its model")
        from models.autoports.ornith_ai_ornith_1_5_9b.tt.precision import load_policy

        dtype = load_policy()["kv_cache_dtype"]
        return {index: LayerCacheSpec(index, key_dtype=dtype, value_dtype=dtype) for index in FULL_LAYERS}
    _validate_model_layout(model)
    if cache is not None and len(cache) != 32:
        raise ValueError("Cache must cover all model layers")
    specs = {}
    for index in FULL_LAYERS:
        layer = model.layers[index]
        key_dtype = dtype_name(layer.kv_dtype if cache is None else cache[index][0].key.dtype)
        value_dtype = dtype_name(layer.kv_dtype if cache is None else cache[index][0].value.dtype)
        expected = dtype_name(layer.kv_dtype)
        if (key_dtype, value_dtype) != (expected, expected):
            raise ValueError("Cache precision differs from the constructed model policy")
        specs[index] = LayerCacheSpec(index, key_dtype=key_dtype, value_dtype=value_dtype)
    return specs


@dataclass(frozen=True)
class PoolLimit:
    num_blocks_upper_bound: int
    bytes_per_block_per_device: int
    usable_bytes_per_device: tuple
    measurement_artifact: str
    reserve_bytes_per_device: int
    allocation_verified: bool = False


@dataclass(frozen=True)
class DramBudget:
    """Four per-device FREE-byte measurements after explicitly named allocations.

    Every required category must appear exactly once in included_categories or
    pending_bytes_per_device. Pending values include peak setup/runtime overhead
    not present at measurement time; zero means explicitly known zero, not unknown.
    Add other named pending allocations (e.g. warmup snapshots) as needed. Reserve
    covers allocator/fragmentation and unaccounted peak margin; it is mandatory.
    By default the target NEW pool is not yet allocated at measurement time.
    For measurement AFTER setup, existing_target_pool_bytes_per_device must be
    actual validated target K/V bytes already included in existing_cache. This
    computes total capacity INCLUDING that retained target pool, not free space
    for a second allocation. Other existing caches are never credited.
    """

    measured_free_bytes_per_device: tuple
    measurement_artifact: str
    included_categories: tuple
    pending_bytes_per_device: object
    reserve_bytes_per_device: int
    existing_target_pool_bytes_per_device: tuple = ()

    def __post_init__(self):
        free = tuple(_integer(v, "measured free bytes") for v in self.measured_free_bytes_per_device)
        if len(free) != 4 or not self.measurement_artifact.strip():
            raise ValueError("Require four device measurements and their evidence artifact")
        included = tuple(self.included_categories)
        pending = dict(self.pending_bytes_per_device)
        if len(set(included)) != len(included) or set(included) & set(pending):
            raise ValueError("Allocations must be accounted exactly once")
        if PERSISTENT_CATEGORIES - (set(included) | set(pending)):
            raise ValueError("Every persistent allocation category must be accounted explicitly")
        pending = {name: _integer(value, f"pending {name}") for name, value in pending.items()}
        target = tuple(_integer(v, "existing target pool bytes") for v in self.existing_target_pool_bytes_per_device)
        if target and (len(target) != 4 or "existing_cache" not in included):
            raise ValueError(
                "Existing target pool credit requires four actual sizes already included in the measurement"
            )
        object.__setattr__(self, "existing_target_pool_bytes_per_device", target or (0, 0, 0, 0))
        object.__setattr__(self, "measured_free_bytes_per_device", free)
        object.__setattr__(self, "included_categories", included)
        object.__setattr__(self, "pending_bytes_per_device", MappingProxyType(pending))
        object.__setattr__(self, "reserve_bytes_per_device", _integer(self.reserve_bytes_per_device, "reserve", 1))

    def pool_limit(self, specs=None):
        specs = get_kv_cache_spec() if specs is None else specs
        if tuple(sorted(specs)) != FULL_LAYERS:
            raise ValueError("Budget must include all eight full-attention layers")
        page_bytes = sum(spec.bytes_per_block for spec in specs.values())
        pending = sum(self.pending_bytes_per_device.values()) + self.reserve_bytes_per_device
        usable = tuple(
            max(0, free + target - pending)
            for free, target in zip(self.measured_free_bytes_per_device, self.existing_target_pool_bytes_per_device)
        )
        return PoolLimit(
            min(usable) // page_bytes, page_bytes, usable, self.measurement_artifact, self.reserve_bytes_per_device
        )


def get_max_tokens_all_users(
    *,
    budget,
    validated_num_blocks,
    validation_artifact,
    max_model_len,
    max_num_seqs,
    num_devices=4,
    tt_data_parallel=1,
    specs=None,
):
    """Validated per-runner token budget; never use the plugin's guessed fallback.

    The later adapter may expose this result class-callably from a persisted,
    configuration-matched measurement. Plugin-added reservation/headroom must be
    budgeted separately; this helper does not authorize exceeding validated blocks.
    """
    if num_devices != 4 or tt_data_parallel != 1:
        raise ValueError("Only one reviewed TP4 runner is supported")
    context = _integer(max_model_len, "max_model_len", 1)
    slots = _integer(max_num_seqs, "max_num_seqs", 1)
    blocks = _integer(validated_num_blocks, "validated_num_blocks", 1)
    if context > 262144 or slots > 32 or not validation_artifact or not str(validation_artifact).strip():
        raise ValueError("Context/slots exceed contract or validation evidence is absent")
    limit = budget.pool_limit(specs)
    if blocks > limit.num_blocks_upper_bound:
        raise ValueError("Validated pool exceeds this configuration's measured budget")
    if blocks < ceil(context / 32):
        raise ValueError(
            "Pool cannot hold advertised context; provide physical-limit evidence, do not silently reduce it"
        )
    return min(blocks, slots * ceil(context / 32)) * 32


@dataclass(frozen=True)
class CacheAllocationPlan:
    batch: int
    context: int
    num_blocks: int
    estimated_kv_bytes_per_device: int
    # Scheduler storage identities, in attention-layer order 3,7,...31.
    # A unique identity owns a distinct K/V pair, not one combined tensor.
    attention_tensor_indices: tuple = ()

    def model_kwargs(self):
        return {"batch": self.batch, "context": self.context, "global_pages": True, "num_blocks": self.num_blocks}


def plan_cache_allocation(*, batch, context, num_blocks, budget=None):
    """Validate scheduler-supplied physical blocks; never substitute batch*context.

    Call model.allocate_cache(**plan.model_kwargs()) during setup, then register
    the exact result. A supplied measured budget rejects an impossible request;
    this metadata plan alone does not establish that allocation will succeed.
    """
    batch = _integer(batch, "batch", 1)
    context = _integer(context, "context", 1)
    num_blocks = _integer(num_blocks, "num_blocks", 1)
    if batch > 32 or context > 262144 or num_blocks < ceil(context / 32):
        raise ValueError("Invalid slot/context/physical-pool capacity")
    if budget is not None and num_blocks > budget.pool_limit().num_blocks_upper_bound:
        raise ValueError("Requested external pool exceeds the measured remaining budget")
    size = num_blocks * sum(spec.bytes_per_block for spec in get_kv_cache_spec().values())
    if budget is not None and any(budget.existing_target_pool_bytes_per_device):
        raise ValueError("A retained-pool budget cannot authorize a second pool allocation")
    return CacheAllocationPlan(batch, context, num_blocks, size)


def plan_cache_specs(per_layer_specs, *, batch, context, budget=None):
    """Validate the pinned plugin's eight ``(shape, dtype, tensor_idx)`` triples.

    MODEL_SETUP and TTModelRunner._build_per_layer_specs emit one triple per
    attention layer, in attention-index order. Those eight ordinal positions map
    to decoder layers FULL_LAYERS. ``tensor_idx`` labels shared layer storage;
    it does NOT identify separate key/value tensors. This model allocates a
    distinct K and V pool for every unique layer identity, preserving BF8 storage
    and BF16 prefill inputs. Cross-layer sharing is unsupported and rejected.

    Return a host-only plan; call model.allocate_cache(**plan.model_kwargs())
    before capture and bind that exact returned cache. Labels are retained in
    attention_tensor_indices for review, never treated as extra block counts.
    No allocation, device import, adapter or alias emulation occurs here.
    """
    if not isinstance(per_layer_specs, (list, tuple)) or len(per_layer_specs) != len(FULL_LAYERS):
        raise ValueError("Expected eight attention-layer cache specification triples")
    blocks, indices = [], []
    for ordinal, entry in enumerate(per_layer_specs):
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise ValueError(f"Attention layer {ordinal} requires a (shape, dtype, tensor_idx) triple")
        shape, dtype, tensor_idx = entry
        if not isinstance(shape, (list, tuple)) or len(shape) != 4:
            raise ValueError("Per-device cache shape must be (num_blocks,1,32,256)")
        shape = tuple(_integer(dimension, "cache shape dimension", 1) for dimension in shape)
        if shape[1:] != (1, BLOCK_SIZE, 256):
            raise ValueError("Cache shape differs from the optimized TP4 attention geometry")
        if dtype_name(dtype) != get_kv_cache_spec()[FULL_LAYERS[ordinal]].key_dtype:
            raise ValueError("Both K and V must match the selected cache storage policy")
        blocks.append(shape[0])
        indices.append(_integer(tensor_idx, "tensor_idx"))
    if len(set(blocks)) != 1:
        raise ValueError("The single attention group requires one physical block count across all eight layers")
    if len(set(indices)) != len(indices):
        raise ValueError("Aliased tensor_idx storage across attention layers is unsupported")
    plan = plan_cache_allocation(batch=batch, context=context, num_blocks=blocks[0], budget=budget)
    return replace(plan, attention_tensor_indices=tuple(indices))


@dataclass(frozen=True)
class CacheBinding:
    cache: object
    context: int
    slot_capacity: int
    global_pages: bool
    num_blocks: int
    kv_bytes_per_device: int
    recurrent_bytes_per_device: int
    fingerprint: tuple


def _tensor(tensor, shape, dtype):
    if tensor is None or tuple(tensor.shape) != tuple(shape) or dtype_name(tensor.dtype) != dtype:
        raise ValueError(f"Expected {dtype} cache tensor with shape {shape}")


def validate_host_page_table(page_table, binding):
    """Validate CPU integer table metadata without importing a tensor package."""
    device = getattr(page_table, "device", None)
    if getattr(device, "type", str(device)) != "cpu":
        raise ValueError("Serving page table must be CPU-resident")
    if str(getattr(page_table, "dtype", "")).split(".")[-1] not in ("int32", "int64"):
        raise ValueError("Serving page table must have int32/int64 dtype")
    expected = (binding.slot_capacity, ceil(binding.context / 32))
    if tuple(getattr(page_table, "shape", ())) != expected:
        raise ValueError(f"Serving page table must have fixed shape {expected}")
    rows = page_table.tolist()
    limit = binding.num_blocks if binding.global_pages else expected[1]
    for row in rows:
        if len(row) != expected[1] or any(
            isinstance(page, bool) or not isinstance(page, Integral) or not -1 <= page < limit for page in row
        ):
            raise ValueError("Serving table contains invalid physical page IDs")
    return tuple(tuple(int(page) for page in row) for row in rows)


class CacheOwnershipRegistry:
    """Host metadata only; caller serializes preparation, traces and readbacks.

    register() validates complete production geometry and stable tensor identities.
    Optional tensor_owner(tensor) returns a host-only mesh identity, compared to
    model.device; use it to validate imported tensors without device transfers.
    seal() prevents rebinding after capture. release_request calls the supplied
    scoped reset with the SERVING cache and does not free shared KV allocations.
    """

    def __init__(self, model, *, tensor_owner=None, allow_reduced=False, strict_tensor_metadata=False):
        _validate_model_layout(model, allow_reduced=allow_reduced)
        self.model, self.tensor_owner = model, tensor_owner
        self.strict_tensor_metadata = strict_tensor_metadata
        self.bindings, self.serving, self.sealed = {}, None, False

    def register(self, cache, *, context, global_pages, num_blocks=None):
        context = _integer(context, "context", 1)
        if context > min(262144, self.model.config.max_position_embeddings) or type(global_pages) is not bool:
            raise ValueError("Invalid context or page-address mode")
        if len(cache) != len(self.model.layers) or not cache[0]:
            raise ValueError("Cache must cover all loaded layers and at least one slot")
        capacity = len(cache[0])
        if not 1 <= capacity <= 32 or any(len(states) != capacity for states in cache):
            raise ValueError("Every layer must have identical slot capacity in [1,32]")
        logical_pages = ceil(context / 32)
        physical_pages = _integer(num_blocks, "num_blocks", 1) if num_blocks is not None else capacity * logical_pages
        if physical_pages < logical_pages or (not global_pages and physical_pages != capacity * logical_pages):
            raise ValueError("Physical pages do not match the declared address mode/context")
        seen, fingerprint, kv_bytes, recurrent_bytes = set(), [], 0, 0
        for index, (layer, states) in enumerate(zip(self.model.layers, cache)):
            first = states[0]
            if layer.kind == "full_attention":
                for state in states:
                    if state.key is not first.key or state.value is not first.value or state.key is state.value:
                        raise ValueError("Full-attention slots must share distinct K and V tensors in one layer pool")
                    if bool(state.global_pages) != global_pages or state.page_count not in (None, logical_pages):
                        raise ValueError("Cache page metadata differs from the declared binding")
                for slot, state in enumerate(states):
                    if not global_pages and state.page_offset != slot * logical_pages:
                        raise ValueError("Private slot regions must be disjoint and ordered")
                tensors = (
                    (first.key, (physical_pages, 1, 32, 256), dtype_name(layer.kv_dtype)),
                    (first.value, (physical_pages, 1, 32, 256), dtype_name(layer.kv_dtype)),
                )
            else:
                if any(
                    st.conv is not first.conv or st.recurrent is not first.recurrent or st.state_row != slot
                    for slot, st in enumerate(states)
                ):
                    raise ValueError("Linear slots must share ordered pooled convolution/recurrent state")
                tensors = (
                    (first.conv, (3, 1, capacity, 2048), "bfloat16"),
                    (first.recurrent, (capacity, 8, 128, 128), "float32"),
                )
            for tensor, shape, dtype in tensors:
                _tensor(tensor, shape, dtype)
                if self.strict_tensor_metadata:
                    layout = str(getattr(tensor, "layout", "")).lower().split(".")[-1]
                    if layout not in ("tile", "tile_layout"):
                        raise ValueError("Pooled serving cache tensors must retain tiled layout")
                    allocated = getattr(tensor, "is_allocated", None)
                    if not callable(allocated) or not allocated():
                        raise ValueError("Serving cache tensor must own allocated device storage")
                if id(tensor) in seen:
                    raise ValueError(
                        "This architecture requires distinct layer/tensor storage; aliases need explicit support"
                    )
                if self.tensor_owner is not None and self.tensor_owner(tensor) != self.model.device:
                    raise ValueError("Cache tensor belongs to another mesh")
                seen.add(id(tensor))
                size = storage_bytes(shape, tensor.dtype)
                if layer.kind == "full_attention":
                    kv_bytes += size
                else:
                    recurrent_bytes += size
            fingerprint.append(
                tuple(
                    (
                        id(st),
                        id(st.key),
                        id(st.value),
                        id(st.conv),
                        id(st.recurrent),
                        st.state_row,
                        st.page_offset,
                        st.page_count,
                        st.global_pages,
                    )
                    for st in states
                )
            )
        binding = CacheBinding(
            cache, context, capacity, global_pages, physical_pages, kv_bytes, recurrent_bytes, tuple(fingerprint)
        )
        old = self.bindings.get(id(cache))
        if old is not None:
            if (old.context, old.global_pages, old.num_blocks, old.fingerprint) != (
                context,
                binding.global_pages,
                binding.num_blocks,
                binding.fingerprint,
            ):
                raise ValueError("Registered cache backing or geometry changed")
            return old
        if self.sealed:
            raise RuntimeError("Register cache backing before capture")
        self.bindings[id(cache)] = binding
        return binding

    def register_inferred(self, cache, *, context):
        """Infer addressing from actual full-attention tensors, never batch*context."""
        if len(cache) != len(self.model.layers) or any(not states for states in cache):
            raise ValueError("Cache must cover every loaded layer")
        full = [states[0] for layer, states in zip(self.model.layers, cache) if layer.kind == "full_attention"]
        if full:
            modes = {state.global_pages for state in full}
            blocks = {int(state.key.shape[0]) for state in full}
            if len(modes) != 1 or len(blocks) != 1:
                raise ValueError("The single KV group needs consistent addressing and physical capacity")
            global_pages, num_blocks = next(iter(modes)), next(iter(blocks))
        else:
            # Reduced linear-only diagnostics have no physical KV pool. This
            # placeholder is used solely to validate their fixed host table.
            global_pages, num_blocks = True, len(cache[0]) * ceil(context / 32)
        return self.register(cache, context=context, global_pages=global_pages, num_blocks=num_blocks)

    def bind_serving(self, cache):
        binding = self.bindings.get(id(cache))
        if binding is None or binding.cache is not cache:
            raise ValueError("Register the exact serving cache first")
        if self.sealed and self.serving is not binding:
            raise RuntimeError("Serving cache cannot change after capture")
        self.serving = binding
        return binding

    def seal(self):
        if self.serving is None:
            raise ValueError("Bind the serving cache before sealing")
        self.sealed = True

    def release_request(self, slot, *, reset_slots):
        if self.serving is None:
            raise ValueError("No serving cache bound")
        slot = _integer(slot, "slot")
        if slot >= self.serving.slot_capacity:
            raise ValueError("Released slot exceeds serving capacity")
        return reset_slots([slot], kv_cache=self.serving.cache, clear_kv=False)
