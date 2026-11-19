# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Model cache storage contract; independent of vLLM classes and scheduling.

Allocate/bind before generator preparation. K/V tensors are physical per-chip
pools, shared across request slots. Recurrent state is separate per request.
No cache allocation or replacement belongs in a captured decode loop.
"""

from dataclasses import dataclass
from numbers import Integral

import torch

import ttnn

PAGE_SIZE = 32
PAGE_WINDOW = 8
HEAD_SIZE = 256
LOCAL_KV_HEADS = 1
MAX_SLOTS = 32
AGGREGATE_TOKEN_CAPACITY = 262144


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _context(model, context_length):
    context = model.max_seq_len if context_length is None else _integer(context_length, "context_length", 1)
    if context > model.max_seq_len:
        raise ValueError("Cache context exceeds the configured model context")
    return context


def _full_layers(model):
    return [(index, layer) for index, layer in zip(model.layer_indices, model.layers) if layer.kind == "full_attention"]


def _dtype_name(dtype):
    if dtype == ttnn.bfloat16:
        return "bfloat16"
    if dtype == ttnn.bfloat8_b:
        return "bfloat8_b"
    raise ValueError(f"Unsupported selected KV dtype: {dtype}")


def _canonical_dtype(dtype):
    if dtype == torch.bfloat16 or dtype == ttnn.bfloat16 or dtype in ("bf16", "bfloat16"):
        return ttnn.bfloat16
    # BFP8 is not torch.float8: accepting float8 would misstate its storage.
    if dtype == ttnn.bfloat8_b or dtype in ("bfp8", "bfloat8_b"):
        return ttnn.bfloat8_b
    raise ValueError(f"Cache dtype {dtype!r} does not identify a supported TT cache format")


@dataclass(frozen=True)
class LayerCacheSpec:
    shape: tuple
    dtype: object
    tensor_idx: int


def describe_cache(model):
    """Return model-neutral descriptors for later adapter translation.

    Names use attention ordinals, as the pinned runner sizes its list by the
    HF attention-layer count; hf_layer_index retains the original stack index.
    No vLLM import, registration or adapter implementation occurs here.
    """
    context = _context(model, None)
    blocks = ((context + PAGE_SIZE - 1) // PAGE_SIZE + PAGE_WINDOW - 1) // PAGE_WINDOW * PAGE_WINDOW
    layers = []
    for ordinal, (index, layer) in enumerate(_full_layers(model)):
        layers.append(
            dict(
                name=f"model.layers.{ordinal}.self_attn",
                attention_index=ordinal,
                hf_layer_index=index,
                num_kv_heads=int(model.config.num_key_value_heads),
                heads_per_device=LOCAL_KV_HEADS,
                head_size=HEAD_SIZE,
                block_size=PAGE_SIZE,
                dtype=_dtype_name(layer.kv_dtype),
                layout="TILE",
                memory="DRAM",
                default_shape=[blocks, LOCAL_KV_HEADS, PAGE_SIZE, HEAD_SIZE],
                tensor_idx=ordinal,
            )
        )
    return dict(
        context_length=context,
        aggregate_token_capacity=min(context, AGGREGATE_TOKEN_CAPACITY),
        max_request_slots=MAX_SLOTS,
        mesh=[1, 4],
        layers=layers,
        recurrent=dict(
            convolution=dict(local_shape=[1, 3, 2048], dtype="bfloat16", layout="ROW_MAJOR"),
            recurrent=dict(local_shape=[1, 8, 128, 128], dtype="float32", layout="TILE"),
        ),
        page_table=dict(
            dtype="int32", layout="ROW_MAJOR", entries=blocks, padding="read-only aliases of the final logical page"
        ),
    )


def normalize_specs(model, per_layer_specs, *, context_length=None):
    """Validate all specifications before allocating any device storage."""
    context = _context(model, context_length)
    minimum_pages = ((context + PAGE_SIZE - 1) // PAGE_SIZE + PAGE_WINDOW - 1) // PAGE_WINDOW * PAGE_WINDOW
    specs = list(per_layer_specs)
    layers = _full_layers(model)
    if len(specs) != len(layers):
        raise ValueError("Supply one specification per full-attention layer, in attention-layer order")
    normalized, shared = [], {}
    for (_, layer), raw in zip(layers, specs):
        if isinstance(raw, LayerCacheSpec):
            shape, dtype, tensor_idx = raw.shape, raw.dtype, raw.tensor_idx
        else:
            if len(raw) != 3:
                raise ValueError("Each specification is (shape, dtype, tensor_idx)")
            shape, dtype, tensor_idx = raw
        shape = tuple(_integer(value, "shape dimension", 1) for value in shape)
        dtype, tensor_idx = _canonical_dtype(dtype), _integer(tensor_idx, "tensor_idx")
        if len(shape) != 4 or shape[1:] != (LOCAL_KV_HEADS, PAGE_SIZE, HEAD_SIZE):
            raise ValueError("Selected TP4 KV geometry is [physical_pages, 1, 32, 256] per chip")
        if shape[0] < minimum_pages:
            raise ValueError("Physical pool must cover round_up(ceil(context / 32), 8) pages")
        if dtype != layer.kv_dtype:
            raise ValueError("External cache dtype must preserve each selected decoder layer's KV policy")
        spec = LayerCacheSpec(shape, dtype, tensor_idx)
        old = shared.setdefault(tensor_idx, spec)
        if old.shape != shape or old.dtype != dtype:
            raise ValueError("Shared tensor_idx requires identical physical shape and dtype")
        normalized.append(spec)
    return tuple(normalized)


def allocate_kv_cache_per_layer(model, per_layer_specs, *, batch=1, context_length=None):
    """Allocate external cache usable as QwenGenerator(..., kv_cache=cache).

    Same tensor_idx means the exact same K/V objects, never separate copies.
    Slot page tables are independent. Their initial identity maps are only
    placeholders: the scheduler must assign disjoint live page ownership,
    especially when different layers share tensor_idx. See validate_page_ownership.
    """
    batch = _integer(batch, "batch", 1)
    if batch > MAX_SLOTS:
        raise ValueError("At most 32 recurrent request slots are supported")
    context = _context(model, context_length)
    specs = iter(normalize_specs(model, per_layer_specs, context_length=context))
    pools, cache = {}, []
    for layer in model.layers:
        if layer.kind == "linear_attention":
            cache.append([layer.new_recurrent_state() for _ in range(batch)])
            continue
        spec = next(specs)
        if spec.tensor_idx not in pools:
            first = layer.new_kv_state(context_length=context, num_physical_pages=spec.shape[0])
            pools[spec.tensor_idx] = (first["k"], first["v"])
            states = [first]
        else:
            states = []
        while len(states) < batch:
            states.append(
                layer.new_kv_state(
                    context_length=context, num_physical_pages=spec.shape[0], kv_cache=pools[spec.tensor_idx]
                )
            )
        for state in states:
            state["_tensor_idx"] = spec.tensor_idx
        cache.append(states)
    return cache


def allocate_kv_cache(model, shape, dtype, num_layers, *, batch=1, context_length=None):
    """Uniform legacy allocator, rejecting any per-layer dtype disagreement."""
    count = _integer(num_layers, "num_layers")
    if count != len(_full_layers(model)):
        raise ValueError("num_layers counts full-attention layers, not recurrent layers")
    return allocate_kv_cache_per_layer(
        model, [(shape, dtype, i) for i in range(count)], batch=batch, context_length=context_length
    )


def validate_cache(model, cache, *, batch):
    """Validate externally supplied storage before constructing trace inputs.

    Metadata-only: no tensor payload is copied from the device. Logical page
    ownership is validated separately using scheduler-owned CPU page tables.
    """
    batch = _integer(batch, "batch", 1)
    if batch > MAX_SLOTS or len(cache) != len(model.layers):
        raise ValueError("External cache must provide every model layer and 1..32 slots")
    mutable_addresses = set()
    pool_by_address = {}
    for layer, states in zip(model.layers, cache):
        if len(states) != batch:
            raise ValueError("External cache must contain exactly the configured number of slots")
        full = layer.kind == "full_attention"
        first_pool = None
        for state in states:
            if full:
                if "context_length" not in state:
                    raise ValueError("External full-attention state must declare context_length")
                context = _context(model, state["context_length"])
                if context != model.max_seq_len:
                    raise ValueError("Bound cache must cover the generator's configured logical context")
                pages = ((context + 31) // 32 + 7) // 8 * 8
                k, v = state["k"], state["v"]
                if k is v or k.buffer_address() == v.buffer_address():
                    raise ValueError("K and V cannot alias")
                shape = tuple(k.shape)
                if len(shape) != 4 or shape[1:] != (1, 32, 256) or shape[0] < pages:
                    raise ValueError("External KV pool has incompatible geometry")
                for tensor in (k, v):
                    if (
                        tuple(tensor.shape) != shape
                        or tensor.dtype != layer.kv_dtype
                        or tensor.layout != ttnn.TILE_LAYOUT
                        or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG
                    ):
                        raise ValueError(
                            "External KV pool must preserve decoder geometry, dtype, layout and DRAM placement"
                        )
                pool = (k.buffer_address(), v.buffer_address())
                for address in pool:
                    previous = pool_by_address.setdefault(address, pool)
                    if previous != pool:
                        raise ValueError("Shared storage must alias the complete ordered K/V pair")
                if first_pool is not None and pool != first_pool:
                    raise ValueError("Slots must share one aggregate physical pool per full-attention layer")
                first_pool = pool
                fields = [(state["page_table"], (1, pages), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)]
            else:
                fields = [
                    (state["conv"], (1, 3, 2048), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
                    (state["recurrent"], (1, 8, 128, 128), ttnn.float32, ttnn.TILE_LAYOUT),
                ]
            for tensor, expected, dtype, layout in fields:
                if (
                    tuple(tensor.shape) != expected
                    or tensor.dtype != dtype
                    or tensor.layout != layout
                    or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG
                ):
                    raise ValueError("External per-slot state has incompatible shape, dtype, layout or placement")
                address = tensor.buffer_address()
                if address in mutable_addresses:
                    raise ValueError("Mutable page tables/recurrent state cannot alias between slots or layers")
                mutable_addresses.add(address)
    return cache


def validate_page_ownership(cache, page_tables_per_layer, *, slots, prompt_lens):
    """Validate live logical pages, excluding read-only padded table entries.

    Include all live slots, not only this submitted batch. Sharing a physical
    page is unsafe between distinct request/layer owners in this recurrent
    model (automatic prefix caching is excluded). No TT readback is performed.
    """
    full = [states for states in cache if "page_table" in states[0]]
    tables = list(page_tables_per_layer)
    slots, ends = list(slots), list(prompt_lens)
    if len(tables) != len(full) or len(slots) != len(ends) or len(set(slots)) != len(slots):
        raise ValueError("Provide one table per full layer and unique live slots with absolute logical ends")
    owned = {}
    for layer_index, (states, table) in enumerate(zip(full, tables)):
        table = torch.as_tensor(table)
        if table.device.type != "cpu" or table.ndim != 2 or table.dtype not in (torch.int32, torch.int64):
            raise ValueError("Ownership validation requires CPU INT32/INT64 page tables")
        if table.shape[0] != len(slots):
            raise ValueError("Page table rows must match supplied live-slot order")
        for row, (slot, end) in enumerate(zip(slots, ends)):
            slot, end = _integer(slot, "slot"), _integer(end, "prompt_lens")
            if slot >= len(states) or end > states[slot]["context_length"]:
                raise ValueError("Slot or absolute logical end exceeds the allocated cache")
            state = states[slot]
            count = (end + PAGE_SIZE - 1) // PAGE_SIZE
            if count > table.shape[1]:
                raise ValueError("Page table is shorter than the live logical prefix")
            pages = table[row, :count].tolist()
            if len(set(pages)) != len(pages) or any(p < 0 or p >= state["k"].shape[0] for p in pages):
                raise ValueError("Logical pages must be unique and within the physical pool")
            # Address identity also detects externally constructed sharing that
            # did not use this module's optional tensor_idx metadata.
            for tensor in (state["k"], state["v"]):
                for page in pages:
                    key = (tensor.buffer_address(), page)
                    owner = (layer_index, slot)
                    if key in owned and owned[key] != owner:
                        raise ValueError("Distinct live request/layer owners overlap a shared physical page")
                    owned[key] = owner


def release_request(generator, slot):
    """Retire one request without touching shared KV or other live requests.

    Call after all outstanding output leases are consumed/discarded. KV pages
    belong to the scheduler and are overwritten by the next valid prefill;
    zeroing the aggregate pool here would corrupt unrelated live requests.
    Current device penalties are disabled, so dormant masks remain neutral.
    """
    slot = _integer(slot, "slot")
    if slot >= generator.max_batch_size or generator.closed:
        raise ValueError("Cannot release a slot outside a live generator")
    generator.output_pool.require_drained()
    if generator.slot_state.poisoned or generator.reload_executor.poisoned:
        raise RuntimeError("Reset the poisoned generator before reusing request slots")
    if generator.sampler._penalties_active:
        raise ValueError("Slot release requires the current disabled-device-penalties contract")
    try:
        for states, zeros in zip(generator.cache, generator.zeros):
            for key, source in zeros[slot].items():
                ttnn.copy(source, states[slot][key])
        generator._copy_host(torch.zeros(1, dtype=torch.int32), generator.positions[slot])
        generator.counters["position_refreshes"] += 1
        generator._set_active_slots(tuple(s for s in generator.active_slots if s != slot))
        generator.configure_sampling(temperature=0.0, top_k=1, top_p=1.0, seed=0, slots=[slot])
        generator.sampler.seed_manager.release_slot(slot)
        for history in (
            generator.sampler.tt_penalties._prompt_tokens_host,
            getattr(generator, "serving_output_history", None),
        ):
            if history is not None:
                history[slot] = -1
        # Inactive token bits intentionally remain resident; they are masked
        # and authoritative input reload overwrites them before the next use.
    except BaseException:
        generator.slot_state.poisoned = True
        generator.reload_executor.poisoned = True
        raise
