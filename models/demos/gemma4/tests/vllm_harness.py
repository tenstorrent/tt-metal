# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Test harness that reproduces vLLM's hybrid kv-cache layout offline.

Why this exists
---------------

The Gemma4 demo (``test_generator_demo.py``) and existing unit tests
(``test_attention.py``, ``test_layer.py``, ``test_model.py``) all set up
paged attention with a *uniform* layout:

* one ``block_size`` shared by every layer (typically 64), so sliding and
  full layers see identical block dimensions;
* one ``[1, max_num_blocks]`` page table per request, content
  ``arange(max_num_blocks)``, fed identically to every attention call;
* one DRAM buffer per layer (no cross-layer sharing).

vLLM produces something quite different at runtime:

* the per-block byte unifier doubles the smaller-byte spec's block_size
  (for Gemma4-E2B, sliding's ``(64, 256)`` becomes ``(128, 256)`` so it
  matches full's ``(64, 512)``);
* each *kv_cache_group* (= one attention type) has its own block table
  whose width is ``cdiv(max_model_len, group_block_size)`` — different
  widths per group;
* one physical DRAM buffer (``KVCacheTensor``) is *shared* across groups
  (HMA tensor sharing); for Gemma4-E2B's 28-sliding/7-full split, tensors
  0..6 are shared by ``(sliding[i], full[i])`` and tensors 7..27 are
  sliding-only;
* the buffer is allocated at the *first encountered layer's* spec, then
  every other sharer accesses it via ``block_size_override`` against
  ``padded_shape``.

If a model bug only shows up under that combination, the uniform-shape
tests can't catch it — the bug becomes a black-box symptom in the vLLM
server. This harness lets unit tests construct the same KV cache and
page-table layout vLLM would have produced, so prefill/decode
correctness can be asserted end-to-end without the vLLM stack.

API surface
-----------

* :class:`Gemma4VllmLayout` — pure-Python description of the layout that
  ``vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_page_size``
  +  ``unify_kv_cache_spec_page_size`` would produce for a given Gemma4
  config + requested ``block_size``. Includes the unifier output, the
  per-layer ``tensor_idx`` for HMA sharing, and the model's
  ``kv_shared_layer_map``.

* :func:`allocate_vllm_kv_cache` — build a ``list[list[k_tt, v_tt]]``
  matching that layout (HMA + kv-share aliasing), suitable for passing
  directly to ``Gemma4Model.__call__`` as ``kv_caches=`` or to the
  attention layer in unit tests as ``kv_cache=``.

* :class:`Gemma4VllmRequestPool` — block-ID allocator that mimics
  vLLM's ``BlockPool`` invariant of disjoint allocations across groups.
  Generates per-layer ``page_tables_per_layer`` lists padded to
  ``max_num_blocks_per_req``, matching what the plugin's
  ``_block_tables_per_layer`` produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

import ttnn


@dataclass(frozen=True)
class _LayerInfo:
    """One layer's vLLM-side allocation slot."""

    layer_idx: int
    layer_type: str  # "sliding_attention" | "full_attention"
    num_kv_heads_per_dev: int
    head_dim: int
    block_size: int  # post-unifier
    tensor_idx: int  # which physical buffer this layer references


@dataclass
class Gemma4VllmLayout:
    """Pre-computed layout that mirrors vLLM's hybrid kv-cache-groups output.

    Construct via :meth:`from_hf_config`; the resulting object is treated
    as immutable and can be reused across many requests and many tests.
    """

    per_layer: list[_LayerInfo]
    """Layer-aligned list, index ``i`` corresponds to model layer ``i``."""

    tensor_alloc_shape: dict[int, tuple[int, int, int, int]]
    """``tensor_idx`` → ``(num_blocks, num_kv_heads, block_size, head_dim)``
    of the *first* layer encountered for that tensor — the shape the
    allocator actually uses. Other sharers access through
    ``block_size_override`` against this allocation."""

    max_num_blocks_per_req: int
    """``cdiv(max_model_len, requested_block_size)`` — the global cap the
    plugin uses for ``model_input.block_tables`` and per-layer pads."""

    kv_shared_map: dict[int, int]
    """layer_idx → source_idx, as the model's ``kv_shared_layer_map``
    encodes Gemma3n-style KV reuse."""

    num_blocks: int
    """Total blocks per allocated buffer. Each tensor holds this many
    blocks regardless of group; vLLM's allocator divides total memory
    evenly across the ``num_unique_tensors`` buffers."""

    @property
    def num_layers(self) -> int:
        return len(self.per_layer)

    @property
    def num_unique_tensors(self) -> int:
        return len(self.tensor_alloc_shape)

    @property
    def layer_types(self) -> list[str]:
        return [li.layer_type for li in self.per_layer]

    @classmethod
    def from_hf_config(
        cls,
        hf_config: Any,
        *,
        num_blocks: int,
        max_model_len: int,
        requested_block_size: int = 64,
        num_layers: int | None = None,
        tp: int = 1,
        num_devices: int = 1,
        kv_shared_map: dict[int, int] | None = None,
    ) -> "Gemma4VllmLayout":
        """Build the layout for a Gemma4 (or Gemma4-like) HF config.

        Args:
            hf_config: HF config with ``layer_types``, ``num_key_value_heads``,
                ``head_dim``, and the Gemma4-specific ``num_global_key_value_heads``
                / ``global_head_dim`` (falls back to the sliding values when
                unset, matching :class:`Gemma4ForCausalLM.get_kv_cache_spec`).
            num_blocks: blocks per physical buffer (= vLLM's per-tensor
                ``num_blocks`` after memory budget partitioning).
            max_model_len: drives ``max_num_blocks_per_req`` exactly like
                ``cdiv(max_model_len, cache_config.block_size)`` in the plugin.
            requested_block_size: the ``cache_config.block_size`` value
                (e.g. ``64`` from ``server_example_tt.py``); the unifier
                may grow this for the smaller-byte spec but only ever
                doubles, never shrinks.
            num_layers: clip the layout to the first ``num_layers`` layers
                (matches ``Gemma4Model``'s ``num_layers`` override used by
                unit tests). ``None`` means "all layers".
            tp: tensor-parallel factor; folded into ``num_kv_heads_per_dev``.
            num_devices: total devices on the submesh; used to derive the
                per-device kv-head count consistently with the model and
                the plugin's ``_kv_cache_shape``.
            kv_shared_map: explicit override (layer_idx → source_idx).
                When ``None``, derived from ``hf_config.num_kv_shared_layers``
                the same way :class:`Gemma4Model` does in ``__init__``.
        """
        text_config = getattr(hf_config, "text_config", hf_config)
        full_layer_types = list(text_config.layer_types)
        n = num_layers if num_layers is not None else text_config.num_hidden_layers
        layer_types = full_layer_types[:n]

        sliding_kv = int(text_config.num_key_value_heads)
        sliding_hd = int(text_config.head_dim)
        full_kv = int(getattr(text_config, "num_global_key_value_heads", None) or sliding_kv)
        full_hd = int(getattr(text_config, "global_head_dim", None) or sliding_hd)

        def kv_per_dev(num_kv_heads: int) -> int:
            return 1 if num_kv_heads < num_devices else num_kv_heads // min(num_devices, num_kv_heads)

        # Match the plugin's exact formula (``_kv_cache_shape``):
        # ``num_kv_heads // min(num_devices, num_kv_heads)`` — single-device
        # passes through unchanged, multi-device shards by ``num_devices``.
        # The plugin folds TP into ``num_devices``; tests on single device
        # with ``tp>1`` are not realistic, so we just mirror the plugin.
        sliding_kv_per_dev = sliding_kv // min(num_devices, sliding_kv) if sliding_kv else 0
        full_kv_per_dev = full_kv // min(num_devices, full_kv) if full_kv else 0

        # ── Per-block byte unifier ────────────────────────────────────
        # ``unify_kv_cache_spec_page_size`` operates on ``page_size_bytes
        # = block_size * num_kv_heads * head_dim * dtype_bytes`` and
        # doubles the smaller-byte spec's block_size by integer factors
        # until per-block bytes match. We replicate it here without the
        # dtype bytes (it cancels), so the input is just
        # ``block_size * num_kv_heads * head_dim``.
        sliding_units = requested_block_size * sliding_kv_per_dev * sliding_hd
        full_units = requested_block_size * full_kv_per_dev * full_hd

        def _ratio(big: int, small: int) -> int:
            if big == small:
                return 1
            if big % small != 0:
                raise NotImplementedError(
                    f"vLLM unifier cannot unify mismatched page sizes "
                    f"{big} and {small} by block_size doubling — same constraint as upstream."
                )
            return big // small

        if sliding_units >= full_units:
            sliding_block_size = requested_block_size
            full_block_size = requested_block_size * _ratio(sliding_units, full_units)
        else:
            sliding_block_size = requested_block_size * _ratio(full_units, sliding_units)
            full_block_size = requested_block_size

        # ── Group construction (kv_cache_groups order) ───────────────
        # vLLM's ``_get_kv_cache_groups_uniform_page_size`` partitions by
        # *spec* (a layer's class + dtype + block_size + ...). For
        # Gemma4 we have at most two specs: sliding and full. Order of
        # the resulting groups list comes from a dict iteration over
        # ``KVCacheGroupSpec``, which for our get_kv_cache_spec ends up
        # in *layer-index order of first appearance*. For
        # ``layer_types[0]=='sliding_attention'`` (Gemma4-E2B), that
        # means sliding is group 0.
        groups_in_order: list[str] = []
        for lt in layer_types:
            if lt not in groups_in_order:
                groups_in_order.append(lt)

        group_members: dict[str, list[int]] = {lt: [] for lt in groups_in_order}
        for i, lt in enumerate(layer_types):
            group_members[lt].append(i)

        # ── HMA tensor_idx assignment ────────────────────────────────
        # vLLM allocates ``group_size = max(|g|)`` physical buffers.
        # Tensor ``i`` is shared by ``[group[i] for group in groups if
        # i < |group|]``. We expose ``tensor_idx`` per layer = its
        # *position* within its group's member list.
        tensor_idx_per_layer: dict[int, int] = {}
        for lt in groups_in_order:
            for pos, layer_idx in enumerate(group_members[lt]):
                tensor_idx_per_layer[layer_idx] = pos

        per_layer: list[_LayerInfo] = []
        for i, lt in enumerate(layer_types):
            if lt == "sliding_attention":
                bs, kv, hd = sliding_block_size, sliding_kv_per_dev, sliding_hd
            elif lt == "full_attention":
                bs, kv, hd = full_block_size, full_kv_per_dev, full_hd
            else:
                raise ValueError(f"Unsupported layer_type {lt!r} at layer {i}")
            per_layer.append(
                _LayerInfo(
                    layer_idx=i,
                    layer_type=lt,
                    num_kv_heads_per_dev=kv,
                    head_dim=hd,
                    block_size=bs,
                    tensor_idx=tensor_idx_per_layer[i],
                )
            )

        # ── Allocator's first-encountered shape per tensor ──────────
        # ``allocate_vllm_kv_cache_per_layer`` iterates per_layer_specs
        # in *layer-index order* and uses the first layer hitting a
        # given tensor_idx to determine the buffer shape. Replicate
        # that exact order so cross-group HMA sharers land on the
        # right reference shape.
        tensor_alloc_shape: dict[int, tuple[int, int, int, int]] = {}
        for li in per_layer:
            if li.tensor_idx in tensor_alloc_shape:
                continue
            tensor_alloc_shape[li.tensor_idx] = (
                num_blocks,
                li.num_kv_heads_per_dev,
                li.block_size,
                li.head_dim,
            )

        # ── kv-shared layer map ────────────────────────────────────
        if kv_shared_map is None:
            kv_shared_map = _derive_kv_shared_map(text_config, n)

        max_num_blocks_per_req = (max_model_len + requested_block_size - 1) // requested_block_size

        return cls(
            per_layer=per_layer,
            tensor_alloc_shape=tensor_alloc_shape,
            max_num_blocks_per_req=max_num_blocks_per_req,
            kv_shared_map=kv_shared_map,
            num_blocks=num_blocks,
        )


def _derive_kv_shared_map(text_config: Any, num_layers: int) -> dict[int, int]:
    """Replicate :class:`Gemma4Model`'s ``kv_shared_layer_map`` derivation.

    Kept in sync with ``models/demos/gemma4/tt/model.py`` lines around
    "KV sharing map" — the last ``num_kv_shared_layers`` layers reuse
    the most-recent same-type predecessor's KV cache. Updating the
    model side without updating this helper will silently drift; the
    harness will then build a layout that disagrees with the model's
    expectations.
    """
    full_n_layers = int(text_config.num_hidden_layers)
    num_kv_shared = int(getattr(text_config, "num_kv_shared_layers", 0) or 0)
    first_shared_idx = full_n_layers - num_kv_shared
    kv_shared_map: dict[int, int] = {}
    if num_kv_shared <= 0 or first_shared_idx >= num_layers:
        return kv_shared_map
    prev_layers = list(text_config.layer_types)[:first_shared_idx]
    for i in range(first_shared_idx, num_layers):
        lt = text_config.layer_types[i]
        if lt in prev_layers:
            source = len(prev_layers) - 1 - prev_layers[::-1].index(lt)
            if source < num_layers:
                kv_shared_map[i] = source
    return kv_shared_map


def allocate_vllm_kv_cache(
    mesh_device: Any,
    layout: Gemma4VllmLayout,
    *,
    dtype: Any = None,
) -> list[list[Any]]:
    """Allocate a kv_cache structure matching ``layout``.

    Returns a list ``kv[layer_idx] = [k_tt, v_tt]`` where:

    * layers sharing a ``tensor_idx`` (HMA tensor sharing) point at the
      *same* ttnn tensor objects;
    * layers in ``layout.kv_shared_map`` are aliased to their source
      layer's slot (same pattern as
      :class:`Gemma4ForCausalLM.allocate_kv_cache_per_layer`).

    The buffer for each ``tensor_idx`` is allocated at the first layer's
    spec (``layout.tensor_alloc_shape``). Other sharers must use
    ``block_size_override`` against ``padded_shape`` at every paged-cache
    call site — which the model's ``attention/{prefill,decode}.py``
    already does, via ``padded_shape[2] * padded_shape[-1] // head_dim``.
    """
    if dtype is None:
        dtype = ttnn.bfloat16
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    unique_buffers: dict[int, list[Any]] = {}
    kv_per_layer: list[list[Any]] = [None] * layout.num_layers  # type: ignore[list-item]
    for li in layout.per_layer:
        if li.tensor_idx not in unique_buffers:
            shape = layout.tensor_alloc_shape[li.tensor_idx]
            kv_pair = []
            for _ in ("k", "v"):
                kv_pair.append(
                    ttnn.as_tensor(
                        torch.zeros(shape, dtype=torch.bfloat16),
                        device=mesh_device,
                        layout=ttnn.TILE_LAYOUT,
                        dtype=dtype,
                        mesh_mapper=mesh_mapper,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                )
            unique_buffers[li.tensor_idx] = kv_pair
        kv_per_layer[li.layer_idx] = unique_buffers[li.tensor_idx]

    # Apply kv-shared aliasing exactly like ``allocate_kv_cache_per_layer``.
    for layer_idx, source_idx in layout.kv_shared_map.items():
        if 0 <= layer_idx < layout.num_layers and 0 <= source_idx < layout.num_layers:
            kv_per_layer[layer_idx] = kv_per_layer[source_idx]

    return kv_per_layer


@dataclass
class _GroupAlloc:
    """Per-(request, group) block IDs and decode-time append cursor."""

    block_ids: list[int]
    next_token_pos: int  # position where the next decode token's K/V lands


class Gemma4VllmRequestPool:
    """Allocate block IDs from a single global pool, disjointly across groups.

    vLLM's ``BlockPool`` guarantees that no two layer groups ever hold
    the same block ID at the same time (the manager allocates from one
    pool and hands out IDs unique per request). We mimic that with a
    simple bump allocator: each ``allocate_request(num_tokens)`` call
    pulls ``cdiv(num_tokens, group_block_size)`` block IDs from the
    cursor for every group, advancing the cursor as it goes. The
    resulting per-group block ID lists are disjoint by construction.

    This is intentionally simpler than vLLM's free-list semantics (we
    never release blocks back); the harness is for short-lived tests
    where each request runs to completion before the next one starts.
    """

    def __init__(self, layout: Gemma4VllmLayout):
        self._layout = layout
        # Block-size-per-group, in the same group order vLLM constructs.
        bs_per_type: dict[str, int] = {}
        for li in layout.per_layer:
            if li.layer_type not in bs_per_type:
                bs_per_type[li.layer_type] = li.block_size
        self._block_size_per_group = bs_per_type
        self._next_block_id = 0
        # ``request_id -> {layer_type: _GroupAlloc}``
        self._requests: dict[int, dict[str, _GroupAlloc]] = {}
        self._next_req_id = 0

    @property
    def layout(self) -> Gemma4VllmLayout:
        return self._layout

    def allocate_request(self, num_prefill_tokens: int) -> int:
        """Reserve enough blocks across every group to hold ``num_prefill_tokens``.

        Returns the request id, used by :meth:`per_layer_page_tables` and
        :meth:`commit_decode_token`.
        """
        req_id = self._next_req_id
        self._next_req_id += 1
        per_group: dict[str, _GroupAlloc] = {}
        for layer_type, block_size in self._block_size_per_group.items():
            n_blocks = (num_prefill_tokens + block_size - 1) // block_size
            block_ids = list(range(self._next_block_id, self._next_block_id + n_blocks))
            self._next_block_id += n_blocks
            per_group[layer_type] = _GroupAlloc(
                block_ids=block_ids,
                next_token_pos=num_prefill_tokens,
            )
        self._requests[req_id] = per_group
        return req_id

    def reserve_decode_token(self, request_id: int) -> None:
        """Advance each group's append cursor by one and extend block IDs
        when a new block is needed.

        Call before each decode step (vLLM does the equivalent when it
        rebuilds ``block_tables`` for the next decode). Position
        bookkeeping is per-group because their block sizes differ.
        """
        for layer_type, alloc in self._requests[request_id].items():
            bs = self._block_size_per_group[layer_type]
            target_blocks = (alloc.next_token_pos + 1 + bs - 1) // bs
            while len(alloc.block_ids) < target_blocks:
                alloc.block_ids.append(self._next_block_id)
                self._next_block_id += 1
            alloc.next_token_pos += 1

    def per_layer_page_tables(self, request_id: int) -> list[torch.Tensor]:
        """Build ``page_tables_per_layer`` for this request.

        Each entry is shaped ``[1, max_num_blocks_per_req]`` (the
        plugin's pad target), with the request's per-group block IDs in
        the low-index slots and zeros after.
        """
        per_group = self._requests[request_id]
        width = self._layout.max_num_blocks_per_req
        per_layer: list[torch.Tensor] = []
        for li in self._layout.per_layer:
            row = torch.zeros(1, width, dtype=torch.int32)
            block_ids = per_group[li.layer_type].block_ids
            if not block_ids:
                per_layer.append(row)
                continue
            n = min(len(block_ids), width)
            row[0, :n] = torch.tensor(block_ids[:n], dtype=torch.int32)
            per_layer.append(row)
        return per_layer

    def legacy_page_table(self, request_id: int) -> torch.Tensor:
        """Group-0's block table — the "legacy" single page_table view
        kept on ``model_input.block_tables``.

        Required by ``prepare_decode_inputs_host`` for shape-binding the
        decode trace; the per-layer routing inside the model ignores its
        *content*. Mirrors the plugin's
        ``block_tables = block_tables_per_group[0]`` line, post-padding.
        """
        first_group = next(iter(self._block_size_per_group))
        per_group = self._requests[request_id]
        width = self._layout.max_num_blocks_per_req
        row = torch.zeros(1, width, dtype=torch.int32)
        block_ids = per_group[first_group].block_ids
        if block_ids:
            n = min(len(block_ids), width)
            row[0, :n] = torch.tensor(block_ids[:n], dtype=torch.int32)
        return row


__all__ = [
    "Gemma4VllmLayout",
    "Gemma4VllmRequestPool",
    "allocate_vllm_kv_cache",
]
