# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The engine-owned serving state of Qwen3.8-27B and its KV address table (serving_contract.md "KV cache", "Adapter
and table").

State (the device model's TtState runs on these tensors, tt/model.py):
  kv    one KV cache for every served full-attention layer: K and V, per chip [slots * n_attn, 1, capacity / sp, 256]
        bfloat8_b, kv head c on TP column c, ND-sharded [1, 1, 32, 256] ROUND_ROBIN_1D over the DRAM banks
        (tt/attention.py kv_memory_config), user-major (slot s, attention ordinal a at batch s * n_attn + a). Local row
        lr of SP row r holds position (lr // (chunk / sp)) * chunk + r * (chunk / sp) + lr % (chunk / sp).
  gdn   per (slot, linear layer) the Gated DeltaNet conv history (bf16) and recurrent state (fp32), not in the table.

Table: configs 0..n_kv-1 = K heads, n_kv..2 n_kv-1 = V heads (the gpt_oss_d_p GQA layout), each record 32 tokens of
one head = 8 bfp8 tiles (8704 B) in one bank of one chip (device group = that chip). Published on the model's layer
axis: num_layers = the served layers, entries only on the attention layers (linear layers stay unset, size 0).
"""

from __future__ import annotations

import socket
from dataclasses import dataclass, field

import torch

import ttnn
from models.demos.common.prefill.adapter import KvCaches

from ..attention import KV_RECORD, KvCache, kv_memory_config
from ..linear_attention import CONV_K, GdnState

TILE_BYTES = {ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}


@dataclass
class Qwen38KvCaches(KvCaches):
    kv: KvCache | None
    gdn: dict = field(default_factory=dict)  # (slot, global layer) -> GdnState
    attn_layers: list = field(default_factory=list)  # global ids, in cache order
    linear_layers: list = field(default_factory=list)
    slots: int = 1
    capacity: int = 0  # tokens per slot
    chunk: int = 0  # the block-cyclic period (the served chunk)


def allocate(mesh, cfg, layers, slots: int, chunk: int, max_seq: int) -> Qwen38KvCaches:
    """The zero serving state for ``layers`` (global ids), ``slots`` request slots, ``max_seq`` tokens per slot."""
    rows, cols = tuple(mesh.shape)
    capacity = -(-max_seq // chunk) * chunk
    assert chunk % (rows * KV_RECORD) == 0, f"chunk {chunk} over {rows} SP rows is not whole 32-token records"
    assert cfg.n_kv == cols, "one kv head per TP column"
    attn = [i for i in layers if not cfg.is_linear(i)]
    linear = [i for i in layers if cfg.is_linear(i)]

    kv = None
    if attn:
        mapper = ttnn.ShardTensor2dMesh(mesh, mesh_shape=(rows, cols), dims=(2, 1))
        mem = kv_memory_config(mesh, cfg.head_dim)

        def zero_kv():
            return ttnn.from_torch(
                torch.zeros(slots * len(attn), cfg.n_kv, capacity, cfg.head_dim),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=mem,
                mesh_mapper=mapper,
            )

        kv = KvCache(k=zero_kv(), v=zero_kv(), num_layers=len(attn), capacity=capacity)

    col = lambda dim: ttnn.ShardTensor2dMesh(mesh, mesh_shape=(rows, cols), dims=(None, dim))  # noqa: E731
    gdn = {}
    for i in linear:
        for s in range(slots):
            gdn[(s, i)] = GdnState(
                conv=ttnn.from_torch(
                    torch.zeros(1, CONV_K - 1, cfg.conv_dim, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=col(-1),
                ),
                recurrent=ttnn.from_torch(
                    torch.zeros(cfg.lv_heads, cfg.lk_dim, cfg.lv_dim),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=col(0),
                ),
            )
    return Qwen38KvCaches(
        kv=kv, gdn=gdn, attn_layers=attn, linear_layers=linear, slots=slots, capacity=capacity, chunk=chunk
    )


def record_bytes(dtype, head_dim: int) -> int:
    return head_dim // 32 * TILE_BYTES[dtype]


def _config_name(config_id: int, n: int) -> str:
    """Zero-padded, so the protobuf round-trip (std::map, name order) keeps the config ids."""
    return f"{config_id:0{max(2, len(str(n - 1)))}d}"


def build_table(mesh, caches: Qwen38KvCaches, *, first_layer_idx: int, num_layers: int, max_seq: int):
    """The KV chunk address table (module docstring); table layer = global layer - first_layer_idx."""
    D = ttnn.experimental.disaggregation
    rows, cols = tuple(mesh.shape)
    kv = caches.kv
    assert kv is not None, "no attention layer served: nothing to publish"
    n_kv, head_dim = cols, kv.k.shape[-1]
    part = caches.chunk // rows  # tokens of one chunk on one SP row
    local = caches.capacity // rows  # cache rows per chip per (slot, layer)
    shards = local // KV_RECORD  # records per chip per (slot, layer)
    banks = mesh.dram_grid_size().x
    size = record_bytes(kv.k.dtype, head_dim)
    n_attn = len(caches.attn_layers)
    assert max_seq <= caches.capacity, f"max_seq {max_seq} above the cache's {caches.capacity}"

    specs = [(kv.k, h) for h in range(n_kv)] + [(kv.v, h) for h in range(n_kv)]
    configs = {}
    for c in range(len(specs)):
        cfg = D.KvChunkAddressTableConfig()
        cfg.num_layers = num_layers
        cfg.max_sequence_length = max_seq
        cfg.num_slots = caches.slots
        cfg.chunk_n_tokens = KV_RECORD
        cfg.chunk_size_bytes = size
        configs[_config_name(c, len(specs))] = cfg
    table = D.KvChunkAddressTable(configs)

    host = socket.gethostname()
    groups = {}  # (row, col) -> device group index
    for r in range(rows):
        for c in range(cols):
            fid = mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
            groups[(r, c)] = table.add_device_group([fid])
            table.set_fabric_node_host(fid, host_name=host)

    for config_id, (tensor, head) in enumerate(specs):
        base = tensor.buffer_address()  # identical on every chip (lockstep allocation)
        for slot in range(caches.slots):
            for a, layer in enumerate(caches.attn_layers):
                batch = slot * n_attn + a
                for pos in range(0, max_seq, KV_RECORD):
                    r = (pos // part) % rows
                    lr = (pos // caches.chunk) * part + pos % part
                    shard = batch * shards + lr // KV_RECORD  # ROUND_ROBIN_1D: shard j in bank j % banks
                    loc = D.KvCacheLocation()
                    loc.noc_addr = ((shard % banks) << 32) | (base + (shard // banks) * size)
                    loc.size_bytes = size
                    loc.device_group_index = groups[(r, head)]
                    table.set(layer - first_layer_idx, pos, slot, loc, config_id)
    return table
