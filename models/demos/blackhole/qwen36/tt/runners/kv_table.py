# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PUP prefill's engine-owned caches and their KV chunk address table (what tt-d-gen's KV manager migrates).

Attention (8 layers): MiniMax-M3's GQA layout (``models/demos/minimax_m3/tt/attention/kv_cache.py``): per chip
``[slots * 8, 1, max_seq_len, 256]`` bf8, DRAM ND-sharded, shard ``[1, 1, 32, 256]`` round robin over the banks, written
with ``ttnn.experimental.deepseek_prefill.update_padded_kv_cache``. Qwen3.5-9B has 4 KV heads on a (1, 4) column, so KV
head h is chip h, exactly M3's head -> TP-column rule. K is post k-norm + M-RoPE, published in the decode ring's
interleaved-pair RoPE order (:func:`rotate_half_to_interleaved`: device dim 2j = HF dim j, 2j + 1 = HF dim 32 + j over
the 64 rotary dims, tt-blaze ``blaze.weights.qwen35_9b.rope.head_perm``); V raw.

GDN (24 layers): ``gdn_state.GdnStates``, Kimi-K3's Snapshot contract with Qwen3.5's head counts.

Table configs (id order is the prefill <-> decode contract; names are the zero-padded positions, as blaze's decode
tables and M3 use):

    00..03  k head 0..3        Kv        chunk 32 tokens, 8704 B (8 bf8 tiles)   rows: attention layers 3, 7, ..., 31
    04..07  v head 0..3        Kv        same
    08      gdn_recurrent      Snapshot  chunk 32 (synthetic), 16384 B         rows: GDN layers 0-2, 4-6, ..., 28-30
    09      gdn_convolution    Snapshot  chunk 32 (synthetic), 384 B
    10      slot_params        Snapshot  chunk 32, 64 B: one per TP chip       rows: every layer (0..31)

Config 10 is the request's per-slot record (:func:`write_slot_params`): word 0 the int32 M-RoPE position offset decode
applies after the prompt, word 1 the fp32 presence penalty. One page per (layer, slot) on each chip, window 128 = one
32-position chunk per chip, so the KV manager lands chip c's copy on decode chip c of every layer's stage (tt-blaze
``blaze.models.qwen35_9b.slot_params``).

Every config is published on the model's layer axis (32 rows); a row published on one side only makes the KV manager
reject the migration, so decode must publish the same rows.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass

from models.demos.blackhole.qwen36.tt.runners.gdn_state import GdnGeometry, GdnStates

NUM_BANKS_BH = 8
CHUNK_TOKENS = 32
BF8_TILE_BYTES = 1088
KINDS = ("k", "v", "gdn_recurrent", "gdn_convolution", "slot_params")
SLOT_PAGE_BYTES = 64
SLOT_WORDS = SLOT_PAGE_BYTES // 4
SLOT_WINDOW = CHUNK_TOKENS * 4  # one chunk per TP chip
#: Rotary dims of a Qwen3.5-9B attention head (partial_rotary_factor 0.25 of 256).
K_ROPE_DIM = 64


def interleaved_k_perm(head_dim: int = 256, rot: int = K_ROPE_DIM):
    """``perm[ring dim] = HF dim`` within one head (the ring's ``head_perm``)."""
    import torch

    perm = torch.arange(head_dim)
    perm[0:rot:2] = torch.arange(rot // 2)
    perm[1:rot:2] = torch.arange(rot // 2, rot)
    return perm


def rotate_half_to_interleaved_host(k, rot: int = K_ROPE_DIM):
    """Host mirror of :func:`rotate_half_to_interleaved` on ``[..., head_dim]``."""
    import torch

    lead = k.shape[:-1]
    r = k[..., :rot].reshape(*lead, 2, rot // 2).transpose(-2, -1).reshape(*lead, rot)
    return torch.cat([r, k[..., rot:]], dim=-1)


def rotate_half_to_interleaved(x, rot: int = K_ROPE_DIM):
    """Device: K ``[1, 1, n, head_dim]`` (any dtype) -> bf8 TILE with the rotary dims in the ring's interleaved-pair
    order. Done in bf16 before the bf8 cast: bf8 shares one exponent per 16-element row segment, so the bf8 bytes of
    the permuted K are not a permutation of the bf8 bytes."""
    import ttnn

    n, hd = int(x.shape[-2]), int(x.shape[-1])
    y = ttnn.to_layout(ttnn.typecast(x, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    r = ttnn.slice(y, [0, 0, 0, 0], [1, 1, n, rot])
    r = ttnn.reshape(r, [1, n, 2, rot // 2])
    r = ttnn.permute(r, (0, 1, 3, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    r = ttnn.reshape(r, [1, 1, n, rot])
    rest = ttnn.slice(y, [0, 0, 0, rot], [1, 1, n, hd])
    out = ttnn.concat([r, rest], dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.to_layout(out, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.typecast(out, ttnn.bfloat8_b)


def kv_chunk_bytes(head_dim: int) -> int:
    return (head_dim // 32) * BF8_TILE_BYTES


def config_specs(num_kv_heads: int):
    """[(name, kind, head or None, snapshot)] in config-id order."""
    specs = [(f"{i:02d}", "k", h, False) for i, h in enumerate(range(num_kv_heads))]
    specs += [(f"{num_kv_heads + h:02d}", "v", h, False) for h in range(num_kv_heads)]
    n = 2 * num_kv_heads
    specs += [(f"{n:02d}", "gdn_recurrent", None, True), (f"{n + 1:02d}", "gdn_convolution", None, True)]
    specs += [(f"{n + 2:02d}", "slot_params", None, True)]
    return specs


def cache_kind(config_id: int, num_kv_heads: int = 4) -> str:
    specs = config_specs(num_kv_heads)
    if not 0 <= config_id < len(specs):
        return "other"
    kind = specs[config_id][1]
    return "kv" if kind in ("k", "v") else kind


@dataclass
class PupKvCaches:
    """``KvCaches`` of the PUP prefill (subclassed from the runner's ABC at import time when it is available)."""

    k: object
    v: object
    gdn: GdnStates
    num_users: int
    max_seq_len: int
    head_dim: int
    attn_layer_ids: tuple
    gdn_layer_ids: tuple
    slot_params: object = None  # [num_layers * num_users, 16] uint32 ROW_MAJOR interleaved DRAM (config 10)

    @property
    def num_attn_layers(self) -> int:
        return len(self.attn_layer_ids)

    def deallocate(self):
        import ttnn

        ttnn.deallocate(self.k)
        ttnn.deallocate(self.v)
        self.gdn.deallocate()
        if self.slot_params is not None:
            ttnn.deallocate(self.slot_params)

    @property
    def num_layers(self) -> int:
        return len(self.attn_layer_ids) + len(self.gdn_layer_ids)


try:  # register as the runner's opaque handle type without importing it at module load in a host-only test
    from models.demos.common.prefill.adapter import KvCaches as _KvCaches

    _KvCaches.register(PupKvCaches)
except Exception:  # noqa: BLE001
    pass


def allocate(mesh_device, *, attn_layer_ids, gdn_layer_ids, geometry: GdnGeometry, num_users, max_seq_len, head_dim):
    import ttnn

    from models.demos.common.prefill.runners.kv_bank_alignment import check_kv_bank_aligned

    banks = mesh_device.dram_grid_size().x
    check_kv_bank_aligned(max_seq_len, max_seq_len, banks)
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(banks)})
    mc = ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=[1, 1, CHUNK_TOKENS, head_dim],
            grid=grid,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )
    n = num_users * len(attn_layer_ids)

    def _alloc():
        t = ttnn.allocate_tensor_on_device(
            ttnn.Shape([n, 1, max_seq_len, head_dim]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, mesh_device, mc
        )
        try:
            from models.demos.deepseek_v3_b1.micro_ops.dram_zero_fill.op import DRAMZeroFill

            DRAMZeroFill.op(t)
        except Exception:  # noqa: BLE001  (zeros are cosmetic: migration reads only written positions)
            pass
        return t

    k, v = _alloc(), _alloc()
    gdn = GdnStates.allocate(mesh_device, geometry, layer_ids=tuple(gdn_layer_ids), num_slots=num_users)
    import torch

    slot_params = ttnn.from_torch(
        torch.zeros((len(attn_layer_ids) + len(gdn_layer_ids)) * num_users, SLOT_WORDS, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return PupKvCaches(
        k=k,
        v=v,
        gdn=gdn,
        num_users=num_users,
        max_seq_len=max_seq_len,
        head_dim=head_dim,
        attn_layer_ids=tuple(attn_layer_ids),
        gdn_layer_ids=tuple(gdn_layer_ids),
        slot_params=slot_params,
    )


def slot_record(rope_delta: int, presence_penalty: float):
    """The 16-word record of one request (config 10)."""
    import torch

    r = torch.zeros(SLOT_WORDS, dtype=torch.int32)
    r[0] = int(rope_delta)
    r[1] = torch.tensor([float(presence_penalty)], dtype=torch.float32).view(torch.int32)[0]
    return r


def write_slot_params(mesh_device, caches: "PupKvCaches", slot: int, rope_delta: int, presence_penalty: float, host):
    """Write ``slot``'s record into every layer's page (host keeps the whole table: [num_layers * num_users, 16] int32,
    updated in place); one small host -> device copy. Call before the request's first layer completion."""
    import ttnn

    rec = slot_record(rope_delta, presence_penalty)
    for layer in range(caches.num_layers):
        host[layer * caches.num_users + slot] = rec
    h = ttnn.from_torch(
        host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    ttnn.copy_host_to_device_tensor(h, caches.slot_params)
    return h


def mrope_delta(grids, merge: int = 2) -> int:
    """Rope position minus KV position for text after the images (HF Qwen-VL ``get_rope_index``, single turn)."""
    delta = 0
    for g in grids if grids is not None else ():
        t, h, w = (int(v) for v in g)
        delta += max(t, h // merge, w // merge) - t * (h // merge) * (w // merge)
    return delta


def migration_stages(caches: PupKvCaches):
    """One ``KvCacheStage`` per table kind (k, v, gdn_recurrent, gdn_convolution), each in its own layer space."""
    from models.demos.common.prefill.runners.migration import KvCacheStage

    return [
        KvCacheStage(int(caches.k.buffer_address()), 0, caches.num_attn_layers),
        KvCacheStage(int(caches.v.buffer_address()), 0, caches.num_attn_layers),
        KvCacheStage(int(caches.gdn.recurrent.buffer_address()), 0, caches.gdn.num_layers),
        KvCacheStage(int(caches.gdn.convolution.buffer_address()), 0, caches.gdn.num_layers),
        KvCacheStage(int(caches.slot_params.buffer_address()), 0, caches.num_layers),
    ]


def local_layouts(mesh_device, caches: PupKvCaches, num_banks: int = NUM_BANKS_BH):
    """Single-rank stage layouts (the shape the runner's all-gather returns), one per kind."""
    import ttnn

    rows, cols = tuple(mesh_device.shape)
    fnids = [[mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(r, c)) for c in range(cols)] for r in range(rows)]
    host = socket.gethostname()
    out = []
    for stage in migration_stages(caches):
        out.append(
            [
                {
                    "first_layer": stage.first_layer,
                    "count": stage.count,
                    "base_addr": stage.base_addr,
                    "num_banks": num_banks,
                    "host_name": host,
                    "fnids": fnids,
                }
            ]
        )
    return out


def build_table(
    *,
    stage_layouts,
    num_users: int,
    max_seq_len: int,
    head_dim: int,
    num_kv_heads: int,
    attn_layer_ids,
    gdn_layer_ids,
    num_model_layers: int,
    geometry: GdnGeometry,
    disagg=None,
):
    """The PUP KV chunk address table. ``stage_layouts``: one gathered layout per kind (``local_layouts`` single-rank).
    ``disagg`` defaults to ``ttnn.experimental.disaggregation`` (tests pass a recorder)."""
    if disagg is None:
        import ttnn

        disagg = ttnn.experimental.disaggregation
    assert len(stage_layouts) == 5, "one layout per kind: k, v, gdn_recurrent, gdn_convolution, slot_params"
    assert max_seq_len % CHUNK_TOKENS == 0
    specs = config_specs(num_kv_heads)

    def _cfg(kind):
        cfg = disagg.KvChunkAddressTableConfig()
        cfg.num_layers = num_model_layers
        cfg.num_slots = num_users
        if kind in ("k", "v"):
            cfg.max_sequence_length = max_seq_len
            cfg.chunk_n_tokens = CHUNK_TOKENS
            cfg.chunk_size_bytes = kv_chunk_bytes(head_dim)
        elif kind == "slot_params":
            cfg.max_sequence_length = SLOT_WINDOW
            cfg.chunk_n_tokens = CHUNK_TOKENS
            cfg.chunk_size_bytes = SLOT_PAGE_BYTES
            cfg.tensor_type = disagg.KvChunkTensorType.Snapshot
        else:
            cfg.max_sequence_length = geometry.window
            cfg.chunk_n_tokens = geometry.chunk_n_tokens(kind)
            cfg.chunk_size_bytes = geometry.segment_bytes(kind)
            cfg.tensor_type = disagg.KvChunkTensorType.Snapshot
        return cfg

    table = disagg.KvChunkAddressTable({name: _cfg(kind) for name, kind, _, _ in specs})
    hosts = set()

    def _group(stage, members):
        idx = table.add_device_group(members)
        host = stage.get("host_name") or f"host-{stage['host_tag']:08x}"
        for fid in members:
            key = (int(fid.mesh_id), int(fid.chip_id)) if hasattr(fid, "chip_id") else id(fid)
            if key not in hosts:
                table.set_fabric_node_host(fid, host_name=host)
                hosts.add(key)
        return idx

    kv_chunk = kv_chunk_bytes(head_dim)
    chunks_per_seq = max_seq_len // CHUNK_TOKENS
    kind_index = {"k": 0, "v": 1, "gdn_recurrent": 2, "gdn_convolution": 3, "slot_params": 4}
    for config_id, (_, kind, head, _) in enumerate(specs):
        for stage in stage_layouts[kind_index[kind]]:
            count = int(stage["count"])
            if count == 0:
                continue
            base, nb, first, fnids = int(stage["base_addr"]), int(stage["num_banks"]), int(stage["first_layer"]), stage["fnids"]
            sp = len(fnids)
            if kind == "slot_params":
                # page (layer, slot) on every chip; chunk c of the 128-position window = chip c's copy
                for tp_col in range(len(fnids[0])):
                    g = _group(stage, [fnids[r][tp_col] for r in range(sp)])
                    for slot in range(num_users):
                        for layer in range(count):
                            page = layer * num_users + slot
                            loc = disagg.KvCacheLocation()
                            loc.noc_addr = ((page % nb) << 32) | (base + (page // nb) * SLOT_PAGE_BYTES)
                            loc.size_bytes = SLOT_PAGE_BYTES
                            loc.device_group_index = g
                            table.set(first + layer, tp_col * CHUNK_TOKENS, slot, loc, config_id)
                continue
            if kind in ("k", "v"):
                assert sp == 1, "PUP prefill runs on one (1, TP) column: no sequence parallelism"
                g = _group(stage, [fnids[0][head]])
                for slot in range(num_users):
                    for local in range(count):
                        layer = attn_layer_ids[first + local]
                        batch = slot * count + local
                        for c in range(chunks_per_seq):
                            s = batch * chunks_per_seq + c
                            loc = disagg.KvCacheLocation()
                            loc.noc_addr = ((s % nb) << 32) | (base + (s // nb) * kv_chunk)
                            loc.size_bytes = kv_chunk
                            loc.device_group_index = g
                            table.set(layer, c * CHUNK_TOKENS, slot, loc, config_id)
            else:
                seg_bytes = geometry.segment_bytes(kind)
                spl = geometry.shards_per_layer(kind)
                stride = geometry.chunk_n_tokens(kind)
                for tp_col in range(geometry.tensor_parallel_size):
                    g = _group(stage, [fnids[r][tp_col] for r in range(sp)])
                    segs = geometry.local_segments(kind, tp_col)
                    for slot in range(num_users):
                        for local in range(count):
                            layer = gdn_layer_ids[first + local]
                            batch = slot * count + local
                            for seg, shard_in_layer in segs:
                                s = batch * spl + shard_in_layer
                                loc = disagg.KvCacheLocation()
                                loc.noc_addr = ((s % nb) << 32) | (base + (s // nb) * seg_bytes)
                                loc.size_bytes = seg_bytes
                                loc.device_group_index = g
                                table.set(layer, seg * stride, slot, loc, config_id)
    return table


def layer_position_range(layer_idx: int, real_len: int, gdn_layer_ids, geometry: GdnGeometry):
    """Positions one /migrate of ``layer_idx`` covers: the tokens for attention, the whole Snapshot window for GDN
    (config 10, the slot record, is a Snapshot of every layer: the KV manager moves it whole)."""
    if layer_idx in gdn_layer_ids:
        return 0, geometry.window
    return 0, real_len
