# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""KV chunk address table for Gemma 4 prefill (packed dual-family cache).

Config layout (id order is the prefill↔decode contract; workers route by integer
``config_id``). Do not reorder — that silently cross-wires K / V / merged rows::

    config  0 .. 15   ->  k_h0  .. k_h15     local/SWA K, 16 heads, head_dim 256
    config 16 .. 31   ->  v_h0  .. v_h15     local/SWA V, same geometry
    config 32 .. 35   ->  kv_h0 .. kv_h3     global merged [K_roped_rotary | V], row 640

That matches ``tt-blaze/blaze/models/gemma4/kv_migration.py``. Semantic names live
on the specs; protobuf import rebuilds configs via ``std::map`` (lexicographic
name order), so the table is built with zero-padded names (``"00"``..``"35"``).

Two packed NdShard tensors (see ``tt/attention/prefill_kv_cache.py``)
---------------------------------------------------------------------------
Local K/V: ``[num_users * num_layers, nkv_per_dev, sliding_window // sp, 256]``.
Unused slots on full-attention layers are cheap (the ring is 1024 tokens).
Batch fold is user-major: ``slot * num_layers + layer``.

Global merged: ``[num_users * n_global_layers, 1, seq_len // sp, 640]``.
Compact — do not pad unused sliding layers out to full context.
Batch fold: ``slot * n_global_layers + global_index(layer)``.

NdShard ROUND_ROBIN_1D (``buffer_distribution_spec.cpp`` ``iterate_over_shards``)
walks dims 0→1→2: batch, then local head, then 32-token seq blocks. Closed form::

    n_seq_blocks = seq_local // 32
    shard_id     = batch_idx * (n_heads * n_seq_blocks) + local_head * n_seq_blocks
                   + (local_pos // 32)
    bank         = shard_id % num_dram_banks
    offset       = (shard_id // num_dram_banks) * chunk_bytes

A sliding window is an **extent**, not a folded address: local configs are
authored only at positions ``0 .. sliding_window-1`` on sliding layers. Do not
put ``p % sw`` into ``locate``. Local configs are absent on full-attention
layers; global configs are absent on sliding layers.

Prefill mesh default that matches decode's flattened 1×4 TP view:
``mesh_shape=(1, 4)``, ``sp_axis=0`` → device coord ``(0, chip)``.
Local head ``h`` lives on chip ``h // 4`` at ``local_head = h % 4``.
Global head ``h`` lives on chip ``h``.
"""

from __future__ import annotations

from dataclasses import dataclass

# Must match the DRAM NdShard in allocate_kv_cache.
NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32
TILE = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK

SLIDING_ATTENTION = "sliding_attention"
FULL_ATTENTION = "full_attention"

# 31B: 5 sliding + 1 full, repeating.
_DEFAULT_LAYER_PERIOD = (SLIDING_ATTENTION,) * 5 + (FULL_ATTENTION,)

# bf8_b / bf16 TILE byte sizes (32x32). bf8_b = 1024 mantissa + 64 exponent.
_TILE_BYTES = {"bfloat8_b": 1088, "bfloat16": 2048}

# Production 31B attention shape (TP=4).
DEFAULT_LOCAL_N_KV = 16
DEFAULT_GLOBAL_N_KV = 4
DEFAULT_LOCAL_HEAD_DIM = 256
DEFAULT_GLOBAL_HEAD_DIM = 512
DEFAULT_GLOBAL_ROTARY_FACTOR = 0.25
DEFAULT_SLIDING_WINDOW = 1024
DEFAULT_NUM_DRAM_BANKS = 8


def _dtype_key(dtype) -> str:
    if isinstance(dtype, str):
        return dtype
    name = getattr(dtype, "name", None)
    if name in _TILE_BYTES:
        return name
    text = str(dtype)
    for key in _TILE_BYTES:
        if key in text:
            return key
    raise AssertionError(f"unsupported KV cache dtype {dtype!r}; expected bfloat8_b or bfloat16")


def tile_bytes(dtype) -> int:
    return _TILE_BYTES[_dtype_key(dtype)]


def chunk_size_bytes(dtype, row_dim: int, tokens_per_chunk: int = TILE) -> int:
    """Bytes of one ``[1, 1, tokens_per_chunk, row_dim]`` NdShard chunk."""
    if row_dim % TILE or tokens_per_chunk % TILE:
        raise ValueError(
            f"row_dim ({row_dim}) and tokens_per_chunk ({tokens_per_chunk}) must be multiples of {TILE}"
        )
    return (tokens_per_chunk // TILE) * (row_dim // TILE) * tile_bytes(dtype)


def global_row_dim(global_head_dim: int = DEFAULT_GLOBAL_HEAD_DIM, rotary_factor: float = DEFAULT_GLOBAL_ROTARY_FACTOR) -> int:
    """Merged global row: rotary(K) || V. 512 * 0.25 + 512 = 640 at 31B."""
    rotary = int(global_head_dim * rotary_factor)
    if rotary % TILE:
        raise ValueError(
            f"global rotary_dim ({rotary}) must be tile-aligned "
            f"(head_dim={global_head_dim}, factor={rotary_factor})."
        )
    return rotary + global_head_dim


def config_names(local_n_kv_heads: int = DEFAULT_LOCAL_N_KV, global_n_kv_heads: int = DEFAULT_GLOBAL_N_KV) -> tuple[str, ...]:
    """Ordered semantic names. THE ORDER IS THE PREFILL↔DECODE CONTRACT."""
    return (
        tuple(f"k_h{h}" for h in range(local_n_kv_heads))
        + tuple(f"v_h{h}" for h in range(local_n_kv_heads))
        + tuple(f"kv_h{h}" for h in range(global_n_kv_heads))
    )


def stable_config_name(config_id: int, num_configs: int) -> str:
    """Zero-padded decimal name so std::map lexicographic order matches numeric config_id."""
    width = max(2, len(str(max(num_configs - 1, 0))))
    return f"{config_id:0{width}d}"


def default_layer_types(num_layers: int) -> tuple[str, ...]:
    """31B pattern: 5× sliding_attention + 1× full_attention, repeating."""
    period = _DEFAULT_LAYER_PERIOD
    return tuple(period[i % len(period)] for i in range(num_layers))


def layer_types_from_hf(hf_config) -> tuple[str, ...]:
    """Read ``layer_types`` off an HF config (or its ``text_config`` wrapper)."""
    text = getattr(hf_config, "text_config", hf_config)
    types = getattr(text, "layer_types", None)
    if types is None:
        return default_layer_types(int(text.num_hidden_layers))
    return tuple(types)


def is_sliding_layer(layer_types: tuple[str, ...], layer: int) -> bool:
    return layer_types[layer] == SLIDING_ATTENTION


def num_global_layers(layer_types: tuple[str, ...]) -> int:
    return sum(1 for t in layer_types if t == FULL_ATTENTION)


def global_layer_index(layer_types: tuple[str, ...], layer: int) -> int:
    """Dense index of ``layer`` among full-attention layers. Raises if sliding."""
    if is_sliding_layer(layer_types, layer):
        raise ValueError(f"layer {layer} is sliding_attention; it has no global-cache slot")
    return sum(1 for t in layer_types[:layer] if t == FULL_ATTENTION)


def nkv_per_device(n_kv_heads: int, tp: int) -> int:
    return 1 if n_kv_heads < tp else n_kv_heads // tp


def local_batch_idx(slot: int, layer: int, num_layers: int) -> int:
    return slot * num_layers + layer


def global_batch_idx(slot: int, layer: int, layer_types: tuple[str, ...]) -> int:
    return slot * num_global_layers(layer_types) + global_layer_index(layer_types, layer)


def block_cyclic_local_pos(position: int, *, chunk_size: int, sp: int) -> tuple[int, int]:
    """Map a global token position to ``(local_pos, sp_row)`` (GPT-OSS block-cyclic).

    ``sp=1`` is the identity: ``(position, 0)``. Do not fold ``position % sliding_window``.
    """
    if chunk_size % sp:
        raise ValueError(f"chunk_size ({chunk_size}) must be divisible by sp ({sp})")
    tokens_per_chunk_local = chunk_size // sp
    seq_chunk = position // chunk_size
    offset_in_chunk = position % chunk_size
    sp_row = offset_in_chunk // tokens_per_chunk_local
    local_in_chunk = offset_in_chunk % tokens_per_chunk_local
    return seq_chunk * tokens_per_chunk_local + local_in_chunk, sp_row


def shard_id(*, batch_idx: int, local_head: int, local_pos: int, n_heads: int, seq_local: int, tile: int = TILE) -> int:
    """NdShard emit index: batch, then head, then 32-token seq blocks."""
    if seq_local % tile:
        raise ValueError(f"seq_local ({seq_local}) must be a multiple of {tile}")
    n_seq_blocks = seq_local // tile
    return batch_idx * (n_heads * n_seq_blocks) + local_head * n_seq_blocks + (local_pos // tile)


def chunk_noc_addr(*, shard: int, base_addr: int, chunk_bytes: int, num_banks: int) -> tuple[int, int, int]:
    """``(encoded_noc_addr, bank_id, per_bank_offset)`` for ROUND_ROBIN_1D."""
    bank = shard % num_banks
    offset = (shard // num_banks) * chunk_bytes
    return ((bank & 0xFFFFFFFF) << 32) | ((base_addr + offset) & 0xFFFFFFFF), bank, offset


def tp_chip_coord(sp_row: int, chip: int, *, sp_axis: int) -> tuple[int, int]:
    """Mesh coordinate of the chip that owns ``(sp_row, TP chip)``."""
    coord = [0, 0]
    coord[sp_axis] = sp_row
    coord[1 - sp_axis] = chip
    return (coord[0], coord[1])


@dataclass(frozen=True)
class PrefillConfigSpec:
    """One integer config_id. ``label`` is the semantic name (``k_h0``); protobuf uses the padded id."""

    label: str
    family: str  # "local_k" | "local_v" | "global_kv"
    head: int
    row_dim: int
    chunk_size_bytes: int
    seq_extent: int


@dataclass(frozen=True)
class Gemma4PrefillGeom:
    """Device-free geometry for address math and table authorship."""

    num_layers: int
    num_users: int
    seq_len: int
    layer_types: tuple[str, ...]
    mesh_shape: tuple[int, int] = (1, 4)
    sp_axis: int = 0
    chunk_size: int = 0  # 0 => seq_len (one period; sp=1 identity)
    sliding_window: int = DEFAULT_SLIDING_WINDOW
    local_n_kv_heads: int = DEFAULT_LOCAL_N_KV
    global_n_kv_heads: int = DEFAULT_GLOBAL_N_KV
    local_head_dim: int = DEFAULT_LOCAL_HEAD_DIM
    global_head_dim: int = DEFAULT_GLOBAL_HEAD_DIM
    global_rotary_factor: float = DEFAULT_GLOBAL_ROTARY_FACTOR
    num_dram_banks: int = DEFAULT_NUM_DRAM_BANKS
    local_dtype: str = "bfloat8_b"
    global_dtype: str = "bfloat8_b"

    def __post_init__(self):
        if len(self.layer_types) != self.num_layers:
            raise ValueError(f"layer_types length {len(self.layer_types)} != num_layers {self.num_layers}")
        object.__setattr__(self, "chunk_size", self.chunk_size or self.seq_len)

    @property
    def sp(self) -> int:
        return self.mesh_shape[self.sp_axis]

    @property
    def tp(self) -> int:
        return self.mesh_shape[1 - self.sp_axis]

    @property
    def local_nkv_per_dev(self) -> int:
        return nkv_per_device(self.local_n_kv_heads, self.tp)

    @property
    def global_nkv_per_dev(self) -> int:
        return nkv_per_device(self.global_n_kv_heads, self.tp)

    @property
    def local_seq_local(self) -> int:
        return self.sliding_window // self.sp

    @property
    def global_seq_local(self) -> int:
        return self.seq_len // self.sp

    @property
    def global_row_dim(self) -> int:
        return global_row_dim(self.global_head_dim, self.global_rotary_factor)

    @property
    def n_global_layers(self) -> int:
        return num_global_layers(self.layer_types)


def config_specs(geom: Gemma4PrefillGeom) -> tuple[PrefillConfigSpec, ...]:
    """36 specs in decode-contract order."""
    local_csb = chunk_size_bytes(geom.local_dtype, geom.local_head_dim)
    global_csb = chunk_size_bytes(geom.global_dtype, geom.global_row_dim)
    specs: list[PrefillConfigSpec] = []
    for h in range(geom.local_n_kv_heads):
        specs.append(
            PrefillConfigSpec("k_h" + str(h), "local_k", h, geom.local_head_dim, local_csb, geom.sliding_window)
        )
    for h in range(geom.local_n_kv_heads):
        specs.append(
            PrefillConfigSpec("v_h" + str(h), "local_v", h, geom.local_head_dim, local_csb, geom.sliding_window)
        )
    for h in range(geom.global_n_kv_heads):
        specs.append(
            PrefillConfigSpec("kv_h" + str(h), "global_kv", h, geom.global_row_dim, global_csb, geom.seq_len)
        )
    names = config_names(geom.local_n_kv_heads, geom.global_n_kv_heads)
    got = tuple(s.label for s in specs)
    if got != names:
        raise AssertionError(f"config spec labels {got} != contract {names}")
    return tuple(specs)


def layer_owns_config(geom: Gemma4PrefillGeom, spec: PrefillConfigSpec, layer: int) -> bool:
    sliding = is_sliding_layer(geom.layer_types, layer)
    return sliding if spec.family.startswith("local") else not sliding


def locate(
    geom: Gemma4PrefillGeom,
    spec: PrefillConfigSpec,
    *,
    position: int,
    slot: int,
    layer: int,
    base_addr: int = 0,
) -> tuple[int, int, int, tuple[int, int]]:
    """``(noc_addr, bank_id, per_bank_offset, device_coord)``.

    Raises if this config has no cache on ``layer`` (same rule as decode ``locate``).
    Addressing is linear in ``position`` — the window is an extent, not a wrap.
    """
    if not layer_owns_config(geom, spec, layer):
        kind = "local" if spec.family.startswith("local") else "global"
        other = "global" if kind == "local" else "local"
        raise ValueError(f"{kind} config {spec.label} has no cache on {other} layer {layer}")
    if not (0 <= position < spec.seq_extent):
        raise ValueError(f"position {position} is outside {spec.label} extent [0, {spec.seq_extent})")

    local_pos, sp_row = block_cyclic_local_pos(position, chunk_size=geom.chunk_size, sp=geom.sp)
    if spec.family.startswith("local"):
        n_heads = geom.local_nkv_per_dev
        seq_local = geom.local_seq_local
        batch = local_batch_idx(slot, layer, geom.num_layers)
        nkv_per_dev = geom.local_nkv_per_dev
    else:
        n_heads = geom.global_nkv_per_dev
        seq_local = geom.global_seq_local
        batch = global_batch_idx(slot, layer, geom.layer_types)
        nkv_per_dev = geom.global_nkv_per_dev

    chip = spec.head // nkv_per_dev
    local_head = spec.head % nkv_per_dev
    sid = shard_id(
        batch_idx=batch,
        local_head=local_head,
        local_pos=local_pos,
        n_heads=n_heads,
        seq_local=seq_local,
    )
    noc, bank, offset = chunk_noc_addr(
        shard=sid, base_addr=base_addr, chunk_bytes=spec.chunk_size_bytes, num_banks=geom.num_dram_banks
    )
    return noc, bank, offset, tp_chip_coord(sp_row, chip, sp_axis=geom.sp_axis)


def _validate_geom(geom: Gemma4PrefillGeom) -> None:
    sp = geom.sp
    if geom.seq_len % geom.chunk_size:
        raise ValueError(f"seq_len {geom.seq_len} must be a multiple of chunk_size {geom.chunk_size}")
    tokens_per_chunk_local = geom.chunk_size // sp
    if tokens_per_chunk_local % TILE:
        raise ValueError(
            f"chunk_size {geom.chunk_size} / sp {sp} = {tokens_per_chunk_local}, not a multiple of {TILE}"
        )
    if geom.sliding_window % TILE:
        raise ValueError(f"sliding_window ({geom.sliding_window}) must be a multiple of {TILE}")
    if geom.sliding_window % sp:
        raise ValueError(f"sliding_window ({geom.sliding_window}) must be divisible by sp ({sp})")
    if geom.seq_len % (TILE * sp):
        raise ValueError(f"seq_len ({geom.seq_len}) must be a multiple of TILE*sp ({TILE * sp})")
    if geom.local_n_kv_heads % geom.tp and geom.local_n_kv_heads >= geom.tp:
        raise ValueError(f"local_n_kv_heads ({geom.local_n_kv_heads}) must divide TP ({geom.tp})")
    if geom.global_n_kv_heads % geom.tp and geom.global_n_kv_heads >= geom.tp:
        raise ValueError(f"global_n_kv_heads ({geom.global_n_kv_heads}) must divide TP ({geom.tp})")


def _tensor_for_family(kv_cache, family: str):
    if family == "local_k":
        return kv_cache.local_k
    if family == "local_v":
        return kv_cache.local_v
    if family == "global_kv":
        return kv_cache.global_kv
    raise ValueError(f"unknown family {family!r}")


def build_kv_chunk_address_table(
    *,
    mesh_device,
    kv_cache,
    seq_len,
    num_layers,
    mesh_shape,
    sp_axis,
    num_users,
    chunk_size,
    layer_types=None,
    sliding_window: int = DEFAULT_SLIDING_WINDOW,
    local_n_kv_heads: int = DEFAULT_LOCAL_N_KV,
    global_n_kv_heads: int = DEFAULT_GLOBAL_N_KV,
    local_head_dim: int = DEFAULT_LOCAL_HEAD_DIM,
    global_head_dim: int = DEFAULT_GLOBAL_HEAD_DIM,
    global_rotary_factor: float = DEFAULT_GLOBAL_ROTARY_FACTOR,
):
    """Build the Gemma 4 multi-config block-cyclic KV chunk address table (does not serialize).

    ``kv_cache`` is a :class:`Gemma4PrefillKVCache` (``.local_k`` / ``.local_v`` / ``.global_kv``).
    ``chunk_size`` is the block-cyclic period (tokens per ``prefill_chunk``).
    ``layer_types`` defaults to the 31B 5+1 pattern of length ``num_layers``.
    """
    import socket

    import ttnn
    from loguru import logger

    from models.demos.common.prefill.runners.migration import get_num_dram_banks

    types = tuple(layer_types) if layer_types is not None else default_layer_types(num_layers)
    geom = Gemma4PrefillGeom(
        num_layers=num_layers,
        num_users=num_users,
        seq_len=seq_len,
        layer_types=types,
        mesh_shape=tuple(mesh_shape),
        sp_axis=sp_axis,
        chunk_size=chunk_size,
        sliding_window=sliding_window,
        local_n_kv_heads=local_n_kv_heads,
        global_n_kv_heads=global_n_kv_heads,
        local_head_dim=local_head_dim,
        global_head_dim=global_head_dim,
        global_rotary_factor=global_rotary_factor,
        num_dram_banks=get_num_dram_banks(mesh_device),
        local_dtype=kv_cache.local_k.dtype,
        global_dtype=kv_cache.global_kv.dtype,
    )
    _validate_geom(geom)
    specs = config_specs(geom)

    local_seq = geom.local_seq_local
    global_seq = geom.global_seq_local
    n_global = geom.n_global_layers
    for name, t, batch, heads, seq, dim in (
        ("local_k", kv_cache.local_k, num_users * num_layers, geom.local_nkv_per_dev, local_seq, geom.local_head_dim),
        ("local_v", kv_cache.local_v, num_users * num_layers, geom.local_nkv_per_dev, local_seq, geom.local_head_dim),
        ("global_kv", kv_cache.global_kv, num_users * n_global, geom.global_nkv_per_dev, global_seq, geom.global_row_dim),
    ):
        assert t.shape[0] == batch, f"{name} batch dim {t.shape[0]} != {batch}"
        assert t.shape[1] == heads, f"{name} head dim {t.shape[1]} != {heads}"
        assert t.shape[2] == seq, f"{name} seq dim {t.shape[2]} != {seq}"
        assert t.shape[3] == dim, f"{name} row dim {t.shape[3]} != {dim}"

    num_configs = len(specs)
    configs_by_name = {}
    for i, spec in enumerate(specs):
        cfg = ttnn.experimental.disaggregation.KvChunkAddressTableConfig()
        cfg.num_layers = num_layers
        cfg.max_sequence_length = spec.seq_extent
        cfg.num_slots = num_users
        cfg.chunk_n_tokens = TILE
        cfg.chunk_size_bytes = spec.chunk_size_bytes
        configs_by_name[stable_config_name(i, num_configs)] = cfg

    table = ttnn.experimental.disaggregation.KvChunkAddressTable(configs_by_name)
    assert table.num_configs() == num_configs
    for i in range(num_configs):
        want = stable_config_name(i, num_configs)
        assert table.config_name(i) == want, (
            f"config_id {i} name {table.config_name(i)!r} != {want!r} (protobuf-safe naming broken)"
        )

    host_name = socket.gethostname()
    hosts_set = set()
    groups: dict[tuple[int, int], int] = {}

    def _device_group(coord: tuple[int, int]) -> int:
        if coord in groups:
            return groups[coord]
        fabric_node_ids = [mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))]
        group_idx = table.add_device_group(fabric_node_ids)
        for fid in fabric_node_ids:
            key = (int(fid.mesh_id), int(fid.chip_id))
            if key not in hosts_set:
                table.set_fabric_node_host(fid, host_name=host_name)
                hosts_set.add(key)
        groups[coord] = group_idx
        return group_idx

    for config_id, spec in enumerate(specs):
        tensor = _tensor_for_family(kv_cache, spec.family)
        base_addr = tensor.buffer_address()
        for slot in range(num_users):
            for layer in range(num_layers):
                if not layer_owns_config(geom, spec, layer):
                    continue
                for position in range(0, spec.seq_extent, TILE):
                    noc, _bank, _off, coord = locate(
                        geom, spec, position=position, slot=slot, layer=layer, base_addr=base_addr
                    )
                    location = ttnn.experimental.disaggregation.KvCacheLocation()
                    location.noc_addr = noc
                    location.size_bytes = spec.chunk_size_bytes
                    location.device_group_index = _device_group(coord)
                    table.set(layer, position, slot, location, config_id)

    logger.info(
        f"[gemma4-kv-table] multi-config table built "
        f"(configs={num_configs} [{', '.join(s.label for s in specs)}], "
        f"entries={table.total_entries()}, banks={geom.num_dram_banks}, "
        f"local_csb={specs[0].chunk_size_bytes}, global_csb={specs[-1].chunk_size_bytes})"
    )
    return table


def build_and_serialize_kv_chunk_table(
    *,
    mesh_device,
    kv_cache,
    seq_len,
    num_layers,
    mesh_shape,
    sp_axis,
    num_users,
    chunk_size,
    path,
    layer_types=None,
    sliding_window: int = DEFAULT_SLIDING_WINDOW,
    local_n_kv_heads: int = DEFAULT_LOCAL_N_KV,
    global_n_kv_heads: int = DEFAULT_GLOBAL_N_KV,
    local_head_dim: int = DEFAULT_LOCAL_HEAD_DIM,
    global_head_dim: int = DEFAULT_GLOBAL_HEAD_DIM,
    global_rotary_factor: float = DEFAULT_GLOBAL_ROTARY_FACTOR,
) -> str:
    """Build the Gemma 4 multi-config table and serialize it to ``path`` for SET_TABLE."""
    import ttnn
    from loguru import logger

    table = build_kv_chunk_address_table(
        mesh_device=mesh_device,
        kv_cache=kv_cache,
        seq_len=seq_len,
        num_layers=num_layers,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_users=num_users,
        chunk_size=chunk_size,
        layer_types=layer_types,
        sliding_window=sliding_window,
        local_n_kv_heads=local_n_kv_heads,
        global_n_kv_heads=global_n_kv_heads,
        local_head_dim=local_head_dim,
        global_head_dim=global_head_dim,
        global_rotary_factor=global_rotary_factor,
    )
    ttnn.experimental.disaggregation.export_to_protobuf_file(table, path)
    logger.info(
        f"[migration] Gemma 4 KV chunk address table serialized to {path} "
        f"(configs={table.num_configs()}, entries={table.total_entries()})"
    )
    return path
