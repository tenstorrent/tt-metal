# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""gemma4_31b_glx through tt-metal's prefill runner (models/demos/common/prefill): the adapter, the served KV cache
it owns, the runtime the runner drives and the KV chunk address table tt-d-gen migrates by.

Served cache, per layer (serving_contract.md, "KV cache"), allocated once by the adapter and handed to the model:
    global layer (i % 6 == 5): kv [slots, 1, max_seq / SP, 640] bfp8, the packed row [K rotary | V]; TP column t: head t
    sliding layer:             k, v [slots, 4, max_seq / SP, 256] bfp8 (K lanes in adjacent pairs); TP column t: 4t..
ND-sharded [1, 1, 32, width] round robin over every DRAM bank: one 32-token record per shard, so shard
s = (slot * heads + head) * (max_seq / SP / 32) + local_row / 32 sits in bank s % banks at base + s // banks * bytes.
SP row r's local row lr holds position (lr // L) * C + r * L + lr % L (L = C / SP, block-cyclic).

Table (36 configs, gemma4_d_p's names and order): 00..03 global head h, 04..19 sliding K head h, 20..35 sliding V
head h; a config maps only its own layer type. One device group per chip (each chip owns its records alone).

Acks: one per layer per chunk, after the layer's cache writes: the runner's D2H socket record when it passes a
d2h_service, else synchronize_device and the host sink (set_layer_completion_sink).
"""

from __future__ import annotations

import hashlib
import socket
import zlib
from pathlib import Path
from types import SimpleNamespace

from models.demos.common.prefill.adapter import KvCaches, PrefillModelAdapter
from models.demos.common.prefill.runners.migration import get_num_dram_banks

from .settings import settings

MESH_SHAPE = (8, 4)
GLOBAL_HEADS, SLIDING_HEADS = 4, 16
GLOBAL_WIDTH, SLIDING_WIDTH = 640, 256
TILE = 32
BFP8_TILE_BYTES = 1088
GLOBAL_CHUNK_BYTES = GLOBAL_WIDTH // TILE * BFP8_TILE_BYTES
SLIDING_CHUNK_BYTES = SLIDING_WIDTH // TILE * BFP8_TILE_BYTES
CONFIG_NAMES = (
    tuple(f"{h:02d}_global_h{h}" for h in range(GLOBAL_HEADS))
    + tuple(f"{h + 4:02d}_sliding_k_h{h}" for h in range(SLIDING_HEADS))
    + tuple(f"{h + 20:02d}_sliding_v_h{h}" for h in range(SLIDING_HEADS))
)


def cache_capacity(max_seq: int, chunk: int) -> int:
    """Positions the served caches hold: at least two chunks, so a chunk's Q slab is always shorter than the
    per-chip K cache and the sliding ring SDPA takes its chunked path (it rejects the equal-length one;
    gemma4_d_p ring_cache_capacity)."""
    return max(max_seq, 2 * chunk)


def is_global(layer: int) -> bool:
    return layer % 6 == 5


def config_width(config_id: int) -> int:
    return GLOBAL_WIDTH if config_id < GLOBAL_HEADS else SLIDING_WIDTH


class Gemma431bGlxConfig:
    NUM_LAYERS = 60
    FABRIC_PAYLOAD_SIZE = 8192
    MESH_SHAPE = MESH_SHAPE


class Gemma431bGlxKvCaches(KvCaches):
    """The served caches of the layers [0, num_layers): {layer: tensor (global) or (k, v) (sliding)}.
    max_seq_len: the positions they hold (cache_capacity), which the table and the model are built for."""

    def __init__(self, layers: dict, *, max_seq_len: int, num_users: int):
        self.layers, self.max_seq_len, self.num_users = layers, max_seq_len, num_users

    def tensors(self):
        """(layer, tensor) in layer order, a sliding layer's K then V."""
        for layer, c in self.layers.items():
            for t in (c,) if is_global(layer) else c:
                yield layer, t


# ---------------------------------------------------------------- the KV chunk address table
def worker_host_name(hostname: str | None = None) -> str:
    """The host key the migration worker expects (gemma4_d_p kv_chunk_table.worker_host_name)."""
    hostname = socket.gethostname() if hostname is None else hostname
    return f"host-{zlib.crc32(hostname.encode()) & 0x7FFFFFFF:08x}"


def _streams(kv: Gemma431bGlxKvCaches):
    """(config_id, layer, tensor, tp_column, heads_per_device, local_head, chunk_bytes) of every table row."""
    out = []
    for layer, c in kv.layers.items():
        if is_global(layer):
            out += [(h, layer, c, h, 1, 0, GLOBAL_CHUNK_BYTES) for h in range(GLOBAL_HEADS)]
        else:
            k, v = c
            for h in range(SLIDING_HEADS):
                out.append((4 + h, layer, k, h // 4, 4, h % 4, SLIDING_CHUNK_BYTES))
                out.append((20 + h, layer, v, h // 4, 4, h % 4, SLIDING_CHUNK_BYTES))
    return out


def _records(*, max_seq: int, chunk: int, sp: int, slots: int, heads: int, local_head: int):
    """(cp_row, slot, position, shard) of one head's 32-token records, the ROUND_ROBIN_1D walk."""
    slab = chunk // sp
    blocks_local = max_seq // sp // TILE
    for cp_row in range(sp):
        for slot in range(slots):
            first = (slot * heads + local_head) * blocks_local
            for b in range(blocks_local):
                lr = b * TILE
                yield cp_row, slot, (lr // slab) * chunk + cp_row * slab + lr % slab, first + b


def build_table(mesh, kv: Gemma431bGlxKvCaches, chunk: int):
    import ttnn

    D = ttnn.experimental.disaggregation
    sp, _ = tuple(mesh.shape)
    n_layers = len(kv.layers)
    configs = {}
    for i, name in enumerate(CONFIG_NAMES):
        cfg = D.KvChunkAddressTableConfig()
        cfg.num_layers = n_layers
        cfg.max_sequence_length = kv.max_seq_len
        cfg.num_slots = kv.num_users
        cfg.chunk_n_tokens = TILE
        cfg.chunk_size_bytes = config_width(i) // TILE * BFP8_TILE_BYTES
        configs[name] = cfg
    table = D.KvChunkAddressTable(configs)
    names = tuple(table.config_name(i) for i in range(table.num_configs()))
    if names != CONFIG_NAMES:
        raise RuntimeError(f"table config order {names[:4]}... != {CONFIG_NAMES[:4]}...")
    banks = get_num_dram_banks(mesh)
    groups = {}

    def group(cp_row, tp_col):
        if (cp_row, tp_col) not in groups:
            fnid = mesh.get_fabric_node_id(ttnn.MeshCoordinate(cp_row, tp_col))
            groups[(cp_row, tp_col)] = table.add_device_group([fnid])
            table.set_fabric_node_host(fnid, host_name=worker_host_name())
        return groups[(cp_row, tp_col)]

    table_set = table.set
    for config_id, layer, tensor, tp_col, heads, local_head, nbytes in _streams(kv):
        base = int(tensor.buffer_address())
        loc = D.KvCacheLocation()
        loc.size_bytes = nbytes
        for cp_row, slot, pos, shard in _records(
            max_seq=kv.max_seq_len, chunk=chunk, sp=sp, slots=kv.num_users, heads=heads, local_head=local_head
        ):
            loc.device_group_index = group(cp_row, tp_col)
            loc.noc_addr = (shard % banks) << 32 | base + shard // banks * nbytes
            table_set(layer, pos, slot, loc, config_id)
    return table


_TABLE_SEED = "gemma4_31b_glx:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def _table_key(mesh, kv: Gemma431bGlxKvCaches, chunk: int) -> dict:
    """Every input build_table reads."""
    import ttnn

    sp, tp = tuple(mesh.shape)
    nodes = []
    for r in range(sp):
        for t in range(tp):
            f = mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, t))
            nodes.append((r, t, int(f.mesh_id), int(f.chip_id)))
    return dict(
        configs=CONFIG_NAMES,
        max_seq=kv.max_seq_len,
        slots=kv.num_users,
        chunk=chunk,
        banks=get_num_dram_banks(mesh),
        host=worker_host_name(),
        nodes=nodes,
        streams=[(s[0], s[1], int(s[2].buffer_address()), s[3], s[4], s[5]) for s in _streams(kv)],
    )


# ---------------------------------------------------------------- the producer's read-back
def _pcc(a, b) -> float:
    import torch

    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    den = (a.norm() * b.norm()).item()
    return 1.0 if den == 0 and torch.equal(a, b) else (a @ b).item() / den if den else 0.0


def read_record_rows(table, device_map, layer, slot_id, config_id, real_len):
    """Rows [0, real_len) of one config, read through the table: the records of one (device group, bank) are
    contiguous, so each is one UMD read (gemma4_d_p kv_validation.read_cache_head). -> [real_len, width] float."""
    import torch

    import ttnn
    from models.demos.common.prefill.runners.prefill_producer import _decode_bfp8_chunk, _resolve_unique_id

    cfg = table.config(config_id)
    width = config_width(config_id)
    if cfg.chunk_n_tokens != TILE or cfg.chunk_size_bytes != width // TILE * BFP8_TILE_BYTES:
        raise ValueError(f"config {config_id}: not 32-token bfp8 records of width {width}")
    banks = {}
    for pos in range(0, real_len, TILE):
        loc = table.lookup(layer, pos, slot_id, config_id)
        if loc.size_bytes != cfg.chunk_size_bytes:
            raise ValueError(f"missing KV record: layer {layer} config {config_id} pos {pos}")
        banks.setdefault((int(loc.device_group_index), loc.noc_addr >> 32), []).append((loc.noc_addr, pos // TILE))
    out = torch.empty(-(-real_len // TILE), TILE, width)
    for (grp, _), locs in banks.items():
        locs.sort()
        addr = locs[0][0]
        if any(a != addr + i * cfg.chunk_size_bytes for i, (a, _) in enumerate(locs)):
            raise ValueError(f"non-contiguous bank: layer {layer} config {config_id}")
        nodes = table.get_device_group(ttnn.experimental.disaggregation.DeviceGroupIndex(grp)).fabric_node_ids
        raw = ttnn.experimental.disaggregation.read_dram_umd(
            _resolve_unique_id(nodes, device_map), addr, len(locs) * cfg.chunk_size_bytes
        )
        rows = _decode_bfp8_chunk(bytes(raw), width * len(locs)).reshape(TILE, len(locs), width).permute(1, 0, 2)
        out[[i for _, i in locs]] = rows
    return out.reshape(-1, width)[:real_len]


def read_slot_kv_and_check_pcc(table, device_map, slot_id, real_len, trace_dir) -> dict:
    """Every served layer's KV of slot_id, rows [0, real_len), against the golden (trace_dir/kv_cache/layer_<i>:
    key / value [heads, seq, dim], NeoX lanes) mapped to the served layout. -> minimum PCC per cache kind."""
    from safetensors import safe_open

    from .attention import sliding_kv_indices
    from .global_attention import pack_global_kv

    names = tuple(table.config_name(i) for i in range(table.num_configs()))
    if names != CONFIG_NAMES:
        raise ValueError(f"not the gemma4_31b_glx table: configs {names[:4]}...")
    real_len -= real_len % TILE
    out = {"global": 1.0, "sliding_k": 1.0, "sliding_v": 1.0}
    for layer in range(table.config(0).num_layers):
        with safe_open(str(Path(trace_dir) / "kv_cache" / f"layer_{layer}.safetensors"), framework="pt") as f:
            key = f.get_tensor(f"key_cache_layer_{layer}").float()[:, :real_len]
            value = f.get_tensor(f"value_cache_layer_{layer}").float()[:, :real_len]
        if is_global(layer):
            want = pack_global_kv(key, value)
            for h in range(GLOBAL_HEADS):
                got = read_record_rows(table, device_map, layer, slot_id, h, real_len)
                out["global"] = min(out["global"], _pcc(got, want[h]))
            continue
        order = sliding_kv_indices(SLIDING_WIDTH)
        for h in range(SLIDING_HEADS):
            k = read_record_rows(table, device_map, layer, slot_id, 4 + h, real_len)
            v = read_record_rows(table, device_map, layer, slot_id, 20 + h, real_len)
            out["sliding_k"] = min(out["sliding_k"], _pcc(k, key[h][:, order]))
            out["sliding_v"] = min(out["sliding_v"], _pcc(v, value[h]))
    return out


# ---------------------------------------------------------------- the runtime
class Gemma431bGlxRuntime:
    """One chunk at a time, op by op (no trace), into the engine-owned caches."""

    def __init__(self, *, mesh_device, ckpt: str, params):
        self.mesh_device, self.ckpt, self.config = mesh_device, ckpt, params
        self.model = None
        self.kv = None
        self.layer_completion_sink = None
        self.d2h_service = None

    def compile(self, kv_cache):
        """Load the model's weights over the given caches. No warm-up chunk: it would write slot 0."""
        from .model import Model

        p = self.config
        spec = {"paths.hf": self.ckpt, "target.chunk": p.chunk_size, "target.seq": kv_cache.max_seq_len}
        self.model = Model(
            self.mesh_device,
            SimpleNamespace(get=spec.__getitem__),
            range(p.num_layers),
            lm_head=False,
            caches=kv_cache.layers,
        )
        self.kv = kv_cache

    def _check(self, kv_cache):
        if self.model is None:
            raise RuntimeError("compile(kv) must run first")
        if kv_cache is not self.kv:
            raise ValueError("the runtime serves the caches it was compiled with")

    def set_layer_completion_sink(self, sink):
        self.layer_completion_sink = sink

    def set_d2h_ack_service(self, service):
        self.d2h_service = service

    def prefill_chunk(
        self,
        input_tensor,
        kv_cache,
        *,
        slot_id,
        actual_start,
        actual_end,
        request_id=0,
        d2h_service=None,
        metadata_msg=None,
    ):
        """input_tensor: [SP, 1, C / SP] uint32 as the H2D service delivers it (device row order, PAD_ID past
        actual_end); consumed, as is metadata_msg."""
        import ttnn

        self._check(kv_cache)
        p, m = self.config, self.model
        if not 0 <= slot_id < p.num_users:
            raise ValueError(f"slot {slot_id} outside [0, {p.num_users})")
        if actual_start % TILE or not 0 <= actual_start < actual_end <= actual_start + p.chunk_size:
            raise ValueError(f"chunk [{actual_start}, {actual_end}) is not a 32-aligned chunk of <= {p.chunk_size}")
        if actual_start + p.chunk_size > kv_cache.max_seq_len:
            raise ValueError(f"chunk at {actual_start} passes the cache's {kv_cache.max_seq_len} positions")
        d2h = d2h_service if d2h_service is not None else self.d2h_service
        if d2h is not None and metadata_msg is None:
            raise ValueError("D2H layer acks need the chunk's metadata_msg")
        h = m.embed_device(input_tensor, actual_start, actual_end, slot_id)
        for i in m.layers:
            out = m.layer(i, h, actual_start, None, actual_end)
            m.free(h)
            h = out
            # The ack follows the layer's cache writes (same command queue; the host sink after a full sync).
            if d2h is not None:
                ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(d2h, metadata=metadata_msg)
            elif self.layer_completion_sink is not None:
                ttnn.synchronize_device(self.mesh_device)
                self.layer_completion_sink(i, request_id)
        m.free(h)
        ttnn.deallocate(input_tensor)
        if metadata_msg is not None:
            ttnn.deallocate(metadata_msg)

    def kv_migration_stages(self, kv_cache, first_layer_idx, num_my_layers):
        from models.demos.common.prefill.runners.migration import KvCacheStage

        if first_layer_idx != 0 or num_my_layers != len(kv_cache.layers):
            raise ValueError("gemma4_31b_glx serves its layers on one rank")
        return [KvCacheStage(int(t.buffer_address()), layer, 1) for layer, t in kv_cache.tensors()]

    def kv_migration_base_address(self, kv_cache) -> int:
        """The DRAM offset of table entry (layer 0, position 0, slot 0, config 0): shard 0 of layer 0's global cache,
        i.e. its base; 0 when layer 0 is sliding (config 0 maps global layers only: that entry is unset)."""
        return int(kv_cache.layers[0].buffer_address()) & 0xFFFFFFFF if is_global(0) else 0

    def build_kv_chunk_table(self, kv_cache, path, *, first_layer_idx=0, num_my_layers=None, stage_layouts=None):
        from loguru import logger

        import ttnn

        stages = self.kv_migration_stages(
            kv_cache, first_layer_idx, len(kv_cache.layers) if num_my_layers is None else num_my_layers
        )
        if stage_layouts is not None:
            if len(stage_layouts) != len(stages):
                raise ValueError("one stage per cache tensor expected")
            for gathered, st in zip(stage_layouts, stages):
                if len(gathered) != 1 or any(
                    gathered[0][k] != v
                    for k, v in (("rank", 0), ("base_addr", st.base_addr), ("first_layer", st.first_layer))
                ):
                    raise ValueError("a gathered stage does not match its single-rank cache")
        chunk = self.config.chunk_size
        hit = ttnn.experimental.disaggregation.get_or_build_kv_chunk_table(
            _TABLE_SEED,
            _table_key(self.mesh_device, kv_cache, chunk),
            lambda: build_table(self.mesh_device, kv_cache, chunk),
            str(path),
        )
        logger.info(f"[gemma4_31b_glx] KV chunk table {'reused' if hit else 'built'} -> {path}")
        return str(path)

    def release_trace(self):
        self.d2h_service = None
        self.layer_completion_sink = None


# ---------------------------------------------------------------- the adapter
def _validate(params):
    if tuple(params.mesh_shape) != MESH_SHAPE or (params.sp_axis, params.tp_axis) != (0, 1):
        raise ValueError(f"gemma4_31b_glx needs the SP8 x TP4 mesh, got {params.mesh_shape}")
    if params.first_layer_idx != 0 or not (params.is_first_rank and params.is_last_rank):
        raise ValueError("gemma4_31b_glx serves its layers on one rank")
    if params.use_trace:
        raise ValueError("gemma4_31b_glx runs op by op: PREFILL_USE_TRACE=0")
    if params.max_seq_len % params.chunk_size or params.chunk_size % (MESH_SHAPE[0] * MESH_SHAPE[1] * TILE):
        raise ValueError(f"max_seq {params.max_seq_len} / chunk {params.chunk_size}: not the served geometry")
    if settings.get("ATTN_GLOBAL_SDPA") != "composed":
        raise ValueError("serving needs ATTN_GLOBAL_SDPA=composed (the slot-aware global attention)")


class Gemma431bGlxAdapter(PrefillModelAdapter):
    name = "gemma4_31b_glx"
    model_config = Gemma431bGlxConfig
    hf_model_default = settings.get("SERVING_HF_MODEL")
    ttnn_cache_default = ""
    prefill_trace_default = settings.get("SERVING_GOLDEN_DIR")
    pipeline_activation_emb_tp_sharded = False

    def load_hf_config(self):
        from . import weights

        return SimpleNamespace(**weights.text_config(self.hf_model_default))

    def weight_cache_path(self, mesh_shape):
        return None  # weights load from the checkpoint (tt/weights.py)

    def cache_kind(self, config_id: int) -> str:
        return "global_kv" if config_id < GLOBAL_HEADS else ("sliding_k" if config_id < 20 else "sliding_v")

    def cache_layer_rows(self, config_id, num_layers):
        return {layer: layer for layer in range(num_layers) if is_global(layer) == (config_id < GLOBAL_HEADS)}

    def cache_head_dim(self, config_id):
        return config_width(config_id)

    def allocate_kv_cache(self, *, mesh_device, hf_config, params):
        from .attention import _allocate_ring_cache

        _validate(params)
        sp, tp = MESH_SHAPE
        capacity = cache_capacity(params.max_seq_len, params.chunk_size)
        rows, slots = capacity // sp, params.num_users
        layers = {}
        for i in range(params.num_layers):
            if is_global(i):
                layers[i] = _allocate_ring_cache(mesh_device, [slots, 1, rows, GLOBAL_WIDTH], GLOBAL_WIDTH)
            else:
                shape = [slots, SLIDING_HEADS // tp, rows, SLIDING_WIDTH]
                layers[i] = tuple(_allocate_ring_cache(mesh_device, shape, SLIDING_WIDTH) for _ in range(2))
        return Gemma431bGlxKvCaches(layers, max_seq_len=capacity, num_users=slots)

    def build_runtime(self, *, mesh_device, hf_config, params):
        _validate(params)
        return Gemma431bGlxRuntime(mesh_device=mesh_device, ckpt=self.hf_model_default, params=params)

    def read_slot_kv_and_check_pcc(self, table, device_map, slot_id, real_len, trace_dir):
        return read_slot_kv_and_check_pcc(table, device_map, slot_id, real_len, trace_dir)
