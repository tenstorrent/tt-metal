# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash decode model: V4's traced TP pipeline around V4.1 layers.

The deployment is TP4 across 8 stages (a 32-chip mesh opened as ``(8, 4)``). Each stage is a
1x4 submesh; with the profile's pipeline group size of 1 it owns one contiguous run of layers.
What V4 already does is reused: the H2D packet ``[token | pos_sliding | pos_compress]``, on-device
RoPE, D2D sockets, trace variants, the replay thread and the DRISC prefetch rings. This file only
replaces the parts V4.1 changed.

* Layers exchange ``(streams [B, 1, hc, D], pre_mix [B, 1, 1, hc])``. The first ``pre_mix`` is the
  one-hot ``[1, 0, 0, 0]``; the last one collapses the streams (there is no ``hc_head``).
* Engram layers (1 and 14) take their rows on a per-rank H2D socket. The host writes them in
  :meth:`_write_packet`, before the step that receives them.
* A compressed group shares a tiled ``[ring | selected]`` buffer (``sel_tile``); stages with an
  index source of the group also keep ``sel_kv``, ``comp_kv`` and ``idx_key_cache``. A pipeline
  edge sends ``sel_tile`` of each group the two stages share and, when a stage without the
  group's kv source selects from it, the one compressed-KV row and index key the source wrote
  this step -- not the whole caches, which grow with the context.
"""

from __future__ import annotations

import collections
import math
import queue
import threading
from typing import Optional, Sequence

import torch
import ttnn
from loguru import logger

from models.experimental.deepseek_v41_flash.tt.config import (
    decoder_layer_weights,
    dequantized,
    expert_provider,
)
from models.experimental.deepseek_v41_flash.tt.decode.attention import _SELECT_KV_RATE, DeepSeekV41LayerCache
from models.experimental.deepseek_v41_flash.tt.decode.decoder_layer import DeepSeekV41DecoderLayer
from models.experimental.deepseek_v41_flash.tt.decode.moe import DeepSeekV41PreloadedExperts
from models.experimental.deepseek_v41_flash.tt.engram_lookup import (
    EngramHostLookup,
    engram_table_nbytes,
    load_engram_table,
)
from models.experimental.deepseek_v4_flash.tt.common import (
    _MASK_NEG,
    _profile,
    _region,
    _traced_decode_enabled,
    width_sharded_l1_config,
)
from models.experimental.deepseek_v4_flash.tt.decode.attention import (
    CSA_INDEX_BLOCK_SIZE,
    _index_key_pool,
)
from models.experimental.deepseek_v4_flash.tt.decode.hyperconnection import _HC_COLLAPSE_CORES
from models.experimental.deepseek_v4_flash.tt.embedding import DeepSeekV4Embedding
from models.experimental.deepseek_v4_flash.tt.layers import DeepSeekV4RMSNorm
from models.experimental.deepseek_v4_flash.tt.model import (
    DeepSeekV4Model,
    _OUT_SOCKET_CORE,
    _PKT_SOCKET_CORE,
    plan_layer_placement,
)
from models.experimental.deepseek_v4_flash.tt.system_config import load_system_config, set_active_system_config
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache, _as_cache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

# Host RAM kept free above the Engram tables, matching the decoder-layer test.
_ENGRAM_RAM_MARGIN = 1.25
_ROW_NAMES = ("new_kv_row", "new_key_row")


def _host_mem_available() -> int:
    """Bytes of host RAM ``MemAvailable`` currently reports."""
    with open("/proc/meminfo") as meminfo:
        return next(int(line.split()[1]) * 1024 for line in meminfo if line.startswith("MemAvailable:"))


def _device_zeros(shape, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    """An all-zero DRAM tensor. ``from_torch`` takes fp32 for bf16 and int32 for the integer dtypes."""
    host_dtype = {ttnn.bfloat16: torch.float32, ttnn.float32: torch.float32, ttnn.int32: torch.int32}[dtype]
    return ttnn.from_torch(
        torch.zeros(shape, dtype=host_dtype),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


class DeepSeekV41Model(DeepSeekV4Model):
    """V4.1 decode stack on V4's traced pipeline.

    ``tp_size`` defaults to 4 and ``num_stages`` to 8, so a 32-chip mesh runs one 1x4 stage per
    row. ``use_submeshes`` stays on: traced decode sends the packet and the residual over sockets.
    ``weight_dtype`` is the attention and routed-expert dtype (bfloat4_b in the demo); the shared
    expert stays bfloat8_b and the Engram, norms and hyper-connections stay at their own defaults.

    Do not call :meth:`DeepSeekV4Model.__init__`. It builds V4 layers, reads ``compress_rates`` and
    ``mlp_layer_types``, and constructs ``hc_head``, none of which exist on this checkpoint.
    """

    def __init__(
        self,
        config,
        loader: DeepseekV4WeightLoader,
        full_device: ttnn.MeshDevice,
        cache: Optional[WeightCache] = None,
        cache_dir: Optional[str] = None,
        weight_dtype: Optional[ttnn.DataType] = None,
        max_layers: Optional[int] = None,
        use_submeshes: bool = True,
        require_cache: bool = False,
        pipeline_group_size: Optional[int] = None,
        num_prefetch_pages: Optional[int] = None,
        system_config=None,
        tp_size: int = 4,
        num_stages: int = 8,
        submeshes: Optional[Sequence[ttnn.MeshDevice]] = None,
        use_prefetcher: Optional[bool] = None,
    ):
        self.config = config
        self.loader = loader
        self.device = full_device
        if tp_size < 1:
            raise ValueError(f"tp_size must be >= 1, got {tp_size}")
        if not use_submeshes:
            raise NotImplementedError("V4.1 traced decode runs on 1xTP submeshes")
        if full_device.get_num_devices() % tp_size:
            raise ValueError(
                f"mesh device count {full_device.get_num_devices()} must be divisible by tp_size {tp_size}"
            )
        self.tp_size = tp_size
        if system_config is None:
            system_config = load_system_config(mesh_device=full_device).log()
        self.system_config = system_config
        set_active_system_config(system_config)

        self.weight_dtype = weight_dtype if weight_dtype is not None else system_config.decode.ttnn_weight_dtype
        weight_dtype = self.weight_dtype
        self.use_prefetcher = True if use_prefetcher is None else bool(use_prefetcher)
        if num_prefetch_pages is None:
            num_prefetch_pages = system_config.prefetcher.num_prefetch_pages
        self._prefetch_buffers_by_device: dict[int, tuple] = {}
        self._gcb_headroom: dict[int, ttnn.Tensor] = {}
        self._released_gcbs = None
        self._evicted_l1: list = []
        self._l1_holders: list = []
        self._pre_release = None
        self._resident_snapshot: list = []
        if cache is None and cache_dir is not None:
            cache = WeightCache(cache_dir)
        cache = _as_cache(cache)
        if require_cache:
            if not cache.path:
                raise ValueError("require_cache=True needs a populated cache; pass cache= or cache_dir=")
            cache = cache.require(True)
        self.cache = cache

        self.use_submeshes = True
        self.mesh_devices = full_device.get_num_devices()
        pipeline_devices = system_config.pipeline.resolve_num_devices(self.mesh_devices)
        if num_stages < 1 or num_stages * tp_size > pipeline_devices:
            raise ValueError(
                f"num_stages={num_stages} x TP{tp_size} does not fit the {pipeline_devices} pipeline devices"
            )
        pipeline_devices = num_stages * tp_size
        self.pipeline_devices = pipeline_devices
        self.num_submeshes = pipeline_devices // tp_size

        n = config.num_hidden_layers if max_layers is None else min(max_layers, config.num_hidden_layers)
        self.num_layers = n
        if pipeline_group_size is None:
            pipeline_group_size = max(0, system_config.pipeline.group_size)
        self.layer_submesh_ids = plan_layer_placement(self.num_layers, self.num_submeshes, pipeline_group_size)
        self.pipeline_submesh_ids = list(dict.fromkeys(self.layer_submesh_ids))
        self.pipeline_stages = len(self.pipeline_submesh_ids)
        self.pipeline_edges = list(
            dict.fromkeys(
                (self.layer_submesh_ids[li - 1], self.layer_submesh_ids[li])
                for li in range(1, self.num_layers)
                if self.layer_submesh_ids[li - 1] != self.layer_submesh_ids[li]
            )
        )
        idle = self.mesh_devices - self.pipeline_devices
        logger.info(
            f"Using submeshes: {self.num_submeshes} x TP{tp_size} over {self.pipeline_devices} of "
            f"{self.mesh_devices} chips"
            f"{f' ({idle} left idle)' if idle else ''} (pipeline group size "
            f"{pipeline_group_size or self.num_submeshes}, {self.pipeline_stages} populated)"
        )
        if submeshes is not None:
            if len(submeshes) != self.num_submeshes:
                raise ValueError(f"got {len(submeshes)} submeshes for {self.num_submeshes} pipeline stages")
            self.submeshes = list(submeshes)
        else:
            full_device.reshape(ttnn.MeshShape(self.mesh_devices // tp_size, tp_size))
            self.submeshes = [
                full_device.create_submesh(ttnn.MeshShape(1, tp_size), ttnn.MeshCoordinate(i, 0))
                for i in range(self.num_submeshes)
            ]
        self.first_device = self.submeshes[0]
        self.last_device = self.submeshes[-1]
        self.submesh_socket_pairs = {
            (src, dst): self._create_socket_pair(self.submeshes[src], self.submeshes[dst])
            for src, dst in self.pipeline_edges
        }
        # V4.1 has 40 layers, so an 8-stage mesh has no idle row left for a DSpark tap.
        self.mtp_submesh = None
        self._mtp_sender = None
        self._mtp_receiver = None
        self._mtp_pack = None
        self._mtp_hiddens = None

        self._require_engram_ram()
        self.embed_tokens = DeepSeekV4Embedding(loader, self.first_device, cache=cache)
        self.layers: list[DeepSeekV41DecoderLayer] = []
        self.layer_devices: list[ttnn.MeshDevice] = []
        for li in range(n):
            current_device = self.submeshes[self.layer_submesh_ids[li]]
            logger.info(f"Layer {li} is on device {self.layer_submesh_ids[li]}")
            self.layer_devices.append(current_device)
            layer_cache = cache.sub(f"layers.{li}")
            prefetch_buffers = (
                self._prefetch_buffers_for(current_device, weight_dtype, num_prefetch_pages)
                if self.use_prefetcher
                else None
            )
            experts = DeepSeekV41PreloadedExperts(
                config,
                expert_provider(loader, li),
                current_device,
                dtype=weight_dtype,
                cache=layer_cache.sub("ffn"),
                tp_size=tp_size,
            )
            self.layers.append(
                DeepSeekV41DecoderLayer(
                    config,
                    li,
                    decoder_layer_weights(loader, li, config),
                    current_device,
                    experts,
                    cache=layer_cache,
                    weight_dtype=weight_dtype,
                    tp_size=tp_size,
                    use_prefetcher=self.use_prefetcher,
                    prefetch_buffers=prefetch_buffers,
                )
            )
            _profile(current_device)
        if self.layer_devices:
            self.last_device = self.layer_devices[-1]

        self.sliding_window = config.sliding_window
        self.hc_head = None
        self.norm = DeepSeekV4RMSNorm(
            dequantized(loader, "norm.weight"),
            config.rms_norm_eps,
            self.last_device,
            cache.file("norm"),
            sharded=True,
        )
        self._engram_layers = [(layer.layer_idx, layer.engram) for layer in self.layers if layer.engram is not None]
        self._engram_lookup = None
        self._decode_max_seq = None
        self._context_limit = None
        self._csa_entries_copied = False
        self._paged = None
        self._traced_captured = False
        self._traces_compiled = False
        self._eager_decode = not _traced_decode_enabled()
        self._eager_pending: collections.deque = collections.deque()
        self._eager_outs: list = []
        self._replay_queue: queue.Queue = queue.Queue()
        self._replay_thread: Optional[threading.Thread] = None
        self._pkt_socket = None
        self._out_socket = None
        self._out_plan = None
        self._out_torch_dtype = None
        self._resident: list[Optional[int]] = []
        self._session_pos: dict[int, int] = {}
        self._decode_batch = 1
        self._cache_handoff: dict[tuple[int, int], tuple[list, list]] = {}

    def _engram_ids(self) -> list[int]:
        return [i for i in self.config.engram_layer_ids if i < self.num_layers]

    def _require_engram_ram(self) -> None:
        """Refuse to build when the Engram tables (plus the same headroom the tests use) do not fit."""
        ids = self._engram_ids()
        if not ids:
            return
        need = sum(engram_table_nbytes(self.loader, i) for i in ids)
        available = _host_mem_available()
        if available < _ENGRAM_RAM_MARGIN * need:
            raise MemoryError(
                f"Engram tables for layers {ids} need {_ENGRAM_RAM_MARGIN * need / 1e9:.0f} GB with headroom; "
                f"{available / 1e9:.0f} GB is available"
            )

    @property
    def paged(self) -> bool:
        """True once :meth:`prepare_static_decode` has built the shared compressed-KV pools.

        One session, sized up front for ``max_seq``. The index-key and compressed-KV pools are
        paged in ``CSA_INDEX_BLOCK_SIZE``-row blocks; the sliding ring stays dense.
        """
        return self._decode_max_seq is not None

    def prepare_static_decode(
        self,
        rope: dict,
        max_seq: int,
        lm_head=None,
        num_sessions: int = 1,
        total_tokens: int | None = None,
        block_size: int = 32,
        batch: int = 1,
    ) -> None:
        """Allocate the traced-decode buffers.

        ``rope["main"]`` and ``rope["compress"]`` are ``(cos_half, sin_half)`` tables ``[max_seq, Rd/2]``
        from the checkpoint's ``precompute_freqs_cis`` (sliding: no YaRN; compressed: YaRN). The device
        rebuilds one row per step from them. ``block_size`` and ``total_tokens`` are accepted so the V4
        demo call works; the index pool uses :data:`CSA_INDEX_BLOCK_SIZE` and is sized for ``max_seq``.
        ``batch`` and ``num_sessions`` are 1: the select op decodes one user, and there is no multi-session
        page-table swap.
        """
        del total_tokens
        if batch != 1:
            raise NotImplementedError(f"V4.1 decode supports batch 1, got batch={batch}")
        if num_sessions != 1:
            raise NotImplementedError(f"V4.1 decode supports one session, got num_sessions={num_sessions}")
        if max_seq < 2:
            raise ValueError(f"max_seq must be at least 2 so RoPE can be recovered from row 1, got {max_seq}")
        cfg = self.config
        ratios = {int(r) for r in cfg.compress_ratios[: self.num_layers] if int(r)}
        limit = cfg.candidate_topk_blocks * cfg.candidate_block_size if ratios else None
        self._context_limit = limit
        if limit is not None and max_seq > limit:
            logger.warning(
                f"max_seq {max_seq} exceeds the {limit}-token compressed-KV context; sessions are capped there"
            )
            max_seq = limit
        for ratio in ratios:
            if max_seq % ratio:
                raise ValueError(f"max_seq ({max_seq}) must be a multiple of compress ratio {ratio}")
        self._decode_batch = batch
        self._lm_head_traced = lm_head
        self._decode_max_seq = max_seq
        # Variant axis 1 is the r=2 pool phase. Stages with no r=2 source ignore it.
        has_r2 = 2 in ratios
        self._pool_crs = [2] if has_r2 else []
        self._pool_phases = [(False,), (True,)] if has_r2 else [()]
        logger.info(
            f"V4.1 traced decode: max_seq {max_seq}, index blocks of {CSA_INDEX_BLOCK_SIZE} "
            f"(prepare block_size={block_size} is unused)"
        )

        rope_gen = {}
        for family in ("main", "compress"):
            cos_h, sin_h = rope[family]
            scaling = float(cos_h[0, 0].item())
            inv_freq = torch.atan2(sin_h[1].float(), cos_h[1].float()).repeat_interleave(2)
            rope_gen[family] = (inv_freq.reshape(1, 1, 1, -1), scaling)

        alignment = self.system_config.pipeline.pcie_alignment
        self._pkt_int_prefix = 3 * batch
        self._pkt_page_bytes = math.ceil(self._pkt_int_prefix * 4 / alignment) * alignment
        self._pkt_w = self._pkt_page_bytes // 4

        hc, hidden, window = cfg.hc_mult, cfg.hidden_size, self.sliding_window
        ids = self.layer_submesh_ids
        self._kv_source_of = {
            li: max(s for s in cfg.kv_source_layer_ids if s <= li)
            for li in range(self.num_layers)
            if cfg.compress_ratios[li]
        }
        # The first index source of a group on each stage that lacks the group's kv source
        # writes the rows that source sent.
        self._row_writers = set()
        for li in range(self.num_layers):
            src = self._kv_source_of.get(li)
            if src is None or li == src or li not in cfg.index_source_layer_ids or ids[li] == ids[src]:
                continue
            if not any((ids[w], self._kv_source_of[w]) == (ids[li], src) for w in self._row_writers):
                self._row_writers.add(li)
        self._row_groups = {self._kv_source_of[li] for li in self._row_writers}
        self.submeshes_io = []
        for stage in self.pipeline_submesh_ids:
            device = self.submeshes[stage]
            layers_here = [li for li in range(self.num_layers) if ids[li] == stage]
            stage_ratios = sorted({int(cfg.compress_ratios[li]) for li in layers_here if cfg.compress_ratios[li]})
            kv_ratios = sorted({int(cfg.compress_ratios[li]) for li in layers_here if li in cfg.kv_source_layer_ids})
            index_ratios = sorted(
                {int(cfg.compress_ratios[li]) for li in layers_here if li in cfg.index_source_layer_ids}
            )
            row_ratios = sorted({int(cfg.compress_ratios[li]) for li in layers_here if li in self._row_writers})
            sm = {
                "device": device,
                "index": stage,
                "layers": layers_here,
                # The decode streams [B, 1, hc, D]: ROW_MAJOR, width-sharded over the hyperconnection's
                # collapse cores, the layout engram and every hyperconnection consume in place.
                "streams_memory_config": width_sharded_l1_config(
                    batch * hc, hidden, device, num_cores=_HC_COLLAPSE_CORES, tile_height=1
                ),
                "rope_invfreq": {},
                "scaches": {},
                "ratios": stage_ratios,
                "kv_ratios": kv_ratios,
                "index_ratios": index_ratios,
                "row_ratios": row_ratios,
                "has_sliding": any(cfg.compress_ratios[li] == 0 for li in layers_here),
                "has_compressed": bool(stage_ratios),
                "has_r2_pool": 2 in kv_ratios or 2 in row_ratios,
                "has_r2_select": 2 in index_ratios,
                "traces": {},
                "tids": {},
                "outputs": {},
            }
            families = set()
            if sm["has_sliding"]:
                families.add("main")
            if stage_ratios:
                families.add("compress")
            for family in families:
                inv_freq, scaling = rope_gen[family]
                sm["rope_invfreq"][family] = (
                    ttnn.from_torch(inv_freq, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device),
                    scaling,
                )
            if stage_ratios:
                mask_idx = torch.arange(window + cfg.index_topk, dtype=torch.float32).reshape(1, 1, 1, -1)
                sm["mask_idx"] = ttnn.from_torch(mask_idx, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
            if 0 in layers_here:
                sm["pkt"] = _device_zeros([1, 1, 1, self._pkt_w], device, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                one_hot = torch.zeros(1, 1, 1, hc, dtype=torch.float32)
                one_hot[0, 0, 0, 0] = 1
                sm["pre_mix_init"] = ttnn.from_torch(
                    one_hot, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                )
                if self._pkt_socket is None:
                    self._drop_gcb_headroom(device)
                    self._pkt_socket = ttnn.H2DSocket(
                        device,
                        ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(*_PKT_SOCKET_CORE)),
                        ttnn.BufferType.L1,
                        self.system_config.pipeline.h2d_fifo_bytes,
                        ttnn.H2DMode.HOST_PUSH,
                    )
                    self._pkt_socket.set_page_size(self._pkt_page_bytes)
            if any(li > 0 and ids[li - 1] != stage for li in layers_here):
                sm["streams_in"] = _device_zeros([batch, 1, hc, hidden], device, layout=ttnn.ROW_MAJOR_LAYOUT)
                sm["pkt_in"] = _device_zeros([1, 1, 1, self._pkt_w], device, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                sm["pre_mix_in"] = _device_zeros([batch, 1, 1, hc], device)
            self.submeshes_io.append(sm)

        self._allocate_shared_caches(max_seq)
        self._output_sm_index = self.pipeline_submesh_ids.index(ids[self.num_layers - 1])
        self._traced_captured = False
        self._traces_compiled = False
        if self._out_socket is None:
            out_device = self.submeshes_io[self._output_sm_index]["device"]
            self._drop_gcb_headroom(out_device)
            self._out_socket = ttnn.D2HSocket(
                out_device,
                ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(*_OUT_SOCKET_CORE)),
                self.system_config.pipeline.d2h_fifo_bytes,
            )
        if self._engram_layers:
            tables = {layer_idx: load_engram_table(self.loader, layer_idx) for layer_idx, _ in self._engram_layers}
            self._engram_lookup = EngramHostLookup(cfg, self.loader.snapshot_dir, tables, batch=batch, max_seq=max_seq)
            logger.info(f"Engram host lookup: layers {[layer_idx for layer_idx, _ in self._engram_layers]}")

    def _allocate_shared_caches(self, max_seq: int) -> None:
        """Per-stage KV. A group is aliased inside a stage and handed over across a pipeline edge.

        ``win_kv`` / ``win_gate`` stay on the stage that owns the kv source. Every stage of the
        group gets ``sel_tile``; a stage with an index source of the group also gets ``sel_kv``,
        ``comp_kv``, ``idx_key_cache`` and its own identity page table. Groups some stage selects
        from without holding their source carry ``new_kv_row`` / ``new_key_row`` on every stage.
        """
        cfg = self.config
        window, head_dim = self.sliding_window, cfg.head_dim
        rows = window + cfg.index_topk
        by_stage: dict[int, dict[int, dict]] = {}
        for sm in self.submeshes_io:
            stage = sm["index"]
            device = sm["device"]
            shared_here: dict[int, dict] = {}
            for li in sm["layers"]:
                ratio = int(cfg.compress_ratios[li])
                ring = _device_zeros((1, 1, window, head_dim), device)
                if not ratio:
                    sm["scaches"][li] = DeepSeekV41LayerCache(kv=ring)
                    continue
                src = self._kv_source_of[li]
                if src not in shared_here:
                    owns_window = src in sm["layers"] and ratio > 1
                    group = {
                        "win_kv": _device_zeros((1, 1, ratio, head_dim), device) if owns_window else None,
                        "win_gate": _device_zeros((1, 1, ratio, head_dim), device) if owns_window else None,
                        "sel_tile": _device_zeros((1, 1, rows, head_dim), device),
                    }
                    if src in self._row_groups:
                        group["new_kv_row"] = _device_zeros((1, 1, 1, head_dim), device, layout=ttnn.ROW_MAJOR_LAYOUT)
                        group["new_key_row"] = _device_zeros(
                            (1, 1, 1, cfg.index_head_dim), device, layout=ttnn.ROW_MAJOR_LAYOUT
                        )
                    selects_here = any(
                        w in cfg.index_source_layer_ids and self._kv_source_of.get(w) == src for w in sm["layers"]
                    )
                    if selects_here:
                        blocks = -(-(max_seq // ratio) // CSA_INDEX_BLOCK_SIZE)
                        group.update(
                            idx_key_cache=_index_key_pool(device, blocks, CSA_INDEX_BLOCK_SIZE, cfg.index_head_dim),
                            comp_kv=_device_zeros(
                                (blocks, 1, CSA_INDEX_BLOCK_SIZE, head_dim), device, layout=ttnn.ROW_MAJOR_LAYOUT
                            ),
                            idx_page_table=ttnn.from_torch(
                                torch.arange(blocks, dtype=torch.int32).reshape(1, blocks),
                                dtype=ttnn.int32,
                                layout=ttnn.ROW_MAJOR_LAYOUT,
                                device=device,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            ),
                            sel_kv=_device_zeros((1, 1, rows, head_dim), device, layout=ttnn.ROW_MAJOR_LAYOUT),
                        )
                    shared_here[src] = group
                sm["scaches"][li] = DeepSeekV41LayerCache(kv=ring, **shared_here[src])
            by_stage[stage] = shared_here
        self._cache_handoff = {}
        for src_stage, dst_stage in self.pipeline_edges:
            common = sorted(set(by_stage[src_stage]) & set(by_stage[dst_stage]))
            send, recv = [], []
            for src in common:
                for name in ("sel_tile",) + (_ROW_NAMES if src in self._row_groups else ()):
                    send.append(by_stage[src_stage][src][name])
                    recv.append(by_stage[dst_stage][src][name])
            self._cache_handoff[(src_stage, dst_stage)] = (send, recv)
            if common:
                logger.info(f"shared KV {src_stage} -> {dst_stage}: groups {common}")

    def _variant_key(self, pos: int) -> tuple[bool, int, bool]:
        """``(causal, r=2 pool phase, r=2 select)`` for absolute ``pos``.

        Causal starts at ``sliding_window - 1`` when the profile asks for it. An r=2 group pools on
        odd positions (``(pos + 1) % 2 == 0``) and selects once ``pos >= 1``. Ratio 1 pools and
        selects on every step, so it is not a variant bit.
        """
        causal = self._SDPA_CAUSAL and pos >= self.sliding_window - 1
        if not self._pool_crs:
            return (causal, 0, False)
        return (causal, int((pos + 1) % 2 == 0), pos >= 1)

    def _reachable_variants(self) -> list[tuple[bool, int, bool]]:
        """The five (or fewer) variants a position can ask for. ``(False, 1, False)`` and
        ``(True, *, False)`` never occur: the only non-select step is position 0."""
        if not self._pool_crs:
            modes = [False, True] if self._SDPA_CAUSAL else [False]
            return [(causal, 0, False) for causal in modes]
        variants = [(False, 0, False), (False, 0, True), (False, 1, True)]
        if self._SDPA_CAUSAL:
            variants += [(True, 0, True), (True, 1, True)]
        return variants

    @staticmethod
    def _sm_pool_key(sm: dict, pool_flags: dict, causal: bool, index_sparse: bool = False) -> tuple:
        """Drop variant bits this submesh's layers do not branch on.

        ``index_sparse`` is the r=2 select bit. A sliding-only stage collapses to one trace; a
        ratio-1 stage keeps only the causal bit.
        """
        return (
            causal and sm["has_compressed"],
            bool(pool_flags.get(2, False)) and sm["has_r2_pool"],
            index_sparse and sm["has_r2_select"],
        )

    def _capture_packet_pos(self, pos: int, causal: bool, index_sparse: bool) -> int:
        """A position the dry run can execute. Replay reads the real one from the packet.

        The phase is not an argument: a pool trace may be compiled at a non-pool position, and
        the cache row it dirties is overwritten once prefill runs from 0. The select dry run stays
        at or above position 1 so the index is not negative, and a causal dry run stays inside the
        sliding window.
        """
        if not index_sparse:
            packet_pos = 0
        elif not causal:
            packet_pos = 1
        else:
            packet_pos = self.sliding_window - 1
        if self._decode_max_seq:
            packet_pos = min(packet_pos, self._decode_max_seq - 1)
        return max(packet_pos, 0)

    def _copy_dense_csa_entries(self) -> None:
        """No-op. V4 copies dense CSA rows into ``comp_kv`` when its indexer turns on; V4.1 writes
        ``comp_kv`` from the select itself, and ``variant[2]`` here means "r=2 select", not that copy.
        """
        return

    def _write_packet(self, token_id, pos: int) -> None:
        """The V4 packet, then each Engram layer's rows in layer order. Compile runs push zeros and do not
        advance the hash.

        The packet goes first so layer 0 can start while the host looks up the first Engram layer's rows;
        each later layer's lookup runs only after the earlier layer's rows are on their socket.
        """
        DeepSeekV4Model._write_packet(self, token_id, pos)
        self._write_engram_rows(token_id, pos)

    def _write_engram_rows(self, token_id, pos: int) -> None:
        if not self._engram_layers:
            return
        if self._eager_decode or self._traces_compiled:
            ids = torch.tensor(self._step_tokens(token_id), dtype=torch.long).reshape(-1, 1)
            lookup = self._engram_lookup
            hashes = lookup.hash(ids, pos)  # [B, 1, n_engram_layers, n_hash_cols]
            for layer_idx, engram in self._engram_layers:
                ids_of_layer = hashes[:, :, lookup.layout.layer_ids.index(layer_idx)]
                engram.write_rows(lookup.gather(layer_idx, ids_of_layer).flatten(-2))
        else:
            for _, engram in self._engram_layers:
                engram.write_rows(torch.zeros(engram.batch, engram.kin, dtype=torch.bfloat16))

    def open_session(self) -> int:
        if self._decode_max_seq is None:
            raise RuntimeError("call prepare_static_decode() before using sessions")
        if self._session_pos:
            raise RuntimeError("V4.1 traced decode seats one session")
        self._session_pos[0] = 0
        return 0

    def activate_session(self, sid: int) -> None:
        self.activate_sessions([sid])

    def activate_sessions(self, sids) -> None:
        sids = list(sids)
        if len(sids) != self._decode_batch:
            raise ValueError(f"batch decodes {self._decode_batch} users at a time, got {len(sids)} sessions")
        for sid in sids:
            if sid not in self._session_pos:
                raise KeyError(f"no such session {sid}")
        self._resident = sids

    def ensure_session_capacity(self, pos: int) -> None:
        if not self._resident or None in self._resident:
            raise RuntimeError("call activate_sessions() before decoding")
        if self._context_limit is not None and pos >= self._context_limit:
            raise RuntimeError(f"position {pos} exceeds the {self._context_limit}-token compressed-KV context")
        if self._decode_max_seq is not None and pos >= self._decode_max_seq:
            raise RuntimeError(f"position {pos} is past max_seq {self._decode_max_seq}")

    def session_usage(self) -> dict:
        if self._decode_max_seq is None:
            raise RuntimeError("call prepare_static_decode() before using sessions")
        return {"sessions": (len(self._session_pos), 1), "max_seq": self._decode_max_seq}

    def _step_context(self, sm: dict, pkt: ttnn.Tensor, causal: bool, r2_pool: bool, r2_select: bool) -> dict:
        """Token, positions, RoPE rows and the per-ratio mask or ``cur_pos`` for one step.

        Pool and select tensors are built only for the ratios this variant actually runs, so a
        stage that ignores the r=2 bits has one op sequence across those variants.
        """
        b = self._decode_batch
        token = ttnn.typecast(ttnn.reshape(ttnn.slice(pkt, [0, 0, 0, 0], [1, 1, 1, b]), [1, b]), ttnn.uint32)
        sliding_pos = ttnn.reshape(ttnn.slice(pkt, [0, 0, 0, b], [1, 1, 1, 2 * b]), [b])
        compress_pos = ttnn.reshape(ttnn.slice(pkt, [0, 0, 0, 2 * b], [1, 1, 1, 3 * b]), [b])
        pos_f = ttnn.typecast(ttnn.to_layout(ttnn.reshape(compress_pos, [1, 1, 1, b]), ttnn.TILE_LAYOUT), ttnn.float32)
        rope = {family: self._device_rope(inv, scale, pos_f) for family, (inv, scale) in sm["rope_invfreq"].items()}
        ctx = {
            "token": token,
            "sliding_pos": sliding_pos,
            "rope": rope,
            "sliding_curpos": None,
            "mask": {},
            "curpos": {},
            "win_slot": {},
            "win_rope": {},
            "entry": {},
            "index_pos": {},
        }
        if sm["has_sliding"]:
            ctx["sliding_curpos"] = self._device_index(ttnn.clamp(pos_f, max=float(self.sliding_window - 1)))
        for ratio in sm["ratios"]:
            if causal:
                ctx["curpos"][ratio] = self._compressed_curpos(pos_f, ratio)
            else:
                ctx["mask"][ratio] = self._compressed_mask(sm["mask_idx"], pos_f, ratio)
        for ratio in sm["kv_ratios"]:
            if ratio > 1:
                ctx["win_slot"][ratio] = self._win_slot(pos_f, ratio)
            if ratio == 1 or (ratio == 2 and r2_pool):
                win_pos = ttnn.add(pos_f, float(1 - ratio))
                inv, scale = sm["rope_invfreq"]["compress"]
                cos_win, sin_win, _ = self._device_rope(inv, scale, win_pos)
                ctx["win_rope"][ratio] = (cos_win, sin_win)
                ctx["entry"][ratio] = self._entry_index(pos_f, ratio)
        for ratio in sm["row_ratios"]:
            if ratio not in ctx["entry"] and (ratio == 1 or (ratio == 2 and r2_pool)):
                ctx["entry"][ratio] = self._entry_index(pos_f, ratio)
        for ratio in sm["index_ratios"]:
            if ratio == 1 or (ratio == 2 and r2_select):
                ctx["index_pos"][ratio] = self._index_position(pos_f, ratio)
        return ctx

    def _compressed_mask(self, idx: ttnn.Tensor, pos_f: ttnn.Tensor, ratio: int) -> ttnn.Tensor:
        """Additive mask ``[1, 1, 1, window + index_topk]``: valid on ``[0, pos]`` and on the selected rows."""
        selected = ttnn.clamp(self._group_count(pos_f, ratio), max=float(self.config.index_topk))
        window = float(self.sliding_window)
        ring_hole = ttnn.multiply(ttnn.gt(idx, pos_f), ttnn.lt(idx, window))
        past = ttnn.ge(idx, ttnn.add(window, selected))
        return ttnn.typecast(ttnn.multiply(ttnn.add(ring_hole, past), _MASK_NEG), ttnn.bfloat16)

    def _compressed_curpos(self, pos_f: ttnn.Tensor, ratio: int) -> ttnn.Tensor:
        selected = ttnn.clamp(self._group_count(pos_f, ratio), max=float(self.config.index_topk))
        return self._device_index(ttnn.add(selected, float(self.sliding_window - 1)))

    def _group_count(self, pos_f: ttnn.Tensor, ratio: int) -> ttnn.Tensor:
        """``(pos + 1) // ratio``, the closed groups at this position."""
        return ttnn.floor(ttnn.multiply(ttnn.add(pos_f, 1.0), 1.0 / ratio))

    def _entry_index(self, pos_f: ttnn.Tensor, ratio: int) -> ttnn.Tensor:
        return self._device_index(ttnn.floor(ttnn.multiply(pos_f, 1.0 / ratio)))

    def _win_slot(self, pos_f: ttnn.Tensor, ratio: int) -> ttnn.Tensor:
        quot = ttnn.floor(ttnn.multiply(pos_f, 1.0 / ratio))
        return self._device_index(ttnn.subtract(pos_f, ttnn.multiply(quot, float(ratio))))

    def _index_position(self, pos_f: ttnn.Tensor, ratio: int) -> ttnn.Tensor:
        """``4 * n - 1``: the select op counts ``(cur_pos + 1) // 4`` keys."""
        n = self._group_count(pos_f, ratio)
        return self._device_index(ttnn.subtract(ttnn.multiply(n, float(_SELECT_KV_RATE)), 1.0))

    def _attention_inputs(self, attn, ctx: dict, causal: bool, r2_pool: bool, r2_select: bool) -> dict:
        ratio = attn.compress_ratio
        cos, sin, neg_sin = ctx["rope"]["main" if ratio == 0 else "compress"]
        inputs = {"cos": cos, "sin": sin, "neg_sin": neg_sin, "sliding_pos": ctx["sliding_pos"]}
        if ratio == 0:
            inputs["sdpa_cur_pos"] = ctx["sliding_curpos"]
            return inputs
        if attn.is_kv_source:
            pool = ratio == 1 or (ratio == 2 and r2_pool)
            inputs["pool"] = pool
            if ratio > 1:
                inputs["win_slot"] = ctx["win_slot"][ratio]
            if pool:
                inputs["cos_win"], inputs["sin_win"] = ctx["win_rope"][ratio]
                inputs["entry"] = ctx["entry"][ratio]
        if attn.is_index_source and (ratio == 1 or (ratio == 2 and r2_select)):
            inputs["select"] = True
            inputs["index_pos"] = ctx["index_pos"][ratio]
            if attn.layer_idx in self._row_writers and (ratio == 1 or r2_pool):
                inputs["received_entry"] = ctx["entry"][ratio]
        if causal:
            inputs["sdpa_cur_pos"] = ctx["curpos"][ratio]
        else:
            inputs["mask"] = ctx["mask"][ratio]
        return inputs

    def _collapse_streams(self, streams: ttnn.Tensor, pre_mix: ttnn.Tensor) -> ttnn.Tensor:
        """``sum_j pre_mix[j] * streams[j]`` in fp32, one stream at a time.

        A ``ttnn.sum`` over ``hc`` (4) in a tile would add the tile's padding. The ROW_MAJOR sharded
        streams are tiled in DRAM first. The result is ``[B, 1, 1, D]`` bf16, the final norm's input.
        """
        batch, hidden, hc = self._decode_batch, self.config.hidden_size, self.config.hc_mult
        streams = ttnn.to_layout(ttnn.to_memory_config(streams, ttnn.DRAM_MEMORY_CONFIG), ttnn.TILE_LAYOUT)
        streams_f = ttnn.typecast(streams, ttnn.float32)
        weights = ttnn.typecast(pre_mix, ttnn.float32)
        total = None
        for j in range(hc):
            row = ttnn.slice(streams_f, [0, 0, j, 0], [batch, 1, j + 1, hidden])
            weight = ttnn.slice(weights, [0, 0, 0, j], [batch, 1, 1, j + 1])
            term = ttnn.multiply(row, weight)
            total = term if total is None else ttnn.add(total, term)
        return ttnn.typecast(total, ttnn.bfloat16)

    def _decode_submesh_static(
        self, sm: dict, pool_flags: dict[int, bool], causal: bool, index_sparse: bool = False
    ) -> ttnn.Tensor:
        """One submesh's layers for this variant. ``index_sparse`` is the r=2 select bit.

        Entering a stage receives streams, the packet, ``pre_mix`` and the shared KV groups, in
        that order. Leaving sends the same. The global-last layer collapses, normalises, applies
        ``lm_head`` and pushes the logits out the D2H socket.
        """
        r2_pool = bool(pool_flags.get(2, False))
        r2_select = bool(index_sparse)
        stage = sm["index"]
        ids = self.layer_submesh_ids
        streams = None
        pre_mix = None
        pkt = None
        ctx = None
        out = None
        persistent = {sm.get("pre_mix_init"), sm.get("pre_mix_in")}
        for li in sm["layers"]:
            layer = self.layers[li]
            is_first = li == 0
            is_last = li == self.num_layers - 1
            recv = li > 0 and ids[li - 1] != stage
            if self.use_prefetcher:
                # One layer ahead: a run's first layer is queued before its receive blocks,
                # and each layer queues its successor on this stage before decoding. Every ring
                # is a FIFO consumed in layer order, so a request may only follow its layer's
                # predecessors' -- which holds since each layer is queued exactly once, in order.
                if is_first or recv:
                    layer.prefetch_weights()
                if not is_last and ids[li + 1] == stage:
                    self.layers[li + 1].prefetch_weights()
            if is_first:
                pkt = sm["pkt"]
                ttnn.experimental.recv_async_h2d(pkt, self._pkt_socket)
                if self.tp_size > 1:
                    pkt = ttnn.broadcast(pkt, ttnn.MeshCoordinate(0, 0), cluster_axis=1, topology=ttnn.Topology.Ring)
            elif recv:
                _, receiver = self.submesh_socket_pairs[(ids[li - 1], stage)]
                ttnn.experimental.recv_direct_async(sm["streams_in"], receiver)
                ttnn.experimental.recv_direct_async(sm["pkt_in"], receiver)
                ttnn.experimental.recv_direct_async(sm["pre_mix_in"], receiver)
                _, recv_caches = self._cache_handoff[(ids[li - 1], stage)]
                for cache in recv_caches:
                    ttnn.experimental.recv_direct_async(cache, receiver)
                pkt = sm["pkt_in"]
                streams = ttnn.to_memory_config(sm["streams_in"], sm["streams_memory_config"])
                pre_mix = sm["pre_mix_in"]
            if ctx is None:
                ctx = self._step_context(sm, pkt, causal, r2_pool, r2_select)
            if is_first:
                with _region("EMBED"):
                    embedded = ttnn.embedding(
                        ctx["token"], self.embed_tokens.embedding_weight, layout=ttnn.ROW_MAJOR_LAYOUT
                    )
                hidden = embedded.shape[-1]
                streams = ttnn.repeat(
                    ttnn.reshape(embedded, [self._decode_batch, 1, 1, hidden]),
                    ttnn.Shape([1, 1, self.config.hc_mult, 1]),
                    memory_config=sm["streams_memory_config"],
                )
                pre_mix = sm["pre_mix_init"]
            streams, pre_mix = layer.decode(
                streams,
                pre_mix,
                sm["scaches"][li],
                **self._attention_inputs(layer.self_attn, ctx, causal, r2_pool, r2_select),
            )
            if is_last:
                hidden = self.norm(self._collapse_streams(streams, pre_mix))
                if self._lm_head_traced is not None:
                    hidden = self._lm_head_traced(hidden)
                out = hidden
                self._send_output(out)
            elif ids[li + 1] != stage:
                sender, _ = self.submesh_socket_pairs[(stage, ids[li + 1])]
                streams_rm = ttnn.to_memory_config(streams, ttnn.DRAM_MEMORY_CONFIG)
                pre_send = ttnn.to_memory_config(pre_mix, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.experimental.send_direct_async(streams_rm, sender)
                ttnn.experimental.send_direct_async(pkt, sender)
                ttnn.experimental.send_direct_async(pre_send, sender)
                send_caches, _ = self._cache_handoff[(stage, ids[li + 1])]
                for cache in send_caches:
                    ttnn.experimental.send_direct_async(cache, sender)
                streams.deallocate()
                streams_rm.deallocate()
                if pre_mix not in persistent and pre_mix is not pre_send:
                    pre_mix.deallocate()
                if pre_send not in persistent:
                    pre_send.deallocate()
        return out if out is not None else streams
