# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash decode model: V4's traced TP pipeline around V4.1 layers.

The deployment is TP4 across 8 stages (a 32-chip mesh opened as ``(8, 4)``). Each stage is a
1x4 submesh; with the profile's pipeline group size of 1 it owns one contiguous run of layers.
What V4 already does is reused: the H2D packet, on-device RoPE, D2D sockets, trace variants, the
replay thread and the DRISC prefetch rings. This file only replaces the parts V4.1 changed.

* Several sessions stay resident. Each owns a slot of every sliding ring and compressor window and a
  row of every compressed-KV page table, over block pools all sessions share. The packet
  ``[token | ring row | pos | slot | slot's ring base | slot's window base]`` names the step's slot,
  so switching sessions is free: no device work and no wait for the steps in flight.

* Layers exchange ``(streams [B, 1, hc, D], pre_mix [B, 1, 1, hc])``. The first ``pre_mix`` is the
  one-hot ``[1, 0, 0, 0]``; the last one collapses the streams (there is no ``hc_head``).
* Engram layers (1 and 14) take their rows on a per-rank H2D socket. The host writes them in
  :meth:`_write_packet`, before the step that receives them.
* A compressed group shares a tiled ``[ring | selected]`` buffer (``sel_tile``); stages with an
  index source of the group also keep ``sel_kv``, ``comp_kv`` and ``idx_key_cache``. A pipeline
  edge sends ``sel_tile`` of each group the two stages share and, when a stage without the
  group's kv source selects from it, the one compressed-KV row and index key the source wrote
  this step -- not the whole caches, which grow with the context.
* Past ``candidate_topk_blocks * candidate_block_size`` entries the select is two-level (see
  :mod:`.decode.attention`), a variant of its own. The candidate source's blocks and pool count
  cross the edges of its group, and the first consumer on each stage builds that stage's pool.

Prefill (:meth:`DeepSeekV41Model.build_prefill`) builds a
:class:`~.prefill.model.DeepSeekV41PrefillModel` on the decode stages, sharing the routed experts,
embedding, ``lm_head`` and Engram lookup, and :meth:`DeepSeekV41Model.commit_prefill_state` writes
its per-layer state into the decode caches. The prefill stages are the decode stages, so give the
decode model a contiguous ``layer_submesh_ids`` (e.g. ``plan_layer_placement(n, stages, 1)``).
"""

from __future__ import annotations

import collections
import math
import queue
import threading
from typing import Callable, Optional, Sequence

import torch
import ttnn
from loguru import logger

from models.experimental.deepseek_v41_flash.tt.config import (
    decoder_layer_weights,
    dequantized,
    expert_provider,
)
from models.experimental.deepseek_v41_flash.tt.decode.attention import (
    _SELECT_KV_RATE,
    DeepSeekV41LayerCache,
    candidate_buffers,
    uses_candidate_pool,
)
from models.experimental.deepseek_v41_flash.tt.decode.decoder_layer import DeepSeekV41DecoderLayer
from models.experimental.deepseek_v41_flash.tt.decode.engram import open_rows_sockets, rows_page_bytes
from models.experimental.deepseek_v41_flash.tt.decode.dspark import DeepSeekV41DSpark, host_to_device
from models.experimental.deepseek_v41_flash.tt.decode.moe import DeepSeekV41PreloadedExperts
from models.experimental.deepseek_v41_flash.tt.engram_lookup import (
    EngramHostLookup,
    engram_table_nbytes,
    load_engram_table,
)
from models.experimental.deepseek_v41_flash.tt.prefill.model import DeepSeekV41PrefillModel
from models.experimental.deepseek_v4_flash.tt.common import (
    _HIFI4,
    _MASK_NEG,
    _profile,
    _region,
    _trace_capture_guard,
    _traced_decode_enabled,
    width_sharded_l1_config,
)
from models.experimental.deepseek_v4_flash.tt.decode.attention import (
    CSA_INDEX_BLOCK_SIZE,
    _index_key_pool,
    _one_row_per_user,
    _update_cache_at,
)
from models.experimental.deepseek_v4_flash.tt.decode.hyperconnection import _HC_COLLAPSE_CORES
from models.experimental.deepseek_v4_flash.tt.decode.paged_cache import (
    ZERO_BLOCK,
    PagedCacheFull,
    PagedGroup,
    PagedKVManager,
    plan_pool_blocks,
)
from models.experimental.deepseek_v4_flash.tt.embedding import DeepSeekV4Embedding
from models.experimental.deepseek_v4_flash.tt.layers import DeepSeekV4RMSNorm
from models.experimental.deepseek_v4_flash.tt.model import (
    DeepSeekV4Model,
    _OUT_SOCKET_CORE,
    _PKT_SOCKET_CORE,
    plan_layer_placement,
)
from models.experimental.deepseek_v4_flash.tt.prefill.attention import ALIGNMENT, PrefillAttentionState
from models.experimental.deepseek_v4_flash.tt.system_config import load_system_config, set_active_system_config
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache, _as_cache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

# Host RAM kept free above the Engram tables, matching the decoder-layer test.
_ENGRAM_RAM_MARGIN = 1.25
_ROW_NAMES = ("new_kv_row", "new_key_row")
_CANDIDATE_NAMES = ("cand_blocks", "pool_count")
# Rows of a session slot in the shared compressor windows: one tile, so a slot starts on a tile row.
_WIN_SLOT_ROWS = ttnn.TILE_SIZE
# Tokens past the current position a session's blocks are grown for (when the pool can spare them), so
# the device page tables are rewritten every 16 ratio-1 blocks instead of on every block.
_GROW_AHEAD_TOKENS = 16 * CSA_INDEX_BLOCK_SIZE


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
    ``layer_submesh_ids`` (one non-decreasing stage index per built layer) overrides V4's even
    :func:`plan_layer_placement`, e.g. with the prefill placement when a prefill model shares the stages.

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
        layer_submesh_ids: Optional[Sequence[int]] = None,
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
        if layer_submesh_ids is None:
            self.layer_submesh_ids = plan_layer_placement(self.num_layers, self.num_submeshes, pipeline_group_size)
        else:
            self.layer_submesh_ids = list(layer_submesh_ids)
            if len(self.layer_submesh_ids) != n:
                raise ValueError(f"got {len(self.layer_submesh_ids)} layer placements for {n} layers")
            if self.layer_submesh_ids != sorted(self.layer_submesh_ids) or not all(
                0 <= s < self.num_submeshes for s in self.layer_submesh_ids
            ):
                raise ValueError(
                    f"layer placement {self.layer_submesh_ids} must be non-decreasing stage indices below "
                    f"{self.num_submeshes}"
                )
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
        engram_sockets = {
            li: open_rows_sockets(self.submeshes[self.layer_submesh_ids[li]], rows_page_bytes(config))
            for li in config.engram_layer_ids
            if li < n
        }
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
                    engram_sockets=engram_sockets.get(li),
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
        self._candidate_group = None
        self._candidate_layers = set()
        self._pool_builders = set()
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
        self._max_sessions = 1
        self._slot_of: dict[int, int] = {}
        self._paged_ratios: dict[int, PagedGroup] = {}
        self._packet_sids: collections.deque = collections.deque()
        self._packet_slot = 0
        # Held by whoever touches the pools' book-keeping or queues work on cq 0: the replay thread and
        # the posting thread both do.
        self._cq_lock = threading.RLock()
        self._decode_batch = 1
        self._cache_handoff: dict[tuple[int, int], tuple[list, list]] = {}
        self._spec_hist = None
        self.dspark: Optional[DeepSeekV41DSpark] = None

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

        Up to ``num_sessions`` sessions share them. The index-key and compressed-KV pools are paged in
        ``CSA_INDEX_BLOCK_SIZE``-row blocks, sized for ``total_tokens`` across all sessions; each session
        owns a slot of every sliding ring and compressor window.
        """
        return self._decode_max_seq is not None

    @property
    def _slotted(self) -> bool:
        """Whether rings and windows hold several session slots (read through the ``*_view`` copies)."""
        return self._max_sessions > 1

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
        rebuilds one row per step from them. ``block_size`` is accepted so the V4 demo call works; the
        index pools use :data:`CSA_INDEX_BLOCK_SIZE`.

        ``num_sessions`` sessions can be open at once, each up to ``max_seq`` tokens, sharing index-key and
        compressed-KV pools sized for ``total_tokens`` (default ``num_sessions * max_seq``, at least
        ``max_seq``). ``batch`` is 1: a step decodes one session. Switching sessions costs no device work:
        every session has its own ring and window slot and its own row of each page table, and the
        step's packet names the slot.
        """
        if batch != 1:
            raise NotImplementedError(f"V4.1 decode supports batch 1, got batch={batch}")
        if num_sessions < 1:
            raise ValueError(f"num_sessions must be at least 1, got {num_sessions}")
        if max_seq < 2:
            raise ValueError(f"max_seq must be at least 2 so RoPE can be recovered from row 1, got {max_seq}")
        total_tokens = num_sessions * max_seq if total_tokens is None else total_tokens
        if total_tokens < max_seq:
            raise ValueError(f"total_tokens ({total_tokens}) must hold one max_seq ({max_seq}) session")
        cfg = self.config
        ratios = {int(r) for r in cfg.compress_ratios[: self.num_layers] if int(r)}
        self._context_limit = None
        for ratio in ratios:
            if max_seq % ratio:
                raise ValueError(f"max_seq ({max_seq}) must be a multiple of compress ratio {ratio}")
        self._decode_batch = batch
        self._lm_head_traced = lm_head
        self._decode_max_seq = max_seq
        self._max_sessions = num_sessions
        # One page-table geometry per ratio: every group of a ratio, on every stage, maps a session's
        # entries the same way, so one allocation serves all of them.
        self._paged_ratios = {
            r: PagedGroup(
                layer_type=f"ratio {r}",
                block_size=CSA_INDEX_BLOCK_SIZE,
                sliding_window=0,
                max_seq=max_seq,
                compress_rate=r,
            )
            for r in sorted(ratios)
        }
        groups = {g.layer_type: g for g in self._paged_ratios.values()}
        self._pool_blocks = plan_pool_blocks(groups, num_sessions, total_tokens)
        self._paged = PagedKVManager(groups, self._pool_blocks)
        self._slot_of = {}
        self._session_pos = {}
        self._resident = []
        self._packet_sids.clear()
        # Variant axis 1 is the r=2 pool phase. Stages with no r=2 source ignore it.
        has_r2 = 2 in ratios
        self._pool_crs = [2] if has_r2 else []
        self._pool_phases = [(False,), (True,)] if has_r2 else [()]
        logger.info(
            f"V4.1 traced decode: {num_sessions} session(s) of up to {max_seq} tokens sharing {total_tokens}, "
            f"index blocks of {CSA_INDEX_BLOCK_SIZE} (pool blocks {self._pool_blocks}; prepare "
            f"block_size={block_size} is unused)"
        )

        rope_gen = {}
        for family in ("main", "compress"):
            cos_h, sin_h = rope[family]
            scaling = float(cos_h[0, 0].item())
            inv_freq = torch.atan2(sin_h[1].float(), cos_h[1].float()).repeat_interleave(2)
            rope_gen[family] = (inv_freq.reshape(1, 1, 1, -1), scaling)

        alignment = self.system_config.pipeline.pcie_alignment
        # [token | ring row | pos | slot | slot's first ring row | slot's first window row], see _build_packet.
        self._pkt_int_prefix = 6
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
        # Past the candidate pool, the candidate source's blocks reach every stage of its group and the
        # first consumer on each stage builds that stage's pool.
        source = cfg.candidate_source_layer_id
        self._candidate_group = None
        self._candidate_layers = set()
        self._pool_builders = set()
        if 0 <= source < self.num_layers and uses_candidate_pool(cfg, max_seq):
            self._candidate_group = self._kv_source_of[source]
            self._candidate_layers = {source}
            for li in range(source + 1, self.num_layers):
                if li not in cfg.index_source_layer_ids:
                    continue
                if self._kv_source_of[li] != self._candidate_group:
                    raise NotImplementedError(f"candidate consumer {li} reads another group than layer {source}")
                self._candidate_layers.add(li)
                if not any(ids[w] == ids[li] for w in self._pool_builders):
                    self._pool_builders.add(li)
            logger.info(
                f"two-level select past {cfg.candidate_topk_blocks * cfg.candidate_block_size} entries: "
                f"source {source}, pool builders {sorted(self._pool_builders)}"
            )
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
                "has_candidates": any(li in self._candidate_layers for li in layers_here),
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

        self._allocate_shared_caches()
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
            self._engram_lookup = EngramHostLookup(
                cfg, self.loader.snapshot_dir, tables, batch=batch, max_seq=max_seq, sessions=num_sessions
            )
            logger.info(f"Engram host lookup: layers {[layer_idx for layer_idx, _ in self._engram_layers]}")

    def _allocate_shared_caches(self) -> None:
        """Per-stage KV. A group is aliased inside a stage and handed over across a pipeline edge.

        ``win_kv`` / ``win_gate`` stay on the stage that owns the kv source. Every stage of the
        group gets ``sel_tile``; a stage with an index source of the group also gets ``sel_kv``,
        ``comp_kv``, ``idx_key_cache`` and the stage's page table of the group's ratio. Groups some
        stage selects from without holding their source carry ``new_kv_row`` / ``new_key_row`` on every
        stage. With a two-level select, every stage of the candidate source's group carries its
        ``cand_blocks`` / ``pool_count``, and a stage with a pool builder also gets the pool.

        With several sessions every ring is ``[1, 1, N * window, Dh]`` and every window
        ``[1, 1, N * 32, Dh]``, read through one ``ring_view`` / ``win_view_*`` per stage (see
        :class:`DeepSeekV41LayerCache`). ``comp_kv`` / ``idx_key_cache`` hold the ratio's
        ``plan_pool_blocks`` blocks, and ``sm["page_tables"][ratio]`` (INT32 ``[N, L]``) maps slot
        ``s``'s logical blocks to them; :meth:`_write_page_tables` refreshes it.
        """
        cfg = self.config
        window, head_dim = self.sliding_window, cfg.head_dim
        rows = window + cfg.index_topk
        slots = self._max_sessions
        slotted = self._slotted
        by_stage: dict[int, dict[int, dict]] = {}
        for sm in self.submeshes_io:
            stage = sm["index"]
            device = sm["device"]
            shared_here: dict[int, dict] = {}
            sm["page_tables"] = {}
            views = {}
            if slotted and sm["has_sliding"]:
                views["ring_view"] = _device_zeros((1, 1, window, head_dim), device)
            if slotted and any(
                li in cfg.kv_source_layer_ids and int(cfg.compress_ratios[li]) > 1 for li in sm["layers"]
            ):
                views["win_view_kv"] = _device_zeros((1, 1, _WIN_SLOT_ROWS, head_dim), device)
                views["win_view_gate"] = _device_zeros((1, 1, _WIN_SLOT_ROWS, head_dim), device)
            for li in sm["layers"]:
                ratio = int(cfg.compress_ratios[li])
                ring = _device_zeros((1, 1, slots * window, head_dim), device)
                if not ratio:
                    sm["scaches"][li] = DeepSeekV41LayerCache(kv=ring, ring_view=views.get("ring_view"))
                    continue
                src = self._kv_source_of[li]
                if src not in shared_here:
                    owns_window = src in sm["layers"] and ratio > 1
                    win_rows = slots * _WIN_SLOT_ROWS if slotted else ratio
                    group = {
                        "win_kv": _device_zeros((1, 1, win_rows, head_dim), device) if owns_window else None,
                        "win_gate": _device_zeros((1, 1, win_rows, head_dim), device) if owns_window else None,
                        "sel_tile": _device_zeros((1, 1, rows, head_dim), device),
                    }
                    if owns_window:
                        group["win_view_kv"] = views.get("win_view_kv")
                        group["win_view_gate"] = views.get("win_view_gate")
                    if src in self._row_groups:
                        group["new_kv_row"] = _device_zeros((1, 1, 1, head_dim), device, layout=ttnn.ROW_MAJOR_LAYOUT)
                        group["new_key_row"] = _device_zeros(
                            (1, 1, 1, cfg.index_head_dim), device, layout=ttnn.ROW_MAJOR_LAYOUT
                        )
                    selects_here = any(
                        w in cfg.index_source_layer_ids and self._kv_source_of.get(w) == src for w in sm["layers"]
                    )
                    if selects_here:
                        paged = self._paged_ratios[ratio]
                        blocks = self._pool_blocks[paged.layer_type]
                        if ratio not in sm["page_tables"]:
                            sm["page_tables"][ratio] = _device_zeros(
                                (slots, paged.logical_blocks), device, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
                            )
                        group.update(
                            idx_key_cache=_index_key_pool(device, blocks, CSA_INDEX_BLOCK_SIZE, cfg.index_head_dim),
                            comp_kv=_device_zeros(
                                (blocks, 1, CSA_INDEX_BLOCK_SIZE, head_dim), device, layout=ttnn.ROW_MAJOR_LAYOUT
                            ),
                            idx_page_table=sm["page_tables"][ratio],
                            sel_kv=_device_zeros((1, 1, rows, head_dim), device, layout=ttnn.ROW_MAJOR_LAYOUT),
                        )
                    if src == self._candidate_group:
                        builds_pool = any(w in self._pool_builders for w in sm["layers"])
                        group.update(candidate_buffers(device, cfg, with_pool=builds_pool))
                    shared_here[src] = group
                sm["scaches"][li] = DeepSeekV41LayerCache(kv=ring, **shared_here[src])
            by_stage[stage] = shared_here
        self._cache_handoff = {}
        for src_stage, dst_stage in self.pipeline_edges:
            common = sorted(set(by_stage[src_stage]) & set(by_stage[dst_stage]))
            send, recv = [], []
            for src in common:
                names = ("sel_tile",) + (_ROW_NAMES if src in self._row_groups else ())
                names += _CANDIDATE_NAMES if src == self._candidate_group else ()
                for name in names:
                    send.append(by_stage[src_stage][src][name])
                    recv.append(by_stage[dst_stage][src][name])
            self._cache_handoff[(src_stage, dst_stage)] = (send, recv)
            if common:
                logger.info(f"shared KV {src_stage} -> {dst_stage}: groups {common}")

    def _two_level(self, pos: int) -> bool:
        """Whether the candidate source's group has outgrown the candidate pool at ``pos``."""
        if self._candidate_group is None:
            return False
        ratio = int(self.config.compress_ratios[self._candidate_group])
        return (pos + 1) // ratio > self.config.candidate_topk_blocks * self.config.candidate_block_size

    def _variant_key(self, pos: int) -> tuple[bool, int, int]:
        """``(causal, r=2 pool phase, select mode)`` for absolute ``pos``.

        Causal starts at ``sliding_window - 1`` when the profile asks for it. An r=2 group pools on
        odd positions (``(pos + 1) % 2 == 0``) and selects once ``pos >= 1``. Ratio 1 pools and
        selects on every step, so it is not a variant bit. The select mode is 0 before the r=2
        select, 1 after it and 2 once the select is two-level (:meth:`_two_level`), which V4's
        code reads as the truthy indexer bit.
        """
        causal = self._SDPA_CAUSAL and pos >= self.sliding_window - 1
        if self._two_level(pos):
            return (causal, int(bool(self._pool_crs) and (pos + 1) % 2 == 0), 2)
        if not self._pool_crs:
            return (causal, 0, 0)
        return (causal, int((pos + 1) % 2 == 0), int(pos >= 1))

    def _reachable_variants(self) -> list[tuple[bool, int, int]]:
        """The variants a position can ask for. ``(False, 1, 0)`` and ``(True, *, 0)`` never occur: the
        only non-select step is position 0. Two-level steps are past the sliding window, so causal when
        the profile is."""
        if not self._pool_crs:
            modes = [False, True] if self._SDPA_CAUSAL else [False]
            variants = [(causal, 0, 0) for causal in modes]
        else:
            variants = [(False, 0, 0), (False, 0, 1), (False, 1, 1)]
            if self._SDPA_CAUSAL:
                variants += [(True, 0, 1), (True, 1, 1)]
        if self._candidate_group is not None:
            variants += [(self._SDPA_CAUSAL, phase, 2) for phase in range(len(self._pool_phases))]
        return variants

    @staticmethod
    def _sm_pool_key(sm: dict, pool_flags: dict, causal: bool, index_sparse: int = 0) -> tuple:
        """Drop variant bits this submesh's layers do not branch on.

        ``index_sparse`` is the select mode of :meth:`_variant_key`. A sliding-only stage collapses to
        one trace; a ratio-1 stage keeps only the causal bit, and the two-level bit if it holds the
        candidate source or a consumer.
        """
        return (
            causal and sm["has_compressed"],
            bool(pool_flags.get(2, False)) and sm["has_r2_pool"],
            bool(index_sparse) and sm["has_r2_select"],
            index_sparse == 2 and sm["has_candidates"],
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

    def _build_packet(self, token_id, pos: int) -> torch.Tensor:
        """One INT32 socket page ``[1, 1, 1, _pkt_w]``: ``[token | slot * W + pos % W | pos | slot |
        slot * W | slot * 32]`` for session slot ``_packet_slot``, then padding.

        The slot fields are what makes a session switch free: the traces read the slot's ring rows,
        window rows and page-table row from here, so nothing on device changes between sessions.
        """
        w, slot = self.sliding_window, self._packet_slot
        packet = torch.zeros(1, 1, 1, self._pkt_w, dtype=torch.int32)
        packet[0, 0, 0, : self._pkt_int_prefix] = torch.tensor(
            self._step_tokens(token_id) + [slot * w + pos % w, pos, slot, slot * w, slot * _WIN_SLOT_ROWS],
            dtype=torch.int32,
        )
        return packet

    def _write_packet(self, token_id, pos: int, sid: Optional[int] = None) -> None:
        """The packet of session ``sid`` (the resident one when ``None``, as in the compile runs), then each
        Engram layer's rows in layer order. Compile runs push zeros and do not advance the hash.

        The packet goes first so layer 0 can start while the host looks up the first Engram layer's rows;
        each later layer's lookup runs only after the earlier layer's rows are on their socket.
        """
        if sid is None:
            sid = self._resident[0] if self._resident and self._resident[0] is not None else None
        self._packet_slot = self._slot_of.get(sid, 0)
        DeepSeekV4Model._write_packet(self, token_id, pos)
        self._write_engram_rows(token_id, pos, self._packet_slot)

    def _write_engram_rows(self, token_id, pos: int, slot: int = 0) -> None:
        if not self._engram_layers:
            return
        if self._eager_decode or self._traces_compiled:
            ids = torch.tensor(self._step_tokens(token_id), dtype=torch.long).reshape(-1, 1)
            lookup = self._engram_lookup
            hashes = lookup.hash_session(slot, ids, pos)  # [B, 1, n_engram_layers, n_hash_cols]
            for layer_idx, engram in self._engram_layers:
                ids_of_layer = hashes[:, :, lookup.layout.layer_ids.index(layer_idx)]
                engram.write_rows(lookup.gather(layer_idx, ids_of_layer).flatten(-2))
        else:
            for _, engram in self._engram_layers:
                engram.write_rows(torch.zeros(engram.batch, engram.kin, dtype=torch.bfloat16))

    # -- sessions: slots of the rings and windows, rows of the shared block pools' page tables -- #
    def open_session(self) -> int:
        """Claim the lowest free session slot and return the session's id.

        Host-only. The slot's ring and window rows are the session's own, and its compressed blocks are
        handed out as its windows close (:meth:`ensure_session_capacity`). Nothing needs zeroing: as
        :meth:`reset_session` explains, a sequence reads only rows it wrote itself.
        """
        paged = self._require_paged()
        free = sorted(set(range(self._max_sessions)) - set(self._slot_of.values()))
        if not free:
            raise PagedCacheFull(f"all {self._max_sessions} session slots are in use")
        sid = paged.open_session()
        self._slot_of[sid] = free[0]
        self._session_pos[sid] = 0
        return sid

    def close_session(self, sid: int) -> None:
        """Return session ``sid``'s blocks to the pools and free its slot."""
        paged = self._require_paged()
        if sid not in self._slot_of:
            raise KeyError(f"no such session {sid}")
        self._wait_replays_dispatched()
        if sid in self._resident:
            self._resident[self._resident.index(sid)] = None
        paged.close_session(sid)
        del self._slot_of[sid]
        self._session_pos.pop(sid, None)
        self._write_page_tables()

    def activate_session(self, sid: int) -> None:
        self.activate_sessions([sid])

    def activate_sessions(self, sids) -> None:
        """Make ``sids`` (one session) the one the next posted steps and the next prefill belong to.

        Host-only and free: the device is not touched and nothing waits for the steps in flight. Each
        posted step carries its session (see :meth:`replay_traced`), and its packet names that session's
        slot. The Engram lookup switches to the session's token history for the prefill.
        """
        sids = list(sids)
        if len(sids) != self._decode_batch:
            raise ValueError(f"batch decodes {self._decode_batch} users at a time, got {len(sids)} sessions")
        for sid in sids:
            if sid not in self._slot_of:
                raise KeyError(f"no such session {sid}")
        self._resident = sids
        if self._engram_lookup is not None:
            self._engram_lookup.use_session(self._slot_of[sids[0]])

    def ensure_session_capacity(self, pos: int, sid: Optional[int] = None) -> None:
        """Give session ``sid`` (the resident one when ``None``) blocks for every entry a step at ``pos``
        writes, growing ahead by up to :data:`_GROW_AHEAD_TOKENS` when the pools can spare it, and
        refresh the page tables whose rows changed.

        Raises :class:`PagedCacheFull` past ``max_seq`` or when a pool runs out.
        """
        paged = self._require_paged()
        if sid is None:
            if not self._resident or None in self._resident:
                raise RuntimeError("call activate_sessions() before decoding")
            sid = self._resident[0]
        if pos >= self._decode_max_seq:
            raise PagedCacheFull(f"position {pos} is past max_seq {self._decode_max_seq}")
        with self._cq_lock:
            ahead = min(pos + _GROW_AHEAD_TOKENS, self._decode_max_seq - 1)
            try:
                changed = paged.ensure_capacity(sid, ahead)
            except PagedCacheFull:
                if ahead == pos:
                    raise
                # The failed attempt may have grown some groups before running out.
                paged.ensure_capacity(sid, pos)
                changed = [g.layer_type for g in self._paged_ratios.values()]
            if changed:
                self._write_page_tables(changed)

    def _write_page_tables(self, groups=None) -> None:
        """Copy every open session's page-table row into slot ``slot``'s row of the INT32 ``[N, L]``
        device tables of ``groups`` (all when ``None``), on every stage that has one. Free slots map to
        ``ZERO_BLOCK``.

        Rows of other sessions are rewritten with their unchanged contents. The write is queued on the
        command queue the traces run on, so it lands between steps: after every step dispatched before
        it and before every step dispatched after it.
        """
        paged = self._require_paged()
        with self._cq_lock:
            for ratio, group in self._paged_ratios.items():
                if groups is not None and group.layer_type not in groups:
                    continue
                table = torch.full((self._max_sessions, group.logical_blocks), ZERO_BLOCK, dtype=torch.int32)
                for sid, slot in self._slot_of.items():
                    table[slot] = paged.page_row(sid, group.layer_type)[0]
                host = ttnn.from_torch(table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
                for sm in self.submeshes_io:
                    device_table = sm["page_tables"].get(ratio)
                    if device_table is not None:
                        ttnn.copy_host_to_device_tensor(host, device_table)

    def session_usage(self) -> dict:
        """``"ratio r" -> (blocks in use, blocks in pool)`` of the shared ``comp_kv`` / ``idx_key_cache``
        pools of each ratio (the zero block counts as used)."""
        return self._require_paged().usage()

    def session_tokens_left(self) -> int:
        """Tokens the tightest ratio's pool can still admit across all open sessions."""
        return self._require_paged().tokens_left()

    def reset_static_caches(self) -> None:
        """No-op: as :meth:`reset_session` explains, a fresh sequence reads only rows it wrote itself."""
        return

    def reset_session(self, sid: int) -> None:
        """Rewind session ``sid`` to position 0 and return its compressed blocks to the pools.

        Nothing on device needs clearing: every row a step reads was written earlier in the same
        sequence (the ring is masked below ``sliding_window``, a select scores only the groups closed
        so far, a ratio-2 window slot is written before its group pools, and the Engram history is
        rewritten from position 0).
        """
        paged = self._require_paged()
        if sid not in self._slot_of:
            raise KeyError(f"no such session {sid}")
        self._wait_replays_dispatched()
        paged.reset_session(sid)
        self._session_pos[sid] = 0
        self._write_page_tables()

    def truncate_session(self, sid: int, pos: int) -> None:
        """Rewind session ``sid`` to position ``pos`` (at most its current one), keeping the compressed entries
        of the positions before ``pos`` and returning the blocks past them to the pools.

        The ring is left stale: it holds the window before the old position. Only a prefill that rebuilds the
        window (:meth:`seed_prefill` with ``tails=False``, then :meth:`commit_prefill_state` from ``pos``) may
        follow, not a decode step."""
        paged = self._require_paged()
        if sid not in self._slot_of:
            raise KeyError(f"no such session {sid}")
        if not 0 <= pos <= self._session_pos[sid]:
            raise ValueError(f"cannot truncate session {sid} at {self._session_pos[sid]} to {pos}")
        self._wait_replays_dispatched()
        paged.truncate_session(sid, pos)
        self._session_pos[sid] = pos
        self._write_page_tables()

    # -- replay: every posted step is bound to the session resident when it was posted -- #
    def replay_traced(self, pos: int) -> None:
        """V4's :meth:`~DeepSeekV4Model.replay_traced` for the resident session, which the step keeps:
        its packet (:meth:`write_step_packet`) and its capacity / position book-keeping
        (:meth:`_execute_traces`) are for that session even if another is activated meanwhile."""
        if not self._resident or None in self._resident:
            raise RuntimeError("call activate_sessions() before decoding")
        sid = self._resident[0]
        if self._eager_decode:
            self._packet_sids.append(sid)
            self._eager_pending.append((pos, sid))
            return
        if not self._traced_captured:
            raise RuntimeError("call decode_traced() once to capture the traces before replay_traced()")
        self._ensure_replay_thread()
        self._packet_sids.append(sid)
        self._replay_queue.put((pos, sid))

    def write_step_packet(self, token_id, pos: int) -> None:
        """Push the packet of the oldest posted step, in its session's slot (eager: then run it)."""
        sid = self._packet_sids.popleft()
        self._write_packet(token_id, pos, sid)
        if self._eager_decode:
            posted = self._eager_pending.popleft()
            assert posted == (pos, sid), f"eager decode: packet for {(pos, sid)}, but {posted} was posted next"
            self._execute_traces(posted)

    def _execute_traces(self, step) -> None:
        """Replay the variant of ``step`` = ``(pos, sid)`` on every submesh, after growing that session's
        blocks for it (normally already done when it was posted)."""
        pos, sid = step
        self.ensure_session_capacity(pos, sid)
        self._session_pos[sid] = pos + 1
        variant = self._variant_key(pos)
        if self._eager_decode:
            self._run_eager_step(variant)
            return
        with self._cq_lock:
            for sm in self.submeshes_io:
                ttnn.execute_trace(sm["device"], sm["tids"][variant], cq_id=0, blocking=False)

    def _step_context(self, sm: dict, pkt: ttnn.Tensor, causal: bool, r2_pool: bool, r2_select: bool) -> dict:
        """Token, positions, RoPE rows and the per-ratio mask or ``cur_pos`` for one step.

        Pool and select tensors are built only for the ratios this variant actually runs, so a
        stage that ignores the r=2 bits has one op sequence across those variants.
        """
        b = self._decode_batch

        def field(i: int) -> ttnn.Tensor:  # packet int i as INT32 ROW_MAJOR [1]
            return ttnn.reshape(ttnn.slice(pkt, [0, 0, 0, i], [1, 1, 1, i + 1]), [1])

        token = ttnn.typecast(ttnn.reshape(ttnn.slice(pkt, [0, 0, 0, 0], [1, 1, 1, b]), [1, b]), ttnn.uint32)
        sliding_pos = field(1)
        compress_pos = field(2)
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
            "page_row": None,
            "ring_offset": None,
            "win_offset": None,
        }
        win_base_f = None
        if self._slotted:
            ctx["page_row"], ctx["ring_offset"], ctx["win_offset"] = field(3), field(4), field(5)
            if any(r > 1 for r in sm["kv_ratios"]):
                win_base_f = ttnn.typecast(
                    ttnn.to_layout(ttnn.reshape(ctx["win_offset"], [1, 1, 1, 1]), ttnn.TILE_LAYOUT), ttnn.float32
                )
        if sm["has_sliding"]:
            ctx["sliding_curpos"] = self._device_index(ttnn.clamp(pos_f, max=float(self.sliding_window - 1)))
        for ratio in sm["ratios"]:
            if causal:
                ctx["curpos"][ratio] = self._compressed_curpos(pos_f, ratio)
            else:
                ctx["mask"][ratio] = self._compressed_mask(sm["mask_idx"], pos_f, ratio)
        for ratio in sm["kv_ratios"]:
            if ratio > 1:
                ctx["win_slot"][ratio] = self._win_slot(pos_f, ratio, base_f=win_base_f)
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
        if sm.get("hist_layers"):
            ctx["hist_slot"] = self._win_slot(pos_f, self._spec_hist)
        if sm.get("taps"):
            ctx["stash_slot"] = self._win_slot(pos_f, self.dspark.stash_rows)
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

    def _win_slot(self, pos_f: ttnn.Tensor, ratio: int, base_f: Optional[ttnn.Tensor] = None) -> ttnn.Tensor:
        """``pos % ratio``, plus the FP32 ``[1, 1, 1, 1]`` ``base_f`` row when given."""
        quot = ttnn.floor(ttnn.multiply(pos_f, 1.0 / ratio))
        slot = ttnn.subtract(pos_f, ttnn.multiply(quot, float(ratio)))
        if base_f is not None:
            slot = ttnn.add(slot, base_f)
        return self._device_index(slot)

    def _index_position(self, pos_f: ttnn.Tensor, ratio: int) -> ttnn.Tensor:
        """``4 * n - 1``: the select op counts ``(cur_pos + 1) // 4`` keys."""
        n = self._group_count(pos_f, ratio)
        return self._device_index(ttnn.subtract(ttnn.multiply(n, float(_SELECT_KV_RATE)), 1.0))

    def _attention_inputs(
        self, attn, ctx: dict, causal: bool, r2_pool: bool, r2_select: bool, two_level: bool = False
    ) -> dict:
        ratio = attn.compress_ratio
        cos, sin, neg_sin = ctx["rope"]["main" if ratio == 0 else "compress"]
        inputs = {
            "cos": cos,
            "sin": sin,
            "neg_sin": neg_sin,
            "sliding_pos": ctx["sliding_pos"],
            "ring_offset": ctx["ring_offset"],
        }
        if ratio == 0:
            inputs["sdpa_cur_pos"] = ctx["sliding_curpos"]
            return inputs
        inputs["page_row"] = ctx["page_row"]
        if attn.is_kv_source:
            pool = ratio == 1 or (ratio == 2 and r2_pool)
            inputs["pool"] = pool
            if ratio > 1:
                inputs["win_slot"] = ctx["win_slot"][ratio]
                inputs["hist_slot"] = ctx.get("hist_slot")
                inputs["win_offset"] = ctx["win_offset"]
            if pool:
                inputs["cos_win"], inputs["sin_win"] = ctx["win_rope"][ratio]
                inputs["entry"] = ctx["entry"][ratio]
        if attn.is_index_source and (ratio == 1 or (ratio == 2 and r2_select)):
            inputs["select"] = True
            inputs["index_pos"] = ctx["index_pos"][ratio]
            if attn.layer_idx in self._row_writers and (ratio == 1 or r2_pool):
                inputs["received_entry"] = ctx["entry"][ratio]
            if two_level and attn.layer_idx in self._candidate_layers:
                inputs["candidates"] = True
                inputs["build_pool"] = attn.layer_idx in self._pool_builders
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
        self, sm: dict, pool_flags: dict[int, bool], causal: bool, index_sparse: int = 0
    ) -> ttnn.Tensor:
        """One submesh's layers for this variant. ``index_sparse`` is the select mode of
        :meth:`_variant_key`.

        Entering a stage receives streams, the packet, ``pre_mix`` and the shared KV groups, in
        that order. Leaving sends the same. The global-last layer collapses, normalises, applies
        ``lm_head`` and pushes the logits out the D2H socket.
        """
        r2_pool = bool(pool_flags.get(2, False))
        r2_select = bool(index_sparse)
        two_level = index_sparse == 2
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
            if li in sm.get("taps", {}):
                row = self._collapse_streams(streams, sm["tap_weights"])
                b, hidden = row.shape[0], row.shape[-1]
                row = ttnn.reshape(ttnn.to_layout(row, ttnn.ROW_MAJOR_LAYOUT), [1, 1, b, hidden])
                for j, piece in enumerate(sm["taps"][li]):
                    w = piece.shape[-1]
                    part = ttnn.slice(row, [0, 0, 0, j * w], [1, 1, b, (j + 1) * w])
                    _update_cache_at(piece, _one_row_per_user(part), ctx["stash_slot"])
            streams, pre_mix = layer.decode(
                streams,
                pre_mix,
                sm["scaches"][li],
                **self._attention_inputs(layer.self_attn, ctx, causal, r2_pool, r2_select, two_level),
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

    # -- prefill: the decode stages, experts and Engram lookup; the decode caches as its state -- #
    def build_prefill(self, rope: dict, lm_head=None, **kwargs) -> DeepSeekV41PrefillModel:
        """Build the :class:`DeepSeekV41PrefillModel` on this model's stages.

        Every prefill layer lives on its decode layer's submesh and reads that layer's routed experts in
        place. The embedding table, ``lm_head`` (when given), the D2D socket pairs and the Engram host
        lookup are shared too, so a prefill and the decode steps after it hash one token history.
        ``kwargs`` go to the prefill constructor.

        Call it after :meth:`prepare_static_decode` (which builds the Engram lookup) and before the decode
        traces are captured.
        """
        if self._decode_max_seq is None:
            raise RuntimeError("call prepare_static_decode() before build_prefill()")
        self.prefill_model = DeepSeekV41PrefillModel(
            self.config,
            self.loader,
            self.first_device,
            rope,
            num_layers=self.num_layers,
            tp_size=self.tp_size,
            layer_devices=self.layer_devices,
            engram_lookup=self._engram_lookup,
            experts=[layer.mlp.experts for layer in self.layers],
            embedding_weight=self.embed_tokens.embedding_weight,
            lm_head=lm_head,
            **kwargs,
        )
        # Each receiver FIFO is permanent L1 on its stage, so a second set would cost prefill's ops that room.
        self.prefill_model.shared_socket_pairs = {
            (id(self.submeshes[from_id]), id(self.submeshes[to_id])): pair
            for (from_id, to_id), pair in self.submesh_socket_pairs.items()
        }
        # Same for the Engram rows FIFOs (one per device of the stage, ~100 KB of a TP4 stage's L1 per layer).
        self.prefill_model.shared_engram_sockets = {
            li: layer.engram.sockets for li, layer in enumerate(self.layers) if layer.engram is not None
        }
        self.prefill_model.keep_decode_prefetcher = bool(self._prefetch_buffers_by_device)
        return self.prefill_model

    def commit_prefill_state(
        self,
        states: list[PrefillAttentionState],
        session_id: int,
        progress: Optional[Callable[[str], None]] = None,
        bias_slots: Optional[dict] = None,
        start: int = 0,
    ) -> int:
        """Write a prefill's per-layer ``states`` into the decode caches of ``session_id``; returns ``T``.

        ``bias_slots`` is accepted so V4's callers work and must be ``None``: V4.1 has no CSA position bias.

        ``start > 0`` commits a prefill continued from the session's current position ``start`` (seeded by
        :meth:`seed_prefill`): the rings are rewritten whole, the paged pools only from the block holding entry
        ``start // rate`` on, and the session is not rewound. A bounded replay may end at ``start`` itself, and
        then rewrites only the rings.

        For a prompt of ``T`` tokens (a multiple of ``sliding_window``, so ring slot ``j`` holds token
        ``T - W + j``, and of every compress ratio, so no compressor window is half full):

        * every layer: ``kv_tail`` -> the session's slot of its ring ``kv``;
        * every stage holding a group's ``comp_kv`` (it has an index source of the group): the kv source's
          ``compressed_kv`` -> ``comp_kv`` and ``idx_keys`` -> ``idx_key_cache``, each logical block into the
          physical block the session's page-table row maps it to. A source on another stage is copied over
          through the host.

        ``win_kv`` / ``win_gate``, ``sel_kv`` / ``sel_tile`` and the carried rows need nothing: the next step
        writes them before it reads them. The caches are written in place, so captured traces stay valid.
        """
        note = progress or (lambda message: None)
        if bias_slots:
            raise ValueError("V4.1 has no CSA position bias: commit_prefill_state takes no bias_slots")
        if not self.paged:
            raise RuntimeError("call prepare_static_decode() first")
        if session_id not in self._slot_of:
            raise KeyError(f"no such session {session_id}")
        if len(states) < self.num_layers:
            raise ValueError(f"need {self.num_layers} layer states, got {len(states)}")
        lengths = {state.seq_len for state in states[: self.num_layers]}
        if len(lengths) != 1:
            raise ValueError(f"layer states disagree on the prompt length: {sorted(lengths)}")
        total = lengths.pop()
        if total <= 0 or total % ALIGNMENT or total % self.sliding_window:
            raise ValueError(
                f"prefilled length {total} must be a positive multiple of {ALIGNMENT} and of the "
                f"{self.sliding_window}-token sliding window"
            )
        if start and (start != self._session_pos[session_id] or start > total):
            raise ValueError(
                f"a prefill continued from {start} up to {total} does not continue session {session_id} "
                f"at position {self._session_pos[session_id]}"
            )
        self.ensure_session_capacity(total - 1, session_id)
        ring_row = self._slot_of[session_id] * self.sliding_window
        for sm in self.submeshes_io:
            device = sm["device"]
            committed = set()
            for li in sm["layers"]:
                state = states[li]
                if list(state.kv_tail.device().get_device_ids()) != list(device.get_device_ids()):
                    raise RuntimeError(f"prefill layer {li}'s state is not on its decode submesh")
                scache = sm["scaches"][li]
                note(f"commit layer {li + 1}/{self.num_layers} on decode submesh {sm['index']}")
                ring = ttnn.to_layout(state.kv_tail, ttnn.TILE_LAYOUT)
                ttnn.fill_cache(scache.kv, ring, 0, update_idx=ring_row)
                ttnn.deallocate(ring)
                src = self._kv_source_of.get(li)
                if src is None or src in committed or scache.comp_kv is None:
                    continue
                committed.add(src)
                source = states[src]
                if source.num_entries != total // self.config.compress_ratios[src]:
                    raise ValueError(f"kv source {src} holds {source.num_entries} entries for {total} tokens")
                entries = self._on_stage(source.compressed_kv, device)
                keys = self._on_stage(source.idx_keys, device)
                ratio = int(self.config.compress_ratios[src])
                self._commit_paged_entries(
                    scache, self._paged_ratios[ratio].layer_type, session_id, entries, keys, start // ratio
                )
                for copy, original in ((entries, source.compressed_kv), (keys, source.idx_keys)):
                    if copy is not original:
                        ttnn.deallocate(copy)
        for sm in self.submeshes_io:
            ttnn.synchronize_device(sm["device"])
        self._session_pos[session_id] = total
        note("prefill state committed")
        return total

    def _commit_paged_entries(
        self,
        scache,
        group: str,
        session_id: int,
        entries: ttnn.Tensor,
        keys: ttnn.Tensor,
        first_entry: int = 0,
    ) -> None:
        """Compressed entries (ROW_MAJOR ``[1, 1, n, Dh]``) into ``scache.comp_kv`` and their index keys
        (ROW_MAJOR ``[1, 1, n, index_head_dim]``) into ``scache.idx_key_cache``, entry ``w`` at row
        ``w % 64`` of the physical block session ``session_id``'s row of ``group`` maps logical block
        ``w // 64`` to. Blocks before the one holding ``first_entry`` are left as they are.

        ``comp_kv`` is ROW_MAJOR, which ``fill_cache`` does not take, so each run of consecutive physical
        blocks is one ``slice_write``; the tiled keys go one block per ``fill_cache``.
        """
        n = keys.shape[2]
        if entries.shape[2] != n:
            raise ValueError(f"{n} index keys for {entries.shape[2]} compressed entries")
        block = CSA_INDEX_BLOCK_SIZE
        blocks = -(-n // block)
        lo = first_entry // block
        if lo >= blocks:
            return
        page_row = self._require_paged().page_row(session_id, group)[0].tolist()
        if ZERO_BLOCK in page_row[lo:blocks]:
            raise RuntimeError(f"{group}: {n} entries reach an unmapped block; capacity was not ensured")
        device = scache.idx_key_cache.device()
        dh, key_dim = entries.shape[3], keys.shape[3]
        if lo:
            entries = ttnn.slice(entries, [0, 0, lo * block, 0], [1, 1, n, dh])
            keys = ttnn.slice(keys, [0, 0, lo * block, 0], [1, 1, n, key_dim])

        stacked = self._stack_rows([entries], (blocks - lo) * block, dh, device)
        kv_blocks = ttnn.reshape(stacked, [blocks - lo, 1, block, dh])
        start = lo
        while start < blocks:
            end = start + 1
            while end < blocks and page_row[end] == page_row[end - 1] + 1:
                end += 1
            first = page_row[start]
            run = ttnn.slice(kv_blocks, [start - lo, 0, 0, 0], [end - lo, 1, block, dh])
            ttnn.experimental.slice_write(
                run, scache.comp_kv, [first, 0, 0, 0], [first + end - start, 1, block, dh], [1, 1, 1, 1]
            )
            if run.buffer_address() != kv_blocks.buffer_address():  # a slice of every block is kv_blocks itself
                ttnn.deallocate(run)
            start = end
        ttnn.deallocate(stacked)

        keys_rm = self._stack_rows([keys], (blocks - lo) * block, key_dim, device)
        keys_tiled = ttnn.to_layout(keys_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(keys_rm)
        for logical in range(lo, blocks):
            rows = ttnn.slice(
                keys_tiled, [0, 0, (logical - lo) * block, 0], [1, 1, (logical - lo + 1) * block, key_dim]
            )
            ttnn.fill_cache(scache.idx_key_cache, rows, page_row[logical])
            if rows.buffer_address() != keys_tiled.buffer_address():  # a slice of the one block is keys_tiled
                ttnn.deallocate(rows)
        ttnn.deallocate(keys_tiled)
        if lo:
            ttnn.deallocate(entries)
            ttnn.deallocate(keys)

    def _gather_paged_entries(self, scache, group: str, session_id: int, n: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """The inverse of :meth:`_commit_paged_entries`: session ``session_id``'s first ``n`` compressed entries
        and index keys out of ``scache.comp_kv`` / ``scache.idx_key_cache``, as new bf16 ROW_MAJOR
        ``[1, 1, n, Dh]`` / ``[1, 1, n, index_head_dim]`` tensors (one slice per run of consecutive blocks)."""
        block = CSA_INDEX_BLOCK_SIZE
        blocks = -(-n // block)
        page_row = self._require_paged().page_row(session_id, group)[0].tolist()
        if ZERO_BLOCK in page_row[:blocks]:
            raise RuntimeError(f"{group}: session {session_id}'s first {n} entries reach an unmapped block")
        dh, key_dim = scache.comp_kv.shape[3], scache.idx_key_cache.shape[3]
        kv_runs, key_runs = [], []
        start = 0
        while start < blocks:
            end = start + 1
            while end < blocks and page_row[end] == page_row[end - 1] + 1:
                end += 1
            first, count = page_row[start], end - start
            kv = ttnn.slice(scache.comp_kv, [first, 0, 0, 0], [first + count, 1, block, dh])
            kv_runs.append(ttnn.reshape(kv, [1, 1, count * block, dh]))
            keys = ttnn.slice(scache.idx_key_cache, [first, 0, 0, 0], [first + count, 1, block, key_dim])
            keys_rm = ttnn.to_layout(keys, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(keys)
            key_runs.append(ttnn.reshape(keys_rm, [1, 1, count * block, key_dim]))
            start = end
        out = []
        for runs, width in ((kv_runs, dh), (key_runs, key_dim)):
            whole = runs[0] if len(runs) == 1 else ttnn.concat(runs, dim=2)
            out.append(ttnn.slice(whole, [0, 0, 0, 0], [1, 1, n, width]))
            # A slice of all of ``whole`` (n a multiple of the block) is ``whole`` itself.
            kept = out[-1].buffer_address()
            for run in {r.buffer_address(): r for r in (*runs, whole)}.values():
                if run.buffer_address() != kept:
                    ttnn.deallocate(run)
        return out[0], out[1]

    def seed_prefill(self, traced, session_id: int, tails: bool = True) -> None:
        """Load session ``session_id``'s decode state at its position ``P`` into the buffers of ``traced`` (a
        :class:`~.prefill.model.DeepSeekV41TracedPrefill` built by :meth:`build_prefill`), so its ``run`` can
        continue the session from ``P``: pass ``start=P, seed=lambda t: model.seed_prefill(t, sid)``.

        ``P`` must be a positive multiple of ``sliding_window``: ring slot ``j`` then holds token ``P - W + j``,
        which is the prefill ``tail`` as it stands after ``P`` tokens. A kv source's first ``P // rate``
        entries and index keys are gathered out of the paged pools of its stage into its ``entries`` /
        ``idx_keys``, and into each prefill stage's ``mirrors`` of them: from that stage's own pools when its
        decode stage holds them, else copied over through the host. Reads only; the caches stay as they are.

        ``tails=False`` skips the rings, for a bounded replay (``run(..., bounded_replay=True)`` from ``P - W``),
        which rebuilds the window itself and so also follows a :meth:`truncate_session`.
        """
        if session_id not in self._slot_of:
            raise KeyError(f"no such session {session_id}")
        pos, w = self._session_pos[session_id], self.sliding_window
        if pos <= 0 or pos % w:
            raise ValueError(f"session {session_id} is at {pos}: a prefill continues from multiples of {w}")
        self._wait_replays_dispatched()
        ring_row = self._slot_of[session_id] * w

        def write_rows(dst: ttnn.Tensor, rows: ttnn.Tensor) -> None:
            end = [1, 1, rows.shape[2], dst.shape[3]]
            ttnn.experimental.slice_write(rows, dst, [0, 0, 0, 0], end, [1, 1, 1, 1])
            ttnn.deallocate(rows)

        def pools_of(sm: dict, src: int):
            return next(
                (
                    sm["scaches"][li]
                    for li in sm["layers"]
                    if self._kv_source_of.get(li) == src and sm["scaches"][li].comp_kv is not None
                ),
                None,
            )

        def entries_of(sm: dict, src: int):
            ratio = int(self.config.compress_ratios[src])
            group = self._paged_ratios[ratio].layer_type
            return self._gather_paged_entries(pools_of(sm, src), group, session_id, pos // ratio)

        sources = {}  # kv source -> its (entries, keys) on its own stage
        for sm in self.submeshes_io:
            for li in sm["layers"]:
                if tails:
                    kv = sm["scaches"][li].kv
                    ring = ttnn.slice(kv, [0, 0, ring_row, 0], [1, 1, ring_row + w, kv.shape[3]])
                    tail = ttnn.to_layout(ring, ttnn.ROW_MAJOR_LAYOUT)
                    if ring.buffer_address() != kv.buffer_address():  # a slice of the whole ring is kv itself
                        ttnn.deallocate(ring)
                    write_rows(traced.buffers[li].tail, tail)
                if li in self.config.kv_source_layer_ids:
                    entries, keys = entries_of(sm, li)
                    sources[li] = (entries, keys)
                    bufs = traced.buffers[li]
                    for dst, rows in ((bufs.entries, entries), (bufs.idx_keys, keys)):
                        ttnn.experimental.slice_write(
                            rows, dst, [0, 0, 0, 0], [1, 1, rows.shape[2], dst.shape[3]], [1, 1, 1, 1]
                        )
        for stage in traced.stages:
            ids = list(stage.device.get_device_ids())
            sm = next(sm for sm in self.submeshes_io if list(sm["device"].get_device_ids()) == ids)
            for src, mirror in stage.io.mirrors.items():
                local = entries_of(sm, src) if pools_of(sm, src) is not None else None
                for dst, rows, remote in zip(mirror, local or (None, None), sources[src]):
                    write_rows(dst, rows if rows is not None else self._on_stage(remote, stage.device))
        for entries, keys in sources.values():
            ttnn.deallocate(entries)
            ttnn.deallocate(keys)

    @staticmethod
    def _on_stage(tensor: ttnn.Tensor, device: ttnn.MeshDevice) -> ttnn.Tensor:
        """A replicated bf16 ROW_MAJOR ``tensor`` on ``device``: itself when it is there, else a copy through the host."""
        if list(tensor.device().get_device_ids()) == list(device.get_device_ids()):
            return tensor
        return ttnn.from_torch(
            ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if device.get_num_devices() > 1 else None,
        )

    # -- DSpark speculative decode -- #
    def prepare_spec_decode(self, rope_main, drafter: bool = True, hist: int = 32, **drafter_kwargs) -> None:
        """Allocate what speculative decode adds to :meth:`prepare_static_decode`. Call it after that and
        before anything is compiled.

        A verify round decodes ``[bonus, draft_1 .. draft_k]`` at consecutive positions, and the rows the
        rejected drafts wrote must leave no trace. Every stage keeps a shadow of each sliding ring, whose
        rows :meth:`_spec_fixup` puts back over the rejected slots. A ratio > 1 kv source also keeps a
        ``hist``-deep history of its compressor-window rows, from which the window slot of the last
        verified position is restored. Everything else a step writes is keyed by position, so the
        accepted token at that position rewrites it before anything reads it.

        ``drafter`` builds :class:`DeepSeekV41DSpark` on the output stage around the ``lm_head`` given to
        :meth:`prepare_static_decode` (``rope_main`` is the sliding RoPE table, ``drafter_kwargs`` go to the
        constructor) and taps the stream mean at its target layers into its stash every step.
        """
        if self._traces_compiled or self._traced_captured:
            raise RuntimeError("call prepare_spec_decode() before the decode traces are compiled")
        if self._slotted:
            raise NotImplementedError("speculative decode restores one session's rings: prepare with num_sessions=1")
        cfg = self.config
        window, head_dim = self.sliding_window, cfg.head_dim
        self._spec_hist = hist
        for sm in self.submeshes_io:
            device = sm["device"]
            sm["ring_shadow"] = {li: _device_zeros((1, 1, window, head_dim), device) for li in sm["layers"]}
            sm["fix_mask"] = _device_zeros((1, 1, window, 1), device)
            sm["fix_keep"] = _device_zeros((1, 1, window, 1), device)
            sm["hist_layers"] = [
                li for li in sm["layers"] if li in cfg.kv_source_layer_ids and int(cfg.compress_ratios[li]) > 1
            ]
            for li in sm["hist_layers"]:
                scache = sm["scaches"][li]
                scache.win_hist_kv = _device_zeros((1, 1, hist, head_dim), device)
                scache.win_hist_gate = _device_zeros((1, 1, hist, head_dim), device)
            hist_ratios = sorted({int(cfg.compress_ratios[li]) for li in sm["hist_layers"]})
            if hist_ratios:
                sm["fix_hist"] = _device_zeros((1, 1, 1, hist), device)
                sm["fix_slot"] = {r: _device_zeros((1, 1, r, 1), device) for r in hist_ratios}
                sm["fix_slot_keep"] = {r: _device_zeros((1, 1, r, 1), device) for r in hist_ratios}
        self._spec_tids = {}
        self._spec_draft_out = None
        if not drafter:
            return
        if self._lm_head_traced is None:
            raise ValueError("the drafter shares the decode lm_head: pass lm_head to prepare_static_decode()")
        out_sm = self.submeshes_io[self._output_sm_index]
        self.dspark = DeepSeekV41DSpark(
            cfg,
            self.loader,
            out_sm["device"],
            self._lm_head_traced,
            rope_main,
            cache=self.cache,
            tp_size=self.tp_size,
            **drafter_kwargs,
        )
        taps = dict(zip(self.dspark.targets, self.dspark.stash))
        elsewhere = [li for li in taps if li not in out_sm["layers"]]
        if elsewhere:
            raise ValueError(f"DSpark target layers {elsewhere} are not on the output stage {out_sm['index']}")
        out_sm["taps"] = taps
        out_sm["tap_weights"] = ttnn.from_torch(
            torch.full((1, 1, 1, cfg.hc_mult), 1.0 / cfg.hc_mult),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=out_sm["device"],
        )

    def _spec_fixup(self, sm: dict) -> None:
        """Undo the previous round's rejected rows on ``sm``, then snapshot its rings for this round.

        ``fix_mask`` marks the rejected positions' ring slots (``fix_keep`` is its complement); they get
        their shadow rows back. ``fix_hist`` picks the last verified position's row out of each window
        history and ``fix_slot`` is the window slot it goes back to. Every product is with 0 or 1, so the
        restore is exact.
        """
        for li in sm["layers"]:
            ring, shadow = sm["scaches"][li].kv, sm["ring_shadow"][li]
            ttnn.copy(ttnn.add(ttnn.multiply(ring, sm["fix_keep"]), ttnn.multiply(shadow, sm["fix_mask"])), ring)
            ttnn.copy(ring, shadow)
        for li in sm["hist_layers"]:
            scache = sm["scaches"][li]
            ratio = int(self.config.compress_ratios[li])
            slot, keep = sm["fix_slot"][ratio], sm["fix_slot_keep"][ratio]
            for win, hist in ((scache.win_kv, scache.win_hist_kv), (scache.win_gate, scache.win_hist_gate)):
                row = ttnn.matmul(sm["fix_hist"], hist, compute_kernel_config=_HIFI4)
                ttnn.copy(ttnn.add(ttnn.multiply(win, keep), ttnn.multiply(slot, row)), win)

    def capture_spec_traces(self, token_id, pos: int) -> None:
        """Compile and then capture the decode variants, every stage's fixup and the drafter's
        ``write_main`` / ``draft``.

        All compile runs come before the first capture on any chip (see :meth:`_capture_traces`). They
        write zeros or the rows of ``pos``, which prefill overwrites.
        """
        if self._spec_hist is None:
            raise RuntimeError("call prepare_spec_decode() first")
        self.compile_traces(token_id, pos)
        for sm in self.submeshes_io:
            self._spec_fixup(sm)
        if self.dspark is not None:
            self.dspark.write_main()
            for out in self.dspark.draft():
                out.deallocate(True)
        for sm in self.submeshes_io:
            ttnn.synchronize_device(sm["device"])
        self._capture_traces(token_id, pos)
        for sm in self.submeshes_io:
            tid = ttnn.begin_trace_capture(sm["device"], cq_id=0)
            with _trace_capture_guard():
                self._spec_fixup(sm)
            ttnn.end_trace_capture(sm["device"], tid, cq_id=0)
            sm["fix_tid"] = tid
        if self.dspark is not None:
            device = self.dspark.device
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            with _trace_capture_guard():
                self.dspark.write_main()
            ttnn.end_trace_capture(device, tid, cq_id=0)
            self._spec_tids["write_main"] = tid
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            with _trace_capture_guard():
                self._spec_draft_out = self.dspark.draft()
            ttnn.end_trace_capture(device, tid, cq_id=0)
            self._spec_tids["draft"] = tid

    def spec_fixup(self, rejected, frontier: int) -> None:
        """Queue every stage's fixup ahead of a verify round: restore the ring slots of the previous round's
        ``rejected`` positions and the window rows of the last verified position ``frontier``."""
        self._wait_replays_dispatched()
        window, hist = self.sliding_window, self._spec_hist
        mask = torch.zeros(1, 1, window, 1)
        for p in rejected:
            mask[0, 0, p % window, 0] = 1
        pick = torch.zeros(1, 1, 1, hist)
        pick[0, 0, 0, frontier % hist] = 1
        for sm in self.submeshes_io:
            host_to_device(mask, sm["fix_mask"])
            host_to_device(1 - mask, sm["fix_keep"])
            if sm["hist_layers"]:
                host_to_device(pick, sm["fix_hist"])
                for ratio, slot in sm["fix_slot"].items():
                    one_hot = torch.zeros(1, 1, ratio, 1)
                    one_hot[0, 0, frontier % ratio, 0] = 1
                    host_to_device(one_hot, slot)
                    host_to_device(1 - one_hot, sm["fix_slot_keep"][ratio])
            ttnn.execute_trace(sm["device"], sm["fix_tid"], cq_id=0, blocking=False)

    def spec_write_main(self, positions, written) -> None:
        """Queue the drafter's ``write_main`` for verified ``positions`` (``written[j]`` False keeps row
        ``j`` out of the rings). Their stash rows must still be in the stash."""
        self._wait_replays_dispatched()
        self.dspark.set_main_rows(positions, written)
        ttnn.execute_trace(self.dspark.device, self._spec_tids["write_main"], cq_id=0, blocking=False)

    def spec_seed(self, positions) -> None:
        """Write every one of ``positions`` into the drafter rings, ``main_rows`` at a time."""
        positions = list(positions)
        rows = self.dspark.main_rows
        for i in range(0, len(positions), rows):
            chunk = positions[i : i + rows]
            self.spec_write_main(chunk, [True] * len(chunk))

    def spec_draft(self, token: int, frontier: int) -> tuple[list[int], list[float]]:
        """Draft from ``token``, the next input after the last verified position ``frontier``: ``block_size``
        proposals for positions ``frontier + 2 ..`` and their confidence logits. The drafter rings must
        hold every verified position up to ``frontier``."""
        self._wait_replays_dispatched()
        d = self.dspark
        d.set_block(token, frontier)
        ttnn.execute_trace(d.device, self._spec_tids["draft"], cq_id=0, blocking=False)
        ids, conf = (ttnn.to_torch(ttnn.get_device_tensors(t)[0]) for t in self._spec_draft_out)
        return [int(i) for i in ids.flatten()], [float(c) for c in conf.flatten()]
