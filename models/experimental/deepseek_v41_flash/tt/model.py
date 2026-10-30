# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash decode model: V4's traced TP pipeline around V4.1 layers.

The deployment is TP4 across ``DEEPSEEK_V4_NUM_STAGES`` stages (default 8: a 32-chip mesh opened
as ``(8, 4)``). Each stage is a 1x4 submesh; with the profile's pipeline group size of 1 it owns
one contiguous run of layers.
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
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
import ttnn
from loguru import logger
from tqdm import tqdm

from models.experimental.deepseek_v41_flash.tt.config import (
    decoder_layer_weights,
    dequantized,
    expert_provider,
    pipeline_stage_count,
)
from models.experimental.deepseek_v41_flash.tt.decode.attention import (
    _SELECT_KV_RATE,
    DeepSeekV41LayerCache,
    _rows_on_one_core,
    candidate_buffers,
    key_split_layers,
    uses_candidate_pool,
)
from models.experimental.deepseek_v41_flash.tt.decode.decoder_layer import DeepSeekV41DecoderLayer
from models.experimental.deepseek_v41_flash.tt.decode.engram import open_rows_sockets, rows_page_bytes
from models.experimental.deepseek_v41_flash.tt.decode.dspark import DeepSeekV41DSpark, host_to_device
from models.experimental.deepseek_v41_flash.tt.decode.moe import DeepSeekV41PreloadedExperts
from models.experimental.deepseek_v41_flash.tt.engram_lookup import (
    EngramHostLookup,
    engram_ram_needed,
    engram_table_nbytes,
    load_engram_table,
)
from models.experimental.deepseek_v41_flash.tt.lm_head import DeepSeekV41DeviceSampler, DeepSeekV41LMHead
from models.experimental.deepseek_v41_flash.tt.prefill.model import DeepSeekV41PrefillModel
from models.experimental.deepseek_v41_flash.tt.vision import extend_embedding, image_mask
from models.experimental.deepseek_v4_flash.tt import common as v4_common
from models.experimental.deepseek_v4_flash.tt.common import (
    CCL_TOPOLOGY,
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
    make_rope_table,
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


# Row width of the logits pages a decode step sends the drafter: the base model's D2H staging width, whose
# untilize fits beside the prefetcher rings; it divides the 129280-wide vocab.
_LOGIT_PAGE = 2020


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


def _replica_rows(tensor: ttnn.Tensor) -> torch.Tensor:
    """A replicated device ``tensor`` read back from its first chip, leading dims folded: ``[rows, width]``."""
    host = ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])
    return host.reshape(-1, host.shape[-1])


@dataclass(eq=False)
class SessionSnapshot:
    """One session's decode state in host memory, from :meth:`DeepSeekV41Model.export_session`.

    ``rings``: layer -> its ring slot ``[sliding_window, head_dim]``. ``windows``: ratio > 1 kv source -> its
    compressor window slot ``(kv, gate)``, ``[_WIN_SLOT_ROWS, head_dim]`` each (the whole ``[ratio, head_dim]``
    windows with one session). ``entries``: kv source -> its first ``pos // ratio`` compressed entries
    ``[n, head_dim]`` and index keys ``[n, index_head_dim]``. ``engram``: the compressed-token history ``[1, pos]``.
    """

    pos: int
    rings: dict[int, torch.Tensor]
    windows: dict[int, tuple[torch.Tensor, torch.Tensor]]
    entries: dict[int, tuple[torch.Tensor, torch.Tensor]]
    engram: Optional[torch.Tensor]

    @property
    def nbytes(self) -> int:
        tensors = [*self.rings.values()]
        tensors += [t for pair in (*self.windows.values(), *self.entries.values()) for t in pair]
        if self.engram is not None:
            tensors.append(self.engram)
        return sum(t.numel() * t.element_size() for t in tensors)


class DeepSeekV41Model(DeepSeekV4Model):
    """V4.1 decode stack on V4's traced pipeline.

    ``tp_size`` defaults to 4 and ``num_stages`` to :func:`pipeline_stage_count`
    (``DEEPSEEK_V4_NUM_STAGES``, default 8), so a 32-chip mesh runs one 1x4 stage per row.
    ``use_submeshes`` stays on: traced decode sends the packet and the residual over sockets.
    ``weight_dtype`` is the attention and routed-expert dtype (bfloat4_b in the demo); the shared
    expert stays bfloat8_b and the Engram, norms and hyper-connections stay at their own defaults.
    ``layer_submesh_ids`` (one non-decreasing stage index per built layer) overrides V4's even
    :func:`plan_layer_placement`, e.g. with the prefill placement when a prefill model shares the stages.

    ``verify_rows > 1`` builds every layer for DSpark's parallel verify (:meth:`decode_verify`), one step
    over ``verify_rows`` consecutive positions.

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
        num_stages: Optional[int] = None,
        submeshes: Optional[Sequence[ttnn.MeshDevice]] = None,
        use_prefetcher: Optional[bool] = None,
        layer_submesh_ids: Optional[Sequence[int]] = None,
        image_rows: int = 0,
        verify_rows: int = 1,
    ):
        self.config = config
        self.image_rows = image_rows
        self.verify_rows = verify_rows
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
        if num_stages is None:
            num_stages = pipeline_stage_count()
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
        # A verify block's rows outnumber a socket's pages, so decode_verify writes them from a second thread.
        engram_sockets = {
            li: open_rows_sockets(self.submeshes[self.layer_submesh_ids[li]], rows_page_bytes(config))
            for li in config.engram_layer_ids
            if li < n
        }
        self.embed_tokens = DeepSeekV4Embedding(loader, self.first_device, cache=cache)
        if image_rows:
            # Image spans embed as ids past the vocabulary (see :mod:`.vision`), rows filled by
            # :func:`.vision.write_embedding_rows`. Prefill and decode read the one table.
            self.embed_tokens.embedding_weight = extend_embedding(
                self.embed_tokens.embedding_weight, image_rows, config.vocab_size
            )
        self.layer_devices: list[ttnn.MeshDevice] = [self.submeshes[s] for s in self.layer_submesh_ids[:n]]

        self.layers: list[DeepSeekV41DecoderLayer] = [None] * n

        def build_stage(layer_ids: list[int]) -> None:
            for li in layer_ids:
                current_device = self.layer_devices[li]
                logger.info(f"Layer {li} is on device {self.layer_submesh_ids[li]}")
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
                self.layers[li] = DeepSeekV41DecoderLayer(
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
                    verify_rows=verify_rows,
                )
                _profile(current_device)

        # Stages are disjoint submeshes (own allocator and command queues), so each builds on its own thread; a
        # stage's layers stay in order, keeping every device's allocation order (and addresses) as when serial.
        stage_layers: dict[int, list[int]] = {}
        for li in range(n):
            stage_layers.setdefault(self.layer_submesh_ids[li], []).append(li)
        with ThreadPoolExecutor(max_workers=max(1, len(stage_layers)), thread_name_prefix="decode-build") as pool:
            for stage in [pool.submit(build_stage, ids) for ids in stage_layers.values()]:
                stage.result()
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
        self.device_sampler: Optional[DeepSeekV41DeviceSampler] = None
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
        # Depth of the ratio > 1 compressor windows under speculative decode (None before prepare_spec_decode).
        self.spec_window_rows: Optional[int] = None
        self.dspark: Optional[DeepSeekV41DSpark] = None
        # Steps posted while set leave the drafter's receive of their row to spec_release_writes.
        self._spec_hold_writes = False
        # Outputs of dispatched steps nobody wants (a streamed round's rows past its first reject), read and
        # dropped ahead of the next read_decoded_output.
        self._stale_outputs = 0

    def _engram_ids(self) -> list[int]:
        return [i for i in self.config.engram_layer_ids if i < self.num_layers]

    def _require_engram_ram(self) -> None:
        """Refuse to build when the Engram tables (plus the same headroom the tests use) do not fit."""
        ids = self._engram_ids()
        if not ids:
            return
        need = engram_ram_needed(self.loader, ids)
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
        device_sampling: bool = False,
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

        ``device_sampling`` picks each step's token on the output stage (:attr:`device_sampler`, greedy until its
        ``set_sampling``): :meth:`read_decoded_output` then returns ``[1, 1, words]`` int32, the token first, and no
        logits leave the stage.
        """
        if batch != 1:
            raise NotImplementedError(f"V4.1 decode supports batch 1, got batch={batch}")
        if device_sampling and lm_head is None:
            raise ValueError("device_sampling samples from the folded-in lm_head; pass lm_head")
        if device_sampling and self.verify_rows > 1:
            raise NotImplementedError("the verify blocks read their rows' logits off the decode socket")
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
        self.device_sampler = None
        if device_sampling:
            self.device_sampler = DeepSeekV41DeviceSampler(
                lm_head.device, lm_head.weight.shape[-1], self.system_config.pipeline.pcie_alignment
            )
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

        self._rope_halves = {family: rope[family] for family in ("main", "compress")}
        rope_gen = {}
        for family in ("main", "compress"):
            cos_h, sin_h = rope[family]
            scaling = float(cos_h[0, 0].item())
            inv_freq = torch.atan2(sin_h[1].float(), cos_h[1].float()).repeat_interleave(2)
            rope_gen[family] = (inv_freq.reshape(1, 1, 1, -1), scaling)

        alignment = self.system_config.pipeline.pcie_alignment
        # [token | ring row | pos | slot | slot's first ring row | slot's first window row | drafter ring row],
        # see _build_packet.
        self._pkt_int_prefix = 7
        # A verify block's packet (_verify_packet) shares the socket, so the page holds the larger of the two.
        t = self.verify_rows
        self._verify_fields = {}
        if t > 1:
            sizes = {"token": t, "pos": t, "sliding_pos": t, "ring_rows": t * t, "sliding_curpos": t}
            sizes.update(win_prev=1, stash_slot=t, dspark_slot=t)
            for r in sorted(ratios):
                sizes.update({f"{name}{r}": t for name in ("win_slot", "entry", "start", "index_pos", "curpos")})
            offset = 0
            for name, size in sizes.items():
                self._verify_fields[name] = (offset, size)
                offset += size
        packet_ints = max([self._pkt_int_prefix] + [o + n for o, n in self._verify_fields.values()])
        self._pkt_page_bytes = math.ceil(packet_ints * 4 / alignment) * alignment
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
        split_layers = key_split_layers(cfg, range(self.num_layers), max_seq) if self.tp_size > 1 else set()
        for layer in self.layers:
            if layer.self_attn.is_index_source:
                layer.self_attn.indexer.split_keys = layer.layer_idx in split_layers
        logger.info(f"key-split selects: layers {sorted(split_layers)}")
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
                        # Whole packets: a verify-sized page need not divide the configured size.
                        self.system_config.pipeline.h2d_fifo_bytes // self._pkt_page_bytes * self._pkt_page_bytes,
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
            ids = [layer_idx for layer_idx, _ in self._engram_layers]
            total = sum(engram_table_nbytes(self.loader, i) for i in ids)
            t0 = time.perf_counter()
            # Drawn on a terminal only (disable=None); a log file gets the summary line below.
            with tqdm(
                total=total, unit="B", unit_scale=True, desc=f"Engram tables {ids}", mininterval=0.5, disable=None
            ) as bar:
                lock = threading.Lock()

                def advance(n: int) -> None:  # the table readers are 16 threads
                    with lock:
                        bar.update(n)

                tables = {}
                for layer_idx in ids:
                    bar.set_postfix_str(f"layer {layer_idx}")
                    tables[layer_idx] = load_engram_table(self.loader, layer_idx, progress=advance)
            seconds = time.perf_counter() - t0
            logger.info(
                f"Engram tables of layers {ids} read: {total / 1e9:.0f} GB in {seconds:.0f}s "
                f"({total / 1e9 / max(seconds, 1e-9):.1f} GB/s)"
            )
            self._engram_lookup = EngramHostLookup(
                cfg, self.loader.snapshot_dir, tables, batch=batch, max_seq=max_seq, sessions=num_sessions
            )
            logger.info(f"Engram host lookup: layers {ids}")

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
                            # Filled on device: a pool sized for the shared budget is tens of GB per chip,
                            # which zeros built on the host take minutes to write.
                            comp_kv=ttnn.moreh_full(
                                [blocks, 1, CSA_INDEX_BLOCK_SIZE, head_dim],
                                0.0,
                                device,
                                dtype=ttnn.bfloat16,
                                layout=ttnn.ROW_MAJOR_LAYOUT,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
        slot * W | slot * 32 | pos % drafter ring rows]`` for session slot ``_packet_slot``, then padding.

        The slot fields are what makes a session switch free: the traces read the slot's ring rows,
        window rows and page-table row from here, so nothing on device changes between sessions. The
        drafter's ring row (0 without a drafter) is where :meth:`_spec_write_step` writes the step's row.
        """
        w, slot = self.sliding_window, self._packet_slot
        drafter_row = pos % self.dspark.ring_rows if self.dspark is not None else 0
        values = self._step_tokens(token_id) + [slot * w + pos % w, pos, slot, slot * w, slot * _WIN_SLOT_ROWS]
        values.append(drafter_row)
        packet = torch.zeros(1, 1, 1, self._pkt_w, dtype=torch.int32)
        packet[0, 0, 0, : len(values)] = torch.tensor(values, dtype=torch.int32)
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
            image = (ids >= self.config.vocab_size).unsqueeze(-1)  # an image-span token: no Engram contribution
            for layer_idx, engram in self._engram_layers:
                ids_of_layer = hashes[:, :, lookup.layout.layer_ids.index(layer_idx)]
                rows = lookup.gather(layer_idx, ids_of_layer).flatten(-2)
                engram.write_rows(rows.masked_fill(image, 0) if bool(image.any()) else rows)
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

    def export_session(self, sid: int) -> SessionSnapshot:
        """Copy session ``sid``'s decode state at its position ``P`` into host memory, for :meth:`import_session`.

        A session's state is its slot of every ring and compressor window, the first ``P // ratio`` entries of each
        group's paged pools and its Engram history; everything else a step reads, it writes first (see
        :meth:`reset_session`). The pools are read from one stage per group, since every stage holding a group keeps
        the same entries. Reads only: the session can go on, or be reset. Call it with no step in flight.
        """
        self._require_paged()
        if sid not in self._slot_of:
            raise KeyError(f"no such session {sid}")
        self._wait_replays_dispatched()
        pos, slot, w = self._session_pos[sid], self._slot_of[sid], self.sliding_window

        def slot_rows(t: ttnn.Tensor, start: int, n: int) -> torch.Tensor:
            part = ttnn.slice(t, [0, 0, start, 0], [1, 1, start + n, t.shape[3]])
            rows = _replica_rows(part)
            if part.buffer_address() != t.buffer_address():  # a slice of all of ``t`` is ``t`` itself
                ttnn.deallocate(part)
            return rows

        rings, windows, entries = {}, {}, {}
        for sm in self.submeshes_io:
            for li in sm["layers"]:
                scache = sm["scaches"][li]
                rings[li] = slot_rows(scache.kv, slot * w, w)
                src = self._kv_source_of.get(li)
                if li == src and scache.win_kv is not None:
                    win_rows = _WIN_SLOT_ROWS if self._slotted else scache.win_kv.shape[2]
                    windows[li] = tuple(
                        slot_rows(t, slot * win_rows, win_rows) for t in (scache.win_kv, scache.win_gate)
                    )
                if src is None or src in entries or scache.comp_kv is None:
                    continue
                ratio = int(self.config.compress_ratios[src])
                if pos // ratio == 0:
                    continue
                kv, keys = self._gather_paged_entries(scache, self._paged_ratios[ratio].layer_type, sid, pos // ratio)
                entries[src] = (_replica_rows(kv), _replica_rows(keys))
                ttnn.deallocate(kv)
                ttnn.deallocate(keys)
        engram = self._engram_lookup.export_history(slot, pos) if self._engram_lookup is not None else None
        return SessionSnapshot(pos, rings, windows, entries, engram)

    def import_session(self, sid: int, snapshot: SessionSnapshot) -> None:
        """Load ``snapshot`` (:meth:`export_session` of this model, from any session) into session ``sid``, which must
        be at position 0 (opened or reset), and move the session to the snapshot's position.

        Raises :class:`PagedCacheFull` when the pools cannot hold the snapshot's entries; the session keeps whatever
        blocks it grew, so reset it then. Call it with no step in flight.
        """
        self._require_paged()
        if sid not in self._slot_of:
            raise KeyError(f"no such session {sid}")
        if self._session_pos[sid]:
            raise ValueError(f"session {sid} is at {self._session_pos[sid]}: import into a session at position 0")
        pos = snapshot.pos
        if pos >= self._decode_max_seq:
            raise PagedCacheFull(f"a {pos}-token snapshot is past max_seq {self._decode_max_seq}")
        self._wait_replays_dispatched()
        if pos:
            self.ensure_session_capacity(pos - 1, sid)
        slot, w = self._slot_of[sid], self.sliding_window
        for sm in self.submeshes_io:
            device = sm["device"]
            mapper = ttnn.ReplicateTensorToMesh(device) if device.get_num_devices() > 1 else None

            def write_rows(dst: ttnn.Tensor, rows: torch.Tensor, start: int) -> None:
                if rows.shape[0] == dst.shape[2]:
                    host_to_device(rows.reshape(tuple(dst.shape)).float(), dst)
                    return
                src_rows = ttnn.from_torch(
                    rows.reshape(1, 1, *rows.shape).float(),
                    dtype=dst.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=mapper,
                )
                ttnn.fill_cache(dst, src_rows, 0, update_idx=start)
                ttnn.deallocate(src_rows)

            written = set()
            for li in sm["layers"]:
                scache = sm["scaches"][li]
                write_rows(scache.kv, snapshot.rings[li], slot * w)
                src = self._kv_source_of.get(li)
                if li == src and scache.win_kv is not None:
                    for dst, rows in zip((scache.win_kv, scache.win_gate), snapshot.windows[li]):
                        write_rows(dst, rows, slot * rows.shape[0])
                if src is None or src in written or scache.comp_kv is None or src not in snapshot.entries:
                    continue
                written.add(src)
                kv, keys = (
                    ttnn.from_torch(
                        rows.reshape(1, 1, *rows.shape).float(),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=mapper,
                    )
                    for rows in snapshot.entries[src]
                )
                ratio = int(self.config.compress_ratios[src])
                self._commit_paged_entries(scache, self._paged_ratios[ratio].layer_type, sid, kv, keys)
                ttnn.deallocate(kv)
                ttnn.deallocate(keys)
        for sm in self.submeshes_io:
            ttnn.synchronize_device(sm["device"])
        if snapshot.engram is not None and self._engram_lookup is not None:
            self._engram_lookup.import_history(slot, snapshot.engram)
        self._session_pos[sid] = pos

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
        self._replay_queue.put((pos, sid, self._spec_hold_writes))

    def write_step_packet(self, token_id, pos: int) -> None:
        """Push the packet of the oldest posted step, in its session's slot (eager: then run it)."""
        sid = self._packet_sids.popleft()
        self._write_packet(token_id, pos, sid)
        if self._eager_decode:
            posted = self._eager_pending.popleft()
            assert posted == (pos, sid), f"eager decode: packet for {(pos, sid)}, but {posted} was posted next"
            self._execute_traces((pos, sid, False))

    def _execute_traces(self, step) -> None:
        """Replay the variant of ``step`` = ``(pos, sid, held)`` on every submesh, after growing that session's
        blocks for it (normally already done when it was posted). With a drafter, the drafter's receive of the
        step's row follows unless ``held`` (:meth:`spec_release_writes` queues it then)."""
        pos, sid, held = step
        self.ensure_session_capacity(pos, sid)
        self._session_pos[sid] = pos + 1
        variant = self._variant_key(pos)
        if self._eager_decode:
            self._run_eager_step(variant)
            return
        with self._cq_lock:
            for sm in self.submeshes_io:
                ttnn.execute_trace(sm["device"], sm["tids"][variant], cq_id=0, blocking=False)
            if self.dspark is not None and not held:
                if "write_step" not in self._spec_tids:
                    raise RuntimeError("the decode step sends the drafter its row: call capture_spec_traces() first")
                ttnn.execute_trace(self.dspark.device, self._spec_tids["write_step"], cq_id=0, blocking=False)

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
            "win_prev": {},
            "win_rope": {},
            "entry": {},
            "index_pos": {},
            # Even one session's page table is not the identity (block 0 is ZERO_BLOCK), so the paged
            # ops always read the slot's row.
            "page_row": field(3),
            "ring_offset": None,
            "win_offset": None,
            # Image-span tokens (ids past the vocabulary) rank their experts with ``bias_vl``.
            "image_mask": image_mask(token, self.config.vocab_size) if self.image_rows else None,
        }
        win_base_f = None
        if self._slotted:
            ctx["ring_offset"], ctx["win_offset"] = field(4), field(5)
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
        win_rows = self.spec_window_rows
        for ratio in sm["kv_ratios"]:
            if ratio > 1:
                ctx["win_slot"][ratio] = self._win_slot(pos_f, win_rows or ratio, base_f=win_base_f)
            if ratio > 1 and win_rows and r2_pool:
                prev = ttnn.add(pos_f, float(win_rows - 1))
                prev = ttnn.subtract(
                    prev, ttnn.multiply(ttnn.floor(ttnn.multiply(prev, 1.0 / win_rows)), float(win_rows))
                )
                ctx["win_prev"][ratio] = ttnn.typecast(ttnn.eq(sm["win_cols"], prev), ttnn.bfloat16)
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
                inputs["win_prev"] = ctx["win_prev"].get(ratio)
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
        streams are tiled in DRAM first. The result is ``[B, S, 1, D]`` bf16, the final norm's input;
        ``pre_mix`` is ``[B, S, 1, hc]`` or one ``[1, 1, 1, hc]`` row for all of them.
        """
        batch, s, hc, hidden = streams.shape
        wb, ws = pre_mix.shape[0], pre_mix.shape[1]
        streams = ttnn.to_layout(ttnn.to_memory_config(streams, ttnn.DRAM_MEMORY_CONFIG), ttnn.TILE_LAYOUT)
        streams_f = ttnn.typecast(streams, ttnn.float32)
        weights = ttnn.typecast(pre_mix, ttnn.float32)
        total = None
        for j in range(hc):
            row = ttnn.slice(streams_f, [0, 0, j, 0], [batch, s, j + 1, hidden])
            weight = ttnn.slice(weights, [0, 0, 0, j], [wb, ws, 1, j + 1])
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
        ``lm_head`` and pushes the logits out the D2H socket. With a drafter, the output stage stashes the
        stream means at its target layers and, after the last, sends them to the drafter
        (:meth:`_send_step_row`), then its logits (:meth:`_send_step_logits`).
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
        tapped = {}
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
                    pkt = ttnn.broadcast(pkt, ttnn.MeshCoordinate(0, 0), cluster_axis=1, topology=CCL_TOPOLOGY)
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
                tapped[li] = ttnn.to_memory_config(row, ttnn.DRAM_MEMORY_CONFIG)
                if len(tapped) == len(sm["taps"]):
                    self._send_step_row(tapped, sm["taps"], pkt)
            streams, pre_mix = layer.decode(
                streams,
                pre_mix,
                sm["scaches"][li],
                image_mask=ctx["image_mask"],
                **self._attention_inputs(layer.self_attn, ctx, causal, r2_pool, r2_select, two_level),
            )
            if is_last:
                hidden = self.norm(self._collapse_streams(streams, pre_mix))
                if self.device_sampler is not None:
                    hidden = self.device_sampler(self._lm_head_traced.linear(hidden))
                elif self._lm_head_traced is not None:
                    hidden = self._lm_head_traced(hidden)
                out = hidden
                self._send_output(out)
                if "taps" in sm:
                    self._send_step_logits(out)
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
    def build_prefill(self, rope: dict, lm_head=None, kv_only: bool = False, **kwargs) -> DeepSeekV41PrefillModel:
        """Build the :class:`DeepSeekV41PrefillModel` on this model's stages.

        Every prefill layer lives on its decode layer's submesh and reads that layer's routed experts in
        place. The embedding table, ``lm_head`` (when given), the D2D socket pairs and the Engram host
        lookup are shared too, so a prefill and the decode steps after it hash one token history.
        ``kwargs`` go to the prefill constructor.

        ``kv_only`` builds only the layers up to the last kv source (the prefill model's ``kv_only`` mode), on
        the stages that hold them, and no head (``lm_head`` is ignored). :meth:`commit_prefill_state` then fills
        the shared compressed KV and the rings of those layers; the caller replays the last ``sliding_window``
        prompt tokens through decode to fill the other layers' rings and get the logits.

        Call it after :meth:`prepare_static_decode` (which builds the Engram lookup) and before the decode
        traces are captured.
        """
        if self._decode_max_seq is None:
            raise RuntimeError("call prepare_static_decode() before build_prefill()")
        n = self.num_layers
        if kv_only:
            n = max(i for i in self.config.kv_source_layer_ids if i < self.num_layers) + 1
            lm_head = None
        self.prefill_model = DeepSeekV41PrefillModel(
            self.config,
            self.loader,
            self.first_device,
            rope,
            num_layers=n,
            tp_size=self.tp_size,
            layer_devices=self.layer_devices[:n],
            engram_lookup=self._engram_lookup,
            experts=[layer.mlp.experts for layer in self.layers[:n]],
            embedding_weight=self.embed_tokens.embedding_weight,
            lm_head=lm_head,
            kv_only=kv_only,
            image_rows=self.image_rows,
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

        A KV-only prefill (``build_prefill(kv_only=True)``) has states for the layers up to the last kv source
        only. The rings of the layers after them are zeroed: once the ring is full, decode reads every row, so the
        rows must not hold another session's keys. The caller replays the last ``sliding_window`` prompt tokens
        through decode, which writes those rings again.
        """
        note = progress or (lambda message: None)
        if bias_slots:
            raise ValueError("V4.1 has no CSA position bias: commit_prefill_state takes no bias_slots")
        if not self.paged:
            raise RuntimeError("call prepare_static_decode() first")
        if session_id not in self._slot_of:
            raise KeyError(f"no such session {session_id}")
        kv_sources = [i for i in self.config.kv_source_layer_ids if i < self.num_layers]
        if len(states) <= max(kv_sources, default=-1) or len(states) > self.num_layers:
            raise ValueError(
                f"need the states of layers 0..{max(kv_sources, default=0)} (the last kv source) up to all "
                f"{self.num_layers}, got {len(states)}"
            )
        lengths = {state.seq_len for state in states}
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
            zero_rings = {}  # head dim -> a zero [1, 1, W, Dh] ring for the layers without a state
            for li in sm["layers"]:
                scache = sm["scaches"][li]
                if li < len(states):
                    state = states[li]
                    if list(state.kv_tail.device().get_device_ids()) != list(device.get_device_ids()):
                        raise RuntimeError(f"prefill layer {li}'s state is not on its decode submesh")
                    note(f"commit layer {li + 1}/{self.num_layers} on decode submesh {sm['index']}")
                    ring = ttnn.to_layout(state.kv_tail, ttnn.TILE_LAYOUT)
                    ttnn.fill_cache(scache.kv, ring, 0, update_idx=ring_row)
                    ttnn.deallocate(ring)
                else:
                    note(f"zero the ring of layer {li + 1}/{self.num_layers} on decode submesh {sm['index']}")
                    dh = scache.kv.shape[3]
                    if dh not in zero_rings:
                        zero_rings[dh] = ttnn.from_torch(
                            torch.zeros(1, 1, self.sliding_window, dh),
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
                        )
                    ttnn.fill_cache(scache.kv, zero_rings[dh], 0, update_idx=ring_row)
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
            for zeros in zero_rings.values():
                ttnn.deallocate(zeros)
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
                if li not in traced.buffers:  # past a kv-only prefill model's last layer
                    continue
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
    def prepare_spec_decode(self, rope_main, drafter: bool = True, **drafter_kwargs) -> None:
        """Allocate what speculative decode adds to :meth:`prepare_static_decode`. Call it after that and
        before anything is compiled.

        A verify round decodes ``[bonus, draft_1 .. draft_k]`` at consecutive positions, and the rows the
        rejected drafts wrote must leave no trace. Every stage keeps a shadow of each sliding ring, whose
        rows :meth:`_spec_fixup` puts back over the rejected slots. A ratio > 1 kv source's compressor window
        grows to :attr:`spec_window_rows` rows (slot ``pos % rows``): a round's rejected rows, fewer than
        that, then never overwrite the row of the last verified position, which the next group pairs with
        through ``win_prev``, and the next round rewrites them before they are read. Everything else a step
        writes is keyed by position, so the accepted token at that position rewrites it before anything
        reads it.

        ``drafter`` builds :class:`DeepSeekV41DSpark` on its own 1xTP stage, the first mesh row the pipeline
        leaves idle (build with ``num_stages`` below the mesh's row count), with its own copy of the decode
        ``lm_head`` (``rope_main`` is the sliding RoPE table, ``drafter_kwargs`` go to the constructor).
        The output stage taps the stream mean at the target layers into the stash every step; the stash
        stays there, and each :meth:`spec_write_main` sends the rows it gathers over one D2D socket. The
        drafter's prefetcher rings are on its own row, so the caller opens a prefetcher session on
        ``dspark.device`` (the model's session predates it).

        A model built with ``verify_rows > 1`` also gets what :meth:`decode_verify` needs: every compressed
        group's ``verify_rows``-row selections on each stage, the stages' verify streams layout and the
        buffers of its traces (:meth:`_prepare_verify_traces`).
        """
        if self._traces_compiled or self._traced_captured:
            raise RuntimeError("call prepare_spec_decode() before the decode traces are compiled")
        if self._slotted:
            raise NotImplementedError("speculative decode restores one session's rings: prepare with num_sessions=1")
        if self.device_sampler is not None:
            raise NotImplementedError(
                "speculative decode accepts against the steps' logits: prepare without device_sampling"
            )
        if self.image_rows:
            raise NotImplementedError("the drafter's token table has no image rows: build without image_rows")
        if self._decode_batch != 1:
            raise NotImplementedError("speculative decode steps one user: prepare with batch=1")
        cfg = self.config
        window, head_dim = self.sliding_window, cfg.head_dim
        win_rows = _WIN_SLOT_ROWS
        self.spec_window_rows = win_rows
        for sm in self.submeshes_io:
            device = sm["device"]
            sm["ring_shadow"] = {li: _device_zeros((1, 1, window, head_dim), device) for li in sm["layers"]}
            sm["fix_mask"] = _device_zeros((1, 1, window, 1), device)
            sm["fix_keep"] = _device_zeros((1, 1, window, 1), device)
            sources = [li for li in sm["layers"] if li in cfg.kv_source_layer_ids and int(cfg.compress_ratios[li]) > 1]
            sm["deep_windows"] = bool(sources)
            for li in sources:
                old_kv, old_gate = sm["scaches"][li].win_kv, sm["scaches"][li].win_gate
                win_kv = _device_zeros((1, 1, win_rows, head_dim), device)
                win_gate = _device_zeros((1, 1, win_rows, head_dim), device)
                # Every layer of the group aliases the source's window.
                for scache in sm["scaches"].values():
                    if scache.win_kv is old_kv:
                        scache.win_kv, scache.win_gate = win_kv, win_gate
                ttnn.deallocate(old_kv)
                ttnn.deallocate(old_gate)
            if sources:
                sm["win_cols"] = ttnn.from_torch(
                    torch.arange(win_rows, dtype=torch.float32).reshape(1, 1, 1, win_rows),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )
        t = self.verify_rows
        if t > 1:
            split = [
                layer.layer_idx
                for layer in self.layers
                if layer.self_attn.is_index_source and layer.self_attn.indexer.split_keys
            ]
            if split:
                raise NotImplementedError(f"a verify step selects unsplit, but layers {split} split their keys")
            hc = cfg.hc_mult
            for sm in self.submeshes_io:
                device = sm["device"]
                # The Engram's verify layout, which every layer of a verify step keeps the streams in.
                sm["verify_streams_memory_config"] = width_sharded_l1_config(
                    t * hc, cfg.hidden_size, device, num_cores=_HC_COLLAPSE_CORES, tile_height=1
                )
                # The one session's slot, its row of every page table.
                sm["verify_page_row"] = _device_zeros([1], device, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                selections = {}
                for li in sm["layers"]:
                    if not cfg.compress_ratios[li]:
                        continue
                    src = self._kv_source_of[li]
                    if src not in selections:
                        rows = t * (window + cfg.index_topk)
                        selections[src] = (
                            _device_zeros((1, 1, rows, head_dim), device, layout=ttnn.ROW_MAJOR_LAYOUT),
                            _device_zeros((1, 1, rows, head_dim), device),
                        )
                    scache = sm["scaches"][li]
                    scache.sel_kv_rows, scache.sel_tile_rows = selections[src]
                if 0 in sm["layers"]:
                    one_hot = torch.zeros(1, t, 1, hc)
                    one_hot[..., 0] = 1
                    sm["verify_pre_mix_init"] = ttnn.from_torch(
                        one_hot, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                    )
            self._prepare_verify_traces()
        self._spec_tids = {}
        self._spec_verified_in = None
        self._spec_step_in = None
        if not drafter:
            return
        if self._lm_head_traced is None:
            raise ValueError("the drafter copies the decode lm_head: pass lm_head to prepare_static_decode()")
        drafter_row = self.num_submeshes
        if drafter_row >= self.mesh_devices // self.tp_size:
            raise ValueError(
                f"the drafter runs on the row after the pipeline's {self.num_submeshes}, but the mesh has "
                f"{self.mesh_devices // self.tp_size}: build with fewer num_stages"
            )
        out_sm = self.submeshes_io[self._output_sm_index]
        device = self.device.create_submesh(ttnn.MeshShape(1, self.tp_size), ttnn.MeshCoordinate(drafter_row, 0))
        link = self._create_socket_pair(out_sm["device"], device)
        logger.info(f"DSpark drafter on idle row {drafter_row}, its stash on output stage {out_sm['index']}")
        lm_head = DeepSeekV41LMHead(
            dequantized(self.loader, "lm_head.weight", translate=True),
            device,
            self.cache.file("lm_head"),
            dtype=self._lm_head_traced.weight.dtype,
        )
        drafter_kwargs = {"use_prefetcher": self.use_prefetcher, **drafter_kwargs}
        self.dspark = DeepSeekV41DSpark(
            cfg,
            self.loader,
            device,
            lm_head,
            rope_main,
            cache=self.cache,
            tp_size=self.tp_size,
            stash_device=out_sm["device"],
            link=link,
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
        # Where a decode step's taps, packet and logits land on the drafter (_spec_write_step).
        vocab = cfg.vocab_size
        if vocab % _LOGIT_PAGE:
            raise ValueError(f"the step's logits go to the drafter in {_LOGIT_PAGE}-wide pages, but V is {vocab}")
        drafter = self.dspark.device
        self._spec_step_in = (
            _device_zeros((1, 1, 1, len(taps) * cfg.hidden_size), drafter, layout=ttnn.ROW_MAJOR_LAYOUT),
            _device_zeros((1, 1, 1, self._pkt_w), drafter, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            _device_zeros((1, 1, vocab // _LOGIT_PAGE, _LOGIT_PAGE), drafter, layout=ttnn.ROW_MAJOR_LAYOUT),
        )
        if t > 1:
            # Where the verify step's taps and packet land on the drafter (_spec_write_verified).
            self._spec_verified_in = (
                _device_zeros((1, 1, t, len(taps) * cfg.hidden_size), self.dspark.device, layout=ttnn.ROW_MAJOR_LAYOUT),
                _device_zeros((1, 1, 1, self._pkt_w), self.dspark.device, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            )

    def _spec_fixup(self, sm: dict) -> None:
        """Undo the previous round's rejected rows on ``sm``, then snapshot its rings for this round.

        ``fix_mask`` marks the rejected positions' ring slots (``fix_keep`` is its complement); they get
        their shadow rows back. Every product is with 0 or 1, so the restore is exact. The compressor
        windows need no restore (see :meth:`prepare_spec_decode`).
        """
        for li in sm["layers"]:
            ring, shadow = sm["scaches"][li].kv, sm["ring_shadow"][li]
            ttnn.copy(ttnn.add(ttnn.multiply(ring, sm["fix_keep"]), ttnn.multiply(shadow, sm["fix_mask"])), ring)
            ttnn.copy(ring, shadow)

    def capture_spec_traces(self, token_id, pos: int) -> None:
        """Compile and then capture the decode variants, every stage's fixup and the drafter's
        ``write_main`` / ``draft`` / both ``accept_draft`` / :meth:`_spec_write_step` (and, with traced verify,
        :meth:`_spec_write_verified`).

        All compile runs come before the first capture on any chip (see :meth:`_capture_traces`). They
        write zeros or the rows of ``pos``, which prefill overwrites. With ``verify_rows > 1`` the verify
        step's traces are compiled (after the decode variants, whose run fixes the output socket's pages) and
        captured too; with traced decode off, :meth:`_compile_verify` warms the eager verify instead.
        """
        if self.spec_window_rows is None:
            raise RuntimeError("call prepare_spec_decode() first")
        verify = self.verify_rows > 1 and self._lm_head_traced is not None
        if verify and self._eager_decode:
            self._compile_verify(token_id)
        self.compile_traces(token_id, pos)
        verify_variants = self._compile_verify_traces(token_id) if verify and not self._eager_decode else []
        for sm in self.submeshes_io:
            self._spec_fixup(sm)
        if self.dspark is not None:
            self.dspark.send_main()
            self.dspark.write_main()
            self.dspark.refresh_uniforms(1)
            self.dspark.draft()
            for sampled in (False, True):
                self.dspark.accept_draft(sampled)
            ttnn.synchronize_device(self.dspark.device)
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
            stash_device = self.dspark.stash_device
            tid = ttnn.begin_trace_capture(stash_device, cq_id=0)
            with _trace_capture_guard():
                self.dspark.send_main()
            ttnn.end_trace_capture(stash_device, tid, cq_id=0)
            self._spec_tids["send_main"] = tid
            device = self.dspark.device
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            with _trace_capture_guard():
                self.dspark.write_main()
            ttnn.end_trace_capture(device, tid, cq_id=0)
            self._spec_tids["write_main"] = tid
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            with _trace_capture_guard():
                self.dspark.draft()
            ttnn.end_trace_capture(device, tid, cq_id=0)
            self._spec_tids["draft"] = tid
            for sampled in (False, True):
                tid = ttnn.begin_trace_capture(device, cq_id=0)
                with _trace_capture_guard():
                    self.dspark.accept_draft(sampled)
                ttnn.end_trace_capture(device, tid, cq_id=0)
                self._spec_tids["accept_sampled" if sampled else "accept_greedy"] = tid
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            with _trace_capture_guard():
                self._spec_write_step()
            ttnn.end_trace_capture(device, tid, cq_id=0)
            self._spec_tids["write_step"] = tid
            if verify_variants:
                tid = ttnn.begin_trace_capture(device, cq_id=0)
                with _trace_capture_guard():
                    self._spec_write_verified()
                ttnn.end_trace_capture(device, tid, cq_id=0)
                self._spec_tids["write_verified"] = tid
        self._capture_verify_traces(verify_variants)

    def spec_fixup(self, rejected) -> None:
        """Queue every stage's fixup ahead of a verify round: restore the ring slots of the previous round's
        ``rejected`` positions."""
        self._wait_replays_dispatched()
        window = self.sliding_window
        mask = torch.zeros(1, 1, window, 1)
        for p in rejected:
            mask[0, 0, p % window, 0] = 1
        for sm in self.submeshes_io:
            host_to_device(mask, sm["fix_mask"])
            host_to_device(1 - mask, sm["fix_keep"])
            ttnn.execute_trace(sm["device"], sm["fix_tid"], cq_id=0, blocking=False)

    def spec_write_main(self, positions) -> None:
        """Queue the drafter's ``write_main`` for verified ``positions``, rejected ones included (the draft
        masks rows past its frontier). Their stash rows must still be in the stash."""
        self._wait_replays_dispatched()
        self.dspark.set_main_rows(positions)
        ttnn.execute_trace(self.dspark.stash_device, self._spec_tids["send_main"], cq_id=0, blocking=False)
        ttnn.execute_trace(self.dspark.device, self._spec_tids["write_main"], cq_id=0, blocking=False)

    def spec_verify_async(self, tokens, start: int) -> None:
        """Dispatch the pipelined verify of ``tokens`` at ``start, start + 1, ..`` without reading anything back,
        for the caller to read the steps' outputs (:meth:`read_decoded_output`) and accept row by row.

        With a drafter, the drafter's receive of each step's row is held: the caller queues them in step order
        (:meth:`spec_release_writes`), one before reading each output, so a draft queued between two releases
        sees exactly the rows released before it. Every step must be released and read eventually: its output
        stage waits on both."""
        if self._eager_decode:
            raise RuntimeError("a streamed verify replays the decode traces: build with traced decode")
        tokens = list(tokens)
        positions = range(start, start + len(tokens))
        self.ensure_session_capacity(positions[-1])
        self._spec_hold_writes = True
        try:
            self.replay_traced_ahead(positions)
        finally:
            self._spec_hold_writes = False
        for token, pos in zip(tokens, positions):
            self.write_step_packet(token, pos)

    def spec_release_writes(self, steps: int = 1) -> None:
        """Queue the drafter's receive of the next ``steps`` held rows (:meth:`spec_verify_async`)."""
        with self._cq_lock:
            for _ in range(steps):
                ttnn.execute_trace(self.dspark.device, self._spec_tids["write_step"], cq_id=0, blocking=False)

    def spec_discard_outputs(self, steps: int) -> None:
        """Mark the next ``steps`` unread outputs as unwanted: the next read drops them first."""
        self._stale_outputs += steps

    def drain_stale_outputs(self) -> None:
        """Read and drop the outputs :meth:`spec_discard_outputs` marked, so no step is left waiting on the host."""
        while self._stale_outputs:
            self._stale_outputs -= 1
            super().read_decoded_output()

    def read_decoded_output(self) -> torch.Tensor:
        """V4's :meth:`~DeepSeekV4Model.read_decoded_output`, past the stale outputs (:meth:`drain_stale_outputs`)."""
        self.drain_stale_outputs()
        return super().read_decoded_output()

    def _send_step_row(self, tapped: dict, taps: dict, pkt: ttnn.Tensor) -> None:
        """On the output stage, inside a decode step: send the step's target-layer means ``[1, 1, 1, targets * D]``
        (``tapped`` by layer, in the stash's target order) and its packet to the drafter, which writes them into
        its rings (:meth:`_spec_write_step`), rejected verify rows included.

        The sends wait for the drafter's receives. A replay queues those as the drafter's trace
        (:meth:`_execute_traces`, or :meth:`spec_release_writes` for a held step); a compile run or an eager
        step queues them eagerly right here: the stage stalls on this send, and with it the host's enqueue of
        the rest of the step once its queue fills. A capture records only the sends.
        """
        rows = ttnn.concat([tapped[target] for target in taps], dim=-1)
        sender = self.dspark.link[0]
        ttnn.experimental.send_direct_async(rows, sender)
        ttnn.experimental.send_direct_async(pkt, sender)
        rows.deallocate(True)
        for part in tapped.values():
            part.deallocate(True)
        if not v4_common._IN_TRACE_CAPTURE:
            self._spec_write_step()

    def _send_step_logits(self, logits: ttnn.Tensor) -> None:
        """On the output stage, after :meth:`_send_step_row`: send the step's logits ``[1, 1, 1, V]`` to the
        drafter, as ``[1, 1, V / 2020, 2020]`` ROW_MAJOR pages, for its device accept. :meth:`_spec_write_step`
        already waits for them.

        Reshaped before the untilize, as in :meth:`_send_output`: untilizing one vocab-wide row needs circular
        buffers past the L1 the stage's prefetcher rings leave."""
        pages = ttnn.to_layout(ttnn.reshape(logits, [1, 1, -1, _LOGIT_PAGE]), ttnn.ROW_MAJOR_LAYOUT)
        ttnn.experimental.send_direct_async(pages, self.dspark.link[0])
        pages.deallocate(True)

    def _spec_write_step(self) -> None:
        """On the drafter: receive one decode step's row, packet and logits (:meth:`_send_step_row`,
        :meth:`_send_step_logits`), write the row's ``main_kv`` into the drafter rings at the packet's position and
        drafter ring row (:meth:`_build_packet`) and keep the logits for the accept."""
        rows, pkt, logits = self._spec_step_in
        receiver = self.dspark.link[1]
        ttnn.experimental.recv_direct_async(rows, receiver)
        ttnn.experimental.recv_direct_async(pkt, receiver)
        positions = ttnn.typecast(ttnn.reshape(ttnn.slice(pkt, [0, 0, 0, 2], [1, 1, 1, 3]), [1, 1]), ttnn.uint32)
        ring_row = ttnn.slice(pkt, [0, 0, 0, 6], [1, 1, 1, 7])
        self.dspark.write_rows(rows, positions, ttnn.reshape(ring_row, [1]))
        ttnn.experimental.recv_direct_async(logits, receiver)
        self.dspark.write_logits(ttnn.reshape(logits, [1, 1, 1, self.config.vocab_size]), ring_row)

    def _spec_write_verified(self) -> None:
        """On the drafter: receive what the output stage's verify step sends after its last tap (the block's
        target-layer means and its packet, see :meth:`_verify_submesh_static`) and write the block into the
        drafter rings, rejected rows included, as :meth:`spec_write_main` would for its positions."""
        rows, pkt = self._spec_verified_in
        receiver = self.dspark.link[1]
        ttnn.experimental.recv_direct_async(rows, receiver)
        ttnn.experimental.recv_direct_async(pkt, receiver)
        t = self.verify_rows
        (pos_at, _), (slot_at, _) = self._verify_fields["pos"], self._verify_fields["dspark_slot"]
        positions = ttnn.typecast(
            ttnn.reshape(ttnn.slice(pkt, [0, 0, 0, pos_at], [1, 1, 1, pos_at + t]), [1, t]), ttnn.uint32
        )
        slots = ttnn.reshape(ttnn.slice(pkt, [0, 0, 0, slot_at], [1, 1, 1, slot_at + t]), [t])
        self.dspark.write_rows(rows, positions, slots)

    def spec_seed(self, positions) -> None:
        """Write every one of ``positions`` into the drafter rings, ``main_rows`` at a time."""
        positions = list(positions)
        rows = self.dspark.main_rows
        for i in range(0, len(positions), rows):
            self.spec_write_main(positions[i : i + rows])

    def spec_draft(
        self, token: int, frontier: int, seed: Optional[int] = None, logits: bool = False
    ) -> tuple[list[int], list[float], Optional[torch.Tensor]]:
        """Draft from ``token``, the next input after the last verified position ``frontier``: ``block_size``
        proposals for positions ``frontier + 2 ..``, their confidence logits and, with ``logits``, the rows'
        logits ``[block_size, V]``. The drafter rings must hold every verified position up to ``frontier``.

        The draft samples at the drafter's temperature (``dspark.set_sampling``), from uniforms drawn on
        device with ``seed`` first (``None`` keeps the last draw, which a greedy draft ignores)."""
        self._wait_replays_dispatched()
        d = self.dspark
        d.set_block(token, frontier)
        if seed is not None:
            d.refresh_uniforms(seed)
        ttnn.execute_trace(d.device, self._spec_tids["draft"], cq_id=0, blocking=False)
        return self._read_draft(logits)

    def spec_accept_draft(
        self, start: int, drafts: int, sampled: bool, seed: Optional[int] = None
    ) -> tuple[int, int, list[int]]:
        """Accept the round just verified at ``start .. start + drafts`` against the last draft and draft on
        from the result, on device (``dspark.accept_draft``): ``(m, token, proposals)``, ``m`` the drafts
        accepted, ``token`` the next input (at ``start + m + 1``) and the ``block_size`` proposals after it.
        ``sampled`` picks the sampled accept over the greedy one; ``seed`` draws fresh uniforms first."""
        self._wait_replays_dispatched()
        d = self.dspark
        d.set_round(start, drafts)
        if seed is not None:
            d.refresh_uniforms(seed)
        tid = self._spec_tids["accept_sampled" if sampled else "accept_greedy"]
        ttnn.execute_trace(d.device, tid, cq_id=0, blocking=False)
        accept, ids = (ttnn.to_torch(ttnn.get_device_tensors(d.out[name])[0]).flatten() for name in ("accept", "ids"))
        m, token = (int(v) for v in accept)
        return m, token, [int(i) for i in ids]

    def _read_draft(self, logits: bool) -> tuple[list[int], list[float], Optional[torch.Tensor]]:
        """The last draft's proposals, confidence logits and (with ``logits``) rows' logits, off the drafter."""
        d = self.dspark
        ids, conf = (ttnn.to_torch(ttnn.get_device_tensors(d.out[name])[0]) for name in ("ids", "conf"))
        rows = None
        if logits:
            rows = ttnn.to_torch(ttnn.get_device_tensors(d.out["rows"])[0]).float().reshape(d.block, -1)
        return [int(i) for i in ids.flatten()], [float(c) for c in conf.flatten()], rows

    # -- DSpark parallel verify: one step over verify_rows consecutive positions -- #
    def _verify_groups(self, sm: dict) -> set[int]:
        """The compressed groups (by kv source) ``sm``'s layers read."""
        return {self._kv_source_of[li] for li in sm["layers"] if self.config.compress_ratios[li]}

    def _group_holder(self, sm: dict, src: int):
        """The cache on ``sm`` whose ``sel_tile_rows`` / ``new_kv_rows`` / ``new_key_rows`` hold group ``src``'s
        block: the kv source's own when it is on ``sm``, else that of the group's first layer there."""
        if src in sm["layers"]:
            return sm["scaches"][src]
        return next(
            sm["scaches"][li]
            for li in sm["layers"]
            if self.config.compress_ratios[li] and self._kv_source_of[li] == src
        )

    def _verify_variant(self, pos: int) -> tuple[bool, bool]:
        """``(causal, odd)`` of the verify block at ``pos``: whether the compressed layers bound the KV axis with
        ``cur_pos`` rather than a mask, and whether the block starts at an odd position (which rows close an
        r=2 group)."""
        return pos >= self.sliding_window - 1, pos % 2 == 1

    def _verify_closing(self, ratio: int, odd: bool) -> tuple[int, ...]:
        """The rows of a verify block that close a ``ratio`` group (1 or 2), for a block starting at an odd
        position or not."""
        return tuple(j for j in range(self.verify_rows) if (int(odd) + j + 1) % ratio == 0)

    def _prepare_verify_traces(self) -> None:
        """Allocate what the traced verify step reads beyond the eager step's buffers.

        Every stage gets the RoPE tables of its families, bf16 ROW_MAJOR ``[L, Rd]`` cos / sin / -sin looked up
        by position (so a row equals the eager step's host-built one), and a stage entered from another one the
        receive buffers of the streams ``[1, T, hc, D]`` and ``pre_mix`` ``[1, T, 1, hc]``. A group whose new
        rows cross an edge (``_row_groups``) gets ROW_MAJOR ``[1, 1, n, Dh]`` / ``[1, 1, n, index_head_dim]``
        receive buffers for each count ``n`` of groups a block closes.

        A block of at least ``ratio`` rows closes a group of every ratio and selects, so its programs depend
        only on :meth:`_verify_variant`; ratios above 2 would add phases.
        """
        cfg, t = self.config, self.verify_rows
        ratios = {int(r) for r in cfg.compress_ratios[: self.num_layers] if int(r)}
        if not ratios <= {1, 2} or t < max(ratios, default=1):
            raise NotImplementedError(f"traced verify covers compress ratios 1 and 2 in {t} rows, got {sorted(ratios)}")
        hc, hidden = cfg.hc_mult, cfg.hidden_size
        tables = {}
        for family, (cos_half, sin_half) in self._rope_halves.items():
            cos, sin = make_rope_table(cos_half, sin_half)
            tables[family] = [x.reshape(x.shape[-2], x.shape[-1]) for x in (cos, sin, -sin)]
        by_stage = {sm["index"]: sm for sm in self.submeshes_io}
        self._verify_edges = {
            (a, b): sorted(self._verify_groups(by_stage[a]) & self._verify_groups(by_stage[b]))
            for a, b in self.pipeline_edges
        }
        for sm in self.submeshes_io:
            sm["verify_rope"] = {
                family: [
                    ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=sm["device"])
                    for x in tables[family]
                ]
                for family in sm["rope_invfreq"]
            }
            sm["verify_traces"], sm["verify_tids"], sm["verify_new_rows"] = {}, {}, {}
        for (_, stage), groups in self._verify_edges.items():
            sm = by_stage[stage]
            device = sm["device"]
            sm["verify_streams_in"] = _device_zeros([1, t, hc, hidden], device, layout=ttnn.ROW_MAJOR_LAYOUT)
            sm["verify_pre_mix_in"] = _device_zeros([1, t, 1, hc], device)
            for src in groups:
                if src not in self._row_groups:
                    continue
                for odd in (False, True):
                    n = len(self._verify_closing(int(cfg.compress_ratios[src]), odd))
                    if (src, n) not in sm["verify_new_rows"]:
                        sm["verify_new_rows"][(src, n)] = (
                            _device_zeros([1, 1, n, cfg.head_dim], device, layout=ttnn.ROW_MAJOR_LAYOUT),
                            _device_zeros([1, 1, n, cfg.index_head_dim], device, layout=ttnn.ROW_MAJOR_LAYOUT),
                        )

    def _verify_packet(self, ids: list[int], pos: int) -> torch.Tensor:
        """The verify block's socket page ``[1, 1, 1, _pkt_w]`` (INT32), laid out by ``_verify_fields``: per row the
        token, position, ring slot, causal row copies (:meth:`~.decode.attention.DeepSeekV41Attention._write_causal_rows`),
        sliding ``cur_pos``, stash slots and drafter ring slots, the compressor-window slot of ``pos - 1``, and per
        ratio the window slot (modulo :attr:`spec_window_rows`), select position and compressed ``cur_pos``, then the
        entry and first position of each group the block closes, in row order. These are
        :meth:`~.decode.attention.DeepSeekV41Attention.decode_inputs`' integers for ``rows=T``."""
        t, w, topk = self.verify_rows, self.sliding_window, self.config.index_topk
        win_rows = self.spec_window_rows
        positions = list(range(pos, pos + t))
        values = {
            "token": ids,
            "pos": positions,
            "sliding_pos": [p % w for p in positions],
            "ring_rows": [(pos + i) % w if i <= j else -1 for j in range(t) for i in range(t)],
            "sliding_curpos": [min(p, w - 1) for p in positions],
            "win_prev": [(pos - 1) % win_rows],
            "stash_slot": [p % self.dspark.stash_rows for p in positions] if self.dspark is not None else [],
            "dspark_slot": [p % self.dspark.ring_rows for p in positions] if self.dspark is not None else [],
        }
        for r in sorted({int(r) for r in self.config.compress_ratios[: self.num_layers] if int(r)}):
            counts = [(p + 1) // r for p in positions]
            closing = [p for p in positions if (p + 1) % r == 0]
            values[f"win_slot{r}"] = [p % (win_rows if r > 1 else r) for p in positions]
            values[f"entry{r}"] = [p // r for p in closing]
            values[f"start{r}"] = [p + 1 - r for p in closing]
            values[f"index_pos{r}"] = [_SELECT_KV_RATE * max(n, 1) - 1 for n in counts]
            values[f"curpos{r}"] = [w + min(n, topk) - 1 for n in counts]
        packet = torch.zeros(1, 1, 1, self._pkt_w, dtype=torch.int32)
        for name, (offset, _) in self._verify_fields.items():
            row = values[name]
            packet[0, 0, 0, offset : offset + len(row)] = torch.tensor(row, dtype=torch.int32)
        return packet

    def _verify_context(self, sm: dict, pkt: ttnn.Tensor, causal: bool, odd: bool) -> dict:
        """:meth:`_step_context` for a verify block: the per-row layer inputs ``sm`` needs, from the packet.

        Integers are slices of the packet. RoPE rows are looked up in ``sm["verify_rope"]``: ROW_MAJOR
        ``[1, 1, T, Rd]`` at the rows' positions, TILE ``[1, 1, n, Rd]`` at the starts of the ``n`` closing
        groups. A masked variant builds each ratio's ``[T, 1, 1, window + index_topk]`` mask from the positions.
        """
        t = self.verify_rows

        def field(name: str, n: Optional[int] = None) -> ttnn.Tensor:  # INT32 ROW_MAJOR [n]
            offset, size = self._verify_fields[name]
            n = size if n is None else n
            return ttnn.reshape(ttnn.slice(pkt, [0, 0, 0, offset], [1, 1, 1, offset + n]), [n])

        def rope_rows(tables, name: str, n: int) -> list[ttnn.Tensor]:  # ROW_MAJOR [1, 1, n, Rd] each
            idx = ttnn.typecast(ttnn.reshape(field(name, n), [1, n]), ttnn.uint32)
            return [
                ttnn.reshape(ttnn.embedding(idx, table, layout=ttnn.ROW_MAJOR_LAYOUT), [1, 1, n, table.shape[-1]])
                for table in tables
            ]

        ctx = {
            "token": ttnn.typecast(ttnn.reshape(field("token"), [1, t]), ttnn.uint32),
            "sliding_pos": field("sliding_pos"),
            "ring_rows": field("ring_rows"),
            "rope": {family: rope_rows(tables, "pos", t) for family, tables in sm["verify_rope"].items()},
            "sliding_curpos": field("sliding_curpos") if sm["has_sliding"] else None,
            "page_row": sm["verify_page_row"],
            "win_prev": None,  # bf16 one-hot [1, 1, 1, spec_window_rows] of the slot of pos - 1
            "stash_slot": field("stash_slot") if sm.get("taps") else None,
            "mask": {},
            "curpos": {},
            "win_slot": {},
            "win_rope": {},
            "entry": {},
            "index_pos": {},
        }
        if sm["deep_windows"]:
            prev = ttnn.typecast(
                ttnn.to_layout(ttnn.reshape(field("win_prev"), [1, 1, 1, 1]), ttnn.TILE_LAYOUT), ttnn.float32
            )
            ctx["win_prev"] = ttnn.typecast(ttnn.eq(sm["win_cols"], prev), ttnn.bfloat16)
        if sm["ratios"] and not causal:
            pos_row = ttnn.typecast(
                ttnn.to_layout(ttnn.reshape(field("pos"), [1, 1, 1, t]), ttnn.TILE_LAYOUT), ttnn.float32
            )
            pos_f = ttnn.transpose(pos_row, -2, -1)  # [1, 1, T, 1]
        for ratio in sm["ratios"]:
            if causal:
                ctx["curpos"][ratio] = field(f"curpos{ratio}")
            else:
                mask = self._compressed_mask(sm["mask_idx"], pos_f, ratio)  # [1, 1, T, window + index_topk]
                ctx["mask"][ratio] = ttnn.reshape(mask, [t, 1, 1, mask.shape[-1]])
        for ratio in sorted(set(sm["kv_ratios"]) | set(sm["row_ratios"])):
            ctx["entry"][ratio] = field(f"entry{ratio}", len(self._verify_closing(ratio, odd)))
        for ratio in sm["kv_ratios"]:
            if ratio > 1:
                ctx["win_slot"][ratio] = field(f"win_slot{ratio}")
            n = len(self._verify_closing(ratio, odd))
            ctx["win_rope"][ratio] = [
                ttnn.to_layout(x, ttnn.TILE_LAYOUT)
                for x in rope_rows(sm["verify_rope"]["compress"][:2], f"start{ratio}", n)
            ]
        for ratio in sm["index_ratios"]:
            ctx["index_pos"][ratio] = field(f"index_pos{ratio}")
        return ctx

    def _verify_attention_inputs(self, attn, ctx: dict, causal: bool, odd: bool) -> dict:
        """What :meth:`_decode_verify_eager` passes ``attn``'s decode, from the device-built ``ctx``."""
        ratio = attn.compress_ratio
        cos, sin, neg_sin = ctx["rope"]["main" if ratio == 0 else "compress"]
        inputs = {
            "cos": cos,
            "sin": sin,
            "neg_sin": neg_sin,
            "sliding_pos": ctx["sliding_pos"],
            "ring_rows": ctx["ring_rows"],
        }
        if ratio == 0:
            inputs["sdpa_cur_pos"] = ctx["sliding_curpos"]
            return inputs
        inputs["page_row"] = ctx["page_row"]
        closing = self._verify_closing(ratio, odd)
        if attn.is_kv_source:
            inputs["pool_rows"] = closing
            if ratio > 1:
                inputs["win_slot"] = ctx["win_slot"][ratio]
                inputs["win_prev"] = ctx["win_prev"]
            inputs["cos_win"], inputs["sin_win"] = ctx["win_rope"][ratio]
            inputs["entry"] = ctx["entry"][ratio]
        if attn.is_index_source:
            inputs["select"] = True
            inputs["index_pos"] = ctx["index_pos"][ratio]
            if attn.layer_idx in self._row_writers:
                inputs["received_entry"] = ctx["entry"][ratio]
        if causal:
            inputs["sdpa_cur_pos"] = ctx["curpos"][ratio]
        else:
            inputs["mask"] = ctx["mask"][ratio]
        return inputs

    def _verify_submesh_static(self, sm: dict, causal: bool, odd: bool) -> Optional[ttnn.Tensor]:
        """One stage of the traced verify step (variant ``(causal, odd)``, see :meth:`_verify_variant`).

        :meth:`_decode_submesh_static` for a ``verify_rows``-row block, with :meth:`_decode_verify_eager`'s layer
        loop. The first stage takes the packet off the H2D socket; entering a later stage receives the streams,
        the packet, ``pre_mix`` and each group of ``_verify_edges``: its ``sel_tile_rows`` and, for a group in
        ``_row_groups``, its new rows ROW_MAJOR into ``verify_new_rows``, which every layer of the group then reads.
        Leaving sends the same. The last layer's logits ``[1, 1, T, V]`` go out the D2H socket in the decode
        output's pages, and are returned. On the output stage, the block's target-layer means and the packet go to
        the drafter (:meth:`_spec_write_verified`) right after the last tap.
        """
        cfg, t, stage = self.config, self.verify_rows, sm["index"]
        hc, hidden = cfg.hc_mult, cfg.hidden_size
        ids, layers = self.layer_submesh_ids, sm["layers"]
        first, last = layers[0], layers[-1]
        if self.use_prefetcher:
            self.layers[first].prefetch_weights()
        for li in layers:
            if cfg.compress_ratios[li]:
                sm["scaches"][li].new_kv_rows = sm["scaches"][li].new_key_rows = None
        if first == 0:
            pkt = sm["pkt"]
            ttnn.experimental.recv_async_h2d(pkt, self._pkt_socket)
            if self.tp_size > 1:
                pkt = ttnn.broadcast(pkt, ttnn.MeshCoordinate(0, 0), cluster_axis=1, topology=CCL_TOPOLOGY)
        else:
            edge = (ids[first - 1], stage)
            _, receiver = self.submesh_socket_pairs[edge]
            ttnn.experimental.recv_direct_async(sm["verify_streams_in"], receiver)
            ttnn.experimental.recv_direct_async(sm["pkt_in"], receiver)
            ttnn.experimental.recv_direct_async(sm["verify_pre_mix_in"], receiver)
            for src in self._verify_edges[edge]:
                ttnn.experimental.recv_direct_async(self._group_holder(sm, src).sel_tile_rows, receiver)
                if src in self._row_groups:
                    n = len(self._verify_closing(int(cfg.compress_ratios[src]), odd))
                    kv_rows, key_rows = sm["verify_new_rows"][(src, n)]
                    ttnn.experimental.recv_direct_async(kv_rows, receiver)
                    ttnn.experimental.recv_direct_async(key_rows, receiver)
                    for li in layers:
                        if self._kv_source_of.get(li) == src:
                            sm["scaches"][li].new_kv_rows, sm["scaches"][li].new_key_rows = kv_rows, key_rows
            pkt = sm["pkt_in"]
            streams = ttnn.to_memory_config(sm["verify_streams_in"], sm["verify_streams_memory_config"])
            pre_mix = sm["verify_pre_mix_in"]
        ctx = self._verify_context(sm, pkt, causal, odd)
        if first == 0:
            with _region("EMBED"):
                embedded = ttnn.embedding(
                    ctx["token"], self.embed_tokens.embedding_weight, layout=ttnn.ROW_MAJOR_LAYOUT
                )
            streams = ttnn.repeat(
                ttnn.reshape(embedded, [1, t, 1, hidden]),
                ttnn.Shape([1, 1, hc, 1]),
                memory_config=sm["verify_streams_memory_config"],
            )
            pre_mix = sm["verify_pre_mix_init"]
        persistent = {sm.get("verify_pre_mix_init"), sm.get("verify_pre_mix_in")}
        tapped = {}
        for li in layers:
            layer = self.layers[li]
            if self.use_prefetcher and li != last:
                self.layers[li + 1].prefetch_weights()
            if li in sm.get("taps", {}):
                row = self._collapse_streams(streams, sm["tap_weights"])  # [1, T, 1, D]
                row = ttnn.reshape(ttnn.to_layout(row, ttnn.ROW_MAJOR_LAYOUT), [1, 1, t, hidden])
                for j, piece in enumerate(sm["taps"][li]):
                    w = piece.shape[-1]
                    part = _rows_on_one_core(ttnn.slice(row, [0, 0, 0, j * w], [1, 1, t, (j + 1) * w]))
                    ttnn.experimental.paged_update_cache(piece, part, update_idxs_tensor=ctx["stash_slot"])
                    ttnn.deallocate(part)
                tapped[li] = ttnn.to_memory_config(row, ttnn.DRAM_MEMORY_CONFIG)
                if len(tapped) == len(sm["taps"]):
                    # The whole block goes to the drafter's _spec_write_verified, in the stash's target order.
                    rows = ttnn.concat([tapped[target] for target in sm["taps"]], dim=-1)
                    ttnn.experimental.send_direct_async(rows, self.dspark.link[0])
                    ttnn.experimental.send_direct_async(pkt, self.dspark.link[0])
                    rows.deallocate(True)
                    for part in tapped.values():
                        part.deallocate(True)
            streams, pre_mix = layer.decode(
                streams,
                pre_mix,
                sm["scaches"][li],
                **self._verify_attention_inputs(layer.self_attn, ctx, causal, odd),
            )
        if last == self.num_layers - 1:
            collapsed = self._collapse_streams(streams, pre_mix)  # [1, T, 1, D]
            logits = self._lm_head_traced(self.norm(ttnn.reshape(collapsed, [1, 1, t, hidden])))
            # The decode output's pages: _send_output's staging, then T outputs' rows of the socket's width.
            rows, cols = self._out_plan
            out_rm = ttnn.to_layout(ttnn.reshape(logits, [1, 1, -1, 2020]), ttnn.ROW_MAJOR_LAYOUT)
            ttnn.experimental.send_async_d2h(ttnn.reshape(out_rm, [t * rows, cols]), self._out_socket)
            return logits
        edge = (stage, ids[last + 1])
        sender, _ = self.submesh_socket_pairs[edge]
        streams_rm = ttnn.to_memory_config(streams, ttnn.DRAM_MEMORY_CONFIG)
        pre_send = ttnn.to_memory_config(pre_mix, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.experimental.send_direct_async(streams_rm, sender)
        ttnn.experimental.send_direct_async(pkt, sender)
        ttnn.experimental.send_direct_async(pre_send, sender)
        for src in self._verify_edges[edge]:
            holder = self._group_holder(sm, src)
            ttnn.experimental.send_direct_async(holder.sel_tile_rows, sender)
            if src in self._row_groups:
                for rows in (holder.new_kv_rows, holder.new_key_rows):
                    # The receive buffers are ROW_MAJOR; a source stage's rows come out of its layers tiled.
                    rows = ttnn.to_memory_config(rows, ttnn.DRAM_MEMORY_CONFIG)
                    if rows.layout != ttnn.ROW_MAJOR_LAYOUT:
                        rows = ttnn.to_layout(rows, ttnn.ROW_MAJOR_LAYOUT)
                    ttnn.experimental.send_direct_async(rows, sender)
        streams.deallocate()
        streams_rm.deallocate()
        if pre_mix not in persistent and pre_mix is not pre_send:
            pre_mix.deallocate()
        if pre_send not in persistent:
            pre_send.deallocate()
        return None

    def _read_verify_output(self) -> torch.Tensor:
        """The verify block's logits ``[T, V]`` off the D2H socket (see :meth:`_verify_submesh_static`)."""
        rows, cols = self._out_plan
        out = torch.empty(self.verify_rows * rows, cols, dtype=self._out_torch_dtype)
        self._out_socket.read_tensor(out)
        return out.reshape(self.verify_rows, -1)

    def _run_verify(self, dispatch: Callable[[], None], ids: list[int], pos: int, engram_rows: dict) -> torch.Tensor:
        """``dispatch`` every stage's verify step (its programs or its traces), feed it the block's packet and
        Engram rows ``{layer: [T, kin]}``, and return its logits ``[T, V]``.

        A block's rows outnumber an Engram socket's pages and a write blocks until the device drains a page, so
        the packet and the rows go from a second thread while this one dispatches (``write_tensor`` drops the GIL).
        """
        packet = self._verify_packet(ids, pos)

        def feed() -> None:
            self._pkt_socket.write_tensor(packet)
            for layer_idx, engram in self._engram_layers:
                engram.write_rows(engram_rows[layer_idx])

        writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="verify_feed")
        try:
            fed = writer.submit(feed)
            dispatch()
            logits = self._read_verify_output()
            fed.result()
            return logits
        finally:
            # A stage that failed before its recv_async_h2d leaves the write blocked for good; do not wait on it.
            writer.shutdown(wait=False)

    def _compile_verify_traces(self, token_id) -> list[tuple[bool, bool]]:
        """Pass 1 of the verify traces: one executed run of :meth:`_verify_submesh_static` on every stage per
        reachable variant, at a block start of that variant, then rewind the resident session. Returns the
        variants. Like :meth:`_compile_plan`, every run must come before the first capture on any chip."""
        t, window = self.verify_rows, self.sliding_window
        starts = {(False, False): 0, (False, True): 1, (True, True): window - 1, (True, False): window}
        variants = []
        for variant, start in starts.items():
            if self._verify_variant(start) != variant or start + t > self._decode_max_seq:
                continue
            if self._two_level(start + t - 1):
                continue
            variants.append(variant)
            self.ensure_session_capacity(start + t - 1)
            logger.info(f"[traced-verify] compiling variant causal={variant[0]} odd={variant[1]} at {start}")
            outs = []
            zeros = {li: torch.zeros(t, engram.kin, dtype=torch.bfloat16) for li, engram in self._engram_layers}

            def dispatch() -> None:
                outs.extend(self._verify_submesh_static(sm, *variant) for sm in self.submeshes_io)
                if self.dspark is not None:
                    self._spec_write_verified()

            self._run_verify(dispatch, [int(token_id)] * t, start, zeros)
            for out in outs:
                if out is not None:
                    out.deallocate(True)
        self.reset_session(self._resident[0])
        return variants

    def _capture_verify_traces(self, variants: list[tuple[bool, bool]]) -> None:
        """Pass 2: capture each stage once per variant bit its layers branch on (compressed layers on
        ``causal``, r=2 layers on ``odd``) and bind every variant to its trace."""
        for causal, odd in variants:
            for sm in self.submeshes_io:
                key = (causal and sm["has_compressed"], odd and 2 in sm["ratios"])
                if key not in sm["verify_traces"]:
                    logger.info(f"[traced-verify] capturing submesh {sm['index']} causal={key[0]} odd={key[1]}")
                    tid = ttnn.begin_trace_capture(sm["device"], cq_id=0)
                    with _trace_capture_guard():
                        out = self._verify_submesh_static(sm, causal, odd)
                    ttnn.end_trace_capture(sm["device"], tid, cq_id=0)
                    # The last stage's logits are persistent: every replay rewrites them in place.
                    sm["verify_traces"][key] = (tid, out)
                sm["verify_tids"][(causal, odd)] = sm["verify_traces"][key][0]

    def _compile_verify(self, token_id) -> None:
        """Run :meth:`decode_verify` once at each block start that takes its own set of programs, then rewind
        the resident session to position 0.

        A program keeps some buffers for as long as it is cached (a tiled reshape's page map, a CCL's
        scratch), allocated on its first call. Allocated once a trace exists, one can sit where the trace's
        intermediates go, and the next replay overwrites it: a reshape then reads a garbage page map and hangs.
        Verify runs eagerly between replays, so its programs are cached here, before any capture.

        The programs vary with which rows close a group (``pos`` mod each compress ratio) and whether the
        block starts past the sliding window; the starts below cover each once.
        """
        t, window = self.verify_rows, self.sliding_window
        max_seq = self._rope_halves["main"][0].shape[0]
        starts = {0, window - 1, window}
        for ratio in {int(r) for r in self.config.compress_ratios if r > 1}:
            # Row j of the block at ratio - 1 - j closes the first group.
            starts.update(ratio - 1 - j for j in range(min(t, ratio)))
        for start in sorted(starts):
            if start + t <= max_seq and not self._two_level(start + t - 1):
                self.decode_verify([token_id] * t, start)
        self.reset_session(self._resident[0])

    def decode_verify(self, token_ids, pos: int) -> torch.Tensor:
        """One parallel-verify step: the ``verify_rows`` tokens ``token_ids`` at positions ``pos ..``, all of
        them through every layer at once. Returns their logits ``[verify_rows, V]``; row ``j``'s argmax is the
        target's token for ``pos + j + 1``.

        Row ``j`` sees what the single-row step at ``pos + j`` would after rows ``0 .. j - 1``, and the step
        leaves every cache, the speculative state (rings since the last :meth:`spec_fixup`, stash) and the
        Engram hash as ``verify_rows`` single-row steps do, so a round's :meth:`spec_fixup` is unchanged. With a
        drafter whose traces are captured, the step also writes its rows into the drafter rings, rejected ones
        included: the replay hands them over device to device (:meth:`_spec_write_verified`), queued with the
        verify and before its logits are read; the eager step follows with :meth:`spec_write_main`.

        Once :meth:`capture_spec_traces` has captured the verify traces it replays them, like a traced decode
        step: the packet (:meth:`_verify_packet`) carries the rows' integers, each stage builds its inputs on
        device and the streams, ``pre_mix`` and shared groups cross the stage edges over the D2D sockets.
        Otherwise it runs eagerly (:meth:`_decode_verify_eager`). A block whose last row selects two-level
        raises; verify it with :meth:`decode_prompt_traced` instead.
        """
        t = self.verify_rows
        ids = [int(i) for i in token_ids]
        if t < 2 or "verify_page_row" not in self.submeshes_io[0]:
            raise RuntimeError("build with verify_rows > 1 and call prepare_spec_decode() first")
        if len(ids) != t:
            raise ValueError(f"a verify step takes {t} tokens, got {len(ids)}")
        if self._two_level(pos + t - 1):
            raise NotImplementedError(f"the block at {pos} reaches the two-level select, which decodes one row")
        if self._lm_head_traced is None:
            raise RuntimeError("decode_verify needs the lm_head passed to prepare_static_decode()")
        self._wait_replays_dispatched()
        sid = self._resident[0]
        with self._cq_lock:
            self.ensure_session_capacity(pos + t - 1, sid)
            self._session_pos[sid] = pos + t
            engram_rows = {}
            if self._engram_layers:
                lookup = self._engram_lookup
                # One run of one session: [1, T, n_engram_layers, n_hash_cols].
                hashes = lookup.hash_session(self._slot_of.get(sid, 0), torch.tensor([ids], dtype=torch.long), pos)
                for layer_idx, _ in self._engram_layers:
                    ids_of_layer = hashes[:, :, lookup.layout.layer_ids.index(layer_idx)]
                    engram_rows[layer_idx] = lookup.gather(layer_idx, ids_of_layer).flatten(-2)
            variant = self._verify_variant(pos)
            if variant not in self.submeshes_io[0]["verify_tids"]:
                logits = self._decode_verify_eager(ids, pos, engram_rows)
                if self.dspark is not None and "write_main" in self._spec_tids:
                    self.spec_write_main(range(pos, pos + t))
                return logits

            def replay() -> None:
                for sm in self.submeshes_io:
                    ttnn.execute_trace(sm["device"], sm["verify_tids"][variant], cq_id=0, blocking=False)
                if self.dspark is not None:
                    tid = self._spec_tids["write_verified"]
                    ttnn.execute_trace(self.dspark.device, tid, cq_id=0, blocking=False)

            return self._run_verify(replay, ids, pos, engram_rows)

    def _decode_verify_eager(self, ids: list[int], pos: int, engram_rows: dict) -> torch.Tensor:
        """:meth:`decode_verify` one stage after the other: the layers' per-row inputs are built on the host
        (:meth:`~.decode.attention.DeepSeekV41Attention.decode_inputs`), and the streams, ``pre_mix`` and the
        shared groups' selections and new rows cross the stage edges through the host."""
        carried, groups = None, {}
        last = len(self.submeshes_io) - 1
        # A block's rows outnumber the socket's pages, and a write blocks until the device drains a page, so
        # they go from this thread while the stage that drains them is dispatched (write_tensor drops the GIL).
        writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engram_rows")
        try:
            for i, sm in enumerate(self.submeshes_io):
                writes = [
                    writer.submit(self.layers[li].engram.write_rows, engram_rows[li])
                    for li in sm["layers"]
                    if self.layers[li].engram is not None
                ]
                carried = self._verify_submesh(sm, ids, pos, carried, groups)
                for write in writes:
                    write.result()
                if i == last:
                    return carried
                # The groups both stages of the edge read, as the traced step's _cache_handoff sends them.
                groups = {}
                for src in self._verify_groups(sm) & self._verify_groups(self.submeshes_io[i + 1]):
                    holder = self._group_holder(sm, src)
                    rows = None
                    if src in self._row_groups and holder.new_kv_rows is not None:
                        rows = tuple(_replica_rows(r) for r in (holder.new_kv_rows, holder.new_key_rows))
                    groups[src] = (ttnn.to_torch(ttnn.get_device_tensors(holder.sel_tile_rows)[0]), rows)
        finally:
            # A stage that failed before its recv_async_h2d leaves the write blocked for good; do not wait on it.
            writer.shutdown(wait=False)

    def _verify_submesh(self, sm: dict, ids: list[int], pos: int, carried, groups: dict):
        """One stage of :meth:`_decode_verify_eager`.

        ``carried`` is the previous stage's host ``(streams [1, T, hc, D], pre_mix [1, T, 1, hc])`` (``None``
        on the first stage, which embeds ``ids``) and ``groups`` its shared groups, kv source ->
        ``(sel_tile_rows, (new kv rows [n, Dh], new index keys [n, index_head_dim]) or None)``. Returns the
        same pair for the next stage, or the logits ``[T, V]`` after the last layer. The stage's Engram rows
        are written by the caller, alongside this dispatch.
        """
        cfg, device, t = self.config, sm["device"], self.verify_rows
        hc, hidden = cfg.hc_mult, cfg.hidden_size
        positions = range(pos, pos + t)

        def int32_rows(values):
            return ttnn.from_torch(
                torch.tensor(values, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )

        if carried is None:
            tokens = ttnn.from_torch(
                torch.tensor([ids], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )
            embedded = ttnn.embedding(tokens, self.embed_tokens.embedding_weight, layout=ttnn.ROW_MAJOR_LAYOUT)
            streams = ttnn.repeat(
                ttnn.reshape(embedded, [1, t, 1, hidden]),
                ttnn.Shape([1, 1, hc, 1]),
                memory_config=sm["verify_streams_memory_config"],
            )
            pre_mix = sm["verify_pre_mix_init"]
        else:
            host_streams, host_pre = carried
            streams = ttnn.from_torch(
                host_streams,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=sm["verify_streams_memory_config"],
            )
            pre_mix = ttnn.from_torch(host_pre, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        copied = set()
        for li in sm["layers"]:
            if not cfg.compress_ratios[li]:
                continue
            scache, src = sm["scaches"][li], self._kv_source_of[li]
            scache.new_kv_rows = scache.new_key_rows = None
            if src not in groups:
                continue
            selections, rows = groups[src]
            if src not in copied:
                host_to_device(selections, scache.sel_tile_rows)
                copied.add(src)
            if rows is not None:
                scache.new_kv_rows, scache.new_key_rows = (
                    ttnn.from_torch(
                        r.reshape(1, 1, *r.shape), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
                    )
                    for r in rows
                )
        stash_slots = int32_rows([p % self.dspark.stash_rows for p in positions]) if sm.get("taps") else None
        win_rows = self.spec_window_rows
        win_slot = win_prev = None
        if sm["deep_windows"]:
            win_slot = int32_rows([p % win_rows for p in positions])
            one_hot = torch.zeros(1, 1, 1, win_rows)
            one_hot[0, 0, 0, (pos - 1) % win_rows] = 1
            win_prev = ttnn.from_torch(one_hot, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        layers = sm["layers"]
        for li in layers:
            layer = self.layers[li]
            if self.use_prefetcher:
                # As the traced step: the stage's first layer, then each layer queues its successor.
                if li == layers[0]:
                    layer.prefetch_weights()
                if li != layers[-1]:
                    self.layers[li + 1].prefetch_weights()
            if li in sm.get("taps", {}):
                row = self._collapse_streams(streams, sm["tap_weights"])  # [1, T, 1, D]
                row = ttnn.reshape(ttnn.to_layout(row, ttnn.ROW_MAJOR_LAYOUT), [1, 1, t, hidden])
                for j, piece in enumerate(sm["taps"][li]):
                    w = piece.shape[-1]
                    part = _rows_on_one_core(ttnn.slice(row, [0, 0, 0, j * w], [1, 1, t, (j + 1) * w]))
                    ttnn.experimental.paged_update_cache(piece, part, update_idxs_tensor=stash_slots)
                    ttnn.deallocate(part)
            attn = layer.self_attn
            ratio = attn.compress_ratio
            inputs = attn.decode_inputs(pos, *self._rope_halves["main" if ratio == 0 else "compress"], rows=t)
            if ratio:
                inputs["page_row"] = sm["verify_page_row"]
                if attn.is_kv_source and ratio > 1:
                    inputs["win_slot"], inputs["win_prev"] = win_slot, win_prev
                closing = [p // ratio for p in positions if (p + 1) % ratio == 0]
                if li in self._row_writers and inputs.get("select") and closing:
                    inputs["received_entry"] = int32_rows(closing)
            streams, pre_mix = layer.decode(streams, pre_mix, sm["scaches"][li], **inputs)
            if li == self.num_layers - 1:
                collapsed = self._collapse_streams(streams, pre_mix)  # [1, T, 1, D]
                logits = self._lm_head_traced(self.norm(ttnn.reshape(collapsed, [1, 1, t, hidden])))
                return _replica_rows(logits)
        out = _replica_rows(streams).reshape(1, t, hc, hidden), _replica_rows(pre_mix).reshape(1, t, 1, hc)
        ttnn.deallocate(streams)
        return out
