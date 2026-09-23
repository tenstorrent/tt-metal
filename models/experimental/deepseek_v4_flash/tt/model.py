"""DeepSeek-V4-Flash full model: pipeline placement, weight loading and the traced,
paged multi-session decode path.

ttnn port of ``DeepseekV4Model`` from ``modular_deepseek_v4.py``: the embedding, the stack
of :class:`DeepSeekV4DecoderLayer`, the final :class:`DeepSeekV4HyperHead` stream collapse
and the model's shared final RMSNorm, driven straight off the safetensors checkpoint (via
:class:`DeepseekV4WeightLoader` + the ``quant`` dequantizers).

Shapes are written as ``[B, 1, H, Dh]``. Letters this module uses:

* ``B`` -- users decoded per step (the decode batch); 1 on the single-user path.
* ``D`` -- ``hidden_size``; ``hc`` -- ``hc_mult``, the hyper-connection residual-stream
  count, so a residual stream is ``[B, 1, hc, D]``.
* ``Rd`` -- ``qk_rope_head_dim``, the trailing RoPE slice of a head.
* ``W`` -- ``sliding_window`` (128 by default), the sliding ring's capacity in rows.
* ``cr`` -- ``compress_rates[layer_type]``, source tokens per compressed entry.
* ``F`` -- a compressor window buffer's feature width (``2 * Dh`` for CSA).
* ``N`` -- a step output's last dim (``V`` vocab with a folded-in ``lm_head``, else ``D``).

Three deviations from the reference, all forced by the on-device decode scope:

* The rotary tables are *inputs* (built host-side by the YaRN rotary in the system
  interpreter -- see ``test_full_model_decode_demo.py``), not an owned
  ``DeepseekV4RotaryEmbedding``: ttnn has no rope-init. The traced path rebuilds the
  equivalent rows on device from the position (:meth:`DeepSeekV4Model._device_rope`).
* The additive compressed-window masks are generated on device from constant index
  tables (:meth:`DeepSeekV4Model._device_mask`), since device attention consumes a plain
  additive mask.
* Every layer's weights are resident at once (the reference holds the whole stack too),
  so the real 43-layer checkpoint wants a populated weight ``cache`` or ``max_layers``.
"""

import contextlib
import math
import os
import queue
import threading
from typing import Optional

import torch
import ttnn
from loguru import logger

from .attention import (
    CSA_MAX_COMPRESSED_ENTRIES,
    PAGED_KV_LAYER_TYPES,
    _StaticLayerCache,
    build_static_layer_cache,
    dense_kv_context_limit,
    dense_kv_rows,
)
from .attention_csa import _scatter_window_rows
from .decode_prefetch import make_decode_prefetch_buffers
from .paged_cache import (
    PagedCacheFull,
    PagedGroup,
    PagedKVManager,
    PagedLayerView,
    build_groups,
    plan_pool_blocks,
)
from .common import DeepSeekV4Module, _MASK_NEG, _profile, _trace_capture_guard
from .decoder_layer import DeepSeekV4DecoderLayer
from .embedding import DeepSeekV4Embedding
from .hyperconnection import DeepSeekV4HyperHead
from .layers import DeepSeekV4RMSNorm
from .moe import DeepSeekV4HashRouter, DeepSeekV4PreloadedExperts
from .quant import dequantize_weight
from .system_config import SystemConfig, load_system_config, set_active_system_config
from .weight_cache import WeightCache, _as_cache
from .weight_loader import DeepseekV4WeightLoader


def plan_layer_placement(num_layers: int, num_devices: int, group_size: int) -> list[int]:
    """Map every layer to a device, given a *pipeline group size* (PGS).

    The devices are cut into groups of ``PGS`` consecutive devices; the layer stack is
    cut into the same number of *contiguous* chunks (one per group, capped at the layer
    count so no group is empty), and each group round-robins its own chunk over its own
    devices. Groups therefore run strictly one after another: the model is done at the
    end of the last group's chunk, and if ``num_devices`` is not a multiple of ``PGS``
    the trailing devices are left idle.

    With 40 layers on 8 devices: ``PGS=1`` -> 8 groups of one device, 5 contiguous
    layers each (device 0 owns 0-4, ..., device 7 owns 35-39); ``PGS=4`` -> 2 groups of
    four, group 0 (devices 0-3) round-robins layers 0-19 and group 1 (devices 4-7)
    layers 20-39 (``l20 -> d4``, ``l21 -> d5``, ``l24 -> d4``); ``PGS >= 8`` (or a
    non-positive PGS) -> one group over all 8 devices, i.e. plain ``li -> li % 8``.

    Returns ``[num_layers]``: index ``li`` holds the placement device's id within the
    mesh.
    """
    if num_layers <= 0 or num_devices <= 0:
        return []
    g = num_devices if group_size <= 0 else min(group_size, num_devices)
    num_groups = max(1, min(num_devices // g, num_layers))
    base, extra = divmod(num_layers, num_groups)
    ids: list[int] = []
    for gi in range(num_groups):
        count = base + (1 if gi < extra else 0)
        ids.extend(gi * g + (j % g) for j in range(count))
    return ids


def _window_indices(compress_rate: int, pos: int) -> tuple[int, int]:
    """``(slot, window)`` for the compressor at absolute ``pos``, both scalars.

    ``slot`` (``pos % cr``) is where this token's projection goes in the one-window
    buffer ``[B*cr, 1, 1, F]``, and ``window`` is the index of the window that closes at
    ``pos`` -- i.e. the entry the pool appends. ``window`` is ``-1`` before the first
    window closes, in which case nothing is pooled (see
    :meth:`DeepSeekV4Model._compressor_pool_due`).
    """
    return pos % compress_rate, (pos + 1) // compress_rate - 1


# --- Host -> device per-step packet socket (``recv_async_h2d``) ---------------- #
# The per-step input packet is streamed into the traced decode over an H2D PCIe socket,
# so the receive is a device op *inside* each submesh-0 trace rather than a host-side
# ``copy_host_to_device_tensor`` around it. The socket moves whole pages, so the packet's
# page (its single row) must be PCIe-aligned and the three meaningful INT32 slots are
# padded out to it; alignment and FIFO page count are ``pipeline.pcie_alignment`` /
# ``pipeline.h2d_fifo_pages``, and the FIFO holds many steps' packets so the host can run
# ahead of the device without blocking in ``H2DSocket::write`` while it drains.
#
# Receiver core for the packet socket, disjoint from the (0,0) / (0,1) cores the
# cross-submesh direct sockets use.
_PKT_SOCKET_CORE = (0, 2)

# --- Device -> host output socket (``send_async_d2h``) ------------------------- #
# The step's output (logits, or the pre-head hidden when no lm_head is folded in) is
# streamed back over a D2H PCIe socket by an op inside the last submesh's trace, so
# the host reads it off the socket instead of issuing a ``to_torch`` readback.
_OUT_SOCKET_CORE = (0, 3)
# The FIFO lives in physically-contiguous pinned host memory and its size is
# ``pipeline.d2h_fifo_bytes``. Without an IOMMU the driver pins a single system page at a
# time, so the FIFO is one 4 KB page minus the trailing bytes_acked counter, PCIe-aligned.
# Asking for more is not merely wasteful, it does not construct: on an IOMMU-less host
# ``D2HSocket`` fails in ``PinnedMemory`` with "Failed to pin pages for DMA buffer" for any
# size past one page.
#
# A whole output does not have to fit: both sides move one page at a time, and the sender
# kernel waits for the host to drain when it runs ahead, so the size of a transfer is
# unbounded by the FIFO. What the FIFO does bound is how far the *device* may run ahead of
# the host -- barely at all -- so a step's output has to be read back promptly, and by
# someone who is not simultaneously dispatching the next step (see
# :meth:`DeepSeekV4Model.decode_traced_async`). One output row, which *is* one socket page,
# has to fit the FIFO, so the FIFO size doubles as the page cap: the output is reshaped
# into rows of at most that size (see :func:`_d2h_page_plan`) rather than sent as one
# enormous vocab-wide page.


def _d2h_page_plan(numel: int, elem_bytes: int, page_cap_bytes: int, pcie_alignment: int) -> tuple[int, int]:
    """``(rows, cols)`` to reshape a flat ``numel`` output into for a D2H socket.

    ``send_async_d2h`` streams whole tensor pages, and a row-major tensor's page is
    one row, so the row width *is* the socket page size: it has to be PCIe-aligned
    and divide the output evenly. Returns ``[rows, cols]``, the widest such row up to
    ``page_cap_bytes``.
    """
    for cols in range(min(numel, page_cap_bytes // elem_bytes), 0, -1):
        if numel % cols == 0 and (cols * elem_bytes) % pcie_alignment == 0:
            return numel // cols, cols
    raise ValueError(
        f"cannot page a {numel}-element ({elem_bytes} B/elem) output for a D2H socket: no row width "
        f"divides it into {pcie_alignment} B-aligned pages of at most {page_cap_bytes} B"
    )


def _dspark_enabled() -> bool:
    """Whether to build the idle-row DSpark MTP link (see :class:`DeepSeekV4Model`).

    ``DEEPSEEK_V4_DSPARK=0`` turns it off. Every step of a pipeline that leaves chips
    idle otherwise taps layers 40-42 into a packed tensor, sends it over a D2D socket and
    replays an extra recv submesh on the idle row -- which only the DSpark drafting tests
    use (``test_dspark_flash_accept_rate.py`` reads it back with
    :meth:`DeepSeekV4Model.read_mtp_hiddens`).
    """
    return os.environ.get("DEEPSEEK_V4_DSPARK", "1") not in ("0", "", "false", "False")


class DeepSeekV4Model(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4Model`` (decode: the prompt is prefilled one token at a
    time by replaying :meth:`decode` / :meth:`decode_traced`).

    Builds the embedding, the ``num_hidden_layers`` decoder stack, the final
    :class:`DeepSeekV4HyperHead` and the shared RMSNorm from the checkpoint. There is no
    monolithic ``forward``: :meth:`decode` embeds the ids, expands them to the ``hc_mult``
    residual-stream stack, runs every decoder layer with the RoPE rows + additive mask
    built for that position, collapses the streams and normalises, returning
    ``[B, 1, 1, D]`` -- the reference's pre-``lm_head`` hidden state. Apply an external
    ``lm_head`` (:class:`.layers.Linear`, as the demos do) for logits ``[B, 1, 1, V]``.

    ``rope`` matches the bundle emitted by the reference rotary::

        rope["main"]    = (cos_half, sin_half)          # sliding layers
        rope["compress"]= (cos_half, sin_half)          # CSA / HCA layers
        rope["win"][cr] = (cos_half, sin_half)          # per compress-rate windows
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
        use_submeshes: bool = False,
        require_cache: bool = False,
        pipeline_group_size: Optional[int] = None,
        num_prefetch_pages: Optional[int] = None,
        system_config: Optional[SystemConfig] = None,
        tp_size: int = 1,
    ):
        """Build the V4-Flash model off the checkpoint.

        Caching: pass either a pre-built ``cache`` :class:`WeightCache` or a ``cache_dir``
        (the model builds ``WeightCache(cache_dir)`` and owns the per-layer ``layers.N`` /
        head namespacing internally). ``None`` for both disables caching, so every weight
        -- the ``[V, D]`` embedding, each layer's projections and ``[E, D]`` gate, the
        ``[2I, D]`` expert weights and the final head -- is converted from the checkpoint.

        ``require_cache=True`` asserts the converted-tile cache is fully populated: any
        tile-cached weight that would otherwise be (re)loaded from the HF checkpoint raises
        instead. The small host-side scalars (attention sinks, the HC ``scale`` triplets,
        the hash router's ``tid2eid`` table) and the locally-computed RoPE rotate matrix
        have no tile cache by design and are always materialised, so they are exempt.

        ``tp_size`` groups adjacent chips into ``1 x tp_size`` pipeline stages and forwards
        it to attention and MoE; their outputs are replicated, so every rank-to-rank socket
        carries the corresponding copy to the next stage. TP4 uses two stages (8 chips) on
        both an 8-chip mesh and a larger Galaxy mesh, the remaining chips staying idle.

        The attention projections (with their compressor) and the MoE shared expert run on
        DRISC-prefetched weights instead of a DRAM->L1 copy per call, so decode must run
        inside :meth:`prefetcher_session`. One GCB per device is shared by every prefetched
        weight on it (see :func:`make_decode_prefetch_buffers`), so the cost is 288 KB of L1
        per receiver core for the whole model rather than per layer, and the prefetcher stays
        on under TP for every projection whose per-rank B-core count still matches that GCB
        (see :class:`~.attention.DeepSeekV4Attention`).

        ``system_config`` is the per-machine tuning profile (see :mod:`.system_config`); it
        defaults to the one matching ``full_device``'s device count and supplies every
        hardware knob left unset here -- pipeline group size, prefetcher depth, socket sizes,
        weight precision, the MoE expert block size and the SDPA program config. The explicit
        arguments above still win, so a caller (or a test) can pin one value without writing
        a profile. The resolved profile is published process-wide with
        :func:`set_active_system_config` so the leaf modules built below pick up the same one.
        """
        self.config = config
        self.loader = loader
        self.device = full_device
        if tp_size < 1:
            raise ValueError(f"tp_size must be >= 1, got {tp_size}")
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
        # The DRISC weight prefetcher is always on: every decode projection streams its
        # weights through a shared GCB rather than copying DRAM -> L1 per call.
        self.use_prefetcher = True
        if num_prefetch_pages is None:
            num_prefetch_pages = system_config.prefetcher.num_prefetch_pages
        self._prefetch_buffers_by_device: dict[int, dict] = {}
        if cache is None and cache_dir is not None:
            cache = WeightCache(cache_dir)
        cache = _as_cache(cache)
        if require_cache:
            if not cache.path:
                raise ValueError(
                    "require_cache=True needs a populated cache; pass cache=WeightCache(dir) or cache_dir=..."
                )
            cache = cache.require(True)
        self.cache = cache

        self.use_submeshes = use_submeshes
        self.mesh_devices = full_device.get_num_devices()
        pipeline_devices = system_config.pipeline.resolve_num_devices(self.mesh_devices)
        # TP4 latency path: two 1x4 stages (8 chips). Extra chips on a larger mesh
        # (e.g. Galaxy32) stay idle so they do not add socket hops.
        if use_submeshes and tp_size == 4:
            pipeline_devices = min(pipeline_devices, 2 * tp_size)
        if pipeline_devices % tp_size:
            raise ValueError(f"pipeline uses {pipeline_devices} devices, which is not divisible by tp_size {tp_size}")
        self.pipeline_devices = pipeline_devices
        self.num_submeshes = pipeline_devices // tp_size

        # Layer -> submesh placement is set by the *pipeline group size* (PGS, see
        # :func:`plan_layer_placement`). ``layer_submesh_ids[li]`` is the mapping;
        # ``pipeline_submesh_ids`` lists the populated submeshes in the order the stack
        # first visits them, and ``pipeline_stages`` counts them. PGS >= num_submeshes
        # (or 0) collapses to plain round-robin over the mesh, whose dataflow is the
        # familiar ring 0 -> 1 -> ... -> (S-1) -> 0; the shipped profiles use PGS=1, i.e.
        # one contiguous slice of layers per device.
        n = config.num_hidden_layers if max_layers is None else min(max_layers, config.num_hidden_layers)
        self.num_layers = n
        if pipeline_group_size is None:
            pipeline_group_size = max(0, system_config.pipeline.group_size)
        if use_submeshes:
            self.layer_submesh_ids = plan_layer_placement(self.num_layers, self.num_submeshes, pipeline_group_size)
        else:
            self.layer_submesh_ids = [0] * self.num_layers
        self.pipeline_submesh_ids = list(dict.fromkeys(self.layer_submesh_ids))
        self.pipeline_stages = len(self.pipeline_submesh_ids)
        # Directed submesh handoffs the stack actually needs: one per distinct
        # ``(device of layer li-1) -> (device of layer li)`` transition.
        self.pipeline_edges = list(
            dict.fromkeys(
                (self.layer_submesh_ids[li - 1], self.layer_submesh_ids[li])
                for li in range(1, self.num_layers)
                if self.layer_submesh_ids[li - 1] != self.layer_submesh_ids[li]
            )
        )

        if use_submeshes:
            idle = self.mesh_devices - self.pipeline_devices
            logger.info(
                f"Using submeshes: {self.num_submeshes} x TP{tp_size} over {self.pipeline_devices} of "
                f"{self.mesh_devices} chips"
                f"{f' ({idle} left idle by pipeline.max_devices)' if idle else ''} (pipeline group size "
                f"{pipeline_group_size or self.num_submeshes}, {self.pipeline_stages} populated)"
            )
            self.submeshes = []
            if tp_size == 1:
                full_device.reshape(ttnn.MeshShape(1, self.mesh_devices))
                for i in range(self.num_submeshes):
                    self.submeshes.append(full_device.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, i)))
            else:
                full_device.reshape(ttnn.MeshShape(self.mesh_devices // tp_size, tp_size))
                for i in range(self.num_submeshes):
                    self.submeshes.append(
                        full_device.create_submesh(ttnn.MeshShape(1, tp_size), ttnn.MeshCoordinate(i, 0))
                    )
            self.first_device = self.submeshes[0]
            self.last_device = self.submeshes[-1]

            # Socket pairs between submeshes for copying hidden_states, one directed pair
            # per handoff the placement needs (``pipeline_edges``), reused for every step.
            # Under plain round-robin those edges are the ring
            # 0 -> 1 -> ... -> (S-1) -> 0 (the wrap-around included, since submesh 0 is
            # revisited for layers S, 2S, ...); with PGS=1 each device owns one
            # contiguous run, so they are just the group-to-group handoffs.
            self.submesh_socket_pairs = {}
            for from_id, to_id in self.pipeline_edges:
                self.submesh_socket_pairs[(from_id, to_id)] = self._create_socket_pair(
                    self.submeshes[from_id], self.submeshes[to_id]
                )
        else:
            self.first_device = full_device
            self.last_device = full_device

        # MTP/DSpark *link* (live on tp4_32chip): one D2D socket from the submesh that
        # owns layers 40/41/42; those three residuals are slice-written into one packed
        # tensor and sent as a single socket payload, which the drafting test reads back
        # with :meth:`read_mtp_hiddens`. Built only when the mesh is larger than the
        # target pipeline (Galaxy 32-chip TP4 leaves 24 chips idle) and all three tap
        # layers share a submesh.
        self.mtp_submesh = None
        self._mtp_sender = None
        self._mtp_receiver = None
        self._mtp_pack = None
        self._mtp_hiddens = None
        self._dspark_tap_ids = tuple(i for i in (40, 41, 42) if i < self.num_layers)
        if (
            _dspark_enabled()
            and use_submeshes
            and self.mesh_devices > self.pipeline_devices
            and len(self._dspark_tap_ids) == 3
            and len({self.layer_submesh_ids[i] for i in self._dspark_tap_ids}) == 1
        ):
            self._init_mtp_link()

        n = self.num_layers

        self.embed_tokens = DeepSeekV4Embedding(loader, self.first_device, cache=cache)

        self.layers: list[DeepSeekV4DecoderLayer] = []
        self.layer_devices: list[ttnn.MeshDevice] = []
        layer_weights = {
            li: self._build_layer_weights(li, config.layer_types[li], config.mlp_layer_types[li] == "hash_moe")
            for li in range(n)
        }
        for li in range(n):
            if self.use_submeshes:
                layer_device_id = self._submesh_id_for_layer(li)
                current_device = self.submeshes[layer_device_id]
                logger.info(f"Layer {li} is on device {layer_device_id}")
            else:
                current_device = self.device
            self.layer_devices.append(current_device)
            is_hash = config.mlp_layer_types[li] == "hash_moe"
            layer_cache = cache.sub(f"layers.{li}")
            weights = layer_weights[li]
            prefetch_buffers = self._prefetch_buffers_for(current_device, weight_dtype, num_prefetch_pages)
            gate = (
                self._hash_gate(li, prefetch_buffers=prefetch_buffers, weight_dtype=weight_dtype) if is_hash else None
            )
            experts = DeepSeekV4PreloadedExperts(
                config,
                self._expert_provider(li),
                current_device,
                dtype=weight_dtype,
                cache=layer_cache.sub("mlp"),
                tp_size=tp_size,
            )
            self.layers.append(
                DeepSeekV4DecoderLayer(
                    config,
                    li,
                    weights,
                    current_device,
                    experts=experts,
                    gate=gate,
                    cache=layer_cache,
                    weight_dtype=weight_dtype,
                    use_prefetcher=self.use_prefetcher,
                    prefetch_buffers=prefetch_buffers,
                    tp_size=tp_size,
                )
            )
            _profile(current_device)

        # The head (hc_head / norm / external lm_head) must live where the *last*
        # decoder layer's output lands, not unconditionally on the final submesh --
        # otherwise a capped (``max_layers``) stack would end on a lower submesh
        # than the head and mismatch devices.
        if self.layer_devices:
            self.last_device = self.layer_devices[-1]

        self.sliding_window = config.sliding_window
        self._decode_max_seq: Optional[int] = None
        # Tokens the dense CSA KV buffers can hold (None: no CSA layer, or not prepared).
        self._context_limit: Optional[int] = None
        # Paged multi-session decode state (see :meth:`prepare_static_decode`).
        self._paged: Optional[PagedKVManager] = None
        # Traced-decode replay state. :meth:`prepare_static_decode` fills in the buffers
        # and re-arms capture, but the thread handle and its queue are owned here so that
        # :meth:`shutdown` stays callable on a model that never prepared a traced decode
        # or that failed part-way through preparing one -- creating them
        # in ``prepare_static_decode`` would make the unwind itself raise and mask the
        # error being unwound.
        self._traced_captured = False
        self._replay_queue: queue.Queue = queue.Queue()
        self._replay_thread: Optional[threading.Thread] = None
        # H2D socket carrying the per-step input packet into the traced decode, and the
        # D2H socket carrying the step output back out (both allocated by
        # :meth:`prepare_static_decode`).
        self._pkt_socket = None
        self._out_socket = None
        self._out_plan: Optional[tuple[int, int]] = None  # (rows, cols) of one output
        self._out_torch_dtype: Optional[torch.dtype] = None
        self._paged_groups: dict[str, PagedGroup] = {}
        # The sessions occupying the batch slots, slot order (``None`` where a session
        # was closed without another taking its place). One entry unless
        # :meth:`prepare_static_decode` was given ``batch > 1``.
        self._resident: list[Optional[int]] = []
        # Absolute position each session will decode next, so a batch whose users have
        # drifted apart is caught rather than silently decoded at one user's phase.
        self._session_pos: dict[int, int] = {}
        # Users per step. Every shape below the packet is parameterised by it, and 1
        # reproduces the single-user path exactly.
        self._decode_batch = 1
        # Compressor window buffers held while a batch is not resident, one block per seat
        # group rather than per session, because a group is swapped as a unit (see
        # :meth:`activate_sessions`). Keyed by the group's ordered session ids, with
        # ``_group_of`` the reverse map and ``_resident_group`` the block currently seated.
        self._group_state: dict[tuple[int, ...], dict] = {}
        self._group_of: dict[int, tuple[int, ...]] = {}
        self._resident_group: Optional[dict] = None
        self._free_group_state: list[dict] = []
        self._empty_row: dict = {}

        self.hc_head = DeepSeekV4HyperHead(
            config,
            {
                "hc_fn": self._thunk("hc_head.hc_fn"),
                "hc_base": self._thunk("hc_head.hc_base"),
                "hc_scale": self._thunk("hc_head.hc_scale"),
            },
            self.last_device,
            cache=cache.sub("hc_head"),
        )
        self.norm = DeepSeekV4RMSNorm(
            self._thunk("norm.weight"), config.rms_norm_eps, self.last_device, cache.file("norm"), sharded=True
        )

    # -- weight plumbing (lazy dequant; a populated tile cache skips the read) -- #
    def _thunk(self, name: str):
        """Zero-arg thunk dequantizing checkpoint weight ``name`` (an HF name, i.e. one
        with ``layers.N``). Calling it returns a host ``torch.Tensor`` at the checkpoint's
        own ``[K, N]``-style shape; the thunk itself is handed to the sub-modules so a
        tile-cache hit avoids the read entirely."""
        loader = self.loader
        return lambda: dequantize_weight(loader.get_tensor(name), loader.get_scale(name))

    @staticmethod
    def _attn_keys(layer_type: str) -> list[str]:
        """The ``self_attn.*`` weight keys (relative names, sans ``layers.N``) of one
        attention: the replicated q_a/kv projections, the row-parallel o_b output, the
        norm weights and the sinks. Sliding layers have no compressor, so they get the
        first eight only; CSA/HCA layers add the four ``compressor.*`` keys, one per
        ``[K, N]`` projection plus the ``position_bias`` vector. CSA also adds the
        lightning-indexer tensors; without them the indexer is never built and the
        long-sequence trace stays on dense SDPA."""
        keys = [
            "q_a_proj.weight",
            "q_a_norm.weight",
            "q_b_proj.weight",
            "kv_proj.weight",
            "kv_norm.weight",
            "o_a_proj.weight",
            "o_b_proj.weight",
            "sinks",
        ]
        if layer_type != "sliding_attention":
            keys += [
                "compressor.kv_proj.weight",
                "compressor.gate_proj.weight",
                "compressor.kv_norm.weight",
                "compressor.position_bias",
            ]
        if layer_type == "compressed_sparse_attention":
            keys += [
                "compressor.indexer.kv_proj.weight",
                "compressor.indexer.gate_proj.weight",
                "compressor.indexer.kv_norm.weight",
                "compressor.indexer.position_bias",
                "compressor.indexer.q_b_proj.weight",
                "compressor.indexer.weights_proj.weight",
            ]
        return keys

    def _build_layer_weights(self, layer_idx: int, layer_type: str, is_hash: bool) -> dict:
        """The decoder layer's weight dict: name -> lazy dequant thunk (no tensors are
        read here). Keys are the module-relative names :class:`DeepSeekV4DecoderLayer`
        expects: ``self_attn.*`` (per :meth:`_attn_keys`), the router gate
        (``[E, D]``) plus its ``e_score_correction_bias`` -- omitted for the static
        ``hash_moe`` router, which reads its frozen ``tid2eid`` table instead -- the
        shared expert's ``gate/up`` ``[I, D]`` + ``down`` ``[D, I]``, both
        hyper-connections and both layernorms."""
        weights: dict = {}
        for k in self._attn_keys(layer_type):
            weights[f"self_attn.{k}"] = self._thunk(f"layers.{layer_idx}.self_attn.{k}")
        weights["mlp.gate.weight"] = self._thunk(f"layers.{layer_idx}.mlp.gate.weight")
        if not is_hash:
            weights["mlp.gate.e_score_correction_bias"] = self._thunk(
                f"layers.{layer_idx}.mlp.gate.e_score_correction_bias"
            )
        for k in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
            weights[f"mlp.shared_experts.{k}"] = self._thunk(f"layers.{layer_idx}.mlp.shared_experts.{k}")
        for hc in ("attn_hc", "ffn_hc"):
            for p in ("fn", "base", "scale"):
                weights[f"{hc}.{p}"] = self._thunk(f"layers.{layer_idx}.{hc}.{p}")
        for k in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            weights[k] = self._thunk(f"layers.{layer_idx}.{k}")
        return weights

    def _create_socket_pair(self, from_submesh, to_submesh):
        """Directed L1 D2D socket pair ``(sender, receiver)`` between two 1xTP submeshes,
        same core map as the pipeline handoffs: cores (0,0) and (0,1) of every rank, one
        socket each, in ``pipeline.socket_l1_bytes`` of L1 per core. Carries the pipeline
        handoff payload -- residual streams ``[B, 1, hc, D]`` row-major plus the fused
        packet ``[1,1,1,_pkt_w]`` -- and, on the tap submesh, the packed ``[B, 3, hc, D]``
        MTP residuals."""
        socket_memconfig = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, self.system_config.pipeline.socket_l1_bytes)
        socket_connections = []
        for coord in ttnn.MeshCoordinateRange(from_submesh.shape):
            socket_connections.append(
                ttnn.SocketConnection(
                    ttnn.MeshCoreCoord(coord, ttnn.CoreCoord(0, 0)),
                    ttnn.MeshCoreCoord(coord, ttnn.CoreCoord(0, 0)),
                )
            )
            socket_connections.append(
                ttnn.SocketConnection(
                    ttnn.MeshCoreCoord(coord, ttnn.CoreCoord(0, 1)),
                    ttnn.MeshCoreCoord(coord, ttnn.CoreCoord(0, 1)),
                )
            )
        socket_config = ttnn.SocketConfig(socket_connections, socket_memconfig)
        return ttnn.create_socket_pair(from_submesh, to_submesh, socket_config)

    def _init_mtp_link(self) -> None:
        """Park DSpark/MTP on the first idle 1xTP row and open one socket to it.

        Sets ``mtp_submesh`` (a 1xTP submesh on the row after the pipeline's) and the
        ``(sender, receiver)`` socket pair from the tap submesh to it; the payload that
        pair carries is the packed residuals ``[B, 3, hc, D]``. Called from
        :meth:`__init__` only under the conditions checked there.
        """
        tap_sm = self.layer_submesh_ids[self._dspark_tap_ids[0]]
        mtp_row = self.num_submeshes
        if self.tp_size > 1:
            self.mtp_submesh = self.device.create_submesh(
                ttnn.MeshShape(1, self.tp_size), ttnn.MeshCoordinate(mtp_row, 0)
            )
        else:
            self.mtp_submesh = self.device.create_submesh(
                ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, self.pipeline_devices)
            )
        self._mtp_sender, self._mtp_receiver = self._create_socket_pair(self.submeshes[tap_sm], self.mtp_submesh)
        logger.info(
            f"DSpark MTP link: pipeline submesh {tap_sm} -> idle row {mtp_row} "
            f"(layers {self._dspark_tap_ids} packed on one D2D socket)"
        )

    def _ensure_mtp_buffers(self, batch: int) -> None:
        """Persistent ``[B, 3, hc, D]`` pack on the tap submesh and recv on MTP."""
        if self.mtp_submesh is None:
            return
        n = len(self._dspark_tap_ids)
        hc, d = self.config.hc_mult, self.config.hidden_size
        shape = [batch, n, hc, d]
        if self._mtp_pack is not None and list(self._mtp_pack.shape) == shape:
            return
        tap_sm = self.layer_submesh_ids[self._dspark_tap_ids[0]]
        src = self.submeshes[tap_sm]
        zeros = torch.zeros(shape, dtype=torch.float32)
        self._mtp_pack = ttnn.from_torch(zeros, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=src)
        self._mtp_hiddens = ttnn.from_torch(
            zeros, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mtp_submesh
        )

    def _slice_write_mtp_hidden(self, streams, slot: int) -> None:
        """Copy one layer residual ``[B, 1, hc, D]`` into slot ``slot`` of the pack."""
        packed = ttnn.to_layout(streams, ttnn.ROW_MAJOR_LAYOUT)
        b, _, hc, d = packed.shape
        ttnn.experimental.slice_write(
            packed,
            self._mtp_pack,
            [0, slot, 0, 0],
            [b, slot + 1, hc, d],
            [1, 1, 1, 1],
        )
        if packed is not streams:
            packed.deallocate()

    def _send_mtp_pack(self) -> None:
        """One socket send of the packed ``[B, 3, hc, D]`` last-3-layer residuals to the
        MTP chip. Must be matched by a :meth:`_recv_mtp_pack` on the MTP submesh's trace."""
        ttnn.experimental.send_direct_async(self._mtp_pack, self._mtp_sender)

    def _recv_mtp_pack(self) -> None:
        """Post the receive of the packed residuals ``[B, 3, hc, D]`` ROW_MAJOR bf16 into
        ``_mtp_hiddens`` on the MTP submesh -- the whole body of that submesh's decode
        trace. Order must match :meth:`_send_mtp_pack`."""
        ttnn.experimental.recv_direct_async(self._mtp_hiddens, self._mtp_receiver)

    def read_mtp_hiddens(self) -> torch.Tensor:
        """Host copy of the packed layer-40/41/42 residuals, ``[B, 3, hc, D]``.

        Rank 0 of the MTP submesh (residuals are replicated across the 1xTP stage).
        Valid after a traced decode step that ran the tap.
        """
        if self._mtp_hiddens is None:
            raise RuntimeError("MTP hidden tap is not allocated (need idle chips and layers 40-42)")
        packed = self._mtp_hiddens
        copies = ttnn.to_torch(packed, mesh_composer=ttnn.ConcatMeshToTensor(self.mtp_submesh, dim=0))
        b = packed.shape[0]
        return copies[:b].contiguous()

    def _submesh_id_for_layer(self, layer_idx: int) -> int:
        """The submesh layer ``layer_idx`` lives on -- an index into ``submeshes``, i.e.
        entry ``layer_idx`` of the ``[num_layers]`` placement computed in
        :meth:`__init__` (see :func:`plan_layer_placement`)."""
        return self.layer_submesh_ids[layer_idx]

    def _next_layer_on_submesh(self, layer_idx: int) -> Optional[int]:
        """The next global layer placed on the same submesh as ``layer_idx`` (the one
        whose weights are worth prefetching while this device waits), or ``None`` if the
        ``[num_layers]`` placement holds no later layer on that submesh."""
        k = self.layer_submesh_ids[layer_idx]
        for li in range(layer_idx + 1, self.num_layers):
            if self.layer_submesh_ids[li] == k:
                return li
        return None

    def _prefetch_buffers_for(self, device, weight_dtype, num_prefetch_pages) -> dict:
        """The GCB for ``device``, built on first use and reused after.

        One buffer per device, not per layer or per weight: a GCB is a permanent L1 allocation,
        and every layer's weights have the same ``[K, N]`` shapes, so they can all stream
        through the same ring. Building one per layer would multiply 288 KB per receiver core
        by the layer count and exhaust L1 -- and long before that, the DRISC senders' state
        zone, which holds only about six GCBs per device however small they are.
        """
        key = id(device)
        if key not in self._prefetch_buffers_by_device:
            self._prefetch_buffers_by_device[key] = (
                device,
                make_decode_prefetch_buffers(device, weight_dtype, num_prefetch_pages),
            )
        return self._prefetch_buffers_by_device[key][1]

    @contextlib.contextmanager
    def prefetcher_session(self):
        """Run the DRISC senders for the duration of a decode run.

        Required around every :meth:`decode`, whose step returns ``[B, 1, 1, D]``. One
        session should span a whole generation rather than a single step, because starting
        and stopping the senders is not free and the GCB ring state carries across steps.

        Opened on every device holding prefetch buffers (one per submesh under
        ``use_submeshes``). Entry fences against the weight uploads already on the command
        queue with ``wait_for_cq_on_tensor_prefetcher``, so a sender cannot read a weight
        buffer before its write has landed.

        A failure anywhere inside the session -- not just a rejected ``matmul_decode``, but
        any op that throws while building or launching its program -- leaves the requests
        that ``prefetch_weights`` hoisted for the next layer sitting with the DRISC senders,
        with no matmul left to drain them. A clean stop cannot retire that: its sentinel
        queues behind the orphaned requests, so the kernel blocks on a full GCB and never
        reaches it. The exception path therefore force-stops (abandoning the kernels) and
        skips the device sync, both of which would otherwise hang and bury the error. Forcing
        leaves DRISC kernels running, so the device must be closed or reset before another
        session is opened -- fine, because this path only runs while the caller is unwinding.
        """
        devices = [device for device, _ in self._prefetch_buffers_by_device.values()]
        for device in devices:
            ttnn.experimental.start_tensor_prefetcher(device)
        for device in devices:
            ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        try:
            yield
        except BaseException:
            for device in devices:
                # Already stopped if the failure came out of LinearDecode.forward, which
                # retires the senders itself; stopping twice is a no-op.
                with contextlib.suppress(Exception):
                    ttnn.experimental.stop_tensor_prefetcher(device, force=True)
            raise
        for device in devices:
            ttnn.experimental.stop_tensor_prefetcher(device)
            ttnn.synchronize_device(device)

    def _expert_provider(self, layer_idx: int):
        """Host expert provider for one routed MoE layer: ``(gate_up [2I, D], down
        [D, I])`` per expert id, the fused layout :class:`DeepSeekV4PreloadedExperts`
        wants."""

        def provider(e: int):
            """Expert ``e`` as ``(gate_up [2I, D], down [D, I])`` host float32 torch
            tensors -- the HF packed layout, gate and up concatenated on dim 0."""
            base = f"layers.{layer_idx}.mlp.experts.{e}"
            gate = self._thunk(f"{base}.gate_proj.weight")()
            up = self._thunk(f"{base}.up_proj.weight")()
            down = self._thunk(f"{base}.down_proj.weight")()
            return torch.cat([gate, up], dim=0).float(), down.float()

        return provider

    def _hash_gate(self, layer_idx: int, prefetch_buffers=None, weight_dtype=None) -> DeepSeekV4HashRouter:
        """The static ``hash_moe`` router for ``layer_idx``, on the device that holds the
        layer. Its weights are the gate Linear ``gate.weight`` ``[E, D]`` plus the frozen
        ``gate.tid2eid`` ``[V, top_k]`` int64 token-id -> expert-id table, which has no
        tile cache and is read from the checkpoint here."""
        weights = {
            "gate.weight": self._thunk(f"layers.{layer_idx}.mlp.gate.weight"),
            "gate.tid2eid": self.loader.get_tensor(f"layers.{layer_idx}.mlp.gate.tid2eid").long(),
        }
        if self.use_submeshes:
            this_device = self.submeshes[self._submesh_id_for_layer(layer_idx)]
        else:
            this_device = self.first_device
        return DeepSeekV4HashRouter(
            self.config,
            weights,
            this_device,
            use_prefetcher=self.use_prefetcher,
            prefetch_buffers=prefetch_buffers,
            weight_dtype=weight_dtype if weight_dtype is not None else ttnn.bfloat16,
        )

    # -- compressor pooling schedule -------------------------------------------- #
    #
    # A CSA/HCA compressor emits a new compressed entry once every ``compress_rate``
    # tokens, and the block-bias exposes entries ``w < (pos+1)//compress_rate`` --
    # constant between two window closures. So the pool runs only on the steps that
    # close a window, and pools *only* that window's projections into one new entry of
    # the layer's paged KV axis. Together that makes a
    # step ``O(compress_rate)`` instead of ``O(max_seq)``, so throughput is flat in
    # ``max_seq``. Pooling off-closure is not merely slower but wrong -- the window
    # buffer is only fully written at a closure -- so there is no A/B switch here.

    # -- SDPA masking mode ------------------------------------------------------- #
    #
    # Once the sliding ring is full, a CSA/HCA layer's valid KV set is a contiguous
    # prefix, so SDPA-decode can be bounded by a single
    # ``cur_pos`` in causal mode instead of an additive mask; the causal kernel then
    # derives its chunk range from the position and skips the rest. Sub-window steps
    # (whose valid set has a hole) keep the mask. The mask is *data*, not control flow,
    # so the masked kernel always walks the KV axis it was captured against -- but those
    # traces only ever run for ``pos < sliding_window``, so they are captured at
    # ``max_seqlen == sliding_window`` (the ring plus the compressor entries that exist
    # in that prefix) rather than the full ``--max-context``. Causal attention therefore
    # tracks the actual position instead of ``max_seq``, and drops the per-step
    # per-layer head-broadcast of the mask row.
    # ``attention.sdpa_causal: false`` in the system profile (or
    # ``DEEPSEEK_V4_SDPA_CAUSAL=0``) forces the mask everywhere -- the previous
    # behaviour, and on the traced path it collapses the capture back to one variant
    # per pool phase.
    @property
    def _SDPA_CAUSAL(self) -> bool:
        """Whether compressor layers use causal SDPA (a ``cur_pos`` bound on the
        ``[W | compressor]`` KV axis) rather than the additive mask:
        ``attention.sdpa_causal`` of the active system profile."""
        return self.system_config.attention.sdpa_causal

    @property
    def _masked_decode_max_seq(self) -> int:
        """Context length baked into the masked SDPA traces.

        With causal SDPA those traces only run below ``sliding_window``, so that
        window is enough: the ``[W | W//cr]`` axis has ``W + W/cr`` rows instead of
        ``W + max_seq/cr``. With it disabled the mask is used at every position and the
        axis stays the full ``max_seq``.
        """
        if self._SDPA_CAUSAL:
            return self.sliding_window
        assert self._decode_max_seq is not None
        return self._decode_max_seq

    def _compressor_pool_due(self, layer_type: str, pos: int) -> bool:
        """Does the step at absolute ``pos`` close a window for ``layer_type`` -- i.e.
        append one ``[Dh]`` entry to that layer's paged KV axis?"""
        if layer_type == "sliding_attention":
            return False
        return (pos + 1) % self.config.compress_rates[layer_type] == 0

    def _compress_rates_for(self, layer_types) -> list[int]:
        """Sorted distinct compress rates among ``layer_types`` (sliding layers have none):
        the ``cr`` of each compressor family's ``[B*cr, 1, 1, F]`` window buffer."""
        return sorted({self.config.compress_rates[t] for t in layer_types if t != "sliding_attention"})

    def _build_pool_phases(self, crs: list[int]) -> tuple[list[tuple], dict[int, int]]:
        """The distinct pooling patterns over a window period, and the pos -> phase map.

        A step's pattern is ``tuple((pos+1) % cr == 0 for cr in crs)``, a ``[len(crs)]``
        tuple of booleans, which repeats with period ``lcm(crs)``. Far fewer than
        ``2 ** len(crs)`` patterns are reachable, because the rates divide one another: with
        the default ``{CSA: 4, HCA: 128}`` an HCA closure *always* coincides with a CSA
        closure (4 | 128), so the reachable set is three phases -- pool nothing, pool CSA,
        pool CSA+HCA -- and never "HCA alone". The traced path captures one trace variant per
        phase, so this directly bounds the trace-memory cost. Returns ``(phases, phase_of)``:
        the ``[n_phases, len(crs)]`` boolean table and the ``pos % period -> phase`` map.
        """
        if not crs:
            return [()], {0: 0}
        phases: list[tuple] = []
        phase_of: dict[int, int] = {}
        for p in range(math.lcm(*crs)):
            key = tuple((p + 1) % cr == 0 for cr in crs)
            if key not in phases:
                phases.append(key)
            phase_of[p] = phases.index(key)
        return phases, phase_of

    def _pool_phase_index(self, pos: int) -> int:
        """Index into the ``[n_phases]`` :attr:`_pool_phases` table of the pooling
        schedule to use at ``pos``."""
        return self._pool_phase_of[pos % self._pool_period]

    def _sdpa_causal_step(self, pos: int) -> bool:
        """Whether this step's compressor layers use causal SDPA. Layer-type
        independent (only ``pos`` vs. the ring capacity matters), so one boolean
        selects the trace variant for the whole stack."""
        return self._SDPA_CAUSAL and pos + 1 >= self.sliding_window

    def _variant_key(self, pos: int) -> tuple[bool, int, bool]:
        """The ``[3]``-tuple identifying the captured trace variant to replay at ``pos``:
        (SDPA mode, pool phase, CSA indexer).

        The third entry is true once the sequence is long enough that CSA top-k is
        legal (``seq_len >= compress_rate * index_topk``, unless the profile overrides
        it). Shorter positions keep dense causal SDPA and are a different capture:
        top-k refuses a valid length below ``k``, and the two programs are not the
        same op sequence.
        """
        return (self._sdpa_causal_step(pos), self._pool_phase_index(pos), self._index_sparse_step(pos))

    def _indexer_active(self) -> bool:
        """True when some CSA layer actually built a lightning indexer."""
        for layer in self.layers:
            indexer = getattr(getattr(layer.self_attn, "compressor", None), "indexer", None)
            if indexer is not None:
                return True
        return False

    def _index_dense_limit(self) -> int:
        """Sequence length at which CSA switches from dense SDPA to the indexer.

        ``attention.index_dense_max_positions`` overrides this; ``0`` means
        ``compress_rate * index_topk``. Below the limit there are fewer closed
        windows than ``index_topk``.
        """
        configured = int(self.system_config.attention.index_dense_max_positions)
        cr = int(self.config.compress_rates.get("compressed_sparse_attention", 0))
        floor = cr * int(getattr(self.config, "index_topk", 0))
        if configured <= 0:
            return floor
        # Top-k rejects a valid length below k, so the dense region cannot shrink
        # past the first position where ``index_topk`` windows are closed.
        return max(configured, floor)

    def _index_sparse_step(self, pos: int) -> bool:
        """Whether CSA at ``pos`` attends through the lightning indexer."""
        limit = self._index_dense_limit()
        if limit <= 0 or not self._indexer_active():
            return False
        return (pos + 1) >= limit

    def _capture_packet_pos(self, pos: int, causal: bool, index_sparse: bool) -> int:
        """Position a compile run may execute at.

        Replay reads the real position from the packet, so this value is only the
        dry run's. Masked traces address the short page table (``sliding_window``
        rows). Feeding them ``DEEPSEEK_V4_START_POS`` walks the compressor write
        off that table and the paged update never completes. Every dry run is
        also at least at the first closed CSA window, so the index-cache row is
        not negative. The indexer dry run needs at least ``index_topk`` closed
        windows.
        """
        # A pooling dry run still executes the cache write. Below the first
        # closed CSA window that row is negative and the index-cache update
        # never returns.
        first_closed = max(int(self.config.compress_rates.get("compressed_sparse_attention", 1)) - 1, 0)
        if index_sparse:
            packet_pos = max(pos, self._index_dense_limit() - 1, first_closed)
        elif self._SDPA_CAUSAL and not causal:
            packet_pos = min(max(pos, first_closed), max(self.sliding_window - 1, 0))
        else:
            packet_pos = max(pos, first_closed)
        if self._decode_max_seq:
            packet_pos = min(packet_pos, self._decode_max_seq - 1)
        return max(packet_pos, 0)

    def _indexer_head_dim(self, li: int) -> int | None:
        """``index_head_dim`` when layer ``li`` has an indexer, else ``None``."""
        indexer = getattr(getattr(self.layers[li].self_attn, "compressor", None), "indexer", None)
        return None if indexer is None else indexer.head_dim

    def _reachable_phases(self, causal: bool) -> list[int]:
        """Pool-phase indices (``[n_phases]`` at most) that can co-occur with this SDPA mode.

        Causal mode runs for every ``pos >= sliding_window - 1``, so every phase
        eventually occurs there. The mask fallback only runs *below* that, so phases
        whose closures never land in that prefix need no masked variant — with the
        default rates the HCA closure first lands at ``pos == 127``, i.e. exactly at
        the switch, so the "pool CSA+HCA" phase is causal-only and the masked family
        is two variants rather than three.
        """
        if causal or not self._SDPA_CAUSAL:
            return list(range(len(self._pool_phases)))
        return sorted({self._pool_phase_index(p) for p in range(max(self.sliding_window - 1, 0))})

    def _reachable_variants(self) -> list[tuple[bool, int, bool]]:
        """Every (SDPA mode, pool phase, indexer) triple a step can actually ask for.

        The indexer trace exists only past the dense limit, which is above the sliding
        window, so it has no masked twin: HCA on that trace is causal whenever causal
        SDPA is enabled. Sub-limit positions keep the dense family, including the masked
        captures below the sliding window.
        """
        modes = [False, True] if self._SDPA_CAUSAL else [False]
        dense = [(causal, phase, False) for causal in modes for phase in self._reachable_phases(causal)]
        max_seq = self._decode_max_seq or 0
        limit = self._index_dense_limit()
        if not self._indexer_active() or limit <= 0 or max_seq < limit:
            return dense
        long_causal = bool(self._SDPA_CAUSAL)
        sparse = [(long_causal, phase, True) for phase in self._reachable_phases(long_causal)]
        return dense + sparse

    @staticmethod
    def _sm_pool_key(sm: dict, pool_flags: dict[int, bool], causal: bool, index_sparse: bool = False) -> tuple:
        """A submesh's slice of a global variant -- a tuple of the rates its own layers
        use, the SDPA mode (which only compressor layers observe), and whether this
        submesh's CSA layers run the indexer.

        Submeshes that host no compressor layer (or only one of the rates) collapse
        several global variants onto the same capture — a sliding-only submesh is
        captured exactly once however many variants the stack has. A submesh with no
        CSA layer likewise ignores the indexer bit.
        """
        return (
            causal and bool(sm["pool_crs"]),
            tuple(pool_flags[cr] for cr in sm["pool_crs"]),
            index_sparse and bool(sm.get("has_csa")),
        )

    # ------------------------------------------------------------------ #
    # Paged multi-session decode
    #
    # Several conversations share one captured trace. A trace bakes in the *addresses* of
    # the buffers it touches, so per-session state cannot live in per-session buffers --
    # the trace would only ever see the first session's. Two mechanisms cover the two kinds
    # of state:
    #
    #   * The KV caches (all the memory that matters) move behind a block pool per layer
    #     plus a ``page_table`` tensor per (submesh, layer type). Switching sessions rewrites
    #     the table's *contents* with that session's logical->physical block row, so blocks
    #     are handed out on demand and N conversations share a total token budget instead of
    #     reserving ``N x max_context`` (see :mod:`.paged_cache`).
    #   * The compressor window buffers (one window of projections, a few KB) stay dense and
    #     are copied in and out of the trace-addressed buffers when the seated batch changes
    #     -- which a round-robin over batches does every step, so the copy is held to one per
    #     buffer by keeping the state per *group* of co-seated sessions rather than per
    #     session (see :meth:`activate_sessions`).
    #
    # Everything either needs is allocated up front by :meth:`prepare_static_decode`:
    # allocating device buffers once a trace exists on the device is unsafe, so opening a
    # session and seating it only claim pre-built blocks and do host-side book-keeping.
    # ------------------------------------------------------------------ #
    @property
    def paged(self) -> bool:
        """Is this model set up for paged (multi-session) traced decode?"""
        return self._paged is not None

    def _require_paged(self) -> PagedKVManager:
        """The paged manager, or a raise naming the call that would have created it."""
        if self._paged is None:
            raise RuntimeError("call prepare_static_decode() before using sessions")
        return self._paged

    def open_session(self) -> int:
        """Claim a session slot -- its ring blocks in every group's
        ``[num_blocks, 1, block_size, Dh]`` pool -- and return its id.

        Purely host-side: the device buffers were all allocated by
        :meth:`prepare_static_decode`. A fresh session needs no cache zeroing -- every
        step masks (or, in causal mode, bounds itself below) the rows it has not
        written yet, so a recycled block is never read.
        """
        paged = self._require_paged()
        if len(self._session_pos) >= self._max_sessions:
            raise PagedCacheFull(f"all {self._max_sessions} session slots are in use")
        sid = paged.open_session()
        self._session_pos[sid] = 0
        return sid

    def close_session(self, sid: int) -> None:
        """Release a session's blocks back to the pool, and its seat group's window block
        once the last member of that group has gone."""
        paged = self._require_paged()
        if sid in self._resident:
            # Vacated, not removed: the slot a session sat in is where its window rows
            # live, so compacting the list would misalign every slot after it.
            self._resident[self._resident.index(sid)] = None
        self._session_pos.pop(sid, None)
        paged.close_session(sid)
        # A group's block belongs to the group, so it is only recycled once every session
        # in it has gone -- while any member is open, its rows are still that member's.
        key = self._group_of.pop(sid, None)
        if key is not None and all(s not in self._session_pos for s in key):
            state = self._group_state.pop(key)
            if state is self._resident_group:
                self._resident_group = None
            self._free_group_state.append(state)

    def reset_session(self, sid: int) -> None:
        """Rewind a session to position 0: free its compressed blocks and clear its
        compressor window state (keeping its sliding-ring blocks ``[B, 1, W, Dh]``, whose
        stale rows are masked until rewritten)."""
        paged = self._require_paged()
        paged.reset_session(sid)
        self._session_pos[sid] = 0
        if sid in self._resident:
            self._clear_compressor_slot(self._resident.index(sid), None)
        else:
            key = self._group_of.get(sid)
            # A session that has never been seated has no rows anywhere yet: the block it
            # will get is blanked when its group claims one.
            if key is not None:
                self._clear_compressor_slot(key.index(sid), self._group_state[key])
        self._write_page_tables(list(self._paged_groups))

    def activate_session(self, sid: int) -> None:
        """Make ``sid`` the session the next :meth:`decode_traced` steps belong to.

        The single-user form of :meth:`activate_sessions`, and only valid on a
        single-user model -- a batched step decodes every slot, so it needs a session
        for each. A step's output ``[B, 1, N]`` then belongs to this one session.
        """
        self.activate_sessions([sid])

    def activate_sessions(self, sids) -> None:
        """Seat ``sids`` in the batch slots the next steps decode, in slot order.

        Points every page table's row ``u`` at ``sids[u]``'s blocks and swaps their
        compressor window state into the rows the traces address. Exactly as many
        sessions as the captured batch decodes are required: a trace decodes all of its
        slots unconditionally, so an empty slot would still write KV somewhere, and the
        only somewhere it could write is another session's blocks.

        The resident sessions step in lockstep -- one call to :meth:`decode_traced`
        advances them all -- so they must agree on the position they resume at (see
        :meth:`_variant_key`: one trace bakes in one pooling schedule for the whole
        batch). A set that disagrees, or that repeats a session, raises here rather than
        silently decoding some of them at another user's phase.

        The set is also the unit the window state is *kept* in: these sessions become a
        seat group, swapped in as one block. So a session may be re-seated with the same
        companions in any order, but not mixed into a different set -- its state is held
        at a fixed slot of that group's block, and a new set would read another user's
        window. Tying it this way is a cost decision: a per-session swap moves one row per
        slot per compressor buffer, which on a 43-layer stack is thousands of device ops
        on every step's critical path (measured at ~100 ms, more than the step itself),
        while a block swap is one copy per buffer regardless of batch.
        """
        paged = self._require_paged()
        sids = list(sids)
        if len(sids) != self._decode_batch:
            raise ValueError(f"batch decodes {self._decode_batch} users at a time, got {len(sids)} sessions")
        if len(set(sids)) != len(sids):
            raise ValueError(f"a session cannot occupy two slots at once: {sids}")
        for sid in sids:
            if not paged.has_session(sid):
                raise KeyError(f"no such session {sid}")
        at = {sid: self._session_pos.get(sid, 0) for sid in sids}
        if len(set(at.values())) > 1:
            raise ValueError(f"resident sessions must resume at one position, got {at}")
        if sids == self._resident:
            return

        group = self._seat_group(sids)
        if group is not self._resident_group:
            if self._resident_group is not None:
                self._save_group_state(self._resident_group)
            self._load_group_state(group)
        self._resident = sids
        self._resident_group = group
        self._write_page_tables(list(self._paged_groups))

    def ensure_session_capacity(self, pos: int) -> None:
        """Give every resident session blocks for the rows a step at ``pos`` touches,
        refreshing only the page tables whose rows actually changed (a compressor group
        grows one ``[block_size, Dh]`` block every ``compress_rate * block_size`` tokens)."""
        paged = self._require_paged()
        if not self._resident or None in self._resident:
            raise RuntimeError("call activate_sessions() before decoding")
        if self._context_limit is not None and pos >= self._context_limit:
            raise PagedCacheFull(
                f"position {pos} exceeds the {self._context_limit}-token context the dense CSA KV "
                f"({CSA_MAX_COMPRESSED_ENTRIES} compressed entries) can hold"
            )
        # Every session is grown before anything is written: a table refresh publishes
        # all of its rows at once, so a short-circuit here would leave the rest of the
        # batch pointing at blocks it has not been given yet.
        changed = {group for sid in self._resident for group in paged.ensure_capacity(sid, pos)}
        if changed:
            self._write_page_tables(sorted(changed))

    def session_usage(self) -> dict:
        """Per-group ``(blocks used, pool size)`` of the ``[num_blocks, 1, block_size, Dh]``
        pools, for status reporting."""
        return self._require_paged().usage()

    @property
    def context_limit(self) -> Optional[int]:
        """Longest per-session context the dense CSA KV buffers allow, or ``None`` when
        unbounded by them (no CSA layer). Set by :meth:`prepare_static_decode`."""
        return self._context_limit

    def session_tokens_left(self) -> int:
        """Tokens the shared pool can still admit across all open sessions."""
        return self._require_paged().tokens_left()

    # -- paged device state ----------------------------------------------------- #
    def _paged_view(self, sm: dict, li: int, causal: bool = True) -> Optional[PagedLayerView]:
        """The pool + page table layer ``li`` reads its KV through, or ``None`` for a
        sliding / CSA layer, whose KV is the dense ``scache.kv``. The view carries the layer's pool
        ``[num_blocks, 1, block_size, Dh]``, its ``[B, logical_blocks]`` page-table row for
        this trace family, and the ring's ``position_modulo``.

        Masked traces bake in the short page-table prefix sized for
        ``sliding_window``; causal traces use the full ``max_seq`` table. Both
        views share the same pool, and :meth:`_write_page_tables` keeps the prefix
        in sync with the full row.
        """
        group = self._paged_groups.get(self.config.layer_types[li])
        if group is None:
            return None
        tables = sm["page_tables"] if causal else sm["page_tables_masked"]
        return PagedLayerView(sm["pools"][li], tables[group.layer_type], group.position_modulo)

    def _write_page_tables(self, groups) -> None:
        """Copy the resident sessions' page-table rows into the persistent device
        tables of every submesh that hosts a layer of those groups.

        Row ``u`` of a table is slot ``u``'s mapping, which is how the paged ops give
        each user of a step its own blocks out of the shared pool. Tables are
        ``[B, logical_blocks]`` INT32 (``[B, n_masked]`` for the masked prefix).
        """
        paged = self._require_paged()
        if len(self._resident) != self._decode_batch or None in self._resident:
            # No full batch is seated (nothing has been activated yet, or a session was
            # just closed), so there are no rows to publish. The next
            # :meth:`activate_sessions` writes every table anyway.
            return
        for group in groups:
            rows = torch.cat([paged.page_row(sid, group) for sid in self._resident])
            table_row = ttnn.from_torch(rows, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            g = self._paged_groups[group]
            n_masked = g.logical_blocks_for(self._masked_decode_max_seq)
            short_row = (
                ttnn.from_torch(rows[:, :n_masked], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
                if n_masked < g.logical_blocks
                else None
            )
            for sm in self.submeshes_io:
                table = sm["page_tables"].get(group)
                if table is not None:
                    ttnn.copy_host_to_device_tensor(table_row, table)
                short = sm.get("page_tables_masked", {}).get(group)
                if short is not None and short is not table and short_row is not None:
                    ttnn.copy_host_to_device_tensor(short_row, short)

    def _compressor_slots(self):
        """``(submesh, layer, buffer name)`` for every per-session buffer the layer actually
        allocated, in a fixed order: the dense KV ``kv`` (sliding / CSA, TILE DRAM
        ``[1, 1, rows, Dh]``), then the compressor windows. CSA keeps all four
        (``win_kv``/``win_gate``/``prev_kv``/``prev_gate``, ROW_MAJOR L1 WIDTH_SHARDED
        ``[B*cr, 1, 1, 2*Dh]``); HCA keeps only ``win_kv``/``win_gate`` (TILE DRAM
        ``[B, 1, cr, Dh]``); sliding layers keep none."""
        for sm in self.submeshes_io:
            for li, scache in sm["scaches"].items():
                for name in (
                    "kv",
                    "win_kv",
                    "win_gate",
                    "prev_kv",
                    "prev_gate",
                    "idx_win_kv",
                    "idx_win_gate",
                    "idx_prev_kv",
                    "idx_prev_gate",
                    "idx_key_cache",
                ):
                    if getattr(scache, name) is not None:
                        yield sm, li, name

    @staticmethod
    def _empty_compressor_fill(name: str) -> float:
        """The value an unwritten compressor window buffer holds.

        ``prev_gate`` starts at ``_MASK_NEG`` rather than 0 so the first window's
        absent Ca half carries softmax weight 0 (see :class:`_StaticLayerCache`); every
        other window buffer -- ``[B*cr, 1, 1, F]`` on CSA, ``[B, 1, cr, Dh]`` on HCA --
        is filled with 0.
        """
        return _MASK_NEG if name in ("prev_gate", "idx_prev_gate") else 0.0

    def _window_host_tensor(self, buf: ttnn.Tensor, shape, fill, device, mesh_mapper=None) -> ttnn.Tensor:
        """DRAM INTERLEAVED clone of a compressor window, used for held-aside group
        state and the per-slot blanking source: ``[B*cr, 1, 1, F]`` for a packed CSA
        window, ``[B, 1, cr, Dh]`` (or one row of it) for a TILE one.

        CSA resident windows are ROW_MAJOR L1 WIDTH_SHARDED so ``csa_pool_window`` can
        consume them in place. Cloning that spec once per seat group (``num_sessions //
        batch`` extra copies of every CSA layer's four windows) exhausts L1 -- the
        single-user path never allocates those copies. DRAM keeps the same shape and
        layout; :meth:`_copy_window` moves them with ``to_memory_config(...,
        output_tensor=)`` so the swap never allocates.
        """
        return ttnn.from_torch(
            torch.full(list(shape), fill),
            dtype=buf.dtype,
            layout=buf.layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

    def _state_mesh_mapper(self, name: str, device):
        """Mesh placement of a held-aside copy of buffer ``name``.

        Compressor buffers, including the index-key cache, are replicated across
        the TP mesh. Paged KV lives in the block pool, not in one of these copies.
        """
        del name
        if self.tp_size <= 1:
            return None
        return ttnn.ReplicateTensorToMesh(device)

    def _copy_window(self, src: ttnn.Tensor, dest: ttnn.Tensor) -> None:
        """Write ``src`` into preallocated ``dest`` (same shape, ``[B*cr, 1, 1, F]`` or
        ``[B, 1, cr, Dh]``), including L1-sharded CSA <-> DRAM, without allocating."""
        ttnn.to_memory_config(src, dest.memory_config(), output_tensor=dest)

    def _build_group_state(self) -> dict:
        """One seat group's held-aside compressor window buffers, at their empty values.

        Shaped like the resident buffers, batch and all -- ``[B*cr, 1, 1, F]`` /
        ``[B, 1, cr, Dh]``, keyed by (submesh index, layer, buffer name): a group is seated
        and unseated as a unit (see :meth:`activate_sessions`), so one whole-buffer copy per
        direction replaces a copy per slot. The memory is the same either way --
        ``num_sessions // batch`` groups of ``batch`` rows -- but these copies are DRAM, so
        they do not compete with the resident L1 windows.

        Allocating runs only from :meth:`prepare_static_decode`; a group claimed later is
        blanked in place by :meth:`_empty_group_state`, which allocates nothing and so
        stays legal once traces exist.

        The state lives on device and the swap moves it with device ops only. Holding it on
        host would be simpler, but reading a device buffer back blocks the host until the
        command queue drains -- and a caller that pipelines steps (see
        :meth:`decode_traced_async`) has replays in flight whose output nobody has read
        yet, so their in-trace D2H sends are waiting on the very host that would now be
        waiting on the queue. That deadlocks.
        """
        state = {}
        for sm, li, name in self._compressor_slots():
            buf = getattr(sm["scaches"][li], name)
            state[(sm["index"], li, name)] = self._window_host_tensor(
                buf,
                buf.shape,
                self._empty_compressor_fill(name),
                sm["device"],
                mesh_mapper=self._state_mesh_mapper(name, sm["device"]),
            )
        return state

    def _build_empty_rows(self) -> dict:
        """One empty slot's worth of rows per compressor buffer, the source for blanking it.

        A TILE DRAM window carries the batch on dim 0, so a slot is one row of it. A
        packed CSA window (ROW_MAJOR L1 WIDTH_SHARDED ``[B*cr, 1, 1, F]``) is user-major,
        so a slot is the ``cr`` rows of one user, and the source is INTERLEAVED because
        that is what ``indexed_fill`` scatters from (see :meth:`_blank_window_slots`).
        """
        rows = {}
        for sm, li, name in self._compressor_slots():
            buf = getattr(sm["scaches"][li], name)
            slot_rows = buf.shape[0] // self._decode_batch if buf.is_sharded() else 1
            rows[(sm["index"], li, name)] = self._window_host_tensor(
                buf,
                [slot_rows, *list(buf.shape)[1:]],
                self._empty_compressor_fill(name),
                sm["device"],
                mesh_mapper=self._state_mesh_mapper(name, sm["device"]),
            )
        return rows

    def _blank_window_slots(self, target: ttnn.Tensor, key: tuple, slots, device) -> None:
        """Reset ``slots`` of a packed CSA window buffer to their empty values.

        User ``u`` owns rows ``[u*cr, (u+1)*cr)``, so blanking a slot is a scatter of
        that user's block rather than a whole-buffer fill (which would wipe the other
        users) -- and ``ttnn.fill`` cannot write a ROW_MAJOR sharded tensor anyway.
        """
        cr = target.shape[0] // self._decode_batch
        for slot in slots:
            index = ttnn.from_torch(
                torch.arange(cr, dtype=torch.int32) + slot * cr,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
            _scatter_window_rows(target, self._empty_row[key], index)
            ttnn.deallocate(index)

    def _seat_group(self, sids: list[int]) -> dict:
        """The held-aside block for the group ``sids``, claiming a free one on first seat.

        The block holds every compressor layer's ``[B*cr, 1, 1, F]`` / ``[B, 1, cr, Dh]``
        windows, keyed by (submesh index, layer, name). A session is grouped by the set (and
        slot order) it is first seated with and stays there, because that is where its rows
        are saved; seating it with anyone else would read another user's window, so it raises
        instead.
        """
        key = tuple(sids)
        group = self._group_state.get(key)
        if group is not None:
            return group
        for sid in sids:
            held = self._group_of.get(sid)
            if held is not None:
                raise ValueError(
                    f"session {sid} is seated as slot {held.index(sid)} of group {held} and cannot be "
                    f"re-seated as {key}: its compressor window state is held at that slot of that "
                    "group's block. Close and reopen the session to move it."
                )
        if not self._free_group_state:
            raise PagedCacheFull(
                f"all {self._max_sessions // self._decode_batch} seat groups are in use; "
                "close a group's sessions to release one"
            )
        group = self._free_group_state.pop()
        # Handed out blank: these sessions have never been seated, so they must find empty
        # windows, and a block released by :meth:`close_session` still holds what its last
        # members wrote.
        self._empty_group_state(group)
        self._group_state[key] = group
        for sid in sids:
            self._group_of[sid] = key
        return group

    def _empty_group_state(self, state: dict) -> None:
        """Blank a group's held-aside buffers in place, for a group of fresh sessions: the
        ``[B*cr, 1, 1, F]`` CSA windows by scatter, the ``[B, 1, cr, Dh]`` TILE ones by fill."""
        for sm, li, name in self._compressor_slots():
            key = (sm["index"], li, name)
            if state[key].layout == ttnn.ROW_MAJOR_LAYOUT:
                self._blank_window_slots(state[key], key, range(self._decode_batch), sm["device"])
            else:
                ttnn.fill(state[key], self._empty_compressor_fill(name), output_tensor=state[key])

    def _clear_compressor_slot(self, slot: int, state: Optional[dict]) -> None:
        """Reset one batch slot's compressor window rows to their empty values.

        Writes the resident buffers when ``state`` is ``None``, else that slot's row of a
        group's held-aside copy. Packed CSA windows (ROW_MAJOR ``[B*cr, 1, 1, F]``, L1
        or DRAM) blank by scatter; TILE HCA windows blank with ``fill_cache``.
        """
        for sm, li, name in self._compressor_slots():
            key = (sm["index"], li, name)
            target = getattr(sm["scaches"][li], name) if state is None else state[key]
            if target.layout == ttnn.ROW_MAJOR_LAYOUT:
                self._blank_window_slots(target, key, (slot,), sm["device"])
            else:
                ttnn.fill_cache(target, self._empty_row[key], slot)

    def _save_group_state(self, state: dict) -> None:
        """Hold the resident batch's window buffers ``[B*cr, 1, 1, F]`` /
        ``[B, 1, cr, Dh]`` aside as ``state``."""
        for sm, li, name in self._compressor_slots():
            self._copy_window(getattr(sm["scaches"][li], name), state[(sm["index"], li, name)])

    def _load_group_state(self, state: dict) -> None:
        """Put a group's held-aside window buffers ``[B*cr, 1, 1, F]`` / ``[B, 1, cr, Dh]``
        back into the resident ones."""
        for sm, li, name in self._compressor_slots():
            self._copy_window(state[(sm["index"], li, name)], getattr(sm["scaches"][li], name))

    # ------------------------------------------------------------------ #
    # Traced decode (one reusable trace per submesh / device)
    #
    # Decode is traced: a host-dispatched step would re-dispatch ~43 layers' worth of ops,
    # rebuild the RoPE rows / masks from host, read the MoE routing weights back to host,
    # and host-copy the residual streams across submeshes. Instead the model captures
    # one ``ttnn`` trace per submesh (so each device replays its own slice of the stack)
    # and, between replays, writes the tiny per-step inputs onto submesh 0 *only*, fused
    # into ONE fixed-shape INT32 packet (tokens + the two cache positions). RoPE rows and
    # additive masks are not in it: both are generated on device from the position, and the
    # streams and packet are socket-copied between submeshes from inside the traces
    # themselves, where each submesh splits the packet into the individual inputs (no
    # per-step host op dispatch past submesh 0). All cross-token state lives in fixed-size
    # in-place caches (:class:`_StaticLayerCache`), so a single capture serves every step.
    # See :meth:`prepare_static_decode` / :meth:`decode_traced`.
    # ------------------------------------------------------------------ #

    def _build_static_layer_cache(self, li: int, device: ttnn.MeshDevice) -> "_StaticLayerCache":
        """Allocate layer ``li``'s compressor window buffers *empty* for ``_decode_batch``
        users, as :func:`build_static_layer_cache` does (``[B*cr, 1, 1, 2*Dh]`` ROW_MAJOR L1
        WIDTH_SHARDED on CSA, ``[B, 1, cr, Dh]`` TILE DRAM on HCA). The KV is in the block
        pools (:meth:`_build_block_pool`)."""
        assert self._decode_max_seq is not None, "set max_seq via prepare_static_decode first"
        return build_static_layer_cache(
            device,
            self.config.layer_types[li],
            self.config.head_dim,
            self._decode_max_seq,
            self.config.compress_rates,
            self.sliding_window,
            batch=self._decode_batch,
            index_head_dim=self._indexer_head_dim(li),
        )

    def _build_block_pool(self, li: int, device: ttnn.MeshDevice) -> ttnn.Tensor:
        """One layer's KV block pool ``[num_blocks, 1, block_size, Dh]`` (all-zero).

        Block ``0`` is the shared zero block every unmapped page-table entry points at,
        so it must stay zero -- :class:`PagedKVManager` never hands it out.
        """
        group = self._paged_groups[self.config.layer_types[li]]
        num_blocks = self._paged.pools[group.layer_type].num_blocks
        return ttnn.from_torch(
            torch.zeros(num_blocks, 1, group.block_size, self.config.head_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _alloc_page_table(self, n_blocks: int, device: ttnn.MeshDevice) -> ttnn.Tensor:
        """Persistent ``[B, n_blocks]`` INT32 ROW_MAJOR page table (all zeros = every
        slot pointing at the pool's zero block)."""
        return ttnn.from_torch(
            torch.zeros(self._decode_batch, n_blocks, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

    def _build_page_tables(self, layer_types, device: ttnn.MeshDevice) -> tuple[dict, dict]:
        """Persistent ``[B, logical_blocks]`` INT32 page tables, one per layer type on
        this submesh: row ``u`` is the mapping the paged ops read for slot ``u``. The
        traces bake in these addresses; :meth:`activate_sessions` rewrites their contents.

        Returns ``(full, masked)``. Causal captures address the full ``max_seq`` tables;
        masked captures address a prefix sized for ``sliding_window`` (the only context
        those traces ever see). Sliding groups already fit in that prefix, so the two
        dicts alias the same tensor there.
        """
        full, masked = {}, {}
        for lt in layer_types:
            g = self._paged_groups[lt]
            full[lt] = self._alloc_page_table(g.logical_blocks, device)
            n_masked = g.logical_blocks_for(self._masked_decode_max_seq)
            masked[lt] = full[lt] if n_masked >= g.logical_blocks else self._alloc_page_table(n_masked, device)
        return full, masked

    def reset_static_caches(self) -> None:
        """Zero every compressor window buffer so a fresh sequence can start at position 0.

        The captured traces address these buffers directly, so they are zeroed in place:
        reallocating them would invalidate every capture, and even a temporary device
        allocation is unsafe while a trace exists. (``fill`` still logs the allocator's
        "unsafe with an active trace" warning for its own scratch; the cache buffers
        themselves are untouched by it.)

        ``prev_gate`` is refilled with ``_MASK_NEG`` rather than 0, matching how
        :func:`build_static_layer_cache` allocates it: it gates window 0's absent Ca half,
        which a 0 fill would give real softmax weight instead of none.

        The KV caches live in the block pools; use :meth:`reset_session` to rewind one
        session.
        """
        if not getattr(self, "submeshes_io", None):
            raise RuntimeError("call prepare_static_decode() before reset_static_caches()")
        for sm in self.submeshes_io:
            for scache in sm["scaches"].values():
                for name in _StaticLayerCache.__slots__:
                    buf = getattr(scache, name)
                    if buf is not None:
                        ttnn.fill(buf, _MASK_NEG if name == "prev_gate" else 0.0, output_tensor=buf)

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
        """Allocate the traced-decode state (the prompt is prefilled by replaying
        :meth:`decode_traced` once per prompt token into these empty caches).

        Builds, per submesh: the fixed-size in-place caches (empty / all-zero), the constant
        window-RoPE and mask index tables, and the persistent socket recv buffers (residual
        streams ``[B, 1, hc, D]`` row-major, plus the fused per-step packet
        ``[1,1,1,_pkt_w]`` INT32). Submesh 0 additionally gets the H2D socket the packet
        arrives on -- the only host->device traffic of a traced step. ``max_seq`` must be a
        multiple of every compress-rate (the caller pads it) so each compressor's fixed
        capacity tiles cleanly into windows. ``lm_head`` (optional) is folded into the last
        submesh's trace, turning its ``[B, 1, 1, D]`` hidden into ``[B, 1, 1, V]`` logits so
        a step returns them directly.

        KV is always paged: each layer gets a ``[num_blocks, 1, block_size, Dh]`` pool,
        sized (with ``total_tokens``, defaulting to one full ``max_seq``) for that many
        concurrent conversations sharing the budget. Everything a session needs is allocated
        here, before any trace exists, because allocating on a device that holds a trace is
        unsafe; :meth:`open_session` then only claims a slot, and ``block_size`` sets the
        same row count for every layer type (see :func:`.paged_cache.build_groups`).

        ``batch`` > 1 decodes that many users per step, one per slot: the packet carries a
        token each, every cache and page table gains a leading user dimension, and a step
        returns one output row per user. The users share the step's *position* -- a trace
        bakes in one compressor-pooling schedule and one SDPA mode for the whole batch (see
        :meth:`_variant_key`) -- so they advance in lockstep. Everything below is
        parameterised by it, and ``batch=1`` is the single-user path unchanged.
        """
        if not self.use_submeshes:
            raise NotImplementedError("traced decode requires use_submeshes=True")
        if batch < 1:
            raise ValueError(f"batch must be at least 1, got {batch}")
        if num_sessions < 1:
            raise ValueError(f"paged decode needs at least one session, got num_sessions={num_sessions}")
        if num_sessions < batch:
            raise ValueError(f"a batch of {batch} needs at least that many sessions, got num_sessions={num_sessions}")
        if batch != 1:
            # TODO: batch the dense sliding / CSA KV buffers.
            raise NotImplementedError(f"the dense sliding / CSA KV supports batch 1 only, got batch={batch}")
        self._decode_batch = batch
        cfg = self.config
        for cr in {cfg.compress_rates[t] for t in cfg.layer_types[: self.num_layers] if t != "sliding_attention"}:
            assert max_seq % cr == 0, f"max_seq ({max_seq}) must be a multiple of compress_rate {cr}"
        self._context_limit = dense_kv_context_limit(cfg.layer_types[: self.num_layers], cfg.compress_rates)
        if self._context_limit is not None and max_seq > self._context_limit:
            logger.warning(
                f"max_seq {max_seq} exceeds the {self._context_limit}-token context the dense CSA KV can hold; "
                "sessions are capped there"
            )
        # Only HCA layers are paged; sliding and CSA layers keep a dense ``scache.kv``.
        self._paged_groups = build_groups(
            [t for t in cfg.layer_types[: self.num_layers] if t in PAGED_KV_LAYER_TYPES],
            cfg.compress_rates,
            self.sliding_window,
            max_seq,
            block_size=block_size,
        )
        pool_blocks = plan_pool_blocks(self._paged_groups, num_sessions, total_tokens or max_seq)
        self._paged = PagedKVManager(self._paged_groups, pool_blocks)
        logger.info(
            "paged decode: "
            + ", ".join(
                f"{name} {pool_blocks[name]} blocks of {g.block_size} rows "
                f"({g.block_size * (g.compress_rate or 1)} tokens each, {g.logical_blocks} per session, "
                f"{g.axis_rows}-row axis)"
                for name, g in self._paged_groups.items()
            )
        )
        if self._SDPA_CAUSAL:
            logger.info(
                f"masked SDPA capture max_seqlen={self.sliding_window}: "
                + ", ".join(
                    f"{name} {g.kv_len_for(self.sliding_window)}-row axis "
                    f"({g.logical_blocks_for(self.sliding_window)} blocks)"
                    for name, g in self._paged_groups.items()
                )
            )
        self._lm_head_traced = lm_head
        self._decode_max_seq = max_seq
        self._pool_crs = self._compress_rates_for(cfg.layer_types[: self.num_layers])
        self._pool_period = math.lcm(*self._pool_crs) if self._pool_crs else 1
        self._pool_phases, self._pool_phase_of = self._build_pool_phases(self._pool_crs)

        rd = cfg.qk_rope_head_dim
        hc, d, w = cfg.hc_mult, cfg.hidden_size, self.sliding_window
        ids = self.layer_submesh_ids

        # --- Canonical per-step input packet layout (shared by every submesh) --- #
        # All per-step inputs are fused into ONE tiny fixed-shape INT32 packet
        # ``[1, 1, 1, _pkt_w]`` (ROW_MAJOR) -- 16 INT32s at ``batch=1``, wider once ``B``
        # grows -- a single persistent buffer on submesh 0, streamed in from host *only*
        # there over an H2D socket and then flowed downstream over the existing
        # device-to-device socket (see :meth:`_decode_submesh_static`), so no submesh past
        # the first sees any host traffic at all.
        #
        #   [0, B)     : one token per user (INT32; embedding/hash typecast to uint32)
        #   [B, 2B)    : pos_sliding, the same value per user
        #   [2B, 3B)   : pos_compress, the same value per user
        #
        # The two position regions are B wide even though a step's users share one position,
        # because the ops that consume them want a value per user (``paged_update_cache``'s
        # update index, SDPA-decode's ``cur_pos``). Repeating them on host costs a few INT32s
        # of an already-padded page and saves the device a broadcast per step.
        #
        # The per-step RoPE rows and additive masks are *not* in the packet: both are
        # generated on device from ``pos_compress`` against constant tables (see
        # :meth:`_device_rope` and :meth:`_device_mask`). Slots past the prefix are padding
        # -- the packet's row is one H2D socket page, so its width is rounded up to the PCIe
        # alignment rather than set by the payload, and nothing reads them.
        alignment = self.system_config.pipeline.pcie_alignment
        self._pkt_int_prefix = 3 * batch  # [tokens | pos_sliding | pos_compress]
        self._pkt_page_bytes = math.ceil(self._pkt_int_prefix * 4 / alignment) * alignment
        self._pkt_w = self._pkt_page_bytes // 4

        # --- On-device RoPE generation constants ------------------------------- #
        # RoPE is ``cos/sin(pos * inv_freq) * attention_scaling``, with ``inv_freq`` and
        # ``attention_scaling`` position-independent per family ("main" sliding,
        # "compress" CSA/HCA). Both are recovered from the host ``rope`` tables so the
        # device output matches them exactly: at p=0 the table is ``scaling`` (sin=0), and
        # ``inv_freq[j] = atan2(sin_half[1,j], cos_half[1,j])`` (all |inv_freq| < pi).
        # Stored already interleaved-by-2 to match ``make_rope_table``'s expansion.
        self._rope_gen: dict[str, tuple[torch.Tensor, float]] = {}
        for rt in ("main", "compress"):
            cos_h, sin_h = rope[rt]
            scaling = float(cos_h[0, 0].item())
            inv_freq_half = torch.atan2(sin_h[1].float(), cos_h[1].float())  # [rd/2]
            inv_freq_full = inv_freq_half.repeat_interleave(2).reshape(1, 1, 1, -1)  # [1,1,1,rd]
            self._rope_gen[rt] = (inv_freq_full, scaling)

        def _dev_zeros(shape, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            """All-zero device tensor of ``shape`` (e.g. the packet ``[1,1,1,_pkt_w]`` INT32
            or a residual stream ``[B, 1, hc, D]``); ``dtype`` is mapped to the host dtype
            ``from_torch`` needs (bf16 <- float32, uint32/int32 <- int32)."""
            tt_dtype = {ttnn.bfloat16: torch.float32, ttnn.uint32: torch.int32, ttnn.int32: torch.int32}[dtype]
            return ttnn.from_torch(torch.zeros(shape, dtype=tt_dtype), dtype=dtype, layout=layout, device=device)

        self.submeshes_io = []
        for k in self.pipeline_submesh_ids:
            device = self.submeshes[k]
            layers_k = [li for li in range(self.num_layers) if ids[li] == k]
            types = {cfg.layer_types[li] for li in layers_k}
            crs = {cfg.compress_rates[t] for t in types if t != "sliding_attention"}
            sm = {
                "device": device,
                "index": k,
                "layers": layers_k,
                "rope_invfreq": {},
                "mask_gen": {},
                "scaches": {li: self._build_static_layer_cache(li, device) for li in layers_k},
                # Paged mode: one block pool per layer, and one page table per layer
                # type (every layer of a type shares the mapping, not the data).
                "pools": {
                    li: self._build_block_pool(li, device)
                    for li in layers_k
                    if cfg.layer_types[li] in PAGED_KV_LAYER_TYPES
                },
                "page_tables": {},
                "page_tables_masked": {},
                "pool_crs": self._compress_rates_for(types),
                "has_csa": "compressed_sparse_attention" in types,
                "traces": {},  # local variant key -> (trace id, persistent output)
                "tids": {},  # global variant key -> trace id
                "outputs": {},  # global variant key -> persistent output
            }
            sm["page_tables"], sm["page_tables_masked"] = self._build_page_tables(
                [t for t in types if t in PAGED_KV_LAYER_TYPES], device
            )
            # Per-family inv_freq constants for the rope families this submesh uses.
            for rt in ({"main"} if "sliding_attention" in types else set()) | ({"compress"} if crs else set()):
                inv_freq_full, scaling = self._rope_gen[rt]
                sm["rope_invfreq"][rt] = (
                    ttnn.from_torch(inv_freq_full, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device),
                    scaling,
                )
            # Per-layer-type constant index tables for on-device mask generation. The mask
            # row is ``invalid * _MASK_NEG`` with ``invalid = (A > pos)`` over the sliding
            # columns OR ``(B >= (pos+1)//cr)`` over the compressor columns; the two
            # regions are packed into full-width A / B tables with ``-1`` fillers in the
            # *other* region (``-1`` is never ``> pos`` nor ``>= thr``), so a single
            # compare per table covers each region without a tile-boundary ``concat``.
            # Sized for the masked capture only (causal variants never read them): those
            # traces run exclusively at ``pos < W``, so ``sliding_window`` rows are enough
            # -- the ring plus the compressor entries that close inside that prefix.
            masked_max_seq = self._masked_decode_max_seq
            for lt in types:
                if lt == "sliding_attention":
                    # Always causal (``min(pos, W - 1)``, see ``_build_step_ctx``): no mask.
                    sm["mask_gen"][lt] = (None, None, None)
                    continue
                cr = cfg.compress_rates[lt]
                dense_rows = dense_kv_rows(lt, w)
                n_win_cap = masked_max_seq // cr
                if dense_rows is not None:
                    n_win_cap = min(n_win_cap, dense_rows - w)
                a = torch.cat([torch.arange(w), torch.full((n_win_cap,), -1)]).float()
                b = torch.cat([torch.full((w,), -1), torch.arange(n_win_cap)]).float()
                # SDPA reads rows past the valid axis -- the rest of a dense buffer, or the
                # unmapped tail of a paged block. Filling A there with a position no step
                # can reach makes ``A > pos`` true, so they are masked out however the
                # compressor compare falls.
                kv_len = dense_rows if dense_rows is not None else self._paged_groups[lt].kv_len_for(masked_max_seq)
                pad = kv_len - a.numel()
                if pad:
                    a = torch.cat([a, torch.full((pad,), float(masked_max_seq))])
                    b = torch.cat([b, torch.full((pad,), -1.0)])
                a_tt = ttnn.from_torch(
                    a.reshape(1, 1, 1, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
                )
                b_tt = ttnn.from_torch(
                    b.reshape(1, 1, 1, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
                )
                sm["mask_gen"][lt] = (a_tt, b_tt, cr)
            # Submesh 0 owns global layer 0, whose per-step inputs (token + positions) stream
            # in from the host over the H2D socket into the tiny fused packet; everything
            # downstream is fed over the device-to-device sockets. A submesh needs recv
            # buffers only for the layers whose *predecessor* sits on another submesh --
            # under round-robin that is every layer but global 0 (submesh 0 is revisited for
            # layers S, 2S, ...), while with PGS=1 a device's contiguous run of layers hands
            # off locally and only its first layer receives.
            if 0 in layers_k:
                sm["pkt"] = _dev_zeros([1, 1, 1, self._pkt_w], device, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                # The one host->device transfer of the whole traced decode. Created
                # here (before any capture) because the socket allocates L1 on its
                # receiver core, which is unsafe once a trace exists on the device.
                if self._pkt_socket is None:
                    self._pkt_socket = ttnn.H2DSocket(
                        device,
                        ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(*_PKT_SOCKET_CORE)),
                        ttnn.BufferType.L1,
                        self.system_config.pipeline.h2d_fifo_bytes,
                        ttnn.H2DMode.HOST_PUSH,
                    )
                    # ``recv_async_h2d`` cross-checks this against the packet's aligned
                    # page size on every program-cache miss.
                    self._pkt_socket.set_page_size(self._pkt_page_bytes)
            if any(li > 0 and ids[li - 1] != k for li in layers_k):
                # Residual handoff is row-major so the socket does not ship tile padding
                # Tilized again after recv.
                sm["streams_in"] = _dev_zeros([batch, 1, hc, d], device, layout=ttnn.ROW_MAJOR_LAYOUT)
                sm["pkt_in"] = _dev_zeros([1, 1, 1, self._pkt_w], device, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            self.submeshes_io.append(sm)
        if self.mtp_submesh is not None:
            self._ensure_mtp_buffers(batch)
            self.submeshes_io.append(
                {
                    "device": self.mtp_submesh,
                    "index": "mtp",
                    "layers": [],
                    "mtp_recv": True,
                    "rope_invfreq": {},
                    "mask_gen": {},
                    "scaches": {},
                    "pools": {},
                    "page_tables": {},
                    "page_tables_masked": {},
                    "pool_crs": [],
                    "traces": {},
                    "tids": {},
                    "outputs": {},
                }
            )
        # Where the global-last layer (num_layers-1) landed: its trace produces the final
        # head output, which it streams to the host over the D2H socket below.
        self._output_sm_index = self.pipeline_submesh_ids.index(ids[self.num_layers - 1])
        # Re-arm capture. The replay queue/thread stay owned by ``__init__`` (see there),
        # so a model whose prepare never finished can still be unwound.
        self._traced_captured = False

        # The step output's return path. The page size is only known once the trace
        # builds the output tensor, so it is set on first use (see
        # :meth:`_send_output`). Created here because the socket allocates L1 on its
        # sender core, which is unsafe once a trace exists.
        if self._out_socket is None:
            self._out_socket = ttnn.D2HSocket(
                self.submeshes_io[self._output_sm_index]["device"],
                ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(*_OUT_SOCKET_CORE)),
                self.system_config.pipeline.d2h_fifo_bytes,
            )

        # The held-aside compressor buffers, allocated now: once a trace exists on a
        # device, allocating buffers there can corrupt it, so seating may only claim one
        # of these pre-built sets. One set per *seat group* rather than per session (see
        # :meth:`activate_sessions`), each the full width of the resident buffers -- so
        # the total is the same ``num_sessions`` rows either way, but a switch moves each
        # buffer in one copy instead of one per slot. Copies live in DRAM: the resident
        # CSA windows are L1 WIDTH_SHARDED, and cloning that spec per group does not fit.
        self._max_sessions = num_sessions
        self._free_group_state = [self._build_group_state() for _ in range(num_sessions // batch)]
        # A single empty row per buffer, to blank one slot without disturbing the rest of
        # the batch (see :meth:`reset_session`). Never written to.
        self._empty_row = self._build_empty_rows()

    def _device_rope(self, inv_freq: ttnn.Tensor, scaling: float, pos_f: ttnn.Tensor) -> tuple:
        """Generate one decode step's RoPE rows on device from the absolute position.

        ``inv_freq`` ``[1,1,1,Rd]`` (FP32, interleaved-by-2) and ``scaling`` are the
        constants for one family; ``pos_f`` ``[1,1,1,1]`` (FP32) is the absolute
        position. Returns ``(cos, sin, neg_sin)`` bf16 tiles ``[1,1,1,Rd]`` equal to the
        host ``make_rope_table`` rows. The raw angle ``pos * inv_freq`` can reach thousands
        of radians, so it is range-reduced to ``[0, 2*pi)`` before ``sin``/``cos`` to keep
        the device transcendentals accurate."""
        two_pi = 6.283185307179586
        angle = ttnn.multiply(inv_freq, pos_f)  # [1,1,1,Rd] (broadcast)
        angle = ttnn.subtract(angle, ttnn.multiply(ttnn.floor(ttnn.multiply(angle, 1.0 / two_pi)), two_pi))
        cos = ttnn.typecast(ttnn.multiply(ttnn.cos(angle), scaling), ttnn.bfloat16)
        sin = ttnn.multiply(ttnn.sin(angle), scaling)
        neg_sin = ttnn.typecast(ttnn.neg(sin), ttnn.bfloat16)
        return cos, ttnn.typecast(sin, ttnn.bfloat16), neg_sin

    def _device_mask(
        self, a: ttnn.Tensor, b: Optional[ttnn.Tensor], cr: Optional[int], pos_f: ttnn.Tensor
    ) -> ttnn.Tensor:
        """Generate one decode step's additive attention mask on device from the
        absolute position. ``a`` / ``b`` are the constant index tables built in
        :meth:`prepare_static_decode`; the row is ``invalid * _MASK_NEG`` with
        ``invalid = (a > pos)`` over the sliding columns plus, for CSA/HCA layers,
        ``(b >= (pos+1)//cr)`` over the compressor columns. The two regions never both
        fire at a column (the ``-1`` fillers compare false), so the indicators add to
        a clean 0/1 mask. Returns a bf16 tile ``[1,1,1,W(+n_win_cap)]``."""
        invalid = ttnn.gt(a, pos_f)  # sliding: slot index > pos  (broadcast over [1,1,1,1])
        if b is not None:
            thr = ttnn.floor(ttnn.multiply(ttnn.add(pos_f, 1.0), 1.0 / cr))  # (pos+1)//cr
            invalid = ttnn.add(invalid, ttnn.ge(b, thr))  # compressor: window >= completed count
        return ttnn.typecast(ttnn.multiply(invalid, _MASK_NEG), ttnn.bfloat16)

    def _device_index(self, value_f: ttnn.Tensor) -> ttnn.Tensor:
        """A device-computed FP32 tile ``[1,1,1,batch]`` as the INT32 ``[batch]``
        row-major tensor the in-place cache writers and SDPA-decode take for a per-user
        index. One entry per user because those ops index each batch slot's cache
        separately, even though the users of a step share a position."""
        return ttnn.reshape(
            ttnn.to_layout(ttnn.typecast(value_f, ttnn.int32), ttnn.ROW_MAJOR_LAYOUT), [self._decode_batch]
        )

    def _device_causal_pos(self, cr: int, pos_row_f: ttnn.Tensor) -> ttnn.Tensor:
        """Generate one decode step's causal SDPA ``cur_pos`` on device from the
        absolute position: ``sliding_window + (pos+1)//cr - 1``, the inclusive last
        valid index on the ``[sliding | compressor]`` KV axis."""
        thr = ttnn.floor(ttnn.multiply(ttnn.add(pos_row_f, 1.0), 1.0 / cr))  # (pos+1)//cr
        return self._device_index(ttnn.add(thr, float(self.sliding_window - 1)))

    def _device_compressor_indices(self, cr: int, pos_row_f: ttnn.Tensor, pos_f: ttnn.Tensor):
        """Device twins of :func:`_window_indices`, plus the closing window's position.

        Returns ``(win_slot, win_row, win_pos_f)``: the INT32 ``[batch]`` slot
        ``pos % cr`` this token's projection is written to in the one-window buffer, the
        INT32 ``[batch]`` row ``sliding_window + w`` of the paged KV axis the pooled
        entry lands in, and the FP32 ``[1,1,1,1]`` absolute position ``w * cr`` to RoPE
        that entry at, for the window ``w = (pos+1)//cr - 1`` closing at this step.

        The two indices come off the per-user ``pos_row_f`` because the cache writers
        take one index per batch slot; ``win_pos_f`` comes off the scalar ``pos_f``
        because a RoPE row is shared by the whole batch (the users are at one position).

        A trace cannot branch on the device-side position, so all three are pure
        arithmetic. On steps that do not close a window ``w`` is one short (or ``-1``),
        but nothing consumes these then (``pool_compressor`` is ``False``).
        """
        thr = ttnn.floor(ttnn.multiply(ttnn.add(pos_row_f, 1.0), 1.0 / cr))  # (pos+1)//cr == w+1
        slot_f = ttnn.subtract(pos_row_f, ttnn.multiply(ttnn.floor(ttnn.multiply(pos_row_f, 1.0 / cr)), float(cr)))
        # At batch 1 the per-user row *is* the scalar, so reuse it rather than emitting
        # the same arithmetic twice.
        scalar_thr = thr if pos_row_f is pos_f else ttnn.floor(ttnn.multiply(ttnn.add(pos_f, 1.0), 1.0 / cr))
        win_pos_f = ttnn.multiply(ttnn.subtract(scalar_thr, 1.0), float(cr))
        return (
            self._device_index(slot_f),
            self._device_index(ttnn.add(thr, float(self.sliding_window - 1))),
            win_pos_f,
        )

    def _decode_submesh_static(
        self, sm: dict, pool_flags: dict[int, bool], causal: bool, index_sparse: bool = False
    ) -> ttnn.Tensor:
        """Run one submesh's layers over the per-step input packet / in-place caches
        (shared by the compile run and the trace capture).

        ``pool_flags`` maps each compress rate to whether this trace variant re-pools that
        compressor; ``causal`` selects causal SDPA (bounded by an on-device ``cur_pos``)
        over the additive mask for the compressor layers; ``index_sparse`` selects the
        CSA lightning-indexer program instead of dense SDPA. All three are fixed at
        capture time (a trace is a flat op sequence, so it cannot branch on the
        device-side position), which is why the capture emits one variant per
        (SDPA mode, window phase, indexer) triple -- see :meth:`_capture_traces`. Returns
        the global-last layer's head output
        ``[B, 1, 1, D]`` (``[B, 1, 1, V]`` with a folded-in ``lm_head``), or ``None`` for
        the MTP submesh, whose only job is the receive.

        The dataflow follows the pipeline-group placement (:func:`plan_layer_placement`)
        layer by layer, so this drives a recv / run / send cycle *per layer* rather than
        once per submesh:

          * The per-step inputs are ONE tiny fused INT32 packet whose first three slots are
            ``[token, pos_sliding, pos_compress]``. Global layer 0 (on submesh 0) receives
            it from the host over the H2D socket; any other layer whose predecessor lives
            on a *different* submesh receives the streams + packet from that submesh.
          * Each layer splits the packet on device and generates its RoPE rows and additive
            mask from ``pos_compress``.
          * Unless it is the global-last layer, it forwards the streams + packet to the
            submesh holding the next layer -- or, when that is this same submesh, hands them
            to the next iteration with no socket traffic. The global-last layer applies the
            head and streams the output out over the D2H socket.
          * On a 32-chip TP4 mesh the last-3-layer residuals are slice-written into one
            pack and sent on a single D2D socket to the idle MTP submesh (whose trace is
            only that receive).

        So plain round-robin sends on every layer boundary (the ring), while PGS=1 makes a
        device's contiguous run of layers chain locally.
        """
        logger.info(f"decode_submesh_static with sparse_indexer: {index_sparse}")
        if sm.get("mtp_recv"):
            self._recv_mtp_pack()
            return None
        cfg = self.config
        k = sm["index"]
        ids = self.layer_submesh_ids
        streams = None  # carried across layers that chain locally on this submesh
        pkt = None
        out = None
        # Every position-derived tensor a step needs -- the split token/positions, the RoPE
        # rows, the additive mask, the causal ``cur_pos`` and the compressor window
        # indices/RoPE -- is a pure function of this step's single token and position, so it
        # is *identical* for every layer this submesh holds. Build them once from the first
        # packet the submesh sees and reuse them across its layers (deduped by rope family /
        # layer type) rather than regenerating ~15 tiny eltwise/typecast/slice ops per
        # layer. On a device that owns several layers this removes the bulk of the per-layer
        # overhead ops (and their fixed per-op launch cost); the tensors are read-only, so
        # sharing them is exact.
        step_ctx: dict = {}

        b = self._decode_batch

        def _build_step_ctx(pkt) -> dict:
            """Split one packet ``[1,1,1,_pkt_w]`` INT32 ROW_MAJOR into the step's shared
            tensors: the uint32 ``[1,B]`` token row, the INT32 ``[B]`` sliding/compress
            position rows, one RoPE row triple per family on this submesh, and per
            layer-type additive masks / causal ``cur_pos`` / compressor window indices."""
            token = ttnn.typecast(
                ttnn.reshape(ttnn.slice(pkt, [0, 0, 0, 0], [1, 1, 1, b]), [1, b]), ttnn.uint32
            )  # [1,B]
            sliding_pos = ttnn.reshape(ttnn.slice(pkt, [0, 0, 0, b], [1, 1, 1, 2 * b]), [b])
            compress_pos = ttnn.reshape(ttnn.slice(pkt, [0, 0, 0, 2 * b], [1, 1, 1, 3 * b]), [b])
            # Two views of the same position. The scalar drives everything the batch
            # shares -- the RoPE rows and the additive mask, both broadcast over users --
            # while the per-user row drives the indices the cache writers and SDPA take
            # one of per slot. They coincide at batch 1, where the row *is* the scalar.
            pos_row_f = ttnn.typecast(
                ttnn.to_layout(ttnn.reshape(compress_pos, [1, 1, 1, b]), ttnn.TILE_LAYOUT), ttnn.float32
            )
            pos_f = (
                pos_row_f
                if b == 1
                else ttnn.typecast(
                    ttnn.to_layout(
                        ttnn.reshape(ttnn.slice(pkt, [0, 0, 0, 2 * b], [1, 1, 1, 2 * b + 1]), [1, 1, 1, 1]),
                        ttnn.TILE_LAYOUT,
                    ),
                    ttnn.float32,
                )
            )
            # One RoPE row triple per family present on this submesh ("main" / "compress").
            rope = {rt: self._device_rope(inv, sc, pos_f) for rt, (inv, sc) in sm["rope_invfreq"].items()}
            masks: dict = {}
            curpos: dict = {}
            win_rope: dict = {}
            win_idx: dict = {}
            for lt, (a, b_tbl, cr) in sm["mask_gen"].items():
                # Bound the KV axis by a position (causal) where the valid set is a
                # contiguous prefix, else by the additive mask. A sliding ring is always
                # a prefix: rows ``[0, min(pos, W - 1)]``.
                if lt == "sliding_attention":
                    masks[lt] = None
                    curpos[lt] = self._device_index(ttnn.clamp(pos_row_f, max=float(self.sliding_window - 1)))
                else:
                    use_causal = causal
                    masks[lt] = None if use_causal else self._device_mask(a, b_tbl, cr, pos_f)
                    curpos[lt] = self._device_causal_pos(cr, pos_row_f) if use_causal else None
                if lt != "sliding_attention":
                    # Incremental pooling emits one entry per closure, so generate just
                    # that entry's RoPE row (the "compress" family already in scope).
                    ws, wr, win_pos_f = self._device_compressor_indices(cr, pos_row_f, pos_f)
                    inv, sc = sm["rope_invfreq"]["compress"]
                    cw, sw, _ = self._device_rope(inv, sc, win_pos_f)
                    win_idx[lt] = (ws, wr)
                    win_rope[lt] = (cw, sw)
            return {
                "token": token,
                "sliding_pos": sliding_pos,
                "compress_pos": compress_pos,
                "rope": rope,
                "masks": masks,
                "curpos": curpos,
                "win_rope": win_rope,
                "win_idx": win_idx,
            }

        for li in sm["layers"]:
            layer = self.layers[li]
            is_first = li == 0
            is_last = li == self.num_layers - 1
            recv = li > 0 and ids[li - 1] != k

            # Obtain this layer's per-step packet (and, when it arrives over a socket, its
            # input streams) -- from the host buffer for global layer 0, from the
            # predecessor submesh when that layer sits elsewhere, else from the previous
            # iteration on this submesh.
            if is_first:
                # Stream this step's packet in from the host over the H2D socket. The op is
                # part of the trace, so replay needs no host-side dispatch: the kernel parks
                # on the socket until :meth:`_write_packet` pushes the page (which the host
                # may well have done already).
                pkt = sm["pkt"]
                ttnn.experimental.recv_async_h2d(pkt, self._pkt_socket)
                if self.tp_size > 1:
                    pkt = ttnn.broadcast(pkt, ttnn.MeshCoordinate(0, 0), cluster_axis=1)
            elif recv:
                # Receive the residual streams + fused packet from the submesh holding the
                # previous layer into the persistent buffers. Streams arrive row-major (no
                # tile padding on the wire). Captured inside the trace, so the copies need
                # no host-side dispatch at replay. Order must match the sender below.
                _, receiver_socket = self.submesh_socket_pairs[(ids[li - 1], k)]
                ttnn.experimental.recv_direct_async(sm["streams_in"], receiver_socket)
                ttnn.experimental.recv_direct_async(sm["pkt_in"], receiver_socket)
                pkt = sm["pkt_in"]

            # Build the shared per-step tensors from the first packet this submesh sees;
            # every later layer reuses them (same token, same position within a step).
            if not step_ctx:
                step_ctx = _build_step_ctx(pkt)
            token = step_ctx["token"]
            sliding_pos = step_ctx["sliding_pos"]
            compress_pos = step_ctx["compress_pos"]
            lt = cfg.layer_types[li]
            rope_type = "main" if lt == "sliding_attention" else "compress"
            cos, sin, neg_sin = step_ctx["rope"][rope_type]
            mask = step_ctx["masks"][lt]
            sdpa_cur_pos = step_ctx["curpos"][lt]
            if lt == "sliding_attention":
                cos_win = sin_win = win_slot = win_row = None
            else:
                win_slot, win_row = step_ctx["win_idx"][lt]
                cos_win, sin_win = step_ctx["win_rope"][lt]

            if is_first:
                # ``[1,B]`` ids -> ``[1,B,D]``, laid back out as one user per batch row
                # (the layout every block below runs on) and repeated over the streams.
                inputs_embeds = self.embed_tokens(token)
                dd = inputs_embeds.shape[-1]
                streams = ttnn.repeat(ttnn.reshape(inputs_embeds, [b, 1, 1, dd]), ttnn.Shape([1, 1, cfg.hc_mult, 1]))
            elif recv:
                streams = ttnn.to_layout(sm["streams_in"], ttnn.TILE_LAYOUT)
            # else the ``streams`` carried from the prior layer on this submesh are reused.

            streams = layer.decode_static(
                streams,
                cos,
                sin,
                neg_sin,
                cos_win,
                sin_win,
                mask,
                sm["scaches"][li],
                sliding_pos,
                compress_pos,
                paged=self._paged_view(sm, li, causal),
                hash_token=token if layer.mlp.is_hash else None,
                pool_compressor=(lt != "sliding_attention" and pool_flags[cfg.compress_rates[lt]]),
                sdpa_cur_pos=sdpa_cur_pos,
                win_slot=win_slot,
                win_row=win_row,
                index_sparse=index_sparse and lt == "compressed_sparse_attention",
            )
            if self.mtp_submesh is not None and li in self._dspark_tap_ids:
                self._slice_write_mtp_hidden(streams, self._dspark_tap_ids.index(li))
                if li == self._dspark_tap_ids[-1]:
                    self._send_mtp_pack()

            if is_last:
                streams = self.norm(self.hc_head(streams))
                if self._lm_head_traced is not None:
                    streams = self._lm_head_traced(streams)
                out = streams
                # Stream the step's output back to the host from inside the trace, so the
                # host reads it off the socket instead of dispatching a readback (see
                # :meth:`read_decoded_output`).
                self._send_output(out)
            elif ids[li + 1] != k:
                # Send the residual streams + fused packet to the submesh holding the next
                # layer. Streams go row-major to skip tile padding. Captured inside the
                # trace, so dispatched on device at replay (no host round-trip). Order must
                # match the receiver above.
                sender_socket, _ = self.submesh_socket_pairs[(k, ids[li + 1])]
                streams_rm = ttnn.to_layout(streams, ttnn.ROW_MAJOR_LAYOUT)
                ttnn.experimental.send_direct_async(streams_rm, sender_socket)
                ttnn.experimental.send_direct_async(pkt, sender_socket)
                streams.deallocate()
                streams_rm.deallocate()
                next_on_device = self._next_layer_on_submesh(li)
                if next_on_device is not None:
                    self.layers[next_on_device].prefetch_weights(index_sparse=index_sparse)
        return out if out is not None else streams

    def _step_tokens(self, token_id) -> list[int]:
        """The step's ``[B]`` token list, from either a scalar or a length-``batch``
        sequence (a scalar is accepted at any batch size, and feeds every user the
        same token)."""
        b = self._decode_batch
        if isinstance(token_id, int):
            return [token_id] * b
        tokens = [int(t) for t in token_id]
        if len(tokens) != b:
            raise ValueError(f"expected {b} tokens for a batch of {b}, got {len(tokens)}")
        return tokens

    def _build_packet(self, token_id, pos: int) -> torch.Tensor:
        """Host-build the whole fused packet as one INT32 socket page
        ``[1,1,1,_pkt_w]``: ``[tokens | pos_sliding | pos_compress]`` then padding, each
        region ``batch`` wide. The per-step RoPE rows and additive masks are *not* in the
        packet -- they are generated on device from ``pos_compress`` (see
        :meth:`_device_rope` / :meth:`_device_mask`)."""
        b, w = self._decode_batch, self.sliding_window
        packet = torch.zeros(1, 1, 1, self._pkt_w, dtype=torch.int32)
        packet[0, 0, 0, : self._pkt_int_prefix] = torch.tensor(
            self._step_tokens(token_id) + [pos % w] * b + [pos] * b, dtype=torch.int32
        )
        return packet

    def _write_packet(self, token_id, pos: int) -> None:
        """Push one step's fused packet into the H2D socket FIFO.

        This is the only host->device transfer of a traced step, of the packet
        ``[1,1,1,_pkt_w]`` built by :meth:`_build_packet`, and it is *not* a
        device op: the write goes straight over PCIe into the socket's L1 FIFO,
        independent of the command queue, so it can be issued before (or while) the
        traces replay. Submesh 0's in-trace ``recv_async_h2d`` pops the page into the
        persistent ``pkt`` buffer, from where it flows to the rest of the stack over
        the device-to-device sockets (so hash-MoE layers on any submesh see the token).

        One page is consumed per submesh-0 program run, so every push must be matched
        by exactly one compile run or trace replay, and vice versa.
        """
        self._pkt_socket.write_tensor(self._build_packet(token_id, pos))

    def _send_output(self, out: ttnn.Tensor) -> None:
        """Push one step's output tensor ``[B, 1, 1, N]`` into the D2H socket (inside the
        trace).

        ``send_async_d2h`` streams whole pages out of a row-major tensor, so ``out`` is
        untilized and reshaped into PCIe-aligned rows first -- one vocab-wide row would be
        a quarter-megabyte page for the sender kernel to stage in L1. The staging reshape
        to 2020 columns needs that width to divide the output's element count (true for the
        129280-wide logits: ``64 * 2020``); the row width the socket actually uses is
        re-derived below by :func:`_d2h_page_plan`.

        The socket's page size is fixed on the first call, when the output's real shape and
        dtype are finally known; the op re-checks it against the tensor on every
        program-cache miss.
        """
        out = ttnn.reshape(out, [1, 1, -1, 2020])
        out_rm = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        if self._out_plan is None:
            numel = math.prod(tuple(out_rm.shape))
            self._out_plan = _d2h_page_plan(
                numel,
                out_rm.element_size(),
                # One row is one socket page, so it has to fit the FIFO.
                page_cap_bytes=self.system_config.pipeline.d2h_fifo_bytes,
                pcie_alignment=self.system_config.pipeline.pcie_alignment,
            )
            self._out_torch_dtype = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}[out_rm.dtype]
            self._out_socket.set_page_size(self._out_plan[1] * out_rm.element_size())
        ttnn.experimental.send_async_d2h(ttnn.reshape(out_rm, list(self._out_plan)), self._out_socket)

    def read_decoded_output(self) -> torch.Tensor:
        """Stage 3 of a step: read its output off the D2H socket, as ``[B, 1, N]`` -- row
        ``u`` is slot ``u``'s user (``N`` = ``V`` with a folded-in ``lm_head``, else ``D``).

        Returns the oldest in-flight step's output.

        Blocks until the device has pushed every page of that step, so this is where a
        traced step synchronizes. Outputs are read in the order they were dispatched, and
        each read returns its own buffer, so several steps may be in flight at once (see
        :meth:`decode_traced_async`).
        """
        rows, cols = self._out_plan
        out = torch.empty(rows, cols, dtype=self._out_torch_dtype)
        self._out_socket.read_tensor(out)
        # The pages carry the output in its own element order, so the users are already
        # contiguous one after another and splitting the leading dim recovers them.
        return out.reshape(self._decode_batch, 1, -1)

    def _capture_traces(self, token_id, pos: int) -> None:
        """Capture the decode traces: per submesh, one variant per (SDPA mode, phase, indexer).

        ``token_id`` / ``pos`` describe the step about to be replayed. Each compile run
        is fed a packet (see :meth:`_write_packet`) at :meth:`_capture_packet_pos`,
        not necessarily ``pos``: masked traces only have a ``sliding_window`` page
        table, and the indexer dry run needs ``index_topk`` closed windows. Replay
        still reads the real position from the packet.

        A ttnn trace is a flat, fixed sequence of device ops, so it cannot skip the
        compressor pool on the steps that do not close a window, nor switch between causal
        and masked SDPA -- the position both would have to branch on exists only as a
        device tensor at replay. But the *choice of trace* is a host-side decision
        (``decode_traced`` knows ``pos`` before it dispatches), so both are baked into the
        capture instead: one variant per entry of :meth:`_reachable_variants`, selected per
        step by :meth:`_variant_key`. With the default rates that is five (three causal
        phases -- pool nothing / CSA / CSA+HCA -- plus the two masked phases reachable
        below the sliding window). When the CSA lightning indexer is attached, positions
        at or above ``compress_rate * index_topk`` add one more family: the same causal
        pool phases, but with CSA attending through the indexer instead of dense SDPA.
        That short-sequence prefix stays on the five dense captures.

        Variants are deduplicated per submesh via :meth:`_sm_pool_key`: a submesh only
        distinguishes what its own layers observe, so a sliding-only submesh is captured
        once and replays that single trace for every variant. A submesh with no CSA
        layer ignores the indexer bit the same way.

        Ordering matters twice over. *Every* variant's compile run (which JITs the programs
        -- trace capture itself cannot) has to be issued before the *first* capture: once a
        trace exists on a device, allocating device buffers on it is unsafe ("these buffers
        may be corrupted once a trace is executed"), and a compile run allocates freely. So
        the two passes below are not interleaved.

        Each submesh is captured independently -- capture only fixes program shapes / buffer
        addresses, so the (stale) compile-run inputs are immaterial: any cache rows the
        compile run writes are at the *same* device-indexed slots a later replay overwrites
        with real values. The real per-step results always come from the
        :meth:`decode_traced` replay loop, never the capture run.

        The compile runs are issued for *all* submeshes before synchronizing, because each
        submesh's slice contains the cross-submesh socket send/recv: a lone ``send_async``
        followed by a blocking per-submesh ``synchronize_device`` would deadlock (the residual
        streams exceed the socket's L1 buffer, so the send cannot drain until the next submesh
        posts its matching ``recv_async``). Issuing every submesh first lets the sends and
        receives pair up across devices, after which a single sync drains them. Trace capture
        only records ops (it does not execute them), so the capture loop is free of that
        hazard. Each trace's output ``[B, 1, N]`` is persistent and rewritten in place by
        every replay.
        """
        # Plan first: which submeshes need their own capture for each phase, and
        # which just alias an earlier phase's trace.
        plan = []
        planned_keys: list[set] = [set() for _ in self.submeshes_io]
        for variant in self._reachable_variants():
            causal, phase_idx, index_sparse = variant
            flags = dict(zip(self._pool_crs, self._pool_phases[phase_idx]))
            pending = []
            for i, sm in enumerate(self.submeshes_io):
                key = self._sm_pool_key(sm, flags, causal, index_sparse)
                if key not in planned_keys[i]:
                    planned_keys[i].add(key)
                    pending.append(sm)
            plan.append((variant, flags, pending))

        # Pass 1 -- every compile run, while no trace exists yet. The run is issued for
        # *all* submeshes, not just the pending ones: the slices contain the cross-submesh
        # socket send/recv, so a submesh sitting the round out would leave its neighbours'
        # sends unpaired. Re-running an already-planned submesh is harmless (the cache rows
        # it dirties are the same device-indexed slots a later replay overwrites, and they
        # stay block-bias-masked until then).
        for variant, flags, pending in plan:
            if not pending:
                continue
            causal, phase_idx, index_sparse = variant
            compile_outs = []
            # The compile run *executes*, so submesh 0's in-trace ``recv_async_h2d`` would
            # park forever without a page of its own. The packet position has to be one
            # this variant can execute (see :meth:`_capture_packet_pos`).
            packet_pos = self._capture_packet_pos(pos, causal, index_sparse)
            if packet_pos > pos:
                self.ensure_session_capacity(packet_pos)
            self._write_packet(token_id, packet_pos)
            for sm in self.submeshes_io:
                logger.info(
                    f"[traced-decode] compiling submesh {sm['index']} "
                    f"({len(sm['layers'])} layers) phase {phase_idx} pool={flags} "
                    f"causal={causal} index_sparse={index_sparse} packet_pos={packet_pos}"
                )
                compile_outs.append(self._decode_submesh_static(sm, flags, causal, index_sparse))  # JITs the programs
            # The run also *sends* an output, so drain it: an unread output would sit in
            # the socket FIFO and eventually backpressure the sender kernel. Discarded --
            # the real per-step outputs all come from the replay loop.
            self.read_decoded_output()
            for out in compile_outs:
                if out is not None:
                    out.deallocate(True)

        # Pass 2 -- record the captures and bind every variant to a trace.
        for variant, flags, pending in plan:
            causal, phase_idx, index_sparse = variant
            for sm in pending:
                device = sm["device"]
                logger.info(
                    f"[traced-decode] capturing submesh {sm['index']} "
                    f"({len(sm['layers'])} layers) phase {phase_idx} pool={flags} "
                    f"causal={causal} index_sparse={index_sparse}"
                )
                tid = ttnn.begin_trace_capture(device, cq_id=0)
                with _trace_capture_guard():
                    out = self._decode_submesh_static(sm, flags, causal, index_sparse)
                ttnn.end_trace_capture(device, tid, cq_id=0)
                # ``out`` is persistent; overwritten in place by every execute_trace.
                sm["traces"][self._sm_pool_key(sm, flags, causal, index_sparse)] = (tid, out)
            for sm in self.submeshes_io:
                sm["tids"][variant], sm["outputs"][variant] = sm["traces"][
                    self._sm_pool_key(sm, flags, causal, index_sparse)
                ]
        self._traced_captured = True

    def decode_traced(self, token_id, pos: int) -> torch.Tensor:
        """One traced decode step: feed ``token_id`` at absolute position ``pos`` and
        return the step's output, read back to host.

        Requires a prior :meth:`prepare_static_decode`. Equivalent to
        :meth:`decode_traced_async` followed by :meth:`read_decoded_output`, i.e. it
        blocks until the output has arrived. The result is ``[B, 1, V]`` logits if an
        ``lm_head`` was passed to :meth:`prepare_static_decode`, else the pre-head hidden
        ``[B, 1, D]``; its dtype is the device dtype (bf16), so cast before doing host math
        on it.

        ``token_id`` is one token for a single-user model, or one per user (in slot order)
        for a batched one; a scalar at batch > 1 feeds every user the same token. The users
        share ``pos`` -- see :meth:`prepare_static_decode`.
        """
        self.decode_traced_async(token_id, pos)
        return self.read_decoded_output()

    def decode_traced_async(self, token_id, pos: int) -> None:
        """Dispatch one traced decode step without waiting for its output.

        Captures the per-submesh traces lazily on the first call, then (every call) queues
        ``execute_trace`` on the replay thread *before* pushing this step's input packet onto
        the H2D socket. The packet receive, residual-stream handoffs and output send all
        happen from inside the traces. ``token_id`` / ``pos`` as in :meth:`decode_traced`.

        Nothing is returned: the output ``[B, 1, N]`` is in flight to the host, to be picked
        up by :meth:`read_decoded_output`. Every dispatched step must be read back exactly
        once, in dispatch order.

        Whoever reads the outputs must not be the thread that dispatches, or the steps in
        flight have to be kept to a handful. The output socket's FIFO is a single pinned host
        page (``pipeline.d2h_fifo_bytes``), so a step whose output nobody is reading stalls
        the sender kernel inside the last submesh's trace; further steps back up behind it
        through the cross-submesh sockets until every submesh's command queue is full, at
        which point a host that is still dispatching blocks on a full queue while the device
        waits for it to read. That is a deadlock, and how far ahead a single-threaded caller
        can safely run shrinks as the submesh pipeline deepens (on a 43-layer stack across
        eight submeshes: four steps in flight run, six wedge).

        Calling :meth:`read_decoded_output` from a thread of its own lifts the limit
        altogether -- the socket read releases the GIL, so the reader can sit on the socket
        while this method keeps dispatching. Then a step per session is fine; see
        ``_OutputReader`` in ``tests/test_multi_user_paged_decode_demo.py``.

        In paged mode the step belongs to whichever session is active (see
        :meth:`activate_session`), and its blocks are grown here as the compressor windows
        close.
        """
        if not getattr(self, "submeshes_io", None):
            raise RuntimeError("call prepare_static_decode() before decode_traced()")
        self.ensure_session_capacity(pos)
        # Capture first: the compile runs inside consume a packet each, so pushing this
        # step's packet before them would hand it to a compile run instead of to the
        # replay below.
        if not self._traced_captured:
            self._capture_traces(token_id, pos)
        # Kick the traces before the host packet so execute_trace is already on the
        # command queue (device parked on in-trace recv) while this thread writes PCIe.
        self.replay_traced(pos)
        self.write_step_packet(token_id, pos)

    # A step's three host-side stages, each split out so a pipelined caller can drive
    # them independently — they touch disjoint state, so they can run concurrently (on
    # separate threads) for different steps: push step n+1's packet while step n's
    # traces replay and step n-1's output is read back.
    def _ensure_replay_thread(self) -> None:
        """Start the daemon thread that drains the replay queue, once."""
        if self._replay_thread is not None:
            return

        def _run() -> None:
            """Dispatch every queued position until the ``None`` sentinel arrives."""
            for pos in iter(self._replay_queue.get, None):
                self._execute_traces(pos)

        self._replay_thread = threading.Thread(target=_run, name="decode-replay", daemon=True)
        self._replay_thread.start()

    def _execute_traces(self, pos: int) -> None:
        """Replay the variant step ``pos`` selects on every submesh (replay-thread body).

        Grows the resident sessions' blocks first, since the traces about to run read
        them, and advances their recorded positions. Non-blocking: each device's trace is
        queued on cq 0 and its output lands in that submesh's persistent trace output
        ``[B, 1, N]`` and, from the last submesh, in the D2H socket.
        """
        self.ensure_session_capacity(pos)
        for sid in self._resident:
            self._session_pos[sid] = pos + 1
        variant = self._variant_key(pos)
        for sm in self.submeshes_io:
            ttnn.execute_trace(sm["device"], sm["tids"][variant], cq_id=0, blocking=False)

    def shutdown(self) -> None:
        """Stop the traced-decode replay thread started by :meth:`_ensure_replay_thread`.

        That thread is a daemon whose target closes over ``self``, and nothing ever tells
        it to stop. CPython does not unwind a daemon thread's frame at shutdown, so the
        closure keeps the whole model -- every weight tensor, cache and trace buffer --
        alive until the process exits, where nanobind's ``Py_AtExit`` leak check reports
        the entire graph (``nanobind: leaked N instances!``). Stopping the thread releases
        that reference so the model can be collected normally.

        Idempotent, and safe whether or not a traced decode ever ran. Not safe to call
        concurrently with :meth:`decode_traced`/:meth:`replay_traced`, where the sentinel
        could overtake a step queued after it.
        """
        thread, self._replay_thread = self._replay_thread, None
        if thread is None:
            return
        # Drain what is already queued, then end the loop. The thread's only work is
        # non-blocking ``execute_trace`` dispatch, so joining cannot park on the device.
        self._replay_queue.put(None)
        thread.join()

    def write_step_packet(self, token_id, pos: int) -> None:
        """Stage 1 of a step: push its input packet ``[1,1,1,_pkt_w]`` to the device.

        Talks only to the H2D socket (a direct PCIe write, no command queue work), and
        may run ahead of the replays by as much as the socket FIFO holds.
        """
        self._write_packet(token_id, pos)

    def replay_traced(self, pos: int) -> None:
        """Stage 2 of a step: queue the traces for a step at ``pos`` on the replay thread.

        Returns immediately; ``execute_trace`` runs on that thread (``blocking=False``
        on the device), producing that submesh's ``[B, 1, N]`` trace output. Call this
        *before* :meth:`write_step_packet` so the traces can already be waiting on
        in-trace recv when the packet lands.

        Requires the traces to already be captured: dispatch one blocking
        :meth:`decode_traced` first (the compile/capture path).
        """
        if not self._traced_captured:
            raise RuntimeError("call decode_traced() once to capture the traces before replay_traced()")
        if self._variant_key(pos)[2]:
            logger.info(f"indexer trace pos={pos}")
        self._ensure_replay_thread()
        self._replay_queue.put(pos)

    def replay_traced_ahead(self, positions) -> None:
        """Queue ``execute_trace`` for every ``pos`` in ``positions`` on the replay thread.

        Call once before feeding packets so the command queues already hold the
        traces (device parked on in-trace recv, each step producing a ``[B, 1, N]``
        trace output) while the host writes H2D data.
        """
        for pos in positions:
            self.replay_traced(int(pos))
