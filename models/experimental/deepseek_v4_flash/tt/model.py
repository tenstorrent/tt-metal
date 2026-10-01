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

import collections
import contextlib
import gc
import math
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch
import ttnn
from loguru import logger

from .decode.attention import (
    CSA_INDEX_BLOCK_SIZE,
    CSA_MAX_COMPRESSED_ENTRIES,
    PAGED_KV_LAYER_TYPES,
    _StaticLayerCache,
    build_static_layer_cache,
    dense_kv_context_limit,
    dense_kv_rows,
)
from .decode.attention_csa import _scatter_window_rows
from .decode.decode_prefetch import make_decode_prefetch_buffers
from .decode.paged_cache import (
    PagedCacheFull,
    PagedGroup,
    PagedKVManager,
    PagedLayerView,
    build_groups,
    plan_pool_blocks,
)
from .common import DeepSeekV4Module, _MASK_NEG, _profile, _trace_capture_guard
from .decode.decoder_layer import DeepSeekV4DecoderLayer, _strip_prefix
from .embedding import DeepSeekV4Embedding
from .decode.hyperconnection import DeepSeekV4HyperHead
from .layers import DeepSeekV4RMSNorm, Linear, build_gcb_from_recipe, take_gcb_recipe
from .decode.moe import DeepSeekV4HashRouter, DeepSeekV4PreloadedExperts
from .prefill.attention import (
    ALIGNMENT,
    COMPRESSED_SPARSE_ATTENTION,
    HEAVILY_COMPRESSED_ATTENTION,
    SLIDING_ATTENTION,
    PrefillAttentionState,
    PrefillStaticStep,
)
from .prefill.decoder_layer import DeepSeekV4PrefillDecoderLayer, load_norm_gamma
from .prefill.hyperconnection import flatten_streams, wide_rms_norm
from .quant import dequantize_weight
from .system_config import SystemConfig, active_system_config, load_system_config, set_active_system_config
from .weight_cache import WeightCache, _as_cache, _load_weight, _materialize
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


def _create_socket_pair(from_submesh, to_submesh, socket_l1_bytes: int):
    """Directed L1 D2D socket pair ``(sender, receiver)`` between two 1xTP submeshes.

    Cores (0,0) and (0,1) of every rank, one socket eachf, rank ``r`` to rank ``r``, with
    ``socket_l1_bytes`` of L1 per core. The ``*_direct_async`` ops only push a handshake page
    through the FIFO and write the payload straight into the receiver's tensor, so the payload
    may be far larger than the FIFO.
    """
    socket_memconfig = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, socket_l1_bytes)
    socket_connections = []
    for coord in ttnn.MeshCoordinateRange(from_submesh.shape):
        for core in (ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1)):
            socket_connections.append(
                ttnn.SocketConnection(ttnn.MeshCoreCoord(coord, core), ttnn.MeshCoreCoord(coord, core))
            )
    socket_config = ttnn.SocketConfig(socket_connections, socket_memconfig)
    return ttnn.create_socket_pair(from_submesh, to_submesh, socket_config)


_PACKAGE = __name__.rsplit(".tt.", 1)[0]


def _is_gcb(value) -> bool:
    return type(value).__name__ == "global_circular_buffer"


def _is_l1_tensor(value) -> bool:
    return (
        isinstance(value, ttnn.Tensor)
        and value.storage_type() == ttnn.StorageType.DEVICE
        and value.is_allocated()
        and value.memory_config().buffer_type == ttnn.BufferType.L1
    )


def _is_device_tensor(value) -> bool:
    return isinstance(value, ttnn.Tensor) and value.storage_type() == ttnn.StorageType.DEVICE and value.is_allocated()


def _gcb_holders(root, match: Callable = _is_gcb, skip: Sequence = (), replaceable: bool = True) -> dict[int, tuple]:
    """``id(value) -> (value, [holder])`` for every ``match``-ing value (GCBs by default) reachable from
    ``root`` through this package's objects (``__dict__`` or ``__slots__``), dicts, lists and tuples; a holder is
    ``(container, key, is_attribute)``. Objects in ``skip`` are not entered. With ``replaceable`` (the holders
    will be reassigned) a match held in a tuple raises."""
    found: dict[int, tuple] = {}
    seen: set[int] = {id(obj) for obj in skip}
    stack = [root]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, dict):
            items, is_attr = list(obj.items()), False
        elif isinstance(obj, (list, tuple)):
            items, is_attr = list(enumerate(obj)), False
        elif type(obj).__module__.startswith(_PACKAGE) and hasattr(obj, "__dict__"):
            items, is_attr = list(vars(obj).items()), True
        elif type(obj).__module__.startswith(_PACKAGE) and hasattr(type(obj), "__slots__"):
            items = [(name, getattr(obj, name)) for name in type(obj).__slots__ if hasattr(obj, name)]
            is_attr = True
        else:
            continue
        for key, value in items:
            if not match(value):
                stack.append(value)
                continue
            if replaceable and isinstance(obj, tuple):
                raise RuntimeError(f"a {type(value).__name__} held in a tuple cannot be released and restored in place")
            found.setdefault(id(value), (value, []))[1].append((obj, key, is_attr))
    return found


def prefill_bias_slots(prefill) -> dict:
    """``layer -> (c_bias_slots, i_bias_slots)`` of a prefill model's CSA layers, for
    :meth:`DeepSeekV4Model.commit_prefill_state`."""
    return {
        li: (getattr(layer.self_attn, "c_bias_slots", None), getattr(layer.self_attn, "i_bias_slots", None))
        for li, layer in enumerate(prefill.layers)
        if prefill.config.layer_types[li] == COMPRESSED_SPARSE_ATTENTION
    }


def _held(holder: tuple):
    container, key, is_attr = holder
    return getattr(container, key) if is_attr else container[key]


def _assign_holder(holder: tuple, value) -> None:
    container, key, is_attr = holder
    if is_attr:
        object.__setattr__(container, key, value)
    else:
        container[key] = value


def _dspark_enabled() -> bool:
    """Whether to build the idle-row DSpark MTP link (see :class:`DeepSeekV4Model`).

    ``DEEPSEEK_V4_DSPARK=0`` turns it off. Every step of a pipeline that leaves chips
    idle otherwise taps layers 40-42 into a packed tensor, sends it over a D2D socket and
    replays an extra recv submesh on the idle row -- which only the DSpark drafting tests
    use (``test_dspark_flash_accept_rate.py`` reads it back with
    :meth:`DeepSeekV4Model.read_mtp_hiddens`).
    """
    return os.environ.get("DEEPSEEK_V4_DSPARK", "1") not in ("0", "", "false", "False")


def _traced_decode_enabled() -> bool:
    """Whether :meth:`DeepSeekV4Model.decode_traced` captures and replays traces.

    ``DEEPSEEK_V4_TRACED_DECODE=0`` runs every step eagerly instead: the same per-submesh
    :meth:`DeepSeekV4Model._decode_submesh_static` program a trace capture records, with the
    same variant selection, packet and output sockets, just dispatched op by op. Profiler
    reads inside the layers then run, which a replayed trace cannot do.
    """
    return os.environ.get("DEEPSEEK_V4_TRACED_DECODE", "1") not in ("0", "false", "False")


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
        num_stages: Optional[int] = None,
        submeshes: Optional[Sequence[ttnn.MeshDevice]] = None,
        use_prefetcher: Optional[bool] = None,
    ):
        """Build the V4-Flash model off the checkpoint.

        ``submeshes`` (with ``use_submeshes``) reuses ``1 x tp_size`` stage meshes the caller already created on
        ``full_device`` -- e.g. the ones a prefill model ran on, so its state can be committed on device -- instead
        of creating them here.

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
        both an 8-chip mesh and a larger Galaxy mesh, the remaining chips staying idle,
        unless ``num_stages`` asks for more (8 x TP4 spans a whole 32-chip Galaxy, which is
        what lets the prefill model share the decode chips, see :meth:`build_prefill`).

        The attention projections (with their compressor) and the MoE shared expert run on
        DRISC-prefetched weights instead of a DRAM->L1 copy per call, so decode must run
        inside :meth:`prefetcher_session`. One GCB per device is shared by every prefetched
        weight on it (see :func:`make_decode_prefetch_buffers`), so the cost is 288 KB of L1
        per receiver core for the whole model rather than per layer, and the prefetcher stays
        on under TP for every projection whose per-rank B-core count still matches that GCB
        (see :class:`~.decode.attention.DeepSeekV4Attention`). ``use_prefetcher=False`` builds no
        GCB at all: every projection copies its weight DRAM -> L1 per call instead (slower), and
        :meth:`prefetcher_session` is a no-op. ``None`` keeps the prefetcher on.

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
        self.use_prefetcher = True if use_prefetcher is None else bool(use_prefetcher)
        if num_prefetch_pages is None:
            num_prefetch_pages = system_config.prefetcher.num_prefetch_pages
        self._prefetch_buffers_by_device: dict[int, dict] = {}
        # GCBs dropped by :meth:`release_prefetch_buffers`: ``[(serial, recipe, holders)]``, or None.
        self._released_gcbs: Optional[list] = None
        self._evicted_l1: list = []
        # The holders of the L1 tensors the first release parked; a release after the decode capture parks these.
        self._l1_holders: list = []
        # The device tensors that outlived release_prefetch_buffers (allocated before any prefill capture), held
        # by id until snapshot_resident_state so no id is reused, and the host copies it took of every other one.
        self._pre_release: Optional[dict[int, ttnn.Tensor]] = None
        self._resident_snapshot: list = []
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
        if num_stages is not None:
            if num_stages < 1 or num_stages * tp_size > pipeline_devices:
                raise ValueError(
                    f"num_stages={num_stages} x TP{tp_size} does not fit the {pipeline_devices} pipeline devices"
                )
            pipeline_devices = num_stages * tp_size
        elif use_submeshes and tp_size == 4:
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
            if submeshes is not None:
                if len(submeshes) != self.num_submeshes:
                    raise ValueError(f"got {len(submeshes)} submeshes for {self.num_submeshes} pipeline stages")
                self.submeshes = list(submeshes)
            elif tp_size == 1:
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
            prefetch_buffers = (
                self._prefetch_buffers_for(current_device, weight_dtype, num_prefetch_pages)
                if self.use_prefetcher
                else None
            )
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
                    matmul_decode=True,
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
        self._csa_entries_copied = False
        # Paged multi-session decode state (see :meth:`prepare_static_decode`).
        self._paged: Optional[PagedKVManager] = None
        # Traced-decode replay state. :meth:`prepare_static_decode` fills in the buffers
        # and re-arms capture, but the thread handle and its queue are owned here so that
        # :meth:`shutdown` stays callable on a model that never prepared a traced decode
        # or that failed part-way through preparing one -- creating them
        # in ``prepare_static_decode`` would make the unwind itself raise and mask the
        # error being unwound.
        self._traced_captured = False
        self._traces_compiled = False
        self._eager_decode = not _traced_decode_enabled()
        # Eager mode: positions posted by :meth:`replay_traced`, each run once its packet is
        # written, and the previous step's outputs, freed once that step has been read.
        self._eager_pending: collections.deque = collections.deque()
        self._eager_outs: list = []
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
        """Directed L1 D2D socket pair ``(sender, receiver)`` between two 1xTP submeshes
        (see :func:`_create_socket_pair`). Carries the pipeline handoff payload -- residual
        streams ``[B, 1, hc, D]`` row-major plus the fused packet ``[1,1,1,_pkt_w]`` -- and,
        on the tap submesh, the packed ``[B, 3, hc, D]`` MTP residuals."""
        return _create_socket_pair(from_submesh, to_submesh, self.system_config.pipeline.socket_l1_bytes)

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

    @property
    def prefetch_buffers_released(self) -> bool:
        return self._released_gcbs is not None

    def release_prefetch_buffers(self) -> None:
        """Stop the DRISC senders and free every weight-prefetch GCB, returning their L1 to the device.

        The GCBs are permanent L1 allocations on the receiver cores (hundreds of KB per core), which is room
        prefill's wide ops need. Every holder of a GCB (the per-device mapping, each ``LinearDecode``) is
        cleared so the last reference goes and the buffer is freed; :meth:`restore_prefetch_buffers` builds
        identical ones and hands them back. Decode's other resident L1 tensors (the CSA windows, position
        bias, ...) are parked on host the same way and uploaded back. Decode cannot run in between.

        With the prefetcher on, only before the first :meth:`decode_traced`: stopping the prefetcher discards
        the prefetch requests the decode traces recorded, so captured traces would stall on replay. With it off
        the release may be repeated after the capture (e.g. around a prefill capture): it then parks the same
        L1 tensors the first release did, and :meth:`restore_prefetch_buffers` checks each comes back at the
        address the traces recorded. Idempotent.
        """
        if self._released_gcbs is not None:
            return
        if self._traced_captured and self._prefetch_buffers_by_device:
            raise RuntimeError(
                "release the prefetch buffers before the first decode_traced(): stopping the prefetcher drops the "
                "prefetch requests the captured decode traces replay"
            )
        if self._traced_captured and not self._l1_holders:
            raise RuntimeError("after the decode capture, only the L1 tensors parked before it can be parked again")
        devices = [device for device, _ in self._prefetch_buffers_by_device.values()]
        for device in devices:
            ttnn.synchronize_device(device)
            ttnn.experimental.stop_tensor_prefetcher(device)
        released = []
        for gcb, holders in _gcb_holders(self).values():
            serial, recipe = take_gcb_recipe(gcb)
            for holder in holders:
                _assign_holder(holder, None)
            released.append((serial, recipe, holders))
        gcb = None
        # A cached matmul_decode program keeps a copy of its GCB in its operation attributes, which holds the
        # L1 allocation alive after every Python reference is gone.
        for device in devices:
            device.clear_program_cache()
        if self._traced_captured:
            # The capture's outputs are L1 tensors too (held in tuples); they stay where the traces put them.
            l1 = [(_held(holders[0]), holders) for holders in self._l1_holders]
        else:
            l1 = list(_gcb_holders(self, _is_l1_tensor).values())
        evicted = []
        for tensor, holders in l1:
            memory_config, device, address = tensor.memory_config(), tensor.device(), tensor.buffer_address()
            parked = ttnn.from_device(tensor)
            ttnn.deallocate(tensor)
            for holder in holders:
                _assign_holder(holder, parked)
            evicted.append((memory_config, device, parked, holders, address))
        tensor = parked = l1 = None
        gc.collect()
        self._released_gcbs = released
        self._evicted_l1 = evicted
        self._l1_holders = [holders for *_, holders, _ in evicted]
        self._pre_release = {
            key: tensor for key, (tensor, _) in _gcb_holders(self, _is_device_tensor, replaceable=False).items()
        }
        if os.environ.get("DEEPSEEK_V4_DUMP_L1", "0") == "1":
            for i, device in enumerate(devices or self.submeshes):
                ttnn.synchronize_device(device)
                ttnn.dump_device_memory_state(device, prefix=f"after_release_sm{i}_")
        logger.info(
            f"prefetch buffers released: {len(released)} GCB(s) and {len(evicted)} L1 tensor(s) parked on host "
            f"on {len(devices)} device(s)"
        )

    def restore_prefetch_buffers(self) -> None:
        """Rebuild the GCBs :meth:`release_prefetch_buffers` freed (in their original order) and restart the
        DRISC senders. Idempotent."""
        released, self._released_gcbs = self._released_gcbs, None
        if released is None:
            return
        self._prefill_footprint = self._allocated_buffers()
        for _, recipe, holders in sorted(released, key=lambda entry: entry[0]):
            gcb = build_gcb_from_recipe(recipe)
            for holder in holders:
                _assign_holder(holder, gcb)
        evicted, self._evicted_l1 = self._evicted_l1, []
        for memory_config, device, parked, holders, address in evicted:
            tensor = ttnn.to_device(parked, device, memory_config=memory_config)
            if self._traced_captured and tensor.buffer_address() != address:
                raise RuntimeError(
                    f"an L1 tensor parked after the decode capture came back at {tensor.buffer_address():#x}, not at "
                    f"{address:#x} where the decode traces address it: something allocated in between stayed"
                )
            for holder in holders:
                _assign_holder(holder, tensor)
        devices = [device for device, _ in self._prefetch_buffers_by_device.values()]
        for device in devices:
            ttnn.experimental.start_tensor_prefetcher(device)
        for device in devices:
            ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        logger.info(f"prefetch buffers restored: {len(released)} GCB(s)")

    def snapshot_resident_state(self) -> None:
        """Keep host copies of every decode tensor a prefill replay may overwrite, for :meth:`reload_resident_state`.

        For a prefill captured between :meth:`release_prefetch_buffers` and :meth:`restore_prefetch_buffers`
        and replayed between decode steps. Its traces were recorded while only the tensors that outlived the
        release were allocated, so everything decode allocated after that (the restored L1 tensors, and what
        the decode capture created) may sit under prefill's buffers. Call once after the decode traces are
        captured; the prefill model itself is not copied. Prefetcher-off models only: a GCB's config pages
        are not reachable from here.
        """
        if self._pre_release is None:
            raise RuntimeError("call release_prefetch_buffers() before the prefill capture, and snapshot only once")
        if self._prefetch_buffers_by_device:
            raise RuntimeError("a prefill replay would overwrite the GCB config pages, which cannot be snapshot")
        skip = [self.prefill_model] if getattr(self, "prefill_model", None) is not None else []
        tensors = [
            tensor
            for key, (tensor, _) in _gcb_holders(self, _is_device_tensor, skip=skip, replaceable=False).items()
            if key not in self._pre_release
        ]
        self._pre_release = None
        self._resident_snapshot = [(ttnn.from_device(tensor), tensor) for tensor in tensors]
        logger.info(f"resident decode state: {len(tensors)} tensor(s) copied to host")
        self._audit_unsnapshot_buffers(tensors)

    def _allocated_buffers(self) -> dict[int, dict[tuple, int]]:
        """``submesh index -> {(buffer type, address): bytes per bank}`` of every buffer its allocator holds."""
        return {
            i: {
                (str(info.buffer_type), info.address): info.max_size_per_bank
                for info in ttnn._ttnn.reports.get_buffers(device)
            }
            for i, device in enumerate(self.submeshes)
        }

    def _audit_unsnapshot_buffers(self, snapshot: list) -> None:
        """Log every buffer decode allocated after the prefill capture that the snapshot does not cover.

        A prefill replay may overwrite any of them and :meth:`reload_resident_state` cannot put them back: these
        are buffers held inside ops (global semaphores, program-cache buffers) rather than by a model tensor.
        """
        footprint = getattr(self, "_prefill_footprint", None)
        if footprint is None:
            return
        covered = {(str(t.memory_config().buffer_type), t.buffer_address()) for t in snapshot}
        total = 0
        for i, buffers in self._allocated_buffers().items():
            missing = sorted(
                (key, size) for key, size in buffers.items() if key not in footprint.get(i, {}) and key not in covered
            )
            total += len(missing)
            for (buffer_type, address), size in missing:
                logger.warning(
                    f"submesh {i}: {buffer_type} buffer at {address:#x} ({size} B/bank) allocated after the prefill "
                    "capture is not in the resident snapshot"
                )
        logger.info(f"resident snapshot audit: {total} buffer(s) a prefill replay could overwrite unrestored")

    def reload_resident_state(self) -> None:
        """Write :meth:`snapshot_resident_state`'s copies back in place, after a prefill replay and before decode.

        The addresses do not change, so the decode traces stay valid. Per-sequence state among them is then
        reset / committed as usual (:meth:`reset_static_caches`, :meth:`commit_prefill_state`).
        """
        for host, tensor in self._resident_snapshot:
            ttnn.copy_host_to_device_tensor(host, tensor)

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
            matmul_decode=True,
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

    def _csa_dense_entries(self) -> int:
        """Compressed entries the dense CSA ``scache.kv`` holds after the ring.

        Without an indexer that is the fixed dense cap. With one, dense CSA runs up to
        :meth:`_index_dense_limit` (or ``max_seq``), so the buffer covers every window
        closed by then, rounded to whole ``CSA_INDEX_BLOCK_SIZE`` blocks for the one-off
        copy into ``comp_kv`` at the switch.
        """
        if not self._indexer_active():
            return CSA_MAX_COMPRESSED_ENTRIES
        cr = int(self.config.compress_rates["compressed_sparse_attention"])
        tokens = min(self._index_dense_limit(), self._decode_max_seq)
        entries = -(-tokens // cr)
        blocks = -(-entries // CSA_INDEX_BLOCK_SIZE)
        return max(CSA_MAX_COMPRESSED_ENTRIES, blocks * CSA_INDEX_BLOCK_SIZE)

    def _copy_dense_csa_entries(self) -> None:
        """Copy every dense CSA entry (``scache.kv`` rows past the ring) into ``comp_kv``.

        Runs once, eagerly, before the first indexer-trace step: the dense trace appends
        pooled entries to ``scache.kv`` only, while the indexer gathers from ``comp_kv``.
        Queued on cq 0 from the replay thread, so it lands after the last dense step.
        """
        w = self.sliding_window
        for sm in self.submeshes_io:
            for scache in sm["scaches"].values():
                if scache.comp_kv is None:
                    continue
                bsz, heads, rows, dh = scache.kv.shape
                entries = rows - w
                n_blocks = entries // CSA_INDEX_BLOCK_SIZE
                dense = ttnn.slice(scache.kv, [0, 0, w, 0], [bsz, heads, rows, dh])
                dense_rm = ttnn.to_layout(dense, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(dense)
                blocks = ttnn.reshape(dense_rm, [n_blocks, 1, CSA_INDEX_BLOCK_SIZE, dh])
                ttnn.experimental.slice_write(
                    blocks, scache.comp_kv, [0, 0, 0, 0], [n_blocks, 1, CSA_INDEX_BLOCK_SIZE, dh], [1, 1, 1, 1]
                )
                ttnn.deallocate(dense_rm)

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
    #     reserving ``N x max_context`` (see :mod:`.decode.paged_cache`).
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
        self._wait_replays_dispatched()
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
        self._wait_replays_dispatched()
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

        self._wait_replays_dispatched()
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
                    # The indexer's paged pools are only read below the step's closed-window
                    # count, so they need no blanking; they are not saved per session either.
                    if name == "idx_key_cache" and scache.idx_page_table is not None:
                        continue
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
            csa_dense_entries=self._csa_dense_entries(),
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

        ``prev_gate`` / ``idx_prev_gate`` are refilled with ``_MASK_NEG`` rather than 0, matching
        how :func:`build_static_layer_cache` allocates them: they gate window 0's absent Ca half,
        which a 0 fill would give real softmax weight instead of none.

        ``idx_page_table`` is left alone: it is the constant identity mapping the indexer
        traces read ``idx_key_cache`` / ``comp_kv`` through, not per-sequence state, and
        zeroing it points every block at block 0.

        The KV caches live in the block pools; use :meth:`reset_session` to rewind one
        session.
        """
        if not getattr(self, "submeshes_io", None):
            raise RuntimeError("call prepare_static_decode() before reset_static_caches()")
        for sm in self.submeshes_io:
            for scache in sm["scaches"].values():
                for name in _StaticLayerCache.__slots__:
                    if name == "idx_page_table":
                        continue
                    buf = getattr(scache, name)
                    if buf is not None:
                        ttnn.fill(buf, self._empty_compressor_fill(name), output_tensor=buf)

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
        same row count for every layer type (see :func:`.decode.paged_cache.build_groups`).

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
        if self._indexer_active():
            # CSA attends the indexer's top-k gathered from max_seq-sized pools.
            self._context_limit = None
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
                dense_rows = dense_kv_rows(lt, w, self._csa_dense_entries())
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
        self._traces_compiled = False

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
                    if curpos[lt] is not None and lt == "compressed_sparse_attention" and index_sparse:
                        # The selected KV holds at most CSA_MAX_COMPRESSED_ENTRIES entries.
                        last_row = float(self.sliding_window + CSA_MAX_COMPRESSED_ENTRIES - 1)
                        thr = ttnn.floor(ttnn.multiply(ttnn.add(pos_row_f, 1.0), 1.0 / cr))
                        curpos[lt] = self._device_index(
                            ttnn.clamp(ttnn.add(thr, float(self.sliding_window - 1)), max=last_row)
                        )
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
                    pkt = ttnn.broadcast(pkt, ttnn.MeshCoordinate(0, 0), cluster_axis=1, topology=ttnn.Topology.Ring)
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
                if self.use_prefetcher and next_on_device is not None:
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
        plan = self._capture_plan()
        if not self._traces_compiled:
            self._compile_plan(plan, token_id, pos)

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

    def compile_traces(self, token_id, pos: int) -> None:
        """Run every decode variant's compile pass now, without capturing (the first :meth:`decode_traced` then
        only captures).

        For a prefill captured after this, between :meth:`release_prefetch_buffers` and
        :meth:`restore_prefetch_buffers`: the buffers decode's ops keep for themselves (global semaphores,
        cached constants) are allocated here, before prefill lays out its memory, so a prefill replay cannot
        overwrite them; :meth:`snapshot_resident_state` only reaches the model's own tensors.
        """
        if self._traced_captured:
            raise RuntimeError("the decode traces are already captured")
        if not self._traces_compiled and not self._eager_decode:
            self.ensure_session_capacity(pos)
            self._compile_plan(self._capture_plan(), token_id, pos)

    def _capture_plan(self) -> list:
        """``[(variant, pool flags, submeshes to capture)]``: which submeshes need their own capture for each
        phase, and which just alias an earlier phase's trace."""
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
        return plan

    def _compile_plan(self, plan: list, token_id, pos: int) -> None:
        """Pass 1 of :meth:`_capture_traces`: one executed (compile) run per planned variant."""
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
        self._traces_compiled = True

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
        if not self._traced_captured and not self._eager_decode:
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
                try:
                    self._execute_traces(pos)
                finally:
                    self._replay_queue.task_done()

        self._replay_thread = threading.Thread(target=_run, name="decode-replay", daemon=True)
        self._replay_thread.start()

    def _wait_replays_dispatched(self) -> None:
        """Block until the replay thread has dispatched every step queued so far.

        Its entries read the resident sessions when they run, not when they were queued,
        so this must come before anything that changes who is seated. It only waits for
        the non-blocking ``execute_trace`` calls to be issued, not for the device to run them.
        """
        thread = self._replay_thread
        if thread is None or thread is threading.current_thread():
            return
        done = self._replay_queue.all_tasks_done
        with done:
            while self._replay_queue.unfinished_tasks and thread.is_alive():
                done.wait(0.1)

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
        if variant[2] and not self._csa_entries_copied:
            self._copy_dense_csa_entries()
        self._csa_entries_copied = variant[2]
        if self._eager_decode:
            self._run_eager_step(variant)
            return
        for sm in self.submeshes_io:
            ttnn.execute_trace(sm["device"], sm["tids"][variant], cq_id=0, blocking=False)

    def _run_eager_step(self, variant: tuple[bool, int, bool]) -> None:
        """Eager body of :meth:`_execute_traces`: the capture's compile run at the real position.

        Every submesh is issued before any output is read, as in :meth:`_capture_traces`,
        so the cross-submesh sends and receives pair up. The outputs are persistent until
        the next step, by which time :meth:`read_decoded_output` has drained this one.
        """
        for out in self._eager_outs:
            out.deallocate(True)
        self._eager_outs = []
        causal, phase_idx, index_sparse = variant
        flags = dict(zip(self._pool_crs, self._pool_phases[phase_idx]))
        for sm in self.submeshes_io:
            out = self._decode_submesh_static(sm, flags, causal, index_sparse)
            if out is not None:
                self._eager_outs.append(out)

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

        In eager mode this also runs the step :meth:`replay_traced` posted for ``pos``;
        the packet goes first, as in a compile run, since the eager layers may sync.
        """
        self._write_packet(token_id, pos)
        if self._eager_decode:
            posted = self._eager_pending.popleft()
            assert posted == pos, f"eager decode: packet for pos {pos}, but pos {posted} was posted next"
            self._execute_traces(pos)

    def replay_traced(self, pos: int) -> None:
        """Stage 2 of a step: queue the traces for a step at ``pos`` on the replay thread.

        Returns immediately; ``execute_trace`` runs on that thread (``blocking=False``
        on the device), producing that submesh's ``[B, 1, N]`` trace output. Call this
        *before* :meth:`write_step_packet` so the traces can already be waiting on
        in-trace recv when the packet lands.

        Requires the traces to already be captured: dispatch one blocking
        :meth:`decode_traced` first (the compile/capture path).
        """
        if self._eager_decode:
            self._eager_pending.append(pos)
            return
        if not self._traced_captured:
            raise RuntimeError("call decode_traced() once to capture the traces before replay_traced()")
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

    def decode_prompt_traced(
        self, token_ids, start_pos: int, on_output: Optional[Callable[[int, torch.Tensor], None]] = None
    ) -> Optional[torch.Tensor]:
        """Feed known tokens ``token_ids`` at ``start_pos, start_pos + 1, ...`` through traced decode
        (prefill-via-decode) without waiting for one step's output before feeding the next.

        Every step's traces are queued up front, a writer thread pushes the input packets (the H2D
        FIFO paces it) and the calling thread reads the outputs in order, so the submesh pipeline holds
        as many steps as it has stages. ``on_output(i, out)`` sees step ``i``'s ``[B, 1, N]`` output as
        it arrives. Returns the last step's output (``None`` for no tokens).
        """
        ids = [token_ids[i] for i in range(len(token_ids))]
        if not ids:
            return None
        start_pos = int(start_pos)
        out = None
        if not self._traced_captured or self._eager_decode:
            # The capture consumes packets itself, and eager steps run on the packet writer: one at a time.
            out = self.decode_traced(ids[0], start_pos)
            if on_output is not None:
                on_output(0, out)
            if self._eager_decode:
                for i in range(1, len(ids)):
                    out = self.decode_traced(ids[i], start_pos + i)
                    if on_output is not None:
                        on_output(i, out)
                return out
            ids, first = ids[1:], 1
            if not ids:
                return out
        else:
            first = 0
        positions = range(start_pos + first, start_pos + first + len(ids))
        self.ensure_session_capacity(positions[-1])
        self.replay_traced_ahead(positions)

        error: list[BaseException] = []

        def _write() -> None:
            try:
                for token_id, pos in zip(ids, positions):
                    self.write_step_packet(token_id, pos)
            except BaseException as exc:  # surfaced by the caller after the join
                error.append(exc)

        writer = threading.Thread(target=_write, name="decode-prompt-writer", daemon=True)
        writer.start()
        try:
            for i in range(len(ids)):
                if error:
                    break
                out = self.read_decoded_output()
                if on_output is not None:
                    on_output(first + i, out)
        finally:
            writer.join()
        if error:
            raise error[0]
        return out

    # -- prefill: the decode chips, the decode experts, the decode buffers as its state -- #
    def build_prefill(self, rope: dict, weights: dict, lm_head=None, **kwargs) -> "DeepSeekV4PrefillModel":
        """Build the :class:`DeepSeekV4PrefillModel` on this model's own pipeline stages.

        Every prefill layer lives on the submesh of the decode layer it mirrors and reads that layer's
        routed experts in place; the embedding table (and ``lm_head``, when given) are shared too, so the
        prefill adds only its own attention / router / shared-expert layouts. ``weights`` is
        :func:`~.prefill.weights.checkpoint_weights`; ``kwargs`` go to the prefill constructor.

        Build it right after :meth:`prepare_static_decode` and before the first :meth:`decode_traced`
        (and :meth:`DeepSeekV4PrefillModel.prepare_traced_prefill` there too, when prefill is traced, with the
        prefetch buffers released so the capture sees the L1 prefill runs with), so its weights and
        persistent buffers are allocated before any decode trace exists.
        """
        if not self.use_submeshes:
            raise NotImplementedError("prefill shares the decode pipeline stages: build with use_submeshes=True")
        self.prefill_model = DeepSeekV4PrefillModel(
            self.config,
            weights,
            self.first_device,
            rope,
            experts=[layer.mlp.experts for layer in self.layers],
            num_layers=self.num_layers,
            tp_size=self.tp_size,
            layer_devices=self.layer_devices,
            embedding_weight=self.embed_tokens.embedding_weight,
            lm_head=lm_head,
            **kwargs,
        )
        # Each receiver FIFO is permanent L1 on its stage, so a second set would cost prefill's ops that room.
        self.prefill_model.shared_socket_pairs = {
            (id(self.submeshes[from_id]), id(self.submeshes[to_id])): pair
            for (from_id, to_id), pair in self.submesh_socket_pairs.items()
        }
        return self.prefill_model

    def prefill(
        self,
        input_ids,
        session_id: int,
        chunk_size: int = 1024,
        traced: bool = False,
        on_chunk: Optional[Callable[[int, int, int, float], None]] = None,
        progress: Optional[Callable[[str], None]] = None,
        capture_token: int = 0,
    ) -> ttnn.Tensor:
        """Prefill ``input_ids`` (a multiple of ``ALIGNMENT`` tokens) into session ``session_id``.

        With the prefetch GCBs released (:meth:`release_prefetch_buffers`, a no-op if they already are), runs
        :meth:`build_prefill`'s model over the prompt (replaying its traces with ``traced=True``); then restores
        the GCBs, captures the decode traces with a throw-away step of ``capture_token`` if they do not exist
        yet (its scratch cache writes are overwritten next), and commits the attention state into the decode
        buffers (:meth:`commit_prefill_state`). The next :meth:`decode_traced` continues at ``len(input_ids)``.
        Returns the prompt's last-token logits ``[1, 1, 1, V]``: on the last stage, or on the host with ``traced``.
        """
        prefill = getattr(self, "prefill_model", None)
        if prefill is None:
            raise RuntimeError("call build_prefill() first")
        self.release_prefetch_buffers()
        if traced:
            logits, states = prefill.prefill_traced(input_ids, on_chunk=on_chunk)
        else:
            logits, states = prefill.prefill(input_ids, chunk_size=chunk_size, on_chunk=on_chunk)
        prefill.synchronize("prefill done")
        self.restore_prefetch_buffers()
        if not self._traced_captured:
            self.decode_traced(capture_token, 0)
        self.commit_prefill_state(states, session_id, progress=progress)
        return logits

    def commit_prefill_state(
        self,
        states: list[PrefillAttentionState],
        session_id: int,
        progress: Optional[Callable[[str], None]] = None,
        bias_slots: Optional[dict] = None,
    ) -> int:
        """Write a prefill's per-layer ``states`` into the decode buffers of ``session_id``; returns ``T``.

        ``bias_slots`` (``layer -> (c_bias_slots, i_bias_slots)``, see :func:`prefill_bias_slots`) is all this
        needs of the prefill model, so a caller that freed it can still commit; default: the :meth:`build_prefill`
        model's.

        Prefill and decode share every layer's submesh, so this is a device-side rewrite (nothing crosses the
        host). What goes where, for a prompt of ``T`` tokens (``T`` a multiple of ``sliding_window``, so no
        compressor window is half full and ring slot ``j`` holds token ``T - W + j``):

        * sliding / CSA: ``kv_tail`` then ``compressed_kv`` fill ``scache.kv`` rows ``[0, W)`` and
          ``[W, W + T/cr)`` (as many entries as the dense buffer holds; the rest are zeroed);
        * HCA: the same ring + entries axis, block by block through the session's page table, into the pool;
        * CSA overlap: the Ca half of ``prev_kv`` / ``prev_gate``. Prefill keeps the gate with the compressor
          ``position_bias`` added, decode adds it inside ``csa_pool_window``, so the bias is subtracted; the
          Cb half is neutral (zero kv, ``_MASK_NEG`` gate). The half-open ``win_*`` need nothing;
        * CSA lightning indexer (when prefill ran it): the index keys fill ``idx_key_cache`` and the entries
          ``comp_kv`` (identity page table), and the indexer compressor's overlap fills ``idx_prev_*``.

        The session must be active and the decode traces captured (so the throw-away capture step's scratch
        writes are the ones overwritten). The buffers are written in place; the traces keep their addresses.
        """
        note = progress or (lambda message: None)
        config = self.config
        if bias_slots is None:
            prefill = getattr(self, "prefill_model", None)
            if prefill is None:
                raise RuntimeError("call build_prefill() first, or pass bias_slots")
            bias_slots = prefill_bias_slots(prefill)
        if not self.paged:
            raise RuntimeError("call prepare_static_decode() first")
        if len(states) < self.num_layers:
            raise ValueError(f"need {self.num_layers} layer states, got {len(states)}")
        lengths = {state.seq_len for state in states[: self.num_layers]}
        if len(lengths) != 1:
            raise ValueError(f"layer states disagree on the prompt length: {sorted(lengths)}")
        total = lengths.pop()
        if total <= 0 or total % ALIGNMENT:
            raise ValueError(f"prefilled length {total} must be a positive multiple of {ALIGNMENT}")
        with_indexer = all(
            state.idx_keys is not None
            for state, layer_type in zip(states, config.layer_types[: self.num_layers])
            if layer_type == COMPRESSED_SPARSE_ATTENTION
        )
        if self._indexer_active() and not with_indexer and self._index_sparse_step(total):
            raise ValueError(
                f"a prompt of {total} tokens already needs the CSA lightning indexer in decode, whose key cache "
                "prefill does not fill (build the prefill with lightning_indexer=True, or prefill at most "
                "index_topk * compress_rate - 1 tokens)"
            )

        self.ensure_session_capacity(total - 1)
        for sm in self.submeshes_io:
            for li in sm["layers"]:
                state, layer_type = states[li], config.layer_types[li]
                if list(state.kv_tail.device().get_device_ids()) != list(sm["device"].get_device_ids()):
                    raise RuntimeError(f"prefill layer {li}'s state is not on its decode submesh")
                scache = sm["scaches"][li]
                c_bias, i_bias = bias_slots.get(li, (None, None))
                note(f"commit layer {li + 1}/{self.num_layers} ({layer_type}) on decode submesh {sm['index']}")
                if layer_type == HEAVILY_COMPRESSED_ATTENTION:
                    self._commit_paged_kv(sm, li, session_id, state)
                else:
                    self._commit_dense_kv(scache, li, state)
                if layer_type == COMPRESSED_SPARSE_ATTENTION:
                    self._commit_csa_overlap(
                        scache.prev_kv, scache.prev_gate, state.csa_prev_kv, state.csa_prev_gate, c_bias
                    )
                    if state.idx_keys is not None and scache.idx_key_cache is not None:
                        self._commit_index_keys(scache, state)
                        self._commit_csa_overlap(
                            scache.idx_prev_kv,
                            scache.idx_prev_gate,
                            state.idx_prev_kv,
                            state.idx_prev_gate,
                            i_bias,
                        )
        for sm in self.submeshes_io:
            ttnn.synchronize_device(sm["device"])
        note("prefill state committed")
        return total

    @staticmethod
    def _overwrite(dst: ttnn.Tensor, src: ttnn.Tensor) -> None:
        """Copy ``src`` into the persistent ``dst`` (same shape / dtype / layout), keeping ``dst``'s address."""
        if tuple(src.shape) != tuple(dst.shape):
            raise ValueError(f"cannot write a {tuple(src.shape)} tensor into a {tuple(dst.shape)} buffer")
        if src.memory_config() == dst.memory_config():
            ttnn.copy(src, dst)
        else:
            ttnn.to_memory_config(src, dst.memory_config(), output_tensor=dst)

    @staticmethod
    def _stack_rows(parts: list, rows: int, width: int, device) -> ttnn.Tensor:
        """ROW_MAJOR ``[1, 1, n_i, width]`` ``parts`` stacked on the row axis, zero-padded to ``rows`` (a new tensor)."""
        used = sum(p.shape[2] for p in parts)
        if used > rows:
            raise ValueError(f"{used} rows do not fit a {rows}-row buffer")
        if used < rows:
            pad = ttnn.zeros(
                [1, 1, rows - used, width], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )
            parts = [*parts, pad]
        if len(parts) == 1:
            return ttnn.clone(parts[0])
        return ttnn.concat(parts, dim=2)

    def _commit_dense_kv(self, scache, li: int, state: PrefillAttentionState) -> None:
        """Ring + entries into a sliding / CSA layer's dense ``scache.kv`` (TILE ``[1, 1, rows, Dh]``)."""
        w = self.sliding_window
        _, _, rows, dh = scache.kv.shape
        parts, scratch = [state.kv_tail], []
        entries = state.compressed_kv
        if entries is not None:
            fit = min(entries.shape[2], rows - w)
            if fit < entries.shape[2]:
                if state.idx_keys is None:
                    raise ValueError(
                        f"layer {li}: {w + entries.shape[2]} KV rows do not fit the decode buffer's {rows} "
                        f"(the dense CSA buffer holds {rows - w} compressed entries)"
                    )
                # The indexer path reads comp_kv; the dense buffer only needs the entries it can hold.
                entries = ttnn.slice(entries, [0, 0, 0, 0], [1, 1, fit, dh])
                scratch.append(entries)
            parts.append(entries)
        stacked = self._stack_rows(parts, rows, dh, scache.kv.device())
        tiled = ttnn.to_layout(stacked, ttnn.TILE_LAYOUT)
        self._overwrite(scache.kv, tiled)
        for t in (*scratch, stacked, tiled):
            ttnn.deallocate(t)

    def _commit_paged_kv(self, sm: dict, li: int, session_id: int, state: PrefillAttentionState) -> None:
        """Ring + entries into an HCA layer's block pool, one block per ``fill_cache`` at its physical index."""
        layer_type = self.config.layer_types[li]
        group = self._paged_groups[layer_type]
        page_row = self._require_paged().page_row(session_id, layer_type)[0].tolist()
        block, dh = group.block_size, self.config.head_dim
        parts = [state.kv_tail] + ([state.compressed_kv] if state.compressed_kv is not None else [])
        blocks = math.ceil(sum(p.shape[2] for p in parts) / block)
        stacked = self._stack_rows(parts, blocks * block, dh, sm["device"])
        axis = ttnn.to_layout(stacked, ttnn.TILE_LAYOUT)
        ttnn.deallocate(stacked)
        pool = sm["pools"][li]
        for logical in range(blocks):
            physical = page_row[logical]
            if physical == 0:
                raise RuntimeError(f"logical block {logical} of {layer_type} is unmapped: capacity was not ensured")
            rows = ttnn.slice(axis, [0, 0, logical * block, 0], [1, 1, (logical + 1) * block, dh])
            ttnn.fill_cache(pool, rows, physical)
            ttnn.deallocate(rows)
        ttnn.deallocate(axis)

    def _commit_csa_overlap(
        self,
        kv_dst: ttnn.Tensor,
        gate_dst: ttnn.Tensor,
        kv_src: ttnn.Tensor,
        gate_src: ttnn.Tensor,
        bias_slots: list,
    ) -> None:
        """A prefill CSA overlap (fp32 ``[1, 1, 1, cr * dim]``, slot-major Ca, gate + bias) into decode's
        ``[cr, 1, 1, 2 * dim]`` ``prev_*`` windows: Ca half from prefill (bias removed), Cb half neutral."""
        rate = len(bias_slots)
        dim = kv_src.shape[3] // rate
        device = kv_dst.device()

        def slots(src: ttnn.Tensor) -> ttnn.Tensor:  # [1, 1, 1, cr * dim] -> [cr, 1, 1, dim] fp32 TILE
            return ttnn.to_layout(ttnn.reshape(src, [rate, 1, 1, dim]), ttnn.TILE_LAYOUT)

        bias = ttnn.concat([ttnn.slice(b, [0, 0, 0, 0], [1, 1, 1, dim]) for b in bias_slots], dim=0)
        raw_gate = slots(gate_src)
        gate = ttnn.subtract(raw_gate, bias)
        ttnn.deallocate(raw_gate)
        ttnn.deallocate(bias)
        for ca, dst, cb_fill in ((slots(kv_src), kv_dst, 0.0), (gate, gate_dst, _MASK_NEG)):
            cb = ttnn.full([rate, 1, 1, dim], cb_fill, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
            pair = ttnn.concat([ca, cb], dim=-1)
            window = ttnn.to_layout(ttnn.typecast(pair, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)
            self._overwrite(dst, window)
            for t in (ca, cb, pair, window):
                ttnn.deallocate(t)

    def _commit_index_keys(self, scache, state: PrefillAttentionState) -> None:
        """Index keys into ``idx_key_cache`` and the entries into ``comp_kv`` (identity paging: entry ``w`` at
        block ``w // block``, row ``w % block``)."""
        n_blocks, _, block, _ = scache.idx_key_cache.shape
        keys, entries = state.idx_keys, state.compressed_kv
        if keys.shape[2] != entries.shape[2]:
            raise ValueError(f"{keys.shape[2]} index keys for {entries.shape[2]} compressed entries")
        if keys.shape[2] > n_blocks * block:
            raise ValueError(f"{keys.shape[2]} index keys do not fit the decode cache's {n_blocks * block} rows")
        device = scache.idx_key_cache.device()

        def paged(rows: ttnn.Tensor) -> ttnn.Tensor:  # ROW_MAJOR [n_blocks, 1, block, width]
            width = rows.shape[3]
            flat = self._stack_rows([rows], n_blocks * block, width, device)
            return ttnn.reshape(flat, [n_blocks, 1, block, width])

        keys_rm = paged(keys)
        keys_tiled = ttnn.to_layout(keys_rm, ttnn.TILE_LAYOUT)
        self._overwrite(scache.idx_key_cache, keys_tiled)
        ttnn.deallocate(keys_rm)
        ttnn.deallocate(keys_tiled)
        if scache.comp_kv is not None:
            kv_rm = paged(entries)
            self._overwrite(scache.comp_kv, kv_rm)
            ttnn.deallocate(kv_rm)


# ================================================================================================= #
# Prefill: the whole network over a prompt, many tokens per call.
#
# :class:`DeepSeekV4PrefillModel` is the prefill counterpart of :class:`DeepSeekV4Model`, following the
# reference ``DeepseekV4ForCausalLM``::
#
#     streams = embed_tokens(ids) expanded to hc parallel streams        # [1, T, hc, D]
#     for layer in layers: streams = layer(streams)                      # PrefillDecoderLayer
#     hidden  = norm(hc_head(streams))                                   # collapse the streams, RMSNorm
#     logits  = lm_head(hidden)
#
# It composes :class:`~.prefill.decoder_layer.DeepSeekV4PrefillDecoderLayer` with the embedding, a prefill
# :class:`DeepSeekV4PrefillHyperHead`, the final RMSNorm and ``lm_head``. Prompts are multiples of
# ``ALIGNMENT`` (128) tokens; a ragged tail is left to decode.
#
# Devices. ``layer_devices`` places each layer on a ``1 x tp_size`` mesh (attention heads and the MoE
# intermediate width sharded over its ranks, everything else replicated); the embedding sits on the first
# stage and the head on the last. :meth:`DeepSeekV4Model.build_prefill` puts every layer on its decode
# submesh and hands in decode's routed experts, embedding table and ``lm_head``, so the two models share the
# chips and the big weights, and :meth:`DeepSeekV4Model.commit_prefill_state` moves the finished state into
# decode's buffers without leaving the device. The residual streams cross a stage boundary through the host
# (one ``[1, T, hc, D]`` tensor per chunk per boundary).
#
# Weights use the checkpoint's names (no ``model.`` prefix): ``embed_tokens.weight``,
# ``layers.{i}.<decoder-layer keys>``, ``hc_head.{hc_fn,hc_base,hc_scale}``, ``norm.weight`` and
# ``lm_head.weight``, each a torch tensor or a zero-arg thunk. The routed experts come either from
# ``expert_provider(layer_idx)`` (uploaded here) or as ready-made ``experts``.
# ================================================================================================= #


class DeepSeekV4PrefillHyperHead(DeepSeekV4Module):
    """ttnn prefill port of ``DeepseekV4HyperHead`` (the final stream collapse).

    The multi-token counterpart of the decode :class:`~.decode.hyperconnection.DeepSeekV4HyperHead`::

        flat = unweighted_rmsnorm(streams.flatten(2))
        pre  = sigmoid(hc_fn @ flat * hc_scale + hc_base) + eps
        out  = (pre[..., None] * streams).sum(dim=2)

    Unlike the layers' hyper-connections there is no ``post`` / ``comb``: the head only produces the
    collapsed sequence. ``weights`` keys: ``hc_fn`` ``[hc, hc*D]``, ``hc_base`` ``[hc]``, ``hc_scale``
    (a single scalar). The decode head keeps its operands in width-sharded L1, which is sized for a few
    token rows; this one runs on DRAM-interleaved tensors.
    """

    def __init__(
        self,
        config,
        weights: dict,
        device: ttnn.MeshDevice,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.device = device
        self.hc = config.hc_mult
        self.hidden = config.hidden_size
        self.eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        cache = _as_cache(cache)

        self.fn = Linear(weights["hc_fn"], device, cache.file("hc_fn.prefill"), dtype=weight_dtype)  # [hc, hc*D]
        base_src = weights["hc_base"]
        base_file = cache.file("hc_base.prefill")
        base = _materialize(
            lambda: (base_src() if callable(base_src) else base_src).detach().reshape(1, 1, 1, self.hc),
            base_file,
            ttnn.bfloat16,
        )
        self.base = _load_weight(base, device, cache_file_name=base_file)
        scale_src = weights["hc_scale"]
        self.scale = float((scale_src() if callable(scale_src) else scale_src).flatten().tolist()[0])

    def forward(self, streams: ttnn.Tensor) -> ttnn.Tensor:
        """``streams`` ``[1, T, hc, D]`` TILE -> ``[1, 1, T, D]`` TILE, the collapsed sequence."""
        _, t, hc, d = streams.shape
        if hc != self.hc or d != self.hidden or streams.shape[0] != 1:
            raise ValueError(f"expected streams [1, T, {self.hc}, {self.hidden}], got {tuple(streams.shape)}")

        flat = flatten_streams(streams)
        normed = wide_rms_norm(flat, self.norm_eps)
        ttnn.deallocate(flat)
        mixes = self.fn(normed)  # [1, 1, T, hc]
        ttnn.deallocate(normed)
        pre = ttnn.add(ttnn.sigmoid(ttnn.add(ttnn.multiply(mixes, self.scale), self.base)), self.eps)
        ttnn.deallocate(mixes)

        # Weight each stream by its own per-token scalar and sum the streams. Streams go stream-major
        # ([1, hc, T, D]) so the sum runs over a plain outer axis, and ``pre`` to [1, hc, T, 1] so it
        # broadcasts across D: in ``[1, T, hc, D]`` the ``hc`` rows of a token share one tile.
        rows = ttnn.to_layout(streams, ttnn.ROW_MAJOR_LAYOUT)
        stream_major = ttnn.to_layout(ttnn.permute(rows, (0, 2, 1, 3)), ttnn.TILE_LAYOUT)
        ttnn.deallocate(rows)
        weights = ttnn.permute(pre, (0, 3, 2, 1))
        ttnn.deallocate(pre)
        weighted = ttnn.multiply(stream_major, weights)
        ttnn.deallocate(stream_major)
        ttnn.deallocate(weights)
        out = ttnn.sum(weighted, dim=1, keepdim=True)  # [1, 1, T, D]
        ttnn.deallocate(weighted)
        return out


class DeepSeekV4PrefillModel(DeepSeekV4Module):
    """ttnn prefill port of ``DeepseekV4ForCausalLM`` (see the prefill section comment above).

    ``num_layers`` builds only the first layers of the stack (bring-up, or a reduced test model). Weight
    dtypes: ``weight_dtype`` for the attention projections, ``moe_weight_dtype`` for the routers and the
    shared experts, ``expert_dtype`` for the routed experts this class uploads (``None`` takes the system
    profile's default) and ``head_dtype`` for ``lm_head``.

    ``device`` holds the embedding. ``layer_devices`` (one entry per layer, default ``device`` for all)
    places each layer; the head lives with the last layer. ``tp_size`` is the width of the ``1 x tp_size``
    meshes the layers run on (1 for a single device), and ``dense_csa`` lifts the CSA length limit without
    the lightning indexer (see :class:`~.prefill.attention.DeepSeekV4PrefillAttention`); ``lightning_indexer``
    runs the real indexer on CSA layers instead (the model's exact attention at any length, eager and traced).

    ``embedding_weight`` (a ROW_MAJOR ``[V, D]`` table on ``device``) and ``lm_head`` (a :class:`~.layers.Linear`
    on the last layer's device) reuse tensors another model already holds instead of uploading copies.
    """

    def __init__(
        self,
        config,
        weights: dict,
        device: ttnn.MeshDevice,
        rope: dict,
        expert_provider: Optional[Callable[[int], Callable[[int], tuple]]] = None,
        experts: Optional[Sequence[DeepSeekV4PreloadedExperts]] = None,
        num_layers: Optional[int] = None,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
        moe_weight_dtype: ttnn.DataType = ttnn.bfloat16,
        expert_dtype: Optional[ttnn.DataType] = None,
        head_dtype: ttnn.DataType = ttnn.bfloat16,
        tp_size: int = 1,
        layer_devices: Optional[Sequence[ttnn.MeshDevice]] = None,
        dense_csa: bool = False,
        lightning_indexer: bool = False,
        progress: Optional[Callable[..., None]] = None,
        embedding_weight: Optional[ttnn.Tensor] = None,
        lm_head: Optional[Linear] = None,
    ):
        """Upload the embedding, the layers, the head and ``lm_head`` to their devices.

        ``progress(message, important=False)`` is told what the model is about to do, at a fine grain
        (each build step, each layer of each chunk, each stage hand-off, each wait on the device). A caller
        keeps the last message to tell a slow step from a hung one, and logs the ``important`` ones (chunk
        boundaries) always and the rest as it sees fit.
        """
        self._progress = progress or (lambda message, important=False: None)
        self._tag = "build"
        note = self._note
        num_layers = config.num_hidden_layers if num_layers is None else num_layers
        if not 0 < num_layers <= config.num_hidden_layers:
            raise ValueError(f"num_layers {num_layers} is not in [1, {config.num_hidden_layers}]")
        if experts is None and expert_provider is None:
            raise ValueError("pass either expert_provider (to upload the routed experts) or ready-made experts")
        if experts is not None and len(experts) < num_layers:
            raise ValueError(f"got {len(experts)} experts objects for {num_layers} layers")
        if layer_devices is None:
            layer_devices = [device] * num_layers
        if len(layer_devices) != num_layers:
            raise ValueError(f"got {len(layer_devices)} layer devices for {num_layers} layers")
        self.config = config
        self.device = device
        self.layer_devices = list(layer_devices)
        self.head_device = self.layer_devices[-1]
        self.tp_size = tp_size
        self.hidden = config.hidden_size
        self.vocab_size = config.vocab_size
        self.hc = config.hc_mult
        self._traced: Optional[TracedPrefill] = None
        # ``(id(from_device), id(to_device)) -> (sender, receiver)``: D2D socket pairs the traced prefill reuses
        # rather than opening its own (set by :meth:`DeepSeekV4Model.build_prefill` to the decode model's).
        self.shared_socket_pairs: dict = {}
        cache = _as_cache(cache)

        if embedding_weight is not None:
            self.embedding_weight = embedding_weight
        else:
            note(f"embedding table [{config.vocab_size} x {config.hidden_size}] -> device", important=True)
            embed_file = cache.file("embed_tokens.prefill")
            embed = _materialize(weights["embed_tokens.weight"], embed_file, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
            # ``ttnn.embedding`` wants a ROW_MAJOR table.
            self.embedding_weight = _load_weight(
                embed.detach() if embed is not None else None,
                device,
                cache_file_name=embed_file,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        self.layers: list[DeepSeekV4PrefillDecoderLayer] = []
        for i in range(num_layers):
            layer_device = self.layer_devices[i]
            layer_cache = cache.sub(f"layers.{i}")
            layer_experts = experts[i] if experts is not None else None
            if layer_experts is None:
                note(f"layer {i + 1}/{num_layers} ({config.layer_types[i]}): routed experts", important=True)
                layer_experts = DeepSeekV4PreloadedExperts(
                    config,
                    self._noting_provider(expert_provider(i), i, num_layers, config.num_local_experts),
                    layer_device,
                    dtype=expert_dtype,
                    cache=layer_cache.sub("mlp"),
                    tp_size=tp_size,
                )
            self.layers.append(
                DeepSeekV4PrefillDecoderLayer(
                    config,
                    i,
                    _strip_prefix(weights, f"layers.{i}"),
                    layer_device,
                    rope,
                    experts=layer_experts,
                    cache=layer_cache,
                    weight_dtype=weight_dtype,
                    moe_weight_dtype=moe_weight_dtype,
                    tp_size=tp_size,
                    dense_csa=dense_csa,
                    lightning_indexer=lightning_indexer,
                )
            )
            note(f"layer {i + 1}/{num_layers}: built (attention, router, shared expert, hyper-connections)")

        note("final hyper head, norm and lm_head -> device", important=True)
        self.hc_head = DeepSeekV4PrefillHyperHead(
            config, _strip_prefix(weights, "hc_head"), self.head_device, cache=cache.sub("hc_head")
        )
        self.norm_weight = load_norm_gamma(weights["norm.weight"], self.head_device, cache.file("norm.prefill"))
        self.eps = config.rms_norm_eps
        self.lm_head = (
            lm_head
            if lm_head is not None
            else Linear(weights["lm_head.weight"], self.head_device, cache.file("lm_head.prefill"), dtype=head_dtype)
        )

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def devices(self) -> list[ttnn.MeshDevice]:
        """The distinct devices the model uses, in pipeline order."""
        seen: dict[int, ttnn.MeshDevice] = {}
        for dev in [self.device, *self.layer_devices, self.head_device]:
            seen.setdefault(id(dev), dev)
        return list(seen.values())

    def new_state(self) -> list[PrefillAttentionState]:
        """One empty attention state per layer: the start of a prompt."""
        return [layer.new_state() for layer in self.layers]

    def synchronize(self, why: str = "") -> None:
        """Block until every device the model uses has finished its queued work."""
        devices = self.devices
        for i, dev in enumerate(devices):
            self._note(f"{self._tag}: {why or 'synchronize'} - waiting for device {i + 1}/{len(devices)}")
            ttnn.synchronize_device(dev)

    # ------------------------------------------------------------------ progress
    def _note(self, message: str, important: bool = False) -> None:
        """Report what the model is doing (see the constructor's ``progress``)."""
        self._progress(message, important)

    def _noting_provider(self, provider: Callable[[int], tuple], layer: int, num_layers: int, num_experts: int):
        """``provider`` that reports every 32nd expert it is asked for (only a cache miss calls it at all)."""

        def noting(e: int):
            if e % 32 == 0:
                self._note(
                    f"layer {layer + 1}/{num_layers}: reading + quantizing routed expert {e}/{num_experts} "
                    "from the checkpoint (weight-cache miss)"
                )
            return provider(e)

        return noting

    # ------------------------------------------------------------------ pieces
    def _host_ids(self, input_ids) -> torch.Tensor:
        """``input_ids`` (``[T]`` / ``[1, T]`` ints) as a validated ``[1, T]`` int64 torch tensor."""
        ids = torch.as_tensor(input_ids)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        if ids.dim() != 2 or ids.shape[0] != 1:
            raise ValueError(f"expected token ids [T] or [1, T], got {tuple(ids.shape)}")
        if ids.numel() == 0 or ids.shape[1] % ALIGNMENT:
            raise ValueError(f"prompt length {ids.shape[1]} must be a positive multiple of {ALIGNMENT}")
        if int(ids.min()) < 0 or int(ids.max()) >= self.vocab_size:
            raise ValueError(f"token ids must lie in [0, {self.vocab_size})")
        return ids.long()

    def _upload_ids(self, ids: torch.Tensor, device: Optional[ttnn.MeshDevice] = None) -> ttnn.Tensor:
        """``[1, T]`` ids as the uint32 ROW_MAJOR tensor the embedding and hash routers read (on every rank)."""
        device = self.device if device is None else device
        return ttnn.from_torch(
            ids.to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if device.get_num_devices() > 1 else None,
        )

    @staticmethod
    def to_host(tensor: ttnn.Tensor, device: ttnn.MeshDevice) -> torch.Tensor:
        """One rank's copy of a replicated device tensor, as fp32 torch (the tensors here are replicated)."""
        if device.get_num_devices() > 1:
            tensor = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
            return tensor[: tensor.shape[0] // device.get_num_devices()].to(torch.float32)
        return ttnn.to_torch(tensor).to(torch.float32)

    def _handoff(self, streams: ttnn.Tensor, src: ttnn.MeshDevice, dst: ttnn.MeshDevice) -> ttnn.Tensor:
        """Move the residual streams ``[1, T, hc, D]`` to the next pipeline stage, through the host."""
        host = self.to_host(streams, src).to(torch.bfloat16)
        return ttnn.from_torch(
            host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dst,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(dst) if dst.get_num_devices() > 1 else None,
        )

    def embed(self, ids_dev: ttnn.Tensor) -> ttnn.Tensor:
        """``[1, T]`` uint32 ids -> the initial streams ``[1, T, hc, D]`` TILE: the embedding, once per stream."""
        t = ids_dev.shape[1]
        emb = ttnn.embedding(ids_dev, self.embedding_weight, layout=ttnn.ROW_MAJOR_LAYOUT)  # [1, T, D]
        # ROW_MAJOR, where adding the stream axis is a view and repeating along it a page copy.
        emb = ttnn.reshape(emb, [1, t, 1, self.hidden])
        streams = ttnn.repeat(emb, ttnn.Shape([1, 1, self.hc, 1]))
        return ttnn.to_layout(streams, ttnn.TILE_LAYOUT)

    def head(self, streams: ttnn.Tensor, last_only: bool = True, select: Optional[ttnn.Tensor] = None) -> ttnn.Tensor:
        """``streams`` ``[1, T, hc, D]`` -> logits ``[1, 1, 1, V]`` (the last token) or ``[1, 1, T, V]``.

        ``select`` (a one-hot bf16 TILE row ``[1, 1, 1, T]``) picks the token instead of the last one.
        """
        self._note(f"{self._tag}: head (hyper head, final norm, lm_head)")
        hidden = ttnn.rms_norm(self.hc_head(streams), weight=self.norm_weight, epsilon=self.eps)  # [1, 1, T, D]
        if select is not None:
            hidden = ttnn.matmul(select, hidden)
        elif last_only:
            t = hidden.shape[2]
            hidden = ttnn.slice(hidden, [0, 0, t - 1, 0], [1, 1, t, self.hidden])
        return self.lm_head(hidden)

    def _stack(
        self,
        ids: torch.Tensor,
        states: list[PrefillAttentionState],
        on_layer: Optional[Callable[[int, ttnn.Tensor, ttnn.MeshDevice], None]] = None,
    ) -> ttnn.Tensor:
        """The embedding and every layer over one chunk of ``ids`` ``[1, T]``; the final streams (on the head device).

        ``on_layer(index, streams, device)`` (verification) is called with each layer's output streams
        before they move on; it must not deallocate them.
        """
        if len(states) != len(self.layers):
            raise ValueError(f"expected {len(self.layers)} layer states, got {len(states)}")
        ids_on: dict[int, ttnn.Tensor] = {}

        def ids_for(dev: ttnn.MeshDevice) -> ttnn.Tensor:
            if id(dev) not in ids_on:
                ids_on[id(dev)] = self._upload_ids(ids, dev)
            return ids_on[id(dev)]

        self._note(f"{self._tag}: embedding")
        streams = self.embed(ids_for(self.device))
        current = self.device
        stage = 0
        for i, (layer, dev, state) in enumerate(zip(self.layers, self.layer_devices, states)):
            if dev is not current:
                stage += 1
                self._note(f"{self._tag}: stage hand-off to stage {stage} (streams -> host -> next submesh)")
                moved = self._handoff(streams, current, dev)
                ttnn.deallocate(streams)
                streams, current = moved, dev
                self._note(f"{self._tag}: stage hand-off done")
            kind = self.config.layer_types[i].replace("_attention", "")
            self._note(f"{self._tag}: layer {i + 1}/{len(self.layers)} ({kind}, stage {stage}) enqueue")
            out = layer(streams, state, ids_for(dev))
            if on_layer is not None:
                on_layer(i, out, dev)
            ttnn.deallocate(streams)
            streams = out
        for t in ids_on.values():
            ttnn.deallocate(t)
        return streams

    # ------------------------------------------------------------------ public
    def forward(
        self, input_ids, states: Optional[list[PrefillAttentionState]] = None, last_only: bool = True
    ) -> ttnn.Tensor:
        """The network over one chunk of a prompt.

        ``input_ids`` is ``[T]`` / ``[1, T]`` torch ints with ``T`` a multiple of ``ALIGNMENT``;
        ``states`` is this model's per-layer state (:meth:`new_state`), advanced in place -- ``None`` runs a
        whole prompt from scratch and discards it. Returns bf16 TILE logits ``[1, 1, 1, V]`` for the chunk's
        last token, or ``[1, 1, T, V]`` for every token with ``last_only=False``, on the head device
        (replicated across its ranks; read one with :meth:`to_host`).
        """
        ids = self._host_ids(input_ids)
        self._tag = "forward"
        streams = self._stack(ids, self.new_state() if states is None else states)
        logits = self.head(streams, last_only)
        ttnn.deallocate(streams)
        return logits

    def prefill(
        self,
        input_ids,
        chunk_size: int = 1024,
        on_chunk: Optional[Callable[[int, int, int, float], None]] = None,
        states: Optional[list[PrefillAttentionState]] = None,
    ) -> tuple[ttnn.Tensor, list[PrefillAttentionState]]:
        """A whole prompt as consecutive ``chunk_size`` chunks; returns ``(last-token logits, states)``.

        The logits are ``[1, 1, 1, V]`` for the prompt's last token, and ``states`` is what decode (or a
        further chunk) continues from. Only the last chunk pays for the head.

        ``on_chunk(index, start, end, seconds)`` is called after every chunk with its wall time. Giving it
        makes the model synchronize every device before and after each chunk, so ``seconds`` covers all of
        the chunk's device work (the last chunk includes the head); without it the chunks are enqueued back
        to back. ``states`` continues an earlier prefill instead of starting a new one.
        """
        if chunk_size <= 0 or chunk_size % ALIGNMENT:
            raise ValueError(f"chunk_size {chunk_size} must be a positive multiple of {ALIGNMENT}")
        ids = self._host_ids(input_ids)
        total = ids.shape[1]
        states = self.new_state() if states is None else states
        num_chunks = -(-total // chunk_size)
        for index, start in enumerate(range(0, total, chunk_size)):
            end = min(total, start + chunk_size)
            self._tag = f"chunk {index + 1}/{num_chunks}"
            if on_chunk is not None:
                self.synchronize("before the chunk")
            self._note(f"{self._tag} [{start}, {end}): start", important=True)
            t0 = time.perf_counter()
            streams = self._stack(ids[:, start:end], states)
            last = end == total
            logits = self.head(streams, last_only=True) if last else None
            ttnn.deallocate(streams)
            self._note(f"{self._tag}: all work enqueued in {time.perf_counter() - t0:.2f} s")
            if on_chunk is not None:
                self.synchronize("device compute")
                on_chunk(index, start, end, time.perf_counter() - t0)
        return logits, states

    # ------------------------------------------------------------------ traced prefill
    def prepare_traced_prefill(self, max_len: int, chunk_size: int = 1024) -> None:
        """Capture the chunk trace for prompts of up to ``max_len`` tokens (see :class:`TracedPrefill`).

        Allocates persistent state (its compressed-KV buffers sized for ``max_len``) and the H2D / D2D / D2H
        sockets, compiles the ``chunk_size``-token chunk step, then captures it: one trace per stage. Do this right
        after building the model and before any eager :meth:`prefill` / :meth:`forward` (allocating on a device
        that holds a trace is unsafe); :meth:`prefill_traced` then serves any such prompt without compiling or
        capturing again.

        One plan at a time: for a longer ``max_len`` or another ``chunk_size``, :meth:`release_traced_prefill`
        first, then prepare again. Preparing with the prepared ``chunk_size`` and a ``max_len`` it covers is a
        no-op.
        """
        if self._traced is not None and self._traced.prepared:
            if self._traced._chunk_size == chunk_size and max_len <= self._traced._max_len:
                return
            raise RuntimeError(
                f"a traced prefill plan for up to {self._traced._max_len} tokens in chunks of "
                f"{self._traced._chunk_size} is prepared; release_traced_prefill() before preparing another"
            )
        self._traced = TracedPrefill(self)
        self._traced.prepare(max_len, chunk_size)

    def compile_traced_prefill(self, max_len: int, chunk_size: int = 1024) -> None:
        """The first half of :meth:`prepare_traced_prefill`: persistent buffers, sockets and the compile run, no
        capture yet; :meth:`capture_traced_prefill` finishes it."""
        if self._traced is not None and (self._traced.prepared or self._traced.compiled):
            raise RuntimeError("a traced prefill plan is already compiled; release_traced_prefill() first")
        self._traced = TracedPrefill(self)
        self._traced.compile(max_len, chunk_size)

    def capture_traced_prefill(self) -> None:
        """The second half of :meth:`prepare_traced_prefill`: capture the plan :meth:`compile_traced_prefill`
        compiled, one trace per stage."""
        if self._traced is None:
            raise RuntimeError("call compile_traced_prefill(max_len, chunk_size) first")
        self._traced.capture()

    def prefill_traced(
        self,
        input_ids,
        on_chunk: Optional[Callable[[int, int, int, float], None]] = None,
    ) -> tuple[torch.Tensor, list[PrefillAttentionState]]:
        """:meth:`prefill` by replaying the traces of :meth:`prepare_traced_prefill`, the stages pipelined.

        ``input_ids`` is a multiple of ``ALIGNMENT`` tokens, at most the prepared ``max_len``. Returns
        ``(logits, states)``: the last-token logits as a host fp32 ``[1, 1, 1, V]`` tensor (they arrive over the D2H
        socket), and ``states`` read out of persistent buffers, so commit (or copy) them before calling again: the
        next call overwrites them. ``on_chunk`` gets the time between consecutive chunks' logits (see
        :meth:`TracedPrefill.run`).
        """
        if self._traced is None or not self._traced.prepared:
            raise RuntimeError("call prepare_traced_prefill(max_len, chunk_size) first")
        return self._traced.run(input_ids, on_chunk=on_chunk)

    def free_traced_states(self, states: list[PrefillAttentionState]) -> None:
        """Free what :meth:`prefill_traced` allocated for ``states`` (slices; the persistent buffers stay), once
        they are consumed and before the next :meth:`prefill_traced`."""
        self._traced.free_states(states)

    def release_traced_prefill(self) -> None:
        """Release every captured prefill trace and close its sockets, so the devices can allocate freely again
        (e.g. a decode model).

        The persistent buffers stay: the states :meth:`prefill_traced` returned remain valid.
        """
        if self._traced is not None:
            self._traced.release()


# --- Traced prefill ------------------------------------------------------------------------------ #
# Capture the chunk step once per pipeline stage and replay it for every chunk, with the stages running as a
# pipeline: with S stages, S chunks are in flight at once, one in each stage. This is the prefill twin of the
# traced decode (:meth:`DeepSeekV4Model.decode_traced`), and it moves its data the same way:
#
# * H2D packet socket. A chunk's only host input is one INT32 packet -- its token ids and positions (see
#   :meth:`TracedPrefill._layout_packet`) -- pushed into an H2D socket on stage 0 and received inside stage 0's
#   trace (``recv_async_h2d``), then broadcast over the stage's TP ranks.
# * D2D sockets. Every stage's trace ends by sending the residual streams (row-major, so no tile padding goes over
#   the wire) and the packet to the next stage, whose trace starts by receiving them into persistent buffers.
# * D2H output socket. The last stage's trace runs the head on the chunk's last token and streams those logits to
#   the host (``send_async_d2h``), which reads them off the socket instead of issuing a readback.
#
# Nothing crosses the host between stages and no host write sits between two replays, so ``execute_trace`` is
# posted ahead of time (from a replay thread, as decode does) while the host feeds packets and reads logits. A
# stage starts chunk c + 1 as soon as it has handed chunk c on; the sockets are the only synchronization.
#
# A trace is a flat, fixed op sequence over fixed buffers, and the eager prefill is neither: every chunk uploads
# its RoPE tables, mask and ids from the host, its compressed KV grows by a concat, and the SDPA key length grows
# with it. So the traced path makes each of those static:
#
# * State in place. Each layer owns persistent :class:`~.prefill.attention.PrefillStaticBuffers`: the K=V
#   ``tail``, the CSA overlap window and a *FIFO* of compressed entries anchored at its end (a chunk drops the
#   oldest rows of the window it reads and appends its own; SDPA does not care about key order, the mask names
#   the rows).
# * Per-chunk inputs made in the trace from the packet. The RoPE rows are gathered (``ttnn.embedding``) from
#   whole-prompt tables at the packet's positions, and a mask is a per-layer-type constant plus a column cut
#   that depends only on the chunk's start
#   (:meth:`~.prefill.attention.DeepSeekV4PrefillAttention.mask_tables_host`). Nothing is allocated between
#   captures: allocating on a device that holds a trace is unsafe.
# * One shape. Every chunk is ``C = chunk_size`` tokens, and every layer reads its *whole* FIFO: the SDPA key axis
#   is ``sliding_window + C + capacity``, the rows not yet filled masked out. So one trace per stage serves every
#   chunk of every prompt of up to ``max_len`` tokens, at the price of attending over the masked rows early on.
# * Padding. A prompt's last chunk is padded at its end (with repeats of its own tokens). Attention is causal, so
#   no real token sees the padding; what the padding does leave behind is the state after it, the head's last
#   token and the FIFOs' last rows, so the packet names the chunk's last real token (the head picks it), the last
#   chunk's ``[tail | chunk]`` window and Ca overlap rows are kept (``PrefillStaticBuffers``), and
#   :meth:`TracedPrefill._export_states` slices the real state out of them and drops the padding's entries.
#
# One trace covers all the layers of one stage (the embedding on the first, the head on the last).
# :meth:`TracedPrefill.prepare` compiles it eagerly, *then* captures it (a compile run allocates), once per
# ``(max_len, chunk_size)``; :meth:`TracedPrefill.run` replays it for any prompt of up to ``max_len`` tokens (a
# multiple of ``ALIGNMENT``). The result is interchangeable with :meth:`DeepSeekV4PrefillModel.prefill` (same chunk
# boundaries), except that the logits arrive on the host; the states are read out of the persistent buffers, so
# commit (or copy) them (:meth:`DeepSeekV4Model.commit_prefill_state`) before the next run.

# Packets the prefill H2D FIFO holds. One, not several: the FIFO sits in L1 on a worker core, and
# ``indexer_score_dsa``'s circular buffers already reach within ~30 KB of the top of that core's L1, so a
# second packet (a 1024-token chunk's packet is ~9 KB) overlaps them. The host's write blocks until stage 0
# has taken the packet, which the pipeline accounts for (see :meth:`TracedPrefill.run`).
_PREFILL_PKT_FIFO_PACKETS = 1


def _round_up(n: int, multiple: int) -> int:
    return -(-n // multiple) * multiple


@dataclass
class _StageIO:
    """Persistent per-stage device tensors: the D2D receive buffer, the mask constants and the head's row ramp."""

    streams_in: Optional[ttnn.Tensor]  # [1, C, hc, D] bf16 ROW_MAJOR (stages after the first)
    masks: dict  # layer_type -> (static [1, 1, C, K] bf16, threshold [1, 1, 1, K] fp32), K = sw + C + capacity
    index_pads: dict = field(default_factory=dict)  # layer_type -> [1, 1, kv_len, index_head_dim] zero key pad
    ramp: Optional[ttnn.Tensor] = None  # [1, 1, 1, C] fp32 TILE 0 .. C - 1 (last stage): the head's one-hot row


@dataclass
class _Stage:
    """The contiguous run of layers that share one device (one pipeline stage), its buffers, sockets and traces."""

    index: int
    device: ttnn.MeshDevice
    layers: list[int]
    first: bool
    last: bool = False
    types: set = field(default_factory=set)
    pkt: Optional[ttnn.Tensor] = None  # [1, 1, 1, W] INT32 ROW_MAJOR chunk packet (H2D on stage 0, D2D after)
    rope: dict = field(default_factory=dict)  # kind -> (cos, sin) [max_len rounded up to C, rope_dim] bf16 ROW_MAJOR
    io: Optional[_StageIO] = None
    recv: object = None  # receiver socket from the previous stage
    send: object = None  # sender socket to the next stage
    trace: Optional[int] = None


class TracedPrefill:
    """Traced, pipelined execution of a :class:`DeepSeekV4PrefillModel` (see the section comment above)."""

    def __init__(self, model: DeepSeekV4PrefillModel):
        self.model = model
        self.config = model.config
        self.prepared = False
        self.compiled = False
        self.stages: list[_Stage] = []
        self.buffers: dict = {}
        self._entry_capacity: dict = {}  # layer type -> FIFO rows, every one of which each chunk reads
        self._max_len = 0
        self._prompt_len = 0  # the last run's prompt length
        self._chunk_size = 0
        self._pkt_socket = None
        self._out_socket = None
        self._pkt_entry: dict = {}  # compress rate -> packet offset of the chunk's entry positions
        self._pkt_last = 0  # packet offset of the chunk's last real token (its index in the chunk)
        self._pkt_w = 0
        self._pkt_page_bytes = 0
        self._out_plan: Optional[tuple[int, int]] = None  # (rows, cols) of one chunk's logits on the D2H socket

    # ------------------------------------------------------------------ planning
    @staticmethod
    def _plan_chunks(prompt_len: int, chunk_size: int) -> list[tuple[int, int]]:
        """``[(start, real tokens)]``: full chunks, then the remainder (padded to ``chunk_size`` when it runs)."""
        return [(s, min(chunk_size, prompt_len - s)) for s in range(0, prompt_len, chunk_size)]

    def _rates(self, stage: _Stage) -> list[int]:
        """The compress rates of the stage's CSA / HCA layers."""
        return sorted({self.config.compress_rates[lt] for lt in stage.types if lt != SLIDING_ATTENTION})

    def _group_stages(self) -> list[_Stage]:
        stages: list[_Stage] = []
        for li, dev in enumerate(self.model.layer_devices):
            if not stages or stages[-1].device is not dev:
                stages.append(_Stage(index=len(stages), device=dev, layers=[], first=not stages))
            stages[-1].layers.append(li)
            stages[-1].types.add(self.config.layer_types[li])
        stages[-1].last = True
        return stages

    def _layout_packet(self, chunk_size: int) -> None:
        """Fix the INT32 slots of the chunk packet for chunks of ``C = chunk_size`` tokens.

        ``[0, C)`` the token ids, ``[C, 2C)`` the token positions ``start + i`` (slot ``C`` doubles as the chunk's
        start), then per compress rate ``r`` the ``C / r`` positions ``start + w * r`` of the chunk's compressed
        entries, then one slot: the index in the chunk of its last real token (the head's token). The row is one
        H2D socket page, so its width is rounded up to the PCIe alignment.
        """
        offset = 2 * chunk_size
        self._pkt_entry = {}
        for rate in sorted({self.config.compress_rates[lt] for lt in self._entry_capacity}):
            self._pkt_entry[rate] = offset
            offset += chunk_size // rate
        self._pkt_last = offset
        offset += 1
        alignment = active_system_config().pipeline.pcie_alignment
        self._pkt_page_bytes = math.ceil(offset * 4 / alignment) * alignment
        self._pkt_w = self._pkt_page_bytes // 4
        # The widest row of at most 4 KB that tiles the packet, for the broadcast over the TP ranks.
        self._pkt_bcast_width = max(d for d in range(1, min(self._pkt_w, 1024) + 1) if self._pkt_w % d == 0)

    def _packet(self, chunk_ids: torch.Tensor, start: int) -> torch.Tensor:
        """The host packet ``[1, 1, 1, W]`` INT32 of the chunk ``chunk_ids`` (``t <= C`` ints) starting at
        ``start``, padded to ``C`` tokens with repeats of its own."""
        t, c = chunk_ids.numel(), self._chunk_size
        ids = chunk_ids.reshape(-1).to(torch.int32)
        packet = torch.zeros(1, 1, 1, self._pkt_w, dtype=torch.int32)
        row = packet[0, 0, 0]
        row[:c] = ids.repeat(-(-c // t))[:c]
        row[c : 2 * c] = start + torch.arange(c, dtype=torch.int32)
        for rate, offset in self._pkt_entry.items():
            row[offset : offset + c // rate] = start + rate * torch.arange(c // rate, dtype=torch.int32)
        row[self._pkt_last] = t - 1
        return packet

    # ------------------------------------------------------------------ allocation (before any trace)
    def _representative(self, stage: _Stage, layer_type: str):
        """One attention block of ``layer_type`` on ``stage`` (the source of its host / device helpers)."""
        for li in stage.layers:
            if self.config.layer_types[li] == layer_type:
                return self.model.layers[li].self_attn
        raise KeyError(layer_type)

    def _allocate(self, max_len: int, chunk_size: int) -> None:
        model, config = self.model, self.config
        present = {config.layer_types[i] for i in range(model.num_layers)}
        padded = _round_up(max_len, chunk_size)  # the last chunk's padding emits entries too
        self._entry_capacity = {
            lt: _round_up(max(padded // config.compress_rates[lt], 1), ALIGNMENT)
            for lt in (COMPRESSED_SPARSE_ATTENTION, HEAVILY_COMPRESSED_ATTENTION)
            if lt in present
        }
        self.stages = self._group_stages()
        self._layout_packet(chunk_size)

        for li, layer in enumerate(model.layers):
            lt = config.layer_types[li]
            self.buffers[li] = layer.self_attn.new_static_buffers(chunk_size, self._entry_capacity.get(lt, 0))

        for stage in self.stages:
            dev = stage.device
            stage.pkt = ttnn.from_torch(
                torch.zeros(1, 1, 1, self._pkt_w, dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(dev) if dev.get_num_devices() > 1 else None,
            )
            for lt in stage.types:
                attn = self._representative(stage, lt)
                if attn.rope_kind not in stage.rope:
                    stage.rope[attn.rope_kind] = self._rope_tables(attn, max_len, padded)
            stage.io = self._allocate_io(stage, chunk_size)
        self._open_sockets()

    @staticmethod
    def _rope_tables(attn, needed: int, length: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """``(cos, sin)`` ``[length, rope_dim]`` bf16 ROW_MAJOR on ``attn``'s device, the rows every chunk gathers.

        The same values as the eager path's per-chunk tables (:meth:`~.prefill.attention.DeepSeekV4PrefillAttention._rope_tables`)
        for the ``needed`` positions a prompt can reach; rows past the model's table are zero (only padding reads them).
        """
        cos_half, sin_half = attn.rope[attn.rope_kind]
        if cos_half.shape[0] < needed:
            raise ValueError(f"rope table covers {cos_half.shape[0]} positions but the prompt needs {needed}")

        def table(half: torch.Tensor) -> ttnn.Tensor:
            half = half[:length].float()
            half = torch.cat([half, half.new_zeros(length - half.shape[0], half.shape[1])])
            return ttnn.from_torch(
                half.repeat_interleave(2, dim=-1),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=attn.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=attn._replicate,
            )

        return table(cos_half), table(sin_half)

    def _allocate_io(self, stage: _Stage, t: int) -> _StageIO:
        model = self.model
        dev = stage.device
        streams_in = None
        if not stage.first:
            streams_in = ttnn.from_torch(
                torch.zeros(1, t, model.hc, model.hidden, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(dev) if dev.get_num_devices() > 1 else None,
            )
        masks, index_pads = {}, {}
        for lt in stage.types:
            attn = self._representative(stage, lt)
            cap = self._entry_capacity.get(lt, 0)
            static, threshold = attn.mask_tables_host(t, cap)
            masks[lt] = (
                attn._to_device(static.reshape(1, 1, t, -1)),
                attn._to_device(threshold.reshape(1, 1, 1, -1), dtype=ttnn.float32),
            )
            if lt == COMPRESSED_SPARSE_ATTENTION and attn.use_indexer:
                # chunk_start_idx (= cap) must sit strictly inside the key pad; pad past the chunk too.
                index_pads[lt] = attn._zeros_rm(_round_up(cap + t, 64), attn.index_head_dim)
        ramp = None
        if stage.last:
            ramp = ttnn.from_torch(
                torch.arange(t, dtype=torch.float32).reshape(1, 1, 1, t),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(dev) if dev.get_num_devices() > 1 else None,
            )
        return _StageIO(streams_in, masks, index_pads, ramp)

    def _open_sockets(self) -> None:
        """The H2D packet socket on stage 0, the D2D pairs between consecutive stages and the D2H logits socket on
        the last stage. Opened before any trace exists, because each allocates L1 on its cores."""
        pipeline = active_system_config().pipeline
        first, last = self.stages[0], self.stages[-1]
        self._pkt_socket = ttnn.H2DSocket(
            first.device,
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(*_PKT_SOCKET_CORE)),
            ttnn.BufferType.L1,
            _PREFILL_PKT_FIFO_PACKETS * self._pkt_page_bytes,
            ttnn.H2DMode.HOST_PUSH,
        )
        # ``recv_async_h2d`` cross-checks this against the packet's aligned page size on every program-cache miss.
        self._pkt_socket.set_page_size(self._pkt_page_bytes)
        for upstream, downstream in zip(self.stages, self.stages[1:]):
            pair = self.model.shared_socket_pairs.get((id(upstream.device), id(downstream.device)))
            if pair is None:
                pair = _create_socket_pair(upstream.device, downstream.device, pipeline.socket_l1_bytes)
            upstream.send, downstream.recv = pair
        self._out_socket = ttnn.D2HSocket(
            last.device,
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(*_OUT_SOCKET_CORE)),
            pipeline.d2h_fifo_bytes,
        )
        # bf16 logits (see _send_logits); one row is one socket page, which has to fit the FIFO.
        self._out_plan = _d2h_page_plan(
            self.model.vocab_size,
            2,
            page_cap_bytes=pipeline.d2h_fifo_bytes,
            pcie_alignment=pipeline.pcie_alignment,
        )
        self._out_socket.set_page_size(self._out_plan[1] * 2)

    # ------------------------------------------------------------------ the traced body
    def _step_inputs(self, stage: _Stage, pkt: ttnn.Tensor) -> tuple:
        """``(ids, step, made)``: the chunk's ``[1, C]`` uint32 ids and :class:`PrefillStaticStep`, built on device
        from the packet ``pkt``, and every tensor made here (for the caller to free once the layers have run)."""
        t = c = self._chunk_size
        io = stage.io
        caps = {lt: self._entry_capacity.get(lt, 0) for lt in stage.types}
        sw = self.config.sliding_window
        made: list = []

        def keep(tensor: ttnn.Tensor) -> ttnn.Tensor:
            made.append(tensor)
            return tensor

        def slots(offset: int, n: int) -> ttnn.Tensor:
            """``n`` packet slots from ``offset``, as a ``[1, n]`` uint32 ROW_MAJOR tensor."""
            row = ttnn.slice(pkt, [0, 0, 0, offset], [1, 1, 1, offset + n])
            return keep(ttnn.typecast(ttnn.reshape(row, [1, n]), ttnn.uint32))

        def gather(tables: tuple, positions: ttnn.Tensor, n: int) -> tuple:
            """The RoPE rows at ``positions``: ``(cos, sin)`` ``[1, 1, n, rope_dim]`` bf16 TILE."""
            out = []
            for table in tables:
                rows = ttnn.embedding(positions, table, layout=ttnn.ROW_MAJOR_LAYOUT)  # [1, n, rope_dim]
                out.append(keep(ttnn.to_layout(ttnn.reshape(rows, [1, 1, n, rows.shape[-1]]), ttnn.TILE_LAYOUT)))
                ttnn.deallocate(rows)
            return tuple(out)

        ids = slots(0, t)
        positions = slots(c, t)
        rope = {kind: gather(tables, positions, t) for kind, tables in stage.rope.items()}
        entry_rope = {}
        for rate in self._rates(stage):
            n = t // rate
            entry_rope[rate] = gather(stage.rope["compress"], slots(self._pkt_entry[rate], n), n)

        start = ttnn.reshape(ttnn.slice(pkt, [0, 0, 0, c], [1, 1, 1, c + 1]), [1, 1, 1, 1])
        start_f = keep(ttnn.typecast(ttnn.to_layout(start, ttnn.TILE_LAYOUT), ttnn.float32))
        masks, index_cuts, index_pads = {}, {}, {}
        for lt in stage.types:
            cap = caps[lt]
            static, threshold = io.masks[lt]
            cut = ttnn.typecast(ttnn.multiply(ttnn.gt(threshold, start_f), _MASK_NEG), ttnn.bfloat16)
            masks[lt] = keep(ttnn.add(static, cut))  # [1, 1, T, K] + the [1, 1, 1, K] start-dependent cut
            ttnn.deallocate(cut)
            if lt == COMPRESSED_SPARSE_ATTENTION and self._representative(stage, lt).use_indexer:
                # The indexer's causal cut over the FIFO rows is the mask's entry columns.
                index_cuts[lt] = keep(ttnn.slice(masks[lt], [0, 0, 0, sw + t], [1, 1, t, sw + t + cap]))
                index_pads[lt] = io.index_pads[lt]
        step = PrefillStaticStep(
            rope=rope,
            entry_rope=entry_rope,
            masks=masks,
            caps=caps,
            index_cuts=index_cuts,
            index_pads=index_pads,
        )
        return ids, step, made

    def _head_select(self, stage: _Stage, pkt: ttnn.Tensor) -> ttnn.Tensor:
        """The one-hot bf16 TILE row ``[1, 1, 1, C]`` of the chunk's last real token (its packet slot)."""
        last = ttnn.reshape(ttnn.slice(pkt, [0, 0, 0, self._pkt_last], [1, 1, 1, self._pkt_last + 1]), [1, 1, 1, 1])
        last_f = ttnn.typecast(ttnn.to_layout(last, ttnn.TILE_LAYOUT), ttnn.float32)
        hit = ttnn.eq(stage.io.ramp, last_f)
        ttnn.deallocate(last_f)
        select = ttnn.typecast(hit, ttnn.bfloat16)
        ttnn.deallocate(hit)
        return select

    def _stage_forward(self, stage: _Stage) -> None:
        """One stage over one chunk: receive, embed (first stage), the stage's layers, send on.

        Stage 0 receives the packet from the host (H2D) and broadcasts it over its TP ranks; a later stage receives
        the streams and the packet from the stage before it (D2D, in the order they are sent). The last stage runs
        the head on the chunk's last real token and sends the logits to the host (D2H); every other stage sends the
        streams and the packet to the next one. Shared by the compile run and the trace capture.
        """
        model = self.model
        io = stage.io
        pkt = stage.pkt
        if stage.first:
            # Parks on the socket until the host pushes this chunk's packet (it may well have already).
            ttnn.experimental.recv_async_h2d(stage.pkt, self._pkt_socket)
            if stage.device.get_num_devices() > 1:
                # ``broadcast`` corrupts a row-major page larger than one fabric packet (~4.3 KB): send rows of <= 4 KB.
                width = self._pkt_bcast_width
                rows = ttnn.reshape(stage.pkt, [1, 1, self._pkt_w // width, width])
                sent = ttnn.broadcast(rows, ttnn.MeshCoordinate(0, 0), cluster_axis=1, topology=ttnn.Topology.Ring)
                pkt = ttnn.reshape(sent, [1, 1, 1, self._pkt_w])
        else:
            ttnn.experimental.recv_direct_async(io.streams_in, stage.recv)
            ttnn.experimental.recv_direct_async(stage.pkt, stage.recv)
        ids, step, made = self._step_inputs(stage, pkt)
        streams = model.embed(ids) if stage.first else ttnn.to_layout(io.streams_in, ttnn.TILE_LAYOUT)
        for li in stage.layers:
            out = model.layers[li].forward_static(streams, self.buffers[li], step, ids)
            ttnn.deallocate(streams)
            streams = out
        if stage.last:
            select = self._head_select(stage, pkt)
            self._send_logits(model.head(streams, select=select))
            ttnn.deallocate(select)
        else:
            rows = ttnn.to_layout(streams, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.experimental.send_direct_async(rows, stage.send)
            ttnn.experimental.send_direct_async(pkt, stage.send)
            ttnn.deallocate(rows)
        ttnn.deallocate(streams)
        for tensor in made:
            ttnn.deallocate(tensor)
        if pkt is not stage.pkt:
            ttnn.deallocate(pkt)

    def _send_logits(self, logits: ttnn.Tensor) -> None:
        """Stream a chunk's last-token logits ``[1, 1, 1, V]`` to the host over the D2H socket (inside the trace).

        ``send_async_d2h`` sends whole row-major pages and a row is one page, which has to fit the socket FIFO, so
        the logits go as ``(rows, cols)`` (:func:`_d2h_page_plan`) rather than as one vocab-wide row.
        """
        rows, cols = self._out_plan
        if logits.dtype != ttnn.bfloat16:
            logits = ttnn.typecast(logits, ttnn.bfloat16)
        paged = ttnn.to_layout(ttnn.reshape(logits, [1, 1, rows, cols]), ttnn.ROW_MAJOR_LAYOUT)
        ttnn.experimental.send_async_d2h(ttnn.reshape(paged, [rows, cols]), self._out_socket)
        ttnn.deallocate(paged)
        ttnn.deallocate(logits)

    def _read_logits(self) -> torch.Tensor:
        """The oldest unread chunk's logits off the D2H socket, fp32 ``[1, 1, 1, V]`` (blocks until they arrive)."""
        rows, cols = self._out_plan
        out = torch.empty(rows, cols, dtype=torch.bfloat16)
        self._out_socket.read_tensor(out)
        return out.reshape(1, 1, 1, -1).float()

    # ------------------------------------------------------------------ prepare: compile, then capture
    def prepare(self, max_len: int, chunk_size: int = 1024) -> None:
        """Allocate the persistent buffers and sockets, compile the chunk step eagerly, then capture it: one trace
        per stage.

        The traces then serve every prompt of up to ``max_len`` tokens (see the section comment). Must run before
        the model does anything else that allocates on its devices per prompt (eager prefills included).
        ``max_len`` and ``chunk_size`` are multiples of ``ALIGNMENT``; ``max_len`` sizes the FIFOs every chunk reads.
        """
        self.compile(max_len, chunk_size)
        self.capture()

    def compile(self, max_len: int, chunk_size: int = 1024) -> None:
        """The first half of :meth:`prepare`: the persistent buffers, the sockets and the executed compile run.

        Everything prefill keeps on its devices is allocated here, so :meth:`capture` may follow later (e.g. after
        another model's capture), as long as nothing allocated in between overlaps what the compile run used.
        """
        model = self.model
        for name, value in (("max_len", max_len), ("chunk_size", chunk_size)):
            if value <= 0 or value % ALIGNMENT:
                raise ValueError(f"{name}={value} must be a positive multiple of {ALIGNMENT}")
        if self.prepared or self.compiled:
            raise RuntimeError("traced prefill is already prepared; another plan needs another TracedPrefill")
        self._max_len, self._chunk_size = max_len, chunk_size
        model._tag = "traced prepare"
        model._note("traced prefill: allocating persistent buffers and sockets", important=True)
        self._allocate(max_len, chunk_size)
        self._stop_prefetcher()
        logger.info(
            f"[traced-prefill] prompts of up to {max_len} tokens in chunks of {chunk_size}: one trace x "
            f"{len(self.stages)} stage(s); entry buffers {self._entry_capacity}; packet {self._pkt_page_bytes} B"
        )

        # Pass 1: the compile run, while no trace exists (a compile run allocates freely). It executes, so it takes
        # a packet and sends logits, drained here. Every stage is issued before the read, since a stage's send parks
        # until the next stage posts its receive.
        self._pkt_socket.write_tensor(self._packet(torch.zeros(chunk_size, dtype=torch.long), 0))
        for stage in self.stages:
            model._note(f"traced prefill: compiling stage {stage.index}", important=True)
            self._stage_forward(stage)
        self._read_logits()
        for stage in self.stages:
            ttnn.synchronize_device(stage.device)
        self.compiled = True

    def capture(self) -> None:
        """The second half of :meth:`prepare`: capture the compiled chunk step, one trace per stage."""
        if not self.compiled:
            raise RuntimeError("compile() the traced prefill before capturing it")
        if self.prepared:
            raise RuntimeError("traced prefill is already captured")
        model = self.model
        model._tag = "traced prepare"
        self._stop_prefetcher()
        for stage in self.stages:
            model._note(f"traced prefill: capturing stage {stage.index}", important=True)
            tid = ttnn.begin_trace_capture(stage.device, cq_id=0)
            with _trace_capture_guard():
                self._stage_forward(stage)
            ttnn.end_trace_capture(stage.device, tid, cq_id=0)
            stage.trace = tid
        self.prepared = True
        model._note("traced prefill: ready", important=True)

    # ------------------------------------------------------------------ run
    def _reset(self) -> None:
        for li, layer in enumerate(self.model.layers):
            layer.self_attn.reset_static(self.buffers[li])

    def _stop_prefetcher(self) -> None:
        """Retire any DRISC tensor prefetcher on the prefill devices before a prefill trace is captured or replayed.

        The device is synchronized first, so every queued ``matmul_decode`` has consumed its prefetch request and the
        clean stop's sentinel is the only thing left in the senders' FIFO (a no-op if no prefetcher runs there).
        A decode model must not leave a request queued for a matmul it never ran (``LinearDecode.prefetch_queued``):
        the sentinel would queue behind it and the stop would never return.
        """
        for device in {id(d): d for d in [*(s.device for s in self.stages), self.model.head_device]}.values():
            ttnn.synchronize_device(device)
            ttnn.experimental.stop_tensor_prefetcher(device)
            ttnn.synchronize_device(device)

    def release(self) -> None:
        """Release the stage traces and close the sockets; the model cannot replay until :meth:`prepare` runs again."""
        for stage in self.stages:
            ttnn.synchronize_device(stage.device)
            if stage.trace is not None:
                ttnn.release_trace(stage.device, stage.trace)
            stage.trace = None
            stage.send = stage.recv = None
        self._pkt_socket = self._out_socket = None
        self.prepared = self.compiled = False

    def _replay(self, chunks: queue.Queue) -> None:
        """Replay-thread body: post every stage's trace for each queued chunk index, until ``None`` arrives."""
        try:
            for _ in iter(chunks.get, None):
                for stage in self.stages:
                    ttnn.execute_trace(stage.device, stage.trace, cq_id=0, blocking=False)
        except Exception:
            logger.exception("[traced-prefill] replay thread failed; the host will wait on the sockets forever")
            raise

    def run(
        self,
        input_ids,
        on_chunk: Optional[Callable[[int, int, int, float], None]] = None,
    ) -> tuple[torch.Tensor, list[PrefillAttentionState]]:
        """Replay the captured traces over ``input_ids`` (up to the prepared ``max_len``); returns ``(logits, states)``.

        The stages run as a pipeline, one chunk in each: a replay thread posts every chunk's ``execute_trace`` on
        every stage ahead of time, while this thread pushes each chunk's packet into the H2D socket and reads each
        chunk's logits off the D2H socket, with at most one chunk per stage fed but not yet read (the H2D FIFO holds a
        single packet, so the host's next write waits until stage 0 has taken the previous one). ``logits`` is the
        prompt's last-token logits, fp32 ``[1, 1, 1, V]`` on the host;
        ``states`` are read out of the persistent buffers.

        ``on_chunk(index, start, end, seconds)`` is called as each chunk's logits arrive, in chunk order, with the time
        since the previous chunk's arrived (the first chunk: since the run started, so it includes filling the
        pipeline). With the pipeline full that is the time per chunk, and the ``seconds`` add up to the whole prefill.
        """
        if not self.prepared:
            raise RuntimeError("call prepare(max_len, chunk_size) first")
        model = self.model
        ids = model._host_ids(input_ids)
        n = ids.shape[1]
        if n <= 0 or n % ALIGNMENT or n > self._max_len:
            raise ValueError(
                f"traces were captured for prompts of up to {self._max_len} tokens in multiples of {ALIGNMENT}, got {n}"
            )
        self._stop_prefetcher()
        self._reset()
        self._prompt_len = n
        plan = self._plan_chunks(n, self._chunk_size)
        num_chunks = len(plan)
        # One chunk per stage, plus as many as the H2D FIFO can hold. One more and the host's next write blocks
        # on stage 0 while the last stage blocks on the host reading its logits.
        in_flight = len(self.stages) + _PREFILL_PKT_FIFO_PACKETS - 1
        ahead = in_flight + len(self.stages)  # posted replays run ahead of the packets by this many chunks
        model._tag = "traced prefill"
        model._note(
            f"traced prefill: {num_chunks} chunk(s) through {len(self.stages)} stage(s), up to {in_flight} in flight",
            important=True,
        )

        replay: queue.Queue = queue.Queue()
        thread = threading.Thread(target=self._replay, args=(replay,), name="prefill-replay", daemon=True)
        thread.start()
        posted = fed = read = 0  # chunks whose traces are posted / whose packet is written / whose logits are read
        logits = None
        last_arrival = time.perf_counter()

        def feed(index: int, chunk_ids: torch.Tensor) -> None:
            nonlocal fed
            self._pkt_socket.write_tensor(self._packet(chunk_ids, plan[index][0]))
            fed += 1

        def take() -> None:
            nonlocal read, logits, last_arrival
            logits = self._read_logits()
            # Counted before the callback: the unwind below must never ask the socket for logits already read.
            read += 1
            now = time.perf_counter()
            start, t = plan[read - 1]
            model._note(f"traced chunk {read}/{num_chunks} [{start}, {start + t}): logits received")
            if on_chunk is not None:
                on_chunk(read - 1, start, start + t, now - last_arrival)
            last_arrival = now

        try:
            for index, (start, t) in enumerate(plan):
                while posted < min(num_chunks, index + ahead):
                    replay.put(posted)
                    posted += 1
                feed(index, ids[0, start : start + t])
                while fed - read > in_flight:
                    take()
            while read < fed:
                take()
        finally:
            try:
                # An error part-way leaves posted replays parked on their in-trace receives: feed each a dummy packet
                # and drain its logits, so the sockets and the replay thread can unwind.
                while read < posted:
                    if read >= fed:
                        feed(read, torch.zeros(plan[read][1], dtype=torch.long))
                    self._read_logits()
                    read += 1
            finally:
                replay.put(None)
                thread.join()
        model.synchronize("traced prefill done")
        return logits, self._export_states()

    # ------------------------------------------------------------------ hand-off to decode
    def _export_states(self) -> list[PrefillAttentionState]:
        """Per-layer :class:`PrefillAttentionState` of the prompt just run (for the decode commit).

        ``compressed_kv`` / ``idx_keys`` are slices of the FIFOs' rows holding the prompt's ``prompt_len // rate``
        entries (the padding's entries come after them). When the last chunk was padded, ``kv_tail`` and the CSA
        overlaps are sliced out of that chunk's kept window and Ca rows at its last real token; otherwise they are the
        persistent tensors themselves. Consume the states before the next run, then :meth:`free_states`.
        """
        n, c = self._prompt_len, self._chunk_size
        pad = _round_up(n, c) - n
        real = c - pad  # the last chunk's real tokens

        def rows(t: ttnn.Tensor, lo: int, hi: int) -> ttnn.Tensor:
            return ttnn.slice(t, [0, 0, lo, 0], [1, 1, hi, t.shape[3]])

        def overlap(persistent: ttnn.Tensor, kept: ttnn.Tensor, rate: int) -> ttnn.Tensor:
            return rows(kept, real // rate - 1, real // rate) if pad else persistent

        states = []
        for li, layer in enumerate(self.model.layers):
            attn = layer.self_attn
            bufs = self.buffers[li]
            tail = rows(bufs.window, real, real + attn.sliding_window) if pad else bufs.tail
            state = PrefillAttentionState(seq_len=n, kv_tail=tail)
            if not attn.is_sliding:
                rate = attn.rate
                emitted, padding = (n + pad) // rate, pad // rate
                total = bufs.entries.shape[2]
                state.compressed_kv = rows(bufs.entries, total - emitted, total - padding)
                if attn.is_csa:
                    state.csa_prev_kv = overlap(bufs.prev_kv, bufs.ca_kv, rate)
                    state.csa_prev_gate = overlap(bufs.prev_gate, bufs.ca_gate, rate)
                if bufs.idx_keys is not None:
                    total = bufs.idx_keys.shape[2]
                    state.idx_keys = rows(bufs.idx_keys, total - emitted, total - padding)
                    state.idx_prev_kv = overlap(bufs.idx_prev_kv, bufs.idx_ca_kv, rate)
                    state.idx_prev_gate = overlap(bufs.idx_prev_gate, bufs.idx_ca_gate, rate)
            states.append(state)
        return states

    def free_states(self, states: list[PrefillAttentionState]) -> None:
        """Deallocate the tensors :meth:`_export_states` sliced out for ``states`` (not the persistent buffers).

        They were allocated while the traces exist, so they must be gone before the next replay.
        """
        for li, state in enumerate(states):
            # By buffer, not by object: a slice spanning a whole FIFO (the longest prepared prompt, chunk-aligned) is
            # a no-op that returns a new tensor object on the persistent buffer.
            persistent = {t.buffer_address() for t in vars(self.buffers[li]).values() if isinstance(t, ttnn.Tensor)}
            for name, tensor in vars(state).items():
                if isinstance(tensor, ttnn.Tensor) and tensor.buffer_address() not in persistent:
                    ttnn.deallocate(tensor)
                    setattr(state, name, None)
