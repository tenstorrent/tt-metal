import math
import os
from typing import Optional

import torch
import ttnn
from loguru import logger

from .attention import (
    _StaticLayerCache,
    build_static_layer_cache,
    host_decode_mask,
    int32_pos_tensor,
    make_rope_table,
    sdpa_causal_cur_pos,
    sdpa_causal_ok,
)
from .paged_cache import (
    PagedCacheFull,
    PagedGroup,
    PagedKVManager,
    PagedLayerView,
    build_groups,
    plan_pool_blocks,
)
from .common import DeepSeekV4Module, _MASK_NEG, _profile, _region, _trace_capture_guard
from .decoder_layer import DeepSeekV4DecoderLayer
from .embedding import DeepSeekV4Embedding
from .hyperconnection import DeepSeekV4HyperHead
from .layers import DeepSeekV4RMSNorm
from .moe import DeepSeekV4HashRouter, DeepSeekV4PreloadedExperts
from .quant import dequantize_weight
from .weight_cache import WeightCache, _as_cache
from .weight_loader import DeepseekV4WeightLoader

# ---------------------------------------------------------------------------- #
# DeepSeek-V4-Flash full model (prefill, ``past_key_values is None``)
#
# ttnn port of ``DeepseekV4Model`` from ``modular_deepseek_v4.py``. Wires the
# embedding, the stack of :class:`DeepSeekV4DecoderLayer`s, the final
# :class:`DeepSeekV4HyperHead` stream-collapse and the model's shared RMSNorm
# into one module driven straight off the safetensors checkpoint (via
# :class:`DeepseekV4WeightLoader` + the ``quant`` dequantizers).
#
# Differences from the reference, all forced by the prefill-only / on-device
# scope already established by the sub-modules in this file:
#   * The rotary tables are *inputs* (built host-side, e.g. by the YaRN rotary
#     in the system interpreter — see ``test_bf4_decode_demo.py``) rather than
#     produced by an owned ``DeepseekV4RotaryEmbedding``; ttnn has no rope-init.
#   * The additive sliding-window / compressed-window masks are built here on
#     host (mirroring ``create_sliding_window_causal_mask`` + the compressors'
#     ``block_bias``), since the device attention consumes a plain additive mask.
#   * Every layer's weights are resident at once (the reference also holds the
#     whole stack); on the real 43-layer checkpoint cap with ``max_layers`` /
#     a populated ``cache`` or run the per-layer load/free loop in the demo.
# ---------------------------------------------------------------------------- #


def _env_pipeline_group_size() -> int:
    """``DEEPSEEK_V4_PIPELINE_GROUP_SIZE``: devices per pipeline group (see
    :func:`plan_layer_placement`). Unset (or <= 0) means "one group spanning every
    device", i.e. plain round-robin over the whole mesh."""
    raw = os.environ.get("DEEPSEEK_V4_PIPELINE_GROUP_SIZE", "1")
    try:
        pgs = int(raw)
    except ValueError:
        return 0
    return pgs if pgs > 0 else 0


def plan_layer_placement(num_layers: int, num_devices: int, group_size: int) -> list[int]:
    """Map every layer to a device, given a *pipeline group size* (PGS).

    The devices are cut into groups of ``PGS`` consecutive devices. The layer stack is
    cut into the same number of *contiguous* chunks, one per group, and each group
    round-robins its own chunk over its own devices. Groups therefore run strictly one
    after another: the model is done at the end of the last group's chunk.

    With 40 layers on 8 devices:

    * ``PGS=1`` -> 8 groups of one device, so each device owns 5 contiguous layers
      (device 0: layers 0-4, device 1: 5-9, ..., device 7: 35-39).
    * ``PGS=4`` -> 2 groups of four devices. Group 0 (devices 0-3) round-robins
      layers 0-19 (l0->d0, l1->d1, l2->d2, l3->d3, l4->d0, ..., l19->d3); group 1
      (devices 4-7) round-robins layers 20-39 (l20->d4, l21->d5, ..., l24->d4).
    * ``PGS >= 8`` (or unset) -> one group over all 8 devices: plain round-robin,
      ``layer li -> device li % 8``.

    Groups are capped at the layer count (so no group is empty) and, if ``num_devices``
    is not a multiple of ``PGS``, the trailing devices are left idle.
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


def _sliding_causal_mask(seq_len: int, sliding_window: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Additive ``[1, 1, S, S]`` sliding-window causal mask (0 keep / ``_MASK_NEG``)."""
    i = torch.arange(seq_len).view(seq_len, 1)
    j = torch.arange(seq_len).view(1, seq_len)
    keep = (j <= i) & (i - j < sliding_window)
    mask = torch.zeros(seq_len, seq_len, dtype=dtype).masked_fill(~keep, _MASK_NEG)
    return mask.view(1, 1, seq_len, seq_len)


def _block_bias(seq_len: int, n_windows: int, compress_rate: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Additive ``[1, 1, S, n_windows]`` causal block bias over compressed windows.

    Query ``t`` may attend compressed entry ``w`` iff ``w < (t + 1) // compress_rate``
    — the degenerate CSA/HCA top-k for ``seq_len <= index_topk * compress_rate``.
    """
    position_ids = torch.arange(seq_len).unsqueeze(0)
    entry = torch.arange(n_windows).view(1, 1, 1, n_windows)
    threshold = ((position_ids + 1) // compress_rate).view(1, 1, seq_len, 1)
    bias = torch.zeros(1, 1, seq_len, n_windows, dtype=dtype)
    return bias.masked_fill(entry >= threshold, _MASK_NEG)


def _window_indices(compress_rate: int, pos: int) -> tuple[int, int]:
    """``(slot, window)`` for the compressor at absolute ``pos``.

    ``slot`` is where this token's projection goes in the one-window buffer, and
    ``window`` is the index of the window that closes at ``pos`` — i.e. the entry
    the pool appends. ``window`` is ``-1`` before the first window closes, in which
    case nothing is pooled (see :meth:`DeepSeekV4Model._compressor_pool_due`).
    """
    return pos % compress_rate, (pos + 1) // compress_rate - 1


# --- Host -> device per-step packet socket (``recv_async_h2d``) ---------------- #
# The per-step input packet is streamed into the traced decode over an H2D PCIe
# socket, so the receive is a device op *inside* each submesh-0 trace rather than a
# host-side ``copy_host_to_device_tensor`` around it.
#
# The socket moves whole pages over PCIe, so the packet's page (its single row) must
# be PCIe-aligned; the three meaningful INT32 slots are padded out to that page.
_PKT_PCIE_ALIGNMENT = 64
# Room for many steps' packets, so the host can run ahead of the device without
# blocking in ``H2DSocket::write`` while the FIFO drains.
_PKT_FIFO_BYTES = 64 * _PKT_PCIE_ALIGNMENT
# Receiver core for the packet socket, disjoint from the (0,0) / (0,1) cores the
# cross-submesh direct sockets use.
_PKT_SOCKET_CORE = (0, 2)

# --- Device -> host output socket (``send_async_d2h``) ------------------------- #
# The step's output (logits, or the pre-head hidden when no lm_head is folded in) is
# streamed back over a D2H PCIe socket by an op inside the last submesh's trace, so
# the host reads it off the socket instead of issuing a ``to_torch`` readback.
_OUT_SOCKET_CORE = (0, 3)
# The FIFO lives in physically-contiguous pinned host memory. Without an IOMMU the
# driver pins a single system page at a time, so the FIFO is one 4 KB page minus the
# trailing bytes_acked counter, PCIe-aligned. A whole output does not have to fit:
# both sides move one page at a time, and the sender kernel waits for the host to
# drain when it runs ahead.
_OUT_FIFO_BYTES = 4032
# So one output row — which *is* one socket page — has to fit the FIFO. The output is
# reshaped into rows of at most this size (see :func:`_d2h_page_plan`) rather than sent
# as one enormous vocab-wide page.
_OUT_PAGE_CAP_BYTES = _OUT_FIFO_BYTES


def _d2h_page_plan(numel: int, elem_bytes: int) -> tuple[int, int]:
    """``(rows, cols)`` to reshape a flat ``numel`` output into for a D2H socket.

    ``send_async_d2h`` streams whole tensor pages, and a row-major tensor's page is
    one row, so the row width *is* the socket page size: it has to be PCIe-aligned
    and divide the output evenly. Returns the widest such row up to
    ``_OUT_PAGE_CAP_BYTES``.
    """
    for cols in range(min(numel, _OUT_PAGE_CAP_BYTES // elem_bytes), 0, -1):
        if numel % cols == 0 and (cols * elem_bytes) % _PKT_PCIE_ALIGNMENT == 0:
            return numel // cols, cols
    raise ValueError(
        f"cannot page a {numel}-element ({elem_bytes} B/elem) output for a D2H socket: no row width "
        f"divides it into {_PKT_PCIE_ALIGNMENT} B-aligned pages of at most {_OUT_PAGE_CAP_BYTES} B"
    )


class DeepSeekV4Model(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4Model`` (prefill).

    Builds the embedding, the ``num_hidden_layers`` decoder stack, the final
    :class:`DeepSeekV4HyperHead` and the shared RMSNorm from the checkpoint, then
    runs the V4 forward: embed the ids, expand to the ``hc_mult`` residual-stream
    stack, run every decoder layer (building each layer's RoPE tables + additive
    mask from the supplied ``rope`` bundle), collapse the streams and normalise.

    ``rope`` matches the bundle emitted by the reference rotary (see
    ``test_bf4_decode_demo.py``)::

        rope["main"]    = (cos_half, sin_half)          # sliding layers
        rope["compress"]= (cos_half, sin_half)          # CSA / HCA layers
        rope["win"][cr] = (cos_half, sin_half)          # per compress-rate windows

    ``forward`` returns the model's ``last_hidden_state`` ``[B, S, hidden_size]``
    (the reference's pre-``lm_head`` output); apply an external ``lm_head``
    :class:`Linear` for logits.
    """

    def __init__(
        self,
        config,
        loader: DeepseekV4WeightLoader,
        full_device: ttnn.MeshDevice,
        cache: Optional[WeightCache] = None,
        cache_dir: Optional[str] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat4_b,
        max_layers: Optional[int] = None,
        use_submeshes: bool = False,
        require_cache: bool = False,
        pipeline_group_size: Optional[int] = None,
    ):
        """Build the V4-Flash model off the checkpoint.

        Caching: pass either a pre-built ``cache`` :class:`WeightCache` or a
        ``cache_dir`` (the model builds ``WeightCache(cache_dir)`` and owns the
        per-layer ``layers.N`` / head namespacing internally, so callers no longer
        repeat the ``WeightCache(...).sub("layers.N")`` dance). ``None`` for both
        disables caching (every weight is converted from the checkpoint).

        ``require_cache=True`` asserts the converted-tile cache is fully populated:
        any tile-cached weight that would otherwise be (re)loaded from the HF
        checkpoint raises instead. The small host-side scalars (attention sinks,
        the HC ``scale`` triplets, the hash router's ``tid2eid`` table) and the
        locally-computed RoPE rotate matrix have no tile cache by design and are
        always materialised, so they are exempt.
        """
        self.config = config
        self.loader = loader
        self.weight_dtype = weight_dtype
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
        self.require_cache = require_cache

        self.use_submeshes = use_submeshes
        self.num_submeshes = full_device.get_num_devices()

        # Layer -> submesh placement is set by the *pipeline group size* (PGS, see
        # :func:`plan_layer_placement`): the devices are cut into groups of PGS
        # consecutive devices, the stack into one contiguous chunk per group, and each
        # group round-robins its chunk over its own devices. PGS >= num_devices (the
        # default) collapses to plain round-robin over the whole mesh, whose dataflow is
        # the familiar ring 0 -> 1 -> ... -> (S-1) -> 0; PGS=1 gives one contiguous slice
        # of layers per device. ``layer_submesh_ids[li]`` is the mapping;
        # ``pipeline_submesh_ids`` lists the populated submeshes in the order the stack
        # first visits them, and ``pipeline_stages`` counts them.
        n = config.num_hidden_layers if max_layers is None else min(max_layers, config.num_hidden_layers)
        self.num_layers = n
        if pipeline_group_size is None:
            pipeline_group_size = _env_pipeline_group_size()
        self.pipeline_group_size = pipeline_group_size
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
            logger.info(
                f"Using submeshes: {self.num_submeshes} (pipeline group size "
                f"{pipeline_group_size or self.num_submeshes}, {self.pipeline_stages} populated)"
            )
            full_device.reshape(ttnn.MeshShape(1, full_device.get_num_devices()))
            self.submeshes = []
            for i in range(self.num_submeshes):
                self.submeshes.append(full_device.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, i)))
            self.first_device = self.submeshes[0]
            self.last_device = self.submeshes[-1]

            # Create socket pairs between submeshes for copying hidden_states .
            # One directed pair per handoff the placement needs (``pipeline_edges``),
            # reused for all forward passes. Under plain round-robin those edges are the
            # ring 0 -> 1 -> ... -> (S-1) -> 0 (the wrap-around included, since submesh 0
            # is revisited for layers S, 2S, ...); under a smaller pipeline group size
            # they are the per-group rings plus the single group-to-group edge.
            self.submesh_socket_pairs = {}
            socket_memconfig = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 16 * 1024)
            for from_id, to_id in self.pipeline_edges:
                from_submesh = self.submeshes[from_id]
                to_submesh = self.submeshes[to_id]
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
                sender_socket, receiver_socket = ttnn.create_socket_pair(from_submesh, to_submesh, socket_config)
                self.submesh_socket_pairs[(from_id, to_id)] = (sender_socket, receiver_socket)
        else:
            self.first_device = full_device
            self.last_device = full_device

        n = self.num_layers

        self.embed_tokens = DeepSeekV4Embedding(loader, self.first_device, cache=cache)

        self.layers: list[DeepSeekV4DecoderLayer] = []
        self.layer_devices: list[ttnn.MeshDevice] = []
        for li in range(n):
            if self.use_submeshes:
                layer_device_id = self._submesh_id_for_layer(li)
                current_device = self.submeshes[layer_device_id]
                logger.info(f"Layer {li} is on device {layer_device_id}")
            else:
                current_device = self.device
            self.layer_devices.append(current_device)
            layer_type = config.layer_types[li]
            is_hash = config.mlp_layer_types[li] == "hash_moe"
            layer_cache = cache.sub(f"layers.{li}")
            weights = self._build_layer_weights(li, layer_type, is_hash)
            gate = self._hash_gate(li) if is_hash else None
            experts = DeepSeekV4PreloadedExperts(
                config,
                self._expert_provider(li),
                current_device,
                dtype=weight_dtype,
                cache=layer_cache.sub("mlp"),
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
                )
            )
            _profile(current_device)

        # The head (hc_head / norm / external lm_head) must live where the *last*
        # decoder layer's output lands, not unconditionally on the final submesh —
        # otherwise a capped (``max_layers``) stack would end on a lower submesh
        # than the head and mismatch devices.
        if self.layer_devices:
            self.last_device = self.layer_devices[-1]

        # Per-layer decode state (in-place sliding K=V + optional compressor projections).
        self.sliding_window = config.sliding_window
        self._decode_max_seq: Optional[int] = None
        self.kv_caches: list[_StaticLayerCache] = []
        # Paged multi-session decode (traced path; see :meth:`prepare_static_decode`).
        self._paged: Optional[PagedKVManager] = None
        # H2D socket carrying the per-step input packet into the traced decode, and the
        # D2H socket carrying the step output back out (both allocated by
        # :meth:`prepare_static_decode`).
        self._pkt_socket = None
        self._out_socket = None
        self._out_plan: Optional[tuple[int, int]] = None  # (rows, cols) of one output
        self._out_torch_dtype: Optional[torch.dtype] = None
        self._paged_groups: dict[str, PagedGroup] = {}
        self._external_pools: Optional[dict[int, ttnn.Tensor]] = None
        self._active_sid: Optional[int] = None
        # session id -> (submesh index, layer) -> compressor window buffers, held while
        # the session is not the active one (see :meth:`activate_session`).
        self._session_state: dict[int, dict] = {}

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
        loader = self.loader
        return lambda: dequantize_weight(loader.get_tensor(name), loader.get_scale(name))

    @staticmethod
    def _attn_keys(layer_type: str) -> list[str]:
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
        return keys

    def _build_layer_weights(self, layer_idx: int, layer_type: str, is_hash: bool) -> dict:
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

    def _submesh_id_for_layer(self, layer_idx: int) -> int:
        """The submesh layer ``layer_idx`` lives on, per the pipeline-group placement
        computed in :meth:`__init__` (see :func:`plan_layer_placement`)."""
        return self.layer_submesh_ids[layer_idx]

    def _next_layer_on_submesh(self, layer_idx: int) -> Optional[int]:
        """The next global layer placed on the same submesh as ``layer_idx`` (the one
        whose weights are worth prefetching while this device waits), or ``None``."""
        k = self.layer_submesh_ids[layer_idx]
        for li in range(layer_idx + 1, self.num_layers):
            if self.layer_submesh_ids[li] == k:
                return li
        return None

    def _expert_provider(self, layer_idx: int):
        def provider(e: int):
            base = f"layers.{layer_idx}.mlp.experts.{e}"
            gate = self._thunk(f"{base}.gate_proj.weight")()
            up = self._thunk(f"{base}.up_proj.weight")()
            down = self._thunk(f"{base}.down_proj.weight")()
            return torch.cat([gate, up], dim=0).float(), down.float()

        return provider

    def _hash_gate(self, layer_idx: int) -> DeepSeekV4HashRouter:
        weights = {
            "gate.weight": self._thunk(f"layers.{layer_idx}.mlp.gate.weight"),
            "gate.tid2eid": self.loader.get_tensor(f"layers.{layer_idx}.mlp.gate.tid2eid").long(),
        }
        if self.use_submeshes:
            this_device = self.submeshes[self._submesh_id_for_layer(layer_idx)]
        else:
            this_device = self.first_device
        return DeepSeekV4HashRouter(self.config, weights, this_device)

    # -- compressor pooling schedule -------------------------------------------- #
    #
    # A CSA/HCA compressor emits a new compressed entry once every
    # ``compress_rate`` tokens, and the block-bias exposes entries
    # ``w < (pos+1)//compress_rate`` -- constant between two window closures. So the
    # pool only runs on the steps that close a window, and it pools *only* that
    # window's ``compress_rate`` projections, appending one entry to the layer's
    # combined buffer (:class:`_StaticLayerCache`). Both together make a step
    # ``O(compress_rate)`` instead of ``O(max_seq)``, so throughput is flat in
    # ``max_seq``. Pooling off-closure is not merely slower but wrong -- the window
    # buffer is only fully written at a closure -- so there is no A/B switch here.

    # -- SDPA masking mode ------------------------------------------------------- #
    #
    # Once the sliding ring is full, a CSA/HCA layer's valid KV set is a contiguous
    # prefix (see :func:`sdpa_causal_ok`), so SDPA-decode can be bounded by a single
    # ``cur_pos`` in causal mode instead of an additive mask. The mask is *data*, not
    # control flow, so the masked kernel always walks the whole ``max_seq``-sized KV
    # axis; the causal kernel derives its chunk range from the position and skips the
    # rest. That makes attention cost track the actual position rather than
    # ``max_seq``, and drops the per-step per-layer head-broadcast of the mask row.
    # The sub-window steps (whose valid set has a hole) keep the mask.
    # ``DEEPSEEK_V4_SDPA_CAUSAL=0`` forces the mask everywhere -- the previous
    # behaviour, and on the traced path it collapses the capture back to one variant
    # per pool phase.
    _SDPA_CAUSAL = os.environ.get("DEEPSEEK_V4_SDPA_CAUSAL", "1") not in ("0", "", "false", "False")

    def _sdpa_causal_at(self, layer_type: str, pos: int) -> bool:
        """Should ``layer_type`` at absolute ``pos`` use causal SDPA over the mask?"""
        return self._SDPA_CAUSAL and sdpa_causal_ok(self.sliding_window, layer_type, pos)

    def _compressor_pool_due(self, layer_type: str, pos: int) -> bool:
        """Does the step at absolute ``pos`` close a window for ``layer_type``?"""
        if layer_type == "sliding_attention":
            return False
        return (pos + 1) % self.config.compress_rates[layer_type] == 0

    def _compress_rates_for(self, layer_types) -> list[int]:
        """Sorted distinct compress rates among ``layer_types`` (sliding layers have none)."""
        return sorted({self.config.compress_rates[t] for t in layer_types if t != "sliding_attention"})

    def _build_pool_phases(self, crs: list[int]) -> tuple[list[tuple], dict[int, int]]:
        """The distinct pooling patterns over a window period, and the pos -> phase map.

        A step's pattern is ``tuple((pos+1) % cr == 0 for cr in crs)``, which repeats
        with period ``lcm(crs)``. Far fewer than ``2 ** len(crs)`` patterns are
        reachable, because the rates divide one another: with the default
        ``{CSA: 4, HCA: 128}`` an HCA closure *always* coincides with a CSA closure
        (4 | 128), so the reachable set is three phases — pool nothing, pool CSA,
        pool CSA+HCA — and never "HCA alone". The traced path captures one trace
        variant per phase, so this directly bounds the trace-memory cost.
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
        """Index into :attr:`_pool_phases` of the pooling schedule to use at ``pos``."""
        return self._pool_phase_of[pos % self._pool_period]

    def _sdpa_causal_step(self, pos: int) -> bool:
        """Whether this step's compressor layers use causal SDPA. Layer-type
        independent (only ``pos`` vs. the ring capacity matters), so one boolean
        selects the trace variant for the whole stack."""
        return self._SDPA_CAUSAL and pos + 1 >= self.sliding_window

    def _variant_key(self, pos: int) -> tuple[bool, int]:
        """The captured trace variant to replay at ``pos``: (SDPA mode, pool phase)."""
        return (self._sdpa_causal_step(pos), self._pool_phase_index(pos))

    def _reachable_phases(self, causal: bool) -> list[int]:
        """Pool-phase indices that can co-occur with this SDPA mode.

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

    def _reachable_variants(self) -> list[tuple[bool, int]]:
        """Every (SDPA mode, pool phase) pair a step can actually ask for."""
        modes = [False, True] if self._SDPA_CAUSAL else [False]
        return [(causal, phase) for causal in modes for phase in self._reachable_phases(causal)]

    @staticmethod
    def _sm_pool_key(sm: dict, pool_flags: dict[int, bool], causal: bool) -> tuple:
        """A submesh's slice of a global variant: only the rates its own layers use,
        plus the SDPA mode (which only compressor layers observe).

        Submeshes that host no compressor layer (or only one of the rates) collapse
        several global variants onto the same capture — a sliding-only submesh is
        captured exactly once however many variants the stack has.
        """
        return (causal and bool(sm["pool_crs"]), tuple(pool_flags[cr] for cr in sm["pool_crs"]))

    # -- decode KV-cache state -------------------------------------------------- #
    def reset_caches(self, max_seq: int) -> None:
        """Allocate empty fixed-size dense decode buffers for a fresh sequence (the
        eager :meth:`decode` path; the traced path uses :meth:`prepare_static_decode`).

        ``max_seq`` is the longest absolute position + 1 the caller will decode
        (prompt + generation), padded to tile / compress-rate multiples as needed.
        """
        self._decode_max_seq = max_seq
        self.kv_caches = [
            build_static_layer_cache(
                self.layer_devices[li],
                self.sliding_window,
                self.config.layer_types[li],
                self.config.head_dim,
                max_seq,
                self.config.compress_rates,
            )
            for li in range(self.num_layers)
        ]

    # ------------------------------------------------------------------ #
    # Paged multi-session decode
    #
    # Several conversations share one captured trace. A trace bakes in the
    # *addresses* of the buffers it touches, so per-session state cannot live in
    # per-session buffers -- the trace would only ever see the first session's. Two
    # mechanisms cover the two kinds of state:
    #
    #   * The KV caches (all the memory that matters) move behind a block pool per
    #     layer plus a ``page_table`` tensor per (submesh, layer type). The trace
    #     addresses the pool and the table; switching sessions rewrites the table's
    #     *contents* with that session's logical->physical block row. Blocks are
    #     handed out on demand, so N conversations share a total token budget instead
    #     of reserving ``N x max_context`` (see :mod:`.paged_cache`).
    #   * The compressor window buffers (one window of projections, a few KB) stay
    #     dense and are swapped in and out of the trace-addressed buffers on a
    #     session switch, which happens per turn rather than per token.
    #
    # Everything a session needs is allocated up front by
    # :meth:`prepare_static_decode`: allocating device buffers once a trace exists on
    # the device is unsafe, so :meth:`open_session` only claims a pre-built slot and
    # does host-side book-keeping.
    # ------------------------------------------------------------------ #
    @property
    def paged(self) -> bool:
        """Is this model set up for paged (multi-session) traced decode?"""
        return self._paged is not None

    @property
    def active_session(self) -> Optional[int]:
        return self._active_sid

    def _require_paged(self) -> PagedKVManager:
        if self._paged is None:
            raise RuntimeError("call prepare_static_decode(..., num_sessions=N) for paged multi-session decode")
        return self._paged

    def open_session(self) -> int:
        """Claim a session slot (its sliding-ring blocks) and return its id.

        Purely host-side: the device buffers were all allocated by
        :meth:`prepare_static_decode`. A fresh session needs no cache zeroing -- every
        step masks (or, in causal mode, bounds itself below) the rows it has not
        written yet, so a recycled block is never read.
        """
        paged = self._require_paged()
        if not self._free_session_state:
            raise PagedCacheFull(f"all {self._max_sessions} session slots are in use")
        sid = paged.open_session()
        self._session_state[sid] = self._build_session_state(self._free_session_state.pop())
        return sid

    def close_session(self, sid: int) -> None:
        """Release a session's blocks back to the pool."""
        paged = self._require_paged()
        if sid == self._active_sid:
            self._active_sid = None
        paged.close_session(sid)
        state = self._session_state.pop(sid, None)
        if state is not None:
            self._free_session_state.append(state)

    def reset_session(self, sid: int) -> None:
        """Rewind a session to position 0: free its compressed blocks and clear its
        compressor window state (keeping its ring blocks, whose stale rows are masked
        until rewritten)."""
        paged = self._require_paged()
        paged.reset_session(sid)
        if sid == self._active_sid:
            self._clear_active_compressor_state()
        else:
            self._build_session_state(self._session_state[sid])
        self._write_page_tables(sid, list(self._paged_groups))

    def activate_session(self, sid: int) -> None:
        """Make ``sid`` the session the next :meth:`decode_traced` steps belong to:
        point every page table at its blocks and swap its compressor window state into
        the buffers the traces address."""
        paged = self._require_paged()
        if not paged.has_session(sid):
            raise KeyError(f"no such session {sid}")
        if sid == self._active_sid:
            return
        if self._active_sid is not None:
            self._save_active_compressor_state(self._active_sid)
        self._load_compressor_state(sid)
        self._active_sid = sid
        self._write_page_tables(sid, list(self._paged_groups))

    def ensure_session_capacity(self, pos: int) -> None:
        """Give the active session blocks for every row a step at ``pos`` will touch,
        refreshing only the page tables whose row actually changed (a compressor group
        grows one block every ``compress_rate * block_size`` tokens)."""
        paged = self._require_paged()
        if self._active_sid is None:
            raise RuntimeError("call activate_session() before decoding")
        changed = paged.ensure_capacity(self._active_sid, pos)
        if changed:
            self._write_page_tables(self._active_sid, changed)

    def session_usage(self) -> dict:
        """Per-group ``(blocks used, pool size)``, for status reporting."""
        return self._require_paged().usage()

    def session_tokens_left(self) -> int:
        """Tokens the shared pool can still admit across all open sessions."""
        return self._require_paged().tokens_left()

    # -- paged device state ----------------------------------------------------- #
    def _paged_view(self, sm: dict, li: int) -> Optional[PagedLayerView]:
        """The pool + page table layer ``li`` reads its KV through, or ``None`` when
        this model runs the dense caches."""
        if self._paged is None:
            return None
        group = self._paged_groups[self.config.layer_types[li]]
        return PagedLayerView(sm["pools"][li], sm["page_tables"][group.layer_type], group.position_modulo)

    def _write_page_tables(self, sid: int, groups) -> None:
        """Copy ``sid``'s page-table rows into the persistent device tables of every
        submesh that hosts a layer of those groups."""
        paged = self._require_paged()
        for group in groups:
            row = ttnn.from_torch(paged.page_row(sid, group), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            for sm in self.submeshes_io:
                table = sm["page_tables"].get(group)
                if table is not None:
                    ttnn.copy_host_to_device_tensor(row, table)

    def _compressor_slots(self):
        """``(submesh, layer, buffer name)`` for every per-session compressor buffer."""
        for sm in self.submeshes_io:
            for li, scache in sm["scaches"].items():
                for name in ("win_kv", "win_gate", "prev_kv", "prev_gate"):
                    if getattr(scache, name) is not None:
                        yield sm, li, name

    def _build_session_state(self, into: Optional[dict] = None) -> dict:
        """A session's held-aside compressor buffers, cleared to their empty values.

        ``prev_gate`` starts at ``_MASK_NEG`` rather than 0 so the first window's
        absent Ca half carries softmax weight 0 (see :class:`_StaticLayerCache`).
        ``into`` clears and reuses an existing slot's buffers rather than allocating,
        which is what keeps :meth:`open_session` safe once traces exist; the allocating
        form runs only from :meth:`prepare_static_decode`.
        """
        state = {} if into is None else into
        for sm, li, name in self._compressor_slots():
            key = (sm["index"], li, name)
            fill = _MASK_NEG if name == "prev_gate" else 0.0
            if into is None:
                buf = getattr(sm["scaches"][li], name)
                state[key] = ttnn.from_torch(
                    torch.full(list(buf.shape), fill),
                    dtype=buf.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=sm["device"],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            else:
                ttnn.fill(state[key], fill, output_tensor=state[key])
        return state

    def _clear_active_compressor_state(self) -> None:
        """Clear the trace-addressed compressor buffers in place."""
        for sm, li, name in self._compressor_slots():
            buf = getattr(sm["scaches"][li], name)
            ttnn.fill(buf, _MASK_NEG if name == "prev_gate" else 0.0, output_tensor=buf)

    def _save_active_compressor_state(self, sid: int) -> None:
        for sm, li, name in self._compressor_slots():
            ttnn.copy(getattr(sm["scaches"][li], name), self._session_state[sid][(sm["index"], li, name)])

    def _load_compressor_state(self, sid: int) -> None:
        for sm, li, name in self._compressor_slots():
            ttnn.copy(self._session_state[sid][(sm["index"], li, name)], getattr(sm["scaches"][li], name))

    # -- per-layer RoPE tables / masks ------------------------------------------ #
    def _to_tt(self, t: torch.Tensor, device: ttnn.MeshDevice) -> ttnn.Tensor:
        _profile(device)

        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def _rope_rows_decode(
        self, rope: dict, pos: int, layer_type: str, compress_rate: Optional[int], cache: dict, device: ttnn.MeshDevice
    ):
        """Single-position RoPE rows for a decode step.

        Returns ``(cos, sin, neg_sin, cos_win, sin_win)``, all one ``[1,1,1,Rd]`` row:
        ``cos/sin/neg_sin`` at absolute ``pos``, and ``cos_win/sin_win`` at the window
        closing at ``pos`` (``None`` for sliding layers). Incremental pooling emits a
        single compressed entry per closure, so a single window row is all it needs.
        """
        key = f'{"sliding" if layer_type == "sliding_attention" else compress_rate}_{device.id()}'
        if key in cache:
            return cache[key]
        cos_h, sin_h = rope["main"] if layer_type == "sliding_attention" else rope["compress"]
        cos_row, sin_row = cos_h[pos : pos + 1], sin_h[pos : pos + 1]
        cos_full, sin_full = make_rope_table(cos_row, sin_row)
        cos_tt = self._to_tt(cos_full, device)
        sin_tt = self._to_tt(sin_full, device)
        neg_sin_tt = self._to_tt(-sin_full, device)

        cos_win_tt = sin_win_tt = None
        if layer_type != "sliding_attention":
            assert self._decode_max_seq is not None
            if self._decode_max_seq // compress_rate > 0:
                # ``rope["win"][cr]`` row w is the "compress" family at position w*cr;
                # clamp because before the first closure there is no window to pool.
                wi = max(_window_indices(compress_rate, pos)[1], 0)
                cw_h, sw_h = rope["win"][compress_rate]
                cw, sw = make_rope_table(cw_h[wi : wi + 1], sw_h[wi : wi + 1])
                cos_win_tt = self._to_tt(cw, device)
                sin_win_tt = self._to_tt(sw, device)
        out = (cos_tt, sin_tt, neg_sin_tt, cos_win_tt, sin_win_tt)
        cache[key] = out
        return out

    def _copy_streams_between_submeshes(self, streams, from_submesh_id: int, to_submesh_id: int):
        """Move the decode residual streams between two adjacent submeshes over the
        pre-created socket pair — device-to-device, with no host round-trip.

        Used by the eager :meth:`decode` path: allocate a fresh tensor on the target
        submesh, receive into it, and return it (the loop reassigns ``streams``). The
        traced path instead folds the send/recv directly into each submesh's trace
        (see :meth:`_decode_submesh_static`).
        """
        to_submesh = self.submeshes[to_submesh_id]
        sender_socket, receiver_socket = self.submesh_socket_pairs[(from_submesh_id, to_submesh_id)]
        output_tensor = ttnn.allocate_tensor_on_device(streams.spec, to_submesh)
        with _region("PIPELINE_HANDOFF"):
            ttnn.experimental.send_direct_async(streams, sender_socket)
            ttnn.experimental.recv_direct_async(output_tensor, receiver_socket)
        streams.deallocate(True)
        return output_tensor

    def decode(self, token_id: int, pos: int, rope: dict) -> ttnn.Tensor:
        """Generate one step: feed ``token_id`` at absolute position ``pos`` against
        the running KV cache; returns ``[B, 1, 1, hidden]`` (apply ``lm_head`` for logits).

        ``rope`` is the *full* (max-length) host bundle; the needed rows are sliced
        per layer. The prompt is prefilled by calling this once per prompt token at
        ascending positions, so the cache holds positions ``0 .. pos - 1``."""
        ids = torch.tensor([[token_id]], dtype=torch.long)
        ids_tt = ttnn.from_torch(
            ids.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.first_device
        )
        with _region("EMBED"):
            inputs_embeds = self.embed_tokens(ids_tt)  # [B, 1, D]
            b, s, d = inputs_embeds.shape
            streams = ttnn.reshape(inputs_embeds, [b, s, 1, d])
            streams = ttnn.repeat(streams, ttnn.Shape([1, 1, self.config.hc_mult, 1]))  # [B, 1, hc_mult, D]

        rope_cache: dict = {}
        last_submesh_id = 0
        w = self.sliding_window
        if not self.kv_caches:
            raise RuntimeError("call reset_caches(max_seq) before decode()")
        for li, layer in enumerate(self.layers):
            if self.use_submeshes:
                current_submesh_id = self._submesh_id_for_layer(li)
                if current_submesh_id != last_submesh_id:
                    streams = self._copy_streams_between_submeshes(streams, last_submesh_id, current_submesh_id)
                this_device = self.submeshes[current_submesh_id]
            else:
                this_device = self.first_device
            layer_type = self.config.layer_types[li]
            compress_rate = None if layer_type == "sliding_attention" else self.config.compress_rates[layer_type]
            cos_tt, sin_tt, neg_sin_tt, cos_win_tt, sin_win_tt = self._rope_rows_decode(
                rope, pos, layer_type, compress_rate, rope_cache, this_device
            )
            causal = self._sdpa_causal_at(layer_type, pos)
            mask = (
                None
                if causal
                else host_decode_mask(w, layer_type, compress_rate, pos, self._decode_max_seq, this_device)
            )
            win_slot = win_row = None
            if compress_rate is not None:
                slot, wi = _window_indices(compress_rate, pos)
                win_slot = int32_pos_tensor(slot, this_device)
                win_row = int32_pos_tensor(w + max(wi, 0), this_device)
            streams = layer.decode(
                streams,
                cos_tt,
                sin_tt,
                neg_sin_tt,
                cos_win_tt,
                sin_win_tt,
                mask,
                self.kv_caches[li],
                int32_pos_tensor(pos % w, this_device),
                int32_pos_tensor(pos, this_device),
                input_ids=ids,
                pool_compressor=self._compressor_pool_due(layer_type, pos),
                win_slot=win_slot,
                win_row=win_row,
                sdpa_cur_pos=(
                    int32_pos_tensor(sdpa_causal_cur_pos(w, compress_rate, pos), this_device) if causal else None
                ),
            )
            last_submesh_id = current_submesh_id
            _profile(this_device)
            next_on_device = self._next_layer_on_submesh(li)
            if next_on_device is not None:
                self.layers[next_on_device].self_attn.prefetch_weights()
        with _region("HC_HEAD"):
            hidden = self.hc_head(streams)
        with _region("FINAL_NORM"):
            return self.norm(hidden)

    # ------------------------------------------------------------------ #
    # Traced decode (one reusable trace per submesh / device)
    #
    # The eager :meth:`decode` is host-bound: every step re-dispatches ~43
    # layers' worth of ops, rebuilds the RoPE rows / masks from host, reads the
    # MoE routing weights back to host, and host-copies the residual streams
    # across submeshes. The traced path captures one ``ttnn`` trace per submesh
    # (so each device replays its own slice of the stack) and, between replays,
    # writes the tiny per-step inputs onto submesh 0 *only*, fused into ONE
    # fixed-shape INT32 packet (token + cache positions + the additive masks carried
    # as float32-bits-as-int32). The per-step RoPE rows are generated on device from
    # the position (no host build / transport), as are the additive attention masks.
    # The streams and packet are socket-copied between submeshes from inside the
    # traces themselves, where each submesh splits the packet into the individual
    # inputs on device (no per-step host op dispatch past submesh 0). All cross-token
    # state lives in fixed-size in-place caches (:class:`_StaticLayerCache`) so a
    # single capture serves every step.
    # See :meth:`prepare_static_decode` / :meth:`decode_traced`.
    # ------------------------------------------------------------------ #

    def _build_static_layer_cache(self, li: int, device: ttnn.MeshDevice) -> "_StaticLayerCache":
        """Allocate a layer's fixed-size in-place caches *empty* (all-zero)."""
        assert self._decode_max_seq is not None, "set max_seq via reset_caches or prepare_static_decode first"
        return build_static_layer_cache(
            device,
            self.sliding_window,
            self.config.layer_types[li],
            self.config.head_dim,
            self._decode_max_seq,
            self.config.compress_rates,
            paged=self._paged is not None,
        )

    def _external_pool_blocks(self, pools: dict[int, ttnn.Tensor]) -> dict[str, int]:
        """Validate caller-owned pools against this model's geometry and read the pool
        size of each group off them."""
        blocks: dict[str, int] = {}
        for li in range(self.num_layers):
            if li not in pools:
                raise ValueError(f"no external block pool for layer {li}")
            group = self._paged_groups[self.config.layer_types[li]]
            shape = list(pools[li].shape)
            want = [1, group.block_size, self.config.head_dim]
            if shape[1:] != want:
                raise ValueError(
                    f"layer {li} ({group.layer_type}) pool has block shape {shape[1:]}, expected {want} "
                    f"for a {group.block_size}-row block"
                )
            seen = blocks.setdefault(group.layer_type, shape[0])
            if seen != shape[0]:
                raise ValueError(
                    f"every pool of group {group.layer_type} must have the same block count, "
                    f"got {seen} and {shape[0]}"
                )
        return blocks

    def _build_block_pool(self, li: int, device: ttnn.MeshDevice) -> ttnn.Tensor:
        """One layer's KV block pool ``[num_blocks, 1, block_size, Dh]`` (all-zero).

        Block ``0`` is the shared zero block every unmapped page-table entry points at,
        so it must stay zero -- :class:`PagedKVManager` never hands it out.
        """
        if self._external_pools is not None:
            return self._external_pools[li]
        group = self._paged_groups[self.config.layer_types[li]]
        num_blocks = self._paged.pools[group.layer_type].num_blocks
        return ttnn.from_torch(
            torch.zeros(num_blocks, 1, group.block_size, self.config.head_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _build_page_tables(self, layer_types, device: ttnn.MeshDevice) -> dict:
        """Persistent ``[1, logical_blocks]`` INT32 page tables, one per layer type on
        this submesh. The traces bake in these addresses; :meth:`activate_session`
        rewrites their contents."""
        return {
            lt: ttnn.from_torch(
                torch.zeros(1, self._paged_groups[lt].logical_blocks, dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
            for lt in layer_types
        }

    def reset_static_caches(self) -> None:
        """Zero every traced-decode cache so a fresh sequence can start at position 0.

        The captured traces address these buffers directly, so they are zeroed in
        place: reallocating them would invalidate every capture, and even a
        temporary device allocation is unsafe while a trace exists. (``fill`` still
        logs the allocator's "unsafe with an active trace" warning for its own
        scratch; the cache buffers themselves are untouched by it.)

        ``prev_gate`` is refilled with ``_MASK_NEG`` rather than 0, matching how
        :func:`build_static_layer_cache` allocates it: it gates window 0's absent Ca
        half, which a 0 fill would give real softmax weight instead of none.

        In paged mode this only touches the compressor window buffers (the KV caches
        live in the block pools); use :meth:`reset_session` to rewind one session.
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
        num_sessions: int = 0,
        total_tokens: int | None = None,
        block_size: int = 32,
        tokens_per_block: int | None = None,
        pools: dict[int, ttnn.Tensor] | None = None,
    ) -> None:
        """Allocate the traced-decode state (the prompt is prefilled by replaying
        :meth:`decode_traced` once per prompt token into these empty caches).

        Builds, per submesh: the fixed-size in-place caches (empty / all-zero), the
        constant window-RoPE tables, and the persistent socket recv buffers
        (residual streams + the single fused per-step input packet). Submesh 0
        additionally gets the H2D socket the per-step packet arrives on — the only
        host->device traffic of a traced step.
        ``max_seq`` must be a multiple of every compress-rate (the caller
        pads it) so each compressor's fixed capacity tiles cleanly into windows.
        ``lm_head`` (optional) is folded into the last submesh's trace so a step
        returns logits directly.

        ``num_sessions`` > 0 switches the KV caches to the paged multi-session layout:
        each layer gets a block pool instead of a dense buffer, sized (with
        ``total_tokens``, defaulting to one full ``max_seq``) for that many concurrent
        conversations sharing the budget. Everything a session needs is allocated here,
        before any trace exists, because allocating on a device that holds a trace is
        unsafe; :meth:`open_session` then only claims a slot.

        Block geometry comes from either ``block_size`` (the same row count for every
        layer type) or ``tokens_per_block`` (row counts scaled by compress rate, so a
        block spans the same context everywhere -- see :func:`.paged_cache.build_groups`).
        ``pools`` supplies externally allocated block pools keyed by layer index, for a
        caller that owns the KV memory (the vLLM wrapper, whose pool size the serving
        stack decides); their block count then replaces the internal pool plan.
        """
        if not self.use_submeshes:
            raise NotImplementedError("traced decode requires use_submeshes=True")
        cfg = self.config
        for cr in {cfg.compress_rates[t] for t in cfg.layer_types[: self.num_layers] if t != "sliding_attention"}:
            assert max_seq % cr == 0, f"max_seq ({max_seq}) must be a multiple of compress_rate {cr}"
        self._external_pools = pools
        if num_sessions > 0:
            self._paged_groups = build_groups(
                cfg.layer_types[: self.num_layers],
                cfg.compress_rates,
                self.sliding_window,
                max_seq,
                block_size=None if tokens_per_block else block_size,
                tokens_per_block=tokens_per_block,
            )
            pool_blocks = (
                self._external_pool_blocks(pools)
                if pools is not None
                else plan_pool_blocks(self._paged_groups, num_sessions, total_tokens or max_seq)
            )
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
        self._traced_rope = rope
        self._lm_head_traced = lm_head
        self._decode_max_seq = max_seq
        self._cr_caps = {
            cr: (max_seq, max_seq // cr)
            for cr in {cfg.compress_rates[t] for t in cfg.layer_types[: self.num_layers] if t != "sliding_attention"}
        }
        self._pool_crs = self._compress_rates_for(cfg.layer_types[: self.num_layers])
        self._pool_period = math.lcm(*self._pool_crs) if self._pool_crs else 1
        self._pool_phases, self._pool_phase_of = self._build_pool_phases(self._pool_crs)

        rd = cfg.qk_rope_head_dim
        hc, d, w = cfg.hc_mult, cfg.hidden_size, self.sliding_window
        ids = self.layer_submesh_ids

        # --- Canonical per-step input packet layout (shared by every submesh) --- #
        # All per-step inputs are fused into ONE tiny fixed-shape INT32 packet
        # ``[1, 1, 1, 16]`` (ROW_MAJOR), a single persistent buffer on submesh 0
        # streamed in from host *only* there over an H2D socket and then flowed
        # downstream over the existing device-to-device socket (see
        # :meth:`_decode_submesh_static`), so no submesh past the first sees any host
        # traffic at all.
        #
        #   idx 0 : token (INT32; embedding/hash use it typecast to uint32)
        #   idx 1 : pos_sliding (INT32)
        #   idx 2 : pos_compress (INT32)
        #
        # The per-step RoPE rows and additive masks are *not* in the packet — they are
        # both generated on device from ``pos_compress`` against constant tables (see
        # :meth:`_device_rope` and :meth:`_device_mask`).
        # Slots past the prefix are padding: the packet's row is one H2D socket page,
        # so its width is set by the PCIe alignment, not by the payload. Nothing reads
        # them.
        self._pkt_int_prefix = 3  # [token, pos_sliding, pos_compress]
        self._pkt_page_bytes = _PKT_PCIE_ALIGNMENT
        self._pkt_w = self._pkt_page_bytes // 4
        self._pkt_rd = rd

        # --- On-device RoPE generation constants ------------------------------- #
        # RoPE is ``cos/sin(pos * inv_freq) * attention_scaling`` with ``inv_freq`` /
        # ``attention_scaling`` position-independent per family ("main" sliding,
        # "compress" CSA/HCA). Recover them from the host ``rope`` tables (so the
        # device output matches them exactly): at p=0 the table is ``scaling`` (sin=0),
        # and ``inv_freq[j] = atan2(sin_half[1,j], cos_half[1,j])`` (all |inv_freq|<π).
        # Stored already interleaved-by-2 to match ``make_rope_table``'s expansion.
        self._rope_gen: dict[str, tuple[torch.Tensor, float]] = {}
        for rt in ("main", "compress"):
            cos_h, sin_h = rope[rt]
            scaling = float(cos_h[0, 0].item())
            inv_freq_half = torch.atan2(sin_h[1].float(), cos_h[1].float())  # [rd/2]
            inv_freq_full = inv_freq_half.repeat_interleave(2).reshape(1, 1, 1, -1)  # [1,1,1,rd]
            self._rope_gen[rt] = (inv_freq_full, scaling)

        def _dev_zeros(shape, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
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
                "pools": {li: self._build_block_pool(li, device) for li in layers_k} if self._paged else {},
                "page_tables": self._build_page_tables(types, device) if self._paged else {},
                "pool_crs": self._compress_rates_for(types),
                "traces": {},  # local variant key -> (trace id, persistent output)
                "tids": {},  # global variant key -> trace id
                "outputs": {},  # global variant key -> persistent output
            }
            # Per-family inv_freq constants for the rope families this submesh uses.
            for rt in ({"main"} if "sliding_attention" in types else set()) | ({"compress"} if crs else set()):
                inv_freq_full, scaling = self._rope_gen[rt]
                sm["rope_invfreq"][rt] = (
                    ttnn.from_torch(inv_freq_full, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device),
                    scaling,
                )
            # Per-layer-type constant index tables for on-device mask generation. The
            # mask row is ``invalid * _MASK_NEG`` with ``invalid = (A > pos)`` over the
            # sliding columns OR ``(B >= (pos+1)//cr)`` over the compressor columns;
            # the two regions are packed into full-width A / B tables with ``-1``
            # fillers in the *other* region (``-1`` is never ``> pos`` nor ``>= thr``),
            # so a single compare per table covers each region without a tile-boundary
            # ``concat``.
            for lt in types:
                if lt == "sliding_attention":
                    a = torch.arange(w, dtype=torch.float32)  # slot index 0..W-1
                    b = None
                    cr = None
                else:
                    cr = cfg.compress_rates[lt]
                    n_win_cap = self._cr_caps[cr][1]
                    a = torch.cat([torch.arange(w), torch.full((n_win_cap,), -1)]).float()
                    b = torch.cat([torch.full((w,), -1), torch.arange(n_win_cap)]).float()
                # A block wider than the axis leaves a tail of unmapped rows, which SDPA
                # still reads (it covers whole blocks). Filling A past the axis with a
                # position no step can reach makes ``A > pos`` true there, so the tail
                # is masked out however the compressor compare falls.
                pad = (self._paged_groups[lt].kv_len - a.numel()) if self._paged else 0
                if pad:
                    a = torch.cat([a, torch.full((pad,), float(max_seq))])
                    b = torch.cat([b, torch.full((pad,), -1.0)]) if b is not None else None
                a_tt = ttnn.from_torch(
                    a.reshape(1, 1, 1, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
                )
                b_tt = (
                    None
                    if b is None
                    else ttnn.from_torch(
                        b.reshape(1, 1, 1, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
                    )
                )
                sm["mask_gen"][lt] = (a_tt, b_tt, cr)
            # Submesh 0 owns global layer 0, whose per-step inputs (token + positions)
            # stream in from the host over the H2D socket into the tiny fused packet;
            # everything downstream is fed over the device-to-device sockets. A submesh
            # needs recv buffers only for the layers whose
            # *predecessor* sits on another submesh — under round-robin that is every
            # layer but global 0 (submesh 0 is revisited for layers S, 2S, ...), while
            # with a small pipeline group size a device's contiguous run of layers hands
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
                        _PKT_FIFO_BYTES,
                        ttnn.H2DMode.HOST_PUSH,
                    )
                    # ``recv_async_h2d`` cross-checks this against the packet's aligned
                    # page size on every program-cache miss.
                    self._pkt_socket.set_page_size(self._pkt_page_bytes)
            if any(li > 0 and ids[li - 1] != k for li in layers_k):
                sm["streams_in"] = _dev_zeros([1, 1, hc, d], device)
                sm["pkt_in"] = _dev_zeros([1, 1, 1, self._pkt_w], device, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            self.submeshes_io.append(sm)
        # Where the global-last layer (num_layers-1) landed: its trace produces the final
        # head output, which it streams to the host over the D2H socket below.
        self._output_sm_index = self.pipeline_submesh_ids.index(ids[self.num_layers - 1])
        self._traced_captured = False

        # The step output's return path. The page size is only known once the trace
        # builds the output tensor, so it is set on first use (see
        # :meth:`_send_output`). Created here because the socket allocates L1 on its
        # sender core, which is unsafe once a trace exists.
        if self._out_socket is None:
            self._out_socket = ttnn.D2HSocket(
                self.submeshes_io[self._output_sm_index]["device"],
                ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(*_OUT_SOCKET_CORE)),
                _OUT_FIFO_BYTES,
            )

        # Every session's held-aside compressor buffers, allocated now: once a trace
        # exists on a device, allocating buffers there can corrupt it, so
        # :meth:`open_session` may only claim one of these pre-built slots.
        self._max_sessions = num_sessions
        self._free_session_state = [self._build_session_state() for _ in range(num_sessions)]

    def _device_rope(self, inv_freq: ttnn.Tensor, scaling: float, pos_f: ttnn.Tensor) -> tuple:
        """Generate one decode step's RoPE rows on device from the absolute position.

        ``inv_freq`` ``[1,1,1,Rd]`` (FP32, interleaved-by-2) and ``scaling`` are the
        constants for one family; ``pos_f`` ``[1,1,1,1]`` (FP32) is the absolute
        position. Returns ``(cos, sin, neg_sin)`` bf16 tiles equal to the host
        ``make_rope_table`` rows. The raw angle ``pos * inv_freq`` can reach thousands
        of radians, so it is range-reduced to ``[0, 2π)`` before ``sin``/``cos`` to
        keep the device transcendentals accurate."""
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

    @staticmethod
    def _device_index(value_f: ttnn.Tensor) -> ttnn.Tensor:
        """A device-computed FP32 scalar tile ``[1,1,1,1]`` as the INT32 ``[1]`` row-major
        tensor the in-place cache writers and SDPA-decode take for an index."""
        return ttnn.reshape(ttnn.to_layout(ttnn.typecast(value_f, ttnn.int32), ttnn.ROW_MAJOR_LAYOUT), [1])

    def _device_causal_pos(self, cr: int, pos_f: ttnn.Tensor) -> ttnn.Tensor:
        """Generate one decode step's causal SDPA ``cur_pos`` on device from the
        absolute position: ``sliding_window + (pos+1)//cr - 1``, the inclusive last
        valid index on the ``[sliding | compressor]`` KV axis (the device twin of
        :func:`sdpa_causal_cur_pos`)."""
        thr = ttnn.floor(ttnn.multiply(ttnn.add(pos_f, 1.0), 1.0 / cr))  # (pos+1)//cr
        return self._device_index(ttnn.add(thr, float(self.sliding_window - 1)))

    def _device_compressor_indices(self, cr: int, pos_f: ttnn.Tensor):
        """Device twins of :func:`_window_indices`, plus the closing window's position.

        Returns ``(win_slot, win_row, win_pos_f)``: the INT32 ``[1]`` slot ``pos % cr``
        this token's projection is written to in the one-window buffer, the INT32 ``[1]``
        row ``sliding_window + w`` of the combined KV buffer the pooled entry lands in,
        and the FP32 ``[1,1,1,1]`` absolute position ``w * cr`` to RoPE that entry at,
        for the window ``w = (pos+1)//cr - 1`` closing at this step.

        A trace cannot branch on the device-side position, so all three are pure
        arithmetic on ``pos_f``. On steps that do not close a window ``w`` is one short
        (or ``-1``), but nothing consumes these then (``pool_compressor`` is ``False``).
        """
        thr = ttnn.floor(ttnn.multiply(ttnn.add(pos_f, 1.0), 1.0 / cr))  # (pos+1)//cr == w+1
        slot_f = ttnn.subtract(pos_f, ttnn.multiply(ttnn.floor(ttnn.multiply(pos_f, 1.0 / cr)), float(cr)))
        win_pos_f = ttnn.multiply(ttnn.subtract(thr, 1.0), float(cr))
        return (
            self._device_index(slot_f),
            self._device_index(ttnn.add(thr, float(self.sliding_window - 1))),
            win_pos_f,
        )

    def _decode_submesh_static(self, sm: dict, pool_flags: dict[int, bool], causal: bool) -> ttnn.Tensor:
        """Run one submesh's round-robin layers over the per-step input packets /
        in-place caches (shared by the compile run and the trace capture).

        ``pool_flags`` maps each compress rate to whether this trace variant re-pools
        that compressor; ``causal`` selects causal SDPA (bounded by an on-device
        ``cur_pos``) over the additive mask for the compressor layers. Both are fixed
        at capture time (a trace is a flat op sequence, so it cannot branch on the
        device-side position), which is why the capture emits one variant per
        (SDPA mode, window phase) pair — see :meth:`_capture_traces`.

        The dataflow follows the pipeline-group placement (:func:`plan_layer_placement`)
        layer by layer, so this method drives a recv / run / send cycle *per layer*
        rather than once per submesh:

          * The per-step inputs are ONE tiny fused INT32 packet whose first three slots
            are ``[token, pos_sliding, pos_compress]``. Global layer 0 (on submesh 0)
            receives it from the host over the H2D socket; any other layer whose
            predecessor lives on a *different* submesh receives the streams + packet
            from it.
          * Each layer splits the packet on device and generates its RoPE rows and
            additive mask from ``pos_compress``.
          * Unless it is the global-last layer, it forwards the streams + packet to the
            submesh holding the next layer — or, when that is this same submesh, simply
            hands them to the next iteration with no socket traffic. The global-last
            layer applies the head.

        So plain round-robin sends on every layer boundary (the ring), while a small
        pipeline group size makes a device's contiguous run of layers chain locally.
        """
        cfg = self.config
        k = sm["index"]
        ids = self.layer_submesh_ids
        streams = None  # carried across layers that chain locally on this submesh
        pkt = None
        out = None
        # Every position-derived tensor a step needs — the split token/positions, the
        # RoPE rows, the additive mask, the causal ``cur_pos`` and the compressor
        # window indices/RoPE — is a pure function of this step's single token and
        # position, so it is *identical* for every layer this submesh holds. Build them
        # once from the first packet the submesh sees and reuse across its layers
        # (deduped by rope family / layer type), rather than regenerating ~15 tiny
        # eltwise/typecast/slice ops per layer. On a device that owns several layers
        # this removes the bulk of the per-layer overhead ops (and their fixed per-op
        # launch cost); the tensors are read-only, so sharing them is exact.
        step_ctx: dict = {}

        def _build_step_ctx(pkt) -> dict:
            token = ttnn.typecast(
                ttnn.reshape(ttnn.slice(pkt, [0, 0, 0, 0], [1, 1, 1, 1]), [1, 1]), ttnn.uint32
            )  # [1,1]
            sliding_pos = ttnn.reshape(ttnn.slice(pkt, [0, 0, 0, 1], [1, 1, 1, 2]), [1])
            compress_pos = ttnn.reshape(ttnn.slice(pkt, [0, 0, 0, 2], [1, 1, 1, 3]), [1])
            pos_f = ttnn.typecast(
                ttnn.to_layout(ttnn.reshape(compress_pos, [1, 1, 1, 1]), ttnn.TILE_LAYOUT), ttnn.float32
            )
            # One RoPE row triple per family present on this submesh ("main" / "compress").
            rope = {rt: self._device_rope(inv, sc, pos_f) for rt, (inv, sc) in sm["rope_invfreq"].items()}
            masks: dict = {}
            curpos: dict = {}
            win_rope: dict = {}
            win_idx: dict = {}
            for lt, (a, b, cr) in sm["mask_gen"].items():
                # Bound the KV axis by a position (causal) where the valid set is a
                # contiguous prefix, else by the additive mask. Sliding layers take
                # neither here (their ring is bounded inside the attention block).
                use_causal = causal and lt != "sliding_attention"
                masks[lt] = None if use_causal else self._device_mask(a, b, cr, pos_f)
                curpos[lt] = self._device_causal_pos(cr, pos_f) if use_causal else None
                if lt != "sliding_attention":
                    # Incremental pooling emits one entry per closure, so generate just
                    # that entry's RoPE row (the "compress" family already in scope).
                    ws, wr, win_pos_f = self._device_compressor_indices(cr, pos_f)
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

            # Obtain this layer's per-step packet (and, when it arrives over a socket,
            # its input streams) — from the host buffer for global layer 0, from the
            # predecessor submesh when that layer sits elsewhere, else from the previous
            # iteration on this submesh.
            if is_first:
                # Stream this step's packet in from the host over the H2D socket. The
                # op is part of the trace, so replay needs no host-side dispatch: the
                # kernel parks on the socket until :meth:`_write_packet` pushes the
                # page (which the host may well have done already).
                pkt = sm["pkt"]
                ttnn.experimental.recv_async_h2d(pkt, self._pkt_socket)
            elif recv:
                # Receive the residual streams + fused packet from the submesh holding
                # the previous layer into the persistent buffers. Captured inside the
                # trace, so the copies need no host-side dispatch at replay. Order must
                # match the sender below.
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
                inputs_embeds = self.embed_tokens(token)  # [1, 1, D]
                bb, ss, dd = inputs_embeds.shape
                streams = ttnn.repeat(ttnn.reshape(inputs_embeds, [bb, ss, 1, dd]), ttnn.Shape([1, 1, cfg.hc_mult, 1]))
            elif recv:
                streams = sm["streams_in"]
            # else: reuse the ``streams`` carried from the prior layer on this submesh.

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
                paged=self._paged_view(sm, li),
                hash_token=token if layer.mlp.is_hash else None,
                pool_compressor=(lt != "sliding_attention" and pool_flags[cfg.compress_rates[lt]]),
                sdpa_cur_pos=sdpa_cur_pos,
                win_slot=win_slot,
                win_row=win_row,
            )

            if is_last:
                streams = self.norm(self.hc_head(streams))
                if self._lm_head_traced is not None:
                    streams = self._lm_head_traced(streams)
                out = streams
                # Stream the step's output back to the host from inside the trace, so
                # the host reads it off the socket instead of dispatching a readback
                # (see :meth:`read_decoded_output`).
                self._send_output(out)
            elif ids[li + 1] != k:
                # Send the residual streams + fused packet to the submesh holding the
                # next layer. Captured inside the trace, so dispatched on device at
                # replay (no host round-trip). Order must match the receiver above.
                sender_socket, _ = self.submesh_socket_pairs[(k, ids[li + 1])]
                ttnn.experimental.send_direct_async(streams, sender_socket)
                ttnn.experimental.send_direct_async(pkt, sender_socket)
                streams.deallocate()
                next_on_device = self._next_layer_on_submesh(li)
                if next_on_device is not None:
                    self.layers[next_on_device].self_attn.prefetch_weights()
        return out if out is not None else streams

    def _build_packet(self, token_id: int, pos: int) -> torch.Tensor:
        """Host-build the whole fused packet as one INT32 socket page
        ``[1,1,1,_pkt_w]``: ``[token, pos_sliding, pos_compress]`` then padding. The
        per-step RoPE rows and additive masks are *not* in the packet — they are
        generated on device from ``pos_compress`` (see :meth:`_device_rope` /
        :meth:`_device_mask`)."""
        w = self.sliding_window
        packet = torch.zeros(1, 1, 1, self._pkt_w, dtype=torch.int32)
        packet[0, 0, 0, : self._pkt_int_prefix] = torch.tensor([token_id, pos % w, pos], dtype=torch.int32)
        return packet

    def _write_packet(self, token_id: int, pos: int) -> None:
        """Push one step's fused packet into the H2D socket FIFO.

        This is the only host->device transfer of a traced step, and it is *not* a
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
        """Push one step's output tensor into the D2H socket (inside the trace).

        ``send_async_d2h`` streams whole pages out of a row-major tensor, so ``out`` is
        untilized and reshaped into PCIe-aligned rows first — one vocab-wide row would
        be a quarter-megabyte page for the sender kernel to stage in L1.

        The socket's page size is fixed on the first call, when the output's real shape
        and dtype are finally known; the op re-checks it against the tensor on every
        program-cache miss.
        """
        out_rm = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        if self._out_plan is None:
            numel = math.prod(tuple(out_rm.shape))
            self._out_plan = _d2h_page_plan(numel, out_rm.element_size())
            self._out_torch_dtype = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}[out_rm.dtype]
            self._out_socket.set_page_size(self._out_plan[1] * out_rm.element_size())
        ttnn.experimental.send_async_d2h(ttnn.reshape(out_rm, list(self._out_plan)), self._out_socket)

    def read_decoded_output(self) -> torch.Tensor:
        """Stage 3 of a step: read its output off the D2H socket, as ``[1, 1, N]``.

        Returns the oldest in-flight step's output.

        Blocks until the device has pushed every page of that step, so this is where a
        traced step synchronizes. Outputs are read in the order they were dispatched,
        and each read returns its own buffer, so several steps may be in flight at once
        (see :meth:`decode_traced_async`).
        """
        rows, cols = self._out_plan
        out = torch.empty(rows, cols, dtype=self._out_torch_dtype)
        self._out_socket.read_tensor(out)
        return out.reshape(1, 1, rows * cols)

    def _capture_traces(self, token_id: int, pos: int) -> None:
        """Capture the decode traces: per submesh, one variant per (SDPA mode, phase).

        ``token_id`` / ``pos`` describe the step about to be replayed; each compile run
        below is fed that packet over the H2D socket (see :meth:`_write_packet`).

        A ttnn trace is a flat, fixed sequence of device ops, so it cannot skip the
        compressor pool on the steps that do not close a window, nor switch between
        causal and masked SDPA — the position both would have to branch on only exists
        as a device tensor at replay. But the *choice of trace* is a host-side decision
        (``decode_traced`` knows ``pos`` before it dispatches), so both are baked into
        the capture instead: one variant per entry of :meth:`_reachable_variants`,
        selected per step by :meth:`_variant_key`. With the default rates that is five
        (three causal phases — pool nothing / CSA / CSA+HCA — plus the two masked
        phases reachable below the sliding window).

        Variants are deduplicated per submesh via :meth:`_sm_pool_key`: a submesh only
        distinguishes what its own layers observe, so a sliding-only submesh is
        captured once and replays that single trace for every variant.

        Ordering matters twice over. *Every* variant's compile run (which JITs the
        programs — trace capture itself cannot) has to be issued before the *first*
        capture: once a trace exists on a device, allocating device buffers on it is
        unsafe ("these buffers may be corrupted once a trace is executed"), and a
        compile run allocates freely. So the two passes below are not interleaved.

        Each submesh is captured independently — capture only fixes program shapes
        / buffer addresses, so the (stale) compile-run inputs are immaterial: any
        cache rows the compile run writes are at the *same* device-indexed slots a
        later replay overwrites with real values. The real per-step results always
        come from the :meth:`decode_traced` replay loop, never the capture run.

        The compile runs are issued for *all* submeshes before synchronizing,
        because each submesh's slice now contains the cross-submesh socket
        send/recv: a lone ``send_async`` followed by a blocking per-submesh
        ``synchronize_device`` would deadlock (the residual streams exceed the
        socket's L1 buffer, so the send cannot drain until the next submesh posts
        its matching ``recv_async``). Issuing every submesh first lets the sends
        and receives pair up across devices, after which a single sync drains
        them. Trace capture only records ops (it does not execute them), so the
        capture loop is free of this hazard.
        """
        # Plan first: which submeshes need their own capture for each phase, and
        # which just alias an earlier phase's trace.
        plan = []
        planned_keys: list[set] = [set() for _ in self.submeshes_io]
        for variant in self._reachable_variants():
            causal, phase_idx = variant
            flags = dict(zip(self._pool_crs, self._pool_phases[phase_idx]))
            pending = []
            for i, sm in enumerate(self.submeshes_io):
                key = self._sm_pool_key(sm, flags, causal)
                if key not in planned_keys[i]:
                    planned_keys[i].add(key)
                    pending.append(sm)
            plan.append((variant, flags, pending))

        # Pass 1 — every compile run, while no trace exists yet. The run is issued
        # for *all* submeshes, not just the pending ones: the slices contain the
        # cross-submesh socket send/recv, so a submesh sitting the round out would
        # leave its neighbours' sends unpaired. Re-running an already-planned
        # submesh is harmless (the cache rows it dirties are the same
        # device-indexed slots a later replay overwrites, and they stay
        # block-bias-masked until then).
        for variant, flags, pending in plan:
            if not pending:
                continue
            causal, phase_idx = variant
            compile_outs = []
            # The compile run *executes*, so submesh 0's in-trace ``recv_async_h2d``
            # would park forever without a page of its own. Every round gets the
            # upcoming step's packet, which is what the runs saw when the packet was
            # copied to device around the traces rather than received inside them.
            self._write_packet(token_id, pos)
            for sm in self.submeshes_io:
                logger.info(
                    f"[traced-decode] compiling submesh {sm['index']} "
                    f"({len(sm['layers'])} layers) phase {phase_idx} pool={flags} causal={causal}"
                )
                compile_outs.append(self._decode_submesh_static(sm, flags, causal))  # JITs the programs
            # The run also *sends* an output, so drain it: an unread output would sit in
            # the socket FIFO and eventually backpressure the sender kernel. Discarded —
            # the real per-step outputs all come from the replay loop.
            self.read_decoded_output()
            for out in compile_outs:
                out.deallocate(True)

        # Pass 2 — record the captures and bind every variant to a trace.
        for variant, flags, pending in plan:
            causal, phase_idx = variant
            for sm in pending:
                device = sm["device"]
                logger.info(
                    f"[traced-decode] capturing submesh {sm['index']} "
                    f"({len(sm['layers'])} layers) phase {phase_idx} pool={flags} causal={causal}"
                )
                tid = ttnn.begin_trace_capture(device, cq_id=0)
                with _trace_capture_guard():
                    out = self._decode_submesh_static(sm, flags, causal)
                ttnn.end_trace_capture(device, tid, cq_id=0)
                # ``out`` is persistent; overwritten in place by every execute_trace.
                sm["traces"][self._sm_pool_key(sm, flags, causal)] = (tid, out)
            for sm in self.submeshes_io:
                sm["tids"][variant], sm["outputs"][variant] = sm["traces"][self._sm_pool_key(sm, flags, causal)]
        self._traced_captured = True

    def decode_traced(self, token_id: int, pos: int) -> torch.Tensor:
        """One traced decode step: feed ``token_id`` at absolute position ``pos`` and
        return the step's output, read back to host.

        Requires a prior :meth:`prepare_static_decode`. Equivalent to
        :meth:`decode_traced_async` followed by :meth:`read_decoded_output`, i.e. it
        blocks until the output has arrived. The result is ``[1, 1, vocab]`` logits if
        an ``lm_head`` was passed to :meth:`prepare_static_decode`, else the pre-head
        hidden ``[1, 1, hidden]``; its dtype is the device dtype (bf16), so cast before
        doing host math on it.
        """
        self.decode_traced_async(token_id, pos)
        return self.read_decoded_output()

    def decode_traced_async(self, token_id: int, pos: int) -> None:
        """Dispatch one traced decode step without waiting for its output.

        Captures the per-submesh traces lazily on the first call, then (every call)
        pushes this step's input packet onto the H2D socket and replays each submesh's
        trace in order. The packet receive, the residual-stream handoffs between
        submeshes and the output send all happen from *inside* the traces, so a step
        dispatches no device ops from the host at all — the only host work is the two
        socket transfers.

        Nothing is returned: the output is in flight to the host, to be picked up by
        :meth:`read_decoded_output`. Every dispatched step must be read back exactly
        once, in dispatch order.

        In paged mode the step belongs to whichever session is active (see
        :meth:`activate_session`), and its blocks are grown here as the compressor
        windows close.
        """
        if self._paged is not None:
            self.ensure_session_capacity(pos)
        # Capture first: the compile runs inside consume a packet each, so pushing this
        # step's packet before them would hand it to a compile run instead of to the
        # replay below.
        if not self._traced_captured:
            self._capture_traces(token_id, pos)
        self.write_step_packet(token_id, pos)
        self.replay_traced(pos)

    # A step's three host-side stages, each split out so a pipelined caller can drive
    # them independently — they touch disjoint state, so they can run concurrently (on
    # separate threads) for different steps: push step n+1's packet while step n's
    # traces replay and step n-1's output is read back.
    def write_step_packet(self, token_id: int, pos: int) -> None:
        """Stage 1 of a step: push its input packet to the device.

        Talks only to the H2D socket (a direct PCIe write, no command queue work), and
        may run ahead of the replays by as much as the socket FIFO holds.
        """
        self._write_packet(token_id, pos)

    def replay_traced(self, pos: int) -> None:
        """Stage 2 of a step: replay the traces for a step at ``pos``.

        The only stage that touches the command queue (and, in paged mode, the session
        state), so a pipelined caller must keep it on a single thread. Expects the
        step's packet from :meth:`write_step_packet` — pushed before or after this call,
        since the trace's receive just waits for it — and its output to be picked up by
        :meth:`read_decoded_output`.

        Requires the traces to already be captured, which a pipelined caller cannot do
        mid-flight: dispatch one blocking :meth:`decode_traced` first.
        """
        if not self._traced_captured:
            raise RuntimeError("call decode_traced() once to capture the traces before replay_traced()")
        if self._paged is not None:
            self.ensure_session_capacity(pos)
        # Pick the variant whose baked-in compressor-pool schedule and SDPA mode match
        # this position (see :meth:`_capture_traces`).
        variant = self._variant_key(pos)
        for sm in self.submeshes_io:
            ttnn.execute_trace(sm["device"], sm["tids"][variant], cq_id=0, blocking=False)

    def decode_sampled_burst(self, first_token_id: int, start_pos: int, n_steps: int) -> list[int]:
        """Unsupported while the per-step packet arrives over the H2D socket.

        This used to decode ``n_steps`` tokens with greedy sampling done on device,
        re-injecting each sampled id into idx 0 of submesh 0's fused ``pkt`` buffer
        between replays. That feedback was an *eager* write into ``pkt``, which the
        in-trace ``recv_async_h2d`` at the head of every submesh-0 trace now
        overwrites with the host's packet — so the sampled token would never reach
        ``embed_tokens``.

        Restoring it means splitting the packet: keep the host-fed positions on the
        socket and read the token from a separate device-written buffer.
        """
        raise NotImplementedError(
            "on-device sampled bursts are incompatible with the in-trace H2D packet "
            "receive (the socket packet overwrites the device-sampled token slot); "
            "use decode_traced() with host-side sampling"
        )
