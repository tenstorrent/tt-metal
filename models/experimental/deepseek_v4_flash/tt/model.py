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
from .common import DeepSeekV4Module, _MASK_NEG, _profile, _trace_capture_guard
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

        # Layers are distributed across submeshes round-robin: layer ``li`` lives on
        # submesh ``li % pipeline_stages`` (layer 0 -> submesh 0, layer 1 -> submesh 1,
        # ... wrapping back to submesh 0). ``pipeline_stages`` is the number of
        # submeshes actually populated (capped at the layer count for tiny stacks).
        # The decode dataflow therefore forms a *ring* 0 -> 1 -> ... -> (S-1) -> 0,
        # traversed once per round-robin "round".
        n = config.num_hidden_layers if max_layers is None else min(max_layers, config.num_hidden_layers)
        self.num_layers = n
        self.pipeline_stages = min(self.num_layers, self.num_submeshes) if use_submeshes else 1

        if use_submeshes:
            logger.info(f"Using submeshes: {self.num_submeshes}")
            full_device.reshape(ttnn.MeshShape(1, full_device.get_num_devices()))
            self.submeshes = []
            for i in range(self.num_submeshes):
                self.submeshes.append(full_device.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, i)))
            self.first_device = self.submeshes[0]
            self.last_device = self.submeshes[-1]

            # Create socket pairs between submeshes for copying hidden_states .
            # The round-robin layer placement makes the decode dataflow a *ring*:
            # submesh k hands off to submesh (k+1) % pipeline_stages, including the
            # wrap-around edge (S-1 -> 0). One directed pair per ring edge, reused for
            # all forward passes.
            self.submesh_socket_pairs = {}
            socket_memconfig = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 16 * 1024)
            ring_edges = (
                [(k, (k + 1) % self.pipeline_stages) for k in range(self.pipeline_stages)]
                if self.pipeline_stages > 1
                else []
            )
            for from_id, to_id in ring_edges:
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
            self._thunk("norm.weight"), config.rms_norm_eps, self.last_device, cache.file("norm")
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
        """Round-robin layer -> submesh mapping: layer ``li`` lives on submesh
        ``li % pipeline_stages`` (layer 0 -> 0, 1 -> 1, ..., wrapping back to 0)."""
        return layer_idx % self.pipeline_stages

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
            next_layer_for_device_id = li + self.pipeline_stages
            if next_layer_for_device_id < self.num_layers:
                assert (
                    self._submesh_id_for_layer(next_layer_for_device_id) == current_submesh_id
                ), "Next layer is not on the same submesh"
                next_layer = self.layers[next_layer_for_device_id]
                next_layer.self_attn.prefetch_weights()
        return self.norm(self.hc_head(streams))

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
        additionally gets the only host-written per-step state: the position / token
        scalars and the float-bits region. ``max_seq`` must be a multiple of every compress-rate (the caller
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
        num_sm = self.pipeline_stages

        # --- Canonical per-step input packet layout (shared by every submesh) --- #
        # All per-step inputs are fused into ONE tiny fixed-shape INT32 packet
        # ``[1, 1, 1, 3]`` (ROW_MAJOR), a single persistent buffer on submesh 0
        # written from host *only* there and then flowed downstream over the existing
        # socket (see :meth:`_decode_submesh_static`), so no submesh past the first
        # needs a per-step host->device write.
        #
        #   idx 0 : token (INT32; embedding/hash use it typecast to uint32). Placed
        #           first so the on-device sampling loop can re-inject the sampled id
        #           by slicing off idx 0 and re-concatenating (see
        #           :meth:`decode_sampled_burst`).
        #   idx 1 : pos_sliding (INT32)
        #   idx 2 : pos_compress (INT32)
        #
        # The per-step RoPE rows and additive masks are *not* in the packet — they are
        # both generated on device from ``pos_compress`` against constant tables (see
        # :meth:`_device_rope` and :meth:`_device_mask`).
        self._pkt_int_prefix = 3  # [token, pos_sliding, pos_compress]
        self._pkt_w = self._pkt_int_prefix
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
        for k in range(num_sm):
            device = self.submeshes[k]
            layers_k = [li for li in range(self.num_layers) if self._submesh_id_for_layer(li) == k]
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
            # Submesh 0 owns global layer 0, whose per-step inputs are host-written into
            # the tiny fused packet (token + positions); everything downstream is fed
            # over the ring sockets. Under round-robin, submesh 0 *also* revisits the ring
            # for its later layers (8, 16, ...), so it needs recv buffers too. In general
            # every submesh that runs any layer other than global layer 0 receives the
            # residual streams + packet from its ring predecessor.
            if 0 in layers_k:
                sm["pkt"] = _dev_zeros([1, 1, 1, self._pkt_w], device, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            if any(li != 0 for li in layers_k):
                sm["streams_in"] = _dev_zeros([1, 1, hc, d], device)
                sm["pkt_in"] = _dev_zeros([1, 1, 1, self._pkt_w], device, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            self.submeshes_io.append(sm)
        # The global-last layer (num_layers-1) lands on this submesh under round-robin;
        # its trace produces the final head output consumed by :meth:`decode_traced`.
        self._output_sm_index = (self.num_layers - 1) % num_sm
        self._traced_captured = False

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

        Under round-robin placement the decode dataflow is a *ring*: consecutive
        global layers live on consecutive submeshes, so a submesh's layers are *not*
        contiguous in the global order — between two layers on the same submesh the
        residual streams travel all the way around the ring. This method therefore
        drives one recv / run / send cycle *per layer* rather than once per submesh:

          * The per-step inputs are ONE tiny fused INT32 packet ``[1,1,1,3]`` =
            ``[token, pos_sliding, pos_compress]``. Global layer 0 (on submesh 0)
            reads submesh 0's host-written packet; every other layer receives the
            streams + packet from its ring predecessor ``(k-1) % S``.
          * Each layer splits the packet on device and generates its RoPE rows and
            additive mask from ``pos_compress``.
          * Unless it is the global-last layer, it forwards the streams + packet to
            its ring successor ``(k+1) % S``. The global-last layer applies the head.

        For a single-stage stack (``pipeline_stages == 1``) there is no ring: the
        streams simply chain layer-to-layer on submesh 0 with no socket traffic.
        """
        cfg = self.config
        k = sm["index"]
        multi = self.pipeline_stages > 1
        prev_k = (k - 1) % self.pipeline_stages
        next_k = (k + 1) % self.pipeline_stages
        streams = None  # carried across layers only in the single-stage case
        out = None
        for li in sm["layers"]:
            layer = self.layers[li]
            is_first = li == 0
            is_last = li == self.num_layers - 1

            # Obtain this layer's per-step packet (and, when multi-stage, its input
            # streams) — from the host buffer for global layer 0, else over the ring.
            if is_first or not multi:
                pkt = sm["pkt"]
            else:
                # Receive the residual streams + fused packet from the ring
                # predecessor into the persistent buffers. Captured inside the trace,
                # so the copies need no host-side dispatch at replay. Order must match
                # the sender below.
                _, receiver_socket = self.submesh_socket_pairs[(prev_k, k)]
                ttnn.experimental.recv_direct_async(sm["streams_in"], receiver_socket)
                ttnn.experimental.recv_direct_async(sm["pkt_in"], receiver_socket)
                pkt = sm["pkt_in"]

            # Split the packet -> token, sliding position [1], compress position [1].
            token = ttnn.typecast(
                ttnn.reshape(ttnn.slice(pkt, [0, 0, 0, 0], [1, 1, 1, 1]), [1, 1]), ttnn.uint32
            )  # [1,1]
            sliding_pos = ttnn.reshape(ttnn.slice(pkt, [0, 0, 0, 1], [1, 1, 1, 2]), [1])
            compress_pos = ttnn.reshape(ttnn.slice(pkt, [0, 0, 0, 2], [1, 1, 1, 3]), [1])

            # Generate this layer's RoPE rows and additive mask on device from the
            # absolute position (nothing position-dependent is shipped in the packet).
            pos_f = ttnn.typecast(
                ttnn.to_layout(ttnn.reshape(compress_pos, [1, 1, 1, 1]), ttnn.TILE_LAYOUT), ttnn.float32
            )
            lt = cfg.layer_types[li]
            rope_type = "main" if lt == "sliding_attention" else "compress"
            inv_freq, scaling = sm["rope_invfreq"][rope_type]
            cos, sin, neg_sin = self._device_rope(inv_freq, scaling, pos_f)
            a, b, cr = sm["mask_gen"][lt]
            # Bound the KV axis by a position (causal) where this step's valid set is
            # a contiguous prefix, else by the additive mask. Sliding layers take
            # neither here: their ring is bounded by the absolute position inside
            # :meth:`DeepSeekV4Attention.decode_static`.
            use_causal = causal and lt != "sliding_attention"
            mask = None if use_causal else self._device_mask(a, b, cr, pos_f)
            sdpa_cur_pos = self._device_causal_pos(cr, pos_f) if use_causal else None
            if lt == "sliding_attention":
                cos_win = sin_win = win_slot = win_row = None
            else:
                # Incremental pooling emits one entry per closure, so instead of a
                # ``max_seq``-wide window table we generate just that entry's RoPE row.
                # ``rope["win"][cr]`` row w is the "compress" family at ``w * cr``, which
                # is the same generator (and inv_freq) already in scope for this layer.
                win_slot, win_row, win_pos_f = self._device_compressor_indices(cr, pos_f)
                cos_win, sin_win, _ = self._device_rope(inv_freq, scaling, win_pos_f)

            if is_first:
                inputs_embeds = self.embed_tokens(token)  # [1, 1, D]
                bb, ss, dd = inputs_embeds.shape
                streams = ttnn.repeat(ttnn.reshape(inputs_embeds, [bb, ss, 1, dd]), ttnn.Shape([1, 1, cfg.hc_mult, 1]))
            elif multi:
                streams = sm["streams_in"]
            # else (single-stage, li>0): reuse the ``streams`` carried from the prior layer.

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
            elif multi:
                # Send the residual streams + fused packet to the ring successor.
                # Captured inside the trace, so dispatched on device at replay (no
                # host round-trip). Order must match the receiver above.
                sender_socket, _ = self.submesh_socket_pairs[(k, next_k)]
                ttnn.experimental.send_direct_async(streams, sender_socket)
                ttnn.experimental.send_direct_async(pkt, sender_socket)
                streams.deallocate()
                next_layer_for_device_id = li + self.pipeline_stages
                if next_layer_for_device_id < self.num_layers:
                    next_layer = self.layers[next_layer_for_device_id]
                    next_layer.self_attn.prefetch_weights()
        return out if out is not None else streams

    def _build_packet(self, token_id: int, pos: int) -> ttnn.Tensor:
        """Host-build the whole fused packet ``[1,1,1,3]`` as INT32:
        ``[token, pos_sliding, pos_compress]``. The per-step RoPE rows and additive
        masks are *not* in the packet — they are generated on device from
        ``pos_compress`` (see :meth:`_device_rope` / :meth:`_device_mask`)."""
        w = self.sliding_window
        packet = torch.tensor([[[[token_id, pos % w, pos]]]], dtype=torch.int32)  # [1,1,1,3]
        return ttnn.from_torch(packet, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    def _set_step_position_inputs(self, pos: int) -> None:
        """Refresh the *position-dependent* per-step inputs (RoPE rows, masks, cache
        positions) on submesh 0 *only* by rewriting the whole fused packet with a
        placeholder token (``0`` at idx 0). They depend only on ``pos`` (a host-side
        counter), never on a device readback, and flow downstream over the socket.
        The on-device sampling loop overwrites the placeholder token with the sampled
        id (see :meth:`decode_sampled_burst`), so the loop never stalls on device."""
        ttnn.copy_host_to_device_tensor(self._build_packet(0, pos), self.submeshes_io[0]["pkt"])

    def _set_step_inputs(self, token_id: int, pos: int) -> None:
        """Write all per-step inputs (token id, RoPE rows, masks, cache positions) as
        the single fused packet onto submesh 0 *only* (allocation-free on device, so
        it is safe to interleave with ``execute_trace``). The packet is propagated to
        the rest of the stack over the socket (so hash-MoE layers on any submesh see
        the token)."""
        ttnn.copy_host_to_device_tensor(self._build_packet(token_id, pos), self.submeshes_io[0]["pkt"])

    def _capture_traces(self) -> None:
        """Capture the decode traces: per submesh, one variant per (SDPA mode, phase).

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
            for sm in self.submeshes_io:
                logger.info(
                    f"[traced-decode] compiling submesh {sm['index']} "
                    f"({len(sm['layers'])} layers) phase {phase_idx} pool={flags} causal={causal}"
                )
                compile_outs.append(self._decode_submesh_static(sm, flags, causal))  # JITs the programs
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

    def decode_traced(self, token_id: int, pos: int) -> ttnn.Tensor:
        """One traced decode step: feed ``token_id`` at absolute position ``pos``.

        Requires a prior :meth:`prepare_static_decode`. Captures
        the per-submesh traces lazily on the first call, then (every call) refreshes
        the per-step inputs and replays each submesh's trace in order. The residual
        streams are socket-copied between submeshes from *inside* each trace
        (device-to-device, no host hop and no per-step host op dispatch).
        Returns the persistent output tensor of the submesh holding the global-last
        layer — logits ``[1,1,vocab]`` if an ``lm_head`` was passed to
        :meth:`prepare_static_decode`, else the pre-head hidden ``[1, 1, 1, hidden]``.

        The returned tensor is overwritten by the next call, so consume it (e.g.
        ``ttnn.to_torch``) before decoding the following token.

        In paged mode the step belongs to whichever session is active (see
        :meth:`activate_session`), and its blocks are grown here as the compressor
        windows close.
        """
        if self._paged is not None:
            self.ensure_session_capacity(pos)
        self._set_step_inputs(token_id, pos)
        if not self._traced_captured:
            self._capture_traces()
        # Pick the variant whose baked-in compressor-pool schedule and SDPA mode match
        # this position (see :meth:`_capture_traces`).
        variant = self._variant_key(pos)
        for sm in self.submeshes_io:
            ttnn.execute_trace(sm["device"], sm["tids"][variant], cq_id=0, blocking=False)
        # Under round-robin the global-last layer (and thus the head output) does not
        # land on the last submesh in the list, but on ``(num_layers-1) % S``.
        return self.submeshes_io[self._output_sm_index]["outputs"][variant]

    def decode_sampled_burst(self, first_token_id: int, start_pos: int, n_steps: int) -> list[int]:
        """Autoregressively decode ``n_steps`` tokens with greedy (top-1) sampling
        done *on device*, feeding each sampled token back into the next step without
        a device->host round trip, then return all ``n_steps`` token ids in a single
        host transfer.

        Per step (all enqueued on cq0, so ordered without an explicit sync):
        replay each submesh trace -> ``argmax`` the last submesh's logits -> re-inject
        the sampled id into idx 0 of the first submesh's fused ``pkt`` buffer (the one
        ``embed_tokens`` reads). Only the position-dependent inputs (RoPE rows / masks
        / cache positions) are refreshed from the host each step; none of that reads
        back from device, so the loop never stalls on the device.

        Greedy feedback is fully on-device only when the model lives on a single
        submesh (the sampled id and ``pkt`` share a device). Hash-MoE layers are
        supported on device: they gather their expert mask from the packet token with
        :func:`ttnn.embedding`, which the on-device feedback already refreshes.
        """
        if not self.use_submeshes:
            raise NotImplementedError("traced sampling requires use_submeshes=True")
        sm0, sm_last = self.submeshes_io[0], self.submeshes_io[self._output_sm_index]
        if sm0["device"] != sm_last["device"]:
            raise NotImplementedError(
                "on-device sampling feedback currently requires a single submesh "
                "(sampled id and token_in must share a device)"
            )

        if self._paged is not None:
            self.ensure_session_capacity(start_pos)
        self._set_step_inputs(first_token_id, start_pos)
        if not self._traced_captured:
            self._capture_traces()

        pkt = sm0["pkt"]
        w = self._pkt_w
        sampled: list[ttnn.Tensor] = []
        tok_i32: ttnn.Tensor | None = None
        for i in range(n_steps):
            if i > 0:
                if self._paged is not None:
                    self.ensure_session_capacity(start_pos + i)
                # Refresh positions / RoPE / masks (token slot reset to a placeholder)
                # then re-inject the previous step's device-sampled id into idx 0 of
                # the fused packet: slice off the placeholder token and re-concatenate
                # the real one, copied back in place. All eager (outside the trace).
                self._set_step_position_inputs(start_pos + i)
                rest = ttnn.slice(pkt, [0, 0, 0, 1], [1, 1, 1, w])  # everything past the token slot
                fused = ttnn.concat([ttnn.reshape(tok_i32, ttnn.Shape([1, 1, 1, 1])), rest], dim=-1)
                ttnn.copy(fused, pkt)
            variant = self._variant_key(start_pos + i)
            for sm in self.submeshes_io:
                ttnn.execute_trace(sm["device"], sm["tids"][variant], cq_id=0, blocking=False)
            logits_rm = ttnn.to_layout(sm_last["outputs"][variant], ttnn.ROW_MAJOR_LAYOUT)  # [1, 1, vocab]
            tok = ttnn.argmax(logits_rm, dim=-1, keepdim=True)  # [1, 1, 1]
            sampled.append(
                ttnn.reshape(tok if tok.dtype == ttnn.uint32 else ttnn.typecast(tok, ttnn.uint32), ttnn.Shape([1, 1]))
            )
            tok_i32 = tok if tok.dtype == ttnn.int32 else ttnn.typecast(tok, ttnn.int32)

        # One-shot readback: concat all sampled ids and transfer once.
        all_toks = ttnn.concat(sampled, dim=0)  # [n_steps, 1]
        return ttnn.to_torch(all_toks).reshape(-1).to(torch.int64).tolist()
