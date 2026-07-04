from typing import Optional

import torch
import ttnn
from loguru import logger

from .attention import _LayerKVCache, _StaticLayerCache, make_rope_table
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
        if use_submeshes:
            logger.info(f"Using submeshes: {self.num_submeshes}")
            full_device.reshape(ttnn.MeshShape(1, full_device.get_num_devices()))
            self.submeshes = []
            for i in range(self.num_submeshes):
                self.submeshes.append(full_device.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, i)))
            self.first_device = self.submeshes[0]
            self.last_device = self.submeshes[-2]

            # Create socket pairs between submeshes for copying hidden_states .
            # One pair per (from_id, to_id) with from_id != to_id; reused for all forward passes.
            num_submeshes = full_device.get_num_devices()
            self.submesh_socket_pairs = {}
            socket_memconfig = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 16 * 1024)
            for from_id in range(num_submeshes - 1):
                to_id = from_id + 1
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

        n = config.num_hidden_layers if max_layers is None else min(max_layers, config.num_hidden_layers)
        self.num_layers = n

        self.embed_tokens = DeepSeekV4Embedding(loader, self.first_device, cache=cache)

        self.layers: list[DeepSeekV4DecoderLayer] = []
        self.layer_devices: list[ttnn.MeshDevice] = []
        # 43 layers across 7 devices (ceil(43/7) = 7 -> submeshes 0..6), leaving the
        # 8th submesh free for the DSpark speculative module (see ``dspark_device``).
        self.layers_per_device = 7
        for li in range(n):
            if self.use_submeshes:
                layer_device_id = li // self.layers_per_device
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

        # Reserve a submesh not occupied by any decoder layer for the DSpark
        # speculative module. With ``layers_per_device == 7`` the 43-layer stack
        # ends on submesh 6, so submesh 7 is free; fall back to the last submesh
        # (shared) if a larger stack / smaller mesh leaves none spare.
        self.dspark_device = None
        if self.use_submeshes:
            used = {li // self.layers_per_device for li in range(n)}
            free = [i for i in range(self.num_submeshes) if i not in used]
            self.dspark_device = self.submeshes[free[0]] if free else self.submeshes[-1]

        # Per-layer decode state (sliding K=V + optional compressor projections).
        self.sliding_window = config.sliding_window
        self.kv_caches: list[_LayerKVCache] = self._new_caches()

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
            this_device = self.submeshes[layer_idx // self.layers_per_device]
        else:
            this_device = self.first_device
        return DeepSeekV4HashRouter(self.config, weights, this_device)

    # -- decode KV-cache state -------------------------------------------------- #
    def _new_caches(self) -> list["_LayerKVCache"]:
        return [
            _LayerKVCache(self.sliding_window, self.config.layer_types[li] != "sliding_attention")
            for li in range(self.num_layers)
        ]

    def reset_caches(self) -> None:
        """Drop all per-layer decode state (call before decoding a fresh sequence)."""
        self.kv_caches = self._new_caches()

    # -- per-layer RoPE tables / masks ------------------------------------------ #
    def _to_tt(self, t: torch.Tensor, device: ttnn.MeshDevice) -> ttnn.Tensor:
        _profile(device)

        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def _rope_rows_decode(
        self, rope: dict, pos: int, layer_type: str, compress_rate: Optional[int], cache: dict, device: ttnn.MeshDevice
    ):
        """Single-position RoPE rows for a decode step.

        Returns ``(cos, sin, neg_sin, cos_win, sin_win)`` where ``cos/sin/neg_sin``
        are the one ``[1,1,1,Rd]`` rows at absolute position ``pos`` and the window
        tables cover the ``(pos + 1) // compress_rate`` currently-emittable windows
        (``None`` when none / a sliding layer).
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
            n_win = (pos + 1) // compress_rate
            if n_win > 0:
                cw_h, sw_h = rope["win"][compress_rate]
                cw, sw = make_rope_table(cw_h[:n_win], sw_h[:n_win])
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

    def _socket_move(self, tensor, from_submesh_id: int, to_submesh_id: int):
        """Move ``tensor`` up the adjacent-submesh socket chain (device-to-device).

        The pre-built socket pairs only connect ``(k, k+1)``, so a move to a higher
        submesh hops one link at a time. ``from_submesh_id <= to_submesh_id`` (the
        DSpark tap layers and its device are never below their source)."""
        for k in range(from_submesh_id, to_submesh_id):
            tensor = self._copy_streams_between_submeshes(tensor, k, k + 1)
        return tensor

    def to_dspark_device(self, tensor):
        """Socket-transfer a ``last_device`` tensor onto the reserved DSpark submesh
        (one adjacent hop, ``last -> dspark``), avoiding a host round-trip."""
        last_id = self.num_layers and ((self.num_layers - 1) // self.layers_per_device)
        dspark_id = next(i for i in range(self.num_submeshes) if self.submeshes[i] is self.dspark_device)
        return self._socket_move(tensor, last_id, dspark_id)

    def decode_main_hidden(self, token_id: int, pos: int, rope: dict) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        """Like :meth:`decode` but also returns the DSpark ``main_hidden`` tap.

        ``main_hidden`` is the concatenation (over the feature axis) of the
        residual-stream stack's mean over the ``hc_mult`` axis at each layer in
        ``config.dspark_target_layer_ids`` — ``[B, 1, len(target)*hidden]`` — the
        input the DSpark speculative module consumes. Returns ``None`` for the tap
        when no target layer is within the (possibly capped) stack.
        """
        target = set(getattr(self.config, "dspark_target_layer_ids", []) or [])
        return self._decode_impl(token_id, pos, rope, collect_targets=target)

    def decode(self, token_id: int, pos: int, rope: dict) -> ttnn.Tensor:
        hidden, _ = self._decode_impl(token_id, pos, rope, collect_targets=set())
        return hidden

    def _decode_impl(
        self, token_id: int, pos: int, rope: dict, collect_targets: set
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
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
        target_means: list[ttnn.Tensor] = []
        for li, layer in enumerate(self.layers):
            if self.use_submeshes:
                current_submesh_id = li // self.layers_per_device
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
            streams = layer.decode(
                streams,
                cos_tt,
                sin_tt,
                neg_sin_tt,
                cos_win_tt,
                sin_win_tt,
                self.kv_caches[li],
                input_ids=ids,
            )
            if li in collect_targets:
                # DSpark tap: mean over the hc_mult stream axis -> [B, 1, 1, D],
                # tagged with the submesh it currently lives on (target layers can
                # straddle submeshes, e.g. 40/41 on submesh 5, 42 on submesh 6).
                sm_id = current_submesh_id if self.use_submeshes else 0
                target_means.append((sm_id, ttnn.mean(streams, dim=2, keepdim=True)))
            last_submesh_id = current_submesh_id
            _profile(this_device)
        main_hidden = None
        if target_means:
            # Gather every tap onto the last layer's submesh over the device sockets
            # (no host round-trip), then concat on-device along the feature axis.
            dest_id = last_submesh_id if self.use_submeshes else 0
            gathered = [self._socket_move(t, sm_id, dest_id) for sm_id, t in target_means]
            main_hidden = ttnn.concat(gathered, dim=-1) if len(gathered) > 1 else gathered[0]
        return self.norm(self.hc_head(streams)), main_hidden

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
        """Allocate a layer's fixed-size in-place caches *empty* (all-zero).

        There is no prefill: the prompt is fed one token at a time through
        :meth:`decode_traced`, which writes each token's K=V / compressor
        projection into these buffers in place at its absolute position. Unwritten
        ring slots / windows stay zero and are dropped by the per-step decode mask.
        """
        dh = self.config.head_dim
        w = self.sliding_window
        sliding = ttnn.from_torch(
            torch.zeros(1, 1, w, dh),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ckv = cgate = None
        layer_type = self.config.layer_types[li]
        if layer_type != "sliding_attention":
            cr = self.config.compress_rates[layer_type]
            cap = self._cr_caps[cr][0]
            feat = (2 if layer_type == "compressed_sparse_attention" else 1) * dh
            ckv = ttnn.from_torch(
                torch.zeros(1, 1, cap, feat),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            cgate = ttnn.from_torch(
                torch.zeros(1, 1, cap, feat),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return _StaticLayerCache(sliding, ckv, cgate)

    def prepare_static_decode(self, rope: dict, max_seq: int, lm_head=None) -> None:
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
        """
        if not self.use_submeshes:
            raise NotImplementedError("traced decode requires use_submeshes=True")
        cfg = self.config
        for cr in {cfg.compress_rates[t] for t in cfg.layer_types[: self.num_layers] if t != "sliding_attention"}:
            assert max_seq % cr == 0, f"max_seq ({max_seq}) must be a multiple of compress_rate {cr}"
        self._traced_rope = rope
        self._lm_head_traced = lm_head
        self._cr_caps = {
            cr: (max_seq, max_seq // cr)
            for cr in {cfg.compress_rates[t] for t in cfg.layer_types[: self.num_layers] if t != "sliding_attention"}
        }

        rd = cfg.qk_rope_head_dim
        hc, d, w = cfg.hc_mult, cfg.hidden_size, self.sliding_window
        num_sm = (self.num_layers + self.layers_per_device - 1) // self.layers_per_device

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
            layers_k = [li for li in range(self.num_layers) if li // self.layers_per_device == k]
            types = {cfg.layer_types[li] for li in layers_k}
            crs = {cfg.compress_rates[t] for t in types if t != "sliding_attention"}
            sm = {
                "device": device,
                "index": k,
                "layers": layers_k,
                "first": k == 0,
                "last": layers_k and layers_k[-1] == self.num_layers - 1,
                "win_rope": {},
                "rope_invfreq": {},
                "mask_gen": {},
                "scaches": {li: self._build_static_layer_cache(li, device) for li in layers_k},
                "tid": None,
                "output": None,
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
            for cr in crs:
                n_win_cap = self._cr_caps[cr][1]
                cw, sw = make_rope_table(rope["win"][cr][0][:n_win_cap], rope["win"][cr][1][:n_win_cap])
                # Store the compressor RoPE tables DRAM-interleaved -- the layout the fused
                # ``_apply_rope`` op consumes for cos/sin (only X is sharded). The static path
                # uses the full ``n_win_cap`` rows, matching the compressed input rows.
                sm["win_rope"][cr] = (
                    ttnn.from_torch(
                        cw,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    ),
                    ttnn.from_torch(
                        sw,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    ),
                )
            if k == 0:
                # The only per-step host-written state: submesh 0's tiny fused packet
                # (token + positions). Everything downstream is fed over the socket.
                sm["pkt"] = _dev_zeros([1, 1, 1, self._pkt_w], device, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            else:
                # Receive buffers for the residual streams and the fused packet.
                sm["streams_in"] = _dev_zeros([1, 1, hc, d], device)
                sm["pkt_in"] = _dev_zeros([1, 1, 1, self._pkt_w], device, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            self.submeshes_io.append(sm)
        self._traced_captured = False

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

    def _decode_submesh_static(self, sm: dict) -> ttnn.Tensor:
        """Run one submesh's slice of the decode stack over the per-step input
        packets / in-place caches (shared by the compile run and the trace capture).

        The per-step inputs arrive as ONE tiny fused INT32 packet ``[1,1,1,3]`` =
        ``[token, pos_sliding, pos_compress]``. Submesh 0 reads its single persistent
        host-written packet buffer directly; every later submesh receives the packet
        (and the residual streams) over the socket. Each submesh then *splits* the
        packet on device and generates both the RoPE rows and the additive masks from
        ``pos_compress`` — and, unless it is the last, forwards the streams and packet
        unchanged to the next submesh."""
        cfg = self.config
        k = sm["index"]
        if sm["first"]:
            pkt = sm["pkt"]
        else:
            # Receive the residual streams and the fused packet from the previous
            # submesh directly into the persistent buffers. Captured inside this
            # submesh's trace so the cross-submesh copies need no host-side op
            # dispatch at replay time. Order must match the sender below.
            _, receiver_socket = self.submesh_socket_pairs[(k - 1, k)]
            ttnn.experimental.recv_direct_async(sm["streams_in"], receiver_socket)
            ttnn.experimental.recv_direct_async(sm["pkt_in"], receiver_socket)
            pkt = sm["pkt_in"]

        # Split the packet -> token, sliding position [1], compress position [1].
        token = ttnn.typecast(ttnn.reshape(ttnn.slice(pkt, [0, 0, 0, 0], [1, 1, 1, 1]), [1, 1]), ttnn.uint32)  # [1,1]
        sliding_pos = ttnn.reshape(ttnn.slice(pkt, [0, 0, 0, 1], [1, 1, 1, 2]), [1])
        compress_pos = ttnn.reshape(ttnn.slice(pkt, [0, 0, 0, 2], [1, 1, 1, 3]), [1])

        # Generate the per-step RoPE rows and additive masks on device from the
        # absolute position (nothing position-dependent is shipped in the packet).
        pos_f = ttnn.typecast(ttnn.to_layout(ttnn.reshape(compress_pos, [1, 1, 1, 1]), ttnn.TILE_LAYOUT), ttnn.float32)
        rope_views: dict[str, tuple] = {}
        for rt, (inv_freq, scaling) in sm["rope_invfreq"].items():
            rope_views[rt] = self._device_rope(inv_freq, scaling, pos_f)

        types = {cfg.layer_types[li] for li in sm["layers"]}
        mask_views: dict[str, ttnn.Tensor] = {}
        for lt in types:
            a, b, cr = sm["mask_gen"][lt]
            mask_views[lt] = self._device_mask(a, b, cr, pos_f)

        if sm["first"]:
            inputs_embeds = self.embed_tokens(token)  # [1, 1, D]
            b, s, d = inputs_embeds.shape
            streams = ttnn.repeat(ttnn.reshape(inputs_embeds, [b, s, 1, d]), ttnn.Shape([1, 1, cfg.hc_mult, 1]))
        else:
            streams = sm["streams_in"]
        for li in sm["layers"]:
            layer = self.layers[li]
            lt = cfg.layer_types[li]
            rope_type = "main" if lt == "sliding_attention" else "compress"
            cos, sin, neg_sin = rope_views[rope_type]
            if lt == "sliding_attention":
                cos_win = sin_win = None
            else:
                cos_win, sin_win = sm["win_rope"][cfg.compress_rates[lt]]
            streams = layer.decode_static(
                streams,
                cos,
                sin,
                neg_sin,
                cos_win,
                sin_win,
                mask_views[lt],
                sm["scaches"][li],
                sliding_pos,
                compress_pos,
                hash_token=token if layer.mlp.is_hash else None,
            )
        if sm["last"]:
            streams = self.norm(self.hc_head(streams))
            if self._lm_head_traced is not None:
                streams = self._lm_head_traced(streams)
        else:
            # Send the residual streams and the fused packet to the next submesh over
            # the socket pair. Captured inside this submesh's trace, so the
            # cross-submesh copies are dispatched on device at replay time (no host
            # round-trip). Order must match the receiver above.
            sender_socket, _ = self.submesh_socket_pairs[(k, k + 1)]
            ttnn.experimental.send_direct_async(streams, sender_socket)
            ttnn.experimental.send_direct_async(pkt, sender_socket)
        return streams

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
        """Capture one trace per submesh: a compile run (to JIT the programs, which
        trace capture itself cannot do), then the recorded capture.

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
        compile_outs = []
        for sm in self.submeshes_io:
            logger.info(f"[traced-decode] compiling submesh {sm['index']} ({len(sm['layers'])} layers)")
            compile_outs.append(self._decode_submesh_static(sm))  # compile run (JITs the programs)
        for out in compile_outs:
            out.deallocate(True)
        for sm in self.submeshes_io:
            device = sm["device"]
            logger.info(f"[traced-decode] capturing submesh {sm['index']} ({len(sm['layers'])} layers)")
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            with _trace_capture_guard():
                out = self._decode_submesh_static(sm)
            ttnn.end_trace_capture(device, tid, cq_id=0)
            sm["tid"] = tid
            sm["output"] = out  # persistent; overwritten in place by every execute_trace
        self._traced_captured = True

    def decode_traced(self, token_id: int, pos: int) -> ttnn.Tensor:
        """One traced decode step: feed ``token_id`` at absolute position ``pos``.

        Requires a prior :meth:`prepare_static_decode`. Captures
        the per-submesh traces lazily on the first call, then (every call) refreshes
        the per-step inputs and replays each submesh's trace in order. The residual
        streams are socket-copied between submeshes from *inside* each trace
        (device-to-device, no host hop and no per-step host op dispatch).
        Returns the last submesh's persistent output tensor — logits ``[1,1,vocab]``
        if an ``lm_head`` was passed to :meth:`prepare_static_decode`, else the
        pre-head hidden ``[1, 1, 1, hidden]``.

        The returned tensor is overwritten by the next call, so consume it (e.g.
        ``ttnn.to_torch``) before decoding the following token.
        """
        self._set_step_inputs(token_id, pos)
        if not self._traced_captured:
            self._capture_traces()
        for sm in self.submeshes_io:
            ttnn.execute_trace(sm["device"], sm["tid"], cq_id=0, blocking=False)
        return self.submeshes_io[-1]["output"]

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
        sm0, sm_last = self.submeshes_io[0], self.submeshes_io[-1]
        if sm0["device"] != sm_last["device"]:
            raise NotImplementedError(
                "on-device sampling feedback currently requires a single submesh "
                "(sampled id and token_in must share a device)"
            )

        self._set_step_inputs(first_token_id, start_pos)
        if not self._traced_captured:
            self._capture_traces()

        pkt = sm0["pkt"]
        w = self._pkt_w
        sampled: list[ttnn.Tensor] = []
        tok_i32: ttnn.Tensor | None = None
        for i in range(n_steps):
            if i > 0:
                # Refresh positions / RoPE / masks (token slot reset to a placeholder)
                # then re-inject the previous step's device-sampled id into idx 0 of
                # the fused packet: slice off the placeholder token and re-concatenate
                # the real one, copied back in place. All eager (outside the trace).
                self._set_step_position_inputs(start_pos + i)
                rest = ttnn.slice(pkt, [0, 0, 0, 1], [1, 1, 1, w])  # everything past the token slot
                fused = ttnn.concat([ttnn.reshape(tok_i32, ttnn.Shape([1, 1, 1, 1])), rest], dim=-1)
                ttnn.copy(fused, pkt)
            for sm in self.submeshes_io:
                ttnn.execute_trace(sm["device"], sm["tid"], cq_id=0, blocking=False)
            logits_rm = ttnn.to_layout(sm_last["output"], ttnn.ROW_MAJOR_LAYOUT)  # [1, 1, vocab]
            tok = ttnn.argmax(logits_rm, dim=-1, keepdim=True)  # [1, 1, 1]
            sampled.append(
                ttnn.reshape(tok if tok.dtype == ttnn.uint32 else ttnn.typecast(tok, ttnn.uint32), ttnn.Shape([1, 1]))
            )
            tok_i32 = tok if tok.dtype == ttnn.int32 else ttnn.typecast(tok, ttnn.int32)

        # One-shot readback: concat all sampled ids and transfer once.
        all_toks = ttnn.concat(sampled, dim=0)  # [n_steps, 1]
        return ttnn.to_torch(all_toks).reshape(-1).to(torch.int64).tolist()
