# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import ttnn
from models.experimental.tt_symbiote.core.module import TTNNLayerStack, TTNNModule
from models.experimental.tt_symbiote.core.run_config import trace_enabled
from models.experimental.tt_symbiote.modules.dots_ocr_attention import TTNNDotsOCRAttention
from models.experimental.tt_symbiote.modules.dots_ocr_mlp import TTNNDotsOCRMLP
from models.experimental.tt_symbiote.modules.linear import (
    _decode_rmsnorm_program_config,
    _decode_width_sharded_input_memory_config,
    _prefill_block_sharded_input_memory_config,
    _prefill_rmsnorm_program_config,
    _tp_requires_ccl,
)


def _env_on(name: str) -> bool:
    val = os.environ.get(name)
    return val is not None and val.strip().lower() in ("1", "true", "yes", "on")


def _mlp_bfp4_only() -> bool:
    """Legacy alias for ``MLP_WEIGHT_DTYPE=bfp4``. Kept so previously-recorded
    sweeps (``MLP_BFP4_ONLY=1 ...``) still work without modification."""
    return _env_on("MLP_BFP4_ONLY")


def _mlp_weight_dtype_mode() -> str:
    """Debug toggle for the two MLP matmuls' weight dtype. Returns one of:

    - ``'mixed'`` (default): production baseline -- BFP4 for layers 0-6
      and BFP8 for layers 7-27 (the layer-index promotion in
      ``_use_bfp8_decoder_weights``). Protects OCR table accuracy.
    - ``'bfp4'``: BFP4 across ALL 28 layers for both ``gate_up`` and
      ``down_proj``. Smallest weight footprint; known to drop the 'k' in
      'Hodgkin' on this checkpoint (BFP4's 3 mantissa bits aren't enough
      for the late-layer weights).
    - ``'bfp8'``: BFP8 across ALL 28 layers for both ``gate_up`` and
      ``down_proj``. Promotes layers 0-6 from BFP4 to BFP8 (doubles those
      layers' weight footprint); accuracy is at least as good as mixed.

    Controlled by ``MLP_WEIGHT_DTYPE`` (case-insensitive). Accepts the
    aliases ``bf4`` / ``bfloat4_b`` and ``bf8`` / ``bfloat8_b``.

    Legacy: ``MLP_BFP4_ONLY=1`` is honored as an alias for ``bfp4``."""
    raw = (os.environ.get("MLP_WEIGHT_DTYPE") or "").strip().lower()
    if raw in ("bfp4", "bf4", "bfloat4_b"):
        return "bfp4"
    if raw in ("bfp8", "bf8", "bfloat8_b"):
        return "bfp8"
    if _mlp_bfp4_only():
        return "bfp4"
    return "mixed"


def _mlp_gate_up_bfp8_weights() -> bool:
    """Force gate_up weights to BFP8 across all 28 decoder layers (overrides
    the BFP4 default for layers 0-6). Triggered by ``MLP_GATE_UP_BFP8_WEIGHTS=1``
    (this knob alone) or ``MLP_GATE_UP_BFP8_IO=1`` (combo alias that also flips
    the input dtype). Scoped to gate_up only -- down_proj keeps the normal
    layer-index promotion."""
    return _env_on("MLP_GATE_UP_BFP8_WEIGHTS") or _env_on("MLP_GATE_UP_BFP8_IO")


from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedRMSNorm


def _mesh_dp_batch_sharded(device, batch_size: int) -> bool:
    if not hasattr(device, "get_num_devices") or int(device.get_num_devices()) <= 1:
        return False
    num_devices = int(device.get_num_devices())
    if int(batch_size) != num_devices or not hasattr(device, "shape"):
        return False
    mesh_shape = [int(x) for x in device.shape]
    return len(mesh_shape) == 2 and (
        (mesh_shape[0] == num_devices and mesh_shape[1] == 1) or (mesh_shape[1] == num_devices and mesh_shape[0] == 1)
    )


def _take_local_dp_batch(hidden_states, device):
    if len(hidden_states.shape) != 3 or int(hidden_states.shape[0]) <= 1:
        return hidden_states
    if not _mesh_dp_batch_sharded(device, int(hidden_states.shape[0])):
        return hidden_states
    return ttnn.slice(
        hidden_states,
        [0, 0, 0],
        [1, int(hidden_states.shape[-2]), int(hidden_states.shape[-1])],
    )


def _use_bfp8_decoder_weights(layer_idx) -> bool:
    if layer_idx is None:
        return False
    layer_idx = int(layer_idx)
    # Layers 0..6 stay BFP4 for decode speed; later layers are more sensitive
    # for OCR spelling/table tokens.
    return layer_idx >= 7


class TTNNDotsOCRLocalShardRMSNorm(TTNNDistributedRMSNorm):
    def move_weights_to_device_impl(self):
        # Inherit the distributed-RMSNorm weight setup (weight_distributed +
        # compute_kernel_config) from the parent so the interleaved fallback
        # path still works unchanged.
        super().move_weights_to_device_impl()
        # Build a replicated per-device tile-layout weight sized [32, padded_dim]
        # that the sharded multi-core RMSNorm kernel consumes directly. This
        # is the same shape/layout the parent uses for its single-device
        # ``tt_weight_local`` fallback, just promoted to work on every device
        # of the mesh (no mesh_mapper -> replication).
        dim = int(self.torch_layer.weight.shape[0])
        padded_dim = ((dim + 31) // 32) * 32
        weight = self.torch_layer.weight
        if padded_dim != dim:
            weight = torch.nn.functional.pad(weight, (0, padded_dim - dim), value=1.0)
        self.tt_weight_sharded = ttnn.from_torch(
            weight.unsqueeze(0).expand(32, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        self.tt_weight_sharded = ttnn.to_device(self.tt_weight_sharded, self.device)

    def _forward_decode_sharded(self, inp, original_shape):
        # Sharded LayerNorm fast path (decode only, non-TP-CCL). Runs the
        # multi-core sharded RMSNorm kernel by re-sharding the input to the
        # exact ``shard_in_cfg`` the program config expects, then re-sharding
        # the output back to DRAM_MEMORY_CONFIG so downstream QKV/MLP matmuls
        # see the same interleaved input layout as on the unsharded path.
        # That second reshard is what isolates this optimization: nothing
        # outside this method has to change.
        hidden_size = int(self.torch_layer.weight.shape[0])
        eps = getattr(self.torch_layer, "variance_epsilon", getattr(self.torch_layer, "eps", 1e-6))
        if len(original_shape) == 3:
            inp = ttnn.unsqueeze(inp, 1)
        if inp.layout != ttnn.TILE_LAYOUT:
            inp = ttnn.to_layout(inp, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        shard_in_cfg = _decode_width_sharded_input_memory_config(hidden_size)
        if inp.memory_config() != shard_in_cfg:
            if inp.is_sharded():
                inp = ttnn.to_memory_config(inp, shard_in_cfg)
            else:
                inp = ttnn.interleaved_to_sharded(inp, shard_in_cfg)
        # HiFi4 + FP32 dest accumulator + packer L1 accumulator. The multi-core
        # sharded RMSNorm combines partial variances across cores; the partial
        # sums sit in the dest register and the L1 packer buffer between
        # cores, so both knobs need maximum precision to keep the cross-core
        # variance combine accurate. Without ``packer_l1_acc=True`` we saw
        # downstream tokenization drift (``Hodgkin`` -> ``Hodgin`` and
        # ``(reference group`` -> ``( (reference group``).
        sharded_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        tt_out = ttnn.rms_norm(
            inp,
            epsilon=eps,
            weight=self.tt_weight_sharded,
            program_config=_decode_rmsnorm_program_config(hidden_size),
            memory_config=shard_in_cfg,
            compute_kernel_config=sharded_compute_kernel_config,
        )
        # Leave the output L1 width-sharded on the same 16c 8x2 grid we
        # computed on. Downstream consumers (QKV and gate-up matmuls in
        # decode) are configured as DRAM-sharded matmuls that take this
        # exact layout as input; the previous trailing
        # ``sharded_to_interleaved`` here is what removed the umbrella
        # across LN -> matmul. If a caller still needs an interleaved
        # tensor (prefill, non-sharded QKV path), it will reshard itself
        # via ``to_memory_config``.
        if len(original_shape) == 3 and len(tt_out.shape) == 4:
            tt_out = ttnn.reshape(tt_out, [tt_out.shape[0], tt_out.shape[2], tt_out.shape[3]])
        return tt_out

    _PREFILL_SHARDED_GRID_X = 8
    _PREFILL_SHARDED_GRID_Y = 8

    def _forward_prefill_sharded(self, inp, original_shape, logical_seq_len):
        # Sharded LayerNorm for **prefill** (multi-token). Block-sharded on
        # the full 8x8 worker grid: split seq across rows and hidden across
        # columns. This is the 2D path of the sharded LN kernel
        # (``layernorm_device_operation.cpp:282-302``), distinct from the
        # mcast_1d width-sharded path decode uses. The block-sharded layout
        # keeps the per-CB tile count small (66 tiles at seq=2816 /
        # hidden=1536), which is what makes it fit L1 -- the simpler
        # width-shard with block_h=88 blew the 1.5 MB L1 budget by 2x.
        #
        # Downstream prefill QKV / gate_up matmuls run on the 2D-mcast
        # kernel which wants DRAM_INTERLEAVED input, so we sharded ->
        # interleaved before returning -- the reshard cost is recovered by
        # running the LN compute on 64 cores in parallel instead of the
        # single-core interleaved variant.
        hidden_size = int(self.torch_layer.weight.shape[0])
        eps = getattr(self.torch_layer, "variance_epsilon", getattr(self.torch_layer, "eps", 1e-6))
        if len(original_shape) == 3:
            inp = ttnn.unsqueeze(inp, 1)
        if inp.layout != ttnn.TILE_LAYOUT:
            inp = ttnn.to_layout(inp, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # The on-device tensor is padded to a tile boundary. Use the padded
        # shape for the shard spec / program config so block_h * grid_y is
        # tile-aligned; use the logical seq separately to scale the output
        # back to compensate the padded-vs-logical RMS bias.
        seq_len_logical_padded = int(inp.shape[-2])
        seq_len = ((seq_len_logical_padded + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        seq_len_tiles = seq_len // ttnn.TILE_SIZE
        grid_x = self._PREFILL_SHARDED_GRID_X
        grid_y = self._PREFILL_SHARDED_GRID_Y
        shard_in_cfg = _prefill_block_sharded_input_memory_config(hidden_size, seq_len, grid_x, grid_y)
        if inp.memory_config() != shard_in_cfg:
            if inp.is_sharded():
                inp = ttnn.to_memory_config(inp, shard_in_cfg)
            else:
                inp = ttnn.interleaved_to_sharded(inp, shard_in_cfg)
        sharded_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        tt_out = ttnn.rms_norm(
            inp,
            epsilon=eps,
            weight=self.tt_weight_sharded,
            program_config=_prefill_rmsnorm_program_config(hidden_size, seq_len_tiles, grid_x, grid_y),
            memory_config=shard_in_cfg,
            compute_kernel_config=sharded_compute_kernel_config,
        )
        # Convert back to DRAM_INTERLEAVED for the prefill QKV / gate_up
        # matmul kernels. The 2D-mcast kernel does accept BLOCK_SHARDED in0
        # (matmul_device_operation.cpp:686-750) with ``fuse_batch=True``, but
        # for this shape (M=2816, K=1536, N=2048, BF16xBFP8 HiFi2) the sharded
        # path runs +54 μs slower than DRAM mcast (compute-bound at 41% FPU,
        # not bandwidth-bound, so eliminating the in0 DRAM read doesn't help),
        # and routing gate_up's input through L1 silently flips its matmul to
        # HiFi4 (33.8 ms vs 1.9 ms). Keep the S2I.
        tt_out = ttnn.sharded_to_interleaved(tt_out, ttnn.DRAM_MEMORY_CONFIG)
        # Compensate the padded-vs-logical RMS bias. The block-sharded LN
        # kernel computed mean(x^2) over the full padded shape (seq_len),
        # which equals mean_logical(x^2) * N_logical / N_padded when the
        # tile-padding rows are zero (true after tilize and preserved
        # through residual-add chains -- but NOT after embedding lookup
        # if pad-token-id != 0). The LN output is therefore over-amplified
        # by sqrt(N_padded / N_logical); multiplying by sqrt(N_logical /
        # N_padded) restores the correct logical-normalized values on the
        # real rows. For tile-aligned logical seq (logical == padded) the
        # factor is exactly 1.0 and this is a no-op.
        if logical_seq_len != seq_len:
            comp = (float(logical_seq_len) / float(seq_len)) ** 0.5
            tt_out = ttnn.multiply(tt_out, comp)
        if len(original_shape) == 3 and len(tt_out.shape) == 4:
            tt_out = ttnn.reshape(tt_out, [tt_out.shape[0], tt_out.shape[2], tt_out.shape[3]])
        return tt_out

    def forward(self, inp):
        original_shape = inp.shape
        # Sharded LN fast path: decode-shape (M=1) and single-device or pure DP
        # (no TP CCL). Anything else (prefill, TP) uses the parent's
        # interleaved RMSNorm path so this change can't affect those.
        is_decode = len(original_shape) >= 2 and int(original_shape[-2]) == 1
        if is_decode and not _tp_requires_ccl(self.device):
            return self._forward_decode_sharded(inp, original_shape)

        # Prefill block-sharded fast path. Uses the tile-PADDED seq for the
        # shard spec (logical 2814 -> padded 2816 = 88 tiles = 256 * 11), and
        # passes the logical seq into _forward_prefill_sharded so the kernel
        # output can be rescaled to compensate the padded-vs-logical RMS
        # bias (block-shard LN computes mean(x^2) over the full padded
        # shape, which under-estimates the true logical mean by factor
        # N_logical / N_padded -> output over-amplified by sqrt of the
        # reciprocal; we multiply the output by sqrt(N_logical/N_padded)
        # to undo it).
        if not _tp_requires_ccl(self.device):
            logical_seq_len = int(original_shape[-2]) if len(original_shape) >= 2 else 0
            seq_len_padded = ((logical_seq_len + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
            hidden_size = int(self.torch_layer.weight.shape[0])
            grid_x = self._PREFILL_SHARDED_GRID_X
            grid_y = self._PREFILL_SHARDED_GRID_Y
            if (
                seq_len_padded >= grid_y * ttnn.TILE_SIZE
                and seq_len_padded % (grid_y * ttnn.TILE_SIZE) == 0
                and hidden_size % (grid_x * ttnn.TILE_SIZE) == 0
            ):
                return self._forward_prefill_sharded(inp, original_shape, logical_seq_len)

        if len(original_shape) == 3:
            inp = ttnn.unsqueeze(inp, 1)
        if inp.layout != ttnn.TILE_LAYOUT:
            inp = ttnn.to_layout(inp, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        eps = getattr(self.torch_layer, "variance_epsilon", getattr(self.torch_layer, "eps", 1e-6))
        tt_out = ttnn.rms_norm(
            inp,
            epsilon=eps,
            weight=self.weight_distributed,
            compute_kernel_config=self.compute_kernel_config,
        )
        if len(original_shape) == 3 and len(tt_out.shape) == 4:
            tt_out = ttnn.reshape(tt_out, [tt_out.shape[0], tt_out.shape[2], tt_out.shape[3]])
        return tt_out


@trace_enabled
class TTNNDotsOCRDecoderLayer(TTNNModule):
    def __init__(self):
        super().__init__()
        self.input_layernorm = None
        self.post_attention_layernorm = None
        self.self_attn = None
        self.mlp = None

    @classmethod
    def from_torch(cls, torch_layer):
        new_layer = cls()
        new_layer._fallback_torch_layer = torch_layer
        new_layer.attention_type = getattr(torch_layer, "attention_type", "full_attention")
        new_layer.input_layernorm = TTNNDotsOCRLocalShardRMSNorm.from_torch(torch_layer.input_layernorm)
        new_layer.post_attention_layernorm = TTNNDotsOCRLocalShardRMSNorm.from_torch(
            torch_layer.post_attention_layernorm
        )
        new_layer.self_attn = TTNNDotsOCRAttention.from_torch(torch_layer.self_attn)
        new_layer.mlp = TTNNDotsOCRMLP.from_torch(torch_layer.mlp)
        # Attention o_proj weight promotion is unchanged: BFP8 for layers
        # 7-27 (sensitive for OCR table tokens), BFP4 for layers 0-6.
        if _use_bfp8_decoder_weights(getattr(new_layer.self_attn, "layer_idx", None)):
            new_layer.self_attn.o_proj.set_weight_dtype(ttnn.bfloat8_b)

        # MLP weight dtype is driven by ``MLP_WEIGHT_DTYPE`` (a single debug
        # toggle that applies to BOTH gate_up and down_proj):
        #   - 'mixed' (default): BFP4 for L<7, BFP8 for L>=7
        #   - 'bfp4'           : BFP4 for all 28 layers
        #   - 'bfp8'           : BFP8 for all 28 layers
        mlp_mode = _mlp_weight_dtype_mode()
        if mlp_mode == "bfp8":
            new_layer.mlp.set_weight_dtype(ttnn.bfloat8_b)
        elif mlp_mode == "mixed" and _use_bfp8_decoder_weights(getattr(new_layer.self_attn, "layer_idx", None)):
            new_layer.mlp.set_weight_dtype(ttnn.bfloat8_b)
        # mlp_mode == 'bfp4': leave both matmuls at their default BFP4.

        # ``MLP_GATE_UP_BFP8_WEIGHTS=1`` (or the combo alias ``MLP_GATE_UP_BFP8_IO=1``)
        # forces gate_up weights specifically to BFP8 for ALL layers,
        # independent of the layer-index promotion above. Useful for isolating
        # which matmul (gate_up vs down_proj) is the source of a regression
        # when ``MLP_WEIGHT_DTYPE=bfp4`` introduces one. The input-side typecast
        # is decoupled and lives behind ``MLP_GATE_UP_BFP8_INPUT=1`` in
        # dots_ocr_mlp.py so the four (input, weight) combinations can be
        # measured in isolation.
        if _mlp_gate_up_bfp8_weights():
            new_layer.mlp.fused_gate_up_proj.set_weight_dtype(ttnn.bfloat8_b)
        return new_layer

    def call(self, *args, **kwds):
        # Keep only kwargs used by forward — unused kwargs with incompatible
        # dtypes (e.g. UINT8 from bool masks) cause ttnn.copy failures in trace replay.
        filtered = {k: kwds[k] for k in ("past_key_value", "cache_position") if k in kwds}
        return super().call(*args, **filtered)

    def post_trace_execute(self, func_args, func_kwargs, result):
        past_key_value = func_kwargs.get("past_key_value")
        if past_key_value is None or not hasattr(past_key_value, "update_seq_length"):
            return
        hidden_states = func_args[0]
        seq_len = hidden_states.shape[-2]
        layer_idx = self.self_attn.layer_idx
        past_key_value.update_seq_length(layer_idx=layer_idx, seq_len=seq_len)

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        **kwargs,
    ):
        hs = _take_local_dp_batch(hidden_states, self.device)

        if hs.layout != ttnn.TILE_LAYOUT:
            hs = ttnn.to_layout(hs, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if hs.dtype != ttnn.bfloat16:
            hs = ttnn.typecast(hs, ttnn.bfloat16)

        # Attention block
        residual = hs
        hs = self.input_layernorm(hs)

        seq_len = hs.shape[-2]
        is_decode = seq_len == 1
        decode_l1_mc = ttnn.L1_MEMORY_CONFIG if is_decode else None
        attn_out, _ = self.self_attn(
            hidden_states=hs,
            position_embeddings=None,
            attention_mask=None,
            past_key_values=past_key_value,
            cache_position=kwargs.get("cache_position"),
            decode_cur_pos_tt=kwargs.get("decode_cur_pos_tt"),
            decode_cos_sin=kwargs.get("decode_cos_sin"),
        )

        hs = (
            ttnn.add(residual, attn_out, memory_config=decode_l1_mc)
            if decode_l1_mc is not None
            else ttnn.add(residual, attn_out)
        )
        ttnn.deallocate(attn_out)

        # MLP block
        residual = hs
        hs = self.post_attention_layernorm(hs)
        mlp_out = self.mlp(hs)

        hs = (
            ttnn.add(residual, mlp_out, memory_config=decode_l1_mc)
            if decode_l1_mc is not None
            else ttnn.add(residual, mlp_out)
        )
        ttnn.deallocate(mlp_out)
        ttnn.deallocate(residual)

        # CRITICAL: Return tuple — Qwen2Model does layer_outputs[0]
        return (hs,)


class TTNNDotsOCRLayerStack(TTNNLayerStack):
    def call(self, *args, **kwds):
        filtered = {k: kwds[k] for k in ("past_key_value", "cache_position") if k in kwds}
        return super().call(*args, **filtered)

    def move_weights_to_device_impl(self):
        super().move_weights_to_device_impl()
        shared_buf = None
        for layer in self.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "_decode_cur_pos") and attn._decode_cur_pos is not None:
                if shared_buf is None:
                    shared_buf = attn._decode_cur_pos
                else:
                    ttnn.deallocate(attn._decode_cur_pos)
                    attn._decode_cur_pos = shared_buf
        self._shared_decode_cur_pos = shared_buf

    def forward(self, hidden_states, **kwargs):
        seq_len = hidden_states.shape[-2]
        if (
            seq_len == 1
            and getattr(self, "_shared_decode_cur_pos", None) is not None
            and self.layers
            and "decode_cos_sin" not in kwargs
        ):
            attn0 = getattr(self.layers[0], "self_attn", None)
            rotary_setup = getattr(attn0, "_rotary_setup", None) if attn0 is not None else None
            cache_position = kwargs.get("cache_position")
            if rotary_setup is not None and cache_position is not None:
                cur_pos_tt = self._materialize_shared_cur_pos(cache_position)
                if cur_pos_tt is not None:
                    kwargs["decode_cur_pos_tt"] = cur_pos_tt
                    kwargs["decode_cos_sin"] = rotary_setup.get_cos_sin_for_decode(cur_pos_tt)

        for layer in self.layers:
            layer_output = layer.forward(hidden_states, **kwargs)
            hidden_states = layer_output[0]
        return hidden_states

    def _materialize_shared_cur_pos(self, cache_position):
        cp = cache_position
        if hasattr(cp, "ttnn_tensor") and cp.ttnn_tensor is not None:
            cp = cp.ttnn_tensor
        if not isinstance(cp, ttnn.Tensor):
            return None
        if len(cp.shape) > 1:
            total_elems = 1
            for d in cp.shape:
                total_elems *= d
            cp = ttnn.reshape(cp, (total_elems,))
        if cp.shape[0] > 1:
            cp = ttnn.slice(cp, [0], [1])
        ttnn.copy(cp, self._shared_decode_cur_pos)
        return self._shared_decode_cur_pos

    def pre_trace_execute(self, func_args, func_kwargs):
        cache_position = func_kwargs.get("cache_position")
        if cache_position is None:
            return

        cp = cache_position
        if hasattr(cp, "ttnn_tensor") and cp.ttnn_tensor is not None:
            cp = cp.ttnn_tensor

        if len(cp.shape) > 1:
            total = 1
            for d in cp.shape:
                total *= d
            cp = ttnn.reshape(cp, (total,))

        if cp.shape[0] > 1:
            cp = ttnn.slice(cp, [0], [1])

        if hasattr(self, "_shared_decode_cur_pos") and self._shared_decode_cur_pos is not None:
            ttnn.copy(cp, self._shared_decode_cur_pos)
            return

        for layer in self.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "_decode_cur_pos") and attn._decode_cur_pos is not None:
                ttnn.copy(cp, attn._decode_cur_pos)

    def post_trace_execute(self, func_args, func_kwargs, result):
        past_key_value = func_kwargs.get("past_key_value")
        if past_key_value is None or not hasattr(past_key_value, "update_seq_length"):
            return
        hidden_states = func_args[0]
        seq_len = hidden_states.shape[-2]
        for layer in self.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
                layer_idx = layer.self_attn.layer_idx
                past_key_value.update_seq_length(layer_idx=layer_idx, seq_len=seq_len)
