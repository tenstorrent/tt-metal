# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.tt_symbiote.core.module import TTNNLayerStack, TTNNModule
from models.experimental.tt_symbiote.core.run_config import trace_enabled
from models.experimental.tt_symbiote.modules.dots_ocr_attention import TTNNDotsOCRAttention
from models.experimental.tt_symbiote.modules.dots_ocr_mlp import TTNNDotsOCRMLP
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
    def forward(self, inp):
        original_shape = inp.shape
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
        if _use_bfp8_decoder_weights(getattr(new_layer.self_attn, "layer_idx", None)):
            new_layer.self_attn.o_proj.set_weight_dtype(ttnn.bfloat8_b)
            new_layer.mlp.set_weight_dtype(ttnn.bfloat8_b)
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
