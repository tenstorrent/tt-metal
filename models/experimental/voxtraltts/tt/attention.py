# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch
import ttnn

from models.common.rmsnorm import RMSNorm
import models.tt_transformers.tt.attention as transformer_attention
from models.tt_transformers.tt.attention import Attention as TransformerAttention
from models.tt_transformers.tt.common import Mode


class VoxtralTTTextAttention(TransformerAttention):
    """Voxtral-only text attention customizations layered on tt_transformers Attention."""

    def __init__(self, *args, **kwargs) -> None:
        state_dict = kwargs["state_dict"]
        weight_cache_path = kwargs["weight_cache_path"]
        layer_num = kwargs["layer_num"]
        configuration = kwargs["configuration"]

        super().__init__(*args, **kwargs)

        if not self._use_interleaved_wo(configuration):
            return

        layer_name = configuration.get_state_dict_prefix("Attention", layer_num)
        pt_wo = state_dict[f"{layer_name}.wo.weight"].transpose(-1, -2).unsqueeze(0).unsqueeze(0)
        cache_file_name = (
            None
            if configuration.dummy_weights or weight_cache_path is None
            else weight_cache_path / f"{layer_name}.wo_interleaved"
        )

        self.wo = ttnn.as_tensor(
            pt_wo,
            dtype=self.wo_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=2),
            cache_file_name=cache_file_name,
        )

    def _use_interleaved_wo(self, configuration) -> bool:
        return (
            getattr(configuration, "attn_wo_interleaved_weights", False)
            and not self.use_fused_all_gather_matmul
            and not self.TG
        )

    def _use_interleaved_wo_decode(self) -> bool:
        return self._use_interleaved_wo(self.args) and self.prefetcher is None

    def forward_prefill(self, *args, **kwargs) -> ttnn.Tensor:
        original_tt_all_reduce = transformer_attention.tt_all_reduce
        original_nlp_create_qkv_heads = ttnn.experimental.nlp_create_qkv_heads
        original_nlp_concat_heads = ttnn.experimental.nlp_concat_heads
        original_all_gather_async = ttnn.experimental.all_gather_async
        original_linear = ttnn.linear

        def tt_all_reduce_prefill(*reduce_args, **reduce_kwargs):
            cluster_axis = reduce_kwargs.get("cluster_axis", reduce_args[3] if len(reduce_args) > 3 else 0)
            if cluster_axis == 1:
                reduce_kwargs["memory_config"] = self.args.get_attn_qkv_all_reduce_output_mem_config(
                    Mode.PREFILL, 1, None
                )
            elif cluster_axis == 0:
                reduce_kwargs["memory_config"] = self.args.get_attn_dense_output_mem_config(Mode.PREFILL, None)
            return original_tt_all_reduce(*reduce_args, **reduce_kwargs)

        def nlp_create_qkv_heads_prefill(*create_args, **create_kwargs):
            create_kwargs["memory_config"] = self.args.get_attn_create_head_input_mem_config(Mode.PREFILL)
            return original_nlp_create_qkv_heads(*create_args, **create_kwargs)

        def nlp_concat_heads_prefill(*concat_args, **concat_kwargs):
            concat_kwargs["memory_config"] = self.args.get_attn_concat_heads_output_mem_config(Mode.PREFILL, None)
            return original_nlp_concat_heads(*concat_args, **concat_kwargs)

        def all_gather_async_prefill(*gather_args, **gather_kwargs):
            gather_kwargs["memory_config"] = self.args.get_attn_all_gather_output_mem_config(Mode.PREFILL, None)
            return original_all_gather_async(*gather_args, **gather_kwargs)

        def linear_prefill(input_tensor, weight, *linear_args, **linear_kwargs):
            if weight is self.wo:
                linear_kwargs["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
            return original_linear(input_tensor, weight, *linear_args, **linear_kwargs)

        transformer_attention.tt_all_reduce = tt_all_reduce_prefill
        ttnn.experimental.nlp_create_qkv_heads = nlp_create_qkv_heads_prefill
        ttnn.experimental.nlp_concat_heads = nlp_concat_heads_prefill
        ttnn.experimental.all_gather_async = all_gather_async_prefill
        ttnn.linear = linear_prefill
        try:
            return super().forward_prefill(*args, **kwargs)
        finally:
            transformer_attention.tt_all_reduce = original_tt_all_reduce
            ttnn.experimental.nlp_create_qkv_heads = original_nlp_create_qkv_heads
            ttnn.experimental.nlp_concat_heads = original_nlp_concat_heads
            ttnn.experimental.all_gather_async = original_all_gather_async
            ttnn.linear = original_linear

    def forward_decode(self, *args, **kwargs) -> ttnn.Tensor:
        if not self._use_interleaved_wo_decode():
            return super().forward_decode(*args, **kwargs)

        original_tt_all_gather = transformer_attention.tt_all_gather

        def tt_all_gather_interleaved(*gather_args, **gather_kwargs):
            attn_output = original_tt_all_gather(*gather_args, **gather_kwargs)
            return ttnn.to_memory_config(attn_output, ttnn.L1_MEMORY_CONFIG)

        transformer_attention.tt_all_gather = tt_all_gather_interleaved
        try:
            return super().forward_decode(*args, **kwargs)
        finally:
            transformer_attention.tt_all_gather = original_tt_all_gather


# The shared Attention initializer asks configuration.get_state_dict_prefix(self.__class__.__name__, ...).
# Keep the local subclass compatible with the existing tt_transformers state-dict module map.
VoxtralTTTextAttention.__name__ = "Attention"


class VoxtralTTAttention:
    """GQA attention: fused QKV linear, optional RoPE, SDPA, output proj. Audio tokenizer: ``is_causal``, ``use_qk_norm``, ``attn_mask``."""

    def __init__(
        self,
        device,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        state_dict: dict[str, torch.Tensor],
        weight_prefix: str = "attention",
        weight_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        compute_kernel_config=None,
        sdpa_compute_kernel_config=None,
        activation_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        wqkv_program_config=None,
        wo_program_config=None,
        wqkv_mem_config=None,
        wo_mem_config=None,
        wqkv_in0_shard_mem_config=None,
        *,
        is_causal: bool = False,
        use_qk_norm: bool = False,
        qk_norm_eps: float = 1e-6,
        qk_norm_mode: Mode | str = Mode.PREFILL,
    ) -> None:
        self.device = device
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.output_dtype = output_dtype
        self.scale = 1.0 / math.sqrt(head_dim)
        self.compute_kernel_config = compute_kernel_config
        self.sdpa_compute_kernel_config = sdpa_compute_kernel_config or compute_kernel_config
        self.activation_memory_config = activation_memory_config
        self.wqkv_program_config = wqkv_program_config
        self.wo_program_config = wo_program_config
        self.wqkv_in0_shard_mem_config = wqkv_in0_shard_mem_config
        self.is_causal = is_causal
        self.use_qk_norm = use_qk_norm
        self._qk_norm_mode = qk_norm_mode

        self.q_norm: RMSNorm | None = None
        self.k_norm: RMSNorm | None = None
        if use_qk_norm:
            prefix = f"{weight_prefix}." if not weight_prefix.endswith(".") else weight_prefix
            qk_norm_weight_dtype = ttnn.bfloat16
            self.q_norm = RMSNorm(
                device=device,
                dim=num_attention_heads * head_dim,
                eps=qk_norm_eps,
                state_dict=state_dict,
                weight_key="q_norm",
                state_dict_prefix=prefix,
                weight_dtype=qk_norm_weight_dtype,
                is_distributed=None,
                tt_ccl=None,
            )
            self.k_norm = RMSNorm(
                device=device,
                dim=num_key_value_heads * head_dim,
                eps=qk_norm_eps,
                state_dict=state_dict,
                weight_key="k_norm",
                state_dict_prefix=prefix,
                weight_dtype=qk_norm_weight_dtype,
                is_distributed=None,
                tt_ccl=None,
            )

        def get_weight(key: str) -> torch.Tensor:
            if key in state_dict:
                return state_dict[key]
            if f"{key}.weight" in state_dict:
                return state_dict[f"{key}.weight"]
            raise KeyError(f"Missing attention weight for key '{key}'")

        wq = get_weight(f"{weight_prefix}.wq").transpose(-2, -1).contiguous()
        wk = get_weight(f"{weight_prefix}.wk").transpose(-2, -1).contiguous()
        wv = get_weight(f"{weight_prefix}.wv").transpose(-2, -1).contiguous()
        wo = get_weight(f"{weight_prefix}.wo").transpose(-2, -1).contiguous()

        wqkv = torch.cat([wq, wk, wv], dim=-1)

        self.wqkv = ttnn.from_torch(
            wqkv,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=wqkv_mem_config or ttnn.DRAM_MEMORY_CONFIG,
        )
        self.wo = ttnn.from_torch(
            wo,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=wo_mem_config or ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        attention_mask: torch.Tensor | ttnn.Tensor | None = None,
        *,
        attn_mask: ttnn.Tensor | None = None,
        qk_norm_mode: Mode | str | None = None,
        activation_memory_config=None,
        wqkv_program_config=None,
        wo_program_config=None,
        sliding_window_size: int | None = None,
        sdpa_program_config=None,
    ) -> ttnn.Tensor:
        seq_len = hidden_states.shape[-2]
        _qk_mode = self._qk_norm_mode if qk_norm_mode is None else qk_norm_mode
        act_mem = activation_memory_config or self.activation_memory_config

        _lin_kw = {
            "dtype": self.output_dtype,
            "memory_config": act_mem,
        }
        if self.compute_kernel_config is not None:
            _lin_kw["compute_kernel_config"] = self.compute_kernel_config

        wqkv_prog = wqkv_program_config if wqkv_program_config is not None else self.wqkv_program_config
        wo_prog = wo_program_config if wo_program_config is not None else self.wo_program_config

        # 1D-mcast configs (per_core_M=2) require ≥2 fused M-tiles after fuse_batch.
        # M-tiles = prod(batch_dims) × ceil(seq / TILE_SIZE). Auto-fall-back for bsz=1.
        _use_1d_mcast = True
        if wqkv_program_config is None and self.wqkv_program_config is not None:
            _hs_shape = list(hidden_states.shape)
            _m_tiles = math.ceil(_hs_shape[-2] / ttnn.TILE_SIZE)
            for d in _hs_shape[:-2]:
                _m_tiles *= d
            if _m_tiles < 2:
                _use_1d_mcast = False

        _wqkv_kw = dict(_lin_kw)
        if wqkv_prog is not None and (wqkv_program_config is not None or _use_1d_mcast):
            _wqkv_kw["program_config"] = wqkv_prog

        if self.wqkv_in0_shard_mem_config is not None:
            # DS path: shard in0 K-wise; output is L1 WIDTH SHARDED.
            # De-shard after so nlp_create_qkv_heads gets a contiguous interleaved tensor.
            hs_sharded = ttnn.to_memory_config(hidden_states, self.wqkv_in0_shard_mem_config)
            _wqkv_kw["memory_config"] = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
            xqkv_sharded = ttnn.linear(hs_sharded, self.wqkv, **_wqkv_kw)
            ttnn.deallocate(hs_sharded)
            xqkv = ttnn.to_memory_config(xqkv_sharded, act_mem)
            ttnn.deallocate(xqkv_sharded)
        else:
            xqkv = ttnn.linear(hidden_states, self.wqkv, **_wqkv_kw)

        if self.use_qk_norm and self.q_norm is not None and self.k_norm is not None:
            b, one, s, _ = tuple(xqkv.shape)
            q_width = self.num_attention_heads * self.head_dim
            k_width = self.num_key_value_heads * self.head_dim
            v_width = self.num_key_value_heads * self.head_dim
            xq = ttnn.slice(xqkv, [0, 0, 0, 0], [b, one, s, q_width])
            xk = ttnn.slice(xqkv, [0, 0, 0, q_width], [b, one, s, q_width + k_width])
            xv = ttnn.slice(
                xqkv,
                [0, 0, 0, q_width + k_width],
                [b, one, s, q_width + k_width + v_width],
            )
            ttnn.deallocate(xqkv)
            xq = self.q_norm(xq, mode=_qk_mode)
            xk = self.k_norm(xk, mode=_qk_mode)
            xqkv = ttnn.concat([xq, xk, xv], dim=3, memory_config=act_mem)
            ttnn.deallocate(xq)
            ttnn.deallocate(xk)
            ttnn.deallocate(xv)

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.num_attention_heads,
            num_kv_heads=self.num_key_value_heads,
            transpose_k_heads=False,
            memory_config=act_mem,
        )
        ttnn.deallocate(xqkv)

        identity_rope = cos is None or sin is None
        if not identity_rope:
            cos_slice = cos[:, :seq_len]
            sin_slice = sin[:, :seq_len]
            identity_rope = bool(torch.all(cos_slice == 1) and torch.all(sin_slice == 0))

        mask_tt: ttnn.Tensor | None = attn_mask
        mask_owned = False
        if attention_mask is not None:
            if mask_tt is not None:
                ttnn.deallocate(q)
                ttnn.deallocate(k)
                ttnn.deallocate(v)
                raise ValueError("Pass only one of attention_mask and attn_mask.")
            if isinstance(attention_mask, torch.Tensor):
                mask_tt = ttnn.from_torch(
                    attention_mask,
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                mask_owned = True
            else:
                mask_tt = attention_mask

        if mask_tt is not None and sliding_window_size is not None:
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            raise ValueError("attn_mask and sliding_window_size are mutually exclusive.")

        if mask_tt is not None and self.is_causal:
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            raise ValueError("is_causal and attn_mask/attention_mask are mutually exclusive.")

        if not identity_rope:
            assert cos is not None and sin is not None
            q_shape = tuple(q.shape)
            k_shape = tuple(k.shape)

            cos_tt = ttnn.from_torch(
                cos_slice.unsqueeze(1),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=act_mem,
            )
            sin_tt = ttnn.from_torch(
                sin_slice.unsqueeze(1),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=act_mem,
            )

            q_in = q
            if q.dtype != ttnn.bfloat16:
                q_in = ttnn.typecast(q, ttnn.bfloat16)
                ttnn.deallocate(q)
            k_in = k
            if k.dtype != ttnn.bfloat16:
                k_in = ttnn.typecast(k, ttnn.bfloat16)
                ttnn.deallocate(k)

            q_rot = ttnn.experimental.rotary_embedding(q_in, cos_tt, sin_tt)
            k_rot = ttnn.experimental.rotary_embedding(k_in, cos_tt, sin_tt)
            ttnn.deallocate(q_in)
            ttnn.deallocate(k_in)
            ttnn.deallocate(cos_tt)
            ttnn.deallocate(sin_tt)

            if tuple(q_rot.shape) != q_shape:
                q_s = ttnn.slice(q_rot, [0, 0, 0, 0], list(q_shape))
                ttnn.deallocate(q_rot)
                q = q_s
            else:
                q = q_rot

            if tuple(k_rot.shape) != k_shape:
                k_s = ttnn.slice(k_rot, [0, 0, 0, 0], list(k_shape))
                ttnn.deallocate(k_rot)
                k = k_s
            else:
                k = k_rot

        # GQA is handled natively by ttnn SDPA (requires only nqh % nkv == 0), so K/V are
        # passed at num_key_value_heads directly. Materializing repeated KV heads here was
        # responsible for a large slice/repeat/tilize/untilize op cluster (~15% of device
        # time) with no functional benefit.
        use_native_sliding_window = sliding_window_size is not None
        _sdpa_kw = {
            "scale": self.scale,
        }
        if use_native_sliding_window:
            _sdpa_kw["is_causal"] = True
            _sdpa_kw["sliding_window_size"] = int(sliding_window_size)
        else:
            _sdpa_kw["attn_mask"] = mask_tt
            _sdpa_kw["is_causal"] = self.is_causal if mask_tt is None else False
        if sdpa_program_config is not None:
            _sdpa_kw["program_config"] = sdpa_program_config
        if self.sdpa_compute_kernel_config is not None:
            _sdpa_kw["compute_kernel_config"] = self.sdpa_compute_kernel_config

        attn_out = ttnn.transformer.scaled_dot_product_attention(q, k, v, **_sdpa_kw)
        attn_out = ttnn.to_memory_config(attn_out, act_mem)

        if mask_owned and mask_tt is not None and not use_native_sliding_window:
            ttnn.deallocate(mask_tt)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        attn_out = ttnn.experimental.nlp_concat_heads(
            attn_out,
            memory_config=act_mem,
        )

        _wo_kw = dict(_lin_kw)
        if wo_prog is not None and (wo_program_config is not None or _use_1d_mcast):
            _wo_kw["program_config"] = wo_prog
        out = ttnn.linear(attn_out, self.wo, **_wo_kw)
        ttnn.deallocate(attn_out)
        return out
