# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import math

import torch
import ttnn
import models.tt_transformers.tt.decoder as transformer_decoder
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.mlp import MLP as TransformerMLP


class VoxtralTTTextMLP(TransformerMLP):
    """Voxtral-only text MLP customizations layered on tt_transformers MLP."""

    def __init__(self, *args, **kwargs) -> None:
        args_obj = kwargs["args"]
        original_as_tensor = ttnn.as_tensor
        weight_index = 0
        weight_names = ("w1", "w2", "w3")

        def as_tensor_with_voxtral_layout(*tensor_args, **tensor_kwargs):
            nonlocal weight_index
            if weight_index < len(weight_names):
                weight_name = weight_names[weight_index]
                weight_index += 1
                use_interleaved = (
                    weight_name in ("w1", "w3")
                    and getattr(args_obj, "mlp_interleaved_weights", False)
                    or weight_name == "w2"
                    and getattr(args_obj, "mlp_ff2_interleaved_weights", False)
                )
                if use_interleaved:
                    tensor_kwargs["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
                    cache_file_name = tensor_kwargs.get("cache_file_name")
                    if cache_file_name is not None:
                        tensor_kwargs["cache_file_name"] = type(cache_file_name)(
                            str(cache_file_name).replace(f"{weight_name}_sharded", f"{weight_name}_interleaved")
                        )
            return original_as_tensor(*tensor_args, **tensor_kwargs)

        ttnn.as_tensor = as_tensor_with_voxtral_layout
        try:
            super().__init__(*args, **kwargs)
        finally:
            ttnn.as_tensor = original_as_tensor

    def forward(self, x: ttnn.Tensor, mode: Mode) -> ttnn.Tensor:
        use_fused_w1_silu = (
            mode == Mode.DECODE
            and getattr(self.args, "mlp_w1_fuse_silu_decode", False)
            and hasattr(self.args, "get_mlp_ff1_w1_prg_config")
        )
        if not use_fused_w1_silu:
            return super().forward(x, mode)

        original_get_ff1_3_prg_config = self.args.get_mlp_ff1_3_prg_config
        original_mul = ttnn.mul
        ff1_3_call_count = 0

        def get_ff1_3_prg_config(mode_arg, seq_len=1, prefetcher=None):
            nonlocal ff1_3_call_count
            ff1_3_call_count += 1
            if ff1_3_call_count == 1:
                return self.args.get_mlp_ff1_w1_prg_config(mode_arg, seq_len, prefetcher)
            return original_get_ff1_3_prg_config(mode_arg, seq_len, prefetcher)

        def mul_with_optional_fused_silu(*mul_args, **mul_kwargs):
            if mul_kwargs.get("input_tensor_a_activations") == [self.activation_type]:
                mul_kwargs["input_tensor_a_activations"] = []
            return original_mul(*mul_args, **mul_kwargs)

        self.args.get_mlp_ff1_3_prg_config = get_ff1_3_prg_config
        ttnn.mul = mul_with_optional_fused_silu
        try:
            return super().forward(x, mode)
        finally:
            self.args.get_mlp_ff1_3_prg_config = original_get_ff1_3_prg_config
            ttnn.mul = original_mul


# The shared MLP initializer asks args.get_state_dict_prefix(self.__class__.__name__, ...).
# Keep the local subclass compatible with the existing tt_transformers state-dict module map.
VoxtralTTTextMLP.__name__ = "MLP"


@contextlib.contextmanager
def use_voxtral_text_mlp():
    original_mlp = transformer_decoder.MLP
    transformer_decoder.MLP = VoxtralTTTextMLP
    try:
        yield
    finally:
        transformer_decoder.MLP = original_mlp


class VoxtralTTMLP:
    """Voxtral SwiGLU MLP: ``ttnn.linear`` + SiLU-gated path (shared by text, acoustic FM, audio tokenizer blocks).

    Instantiate with the checkpoint key prefix for ``w1``/``w2``/``w3``; hidden/intermediate sizes follow weights.
    """

    def __init__(
        self,
        device,
        state_dict: dict[str, torch.Tensor],
        w1_key: str,
        w2_key: str,
        w3_key: str,
        weight_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        exact_silu: bool = False,
        compute_kernel_config=None,
        activation_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ff1_3_program_config=None,
        ff2_program_config=None,
        w1_mem_config=None,
        w2_mem_config=None,
        w3_mem_config=None,
        ff1_3_in0_shard_mem_config=None,
    ) -> None:
        self.device = device
        self.output_dtype = output_dtype
        self.exact_silu = exact_silu
        self.compute_kernel_config = compute_kernel_config
        self.activation_memory_config = activation_memory_config
        self.ff1_3_program_config = ff1_3_program_config
        self.ff2_program_config = ff2_program_config
        self.ff1_3_in0_shard_mem_config = ff1_3_in0_shard_mem_config

        def get_weight(key: str) -> torch.Tensor:
            if key in state_dict:
                return state_dict[key]
            if f"{key}.weight" in state_dict:
                return state_dict[f"{key}.weight"]
            raise KeyError(f"Missing MLP weight for key '{key}'")

        w1 = get_weight(w1_key).transpose(-2, -1).contiguous()
        w2 = get_weight(w2_key).transpose(-2, -1).contiguous()
        w3 = get_weight(w3_key).transpose(-2, -1).contiguous()

        self.w1 = ttnn.from_torch(
            w1,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=w1_mem_config or ttnn.DRAM_MEMORY_CONFIG,
        )
        self.w2 = ttnn.from_torch(
            w2,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=w2_mem_config or ttnn.DRAM_MEMORY_CONFIG,
        )
        self.w3 = ttnn.from_torch(
            w3,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=w3_mem_config or ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(
        self,
        x: ttnn.Tensor,
        *,
        activation_memory_config=None,
        ff1_3_program_config=None,
        ff2_program_config=None,
    ) -> ttnn.Tensor:
        act_mem = activation_memory_config or self.activation_memory_config

        ff1_3_prog = ff1_3_program_config if ff1_3_program_config is not None else self.ff1_3_program_config
        ff2_prog = ff2_program_config if ff2_program_config is not None else self.ff2_program_config

        # 1D-mcast configs (per_core_M=2) require ≥2 fused M-tiles after fuse_batch.
        # M-tiles = prod(batch_dims) × ceil(seq / TILE_SIZE). Auto-fall-back for bsz=1.
        _use_1d_mcast = True
        if ff1_3_program_config is None and self.ff1_3_program_config is not None:
            shape = list(x.shape)
            _m_tiles = math.ceil(shape[-2] / ttnn.TILE_SIZE)
            for d in shape[:-2]:
                _m_tiles *= d
            if _m_tiles < 2:
                _use_1d_mcast = False

        effective_ff1_3 = (
            ff1_3_prog if ff1_3_prog is not None and (ff1_3_program_config is not None or _use_1d_mcast) else None
        )
        effective_ff2 = ff2_prog if ff2_prog is not None and (ff2_program_config is not None or _use_1d_mcast) else None

        # Tier 1 call-time 2D configs use interleaved activations; instance 1D-mcast width-shards w1/w3 out.
        if ff1_3_program_config is not None:
            use_ff1_3_width_shard = False
        else:
            use_ff1_3_width_shard = effective_ff1_3 is not None
        ff1_3_out_mem = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if use_ff1_3_width_shard else act_mem

        _ff1_3_kw = {"dtype": self.output_dtype, "memory_config": ff1_3_out_mem}
        if self.compute_kernel_config is not None:
            _ff1_3_kw["compute_kernel_config"] = self.compute_kernel_config
        if effective_ff1_3 is not None:
            _ff1_3_kw["program_config"] = effective_ff1_3

        _ff2_kw = {"dtype": self.output_dtype, "memory_config": act_mem}
        if self.compute_kernel_config is not None:
            _ff2_kw["compute_kernel_config"] = self.compute_kernel_config
        if effective_ff2 is not None:
            _ff2_kw["program_config"] = effective_ff2

        # For DS matmuls, in0 must be L1 K-width-sharded; shard x then discard the copy.
        if self.ff1_3_in0_shard_mem_config is not None:
            x_in = ttnn.to_memory_config(x, self.ff1_3_in0_shard_mem_config)
        else:
            x_in = x

        w1_out = ttnn.linear(x_in, self.w1, **_ff1_3_kw)
        w3_out = ttnn.linear(x_in, self.w3, **_ff1_3_kw)

        if x_in is not x:
            ttnn.deallocate(x_in)

        if self.exact_silu:
            w1_act = ttnn.silu(w1_out, memory_config=ff1_3_out_mem)
            ttnn.deallocate(w1_out)
            w2_in = ttnn.mul(
                w1_act,
                w3_out,
                memory_config=ff1_3_out_mem,
                dtype=self.output_dtype,
            )
            ttnn.deallocate(w1_act)
        else:
            w2_in = ttnn.mul(
                w1_out,
                w3_out,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                memory_config=ff1_3_out_mem,
                dtype=self.output_dtype,
            )
            ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)

        # De-shard to L1-interleaved before w2: mcast_in0 requires interleaved in0.
        if use_ff1_3_width_shard:
            w2_in_l1 = ttnn.to_memory_config(w2_in, act_mem)
            ttnn.deallocate(w2_in)
            w2_in = w2_in_l1

        out = ttnn.linear(w2_in, self.w2, **_ff2_kw)
        ttnn.deallocate(w2_in)
        return out
