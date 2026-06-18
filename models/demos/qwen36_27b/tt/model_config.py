# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.6-27B model configuration for single-device P150a (Blackhole).
"""

from dataclasses import dataclass, field
from pathlib import Path

import ttnn


@dataclass
class Qwen36ModelConfig:
    # Model architecture
    hidden_size: int = 5120
    num_hidden_layers: int = 64
    full_attention_interval: int = 4
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6

    # DeltaNet
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    # Standard attention
    num_attention_heads: int = 24
    num_key_value_heads: int = 4
    head_dim: int = 256
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10000000.0

    # FFN
    intermediate_size: int = 17408

    # Weight quantization
    weights_dtype: ttnn.DataType = ttnn.bfloat8_b
    # Dense (attn/deltanet/mlp/lm_head) dtype override for the TP path; falls
    # back to weights_dtype when None.
    dense_dtype: ttnn.DataType = None

    # Inference
    max_batch_size: int = 1
    max_seq_len: int = 8192

    # ---- vLLM + tensor-parallel (TP) path (everything is dense; TP=8) ----
    # When dense_tp is True the whole stack (attn q/k/v/o, deltanet projections,
    # MLP gate/up col-parallel + down row-parallel, lm_head vocab col-parallel)
    # is sharded across `tp_size` devices on a 1xN line mesh (FABRIC_1D, Linear).
    dense_tp: bool = False
    tp_size: int = 8
    # On-device decode attention (no host round-trip; enables trace).
    ondevice_attn: bool = False
    # Fixed contiguous KV cache length for the vLLM contiguous-decode path.
    kv_cache_len: int = 2048
    # Matmul math fidelity: None -> ttnn default (HiFi4); else "LoFi"/"HiFi2"/...
    math_fidelity: str = None

    # Paths
    model_name: str = "Qwen/Qwen3.6-27B"
    cache_path: Path = field(default_factory=lambda: Path("/home/yito/ttwork/tt-metal/models/demos/qwen36_27b/weights"))

    def get_dense_dtype(self, default):
        """Dtype for dense weights. Lets the TP path run dense in (e.g.) bf16
        while keeping a different default; falls back to weights_dtype."""
        return self.dense_dtype if self.dense_dtype is not None else default

    _kcfg_cache = None

    def matmul_kcfg(self):
        """compute_kernel_config for matmuls (None => ttnn default)."""
        if self.math_fidelity is None:
            return None
        if self._kcfg_cache is None:
            self._kcfg_cache = ttnn.WormholeComputeKernelConfig(
                math_fidelity=getattr(ttnn.MathFidelity, self.math_fidelity),
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
        return self._kcfg_cache

    # ---- DeltaNet derived dims (mirror the coder_next config helpers) ----
    @property
    def head_expand_ratio(self) -> int:
        return self.linear_num_value_heads // self.linear_num_key_heads  # 48//16 = 3

    @property
    def linear_key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim  # 16*128 = 2048

    @property
    def linear_value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim  # 48*128 = 6144

    @property
    def conv_dim(self) -> int:
        return self.linear_key_dim * 2 + self.linear_value_dim  # 2048*2 + 6144 = 10240

    @property
    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads  # 24//4 = 6

    @property
    def layer_types(self) -> list[str]:
        types = []
        for i in range(self.num_hidden_layers):
            if (i + 1) % self.full_attention_interval == 0:
                types.append("full_attention")
            else:
                types.append("linear_attention")
        return types

    @property
    def num_deltanet_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == "linear_attention")

    @property
    def num_attention_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == "full_attention")

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def padded_vocab_size(self) -> int:
        tile_size = 32
        return ((self.vocab_size + tile_size - 1) // tile_size) * tile_size
