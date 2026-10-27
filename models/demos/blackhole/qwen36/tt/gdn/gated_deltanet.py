# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.5-9B Gated DeltaNet layer — composes config/weights/state/prefill/decode.

Wraps the experimental ``gated_deltanet_forward_ttnn()`` and the on-device GDN prefill
kernel into a module that manages weight tensors, recurrent state, and conv state.
"""
import ttnn
from models.demos.blackhole.qwen36.tt.gdn.config import GDNConfig
from models.demos.blackhole.qwen36.tt.gdn.decode import recurrent_forward
from models.demos.blackhole.qwen36.tt.gdn.weights import load_gdn_weights


class Qwen36GatedDeltaNet:
    """Gated DeltaNet (linear attention) layer for Qwen3.5-9B.

    Maintains fixed-size recurrent state [B, H, K, V] that replaces the KV cache.
    Also maintains conv states [B, kernel_size-1, D] for causal conv1d history.
    Supports two modes:
      - "recurrent": single-token decode (T=1), O(1) memory
      - "chunk": multi-token prefill (T>1), chunked parallel processing
    """

    def __init__(self, mesh_device, config: GDNConfig, state_dict, tensor_cache_path=None):
        self.device = mesh_device
        self.cfg = config

        # Mirror config-derived scalar dims so the forward bodies read them directly.
        self.num_heads = config.num_heads
        self.num_v_heads = config.num_v_heads
        self.head_k_dim = config.head_k_dim
        self.head_v_dim = config.head_v_dim
        self.conv_kernel_size = config.conv_kernel_size
        self.norm_eps = config.norm_eps
        self.long_prefill_chunk_size = config.long_prefill_chunk_size

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.compute_kernel_config_decode = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.weights = load_gdn_weights(mesh_device, config, state_dict, tensor_cache_path)

        # Persistent B=1 state, updated in place so trace-baked addresses stay valid.
        def zeros(shape):
            return ttnn.zeros(
                shape,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        rec_shape = [1, self.num_v_heads, self.head_k_dim, self.head_v_dim]
        conv_shape = [1, self.conv_kernel_size - 1, config.q_dim + config.k_dim + config.v_dim]
        self.recurrent_state = zeros(rec_shape)
        self.fused_conv_state = zeros(conv_shape)
        self._zero_recurrent = zeros(rec_shape)
        self._zero_conv = zeros(conv_shape)

    def forward(self, x, mode="recurrent", chunk_size=None, valid_len=None):
        return recurrent_forward(self, x, mode=mode, chunk_size=chunk_size, valid_len=valid_len)

    def reset_state_inplace(self):
        """Zero the state in place for a new sequence."""
        ttnn.copy(self._zero_recurrent, self.recurrent_state)
        ttnn.copy(self._zero_conv, self.fused_conv_state)
