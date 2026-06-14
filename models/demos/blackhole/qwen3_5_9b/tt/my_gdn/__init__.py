import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from .weights import load_gdn_weights
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs
from .operations import causal_conv1d_silu, l2norm


class Qwen35GatedDeltaNet(LightweightModule):
    def __init__(
        self,
        args: Qwen35ModelArgs,
        state_dict,
        mesh_device,
    ):
        self.hidden_size = args.dim
        self.num_v_heads = args.linear_num_value_heads
        self.num_k_heads = args.linear_num_key_heads
        self.head_k_dim = args.linear_key_head_dim
        self.head_v_dim = args.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.layer_norm_epsilon = args.norm_eps
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.scale = 1 / (self.head_k_dim**0.5)

        self.weights = load_gdn_weights(mesh_device=mesh_device, state_dict=state_dict, args=args)
        self.mesh_device = mesh_device
        self.recurrent_state = None
        self.cache_params = None

    def chunk_gated_delta_rule(
        self, query, key, value, g, beta, chunk_size=64, initial_state=None, use_qk_l2norm_in_kernel=False
    ):
        initial_dtype = query.dtype
        if use_qk_l2norm_in_kernel:
            query = l2norm(query, dim=-1, eps=1e-6)
            key = l2norm(key, dim=-1, eps=1e-6)

        #
        # beta [1, 32, 2026]
        # g [1, 32, 2026]
        query, key, value, beta, g = [ttnn.transpose(x, 1, 2) for x in (query, key, value, beta, g)]

        sequence_length = query.shape[2]
        # batch_size, num_heads, sequence_length, k_head_dim = key.shape
        # v_head_dim = value.shape[-1]
        pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
        qkv_pad = [(0, 0), (0, 0), (0, pad_size), (0, 0)]  # pad seq (dim 2) at the end only
        bg_pad = [(0, 0), (0, 0), (0, pad_size)]
        query = ttnn.pad(query, padding=qkv_pad, value=0.0)
        key = ttnn.pad(key, padding=qkv_pad, value=0.0)
        value = ttnn.pad(value, padding=qkv_pad, value=0.0)
        beta = ttnn.pad(beta, padding=bg_pad, value=0.0)
        g = ttnn.pad(g, padding=bg_pad, value=0.0)

        total_sequence_length = sequence_length + pad_size
        query = query * self.scale

        v_beta = value * ttnn.unsqueeze(beta, -1)
        k_beta = key * ttnn.unsqueeze(beta, -1)

    def forward_prefill(self, hidden_states):
        """
        hidden_states: [B=1, 1, seq_len, hidden_size]
        """
        # attention masking if there is an attention parameter
        weights = self.weights
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[2]

        # mixed_qkv: [B=1, 1, seq_len, conv_dim] — already channels-last out of the linear,
        # which is exactly the layout the FIR conv wants (taps broadcast over the last dim).
        # We deliberately do NOT transpose to channels-first: that's what torch's Conv1d
        # demands, not us, and flipping here would feed the conv the layout it can't use.
        mixed_qkv = ttnn.linear(hidden_states, weights.wqkv)

        z = ttnn.linear(hidden_states, weights.wz)
        z = ttnn.reshape(z, (batch_size, seq_len, self.num_v_heads, self.head_v_dim))

        # a: [B=1, 1, seq_len, num_v_heads]
        # b: [B=11, 1, seq_len, num_v_heads]
        b = ttnn.linear(hidden_states, weights.wb)
        a = ttnn.linear(hidden_states, weights.wa, dtype=ttnn.float32)

        # Prefill path does not use precomputed states
        if self.cache_params is not None:
            conv_state = mixed_qkv[..., :, -self.conv_dim :]
            # conv_state = update conv state
        else:
            # causal_conv1d_silu's contract is a canonical [B, T, D]; drop the singleton
            # dim 1 that ttnn.linear leaves on. In tile layout this is a metadata-only
            # reshape (the last two dims, which carry the tiles, are untouched).

            mixed_qkv = ttnn.reshape(mixed_qkv, (batch_size, 1, seq_len, self.conv_dim))
            mixed_qkv = causal_conv1d_silu(
                x=mixed_qkv,
                weight_taps=weights.w_taps,
                kernel_size=self.conv_kernel_size,
                mesh_device=self.mesh_device,
            )  # [B, 1, seq_len, conv_dim, dim]

        query, key, value = ttnn.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

        query = ttnn.reshape(query, (batch_size, seq_len, self.num_k_heads, self.head_k_dim))
        key = ttnn.reshape(key, (batch_size, seq_len, self.num_k_heads, self.head_k_dim))
        value = ttnn.reshape(value, (batch_size, seq_len, self.num_v_heads, self.head_v_dim))

        beta = ttnn.squeeze(ttnn.sigmoid(b), 1)
        g = ttnn.squeeze(weights.neg_A_log_exp * ttnn.softplus(a + weights.dt_bias), 1)
        if self.num_v_heads // self.num_k_heads > 1:
            query = ttnn.repeat_interleave(query, self.num_v_heads // self.num_k_heads, dim=2)
            key = ttnn.repeat_interleave(key, self.num_v_heads // self.num_k_heads, dim=2)

            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                use_qk_l2norm_in_kernel=True,
            )
