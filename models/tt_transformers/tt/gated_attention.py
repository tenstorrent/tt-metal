# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Gated Attention for Qwen3.5.

Wraps the standard Attention class and adds:
1. Output gate: sigmoid(x @ gate_weight) * attn_output before WO projection
2. Partial RoPE: custom_rope_fn applies rotation to only the first rotary_dim
   dims (64 out of 256) on host, using correct frequencies for rotary_dim=64.

The standard rotary_embedding_llama cannot handle Qwen3.5's partial RoPE because:
- It uses head_dim=256 frequencies (should be rotary_dim=64)
- Its transformation matrix pairs dims 128 apart (should be 32 apart)
- cos/sin are in Meta interleaved format but Q/K are in HF format

The custom_rope_fn approach uses the Attention class's existing hook to bypass
rotary_embedding_llama entirely. Host roundtrip cost is negligible (~14 KB).
"""

import torch

import ttnn
from models.tt_transformers.tt.attention import Attention
from models.tt_transformers.tt.common import Mode


class GatedAttention(Attention):
    def __init__(
        self,
        mesh_device,
        tt_ccl,
        args,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        transformation_mats,
        configuration,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        prefetcher=None,
    ):
        # Standard Attention handles QKV, WO, KV cache, norms, etc.
        super().__init__(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            transformation_mats=transformation_mats,
            configuration=configuration,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
            prefetcher=prefetcher,
        )

        # Detect multi-device mesh
        self._is_mesh = self.num_devices > 1

        # Load gate weight: (hidden_size, n_heads * head_dim)
        layer_prefix = configuration.get_state_dict_prefix("GatedAttention", layer_num)
        gate_key = f"{layer_prefix}.q_proj_gate.weight"
        if gate_key in state_dict:
            gate_w = state_dict[gate_key].float().T  # (hidden_size, q_size)
            cache_name = (
                None
                if configuration.dummy_weights or weight_cache_path is None
                else weight_cache_path / f"{layer_prefix}.gate_weight"
            )
            self.gate_weight = ttnn.as_tensor(
                gate_w.unsqueeze(0).unsqueeze(0),
                dtype=ttnn.bfloat16,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_name,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1) if self._is_mesh else None,
            )
        else:
            self.gate_weight = None

        # Storage for the input tensor (set in forward, used by hook)
        self._gate_input = None

        # Install the pre-WO hook
        if self.gate_weight is not None:
            self.pre_wo_hook = self._apply_gate

        # Partial RoPE: when the Transformer class patches cos/sin matrices at init
        # (cos=1, sin=0 for dims >= rotary_dim), the standard device-side RoPE handles
        # partial rotation correctly. Only fall back to host-based custom_rope_fn on
        # single-device where the patching may not be applied (e.g. standalone tests).
        partial_factor = getattr(configuration, "partial_rotary_factor", 1.0)
        if partial_factor < 1.0 and not self._is_mesh:
            self._setup_partial_rope(configuration)

    def _setup_partial_rope(self, args):
        """Set up host-based partial RoPE for Qwen3.5.

        Qwen3.5 rotates only the first rotary_dim=64 dims of each head
        (head_dim=256, partial_rotary_factor=0.25). Rotation pairs are
        (dim j, dim j+32) for j in [0, 31], with frequencies computed
        using 1/theta^(2i/64) -- NOT 1/theta^(2i/256).
        """
        rotary_dim = int(args.head_dim * args.partial_rotary_factor)
        half_dim = rotary_dim // 2
        rope_theta = args.rope_theta

        # Precompute inverse frequencies: 32 frequencies for 32 rotation pairs
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        device = self.mesh_device

        is_mesh = self._is_mesh

        def partial_rope_fn(q_tt, k_tt, current_pos):
            """Apply partial RoPE on host. ~14 KB roundtrip, negligible latency."""
            if is_mesh:
                pos = ttnn.to_torch(ttnn.get_device_tensors(current_pos)[0]).int().item()
            else:
                pos = ttnn.to_torch(current_pos).int().item()
            freqs = pos * inv_freq
            cos_val = freqs.cos()
            sin_val = freqs.sin()

            if is_mesh:
                mesh_composer = ttnn.ConcatMeshToTensor(device, dim=1)
                q = ttnn.to_torch(q_tt, mesh_composer=mesh_composer).float()
                k = ttnn.to_torch(k_tt, mesh_composer=mesh_composer).float()
            else:
                q = ttnn.to_torch(q_tt).float()
                k = ttnn.to_torch(k_tt).float()

            def apply_rot(x):
                x1 = x[..., :half_dim]
                x2 = x[..., half_dim:rotary_dim]
                x_rotated = torch.cat(
                    [
                        x1 * cos_val - x2 * sin_val,
                        x2 * cos_val + x1 * sin_val,
                    ],
                    dim=-1,
                )
                return torch.cat([x_rotated, x[..., rotary_dim:]], dim=-1)

            q = apply_rot(q)
            k = apply_rot(k)

            if is_mesh:
                mesh_mapper = ttnn.ShardTensorToMesh(device, dim=1)
                q_out = ttnn.from_torch(
                    q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mesh_mapper
                )
                k_out = ttnn.from_torch(
                    k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mesh_mapper
                )
            else:
                q_out = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
                k_out = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            return q_out, k_out

        self.custom_rope_fn = partial_rope_fn

    def _apply_gate(self, attn_output):
        """Pre-WO hook: multiply attn_output by sigmoid(x @ gate_weight).

        attn_output shape: (1, 1, batch, n_heads * head_dim) in L1
        _gate_input shape: (1, 1, batch, hidden_size) in DRAM
        gate_weight shape: (1, 1, hidden_size, n_heads * head_dim) in DRAM
        """
        gate = ttnn.linear(
            self._gate_input,
            self.gate_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate = ttnn.sigmoid(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Ensure compatible memory configs for multiply
        attn_output = ttnn.to_memory_config(attn_output, ttnn.DRAM_MEMORY_CONFIG)
        result = ttnn.mul(attn_output, gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        ttnn.deallocate(gate)
        ttnn.deallocate(self._gate_input)
        self._gate_input = None
        return result

    def forward(
        self,
        x,
        current_pos,
        rot_mats=None,
        user_id=0,
        mode=Mode.DECODE,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
    ):
        # Save a copy of input for gate computation (the hook reads it later).
        # Attention.forward_decode deallocates x after the QKV matmul, so we
        # must ensure _gate_input is a separate buffer. ttnn.add(x, 0) always
        # creates a new tensor, unlike to_memory_config which may alias when
        # source and destination configs match.
        if self.gate_weight is not None:
            self._gate_input = ttnn.add(x, 0, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Delegate to standard Attention forward
        return super().forward(
            x,
            current_pos,
            rot_mats=rot_mats,
            user_id=user_id,
            mode=mode,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            kv_cache=kv_cache,
        )
