# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Gemma 4 Decoder Block

Custom decoder that extends TransformerBlock with:
- Per-layer input gating mechanism
- Layer scalar (learnable per-layer multiplier)
- Double-wide MLP for KV-shared layers (2x intermediate_size)
- Custom attention class routing
"""


import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.gemma4.tt.gemma4_attention import Gemma4Attention
from models.tt_transformers.tt.decoder import TransformerBlock


class Gemma4Decoder(TransformerBlock):
    """
    Gemma 4 decoder block with per-layer input gating, layer scalar,
    and double-wide MLP for KV-shared layers.
    """

    def __init__(
        self,
        args,
        mesh_device,
        tt_ccl,
        dtype,
        state_dict,
        layer_num,
        weight_cache_path,
        transformation_mats,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        attention_class=None,
        prefetcher=None,
    ):
        # Override the MLP intermediate_size for KV-shared layers before parent __init__
        # We do this by temporarily modifying args.hidden_dim
        self._original_hidden_dim = args.hidden_dim
        layer_intermediate_size = args.get_layer_intermediate_size(layer_num)
        args.hidden_dim = layer_intermediate_size

        # Use Gemma4Attention always
        super().__init__(
            args=args,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            dtype=dtype,
            state_dict=state_dict,
            layer_num=layer_num,
            weight_cache_path=weight_cache_path,
            transformation_mats=transformation_mats,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
            attention_class=Gemma4Attention,
            prefetcher=prefetcher,
        )

        # Restore original hidden_dim
        args.hidden_dim = self._original_hidden_dim

        # Load layer_scalar if present
        layer_scalar_key = f"layers.{layer_num}.layer_scalar"
        if layer_scalar_key in state_dict:
            scalar_val = state_dict[layer_scalar_key].item()
            self.layer_scalar = scalar_val
        else:
            self.layer_scalar = 1.0

        # Per-layer input gating weights
        self._has_per_layer_gating = False
        gate_key = f"layers.{layer_num}.per_layer_input_gate.weight"
        if gate_key in state_dict and args.hidden_size_per_layer_input > 0:
            self._has_per_layer_gating = True
            self._init_per_layer_gating(args, state_dict, layer_num, mesh_device, dtype, weight_cache_path)

    def _init_per_layer_gating(self, args, state_dict, layer_num, mesh_device, dtype, weight_cache_path):
        """Initialize per-layer input gating weights."""
        layer_prefix = f"layers.{layer_num}"
        gating_dim = args.hidden_size_per_layer_input  # 256

        cache_path = args.weight_cache_path(dtype) if not args.dummy_weights else None

        def cache_name(name):
            if cache_path is None:
                return None
            return cache_path / f"{layer_prefix}.{name}"

        # per_layer_input_gate: Linear(hidden_size -> gating_dim)
        gate_w = state_dict[f"{layer_prefix}.per_layer_input_gate.weight"].transpose(-1, -2).unsqueeze(0).unsqueeze(0)
        self.per_layer_input_gate = ttnn.as_tensor(
            gate_w,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            cache_file_name=cache_name("per_layer_input_gate"),
        )

        # per_layer_projection: Linear(gating_dim -> hidden_size)
        proj_w = state_dict[f"{layer_prefix}.per_layer_projection.weight"].transpose(-1, -2).unsqueeze(0).unsqueeze(0)
        self.per_layer_projection = ttnn.as_tensor(
            proj_w,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            cache_file_name=cache_name("per_layer_projection"),
        )

        # post_per_layer_input_norm
        self.post_per_layer_input_norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            eps=args.norm_eps,
            state_dict=state_dict,
            state_dict_prefix=f"{layer_prefix}.",
            weight_cache_path=None if args.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            weight_key="post_per_layer_input_norm",
            add_unit_offset=args.rms_norm_add_unit_offset,
            is_distributed=False,
            tt_ccl=self.tt_ccl,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
        batch_size=1,
        per_layer_input=None,
    ) -> ttnn.Tensor:
        """
        Forward pass with Gemma 4 specific per-layer gating and layer scalar.

        Delegates the core attention + MLP + norms to the base TransformerBlock,
        then applies per-layer input gating and layer scalar on top.
        """
        # Run base transformer block forward (attention + norms + MLP + residuals)
        out = super().forward(
            x,
            current_pos,
            rot_mats_global=rot_mats_global,
            rot_mats_local=rot_mats_local,
            user_id=user_id,
            mode=mode,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            kv_cache=kv_cache,
            batch_size=batch_size,
        )

        # Per-layer input gating (requires embed_tokens_per_layer, not yet implemented)
        if self._has_per_layer_gating and per_layer_input is not None:
            skip_mem_cfg = self.args.get_residual_mem_config(mode, self.prefetcher)
            gating_residual = out

            # gate(h) -> gelu -> multiply by per_layer_input -> project -> norm -> residual
            gate_out = ttnn.linear(
                out,
                self.per_layer_input_gate,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            gate_out = ttnn.gelu(gate_out, fast_and_approx=True)
            gate_out = ttnn.multiply(gate_out, per_layer_input)
            gate_out = ttnn.linear(
                gate_out,
                self.per_layer_projection,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            gate_out = self.post_per_layer_input_norm(gate_out, mode)
            out = ttnn.add(gating_residual, gate_out, memory_config=skip_mem_cfg)
            ttnn.deallocate(gate_out)

        # Layer scalar
        if self.layer_scalar != 1.0:
            out = ttnn.multiply(out, self.layer_scalar)

        return out
