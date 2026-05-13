# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Hybrid Decoder Layer for Qwen3.6-27B on BH GLX 8×4 mesh — Task 7.

Dispatches between:
  - TtQwen36GatedAttention  (layer_types[i] == "full_attention")
  - TtQwen36DeltaNet        (layer_types[i] == "linear_attention")

Architecture
------------
  residual = x
  x = input_layernorm(x)              # DistributedNorm (shard → norm → gather), zero_centered=True
  if linear_attention:
    attn_out, new_dn_state, new_conv_state = deltanet(x, mode, state, conv)
  else:
    attn_out = attention(x, current_pos, rot_mats, mask, mode)
    new_dn_state = new_conv_state = None
  x = residual + attn_out
  residual = x
  x = post_attention_layernorm(x)    # DistributedNorm (shard → norm → gather), zero_centered=True
  x = residual + mlp(x)
  return x, new_dn_state, new_conv_state

Weight key conventions (from HF safetensors, short-key form):
  input_layernorm.weight           [H]
  post_attention_layernorm.weight  [H]
  mlp.gate_proj.weight             [intermediate, H]
  mlp.up_proj.weight               [intermediate, H]
  mlp.down_proj.weight             [H, intermediate]

  For full_attention layers:
    self_attn.q_proj.weight   [n_q*hd*2, H]
    self_attn.k_proj.weight   [n_kv*hd, H]
    self_attn.v_proj.weight   [n_kv*hd, H]
    self_attn.o_proj.weight   [H, n_q*hd]
    self_attn.q_norm.weight   [hd]
    self_attn.k_norm.weight   [hd]

  For linear_attention layers:
    linear_attn.in_proj_qkv.weight  [conv_dim, H]
    linear_attn.in_proj_z.weight    [value_dim, H]
    linear_attn.in_proj_a.weight    [n_v, H]
    linear_attn.in_proj_b.weight    [n_v, H]
    linear_attn.conv1d.weight       [conv_dim, 1, 4]
    linear_attn.A_log               [n_v]
    linear_attn.dt_bias             [n_v]
    linear_attn.norm.weight         [head_v_dim]
    linear_attn.out_proj.weight     [H, value_dim]
"""
from __future__ import annotations

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_6_galaxy.tt.distributed_norm import DistributedNorm
from models.demos.qwen3_6_galaxy.tt.llama_mlp import TtQwen36MLP

_TILE = 32


class TtQwen36DecoderLayer(LightweightModule):
    """Hybrid transformer decoder layer for Qwen3.6-27B on BH GLX 8×4 mesh.

    Dispatches between DeltaNet and GatedAttention based on
    args.linear_attention_pattern[layer_idx].

    Parameters
    ----------
    mesh_device : ttnn.MeshDevice
        Full 8×4 mesh.
    args : TtQwen36ModelArgs
        Model configuration.
    state_dict : dict
        Per-layer weights with short keys (see module docstring).
    layer_idx : int
        Layer index — used to read args.linear_attention_pattern[layer_idx].
    dtype : ttnn.DataType
        Activation dtype (default bfloat16).
    rope_setup : optional
        Qwen36RopeSetup shared across layers (T5). Passed through to attention.
    transformation_mats : optional
        Not used in standalone path; kept for API compatibility.
    paged_attention_config : optional
        Not used in standalone path.
    use_paged_kv_cache : bool
        Not used in standalone path.
    prefetcher : optional
        Not used in standalone path.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        args,
        state_dict: dict,
        layer_idx: int = 0,
        dtype=ttnn.bfloat16,
        rope_setup=None,
        transformation_mats=None,
        paged_attention_config=None,
        use_paged_kv_cache: bool = False,
        prefetcher=None,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.args = args
        self.layer_idx = layer_idx
        self.dtype = dtype

        # Dispatch: read layer type from pattern
        self.layer_type = args.linear_attention_pattern[layer_idx]

        # ------------------------------------------------------------------
        # Norms: DistributedNorm with HiFi4+fp32_dest_acc on H/4=1280 per col.
        # Using zero_centered=True (Qwen3NextRMSNorm: output = (1+w)*norm(x)).
        # ------------------------------------------------------------------
        self.input_layernorm = DistributedNorm(
            mesh_device=mesh_device,
            weight_torch=state_dict["input_layernorm.weight"].float(),
            eps=args.norm_eps,
            zero_centered=True,
        )
        self.post_attention_layernorm = DistributedNorm(
            mesh_device=mesh_device,
            weight_torch=state_dict["post_attention_layernorm.weight"].float(),
            eps=args.norm_eps,
            zero_centered=True,
        )

        # ------------------------------------------------------------------
        # Attention block: DeltaNet or GatedAttention
        # ------------------------------------------------------------------
        if self.layer_type == "linear_attention":
            from models.demos.qwen3_6_galaxy.tt.qwen36_deltanet import TtQwen36DeltaNet

            # Extract linear_attn.* keys as a flat dict for the DeltaNet constructor
            dn_sd = _extract_prefix(state_dict, "linear_attn.")
            self.attention = TtQwen36DeltaNet(
                mesh_device=mesh_device,
                args=args,
                state_dict=dn_sd,
                layer_num=layer_idx,
                dtype=dtype,
            )
        else:
            from models.demos.qwen3_6_galaxy.tt.llama_attention import TtQwen36GatedAttention

            # Extract self_attn.* keys as a flat dict for the attention constructor
            attn_sd = _extract_prefix(state_dict, "self_attn.")
            self.attention = TtQwen36GatedAttention(
                mesh_device=mesh_device,
                args=args,
                state_dict=attn_sd,
                layer_num=layer_idx,
                rope_setup=rope_setup,
                dtype=dtype,
            )

        # ------------------------------------------------------------------
        # MLP (SwiGLU, TP-sharded over 8 mesh rows)
        # ------------------------------------------------------------------
        self.mlp = TtQwen36MLP(
            mesh_device=mesh_device,
            args=args,
            state_dict=state_dict,
            dtype=dtype,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos=None,
        rot_mats=None,
        attention_mask=None,
        kv_cache=None,
        page_table=None,
        deltanet_state=None,
        deltanet_conv_state=None,
        mode: str = "prefill",
    ):
        """Hybrid decoder forward pass.

        Parameters
        ----------
        x : ttnn.Tensor
            [B, T, H] replicated across mesh.
        current_pos : int or ttnn.Tensor
            Current decode position (used for KV cache update in full_attention).
        rot_mats : (cos_tt, sin_tt) or None
            Rotary embedding tensors. Required for full_attention; ignored for
            linear_attention (DeltaNet does not use RoPE).
        attention_mask : ttnn.Tensor or None
            Causal mask for full_attention. DeltaNet ignores this.
        kv_cache : tuple or None
            (k_cache, v_cache) for the attention block.
        page_table : optional
            Paged attention table (not used in test path).
        deltanet_state : ttnn.Tensor or None
            Recurrent state for DeltaNet. None on first call.
        deltanet_conv_state : ttnn.Tensor or None
            Causal conv state for DeltaNet. None on first call.
        mode : str
            "prefill" or "decode".

        Returns
        -------
        (x_out, new_dn_state, new_conv_state)
            x_out:          [B, T, H] replicated across mesh.
            new_dn_state:   Updated DeltaNet recurrent state (or None for full_attention).
            new_conv_state: Updated DeltaNet conv state (or None for full_attention).
        """
        mem = ttnn.DRAM_MEMORY_CONFIG

        # ---  Pre-norm  ---
        orig_shape = list(x.shape)
        B_in = orig_shape[0]
        T_in = orig_shape[1] if len(orig_shape) == 3 else orig_shape[-2]
        H_in = orig_shape[-1]

        residual = x
        x_sharded = _shard_across_cols(x, self.mesh_device)
        x_normed_sharded = self.input_layernorm(x_sharded)
        x_sharded.deallocate(True)
        x_normed = _gather_from_cols(x_normed_sharded, self.mesh_device)
        x_normed_sharded.deallocate(True)
        # Tile-padding fix: _gather_from_cols may pad T to 32 when B*T < 32 (decode).
        # Slice back to the original sequence length so downstream ops see T_in.
        if list(x_normed.shape)[-2] != T_in:
            x_normed = ttnn.slice(x_normed, [0, 0, 0], [B_in, T_in, H_in], memory_config=mem)

        # ---  Attention block dispatch  ---
        if self.layer_type == "linear_attention":
            # DeltaNet: no RoPE, no KV cache, uses recurrent + conv state.
            # Returns updated (dn_state, conv_state) for the next layer.
            result = self.attention.forward(
                x_normed,
                mode=mode,
                recurrent_state=deltanet_state,
                conv_state=deltanet_conv_state,
                return_state=True,
            )
            attn_out_raw, new_dn_state, new_conv_state = result
            # fast_reduce_nc (used inside DeltaNet's _output_proj_and_reduce) always
            # returns tile-padded T (e.g., T=1 becomes T=32). Slice back to T_in
            # so subsequent layers receive the correct logical sequence length.
            raw_T = list(attn_out_raw.shape)[1] if len(list(attn_out_raw.shape)) == 3 else list(attn_out_raw.shape)[-2]
            if raw_T != T_in:
                attn_out = ttnn.slice(attn_out_raw, [0, 0, 0], [B_in, T_in, H_in], memory_config=mem)
                attn_out_raw.deallocate(True)
            else:
                attn_out = attn_out_raw
        else:
            # GatedAttention: needs RoPE tensors, supports KV cache.
            # Pass through the DeltaNet state unchanged (matching the reference
            # HybridDecoderLayer which does conv_state_new = conv_state for
            # full_attention layers so state flows correctly through the stack).
            attn_out = self.attention.forward(
                x_normed,
                current_pos=current_pos,
                rot_mats=rot_mats,
                user_id=0,
                mode=mode,
                page_table=page_table,
                kv_cache=kv_cache,
            )
            # Pass through DeltaNet states unchanged
            new_dn_state = deltanet_state
            new_conv_state = deltanet_conv_state

        x_normed.deallocate(True)

        # ---  First residual add  ---
        x = ttnn.add(residual, attn_out, memory_config=mem)
        residual.deallocate(True)
        attn_out.deallocate(True)

        # ---  Post-norm  ---
        residual2 = x
        x_sharded2 = _shard_across_cols(x, self.mesh_device)
        x_normed2_sharded = self.post_attention_layernorm(x_sharded2)
        x_sharded2.deallocate(True)
        x_normed2 = _gather_from_cols(x_normed2_sharded, self.mesh_device)
        x_normed2_sharded.deallocate(True)
        # Tile-padding fix: same as input_LN — slice back to T_in for decode.
        if list(x_normed2.shape)[-2] != T_in:
            x_normed2 = ttnn.slice(x_normed2, [0, 0, 0], [B_in, T_in, H_in], memory_config=mem)

        # ---  MLP  ---
        mlp_out = self.mlp.forward(x_normed2)
        x_normed2.deallocate(True)
        # Tile-padding fix: fast_reduce_nc inside TtQwen36MLP computes its output
        # shape from padded_shape (a TTNN quirk): input padded [8, 32, H] → output
        # logical [1, 32, H] for dim=0 reduction, even though the TRUE sequence
        # length is T_in=1.  Slice back to T_in before the residual add so that
        # subsequent layers receive the correct logical sequence length.
        mlp_T = list(mlp_out.shape)[-2]
        if mlp_T != T_in:
            mlp_out_sliced = ttnn.slice(mlp_out, [0, 0, 0], [B_in, T_in, H_in], memory_config=mem)
            mlp_out.deallocate(True)
            mlp_out = mlp_out_sliced

        # ---  Second residual add  ---
        x_out = ttnn.add(residual2, mlp_out, memory_config=mem)
        residual2.deallocate(True)
        mlp_out.deallocate(True)

        return x_out, new_dn_state, new_conv_state


# ---------------------------------------------------------------------------
# Helper: extract sub-dict with prefix stripped
# ---------------------------------------------------------------------------


def _extract_prefix(d: dict, prefix: str) -> dict:
    """Return a new dict with only keys starting with prefix, with prefix stripped.

    Example:
        _extract_prefix({"self_attn.q_proj.weight": w, "mlp.gate_proj.weight": w2},
                        "self_attn.")
        → {"q_proj.weight": w}
    """
    return {k[len(prefix) :]: v for k, v in d.items() if k.startswith(prefix)}


# ---------------------------------------------------------------------------
# Helpers: shard/gather for DistributedNorm interface
# ---------------------------------------------------------------------------


def _shard_across_cols(x: ttnn.Tensor, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    """Shard a replicated [B, T, H] tensor across 4 mesh columns.

    Steps:
      1. Convert to host (first device).
      2. Reshape to [B, 1, T, H].
      3. Upload with ShardTensor2dMesh(dims=(None, -1)) → each col gets [B, 1, T, H/4].

    Returns a 4-D TTNN tensor sharded across cluster_axis=1.
    """
    # Bring to host (take first of 32 replicated copies)

    cluster_shape = list(mesh_device.shape)  # [8, 4]
    x_cpu = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    x_cpu = x_cpu[0:1]  # [B, T, H] (first device)

    # Reshape to 4-D
    B, T, H = x_cpu.shape
    x_4d = x_cpu.unsqueeze(1)  # [B, 1, T, H]

    return ttnn.from_torch(
        x_4d,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, -1), mesh_shape=cluster_shape),
    )


def _gather_from_cols(x_sharded: ttnn.Tensor, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    """Gather a column-sharded [B, 1, T, H/4] tensor back to replicated [B, T, H].

    Inverse of _shard_across_cols. After gathering, reshapes from [B, 1, T, H] to [B, T, H]
    and replicates across all mesh devices.
    """
    cluster_shape = list(mesh_device.shape)  # [8, 4]

    # ConcatMesh2dToTensor: concat dim=0 (rows) and dim=-1 (cols)
    # Result: [8*B, 1, T, H] — take rows belonging to first row of mesh
    # then take cols 0..H (all), picking first replicated batch
    x_cpu = ttnn.to_torch(
        x_sharded,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 3), mesh_shape=cluster_shape),
    )
    # x_cpu: [8*B, 1, T, H] → take first B items (from first mesh row)
    B = x_cpu.shape[0] // cluster_shape[0]
    x_cpu = x_cpu[:B, 0, :, :]  # [B, T, H]

    # For decode (B*T < 32), TILE_LAYOUT pads the row dim, corrupting the logical
    # T downstream.  Upload row-major then convert to tile, matching the embed
    # decode path.
    T = x_cpu.shape[1]
    if B * T < 32:
        x_tt = ttnn.from_torch(
            x_cpu,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        return ttnn.to_layout(x_tt, ttnn.TILE_LAYOUT)

    return ttnn.from_torch(
        x_cpu,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
