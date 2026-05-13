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
            # Derive paged cache params from args or explicit constructor params
            _use_paged = (
                use_paged_kv_cache or (paged_attention_config is not None) or getattr(args, "use_paged_kv_cache", False)
            )
            _pac = paged_attention_config or getattr(args, "paged_attention_config", None)
            _block_size = _pac.block_size if _pac is not None else 64
            _max_blocks = _pac.max_num_blocks if _pac is not None else None

            self.attention = TtQwen36GatedAttention(
                mesh_device=mesh_device,
                args=args,
                state_dict=attn_sd,
                layer_num=layer_idx,
                rope_setup=rope_setup,
                dtype=dtype,
                use_paged_kv_cache=_use_paged,
                block_size=_block_size,
                max_num_blocks=_max_blocks,
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
    """Shard a replicated [B, T, H] tensor across 4 mesh columns — fully on-device.

    Uses ttnn.mesh_partition (the no-reduce inverse of all_gather) on cluster_axis=1.
    Trace-safe: no host roundtrip. The input must already be replicated across the
    mesh; we reshape [B, T, H] → [B, 1, T, H] as a view, then partition dim=-1
    across the 4 cols so each col owns [B, 1, T, H/4]. mesh_partition allocates
    a fresh buffer, so the intermediate reshape view can safely go out of scope.
    """
    B, T, H = x.shape[0], x.shape[1], x.shape[2]
    x_4d = ttnn.reshape(x, ttnn.Shape([B, 1, T, H]))
    return ttnn.mesh_partition(
        x_4d,
        dim=-1,
        cluster_axis=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _gather_from_cols(x_sharded: ttnn.Tensor, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    """Gather a column-sharded [B, 1, T, H/4] tensor back to replicated [B, T, H]
    — fully on-device.

    Inverse of _shard_across_cols. Uses ttnn.all_gather on cluster_axis=1 (cols)
    at dim=-1, then reshapes the 4-D result to 3-D. Rows already replicate the
    same content (DistributedNorm replicates across rows), so the result is
    fully replicated across the 8×4 mesh after the col gather.

    Trace-safe: no host writes. The downstream `ttnn.slice(...)` in
    `TtQwen36DecoderLayer.forward` corrects any tile-padding T discrepancy
    (e.g., T=1 decode where the tile dim is internally padded to 32).
    """
    # All-gather along the col axis: [B, 1, T, H/4] per col → [B, 1, T, H]
    # replicated within row. Since rows started identical, output is globally replicated.
    x_gathered = ttnn.all_gather(
        x_sharded,
        dim=-1,
        num_links=1,
        cluster_axis=1,
        topology=ttnn.Topology.Linear,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # ttnn.reshape returns a VIEW that aliases x_gathered's buffer. If we just
    # return the view, x_gathered goes out of scope and its destructor frees
    # the device buffer underneath the still-live view — exactly the
    # use-after-free pattern fixed in commit f634f292720 (qknorm reshape).
    # ttnn.clone copies the data into a fresh, independent buffer, so the
    # source can be safely released. Cost: one extra [B, T, H] copy per gather.
    B = x_gathered.shape[0]
    T = x_gathered.shape[2]
    H = x_gathered.shape[3]
    x_view = ttnn.reshape(x_gathered, ttnn.Shape([B, T, H]))
    x_out = ttnn.clone(x_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x_gathered.deallocate(True)
    return x_out
