# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
On-device **scaled dot-product attention** for Mistral4-style Q-LoRA + compressed KV.

This reuses the same **projection** path as
:func:`~models.tt_transformers.tt.mistral_small_4.attention_full.attention_forward_hybrid_bf16`
(device ``q_a→norm→q_b``, ``kv_a``, ``kv_norm→kv_b``). The SDPA core is always on device; split-head prep and RoPE
can run on device in the no-cache path.

The matmul + softmax core uses ``ttnn.transformer.scaled_dot_product_attention`` in the same spirit as
``models/tt_transformers/tt/attention.py`` (prefill SDPA) and ``models/tt_transformers/tt/multimodal/mistral_24b/vision_attention.py``.

Constraints (v1):
    - ``batch_size == 1`` (matches current parity tests).
    - ``attention_mask``: ``None`` matches HF eager **without** additive mask (SDPA ``is_causal=False``). A **4D**
      additive mask (e.g. from ``transformers.masking_utils.create_causal_mask``) is uploaded as ``attn_mask``
      to ``ttnn.transformer.scaled_dot_product_attention`` with ``is_causal=False`` (ttnn forbids ``is_causal``
      together with ``attn_mask``). Last two dims may be **rectangular** ``[S_q, S_kv]`` when using a KV cache.
      Head dimension must be ``1`` (broadcast) or ``num_attention_heads``. ``BlockMask`` / non-tensor masks are
      not supported.
    - When ``v_head_dim != qk_head_dim``, ``value_states`` are **right-padded** to ``qk_head_dim`` for SDPA,
      then the output is sliced back to ``v_head_dim`` before ``o_proj`` (same idea as HF flash padding).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import ttnn
from models.tt_transformers.tt.mistral_small_4.attention_slice import (
    attention_kv_b_and_k_rot_from_compressed_bf16,
    attention_q_after_q_bottleneck_bf16,
)
from models.tt_transformers.tt.mistral_small_4.linear import linear_bf16_no_bias

_SDPA_CK_PREFIX = "_mistral_small_4_sdpa_compute_kernel_config"


def _sdpa_compute_kernel_config(mesh_device):
    # Blackhole SDPA parity tests use HiFi2 + no fp32 dest acc (see ``test_scaled_dot_product_attention_sprint``).
    # Wormhole text prefill often uses HiFi4 for SDPA; keep that path for non-BH devices.
    key = f"{_SDPA_CK_PREFIX}_{mesh_device.arch()}"
    cfg = getattr(mesh_device, key, None)
    if cfg is None:
        if ttnn.device.is_blackhole(mesh_device):
            cfg = ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
        else:
            cfg = ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
        setattr(mesh_device, key, cfg)
    return cfg


def _sdpa_program_config(*, seq_len: int, mesh_device) -> ttnn.SDPAProgramConfig:
    # Match ``test_sdpa_prefill`` / sprint tests: grid must match the device, not a fixed (8, 8).
    qk = max(32, min(256, ((seq_len + 31) // 32) * 32))
    grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid,
        q_chunk_size=qk,
        k_chunk_size=qk,
        exp_approx_mode=False,
    )


def _attention_mask_4d_to_tt_attn_mask(
    mesh_device,
    attention_mask_4d: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    query_seq_len: int,
    key_seq_len: int,
) -> ttnn.Tensor:
    """
    HF / eager additive mask ``[B, H_bcast, S_q, S_kv]`` → TILE ``ttnn`` tensor for SDPA ``attn_mask``.

    Crops the last two dimensions to ``query_seq_len`` × ``key_seq_len`` (prefill uses ``S_q == S_kv``;
    decode-with-cache uses a short query length and a longer key length).
    """
    m = attention_mask_4d
    if m.ndim != 4:
        raise ValueError(f"attention_mask must be 4D, got shape={tuple(m.shape)} ndim={m.ndim}")
    b, h, sq, sk = (int(m.shape[0]), int(m.shape[1]), int(m.shape[2]), int(m.shape[3]))
    if b != int(batch_size):
        raise ValueError(f"mask batch {b} != hidden_states batch {batch_size}")
    if h not in (1, int(n_heads)):
        raise ValueError(f"mask head dim must be 1 (broadcast) or num_heads={n_heads}, got shape={tuple(m.shape)}")
    if sq < int(query_seq_len) or sk < int(key_seq_len):
        raise ValueError(
            f"mask seq dims ({sq}, {sk}) smaller than required ({query_seq_len}, {key_seq_len}); cannot crop safely"
        )
    m = m[:, :, : int(query_seq_len), : int(key_seq_len)].to(device=torch.device("cpu")).contiguous()
    # Materialize head broadcast explicitly — some SDPA paths are safer with ``[B, nh, S, S]`` than ``[B, 1, S, S]``.
    if int(m.shape[1]) == 1 and int(n_heads) != 1:
        m = m.expand(int(batch_size), int(n_heads), int(query_seq_len), int(key_seq_len)).contiguous()
    m = m.to(dtype=torch.bfloat16)
    return ttnn.from_torch(
        m,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        pad_value=0.0,
    )


def _torch_states_to_tt_nhsd(
    mesh_device,
    x_bshd: torch.Tensor,
    *,
    n_heads: int,
    head_dim: int,
) -> ttnn.Tensor:
    """``[B,S,N*D]`` bf16 → ``[1,N,S,D]`` TILE on device (``B`` must be 1)."""
    b, s, rest = x_bshd.shape
    if b != 1:
        raise ValueError("attention_forward_device_sdpa_bf16 currently requires batch_size==1")
    if int(rest) != n_heads * head_dim:
        raise ValueError(f"expected last dim {n_heads * head_dim}, got {rest}")
    x = x_bshd.to(torch.bfloat16).reshape(1, s, n_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    return ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        pad_value=0.0,
    )


def _torch_rot_mats_to_tt(
    mesh_device, cos: torch.Tensor, sin: torch.Tensor, *, seq_len: int, head_dim: int
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """
    Convert HF rotary embeddings (cos/sin) to ``ttnn`` tensors for ``ttnn.experimental.rotary_embedding_hf``.

    Accepts common HF shapes (e.g. ``[B, S, D]``, ``[B, 1, S, D]``, ``[B, S, 1, D]``) and converts to
    ``[1, 1, S, D]`` bf16 TILE on device.
    """

    def _to_11sd(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # [1,1,S,D]
        elif x.ndim == 3:
            x = x.unsqueeze(1)  # [B,1,S,D]
        elif x.ndim == 4:
            # [B,1,S,D] or [B,S,1,D] -> normalize to [B,1,S,D]
            if int(x.shape[1]) != 1 and int(x.shape[2]) == 1:
                x = x.transpose(1, 2)
        else:
            raise ValueError(f"unsupported rotary mat shape={tuple(x.shape)} ndim={x.ndim}")
        return x[:1, :1, : int(seq_len), : int(head_dim)].contiguous()

    cos_11sd = _to_11sd(cos).to(dtype=torch.bfloat16)
    sin_11sd = _to_11sd(sin).to(dtype=torch.bfloat16)

    cos_tt = ttnn.from_torch(
        cos_11sd,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        pad_value=0.0,
    )
    sin_tt = ttnn.from_torch(
        sin_11sd,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        pad_value=0.0,
    )
    return cos_tt, sin_tt


def attention_forward_device_sdpa_bf16(
    mesh_device,
    hidden_states_bsh: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    position_ids: torch.LongTensor,
    attn: torch.nn.Module,
    attention_mask: torch.Tensor | None = None,
    *,
    past_key_values=None,
    layer_idx: int | None = None,
) -> torch.Tensor:
    """
    Mistral4 attention with **projections + o_proj on device** and **SDPA on device** (``ttnn.transformer``).

    In the no-cache path (``past_key_values is None``), split-head prep + RoPE use
    ``ttnn.experimental.rotary_embedding_hf`` **only when** ``qk_rope_head_dim`` is divisible by **64**
    and ``config.rope_interleave`` is **False** (the kernel path does not implement interleaved RoPE).
    Otherwise RoPE stays on CPU (torch ``apply_rotary_pos_emb`` / ``apply_rotary_pos_emb_interleave``)
    while SDPA remains on device.

    In the HF cache path we keep the legacy torch RoPE so that ``DynamicCache.update`` continues to receive
    rotary-applied keys.

    ``past_key_values`` / ``layer_idx``: same semantics as :func:`~models.tt_transformers.tt.mistral_small_4.attention_full.attention_forward_hybrid_bf16`
    (HF ``Cache.update`` before SDPA).
    """
    from transformers.models.mistral4.modeling_mistral4 import (
        apply_rotary_pos_emb,
        apply_rotary_pos_emb_interleave,
        get_llama_4_attn_scale,
    )

    cfg = attn.config
    if getattr(cfg, "q_lora_rank", None) is None:
        raise ValueError("attention_forward_device_sdpa_bf16 requires q_lora_rank (Q-LoRA path)")

    batch_size, seq_length, _hidden = hidden_states_bsh.shape
    if batch_size != 1:
        raise ValueError("attention_forward_device_sdpa_bf16 currently requires batch_size==1")

    nh = int(attn.num_heads)
    qk_nope = int(attn.qk_nope_head_dim)
    qk_rope = int(attn.qk_rope_head_dim)
    qk_head = int(attn.qk_head_dim)
    v_head = int(attn.v_head_dim)

    q_flat = attention_q_after_q_bottleneck_bf16(
        mesh_device,
        hidden_states_bsh,
        attn.q_a_proj.weight,
        attn.q_a_layernorm.weight.data,
        float(attn.q_a_layernorm.variance_epsilon),
        attn.q_b_proj.weight,
    )
    compressed = linear_bf16_no_bias(mesh_device, hidden_states_bsh, attn.kv_a_proj_with_mqa.weight)
    k_after_b, k_rot = attention_kv_b_and_k_rot_from_compressed_bf16(
        mesh_device,
        compressed,
        int(attn.kv_lora_rank),
        attn.kv_a_layernorm.weight.data,
        float(attn.kv_a_layernorm.variance_epsilon),
        attn.kv_b_proj.weight,
    )

    query_shape = (batch_size, seq_length, nh, qk_head)
    key_shape = (batch_size, seq_length, nh, qk_nope + v_head)

    if past_key_values is not None:
        # Legacy torch path for HF cache interop.
        q_states = q_flat.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [qk_nope, qk_rope], dim=-1)

        k_pass = k_after_b.view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [qk_nope, v_head], dim=-1)

        k_rot = k_rot.reshape(batch_size, 1, seq_length, qk_rope)

        cos, sin = position_embeddings
        if getattr(cfg, "rope_interleave", False):
            q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        else:
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        query_states = query_states * get_llama_4_attn_scale(
            position_ids,
            cfg.rope_parameters.get("llama_4_scaling_beta"),
            cfg.rope_parameters.get("original_max_position_embeddings"),
        ).to(query_states.dtype)

        li = int(attn.layer_idx) if layer_idx is None else int(layer_idx)
        key_states, value_states = past_key_values.update(key_states, value_states, li)

        seq_len_kv = int(key_states.shape[2])
        if v_head != qk_head:
            value_states = F.pad(value_states, (0, qk_head - v_head))

        q_bsh = query_states.transpose(1, 2).reshape(batch_size, seq_length, nh * qk_head).contiguous()
        k_bsh = key_states.transpose(1, 2).reshape(batch_size, seq_len_kv, nh * qk_head).contiguous()
        v_bsh = value_states.transpose(1, 2).reshape(batch_size, seq_len_kv, nh * qk_head).contiguous()

        q_tt = _torch_states_to_tt_nhsd(mesh_device, q_bsh, n_heads=nh, head_dim=qk_head)
        k_tt = _torch_states_to_tt_nhsd(mesh_device, k_bsh, n_heads=nh, head_dim=qk_head)
        v_tt = _torch_states_to_tt_nhsd(mesh_device, v_bsh, n_heads=nh, head_dim=qk_head)
    else:
        seq_len_kv = seq_length
        # ``rotary_embedding_hf`` requires rope head dim divisible by 64 (not 32).
        # Interleaved RoPE matches HF only on the torch path (not ``rotary_embedding_hf`` yet).
        if int(qk_rope) % 64 != 0 or getattr(cfg, "rope_interleave", False):
            q_states = q_flat.view(query_shape).transpose(1, 2)
            q_pass, q_rot = torch.split(q_states, [qk_nope, qk_rope], dim=-1)

            k_pass = k_after_b.view(key_shape).transpose(1, 2)
            k_pass, value_states = torch.split(k_pass, [qk_nope, v_head], dim=-1)

            k_rot_t = k_rot.reshape(batch_size, 1, seq_length, qk_rope)
            cos, sin = position_embeddings
            if getattr(cfg, "rope_interleave", False):
                q_rot, k_rot_t = apply_rotary_pos_emb_interleave(q_rot, k_rot_t, cos, sin)
            else:
                q_rot, k_rot_t = apply_rotary_pos_emb(q_rot, k_rot_t, cos, sin)
            k_rot_t = k_rot_t.expand(*k_pass.shape[:-1], -1)

            query_states = torch.cat((q_pass, q_rot), dim=-1)
            key_states = torch.cat((k_pass, k_rot_t), dim=-1)
            query_states = query_states * get_llama_4_attn_scale(
                position_ids,
                cfg.rope_parameters.get("llama_4_scaling_beta"),
                cfg.rope_parameters.get("original_max_position_embeddings"),
            ).to(query_states.dtype)

            if v_head != qk_head:
                value_states = F.pad(value_states, (0, qk_head - v_head))

            q_bsh = query_states.transpose(1, 2).reshape(batch_size, seq_length, nh * qk_head).contiguous()
            k_bsh = key_states.transpose(1, 2).reshape(batch_size, seq_len_kv, nh * qk_head).contiguous()
            v_bsh = value_states.transpose(1, 2).reshape(batch_size, seq_len_kv, nh * qk_head).contiguous()

            q_tt = _torch_states_to_tt_nhsd(mesh_device, q_bsh, n_heads=nh, head_dim=qk_head)
            k_tt = _torch_states_to_tt_nhsd(mesh_device, k_bsh, n_heads=nh, head_dim=qk_head)
            v_tt = _torch_states_to_tt_nhsd(mesh_device, v_bsh, n_heads=nh, head_dim=qk_head)
        else:
            # Device split-head prep + rotary embedding via ``ttnn.experimental.rotary_embedding_hf``.
            cos, sin = position_embeddings
            cos_tt, sin_tt = _torch_rot_mats_to_tt(mesh_device, cos, sin, seq_len=seq_length, head_dim=qk_rope)

            q_tt_full = ttnn.from_torch(
                q_flat.to(torch.bfloat16).reshape(1, seq_length, nh, qk_head).permute(0, 2, 1, 3).contiguous(),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                pad_value=0.0,
            )
            q_pass_tt, q_rot_tt = ttnn.split(q_tt_full, (qk_nope, qk_rope), dim=-1)
            ttnn.deallocate(q_tt_full)

            kv_tt_full = ttnn.from_torch(
                k_after_b.to(torch.bfloat16)
                .reshape(1, seq_length, nh, qk_nope + v_head)
                .permute(0, 2, 1, 3)
                .contiguous(),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                pad_value=0.0,
            )
            k_pass_tt, v_tt = ttnn.split(kv_tt_full, (qk_nope, v_head), dim=-1)
            ttnn.deallocate(kv_tt_full)

            k_rot_tt = ttnn.from_torch(
                k_rot.to(torch.bfloat16).reshape(1, 1, seq_length, qk_rope).contiguous(),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                pad_value=0.0,
            )
            k_rot_tt = ttnn.repeat(k_rot_tt, ttnn.Shape((1, nh, 1, 1)))

            q_rot_tt = ttnn.experimental.rotary_embedding_hf(q_rot_tt, cos_tt, sin_tt, is_decode_mode=False)
            k_rot_tt = ttnn.experimental.rotary_embedding_hf(k_rot_tt, cos_tt, sin_tt, is_decode_mode=False)
            ttnn.deallocate(cos_tt)
            ttnn.deallocate(sin_tt)

            q_tt = ttnn.concat([q_pass_tt, q_rot_tt], dim=-1)
            k_tt = ttnn.concat([k_pass_tt, k_rot_tt], dim=-1)
            ttnn.deallocate(q_pass_tt)
            ttnn.deallocate(q_rot_tt)
            ttnn.deallocate(k_pass_tt)
            ttnn.deallocate(k_rot_tt)

            scale = get_llama_4_attn_scale(
                position_ids,
                cfg.rope_parameters.get("llama_4_scaling_beta"),
                cfg.rope_parameters.get("original_max_position_embeddings"),
            ).to(dtype=torch.bfloat16)
            scale_tt = ttnn.from_torch(
                scale.reshape(1, 1, seq_length, 1).contiguous(),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                pad_value=0.0,
            )
            q_tt = ttnn.mul(q_tt, scale_tt)
            ttnn.deallocate(scale_tt)

            if v_head != qk_head:
                v_tt = ttnn.pad(v_tt, [(0, 0), (0, 0), (0, 0), (0, qk_head - v_head)], value=0.0)

    tt_attn_mask = None
    if attention_mask is not None:
        tt_attn_mask = _attention_mask_4d_to_tt_attn_mask(
            mesh_device,
            attention_mask,
            batch_size=batch_size,
            n_heads=nh,
            query_seq_len=seq_length,
            key_seq_len=int(seq_len_kv),
        )

    # With ``attn_mask``, ``is_causal`` must be ``False`` (ttnn fatal otherwise). Unmasked HF eager path is also
    # non-causal (no additive mask), so we always use ``is_causal=False`` here.
    sdpa_prog_len = max(seq_length, int(seq_len_kv))
    sdpa_out = ttnn.transformer.scaled_dot_product_attention(
        q_tt,
        k_tt,
        v_tt,
        attn_mask=tt_attn_mask,
        is_causal=False,
        scale=float(attn.scaling),
        program_config=_sdpa_program_config(seq_len=sdpa_prog_len, mesh_device=mesh_device),
        compute_kernel_config=_sdpa_compute_kernel_config(mesh_device),
    )
    ttnn.deallocate(q_tt)
    ttnn.deallocate(k_tt)
    ttnn.deallocate(v_tt)
    if tt_attn_mask is not None:
        ttnn.deallocate(tt_attn_mask)

    y = ttnn.to_torch(sdpa_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    ttnn.deallocate(sdpa_out)
    if y.is_sparse:
        y = y.to_dense()
    # Drop leading mesh / wrapper singletons only — keep ``[B, num_heads, S, head_dim]`` (B==1).
    while y.ndim > 4 and int(y.shape[0]) == 1:
        y = y.squeeze(0)
    if y.ndim == 3:
        y = y.unsqueeze(0)
    if y.ndim != 4:
        raise ValueError(
            f"expected SDPA host tensor ndim 4 [B, heads, S, dim], got shape={tuple(y.shape)} ndim={y.ndim}"
        )
    # Drop tile-padded sequence / head_dim slots (same as ``test_sdpa_prefill`` / ``test_scaled_dot_product_attention_sprint``).
    y = y[:, :, :seq_length, :qk_head]
    y = y[..., :v_head].contiguous()
    attn_bshv = y.transpose(1, 2).reshape(batch_size, seq_length, nh * v_head).contiguous()

    return linear_bf16_no_bias(mesh_device, attn_bshv, attn.o_proj.weight)
