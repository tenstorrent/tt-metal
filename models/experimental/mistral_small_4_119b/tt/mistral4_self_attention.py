# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Mistral4 multi-latent self-attention (prefill) for Small 4 text.

Matches HF ``Mistral4Attention.forward`` with eager-style attention:

* **TTNN**: low-rank Q path, ``kv_a`` RMSNorm, ``kv_b_proj``, RoPE (including interleave),
  Llama-4 position scaling on Q, ``ttnn.transformer.scaled_dot_product_attention``, ``o_proj``.

``position_embeddings`` (cos, sin) are normally produced on the host (HF
``Mistral4RotaryEmbedding``). :class:`~models.experimental.mistral_small_4_119b.tt.text_backbone.TtMistral4DecoderSequence`
can either upload host cos/sin **once** per forward (``use_device_rotary_embedding_table=False``)
or use a persistent **device RoPE table** + :func:`ttnn.embedding` (``True``; no host cos/sin
for the TT path). :class:`TtMistral4SelfAttentionPrefill` still accepts host tensors or
``position_embeddings_tt`` when invoked directly.
``position_ids`` is used for the Llama-4 scaling term (small host tensor → device) and for
table gather when enabled.

**KV / decode (bring-up)**:
:class:`TtMistral4SelfAttentionPrefill` exposes :meth:`forward_prefill_with_kv` and
:meth:`forward_decode_extend_kv`, which snapshot K/V after prefill and extend the cache with
one new token using ``ttnn.concat`` + the same prefill-style SDPA (no ``paged_update_cache`` yet).

``weight_sd`` must be a ``Mistral4Attention.state_dict()``-style map (``q_a_proj.weight``, …).
"""

from __future__ import annotations

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule


def _torch_for_ttnn_upload(t: torch.Tensor) -> torch.Tensor:
    """
    Hub ``safetensors`` may use float8 / float32 linear weights while TTNN host upload expects
    dtypes the pybind path accepts as BF16 tiles. HF ``load_state_dict`` may dequantize
    separately; we slice the raw checkpoint for device weights, so normalize here.
    """
    t = torch.as_tensor(t).detach()
    if t.device.type != "cpu":
        t = t.cpu()
    t = t.contiguous()
    if t.dtype == torch.bfloat16:
        return t
    if t.dtype in (torch.float32, torch.float16):
        return t.to(torch.bfloat16)
    if t.is_floating_point():
        return t.to(torch.float32).to(torch.bfloat16)
    return t.to(torch.float32).to(torch.bfloat16)


def _linear_weight_ttnn(W: torch.Tensor) -> torch.Tensor:
    """``nn.Linear`` ``[out, in]`` → layout for ``ttnn.linear`` (see vision TT)."""
    return _torch_for_ttnn_upload(W).T.contiguous()


def _heads_1hsd_from_linear_11sh_flat(
    x_11sh: ttnn.Tensor,
    *,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> ttnn.Tensor:
    """
    Match HF ``hidden.view(B, S, num_heads, D).transpose(1, 2)`` (``[1,1,S,H*D]`` → ``[1,H,S,D]``).

    A naive ``reshape`` on TILE tensors does **not** follow PyTorch row-major head grouping; we go
    through ROW_MAJOR, split ``H*D`` into ``(H, D)``, then ``permute`` heads before the sequence dim.
    """
    x_rm = ttnn.to_layout(x_11sh, ttnn.ROW_MAJOR_LAYOUT)
    x5 = ttnn.reshape(x_rm, (1, 1, seq_len, num_heads, head_dim))
    # 5D ``(1, 1, S, H, D)`` → permute needs 5 entries: move H before S, keep D last, singleton in dim 3.
    x5p = ttnn.permute(x5, (0, 3, 2, 1, 4))
    x_hsd = ttnn.reshape(x5p, (1, num_heads, seq_len, head_dim))
    return ttnn.to_layout(x_hsd, ttnn.TILE_LAYOUT)


def _rms_weight_ttnn(w1d: torch.Tensor, device) -> ttnn.Tensor:
    w = _torch_for_ttnn_upload(w1d).reshape(1, 1, 1, -1).contiguous()
    return ttnn.from_torch(
        w,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device=device),
    )


def _rotate_half_ttnn(x: ttnn.Tensor) -> ttnn.Tensor:
    last_dim = x.shape[-1]
    half = last_dim // 2
    x1 = ttnn.slice(x, (0, 0, 0, 0), (x.shape[0], x.shape[1], x.shape[2], half))
    x2 = ttnn.slice(x, (0, 0, 0, half), (x.shape[0], x.shape[1], x.shape[2], last_dim))
    neg_x2 = ttnn.mul(x2, -1.0)
    return ttnn.concat([neg_x2, x1], dim=-1)


def _rope_interleave_rearrange_ttnn(x: ttnn.Tensor) -> ttnn.Tensor:
    """HF ``apply_rotary_pos_emb_interleave`` reshape (swap half-channels)."""
    b, h, s, d = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    half = d // 2
    x5 = ttnn.reshape(x, (b, h, s, half, 2))
    x5 = ttnn.permute(x5, (0, 1, 2, 4, 3))
    return ttnn.reshape(x5, (b, h, s, d))


def _apply_rotary_ttnn(
    q_rot: ttnn.Tensor,
    k_rot: ttnn.Tensor,
    cos_11sd: ttnn.Tensor,
    sin_11sd: ttnn.Tensor,
    *,
    interleave: bool,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """RoPE on the rotary slices only (HF ``apply_rotary_pos_emb*`` without extra unsqueeze)."""
    if interleave:
        q_rot = _rope_interleave_rearrange_ttnn(q_rot)
        k_rot = _rope_interleave_rearrange_ttnn(k_rot)
    q_embed = ttnn.add(ttnn.mul(q_rot, cos_11sd), ttnn.mul(_rotate_half_ttnn(q_rot), sin_11sd))
    k_embed = ttnn.add(ttnn.mul(k_rot, cos_11sd), ttnn.mul(_rotate_half_ttnn(k_rot), sin_11sd))
    return q_embed, k_embed


def upload_mistral4_rotary_cos_sin_to_mesh(
    mesh_device,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """
    Upload HF ``(cos, sin)`` to replicated TILE tensors ``[B,1,S,D]`` for :func:`_apply_rotary_ttnn`.

    Use when cos/sin are shared across several decoder layers; the caller must
    :func:`ttnn.deallocate` both tensors after the last consumer.
    """
    cos_t, sin_t = position_embeddings
    cos_t = cos_t.to(dtype=torch.bfloat16)
    sin_t = sin_t.to(dtype=torch.bfloat16)
    if cos_t.dim() == 3:
        cos_t = cos_t.unsqueeze(1)
        sin_t = sin_t.unsqueeze(1)
    elif cos_t.dim() != 4:
        raise ValueError(f"Expected cos rank 3 or 4, got shape {tuple(cos_t.shape)}")
    mapper = ttnn.ReplicateTensorToMesh(mesh_device=mesh_device)
    cos_tt = ttnn.from_torch(
        cos_t,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=mapper,
    )
    sin_tt = ttnn.from_torch(
        sin_t,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=mapper,
    )
    return cos_tt, sin_tt


def _llama4_attn_scale_torch(
    position_ids: torch.Tensor,
    *,
    beta: float | None,
    original_max_position_embeddings: int | None,
) -> torch.Tensor:
    """Host-only helper: ``get_llama_4_attn_scale`` (tiny tensor, uploaded to device)."""
    if beta is None or original_max_position_embeddings is None:
        return torch.ones(
            position_ids.shape[0],
            1,
            position_ids.shape[1],
            1,
            dtype=torch.bfloat16,
            device=position_ids.device,
        )
    scaling = 1.0 + float(beta) * torch.log(
        1.0 + torch.floor(position_ids.float() / float(original_max_position_embeddings))
    )
    return scaling[:, None, :, None].to(dtype=torch.bfloat16)


class TtMistral4SelfAttentionPrefill(LightweightModule):
    """
    Prefill self-attention (layer 0 API; weights are layer-local ``state_dict`` keys).

    Args:
        device: Mesh or single device used for TTNN ops.
        config: ``Mistral4Config`` (``text_config`` from the multimodal checkpoint).
        weight_sd: ``Mistral4Attention`` weights as torch tensors (CPU).
    """

    def __init__(self, device, config, weight_sd: dict[str, torch.Tensor]):
        super().__init__()
        self.device = device
        self.cfg = config
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.scaling = float(self.qk_head_dim**-0.5)
        self.rope_interleave = bool(getattr(config, "rope_interleave", False))
        self.eps = float(getattr(config, "rms_norm_eps", 1e-6))

        mapper = ttnn.ReplicateTensorToMesh(mesh_device=device)

        def _as_mm(name: str):
            t = _linear_weight_ttnn(weight_sd[name])
            return ttnn.from_torch(
                t,
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        self.w_q_a = _as_mm("q_a_proj.weight")
        self.w_q_b = _as_mm("q_b_proj.weight")
        self.w_kv_a = _as_mm("kv_a_proj_with_mqa.weight")
        self.w_o = _as_mm("o_proj.weight")
        self.w_kv_b = _as_mm("kv_b_proj.weight")

        self.w_q_a_ln = _rms_weight_ttnn(weight_sd["q_a_layernorm.weight"], device)
        self.w_kv_a_ln = _rms_weight_ttnn(weight_sd["kv_a_layernorm.weight"], device)

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        try:
            sdpa_grid = device.compute_with_storage_grid_size()
        except Exception:
            sdpa_grid = ttnn.CoreCoord(8, 8)
        self._sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=sdpa_grid,
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        self._sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _qkv_scaled_after_rope(
        self,
        hidden_11SH: ttnn.Tensor,
        *,
        position_ids: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_embeddings_tt: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, int]:
        """
        Projections through RoPE and Llama-4 Q scale; returns
        ``(query_states, key_states, v_states_for_sdpa, seq_len)``.
        """
        _b1, _b2, seq_len, _h = hidden_11SH.shape

        q_a = ttnn.linear(
            hidden_11SH,
            self.w_q_a,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        q_a = ttnn.rms_norm(q_a, epsilon=self.eps, weight=self.w_q_a_ln)
        q_flat = ttnn.linear(
            q_a,
            self.w_q_b,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(q_a)

        kv_a = ttnn.linear(
            hidden_11SH,
            self.w_kv_a,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(hidden_11SH)

        k_lat = ttnn.slice(kv_a, (0, 0, 0, 0), (1, 1, seq_len, self.kv_lora_rank))
        k_rot_raw = ttnn.slice(
            kv_a,
            (0, 0, 0, self.kv_lora_rank),
            (1, 1, seq_len, self.kv_lora_rank + self.qk_rope_head_dim),
        )
        ttnn.deallocate(kv_a)

        k_lat = ttnn.rms_norm(k_lat, epsilon=self.eps, weight=self.w_kv_a_ln)
        k_flat = ttnn.linear(
            k_lat,
            self.w_kv_b,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(k_lat)

        fused = self.qk_nope_head_dim + self.v_head_dim
        q_states = _heads_1hsd_from_linear_11sh_flat(
            q_flat,
            seq_len=seq_len,
            num_heads=self.num_heads,
            head_dim=self.qk_head_dim,
        )
        ttnn.deallocate(q_flat)
        q_pass = ttnn.slice(
            q_states,
            (0, 0, 0, 0),
            (1, self.num_heads, seq_len, self.qk_nope_head_dim),
        )
        q_rot = ttnn.slice(
            q_states,
            (0, 0, 0, self.qk_nope_head_dim),
            (1, self.num_heads, seq_len, self.qk_head_dim),
        )

        k_merged = _heads_1hsd_from_linear_11sh_flat(
            k_flat,
            seq_len=seq_len,
            num_heads=self.num_heads,
            head_dim=fused,
        )
        ttnn.deallocate(k_flat)
        k_pass = ttnn.slice(
            k_merged,
            (0, 0, 0, 0),
            (1, self.num_heads, seq_len, self.qk_nope_head_dim),
        )
        v_states = ttnn.slice(
            k_merged,
            (0, 0, 0, self.qk_nope_head_dim),
            (1, self.num_heads, seq_len, fused),
        )
        ttnn.deallocate(k_merged)

        k_rot_rm = ttnn.to_layout(k_rot_raw, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(k_rot_raw)
        k_rot = ttnn.reshape(k_rot_rm, (1, 1, seq_len, self.qk_rope_head_dim))
        k_rot = ttnn.to_layout(k_rot, ttnn.TILE_LAYOUT)
        k_rot = ttnn.repeat(k_rot, (1, self.num_heads, 1, 1))

        own_rotary = False
        if position_embeddings_tt is not None:
            if position_embeddings is not None:
                raise ValueError("Pass at most one of position_embeddings and position_embeddings_tt.")
            cos_tt, sin_tt = position_embeddings_tt
        elif position_embeddings is not None:
            cos_tt, sin_tt = upload_mistral4_rotary_cos_sin_to_mesh(self.device, position_embeddings)
            own_rotary = True
        else:
            raise ValueError("Provide position_embeddings or position_embeddings_tt.")

        q_rot, k_rot = _apply_rotary_ttnn(q_rot, k_rot, cos_tt, sin_tt, interleave=self.rope_interleave)
        if own_rotary:
            ttnn.deallocate(cos_tt)
            ttnn.deallocate(sin_tt)

        query_states = ttnn.concat([q_pass, q_rot], dim=-1)
        ttnn.deallocate(q_pass)
        ttnn.deallocate(q_rot)
        ttnn.deallocate(q_states)
        key_states = ttnn.concat([k_pass, k_rot], dim=-1)
        ttnn.deallocate(k_pass)
        ttnn.deallocate(k_rot)

        rope_params = getattr(self.cfg, "rope_parameters", None) or {}
        scale_1h1s1 = _llama4_attn_scale_torch(
            position_ids.long(),
            beta=rope_params.get("llama_4_scaling_beta"),
            original_max_position_embeddings=rope_params.get("original_max_position_embeddings"),
        )
        scale_tt = ttnn.from_torch(
            scale_1h1s1,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device=self.device),
        )
        query_states = ttnn.mul(query_states, scale_tt)
        ttnn.deallocate(scale_tt)

        if self.qk_head_dim != self.v_head_dim:
            pad_w = self.qk_head_dim - self.v_head_dim
            v_pad = ttnn.concat(
                [
                    v_states,
                    ttnn.zeros(
                        (1, self.num_heads, seq_len, pad_w),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    ),
                ],
                dim=-1,
            )
            ttnn.deallocate(v_states)
            v_states = v_pad

        return query_states, key_states, v_states, int(seq_len)

    def _sdpa_o(
        self,
        query_states: ttnn.Tensor,
        key_states: ttnn.Tensor,
        v_states: ttnn.Tensor,
        *,
        seq_len: int,
        deallocate_kv: bool = True,
    ) -> ttnn.Tensor:
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query_states,
            key_states,
            v_states,
            is_causal=False,
            scale=self.scaling,
            program_config=self._sdpa_program_config,
            compute_kernel_config=self._sdpa_compute_kernel_config,
        )
        ttnn.deallocate(query_states)
        if deallocate_kv:
            ttnn.deallocate(key_states)
            ttnn.deallocate(v_states)

        if self.qk_head_dim != self.v_head_dim:
            attn_output = ttnn.slice(
                attn_output,
                (0, 0, 0, 0),
                (1, self.num_heads, seq_len, self.v_head_dim),
            )

        attn_11sh = ttnn.experimental.nlp_concat_heads(
            attn_output,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_output)

        out = ttnn.linear(
            attn_11sh,
            self.w_o,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(attn_11sh)
        return out

    def forward(
        self,
        hidden_11SH: ttnn.Tensor,
        *,
        position_ids: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_embeddings_tt: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
    ) -> ttnn.Tensor:
        query_states, key_states, v_states, seq_len = self._qkv_scaled_after_rope(
            hidden_11SH,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            position_embeddings_tt=position_embeddings_tt,
        )
        return self._sdpa_o(query_states, key_states, v_states, seq_len=seq_len, deallocate_kv=True)

    def forward_prefill_with_kv(
        self,
        hidden_11SH: ttnn.Tensor,
        *,
        position_ids: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_embeddings_tt: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """
        Prefill path returning ``(attn_out, key_states, value_states)`` for HF-style decode.

        ``key_states`` / ``value_states`` match the layout stored in ``DynamicCache`` after
        prefill (including optional V padding to ``qk_head_dim``). Caller must deallocate
        the returned KV tensors when no longer needed.
        """
        query_states, key_states, v_states, seq_len = self._qkv_scaled_after_rope(
            hidden_11SH,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            position_embeddings_tt=position_embeddings_tt,
        )
        kv_k = ttnn.clone(key_states)
        kv_v = ttnn.clone(v_states)
        out = self._sdpa_o(query_states, key_states, v_states, seq_len=seq_len, deallocate_kv=True)
        return out, kv_k, kv_v

    def forward_decode_extend_kv(
        self,
        hidden_11SH: ttnn.Tensor,
        *,
        position_ids: torch.Tensor,
        past_key_states: ttnn.Tensor,
        past_value_states: ttnn.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_embeddings_tt: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """
        Single-token decode: append this step's K/V to ``past_*`` and run SDPA.

        Shapes: ``hidden_11SH`` is ``[1,1,1,H]``; ``past_*`` are ``[1, num_heads, T, qk_head_dim]``.
        Returns ``(attn_out, key_states_full, value_states_full)`` with length ``T+1``.
        Caller should deallocate ``past_key_states`` / ``past_value_states`` after consuming
        the returned full tensors.
        """
        query_states, key_new, v_new, seq_len = self._qkv_scaled_after_rope(
            hidden_11SH,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            position_embeddings_tt=position_embeddings_tt,
        )
        if seq_len != 1:
            ttnn.deallocate(query_states)
            ttnn.deallocate(key_new)
            ttnn.deallocate(v_new)
            raise ValueError(f"decode expects seq_len==1 on device, got {seq_len}")

        key_full = ttnn.concat([past_key_states, key_new], dim=2)
        value_full = ttnn.concat([past_value_states, v_new], dim=2)
        ttnn.deallocate(key_new)
        ttnn.deallocate(v_new)

        out = self._sdpa_o(query_states, key_full, value_full, seq_len=1, deallocate_kv=False)
        return out, key_full, value_full
