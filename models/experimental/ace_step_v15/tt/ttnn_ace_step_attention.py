# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ACE-Step 1.5 DiT attention in TTNN: GQA self-attention and cross-attention.

Both modules mirror ``diffusers.models.transformers.ace_step_transformer.AceStepAttention``
and consume its state-dict layout directly (``to_q`` / ``to_k`` / ``to_v`` / ``to_out.0`` /
``norm_q`` / ``norm_k``), so nothing here depends on how Block 0 chooses to stage weights —
only on the upstream -> diffusers rename having happened.

Op order (master doc §3.3, steps 3-7 for self-attn, step 10 for cross-attn)::

    q = to_q(h) -> [1, 16, S, 128]     k = to_k(h) -> [1, 8, S, 128]     v = to_v(h)
    q = norm_q(q); k = norm_k(k)                # RMSNorm over head_dim, per head
    q, k = RoPE(q, k)                           # self-attn only; cross-attn gets NO RoPE
    o = SDPA(q, k, v, scale=128 ** -0.5)        # NEVER causal
    o = to_out[0](concat_heads(o))

Traps encoded here
------------------
* **TRAP-1** sliding layers pass ``sliding_window_size = 2 * config.sliding_window`` (256),
  because TTNN's parameter is the *total* window width. Global layers pass nothing at all.
  ``is_causal`` is hard-wired to ``False`` on every path.
* **TRAP-2** K/V are handed to SDPA as ``[1, 8, S, 128]`` and expanded **in-kernel**. The torch
  reference and the MLX port both ``repeat_interleave`` K/V to 16 heads; doing that on device
  would double K/V memory and bandwidth for nothing.
* **TRAP-5** cross-attention K/V are returned as plain ``[1, 8, enc_L, 128]`` DRAM tensors for
  the caller to hold across the denoising loop, not a per-slot KV cache. Batch-1 only.
* Cross-attention gets ``attn_mask=None`` on purpose: the reference overwrites both
  ``attention_mask`` and ``encoder_attention_mask`` with ``None``, so cross-attn deliberately
  attends to the padded tail of ``encoder_hidden_states``.
"""

from __future__ import annotations

import torch

import ttnn
from models.tt_dit.layers.module import Module

from .ttnn_ace_step_common import (
    AceStepDiTConfig,
    apply_rope,
    capture_tensor,
    linear_compute_config,
    make_linear,
    make_rms_norm,
    norm_compute_config,
    sdpa_compute_config,
)


class AceStepSelfAttention(Module):
    """GQA self-attention with QK-RMSNorm, HF half-split RoPE and windowed SDPA.

    The three reference projections are packed into one ``Linear(hidden, (nq + 2 * nkv) * dh)``
    so the head split is a single ``nlp_create_qkv_heads`` call. ``_prepare_torch_state``
    performs the packing from ``to_q`` / ``to_k`` / ``to_v``, in that order — the concatenation
    order ``nlp_create_qkv_heads`` expects.

    Inputs and outputs are 4D ``[1, 1, S, hidden_size]``.
    """

    def __init__(
        self,
        config: AceStepDiTConfig,
        *,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        rope_composite: bool = False,
        sdpa_program_config=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.rope_composite = rope_composite
        self.sdpa_program_config = sdpa_program_config

        self.qkv = make_linear(
            config.hidden_size,
            config.qkv_width,
            bias=config.attention_bias,
            mesh_device=mesh_device,
            dtype=dtype,
        )
        self.to_out = make_linear(
            config.q_width,
            config.hidden_size,
            bias=config.attention_bias,
            mesh_device=mesh_device,
            dtype=dtype,
        )
        self.norm_q = make_rms_norm(config.head_dim, eps=config.rms_norm_eps, mesh_device=mesh_device, dtype=dtype)
        self.norm_k = make_rms_norm(config.head_dim, eps=config.rms_norm_eps, mesh_device=mesh_device, dtype=dtype)

        self.mm_compute_config = linear_compute_config(mesh_device, dtype)
        self.norm_compute_config = norm_compute_config(mesh_device)
        self.sdpa_compute_config = sdpa_compute_config(mesh_device)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        q = state.pop("to_q.weight", None)
        k = state.pop("to_k.weight", None)
        v = state.pop("to_v.weight", None)
        if q is not None and k is not None and v is not None:
            # nlp_create_qkv_heads consumes [Q | K | V] concatenated on the feature axis.
            state["qkv.weight"] = torch.cat([q, k, v], dim=0)
        out = state.pop("to_out.0.weight", None)
        if out is not None:
            state["to_out.weight"] = out
        # to_out.1 is nn.Dropout(0.0) in the reference: no parameters, nothing to map.

    def forward(
        self,
        x_11SC: ttnn.Tensor,
        *,
        rope: tuple[ttnn.Tensor, ttnn.Tensor],
        window: int | None,
        attn_mask: ttnn.Tensor | None = None,
        capture: dict[str, torch.Tensor] | None = None,
        prefix: str = "self_attn.",
    ) -> ttnn.Tensor:
        """Args:
        x_11SC: normalised, adaLN-modulated hidden states ``[1, 1, S, hidden_size]``.
        rope: ``(cos, sin)``, each ``[1, 1, S, head_dim]``.
        window: ``config.window_for_layer(i)`` — 256 for sliding layers, ``None`` for
            global ones. **Never** pass ``config.sliding_window`` (TRAP-1).
        attn_mask: optional dense additive ``[1, 1, Sq, Sk]`` mask. **Mutually exclusive
            with** ``window`` — see TRAP-7: SDPA has an explicit ``TT_FATAL`` for the
            combination, so a caller that needs a dense mask must bake the
            ``|i-j| <= sliding_window`` band into it and pass ``window=None``. The DiT itself
            never needs this (the on-device window stamp is strictly better); it exists for
            Block 2's ``pad_mode="dense_mask"`` escape hatch.
        capture: optional dict collecting host fp32 snapshots for PCC bisection.
        """
        cfg = self.config
        if window is not None:
            assert window == cfg.sdpa_window_size, (
                f"sliding_window_size must be 2 * config.sliding_window = {cfg.sdpa_window_size} "
                f"(TTNN's window parameter is the TOTAL width; got {window}). See TRAP-1."
            )
            assert attn_mask is None, (
                "SDPA rejects sliding_window_size together with attn_mask (TRAP-7). Bake the "
                "|i-j| <= sliding_window band into attn_mask and pass window=None instead."
            )

        qkv = self.qkv(x_11SC, compute_kernel_config=self.mm_compute_config)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)
        capture_tensor(capture, prefix + "q_pre_norm", q)
        capture_tensor(capture, prefix + "k_pre_norm", k)
        capture_tensor(capture, prefix + "v", v)

        q_n = self.norm_q(q, compute_kernel_config=self.norm_compute_config)
        k_n = self.norm_k(k, compute_kernel_config=self.norm_compute_config)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        capture_tensor(capture, prefix + "q_normed", q_n)
        capture_tensor(capture, prefix + "k_normed", k_n)

        cos, sin = rope
        q_r = apply_rope(q_n, cos, sin, composite=self.rope_composite)
        k_r = apply_rope(k_n, cos, sin, composite=self.rope_composite)
        ttnn.deallocate(q_n)
        ttnn.deallocate(k_n)
        capture_tensor(capture, prefix + "q_rope", q_r)
        capture_tensor(capture, prefix + "k_rope", k_r)

        # TRAP-2: q is [1, 16, S, 128] and k/v stay [1, 8, S, 128]; SDPA expands GQA in-kernel.
        out = ttnn.transformer.scaled_dot_product_attention(
            q_r,
            k_r,
            v,
            attn_mask=attn_mask,
            is_causal=False,  # ACE-Step self-attention is NEVER causal on either layer type.
            scale=cfg.attention_scale,
            sliding_window_size=window,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.sdpa_compute_config,
        )
        ttnn.deallocate(q_r)
        ttnn.deallocate(k_r)
        ttnn.deallocate(v)
        capture_tensor(capture, prefix + "sdpa", out)

        out_flat = ttnn.experimental.nlp_concat_heads(out)
        ttnn.deallocate(out)
        result = self.to_out(out_flat, compute_kernel_config=self.mm_compute_config)
        ttnn.deallocate(out_flat)
        capture_tensor(capture, prefix + "out", result)
        return result


class AceStepCrossAttention(Module):
    """Cross-attention over the packed condition-encoder output.

    Differences from self-attention, all load-bearing (master doc §3.3 step 10):

    * **no RoPE** on either Q or K;
    * **no attention mask** — the reference nulls ``encoder_attention_mask``;
    * the surrounding residual is **unmodulated and ungated** (handled by the block);
    * ``to_k`` / ``to_v`` read ``encoder_hidden_states``, which is timestep-independent, so
      they are hoisted out of the denoising loop via :meth:`compute_kv` (TRAP-5).

    Q and K/V therefore have different sequence lengths (``S`` vs ``enc_L``), which non-causal
    TTNN SDPA supports. ``enc_L`` need not be tile-aligned: SDPA generates a partial-tile K
    mask automatically when ``enc_L % 32 != 0``.
    """

    def __init__(
        self,
        config: AceStepDiTConfig,
        *,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        sdpa_program_config=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.sdpa_program_config = sdpa_program_config

        self.to_q = make_linear(
            config.hidden_size, config.q_width, bias=config.attention_bias, mesh_device=mesh_device, dtype=dtype
        )
        # K/V read the (already condition_embedder-projected) encoder states, so their input
        # width is hidden_size even on the XL variants where encoder_hidden_size differs.
        self.to_k = make_linear(
            config.hidden_size, config.kv_width, bias=config.attention_bias, mesh_device=mesh_device, dtype=dtype
        )
        self.to_v = make_linear(
            config.hidden_size, config.kv_width, bias=config.attention_bias, mesh_device=mesh_device, dtype=dtype
        )
        self.to_out = make_linear(
            config.q_width, config.hidden_size, bias=config.attention_bias, mesh_device=mesh_device, dtype=dtype
        )
        self.norm_q = make_rms_norm(config.head_dim, eps=config.rms_norm_eps, mesh_device=mesh_device, dtype=dtype)
        self.norm_k = make_rms_norm(config.head_dim, eps=config.rms_norm_eps, mesh_device=mesh_device, dtype=dtype)

        self.mm_compute_config = linear_compute_config(mesh_device, dtype)
        self.norm_compute_config = norm_compute_config(mesh_device)
        self.sdpa_compute_config = sdpa_compute_config(mesh_device)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        out = state.pop("to_out.0.weight", None)
        if out is not None:
            state["to_out.weight"] = out

    def compute_kv(
        self,
        encoder_11LC: ttnn.Tensor,
        *,
        capture: dict[str, torch.Tensor] | None = None,
        prefix: str = "cross_attn.",
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Project + head-split + QK-norm the encoder states once, for reuse across all steps.

        Args:
            encoder_11LC: ``condition_embedder``-projected encoder states,
                ``[1, 1, enc_L, hidden_size]``.

        Returns:
            ``(k, v)``, each ``[1, num_key_value_heads, enc_L, head_dim]``.

        # BATCH-1 ASSUMPTION: plain tensors, no user/slot axis and no moving position, so no
        # ttnn.fill_cache machinery. Revisit at batch > 1 (APG guidance needs 2x). See TRAP-5.
        """
        cfg = self.config
        k_f = self.to_k(encoder_11LC, compute_kernel_config=self.mm_compute_config)
        v_f = self.to_v(encoder_11LC, compute_kernel_config=self.mm_compute_config)
        k_h = self.split_heads(k_f, cfg.num_key_value_heads)
        v = self.split_heads(v_f, cfg.num_key_value_heads)
        ttnn.deallocate(k_f)
        ttnn.deallocate(v_f)
        k = self.norm_k(k_h, compute_kernel_config=self.norm_compute_config)
        ttnn.deallocate(k_h)
        capture_tensor(capture, prefix + "k", k)
        capture_tensor(capture, prefix + "v", v)
        return k, v

    @staticmethod
    def split_heads(x_11LC: ttnn.Tensor, num_heads: int) -> ttnn.Tensor:
        """``[1, 1, L, num_heads * head_dim]`` -> ``[1, num_heads, L, head_dim]``.

        ``num_kv_heads=0`` is the tt_dit idiom for splitting a single tensor (see
        ``models/tt_dit/models/transformers/ltx/attention_ltx.py``); the 2nd and 3rd outputs
        are empty.
        """
        out, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            x_11LC,
            num_heads=num_heads,
            num_kv_heads=0,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return out

    def forward(
        self,
        x_11SC: ttnn.Tensor,
        kv: tuple[ttnn.Tensor, ttnn.Tensor],
        *,
        capture: dict[str, torch.Tensor] | None = None,
        prefix: str = "cross_attn.",
    ) -> ttnn.Tensor:
        """Args:
        x_11SC: ``cross_attn_norm(x)`` — RMSNorm only, no adaLN.
        kv: the ``(k, v)`` pair from :meth:`compute_kv`.
        """
        cfg = self.config
        k, v = kv

        q_f = self.to_q(x_11SC, compute_kernel_config=self.mm_compute_config)
        q_h = self.split_heads(q_f, cfg.num_attention_heads)
        ttnn.deallocate(q_f)
        q = self.norm_q(q_h, compute_kernel_config=self.norm_compute_config)
        ttnn.deallocate(q_h)
        capture_tensor(capture, prefix + "q", q)

        out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,  # the reference nulls encoder_attention_mask; padding IS attended.
            is_causal=False,
            scale=cfg.attention_scale,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.sdpa_compute_config,
        )
        ttnn.deallocate(q)
        capture_tensor(capture, prefix + "sdpa", out)

        out_flat = ttnn.experimental.nlp_concat_heads(out)
        ttnn.deallocate(out)
        result = self.to_out(out_flat, compute_kernel_config=self.mm_compute_config)
        ttnn.deallocate(out_flat)
        capture_tensor(capture, prefix + "out", result)
        return result
