# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
Host-side ACE-Step decoder reference forward for parity checks against TTNN.

Expects the ACE-Step-1.5 repo on `sys.path` so `acestep.*` and model configs resolve
the same way as `transformers` `trust_remote_code` checkpoints.
"""

from __future__ import annotations

import logging

_ace_step_log = logging.getLogger(__name__)

from pathlib import Path


def ensure_acestep_repo_on_path(ref_repo_root: str | Path) -> Path:
    root = Path(ref_repo_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ACE-Step reference repo not found: {root}")
    import sys

    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


def patch_seq_attention_mask(*, num_frames: int, patch_size: int, batch: int, device, dtype=None):
    """
    HF `AceStepDiTModel` uses `attention_mask` shaped [B, L] where L equals **patch** sequence length
    (same as `hidden_states.shape[1]` after `proj_in`).
    """
    import torch

    if dtype is None:
        dtype = torch.float32
    ps = int(patch_size)
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    pad_length = (ps - (num_frames % ps)) % ps
    t_pad = num_frames + pad_length
    if t_pad % ps != 0:
        raise RuntimeError(f"internal: padded length {t_pad} not divisible by patch_size {ps}")
    n_patches = t_pad // ps
    return torch.ones((int(batch), n_patches), device=device, dtype=dtype)


def load_hf_decoder_from_checkpoint_dir(model_dir: str | Path, *, ref_repo_root: str | Path, torch_dtype=None):
    """
    Load full HF model from `model_dir` (contains config + weights) and return `.decoder` (AceStepDiTModel).
    """
    import torch
    from transformers import AutoModel

    ensure_acestep_repo_on_path(ref_repo_root)
    model_dir = Path(model_dir).resolve()
    if torch_dtype is None:
        torch_dtype = torch.bfloat16
    m = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True, torch_dtype=torch_dtype).eval()
    dec = m.decoder
    # Eager/SDPA path matches our TTNN debug comparisons (no FA2-specific mask contract).
    if hasattr(dec.config, "_attn_implementation"):
        dec.config._attn_implementation = "sdpa"
    return dec


def hf_decoder_velocity(
    decoder,
    *,
    xt: "torch.Tensor",  # [B, T, 64] float32 or bf16
    context_latents: "torch.Tensor",  # [B, T, 128]
    encoder_hidden_states: "torch.Tensor",  # [B, S_enc, cond_dim]
    t_curr: float,
    device: str | None = None,
    dtype=None,
    encoder_attention_mask_1d: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """
    One decoder forward matching HF `generate_audio` contract (timestep_r == timestep).

    Returns velocity / prediction tensor [B, T, audio_acoustic_hidden_dim] in float32 on CPU
    for easy PCC vs TTNN `to_torch(vt).float()`.
    """
    import torch

    if dtype is None:
        dtype = torch.bfloat16
    dec_dev = next(decoder.parameters()).device
    w_dtype = next(decoder.parameters()).dtype
    # Run reference on same device as decoder weights for speed (usually CPU).
    B = int(xt.shape[0])
    T = int(xt.shape[1])
    ps = int(decoder.config.patch_size)
    attn = patch_seq_attention_mask(num_frames=T, patch_size=ps, batch=B, device=dec_dev, dtype=torch.float32)
    s_enc = int(encoder_hidden_states.shape[1])
    enc_attn = torch.ones((B, s_enc), device=dec_dev, dtype=torch.float32)

    xt_d = xt.to(device=dec_dev, dtype=dtype)
    ctx_d = context_latents.to(device=dec_dev, dtype=dtype)
    enc_d = encoder_hidden_states.to(device=dec_dev, dtype=dtype)

    if encoder_attention_mask_1d is not None:
        enc_attn = encoder_attention_mask_1d.to(device=dec_dev, dtype=torch.float32)

    # HF TimestepEmbedding uses `t_freq.to(t.dtype)`; `t` must match Linear weight dtype or matmul mixes float/bf16.
    ts = torch.full((B,), float(t_curr), device=dec_dev, dtype=w_dtype)

    with torch.inference_mode():
        out = decoder(
            hidden_states=xt_d,
            timestep=ts,
            timestep_r=ts,
            attention_mask=attn,
            encoder_hidden_states=enc_d,
            encoder_attention_mask=enc_attn,
            context_latents=ctx_d,
            use_cache=False,
            past_key_values=None,
        )
    return out[0].detach().float().cpu()


def hf_decoder_intermediates(
    decoder,
    *,
    xt: "torch.Tensor",
    context_latents: "torch.Tensor",
    encoder_hidden_states: "torch.Tensor",
    encoder_attention_mask_1d: "torch.Tensor" | None,
    t_curr: float,
    layer_idx: int = 0,
    device: str | None = None,
    dtype=None,
) -> dict[str, "torch.Tensor"]:
    """
    Run one HF decoder forward and capture intermediate tensors for parity debugging.

    Returns a dict of CPU float32 tensors keyed by stage names:
      - patchify_out
      - temb
      - timestep_proj
      - cond_emb (encoder_hidden_states after condition_embedder)
      - layer{idx}.adaln_self_in
      - layer{idx}.self_q / self_k / self_v
      - layer{idx}.self_attn_out
      - layer{idx}.adaln_mlp_in
      - layer{idx}.cross_attn_out
      - layer{idx}.after_self
      - layer{idx}.after_cross
      - layer{idx}.mlp_out
      - layer{idx}.block_out
      - acoustic_out (final model output)
    """
    import torch

    if dtype is None:
        dtype = torch.bfloat16
    if device is None:
        dec_dev = next(decoder.parameters()).device
    else:
        dec_dev = torch.device(device)
    w_dtype = next(decoder.parameters()).dtype

    B = int(xt.shape[0])
    T = int(xt.shape[1])
    ps = int(decoder.config.patch_size)
    attn = patch_seq_attention_mask(num_frames=T, patch_size=ps, batch=B, device=dec_dev, dtype=torch.float32)

    enc_d = encoder_hidden_states.to(device=dec_dev, dtype=dtype)
    s_enc = int(enc_d.shape[1])
    if encoder_attention_mask_1d is None:
        enc_attn_1d = torch.ones((B, s_enc), device=dec_dev, dtype=torch.float32)
    else:
        enc_attn_1d = encoder_attention_mask_1d.to(device=dec_dev, dtype=torch.float32)

    xt_d = xt.to(device=dec_dev, dtype=dtype)
    ctx_d = context_latents.to(device=dec_dev, dtype=dtype)
    ts = torch.full((B,), float(t_curr), device=dec_dev, dtype=w_dtype)

    captured: dict[str, torch.Tensor] = {}
    hooks = []

    def _save(name: str, x: torch.Tensor):
        captured[name] = x.detach().float().cpu()

    # proj_in output (patchify)
    hooks.append(decoder.proj_in.register_forward_hook(lambda _m, _i, o: _save("patchify_out", o)))

    # time embeds
    hooks.append(decoder.time_embed.register_forward_hook(lambda _m, _i, o: _save("time_embed.out", o[0])))
    hooks.append(decoder.time_embed_r.register_forward_hook(lambda _m, _i, o: _save("time_embed_r.out", o[0])))

    # condition embedder output
    hooks.append(decoder.condition_embedder.register_forward_hook(lambda _m, _i, o: _save("cond_emb", o)))

    # Capture layer math for a single layer (default 0)
    layer = decoder.layers[int(layer_idx)]

    def _layer_hook(_m, inputs, outputs):
        # inputs: (hidden_states, position_embeddings, temb, attention_mask, position_ids, ...)
        hidden_states = inputs[0]
        temb = inputs[2]
        _save(f"layer{layer_idx}.temb", temb)

        # AdaLN (self-attn) input
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (layer.scale_shift_table + temb).chunk(
            6, dim=1
        )
        adaln_self_in = (layer.self_attn_norm(hidden_states) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        _save(f"layer{layer_idx}.adaln_self_in", adaln_self_in)

        # Compute HF self-attn q/k after q_norm/k_norm and after RoPE (for parity pinpointing).
        try:
            import sys

            attn = layer.self_attn
            # Resolve RoPE helper from the already-loaded attention module only.
            # Avoid import_module(type(attn).__module__) — SAST flags dynamic imports.
            mod_name = type(attn).__module__ or ""
            _allowed_prefixes = ("transformers.", "acestep.")
            apply_rope = None
            if mod_name.startswith(_allowed_prefixes):
                mod = sys.modules.get(mod_name)
                if mod is not None:
                    apply_rope = getattr(mod, "apply_rotary_pos_emb", None)
            pos = inputs[1]  # (cos, sin)
            if apply_rope is not None and pos is not None:
                cos, sin = pos
                B = int(hidden_states.shape[0])
                S = int(hidden_states.shape[1])
                H = int(attn.config.num_attention_heads)
                kvH = int(attn.config.num_key_value_heads)
                Dh = int(attn.head_dim)
                # q/k pre-norm from captured linear outputs if present
                q_lin = captured.get(f"layer{layer_idx}.self_attn.q_proj.out", None)
                k_lin = captured.get(f"layer{layer_idx}.self_attn.k_proj.out", None)
                if q_lin is not None and k_lin is not None:
                    q_lin_d = q_lin.to(device=hidden_states.device, dtype=hidden_states.dtype)
                    k_lin_d = k_lin.to(device=hidden_states.device, dtype=hidden_states.dtype)
                    q_states = attn.q_norm(q_lin_d.view(B, S, H, Dh)).transpose(1, 2)  # [B,H,S,Dh]
                    k_states = attn.k_norm(k_lin_d.view(B, S, kvH, Dh)).transpose(1, 2)  # [B,kvH,S,Dh]
                    # Expand kv heads like HF attention backend does (repeat_interleave on head dim groups)
                    if H != kvH:
                        rep = H // kvH
                        k_states = k_states.repeat_interleave(rep, dim=1)
                    _save(f"layer{layer_idx}.self_q_norm", q_states)
                    _save(f"layer{layer_idx}.self_k_norm", k_states)
                    q_rope, k_rope = apply_rope(q_states, k_states, cos, sin)
                    _save(f"layer{layer_idx}.self_q_rope", q_rope)
                    _save(f"layer{layer_idx}.self_k_rope", k_rope)
        except Exception as e:
            # Best-effort debug capture: failures here must not affect the main forward path.
            _save(f"layer{layer_idx}.self_attn_rope_capture_error", f"{type(e).__name__}: {e}")

        # After self-attn residual (gate)
        # outputs is the final layer output (after cross+mlp), but we can reconstruct
        # intermediate states using captured self/cross module outputs.
        block_out = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        _save(f"layer{layer_idx}.block_out", block_out)

        # Reconstruct hidden after self-attn and after cross-attn if module outputs were captured.
        if f"layer{layer_idx}.self_attn_out" in captured:
            self_out = captured[f"layer{layer_idx}.self_attn_out"].to(hidden_states.device, dtype=hidden_states.dtype)
            hs_after_self = (hidden_states + self_out * gate_msa).type_as(hidden_states)
            _save(f"layer{layer_idx}.after_self", hs_after_self)
            if f"layer{layer_idx}.cross_attn_out" in captured:
                cross_out = captured[f"layer{layer_idx}.cross_attn_out"].to(
                    hidden_states.device, dtype=hidden_states.dtype
                )
                hs_after_cross = (hs_after_self + cross_out).type_as(hidden_states)
                _save(f"layer{layer_idx}.after_cross", hs_after_cross)
                # Compute the MLP AdaLN input from the reconstructed post-cross hidden state.
                adaln_mlp_in = (layer.mlp_norm(hs_after_cross) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
                _save(f"layer{layer_idx}.adaln_mlp_in", adaln_mlp_in)

    hooks.append(layer.register_forward_hook(_layer_hook))

    # Hook self-attn projections and outputs if present
    for name, mod in layer.named_modules():
        full = f"layer{layer_idx}.{name}"
        if name.endswith("self_attn.q_proj"):
            hooks.append(mod.register_forward_hook(lambda _m, _i, o, n=full: _save(f"{n}.out", o)))
        if name.endswith("self_attn.k_proj"):
            hooks.append(mod.register_forward_hook(lambda _m, _i, o, n=full: _save(f"{n}.out", o)))
        if name.endswith("self_attn.v_proj"):
            hooks.append(mod.register_forward_hook(lambda _m, _i, o, n=full: _save(f"{n}.out", o)))
        if name.endswith("self_attn"):
            hooks.append(mod.register_forward_hook(lambda _m, _i, o: _save(f"layer{layer_idx}.self_attn_out", o[0])))
        if name.endswith("cross_attn"):
            hooks.append(mod.register_forward_hook(lambda _m, _i, o: _save(f"layer{layer_idx}.cross_attn_out", o[0])))
        if name == "mlp":
            # Capture MLP input (AdaLN'd) and output (raw, before c_gate).
            hooks.append(mod.register_forward_pre_hook(lambda _m, i, n=full: _save(f"{n}.in", i[0])))
            hooks.append(mod.register_forward_hook(lambda _m, _i, o: _save(f"layer{layer_idx}.mlp_out", o)))

    with torch.inference_mode():
        out = decoder(
            hidden_states=xt_d,
            timestep=ts,
            timestep_r=ts,
            attention_mask=attn,
            encoder_hidden_states=enc_d,
            encoder_attention_mask=enc_attn_1d,
            context_latents=ctx_d,
            use_cache=False,
            past_key_values=None,
        )
    _save("acoustic_out", out[0])

    for h in hooks:
        try:
            h.remove()
        except Exception as exc:
            # Best-effort: non-fatal if already released or unavailable.
            _ace_step_log.debug("Best-effort cleanup ignored: %s", exc)

    # Normalize key names to the ones the user cares about
    normalized: dict[str, torch.Tensor] = {}
    if "patchify_out" in captured:
        normalized["patchify_out"] = captured["patchify_out"]
    if "time_embed.out" in captured and "time_embed_r.out" in captured:
        normalized["temb"] = captured["time_embed.out"] + captured["time_embed_r.out"]
    if f"layer{layer_idx}.temb" in captured:
        normalized["timestep_proj"] = captured[f"layer{layer_idx}.temb"]
    if "cond_emb" in captured:
        normalized["cond_emb"] = captured["cond_emb"]
    if f"layer{layer_idx}.adaln_self_in" in captured:
        normalized["adaln_self_in"] = captured[f"layer{layer_idx}.adaln_self_in"]
    if f"layer{layer_idx}.self_attn_out" in captured:
        normalized["self_attn_out"] = captured[f"layer{layer_idx}.self_attn_out"]
    if f"layer{layer_idx}.mlp_out" in captured:
        normalized["mlp_out"] = captured[f"layer{layer_idx}.mlp_out"]
    if f"layer{layer_idx}.adaln_mlp_in" in captured:
        normalized["adaln_mlp_in"] = captured[f"layer{layer_idx}.adaln_mlp_in"]
    if f"layer{layer_idx}.cross_attn_out" in captured:
        normalized["cross_attn_out"] = captured[f"layer{layer_idx}.cross_attn_out"]
    if f"layer{layer_idx}.after_self" in captured:
        normalized["after_self"] = captured[f"layer{layer_idx}.after_self"]
    if f"layer{layer_idx}.after_cross" in captured:
        normalized["after_cross"] = captured[f"layer{layer_idx}.after_cross"]
    if f"layer{layer_idx}.block_out" in captured:
        normalized["block_out"] = captured[f"layer{layer_idx}.block_out"]
    if "acoustic_out" in captured:
        normalized["acoustic_out"] = captured["acoustic_out"]

    # q/k/v linear outputs (before head split), if hooks matched
    qkvs = {
        "self_q": f"layer{layer_idx}.self_attn.q_proj.out",
        "self_k": f"layer{layer_idx}.self_attn.k_proj.out",
        "self_v": f"layer{layer_idx}.self_attn.v_proj.out",
    }
    for k, src in qkvs.items():
        if src in captured:
            normalized[k] = captured[src]

    for k in ("self_q_norm", "self_k_norm", "self_q_rope", "self_k_rope"):
        key = f"layer{layer_idx}.{k}"
        if key in captured:
            normalized[k] = captured[key]

    return normalized


def tensor_stats(name: str, x: "torch.Tensor") -> str:
    import torch

    t = x.detach().float().cpu().reshape(-1)
    fin = torch.isfinite(t)
    if not fin.any():
        return f"{name}: all non-finite shape={tuple(x.shape)}"
    tf = t[fin]
    return (
        f"{name}: shape={tuple(x.shape)} dtype={x.dtype} "
        f"min={float(tf.min()):.4g} max={float(tf.max()):.4g} "
        f"mean={float(tf.mean()):.4g} std={float(tf.std(unbiased=False)):.4g}"
    )


def pearson_pcc(a: "torch.Tensor", b: "torch.Tensor") -> float:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    if a.numel() != b.numel():
        raise ValueError(f"PCC shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp(min=1e-12)
    return float((a @ b) / denom)


def max_abs_diff(a: "torch.Tensor", b: "torch.Tensor") -> float:
    return float((a.detach().float() - b.detach().float()).abs().max().item())
