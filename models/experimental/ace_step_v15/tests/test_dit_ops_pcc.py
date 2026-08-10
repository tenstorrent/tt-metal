# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Op-level PCC test for the ACE-Step 1.5 DiT (Block 1), S=32 (2.56 s).

One assertion per primitive, so a regression points at the op that broke rather than at the
model. Every oracle is the `diffusers` reference module itself (fp32 CPU), never a
hand-written re-derivation — see ``dit_reference.py``.

Covered, in the order the block uses them (master doc §3.3):

  * ``rms_norm_hidden``     RMSNorm over hidden_size=2048
  * ``adaln_composite``     RMSNorm(x) * (1 + scale) + shift, plain ops
  * ``adaln_fused``         the same, folded into the fused norm kernel's affine terms
  * ``modulation_6``        the folded 6-way adaLN table (incl. the time_embed_r fold)
  * ``qkv_split``           fused QKV matmul + nlp_create_qkv_heads
  * ``qk_norm``             RMSNorm over head_dim=128, per head
  * ``rope_device``         ttnn.experimental.rotary_embedding_hf
  * ``rope_composite``      slice/neg/concat fallback
  * ``self_attn``           the whole GQA self-attention module
  * ``cross_attn``          the whole cross-attention module (no RoPE, no mask)
  * ``mlp``                 fused-SwiGLU 2048 -> 6144 -> 2048
  * ``time_sinusoid``       host fp32 256-wide sinusoid
  * ``time_embed``          linear_1 / silu / linear_2
  * ``timestep_proj``       the 6-way projection, chunked on the last dim
  * ``condition_embedder``  the encoder projection
  * ``patchify``            proj_in_conv as reshape + Linear
  * ``unpatchify``          proj_out_conv as Linear + reshape
  * ``norm_out``            the 2-way output adaLN site

At S=32 the sliding window (``|i-j| <= 128``) covers the whole sequence, so it is a no-op
here by construction; the window geometry itself is gated in ``test_dit_banded_pcc.py``.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.ace_step_v15.tests import dit_reference as R
from models.experimental.ace_step_v15.tt.ttnn_ace_step_attention import (
    AceStepCrossAttention,
    AceStepSelfAttention,
)
from models.experimental.ace_step_v15.tt.ttnn_ace_step_common import (
    AceStepDiTConfig,
    Modulation,
    apply_rope,
    build_rope_tables,
    make_rms_norm,
    norm_compute_config,
    rms_norm_modulated,
    timestep_sinusoid,
    to_device,
    to_host,
)
from models.experimental.ace_step_v15.tt.ttnn_ace_step_dit import (
    AceStepMLP,
    AceStepTransformer1DModel,
)

GOLDEN = str(R.GOLDEN_DIR)
TARGET_PCC = 0.999

SEQ_LEN = R.SEQ_LEN_UNIT  # 32 -> 2.56 s
ENC_LEN = 96  # arbitrary; tile-aligned so the padded-K path is not also under test here
SEED = 1234


def _to_11sc(x: torch.Tensor, device) -> ttnn.Tensor:
    """``[1, S, C]`` torch -> ``[1, 1, S, C]`` device tensor.  # BATCH-1 ASSUMPTION"""
    assert x.shape[0] == 1, "# BATCH-1 ASSUMPTION"
    return to_device(x.reshape(1, 1, *x.shape[1:]), device)


class _Results(dict):
    """Ordered ``name -> (pcc, ref_shape, got_shape)`` collector.

    ``comp_pcc(ref, got, pcc=0.0)`` never gates; it just yields the number, so every stage
    is reported and divergence is localised before the final assert.
    """

    def check(self, name: str, ref: torch.Tensor, got: torch.Tensor) -> float:
        _, pcc = comp_pcc(ref.to(torch.float32), got.to(torch.float32), pcc=0.0)
        self[name] = (float(pcc), tuple(ref.shape), tuple(got.shape))
        return float(pcc)


def run_dit_ops_pcc(device, verbose: bool = True):  # noqa: C901, PLR0915
    torch.manual_seed(SEED)
    config = AceStepDiTConfig()
    results = _Results()
    norm_cfg = norm_compute_config(device)

    hidden = config.hidden_size
    heads, kv_heads, head_dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim

    # ---------------------------------------------------------------- shared activations --
    x_nsc = torch.randn(1, SEQ_LEN, hidden) * 0.5
    enc_nsc = torch.randn(1, ENC_LEN, hidden) * 0.5
    x_tt = _to_11sc(x_nsc, device)
    enc_tt = _to_11sc(enc_nsc, device)

    # ================================================================== RMSNorm (hidden) ==
    from diffusers.models.normalization import RMSNorm as RefRMSNorm

    ref_norm = R.randomize_(RefRMSNorm(hidden, eps=config.rms_norm_eps), seed=SEED)
    tt_norm = make_rms_norm(hidden, eps=config.rms_norm_eps, mesh_device=device)
    tt_norm.load_torch_state_dict(ref_norm.state_dict())
    results.check("rms_norm_hidden", ref_norm(x_nsc), to_host(tt_norm(x_tt, compute_kernel_config=norm_cfg)))

    # ============================================================================= adaLN ==
    scale_nsc = torch.randn(1, 1, hidden) * 0.1
    shift_nsc = torch.randn(1, 1, hidden) * 0.1
    ref_adaln = ref_norm(x_nsc) * (1 + scale_nsc) + shift_nsc
    one_plus_scale_tt = to_device((1.0 + scale_nsc).reshape(1, 1, 1, hidden), device)
    shift_tt = to_device(shift_nsc.reshape(1, 1, 1, hidden), device)
    for name, fused in (("adaln_composite", False), ("adaln_fused", True)):
        got = rms_norm_modulated(
            tt_norm, x_tt, one_plus_scale_tt, shift_tt, compute_kernel_config=norm_cfg, fused=fused
        )
        results.check(name, ref_adaln, to_host(got))
        ttnn.deallocate(got)

    # ==================================================================== Modulation fold ==
    # The device half is six broadcast adds; the interesting part is that the folded
    # constants reproduce (scale_shift_table + timestep_proj_t + timestep_proj_r).chunk(6).
    ref_te_r = R.reference_timestep_embedding(config, seed=SEED + 7)
    _, timestep_proj_r = ref_te_r(torch.zeros(1))
    table = torch.randn(1, 6, hidden) / hidden**0.5
    tp_t = torch.randn(1, 6, hidden) * 0.1
    ref_chunks = (table + tp_t + timestep_proj_r).chunk(6, dim=1)

    tt_mod = Modulation(hidden, num_chunks=6, mesh_device=device)
    tt_mod.load_torch_state_dict(Modulation.fold_state(table, timestep_proj_r[0], num_chunks=6))
    per_step = [to_device(tp_t[:, i].reshape(1, 1, 1, hidden), device) for i in range(6)]
    got_chunks = tt_mod(per_step)
    worst = 1.0
    for i, name in enumerate(Modulation.NAMES_6):
        expected = ref_chunks[i] + (1.0 if name.startswith("one_plus_") else 0.0)
        worst = min(worst, results.check(f"modulation_6.{name}", expected, to_host(got_chunks[i])))
    results["modulation_6"] = (worst, (1, 6, hidden), (6, 1, 1, 1, hidden))

    # ================================================== self-attention pieces and whole ==
    ref_self = R.reference_attention(config, is_cross=False, seed=SEED + 1)
    tt_self = AceStepSelfAttention(config, mesh_device=device)
    tt_self.load_torch_state_dict(ref_self.state_dict())

    # qkv projection + head split, before any norm.
    ref_q = ref_self.to_q(x_nsc).unflatten(-1, (heads, head_dim)).permute(0, 2, 1, 3)
    ref_k = ref_self.to_k(x_nsc).unflatten(-1, (kv_heads, head_dim)).permute(0, 2, 1, 3)
    ref_v = ref_self.to_v(x_nsc).unflatten(-1, (kv_heads, head_dim)).permute(0, 2, 1, 3)
    qkv = tt_self.qkv(x_tt, compute_kernel_config=tt_self.mm_compute_config)
    q_tt, k_tt, v_tt = ttnn.experimental.nlp_create_qkv_heads(
        qkv, num_heads=heads, num_kv_heads=kv_heads, transpose_k_heads=False
    )
    ttnn.deallocate(qkv)
    results.check("qkv_split.q", ref_q, to_host(q_tt))
    results.check("qkv_split.k", ref_k, to_host(k_tt))
    results.check("qkv_split.v", ref_v, to_host(v_tt))
    results["qkv_split"] = (
        min(results["qkv_split.q"][0], results["qkv_split.k"][0], results["qkv_split.v"][0]),
        tuple(ref_q.shape),
        tuple(ref_q.shape),
    )

    # QK-RMSNorm over head_dim, applied per head.
    ref_qn = ref_self.norm_q(ref_q.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
    q_n = tt_self.norm_q(q_tt, compute_kernel_config=norm_cfg)
    results.check("qk_norm", ref_qn, to_host(q_n))

    # RoPE. The reference applies it in [B, S, H, D] with sequence_dim=1; identical maths.
    from diffusers.models.embeddings import apply_rotary_emb

    cos_ref, sin_ref = R.rope_for(SEQ_LEN, config)
    ref_qr = apply_rotary_emb(
        ref_qn.permute(0, 2, 1, 3), (cos_ref, sin_ref), use_real=True, use_real_unbind_dim=-2, sequence_dim=1
    ).permute(0, 2, 1, 3)
    cos_tt, sin_tt = build_rope_tables(device, SEQ_LEN, head_dim=head_dim, theta=config.rope_theta, dtype=ttnn.bfloat16)
    for name, composite in (("rope_device", False), ("rope_composite", True)):
        got = apply_rope(q_n, cos_tt, sin_tt, composite=composite)
        results.check(name, ref_qr, to_host(got))
        ttnn.deallocate(got)
    ttnn.deallocate(q_n)
    ttnn.deallocate(q_tt)
    ttnn.deallocate(k_tt)
    ttnn.deallocate(v_tt)

    # Whole self-attention module. At S=32 the |i-j| <= 128 band masks nothing, so the
    # reference mask is all-zeros and the windowed TTNN path must agree exactly.
    mask = R.sliding_mask(SEQ_LEN, config.sliding_window)
    assert bool((mask == 0).all()), "at S=32 the sliding band should mask nothing"
    ref_out = ref_self(hidden_states=x_nsc, image_rotary_emb=(cos_ref, sin_ref), attention_mask=mask)
    got = tt_self(x_tt, rope=(cos_tt, sin_tt), window=config.window_for_layer(0))
    results.check("self_attn", ref_out, to_host(got))
    ttnn.deallocate(got)
    # ...and the global (odd-layer, window=None) path against the unmasked reference.
    ref_out_full = ref_self(hidden_states=x_nsc, image_rotary_emb=(cos_ref, sin_ref), attention_mask=None)
    got = tt_self(x_tt, rope=(cos_tt, sin_tt), window=config.window_for_layer(1))
    results.check("self_attn_global", ref_out_full, to_host(got))
    ttnn.deallocate(got)

    # ==================================================================== cross-attention ==
    ref_cross = R.reference_attention(config, is_cross=True, seed=SEED + 2)
    tt_cross = AceStepCrossAttention(config, mesh_device=device)
    tt_cross.load_torch_state_dict(ref_cross.state_dict())
    ref_cross_out = ref_cross(hidden_states=x_nsc, encoder_hidden_states=enc_nsc, attention_mask=None)
    kv = tt_cross.compute_kv(enc_tt)
    got = tt_cross(x_tt, kv)
    results.check("cross_attn", ref_cross_out, to_host(got))
    ttnn.deallocate(got)
    for t in kv:
        ttnn.deallocate(t)

    # =============================================================================== MLP ==
    ref_mlp = R.reference_mlp(config, seed=SEED + 3)
    tt_mlp = AceStepMLP(config, mesh_device=device)
    tt_mlp.load_torch_state_dict(ref_mlp.state_dict())
    got = tt_mlp(x_tt)
    results.check("mlp", ref_mlp(x_nsc), to_host(got))
    ttnn.deallocate(got)

    # ====================== whole-model pieces: 1-layer model exercises _prepare_torch_state
    # A single layer is enough: proj_in / proj_out / time_embed / condition_embedder /
    # norm_out / mod_out are all model-level, and running one layer keeps the fp32 CPU
    # reference (and the whole `_prepare_torch_state` path) cheap.
    config_1 = dataclasses.replace(config, num_hidden_layers=1)
    ref_model = R.reference_model(config_1, seed=SEED + 4)
    tt_model = AceStepTransformer1DModel(config_1, mesh_device=device)
    tt_model.load_torch_state_dict(ref_model.state_dict())

    # timestep sinusoid (host fp32) -> time_embed -> time_proj
    timestep = torch.tensor([0.9545])
    results.check(
        "time_sinusoid",
        ref_model.time_embed.time_sinusoid(timestep * ref_model.time_embed.scale),
        timestep_sinusoid(timestep),
    )
    ref_temb_t, ref_tp_t = ref_model.time_embed(timestep)
    chunks, temb_tt = tt_model.timestep_proj_chunks(timestep)
    results.check("time_embed", ref_temb_t, to_host(temb_tt))
    worst = 1.0
    for i in range(6):
        worst = min(worst, results.check(f"timestep_proj.{i}", ref_tp_t[:, i], to_host(chunks[i])))
    results["timestep_proj"] = (worst, tuple(ref_tp_t.shape), (6, 1, 1, 1, hidden))

    # condition_embedder
    got = tt_model.project_encoder_hidden_states(enc_tt)
    results.check("condition_embedder", ref_model.condition_embedder(enc_nsc), to_host(got))
    ttnn.deallocate(got)

    # patchify == proj_in_conv, unpatchify == proj_out_conv
    latent_t = 2 * SEQ_LEN
    hs = torch.randn(1, latent_t, config.audio_acoustic_hidden_dim) * 0.5
    ctx = torch.randn(1, latent_t, config.in_channels - config.audio_acoustic_hidden_dim) * 0.5
    patch_in = torch.cat([ctx, hs], dim=-1)
    ref_patch = ref_model.proj_in_conv(patch_in.transpose(1, 2)).transpose(1, 2)
    patched, original_t = tt_model.patchify(_to_11sc(patch_in, device))
    assert original_t == latent_t
    results.check("patchify", ref_patch, to_host(patched))
    ttnn.deallocate(patched)

    h_out = torch.randn(1, SEQ_LEN, hidden) * 0.5
    ref_unpatch = ref_model.proj_out_conv(h_out.transpose(1, 2)).transpose(1, 2)
    got = tt_model.unpatchify(_to_11sc(h_out, device), latent_t)
    results.check("unpatchify", ref_unpatch, to_host(got))
    ttnn.deallocate(got)

    # norm_out: the 2-way adaLN site, with temb_r folded in.
    ref_temb = ref_temb_t + ref_model.time_embed_r(timestep - timestep)[0]
    ref_shift, ref_scale = (ref_model.scale_shift_table + ref_temb.unsqueeze(1)).chunk(2, dim=1)
    ref_norm_out = ref_model.norm_out(h_out) * (1 + ref_scale) + ref_shift
    shift_o, one_plus_scale_o = tt_model.mod_out([temb_tt, temb_tt])
    got = rms_norm_modulated(
        tt_model.norm_out, _to_11sc(h_out, device), one_plus_scale_o, shift_o, compute_kernel_config=norm_cfg
    )
    results.check("norm_out", ref_norm_out, to_host(got))
    ttnn.deallocate(got)

    if verbose:
        print(f"\n=== DiT op-level PCC (S={SEQ_LEN}, target {TARGET_PCC}) ===")
        for name, (pcc, ref_shape, got_shape) in results.items():
            flag = "ok " if pcc >= TARGET_PCC else "FAIL"
            print(f"  [{flag}] {name:28s} pcc={pcc:.6f}  ref{ref_shape} tt{got_shape}")

    failures = {name: pcc for name, (pcc, *_rest) in results.items() if pcc < TARGET_PCC}
    return not failures, results, failures


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_dit_ops_pcc(device):
    passed, results, failures = run_dit_ops_pcc(device)
    assert passed, f"DiT op PCC below {TARGET_PCC}: {failures}"


if __name__ == "__main__":
    import sys
    import time

    dev = None
    for attempt in range(20):
        try:
            dev = ttnn.open_device(device_id=0, l1_small_size=32768)
            break
        except Exception as err:  # device momentarily busy (shared with other blocks)
            print(f"open_device attempt {attempt} failed ({err}); retrying in 45s")
            time.sleep(45)
    if dev is None:
        print("FAILED could not open device")
        sys.exit(1)
    try:
        ok, _results, fails = run_dit_ops_pcc(dev)
    finally:
        ttnn.close_device(dev)
    print(("PASSED" if ok else f"FAILED {fails}"))
    sys.exit(0 if ok else 1)
