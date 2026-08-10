# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""**Device-free** algebra test for the ACE-Step 1.5 DiT (Block 1).

Simulates the *exact* op sequence of ``tt/ttnn_ace_step_dit.py`` in torch fp32 and gates it
against Block 0's **real-weight** goldens. Everything that is a host-side decision is under
test here — the weight folds, the packing orders, the RoPE convention, the mask geometry, the
op ordering — so a failure on hardware after this passes isolates the cause to TTNN op
behaviour or bf16 precision, not to the algebra.

Specifically mirrored:

  * fused QKV ``Linear`` from ``cat([to_q, to_k, to_v], 0)``, then
    ``nlp_create_qkv_heads``' reshape + permute (TRAP-2: K/V stay at 8 heads);
  * per-head RMSNorm over ``head_dim``, fp32 variance;
  * HF **half-split** ``rotate_half`` RoPE (TRAP-8) with ``rope_tables_torch``;
  * non-causal SDPA with the ``|i-j| <= 128`` band (TRAP-1) and in-kernel GQA;
  * ``nlp_concat_heads``' permute + flatten, then ``to_out`` from ``to_out.0.weight``;
  * the folded 6-way ``Modulation`` (``time_embed_r`` constant + the ``(1 + scale)`` offset)
    and the **bare** gates;
  * tt_dit's fused-SwiGLU packing ``cat([up, gate], 0)`` run through
    ``prepare_for_fused_swiglu`` with the kernel's ``silu(even_tile) * odd_tile`` semantics;
  * the host fp32 timestep sinusoid (TRAP-9) and the last-dim 6-way chunk;
  * ``proj_in`` / ``proj_out`` as reshape + ``Linear``, including the ConvTranspose1d in/out
    axis swap and the ``p``-times-tiled bias.

Requires ``golden/dit/`` and ``$ACE_STEP_PIPELINE`` (Block 0); skips cleanly without them.
Runs in seconds and opens **no device**, so it is safe to run while another block has the board.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from models.common.utility_functions import comp_pcc
from models.experimental.ace_step_v15.tests import dit_reference as R
from models.experimental.ace_step_v15.tt.ttnn_ace_step_common import (
    TILE,
    AceStepDiTConfig,
    Modulation,
    conv1d_patch_to_linear,
    conv_transpose1d_patch_to_linear,
    fold_time_embed_r,
    rope_tables_torch,
    timestep_sinusoid,
)
from models.tt_dit.utils.substate import substate
from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

GOLDEN = str(R.GOLDEN_DIR)
#: fp32 simulation of an fp32 reference: anything below this is an algebra bug, not precision.
TARGET_PCC = 0.9999

#: (seq_len, sliding) pairs. Layer 0 is sliding_attention, layer 1 full_attention.
BLOCK_CASES = ((32, True), (32, False), (128, True), (128, False), (256, True), (768, True))
MODEL_CASES = (32, 128, 256, 768)


def _require_goldens(seq_len: int):
    try:
        goldens = R.DitGoldens(seq_len)
        R.real_dit_state_dict("layers.0")
    except (FileNotFoundError, KeyError) as err:
        pytest.skip(f"real-weight goldens unavailable: {err}")
    return goldens


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _fused_swiglu(x: torch.Tensor, ff1_weight: torch.Tensor, ff2_weight: torch.Tensor, inner: int) -> torch.Tensor:
    """tt_dit's fused path: pack -> one matmul -> ``silu(even tile) * odd tile`` -> ff2."""
    weight = prepare_for_fused_swiglu(ff1_weight.transpose(0, 1), ndev=1)  # [dim, 2 * inner]
    wide = x @ weight
    tiles = wide.reshape(*wide.shape[:-1], -1, TILE)
    hidden = (F.silu(tiles[..., 0::2, :]) * tiles[..., 1::2, :]).reshape(*wide.shape[:-1], inner)
    return hidden @ ff2_weight.transpose(0, 1)


def _band_mask(seq_len: int, bound: int) -> torch.Tensor:
    idx = torch.arange(seq_len)
    keep = (idx[:, None] - idx[None, :]).abs() <= bound
    mask = torch.zeros(1, 1, seq_len, seq_len)
    return mask.masked_fill_(~keep[None, None], torch.finfo(torch.float32).min)


def run_block_algebra(seq_len: int, sliding: bool, *, verbose: bool = True):
    goldens = _require_goldens(seq_len)
    config = AceStepDiTConfig.from_diffusers_config(goldens.meta["transformer_config"])
    index = 0 if sliding else 1
    assert config.is_sliding(index) == sliding
    heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
    head_dim, eps = config.head_dim, config.rms_norm_eps
    state = R.real_dit_state_dict(f"layers.{index}")
    results: dict[str, float] = {}

    def check(name, ref, got):
        _, pcc = comp_pcc(ref.float(), got.float(), pcc=0.0)
        results[name] = float(pcc)

    # --- folded modulation: exactly what Modulation holds plus what the model feeds it ---
    timestep_proj_r = goldens["time_embed_r.out1"][0]
    timestep_proj_t = goldens["time_embed.out1"]
    folded = Modulation.fold_state(state["scale_shift_table"], timestep_proj_r, num_chunks=6)
    mods = [folded[n].reshape(1, 1, -1) + timestep_proj_t[:, j] for j, n in enumerate(Modulation.NAMES_6)]
    shift, one_plus_scale, gate, c_shift, one_plus_c_scale, c_gate = mods

    x = goldens[f"layers.{index}.kw_hidden_states"]
    encoder = goldens.get(f"layers.{index}.cross_attn.kw_encoder_hidden_states")
    if encoder is None:
        encoder = goldens["condition_embedder.out"]

    # --- steps 1-8: self-attention ---
    h = _rms_norm(x, state["self_attn_norm.weight"], eps) * one_plus_scale + shift
    if goldens.has(f"layers.{index}.self_attn.kw_hidden_states"):
        check("adaLN(self_attn)", goldens[f"layers.{index}.self_attn.kw_hidden_states"], h)

    qkv_weight = torch.cat(
        [state["self_attn.to_q.weight"], state["self_attn.to_k.weight"], state["self_attn.to_v.weight"]], dim=0
    )
    fused = h @ qkv_weight.transpose(0, 1)
    q_width, kv_width = heads * head_dim, kv_heads * head_dim
    q = fused[..., :q_width].reshape(1, seq_len, heads, head_dim).permute(0, 2, 1, 3)
    k = fused[..., q_width : q_width + kv_width].reshape(1, seq_len, kv_heads, head_dim).permute(0, 2, 1, 3)
    v = fused[..., q_width + kv_width :].reshape(1, seq_len, kv_heads, head_dim).permute(0, 2, 1, 3)
    q = _rms_norm(q, state["self_attn.norm_q.weight"], eps)
    k = _rms_norm(k, state["self_attn.norm_k.weight"], eps)
    if goldens.has(f"layers.{index}.self_attn.norm_q.out"):
        check("qk_norm", goldens[f"layers.{index}.self_attn.norm_q.out"].permute(0, 2, 1, 3), q)

    cos, sin = rope_tables_torch(seq_len, head_dim, config.rope_theta)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin

    mask = _band_mask(seq_len, config.sliding_window) if sliding else None
    if sliding:
        # The band we build must be the reference's band, bit for bit.
        assert torch.equal(mask, R.sliding_mask(seq_len, config.sliding_window))
        assert torch.equal(mask, goldens[f"layers.{index}.kw_attention_mask"])
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=config.attention_scale, enable_gqa=True)
    out = out.permute(0, 2, 1, 3).reshape(1, seq_len, q_width) @ state["self_attn.to_out.0.weight"].transpose(0, 1)
    check("self_attn.out", goldens[f"layers.{index}.self_attn.out"], out)
    x = x + out * gate  # BARE gate

    # --- steps 9-11: cross-attention (unmodulated, ungated, no RoPE, no mask) ---
    h = _rms_norm(x, state["cross_attn_norm.weight"], eps)
    check("cross_attn_norm", goldens[f"layers.{index}.cross_attn.kw_hidden_states"], h)
    enc_len = encoder.shape[1]
    cq = _rms_norm(
        (h @ state["cross_attn.to_q.weight"].transpose(0, 1)).reshape(1, seq_len, heads, head_dim).permute(0, 2, 1, 3),
        state["cross_attn.norm_q.weight"],
        eps,
    )
    ck = _rms_norm(
        (encoder @ state["cross_attn.to_k.weight"].transpose(0, 1))
        .reshape(1, enc_len, kv_heads, head_dim)
        .permute(0, 2, 1, 3),
        state["cross_attn.norm_k.weight"],
        eps,
    )
    cv = (
        (encoder @ state["cross_attn.to_v.weight"].transpose(0, 1))
        .reshape(1, enc_len, kv_heads, head_dim)
        .permute(0, 2, 1, 3)
    )
    cross = F.scaled_dot_product_attention(cq, ck, cv, attn_mask=None, scale=config.attention_scale, enable_gqa=True)
    cross = cross.permute(0, 2, 1, 3).reshape(1, seq_len, q_width) @ state["cross_attn.to_out.0.weight"].transpose(0, 1)
    check("cross_attn.out", goldens[f"layers.{index}.cross_attn.out"], cross)
    x = x + cross  # plain residual

    # --- steps 12-14: SwiGLU MLP ---
    h = _rms_norm(x, state["mlp_norm.weight"], eps) * one_plus_c_scale + c_shift
    check("adaLN(mlp)", goldens[f"layers.{index}.mlp.in0"], h)
    ff1 = torch.cat([state["mlp.up_proj.weight"], state["mlp.gate_proj.weight"]], dim=0)
    ff = _fused_swiglu(h, ff1, state["mlp.down_proj.weight"], config.intermediate_size)
    check("mlp.out", goldens[f"layers.{index}.mlp.out"], ff)
    x = x + ff * c_gate
    check("BLOCK_OUT", goldens[f"layers.{index}.out"], x)

    if verbose:
        kind = "sliding_attention" if sliding else "full_attention"
        print(f"\n=== DiT algebra, block (S={seq_len}, layer {index} / {kind}, enc_L={enc_len}) ===")
        for name, pcc in results.items():
            print(f"  [{'ok  ' if pcc >= TARGET_PCC else 'FAIL'}] {name:22s} pcc={pcc:.8f}")

    failures = {n: p for n, p in results.items() if p < TARGET_PCC}
    return not failures, results, failures


def run_model_algebra(seq_len: int, *, verbose: bool = True):
    goldens = _require_goldens(seq_len)
    config = AceStepDiTConfig.from_diffusers_config(goldens.meta["transformer_config"])
    full = R.real_dit_state_dict()
    top = {k: v for k, v in full.items() if not k.startswith("layers.")}
    patch = config.patch_size
    results: dict[str, float] = {}

    def check(name, ref, got):
        _, pcc = comp_pcc(ref.float(), got.float(), pcc=0.0)
        results[name] = float(pcc)

    # --- timestep embedding: host fp32 sinusoid -> linear_1/silu/linear_2 -> silu -> proj ---
    freq = timestep_sinusoid(
        goldens["kw_timestep"], num_channels=config.time_embed_in_channels, scale=config.time_embed_scale
    )
    h = F.silu(F.linear(freq, top["time_embed.linear_1.weight"], top["time_embed.linear_1.bias"]))
    temb = F.linear(h, top["time_embed.linear_2.weight"], top["time_embed.linear_2.bias"])
    check("temb", goldens["time_embed.out0"], temb)
    flat = F.linear(F.silu(temb), top["time_embed.time_proj.weight"], top["time_embed.time_proj.bias"])
    # The reference unflattens to [B, 6, hidden] and chunks dim=1; chunking the flat last dim
    # is value-identical and keeps every boundary tile-aligned.
    chunks = flat.reshape(1, 1, -1).chunk(config.num_modulation_chunks, dim=-1)
    check("timestep_proj", goldens["time_embed.out1"], torch.cat([c.reshape(1, 1, -1) for c in chunks], dim=1))

    # --- the time_embed_r fold must reproduce the golden constants exactly ---
    r_state = substate(full, "time_embed_r")
    temb_r, timestep_proj_r = fold_time_embed_r(r_state)
    check("time_embed_r.temb", goldens["time_embed_r.out0"], temb_r)
    check("time_embed_r.proj", goldens["time_embed_r.out1"][0], timestep_proj_r)

    # --- proj_in as reshape + Linear ---
    x_ncl = goldens["proj_in_conv.in0"]
    latent_t = x_ncl.shape[-1]
    projected = F.linear(
        x_ncl.transpose(1, 2).reshape(1, latent_t // patch, patch * config.in_channels),
        conv1d_patch_to_linear(top["proj_in_conv.weight"], patch),
        top["proj_in_conv.bias"],
    )
    check("proj_in", goldens["proj_in_conv.out"].transpose(1, 2), projected)

    # --- norm_out (2-way adaLN, temb_r folded) + proj_out as Linear + reshape ---
    folded = Modulation.fold_state(top["scale_shift_table"], temb_r, num_chunks=2)
    out_shift = folded["shift"].reshape(1, 1, -1) + temb
    out_scale = folded["one_plus_scale"].reshape(1, 1, -1) + temb
    pre = goldens["norm_out.in0"]
    modulated = _rms_norm(pre, top["norm_out.weight"], config.rms_norm_eps) * out_scale + out_shift
    lin_w, lin_b = conv_transpose1d_patch_to_linear(top["proj_out_conv.weight"], top["proj_out_conv.bias"], patch)
    seq = modulated.shape[1]
    out = F.linear(modulated, lin_w, lin_b).reshape(1, patch * seq, config.audio_acoustic_hidden_dim)
    check("FINAL_OUT", goldens["out0"], out)

    if verbose:
        print(f"\n=== DiT algebra, model level (S={seq_len}, T={latent_t}) ===")
        for name, pcc in results.items():
            print(f"  [{'ok  ' if pcc >= TARGET_PCC else 'FAIL'}] {name:22s} pcc={pcc:.8f}")

    failures = {n: p for n, p in results.items() if p < TARGET_PCC}
    return not failures, results, failures


@pytest.mark.parametrize(
    ("seq_len", "sliding"), BLOCK_CASES, ids=[f"S{s}-{'sliding' if w else 'full'}" for s, w in BLOCK_CASES]
)
def test_dit_block_algebra(seq_len, sliding):
    passed, _results, failures = run_block_algebra(seq_len, sliding)
    assert passed, f"DiT block algebra below {TARGET_PCC}: {failures}"


@pytest.mark.parametrize("seq_len", MODEL_CASES, ids=[f"S{s}" for s in MODEL_CASES])
def test_dit_model_algebra(seq_len):
    passed, _results, failures = run_model_algebra(seq_len)
    assert passed, f"DiT model algebra below {TARGET_PCC}: {failures}"


if __name__ == "__main__":
    import sys

    ok = True
    for seq_len, sliding in BLOCK_CASES:
        passed, _res, _fail = run_block_algebra(seq_len, sliding)
        ok = ok and passed
    for seq_len in MODEL_CASES:
        passed, _res, _fail = run_model_algebra(seq_len)
        ok = ok and passed
    print("\nPASSED" if ok else "\nFAILED")
    sys.exit(0 if ok else 1)
