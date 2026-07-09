# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end PCC for the Qwen3.6-27B Gated DeltaNet (linear_attention) layer, TP path.

Two tests, one per generation phase. Each test runs the **exact torch implementation
of the model** as the reference and the **exact code path the text demo drives** as the
TTNN device-under-test, then compares full-precision reference vs on-device output via PCC:

* ``test_gdn_layer_prefill`` — chunk-prefill of a T-token prompt (B=1). The TTNN side is
  ``TPGatedDeltaNet.forward_prefill(chunk_size=128, capture_state=True)`` with
  ``_stable_state=True`` — the exact call the demo's captured prefill-chunk trace makes
  (model.py ``_forward_prefill_chunk_tp`` -> layer.py -> tp.py). The reference is the HF
  ``Qwen3_5GatedDeltaNet`` chunk path (``torch_chunk_gated_delta_rule``).

* ``test_gdn_layer_decode`` — seed the recurrent + conv state with a T_p-token prefill
  (mirroring the demo: prefill establishes state, decode continues from it), then decode
  T_d tokens one at a time via ``TPGatedDeltaNet.forward_decode`` — the exact per-step
  call the demo's decode trace makes. The reference is the HF recurrent path
  (``torch_recurrent_gated_delta_rule``) run over the *same* full T_p+T_d sequence; the
  decode-region outputs must match the reference at those positions.

The torch references are the model's own functions, imported straight from
``transformers.models.qwen3_5.modeling_qwen3_5`` (the ``torch_*`` fallbacks, which are what
the model runs when the FLA/causal-conv1d fast kernels are absent), so the reference is not
a hand-rewrite — it is the reference model.

Run:
    MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B \
      pytest models/demos/blackhole/qwen36/tests/test_gdn_layer_tp.py -v -s
"""
import os

import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.test_factory import (
    get_pcc_threshold,
    load_gdn_layer,
    model_path,
    parametrize_mesh_tp,
    replicate_to_device,
    tp_composer,
)
from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs


# --------------------------------------------------------------------------- #
# Torch reference — the model's own Gated DeltaNet forward.
#
# Mirrors transformers Qwen3_5GatedDeltaNet.forward (modeling_qwen3_5.py):
#   mixed_qkv = in_proj_qkv(x); depthwise causal conv1d + SiLU; split q/k/v;
#   reshape per head; GVA repeat_interleave(Nv//Nk); beta = sigmoid(in_proj_b(x));
#   g = -exp(A_log) * softplus(in_proj_a(x) + dt_bias); delta rule (chunk|recurrent)
#   with use_qk_l2norm_in_kernel=True; gated RMSNorm(o) * SiLU(z) [weight only, no +1];
#   out_proj.
# --------------------------------------------------------------------------- #
def _gdn_torch_reference(sd, x, args, delta_rule):
    """Full-sequence GDN reference. ``x``: [1, T, dim] float32. Returns [T, dim] float32.

    ``delta_rule`` is the HF ``torch_chunk_gated_delta_rule`` (prefill) or
    ``torch_recurrent_gated_delta_rule`` (decode) — the two kernels the model dispatches.
    """
    Nk, Nv, Dk, Dv = args.gdn_nk, args.gdn_nv, args.gdn_dk, args.gdn_dv
    key_dim, value_dim = args.gdn_key_dim, args.gdn_value_dim
    K = args.gdn_conv_kernel_size
    conv_dim = 2 * key_dim + value_dim
    B, T, _ = x.shape

    p = "linear_attn."
    Wqkv = sd[f"{p}in_proj_qkv.weight"].float()
    Wz = sd[f"{p}in_proj_z.weight"].float()
    Wa = sd[f"{p}in_proj_a.weight"].float()
    Wb = sd[f"{p}in_proj_b.weight"].float()
    Wout = sd[f"{p}out_proj.weight"].float()
    conv_w = sd[f"{p}conv1d.weight"].float()  # [conv_dim, 1, K]
    A_log = sd[f"{p}A_log"].float()
    dt_bias = sd[f"{p}dt_bias"].float()
    norm_w = sd[f"{p}norm.weight"].float()  # [Dv]

    # 1. projections
    mixed_qkv = F.linear(x, Wqkv)  # [1, T, conv_dim]
    z = F.linear(x, Wz)  # [1, T, value_dim]
    b = F.linear(x, Wb)  # [1, T, Nv]
    a = F.linear(x, Wa)  # [1, T, Nv]

    # 2. depthwise causal conv1d + SiLU (HF: nn.Conv1d(padding=K-1)[..., :T] == left-pad K-1)
    mixed_qkv = mixed_qkv.transpose(1, 2)  # [1, conv_dim, T]
    mixed_qkv = F.conv1d(F.pad(mixed_qkv, (K - 1, 0)), conv_w, bias=None, groups=conv_dim)
    mixed_qkv = F.silu(mixed_qkv).transpose(1, 2)  # [1, T, conv_dim]

    # 3. split + reshape to heads
    q, k, v = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
    q = q.reshape(B, T, Nk, Dk)
    k = k.reshape(B, T, Nk, Dk)
    v = v.reshape(B, T, Nv, Dv)
    rf = Nv // Nk
    q = q.repeat_interleave(rf, dim=2)  # [1, T, Nv, Dk]
    k = k.repeat_interleave(rf, dim=2)

    # 4. beta / g gates
    beta = b.sigmoid()  # [1, T, Nv]
    g = -A_log.exp() * F.softplus(a + dt_bias)  # [1, T, Nv]

    # 5. delta rule (l2norm of q,k + scale happen inside the kernel)
    o, _ = delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )  # [1, T, Nv, Dv]

    # 6. gated RMSNorm over Dv (weight only, NO +1) then SiLU(z) gate
    o = o.reshape(-1, Dv).float()
    var = o.pow(2).mean(-1, keepdim=True)
    o = o * torch.rsqrt(var + 1e-6)
    o = o * norm_w
    o = o * F.silu(z.reshape(-1, Dv).float())
    o = o.reshape(B, T, value_dim)

    # 7. output projection
    return F.linear(o, Wout)[0]  # [T, dim]


def _build_gdn(mesh_device, B, max_seq_len=256):
    """Load one GDN layer's weights and build the TP module the demo uses."""
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=B, max_seq_len=max_seq_len)
    nd = mesh_device.get_num_devices()
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    logger.info(f"devices={nd} gdn layer={li} Nk={args.gdn_nk} Nv={args.gdn_nv} Dk={args.gdn_dk} Dv={args.gdn_dv}")

    sd = load_gdn_layer(args.CKPT_DIR, li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    tw = load_gdn_weights_tp(mesh_device, sd, args)
    gdn = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)
    return args, sd, gdn


@torch.no_grad()
@parametrize_mesh_tp()
def test_gdn_layer_prefill(mesh_device, reset_seeds, ensure_gc, request):
    """GDN chunk-prefill: TTNN forward_prefill (demo trace path) vs HF chunk reference.

    B=1, single prompt of T=128 tokens (one full 128-chunk, valid_len=None — the demo's
    common trace path). Compares the full [T, dim] output.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    T = 128
    args, sd, gdn = _build_gdn(mesh_device, B=1)

    x = torch.randn(1, 1, T, args.dim, dtype=torch.bfloat16)
    x_tt = replicate_to_device(mesh_device, x)

    # ---- TTNN: exactly as the demo's captured prefill chunk drives it ----
    gdn._stable_state = True
    gdn.reset_state()
    out_pf = gdn.forward_prefill(x_tt, chunk_size=128, valid_len=None, capture_state=True)
    pf = ttnn.to_torch(out_pf, mesh_composer=tp_composer(mesh_device))[0, 0].float()  # [T, dim]
    assert pf.shape == (T, args.dim) and not torch.isnan(pf).any() and pf.abs().max() > 0

    # ---- torch reference (model's chunk path) ----
    ref = _gdn_torch_reference(sd, x[0].float(), args, torch_chunk_gated_delta_rule)  # [T, dim]

    passing, pcc = comp_pcc(ref, pf, get_pcc_threshold(request))
    logger.info(f"GDN prefill PCC (T={T}) = {pcc}")
    assert passing, f"GDN prefill PCC too low: {pcc}"


@torch.no_grad()
@parametrize_mesh_tp()
def test_gdn_layer_decode(mesh_device, reset_seeds, ensure_gc, request):
    """GDN decode: seed state with a prefill, then per-token forward_decode vs HF recurrent ref.

    Mirrors the demo: a T_p-token prefill establishes the recurrent + conv state, then
    T_d tokens are decoded one at a time (state carried in place, _stable_state=True). The
    reference runs the model's recurrent kernel over the SAME T_p+T_d sequence; the decode
    outputs must match the reference at positions T_p .. T_p+T_d-1.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule

    T_p, T_d = 128, 8
    T = T_p + T_d
    args, sd, gdn = _build_gdn(mesh_device, B=1)
    composer = tp_composer(mesh_device)

    x = torch.randn(1, 1, T, args.dim, dtype=torch.bfloat16)

    # ---- TTNN: prefill to seed state, then decode token-by-token (demo flow) ----
    gdn._stable_state = True
    gdn.reset_state()
    x_pf = replicate_to_device(mesh_device, x[:, :, :T_p, :])
    gdn.forward_prefill(x_pf, chunk_size=128, valid_len=None, capture_state=True)

    dec_rows = []
    for t in range(T_p, T):
        xt = replicate_to_device(mesh_device, x[:, :, t : t + 1, :])
        ot = gdn.forward_decode(xt)
        dec_rows.append(ttnn.to_torch(ot, mesh_composer=composer)[0, 0, 0].float())  # [dim]
    dec = torch.stack(dec_rows, dim=0)  # [T_d, dim]
    assert dec.shape == (T_d, args.dim) and not torch.isnan(dec).any() and dec.abs().max() > 0

    # ---- torch reference (model's recurrent path) over the full sequence ----
    ref_full = _gdn_torch_reference(sd, x[0].float(), args, torch_recurrent_gated_delta_rule)  # [T, dim]
    ref = ref_full[T_p:T]  # decode region

    passing, pcc = comp_pcc(ref, dec, get_pcc_threshold(request))
    logger.info(f"GDN decode PCC (T_p={T_p}, T_d={T_d}) = {pcc}")
    assert passing, f"GDN decode PCC too low: {pcc}"
