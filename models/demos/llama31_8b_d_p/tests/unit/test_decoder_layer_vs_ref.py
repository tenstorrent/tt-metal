# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""`tt/layer.py` vs an in-test fp32 torch decoder layer. Gate: `G-LAYER`.

The layer: `x + Attention(input_layernorm(x))`, then `x + MLP(post_attention_layernorm(x))`.
`(1,1)` mesh, so TP=1 and neither sublayer's TP collective is executed
(`bringup_log/04_CCL_PLAN.md` §5 rows 1-2 are P8's).

**This is an integration check and it may not stand in for a sublayer gate**
(`BRINGUP_RECIPE.md:1375-1392`). A layer PCC cannot localise: one bad sublayer moves an aggregate
that a dozen other causes also move. `G-RMS`, `G-ROPE`, `G-MLP` and `G-ATTN` are what localise, and
they are met on their own; what this file adds is the two things only assembly can get wrong — the
**residual wiring** and the **norm→sublayer pairing**.

* **Input distribution:** `x` standard normal `[1, 1, S, 4096]`, `S ∈ {128, 512, 2048}`; the seven
  projection weights `randn * 0.02` (the scale both templates use); the two norm gains
  `1 + randn * 0.02`, i.e. centred on the identity the way a real Llama gain is. Stated because it
  must never be chosen to pass. Both weight dtypes are run (bf8_b, the package default, and bf16).
* **Reference dtype policy:** fp32 weights, fp32 activations, fp32 arithmetic throughout; the causal
  mask is built explicitly as `triu(full((S,S), -inf), 1)` and the KV heads are `repeat_interleave`d
  by the GQA group, because the device does neither (recipe P1 trap 3, P5.5).
* **Computed noise floor:** the same fp32 layer with its **inputs and weights** rounded to the
  device dtypes and everything else in fp32 (recipe §2.2). Internal intermediates are *not*
  quantised — the device does store bf16 intermediates, and quantising them too would lower the
  floor and flatter every ratio it gates.
* **Negative control:** the two norm gains swapped (the recipe measured **0.9471** — note again how
  high a broken layer scores, which is §2.1's whole argument).

**On the error-ratio budget (`R-015`, `DEC-042`, and this file's `DEC-051`).** Appendix A gives
`G-LAYER` "PCC >= 0.999, <= 8x floor". This layer contains
`ttnn.transformer.scaled_dot_product_attention`, the fused kernel recipe §2.3 measures at 71x its
own modelled floor and whose slack accounted for the **entire** block-level gap at `G-ATTN`
(26.7-28.6x in-pipeline on this box). §2.3.1 is explicit that a single block-level ratio budget is
**not portable across dtypes** where a fixed-error stage dominates, and prescribes gating on the
**fused-kernel-attributed residual** instead. So this file measures, every run:

1. the raw ratio `(1-measured)/(1-floor)`, asserted where §2.3.1 says it holds (bf8_b);
2. a **kernel-attributed prediction**: the same torch floor layer with the device's *real* SDPA
   output substituted for the torch SDPA at the same stage. Because the substitution propagates
   through the rest of the layer, this handles the residual-add attenuation
   (`BRINGUP_RECIPE.md:1388-1389` measures it at 1.12x-1.73x) **exactly**, instead of assuming
   independent errors add in the layer's output space;
3. the residual `((1-measured) - kernel_excess)/(1-floor)`, asserted at <= 8x at **both** dtypes.
   Everything this package wrote lives in that residual.

Run:
    pytest models/demos/llama31_8b_d_p/tests/unit/test_decoder_layer_vs_ref.py -x -q
"""

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama31_8b_d_p.tests.test_factory import (
    err_ratio,
    llama_config_dims,
    load_hf_state_dict,
    quantize_like_device,
    requires_hf_reference,
)
from models.demos.llama31_8b_d_p.tests.unit.test_reference_model import build_cos_sin, causal_mask
from models.demos.llama31_8b_d_p.tests.unit.test_reference_model import decoder_layer as reference_decoder_layer
from models.demos.llama31_8b_d_p.tests.unit.test_reference_model import (
    hf_decoder_layer,
    random_layer_weights,
    real_layer_weights,
    run_hf_layer,
)
from models.demos.llama31_8b_d_p.tt.attention.config import ProgramConfig
from models.demos.llama31_8b_d_p.tt.attention.prefill import run_sdpa
from models.demos.llama31_8b_d_p.tt.config import MeshConfig, derive_head_dim
from models.demos.llama31_8b_d_p.tt.layer import DecoderLayer, _delta_stats, build_attention_config
from models.demos.llama31_8b_d_p.tt.rope import build_prefill_rope, build_transformation_mat, llama3_freqs

SEQ_LENS = [128, 512, 2048]
ACTIVATION_DTYPE = ttnn.bfloat16  # `DEC-022`
WEIGHT_SCALE = 0.02

PCC_THRESHOLD = 0.999  # `BRINGUP_RECIPE.md:1877` (Appendix A), stated at `:1370-1373`
MAX_BLOCK_ERR_RATIO = 8.0  # `BRINGUP_RECIPE.md:1877`

# `DEC-051`, following `DEC-042`: the raw 8x holds at bf8_b — the package's weight dtype — and does
# **not** hold at bf16, where a smaller floor error turns the same fixed fused-kernel slack into a
# bigger multiple (recipe §2.3.1's own table shows the *more accurate* dtype scoring the worse
# ratio). The kernel-attributed residual is asserted at BOTH dtypes and is the tighter test.
RAW_BLOCK_BUDGET_APPLIES = {ttnn.bfloat8_b: True, ttnn.bfloat16: False}

_DTYPES = [ttnn.bfloat8_b, ttnn.bfloat16]
_DTYPE_IDS = {ttnn.bfloat8_b: "bf8_b", ttnn.bfloat16: "bf16"}

_HF = llama_config_dims()
HIDDEN = _HF["hidden_size"]
NQ = _HF["num_attention_heads"]
NKV = _HF["num_key_value_heads"]
HEAD_DIM = derive_head_dim(_HF)
EPS = _HF["rms_norm_eps"]

_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_NORMS = ("input_layernorm", "post_attention_layernorm")


# ---------------------------------------------------------------------------------------------
# cos/sin: ONE frequency set, both conventions (the `_build_cos_sin` structure of
# `models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py:83`). The reference uses the HF
# pair, the device the Meta pair, and both come from `tt/rope.py::llama3_freqs`, so the test cannot
# silently compare two different RoPEs and call it a pass.
# ---------------------------------------------------------------------------------------------
def _hf_cos_sin(hf, seq_len):
    cos_half, sin_half = llama3_freqs(hf, seq_len)
    return torch.cat([cos_half, cos_half], dim=-1), torch.cat([sin_half, sin_half], dim=-1)


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _rope_hf(t, cos, sin):
    return t * cos + _rotate_half(t) * sin


def _meta_head_index(head_dim=HEAD_DIM):
    """`src` such that `meta = hf[..., src]` — `reverse_permute`'s head-dim map
    (`models/tt_transformers/tt/load_checkpoints.py:891`). A shared permutation of the head dim is
    orthogonal, so `q·kᵀ` and therefore the layer output are unchanged; only stage-level tensors
    need it, and the SDPA substitution below is one of those."""
    half = head_dim // 2
    return torch.tensor([half * (m % 2) + (m // 2) for m in range(head_dim)], dtype=torch.long)


# ---------------------------------------------------------------------------------------------
# The fp32 staged reference. Cross-checked against `test_reference_model.decoder_layer` (which is
# itself cross-checked against HF at `G-REF`) by `test_staged_reference_matches_g_ref`, so the
# stages this file needs cannot drift from the oracle the bring-up already gated.
# ---------------------------------------------------------------------------------------------
def _rms_norm(x, weight, eps=EPS):
    return weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def _torch_layer(x, w, cos, sin, *, sdpa_fn=None, swap_norm_gains=False):
    """One decoder layer in fp32, HF conventions. `x` is `[1, 1, S, hidden]`; returns every stage.

    `sdpa_fn(q_rot, k_rot, v) -> [1, nq, S, head_dim]` replaces the SDPA stage — the seam the
    kernel-attributed prediction uses, at layer level (`DEC-051`) and at model scale (`DEC-053`).
    Taking a callable rather than a precomputed tensor means the substituted chain runs the
    projections **once** per layer, which is what makes a 32-layer attribution affordable.
    `swap_norm_gains` is the negative control.
    """
    seq_len = x.shape[-2]
    gains = _NORMS[::-1] if swap_norm_gains else _NORMS
    x2d = x.float().reshape(seq_len, HIDDEN)

    h = _rms_norm(x2d, w[gains[0]].float())

    def _proj(name, heads):
        return (h @ w[name].float().transpose(-1, -2)).view(1, seq_len, heads, HEAD_DIM).transpose(1, 2)

    q, k, v = _proj("q_proj", NQ), _proj("k_proj", NKV), _proj("v_proj", NKV)
    q_rot, k_rot = _rope_hf(q, cos, sin), _rope_hf(k, cos, sin)

    if sdpa_fn is None:
        group = NQ // NKV
        scores = (q_rot @ k_rot.repeat_interleave(group, dim=1).transpose(-1, -2)) * (HEAD_DIM**-0.5)
        probs = torch.softmax(scores + causal_mask(seq_len)[0], dim=-1)
        ctx_heads = probs @ v.repeat_interleave(group, dim=1)
    else:
        ctx_heads = sdpa_fn(q_rot, k_rot, v).float()

    ctx = ctx_heads.transpose(1, 2).reshape(seq_len, NQ * HEAD_DIM)
    attn_out = ctx @ w["o_proj"].float().transpose(-1, -2)
    mid = x2d + attn_out

    h2 = _rms_norm(mid, w[gains[1]].float())
    mlp_out = F.silu(h2 @ w["gate_proj"].float().transpose(-1, -2)) * (h2 @ w["up_proj"].float().transpose(-1, -2))
    mlp_out = mlp_out @ w["down_proj"].float().transpose(-1, -2)
    out = mid + mlp_out
    return {
        "q_rot": q_rot,
        "k_rot": k_rot,
        "v": v,
        "ctx_heads": ctx_heads,
        "attn_out": attn_out.reshape(1, 1, seq_len, HIDDEN),
        "mid": mid.reshape(1, 1, seq_len, HIDDEN),
        "out": out.reshape(1, 1, seq_len, HIDDEN),
    }


# ---------------------------------------------------------------------------------------------
# Device side
# ---------------------------------------------------------------------------------------------
def _state_dict(w, *, swap_norm_gains=False):
    """This layer's HF-named keys, exactly as `Model` hands them to `DecoderLayer`."""
    gains = _NORMS[::-1] if swap_norm_gains else _NORMS
    state = {f"self_attn.{n}.weight": w[n] for n in _PROJECTIONS[:4]}
    state.update({f"mlp.{n}.weight": w[n] for n in _PROJECTIONS[4:]})
    state.update({f"{name}.weight": w[gain] for name, gain in zip(_NORMS, gains)})
    return state


def _quantize_weights(w, weight_dtype):
    """Quantise each weight as the device stores it: projections `[1,1,in,out]` at `weight_dtype`,
    norm gains `(1,1,hidden/32,32)` at bf16 (`tt/rms_norm.py`)."""
    out = {}
    for name in _PROJECTIONS:
        t = w[name].transpose(-1, -2).unsqueeze(0).unsqueeze(0)
        out[name] = quantize_like_device(t, weight_dtype)[0, 0].transpose(-1, -2)
    for name in _NORMS:
        shaped = w[name].reshape(1, 1, -1, ttnn.TILE_SIZE)
        out[name] = quantize_like_device(shaped, ttnn.bfloat16).reshape(-1)
    return out


def _to_device(t, mesh_device, dtype=ACTIVATION_DTYPE):
    return ttnn.from_torch(
        t,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _from_device(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def _build_layer(mesh_device, hf, w, *, weight_dtype, seq_len, swap_norm_gains=False):
    mesh_config = MeshConfig(tuple(mesh_device.shape), tp=mesh_device.shape[1])
    attention_config = build_attention_config(hf, max_seq_len=max(seq_len, ttnn.TILE_SIZE))
    layer = DecoderLayer(
        mesh_device,
        hf,
        _state_dict(w, swap_norm_gains=swap_norm_gains),
        0,
        mesh_config=mesh_config,
        attention_config=attention_config,
        transformation_mats={"prefill": build_transformation_mat(mesh_device)},
        weight_dtype=weight_dtype,
        max_seq_len=max(seq_len, ttnn.TILE_SIZE),
    )
    return layer, attention_config


def _device_sdpa(mesh_device, attention_config, q_hf, k_hf, v, seq_len):
    """The fused kernel alone, on the floor's own post-RoPE tensors, in Meta head space.

    Q/K are permuted into Meta space because that is where the device's swizzled weights put them;
    the context output is unaffected by a shared orthogonal permutation of the head dim, so the
    result drops straight back into the HF-space reference (the same construction `G-ATTN` uses)."""
    src = _meta_head_index()
    tt_out = run_sdpa(
        _to_device(q_hf[..., src], mesh_device),
        _to_device(k_hf[..., src], mesh_device),
        _to_device(v, mesh_device),
        attention_config,
        ProgramConfig(),
        mesh_device,
        seq_len,
    )
    out = _from_device(tt_out)
    tt_out.deallocate(True)
    return out


# ---------------------------------------------------------------------------------------------
# G-LAYER
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("weight_dtype", _DTYPES, ids=lambda d: _DTYPE_IDS[d])
@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=lambda s: f"s{s}")
def test_decoder_layer_vs_ref(mesh_device, weight_dtype, seq_len, reset_seeds):
    """The whole layer vs the fp32 reference: PCC >= 0.999, and the SDPA-attributed residual <= 8x."""
    hf = llama_config_dims()
    w = random_layer_weights()
    x = torch.randn(1, 1, seq_len, HIDDEN)
    cos, sin = _hf_cos_sin(hf, seq_len)

    ref = _torch_layer(x, w, cos, sin)["out"]

    # --- the floor: inputs and weights rounded to the device dtypes, everything else fp32 -------
    x_q = quantize_like_device(x, ACTIVATION_DTYPE)
    w_q = _quantize_weights(w, weight_dtype)
    cos_q = quantize_like_device(cos[None, None], ttnn.bfloat16)[0, 0]
    sin_q = quantize_like_device(sin[None, None], ttnn.bfloat16)[0, 0]
    floor_stages = _torch_layer(x_q, w_q, cos_q, sin_q)
    _, floor = comp_pcc(ref, floor_stages["out"], 0.0)
    floor = float(floor)

    # --- the device layer ----------------------------------------------------------------------
    layer, attention_config = _build_layer(mesh_device, hf, w, weight_dtype=weight_dtype, seq_len=seq_len)
    rope_mats = build_prefill_rope(mesh_device, hf, seq_len)
    out = _from_device(layer(_to_device(x, mesh_device), rope_mats))

    passing, pcc = comp_pcc(ref, out, PCC_THRESHOLD)
    pcc = float(pcc)
    ratio = err_ratio(pcc, floor)

    # --- attribute the gap to the fused kernel BEFORE judging it (recipe §2.3, §2.3.1) ----------
    # The device's real SDPA output, substituted into the floor layer at the SDPA stage. Because the
    # substitution then flows through o_proj, the residual add, the second norm and the MLP, the
    # residual-add attenuation is handled by construction rather than modelled.
    dev_ctx = _device_sdpa(
        mesh_device, attention_config, floor_stages["q_rot"], floor_stages["k_rot"], floor_stages["v"], seq_len
    )
    predicted_out = _torch_layer(x_q, w_q, cos_q, sin_q, sdpa_fn=lambda *_: dev_ctx)["out"]
    _, predicted = comp_pcc(ref, predicted_out, 0.0)
    predicted = float(predicted)

    floor_err = 1.0 - floor
    kernel_excess = (1.0 - predicted) - floor_err
    residual_ratio = ((1.0 - pcc) - kernel_excess) / floor_err

    logger.info(
        f"[G-LAYER] {_DTYPE_IDS[weight_dtype]} seq={seq_len}: PCC={pcc:.7f} floor={floor:.7f} "
        f"raw_ratio={ratio:.2f}x (threshold {PCC_THRESHOLD}, raw budget {MAX_BLOCK_ERR_RATIO}x "
        f"applies={RAW_BLOCK_BUDGET_APPLIES[weight_dtype]})"
    )
    logger.info(
        f"[G-LAYER] {_DTYPE_IDS[weight_dtype]} seq={seq_len} attribution: floor+deviceSDPA predicts "
        f"PCC={predicted:.7f} vs measured {pcc:.7f}; floor error={floor_err:.3e}; kernel excess "
        f"through the layer={kernel_excess:.3e} ({kernel_excess / floor_err:.2f}x the floor error); "
        f"SDPA-attributed residual ratio={residual_ratio:.2f}x"
    )

    assert passing, f"below threshold {PCC_THRESHOLD}: {pcc}"
    assert residual_ratio <= MAX_BLOCK_ERR_RATIO, (
        f"after attributing the fused SDPA kernel's own excess the layer is still "
        f"{residual_ratio:.2f}x off its floor — that residual is this package's own code "
        f"(norms, projections, RoPE, residual adds, MLP), so investigate before recording a PASS"
    )
    if RAW_BLOCK_BUDGET_APPLIES[weight_dtype]:
        assert ratio <= MAX_BLOCK_ERR_RATIO, f"layer is {ratio:.2f}x off its floor — attribute it before a PASS"


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("weight_dtype", _DTYPES, ids=lambda d: _DTYPE_IDS[d])
def test_decoder_layer_swapped_norm_gains_negative_control(mesh_device, weight_dtype, reset_seeds):
    """**The negative control:** the two norm gains swapped. The recipe measured **0.9471**.

    This is the one bug a layer gate exists to catch and no sublayer gate can: both norms are
    individually correct, both sublayers are individually correct, and the assembly is wrong.
    """
    hf = llama_config_dims()
    seq_len = 512
    w = random_layer_weights()
    x = torch.randn(1, 1, seq_len, HIDDEN)
    cos, sin = _hf_cos_sin(hf, seq_len)
    ref = _torch_layer(x, w, cos, sin)["out"]

    layer, _ = _build_layer(mesh_device, hf, w, weight_dtype=weight_dtype, seq_len=seq_len, swap_norm_gains=True)
    out = _from_device(layer(_to_device(x, mesh_device), build_prefill_rope(mesh_device, hf, seq_len)))
    _, pcc = comp_pcc(ref, out, 0.0)

    logger.info(f"[G-LAYER] control ({_DTYPE_IDS[weight_dtype]}): norm gains swapped -> PCC={float(pcc):.5f}")
    assert float(pcc) < PCC_THRESHOLD, (
        f"swapping the two norm gains still scores {float(pcc)} — the gate is not sensitive to "
        f"which norm feeds which sublayer"
    )


@requires_hf_reference
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_decoder_layer_real_weights_real_input_swapped_norms(mesh_device, reset_seeds):
    """**The control that discriminates: real layer-0 weights, real embedding-scale input.**

    Two weaker arms come first in this file, and *why* they are weak is the finding:

    | input | weights | swapped-gain PCC | max&#124;x&#124; | max&#124;attn_out&#124; |
    |---|---|---|---|---|
    | `randn` | random `*0.02` + random gains | 0.99864 | ~4 | ~0.5 |
    | `randn` | real layer-0 (all nine) | 0.99993 | 5.16 | **0.08** |
    | **real embeddings** | real layer-0 (all nine) | see below | ~0.06 | ~0.08 |

    A negative control on a residual block can only discriminate if the sublayer's output is
    comparable to the residual it is added to: for `y = r + s`, a perturbation of `s` is attenuated
    in `y` by exactly `||y||/||s||` (`BRINGUP_RECIPE.md:1388-1389`). With a standard-normal input
    and the **real** weights that ratio is about **65x**, because `randn` is ~100x larger than what
    layer 0 actually receives — `embed_tokens` rows have an RMS around 0.01, and the norm removes
    the input's scale anyway, so the sublayer output is scale-invariant while the residual is not.
    Feed `randn` and the residual drowns the sublayers; feed the real embedding rows and the
    sublayers dominate, which is the regime the model runs in.

    So this arm drives the device layer with **the real thing**: real layer-0 weights and the real
    `embed_tokens` rows for a fixed token sequence. It is the only place in `G-LAYER` where both the
    weights and the input are the model's own, and therefore the only place the norm-swap control
    can fail. Recipe §2.1(b) is right that the input distribution does not move the *floor*; it
    moves the **control's power**, which is a different quantity and is not interchangeable with it.

    The assertion is the gate's own (`PCC >= 0.999` must reject the swap), not a number invented
    after seeing the measurement.
    """
    hf = llama_config_dims()
    seq_len = 512
    w = real_layer_weights(0)
    # The real layer-0 input: `embed_tokens` rows for a fixed pseudo-random token sequence. Real
    # rows rather than a rescaled Gaussian, because the embedding carries Llama's massive-activation
    # channel structure and a Gaussian of the same RMS does not.
    table = load_hf_state_dict(prefixes=("model.embed_tokens.",))["model.embed_tokens.weight"].float()
    ids = torch.randint(0, table.shape[0], (seq_len,), generator=torch.Generator().manual_seed(0))
    x = table[ids].reshape(1, 1, seq_len, HIDDEN).contiguous()
    del table

    cos, sin = _hf_cos_sin(hf, seq_len)
    stages = _torch_layer(x, w, cos, sin)
    ref = stages["out"]
    attenuation = (ref.norm() / stages["attn_out"].norm()).item()
    logger.info(
        f"[G-LAYER] real weights + real embedding input: max|x|={x.abs().max():.4f} "
        f"(RMS {x.pow(2).mean().sqrt():.5f}), max|attn_out|={stages['attn_out'].abs().max():.4f}, "
        f"max|out|={ref.abs().max():.4f}; residual attenuation ||y||/||attn_out||={attenuation:.2f}x"
    )

    good, _ = _build_layer(mesh_device, hf, w, weight_dtype=ttnn.bfloat8_b, seq_len=seq_len)
    _, pcc_good = comp_pcc(
        ref, _from_device(good(_to_device(x, mesh_device), build_prefill_rope(mesh_device, hf, seq_len))), 0.0
    )
    bad, _ = _build_layer(mesh_device, hf, w, weight_dtype=ttnn.bfloat8_b, seq_len=seq_len, swap_norm_gains=True)
    _, pcc_bad = comp_pcc(
        ref, _from_device(bad(_to_device(x, mesh_device), build_prefill_rope(mesh_device, hf, seq_len))), 0.0
    )

    floor_out = _torch_layer(
        quantize_like_device(x, ACTIVATION_DTYPE),
        _quantize_weights(w, ttnn.bfloat8_b),
        quantize_like_device(cos[None, None], ttnn.bfloat16)[0, 0],
        quantize_like_device(sin[None, None], ttnn.bfloat16)[0, 0],
    )["out"]
    _, floor = comp_pcc(ref, floor_out, 0.0)

    logger.info(
        f"[G-LAYER] real weights + real input (bf8_b): correct={float(pcc_good):.7f} "
        f"floor={float(floor):.7f} ratio={err_ratio(float(pcc_good), float(floor)):.2f}x; "
        f"norm gains swapped={float(pcc_bad):.5f} (recipe measured 0.9471 for this control)"
    )
    assert float(pcc_good) >= PCC_THRESHOLD, f"the correct arm failed the gate on real weights: {pcc_good}"
    assert float(pcc_bad) < PCC_THRESHOLD, (
        f"swapping the two norm gains on real weights and a real input still scores "
        f"{float(pcc_bad)} — the gate is not sensitive to which norm feeds which sublayer"
    )


@torch.no_grad()
def test_staged_reference_matches_g_ref(reset_seeds):
    """This file's staged reference must be the **same math** as the `G-REF` oracle.

    Host-only. Without it, the layer gate would be scored against a reference nothing has ever
    checked — and `G-REF` is where the oracle was checked against HF `LlamaDecoderLayer`
    (bit-exact, `raw/G-REF_20260904T035140Z.log`). Expect bit-equality up to fp32 reassociation.
    """
    hf = llama_config_dims()
    seq_len = 128
    w = random_layer_weights()
    x = torch.randn(1, 1, seq_len, HIDDEN)
    cos, sin = _hf_cos_sin(hf, seq_len)

    staged = _torch_layer(x, w, cos, sin)["out"]
    # The G-REF oracle takes `[1, S, hidden]` and its own cos/sin builder.
    g_ref_cos, g_ref_sin = build_cos_sin(seq_len)
    oracle = reference_decoder_layer(x.reshape(1, seq_len, HIDDEN), w, g_ref_cos, g_ref_sin, causal_mask(seq_len))

    _, pcc = comp_pcc(staged, oracle.reshape(1, 1, seq_len, HIDDEN), 0.0)
    max_delta = (staged - oracle.reshape(1, 1, seq_len, HIDDEN)).abs().max().item()
    cos_delta = (cos - g_ref_cos).abs().max().item()
    logger.info(
        f"[G-LAYER] staged reference vs the G-REF oracle: PCC={float(pcc):.9f} max|delta|={max_delta:.3e}; "
        f"cos tables agree to {cos_delta:.3e} (tt/rope.py's llama3_freqs vs the test's own transcription)"
    )
    assert float(pcc) > 0.99999999, f"the staged reference disagrees with the G-REF oracle: {pcc}"


@torch.no_grad()
def test_staged_reference_matches_hf_decoder_layer(reset_seeds):
    """And the same against HF's own `LlamaDecoderLayer`, which is what `G-LAYER` is nominally
    scored against (`BRINGUP_RECIPE.md:1370-1371` allows either).

    Host-only. Runs the HF module in fp32 with `_attn_implementation="eager"` and an **explicit**
    causal mask, because `attention_mask=None` is silently non-causal (recipe P1 trap 3).
    """
    hf = llama_config_dims()
    seq_len = 128
    w = random_layer_weights()
    x = torch.randn(1, 1, seq_len, HIDDEN)
    cos, sin = _hf_cos_sin(hf, seq_len)

    staged = _torch_layer(x, w, cos, sin)["out"]
    hf_out = run_hf_layer(hf_decoder_layer(w), x.reshape(1, seq_len, HIDDEN), cos, sin, causal_mask(seq_len))

    _, pcc = comp_pcc(staged, hf_out.reshape(1, 1, seq_len, HIDDEN), 0.0)
    logger.info(f"[G-LAYER] staged reference vs HF LlamaDecoderLayer: PCC={float(pcc):.9f}")
    assert float(pcc) > 0.99999999, f"the staged reference disagrees with HF LlamaDecoderLayer: {pcc}"


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_layer_is_causal(mesh_device, reset_seeds):
    """**Causality, asserted directly on the device layer.** Perturb the last token; rows `[:-1]`
    must be unchanged at `max|delta| = 0`.

    Recipe P1 trap 3 makes this a reference-side check; running it on the *device* layer is what
    proves the SDPA `is_causal=True` flag reached the kernel, which no PCC against a causal
    reference can distinguish from a reference that is also non-causal.
    """
    hf = llama_config_dims()
    seq_len = 128
    w = random_layer_weights()
    x = torch.randn(1, 1, seq_len, HIDDEN)
    x_perturbed = x.clone()
    x_perturbed[0, 0, -1] += 1.0

    layer, _ = _build_layer(mesh_device, hf, w, weight_dtype=ttnn.bfloat16, seq_len=seq_len)
    rope_mats = build_prefill_rope(mesh_device, hf, seq_len)
    base = _from_device(layer(_to_device(x, mesh_device), rope_mats))
    perturbed = _from_device(layer(_to_device(x_perturbed, mesh_device), rope_mats))

    delta_prefix = (base[:, :, :-1] - perturbed[:, :, :-1]).abs().max().item()
    delta_last = (base[:, :, -1] - perturbed[:, :, -1]).abs().max().item()
    logger.info(f"[G-LAYER] causality: max|delta| rows[:-1]={delta_prefix:.3e}, last row={delta_last:.3e}")
    assert delta_prefix == 0.0, f"perturbing the last token changed earlier rows by {delta_prefix:.3e} — not causal"
    assert delta_last > 0.0, "perturbing the last token changed nothing at all — the probe is blind"


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_delta_probe_runs_and_survives_garbage(mesh_device, reset_seeds):
    """The `LLAMA_DELTA_PROBE` helper (`DEC-023`) must log on a real tensor and **never raise**.

    It carries the package's one `except Exception` (recipe §0 rule 5 allows it here because a
    bring-up probe must not break a run), so the except path is exercised deliberately rather than
    left as an untested claim.
    """
    tensor = _to_device(torch.randn(1, 1, ttnn.TILE_SIZE, HIDDEN), mesh_device)
    _delta_stats("real_tensor", 0, tensor)
    _delta_stats("not_a_tensor", 7, object())  # must warn, not raise
