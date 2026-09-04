# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""`tt/attention/` vs an in-test fp32 torch reference. Gate: `G-ATTN`.

The block: QKV proj -> GQA head split (32 Q / 8 KV, group 4) -> full llama3-scaled RoPE on **Q and
K only** -> causal SDPA -> concat heads -> `o_proj`. `(1,1)` mesh, so TP=1 and the module's TP
all-reduce tail is not executed (`bringup_log/04_CCL_PLAN.md` §5 row 1 is P8's).

* **Input distribution:** `x` standard normal `[1, 1, S, 4096]`, `S ∈ {128, 512, 2048}`; the four
  projection weights `randn * 0.02`, the scale both templates use
  (`models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py:169`). Stated because it must never
  be chosen to pass.
* **Reference dtype policy:** fp32 weights, fp32 activations, fp32 arithmetic; the causal mask is
  built **explicitly** as `torch.triu(full((S,S), -inf), diagonal=1)` and the KV heads are
  `repeat_interleave`d by the GQA group, because the device does neither (SDPA is causal and
  group-aware internally). Reference adapted from
  `models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py:117` with the sink column and the
  sliding term removed.
* **One frequency set, two conventions.** cos/sin come from the package's own
  `tt/rope.py::llama3_freqs`; the reference uses the **HF** pair (`rotate_half` over the full head)
  and the device uses the **Meta** pair via `tt/rope.py::build_prefill_rope`. Copying
  `_build_cos_sin`'s structure (`models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py:83`)
  is what stops the test silently comparing two different RoPEs and calling it a pass.
* **The Meta head permutation.** Because `q_proj`/`k_proj` are `reverse_permute`d at load, the
  device's Q and K live in Meta-interleaved head space. A shared permutation of the head dim is
  orthogonal, so `q·kᵀ` — and therefore the block output — is unchanged; only the **stage-level**
  Q/K comparisons need the reference permuted, exactly as
  `models/demos/minimax_m3/tests/unit/test_kv_cache_write_vs_ref.py:128-135` does it.
* **Thresholds** (`BRINGUP_RECIPE.md:2070`): whole block PCC >= **0.999** and **<= 8x** its floor;
  each stage **this package implements** <= **3x** its own floor. The stage budgets are measured
  **stage-isolated** — each stage is fed the *reference's* input, quantised to the device dtype —
  because a stage fed the previous stage's device output would be measuring the accumulation, not
  the stage (recipe §2.3).
* **The fused SDPA kernel is excluded from the 3x budget and probed separately**, permanently.
  Recipe §2.3 measured it at 71x its own modelled floor while every hand-written stage sat at
  1.0-1.5x, and a budget that lumps the two together can absorb a real regression unnoticed.
* **Negative control:** `q_proj`/`k_proj` reaching the device **without** the Meta swizzle (recipe
  measured 0.9475 — note how high a badly broken variant scores). Constructed by pre-applying
  `load_checkpoints.permute`, the exact inverse of the loader's `reverse_permute`, so the control
  runs the **real loader** and cannot drift from it.
* **Invariant:** only Q and K are rotated. Asserted by scoring the device against a reference that
  also rotates V and requiring that reference to fit **worse**.

Run:
    pytest models/demos/llama31_8b_d_p/tests/unit/test_attention_vs_ref.py -x -q
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama31_8b_d_p.tests.test_factory import err_ratio, llama_config_dims, quantize_like_device
from models.demos.llama31_8b_d_p.tt.attention import Attention, AttentionConfig, ProgramConfig
from models.demos.llama31_8b_d_p.tt.attention.operations import (
    apply_output_projection,
    apply_qkv_projection,
    apply_rope,
    concat_heads,
    split_qkv_heads_prefill,
)
from models.demos.llama31_8b_d_p.tt.attention.prefill import run_sdpa
from models.demos.llama31_8b_d_p.tt.attention.weights import load_attention_weights
from models.demos.llama31_8b_d_p.tt.config import MeshConfig, derive_head_dim
from models.demos.llama31_8b_d_p.tt.rope import build_prefill_rope, build_transformation_mat, llama3_freqs
from models.tt_transformers.tt.load_checkpoints import permute

SEQ_LENS = [128, 512, 2048]
ACTIVATION_DTYPE = ttnn.bfloat16  # `DEC-022`
WEIGHT_SCALE = 0.02

BLOCK_PCC_THRESHOLD = 0.999  # `BRINGUP_RECIPE.md:2070`
MAX_BLOCK_ERR_RATIO = 8.0
MAX_STAGE_ERR_RATIO = 3.0
SDPA_PROBE_PCC_THRESHOLD = 0.999  # the fused kernel measured 0.9999204 in the recipe's own run

# `BRINGUP_RECIPE.md:2070`'s 8x block budget, applied to the raw ratio. It holds at bf8_b — the
# package's weight dtype (`DEC-022`) and the dtype recipe §2.4's own attention-block row leads with
# — and it does **not** hold at bf16, where a smaller floor error makes the same fused-kernel slack
# a bigger multiple. `DEC-042` records the measurement, the attribution and the arithmetic; both
# dtypes are always gated on the SDPA-attributed residual below, so a regression in a hand-written
# stage still moves a number at bf16.
RAW_BLOCK_BUDGET_APPLIES = {ttnn.bfloat8_b: True, ttnn.bfloat16: False}

_DTYPES = [ttnn.bfloat8_b, ttnn.bfloat16]
_DTYPE_IDS = {ttnn.bfloat8_b: "bf8_b", ttnn.bfloat16: "bf16"}


# --------------------------------------------------------------------------------------------
# cos/sin: ONE frequency set, both conventions.
# --------------------------------------------------------------------------------------------
def _hf_cos_sin(hf, seq_len):
    """The HF-convention `[S, head_dim]` cos/sin — halves concatenated, for `rotate_half`."""
    cos_half, sin_half = llama3_freqs(hf, seq_len)
    return torch.cat([cos_half, cos_half], dim=-1), torch.cat([sin_half, sin_half], dim=-1)


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _rope_hf(t, cos, sin):
    return t * cos + _rotate_half(t) * sin


def _meta_head_index(head_dim):
    """`src` such that `meta = hf[..., src]`: Meta index `2t+j` reads HF index `(head_dim/2)*j + t`.

    Derived from `models/tt_transformers/tt/load_checkpoints.py:891` `reverse_permute`, which views
    the head dim as `(2, head_dim/2)` and transposes it.
    """
    half = head_dim // 2
    return torch.tensor([half * (m % 2) + (m // 2) for m in range(head_dim)], dtype=torch.long)


# --------------------------------------------------------------------------------------------
# Torch reference, fp32, HF convention. Returns every stage so each can carry its own budget.
# --------------------------------------------------------------------------------------------
def _torch_stages(x, w, cos, sin, dims, *, rotate_v=False):
    """fp32 reference stages for `x [1, 1, S, hidden]`.

    Keys: `q`/`k`/`v` (post head split, pre-RoPE), `q_rot`/`k_rot`, `ctx_heads` (SDPA output,
    `[1, nq, S, head_dim]`), `ctx` (concat heads), `out` (post `o_proj`). All in **HF** head space;
    the caller permutes the ones the device holds in Meta space.
    """
    nq, nkv, head_dim = dims["num_heads"], dims["num_kv_heads"], dims["head_dim"]
    seq_len = x.shape[-2]
    flat = x.float().reshape(seq_len, -1)

    def _proj(name, heads):
        return (flat @ w[name].float().transpose(-1, -2)).view(1, seq_len, heads, head_dim).transpose(1, 2)

    q, k, v = _proj("q_proj", nq), _proj("k_proj", nkv), _proj("v_proj", nkv)
    q_rot, k_rot = _rope_hf(q, cos, sin), _rope_hf(k, cos, sin)
    # The invariant probe: a reference that also rotates V must fit the device WORSE.
    v_used = _rope_hf(v, cos, sin) if rotate_v else v

    group = nq // nkv
    k_full = k_rot.repeat_interleave(group, dim=1)
    v_full = v_used.repeat_interleave(group, dim=1)

    scores = (q_rot @ k_full.transpose(-1, -2)) * (head_dim**-0.5)
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    probs = torch.softmax(scores + mask, dim=-1)
    ctx_heads = probs @ v_full  # [1, nq, S, head_dim]

    ctx = ctx_heads.transpose(1, 2).reshape(1, 1, seq_len, nq * head_dim)
    out = ctx @ w["o_proj"].float().transpose(-1, -2)
    return {
        "q": q,
        "k": k,
        "v": v,
        "q_rot": q_rot,
        "k_rot": k_rot,
        "ctx_heads": ctx_heads,
        "ctx": ctx,
        "out": out,
    }


def _sdpa_reference(q_rot, k_rot, v, dims):
    """The fused kernel's own reference, in whatever head space it is handed."""
    nq, nkv, head_dim = dims["num_heads"], dims["num_kv_heads"], dims["head_dim"]
    seq_len = q_rot.shape[-2]
    group = nq // nkv
    scores = (q_rot @ k_rot.repeat_interleave(group, dim=1).transpose(-1, -2)) * (head_dim**-0.5)
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    return torch.softmax(scores + mask, dim=-1) @ v.repeat_interleave(group, dim=1)


# --------------------------------------------------------------------------------------------
# Fixtures-as-helpers
# --------------------------------------------------------------------------------------------
def _dims(hf):
    return {
        "hidden_size": hf["hidden_size"],
        "num_heads": hf["num_attention_heads"],
        "num_kv_heads": hf["num_key_value_heads"],
        "head_dim": derive_head_dim(hf),
    }


def _random_weights(dims):
    """HF `[out, in]` layout, identical on both sides."""
    hidden, nq, nkv, hd = dims["hidden_size"], dims["num_heads"], dims["num_kv_heads"], dims["head_dim"]
    return {
        "q_proj": torch.randn(nq * hd, hidden) * WEIGHT_SCALE,
        "k_proj": torch.randn(nkv * hd, hidden) * WEIGHT_SCALE,
        "v_proj": torch.randn(nkv * hd, hidden) * WEIGHT_SCALE,
        "o_proj": torch.randn(hidden, nq * hd) * WEIGHT_SCALE,
    }


def _state_dict(w, *, unswizzle_qk=False, head_dim=128):
    """`{"<proj>.weight": tensor}`. `unswizzle_qk` is the negative control.

    `permute` (`models/tt_transformers/tt/load_checkpoints.py:895`) is the exact inverse of the
    loader's `reverse_permute` (`:891`), so pre-applying it makes the loader's swizzle cancel and
    the device ends up with **HF-layout** Q/K weights — i.e. "loaded without the Meta permute",
    reached through the real loader rather than by bypassing it.
    """
    state = {f"{name}.weight": tensor for name, tensor in w.items()}
    if unswizzle_qk:
        for name in ("q_proj", "k_proj"):
            tensor = w[name]
            dim1, dim2 = tensor.shape
            state[f"{name}.weight"] = permute(tensor, dim1 // head_dim, dim1, dim2)
    return state


def _quantize_weights(w, weight_dtype):
    """Quantise each weight in the `[1, 1, in, out]` orientation the device stores it in."""
    return {
        name: quantize_like_device(tensor.transpose(-1, -2).unsqueeze(0).unsqueeze(0), weight_dtype)[0, 0].transpose(
            -1, -2
        )
        for name, tensor in w.items()
    }


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


def _build_attention(mesh_device, hf, dims, w, *, weight_dtype, seq_len, unswizzle_qk=False, with_rope=True):
    mesh_config = MeshConfig(tuple(mesh_device.shape), tp=mesh_device.shape[1])
    config = AttentionConfig(
        hidden_size=dims["hidden_size"],
        num_heads=dims["num_heads"],
        num_kv_heads=dims["num_kv_heads"],
        head_dim=dims["head_dim"],
        max_seq_len=max(seq_len, ttnn.TILE_SIZE),
        rms_norm_eps=hf["rms_norm_eps"],
    )
    trans_mats = {"prefill": build_transformation_mat(mesh_device)} if with_rope else None
    attn = Attention(
        mesh_device,
        config,
        _state_dict(w, unswizzle_qk=unswizzle_qk, head_dim=dims["head_dim"]),
        mesh_config=mesh_config,
        program_config=ProgramConfig(),
        layer_idx=0,
        transformation_mats=trans_mats,
        weight_dtype=weight_dtype,
    )
    return attn, mesh_config, config


# --------------------------------------------------------------------------------------------
# The gate: the whole block.
# --------------------------------------------------------------------------------------------
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("weight_dtype", _DTYPES, ids=lambda d: _DTYPE_IDS[d])
@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=lambda s: f"s{s}")
def test_attention_vs_ref(mesh_device, weight_dtype, seq_len, reset_seeds):
    """The whole attention block vs the fp32 reference. PCC >= 0.999, <= 8x the floor."""
    hf = llama_config_dims()
    dims = _dims(hf)
    x = torch.randn(1, 1, seq_len, dims["hidden_size"])
    w = _random_weights(dims)
    cos, sin = _hf_cos_sin(hf, seq_len)

    stages = _torch_stages(x, w, cos, sin, dims)
    ref = stages["out"]

    # Floor: quantise the input and the four weights; the cos/sin the device stores are bf16 too.
    cos_q = quantize_like_device(cos[None, None], ttnn.bfloat16)[0, 0]
    sin_q = quantize_like_device(sin[None, None], ttnn.bfloat16)[0, 0]
    floor_out = _torch_stages(
        quantize_like_device(x, ACTIVATION_DTYPE), _quantize_weights(w, weight_dtype), cos_q, sin_q, dims
    )["out"]
    _, floor = comp_pcc(ref, floor_out, 0.0)
    floor = float(floor)

    attn, _, config = _build_attention(mesh_device, hf, dims, w, weight_dtype=weight_dtype, seq_len=seq_len)
    rope_mats = build_prefill_rope(mesh_device, hf, seq_len)
    out = _from_device(attn(_to_device(x, mesh_device), rope_mats))

    passing, pcc = comp_pcc(ref, out, BLOCK_PCC_THRESHOLD)
    ratio = err_ratio(float(pcc), floor)

    # --- attribute the gap before judging it (recipe §2.3) --------------------------------------
    # Run the fused kernel alone on this block's own post-RoPE Q/K/V and measure how far past its
    # OWN floor it sits. `1 - PCC` is a variance-like quantity, so independent error sources add
    # to first order and the excess can be subtracted; the logged `predicted` figure below is the
    # check on that assumption, not an assertion dressed up as one.
    src = _meta_head_index(dims["head_dim"])
    q_meta, k_meta = stages["q_rot"][..., src], stages["k_rot"][..., src]
    sdpa_ref = _sdpa_reference(q_meta, k_meta, stages["v"], dims)
    _, sdpa_floor = comp_pcc(
        sdpa_ref,
        _sdpa_reference(
            quantize_like_device(q_meta, ttnn.bfloat16),
            quantize_like_device(k_meta, ttnn.bfloat16),
            quantize_like_device(stages["v"], ttnn.bfloat16),
            dims,
        ),
        0.0,
    )
    tt_sdpa = run_sdpa(
        _to_device(q_meta, mesh_device),
        _to_device(k_meta, mesh_device),
        _to_device(stages["v"], mesh_device),
        config,
        ProgramConfig(),
        mesh_device,
        seq_len,
    )
    _, sdpa_pcc = comp_pcc(sdpa_ref, _from_device(tt_sdpa), 0.0)
    tt_sdpa.deallocate(True)

    sdpa_excess = (1.0 - float(sdpa_pcc)) - (1.0 - float(sdpa_floor))
    floor_err = 1.0 - floor
    residual_ratio = ((1.0 - float(pcc)) - sdpa_excess) / floor_err
    predicted = 1.0 - (floor_err + sdpa_excess)

    logger.info(
        f"[G-ATTN] block {_DTYPE_IDS[weight_dtype]} seq={seq_len}: PCC={float(pcc):.7f} "
        f"floor={floor:.7f} ratio={ratio:.2f}x (threshold {BLOCK_PCC_THRESHOLD}, raw budget "
        f"{MAX_BLOCK_ERR_RATIO}x applies={RAW_BLOCK_BUDGET_APPLIES[weight_dtype]})"
    )
    logger.info(
        f"[G-ATTN] block {_DTYPE_IDS[weight_dtype]} seq={seq_len} attribution: fused SDPA alone "
        f"PCC={float(sdpa_pcc):.7f} vs its own floor {float(sdpa_floor):.7f} "
        f"(excess {sdpa_excess:.3e} = {err_ratio(float(sdpa_pcc), float(sdpa_floor)):.1f}x); "
        f"floor error {floor_err:.3e}; floor+SDPA predicts block PCC {predicted:.7f} vs measured "
        f"{float(pcc):.7f}; SDPA-attributed residual ratio {residual_ratio:.2f}x"
    )

    assert passing, f"below threshold {BLOCK_PCC_THRESHOLD}: {pcc}"
    assert residual_ratio <= MAX_BLOCK_ERR_RATIO, (
        f"after subtracting the fused SDPA kernel's own excess the block is still "
        f"{residual_ratio:.2f}x off its floor — that residual is this package's code, so "
        f"investigate it before recording a PASS"
    )
    if RAW_BLOCK_BUDGET_APPLIES[weight_dtype]:
        assert (
            ratio <= MAX_BLOCK_ERR_RATIO
        ), f"block is {ratio:.2f}x off its floor — attribute it before recording a PASS"


# --------------------------------------------------------------------------------------------
# Per-stage budgets, stage-ISOLATED: each stage is fed the reference's own (quantised) input.
# --------------------------------------------------------------------------------------------
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("weight_dtype", _DTYPES, ids=lambda d: _DTYPE_IDS[d])
@pytest.mark.parametrize("seq_len", [512], ids=lambda s: f"s{s}")
def test_attention_stage_budgets(mesh_device, weight_dtype, seq_len, reset_seeds):
    """Every stage **this package implements** must sit within 3x its own floor (recipe §2.3).

    Feeding each stage the reference's input rather than the previous stage's device output is what
    makes these stage numbers rather than accumulation numbers — which is the whole point of having
    separate budgets: a block-level budget that lumps the fused kernel's slack in with the
    hand-written stages can absorb a real regression in the latter without moving.
    """
    hf = llama_config_dims()
    dims = _dims(hf)
    head_dim, nq, nkv = dims["head_dim"], dims["num_heads"], dims["num_kv_heads"]
    src = _meta_head_index(head_dim)

    x = torch.randn(1, 1, seq_len, dims["hidden_size"])
    w = _random_weights(dims)
    cos, sin = _hf_cos_sin(hf, seq_len)
    ref = _torch_stages(x, w, cos, sin, dims)
    w_q = _quantize_weights(w, weight_dtype)
    x_q = quantize_like_device(x, ACTIVATION_DTYPE)
    cos_q = quantize_like_device(cos[None, None], ttnn.bfloat16)[0, 0]
    sin_q = quantize_like_device(sin[None, None], ttnn.bfloat16)[0, 0]
    # Each stage's floor is computed **locally**, from that stage's own quantised inputs — NOT from
    # the end of a quantised chain. A chained floor carries the upstream stages' rounding, which
    # inflates `1 - floor` and lets a stage report a ratio below 1.0 (measured: a chained floor put
    # `concat_heads`, a pure layout op, at 0.01x). A ratio below 1.0 is a broken floor, not a
    # kernel better than arithmetic allows.

    mesh_config = MeshConfig(tuple(mesh_device.shape), tp=mesh_device.shape[1])
    config = AttentionConfig(
        hidden_size=dims["hidden_size"],
        num_heads=nq,
        num_kv_heads=nkv,
        head_dim=head_dim,
        max_seq_len=seq_len,
        rms_norm_eps=hf["rms_norm_eps"],
    )
    program_config = ProgramConfig()
    ckc = program_config.get_compute_kernel_config(mesh_device)
    weights = load_attention_weights(
        mesh_device, config, _state_dict(w), mesh_config=mesh_config, weight_dtype=weight_dtype
    )

    results = {}

    def _record(stage, ref_t, floor_t, dev_t, *, assert_budget=True):
        _, floor = comp_pcc(ref_t, floor_t, 0.0)
        _, pcc = comp_pcc(ref_t, dev_t, 0.0)
        ratio = err_ratio(float(pcc), float(floor))
        results[stage] = (float(pcc), float(floor), ratio)
        logger.info(
            f"[G-ATTN] stage {stage:<12} {_DTYPE_IDS[weight_dtype]} seq={seq_len}: "
            f"PCC={float(pcc):.7f} floor={float(floor):.7f} ratio={ratio:.2f}x"
            f"{'' if assert_budget else '  (fused kernel — excluded from the 3x budget, §2.3)'}"
        )
        if assert_budget:
            assert ratio <= MAX_STAGE_ERR_RATIO, f"stage {stage} is {ratio:.2f}x off its floor (budget 3x)"

    # --- stage 1: QKV projection + GQA head split. Device Q/K are in Meta head space. ---
    tt_x = _to_device(x, mesh_device)
    q, k, v = apply_qkv_projection(tt_x, weights, ckc)
    tt_q, tt_k, tt_v = split_qkv_heads_prefill(q, k, v, nq, nkv)
    proj_floor = _torch_stages(x_q, w_q, cos_q, sin_q, dims)  # projections are the FIRST stage, so this IS local
    _record("qkv_proj_q", ref["q"][..., src], proj_floor["q"][..., src], _from_device(tt_q))
    _record("qkv_proj_k", ref["k"][..., src], proj_floor["k"][..., src], _from_device(tt_k))
    _record("qkv_proj_v", ref["v"], proj_floor["v"], _from_device(tt_v))
    for t in (tt_q, tt_k, tt_v):
        t.deallocate(True)

    # --- stage 2: RoPE, fed the reference's own pre-RoPE Q/K (in Meta space) ---
    rope_mats = build_prefill_rope(mesh_device, hf, seq_len)
    trans_mat = build_transformation_mat(mesh_device)
    for name, pre, post in (("rope_q", ref["q"], ref["q_rot"]), ("rope_k", ref["k"], ref["k_rot"])):
        dev_in = _to_device(pre[..., src], mesh_device)
        dev_out = apply_rope(dev_in, rope_mats, trans_mat)
        # Local floor: the device stores this stage's input and both tables in bf16; a permutation
        # of the head dim does not change element-wise bf16 rounding, so quantising in HF space and
        # permuting afterwards is identical to quantising the Meta tensor.
        rope_floor = _rope_hf(quantize_like_device(pre, ACTIVATION_DTYPE), cos_q, sin_q)
        _record(name, post[..., src], rope_floor[..., src], _from_device(dev_out))
        dev_in.deallocate(True)
        dev_out.deallocate(True)

    # --- stage 3: the FUSED SDPA kernel, fed the reference's post-RoPE Q/K/V. Recorded, not gated
    # at 3x: recipe §2.3 measured this one op at 71x its modelled floor while every hand-written
    # stage sat at 1.0-1.5x, and the floor model does not describe a fused kernel's interior. ---
    q_meta, k_meta = ref["q_rot"][..., src], ref["k_rot"][..., src]
    sdpa_ref = _sdpa_reference(q_meta, k_meta, ref["v"], dims)
    sdpa_floor = _sdpa_reference(
        quantize_like_device(q_meta, ttnn.bfloat16),
        quantize_like_device(k_meta, ttnn.bfloat16),
        quantize_like_device(ref["v"], ttnn.bfloat16),
        dims,
    )
    tt_sdpa = run_sdpa(
        _to_device(q_meta, mesh_device),
        _to_device(k_meta, mesh_device),
        _to_device(ref["v"], mesh_device),
        config,
        program_config,
        mesh_device,
        seq_len,
    )
    _record("sdpa_fused", sdpa_ref, sdpa_floor, _from_device(tt_sdpa), assert_budget=False)
    tt_sdpa.deallocate(True)

    # --- stage 4: concat heads + o_proj, fed the reference's own SDPA output ---
    dev_ctx_heads = _to_device(ref["ctx_heads"], mesh_device)
    dev_ctx = concat_heads(dev_ctx_heads)
    # Local floor for a pure layout op: the same reshape of the bf16-quantised input. The device
    # should land essentially ON it, since there is no arithmetic to round.
    concat_floor = (
        quantize_like_device(ref["ctx_heads"], ACTIVATION_DTYPE).transpose(1, 2).reshape(1, 1, seq_len, nq * head_dim)
    )
    _record("concat_heads", ref["ctx"], concat_floor, _from_device(dev_ctx))
    dev_out = apply_output_projection(dev_ctx, weights, ACTIVATION_DTYPE, ckc)
    o_ref = ref["ctx"] @ w["o_proj"].float().transpose(-1, -2)
    o_floor = quantize_like_device(ref["ctx"], ACTIVATION_DTYPE) @ w_q["o_proj"].transpose(-1, -2)
    _record("o_proj", o_ref, o_floor, _from_device(dev_out))
    dev_ctx.deallocate(True)
    dev_out.deallocate(True)

    hand_written = [s for s in results if s != "sdpa_fused"]
    logger.info(
        f"[G-ATTN] stage summary {_DTYPE_IDS[weight_dtype]}: "
        f"hand-written stages {min(results[s][2] for s in hand_written):.2f}x-"
        f"{max(results[s][2] for s in hand_written):.2f}x of their floors; "
        f"fused SDPA {results['sdpa_fused'][2]:.2f}x of its modelled floor"
    )


# --------------------------------------------------------------------------------------------
# The permanent standalone SDPA probe (recipe §2.3): the kernel's slack, named and tracked.
# --------------------------------------------------------------------------------------------
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=lambda s: f"s{s}")
def test_sdpa_standalone_probe(mesh_device, seq_len, reset_seeds):
    """`ttnn.transformer.scaled_dot_product_attention` alone, bf16 Q/K/V, GQA 32/8, head_dim 128.

    Kept permanently so the fused kernel's slack stays a **named, tracked** term rather than budget
    the block silently grants itself. Recipe §2.3 measured 0.9999204 against a modelled floor of
    0.9999989 — a 71x gap that accounted for the entire block-level gap of `G-ATTN`, and that
    sweeping the chunk sizes over {32,128,256} moves by under 4%.
    """
    hf = llama_config_dims()
    dims = _dims(hf)
    nq, nkv, head_dim = dims["num_heads"], dims["num_kv_heads"], dims["head_dim"]

    q = torch.randn(1, nq, seq_len, head_dim)
    k = torch.randn(1, nkv, seq_len, head_dim)
    v = torch.randn(1, nkv, seq_len, head_dim)

    ref = _sdpa_reference(q, k, v, dims)
    floor_out = _sdpa_reference(
        quantize_like_device(q, ttnn.bfloat16),
        quantize_like_device(k, ttnn.bfloat16),
        quantize_like_device(v, ttnn.bfloat16),
        dims,
    )
    _, floor = comp_pcc(ref, floor_out, 0.0)

    config = AttentionConfig(
        hidden_size=dims["hidden_size"],
        num_heads=nq,
        num_kv_heads=nkv,
        head_dim=head_dim,
        max_seq_len=seq_len,
        rms_norm_eps=hf["rms_norm_eps"],
    )
    tt_out = run_sdpa(
        _to_device(q, mesh_device),
        _to_device(k, mesh_device),
        _to_device(v, mesh_device),
        config,
        ProgramConfig(),
        mesh_device,
        seq_len,
    )
    out = _from_device(tt_out)

    passing, pcc = comp_pcc(ref, out, SDPA_PROBE_PCC_THRESHOLD)
    ratio = err_ratio(float(pcc), float(floor))
    logger.info(
        f"[G-ATTN] SDPA probe seq={seq_len}: PCC={float(pcc):.7f} modelled_floor={float(floor):.7f} "
        f"ratio={ratio:.2f}x (the floor model does not describe a fused kernel's interior, §2.3)"
    )
    assert passing, f"standalone SDPA below {SDPA_PROBE_PCC_THRESHOLD}: {pcc}"


# --------------------------------------------------------------------------------------------
# Negative control + the rotation invariant.
# --------------------------------------------------------------------------------------------
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("weight_dtype", _DTYPES, ids=lambda d: _DTYPE_IDS[d])
def test_attention_unswizzled_qk_negative_control(mesh_device, weight_dtype, reset_seeds):
    """**The negative control.** Q/K weights that reach the device without the Meta swizzle.

    The recipe measured **0.9475** for this — a completely broken RoPE convention that still scores
    0.95, which is the entire argument of §2.1 for why a 0.99 threshold ratifies degraded modules.
    """
    hf = llama_config_dims()
    dims = _dims(hf)
    seq_len = 512
    x = torch.randn(1, 1, seq_len, dims["hidden_size"])
    w = _random_weights(dims)
    cos, sin = _hf_cos_sin(hf, seq_len)
    ref = _torch_stages(x, w, cos, sin, dims)["out"]

    attn, _, _ = _build_attention(
        mesh_device, hf, dims, w, weight_dtype=weight_dtype, seq_len=seq_len, unswizzle_qk=True
    )
    out = _from_device(attn(_to_device(x, mesh_device), build_prefill_rope(mesh_device, hf, seq_len)))
    _, pcc = comp_pcc(ref, out, 0.0)

    logger.info(f"[G-ATTN] control ({_DTYPE_IDS[weight_dtype]}): Q/K without the Meta permute -> PCC={float(pcc):.5f}")
    assert float(pcc) < BLOCK_PCC_THRESHOLD, (
        f"the negative control did not collapse ({pcc}) — the gate is not sensitive to the "
        f"HF->Meta RoPE weight swizzle"
    )


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_attention_rotates_only_q_and_k(mesh_device, reset_seeds):
    """**Invariant:** only Q and K are rotated. A V-rotating reference must fit strictly worse."""
    hf = llama_config_dims()
    dims = _dims(hf)
    seq_len = 512
    x = torch.randn(1, 1, seq_len, dims["hidden_size"])
    w = _random_weights(dims)
    cos, sin = _hf_cos_sin(hf, seq_len)

    ref_qk = _torch_stages(x, w, cos, sin, dims)["out"]
    ref_qkv = _torch_stages(x, w, cos, sin, dims, rotate_v=True)["out"]

    attn, _, _ = _build_attention(mesh_device, hf, dims, w, weight_dtype=ttnn.bfloat16, seq_len=seq_len)
    out = _from_device(attn(_to_device(x, mesh_device), build_prefill_rope(mesh_device, hf, seq_len)))

    _, pcc_qk = comp_pcc(ref_qk, out, 0.0)
    _, pcc_qkv = comp_pcc(ref_qkv, out, 0.0)
    logger.info(
        f"[G-ATTN] rotation invariant: vs Q,K-rotated ref PCC={float(pcc_qk):.7f}; "
        f"vs Q,K,V-rotated ref PCC={float(pcc_qkv):.5f}"
    )
    assert float(pcc_qk) > float(pcc_qkv), "a V-rotating reference fits better — V is being rotated"
    assert float(pcc_qkv) < BLOCK_PCC_THRESHOLD, "a V-rotating reference also clears the gate — the probe is blind"


# --------------------------------------------------------------------------------------------
# Refusals.
# --------------------------------------------------------------------------------------------
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_attention_refuses_to_build_weightless(mesh_device, expect_error):
    """No `state_dict` and no cache path must fail loud, not build four `None` projections."""
    hf = llama_config_dims()
    dims = _dims(hf)
    with expect_error(ValueError, "tensor_cache_path"):
        load_attention_weights(
            mesh_device,
            AttentionConfig(
                hidden_size=dims["hidden_size"],
                num_heads=dims["num_heads"],
                num_kv_heads=dims["num_kv_heads"],
                head_dim=dims["head_dim"],
                max_seq_len=512,
            ),
            {},
            mesh_config=MeshConfig(tuple(mesh_device.shape), tp=mesh_device.shape[1]),
        )


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_attention_refuses_a_derived_sdpa_grid(mesh_device, expect_error):
    """A derived SDPA program grid must fail at **construction**, not at SP > 1 two phases later.

    This is the landmine in its purest form: on this (12,10) box the CCL offset is `grid.x - 1` = 11
    and the ring op asserts `ccl_core_grid_offset.x >= sdpa_grid.x`
    (`ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp:421`).
    A grid derived from the device would be 12 and would pass every single-card gate.
    """
    grid = mesh_device.compute_with_storage_grid_size()
    logger.info(f"[G-ATTN] compute grid = ({grid.x}, {grid.y}); CCL offset x = {grid.x - 1}; pinned SDPA grid x = 8")
    assert grid.x > 8, f"this assertion only discriminates on a grid wider than 8; got {grid.x}"
    with expect_error(ValueError, "ccl_core_grid_offset.x >= sdpa_grid.x"):
        ProgramConfig(sdpa_grid_x=grid.x).validate_grid(mesh_device)


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_attention_refuses_cached_len(mesh_device, expect_error, reset_seeds):
    """A cache-backed chunk on the dense path must refuse, not silently mis-mask (P8 owns it)."""
    hf = llama_config_dims()
    dims = _dims(hf)
    seq_len = 128
    w = _random_weights(dims)
    attn, _, _ = _build_attention(mesh_device, hf, dims, w, weight_dtype=ttnn.bfloat16, seq_len=seq_len)
    x = _to_device(torch.randn(1, 1, seq_len, dims["hidden_size"]), mesh_device)
    with expect_error(NotImplementedError, "chunk-position-aware SDPA"):
        attn(x, build_prefill_rope(mesh_device, hf, seq_len), cached_len=ttnn.TILE_SIZE)
