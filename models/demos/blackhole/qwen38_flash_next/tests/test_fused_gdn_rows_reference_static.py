# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The GDN rows chain reference (``ttnn/fused/gdn_rows_reference.py``) without a device: its geometry against
``ttnn/gdn.py``'s constants, the exact parts (row shifts, history, GQA expand, the prim layouts, the head fold) against
direct index arithmetic, the rounding policies' candidate rules, and the two program references' shapes and
composition on small synthetic rows; then the recurrence references of the verify-rows scan: the serial scan at one
row bitwise the fused gdn_step's ``reference_step`` (the tt/gdn.py oracle), the masked commit in both forms, the chunk
prims' arithmetic against the serial form (a bound justified from the arithmetic, and an exact check on dyadic inputs),
the verify tile's row mask and its selector-driven history against ``ttnn/gdn.py``'s selection tiles."""

from __future__ import annotations

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_rows_reference as ref

T = 64  # two chunks: the FIR boundary between tile rows and the first chunk with a history
R = 5  # the MTP verify's real rows (k = 4 drafts + the base token) of one 32-row tile


def _rows(seed: int = 1, rows: int = T):
    g = torch.Generator().manual_seed(seed)
    projected = torch.zeros(rows, ref.PROJECTION_WIDTH)
    projected[:, : ref.A_COLUMN] = torch.randn(rows, ref.A_COLUMN, generator=g) * 0.6
    projected[:, ref.A_COLUMN : ref.A_COLUMN + ref.HEADS] = torch.randn(rows, ref.HEADS, generator=g) * 1.5 - 1.0
    projected[:, ref.B_COLUMN : ref.B_COLUMN + ref.HEADS] = torch.randn(rows, ref.HEADS, generator=g) * 1.5
    history = torch.zeros(ref.TILE, ref.QKV_WIDTH)
    history[: ref.HISTORY_ROWS] = torch.randn(ref.HISTORY_ROWS, ref.QKV_WIDTH, generator=g) * 0.6
    return {
        "projected": projected.to(torch.bfloat16),
        "history": history.to(torch.bfloat16),
        "conv_weights": (torch.randn(ref.CONV_KERNEL, ref.QKV_WIDTH, generator=g) * 0.5).to(torch.bfloat16),
        "dt_bias": torch.randn(ref.HEADS, generator=g) * 0.5,
        "neg_exp_A": -torch.exp(torch.rand(ref.HEADS, generator=g) * 3.0),
        "norm": (1.0 + torch.randn(ref.HEAD_DIM, generator=g) * 0.1).to(torch.bfloat16),
        "o": torch.randn(ref.HEADS, rows, ref.HEAD_DIM, generator=g) * 0.7,
        "state": torch.randn(ref.HEADS, ref.HEAD_DIM, ref.HEAD_DIM, generator=g) * 0.4,
    }


def _tile_pre(seed: int, rounding: ref.Rounding = ref.DEFAULT):
    """One verify tile (T = 32, every row real) through ``pre_reference`` under ``rounding``, with its state."""

    rows = _rows(seed, rows=ref.TILE)
    pre = ref.pre_reference(
        rows["projected"], rows["history"], rows["conv_weights"], rows["dt_bias"], rows["neg_exp_A"], rounding
    )
    return rows, pre


def _serial(pre, s0, rounding: ref.Rounding, rows: int | None = None, **kwargs):
    n = pre["beta"].shape[0] if rows is None else rows
    v = pre["v"][:n].reshape(n, ref.HEADS, ref.HEAD_DIM)
    return ref.serial_scan_reference(
        pre["q"][:n], pre["k"][:n], v, pre["beta"][:n], pre["g"][:n], s0, rounding=rounding, **kwargs
    )


def _chunk(pre, s0, rounding: ref.Rounding):
    prep = ref.chunk_prep_reference(pre["q_c"], pre["k_c"], pre["v"], pre["g_c"], pre["beta_c"], rounding)
    return ref.chunk_scan_reference(prep, s0, rounding)


def test_geometry_matches_gdn_py():
    assert (ref.HEADS, ref.QK_HEADS, ref.HEAD_DIM) == (
        gdn_module.VALUE_HEADS_PER_DEVICE,
        gdn_module.QK_HEADS_PER_DEVICE,
        gdn_module.HEAD_DIM,
    )
    assert (ref.QK_WIDTH, ref.VALUE_WIDTH, ref.QKV_WIDTH) == (
        gdn_module.QK_WIDTH_PER_DEVICE,
        gdn_module.VALUE_WIDTH_PER_DEVICE,
        gdn_module.QKV_WIDTH_PER_DEVICE,
    )
    assert (ref.A_COLUMN, ref.B_COLUMN, ref.PROJECTION_WIDTH) == (
        gdn_module.A_COLUMN,
        gdn_module.B_COLUMN,
        gdn_module.PROJECTION_WIDTH_PER_DEVICE,
    )
    assert (ref.CONV_KERNEL, ref.HISTORY_ROWS) == (gdn_module.CONV_KERNEL_SIZE, gdn_module.CONV_HISTORY_ROWS)
    assert ref.QK_L2_NORM_EPS == gdn_module.QK_L2_NORM_EPS and ref.RMS_NORM_EPS == gdn_module.RMS_NORM_EPS
    assert ref.QK_SCALE == gdn_module.HEAD_DIM**-0.5 and ref.TILE == gdn_module.CHUNK_SIZE


def test_fir_taps_are_the_row_shifts_of_the_window():
    rows = _rows()
    qkv = ref.split_projection(rows["projected"])["qkv"]
    taps = ref.fir_taps(qkv, rows["history"])
    assert len(taps) == 4 and all(tap.shape == (T, ref.QKV_WIDTH) for tap in taps)
    for t in range(3):
        # tap t row r = qkv row r - (3 - t), or history row t + r before the first new row
        for r in (0, 1, 2, 3, 31, 32, 33, T - 1):
            source = qkv[r - (3 - t)] if r >= 3 - t else rows["history"][t + r]
            assert torch.equal(taps[t][r], source), (t, r)
    assert taps[3] is qkv
    nxt = ref.history_next(qkv)
    assert torch.equal(nxt[:3], qkv[-3:]) and not nxt[3:].any() and nxt.dtype == torch.bfloat16


def test_gqa_expand_and_prim_layouts_are_index_maps():
    x = torch.arange(T * ref.QK_WIDTH, dtype=torch.float32).reshape(T, ref.QK_WIDTH).to(torch.bfloat16)
    heads = ref.gqa_expand(x)
    assert heads.shape == (T, ref.HEADS, ref.HEAD_DIM)
    for hv in range(ref.HEADS):
        assert torch.equal(heads[:, hv], x[:, (hv // 3) * ref.HEAD_DIM : (hv // 3 + 1) * ref.HEAD_DIM])
    qc = ref.to_prim_qk(heads)
    assert qc.shape == (ref.HEADS, T // 32, 32, ref.HEAD_DIM)
    for h, c, r in ((0, 0, 0), (5, 1, 31), (11, 1, 7)):
        assert torch.equal(qc[h, c, r], heads[c * 32 + r, h])
    vec = torch.arange(T * ref.HEADS, dtype=torch.float32).reshape(T, ref.HEADS)
    vc = ref.to_prim_vec(vec)
    assert vc.shape == (ref.HEADS, T // 32, 32, 1) and vc[3, 1, 5, 0] == vec[37, 3]
    head_major = torch.arange(ref.HEADS * T * ref.HEAD_DIM, dtype=torch.float32).reshape(ref.HEADS, T, ref.HEAD_DIM)
    folded = ref.fold_heads(head_major)
    assert folded.shape == (T, ref.VALUE_WIDTH)
    assert torch.equal(folded[9, 4 * 128 : 5 * 128], head_major[4, 9])


def test_rounding_candidates_differ_where_they_should():
    x = torch.tensor(
        [1.0 + 2.0**-9 + 2.0**-12], dtype=torch.float32
    )  # not representable in bf16; below the midpoint
    assert ref.bf16_rne(x).float().item() == 1.0 and ref.bf16_trunc(x).float().item() == 1.0
    # the bf16 ulp at 1.0 is 2^-7: 1 + 2^-8 is the exact midpoint, a hair above it RNE rounds up and truncation drops it
    y = torch.tensor([1.0 + 2.0**-8 + 2.0**-20], dtype=torch.float32)
    assert ref.bf16_rne(y).float().item() == 1.0 + 2.0**-7 and ref.bf16_trunc(y).float().item() == 1.0
    z = torch.tensor([1.0 + 2.0**-11 + 2.0**-20], dtype=torch.float32)
    assert ref.tf32(z).item() == 1.0 + 2.0**-10 or ref.tf32(z).item() == 1.0  # 10 mantissa bits kept
    assert ref.tf32(torch.tensor([1.0 + 2.0**-10])).item() == 1.0 + 2.0**-10
    assert ref.tf32(torch.tensor([1.0 + 2.0**-11])).item() == 1.0
    a = torch.tensor([1.0 + 2.0**-7], dtype=torch.bfloat16)
    b = torch.tensor([1.0 + 2.0**-7], dtype=torch.bfloat16)
    c = torch.tensor([2.0**-9], dtype=torch.bfloat16)
    fused = ref.mac_bf16(a, b, c, ref.Rounding(mac_fused=True))
    split = ref.mac_bf16(a, b, c, ref.Rounding(mac_fused=False))
    assert fused.dtype == split.dtype == torch.bfloat16  # both candidates are defined; the pin tool picks one
    s = ref.multiply_scalar_bf16(torch.ones(1, dtype=torch.bfloat16), ref.QK_SCALE, ref.Rounding(scalar_bf16=True))
    assert s.float().item() == torch.tensor(ref.QK_SCALE).to(torch.bfloat16).float().item()
    d = ref.Rounding()
    assert (d.pack, d.mac_fused, d.fp32_source, d.scalar_bf16, d.mul_zero_clamp) == ("rne", True, "exact", True, True)
    # the bf16 SFPU multiply never emits a negative zero (0 * x = +0); the fp32 row multiply keeps IEEE signs
    zero = torch.zeros(2, dtype=torch.bfloat16)
    neg = torch.tensor([-3.0, -0.5], dtype=torch.bfloat16)
    prod = ref.multiply_bf16(zero, neg)
    assert torch.equal(prod.float().view(torch.int32), torch.zeros(2, dtype=torch.int32))
    assert ref.multiply_bf16(zero, neg, ref.Rounding(mul_zero_clamp=False)).float().view(torch.int32)[0] != 0
    # the pinned zero-sign rule: the fp32 multiply, mac, the typecast and rms_norm return +0 for a zero; the add keeps -0
    assert ref.multiply_fp32_row(torch.tensor([-2.0]), torch.zeros(1, 1)).view(torch.int32).item() == 0
    assert (
        ref.multiply_fp32_row(torch.tensor([-2.0]), torch.zeros(1, 1), ref.Rounding(zero_sign_plus=False))
        .view(torch.int32)
        .item()
        != 0
    )
    assert ref.typecast_to_bf16(torch.tensor([-0.0])).float().view(torch.int32).item() == 0
    assert ref.add_fp32_row(torch.tensor([[-0.0]]), torch.tensor([-0.0])).view(torch.int32).item() != 0
    # mac narrows ties away from zero (a 16-bit destination), the multiply / typecast by RNE: separate rules
    a = torch.tensor([1.375], dtype=torch.bfloat16)
    b = torch.tensor([1.84375], dtype=torch.bfloat16)
    c = torch.tensor([-1.390625], dtype=torch.bfloat16)
    assert torch.addcmul(c.float(), a.float(), b.float()).item() == 1.14453125  # an exact bf16 midpoint
    assert ref.mac_bf16(a, b, c).float().view(torch.int32).item() >> 16 == 0x3F93  # away: the pinned device bits
    assert ref.mac_bf16(a, b, c, ref.Rounding(mac_pack="rne")).float().view(torch.int32).item() >> 16 == 0x3F92
    assert ref.bf16_rna(torch.tensor([-1.14453125])).float().item() == -1.1484375  # symmetric about zero
    assert ref.Rounding().mac_pack == "rna" and ref.Rounding().zero_sign_plus is True


def test_pre_and_post_reference_shapes_and_composition():
    rows = _rows()
    out = ref.pre_reference(
        rows["projected"], rows["history"], rows["conv_weights"], rows["dt_bias"], rows["neg_exp_A"]
    )
    assert out["conv"].shape == (T, ref.QKV_WIDTH) and out["conv"].dtype == torch.bfloat16
    assert out["q"].shape == out["k"].shape == (T, ref.HEADS, ref.HEAD_DIM) and out["q"].dtype == torch.bfloat16
    assert out["v"].shape == (T, ref.VALUE_WIDTH) and torch.equal(out["v"], out["conv"][:, 2 * ref.QK_WIDTH :])
    assert out["beta"].shape == out["g"].shape == (T, ref.HEADS) and out["beta"].dtype == torch.float32
    assert out["q_c"].shape == (ref.HEADS, T // 32, 32, ref.HEAD_DIM) and out["beta_c"].shape == (
        ref.HEADS,
        T // 32,
        32,
        1,
    )
    assert out["z"].shape == (T, ref.VALUE_WIDTH) and out["history_next"].shape == (32, ref.QKV_WIDTH)
    # q carries the composite's second scale: q = rne(k_like * scale) where k_like is the same chain on q's columns
    conv = out["conv"]
    q_once = ref.qk_prepare(conv[:, : ref.QK_WIDTH], composite_scale=False)
    assert torch.equal(out["q"], ref.multiply_scalar_bf16(q_once, ref.QK_SCALE))
    assert torch.equal(out["k"], ref.qk_prepare(conv[:, ref.QK_WIDTH : 2 * ref.QK_WIDTH], composite_scale=False))
    # beta in (0, 1), g <= 0 (neg_exp_A < 0, softplus > 0)
    assert bool(((out["beta"] > 0) & (out["beta"] < 1)).all()) and bool((out["g"] <= 0).all())
    # every q/k head row is unit-norm-ish times the scales (12 heads, 3 copies of each key head before the norm)
    k_norm = out["k"].float().square().sum(-1).sqrt()
    assert torch.allclose(k_norm, torch.full_like(k_norm, ref.QK_SCALE * ref.HEAD_DIM**0.5), atol=0.02)
    gated = ref.post_reference(rows["o"], out["z"], rows["norm"])
    assert gated.shape == (T, ref.VALUE_WIDTH) and gated.dtype == torch.bfloat16
    # the fold: head h of token r sits at columns 128h..: recompute one (r, h) directly
    r, h = 37, 5
    o_bf16 = rows["o"][h, r].to(torch.bfloat16).float()
    unit = o_bf16 * torch.rsqrt(o_bf16.square().mean() + ref.RMS_NORM_EPS) * rows["norm"].float()
    sig = torch.sigmoid(out["z"][r, 128 * h : 128 * (h + 1)].float()).to(torch.bfloat16).float()
    expected = (unit.to(torch.bfloat16).float() * sig).to(torch.bfloat16)
    assert torch.equal(gated[r, 128 * h : 128 * (h + 1)], expected)


# ------------------------------------------------------------------- the recurrence references (the verify-rows scan)


def test_oracle_policies_sit_beside_the_chain_default(expect_error):
    d, o, k = ref.DEFAULT, ref.ORACLE, ref.GDN_STEP
    assert (d.conv_packs, d.qk_l2, d.q_scale_point, d.beta_bf16, d.epilogue) == (
        5,
        "rms_scaled",
        "q_bf16",
        False,
        "chain",
    )
    assert (o.conv_packs, o.qk_l2, o.q_scale_point, o.beta_bf16, o.epilogue) == (
        1,
        "l2_direct",
        "q_fp32",
        True,
        "oracle",
    )
    assert k == ref.Rounding(**{**o.__dict__, "q_scale_point": "o_fp32"})
    # the chain's rules are the same under all three: the oracle path never reaches them
    assert (o.pack, o.mac_fused, o.fp32_source, o.scalar_bf16, o.mul_zero_clamp) == (
        d.pack,
        d.mac_fused,
        d.fp32_source,
        d.scalar_bf16,
        d.mul_zero_clamp,
    )
    rows = _rows(3, rows=ref.TILE)
    with expect_error(ValueError):
        ref.qk_prepare(rows["projected"][:, : ref.QK_WIDTH], composite_scale=True, rounding=ref.ORACLE)


def test_serial_scan_at_one_row_is_reference_step_bitwise():
    """(i) ``serial_scan_reference`` at R = 1 under ORACLE is ``gdn_step.reference_step`` (tt/gdn.py's arithmetic) bit
    for bit: the conv, beta, the decay, the state after the row, the read-out and the gated row."""

    gdn_step = pytest.importorskip("models.demos.blackhole.qwen38_flash_next.ttnn.fused.gdn_step")
    g = torch.Generator().manual_seed(11)
    projected = torch.zeros(1, ref.PROJECTION_WIDTH)
    projected[:, : ref.A_COLUMN] = torch.randn(1, ref.A_COLUMN, generator=g) * 0.6
    projected[:, ref.A_COLUMN : ref.A_COLUMN + ref.HEADS] = torch.randn(1, ref.HEADS, generator=g) * 1.5 - 1.0
    projected[:, ref.B_COLUMN : ref.B_COLUMN + ref.HEADS] = torch.randn(1, ref.HEADS, generator=g) * 1.5
    projected = projected.to(torch.bfloat16)
    older = [(torch.randn(1, ref.QKV_WIDTH, generator=g) * 0.6).to(torch.bfloat16) for _ in range(3)]
    taps = [(torch.randn(ref.QKV_WIDTH, generator=g) * 0.5).to(torch.bfloat16) for _ in range(4)]
    dt_bias = torch.randn(ref.HEADS, generator=g) * 0.5
    neg_exp_a = -torch.exp(torch.rand(ref.HEADS, generator=g) * 3.0)
    norm = (1.0 + torch.randn(ref.HEAD_DIM, generator=g) * 0.1).to(torch.bfloat16)
    state = torch.randn(1, ref.HEADS, ref.HEAD_DIM, ref.HEAD_DIM, generator=g) * 0.4
    new_state, gated, conv, o, beta, decay = gdn_step.reference_step(
        projected, older, taps, dt_bias, neg_exp_a, norm, state
    )

    parts = ref.split_projection(projected)
    history = torch.zeros(ref.TILE, ref.QKV_WIDTH, dtype=torch.bfloat16)
    history[: ref.HISTORY_ROWS] = torch.cat(older, dim=0)  # oldest first = window rows 0..2
    my_conv = ref.conv_silu(ref.fir_taps(parts["qkv"], history), torch.stack(taps), ref.ORACLE)
    assert torch.equal(my_conv, conv)
    q = ref.qk_prepare(my_conv[:, : ref.QK_WIDTH], composite_scale=False, rounding=ref.ORACLE)
    k = ref.qk_prepare(my_conv[:, ref.QK_WIDTH : 2 * ref.QK_WIDTH], composite_scale=False, rounding=ref.ORACLE)
    v = my_conv[:, 2 * ref.QK_WIDTH :].reshape(1, ref.HEADS, ref.HEAD_DIM)
    my_beta, my_g = ref.gates(parts["a"], parts["b"], dt_bias, neg_exp_a, ref.ORACLE)
    assert torch.equal(my_beta, beta.float()) and torch.equal(my_g.exp(), decay)
    my_o, states = ref.serial_scan_reference(q, k, v, my_beta, my_g, state[0], rounding=ref.ORACLE)
    assert states.shape == (2, ref.HEADS, ref.HEAD_DIM, ref.HEAD_DIM) and torch.equal(states[0], state[0])
    assert torch.equal(states[1], new_state[0])
    assert torch.equal(my_o.to(torch.bfloat16), o)
    my_gated = ref.post_reference(my_o.permute(1, 0, 2), parts["z"], norm, ref.ORACLE)
    assert torch.equal(my_gated, gated)
    # GDN_STEP moves only the attention scale (to the read-out, fp32): the state trajectory is the oracle's
    o_step, states_step = ref.serial_scan_reference(q, k, v, my_beta, my_g, state[0], rounding=ref.GDN_STEP)
    assert torch.equal(states_step, states)
    assert torch.allclose(o_step, my_o, rtol=1e-6, atol=1e-6) and not torch.equal(o_step, my_o)


@pytest.mark.parametrize("accepted", range(R))
def test_serial_masked_commit_is_the_prefix_state_bitwise(accepted: int):
    """(ii) the verify-rows scan's commit mode (every row stepped, rows past the prefix masked to the exact identity
    ``S * 1.0 + 0``) lands bitwise the state after the first ``accepted + 1`` rows."""

    rows, pre = _tile_pre(21, ref.GDN_STEP)
    committed = ref.masked_commit_reference(pre, rows["state"], accepted, "serial", ref.GDN_STEP)
    _, prefix = _serial(pre, rows["state"], ref.GDN_STEP, rows=accepted + 1)
    assert torch.equal(committed, prefix[accepted + 1])
    # a full acceptance of the whole tile is the pass's final state
    _, whole = _serial(pre, rows["state"], ref.GDN_STEP)
    assert torch.equal(
        ref.masked_commit_reference(pre, rows["state"], ref.TILE - 1, "serial", ref.GDN_STEP), whole[ref.TILE]
    )


@pytest.mark.parametrize("accepted", range(R))
def test_chunk_masked_commit_equals_the_truncated_chunk_form_exactly(accepted: int):
    """(iii) today's commit (beta and g of rows > a zeroed, the prims over the whole tile) against the prims over the
    truncated tile (rows > a zero in q / k / v / beta / g too, ``mask_rows``).  The tolerance is ZERO: a masked row
    contributes ``v_beta = k_beta = 0`` to every product it enters (``0 * finite`` is an exact zero, ``x + 0 = x``), its
    T_inv row is the identity row (the Horner adds ``I`` to ``N @ out`` whose row is zero) so its ``v_new`` is zero, and
    the decay sums see the same real rows in both forms; the only difference between the two inputs is the real q / k /
    v of the masked rows, which reach the state only through those zero factors (and the discarded ``o`` rows)."""

    rows, pre = _tile_pre(31)
    committed = ref.masked_commit_reference(pre, rows["state"], accepted, "chunk")
    _, truncated = _chunk(ref.mask_rows(pre, accepted + 1), rows["state"], ref.DEFAULT)
    assert torch.equal(committed, truncated)


def test_chunk_passthrough_is_exact_in_fp32_and_tf32_class_through_the_source_path():
    """The all-masked chunk (no committed row; the reference's study case) returns the state itself under the exact
    source rule (``dl = exp(0) * exp(0) = 1``, ``s_upd = 0``); under the ``tf32`` rule the only surviving error is the
    19-bit truncation of S in ``S * dl``: below ``2^-10 x max|S|`` (one ulp of the 10-bit mantissa at the leading
    exponent), typically ``2^-11`` = the design's 4.9e-4 passthrough figure (``gdn.py`` ``commit_rows``)."""

    rows, pre = _tile_pre(41)
    s0 = rows["state"]
    assert torch.equal(ref.masked_commit_reference(pre, s0, -1, "chunk"), s0)
    tf32 = ref.Rounding(fp32_source="tf32")
    error = (ref.masked_commit_reference(pre, s0, -1, "chunk", tf32) - s0).abs().max().item()
    assert 0 < error <= 2.0**-10 * s0.abs().max().item()
    # and the same identity for the serial form's masked rows, exact under both rules (no source path touches S there)
    assert torch.equal(ref.masked_commit_reference(pre, s0, -1, "serial", tf32), s0)


def test_chunk_form_agrees_with_the_serial_form_within_the_reassociation_bound():
    """(iv) the prims' WY / chunk arithmetic against the serial recurrence on one random tile.  Exact fp32 operands:
    the two forms compute the same real numbers with different association (the serial form's 32 sequential fp32
    updates against the chunk form's 32-term contractions, the 15-term Horner inverse and one ``S * dl + k_dec_t @
    v_new``), so they agree to a few fp32 roundings of the largest partial sums: the bound is ``1e-4 x max|S| + 1e-5``
    (the device chunk kernel's own bound against five 1-row steps in the rows-path tests) and the same for ``o``.  Under
    the ``tf32`` source rule every product's operands lose 13 mantissa bits (``2^-11`` relative each) and the
    contractions sum 32..128 such terms: the state stays within ``2e-2 x max|S|`` (the kernel's measured 2.1e-3 state
    error sits inside), reported not pinned."""

    rows, pre = _tile_pre(51)
    s0 = rows["state"]
    o_chunk, s_chunk = _chunk(pre, s0, ref.DEFAULT)
    o_serial, states = _serial(pre, s0, ref.DEFAULT)
    s_scale, o_scale = states[ref.TILE].abs().max().item(), o_serial.abs().max().item()
    s_err = (s_chunk - states[ref.TILE]).abs().max().item()
    o_err = (o_chunk - o_serial.transpose(0, 1)).abs().max().item()
    print(
        f"chunk vs serial (exact fp32): state max abs {s_err:.3e} (scale {s_scale:.3g}), o max abs {o_err:.3e} (scale {o_scale:.3g})"
    )
    assert s_err <= 1e-4 * s_scale + 1e-5 and o_err <= 1e-4 * o_scale + 1e-5
    tf32 = ref.Rounding(fp32_source="tf32")
    _, s_tf32 = _chunk(pre, s0, tf32)
    tf32_err = (s_tf32 - states[ref.TILE]).abs().max().item()
    print(f"chunk (tf32 operands) vs serial (exact): state max abs {tf32_err:.3e}")
    assert 0 < tf32_err <= 2e-2 * s_scale


def test_chunk_and_serial_forms_are_bitwise_on_dyadic_inputs():
    """(iv, exact) one-hot q / k rows (a position shared by four rows of the tile, so the WY inverse has off-diagonal
    terms), small-integer v and state, beta = 1/2, g = 0 (unit decays): every product and sum of both forms is a
    dyadic rational far below 2^24, so both are exact and must agree bit for bit (state, o, and the prefix state of a
    chunk commit against the serial prefix)."""

    g = torch.Generator().manual_seed(61)
    rows = ref.TILE
    q = torch.zeros(rows, ref.HEADS, ref.HEAD_DIM)
    k = torch.zeros(rows, ref.HEADS, ref.HEAD_DIM)
    for j in range(rows):
        for h in range(ref.HEADS):
            k[j, h, ((j + h) % 8) * 16] = 1.0
            q[j, h, ((3 * j + h) % 8) * 16 + 1] = 1.0
    v = torch.randint(-2, 3, (rows, ref.HEADS, ref.HEAD_DIM), generator=g).float()
    beta = torch.full((rows, ref.HEADS), 0.5)
    decay = torch.zeros(rows, ref.HEADS)
    s0 = torch.randint(-3, 4, (ref.HEADS, ref.HEAD_DIM, ref.HEAD_DIM), generator=g).float()
    pre = {
        "q": q.to(torch.bfloat16),
        "k": k.to(torch.bfloat16),
        "v": v.reshape(rows, ref.VALUE_WIDTH).to(torch.bfloat16),
        "beta": beta,
        "g": decay,
    }
    pre.update(
        q_c=ref.to_prim_qk(pre["q"]),
        k_c=ref.to_prim_qk(pre["k"]),
        beta_c=ref.to_prim_vec(beta),
        g_c=ref.to_prim_vec(decay),
    )
    o_chunk, s_chunk = _chunk(pre, s0, ref.DEFAULT)
    o_serial, states = _serial(pre, s0, ref.DEFAULT)
    assert torch.equal(s_chunk, states[rows]) and torch.equal(o_chunk, o_serial.transpose(0, 1))
    assert not torch.equal(s_chunk, s0)  # the rows did move the state
    for accepted in (0, 2, R - 1):
        assert torch.equal(ref.masked_commit_reference(pre, s0, accepted, "chunk"), states[accepted + 1])


def test_mask_rows_zeroes_exactly_the_padding_rows(expect_error):
    """(v) rows >= R of q / k / v / beta / g are zero (bf16 +0 under the multiply's clamp; fp32 IEEE zeros for beta and
    g), rows < R are bitwise the unmasked ones, the prim layouts follow, the other keys pass through."""

    _, pre = _tile_pre(71)
    masked = ref.mask_rows(pre, R)
    for key in ("q", "k", "v", "beta", "g"):
        assert torch.equal(masked[key][:R], pre[key][:R]), key
        assert bool((masked[key][R:] == 0).all()), key
        assert masked[key].dtype == pre[key].dtype and masked[key].shape == pre[key].shape
    for key in ("q", "k", "v"):
        assert not masked[key][R:].float().view(torch.int32).any(), key  # +0 bits, never -0
    assert torch.equal(masked["q_c"], ref.to_prim_qk(masked["q"])) and torch.equal(
        masked["k_c"], ref.to_prim_qk(masked["k"])
    )
    assert torch.equal(masked["beta_c"], ref.to_prim_vec(masked["beta"])) and torch.equal(
        masked["g_c"], ref.to_prim_vec(masked["g"])
    )
    assert masked["z"] is pre["z"] and masked["conv"] is pre["conv"]
    assert not bool((pre["q"][R:] == 0).all())  # the unmasked padding rows are not zero (the FIR reads real rows)
    with expect_error(ValueError):
        ref.mask_rows(pre, 0)


@pytest.mark.parametrize("accepted", range(R))
def test_history_after_commit_matches_the_selector_stack(accepted: int):
    """(vi) the verify commit's next history against ``ttnn/gdn.py``'s selection tiles: ``history_select_stack[a]``
    reshaped to ``[32, 64]`` applied to the ``[history tile | qkv tile]`` buffer (the exact 0/1 select
    ``_advance_history_rows`` runs) picks window rows a + 1 .. a + 3 into rows 0..2 and leaves 3..31 zero."""

    rows = _rows(81, rows=ref.TILE)
    qkv = ref.split_projection(rows["projected"])["qkv"]
    history = rows["history"]
    tiles = gdn_module.rows_window_select_tiles(R)
    select = tiles["history_select_stack"][accepted].reshape(ref.TILE, 2 * ref.TILE)
    buffer = torch.cat([history, qkv], dim=0).float()  # buffer row m: history rows 0..2 at 0..2, qkv row r at 32 + r
    expected = select @ buffer
    actual = ref.history_after_commit(qkv, history, accepted)
    assert actual.dtype == torch.bfloat16 and actual.shape == (ref.TILE, ref.QKV_WIDTH)
    assert torch.equal(actual.float(), expected)
    assert not actual[ref.HISTORY_ROWS :].any()
    # the slab's history (every row committed) is the same rule at accepted = T - 1
    assert torch.equal(ref.history_after_commit(qkv, history, ref.TILE - 1), ref.history_next(qkv))
