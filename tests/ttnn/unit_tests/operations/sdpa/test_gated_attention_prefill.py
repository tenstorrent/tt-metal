# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Op-level harness for the Qwen3.6-27B Gated Attention prefill path.

Issue https://github.com/tenstorrent/tt-metal/issues/50475 names Gated Attention
prefill as the long-context bottleneck: quadratic in ISL, with the constant
doubled by head_dim=256 (2x the usual 128).

Until now the only coverage of this path was model-level
(`models/demos/blackhole/qwen36/tests/test_attention_tp.py`), which needs HF
weights and a 4-card mesh. This file isolates the op chain so it can be
iterated on a single device -- including under ttsim, where no card is touched
at all.

The chain mirrors `models/demos/blackhole/qwen36/tt/attention/tp.py`
`forward_prefill`:

    SDPA(q, k, v, is_causal=True)            -> [1, NH, S, HD]
    nlp_concat_heads(.)                      -> [1,  1, S, NH*HD]
    multiply(., sigmoid(gate_flat))          -> [1,  1, S, NH*HD]

`_concat_heads` (tp.py:311) relies on nlp_concat_heads' column order matching
gate_flat's (column h*HD+d == head h, dim d), so gating post-concat is
bit-identical to per-head gating. Several tests here lock that property down,
because it is the precondition that makes fusing the gate into the SDPA
epilogue legal.

Sizing: the `sim` tier is deliberately tiny so it completes under ttsim, which
runs at a kHz simulated clock. The `qwen36` tier is the real per-device
geometry at TP=4 and is meant for hardware.
"""

import math
import os

import pytest
import torch
from loguru import logger

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_and_get_pcc

# Qwen3.6-27B gated attention, per device at TP=4:
#   24 Q heads / 4 KV heads globally -> 6 Q heads / 1 KV head per device.
#   head_dim = 256 (models/tt_transformers/model_params/Qwen3.6-27B/config.json).
QWEN36_NH = 6
QWEN36_NKV = 1
QWEN36_HD = 256


def running_on_simulator():
    return os.environ.get("TT_METAL_SIMULATOR") is not None


def fa_rand(*shape):
    """Heavy-tailed inputs -- the same generator test_sdpa_prefill.py uses, so
    flash-attention's running-max rescaling is actually exercised."""
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


def gated_attention_reference(Q, K, V, gate_flat, scale):
    """torch reference for the whole chain. Returns [1, 1, S, NH*HD]."""
    b, nh, s, d = Q.shape
    nkv = K.shape[1]
    K_rep = K.repeat_interleave(nh // nkv, dim=1)
    V_rep = V.repeat_interleave(nh // nkv, dim=1)

    attn = torch.nn.functional.scaled_dot_product_attention(Q, K_rep, V_rep, is_causal=True, scale=scale)

    # concat heads: [b, nh, s, d] -> [b, 1, s, nh*d], column h*d+e == head h dim e
    concat = attn.permute(0, 2, 1, 3).reshape(b, 1, s, nh * d)
    return concat * torch.sigmoid(gate_flat)


def make_compute_kernel_config(math_fidelity, fp32_dest_acc_en, packer_l1_acc=False):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
    )


def run_gated_attention_prefill(
    device,
    *,
    b,
    nh,
    nkv,
    s,
    d,
    dtype,
    q_chunk_size,
    k_chunk_size,
    exp_approx_mode,
    math_fidelity,
    fp32_dest_acc_en,
    grid_size=None,
    kv_dtype=None,
    seed=1234,
):
    """Run the ttnn gated-attention prefill chain. Returns (tt_out, reference), both
    torch tensors shaped [b, 1, s, nh*d].

    kv_dtype defaults to dtype. It is separable because K and V are the tensors
    streamed O(S^2) times while Q is read once per q-chunk, so quantizing only
    K/V may buy most of the bandwidth win at a fraction of the accuracy cost.
    """
    torch.manual_seed(seed)

    scale = 1.0 / math.sqrt(d)

    Q = fa_rand(b, nh, s, d)
    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)
    # The gate comes from the 2x-wide q_proj, so it has Q's statistics and Q's
    # flat column order -- not a separate distribution.
    gate = fa_rand(b, 1, s, nh * d)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(
            grid_size if grid_size is not None else device.compute_with_storage_grid_size()
        ),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=exp_approx_mode,
    )
    compute_kernel_config = make_compute_kernel_config(math_fidelity, fp32_dest_acc_en)

    kvt = kv_dtype if kv_dtype is not None else dtype
    tt_q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_k = ttnn.from_torch(K, dtype=kvt, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_v = ttnn.from_torch(V, dtype=kvt, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    # The gate stays bf16 in the model regardless of QWEN_SDPA_BF8 -- it is not
    # part of the quadratic work.
    tt_gate = ttnn.from_torch(gate, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)

    tt_attn = ttnn.transformer.scaled_dot_product_attention(
        tt_q,
        tt_k,
        tt_v,
        is_causal=True,
        scale=scale,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    ttnn.deallocate(tt_q)
    ttnn.deallocate(tt_k)
    ttnn.deallocate(tt_v)

    tt_concat = ttnn.experimental.nlp_concat_heads(tt_attn, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(tt_attn)

    tt_out = ttnn.multiply(
        tt_concat,
        ttnn.sigmoid(tt_gate, memory_config=ttnn.L1_MEMORY_CONFIG),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(tt_concat)
    ttnn.deallocate(tt_gate)

    out = ttnn.to_torch(tt_out)[:, :, :s, : nh * d]
    ref = gated_attention_reference(Q, K, V, gate, scale)
    return out, ref


# ---------------------------------------------------------------------------
# Tier definitions
#
# sim    -- small enough to finish under ttsim (kHz simulated clock). This is
#           the loop used for kernel development without a card.
# qwen36 -- the real per-device geometry at TP=4. Hardware only.
# ---------------------------------------------------------------------------

SIM_SHAPE = dict(b=1, nh=1, nkv=1, s=128, d=QWEN36_HD, q_chunk_size=32, k_chunk_size=32)
SIM_SHAPE_GQA = dict(b=1, nh=2, nkv=1, s=128, d=QWEN36_HD, q_chunk_size=32, k_chunk_size=32)
QWEN36_SHAPE = dict(b=1, nh=QWEN36_NH, nkv=QWEN36_NKV, s=2048, d=QWEN36_HD, q_chunk_size=128, k_chunk_size=128)


BASELINE = dict(
    dtype=ttnn.bfloat16,
    exp_approx_mode=False,
    math_fidelity=ttnn.MathFidelity.HiFi2,
    fp32_dest_acc_en=True,
)
"""What models/demos/blackhole/qwen36/tt/attention/tp.py actually runs today.

tp.py:453 and :766 pass no compute_kernel_config, so the op default at
ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.cpp:51-52 applies:
MathFidelity::HiFi2 with fp32_dest_acc_en=true. Both call sites set
exp_approx_mode=False. This is the most expensive setting available on the
entire O(S^2) term, which is what makes it the first thing to attack.
"""


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [SIM_SHAPE, SIM_SHAPE_GQA], ids=["mha1", "gqa2"])
def test_gated_attention_prefill_sim(device, shape):
    """Baseline correctness at ttsim-sized shapes. No card required."""
    out, ref = run_gated_attention_prefill(device, **shape, **BASELINE)
    passed, msg, pcc = comp_and_get_pcc(ref, out, 0.99)
    logger.info(f"gated attention prefill (sim tier) PCC: {pcc} :: {msg}")
    assert passed, f"PCC below threshold: {msg}"


@pytest.mark.parametrize("shape", [QWEN36_SHAPE], ids=["qwen36_tp4_s2048"])
def test_gated_attention_prefill_qwen36(device, shape):
    """Real Qwen3.6-27B per-device geometry. Hardware tier -- far too slow for ttsim."""
    if running_on_simulator():
        pytest.skip("qwen36 tier is hardware-only; ttsim runs at a kHz clock")
    out, ref = run_gated_attention_prefill(device, **shape, **BASELINE)
    passed, msg, pcc = comp_and_get_pcc(ref, out, 0.99)
    logger.info(f"gated attention prefill (qwen36 tier) PCC: {pcc} :: {msg}")
    assert passed, f"PCC below threshold: {msg}"


# ---------------------------------------------------------------------------
# Invariants that later optimizations must preserve
# ---------------------------------------------------------------------------


def test_gate_after_concat_is_bit_identical_to_per_head_gate(device):
    """tp.py:311-314 applies the gate after nlp_concat_heads and claims that is
    bit-identical to gating each head separately. Fusing the gate into the SDPA
    epilogue (the planned kernel change) depends on that column identity, so
    lock it down here rather than trusting the comment.
    """
    torch.manual_seed(1234)
    b, nh, s, d = 1, 2, 128, QWEN36_HD

    attn = fa_rand(b, nh, s, d)
    gate = fa_rand(b, 1, s, nh * d)

    tt_attn = ttnn.from_torch(attn, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_gate = ttnn.from_torch(gate, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Path A: concat, then gate (what the model does).
    concat = ttnn.experimental.nlp_concat_heads(tt_attn, memory_config=ttnn.L1_MEMORY_CONFIG)
    sig = ttnn.sigmoid(tt_gate, memory_config=ttnn.L1_MEMORY_CONFIG)
    post = ttnn.to_torch(ttnn.multiply(concat, sig, memory_config=ttnn.DRAM_MEMORY_CONFIG))[:, :, :s, : nh * d]

    # Path B: gate per head in head-major layout, then concat.
    gate_heads = gate.reshape(b, s, nh, d).permute(0, 2, 1, 3).contiguous()
    tt_gate_heads = ttnn.from_torch(gate_heads, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    sig_heads = ttnn.sigmoid(tt_gate_heads, memory_config=ttnn.L1_MEMORY_CONFIG)
    gated_heads = ttnn.multiply(tt_attn, sig_heads, memory_config=ttnn.L1_MEMORY_CONFIG)
    pre = ttnn.to_torch(ttnn.experimental.nlp_concat_heads(gated_heads, memory_config=ttnn.DRAM_MEMORY_CONFIG))[
        :, :, :s, : nh * d
    ]

    assert torch.equal(post, pre), "gate-after-concat is NOT bit-identical to per-head gating"


@pytest.mark.parametrize("shape", [SIM_SHAPE_GQA], ids=["gqa2"])
def test_sdpa_grid_size_is_bit_identical(device, shape):
    """tp.py:451 pins the single-pass prefill SDPA to an (8,8) grid while the
    chunked path at tp.py:752 uses the full grid and its comment claims the two
    are bit-identical. Widening the single-pass grid is only safe if that holds.
    """
    full = device.compute_with_storage_grid_size()
    if full.x < 8 or full.y < 8:
        pytest.skip(f"device grid {full.x}x{full.y} is not larger than 8x8")

    out_small, _ = run_gated_attention_prefill(device, **shape, **BASELINE, grid_size=ttnn.CoreCoord(8, 8))
    out_full, _ = run_gated_attention_prefill(device, **shape, **BASELINE, grid_size=full)

    assert torch.equal(out_small, out_full), "SDPA output depends on compute grid size"


# ---------------------------------------------------------------------------
# Precision sweep -- the O(S^2) lever (issue #50475, plan phase 2a)
#
# This does not assert a winner; it records PCC/RMSE for each setting so the
# accuracy cost of each candidate is known before it is measured for speed on
# hardware. Numbers here are at sim-tier shapes; production-ISL accuracy is
# bounded separately by gated_attention_precision_model.py on CPU.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bfp8"])
@pytest.mark.parametrize(
    "math_fidelity",
    [ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.LoFi],
    ids=["hifi4", "hifi2", "lofi"],
)
@pytest.mark.parametrize("fp32_dest_acc_en", [True, False], ids=["fp32acc", "bf16acc"])
@pytest.mark.parametrize("exp_approx_mode", [False, True], ids=["exact_exp", "approx_exp"])
def test_gated_attention_prefill_precision_sweep(device, dtype, math_fidelity, fp32_dest_acc_en, exp_approx_mode):
    shape = dict(SIM_SHAPE_GQA)
    out, ref = run_gated_attention_prefill(
        device,
        **shape,
        dtype=dtype,
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_dest_acc_en,
        exp_approx_mode=exp_approx_mode,
    )
    _, _, pcc = comp_and_get_pcc(ref, out, 0.0)
    rmse = torch.sqrt(((ref - out) ** 2).mean()).item()
    logger.info(
        f"PRECISION_SWEEP dtype={dtype} fidelity={math_fidelity} "
        f"fp32_acc={fp32_dest_acc_en} exp_approx={exp_approx_mode} pcc={pcc} rmse={rmse:.6f}"
    )
    # A floor, not a target: anything below this is broken, not merely lossy.
    assert not torch.isnan(out).any(), "NaN in output"


# ---------------------------------------------------------------------------
# Decode. The issue reports gated-attention decode growing 438 -> 724 us as KV
# length goes 4k -> 128k, so the softmax exp is O(kv_len) work here too.
#
# SDPA decode is a DIFFERENT kernel from prefill
# (sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp), so the prefill
# exp_approx result does not transfer. It is measured separately.
# ---------------------------------------------------------------------------


def run_sdpa_decode(device, *, b, nh, nkv, kv_len, d, exp_approx_mode, dtype=ttnn.bfloat16, seed=7):
    torch.manual_seed(seed)
    scale = 1.0 / math.sqrt(d)
    cur_pos = kv_len - 1

    # decode Q is one token per batch row: [1, b, nh, d]
    Q = fa_rand(1, b, nh, d)
    K = fa_rand(b, nkv, kv_len, d)
    V = fa_rand(b, nkv, kv_len, d)

    tt_q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_k = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_v = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_pos = ttnn.from_torch(
        torch.full((b,), cur_pos, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    grid = device.compute_with_storage_grid_size()
    cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        exp_approx_mode=exp_approx_mode,
        q_chunk_size=0,
        k_chunk_size=0,
    )
    tt_out = ttnn.transformer.scaled_dot_product_attention_decode(
        tt_q, tt_k, tt_v, cur_pos_tensor=tt_pos, scale=scale, program_config=cfg
    )
    out = ttnn.to_torch(tt_out)

    # Reference: attend over keys [0, cur_pos] inclusive.
    q_ref = Q.permute(1, 2, 0, 3)  # [b, nh, 1, d]
    K_rep = K[:, :, : cur_pos + 1, :].repeat_interleave(nh // nkv, dim=1)
    V_rep = V[:, :, : cur_pos + 1, :].repeat_interleave(nh // nkv, dim=1)
    ref = torch.nn.functional.scaled_dot_product_attention(q_ref, K_rep, V_rep, is_causal=False, scale=scale)
    return out.reshape(b, nh, 1, d), ref


@pytest.mark.parametrize("nh", [1, QWEN36_NH], ids=["h1", "h6"])
@pytest.mark.parametrize("s", [128, 512], ids=["s128", "s512"])
def test_sigmoid_gate_fuses_into_multiply(device, s, nh):
    """The output gate is `multiply(attn, sigmoid(gate))`, spelled as two ops:
    a standalone ttnn.sigmoid writing a full [S, NH*HD] temp, then a multiply
    that reads it back.

    ttnn.multiply can apply sigmoid as an activation on operand B instead,
    which removes that temp entirely. This is the same fusion that landed for
    the GDN decode path in #50089 (exp folded into the decay multiply).

    Assert the fused form is BIT-IDENTICAL to the unfused one, so the change is
    a pure op-count reduction with no numerical argument attached to it.
    """
    torch.manual_seed(99)
    attn = fa_rand(1, 1, s, nh * QWEN36_HD)
    gate = fa_rand(1, 1, s, nh * QWEN36_HD)

    tt_attn = ttnn.from_torch(attn, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_gate = ttnn.from_torch(gate, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    unfused = ttnn.to_torch(
        ttnn.multiply(
            tt_attn,
            ttnn.sigmoid(tt_gate, memory_config=ttnn.L1_MEMORY_CONFIG),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    )
    fused = ttnn.to_torch(
        ttnn.multiply(
            tt_attn,
            tt_gate,
            input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    )

    assert torch.equal(unfused, fused), (
        "fused sigmoid-activation multiply is NOT bit-identical to sigmoid+multiply; "
        f"max abs diff {(unfused.float() - fused.float()).abs().max().item()}"
    )


@pytest.mark.parametrize("exp_approx_mode", [False, True], ids=["exact_exp", "approx_exp"])
@pytest.mark.parametrize("kv_len", [256, 1024], ids=["kv256", "kv1k"])
def test_gated_attention_decode_exp_approx(device, kv_len, exp_approx_mode):
    """Is exp_approx_mode free on the decode kernel the way it is on prefill?

    Run both and compare each against torch; the pair of numbers is the answer.
    """
    out, ref = run_sdpa_decode(
        device, b=1, nh=QWEN36_NH, nkv=QWEN36_NKV, kv_len=kv_len, d=QWEN36_HD, exp_approx_mode=exp_approx_mode
    )
    _, _, pcc = comp_and_get_pcc(ref, out, 0.0)
    rmse = torch.sqrt(((ref - out) ** 2).mean()).item()
    logger.info(f"DECODE_SWEEP kv_len={kv_len} exp_approx={exp_approx_mode} pcc={pcc} rmse={rmse:.6f}")
    assert not torch.isnan(out).any(), "NaN in decode output"
    assert pcc > 0.99, f"decode PCC collapsed: {pcc}"


# ---------------------------------------------------------------------------
# Headroom beyond bfloat8_b.
#
# Measured on hardware, bfloat8_b q/k/v takes the prefill chain 393.6 -> 250.3 us
# (-36.4%). These are the next candidates along that axis. Accuracy is settled
# here, bit-exactly; only the speed of each still needs a card.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "q_dt, kv_dt, label",
    [
        (ttnn.bfloat16, ttnn.bfloat16, "bf16/bf16 (today)"),
        (ttnn.bfloat16, ttnn.bfloat8_b, "bf16 Q / bf8 KV"),
        (ttnn.bfloat8_b, ttnn.bfloat8_b, "bf8/bf8 (measured -36%)"),
        (ttnn.bfloat16, ttnn.bfloat4_b, "bf16 Q / bf4 KV"),
        (ttnn.bfloat8_b, ttnn.bfloat4_b, "bf8 Q / bf4 KV"),
    ],
    ids=["bf16", "bf8kv", "bf8", "bf4kv", "bf8_bf4kv"],
)
def test_mixed_precision_kv_headroom(device, q_dt, kv_dt, label):
    """K and V are streamed O(S^2) times; Q is read once per q-chunk. So the
    bandwidth win is almost entirely a K/V property, and quantizing K/V harder
    than Q may be the better trade than quantizing both equally.
    """
    cfg = {**BASELINE, "exp_approx_mode": True, "dtype": q_dt}
    out, ref = run_gated_attention_prefill(device, **SIM_SHAPE_GQA, **cfg, kv_dtype=kv_dt)
    _, _, pcc = comp_and_get_pcc(ref, out, 0.0)
    rmse = torch.sqrt(((ref - out) ** 2).mean()).item()
    logger.info(f"KV_HEADROOM {label:<26} pcc={pcc:.8f} rmse={rmse:.6f}")
    assert not torch.isnan(out).any(), "NaN in output"


@pytest.mark.parametrize("chunk", [64, 128, 256], ids=["c64", "c128", "c256"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bf8"])
def test_chunk_size_under_bf8(device, chunk, dtype, expect_error):
    """tp.py's comments record that q/k_chunk=128 beat 64 and 256 -- but that was
    measured with bf16 circular buffers. bfloat8_b halves CB footprint, so the
    256 case that previously clashed with the resident L1 activation may now fit.
    Accuracy is the cheap half of that question and is answered here; whether 256
    now wins on speed needs the next hardware window.
    """
    shape = dict(SIM_SHAPE_GQA)
    shape.update(s=512, q_chunk_size=chunk, k_chunk_size=chunk)
    cfg = {**BASELINE, "exp_approx_mode": True, "dtype": dtype}

    # bf16 at chunk=256 does not fit L1 at head_dim=256: the statically allocated
    # circular buffers come to 1,676,160 B against a 1,572,864 B limit. bfloat8_b
    # halves the CB footprint and the same config builds. Assert the constraint
    # rather than letting the case fail -- it is the reason tp.py's comment says
    # 256 "clashes with the resident L1 buffer", and it is load-bearing for anyone
    # revisiting chunk sizing.
    if dtype == ttnn.bfloat16 and chunk == 256:
        with expect_error(RuntimeError, "beyond max L1 size"):
            run_gated_attention_prefill(device, **shape, **cfg)
        return

    out, ref = run_gated_attention_prefill(device, **shape, **cfg)
    _, _, pcc = comp_and_get_pcc(ref, out, 0.0)
    logger.info(f"CHUNK_SWEEP dtype={dtype} chunk={chunk} pcc={pcc:.8f}")
    assert pcc > 0.99, f"chunk={chunk} dtype={dtype} PCC collapsed: {pcc}"


# ---------------------------------------------------------------------------
# SDPA concat-heads output fusion.
#
# After bfloat8_b, the epilogue (nlp_concat_heads + the gate multiply) is
# 58.5 us of a 250.3 us chain -- 23%. concat_heads alone is 21.1 us of that,
# and it is pure data movement: SDPA already has every tile, it just writes
# them head-major. `fuse_concat_heads=True` makes the writer place them in
# concat-heads layout instead, so the separate pass disappears.
#
# Nothing about the values changes, so the bar is bit-identity, not PCC.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nh, nkv", [(1, 1), (2, 1), (QWEN36_NH, QWEN36_NKV)], ids=["mha1", "gqa2", "qwen36"])
@pytest.mark.parametrize("s", [128, 512], ids=["s128", "s512"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bf8"])
def test_sdpa_fused_concat_heads_is_bit_identical(device, s, nh, nkv, dtype):
    torch.manual_seed(4321)
    d = QWEN36_HD
    scale = 1.0 / math.sqrt(d)

    Q = fa_rand(1, nh, s, d)
    K = fa_rand(1, nkv, s, d)
    V = fa_rand(1, nkv, s, d)

    def mk(t, dt):
        return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)

    grid = device.compute_with_storage_grid_size()
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        q_chunk_size=32,
        k_chunk_size=32,
        exp_approx_mode=True,
    )
    common = dict(is_causal=True, scale=scale, program_config=pc, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Reference: head-major SDPA, then the separate concat pass.
    ref = ttnn.to_torch(
        ttnn.experimental.nlp_concat_heads(
            ttnn.transformer.scaled_dot_product_attention(mk(Q, dtype), mk(K, dtype), mk(V, dtype), **common),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    )

    fused_t = ttnn.transformer.scaled_dot_product_attention(
        mk(Q, dtype), mk(K, dtype), mk(V, dtype), fuse_concat_heads=True, **common
    )
    assert tuple(fused_t.shape)[-2:] == (s, nh * d), f"fused output shape wrong: {tuple(fused_t.shape)}"
    fused = ttnn.to_torch(fused_t)

    assert ref.shape == fused.shape, f"shape mismatch: {ref.shape} vs {fused.shape}"
    assert torch.equal(ref, fused), (
        "fused concat-heads output differs from SDPA + nlp_concat_heads; "
        f"max abs diff {(ref.float() - fused.float()).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Concat-heads fusion on the CHUNKED/PAGED path.
#
# This is the path vLLM serving actually runs (forward_prefill_paged), so it is
# where the fusion has to work to matter in production. Chunking shifts the
# output ROW via write_offset; fuse_concat_heads remaps the head to a COLUMN
# range. The two are orthogonal, which is why the same writer mapping serves
# both -- this test is what makes that claim checkable rather than asserted.
# ---------------------------------------------------------------------------


def _paged_kv(t, block_size):
    """[1, nkv, S, D] -> [S/block_size, nkv, block_size, D] with an identity page table."""
    _, nkv, s, d = t.shape
    nblocks = s // block_size
    # [1, nkv, nblocks, bs, d] -> [nblocks, nkv, bs, d]
    return (
        t.reshape(1, nkv, nblocks, block_size, d).permute(2, 1, 0, 3, 4).reshape(nblocks, nkv, block_size, d),
        nblocks,
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bf8"])
@pytest.mark.parametrize("nh, nkv", [(2, 1), (QWEN36_NH, QWEN36_NKV)], ids=["gqa2", "qwen36"])
def test_chunked_sdpa_fused_concat_heads_is_bit_identical(device, nh, nkv, dtype):
    torch.manual_seed(777)
    d = QWEN36_HD
    s_total, chunk, block_size = 256, 128, 32
    chunk_start = 128  # second chunk, so write_offset is exercised
    scale = 1.0 / math.sqrt(d)

    Q = fa_rand(1, nh, chunk, d)
    K = fa_rand(1, nkv, s_total, d)
    V = fa_rand(1, nkv, s_total, d)

    k_paged, nblocks = _paged_kv(K, block_size)
    v_paged, _ = _paged_kv(V, block_size)
    page_table = torch.arange(nblocks, dtype=torch.int32).reshape(1, nblocks)

    def mk(t, dt, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, dtype=dt, layout=layout, device=device)

    grid = device.compute_with_storage_grid_size()
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        q_chunk_size=32,
        k_chunk_size=32,
        exp_approx_mode=True,
    )

    def run(fused):
        return ttnn.transformer.chunked_scaled_dot_product_attention(
            mk(Q, dtype),
            mk(k_paged, dtype),
            mk(v_paged, dtype),
            mk(page_table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            chunk_start,
            scale=scale,
            program_config=pc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            fuse_concat_heads=fused,
        )

    ref = ttnn.to_torch(ttnn.experimental.nlp_concat_heads(run(False), memory_config=ttnn.DRAM_MEMORY_CONFIG))
    fused_t = run(True)
    assert tuple(fused_t.shape)[-2:] == (chunk, nh * d), f"fused shape wrong: {tuple(fused_t.shape)}"
    fused = ttnn.to_torch(fused_t)

    assert ref.shape == fused.shape, f"shape mismatch: {ref.shape} vs {fused.shape}"
    assert torch.equal(ref, fused), (
        "chunked fused concat-heads differs from chunked SDPA + nlp_concat_heads; "
        f"max abs diff {(ref.float() - fused.float()).abs().max().item()}"
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bf8"])
@pytest.mark.parametrize(
    "q_chunk, k_chunk", [(128, 128), (128, 256), (64, 256)], ids=["q128k128", "q128k256", "q64k256"]
)
def test_decoupled_chunk_sizes(device, q_chunk, k_chunk, dtype):
    """q_chunk and k_chunk are independent knobs and every prior experiment moved them together.

    q_chunk sets parallelism (B*NQH*Sq/q_chunk chunks, pair-distributed) and k_chunk sets the
    inner-loop blocking. Notably `can_reduce_trigger` (compute_streaming.hpp:1972) requires
    Sk_chunk_t/qkt_subblock_w > 1, which is false at k_chunk=128 and true at 256 -- so the
    early-reduce overlap path is dead code at the shipped default.

    Different k_chunk changes online-softmax blocking, so this is a PCC check, not bit-identity.
    """
    shape = dict(SIM_SHAPE_GQA)
    shape.update(s=512, q_chunk_size=q_chunk, k_chunk_size=k_chunk)
    cfg = {**BASELINE, "exp_approx_mode": True, "dtype": dtype}
    out, ref = run_gated_attention_prefill(device, **shape, **cfg)
    _, _, pcc = comp_and_get_pcc(ref, out, 0.0)
    logger.info(f"DECOUPLED_CHUNK dtype={dtype} q={q_chunk} k={k_chunk} pcc={pcc:.8f}")
    assert pcc > 0.99, f"q={q_chunk} k={k_chunk} {dtype} PCC collapsed: {pcc}"


@pytest.mark.parametrize("max_cores", [16, 24], ids=["cap16_default", "cap24_wide"])
@pytest.mark.parametrize("kv_len", [256, 1024], ids=["kv256", "kv1k"])
def test_decode_wide_kv_reduction(device, kv_len, max_cores):
    """max_cores_per_head_batch sets the flash-decode split-K width, and therefore the depth of
    the tree reduction that merges the partial (out, max, sum).

    sdpa_decode_program_factory.cpp:194 switches the cap from the full grid to the struct default
    of 16 as soon as any program config is passed. At B=1/NKV=1 that is 16 cores per head instead
    of 64 -- a 4x narrower reduction on the single-user long-context decode #50475 reports.

    Deeper trees change the float reduction ORDER, so this is a PCC check. It answers "is the
    wide path correct", not "is it faster" -- ttsim cannot answer the latter.

    24 is the measured ceiling, not 64. MAX_TREE_REDUCTION_ROUNDS=6 allows 64, but at head_dim=256
    L1 binds first: cap 32 asks for ~1.92 MB of statically allocated CBs against a 1.57 MB limit,
    and fails identically at kv 512, 1024 and 2048. The struct default of 16 is therefore partly
    protective rather than merely conservative.
    """
    torch.manual_seed(11)
    b, nh, nkv, d = 1, QWEN36_NH, QWEN36_NKV, QWEN36_HD
    scale = 1.0 / math.sqrt(d)
    cur_pos = kv_len - 1

    Q = fa_rand(1, b, nh, d)
    K = fa_rand(b, nkv, kv_len, d)
    V = fa_rand(b, nkv, kv_len, d)

    grid = device.compute_with_storage_grid_size()
    cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        exp_approx_mode=True,
        q_chunk_size=0,
        k_chunk_size=0,
        max_cores_per_head_batch=max_cores,
    )
    out = ttnn.to_torch(
        ttnn.transformer.scaled_dot_product_attention_decode(
            ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
            ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
            ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
            cur_pos_tensor=ttnn.from_torch(
                torch.full((b,), cur_pos, dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            ),
            scale=scale,
            program_config=cfg,
        )
    ).reshape(b, nh, 1, d)

    q_ref = Q.permute(1, 2, 0, 3)
    K_rep = K[:, :, : cur_pos + 1, :].repeat_interleave(nh // nkv, dim=1)
    V_rep = V[:, :, : cur_pos + 1, :].repeat_interleave(nh // nkv, dim=1)
    ref = torch.nn.functional.scaled_dot_product_attention(q_ref, K_rep, V_rep, is_causal=False, scale=scale)

    _, _, pcc = comp_and_get_pcc(ref, out, 0.0)
    logger.info(f"DECODE_WIDE kv_len={kv_len} max_cores_per_head_batch={max_cores} pcc={pcc:.8f}")
    assert not torch.isnan(out).any(), "NaN in wide-reduction decode output"
    assert pcc > 0.99, f"wide reduction PCC collapsed at max_cores={max_cores}: {pcc}"
