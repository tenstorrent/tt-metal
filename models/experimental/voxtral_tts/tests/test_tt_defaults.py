# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The shipped TTNN configuration, pinned.

THIS IS THE BLACKHOLE p150 FORK, and most of what it pins is the OPPOSITE of the parent branch.
Six N150 decisions reversed here (STATUS.md 6.39-6.45) and the guards below exist so that
re-introducing any of them fails loudly rather than quietly costing 4-7 ms/frame. The one-line
reason is with each; the measurement is in the STATUS section named.

There are no runtime toggles left -- every alternative that was measured and rejected has been
deleted rather than parked behind a flag, so what the code does is what it does. What survives is
a handful of CONSTANTS, and each one encodes a measurement that is expensive to rediscover. This
test is what makes an accidental edit to one of them fail loudly instead of quietly changing the
model. The reason for each lives in the module that owns it; the one-liners here are pointers.

Needs no device and no checkpoint -- it only imports the modules.

    pytest -svv models/experimental/voxtral_tts/tests/test_tt_defaults.py
"""

import pytest

ttnn = pytest.importorskip("ttnn")

from models.experimental.voxtral_tts.tt import ttnn_voxtral_flow as flow  # noqa: E402
from models.experimental.voxtral_tts.tt import ttnn_voxtral_gpt as gpt  # noqa: E402


def test_block1_weights_are_bfp8_except_w2():
    """Block 1 is BFP8 everywhere except w2. Decode is bandwidth-limited on weight bytes, so dtype
    is the single biggest speed lever, and every matrix worth halving has been.

    W2 IS bf16 FOR ACCURACY, NOT FOR THE HANG -- and this test exists mostly to keep those two apart,
    because they were conflated for months. BFP8 on w2 used to wedge the card; that was ttnn's conv
    `halo_gather` kernel reached via the CODEC (STATUS.md 6.12-6.13), it is fixed, and BFP8 here is
    now safe. It is simply a bad trade: measured on all 15 prompts it costs 0.24 pp of mean
    worst-sample and 0.40 pp of p90 for 2.5 ms/step -- 77% of the precision stack's whole accuracy
    cost for 15% of its speed, and 8x the worst ratio of the other two (STATUS.md 6.16).

    If you flip it back, expect mean/p90 1.17%/1.75% instead of 0.93%/1.35%, and note the codec test
    below becomes load-bearing again."""
    assert gpt.WEIGHT_DTYPE == ttnn.bfloat16        # w2 -- accuracy, see above
    assert gpt.FF_WEIGHT_DTYPE == ttnn.bfloat8_b    # FF1, FF3 -- 11.1 ms for 0.04 pp, best trade
    assert gpt.ATTN_WEIGHT_DTYPE == ttnn.bfloat8_b  # wqkv, wo -- 3.3 ms for 0.04 pp


def test_codec_output_projection_does_not_use_conv1d():
    """The codec's output projection must NOT call ttnn.conv1d: its halo_gather kernel issues an
    out-of-range NOC write on the second execution of that shape and hangs the card.

    This no longer pairs with the w2 test -- w2 is bf16 again, so nothing in the shipped config
    triggers that kernel. It stands on its own two feet: the matmul form is FASTER than the conv it
    replaced (3.45 vs 4.29 ms, STATUS.md 6.14), and it is what makes w2-in-BFP8 survivable for anyone
    who flips that line. STATUS.md 6.12-6.14."""
    import inspect

    from models.experimental.voxtral_tts.tt import ttnn_voxtral_codec as codec

    src = inspect.getsource(codec.TtVoxtralCodecDecoder._graph)
    init = inspect.getsource(codec.TtVoxtralCodecDecoder.__init__)
    assert '_conv1d(x, "out"' not in src, "the output projection is back on ttnn.conv1d -- see 6.13"
    assert "_out_taps" in init
    # And its prefix must come from ttnn.gather, not _pad_causal's six single-row slices: one such
    # slice of the 16 MiB input costs 0.381 ms, so the six of them were 2.28 of the op's 6.26 ms.
    # Bit-identical either way, so only the clock catches a revert. STATUS.md 6.14.
    assert "self._pad_causal(" not in src, "the projection is back on the slice-built pad -- see 6.14"
    assert "ttnn.gather(" in src and "_out_prefix_idx" in init


def test_block1_math_config_keeps_fp32_accumulation():
    """RMSNorm's mean-of-squares needs it. Dropping the compute config makes that op 2.4x faster
    and takes model decode PCC from 0.99991 to 0.992, worst sample 1.7% -> 18.9%."""
    assert gpt.COMPUTE_CONFIG.fp32_dest_acc_en is True
    assert gpt.COMPUTE_CONFIG.math_fidelity == ttnn.MathFidelity.HiFi4


def test_block2_weights_are_bfp8_but_fidelity_stays_high():
    """BFP8 weights are 1.23x for one extra differing code in 222. Lowering the MATH fidelity is
    the opposite trade -- ~4 ms for 10-20x the code errors."""
    assert flow.WEIGHT_DTYPE == ttnn.bfloat8_b
    assert flow.COMPUTE_CONFIG.math_fidelity == ttnn.MathFidelity.HiFi4
    assert flow.COMPUTE_CONFIG.fp32_dest_acc_en is True


def test_prefill_padding_stays_on_the_tile_grid():
    """Prefill's causal mask is cut at this boundary; a ragged value would misalign it silently
    rather than raise. 128 itself is a kernel-shape-churn choice, not a hardware limit."""
    assert gpt.PREFILL_MULTIPLE % gpt.TILE == 0


def test_block2_semantic_head_stays_fp32():
    """It produces an INDEX, not a value. Measured over 64 hidden states, bf16 weights pick a
    DIFFERENT index on 4 of them; fp32 matches the host answer on all 64, for 0.2 ms."""
    assert flow.SEMANTIC_DTYPE == ttnn.float32


def test_fused_qkv_width_matches_the_head_config():
    """Decode's head op reads ONE fused projection and splits it by head count; a mismatch would
    mis-slice q/k/v rather than raise."""
    from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
        HEAD_DIM,
        N_HEADS,
        N_KV_HEADS,
    )

    assert gpt._QKV_WIDTH == (N_HEADS + 2 * N_KV_HEADS) * HEAD_DIM


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-svv", __file__]))


# ---------------------------------------------------------------------------------------
# p150 REVERSALS. Each of these was the SHIPPED choice on the N150 and is wrong here.
# ---------------------------------------------------------------------------------------
def test_no_width_sharded_norms_anywhere():
    """Both blocks' decode RMSNorms are the plain interleaved op. Width-sharding them wins ~4.4
    ms/frame on Wormhole and LOSES ~4.5 on Blackhole: the reshard is the tax (6.9's phrase), and
    the p150's interleaved kernel made the reduction cheap enough that the tax stops paying for
    itself. It is also CLOSER to fp32 truth here, not further. STATUS.md 6.39/6.40."""
    for mod in (gpt, flow):
        assert not hasattr(mod, "_NORM_SHARD"), f"{mod.__name__}: sharded norm is back -- 6.39/6.40"
        assert not hasattr(mod, "_NORM_PRG")
    assert not hasattr(gpt.TtVoxtralGPT, "_norm_dec"), "decode norm split back out -- 6.39"


def test_wo_does_not_get_the_n150_hand_tuned_config_back():
    """6.25 hand-tuned _WO_PRG for the N150 (+0.196 ms/frame); 6.43 found it buys nothing here and
    deleted it. wo DOES now carry a program config again, but not that one -- it takes the shared
    12x6 decode config from 6.52, whose reason is the reduction-depth collapse, not wo's shape.
    Keep the two claims apart: the N150 constant stays dead."""
    assert not hasattr(gpt, "_WO_PRG"), "the N150's hand-tuned wo config is back -- 6.43"
    assert not hasattr(gpt, "_WO_GRID")
    assert gpt.DECODE_PRG["wo"] is gpt._PRG_WO


def test_silu_is_fused_by_the_program_config_not_the_activation_kwarg():
    """activation="silu" is NOT fused on this chip: 98.8 us against a plain matmul's 85.5, the
    same +14.9 as a separate ttnn.silu, and UnaryWithParam/UnaryOpType behave identically. Only a
    program config's fused_activation folds it in (88.1). Worth 2.42 ms/frame over the 47 w1 calls
    across both blocks, and slightly MORE accurate (PCC 0.9999984 vs 0.9999970). STATUS.md 6.52."""
    import inspect

    assert gpt._PRG_W1.fused_activation is not None, "w1 lost its fused silu -- 6.52"
    assert gpt._PRG_W3.fused_activation is None, "w3 must NOT have an activation"
    for fn in (gpt.TtVoxtralGPT._layer_step, flow.TtVoxtralFlow._block):
        # comments explain WHY the kwarg is gone and name it; strip them so only code is checked
        code = "\n".join(ln.split("#")[0] for ln in inspect.getsource(fn).splitlines())
        assert 'activation="silu"' not in code, (
            f"{fn.__qualname__} is back on the unfused activation kwarg -- 6.52")


def test_out_subblock_w_is_the_largest_legal_one():
    """The helper's comment says "biggest that fits" and for a while the code did not do that: the
    candidate tuple was (4, 2, 1), which skips 3. ttnn's own SUBBLOCK_HW_CHOICES lists {3,1}
    explicitly, so wqkv's per_core_N=3 fell all the way to out_subblock_w=1 -- three passes through
    the destination registers where one would do. It measured perf-neutral (59.3 vs 59.2 us,
    because subblock width is compute-side and the ALUs are ~99.6% idle at batch 1, §6.53) and the
    paired gate was clean, so it is fixed for correctness of intent rather than for speed. A future
    shape with per_core_N of 3, 9 or 15 would otherwise silently drop to 1 with no indication.

    The two rules, both enforced by ttnn as hard errors:
      out_subblock_h * out_subblock_w <= 4   (8 normally; fp32_dest_acc_en halves the dest file)
      per_core_N % out_subblock_w == 0
    """
    for name, cfg in gpt.DECODE_PRG.items():
        w, n, h = cfg.out_subblock_w, cfg.per_core_N, cfg.out_subblock_h
        assert h * w <= 4, f"{name}: h*w={h*w} exceeds the fp32_dest_acc_en limit of 4"
        assert n % w == 0, f"{name}: per_core_N={n} is not divisible by out_subblock_w={w}"
        bigger = [s for s in range(w + 1, 5) if n % s == 0 and h * s <= 4]
        assert not bigger, (
            f"{name}: out_subblock_w={w}, but {bigger[0]} is also legal and divides "
            f"per_core_N={n} -- the candidate list has a hole in it again")


def test_decode_matmul_configs_assume_one_tile_of_rows():
    """per_core_M=1 and fuse_batch=True are only valid for a single tile of rows -- Block 1's 1 and
    Block 2's 3-or-6. Prefill has many, so _mlp must reach it WITHOUT these configs. STATUS.md
    6.52."""
    import inspect

    for p in gpt.DECODE_PRG.values():
        assert p.per_core_M == 1, "a decode config grew rows; prefill would silently share it"
    prefill = inspect.getsource(gpt.TtVoxtralGPT._layer)
    assert "DECODE_PRG" not in prefill, "prefill must keep the ttnn heuristic -- 6.52"
    assert "self._mlp(x, self._norm(x, w[\"fn\"]), w, ttnn.DRAM_MEMORY_CONFIG)" in prefill, (
        "prefill's _mlp call gained an argument -- check it is not a program config")


def test_kv_cache_uses_two_writes_not_the_fused_one():
    """paged_fused_update_cache is +0.454 ms/frame on the N150 (6.20/6.22) and 0.687 ms/step
    SLOWER here. _V_SHARD existed only to let it accept K and V on different cores, so it goes
    too -- and with it the failure mode where RoPE on a core whose cos/sin table lives elsewhere
    returns 3.4e38 from uninitialised L1 instead of raising. STATUS.md 6.44."""
    import inspect

    src = inspect.getsource(gpt.TtVoxtralGPT._layer_step)
    assert "paged_fused_update_cache" not in src, "fused cache write is back -- 6.44"
    assert src.count("paged_update_cache") == 2, "expected exactly two cache writes"
    assert not hasattr(gpt, "_V_SHARD"), "_V_SHARD is back; it has no consumer without the fused op"


def test_block2_uses_the_fused_head_split_and_sdpa():
    """Block 2's interior is 2 ops, not 13. A small op costs 3.4x more here (67.7 us against the
    N150's ~20), so op count dominates and both 6.31's 9-op hand-rolled split and 6.37's
    rejection of sdpa reverse -- together worth 6.586 ms/frame. The fused split is accuracy-
    IDENTICAL; sdpa is 1.57x the velocity error (against the 6.48x that got it rejected on the
    N150) with the acoustic codes unmoved. STATUS.md 6.45."""
    import inspect

    src = inspect.getsource(flow.TtVoxtralFlow._block)
    assert "nlp_create_qkv_heads" in src, "hand-rolled 9-op split is back -- 6.45"
    assert "scaled_dot_product_attention" in src, "hand-rolled attention interior is back -- 6.45"
    assert "scale=1.0" in src, (
        "sdpa MUST take scale=1.0 -- SCALE is folded into wqkv's q rows ([flow-09]), so the "
        "default applies 1/sqrt(d) twice: 3.8e-01 relative error (6.37)")
    assert not hasattr(flow, "REP"), "the GQA row fold is back; sdpa handles GQA natively"


def test_sdpa_decode_keeps_its_program_config():
    """The one N150 program config that DID survive. k=512 on 8x2 is 1.751x over the default and
    is the only candidate exact at all 13 probe positions -- k=128 is faster still and degrades at
    pos 128 and 1000. 6.27's rule reproduced on new hardware: a position sweep, not a gate run, is
    what makes an sdpa config safe. STATUS.md 6.46."""
    assert gpt._SDPA_PRG.k_chunk_size == 512
    assert gpt._SDPA_PRG.q_chunk_size == gpt.TILE
