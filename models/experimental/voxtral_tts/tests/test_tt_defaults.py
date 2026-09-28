# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The shipped TTNN configuration, pinned.

Each test guards one constant or source-level invariant of the p150 fork so an accidental edit fails
loudly; its docstring names where the measurement lives. see VOXTRAL_TTS_BRINGUP.md [test-05]
Needs no device and no checkpoint -- it only imports the modules.

    pytest -svv models/experimental/voxtral_tts/tests/test_tt_defaults.py
"""

import pytest

ttnn = pytest.importorskip("ttnn")

from models.experimental.voxtral_tts.tt import ttnn_voxtral_flow as flow  # noqa: E402
from models.experimental.voxtral_tts.tt import ttnn_voxtral_gpt as gpt
from models.experimental.voxtral_tts.tt import ttnn_voxtral_pipeline as pipeline  # noqa: E402


def test_backbone_weights_are_bfp8_except_w2():
    """The backbone is BFP8 except w2, which is bf16 for ACCURACY, not for the old hang.
    see VOXTRAL_TTS_STATUS.md §6.16 (the trade) and §6.12-§6.13 (the hang)"""
    assert gpt.WEIGHT_DTYPE == ttnn.bfloat16        # w2 -- accuracy, see above
    assert gpt.FF_WEIGHT_DTYPE == ttnn.bfloat8_b    # FF1, FF3 -- see §6.16
    assert gpt.ATTN_WEIGHT_DTYPE == ttnn.bfloat8_b  # wqkv, wo -- see §6.16


def test_codec_output_projection_does_not_use_conv1d():
    """The codec's output projection must not call ttnn.conv1d, whose halo_gather hangs the card.
    see VOXTRAL_TTS_STATUS.md §6.12-§6.14 and VOXTRAL_TTS_CODEC.md [codec-17]"""
    import inspect

    from models.experimental.voxtral_tts.tt import ttnn_voxtral_codec as codec

    src = inspect.getsource(codec.TtVoxtralCodecDecoder._graph)
    init = inspect.getsource(codec.TtVoxtralCodecDecoder.__init__)
    assert '_conv1d(x, "out"' not in src, "the output projection is back on ttnn.conv1d -- see 6.13"
    assert "_out_taps" in init
    # Its prefix comes from ttnn.gather, not _pad_causal's slices; only the clock catches a revert.
    # see VOXTRAL_TTS_STATUS.md §6.14
    assert "self._pad_causal(" not in src, "the projection is back on the slice-built pad -- see 6.14"
    assert "ttnn.gather(" in src and "_out_prefix_idx" in init


def test_backbone_math_config_keeps_fp32_accumulation():
    """RMSNorm's mean-of-squares needs fp32 accumulation. see VOXTRAL_TTS_BACKBONE.md [gpt-12]"""
    assert gpt.COMPUTE_CONFIG.fp32_dest_acc_en is True
    assert gpt.COMPUTE_CONFIG.math_fidelity == ttnn.MathFidelity.HiFi4


def test_flow_model_weights_are_bfp8_but_fidelity_stays_high():
    """The flow model takes BFP8 weights but keeps HiFi4 + fp32 accumulation.
    see VOXTRAL_TTS_FLOW.md [flow-05] (weights) and [flow-03] (fidelity)"""
    assert flow.WEIGHT_DTYPE == ttnn.bfloat8_b
    assert flow.COMPUTE_CONFIG.math_fidelity == ttnn.MathFidelity.HiFi4
    assert flow.COMPUTE_CONFIG.fp32_dest_acc_en is True


def test_prefill_padding_stays_on_the_tile_grid():
    """Prefill's causal mask is cut at this boundary; a ragged value misaligns it silently.
    see VOXTRAL_TTS_BACKBONE.md [gpt-02]"""
    assert gpt.PREFILL_MULTIPLE % gpt.TILE == 0


def test_flow_model_semantic_head_stays_fp32():
    """The semantic head produces an index, so it stays fp32. see VOXTRAL_TTS_FLOW.md [flow-08]"""
    assert flow.SEMANTIC_DTYPE == ttnn.float32


def test_fused_qkv_width_matches_the_head_config():
    """The fused q/k/v projection is split by head count; a mismatch mis-slices silently.
    see VOXTRAL_TTS_BACKBONE.md [gpt-10]"""
    from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
        HEAD_DIM,
        N_HEADS,
        N_KV_HEADS,
    )

    assert gpt._QKV_WIDTH == (N_HEADS + 2 * N_KV_HEADS) * HEAD_DIM


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-svv", __file__]))


# ---------------------------------------------------------------------------------------
# p150 reversals of N150 choices. see VOXTRAL_TTS_BRINGUP.md [test-05]
# ---------------------------------------------------------------------------------------
def test_sharded_norm_is_decode_only_and_legally_shaped():
    """The width-sharded decode norm: decode only (prefill falls back to interleaved), and
    cores * block_w == 96 tiles. see VOXTRAL_TTS_STATUS.md §6.67 and
    VOXTRAL_TTS_BACKBONE.md [gpt-28]"""
    nc = gpt._NORM_GRID[0] * gpt._NORM_GRID[1]
    assert nc * gpt._NORM_PRG.block_w == gpt.DIM // gpt.TILE, (
        f"{nc} cores x block_w {gpt._NORM_PRG.block_w} != {gpt.DIM // gpt.TILE} tiles")
    assert gpt._NORM_PRG.block_h == 1, "the shard assumes one tile of rows"
    import inspect
    src = inspect.getsource(gpt.sharded_norm)
    assert "x.shape[-2] > TILE" in src, "the prefill fallback is gone -- prefill will fail"
    assert flow.TtVoxtralFlow._norm.__module__ == flow.__name__


def test_wo_does_not_get_the_n150_hand_tuned_config_back():
    """The N150's hand-tuned _WO_PRG stays deleted; wo takes the shared decode config instead.
    see VOXTRAL_TTS_STATUS.md §6.43, §6.52, §6.78 and VOXTRAL_TTS_BACKBONE.md [gpt-20]"""
    assert not hasattr(gpt, "_WO_PRG"), "the N150's hand-tuned wo config is back -- 6.43"
    assert not hasattr(gpt, "_WO_GRID")
    assert gpt.DECODE_PRG["wo"] is gpt._PRG_WO


def test_decode_matmul_grid_fits_the_smaller_card_and_keeps_the_12x6_split():
    """_MM_GRID fits the 11x10 card and keeps 12x6's per_core_N, so the output stays bit-identical;
    changing any per_core_N is a model change. see VOXTRAL_TTS_STATUS.md §6.78, [gpt-29]"""
    import math

    assert gpt._MM_GRID[0] <= 11 and gpt._MM_GRID[1] <= 10, (
        f"_MM_GRID {gpt._MM_GRID} does not fit the 11x10 p150b this port runs on")
    was = 12 * 6
    ntiles = {"wqkv": 6144, "wo": 3072, "w1": 9216, "w3": 9216, "w2": 3072}
    for name, n in ntiles.items():
        exp = math.ceil(n // gpt.TILE / was)
        assert gpt.DECODE_PRG[name].per_core_N == exp, (
            f"{name}: per_core_N {gpt.DECODE_PRG[name].per_core_N} != 12x6's {exp} -- not bit-exact")
        cores = math.ceil(n // gpt.TILE / exp)
        assert cores <= gpt._MM_GRID[0] * gpt._MM_GRID[1], f"{name} needs {cores} cores"


def test_silu_is_fused_by_the_program_config_not_the_activation_kwarg():
    """SiLU is fused only by the program config's fused_activation; the activation kwarg is not
    fused on this chip. see VOXTRAL_TTS_STATUS.md §6.52, VOXTRAL_TTS_BACKBONE.md [gpt-26]"""
    import inspect

    assert gpt._PRG_W1.fused_activation is not None, "w1 lost its fused silu -- 6.52"
    assert gpt._PRG_W3.fused_activation is None, "w3 must NOT have an activation"
    for fn in (gpt.TtVoxtralGPT._layer_step, flow.TtVoxtralFlow._block):
        # comments explain WHY the kwarg is gone and name it; strip them so only code is checked
        code = "\n".join(ln.split("#")[0] for ln in inspect.getsource(fn).splitlines())
        assert 'activation="silu"' not in code, (
            f"{fn.__qualname__} is back on the unfused activation kwarg -- 6.52")


def test_out_subblock_w_is_the_largest_legal_one():
    """out_subblock_w is the largest legal width: h * w <= 4 (fp32_dest_acc_en) and
    per_core_N % w == 0. see VOXTRAL_TTS_STATUS.md §6.61
    """
    for name, cfg in gpt.DECODE_PRG.items():
        w, n, h = cfg.out_subblock_w, cfg.per_core_N, cfg.out_subblock_h
        assert h * w <= 4, f"{name}: h*w={h*w} exceeds the fp32_dest_acc_en limit of 4"
        assert n % w == 0, f"{name}: per_core_N={n} is not divisible by out_subblock_w={w}"
        bigger = [s for s in range(w + 1, 5) if n % s == 0 and h * s <= 4]
        assert not bigger, (
            f"{name}: out_subblock_w={w}, but {bigger[0]} is also legal and divides "
            f"per_core_N={n} -- the candidate list has a hole in it again")


def test_residual_rides_in_as_bias_on_the_decode_path_only():
    """Residual-as-bias is valid only at one row, so decode takes it and prefill must not.
    see VOXTRAL_TTS_STATUS.md §6.62, VOXTRAL_TTS_BACKBONE.md [gpt-27]"""
    import inspect

    step = inspect.getsource(gpt.TtVoxtralGPT._layer_step)
    assert "bias=" in step, "wo's residual is back to a separate add -- 6.62"
    mlp = inspect.getsource(gpt.TtVoxtralGPT._mlp)
    assert "if prg:" in mlp and "bias=" in mlp, "w2's residual bias is gone -- 6.62"
    assert "ttnn.add_(x, ttnn.linear(u, w[\"w2\"]" in mlp, (
        "the prefill fallback add is gone; prefill must NOT take the bias path")
    prefill = inspect.getsource(gpt.TtVoxtralGPT._layer)
    assert "bias=" not in prefill, "prefill is using residual-as-bias, which is WRONG for M>1"


def test_trace_capture_aims_the_cache_write_at_the_first_frames_slot():
    """Both graph() runs in _trace_capture write K/V at `pos`, so it must be aimed at pos0 first,
    and the trace released in a finally. see VOXTRAL_TTS_STATUS.md §6.65 and
    VOXTRAL_TTS_BRINGUP.md [pipe-05]"""
    import inspect

    src = inspect.getsource(pipeline.TtVoxtralPipeline._trace_capture)
    assert "copy_host_to_device_tensor" in src and "pos0" in src, (
        "the capture no longer aims `pos` at pos0 -- it will corrupt KV cache slot 0")
    # the CALL, not `def graph():` -- the definition necessarily comes first
    i_call = src.index("\n        graph()")
    i_aim = src.index('torch.tensor([pos0]')
    assert i_aim < i_call, "`pos` must be aimed BEFORE the warm-up graph() runs"
    gen = inspect.getsource(pipeline.TtVoxtralPipeline.generate)
    assert "finally:" in gen and "_trace_release" in gen, (
        "the trace must be released in a finally -- the next generate() prefills, which allocates")


def test_decode_matmul_configs_assume_one_tile_of_rows():
    """Decode program configs assume one tile of rows, so prefill's _mlp must not get them.
    see VOXTRAL_TTS_STATUS.md §6.52, VOXTRAL_TTS_BACKBONE.md [gpt-26]"""
    import inspect

    for p in gpt.DECODE_PRG.values():
        assert p.per_core_M == 1, "a decode config grew rows; prefill would silently share it"
    prefill = inspect.getsource(gpt.TtVoxtralGPT._layer)
    assert "DECODE_PRG" not in prefill, "prefill must keep the ttnn heuristic -- 6.52"
    assert "self._mlp(x, self._norm(x, w[\"fn\"]), w, ttnn.DRAM_MEMORY_CONFIG)" in prefill, (
        "prefill's _mlp call gained an argument -- check it is not a program config")


def test_kv_cache_uses_two_writes_not_the_fused_one():
    """Two paged_update_cache writes, not the fused one, and no _V_SHARD.
    see VOXTRAL_TTS_STATUS.md §6.44, VOXTRAL_TTS_BACKBONE.md [gpt-19]"""
    import inspect

    src = inspect.getsource(gpt.TtVoxtralGPT._layer_step)
    assert "paged_fused_update_cache" not in src, "fused cache write is back -- 6.44"
    assert src.count("paged_update_cache") == 2, "expected exactly two cache writes"
    assert not hasattr(gpt, "_V_SHARD"), "_V_SHARD is back; it has no consumer without the fused op"


def test_flow_model_hand_rolls_the_head_split_and_keeps_sdpa():
    """The flow model splits heads with nine L1-pinned ops and attends with sdpa.
    see VOXTRAL_TTS_STATUS.md §6.72, VOXTRAL_TTS_FLOW.md [flow-10]"""
    import inspect

    blk = inspect.getsource(flow.TtVoxtralFlow._block)
    assert "_split_heads" in blk, "the head split left _block -- 6.72"
    assert "scaled_dot_product_attention" in blk, "hand-rolled attention interior is back -- 6.45"
    assert "scale=1.0" in blk, (
        "sdpa MUST take scale=1.0 -- SCALE is folded into wqkv's q rows ([flow-09]), so the "
        "default applies 1/sqrt(d) twice: 3.8e-01 relative error (6.37)")

    # ast, not a `#`-strip: _split_heads names the op it does not call in its docstring, and
    # dropping the docstring node before unparsing leaves executable code only.
    import ast, textwrap

    fn = ast.parse(textwrap.dedent(inspect.getsource(flow._split_heads))).body[0]
    if ast.get_docstring(fn):
        fn.body = fn.body[1:]
    code = ast.unparse(fn)
    assert "nlp_create_qkv_heads" not in code, "the fused split is back -- 6.72 measured it slower"
    assert code.count("memory_config=_L1") == 2, (
        "both the slice and the permute must pin _L1: DRAM outputs cost 9.6 us/split and "
        "silently undo [flow-02]")
    assert "HANDSPLIT" not in code, "the A/B env switch is back; this branch ships one path (6.72)"
    assert not hasattr(flow, "REP"), "the GQA row fold is back; sdpa handles GQA natively"


def test_sdpa_decode_keeps_its_program_config():
    """sdpa_decode keeps its N150 program config (k_chunk 512), chosen by a position sweep.
    see VOXTRAL_TTS_STATUS.md §6.46, VOXTRAL_TTS_BACKBONE.md [gpt-21]"""
    assert gpt._SDPA_PRG.k_chunk_size == 512
    assert gpt._SDPA_PRG.q_chunk_size == gpt.TILE
