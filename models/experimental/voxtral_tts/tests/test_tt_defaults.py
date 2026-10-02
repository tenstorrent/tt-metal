# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The shipped TTNN configuration, pinned.

Each test guards one constant or source-level invariant of the p150 fork so an accidental edit fails
loudly; its docstring says what the choice protects.
Needs no device and no checkpoint -- it only imports the modules.

    pytest -svv models/experimental/voxtral_tts/tests/test_tt_defaults.py
"""

import os

import pytest

ttnn = pytest.importorskip("ttnn")

from models.experimental.voxtral_tts.tt import ttnn_voxtral_flow as flow  # noqa: E402
from models.experimental.voxtral_tts.tt import ttnn_voxtral_gpt as gpt
from models.experimental.voxtral_tts.tt import ttnn_voxtral_pipeline as pipeline  # noqa: E402


def _flat(text):
    """Source with all whitespace removed, so a formatter re-wrapping a call cannot hide it."""
    return "".join(text.split())


def test_backbone_weights_are_bfp8_except_w2():
    """The backbone's weights are BFP8, since decode is bound by weight bytes, except w2, which
    stays bf16 for accuracy."""
    assert gpt.WEIGHT_DTYPE == ttnn.bfloat16  # w2
    assert gpt.FF_WEIGHT_DTYPE == ttnn.bfloat8_b  # FF1, FF3
    assert gpt.ATTN_WEIGHT_DTYPE == ttnn.bfloat8_b  # wqkv, wo


def test_codec_output_projection_does_not_use_conv1d():
    """The codec's output projection must not call ttnn.conv1d, whose halo_gather hangs the card."""
    import inspect

    from models.experimental.voxtral_tts.tt import ttnn_voxtral_codec as codec

    src = inspect.getsource(codec.TtVoxtralCodecDecoder._graph)
    init = inspect.getsource(codec.TtVoxtralCodecDecoder.__init__)
    assert _flat('_conv1d(x, "out"') not in _flat(src), "the output projection is back on ttnn.conv1d"
    assert "_out_taps" in init
    # Its prefix comes from ttnn.gather, not _pad_causal's slices, which are slower and give the same
    # values, so only timing would catch a revert.
    assert "self._pad_causal(" not in src, "the projection is back on the slice-built pad"
    assert "ttnn.gather(" in src and "_out_prefix_idx" in init


def test_codec_convs_default_to_matmuls():
    """The codec's four convs run as tap matmuls by default, not ttnn.conv1d/conv_transpose2d: the halo
    path hung a chip twice on 2026-10-01/02 (a second pipeline's codec warming up, buckets 512 and 640)."""
    import inspect

    from models.experimental.voxtral_tts.tt import ttnn_voxtral_codec as codec

    assert codec.CONV_IMPL == "matmul" or os.environ.get("VOXTRAL_CODEC_CONV"), "codec back on ttnn conv ops"
    src = _flat(inspect.getsource(codec.TtVoxtralCodecDecoder._graph))
    assert _flat("self._conv1d_mm(") in src and _flat("self._conv_transpose_mm(") in src
    assert codec.LATENT_PAD % 32 == 0 and codec.LATENT_PAD >= codec.LATENT_DIM


def test_backbone_math_config_keeps_fp32_accumulation():
    """RMSNorm's mean-of-squares needs fp32 accumulation: its small per-op error compounds through
    every layer."""
    assert gpt.COMPUTE_CONFIG.fp32_dest_acc_en is True
    assert gpt.COMPUTE_CONFIG.math_fidelity == ttnn.MathFidelity.HiFi4


def test_flow_model_weights_are_bfp8_but_fidelity_stays_high():
    """The flow model takes BFP8 weights, which barely move its codes, but keeps HiFi4 + fp32
    accumulation, since lower fidelity multiplies the code errors."""
    assert flow.WEIGHT_DTYPE == ttnn.bfloat8_b
    assert flow.COMPUTE_CONFIG.math_fidelity == ttnn.MathFidelity.HiFi4
    assert flow.COMPUTE_CONFIG.fp32_dest_acc_en is True


def test_prefill_padding_stays_on_the_tile_grid():
    """Prefill's causal mask is cut at this boundary; a ragged value misaligns it silently."""
    assert gpt.PREFILL_MULTIPLE % gpt.TILE == 0


def test_flow_model_semantic_head_stays_fp32():
    """The semantic head feeds an argmax, so it stays fp32: two close logits ranked the other way
    round change the semantic code outright."""
    assert flow.SEMANTIC_DTYPE == ttnn.float32


def test_fused_qkv_width_matches_the_head_config():
    """The fused q/k/v projection is split by head count; a mismatch mis-slices silently."""
    from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
        HEAD_DIM,
        N_HEADS,
        N_KV_HEADS,
    )

    assert gpt._QKV_WIDTH == (N_HEADS + 2 * N_KV_HEADS) * HEAD_DIM


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-svv", __file__]))


# ---------------------------------------------------------------------------------------
# p150 reversals of N150 choices.
# ---------------------------------------------------------------------------------------
def test_sharded_norm_is_decode_only_and_legally_shaped():
    """The width-sharded decode norm: decode only (prefill falls back to interleaved), and
    cores * block_w == 96 tiles."""
    nc = gpt._NORM_GRID[0] * gpt._NORM_GRID[1]
    assert (
        nc * gpt._NORM_PRG.block_w == gpt.DIM // gpt.TILE
    ), f"{nc} cores x block_w {gpt._NORM_PRG.block_w} != {gpt.DIM // gpt.TILE} tiles"
    assert gpt._NORM_PRG.block_h == 1, "the shard assumes one tile of rows"
    import inspect

    src = inspect.getsource(gpt.sharded_norm)
    assert "x.shape[-2] > TILE" in src, "the prefill fallback is gone -- prefill will fail"
    assert flow.TtVoxtralFlow._norm.__module__ == flow.__name__


def test_wo_does_not_get_the_n150_hand_tuned_config_back():
    """The N150's hand-tuned _WO_PRG stays deleted: it gains nothing on the p150, so wo takes the
    shared decode config."""
    assert not hasattr(gpt, "_WO_PRG"), "the N150's hand-tuned wo config is back"
    assert not hasattr(gpt, "_WO_GRID")
    assert "wo" in gpt._DECODE_SPLIT


def test_decode_grid_follows_the_device_and_keeps_the_12x6_split(expect_error):
    """decode_grid picks a grid that fits the device, and every grid it picks keeps 12x6's
    per_core_N, so the output is bit-identical; changing any per_core_N is a model change."""
    import math
    from types import SimpleNamespace as Grid

    assert gpt.decode_grid(Grid(x=13, y=10)) == (12, 6)
    assert gpt.decode_grid(Grid(x=11, y=10)) == (11, 7)
    assert gpt.decode_grid(Grid(x=10, y=10)) == (10, 8)
    with expect_error(RuntimeError, "no rectangle"):
        gpt.decode_grid(Grid(x=7, y=10))
    ntiles = {"wqkv": 6144, "wo": 3072, "w1": 9216, "w3": 9216, "w2": 3072}
    for grid in ((12, 6), (11, 7), (10, 8)):
        cfgs = gpt.decode_program_configs(grid)
        for name, n in ntiles.items():
            exp = math.ceil(n // gpt.TILE / (12 * 6))
            assert cfgs[name].per_core_N == exp, f"{grid} {name}: per_core_N {cfgs[name].per_core_N} != 12x6's {exp}"
            cores = math.ceil(n // gpt.TILE / exp)
            assert cores <= grid[0] * grid[1], f"{grid} {name} needs {cores} cores"


def test_silu_is_fused_by_the_program_config_not_the_activation_kwarg():
    """SiLU is fused only by the program config's fused_activation; the activation kwarg is not
    fused on this chip."""
    import inspect

    cfgs = gpt.decode_program_configs((11, 7))
    assert cfgs["w1"].fused_activation is not None, "w1 lost its fused silu"
    assert cfgs["w3"].fused_activation is None, "w3 must NOT have an activation"
    for fn in (gpt.TtVoxtralGPT._layer_step, flow.TtVoxtralFlow._block):
        # comments explain WHY the kwarg is gone and name it; strip them so only code is checked
        code = "\n".join(ln.split("#")[0] for ln in inspect.getsource(fn).splitlines())
        assert 'activation="silu"' not in code, f"{fn.__qualname__} is back on the unfused activation kwarg"


def test_out_subblock_w_is_the_largest_legal_one():
    """out_subblock_w is the largest legal width: h * w <= 4 (fp32_dest_acc_en) and
    per_core_N % w == 0."""
    for name, cfg in gpt.decode_program_configs((11, 7)).items():
        w, n, h = cfg.out_subblock_w, cfg.per_core_N, cfg.out_subblock_h
        assert h * w <= 4, f"{name}: h*w={h*w} exceeds the fp32_dest_acc_en limit of 4"
        assert n % w == 0, f"{name}: per_core_N={n} is not divisible by out_subblock_w={w}"
        bigger = [s for s in range(w + 1, 5) if n % s == 0 and h * s <= 4]
        assert not bigger, (
            f"{name}: out_subblock_w={w}, but {bigger[0]} is also legal and divides "
            f"per_core_N={n} -- the candidate list has a hole in it again"
        )


def test_residual_rides_in_as_bias_on_the_decode_path_only():
    """Residual-as-bias is valid only at ONE row, so the batch-1 decode path takes it and both
    prefill and batched decode (max_batch > 1, one row per user) must use a real add."""
    import inspect

    step = inspect.getsource(gpt.TtVoxtralGPT._layer_step)
    assert "bias=" in step, "wo's residual is back to a separate add"
    assert "if B == 1:" in step and "ttnn.add(" in step, "wo must add the residual per row when B > 1"
    mlp = inspect.getsource(gpt.TtVoxtralGPT._mlp)
    assert "if prg and self.max_batch == 1:" in mlp and "bias=" in mlp, "w2's residual bias is gone"
    assert _flat('ttnn.add_(x, ttnn.linear(u, w["w2"]') in _flat(
        mlp
    ), "the prefill fallback add is gone; prefill must NOT take the bias path"
    prefill = inspect.getsource(gpt.TtVoxtralGPT._layer)
    assert "bias=" not in prefill, "prefill is using residual-as-bias, which is WRONG for M>1"


def test_trace_capture_aims_the_cache_write_at_the_first_frames_slot():
    """Both graph() runs in _trace_capture write K/V at `pos`, so it must be aimed at pos0 first,
    and the trace released in a finally."""
    import inspect

    src = inspect.getsource(pipeline.TtVoxtralPipeline._trace_capture)
    assert (
        "copy_host_to_device_tensor" in src and "pos0" in src
    ), "the capture no longer aims `pos` at pos0 -- it will corrupt KV cache slot 0"
    # the CALL, not `def graph():` -- the definition necessarily comes first
    i_call = src.index("\n        graph()")
    i_aim = src.index("torch.tensor([pos0]")
    assert i_aim < i_call, "`pos` must be aimed BEFORE the warm-up graph() runs"
    gen = inspect.getsource(pipeline.TtVoxtralPipeline.generate)
    assert (
        "finally:" in gen and "_trace_release" in gen
    ), "the trace must be released in a finally -- the next generate() prefills, which allocates"


def test_decode_matmul_configs_assume_one_tile_of_rows():
    """Decode program configs assume one tile of rows, so prefill's _mlp must not get them."""
    import inspect

    for p in gpt.decode_program_configs((11, 7)).values():
        assert p.per_core_M == 1, "a decode config grew rows; prefill would silently share it"
    prefill = inspect.getsource(gpt.TtVoxtralGPT._layer)
    assert "decode_prg" not in prefill, "prefill must keep the ttnn heuristic"
    assert _flat('self._mlp(x, self._norm(x, w["fn"]), w, ttnn.DRAM_MEMORY_CONFIG)') in _flat(
        prefill
    ), "prefill's _mlp call gained an argument -- check it is not a program config"


def test_kv_cache_uses_two_writes_not_the_fused_one():
    """Two paged_update_cache writes, not the fused one, which is slower here; without it _V_SHARD
    has no consumer."""
    import inspect

    src = inspect.getsource(gpt.TtVoxtralGPT._layer_step)
    assert "paged_fused_update_cache" not in src, "fused cache write is back"
    assert src.count("paged_update_cache") == 2, "expected exactly two cache writes"
    assert not hasattr(gpt, "_V_SHARD"), "_V_SHARD is back; it has no consumer without the fused op"


def test_flow_model_hand_rolls_the_head_split_and_keeps_sdpa():
    """The flow model splits heads with nine L1-pinned ops, faster than nlp_create_qkv_heads, and
    attends with sdpa."""
    import inspect

    blk = inspect.getsource(flow.TtVoxtralFlow._block)
    assert "_split_heads" in blk, "the head split left _block"
    assert "scaled_dot_product_attention" in blk, "hand-rolled attention interior is back"
    assert (
        "scale=1.0" in blk
    ), "sdpa MUST take scale=1.0 -- SCALE is folded into wqkv's q rows, so the default applies 1/sqrt(d) twice"

    # ast, not a `#`-strip: _split_heads names the op it does not call in its docstring, and
    # dropping the docstring node before unparsing leaves executable code only.
    import ast, textwrap

    fn = ast.parse(textwrap.dedent(inspect.getsource(flow._split_heads))).body[0]
    if ast.get_docstring(fn):
        fn.body = fn.body[1:]
    code = ast.unparse(fn)
    assert "nlp_create_qkv_heads" not in code, "the fused split is back, and it is slower"
    assert (
        code.count("memory_config=_L1") == 2
    ), "both the slice and the permute must pin _L1: DRAM outputs are slower and silently undo the L1 placement"
    assert "HANDSPLIT" not in code, "the A/B env switch is back; this branch ships one path"
    assert not hasattr(flow, "REP"), "the GQA row fold is back; sdpa handles GQA natively"


def test_sdpa_decode_keeps_its_program_config():
    """sdpa_decode keeps the N150's program config (k_chunk 512): the faster candidates are not
    exact at every cache position."""
    assert gpt._SDPA_PRG.k_chunk_size == 512
    assert gpt._SDPA_PRG.q_chunk_size == gpt.TILE
