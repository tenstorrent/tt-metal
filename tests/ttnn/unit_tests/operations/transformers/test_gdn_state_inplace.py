# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pre-allocated state outputs of the two GDN prefill ops (QWEN36_GDN_STATE_INPLACE).

- ttnn.transformer.chunk_gated_delta_rule(..., final_state_output=S): the kernel writes the final state
  into S and returns S. S may be initial_state itself (in-place state update).
- ttnn.experimental.kda.qkv_causal_conv1d_silu(..., return_conv_state=True, conv_state_output=H): the op
  writes new_state into H and returns H. H may be history itself (in-place conv-state update; the reader
  then writes each block's new_state from the core that reads that block's history).

Contract checked here: every output is torch.equal to the returned-tensor path, the returned state is the
caller's buffer (no new state tensor), inputs other than the aliased state are untouched, and program-cache
hits stay correct when aliased and non-aliased calls with the same specs alternate.
"""

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.common.utility_functions import is_blackhole

pytestmark = pytest.mark.skipif(
    not is_blackhole(), reason="the fused/phased GDN paths and the tiled conv are Blackhole-only"
)

# Qwen3.5-2B GDN layer shapes.
HK, HV, DK, DV = 16, 16, 128, 128
CONV_WIDTHS = (2048, 2048, 2048)


def _dev(t, dtype, device, mem=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)


def _padded(t):
    return t.cpu().to_torch_with_padded_shape()


def _mem(name):
    return ttnn.L1_MEMORY_CONFIG if name == "l1" else ttnn.DRAM_MEMORY_CONFIG


# ---------------------------------------------------------------------------------------------------------
# chunk_gated_delta_rule final_state_output
# ---------------------------------------------------------------------------------------------------------


def _fla_const(device, gb_flat):
    c = 32
    eye = torch.eye(c)
    tril = torch.tril(torch.ones(c, c))
    ones = torch.ones(c, c)
    ii = torch.arange(32).unsqueeze(1)
    jj = torch.arange(32).unsqueeze(0)
    lo_i, lo_j = ii < 16, jj < 16
    masks = torch.cat([(lo_i & lo_j).float(), (~lo_i & ~lo_j).float(), (~lo_i & lo_j).float()], dim=1)
    up = lambda t: _dev(t.reshape(1, 1, *t.shape), ttnn.float32, device)
    sel = None
    if gb_flat:
        sel_t = torch.zeros(32, 32 * HV)
        for h in range(HV):
            sel_t[h, h * 32] = 1.0
        sel = up(sel_t)
    return dict(eye=up(eye), tril=up(tril), ones=up(ones), masks=up(masks), sel=sel)


def _fla_inputs(device, seq, seed):
    """The model's call shape: flat q/k/v [1,T,H*D] bf16 (in-kernel L2 norm), g/beta [1,T,HV] fp32."""
    torch.manual_seed(seed)
    q = torch.randn(1, seq, HK * DK).to(torch.bfloat16)
    k = torch.randn(1, seq, HK * DK).to(torch.bfloat16)
    v = (0.5 * torch.randn(1, seq, HV * DV)).to(torch.bfloat16)
    g = -F.softplus(torch.randn(1, seq, HV)) * 0.5
    beta = torch.sigmoid(torch.randn(1, seq, HV))
    s0 = 0.05 * torch.randn(1, HV, DK, DV)
    tensors = (
        _dev(q, ttnn.bfloat16, device),
        _dev(k, ttnn.bfloat16, device),
        _dev(v, ttnn.bfloat16, device),
        _dev(g, ttnn.float32, device),
        _dev(beta, ttnn.float32, device),
    )
    return tensors, s0


_FLA_PATHS = {
    "fused_nv1np5": lambda: ttnn.ChunkGdnFusedProgramConfig(num_producers=5, num_receivers=1, row_local=True),
    "fused_nv2np4": lambda: ttnn.ChunkGdnFusedProgramConfig(num_producers=4, num_receivers=2),
    "phased": lambda: ttnn.ChunkGdnPhasedProgramConfig(),
}


def _run_fla(tensors, const, path, o_mem, initial_state, final_state_output=None):
    q, k, v, g, beta = tensors
    kwargs = dict(const)
    if kwargs["sel"] is None:
        del kwargs["sel"]
    if final_state_output is not None:
        kwargs["final_state_output"] = final_state_output
    return ttnn.transformer.chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=32,
        output_head_major=True,
        program_config=_FLA_PATHS[path](),
        wy_inverse=ttnn.ChunkGdnWyInverse.SFPU,
        memory_config=_mem(o_mem),
        **kwargs,
    )


@pytest.mark.parametrize("seq", [256, 2048], ids=["T256", "T2048"])
@pytest.mark.parametrize("o_mem", ["l1", "dram"])
@pytest.mark.parametrize("path", list(_FLA_PATHS))
def test_fla_final_state_output(device, path, o_mem, seq):
    if path == "phased" and o_mem == "l1":
        pytest.skip("phased puts its seven prep intermediates on memory_config too; not a model configuration")
    const = _fla_const(device, gb_flat=path.startswith("fused"))
    tensors, s0 = _fla_inputs(device, seq, seed=7)
    s0_tt = _dev(s0, ttnn.float32, device)

    # Returned-tensor path (reference).
    o_ref_tt, fs_ref_tt = _run_fla(tensors, const, path, o_mem, s0_tt)
    o_ref, fs_ref = ttnn.to_torch(o_ref_tt), ttnn.to_torch(fs_ref_tt)
    assert fs_ref_tt.memory_config().buffer_type == _mem(o_mem).buffer_type

    # Separate pre-allocated DRAM output (o stays on o_mem).
    out = _dev(torch.zeros(1, HV, DK, DV), ttnn.float32, device)
    o_tt, fs_tt = _run_fla(tensors, const, path, o_mem, s0_tt, final_state_output=out)
    assert fs_tt.buffer_address() == out.buffer_address(), "final_state must be the caller's buffer"
    assert list(fs_tt.shape) == [1, HV, DK, DV]
    assert fs_tt.memory_config().buffer_type == ttnn.BufferType.DRAM
    assert torch.equal(ttnn.to_torch(o_tt), o_ref)
    assert torch.equal(ttnn.to_torch(out), fs_ref)
    assert torch.equal(ttnn.to_torch(s0_tt), s0.to(torch.float32)), "initial_state must be untouched"

    # In place: initial_state and final_state_output are the same (persistent) buffer.
    st = _dev(s0, ttnn.float32, device)
    o_tt, fs_tt = _run_fla(tensors, const, path, o_mem, st, final_state_output=st)
    assert fs_tt.buffer_address() == st.buffer_address()
    assert torch.equal(ttnn.to_torch(o_tt), o_ref)
    assert torch.equal(ttnn.to_torch(st), fs_ref)

    # Second in-place call (program-cache hit): chains from the state the first call left.
    o2_ref_tt, fs2_ref_tt = _run_fla(tensors, const, path, o_mem, _dev(fs_ref, ttnn.float32, device))
    o_tt, _ = _run_fla(tensors, const, path, o_mem, st, final_state_output=st)
    assert torch.equal(ttnn.to_torch(o_tt), ttnn.to_torch(o2_ref_tt))
    assert torch.equal(ttnn.to_torch(st), ttnn.to_torch(fs2_ref_tt))

    # Non-aliased again after the aliased entry (same specs): each buffer is bound to its own slot.
    out2 = _dev(torch.zeros(1, HV, DK, DV), ttnn.float32, device)
    o_tt, fs_tt = _run_fla(tensors, const, path, o_mem, s0_tt, final_state_output=out2)
    assert fs_tt.buffer_address() == out2.buffer_address()
    assert torch.equal(ttnn.to_torch(o_tt), o_ref)
    assert torch.equal(ttnn.to_torch(out2), fs_ref)
    assert torch.equal(ttnn.to_torch(s0_tt), s0.to(torch.float32))


def test_fla_final_state_output_bh_kv_shape(device):
    """A [B*HV, K, V] output is accepted and returned as given."""
    const = _fla_const(device, gb_flat=True)
    tensors, s0 = _fla_inputs(device, 256, seed=3)
    s0_tt = _dev(s0, ttnn.float32, device)
    _, fs_ref_tt = _run_fla(tensors, const, "fused_nv1np5", "dram", s0_tt)
    out = _dev(torch.zeros(HV, DK, DV), ttnn.float32, device)
    _, fs_tt = _run_fla(tensors, const, "fused_nv1np5", "dram", s0_tt, final_state_output=out)
    assert fs_tt.buffer_address() == out.buffer_address() and list(fs_tt.shape) == [HV, DK, DV]
    assert torch.equal(ttnn.to_torch(out).reshape(1, HV, DK, DV), ttnn.to_torch(fs_ref_tt))


@pytest.mark.parametrize("case", ["no_final_state", "bf16", "bad_shape", "mono"])
def test_fla_final_state_output_rejects(device, case, expect_error):
    const = _fla_const(device, gb_flat=False)
    torch.manual_seed(0)
    T = 64
    q = _dev(F.normalize(torch.randn(1, T, 4, DK), dim=-1).to(torch.bfloat16), ttnn.bfloat16, device)
    k = _dev(F.normalize(torch.randn(1, T, 4, DK), dim=-1).to(torch.bfloat16), ttnn.bfloat16, device)
    v = _dev(torch.randn(1, T, 4, DV).to(torch.bfloat16), ttnn.bfloat16, device)
    g = _dev(-F.softplus(torch.randn(1, T, 4)), ttnn.float32, device)
    beta = _dev(torch.sigmoid(torch.randn(1, T, 4)), ttnn.float32, device)
    out_shape, out_dtype, pcfg, ofs = [1, 4, DK, DV], ttnn.float32, ttnn.ChunkGdnPhasedProgramConfig(), True
    if case == "no_final_state":
        ofs = False
    elif case == "bf16":
        out_dtype = ttnn.bfloat16
    elif case == "bad_shape":
        out_shape = [1, 4, DK, 2 * DV]
    else:
        pcfg = ttnn.ChunkGdnMonoProgramConfig()
    out = _dev(torch.zeros(out_shape), out_dtype, device)
    kwargs = {n: const[n] for n in ("eye", "tril", "ones", "masks")}
    with expect_error(RuntimeError, "final_state_output"):
        ttnn.transformer.chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            output_final_state=ofs,
            chunk_size=32,
            program_config=pcfg,
            final_state_output=out,
            **kwargs,
        )


# ---------------------------------------------------------------------------------------------------------
# qkv_causal_conv1d_silu conv_state_output
# ---------------------------------------------------------------------------------------------------------


def _conv_inputs(device, seq, seed, x_mem):
    torch.manual_seed(seed)
    C = sum(CONV_WIDTHS)
    x = torch.randn(1, seq, C).to(torch.bfloat16)
    hist = torch.randn(1, 3, C).to(torch.bfloat16)
    taps = [(0.5 * torch.randn(1, 1, C)).to(torch.bfloat16) for _ in range(4)]
    x_tt = _dev(x, ttnn.bfloat16, device, _mem(x_mem))
    taps_tt = [_dev(t, ttnn.bfloat16, device) for t in taps]
    return x, hist, x_tt, taps_tt


def _run_conv(x_tt, history, taps_tt, block_tiles=4, conv_state_output=None):
    kwargs = {} if conv_state_output is None else {"conv_state_output": conv_state_output}
    return ttnn.experimental.kda.qkv_causal_conv1d_silu(
        x_tt,
        history,
        *taps_tt,
        *CONV_WIDTHS,
        program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=32 * block_tiles),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        return_conv_state=True,
        **kwargs,
    )


def _assert_qkv_equal(outs, refs):
    for name, a, b in zip("qkv", outs[:3], refs[:3]):
        assert torch.equal(ttnn.to_torch(a), b), f"{name} differs"


@pytest.mark.parametrize("block_tiles", [4, 8], ids=["B4", "B8"])
@pytest.mark.parametrize("x_mem", ["dram", "l1"])
@pytest.mark.parametrize("seq", [32, 96, 2048, 4096], ids=lambda s: f"T{s}")
def test_conv_state_output(device, seq, x_mem, block_tiles):
    C = sum(CONV_WIDTHS)
    x, hist, x_tt, taps_tt = _conv_inputs(device, seq, seed=11, x_mem=x_mem)
    hist_tt = _dev(hist, ttnn.bfloat16, device)

    # Returned-tensor path (reference): q, k, v, new_state (padded, incl. the zero rows 3-31).
    ref = _run_conv(x_tt, hist_tt, taps_tt, block_tiles)
    ref_qkv = [ttnn.to_torch(t) for t in ref[:3]]
    ref_ns = _padded(ref[3])
    assert torch.equal(ref_ns[:, :3, :], x[:, seq - 3 :, :])

    # Separate pre-allocated output (non-zero garbage first, to prove every byte is written).
    out = _dev(torch.full((1, 3, C), 7.0), ttnn.bfloat16, device)
    outs = _run_conv(x_tt, hist_tt, taps_tt, block_tiles, conv_state_output=out)
    assert outs[3].buffer_address() == out.buffer_address(), "new_state must be the caller's buffer"
    _assert_qkv_equal(outs, ref_qkv)
    assert torch.equal(_padded(out), ref_ns)
    assert torch.equal(ttnn.to_torch(hist_tt), hist), "history must be untouched"

    # history=None with an output (non-aliased).
    ref_none = _run_conv(x_tt, None, taps_tt, block_tiles)
    out_none = _dev(torch.full((1, 3, C), 5.0), ttnn.bfloat16, device)
    outs = _run_conv(x_tt, None, taps_tt, block_tiles, conv_state_output=out_none)
    assert outs[3].buffer_address() == out_none.buffer_address()
    _assert_qkv_equal(outs, [ttnn.to_torch(t) for t in ref_none[:3]])
    assert torch.equal(_padded(out_none), _padded(ref_none[3]))

    # In place: history and conv_state_output are the same (persistent) buffer.
    st = _dev(hist, ttnn.bfloat16, device)
    outs = _run_conv(x_tt, st, taps_tt, block_tiles, conv_state_output=st)
    assert outs[3].buffer_address() == st.buffer_address()
    _assert_qkv_equal(outs, ref_qkv)
    assert torch.equal(_padded(st), ref_ns)

    # Second in-place call (cache hit) chains from the carried state: history = x[T-3:].
    ref2 = _run_conv(x_tt, _dev(x[:, seq - 3 :, :], ttnn.bfloat16, device), taps_tt, block_tiles)
    outs = _run_conv(x_tt, st, taps_tt, block_tiles, conv_state_output=st)
    _assert_qkv_equal(outs, [ttnn.to_torch(t) for t in ref2[:3]])
    assert torch.equal(_padded(st), _padded(ref2[3]))

    # Non-aliased again after the aliased entry (same specs): each buffer is bound to its own slot.
    out2 = _dev(torch.full((1, 3, C), 3.0), ttnn.bfloat16, device)
    outs = _run_conv(x_tt, hist_tt, taps_tt, block_tiles, conv_state_output=out2)
    assert outs[3].buffer_address() == out2.buffer_address()
    _assert_qkv_equal(outs, ref_qkv)
    assert torch.equal(_padded(out2), ref_ns)
    assert torch.equal(ttnn.to_torch(hist_tt), hist)


@pytest.mark.parametrize("case", ["no_return_state", "shape", "fp32"])
def test_conv_state_output_rejects(device, case, expect_error):
    C = sum(CONV_WIDTHS)
    x, hist, x_tt, taps_tt = _conv_inputs(device, 64, seed=1, x_mem="dram")
    hist_tt = _dev(hist, ttnn.bfloat16, device)
    out = _dev(torch.zeros(1, 3, C), ttnn.bfloat16, device)
    kwargs = dict(return_conv_state=True, conv_state_output=out)
    if case == "no_return_state":
        kwargs["return_conv_state"] = False
    elif case == "shape":
        kwargs["conv_state_output"] = _dev(torch.zeros(1, 3, C // 2), ttnn.bfloat16, device)
    else:
        kwargs["conv_state_output"] = _dev(torch.zeros(1, 3, C), ttnn.float32, device)
    with expect_error(RuntimeError, "conv_state_output"):
        ttnn.experimental.kda.qkv_causal_conv1d_silu(x_tt, hist_tt, *taps_tt, *CONV_WIDTHS, **kwargs)


def test_conv_tiled_plan_inplace_stage():
    base = ttnn._ttnn.operations.experimental.kda.qkv_causal_conv1d_silu_tiled_program_plan(
        2048, *CONV_WIDTHS, grid_x=13, grid_y=10, channel_chunk_size=128, return_conv_state=True
    )
    inplace = ttnn._ttnn.operations.experimental.kda.qkv_causal_conv1d_silu_tiled_program_plan(
        2048,
        *CONV_WIDTHS,
        grid_x=13,
        grid_y=10,
        channel_chunk_size=128,
        return_conv_state=True,
        conv_state_inplace=True,
    )
    sb, si = base["scratchpad"], inplace["scratchpad"]
    assert sb["stage_bytes"] == 0 and not base["conv_state_inplace"]
    assert inplace["conv_state_inplace"]
    assert si["stage_offset"] == si["state_offset"] + si["state_bytes"]  # the reader derives stage = state + tile
    assert si["stage_offset"] % 64 == 0 and si["stage_bytes"] == 4 * 256
    assert si["bytes"] == sb["bytes"] + si["stage_bytes"]
    assert base["step_start"] == inplace["step_start"] and base["step_count"] == inplace["step_count"]
