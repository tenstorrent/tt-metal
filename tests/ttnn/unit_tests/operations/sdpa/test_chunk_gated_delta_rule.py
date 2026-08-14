# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Op-level harness for `ttnn.transformer.chunk_gated_delta_rule` (Qwen3.6 GDN prefill core).

The fused chunked gated-delta-rule op *is* the GDN prefill inner loop -- WY
preprocessing, the intra-chunk Woodbury inverse and the sequential inter-chunk
scan -- in one device op. It is the next kernel to be optimized, and until this
file existed it had **no ttnn-level test at all**: the only coverage was
model-level (`models/demos/blackhole/qwen36/tests/test_gdn_tp.py`), which needs
HF weights and a 4-card mesh, and which checks the op against *another ttnn
path* (the seq adapter) rather than against math.

This file closes that gap with a harness that:

* needs no Tenstorrent card -- the full Qwen3.6 per-device GDN geometry
  (12 value heads / 4 key heads, Dk=Dv=128, chunk_size=32) at T=128 completes
  in a few seconds under ttsim,
* compares against the pure-torch FLA reference in
  `models/experimental/gated_attention_gated_deltanet/torch_functional/delta_rule_ops.py`
  -- an *independent* golden, not another ttnn implementation,
* covers both `initial_state=None` and a supplied non-zero state. That axis is
  load-bearing: the op always materializes a state buffer (it fills zeros when
  the caller passes none) and a planned optimization changes exactly that path,
  so a regression there would otherwise only surface in multi-bucket prefill,
* records a deterministic structural baseline (dispatched device ops + peak L1)
  via `ttnn.graph`, so later kernel changes can be diffed against it with no
  card and no timing noise.

Op contract, as measured here rather than as documented:

* `eye`/`tril`/`ones`/`masks` are genuinely optional. Omitting them (as this
  file does throughout) works: the op builds them itself. They exist only so
  traced callers can own them, because building them does a host upload, which
  is illegal under trace capture.
* `use_qk_l2norm=True` is rejected by a TT_FATAL at the top level. GDN *does*
  L2-normalize q/k; the model adapter does it on host before the call.
* q/k/v are cast to bfloat16 inside the op whatever is passed in (g/beta/state
  stay fp32). That is what sets the PCC ceiling here, not the algorithm.
* `o` comes back ROW_MAJOR as [B, T, HV, V] unless `output_head_major=True`.
* There are two implementations behind the op, selected by the `QWEN_GDN_PHASED`
  environment variable (default on): a phase-split prep+scan pair and a
  monolithic single-kernel op, documented as computing the same math. Both are
  parametrized here, because "same math" is exactly the kind of claim that
  decays silently.

Sizing warning, learned the hard way. Do NOT shrink the head dim to make these
cheaper: under ttsim the phased (default) path completes at Dk=Dv=128 in ~1-2 s,
but never completes at Dk=Dv=64 or 32 -- >2 min with no progress at
B1/T64/HV3/chunk32, where the monolithic path on the same input takes 2 s. That
was reproduced on the 2026-08-14 03:25 build for HV in {1, 3, 12}, so it tracks
the head dim, not the head count. It has not been checked on hardware and may
well be a simulator-only artifact, but it makes small-D shapes useless here.
The real per-device geometry is cheap enough anyway, so this file just uses it.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.gated_attention_gated_deltanet.torch_functional.delta_rule_ops import (
    chunk_gated_delta_rule as torch_chunk_gated_delta_rule,
)
from models.experimental.gated_attention_gated_deltanet.torch_functional.delta_rule_ops import l2_norm
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_and_get_pcc

# Opening a ttsim device costs more than most of the tests in here; keep one.
pytestmark = pytest.mark.use_module_device

# Qwen3.6-27B GDN, per device at TP=4 (models/demos/blackhole/qwen36/tt/gdn).
QWEN36_HV = 12  # value heads
QWEN36_H = 4  # key/query heads -> GVA ratio 3
QWEN36_D = 128  # linear_key_head_dim == value_head_dim
FUSED_CHUNK_SIZE = 32  # what models/demos/blackhole/qwen36/tt/gdn/fused_chunk.py pins

# Blackhole worker L1, from tt_metal/soc_descriptors/blackhole_140_arch.yaml.
L1_LIMIT_BYTES = 1572864


# ---------------------------------------------------------------------------
# Inputs / reference
# ---------------------------------------------------------------------------


def gdn_inputs(*, b, t, h, hv, dk, dv, seed=1234, with_state=False):
    """GDN-shaped random inputs in the op's (FLA) layout.

    g is a log-space decay, so it must be <= 0 -- the model produces it as
    -softplus(a) * A_log.exp(). beta is a sigmoid, so it lives in (0, 1).
    Feeding plain randn for either makes the state diverge across chunks and
    turns any PCC into noise, which is why the generator is not just randn.
    """
    torch.manual_seed(seed)
    q = torch.randn(b, t, h, dk)
    k = torch.randn(b, t, h, dk)
    v = torch.randn(b, t, hv, dv)
    g = -torch.nn.functional.softplus(torch.randn(b, t, hv))
    beta = torch.sigmoid(torch.randn(b, t, hv))
    state = torch.randn(b, hv, dk, dv) * 0.1 if with_state else None
    return q, k, v, g, beta, state


def torch_reference(q, k, v, g, beta, *, chunk_size, initial_state, scale=None):
    """The FLA torch golden, with the GVA head expansion the op does internally.

    The op head-splits q/k to [B*H, T, D] and then `repeat_interleave(G, dim=0)`,
    so value head hv reads key head hv // G. `repeat_interleave` on the head axis
    of the [B, T, H, D] host tensor is the same mapping.
    """
    ratio = v.shape[2] // q.shape[2]
    q_e = q.repeat_interleave(ratio, dim=2) if ratio > 1 else q
    k_e = k.repeat_interleave(ratio, dim=2) if ratio > 1 else k
    return torch_chunk_gated_delta_rule(
        q_e,
        k_e,
        v,
        g,
        beta,
        chunk_size=chunk_size,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
    )


def to_tt(device, x, dtype):
    return ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def run_chunk_gated_delta_rule(device, *, b, t, h, hv, dk, dv, chunk_size, with_state=False, seed=1234, scale=None):
    """Run the op and its golden. Returns (o, final_state, o_ref, final_state_ref) as torch tensors."""
    q, k, v, g, beta, state = gdn_inputs(b=b, t=t, h=h, hv=hv, dk=dk, dv=dv, seed=seed, with_state=with_state)

    # q/k/v are cast to bf16 inside the op regardless, so hand them over already
    # bf16 -- host and device then see the same values going in. g/beta are fp32.
    o, final_state = ttnn.transformer.chunk_gated_delta_rule(
        to_tt(device, q, ttnn.bfloat16),
        to_tt(device, k, ttnn.bfloat16),
        to_tt(device, v, ttnn.bfloat16),
        to_tt(device, g, ttnn.float32),
        to_tt(device, beta, ttnn.float32),
        scale=scale,
        initial_state=to_tt(device, state, ttnn.float32) if state is not None else None,
        output_final_state=True,
        chunk_size=chunk_size,
    )
    o_ref, state_ref = torch_reference(q, k, v, g, beta, chunk_size=chunk_size, initial_state=state, scale=scale)
    return ttnn.to_torch(o).float(), ttnn.to_torch(final_state).float(), o_ref, state_ref


# The real per-device geometry at TP=4. It is cheap enough under ttsim (a few
# seconds at T=128) that there is no reason to test a scaled-down stand-in.
QWEN36_SHAPE = dict(b=1, h=QWEN36_H, hv=QWEN36_HV, dk=QWEN36_D, dv=QWEN36_D, chunk_size=FUSED_CHUNK_SIZE)


@pytest.fixture(params=["phased", "monolithic"])
def gdn_path(request, monkeypatch):
    """Select the implementation behind the op.

    `QWEN_GDN_PHASED` is read fresh on every call (chunk_gated_delta_rule.cpp
    reads it per invocation, not into a static), so it can be toggled per test.
    Both are pinned to the same golden here rather than to each other.
    """
    monkeypatch.setenv("QWEN_GDN_PHASED", "1" if request.param == "phased" else "0")
    return request.param


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("t", [64, 128], ids=["t64", "t128"])
@pytest.mark.parametrize("with_state", [False, True], ids=["no_state", "with_state"])
def test_chunk_gated_delta_rule(device, gdn_path, t, with_state):
    """PCC against the torch FLA reference at the real Qwen3.6 GDN geometry.

    The `with_state` axis is not decoration: `initial_state=None` and a supplied
    state take different routes through the op (the None case builds a device-side
    zero buffer), and carrying a state in is what batched/bucketed prefill does.
    """
    o, final_state, o_ref, state_ref = run_chunk_gated_delta_rule(device, t=t, with_state=with_state, **QWEN36_SHAPE)

    passed_o, msg_o, pcc_o = comp_and_get_pcc(o_ref, o, 0.99)
    passed_s, msg_s, pcc_s = comp_and_get_pcc(state_ref, final_state, 0.99)
    logger.info(f"GDN_PCC path={gdn_path} T={t} with_state={with_state} o={pcc_o:.6f} final_state={pcc_s:.6f}")
    assert not torch.isnan(o).any(), "NaN in output"
    assert o.shape == o_ref.shape, f"o shape {tuple(o.shape)} != reference {tuple(o_ref.shape)}"
    assert passed_o, f"o PCC below threshold: {msg_o}"
    assert passed_s, f"final_state PCC below threshold: {msg_s}"


def test_chunk_size_is_an_internal_tiling_choice(device, gdn_path):
    """The answer must not depend on chunk_size; only the tiling does.

    The torch reference obeys that exactly (chunk 32 vs 64 agree to PCC 1-3e-11,
    asserted below so this test cannot be fooled by a broken golden).
    `fused_chunk.py` pins the op to 32 anyway, with a comment that at 64 "the
    bottom-right 32x32 sub-block can be ill-conditioned enough that the fp32
    block inverse loses precision on some chunks". So: 32 is asserted, because it
    is what production runs; 64 is measured and logged, because its accuracy is a
    property that has moved before and is worth watching rather than trusting.
    """
    shape = dict(QWEN36_SHAPE)
    shape.pop("chunk_size")

    o32, s32, ref32, refs32 = run_chunk_gated_delta_rule(device, t=64, chunk_size=32, **shape)
    o64, s64, ref64, refs64 = run_chunk_gated_delta_rule(device, t=64, chunk_size=64, **shape)

    _, _, pcc32 = comp_and_get_pcc(ref32, o32, 0.99)
    _, _, pcc64 = comp_and_get_pcc(ref64, o64, 0.99)
    _, _, pcc32_s = comp_and_get_pcc(refs32, s32, 0.99)
    _, _, pcc64_s = comp_and_get_pcc(refs64, s64, 0.99)
    logger.info(f"GDN_CHUNK_SIZE path={gdn_path} chunk32 o={pcc32:.6f} state={pcc32_s:.6f}")
    logger.info(f"GDN_CHUNK_SIZE path={gdn_path} chunk64 o={pcc64:.6f} state={pcc64_s:.6f}")
    # Guards the claim above: any chunk-size sensitivity seen here is the op's.
    _, _, ref_invariance = comp_and_get_pcc(ref32, ref64, 0.99)
    assert ref_invariance > 0.9999, f"the torch reference should not depend on chunk_size: {ref_invariance}"
    assert pcc32 >= 0.99, f"chunk_size=32 (what production uses) regressed: o PCC {pcc32}"
    assert pcc32_s >= 0.99, f"chunk_size=32 (what production uses) regressed: state PCC {pcc32_s}"


def test_l2norm_is_a_host_responsibility(device, expect_error):
    """`use_qk_l2norm=True` is rejected at the top level; callers pre-normalize.

    Worth pinning down: GDN *does* L2-normalize q/k, so an op that takes a
    `use_qk_l2norm` argument reads as though it were supported, and silently
    skipping the normalization would be a subtle accuracy bug rather than a
    crash. The second half checks the workaround the model adapter uses --
    normalize on host -- actually reproduces the reference's own l2-normed path.
    """
    t = 64
    h, hv, d = QWEN36_H, QWEN36_HV, QWEN36_D
    q, k, v, g, beta, _ = gdn_inputs(b=1, t=t, h=h, hv=hv, dk=d, dv=d)
    args = (
        to_tt(device, q, ttnn.bfloat16),
        to_tt(device, k, ttnn.bfloat16),
        to_tt(device, v, ttnn.bfloat16),
        to_tt(device, g, ttnn.float32),
        to_tt(device, beta, ttnn.float32),
    )

    with expect_error(RuntimeError, "use_qk_l2norm not yet supported"):
        ttnn.transformer.chunk_gated_delta_rule(*args, chunk_size=FUSED_CHUNK_SIZE, use_qk_l2norm=True)

    o, _ = ttnn.transformer.chunk_gated_delta_rule(
        to_tt(device, l2_norm(q, dim=-1), ttnn.bfloat16),
        to_tt(device, l2_norm(k, dim=-1), ttnn.bfloat16),
        args[2],
        args[3],
        args[4],
        chunk_size=FUSED_CHUNK_SIZE,
        output_final_state=True,
    )
    ratio = hv // h
    o_ref, _ = torch_chunk_gated_delta_rule(
        q.repeat_interleave(ratio, dim=2),
        k.repeat_interleave(ratio, dim=2),
        v,
        g,
        beta,
        chunk_size=FUSED_CHUNK_SIZE,
        use_qk_l2norm=True,
    )
    passed, msg, pcc = comp_and_get_pcc(o_ref, ttnn.to_torch(o).float(), 0.99)
    logger.info(f"GDN_PCC host_l2norm o={pcc:.6f}")
    assert passed, f"host-normalized q/k disagrees with the l2-normed reference: {msg}"


# ---------------------------------------------------------------------------
# Structural baseline (card-free, deterministic)
#
# ttsim wall time is not a performance model and hardware timings are one-offs
# quoted in commit messages. What is checkable anywhere is *which* device ops a
# call dispatches and how much L1 it peaks at. Recording both gives the coming
# kernel work something exact to diff against -- "the composite still costs this
# many dispatches", "the new CB layout did not blow the L1 budget" -- with no
# card and no flakiness.
# ---------------------------------------------------------------------------


def capture_chunk_gated_delta_rule(device, *, t, with_state, **shape):
    """Run the op under ttnn.graph capture. Returns (device_op_names, peak_L1_bytes).

    The program cache is cleared first, and that is load-bearing rather than
    hygiene: circular-buffer allocations are only emitted into the graph on a
    program-cache *miss*, so a capture that hits the cache reports peak L1 = 0.
    Without the clear, the number this test prints would depend on which tests
    ran before it -- and a 0 would sail through the budget assertion below.
    """
    kwargs = dict(QWEN36_SHAPE)
    kwargs.update(shape)
    q, k, v, g, beta, state = gdn_inputs(
        b=kwargs["b"], t=t, h=kwargs["h"], hv=kwargs["hv"], dk=kwargs["dk"], dv=kwargs["dv"], with_state=with_state
    )
    tt_args = (
        to_tt(device, q, ttnn.bfloat16),
        to_tt(device, k, ttnn.bfloat16),
        to_tt(device, v, ttnn.bfloat16),
        to_tt(device, g, ttnn.float32),
        to_tt(device, beta, ttnn.float32),
    )
    tt_state = to_tt(device, state, ttnn.float32) if state is not None else None

    device.disable_and_clear_program_cache()
    device.enable_program_cache()

    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    ttnn.transformer.chunk_gated_delta_rule(
        *tt_args,
        initial_state=tt_state,
        output_final_state=True,
        chunk_size=kwargs["chunk_size"],
    )
    captured = ttnn.graph.end_graph_capture()

    names = sorted(
        {
            n["params"]["name"]
            for n in captured
            if n.get("node_type") == "function_start" and "name" in n.get("params", {})
        }
    )
    device_ops = [n for n in names if "DeviceOperation" in n or n.endswith("Operation")]
    return device_ops, ttnn.graph.extract_peak_L1_memory_usage(captured)


@pytest.mark.parametrize("with_state", [False, True], ids=["no_state", "with_state"])
def test_graph_capture_baseline(device, gdn_path, with_state):
    """Record the dispatched device ops and the peak L1 of one op call.

    The op set is logged, not asserted: it is a baseline to diff against, and
    changing it is the point of the coming work. Peak L1 *is* asserted, because
    overflowing the worker budget is a crash rather than a regression -- the
    failure mode this workstream has already hit twice on the attention side.

    The budget check is not incidental to the GDN kernels: attributing every
    circular-buffer allocation in the capture to its enclosing device op puts
    the entire peak on ChunkGdnPrepOperation (1,003,520 B of 1,003,520 B at the
    Qwen geometry, chunk 32, T=64; the scan's CBs total 196,608 B and every
    data-movement op in the composite is under 50 KB). So this number moves when
    the prep kernel's CB layout moves, which is the point.
    """
    device_ops, peak_l1 = capture_chunk_gated_delta_rule(device, t=64, with_state=with_state)
    logger.info(f"GDN_GRAPH path={gdn_path} with_state={with_state} device_ops={device_ops}")
    logger.info(
        f"GDN_PEAK_L1 path={gdn_path} with_state={with_state} bytes={peak_l1} "
        f"({peak_l1 / L1_LIMIT_BYTES * 100:.1f}% of limit)"
    )
    assert device_ops, "graph capture recorded no device operations"
    # A zero here means the capture saw no circular-buffer allocations at all, i.e.
    # the budget assertion below is measuring nothing. Fail instead of passing blind.
    assert peak_l1 > 0, "graph capture reported no L1 usage; the budget check would be vacuous"
    assert peak_l1 < L1_LIMIT_BYTES, f"peak L1 {peak_l1} exceeds the {L1_LIMIT_BYTES} B worker limit"
