# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Card-free regression guards for the Qwen3.6 gated-attention optimizations.

Timing measurements are one-offs: they are taken once on a card, quoted in a
commit message, and then nothing stops a later change from silently undoing
them. These tests make the *structural* properties the optimizations depend on
permanent and checkable with no Tenstorrent device at all.

Three kinds of guard, in decreasing order of strength:

1. **Op-set assertions** (exact, deterministic). `ttnn.graph` captures which
   device operations a chain actually dispatches. "The concat-heads fusion
   removed NLPConcatHeads" is then a fact the test enforces, not a claim in a
   changelog. This is strictly stronger than any timing threshold: it cannot
   flake, and it fails the moment someone reverts the fusion.

2. **Peak-L1 assertions.** `extract_peak_L1_memory_usage` catches the failure
   mode that actually bit this work twice -- bf16 at q/k_chunk=256 wanting
   1,676,160 B against a 1,572,864 B limit, and decode's
   max_cores_per_head_batch=32 wanting ~1.92 MB. Both were discovered by
   crashing. A budget assertion finds them before the crash.

3. **A ttsim work proxy** (see `test_ttsim_ranks_work_removal`). ttsim executes
   a deterministic instruction stream with no pipeline, NoC or DRAM latency
   model, so its wall time is a noise-free proxy for *instructions retired* --
   not for time on silicon.

   Measured boundary of that proxy, both against known hardware answers:

     change that removes work (concat-heads fusion)  ttsim -11.8%   hw -7.6%   ranks it
     pure memory placement    (L1 vs DRAM output)    ttsim  +0.5%   hw -9 us   blind

   So it is usable for the remaining GDN work, which is largely instruction
   count (DST batching, redundant reconfig removal, dispatch elimination), and
   useless for anything whose win is DRAM latency. Use it as a directional
   guard, never as a substitute for a Tracy capture.
"""

import math
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn

QWEN36_NH = 6
QWEN36_NKV = 1
QWEN36_HD = 256

# Blackhole worker L1, from tt_metal/soc_descriptors/blackhole_140_arch.yaml.
L1_LIMIT_BYTES = 1572864


def _inputs(device, nh=2, nkv=1, s=128, d=QWEN36_HD, seed=0):
    torch.manual_seed(seed)
    mk = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return (
        mk(torch.randn(1, nh, s, d)),
        mk(torch.randn(1, nkv, s, d)),
        mk(torch.randn(1, nkv, s, d)),
        mk(torch.randn(1, 1, s, nh * d)),
    )


def _capture_chain(device, *, fused, nh=2, s=128, d=QWEN36_HD, out_mc=None):
    """Run the gated-attention chain under graph capture. Returns (device_op_names, peak_L1)."""
    q, k, v, gate = _inputs(device, nh=nh, s=s, d=d)
    grid = device.compute_with_storage_grid_size()
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        q_chunk_size=32,
        k_chunk_size=32,
        exp_approx_mode=True,
    )
    mc = out_mc if out_mc is not None else ttnn.DRAM_MEMORY_CONFIG

    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    attn = ttnn.transformer.scaled_dot_product_attention(
        q, k, v, is_causal=True, scale=1.0 / math.sqrt(d), program_config=pc, memory_config=mc, fuse_concat_heads=fused
    )
    if not fused:
        concat = ttnn.experimental.nlp_concat_heads(attn, memory_config=mc)
        ttnn.deallocate(attn)
        attn = concat
    gated = ttnn.multiply(
        attn, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID], memory_config=ttnn.DRAM_MEMORY_CONFIG
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
    peak_l1 = ttnn.graph.extract_peak_L1_memory_usage(captured)

    ttnn.deallocate(attn)
    ttnn.deallocate(gated)
    return device_ops, peak_l1


# ---------------------------------------------------------------------------
# 1. Op-set guards
# ---------------------------------------------------------------------------


def test_fusion_removes_the_concat_heads_dispatch(device):
    """The whole point of fuse_concat_heads is that one device op stops existing.

    Asserted structurally rather than by timing: this cannot flake, and it fails
    immediately if the fusion is reverted or silently stops applying (e.g. a
    guard in the device op starts rejecting the flag and it falls back).
    """
    unfused_ops, _ = _capture_chain(device, fused=False)
    fused_ops, _ = _capture_chain(device, fused=True)

    assert "NLPConcatHeadsDeviceOperation" in unfused_ops, f"baseline should concat heads; got {unfused_ops}"
    assert "NLPConcatHeadsDeviceOperation" not in fused_ops, f"fusion did not remove the concat op; got {fused_ops}"
    assert len(fused_ops) == len(unfused_ops) - 1, f"expected exactly one fewer op: {unfused_ops} -> {fused_ops}"
    # The remaining two are the ones that must survive.
    assert "SDPAOperation" in fused_ops and "BinaryNgDeviceOperation" in fused_ops, fused_ops


def test_gate_is_a_single_dispatch(device):
    """sigmoid is folded into the multiply as an operand activation, so the gate
    must cost exactly one binary op and no standalone unary."""
    ops, _ = _capture_chain(device, fused=True)
    assert "UnaryDeviceOperation" not in ops, f"a standalone sigmoid reappeared: {ops}"
    assert sum(1 for o in ops if o == "BinaryNgDeviceOperation") == 1, ops


# ---------------------------------------------------------------------------
# 2. L1 budget guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nh", [2, QWEN36_NH], ids=["gqa2", "qwen36"])
def test_chain_stays_within_l1(device, nh):
    """Two of this work's failures were L1 overflows found by crashing: bf16 at
    q/k_chunk=256 (1,676,160 B) and decode at max_cores_per_head_batch=32
    (~1.92 MB). Both against a 1,572,864 B limit. Assert the budget instead."""
    _, peak = _capture_chain(device, fused=True, nh=nh)
    logger.info(f"PEAK_L1 nh={nh} bytes={peak} ({peak / L1_LIMIT_BYTES * 100:.1f}% of limit)")
    assert peak < L1_LIMIT_BYTES, f"peak L1 {peak} exceeds the {L1_LIMIT_BYTES} B worker limit"


# ---------------------------------------------------------------------------
# 3. ttsim work proxy
# ---------------------------------------------------------------------------


def _time_chain(device, fused, reps=5):
    for _ in range(2):  # warm the program cache; first call includes JIT
        _capture_chain(device, fused=fused)
    ttnn.synchronize_device(device)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _capture_chain(device, fused=fused)
        ttnn.synchronize_device(device)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


@pytest.mark.parametrize("device_params", [{}], indirect=False)
def test_ttsim_ranks_work_removal(device, device_params):
    """ttsim wall time is a proxy for instructions retired, so a change that
    removes a whole device op must show up as less simulated work.

    Deliberately loose (>2%): this guards against the fusion silently ceasing to
    apply, not against small regressions. For anything finer, or for any change
    whose win is memory latency, use hardware -- see the module docstring for the
    measured boundary of this proxy.
    """
    unfused = _time_chain(device, fused=False)
    fused = _time_chain(device, fused=True)
    delta = (fused / unfused - 1) * 100
    logger.info(f"TTSIM_WORK unfused={unfused * 1000:.1f}ms fused={fused * 1000:.1f}ms delta={delta:+.1f}%")
    assert delta < -2.0, f"fusion should reduce simulated work; measured {delta:+.1f}%"
