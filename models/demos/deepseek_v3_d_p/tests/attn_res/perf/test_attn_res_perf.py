# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 AttnRes performance: a boundary layer and an ordinary layer, in isolation, untraced and traced.

Galaxy (SP8xTP4) runs K3's prefill chunk and is gated. LoudBox (SP2xTP4, 1280 tokens) runs the same programs at
Galaxy's TP width and 640 rows per chip, so every per-chip shape matches Galaxy's and only the SP axis is shorter; it
reports without gating until it has CI targets. SP4xTP4 runs only on a Galaxy carved by hand (`SUBTORUS_4X4_ENV`) and
reports.

There is no QuietBox arm: QuietBox 2 has no layout AttnRes can run at a Galaxy-like shape.
  * TP2: at a 3584-wide shard `attn_res_gather_softmax` needs 1658880 B of L1 per core against 1572864 B.
  * TP4 on 2D torus-X: the four chips are wired 2x2, so a 1x4 TORUS_X request comes up unwrapped.
  * TP4 on a 1D ring: `attn_res_gather_softmax` requires a 2D fabric.

The layers are those of the last pipeline rank (24/24/24/21 over 4 Galaxies), which inherits sealed snapshots from
the upstream ranks:

  * **Boundary layer** (the rank's first, layer 72): the most expensive AttnRes layer of the stack. Its programs are
    the 25-site `inter_block` over the inherited set with the live stream sealed onto it (the read that straddles the
    rank boundary is its first, closed, site) and the post-attention read.
  * **Ordinary layer** (73-80, averaged): two reads, each settling the module write the walk deferred into it. 85 of
    the 93 layers are this.

Each step issues exactly the programs `TtAttnResWalk` issues for those layers, on fixed inputs: `inter_block` and
`merge` borrow their inputs and return new tensors, so a step repeats and replays unchanged, and its eager and traced
outputs are comparable. The walk's own bookkeeping is host-side Python around those calls and is not timed.

Measured as tests/measure does: untraced fastest of 10 after 10 warm-up, traced one replay, both with the CPU clock
held; the traced output must match the eager one. The ordinary step runs layers 73-80 and is divided by 8.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.attn_res import EPS
from models.demos.deepseek_v3_d_p.tests.attn_res.model.harness import (
    EXACT_BOX,
    HIDDEN_SIZE,
    L1_SMALL_SIZE,
    PER_CHIP_TOKENS,
    generator,
    place,
    random_hidden,
    random_queries,
)
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric2d_device_params,
    torus_xy_device_params,
    tp_axis_is_wrapped,
)
from models.demos.deepseek_v3_d_p.tests.measure.gate import MARGIN, gate
from models.demos.deepseek_v3_d_p.tests.measure.timing import free
from models.demos.deepseek_v3_d_p.tests.measure.trace import measure_host
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res import TtAttnRes
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res_stream import BLOCK_SIZE, _block_sites
from models.demos.deepseek_v3_d_p.utils.prefill_summary_utils import emit_perf_metrics
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import require_high_power
from tt_py_test_utils_common.perf.cpu_clock import hold_full_cpu_clock

pytestmark = [run_for_blackhole(), pytest.mark.perf, pytest.mark.timeout(1800)]

# The first measured layer: the last rank's first, under the 24/24/24/21 split of K3's 93 layers.
LAYER = 72
RANK_LAYERS = 93 - LAYER
INHERITED_SNAPSHOTS = LAYER // BLOCK_SIZE
# Ordinary layers timed in one step, after the boundary layer; the step's time is divided by this.
ORDINARY_LAYERS = 8
_GALAXY = "SP8xTP4"
# Host-time targets (ms) per (layer, mode) on Galaxy, as tests/measure measures them: the mean of 3 runs on one
# Galaxy (off the CI pool; samples within 1% of each other). The untraced targets are then scaled by the speedup of
# running AttnRes as one program per mesh (#66039: ordinary 175 -> 108 us, boundary 400 -> 380 us on one Galaxy).
# LoudBox SP2xTP4 and SP4xTP4 report until a CI run gives their targets.
_TARGETS_MS = {
    ("boundary", "untraced"): 0.459,
    ("boundary", "traced"): 0.446,
    ("ordinary", "untraced"): 0.106,
    ("ordinary", "traced"): 0.128,
}
# A layout's prefill chunk where it is not PER_CHIP_TOKENS a chip: a 4x4 stage takes the same chunk as the 8x4 mesh,
# so each chip holds twice the rows.
_CHUNK_TOKENS = {"SP4xTP4": PER_CHIP_TOKENS * 8}
_TRACE_REGION_SIZE = 128 * 1024 * 1024


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(l1_small_size=L1_SMALL_SIZE, trace_region_size=_TRACE_REGION_SIZE, **EXACT_BOX),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4-torus-xy",
        ),
        # One 4x4 pipeline stage of SC1, on a Galaxy carved with `SUBTORUS_4X4_ENV`.
        pytest.param(
            (4, 4),
            torus_xy_device_params(l1_small_size=L1_SMALL_SIZE, trace_region_size=_TRACE_REGION_SIZE, **EXACT_BOX),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 4), topology="mesh-4x4"),
            id="SP4xTP4-torus-xy",
        ),
        # Unwrapped 2D, the fabric the AttnRes model suite runs on LoudBox; `attn_res_gather_softmax` needs 2D.
        pytest.param(
            (2, 4),
            fabric2d_device_params(l1_small_size=L1_SMALL_SIZE, trace_region_size=_TRACE_REGION_SIZE, **EXACT_BOX),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="SP2xTP4-fabric-2d",
        ),
        # No QuietBox arm: it has no layout AttnRes can run at a Galaxy-like shape. See the module docstring.
    ],
    indirect=True,
)
@pytest.mark.parametrize("layer", ["boundary", "ordinary"])
def test_attn_res_layer_perf(request, mesh_device, device_params, layer):
    """Gate the host time of one AttnRes layer's programs, untraced and traced, on Galaxy with its TP axis wrapped
    (anything else reports)."""
    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS)
    layout = f"SP{op.sp_factor}xTP{op.tp_factor}"
    if layout == _GALAXY:
        require_high_power(f"kimi_k3 AttnRes {layout} perf")
    num_tokens = _CHUNK_TOKENS.get(layout, PER_CHIP_TOKENS * op.sp_factor)

    owned = []  # every tensor the test allocates outside the step, freed even if the measurement fails

    def own(tensor):
        owned.append(tensor)
        return tensor

    def boundary():
        """open_layer(0) on an inheriting walk, then the post-attention read (nothing deferred yet)."""
        batch = op.inter_block(sealed, [q_pre[0], *sites], num_closed=1)
        try:
            return op.merge(*batch[:3], running_sum, sites[0], site=0)
        finally:
            free(batch)

    def ordinary():
        """Layers 73-80: each a pre- and a post-attention read, settling the deferred write."""
        out = None
        for site in range(1, 1 + 2 * ORDINARY_LAYERS):
            free(out)
            out = None
            out, total = op.merge(*batches[:3], running_sum, sites[site], site=site, pending=pending)
            free(total)
        return out

    clock = hold_full_cpu_clock()
    request.addfinalizer(clock.restore)
    try:
        rng = generator(0)
        q_pre = [own(op.to_query(q)) for q in random_queries(rng, RANK_LAYERS)]
        q_post = [own(op.to_query(q)) for q in random_queries(rng, RANK_LAYERS)]
        q_out = own(op.to_query(random_queries(rng, 1)[0]))
        # The sealed set the boundary layer reads: the inherited snapshots, then the live stream sealed onto them.
        hidden = lambda: own(place(op, random_hidden(rng, num_tokens).unsqueeze(0).unsqueeze(0)))
        inherited = own(place(op, torch.randn(1, INHERITED_SNAPSHOTS, num_tokens, HIDDEN_SIZE, generator=rng)))
        sealed = [inherited, hidden()]
        # Stand-ins for the module outputs the reads fold in: the running sum and the write deferred into it.
        running_sum, pending = hidden(), hidden()
        # The block the rank's first seal opens: layer 72's post-attention read (site 0), then two sites per layer.
        sites = _block_sites(RANK_LAYERS, q_pre, q_post, q_out, BLOCK_SIZE)[0]
        assert len(sites) >= 1 + 2 * ORDINARY_LAYERS
        if layer == "ordinary":
            batches = [own(t) for t in op.inter_block(sealed, sites)[:3]]
        step, per_call = (boundary, 1) if layer == "boundary" else (ordinary, ORDINARY_LAYERS)
        untraced, traced, trace_pcc = measure_host(mesh_device, step)
    finally:
        free(owned)
    untraced = [ms / per_call for ms in untraced]
    traced = [ms / per_call for ms in traced]

    gated = layout == _GALAXY and tp_axis_is_wrapped(mesh_device)
    results = {}
    for mode, samples in (("untraced", untraced), ("traced", traced)):
        target = _TARGETS_MS.get((layer, mode)) if gated else None
        results[mode] = (samples, min(samples), target, MARGIN[mode])
        emit_perf_metrics(
            f"kimi_k3_attn_res_{layer}_layer_{layout}_{mode}",
            f"Kimi-K3 AttnRes {layer} layer perf ({layout}, {mode})",
            {
                "layout": layout,
                "layer": layer,
                "mode": mode,
                "fabric_config": ttnn.get_fabric_config(mesh_device).name,
                "tokens": num_tokens,
                "first_layer": LAYER,
                "layers_per_sample": per_call,
                "inherited_snapshots": INHERITED_SNAPSHOTS,
                "samples_ms": samples,
                "measured_ms": min(samples),
                "target_ms": target,
                "margin_pct": MARGIN[mode] * 100.0,
            },
        )
    gate(f"kimi_k3 AttnRes {layer} layer {layout}", results, trace_pcc)
