# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attention of block type sliding (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.sliding.attention.test.1): golden is s4096 chunk 1 (start 2048, chunk 2048 > sliding window 1024),
``attn_norm`` [2048, 2816] -> ``attn_out`` [2048, 2816] bf16; the KV prefix [0, 2048) comes from the golden state.
The gated metric stays PCC >= 0.99, but on this golden PCC misses two stateful bugs: ignoring the KV prefix scores
0.9979 and RoPE positions counted from 0 instead of the chunk start score 0.9959 (dropping the window: 0.9872). Both
bugs only touch the first ``sliding_window`` query rows, the only rows that attend into the prefix. Extra checks:
relative L2 over the whole chunk <= 0.03 and over those first rows <= 0.03. CPU measurements (whole / first rows):
reference 0.0019 / 0.0019; bf16 activations + bf16 weights 0.0028; bf16 + bfp8 q/k/v/o weights 0.0055; 1% noise 0.010;
no prefix 0.064 / 0.090; RoPE from 0 0.091 / 0.128; no window 0.16; window 992 instead of 1024 0.0065 (not caught,
the lost keys are nearly irrelevant); interleaved instead of rotate-half RoPE 0.38; missing v norm 16.8.
This test checks the returned attn_out only. The K/V this chunk writes to the state is checked by the state metrics.
"""

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing.component import _step, module_under_test
from models.demos.common.bringup.testing.harness import (
    compare,
    component_golden,
    default_mode,
    device_ctx,
    mesh_parametrize,
    reference_ctx,
    spec,
    threshold,
)

S = spec()
STEP = "attention"
LAYER = 0
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.03  # ||got - want|| / ||want||, whole chunk
MAX_REL_L2_PREFIX_ROWS = 0.03  # same, over the first sliding_window rows (the rows that read the KV prefix)


@mesh_parametrize
def test_component(mesh_device):
    g, c = component_golden(S)
    assert c * g.chunk > 0, "component chunk must start after 0 so attention reads a real KV prefix"
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    st = _step(ref, LAYER, STEP)
    gl = g.layer(c, LAYER)
    inputs = [gl[i].float() if gl[i].is_floating_point() else gl[i] for i in st.inputs]
    want = gl[st.output]
    fn = module_under_test(S, ref, mesh_device, LAYER, STEP)
    out = fn(reference_ctx(ref, LAYER, g, c), device_ctx(LAYER, g, c), *inputs)

    mode = COMPARE or default_mode(want)
    thr = threshold(S, "component") if THRESHOLD is None else THRESHOLD
    _, ok = compare(f"pcc_{STEP}_L{LAYER:02d}", out, want, mode, thr)
    assert ok, "PCC below threshold"

    # Error-size checks (PCC barely sees prefix and RoPE-position bugs). Informational metrics, not in the runner's list.
    got = out.float().reshape(want.shape)
    w = want.float()
    window = int(S.get("checkpoint.config.sliding_window", 1024))
    rows = min(window, w.shape[0])
    rel = ((got - w).norm() / w.norm()).item()
    rel_head = ((got[:rows] - w[:rows]).norm() / w[:rows].norm()).item()
    metrics.record(f"rel_l2_{STEP}_L{LAYER:02d}", rel)
    metrics.record(f"rel_l2_prefix_rows_{STEP}_L{LAYER:02d}", rel_head)
    print(f"rel_l2={rel:.6f} (<= {MAX_REL_L2}) rel_l2_first_{rows}_rows={rel_head:.6f} (<= {MAX_REL_L2_PREFIX_ROWS})")
    assert torch.isfinite(got).all(), "non-finite output"
    assert rel <= MAX_REL_L2, f"relative L2 error {rel:.4f} > {MAX_REL_L2}"
    assert rel_head <= MAX_REL_L2_PREFIX_ROWS, (
        f"relative L2 error on the first {rows} rows {rel_head:.4f} > {MAX_REL_L2_PREFIX_ROWS} "
        "(KV prefix ignored or RoPE positions not offset by the chunk start?)"
    )
