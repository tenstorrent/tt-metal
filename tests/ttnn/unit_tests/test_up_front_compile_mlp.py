# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Better validation of Tier-1 up-front precompile on a real model layer.

Drives the actual Falcon-7B MLP module (models/demos/ttnn_falcon7b TtFalconMLP:
two ttnn.linear with a fused gelu) through collect -> parallel compile -> warm
forward, and checks PCC vs a torch reference. This exercises matmul + fused
activation (not just eltwise), so it validates that the device-op funnel hook
captures real model programs and that a post-precompile run is correct.

Offline: builds the module with a small random-weight FalconConfig (no HF
download). Weights are oriented the same way the real preprocessor does
(transpose, see models/demos/ttnn_falcon7b/tt/common.py).

Run on a FRESH cache to make the parallel compile do real work, e.g.:
    TT_METAL_CACHE=/tmp/upfront_mlp_$$ scripts/run_safe_pytest.sh \
        tests/ttnn/unit_tests/test_up_front_compile_mlp.py
"""

import time
from types import SimpleNamespace

import torch
import transformers

import ttnn
from models.demos.ttnn_falcon7b.tt.falcon_mlp import TtFalconMLP
from models.demos.ttnn_falcon7b.tt.model_config import get_model_config
from tests.ttnn.utils_for_testing import assert_with_pcc


def _build_falcon_mlp(device, hidden=2048, seq=128):
    """Return (tt_mlp, tt_input, torch_output) for a small random-weight Falcon MLP."""
    torch.manual_seed(0)
    cfg = transformers.FalconConfig(hidden_size=hidden, num_attention_heads=32, num_hidden_layers=1, bias=False)
    torch_mlp = transformers.models.falcon.modeling_falcon.FalconMLP(cfg).eval()

    torch_in = (torch.rand(1, 1, seq, hidden) * 2) - 1
    with torch.no_grad():
        torch_out = torch_mlp(torch_in)

    model_config = get_model_config("BFLOAT16-DRAM")

    def _w(linear):
        # ttnn.linear expects [in, out]; torch Linear stores [out, in] -> transpose.
        return ttnn.from_torch(
            linear.weight.T.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

    parameters = SimpleNamespace(
        dense_h_to_4h=SimpleNamespace(weight=_w(torch_mlp.dense_h_to_4h)),
        dense_4h_to_h=SimpleNamespace(weight=_w(torch_mlp.dense_4h_to_h)),
    )
    tt_mlp = TtFalconMLP(model_config, parameters)
    tt_in = ttnn.from_torch(torch_in, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return tt_mlp, tt_in, torch_out


def test_falcon_mlp_up_front_compile(device):
    tt_mlp, tt_in, torch_out = _build_falcon_mlp(device)

    # --- Phase 1: collect (NO_DISPATCH) — capture the MLP's programs, run nothing ---
    ttnn.graph.up_front_clear()
    ttnn.graph.up_front_begin_collect()
    try:
        tt_mlp(tt_in)
    finally:
        ttnn.graph.up_front_end_collect()
    n_collected = ttnn.graph.up_front_num_collected()
    n_unique = ttnn.graph.up_front_num_unique()
    print(f"\nMLP collect: {n_collected} ops -> {n_unique} unique programs")
    assert n_collected >= 2, "expected at least the two dense linears to be captured"

    # --- Phase 2: parallel compile — warm the on-disk kernel cache ---
    t0 = time.perf_counter()
    num_programs, num_errors, workers, wall = ttnn.graph.up_front_compile(device, 4)
    print(
        f"MLP parallel compile: {num_programs} programs in {wall:.2f}s "
        f"(workers={workers}, errors={num_errors}); call wall {time.perf_counter() - t0:.2f}s"
    )
    assert num_errors == 0, "parallel compile reported errors"
    assert num_programs >= 1

    # --- Phase 3: warm forward — must be correct vs the torch reference ---
    t1 = time.perf_counter()
    out = tt_mlp(tt_in)
    ttnn.synchronize_device(device)
    print(f"MLP warm forward: {(time.perf_counter() - t1) * 1000:.1f} ms")

    passed, pcc = assert_with_pcc(torch_out, ttnn.to_torch(out).to(torch_out.dtype), 0.99)
    print(f"MLP warm-forward PCC vs torch = {pcc}")


def test_falcon_mlp_cold_baseline(device):
    """A/B baseline: NO precompile. Time the first (cold) forward — it JIT-compiles
    inline, serially. Run this in its OWN fresh-cache invocation and compare its
    first-forward time against the up-front path's (compile up-front + ~ms forward):

        TT_METAL_CACHE=/tmp/a_$$ run_safe_pytest ... -k cold_baseline
        TT_METAL_CACHE=/tmp/b_$$ run_safe_pytest ... -k up_front_compile
    """
    tt_mlp, tt_in, torch_out = _build_falcon_mlp(device)

    t0 = time.perf_counter()
    out = tt_mlp(tt_in)
    ttnn.synchronize_device(device)
    print(f"MLP cold-inline first forward: {time.perf_counter() - t0:.2f}s")

    passed, pcc = assert_with_pcc(torch_out, ttnn.to_torch(out).to(torch_out.dtype), 0.99)
    print(f"MLP cold-baseline PCC vs torch = {pcc}")
