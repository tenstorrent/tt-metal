# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sliding-window geometry test for the ACE-Step 1.5 DiT (Block 1), **S=256** (20.48 s).

This is the TRAP-1 regression gate. It exists as a separate file because it is the only test
where the window is not a no-op: ACE-Step keeps ``|i - j| <= 128``, so any sequence shorter
than ~130 is fully in-band and a wrong window size scores a perfect PCC. S=256 is the
smallest tile-aligned reference duration where the band actually bites.

Three checks:

1.  ``window_mapping`` — a bare SDPA probe. TTNN's ``sliding_window_size`` is the **total**
    window width, i.e. non-causal it keeps ``|i - j| <= W / 2``
    (``ttnn/cpp/.../sdpa/device/kernels/sliding_window_geometry.hpp``: ``half_window = W / 2``).
    Asserts ``W=256`` matches the ``|i-j| <= 128`` dense reference, **and** that ``W=128``
    (the value straight out of ``config.json``) does *not* — otherwise the trap could
    silently return.
2.  ``window_is_active`` — the correct band must differ measurably from unmasked attention at
    this sequence length, proving the window is doing something.
3.  ``block`` — the full block at S=256, both layer types, against the reference with the
    reference's own ``_create_4d_mask``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.ace_step_v15.tests import dit_reference as R
from models.experimental.ace_step_v15.tests.test_dit_block_pcc import run_dit_block_pcc
from models.experimental.ace_step_v15.tt.ttnn_ace_step_common import (
    AceStepDiTConfig,
    sdpa_compute_config,
    to_device,
    to_host,
)

GOLDEN = str(R.GOLDEN_DIR)
TARGET_PCC = 0.998

SEQ_LEN = R.SEQ_LEN_BANDED  # 256 -> 20.48 s; the window is a no-op below S ~= 130
SEED = 1234

#: A wrong window must score at most this against the correct band. The measured value for
#: W=128 (i.e. |i-j| <= 64 against |i-j| <= 128) is ~0.762.
WRONG_WINDOW_MAX_PCC = 0.95


def _dense_band(seq_len: int, bound: int) -> torch.Tensor:
    """Additive ``[1, 1, S, S]`` mask keeping ``|i - j| <= bound``, ``finfo.min`` elsewhere."""
    idx = torch.arange(seq_len)
    keep = (idx[:, None] - idx[None, :]).abs() <= bound
    mask = torch.full((1, 1, seq_len, seq_len), torch.finfo(torch.float32).min, dtype=torch.float32)
    mask.masked_fill_(keep[None, None], 0.0)
    return mask


def run_window_mapping_pcc(device, *, seq_len: int = SEQ_LEN, verbose: bool = True):
    """Probe ``sliding_window_size`` semantics directly, with no model weights involved."""
    torch.manual_seed(SEED)
    config = AceStepDiTConfig()
    heads, kv_heads, head_dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim

    q = torch.randn(1, heads, seq_len, head_dim) * 0.5
    k = torch.randn(1, kv_heads, seq_len, head_dim) * 0.5
    v = torch.randn(1, kv_heads, seq_len, head_dim) * 0.5
    q_tt, k_tt, v_tt = (to_device(t, device) for t in (q, k, v))

    def torch_sdpa(bound: int | None) -> torch.Tensor:
        mask = None if bound is None else _dense_band(seq_len, bound)
        # TRAP-2 reference side: torch needs enable_gqa to expand 8 kv heads to 16 q heads.
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=config.attention_scale, enable_gqa=True)

    def ttnn_sdpa(window: int | None) -> torch.Tensor:
        out = ttnn.transformer.scaled_dot_product_attention(
            q_tt,
            k_tt,
            v_tt,
            is_causal=False,
            scale=config.attention_scale,
            sliding_window_size=window,
            compute_kernel_config=sdpa_compute_config(device),
        )
        host = to_host(out)
        ttnn.deallocate(out)
        return host

    ref_128 = torch_sdpa(128)  # what ACE-Step wants
    ref_64 = torch_sdpa(64)  # what passing config.sliding_window straight through gives
    ref_none = torch_sdpa(None)

    results: dict[str, float] = {}
    got_256 = ttnn_sdpa(config.sdpa_window_size)  # 256
    got_128 = ttnn_sdpa(config.sliding_window)  # 128 -- the trap
    got_none = ttnn_sdpa(None)

    _, results["W=256 vs |i-j|<=128 (correct)"] = comp_pcc(ref_128, got_256, pcc=0.0)
    _, results["W=128 vs |i-j|<=128 (the trap)"] = comp_pcc(ref_128, got_128, pcc=0.0)
    _, results["W=128 vs |i-j|<=64 (what it is)"] = comp_pcc(ref_64, got_128, pcc=0.0)
    _, results["W=None vs unmasked (control)"] = comp_pcc(ref_none, got_none, pcc=0.0)
    _, results["|i-j|<=128 vs unmasked (band active?)"] = comp_pcc(ref_128, ref_none, pcc=0.0)

    for t in (q_tt, k_tt, v_tt):
        ttnn.deallocate(t)

    if verbose:
        print(f"\n=== SDPA sliding_window_size mapping (S={seq_len}, {heads}q/{kv_heads}kv) ===")
        for name, pcc in results.items():
            print(f"  {name:42s} pcc={float(pcc):.6f}")

    checks = {
        "correct_mapping": float(results["W=256 vs |i-j|<=128 (correct)"]) >= TARGET_PCC,
        "trap_still_detected": float(results["W=128 vs |i-j|<=128 (the trap)"]) < WRONG_WINDOW_MAX_PCC,
        "half_width_confirmed": float(results["W=128 vs |i-j|<=64 (what it is)"]) >= TARGET_PCC,
        "unwindowed_control": float(results["W=None vs unmasked (control)"]) >= TARGET_PCC,
        "band_is_active": float(results["|i-j|<=128 vs unmasked (band active?)"]) < 0.999,
    }
    failures = {name: ok for name, ok in checks.items() if not ok}
    return not failures, results, failures


def run_dit_banded_pcc(device, *, verbose: bool = True):
    ok_map, map_results, map_failures = run_window_mapping_pcc(device, verbose=verbose)

    block_results = {}
    block_failures = {}
    for sliding in (True, False):
        ok, res, fails = run_dit_block_pcc(device, seq_len=SEQ_LEN, sliding=sliding, verbose=verbose)
        key = "sliding_attention" if sliding else "full_attention"
        block_results[key] = res
        if not ok:
            block_failures[key] = fails

    failures = {}
    if map_failures:
        failures["window_mapping"] = map_failures
    if block_failures:
        failures["block"] = block_failures
    return not failures, {"window": map_results, "block": block_results}, failures


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_window_mapping(device):
    passed, results, failures = run_window_mapping_pcc(device)
    assert passed, (
        "sliding-window geometry check failed (see TRAP-1: sliding_window_size is the TOTAL "
        f"width, so pass 2 * config.sliding_window): {failures}; measured {results}"
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("sliding", [True, False], ids=["sliding_attention", "full_attention"])
def test_dit_banded_block_pcc(device, sliding):
    passed, _results, failures = run_dit_block_pcc(device, seq_len=SEQ_LEN, sliding=sliding)
    assert passed, f"banded DiT block PCC below {TARGET_PCC} at S={SEQ_LEN}: {failures}"


if __name__ == "__main__":
    import sys
    import time

    dev = None
    for attempt in range(20):
        try:
            dev = ttnn.open_device(device_id=0, l1_small_size=32768)
            break
        except Exception as err:  # device momentarily busy (shared with other blocks)
            print(f"open_device attempt {attempt} failed ({err}); retrying in 45s")
            time.sleep(45)
    if dev is None:
        print("FAILED could not open device")
        sys.exit(1)
    try:
        ok, _results, fails = run_dit_banded_pcc(dev)
    finally:
        ttnn.close_device(dev)
    print("PASSED" if ok else f"FAILED {fails}")
    sys.exit(0 if ok else 1)
