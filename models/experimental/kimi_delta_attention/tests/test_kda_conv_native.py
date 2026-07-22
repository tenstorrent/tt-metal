# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Prototype: native ttnn.conv1d depthwise causal conv + SiLU vs torch ref and vs the composed FIR.
# De-risks perf-improvement #1 (replace the ~57-op composed FIR). Mirrors qwen36 gdn/tp.py::_conv1d_prefill.
# Validates PCC (incl. the bf16 question) + program count at KDA shapes: D=1024 (TP4/chip), D=4096 (TP1).

import pytest
import torch
import torch.nn.functional as F
import ttnn
from loguru import logger

from models.experimental.kimi_delta_attention.tt.ttnn_kda_ops import (
    causal_conv1d_silu_native,
    causal_conv1d_silu_ttnn,
    prepare_conv1d_weight,
)
from models.common.utility_functions import comp_pcc
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program


def _torch_ref(x, w, K):
    """Depthwise causal conv1d + SiLU. x:[1,T,D], w:[D,K] (tap k multiplies x[t-(K-1)+k]). Returns [1,T,D]."""
    D = x.shape[-1]
    xp = F.pad(x.transpose(1, 2), (K - 1, 0))  # left-pad K-1 -> [1,D,T+K-1]
    y = F.conv1d(xp, w.reshape(D, 1, K), groups=D)  # depthwise -> [1,D,T]
    return F.silu(y).transpose(1, 2)  # [1,T,D]


def _native_conv(x_tt, w_th, K, C, device):
    """Exercises the library native path (prepare_conv1d_weight + causal_conv1d_silu_native), single device."""
    w_host = ttnn.from_torch(w_th.reshape(C, 1, 1, K).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    w_prep = prepare_conv1d_weight(w_host, C, K, x_tt.shape[1], device)
    return causal_conv1d_silu_native(x_tt, w_prep, K, C, device)


def _nprog(device, fn):
    ttnn.synchronize_device(device)
    _, recs = profile_realtime_program(device, fn, collect_all=True)
    ids = {r["runtime_id"] for r in recs if r["runtime_id"]}
    us = {}
    for r in recs:
        if r["runtime_id"]:
            us[r["runtime_id"]] = max(us.get(r["runtime_id"], 0), float(r["duration_ns"]))
    return len(ids), sum(us.values()) / 1e3


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("D", [1024, 4096], ids=["D1024_tp4", "D4096_tp1"])
def test_conv_native_pcc(device, D):
    T, K = 640, 4
    torch.manual_seed(0)
    x = torch.randn(1, T, D)
    w = torch.randn(D, K) * 0.3
    y_ref = _torch_ref(x, w, K)

    x_tt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_native = ttnn.to_torch(_native_conv(x_tt, w, K, D, device))
    okn, pccn = comp_pcc(y_ref, y_native, pcc=0.98)
    logger.info(f"[conv_native] D={D} native-vs-torch PCC={pccn}")

    # composed FIR for reference (fp32) — the path we're replacing
    taps = [ttnn.from_torch(w[:, k].reshape(1, 1, D), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device) for k in range(K)]
    y_fir = ttnn.to_torch(causal_conv1d_silu_ttnn(x_tt, taps, K, device))
    okf, pccf = comp_pcc(y_ref, y_fir, pcc=0.98)
    logger.info(f"[conv_native] D={D} composed-FIR-vs-torch PCC={pccf}")

    # program count + device time: native vs composed
    n_native, us_native = _nprog(device, lambda: _native_conv(x_tt, w, K, D, device))
    n_fir, us_fir = _nprog(device, lambda: causal_conv1d_silu_ttnn(x_tt, taps, K, device))
    logger.info(f"[conv_native] D={D} PROGRAMS native={n_native} ({us_native:.0f}us) vs FIR={n_fir} ({us_fir:.0f}us) "
                f"-> {us_fir/us_native:.1f}x faster, {n_fir/n_native:.1f}x fewer")
    assert okn, f"native PCC too low: {pccn}"
