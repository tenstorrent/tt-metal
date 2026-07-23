# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `l_s_t_m` (hexgrad/Kokoro-82M
`predictor.text_encoder.lstms.0`, a single-layer bidirectional `nn.LSTM`).

Reference: `nn.LSTM(input_size=640, hidden_size=256, batch_first=True,
bidirectional=True)`. Returns `(output, (h_n, c_n))`; the PCC test compares the
`output` tensor `[B, T, 2*hidden]`.

The recurrence (forward + reverse cell-by-cell unroll) lives in the shared
`_lstm_scan.run_bilstm` primitive so the trace-enabling masked fixed-capacity
variant is defined once and reused by the token-axis LSTMs (duration_encoder,
text_encoder) too. All fp32 (HiFi4 matmuls) for a clean PCC.
"""

from __future__ import annotations

import ttnn
from models.demos.kokoro_82m._stubs._lstm_scan import run_bilstm

_DRAM = ttnn.DRAM_MEMORY_CONFIG


HF_MODEL_ID = "hexgrad/Kokoro-82M"


def build(device, torch_module):
    """Bind LSTM gate weights and return a native ttnn forward closure."""
    m = torch_module
    if int(getattr(m, "num_layers", 1)) != 1:
        raise RuntimeError("l_s_t_m native port supports a single LSTM layer only")
    H = int(m.hidden_size)
    bidirectional = bool(m.bidirectional)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def _t(x):
        return ttnn.from_torch(
            x.contiguous().float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_DRAM,
        )

    def _dir(suffix):
        wih = getattr(m, f"weight_ih_l0{suffix}").detach().float()  # [4H, in]
        whh = getattr(m, f"weight_hh_l0{suffix}").detach().float()  # [4H, H]
        bih = getattr(m, f"bias_ih_l0{suffix}").detach().float()
        bhh = getattr(m, f"bias_hh_l0{suffix}").detach().float()
        return {"wih_t": _t(wih.t()), "whh_t": _t(whh.t()), "bias": _t((bih + bhh).reshape(1, 1, -1))}

    fwd = _dir("")
    rev = _dir("_reverse") if bidirectional else None

    def forward(x, *args, **kwargs):
        if not isinstance(x, ttnn.Tensor):
            x = _t(x)
        if x.get_dtype() != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        # x: [B, T, in]
        return run_bilstm(device, compute_config, x, fwd, rev, H)

    return forward


def l_s_t_m(*args, **kwargs):
    raise RuntimeError(
        "l_s_t_m requires build(device, torch_module) to bind trained weights; " "the bare callable has no parameters."
    )
