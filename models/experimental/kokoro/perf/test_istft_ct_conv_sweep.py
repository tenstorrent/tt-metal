# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Sweep the iSTFT OLA ``conv_transpose2d`` (``_ct_ola_run`` in ``tt_torch_stft.py``) over memory /
program configs to find the fastest strategy that fits L1 across the Kokoro phoneme-length range.

The op: input ``[B=1, H=F, W=1, C=2K=22]`` (NHWC), weight ``[2K, 1, n_fft=20, 1]``, out_channels=1,
stride=(hop=5,1), padding=(n_fft//2=10, 0).  F (= frames) scales with phoneme count:
generator short ~9721, demo ~49321, max-length (510 phonemes) ~138241.

For every (F, config) we run ONE conv_transpose2d, synchronise, record fit/OOM + PCC vs a CPU
``F.conv_transpose2d`` reference, and write a JSON manifest (in execution order, with the device-op
count each config emits) to ``$KOKORO_CT_SWEEP_JSON``.  Run under tracy:

    KOKORO_CT_SWEEP_JSON=ct_sweep.json python -m tracy -r -p -v --op-support-count 10000 -m pytest \
        models/experimental/kokoro/perf/test_istft_ct_conv_sweep.py::test_istft_ct_conv_sweep

then map the Conv2dDeviceOperation rows of the generated CSV (in order) to the manifest configs
with ``_parse_sweep.py`` to get per-config device time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from loguru import logger

import ttnn

from models.experimental.kokoro.tt.tt_torch_stft import preprocess_tt_torch_stft

_N_FFT = 20
_HOP = 5
_WIN = 20
_PAD = _N_FFT // 2
_K = _N_FFT // 2 + 1  # 11
_2K = 2 * _K  # 22

# F values spanning the Kokoro phoneme-length range (frames = har_time_len // hop + 1).
_F_VALUES = [int(v) for v in os.getenv("KOKORO_CT_SWEEP_F", "5001,9721,24305,49321").split(",")]

_SHARD = {
    "height": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    "width": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    "block": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    "auto": None,
}


def _configs_for(n_frames: int) -> list[dict]:
    """Strategy grid for one F.  ``slices`` None => L1 single pass (no dram_slice_config)."""
    ceil4096 = (n_frames + 4096 - 1) // 4096
    grid = []
    for shard in ("height", "width", "block", "auto"):
        for abh in (None, 32):
            # L1 single-pass (no DRAM slicing).
            grid.append({"shard": shard, "abh": abh, "slices": None})
        # DRAM height-sliced variants (only meaningful for the height-sharded reader path).
        if shard == "height":
            grid.append({"shard": shard, "abh": 32, "slices": 1})
            grid.append({"shard": shard, "abh": 32, "slices": ceil4096})
            grid.append({"shard": shard, "abh": None, "slices": ceil4096})
    return grid


def _build_inputs(device, n_frames: int):
    """Return (x_nhwc_rm fp32 [1,1,F,2K], synth_combined weight, torch reference output [1,L_out])."""
    params = preprocess_tt_torch_stft(
        filter_length=_N_FFT, hop_length=_HOP, win_length=_WIN, input_length=n_frames * _HOP, device=device
    )
    torch.manual_seed(0)
    x = torch.randn(1, _2K, n_frames, dtype=torch.float32) * 0.05  # [1, 2K, F]

    # CPU reference: conv_transpose2d with padding=(pad,0) over [1,2K,F,1] -> [1,1,L_out,1].
    w_cpu = ttnn.to_torch(params.synth_combined).float()  # [2K, 1, 20, 1]
    with torch.no_grad():
        ref = torch.nn.functional.conv_transpose2d(
            x.unsqueeze(-1), w_cpu, stride=(_HOP, 1), padding=(_PAD, 0)
        )  # [1, 1, L_out, 1]
    ref = ref.reshape(1, -1)  # [1, L_out]

    x_nhwc = ttnn.from_torch(
        x.permute(0, 2, 1).reshape(1, 1, n_frames, _2K),  # [1,1,F,2K] NHWC
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return params, x_nhwc, ref


def _run_one(device, params, x_nhwc, n_frames: int, cfg: dict):
    """Run one conv_transpose2d config.  Returns (out_torch [1,L_out] or None, n_dev_ops, status)."""
    mc = ttnn.DRAM_MEMORY_CONFIG
    conv_cfg = ttnn.Conv2dConfig(weights_dtype=ttnn.float32)
    conv_cfg.config_tensors_in_dram = True
    conv_cfg.deallocate_activation = False
    conv_cfg.enable_act_double_buffer = False
    if cfg["abh"] is not None:
        conv_cfg.act_block_h_override = int(cfg["abh"])
    if _SHARD[cfg["shard"]] is not None:
        conv_cfg.shard_layout = _SHARD[cfg["shard"]]
    compute_cfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )
    slice_cfg = None
    n_dev_ops = 1
    if cfg["slices"] is not None:
        slice_cfg = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=int(cfg["slices"]))
        n_dev_ops = int(cfg["slices"])
    try:
        y, out_hw = ttnn.conv_transpose2d(
            input_tensor=x_nhwc,
            weight_tensor=params.synth_combined,
            device=device,
            in_channels=_2K,
            out_channels=1,
            batch_size=1,
            input_height=n_frames,
            input_width=1,
            kernel_size=(_N_FFT, 1),
            stride=(_HOP, 1),
            padding=(_PAD, 0),
            output_padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            bias_tensor=None,
            conv_config=conv_cfg,
            compute_config=compute_cfg,
            dram_slice_config=slice_cfg,
            mirror_kernel=True,
            return_output_dim=True,
        )
        ttnn.synchronize_device(device)
        oh, ow = int(out_hw[0]), int(out_hw[1])
        y2 = ttnn.reshape(y, (y.shape[0], oh * ow), memory_config=mc)
        if y2.layout != ttnn.TILE_LAYOUT:
            y2 = ttnn.to_layout(y2, ttnn.TILE_LAYOUT, memory_config=mc)
        out = ttnn.to_torch(y2).float().reshape(1, -1)
        ttnn.deallocate(y)
        ttnn.deallocate(y2)
        return out, n_dev_ops, "ok"
    except RuntimeError as e:
        msg = str(e)
        status = "OOM" if ("L1" in msg or "circular buffer" in msg or "Out of Memory" in msg) else f"ERR:{msg[:40]}"
        return None, n_dev_ops, status


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().double()
    b = b.flatten().double()
    n = min(a.numel(), b.numel())
    a, b = a[:n], b[:n]
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return 1.0 if denom == 0 else float((a @ b).item() / denom)


def test_istft_ct_conv_sweep(device):
    manifest = []
    for n_frames in _F_VALUES:
        params, x_nhwc, ref = _build_inputs(device, n_frames)
        for cfg in _configs_for(n_frames):
            out, n_dev_ops, status = _run_one(device, params, x_nhwc, n_frames, cfg)
            pcc = _pcc(ref, out) if out is not None else None
            label = f"F={n_frames} shard={cfg['shard']} abh={cfg['abh']} slices={cfg['slices']}"
            logger.info(f"[CT-SWEEP] {label} -> {status} pcc={pcc} n_dev_ops={n_dev_ops}")
            manifest.append(
                {
                    "F": n_frames,
                    "shard": cfg["shard"],
                    "abh": cfg["abh"],
                    "slices": cfg["slices"],
                    "status": status,
                    "pcc": pcc,
                    "n_dev_ops": n_dev_ops if status == "ok" else 0,
                }
            )
        ttnn.deallocate(x_nhwc)

    out_path = os.getenv("KOKORO_CT_SWEEP_JSON", "ct_sweep.json")
    Path(out_path).write_text(json.dumps(manifest, indent=2))
    logger.info(f"[CT-SWEEP] wrote manifest to {out_path}")

    # Sanity: every config that ran must be numerically correct.
    for m in manifest:
        if m["status"] == "ok":
            assert m["pcc"] is not None and m["pcc"] > 0.99, f"bad PCC for {m}"
