# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Instrument real model fused SiLU multiplication; timings are not comparable."""

import json
import os
import runpy
from pathlib import Path

from binary_silu_repro import metrics

import ttnn

original = ttnn.multiply
records = []


def host(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def multiply(a, b, *args, **kwargs):
    if not kwargs.get("input_tensor_b_activations") or len(records) >= 3:
        return original(a, b, *args, **kwargs)
    ah, bh = host(a), host(b)
    fused = original(a, b, *args, **kwargs)
    actual = host(fused)
    separate = host(original(a, ttnn.silu(b)))
    records.append(
        {
            "a_range": [float(ah.min()), float(ah.max())],
            "b_range": [float(bh.min()), float(bh.max())],
            "a_shape": list(a.shape),
            "a_padded": list(a.padded_shape),
            "b_padded": list(b.padded_shape),
            "a_dtype": str(a.dtype),
            "b_dtype": str(b.dtype),
            "fused_vs_separate": metrics(separate, actual),
        }
    )
    print("REAL_FUSED_SILU", records[-1], flush=True)
    return fused


if __name__ == "__main__":
    ttnn.multiply = multiply
    os.environ["TAIL_FUSION"] = "both"
    try:
        runpy.run_path(str(Path(__file__).with_name("tail_followup.py")), run_name="__main__")
    finally:
        (Path(__file__).parent / "artifacts/binary_silu_real_inputs.json").write_text(
            json.dumps(records, indent=2) + "\n"
        )
