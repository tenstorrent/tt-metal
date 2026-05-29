# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PCC test for the dots.ocr Qwen2 LM RMSNorm TTNN block.

Loads the seed-0 reference golden produced by
``reference/functional.py::rmsnorm_forward`` (eps=1e-6, dim=1536), runs
:class:`TtRMSNorm` on the open p150 (blackhole) device, and asserts
``comp_pcc > 0.99`` against the golden output.

Run as a pytest (uses the shared ``device`` fixture) or as a standalone script
that opens/closes its own device.
"""
import importlib.util
import os

import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc

# The model dir name (rednote_hilab_dots.ocr) contains a dot, so the tt package
# cannot be imported via the normal dotted module path. Load the block by file path.
_TT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "tt"))
_spec = importlib.util.spec_from_file_location("dots_tt_rmsnorm", os.path.join(_TT_DIR, "rmsnorm.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TtRMSNorm = _mod.TtRMSNorm

GOLDEN_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "reference",
    "golden",
    "rmsnorm.pt",
)


def _run_pcc(device) -> float:
    golden = torch.load(GOLDEN_PATH, map_location="cpu")
    torch_input = golden["input"].to(torch.float32)  # [1, 128, 1536]
    weight = golden["weight"].to(torch.float32)  # [1536]
    ref_output = golden["output"].to(torch.float32)  # [1, 128, 1536]
    eps = float(golden["eps"])
    dim = int(golden["dim"])

    tt_norm = TtRMSNorm(device=device, dim=dim, weight=weight, eps=eps)

    tt_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = tt_norm(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32).reshape(ref_output.shape)

    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, 0.99)
    print(comp_allclose(ref_output, tt_output_torch))
    print(f"comp_pcc(rmsnorm): passing={passing}, message={pcc_message}")
    # comp_pcc returns the PCC either as a numpy float or a "PCC: <x>" string.
    msg = str(pcc_message)
    pcc = float(msg.split("PCC:")[-1].strip()) if "PCC:" in msg else float(pcc_message)
    assert passing, f"rmsnorm output does not meet PCC requirement 0.99: {pcc_message}"
    return pcc


def test_tt_rmsnorm(device):
    pcc = _run_pcc(device)
    print(f"rmsnorm PCC = {pcc}")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        _run_pcc(device)
    finally:
        ttnn.close_device(device)
