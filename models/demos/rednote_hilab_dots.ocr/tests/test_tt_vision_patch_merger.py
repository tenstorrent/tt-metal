# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PCC test for the dots.ocr vision PatchMerger TTNN block.

Loads the seed-0 reference golden produced by
``reference/functional.py::vision_patch_merger_forward`` (LayerNorm eps 1e-6 with
bias -> view(-1, 6144) -> Linear(6144,6144) -> GELU -> Linear(6144,1536), all
biased), runs :class:`TtVisionPatchMerger` on the open p150 (blackhole) device,
and asserts ``comp_pcc > 0.99`` against the golden.

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
_spec = importlib.util.spec_from_file_location(
    "dots_tt_vision_patch_merger", os.path.join(_TT_DIR, "vision_patch_merger.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TtVisionPatchMerger = _mod.TtVisionPatchMerger

GOLDEN_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "reference",
    "golden",
    "vision_patch_merger.pt",
)


def _run_pcc(device) -> float:
    golden = torch.load(GOLDEN_PATH, map_location="cpu", weights_only=False)
    torch_input = golden["input"].to(torch.float32)  # [64, 1536]
    ref_output = golden["output"].to(torch.float32)  # [16, 1536]
    state_dict = golden["state_dict"]
    cfg = golden["config"]

    tt_merger = TtVisionPatchMerger(
        device=device,
        ln_weight=state_dict["ln_q.weight"].to(torch.float32),
        ln_bias=state_dict["ln_q.bias"].to(torch.float32),
        fc1_weight=state_dict["mlp.0.weight"].to(torch.float32),
        fc1_bias=state_dict["mlp.0.bias"].to(torch.float32),
        fc2_weight=state_dict["mlp.2.weight"].to(torch.float32),
        fc2_bias=state_dict["mlp.2.bias"].to(torch.float32),
        context_dim=cfg["context_dim"],
        spatial_merge_size=cfg["spatial_merge_size"],
        ln_eps=cfg["ln_eps"],
    )

    tt_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = tt_merger(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32).reshape(ref_output.shape)

    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, 0.99)
    print(comp_allclose(ref_output, tt_output_torch))
    print(f"comp_pcc(vision_patch_merger): passing={passing}, message={pcc_message}")
    msg = str(pcc_message)
    pcc = float(msg.split("PCC:")[-1].strip()) if "PCC:" in msg else float(pcc_message)
    assert passing, f"vision_patch_merger output does not meet PCC requirement 0.99: {pcc_message}"
    return pcc


def test_tt_vision_patch_merger(device):
    pcc = _run_pcc(device)
    print(f"vision_patch_merger PCC = {pcc}")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        _run_pcc(device)
    finally:
        ttnn.close_device(device)
