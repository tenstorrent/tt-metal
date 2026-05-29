# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PCC test for the dots.ocr vision transformer block (composite) TTNN module.

Loads the seed-0 reference golden produced by
``reference/functional.py::vision_block_forward`` (pre-norm residual:
h = h + attn(norm1(h)); h = h + mlp(norm2(h)); embed_dim 1536, 12 heads,
head_dim 128, intermediate 4224, eps 1e-5, no bias), runs :class:`TtVisionBlock`
on the open p150 (blackhole) device, and asserts ``comp_pcc > 0.99`` against the
golden output.

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
_spec = importlib.util.spec_from_file_location("dots_tt_vision_block", os.path.join(_TT_DIR, "vision_block.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TtVisionBlock = _mod.TtVisionBlock

GOLDEN_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "reference",
    "golden",
    "vision_block.pt",
)


def _run_pcc(device) -> float:
    golden = torch.load(GOLDEN_PATH, map_location="cpu")
    torch_input = golden["input"].to(torch.float32)  # [256, 1536]
    ref_output = golden["output"].to(torch.float32)  # [256, 1536]
    state_dict = golden["state_dict"]
    cu_seqlens = golden["cu_seqlens"]
    rotary_pos_emb = golden["rotary_pos_emb"].to(torch.float32)  # [256, 64]
    cfg = golden["config"]

    seq_length, dim = torch_input.shape
    num_heads = int(cfg["num_heads"])
    head_dim = int(cfg["head_dim"])
    eps = float(cfg["rms_norm_eps"])

    tt_block = TtVisionBlock(
        device=device,
        norm1_weight=state_dict["norm1.weight"].to(torch.float32),
        qkv_weight=state_dict["attn.qkv.weight"].to(torch.float32),
        proj_weight=state_dict["attn.proj.weight"].to(torch.float32),
        norm2_weight=state_dict["norm2.weight"].to(torch.float32),
        fc1_weight=state_dict["mlp.fc1.weight"].to(torch.float32),
        fc3_weight=state_dict["mlp.fc3.weight"].to(torch.float32),
        fc2_weight=state_dict["mlp.fc2.weight"].to(torch.float32),
        rotary_pos_emb=rotary_pos_emb,
        cu_seqlens=cu_seqlens,
        seq_length=seq_length,
        num_heads=num_heads,
        head_dim=head_dim,
        eps=eps,
    )

    tt_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = tt_block(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32).reshape(ref_output.shape)

    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, 0.99)
    print(comp_allclose(ref_output, tt_output_torch))
    print(f"comp_pcc(vision_block): passing={passing}, message={pcc_message}")
    msg = str(pcc_message)
    pcc = float(msg.split("PCC:")[-1].strip()) if "PCC:" in msg else float(pcc_message)
    assert passing, f"vision_block output does not meet PCC requirement 0.99: {pcc_message}"
    return pcc


def test_tt_vision_block(device):
    pcc = _run_pcc(device)
    print(f"vision_block PCC = {pcc}")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        _run_pcc(device)
    finally:
        ttnn.close_device(device)
