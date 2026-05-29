# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PCC test for the dots.ocr DotsVisionTransformer (vision tower) TTNN assembly.

Loads the seed-0 reference golden produced by
``reference/functional.py::vision_tower_forward`` (full DotsVisionTransformer run
at a REDUCED layer count -- the golden's config carries ``num_layers`` (2) vs
``full_num_hidden_layers`` (42) to keep the golden small): patch_embed ->
N x vision_block (bidirectional 2D-RoPE attn over cu_seqlens) -> post_trunk
RMSNorm (eps 1e-5) -> patch_merger (LayerNorm + GELU MLP, merge 2x2).

The patch_embed Conv2d+RMSNorm is the documented host-resident boundary and is
run on host (TtVisionTower.patch_embed); its [num_patches, embed_dim] output is
uploaded to the device tower. Runs :class:`TtVisionTower` on the open p150
(blackhole) device and asserts ``comp_pcc > 0.99`` against the golden output.

Run as a pytest (uses the shared ``device`` fixture) or as a standalone script
that opens/closes its own device.
"""
import importlib.util
import os

import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc

# The model dir name (rednote_hilab_dots.ocr) contains a dot, so the tt package
# cannot be imported via the normal dotted module path. Load by file path.
_TT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "tt"))
_spec = importlib.util.spec_from_file_location("dots_tt_vision_tower", os.path.join(_TT_DIR, "vision_tower.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TtVisionTower = _mod.TtVisionTower

GOLDEN_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "reference",
    "golden",
    "vision_tower.pt",
)


def _run_pcc(device) -> float:
    golden = torch.load(GOLDEN_PATH, map_location="cpu", weights_only=False)
    pixel_values = golden["input"].to(torch.float32)  # [num_patches, ch*tp*ps*ps]
    grid_thw = torch.as_tensor(golden["grid_thw"])  # [num_images, 3]
    ref_output = golden["output"].to(torch.float32)  # [num_patches // merge**2, hidden]
    state_dict = {k: v.to(torch.float32) for k, v in golden["state_dict"].items()}
    cfg = golden["config"]

    num_layers = int(cfg["num_layers"])
    embed_dim = int(cfg["embed_dim"])
    num_heads = int(cfg["num_heads"])
    num_channels = int(cfg["num_channels"])
    temporal_patch_size = int(cfg["temporal_patch_size"])
    patch_size = int(cfg["patch_size"])
    spatial_merge_size = int(cfg["spatial_merge_size"])
    rms_norm_eps = float(cfg["rms_norm_eps"])
    ln_eps = float(cfg["ln_eps"])
    post_norm = bool(cfg["post_norm"])

    tt_tower = TtVisionTower(
        device=device,
        state_dict=state_dict,
        grid_thw=grid_thw,
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_channels=num_channels,
        temporal_patch_size=temporal_patch_size,
        patch_size=patch_size,
        spatial_merge_size=spatial_merge_size,
        rms_norm_eps=rms_norm_eps,
        ln_eps=ln_eps,
        post_norm=post_norm,
    )

    # Host-resident patch_embed (documented boundary) -> device input tokens.
    hidden_states = tt_tower.patch_embed(pixel_values)  # [num_patches, embed_dim]

    tt_input = ttnn.from_torch(
        hidden_states,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = tt_tower(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32).reshape(ref_output.shape)

    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, 0.99)
    print(comp_allclose(ref_output, tt_output_torch))
    print(f"comp_pcc(vision_tower): passing={passing}, message={pcc_message}")
    msg = str(pcc_message)
    pcc = float(msg.split("PCC:")[-1].strip()) if "PCC:" in msg else float(pcc_message)
    assert passing, f"vision_tower output does not meet PCC requirement 0.99: {pcc_message}"
    return pcc


def test_tt_vision_tower(device):
    pcc = _run_pcc(device)
    print(f"vision_tower PCC = {pcc}")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        _run_pcc(device)
    finally:
        ttnn.close_device(device)
