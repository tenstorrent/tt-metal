# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PCC test for the dots.ocr Qwen2 LM rotary-embedding (RoPE) TTNN block.

Loads the seed-0 reference golden produced by
``reference/functional.py::rope_forward`` (theta 1e6, head_dim 128, seq_len 128,
position_ids 0..127), runs :class:`TtRoPE` on the open p150 (blackhole) device,
and asserts ``comp_pcc > 0.99`` for both the cos and sin tables against the
golden.

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
_spec = importlib.util.spec_from_file_location("dots_tt_rope", os.path.join(_TT_DIR, "rope.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TtRoPE = _mod.TtRoPE

GOLDEN_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "reference",
    "golden",
    "rope.pt",
)


def _pcc(ref, got, name):
    passing, pcc_message = comp_pcc(ref, got, 0.99)
    print(comp_allclose(ref, got))
    print(f"comp_pcc(rope/{name}): passing={passing}, message={pcc_message}")
    msg = str(pcc_message)
    pcc = float(msg.split("PCC:")[-1].strip()) if "PCC:" in msg else float(pcc_message)
    assert passing, f"rope {name} does not meet PCC requirement 0.99: {pcc_message}"
    return pcc


def _run_pcc(device) -> float:
    golden = torch.load(GOLDEN_PATH, map_location="cpu", weights_only=False)
    position_ids = golden["position_ids"]  # [1, 128] int64
    ref_cos = golden["cos"].to(torch.float32)  # [1, 128, 128]
    ref_sin = golden["sin"].to(torch.float32)  # [1, 128, 128]
    cfg = golden["config"]
    head_dim = int(cfg["head_dim"])
    rope_theta = float(cfg["rope_theta"])

    tt_rope = TtRoPE(device=device, head_dim=head_dim, rope_theta=rope_theta)

    # Positions as [batch, 1, seq, 1] float so they broadcast against inv_freq.
    pos = position_ids.to(torch.float32).reshape(position_ids.shape[0], 1, position_ids.shape[1], 1)
    tt_pos = ttnn.from_torch(
        pos,
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_cos, tt_sin = tt_rope(tt_pos)
    got_cos = ttnn.to_torch(tt_cos).to(torch.float32).reshape(ref_cos.shape)
    got_sin = ttnn.to_torch(tt_sin).to(torch.float32).reshape(ref_sin.shape)

    pcc_cos = _pcc(ref_cos, got_cos, "cos")
    pcc_sin = _pcc(ref_sin, got_sin, "sin")
    return min(pcc_cos, pcc_sin)


def test_tt_rope(device):
    pcc = _run_pcc(device)
    print(f"rope PCC (min cos/sin) = {pcc}")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        _run_pcc(device)
    finally:
        ttnn.close_device(device)
