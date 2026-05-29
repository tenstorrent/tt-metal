# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PCC test for the dots.ocr LM token-embedding TTNN block.

Loads the reference golden produced by
``reference/functional.py::embedding_forward`` (vocab=151936, hidden=1536),
runs :class:`TtEmbedding` on the open p150 (blackhole) device, and asserts
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
_spec = importlib.util.spec_from_file_location("dots_tt_embedding", os.path.join(_TT_DIR, "embedding.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TtEmbedding = _mod.TtEmbedding

GOLDEN_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "reference",
    "golden",
    "embedding.pt",
)


def _run_pcc(device) -> float:
    golden = torch.load(GOLDEN_PATH, map_location="cpu")
    input_ids = golden["input"].to(torch.int64)  # [1, 128]
    weight = golden["weight"].to(torch.float32)  # [151936, 1536]
    ref_output = golden["output"].to(torch.float32)  # [1, 128, 1536]

    tt_emb = TtEmbedding(device=device, weight=weight)

    # ttnn.embedding consumes row-major uint32 indices.
    tt_input = ttnn.from_torch(
        input_ids.to(torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = tt_emb(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32).reshape(ref_output.shape)

    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, 0.99)
    print(comp_allclose(ref_output, tt_output_torch))
    print(f"comp_pcc(embedding): passing={passing}, message={pcc_message}")
    msg = str(pcc_message)
    pcc = float(msg.split("PCC:")[-1].strip()) if "PCC:" in msg else float(pcc_message)
    assert passing, f"embedding output does not meet PCC requirement 0.99: {pcc_message}"
    return pcc


def test_tt_embedding(device):
    pcc = _run_pcc(device)
    print(f"embedding PCC = {pcc}")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        _run_pcc(device)
    finally:
        ttnn.close_device(device)
