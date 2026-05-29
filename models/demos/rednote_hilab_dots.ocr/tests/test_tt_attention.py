# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PCC test for the dots.ocr Qwen2 LM self-attention TTNN block.

Loads the seed-0 reference golden produced by
``reference/functional.py::attention_forward`` (hidden 1536, GQA 12 query / 2 KV
heads, head_dim 128, QKV bias, o_proj no bias, 1D RoPE theta 1e6, causal eager),
runs :class:`TtAttention` on the open p150 (blackhole) device, and asserts
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
_spec = importlib.util.spec_from_file_location("dots_tt_attention", os.path.join(_TT_DIR, "attention.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TtAttention = _mod.TtAttention

GOLDEN_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "reference",
    "golden",
    "attention.pt",
)


def _run_pcc(device) -> float:
    golden = torch.load(GOLDEN_PATH, map_location="cpu", weights_only=False)
    torch_input = golden["input"].to(torch.float32)  # [1, 128, 1536]
    ref_output = golden["output"].to(torch.float32)  # [1, 128, 1536]
    state_dict = golden["state_dict"]
    cos = golden["cos"].to(torch.float32)  # [1, 128, 128]
    sin = golden["sin"].to(torch.float32)  # [1, 128, 128]
    attention_mask = golden["attention_mask"].to(torch.float32)  # [1, 1, 128, 128]
    cfg = golden["config"]

    _, seq_len, hidden = torch_input.shape
    num_heads = int(cfg["num_attention_heads"])
    num_kv_heads = int(cfg["num_key_value_heads"])
    head_dim = int(cfg["head_dim"])

    tt_attn = TtAttention(
        device=device,
        q_weight=state_dict["q_proj.weight"].to(torch.float32),
        k_weight=state_dict["k_proj.weight"].to(torch.float32),
        v_weight=state_dict["v_proj.weight"].to(torch.float32),
        q_bias=state_dict["q_proj.bias"].to(torch.float32),
        k_bias=state_dict["k_proj.bias"].to(torch.float32),
        v_bias=state_dict["v_proj.bias"].to(torch.float32),
        o_weight=state_dict["o_proj.weight"].to(torch.float32),
        cos=cos.reshape(seq_len, head_dim),
        sin=sin.reshape(seq_len, head_dim),
        attention_mask=attention_mask.reshape(seq_len, seq_len),
        seq_len=seq_len,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    tt_input = ttnn.from_torch(
        torch_input.reshape(seq_len, hidden),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = tt_attn(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32).reshape(ref_output.shape)

    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, 0.99)
    print(comp_allclose(ref_output, tt_output_torch))
    print(f"comp_pcc(attention): passing={passing}, message={pcc_message}")
    msg = str(pcc_message)
    pcc = float(msg.split("PCC:")[-1].strip()) if "PCC:" in msg else float(pcc_message)
    assert passing, f"attention output does not meet PCC requirement 0.99: {pcc_message}"
    return pcc


def test_tt_attention(device):
    pcc = _run_pcc(device)
    print(f"attention PCC = {pcc}")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        _run_pcc(device)
    finally:
        ttnn.close_device(device)
