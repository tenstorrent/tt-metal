# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PCC test for the dots.ocr Qwen2 language-model TTNN assembly.

Loads the seed-0 reference golden produced by
``reference/functional.py::language_model_forward`` (full Qwen2ForCausalLM run
at a REDUCED layer count -- the golden's config carries ``num_layers`` (2) vs
``full_num_hidden_layers`` (28) to keep the golden small): embed_tokens ->
N x decoder_layer (GQA 12q/2kv, QKV bias, 1D RoPE theta 1e6, causal) ->
final RMSNorm (eps 1e-6) -> lm_head (untied hidden->vocab). Runs
:class:`TtLanguageModel` on the open p150 (blackhole) device and asserts
``comp_pcc > 0.99`` against the golden logits.

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
_spec = importlib.util.spec_from_file_location("dots_tt_language_model", os.path.join(_TT_DIR, "language_model.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TtLanguageModel = _mod.TtLanguageModel

GOLDEN_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "reference",
    "golden",
    "language_model.pt",
)


def _run_pcc(device) -> float:
    golden = torch.load(GOLDEN_PATH, map_location="cpu", weights_only=False)
    input_ids = golden["input"].to(torch.int64)  # [1, 64]
    ref_output = golden["output"].to(torch.float32)  # [1, 64, vocab]
    state_dict = {k: v.to(torch.float32) for k, v in golden["state_dict"].items()}
    cfg = golden["config"]

    _, seq_len = input_ids.shape
    num_layers = int(cfg["num_layers"])
    num_heads = int(cfg["num_attention_heads"])
    num_kv_heads = int(cfg["num_key_value_heads"])
    head_dim = int(cfg["head_dim"])
    rope_theta = float(cfg["rope_theta"])
    eps = float(cfg["rms_norm_eps"])
    bias = bool(cfg["attention_bias"])

    tt_model = TtLanguageModel(
        device=device,
        state_dict=state_dict,
        num_layers=num_layers,
        seq_len=seq_len,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rope_theta=rope_theta,
        eps=eps,
        bias=bias,
    )

    # ttnn.embedding consumes row-major uint32 indices.
    tt_input = ttnn.from_torch(
        input_ids.to(torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = tt_model(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32).reshape(ref_output.shape)

    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, 0.99)
    print(comp_allclose(ref_output, tt_output_torch))
    print(f"comp_pcc(language_model, {num_layers} layers): passing={passing}, message={pcc_message}")
    msg = str(pcc_message)
    pcc = float(msg.split("PCC:")[-1].strip()) if "PCC:" in msg else float(pcc_message)
    assert passing, f"language_model output does not meet PCC requirement 0.99: {pcc_message}"
    return pcc


def test_tt_language_model(device):
    pcc = _run_pcc(device)
    print(f"language_model PCC = {pcc}")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        _run_pcc(device)
    finally:
        ttnn.close_device(device)
