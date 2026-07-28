# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC tests for streaming flow encoder + CFM (Stage 3.2).

Validates:
  1. Streaming encoder mu PCC ≥ 0.99 vs reference streaming output
  2. Streaming encoder with context (finalize=False) produces valid output
  3. Chunk mask correctness (block-lower-triangular pattern)

Run:
  pytest models/demos/cosyvoice/tests/pcc/test_streaming.py -v
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

from models.common.utility_functions import comp_pcc

DEMO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = DEMO_ROOT / "model_data" / "golden" / "flow"
FLOW_PT = DEMO_ROOT / "model_data" / "cosyvoice2-0.5B" / "flow.pt"
CV_SRC = DEMO_ROOT / "model_data" / "CosyVoice_src"
MATCHA = CV_SRC / "third_party" / "Matcha-TTS"
CKPT_DIR = DEMO_ROOT / "model_data" / "cosyvoice2-0.5B"

if "pyworld" not in sys.modules:
    _stub = types.ModuleType("pyworld")
    for _n in (
        "wave_to_world",
        "world_to_wave",
        "pythonworld",
        "dio",
        "stft",
        "harvest",
        "cheaptrick",
        "d4c",
        "star",
        "vocoder",
    ):
        setattr(_stub, _n, lambda *a, **k: None)
    sys.modules["pyworld"] = _stub

sys.path.insert(0, str(CV_SRC))
sys.path.append(str(MATCHA))

MODES = ["zero_shot", "cross_lingual", "instruct2", "sft"]
PCC_THRESHOLD = 0.99


@pytest.fixture(scope="module")
def flow_model():
    from models.demos.cosyvoice.tt.flow.flow_matching import FlowEncoderModel
    from models.demos.cosyvoice.tt.flow.weights import load_flow_weights

    components = load_flow_weights(str(FLOW_PT))
    model = FlowEncoderModel(components)
    model.eval()
    return model


@pytest.fixture(scope="module")
def reference_flow():
    from hyperpyyaml import load_hyperpyyaml

    yaml_path = CKPT_DIR / "cosyvoice2.yaml"
    with open(yaml_path, "r") as f:
        configs = load_hyperpyyaml(f, overrides={"llm": None, "hift": None})
    model = configs["flow"]
    sd = torch.load(str(FLOW_PT), map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def test_chunk_mask_pattern():
    """Verify subsequent_chunk_mask produces correct block-lower-triangular pattern."""
    from models.demos.cosyvoice.tt.flow.encoder import subsequent_chunk_mask

    mask = subsequent_chunk_mask(8, 4)
    expected = torch.tensor(
        [
            [True, True, True, True, False, False, False, False],
            [True, True, True, True, False, False, False, False],
            [True, True, True, True, False, False, False, False],
            [True, True, True, True, False, False, False, False],
            [True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True],
        ]
    )
    assert torch.equal(mask, expected), f"Chunk mask mismatch:\n{mask}"


def test_chunk_mask_size25():
    """Verify chunk mask with size=25 matches reference."""
    from cosyvoice.utils.mask import subsequent_chunk_mask as ref_chunk_mask

    from models.demos.cosyvoice.tt.flow.encoder import subsequent_chunk_mask

    for size in [50, 75, 100, 128]:
        ours = subsequent_chunk_mask(size, 25)
        ref = ref_chunk_mask(size, 25, -1)
        assert torch.equal(ours, ref), f"Mismatch at size={size}"


@pytest.mark.parametrize("mode", MODES)
def test_streaming_encoder_mu_pcc(flow_model, reference_flow, mode):
    """Streaming encoder mu PCC ≥ 0.99 vs reference streaming encoder (finalize=True)."""
    golden_path = GOLDEN_DIR / f"{mode}.pt"
    if not golden_path.exists():
        pytest.skip(f"Golden fixture not found: {golden_path}")

    g = torch.load(str(golden_path), map_location="cpu", weights_only=True)

    token = g["token"]
    token_len = g["token_len"]
    prompt_token = g["prompt_token"]
    prompt_token_len = g["prompt_token_len"]
    prompt_feat = g["prompt_feat"]
    prompt_feat_len = g["prompt_feat_len"]
    embedding = g["embedding"]

    with torch.no_grad():
        ref_mu = _reference_encoder_mu(
            reference_flow, token, token_len, prompt_token, prompt_token_len, streaming=True, finalize=True
        )

    with torch.no_grad():
        mu, spks, conds = flow_model(
            token,
            token_len,
            prompt_token,
            prompt_token_len,
            prompt_feat,
            prompt_feat_len,
            embedding,
            streaming=True,
            finalize=True,
        )

    assert mu.shape == ref_mu.shape, f"Shape mismatch: {mu.shape} vs {ref_mu.shape}"
    passing, msg = comp_pcc(ref_mu, mu, PCC_THRESHOLD)
    assert passing, f"[{mode}] streaming mu PCC failed: {msg}"


def _reference_encoder_mu(ref_flow, token, token_len, prompt_token, prompt_token_len, streaming, finalize):
    """Extract mu from reference flow encoder (without running CFM)."""
    import torch.nn.functional as F
    from cosyvoice.utils.mask import make_pad_mask

    embedding = torch.zeros(1, 192)
    embedding = F.normalize(embedding, dim=1)
    spks = ref_flow.spk_embed_affine_layer(embedding)

    token_cat = torch.concat([prompt_token, token], dim=1)
    token_len_cat = prompt_token_len + token_len
    mask = (~make_pad_mask(token_len_cat)).unsqueeze(-1).to(spks.dtype)
    token_emb = ref_flow.input_embedding(torch.clamp(token_cat, min=0)) * mask

    if finalize:
        h, _ = ref_flow.encoder(token_emb, token_len_cat, streaming=streaming)
    else:
        pre_la = ref_flow.pre_lookahead_len
        ctx = token_emb[:, -pre_la:]
        tok = token_emb[:, :-pre_la]
        h, _ = ref_flow.encoder(tok, token_len_cat, context=ctx, streaming=streaming)

    h = ref_flow.encoder_proj(h)
    return h.transpose(1, 2).contiguous()


@pytest.mark.parametrize("mode", ["zero_shot"])
def test_streaming_encoder_context_pcc(flow_model, reference_flow, mode):
    """Streaming encoder with context (finalize=False) mu PCC ≥ 0.99 vs reference."""
    golden_path = GOLDEN_DIR / f"{mode}.pt"
    if not golden_path.exists():
        pytest.skip(f"Golden fixture not found: {golden_path}")

    g = torch.load(str(golden_path), map_location="cpu", weights_only=True)

    token = g["token"]
    token_len = g["token_len"]
    prompt_token = g["prompt_token"]
    prompt_token_len = g["prompt_token_len"]
    prompt_feat = g["prompt_feat"]
    prompt_feat_len = g["prompt_feat_len"]
    embedding = g["embedding"]

    pre_lookahead = 3
    chunk_gen_len = 25 + pre_lookahead
    if token.shape[1] < chunk_gen_len:
        pytest.skip("Token sequence too short for chunk test")

    chunk_token = token[:, :chunk_gen_len]
    chunk_token_len = torch.tensor([chunk_gen_len], dtype=torch.int32)

    with torch.no_grad():
        ref_mu = _reference_encoder_mu(
            reference_flow,
            chunk_token,
            chunk_token_len,
            prompt_token,
            prompt_token_len,
            streaming=True,
            finalize=False,
        )

    with torch.no_grad():
        mu, spks, conds = flow_model(
            chunk_token,
            chunk_token_len,
            prompt_token,
            prompt_token_len,
            prompt_feat,
            prompt_feat_len,
            embedding,
            streaming=True,
            finalize=False,
        )

    assert mu.shape == ref_mu.shape, f"Shape mismatch: {mu.shape} vs {ref_mu.shape}"
    passing, msg = comp_pcc(ref_mu, mu, PCC_THRESHOLD)
    assert passing, f"[{mode}] streaming context mu PCC failed: {msg}"


def test_non_streaming_unchanged(flow_model):
    """Non-streaming path still produces PCC=1.0 vs golden (regression check)."""
    golden_path = GOLDEN_DIR / "zero_shot.pt"
    if not golden_path.exists():
        pytest.skip("Golden fixture not found")

    g = torch.load(str(golden_path), map_location="cpu", weights_only=True)

    with torch.no_grad():
        mu, spks, conds = flow_model(
            g["token"],
            g["token_len"],
            g["prompt_token"],
            g["prompt_token_len"],
            g["prompt_feat"],
            g["prompt_feat_len"],
            g["embedding"],
        )

    passing, msg = comp_pcc(g["mu"], mu, PCC_THRESHOLD)
    assert passing, f"Non-streaming mu PCC regression: {msg}"
