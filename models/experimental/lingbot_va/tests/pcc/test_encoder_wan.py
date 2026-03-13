# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PyTest comparison between HuggingFace UMT5EncoderModel and TT-Metal UMT5Encoder.
Uses Pearson Correlation Coefficient (PCC) as the accuracy metric.
"""

import torch
import pytest
import ttnn
from transformers import UMT5EncoderModel

from models.tt_dit.encoders.umt5.model_umt5 import UMT5Config, UMT5Encoder as TTUMT5Encoder
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.common.metrics import compute_pcc


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

CHECKPOINT_PATH = "models/experimental/lingbot_va/reference/checkpoints/text_encoder"
PCC_THRESHOLD = 0.99
BATCH_SIZE = 1
SEQ_LEN = 512


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────


@pytest.fixture(scope="module")
def mesh_device():
    """Create and destroy TT mesh device."""
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    yield device
    ttnn.close_mesh_device(device)


@pytest.fixture(scope="module")
def hf_model():
    """Load HuggingFace UMT5 encoder."""
    model = UMT5EncoderModel.from_pretrained(
        CHECKPOINT_PATH,
        torch_dtype=torch.bfloat16,
    ).to(device="cpu")

    model.eval()
    return model


# ─────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────


@pytest.mark.timeout(0)
def test_umt5_encoder_comparison(mesh_device, hf_model):
    """Compare TT-Metal UMT5Encoder with HuggingFace UMT5EncoderModel."""

    # ── 1. Extract weights ──
    text_weights = {k: v.cpu() for k, v in hf_model.state_dict().items()}

    # ── 2. Random input ──
    torch.manual_seed(42)

    input_ids = torch.randint(
        low=0,
        high=hf_model.config.vocab_size,
        size=(BATCH_SIZE, SEQ_LEN),
        dtype=torch.long,
    )

    attention_mask = torch.ones((BATCH_SIZE, SEQ_LEN), dtype=torch.long)

    # ── 3. HuggingFace forward ──
    with torch.no_grad():
        text_out = hf_model(input_ids=input_ids, attention_mask=attention_mask)

    text_embed = text_out.last_hidden_state.float()

    # ── 4. Parallel config ──
    encoder_parallel_config = EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=1, mesh_axis=1))

    ccl_manager = CCLManager(
        mesh_device=mesh_device,
        num_links=1,
        topology=ttnn.Topology.Linear,
    )

    # ── 5. Build TT config ──
    umt5_config = UMT5Config(
        vocab_size=hf_model.config.vocab_size,
        embed_dim=hf_model.config.d_model,
        ff_dim=hf_model.config.d_ff,
        kv_dim=hf_model.config.d_kv,
        num_heads=hf_model.config.num_heads,
        num_hidden_layers=hf_model.config.num_layers,
        max_prompt_length=SEQ_LEN,
        layer_norm_eps=hf_model.config.layer_norm_epsilon,
        relative_attention_num_buckets=hf_model.config.relative_attention_num_buckets,
        relative_attention_max_distance=hf_model.config.relative_attention_max_distance,
    )

    # ── 6. Create TT encoder ──
    tt_encoder = TTUMT5Encoder(
        config=umt5_config,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=encoder_parallel_config,
    )

    tt_encoder.load_torch_state_dict(text_weights)

    # ── 7. Convert inputs to TTNN ──
    tt_input = ttnn.from_torch(
        input_ids,
        dtype=ttnn.uint32,
        device=mesh_device,
    )

    tt_mask = ttnn.from_torch(
        attention_mask,
        dtype=ttnn.bfloat16,
        device=mesh_device,
    )

    # ── 8. TT forward ──
    tt_out = tt_encoder(tt_input, attention_mask=tt_mask)
    tt_out = tt_out[-1]

    tt_embed = ttnn.to_torch(tt_out).float()

    while tt_embed.dim() > 3:
        tt_embed = tt_embed.squeeze(0)

    # ── 9. Shape check ──
    assert tt_embed.shape == text_embed.shape, f"Shape mismatch: HF={text_embed.shape}, TT={tt_embed.shape}"

    # ── 10. Metrics ──
    pcc = compute_pcc(text_embed, tt_embed)
    max_err = (text_embed - tt_embed).abs().max().item()
    mean_err = (text_embed - tt_embed).abs().mean().item()

    print("\n" + "=" * 55)
    print("UMT5 ENCODER COMPARISON")
    print("=" * 55)
    print(f"PCC                 : {pcc:.6f}")
    print(f"Max absolute error  : {max_err:.6f}")
    print(f"Mean absolute error : {mean_err:.6f}")
    print("=" * 55)

    # ── 11. Assertion ──
    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < threshold {PCC_THRESHOLD}"
