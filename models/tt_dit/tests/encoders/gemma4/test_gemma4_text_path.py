# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""End-to-end parity for the LTX-2.5 text path: on-device ``encode`` vs a torch reference.

``test_gemma4_parity.py`` clears the 48-layer encoder, and the 2.3 analogue of this test
(``encoders/gemma3/test_gemma_full.py``) clears the 2.3 projection and connectors. Neither
covers the two stages that carry genuinely new 2.5 weights: the aggregate projection (in the
packed text encoder) and the 258 connector tensors (in the split DiT). Their key names are
identical to 2.3's while the values differ, so a convention change would load silently — this
test is what would catch it.

The reference reuses the ``--full`` Gemma-4 hidden states from ``gen_gemma4_reference.py``
(transformers>=5, out-of-process) and feeds them to diffusers' ``LTX2TextConnectors``, which
is configured from the 2.5 DiT metadata. Every connector field there matches 2.3.
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[6]))

import pytest
import torch
from loguru import logger
from safetensors import safe_open

import ttnn
from models.tt_dit.encoders.gemma4.encoder_pair import Gemma4TokenizerEncoderPair
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.test import line_params_req_exact_devices

REFERENCE_FULL = os.environ.get("GEMMA4_REFERENCE_FULL", "/tmp/g4ref/gemma4_reference_full.safetensors")
PROMPT_FILE = os.environ.get("GEMMA4_PROMPT_FILE", "/tmp/g4ref/guitar_prompt.txt")
TEXT_ENCODER = os.environ.get(
    "GEMMA4_CHECKPOINT",
    os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
)
TRANSFORMER = os.environ.get(
    "LTX25_TRANSFORMER",
    os.path.expanduser(
        "~/.cache/ltx-checkpoints/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    ),
)

PROJECTION_PREFIXES = (
    "text_embedding_projection.video_aggregate_embed.",
    "text_embedding_projection.audio_aggregate_embed.",
)


def _connector_block_subkey(sub: str) -> str:
    """Per-block raw Lightricks key → diffusers connector key: only the block-list name and
    the QK-norm names differ (q_norm/k_norm → norm_q/norm_k); the rest is shared."""
    sub = sub.replace("transformer_1d_blocks.", "transformer_blocks.")
    sub = sub.replace(".attn1.q_norm.", ".attn1.norm_q.")
    sub = sub.replace(".attn1.k_norm.", ".attn1.norm_k.")
    return sub


def _raw_to_diffusers(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Raw Lightricks projection + connector weights → diffusers LTX2TextConnectors keys."""
    VP = "model.diffusion_model.video_embeddings_connector."
    AP = "model.diffusion_model.audio_embeddings_connector."
    VA = "text_embedding_projection.video_aggregate_embed."
    AA = "text_embedding_projection.audio_aggregate_embed."
    out: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if k.startswith(VA):
            out["video_text_proj_in." + k[len(VA) :]] = v
        elif k.startswith(AA):
            out["audio_text_proj_in." + k[len(AA) :]] = v
        elif k.startswith(VP):
            out["video_connector." + _connector_block_subkey(k[len(VP) :])] = v
        elif k.startswith(AP):
            out["audio_connector." + _connector_block_subkey(k[len(AP) :])] = v
    return out


def _reference_embeds() -> tuple[torch.Tensor, torch.Tensor, str]:
    """Torch reference for the 2.5 text path. The Gemma-4 stack is not re-run here: its
    ``--full`` hidden states are read back, since generating them needs transformers>=5."""
    pytest.importorskip("diffusers")
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

    for path in (REFERENCE_FULL, PROMPT_FILE, TEXT_ENCODER, TRANSFORMER):
        if not Path(path).exists():
            pytest.skip(f"missing {path}; regenerate with gen_gemma4_reference.py --full")

    prompt = Path(PROMPT_FILE).read_text().strip()

    with safe_open(REFERENCE_FULL, "pt") as handle:
        attention_mask = handle.get_tensor("attention_mask")
        num_states = sum(1 for k in handle.keys() if k.startswith("hidden."))
        # Pack the 49 hidden states D-major (B, seq, 3840, 49) → (B, seq, 188160), which is
        # the column order the checkpoint's projection expects.
        hidden = torch.stack([handle.get_tensor(f"hidden.{i}") for i in range(num_states)], dim=-1)
    hidden = hidden.flatten(2, 3).float()
    logger.info(f"reference hidden {tuple(hidden.shape)} from {num_states} states")

    # Every one of these fields is identical in the 2.3 and 2.5 DiT metadata.
    connectors = (
        LTX2TextConnectors(
            caption_channels=3840,
            text_proj_in_factor=num_states,
            video_connector_num_attention_heads=32,
            video_connector_attention_head_dim=128,
            video_connector_num_layers=8,
            audio_connector_num_attention_heads=32,
            audio_connector_attention_head_dim=64,
            audio_connector_num_layers=8,
            video_gated_attn=True,
            audio_gated_attn=True,
            per_modality_projections=True,
            video_hidden_dim=4096,
            audio_hidden_dim=2048,
            rope_type="split",
            rope_double_precision=False,
            proj_bias=True,
        )
        .float()
        .eval()
    )

    raw = {}
    with safe_open(TEXT_ENCODER, "pt") as handle:
        raw.update({k: handle.get_tensor(k) for k in handle.keys() if k.startswith(PROJECTION_PREFIXES)})
    with safe_open(TRANSFORMER, "pt") as handle:
        raw.update({k: handle.get_tensor(k) for k in handle.keys() if "embeddings_connector." in k})
    inc = connectors.load_state_dict(_raw_to_diffusers(raw), strict=False)
    # A rename upstream would show up here as a missing key rather than as a bad PCC.
    assert not inc.missing_keys, f"unmapped 2.5 connector weights: {inc.missing_keys[:8]}"
    logger.info(f"ref connectors load: unexpected={len(inc.unexpected_keys)}")

    with torch.no_grad():
        # 2.5's packed tokenizer_config pads left, and the --full reference was built that way.
        video, audio, _ = connectors(hidden, attention_mask, padding_side="left")
    return video.float(), audio.float(), prompt


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "device_params",
    [{**line_params_req_exact_devices, "l1_small_size": 8192}],
    indirect=["device_params"],
)
def test_gemma4_text_path_matches_diffusers(*, mesh_device):
    """Projection + both connectors on 2.5 weights, against the host reference.

    A pass here means the embeddings the DiT is conditioned on are faithful to the shipped
    checkpoint, which would move weak prompt adherence out of the port and into prompting.
    """
    v_ref, a_ref, prompt = _reference_embeds()

    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=mesh_device.shape[1], mesh_axis=1),
    )
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    pair = Gemma4TokenizerEncoderPair(
        TEXT_ENCODER,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        transformer_checkpoint=TRANSFORMER,
        mode="av",
        dynamic_load=False,
    )
    pair.ensure_loaded()

    v_dev, a_dev = pair.encode([prompt])[0]
    v_dev, a_dev = v_dev.float(), a_dev.float()

    logger.info(f"VIDEO  ref={tuple(v_ref.shape)} dev={tuple(v_dev.shape)}")
    assert_quality(v_ref, v_dev, pcc=0.999)
    logger.info(f"AUDIO  ref={tuple(a_ref.shape)} dev={tuple(a_dev.shape)}")
    assert_quality(a_ref, a_dev, pcc=0.999)
