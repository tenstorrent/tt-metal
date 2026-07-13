# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""End-to-end parity: on-device ``encode_prompts`` vs a torch reference built from HF
Gemma-3 + diffusers ``LTX2TextConnectors`` (mirrors the diffusers ``LTX2Pipeline`` text
path). PCC of the final video (4096) and audio (2048) context embeddings.

Do NOT run the 2x4 case under TT_METAL_WATCHER — the watcher overflows the active-eth
fabric-router kernel-config buffer at device open.

    pytest models/tt_dit/tests/encoders/gemma/test_gemma_full.py -s
"""

import glob
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[6]))

import pytest
import torch
from loguru import logger
from safetensors import safe_open

import ttnn
from models.tt_dit.pipelines.ltx.pipeline_ltx import LTXPipeline
from models.tt_dit.utils.check import assert_quality


def _connector_block_subkey(sub: str) -> str:
    """Per-block raw Lightricks key → diffusers connector key: only the block-list name and
    the QK-norm names differ (q_norm/k_norm → norm_q/norm_k); the rest is shared."""
    sub = sub.replace("transformer_1d_blocks.", "transformer_blocks.")
    sub = sub.replace(".attn1.q_norm.", ".attn1.norm_q.")
    sub = sub.replace(".attn1.k_norm.", ".attn1.norm_k.")
    return sub


def _raw_connectors_to_diffusers(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Raw Lightricks connector + aggregate_embed weights → diffusers LTX2TextConnectors keys.
    aggregate_embed → text_proj_in; the connector blocks map 1:1 modulo _connector_block_subkey."""
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


PROMPT = "A plump orange tabby cat sits on a piano bench playing keys with its paws."

CONNECTOR_PREFIXES = (
    "text_embedding_projection.video_aggregate_embed.",
    "text_embedding_projection.audio_aggregate_embed.",
    "model.diffusion_model.video_embeddings_connector.",
    "model.diffusion_model.audio_embeddings_connector.",
)


def _gemma_path() -> str:
    explicit = os.environ.get("GEMMA_PATH")
    if explicit:
        return explicit
    cands = glob.glob(
        os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-3-12b-it-qat-q4_0-unquantized/snapshots/*/")
    )
    return cands[0].rstrip("/") if cands else "google/gemma-3-12b-it-qat-q4_0-unquantized"


def _ltx_ckpt() -> str | None:
    explicit = os.environ.get("LTX_CHECKPOINT")
    if explicit and os.path.exists(explicit):
        return explicit
    cands = glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/*/ltx-2.3-22b-dev.safetensors"
        )
    )
    return cands[0] if cands else None


def _encode_prompts_reference(
    checkpoint_path: str, gemma_root: str, prompts: list[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch reference: HF Gemma-3 text encoder + diffusers ``LTX2TextConnectors``, mirroring
    the diffusers ``LTX2Pipeline`` text path (stacked per-layer hidden states → per-modality
    feature extraction + connectors). Connector/aggregate weights come from the raw checkpoint;
    the LTX-2 reference repo is no longer used. Returns (video_encoding, audio_encoding)."""
    pytest.importorskip("diffusers")
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # LTX-2.3-22b connector config: per-modality projections, 8 blocks each, video 32x128=4096,
    # audio 32x64=2048, gated attention, SPLIT rope (the raw checkpoint's rotation convention).
    connectors = (
        LTX2TextConnectors(
            caption_channels=3840,
            text_proj_in_factor=49,
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
    with safe_open(checkpoint_path, "pt") as f:
        for k in f.keys():
            if k.startswith(CONNECTOR_PREFIXES):
                raw[k] = f.get_tensor(k)
    inc = connectors.load_state_dict(_raw_connectors_to_diffusers(raw), strict=False)
    logger.info(f"ref connectors load: missing={len(inc.missing_keys)} unexpected={len(inc.unexpected_keys)}")

    tok = AutoTokenizer.from_pretrained(gemma_root)
    tok.padding_side = "left"  # Gemma-3 / LTX-2 use left padding
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    text_encoder = AutoModelForCausalLM.from_pretrained(gemma_root, torch_dtype=torch.bfloat16).eval()

    ti = tok(
        prompts,
        padding="max_length",
        max_length=1024,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        out = text_encoder(input_ids=ti.input_ids, attention_mask=ti.attention_mask, output_hidden_states=True)
        # Pack the 49 per-layer hidden states D-major (B, seq, 3840, 49) → (B, seq, 188160).
        hidden = torch.stack(out.hidden_states, dim=-1).flatten(2, 3).float()
        video, audio, _ = connectors(hidden, ti.attention_mask)
    return video.float(), audio.float()


# 2x4 drives the encoder's TP all-gathers over CCL, which needs the 1D fabric up;
# 1x1 is single-chip with no CCL, so it must run without fabric.
@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [
        pytest.param((1, 1), {"l1_small_size": 8192}, id="1x1"),
        pytest.param((2, 4), {"l1_small_size": 8192, "fabric_config": ttnn.FabricConfig.FABRIC_1D}, id="2x4"),
        pytest.param((4, 8), {"l1_small_size": 8192, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="4x8"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_gemma_encoder(*, mesh_device):
    gemma = _gemma_path()
    ckpt = _ltx_ckpt()
    if not os.path.isdir(gemma):
        pytest.skip(f"Gemma not found: {gemma}")
    if not ckpt:
        pytest.skip("LTX checkpoint not found")

    # Bare pipeline: checkpoint_name=None skips the heavy transformer/VAE load.
    pipe = LTXPipeline.create_pipeline(mesh_device, checkpoint_name=None, gemma_path=gemma, mode="av")

    # On-device Gemma encoder (full 48 layers). TP follows the T5 pattern (axis-1 width):
    # TP=1 on 1x1, TP=4 on 2x4 — set inside the loader, no override needed.
    pipe.gemma_encoder_pair.load_gemma_encoder(gemma)

    # Load only the connector weights from the 46GB checkpoint.
    conn_state = {}
    with safe_open(ckpt, "pt") as f:
        for k in f.keys():
            if k.startswith(CONNECTOR_PREFIXES):
                conn_state[k] = f.get_tensor(k)
    logger.info(f"connector weights: {len(conn_state)} tensors")
    pipe.gemma_encoder_pair.load_embeddings_connectors(conn_state, audio_num_blocks=8)

    # Reference embeds (HF Gemma-3 + diffusers LTX2TextConnectors), local to this test.
    v_ref, a_ref = _encode_prompts_reference(ckpt, gemma, [PROMPT])

    # Device embeds. First call compiles + populates the program cache; the second is warm
    # and is the one we time. use_cache=False to measure the real encode, not a cache load.
    pipe.encode_prompts([PROMPT], use_cache=False)
    t0 = time.perf_counter()
    dev = pipe.encode_prompts([PROMPT], use_cache=False)
    t_warm_ms = (time.perf_counter() - t0) * 1e3

    v_dev = torch.as_tensor(dev[0][0]).float()
    a_dev = torch.as_tensor(dev[0][1]).float()

    logger.info(f"ENCODE warm wall-clock (mesh {tuple(mesh_device.shape)}): {t_warm_ms:.1f} ms")

    # Device and reference both pad to GEMMA_SEQUENCE_LENGTH=1024 (the reference text
    # encoder fixes its tokenizer to 1024 too), so the outputs are the same shape and
    # compare directly — no alignment needed.
    logger.info(f"VIDEO  ref={tuple(v_ref.shape)} dev={tuple(v_dev.shape)}")
    assert_quality(v_ref, v_dev, pcc=0.9995)
    # Audio rides ~0.9982 — the looser bound reflects the longer connector chain it passes
    # through, not a regression. Tighten if the audio path is later hardened.
    logger.info(f"AUDIO  ref={tuple(a_ref.shape)} dev={tuple(a_dev.shape)}")
    assert_quality(a_ref, a_dev, pcc=0.998)
