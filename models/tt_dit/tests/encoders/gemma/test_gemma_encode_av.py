# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
End-to-end validation of the on-device Gemma encode path for LTX AV.

Compares ``encode_prompts_device`` (TTNN GemmaEncoder + on-device video/audio
embeddings connectors) against ``encode_prompts_reference`` (official LTX-2 CPU
PromptEncoder) for the same prompt, reporting PCC of the final video (4096-dim)
and audio (2048-dim) context embeddings.

Single Blackhole chip (1x1 mesh, tp=1):
    pytest models/tt_dit/tests/encoders/gemma/test_gemma_encode_av.py -s
"""

import glob
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[6]))

import pytest
import torch
from loguru import logger
from safetensors import safe_open

from models.tt_dit.pipelines.ltx.pipeline_ltx_av import LTXAVPipeline

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


def pcc(a, b):
    a_f, b_f = a.flatten().float(), b.flatten().float()
    n = min(a_f.numel(), b_f.numel())
    a_f, b_f = a_f[:n], b_f[:n]
    a_m, b_m = a_f - a_f.mean(), b_f - b_f.mean()
    d = (a_m.pow(2).sum() * b_m.pow(2).sum()).sqrt()
    return ((a_m * b_m).sum() / d).item() if d > 0 else 0.0


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=["device_params"])
def test_encode_prompts_device_vs_reference(*, mesh_device):
    gemma = _gemma_path()
    ckpt = _ltx_ckpt()
    if not os.path.isdir(gemma):
        pytest.skip(f"Gemma not found: {gemma}")
    if not ckpt:
        pytest.skip("LTX checkpoint not found")

    # Bare pipeline: checkpoint_name=None skips the heavy transformer/VAE load.
    pipe = LTXAVPipeline.create_pipeline(mesh_device, checkpoint_name=None, gemma_path=gemma, mode="av")

    # On-device Gemma encoder (full 48 layers, tp from the 1x1 mesh = 1).
    pipe.load_gemma_encoder(gemma, num_layers=48, sequence_length=1024)

    # Load only the connector weights from the 46GB checkpoint.
    conn_state = {}
    with safe_open(ckpt, "pt") as f:
        for k in f.keys():
            if k.startswith(CONNECTOR_PREFIXES):
                conn_state[k] = f.get_tensor(k)
    logger.info(f"connector weights: {len(conn_state)} tensors")
    pipe.load_embeddings_connectors(conn_state, audio_num_blocks=8)

    # Reference embeds (official CPU PromptEncoder; cached to ~/.cache/tt-dit after first run).
    pipe.checkpoint_name = ckpt
    ref = pipe.encode_prompts_reference([PROMPT])
    v_ref = ref[0].video_encoding.float()
    a_ref = ref[0].audio_encoding.float()

    # Device embeds.
    dev = pipe.encode_prompts_device([PROMPT])
    v_dev = torch.as_tensor(dev[0][0]).float()
    a_dev = torch.as_tensor(dev[0][1]).float()

    logger.info(f"VIDEO  ref={tuple(v_ref.shape)} dev={tuple(v_dev.shape)}  PCC={pcc(v_dev, v_ref):.4f}")
    logger.info(f"AUDIO  ref={tuple(a_ref.shape)} dev={tuple(a_dev.shape)}  PCC={pcc(a_dev, a_ref):.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-v"])
