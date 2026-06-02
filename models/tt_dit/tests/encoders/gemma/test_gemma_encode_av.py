# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
End-to-end validation of the on-device Gemma encode path for LTX AV.

Compares ``encode_prompts`` (TTNN GemmaEncoder + on-device video/audio
embeddings connectors) against the official LTX-2 CPU ``PromptEncoder`` reference
(wrapped locally in ``_encode_prompts_reference``) for the same prompt, reporting
PCC of the final video (4096-dim) and audio (2048-dim) context embeddings.

Parametrized over two meshes (Blackhole): 1x1 (TP=1, no fabric) and 2x4 (TP=4,
FABRIC_1D). Asserts video PCC > 0.999 and audio PCC > 0.998, and logs warm encode
wall-clock. Do NOT run the 2x4 case under TT_METAL_WATCHER — the watcher overflows
the active-eth fabric-router kernel-config buffer at device open.
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

import ttnn
from models.tt_dit.pipelines.ltx.pipeline_ltx import LTXPipeline

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


def _encode_prompts_reference(checkpoint_path: str, gemma_root: str, prompts: list[str]) -> list:
    """Official LTX-2 CPU ``PromptEncoder`` reference, kept here for validation only —
    the production pipeline no longer depends on the reference package."""
    for p in ("LTX-2/packages/ltx-core/src", "LTX-2/packages/ltx-pipelines/src"):
        if p not in sys.path:
            sys.path.insert(0, p)
    torch.cuda.synchronize = lambda *a, **kw: None  # noqa: ARG005 — no CUDA on the TT host
    from ltx_pipelines.utils.blocks import PromptEncoder

    encoder = PromptEncoder(
        checkpoint_path=checkpoint_path,
        gemma_root=gemma_root,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    try:
        return encoder(prompts)
    finally:
        del encoder


# 2x4 drives the encoder's TP all-gathers over CCL, which needs the 1D fabric up;
# 1x1 is single-chip with no CCL, so it must run without fabric.
@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [
        pytest.param((1, 1), {"l1_small_size": 8192}, id="1x1"),
        pytest.param((2, 4), {"l1_small_size": 8192, "fabric_config": ttnn.FabricConfig.FABRIC_1D}, id="2x4"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_encode_prompts_vs_reference(*, mesh_device):
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

    # Reference embeds (official CPU PromptEncoder), local to this test.
    ref = _encode_prompts_reference(ckpt, gemma, [PROMPT])
    v_ref = ref[0].video_encoding.float()
    a_ref = ref[0].audio_encoding.float()

    # Device embeds. First call compiles + populates the program cache; the second is warm
    # and is the one we time. use_cache=False to measure the real encode, not a cache load.
    import time

    pipe.encode_prompts([PROMPT], use_cache=False)
    t0 = time.perf_counter()
    dev = pipe.encode_prompts([PROMPT], use_cache=False)
    t_warm_ms = (time.perf_counter() - t0) * 1e3

    v_dev = torch.as_tensor(dev[0][0]).float()
    a_dev = torch.as_tensor(dev[0][1]).float()

    logger.info(f"ENCODE warm wall-clock (mesh {tuple(mesh_device.shape)}): {t_warm_ms:.1f} ms")

    v_pcc = pcc(v_dev, v_ref)
    a_pcc = pcc(a_dev, a_ref)
    logger.info(f"VIDEO  ref={tuple(v_ref.shape)} dev={tuple(v_dev.shape)}  PCC={v_pcc:.4f}")
    logger.info(f"AUDIO  ref={tuple(a_ref.shape)} dev={tuple(a_dev.shape)}  PCC={a_pcc:.4f}")

    # Audio rides ~0.999; the looser bound reflects the longer connector chain it
    # passes through, not a regression. Tighten if the audio path is later hardened.
    assert v_pcc > 0.999, f"video context PCC {v_pcc:.4f} below 0.999"
    assert a_pcc > 0.998, f"audio context PCC {a_pcc:.4f} below 0.998"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-v"])
