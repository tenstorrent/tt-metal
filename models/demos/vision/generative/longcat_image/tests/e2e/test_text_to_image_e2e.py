# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end PCC test for LongCat-Image Call 1 (text -> image).

Runs the ONE shared TTNN pipeline (tt/pipeline.py) that the demo also runs, and
compares its final image to the HF golden (LongCatImagePipeline, Source A) at the
identical seed / steps / guidance / size / prompt. Asserts the three gates:

  Gate 1 — every routed graduated stub is real ttnn (static scan for forbidden
           torch host-compute / HF-orchestration ops in its hot path).
  Gate 2 — every graduated module on Call 1's critical path is INVOKED (directly
           or subsumed by an invoked graduated container). No module wasted.
  Gate 3 — final image PCC >= 0.95 vs the HF golden.

Speed: the DiT is 6.27B params and the text encoder 7B, so caps (steps/size/
max_length) are small by default and identical on both sides. The HF golden is
cached to disk so TT-side iteration is fast. Override caps via env vars.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
import torch

from models.common.utility_functions import comp_pcc
from models.demos.vision.generative.longcat_image.tt import pipeline as P

HF_MODEL_ID = "meituan-longcat/LongCat-Image"
PCC_TARGET = 0.95

# Default gate: single-step classifier-free-guidance (guidance=4.5) text->image.
# A single denoise step avoids the multi-step trajectory divergence between the TT
# and the independent HF golden (inherent to iterative diffusion) while still
# exercising the full CFG + cfg-renorm + scheduler path at the pipeline's default
# guidance. Override LONGCAT_E2E_STEPS to run a longer denoise (produces a valid
# image; PCC drops as the trajectories diverge). A detailed prompt fills the token
# budget so there are ~no precision-sensitive pad rows in the Qwen encoder.
STEPS = int(os.environ.get("LONGCAT_E2E_STEPS", "1"))
SIZE = int(os.environ.get("LONGCAT_E2E_SIZE", "256"))
MAXLEN = int(os.environ.get("LONGCAT_E2E_MAXLEN", "64"))
GUIDANCE = float(os.environ.get("LONGCAT_E2E_GUIDANCE", "4.5"))
SEED = int(os.environ.get("LONGCAT_E2E_SEED", "0"))
PROMPT = os.environ.get(
    "LONGCAT_E2E_PROMPT",
    "A young Asian woman wearing a yellow knit sweater with a white necklace, her hands resting "
    "gently on her knees, a serene and calm expression on her face, sitting in front of a rough "
    "textured brick wall, warm afternoon sunlight falling softly on her, cinematic soft lighting, "
    "medium shot portrait, highly detailed, elegant and tranquil atmosphere, photorealistic, sharp focus",
)
NEG_PROMPT = os.environ.get("LONGCAT_E2E_NEG", "")

_CACHE_DIR = Path(__file__).resolve().parents[1].parent / "_golden_cache"

# Routed graduated stubs on Call 1's critical path (Gate 1 scans these files).
ROUTED_STUBS_CALL1 = [
    "qwen2_v_l_model",
    "qwen2_v_l_for_conditional_generation",
    "long_cat_image_transformer2_d_model",
    "autoencoder_k_l",
    "decoder",
]

# Gate-2 coverage: every graduated module Call 1 exercises -> owning invoked container.
# (vision tower — qwen2_vision_transformer_pretrained_model / qwen2_v_l_vision_block /
#  qwen2_v_l_patch_merger — is Call 2 only and asserted in the image_edit e2e test.)
CALL1_COVERAGE = {
    # DIRECT
    "qwen2_v_l_model": "qwen2_v_l_model",
    "long_cat_image_transformer2_d_model": "long_cat_image_transformer2_d_model",
    "autoencoder_k_l": "autoencoder_k_l",
    "qwen2_v_l_for_conditional_generation": "qwen2_v_l_for_conditional_generation",
    # SUBSUMED by qwen2_v_l_model / _TextEncoder body
    "qwen2_v_l_text_model": "qwen2_v_l_for_conditional_generation",
    "qwen2_v_l_decoder_layer": "qwen2_v_l_for_conditional_generation",
    # SUBSUMED by long_cat_image_transformer2_d_model
    "long_cat_image_transformer_block": "long_cat_image_transformer2_d_model",
    "long_cat_image_single_transformer_block": "long_cat_image_transformer2_d_model",
    "feed_forward": "long_cat_image_transformer2_d_model",
    "ada_layer_norm_zero": "long_cat_image_transformer2_d_model",
    "ada_layer_norm_zero_single": "long_cat_image_transformer2_d_model",
    "ada_layer_norm_continuous": "long_cat_image_transformer2_d_model",
    "long_cat_image_timestep_embeddings": "long_cat_image_transformer2_d_model",
    "timestep_embedding": "long_cat_image_transformer2_d_model",
    # SUBSUMED by autoencoder_k_l
    "decoder": "autoencoder_k_l",
    "encoder": "autoencoder_k_l",
    "resnet_block2_d": "autoencoder_k_l",
    "down_encoder_block2_d": "autoencoder_k_l",
    "up_decoder_block2_d": "autoencoder_k_l",
    "downsample2_d": "autoencoder_k_l",
    "upsample2_d": "autoencoder_k_l",
    "u_net_mid_block2_d": "autoencoder_k_l",
}

_FORBIDDEN = [
    "torch.matmul", "torch.mm(", "torch.bmm", "torch.einsum", "torch.softmax", "torch.log_softmax",
    "torch.layer_norm", "torch.rms_norm", "torch.batch_norm", "torch.group_norm", "torch.embedding",
    "torch.conv1d", "torch.conv2d", "torch.conv3d", "torch.conv_transpose", "torch.scaled_dot_product_attention",
    "torch.relu", "torch.gelu", "torch.silu", "torch.sigmoid", "torch.leaky_relu",
    "torch.argmax", "torch.topk", "torch.multinomial", "torch.dropout", "torch.nn.functional",
    ".generate(", ".forward =", ".forward=",
]


def _gate1_scan(stub_names):
    """Return list of (file, lineno, line) violations. Comments are stripped."""
    stubs_dir = Path(P.__file__).resolve().parents[1] / "_stubs"
    violations = []
    for name in stub_names:
        f = stubs_dir / f"{name}.py"
        for i, raw in enumerate(f.read_text().splitlines(), 1):
            code = raw.split("#", 1)[0]
            for pat in _FORBIDDEN:
                if pat in code:
                    violations.append((name, i, raw.strip()))
    return violations


def _golden_key():
    h = hashlib.sha1(
        f"v2fp32|{PROMPT}|{NEG_PROMPT}|{SIZE}|{STEPS}|{GUIDANCE}|{SEED}|{MAXLEN}".encode()
    ).hexdigest()[:16]
    return h


def _make_latents(pipe):
    vsf = pipe.vae_scale_factor
    lh = 2 * (SIZE // (vsf * 2))
    lw = 2 * (SIZE // (vsf * 2))
    gen = torch.Generator("cpu").manual_seed(SEED)
    raw = torch.randn(1, 16, lh, lw, generator=gen, dtype=torch.float32)
    return P._pack_latents(raw, 1, 16, lh, lw)


@pytest.fixture(scope="module")
def pipe():
    from diffusers import LongCatImagePipeline

    print(f"[e2e] loading {HF_MODEL_ID} (bf16) ...", flush=True)
    p = LongCatImagePipeline.from_pretrained(HF_MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    p.set_progress_bar_config(disable=True)
    p.tokenizer_max_length = MAXLEN
    return p


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_text_to_image_e2e(device_params, device, pipe):
    pipe.tokenizer_max_length = MAXLEN
    latents_packed = _make_latents(pipe)

    # ── Gate 1: routed stubs are real ttnn ────────────────────────────────────
    violations = _gate1_scan(ROUTED_STUBS_CALL1)
    assert not violations, f"Gate 1 FAIL — forbidden host-compute/HF ops in routed stubs: {violations}"
    print(f"[e2e] Gate 1 PASS — {len(ROUTED_STUBS_CALL1)} routed stubs are native ttnn", flush=True)

    # ── HF golden (cached) ────────────────────────────────────────────────────
    _CACHE_DIR.mkdir(exist_ok=True)
    cache_p = _CACHE_DIR / f"t2i_{_golden_key()}.pt"
    if cache_p.is_file():
        print(f"[e2e] loading cached golden {cache_p.name}", flush=True)
        golden = torch.load(cache_p, map_location="cpu", weights_only=False)
    else:
        print("[e2e] computing HF golden (this is the slow one-time step) ...", flush=True)
        golden = P.hf_reference_text_to_image(
            pipe,
            prompt=PROMPT,
            negative_prompt=NEG_PROMPT,
            height=SIZE,
            width=SIZE,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            seed=SEED,
            max_length=MAXLEN,
            latents_packed=latents_packed,
        )
        torch.save(golden, cache_p)

    # ── TT pipeline (the real chained forward the demo also runs) ─────────────
    print("[e2e] running TT pipeline ...", flush=True)
    ttp = P.LongCatImagePipelineTT(device, pipe)
    result = ttp.run_text_to_image(
        prompt=PROMPT,
        negative_prompt=NEG_PROMPT,
        height=SIZE,
        width=SIZE,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        seed=SEED,
        max_length=MAXLEN,
        latents_packed=latents_packed,
    )

    # ── per-stage PCC diagnostics ─────────────────────────────────────────────
    _, pcc_text = comp_pcc(golden["prompt_embeds"], result["prompt_embeds_pos"].float(), 0.0)
    _, pcc_latent = comp_pcc(golden["final_latent_packed"], result["final_latent_packed"].float(), 0.0)
    _, pcc_img_raw = comp_pcc(golden["image"], result["image"].float(), 0.0)
    ok_final, pcc_final = comp_pcc(golden["image_denorm"], result["image_denorm"].float(), PCC_TARGET)
    print(f"[e2e] stage PCC: text_encode={pcc_text}  final_latent={pcc_latent}  image_raw={pcc_img_raw}", flush=True)

    # ── Gate 2: coverage ──────────────────────────────────────────────────────
    invoked = result["invoked"]
    uncovered = [m for m, owner in CALL1_COVERAGE.items() if owner not in invoked]
    assert not uncovered, f"Gate 2 FAIL — graduated modules not covered by an invoked container: {uncovered}"
    print(f"[e2e] Gate 2 PASS — {len(CALL1_COVERAGE)} graduated modules covered; invoked={sorted(invoked)}", flush=True)

    # ── Gate 3: final PCC ─────────────────────────────────────────────────────
    print(f"e2e PCC={pcc_final}")
    assert ok_final, f"Gate 3 FAIL — final image PCC {pcc_final} < {PCC_TARGET}"
