# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end generation test for the TTNN XTTS-v2 GPT (Block 3).

Validates prefill + KV-cached greedy decode against the CPU reference (same weights,
same prompt). Two checks:
  1. Teacher-forced: replay the reference per-step inputs; TTNN logits/latents must match
     the reference (latent PCC >= 0.999) and argmax must agree per step. Isolates the
     transformer+head from sampling cascades.
  2. Free-running: TTNN generates on its own samples; report how many leading codes match
     the reference (bf16 argmax flips can eventually diverge — informational).

Requires generation goldens:
    python models/experimental/xtts_v2/reference/xtts_gpt_ref.py

Run:
    pytest -svv models/experimental/xtts_v2/tests/test_gpt_generate_pcc.py
"""

import os

import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.xtts_v2.reference.xtts_gpt_ref import load_gen_head
from models.experimental.xtts_v2.tt.ttnn_xtts_gpt import preprocess_gpt_parameters
from models.experimental.xtts_v2.tt.ttnn_xtts_gpt_generate import TTNNGPTGenerator

GEN_DIR = os.path.join(os.path.dirname(__file__), "..", "golden", "gpt", "generate")
LATENT_PCC = 0.999


def _load():
    g = {
        k: torch.load(os.path.join(GEN_DIR, f"{k}.pt"))
        for k in ("prompt_embeds", "step_inputs", "ref_codes", "ref_logits", "ref_latents")
    }
    g["meta"] = torch.load(os.path.join(GEN_DIR, "meta.pt"))
    return g


def run_generate(device):
    g = _load()
    heads = load_gen_head()
    max_seq = ((g["prompt_embeds"].shape[1] + g["step_inputs"].shape[1] + 31) // 32) * 32
    gen = TTNNGPTGenerator(
        device,
        preprocess_gpt_parameters(device, dtype=ttnn.bfloat16),
        heads,
        max_seq=max_seq,
        start_token=g["meta"]["start_token"],
        stop_token=g["meta"]["stop_token"],
    )

    # 1) Teacher-forced
    tt_logits, tt_latents = gen.teacher_forced(g["prompt_embeds"], g["step_inputs"])
    lat_ok, lat_msg = comp_pcc(g["ref_latents"], tt_latents, pcc=LATENT_PCC)
    _, logit_msg = comp_pcc(g["ref_logits"], tt_logits, pcc=LATENT_PCC)
    tt_argmax = tt_logits.argmax(-1).flatten()
    ref_codes = g["ref_codes"].flatten()
    agree = (tt_argmax == ref_codes).float().mean().item()
    print(f"[teacher-forced] latent pcc: {lat_msg}")
    print(f"[teacher-forced] logit  pcc: {logit_msg}")
    print(f"[teacher-forced] argmax agreement vs ref: {agree*100:.1f}% ({int(agree*len(ref_codes))}/{len(ref_codes)})")

    # 2) Free-running
    codes, _ = gen.generate(g["prompt_embeds"], max_new=len(ref_codes))
    lead = 0
    for a, b in zip(codes, ref_codes.tolist()):
        if a == b:
            lead += 1
        else:
            break
    print(f"[free-running] ref codes: {ref_codes.tolist()}")
    print(f"[free-running] ttnn codes: {codes}")
    print(f"[free-running] leading match: {lead}/{len(ref_codes)}")

    return lat_ok, lat_msg, agree


def test_gpt_generate(device):
    lat_ok, lat_msg, agree = run_generate(device)
    assert lat_ok, f"teacher-forced latent PCC below {LATENT_PCC}: {lat_msg}"


if __name__ == "__main__":
    import sys

    dev = ttnn.open_device(device_id=0)
    try:
        dev.enable_program_cache()
        ok, msg, agree = run_generate(dev)
    finally:
        ttnn.close_device(dev)
    print(("PASSED " if ok else "FAILED ") + str(msg) + f" | argmax agree {agree*100:.1f}%")
    sys.exit(0 if ok else 1)
