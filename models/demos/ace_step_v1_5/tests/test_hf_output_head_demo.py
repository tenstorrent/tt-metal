# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class VariantSpec:
    name: str
    best_use_case: str
    repo_id: str
    subfolder: str


VARIANTS: list[VariantSpec] = [
    VariantSpec(
        name="ACE-Step v1.5 Turbo",
        best_use_case="Fast generation (2–10 seconds). Great for quick prototyping.",
        repo_id="ACE-Step/Ace-Step1.5",
        subfolder="acestep-v15-turbo",
    ),
    # The public docs mention SFT/Base. If they are not present in the repo snapshot, we skip.
    VariantSpec(
        name="ACE-Step v1.5 SFT",
        best_use_case="Highest quality and best prompt adherence. Uses more steps for better sound.",
        repo_id="ACE-Step/acestep-v15-sft",
        subfolder="",
    ),
    VariantSpec(
        name="ACE-Step v1.5 Base",
        best_use_case="The Swiss Army Knife for advanced tasks like track extraction and expansion.",
        repo_id="ACE-Step/acestep-v15-base",
        subfolder="",
    ),
]


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.name)
def test_hf_output_head_demo_variants_smoke(variant: VariantSpec):
    """
    Smoke test: download HF weights (cached), run the demo forward, and validate output shape.

    If a variant subfolder is not present in the HF snapshot, this test is skipped.
    """
    from huggingface_hub import snapshot_download

    snapshot_dir = snapshot_download(repo_id=variant.repo_id)
    import os

    model_dir = os.path.join(snapshot_dir, variant.subfolder)
    if not os.path.exists(model_dir):
        pytest.skip(f"Variant not present in snapshot: {variant.subfolder}")

    from models.demos.ace_step_v1_5.torch_ref.hf_output_head_demo import run_hf_output_head_demo

    sig = run_hf_output_head_demo(
        repo_id=variant.repo_id,
        subfolder=variant.subfolder,
        seed=0,
        batch=1,
        original_seq_len=257,
        input_pt=None,
        noise_std=1.0,
        offline=False,
    )

    assert sig["repo_id"] == variant.repo_id
    assert sig["model_dir"].endswith(variant.subfolder)

    B, T, C = sig["output"]["shape"]
    assert B == 1
    assert T == 257
    # ACE turbo uses 64; other variants should still produce acoustic dim outputs.
    assert C > 0
