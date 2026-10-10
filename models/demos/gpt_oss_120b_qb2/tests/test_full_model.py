# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full-stack agreement against a pinned, independently generated HF reference."""

import hashlib
import json
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.gpt_oss_120b_qb2.tt.generator import build_generator
from models.demos.gpt_oss_120b_qb2.tt.model import HF_CONTEXT_LENGTH, MODEL_REVISION
from models.demos.utils.trace_region_sizes import TRACE_MODEL_KEY_PARAM

REFERENCE_SHA256 = "7e722ad241eee84148ed62b5accee20bc642a4a1de4cab98ae146a166ee9d2bc"


def load_reference(path):
    """Read the original 100-token reference without executing pickle globals."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != REFERENCE_SHA256:
        raise ValueError(f"GPT-OSS reference SHA256 must be {REFERENCE_SHA256}, got {digest}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["hf_model_id"] != f"openai/gpt-oss-120b@{MODEL_REVISION}" or len(payload["entries"]) != 1:
        raise ValueError(f"Expected one openai/gpt-oss-120b@{MODEL_REVISION} reference entry")
    entry = payload["entries"][0]
    if entry["generated_tokens"].shape != (1, 100) or entry["topk_tokens"].shape != (100, 100):
        raise ValueError("Expected exactly 100 teacher-forced tokens and 100 reference candidates per step")
    return entry


def agreement(predicted, reference):
    predicted = torch.as_tensor(predicted).reshape(-1, 1)
    assert predicted.shape[0] == reference.shape[0] == 100
    return {f"top{k}": float((predicted == reference[:, :k]).any(dim=1).float().mean()) for k in (1, 5, 100)}


def require_agreement(metrics):
    assert metrics["top5"] >= 0.98, metrics
    assert metrics["top100"] == 1.0, metrics


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (1, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "require_exact_physical_num_devices": True,
                TRACE_MODEL_KEY_PARAM: "gpt-oss-120b",
            },
            id="p150x4",
        )
    ],
    indirect=True,
)
def test_full_stack_prefill_and_traced_teacher_forcing(mesh_device, device_params, reset_seeds):
    """All 36 layers, full context capacity, and each of the original 100 reference steps."""
    del device_params, reset_seeds
    reference = load_reference(os.environ["GPT_OSS_120B_REFERENCE"])
    prompt = reference["prompt_tokens"][0].long().tolist()
    forced = reference["generated_tokens"][0].long().tolist()
    candidates = reference["topk_tokens"].long()
    generator = build_generator(
        model_dir=Path(__file__).parents[1],
        mesh_device=mesh_device,
        snapshot_path=os.environ["GPT_OSS_120B_SNAPSHOT"],
        tensor_cache_path=os.environ["TT_METAL_CACHE"],
        max_context_length=HF_CONTEXT_LENGTH,
        max_batch_size=1,
    )
    results = {"checkpoint_revision": MODEL_REVISION, "reference_sha256": REFERENCE_SHA256}
    output = Path(os.environ.get("GPT_OSS_120B_RESULTS", "generated/test_reports/gpt_oss_120b_qb2"))
    output.mkdir(parents=True, exist_ok=True)
    try:
        # A causal prefill of the same teacher-forced sequence predicts all 100
        # reference steps; the final prompt row predicts generated token zero.
        logits = generator.prefill_logits(prompt + forced[:-1])
        predicted = logits[0, len(prompt) - 1 :].argmax(dim=-1)
        results["prefill"] = agreement(predicted, candidates)
        require_agreement(results["prefill"])
        del logits

        rows = []
        results["teacher_forcing"] = rows
        for repetition in range(4):
            started = time.perf_counter()
            predicted = generator.generate(
                prompt,
                len(forced),
                next_input=lambda step, _: forced[step],
                enable_trace=True,
                sampling_mode="device",
            )
            elapsed = time.perf_counter() - started
            metrics = agreement(predicted, candidates)
            rows.append(
                {
                    "repetition": repetition,
                    "warmup": repetition == 0,
                    "elapsed_s": elapsed,
                    **metrics,
                    "predicted": predicted,
                }
            )
            require_agreement(metrics)
            evidence = generator.trace_evidence.to_dict()
            assert evidence["model_execute_submissions"] == len(forced) - 1
            assert evidence["sampling_execute_submissions"] == len(forced) - 1
            assert evidence["full_logits_readbacks"] == 0
    finally:
        (output / "reference_agreement.json").write_text(json.dumps(results, indent=2) + "\n")
        generator.teardown()
