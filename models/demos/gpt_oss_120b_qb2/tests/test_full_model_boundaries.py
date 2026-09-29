# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full-model page, ring and chunk boundaries against independent HF logits."""

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.gpt_oss_120b_qb2.tt.generator import build_generator
from models.demos.gpt_oss_120b_qb2.tt.model import HF_CONTEXT_LENGTH, MODEL_REVISION
from models.demos.utils.trace_region_sizes import TRACE_MODEL_KEY_PARAM

BOUNDARIES = [63, 64, 65, 511, 512, 513, 767, 768, 769, 8191, 8192, 8193]


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
def test_full_model_boundary_predictions(mesh_device, device_params, reset_seeds):
    """Compare prefill and traced final-token decode at every frozen boundary."""
    del device_params, reset_seeds
    root = Path(os.environ["GPT_OSS_120B_BOUNDARY_REFERENCE"])
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["checkpoint_revision"] == MODEL_REVISION
    assert manifest["transformers"] == "5.12.1"
    assert manifest["layers"] == 36 and manifest["context_capacity"] == HF_CONTEXT_LENGTH
    assert manifest["boundaries"] == BOUNDARIES
    path = root / "full-model-logits.pt"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["files"][path.name]["sha256"]
    reference = torch.load(path, map_location="cpu", weights_only=True)
    assert reference["boundaries"] == BOUNDARIES
    assert reference["tokens"].shape == (1, max(BOUNDARIES))
    assert reference["top100"].shape == (1, len(BOUNDARIES), 100)
    tokens = reference["tokens"][0].long().tolist()
    generator = build_generator(
        model_dir=Path(__file__).parents[1],
        mesh_device=mesh_device,
        snapshot_path=os.environ["GPT_OSS_120B_SNAPSHOT"],
        tensor_cache_path=os.environ["TT_METAL_CACHE"],
        max_context_length=HF_CONTEXT_LENGTH,
        max_batch_size=1,
    )
    records = []
    output = Path(os.environ["GPT_OSS_120B_RESULTS"]) / "full-model-boundaries.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for index, boundary in enumerate(BOUNDARIES):
            candidates = reference["top100"][0, index].tolist()
            prefill = generator.generate(tokens[:boundary], 1, enable_trace=True, sampling_mode="device")[0]
            decoded = generator.generate(
                tokens[: boundary - 1],
                2,
                next_input=lambda step, predicted: tokens[boundary - 1],
                enable_trace=True,
                sampling_mode="device",
            )[1]
            evidence = generator.trace_evidence.to_dict()
            row = {
                "boundary": boundary,
                "prefill": {"token": prefill, **{f"top{k}": prefill in candidates[:k] for k in (1, 5, 100)}},
                "decode": {"token": decoded, **{f"top{k}": decoded in candidates[:k] for k in (1, 5, 100)}},
                "trace": evidence,
            }
            records.append(row)
            assert evidence["model_execute_submissions"] == 1
            assert evidence["sampling_execute_submissions"] == 1
            assert evidence["full_logits_readbacks"] == 0
        # The frozen 0.98 / 1.0 limits require all twelve rows to pass.
        for mode in ("prefill", "decode"):
            assert sum(row[mode]["top5"] for row in records) / len(records) >= 0.98, records
            assert all(row[mode]["top100"] for row in records), records
    finally:
        output.write_text(json.dumps({"checkpoint_revision": MODEL_REVISION, "cases": records}, indent=2) + "\n")
        generator.teardown()
