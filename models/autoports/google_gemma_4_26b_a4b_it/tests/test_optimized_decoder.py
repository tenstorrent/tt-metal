# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight default-path coverage, including mixed expert-batch tails."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from models.autoports.google_gemma_4_26b_a4b_it.tests import run_decoder, run_optimized_contract


@pytest.fixture(scope="module")
def text_activations(tmp_path_factory):
    root = Path(__file__).parents[1] / "doc/optimized_decoder"
    paths = {layer: root / f"actual_text_layer{layer}_4096_128.pt" for layer in (0, 5)}
    if not all(path.exists() for path in paths.values()):
        root = tmp_path_factory.mktemp("text_activations")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "models.autoports.google_gemma_4_26b_a4b_it.tests.create_optimized_activation_fixture",
                "--output-dir",
                str(root),
                "--case",
                "128",
                "1",
                "--allow-tokenizer-download",
            ],
            check=True,
        )
        paths = {layer: root / f"actual_text_layer{layer}_128_1.pt" for layer in (0, 5)}
    return {layer: (path, torch.load(path, map_location="cpu", weights_only=True)) for layer, path in paths.items()}


@pytest.mark.parametrize("layer", [0, 5], ids=["sliding_attention", "full_attention"])
@pytest.mark.parametrize("length", [33, 65])
def test_optimized_real_weights(layer, length, tmp_path, monkeypatch, text_activations):
    source_path, source = text_activations[layer]
    metadata = dict(source["metadata"], length=length, steps=8)
    metadata["slice_source"] = str(source_path)
    metadata["slice_source_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    fixture_path = tmp_path / "inputs.pt"
    torch.save(
        {
            "metadata": metadata,
            "prefill": source["prefill"][:, :length].clone(),
            "decode": source["prefill"][:, length : length + 8].clone(),
        },
        fixture_path,
    )
    outputs = {}
    for decoder in ("fused", "optimized") if length == 65 else ("optimized",):
        report, tensors = tmp_path / f"{decoder}.json", tmp_path / f"{decoder}.pt"
        selector = ["--contract", "run_decoder"] if decoder == "optimized" else ["--decoder", "fused"]
        monkeypatch.setattr(
            "sys.argv",
            [
                "decoder_contract",
                *selector,
                "--layer",
                str(layer),
                "--length",
                str(length),
                "--real",
                "--input-fixture",
                str(fixture_path),
                "--decode",
                "--steps",
                "8",
                "--verify-program-cache",
                "--output",
                str(report),
                "--save-output-tensors",
                str(tensors),
            ],
        )
        (run_optimized_contract.main if decoder == "optimized" else run_decoder.main)()
        result = json.loads(report.read_text())
        assert result["decoder"] == decoder
        assert result["passed"] and result["decode"]["passed"] and result["decode"]["repeated_equal"]
        assert result["runtime_prefill_audit"] == result["decode"]["runtime_decode_audit"] == "clean"
        if decoder == "optimized":
            assert result["functional_fallback"] == "forbidden"
        outputs[decoder] = torch.load(tensors, weights_only=True)
    if length == 65:
        pairs = [(outputs["fused"]["prefill"], outputs["optimized"]["prefill"])]
        assert len(outputs["fused"]["decode"]) == len(outputs["optimized"]["decode"]) == 8
        pairs.extend(zip(outputs["fused"]["decode"], outputs["optimized"]["decode"]))
        for reference, actual in pairs:
            assert torch.isfinite(reference).all() and torch.isfinite(actual).all()
            pcc = torch.corrcoef(torch.stack((reference.flatten().double(), actual.flatten().double())))[0, 1]
            assert pcc >= 0.995
