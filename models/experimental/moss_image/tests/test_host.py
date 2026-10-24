# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch
from safetensors.torch import save_file

from models.experimental.moss_image.tests.assets import Capture, digest, metrics
from models.experimental.moss_image.tt.checkpoint import PIPELINE_SHA256, SafeWeights
from models.experimental.moss_image.tt.dit import rotary_metadata, time_metadata
from models.experimental.moss_image.tt.text_encoder import format_prompt
from models.experimental.moss_image.tt.vendor.pipeline_moss_image import _RoPE


def test_rotary_matches_official():
    for height, width, length in ((16, 16, 23), (64, 64, 1), (16, 24, 37)):
        expected = _RoPE(128, 10000.0)(height, width, length, torch.device("cpu")).bfloat16()
        assert torch.equal(rotary_metadata(height, width, length), expected)


def test_time_embedding_matches_official_phase():
    for timestep in (1000.0, 24.0, 853.25):
        phase = torch.tensor(timestep).double() * 10000 ** (-torch.arange(128, dtype=torch.float64) / 128)
        expected = torch.cat((phase.cos(), phase.sin())).float().bfloat16().reshape(1, 1, 256)
        assert torch.equal(time_metadata(timestep), expected)


def test_invalid_prompt_limits_fail_before_tokenization():
    with pytest.raises(
        ValueError, match="4096"
    ):  # allow-pytest.raises: host-only checks run without native TTNN fixtures
        format_prompt(None, "fox", 4097)
    with pytest.raises(
        TypeError, match="string"
    ):  # allow-pytest.raises: host-only checks run without native TTNN fixtures
        format_prompt(None, ["fox"])


def test_added_unindexed_shard_cannot_override(tmp_path):
    folder = tmp_path / "transformer"
    folder.mkdir()
    save_file({"weight": torch.ones(2, 2)}, folder / "part.safetensors")
    (folder / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "part.safetensors"}})
    )
    save_file({"weight": torch.zeros(2, 2)}, folder / "override.safetensors")
    assert torch.equal(SafeWeights(folder).get("weight"), torch.ones(2, 2))


@pytest.mark.parametrize("name", ["../part.safetensors", "/part.safetensors", "part.bin"])
def test_invalid_weight_index_paths(tmp_path, name):
    (tmp_path / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": name}})
    )
    with pytest.raises(
        ValueError, match="Invalid checkpoint shard path"
    ):  # allow-pytest.raises: host-only checks run without native TTNN fixtures
        SafeWeights(tmp_path)


def fake_capture(tmp_path):
    tensor = tmp_path / "tensor.pt"
    torch.save(torch.ones(2, 2), tensor)
    manifest = {
        "pipeline_sha256": PIPELINE_SHA256,
        "status": "complete",
        "guidance_scale": 3.0,
        "negative_skip_layers": [22],
        "checkpoint": {"declared_variant": "Base"},
        "arguments": {
            "variant": "Base",
            "height": 1024,
            "width": 1024,
            "steps": 50,
            "seed": 42,
            "prompt": "the quick brown fox jumps over the lazy dog",
        },
        "artifacts": {"tensor": {"path": "tensor.pt", "sha256": digest(tensor)}},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


@pytest.mark.parametrize("damage", ["missing", "tampered", "escape"])
def test_capture_file_integrity(tmp_path, damage):
    path, manifest = fake_capture(tmp_path)
    if damage == "escape":
        manifest["artifacts"]["tensor"]["path"] = "../tensor.pt"
        path.write_text(json.dumps(manifest))
    capture = Capture(tmp_path, digest(path), "Base")
    if damage == "missing":
        (tmp_path / "tensor.pt").unlink()
    elif damage == "tampered":
        (tmp_path / "tensor.pt").write_bytes(b"tampered")
    with pytest.raises(
        (ValueError, FileNotFoundError)
    ):  # allow-pytest.raises: host-only checks run without native TTNN fixtures
        capture.load("tensor")


def test_manifest_pin_and_guidance_mismatch(tmp_path):
    path, manifest = fake_capture(tmp_path)
    with pytest.raises(
        ValueError, match="manifest digest"
    ):  # allow-pytest.raises: host-only checks run without native TTNN fixtures
        Capture(tmp_path, "0" * 64, "Base")
    manifest["guidance_scale"] = 1.0
    path.write_text(json.dumps(manifest))
    with pytest.raises(
        ValueError, match="guidance"
    ):  # allow-pytest.raises: host-only checks run without native TTNN fixtures
        Capture(tmp_path, digest(path), "Base")


def test_pcc_does_not_hide_gain_error():
    reference = torch.arange(10).float()
    readings = metrics(reference * 1.1, reference)
    assert readings["pcc"] > 0.999999
    assert readings["relative_rms"] > 0.09


def test_nonfinite_comparison_fails():
    with pytest.raises(
        ValueError, match="Nonfinite"
    ):  # allow-pytest.raises: host-only checks run without native TTNN fixtures
        metrics(torch.tensor([float("nan")]), torch.ones(1))


@pytest.mark.parametrize(
    "overrides",
    [
        {"height": 255},
        {"width": 257},
        {"height": 256.0},
        {"steps": 0},
        {"steps": 1001},
        {"steps": 1.5},
        {"guidance_scale": float("nan")},
        {"guidance_scale": 0.5},
        {"prompt": ["fox"]},
    ],
)
def test_invalid_generation_options_fail_on_host(overrides):
    from models.experimental.moss_image.demo import validate_generation_options

    options = dict(height=1024, width=1024, steps=50, prompt="fox", negative_prompt="", guidance_scale=3.0)
    options.update(overrides)
    with pytest.raises((ValueError, TypeError)):  # allow-pytest.raises: pure host validation, no native fixtures
        validate_generation_options(**options)


@pytest.mark.parametrize("ci", [None, "true"])
def test_missing_cuda_reference_skips(tmp_path, monkeypatch, ci):
    from types import SimpleNamespace

    from models.experimental.moss_image.tests.conftest import moss_assets

    monkeypatch.setenv("MOSS_IMAGE_ASSET_ROOT", str(tmp_path))
    if ci is None:
        monkeypatch.delenv("CI", raising=False)
    else:
        monkeypatch.setenv("CI", ci)
    with pytest.raises(
        pytest.skip.Exception, match="Missing pinned MOSS CUDA reference"
    ):  # allow-pytest.raises: host-only fixture validation
        # Exercise the real fixture without requesting a hardware fixture.
        moss_assets.__wrapped__(SimpleNamespace(param="Base"))


def test_available_cuda_reference_loads(tmp_path):
    from models.experimental.moss_image.tests.conftest import _load_assets

    directory = tmp_path / "cuda-Base-1024-50"
    directory.mkdir()
    path, _ = fake_capture(directory)
    checkpoint = tmp_path / "checkpoints" / "Base"
    checkpoint.mkdir(parents=True)
    actual_checkpoint, capture = _load_assets(
        tmp_path, "Base", {"capture": directory.name, "manifest_sha256": digest(path)}
    )
    assert actual_checkpoint == checkpoint
    assert torch.equal(capture.load("tensor"), torch.ones(2, 2))


def test_missing_checkpoint_skips(tmp_path):
    from models.experimental.moss_image.tests.conftest import _load_assets

    path, _ = fake_capture(tmp_path)
    with pytest.raises(
        pytest.skip.Exception, match="Missing MOSS Base checkpoint"
    ):  # allow-pytest.raises: host-only fixture validation
        _load_assets(tmp_path, "Base", {"capture": ".", "manifest_sha256": digest(path)})


@pytest.mark.parametrize("checkpoint_available", [False, True])
def test_corrupt_available_reference_fails(tmp_path, checkpoint_available):
    from models.experimental.moss_image.tests.conftest import _load_assets

    path, _ = fake_capture(tmp_path)
    pin = {"capture": ".", "manifest_sha256": digest(path)}
    path.write_bytes(b"corrupt manifest")
    if checkpoint_available:
        (tmp_path / "checkpoints" / "Base").mkdir(parents=True)
    with pytest.raises(ValueError, match="manifest digest"):  # allow-pytest.raises: host asset validation
        _load_assets(tmp_path, "Base", pin)
