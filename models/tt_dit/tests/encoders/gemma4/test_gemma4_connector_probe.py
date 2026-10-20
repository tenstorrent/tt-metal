# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The connector probe: a connector cache that embeds a prompt to zeros or non-finite values is
moved aside and rebuilt from the checkpoint when it loads.

The host tests drive the probe with stand-ins for the device pieces. The device test loads a
known-bad cache (task #117's 4x8 ltx25_ic_fused connectors, kept as ``*.bad-t117``), named by
``LTX25_BAD_CONNECTOR_CACHE``: the directory that holds ``video_connector`` and
``audio_connector``, each with one key directory inside.
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[6]))

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.encoders.gemma4 import encoder_pair as ep
from models.tt_dit.encoders.gemma4.encoder_pair import Gemma4TokenizerEncoderPair
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import cache as cache_module
from models.tt_dit.utils.test import line_params_req_exact_devices

LTX25 = os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.5")
TEXT_ENCODER = os.environ.get(
    "GEMMA4_CHECKPOINT", f"{LTX25}/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
)
TRANSFORMER = os.environ.get(
    "LTX25_TRANSFORMER", f"{LTX25}/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
)
BAD_CACHE = os.environ.get("LTX25_BAD_CONNECTOR_CACHE")

GOOD = (torch.full((1, 8, 4096), 0.25), torch.full((1, 8, 2048), 0.25))


def test_broken_axes():
    zeros = (torch.zeros(1, 8, 4096), torch.zeros(1, 8, 2048))
    assert ep._broken_axes(GOOD) == []
    assert ep._broken_axes(zeros) == ["video", "audio"]
    assert ep._broken_axes((GOOD[0], torch.zeros(1, 8, 2048))) == ["audio"]  # the #117 audio cache
    assert ep._broken_axes((GOOD[0], torch.full((1, 8, 2048), float("inf")))) == ["audio"]
    assert ep._broken_axes((torch.full((1, 8, 4096), float("nan")), None)) == ["video"]


class _Connector:
    def __init__(self):
        self.deallocated = False

    def deallocate_weights(self):
        self.deallocated = True


class _Loaded:
    def is_loaded(self):
        return True


def _host_pair(tmp_path, monkeypatch, outputs):
    """A pair whose device encode returns ``outputs`` in turn and whose cache dirs live under tmp_path."""
    pair = Gemma4TokenizerEncoderPair.__new__(Gemma4TokenizerEncoderPair)
    pair.gemma_encoder = _Loaded()
    pair.video_connector, pair.audio_connector = _Connector(), _Connector()
    pair.parallel_config = None
    pair.mesh_device = type("Mesh", (), {"shape": (4, 8)})()
    pair.transformer_checkpoint = "ckpt.safetensors"
    pair._connectors_probed = False
    pair._encode_one = lambda prompt, *, traced: (assert_untraced(traced), outputs.pop(0))[1]
    monkeypatch.setattr(ttnn, "distributed_context_barrier", lambda: None)
    monkeypatch.setattr(
        cache_module,
        "model_cache_dir",
        lambda *, model_name, subfolder, dtype, **_: tmp_path / model_name / subfolder / f"TP8_1_mesh4x8_{dtype}",
    )
    for axis in ("video", "audio"):
        d = tmp_path / "conn" / f"{axis}_connector" / "TP8_1_mesh4x8_float32"
        d.mkdir(parents=True)
        (d / "cache_dict.json").write_text("{}")
    return pair


def assert_untraced(traced):
    assert traced is False, "the probe must not capture a trace"


def test_probe_leaves_a_good_cache_alone(tmp_path, monkeypatch):
    pair = _host_pair(tmp_path, monkeypatch, [GOOD])
    rebuilt = []
    pair._probe_connectors(rebuilt.append, "conn")
    assert rebuilt == [] and pair._connectors_probed
    assert not list(tmp_path.glob("conn/*/*.bad-*"))


def test_probe_moves_a_zero_cache_aside_and_rebuilds(tmp_path, monkeypatch):
    zero_video = (torch.zeros(1, 8, 4096), GOOD[1])
    pair = _host_pair(tmp_path, monkeypatch, [zero_video, GOOD])
    old_video = pair.video_connector
    rebuilt = []
    pair._probe_connectors(lambda axis: rebuilt.append(axis) or _Connector(), "conn")
    assert rebuilt == ["video"] and pair._connectors_probed
    assert old_video.deallocated and not pair.audio_connector.deallocated
    assert [p.parent.name for p in tmp_path.glob("conn/*/TP8_1_mesh4x8_float32.bad-*")] == ["video_connector"]
    assert not (tmp_path / "conn/video_connector/TP8_1_mesh4x8_float32").exists()
    assert (tmp_path / "conn/audio_connector/TP8_1_mesh4x8_float32").exists()


def test_probe_fails_loudly_when_the_rebuild_is_still_broken(tmp_path, monkeypatch, expect_error):
    zeros = (torch.zeros(1, 8, 4096), torch.zeros(1, 8, 2048))
    pair = _host_pair(tmp_path, monkeypatch, [zeros, zeros])
    with expect_error(RuntimeError, "still embeds"):
        pair._probe_connectors(lambda axis: _Connector(), "conn")
    assert not pair._connectors_probed


def test_probe_skips_a_kernel_capture_pass(tmp_path, monkeypatch):
    pair = _host_pair(tmp_path, monkeypatch, [])
    monkeypatch.setenv("TT_METAL_KERNEL_CAPTURE_ONLY", "1")
    pair._probe_connectors(lambda axis: pytest.fail("rebuilt in a capture-only pass"), "conn")


@pytest.mark.timeout(1800)  # rebuilding both connectors from the checkpoint takes minutes
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "device_params",
    [{**line_params_req_exact_devices, "l1_small_size": 8192}],
    indirect=["device_params"],
)
def test_bad_connector_cache_is_rebuilt_on_load(*, mesh_device, tmp_path, monkeypatch):
    """Serve the #117 cache through a scratch cache root: the probe must move it aside, rebuild
    it from the checkpoint, and the pair must then embed a prompt to finite, non-degenerate values."""
    if not BAD_CACHE:
        pytest.skip("LTX25_BAD_CONNECTOR_CACHE not set")
    for path in (TEXT_ENCODER, TRANSFORMER, BAD_CACHE):
        if not Path(path).exists():
            pytest.skip(f"missing {path}")

    connector_name = Path(TRANSFORMER).name.removesuffix(".safetensors")
    root = tmp_path / "cache"
    root.mkdir()
    # Everything but this checkpoint's connectors comes from the real cache, so Gemma is not re-tilized.
    real_root = os.environ.get("TT_DIT_CACHE_DIR")
    if real_root:
        for entry in Path(real_root).iterdir():
            if entry.name != connector_name:
                (root / entry.name).symlink_to(entry)
    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(root))

    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=mesh_device.shape[1], mesh_axis=1),
    )
    bad_files = {}
    for axis in ("video", "audio"):
        (src,) = [d for d in (Path(BAD_CACHE) / f"{axis}_connector").iterdir() if d.is_dir()]
        key_dir = cache_module.model_cache_dir(
            model_name=connector_name,
            subfolder=f"{axis}_connector",
            parallel_config=parallel_config,
            mesh_shape=tuple(mesh_device.shape),
            mesh_device=mesh_device,
            dtype="float32",
        )
        key_dir.mkdir(parents=True)
        for f in src.iterdir():
            (key_dir / f.name).symlink_to(f)
        bad_files[axis] = (key_dir, sorted(src.iterdir()))

    pair = Gemma4TokenizerEncoderPair(
        TEXT_ENCODER,
        mesh_device=mesh_device,
        ccl_manager=CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear),
        parallel_config=parallel_config,
        transformer_checkpoint=TRANSFORMER,
    )
    pair.ensure_loaded()

    for axis, (key_dir, sources) in bad_files.items():
        aside = list(key_dir.parent.glob(f"{key_dir.name}.bad-*"))
        assert all(f.exists() for f in sources), "the original bad cache was touched"
        assert len(aside) == 1, f"the bad {axis} connector cache was not moved aside"
        assert all(p.is_symlink() for p in aside[0].iterdir()), f"the {axis} cache moved aside is not the bad one"
        assert (key_dir / cache_module.CACHE_DICT_FILE).is_file(), f"the {axis} connector cache was not rebuilt"
        assert not any(p.is_symlink() for p in key_dir.iterdir()), f"the {axis} cache was not rewritten"

    video, audio = pair.encode(["a cat playing a piano in a jazz bar"])[0]
    logger.info(f"after rebuild: video absmean {video.abs().mean():.4f} | audio absmean {audio.abs().mean():.4f}")
    assert ep._broken_axes((video, audio)) == []
    assert video.std() > 0.01 and audio.std() > 0.01
