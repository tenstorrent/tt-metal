# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2.5 conv VAE: split-file key mapping and path resolution (host only)."""

import json
import os
import struct

import pytest

from models.tt_dit.models.vae.vae_ltx import vae_key_map
from models.tt_dit.utils import ltx as ltx_utils

PCS = ["per_channel_statistics.mean-of-means", "per_channel_statistics.std-of-means"]
VAE = ["decoder.conv_in.conv.weight", "decoder.up_blocks.0.res_blocks.0.conv1.conv.bias", "encoder.conv_in.conv.weight"]


def test_split_and_monolith_keys_map_alike():
    monolith = [f"vae.{k}" for k in VAE + PCS] + ["model.diffusion_model.x", "audio_vae.decoder.conv_in.conv.weight"]
    split = VAE + PCS
    for part in ("decoder", "encoder"):
        want = {k.removeprefix(f"{part}."): None for k in VAE if k.startswith(f"{part}.")} | dict.fromkeys(PCS)
        assert set(vae_key_map(monolith, part).values()) == set(want)
        assert set(vae_key_map(split, part).values()) == set(want)


def test_monolith_ignores_bare_keys_of_other_components():
    keys = ["vae.decoder.conv_in.conv.weight", "decoder.conv_in.conv.weight"]
    assert vae_key_map(keys, "decoder") == {"vae.decoder.conv_in.conv.weight": "conv_in.conv.weight"}


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    for var in ("LTX25_VIDEO_VAE", "LTX25_VAE_FALLBACK_23"):
        monkeypatch.delenv(var, raising=False)
    root = tmp_path / "ltx-2.5"
    root.mkdir()
    monkeypatch.setenv("LTX25_ROOT", str(root))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(ltx_utils, "LTX25_VIDEO_VAE_CONV_DEFAULT", str(tmp_path / "default" / "conv.safetensors"))
    monkeypatch.setattr(ltx_utils, "default_ltx_checkpoint", lambda name: str(tmp_path / "hub" / name))
    return tmp_path


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return str(path)


def test_conv_vae_resolution_order(isolated, monkeypatch):
    default = _touch(isolated / "default" / "conv.safetensors")
    assert ltx_utils.default_ltx25_video_vae() == default
    in_root = _touch(isolated / "ltx-2.5" / ltx_utils.LTX25_VIDEO_VAE_CONV)
    assert ltx_utils.default_ltx25_video_vae() == in_root
    explicit = _touch(isolated / "explicit.safetensors")
    monkeypatch.setenv("LTX25_VIDEO_VAE", explicit)
    assert ltx_utils.default_ltx25_video_vae() == explicit


def test_missing_explicit_conv_vae_raises(isolated, monkeypatch, expect_error):
    monkeypatch.setenv("LTX25_VIDEO_VAE", str(isolated / "absent.safetensors"))
    with expect_error(FileNotFoundError, "LTX25_VIDEO_VAE"):
        ltx_utils.default_ltx25_video_vae()


def test_ltx23_fallback_needs_opt_in(isolated, monkeypatch):
    monolith = _touch(isolated / "home" / ".cache" / "ltx-checkpoints" / "ltx-2.3-22b-distilled-1.1.safetensors")
    assert ltx_utils.default_ltx25_video_vae() is None
    monkeypatch.setenv("LTX25_VAE_FALLBACK_23", "1")
    assert ltx_utils.default_ltx25_video_vae() == monolith


def _header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


CONV = os.environ.get("LTX25_VIDEO_VAE", ltx_utils.LTX25_VIDEO_VAE_CONV_DEFAULT)
MONOLITH = os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.3-22b-distilled-1.1.safetensors")


@pytest.mark.skipif(not (os.path.isfile(CONV) and os.path.isfile(MONOLITH)), reason="needs both real VAE files")
def test_real_ltx25_conv_vae_matches_ltx23_layout():
    conv, mono = _header(CONV), _header(MONOLITH)
    for part in ("decoder", "encoder"):
        a = {short: conv[k]["shape"] for k, short in vae_key_map(conv, part).items()}
        b = {short: mono[k]["shape"] for k, short in vae_key_map(mono, part).items()}
        assert a and a == b
