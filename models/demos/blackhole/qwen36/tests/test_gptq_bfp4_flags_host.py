# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host tests (ttnn importable, no device) for the GPTQ-BFP4 wiring: QWEN36_GPTQ_BFP4 preset defaults,
QWEN36_BFP4_GDN_OUT, the GPTQ tensor-cache dir and the TP guard. Also checks that ttnn's own host-side bfloat4_b
conversion leaves GPTQ output unchanged (the fixed point the offline pipeline relies on).

Run: pytest models/demos/blackhole/qwen36/tests/test_gptq_bfp4_flags_host.py -q
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.qwen36.tt import tp_common as tpc  # noqa: E402
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs  # noqa: E402
from models.demos.blackhole.qwen36.utils import gptq_bfp4 as G  # noqa: E402

FLAGS = [
    "QWEN36_GPTQ_BFP4",
    "QWEN36_BFP4_GDN_IN",
    "QWEN36_BFP4_GDN_OUT",
    "QWEN36_BFP4_MLP_DOWN",
    "QWEN36_MLP_GATEUP_BFP8",
    "QWEN36_BFP4_ATTN",
]


@pytest.fixture
def clean_env(monkeypatch):
    for f in FLAGS:
        monkeypatch.delenv(f, raising=False)
    return monkeypatch


def _state():
    return dict(
        gdn_in=tpc.bfp4_gdn_in_enabled(),
        gdn_out=tpc.bfp4_gdn_out_enabled(),
        down=tpc.resolve_mlp_down_dtype() == ttnn.bfloat4_b,
        gateup=tpc.resolve_mlp_gateup_dtype() == ttnn.bfloat4_b,
        attn=tpc.bfp4_attn_enabled(),
    )


def test_defaults_unchanged_without_preset(clean_env):
    assert _state() == dict(gdn_in=False, gdn_out=False, down=False, gateup=False, attn=False)


def test_preset_turns_on_mlp_and_gdn_bfp4_only(clean_env):
    clean_env.setenv("QWEN36_GPTQ_BFP4", "1")
    assert _state() == dict(gdn_in=True, gdn_out=True, down=True, gateup=True, attn=False)


def test_explicit_flags_beat_preset(clean_env):
    clean_env.setenv("QWEN36_GPTQ_BFP4", "1")
    clean_env.setenv("QWEN36_BFP4_GDN_OUT", "0")
    clean_env.setenv("QWEN36_MLP_GATEUP_BFP8", "1")
    assert _state() == dict(gdn_in=True, gdn_out=False, down=True, gateup=False, attn=False)


def test_gdn_out_flag_alone(clean_env):
    clean_env.setenv("QWEN36_BFP4_GDN_OUT", "1")
    assert _state() == dict(gdn_in=False, gdn_out=True, down=False, gateup=False, attn=False)


def _write_marker(d, tp):
    Path(d).mkdir(parents=True, exist_ok=True)
    (Path(d) / G.MARKER_NAME).write_text(json.dumps({"format": G.MARKER_FORMAT, "tp": tp, "targets": ["mlp_down"]}))


def _stub(ckpt, num_devices=8):
    return SimpleNamespace(
        checkpoint_dir=str(ckpt), num_devices=num_devices, cluster_shape=(1, num_devices), weight_dtype=ttnn.bfloat8_b
    )


def test_cache_dir_separate_for_gptq_checkpoint(clean_env, tmp_path):
    plain, gq = _stub(tmp_path / "plain"), _stub(tmp_path / "gptq")
    _write_marker(gq.checkpoint_dir, 8)
    for s in (plain, gq):
        s.model_cache_path = tmp_path / "cache"
        Qwen36ModelArgs._check_gptq_checkpoint(s, mesh_device=object())
    p0 = Qwen36ModelArgs.weight_cache_path(plain)
    p1 = Qwen36ModelArgs.weight_cache_path(gq)
    assert p0 == tmp_path / "cache" / "tensor_cache_bfp8_mesh1x8"  # unchanged for normal checkpoints
    assert p1.parent == p0.parent and p1.name.startswith("tensor_cache_bfp8_mesh1x8_gptq") and p1 != p0


def test_preset_guards(clean_env, tmp_path):
    clean_env.setenv("QWEN36_GPTQ_BFP4", "1")
    with pytest.raises(ValueError, match="needs a GPTQ checkpoint"):
        Qwen36ModelArgs._check_gptq_checkpoint(_stub(tmp_path / "plain"), mesh_device=object())
    _write_marker(tmp_path / "g4", 4)
    with pytest.raises(ValueError, match="TP=4"):
        Qwen36ModelArgs._check_gptq_checkpoint(_stub(tmp_path / "g4", 8), mesh_device=object())
    _write_marker(tmp_path / "g8", 8)
    Qwen36ModelArgs._check_gptq_checkpoint(_stub(tmp_path / "g8", 8), mesh_device=object())
    # TP=1 (single device) is a mismatch too: the TP8 checkpoint's per-shard GPTQ groups do not apply.
    with pytest.raises(ValueError, match="running TP=1"):
        Qwen36ModelArgs._check_gptq_checkpoint(_stub(tmp_path / "g8", 1), mesh_device=object())


@pytest.mark.parametrize("exp_search", [False, True])
def test_ttnn_bfp4_conversion_is_identity_on_gptq_output(exp_search):
    """Real ttnn host conversion (no device) of a GPTQ-quantized [K, N] weight: to_torch(from_torch(bfloat4_b)) == input."""
    g = torch.Generator().manual_seed(0)
    N, K = 256, 128
    X = torch.randn(1024, K, generator=g) * torch.exp(torch.randn(K, generator=g))
    W = torch.randn(N, K, generator=g) * torch.exp(torch.randn(N, 1, generator=g))
    Q = G.gptq_quantize(W, X.t() @ X, G.consecutive_groups(N), blocksize=32, exp_search=exp_search).to(torch.bfloat16)
    w_kn = Q.t().contiguous()  # what shard_w hands to ttnn
    back = ttnn.to_torch(ttnn.from_torch(w_kn, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT))
    assert torch.equal(back.to(torch.bfloat16), w_kn)
