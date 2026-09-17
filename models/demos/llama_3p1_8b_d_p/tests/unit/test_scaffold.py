# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Adapter contract tests for ``llama_3p1_8b_d_p``. Device-free.

Deliberately narrow: these assert *behavior that can regress*, not restated constants. The dim
values live in ``reference/llama_3p1_8b_config.py`` and a test that re-typed them there would only
check that copy-paste worked.
"""

import subprocess
import sys

ADAPTER_MODULE = "models.demos.llama_3p1_8b_d_p.tt.runners.adapters.llama_3p1_8b"

# Anything in this set at adapter-import time breaks the H2D producers, which import the module only
# to read the registry. The prefill engine's docs make this a hard contract.
FORBIDDEN_AT_IMPORT = ("torch", "ttnn", "transformers", "safetensors")


def test_adapter_is_import_light():
    """Import the adapter in a clean interpreter and assert none of the heavy stacks came with it.

    This is the test most likely to earn its keep: the natural way to write this adapter is to put
    `import ttnn` or `from transformers import AutoConfig` at the top, and nothing else in the repo
    would complain.

    Runs in a subprocess because pytest's own collection has already imported torch by this point,
    so checking ``sys.modules`` in-process would prove nothing.
    """
    probe = (
        "import sys;"
        f"import {ADAPTER_MODULE};"
        f"leaked=[m for m in {FORBIDDEN_AT_IMPORT!r} if m in sys.modules];"
        "print(','.join(leaked))"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "", f"adapter import pulled in heavy modules: {out}"


def test_weight_cache_path_uses_sp_times_tp(tmp_path, monkeypatch):
    """`{name}_{arch}_{N}dev / {sp}x{tp}`, with N = sp*tp.

    N must NOT come from ``ttnn.get_num_devices()``: the runner calls this from ``_print_config``
    before ``open_mesh_device``, and with co-located migration workers ``GetNumAvailableDevices``
    can throw ``unordered_map::at`` and abort Gate 2.
    """
    from models.demos.llama_3p1_8b_d_p.tt.runners.adapters.llama_3p1_8b import Llama31PrefillAdapter

    monkeypatch.setenv("PREFILL_TTNN_CACHE", str(tmp_path))
    path = Llama31PrefillAdapter().weight_cache_path((4, 8))
    assert path is not None
    assert path.parent.name in ("llama_3p1_8b_bh_32dev", "llama_3p1_8b_wh_32dev")
    assert path.name == "4x8"
    assert path.is_dir()


def test_weight_cache_disabled_returns_none(monkeypatch):
    """An empty PREFILL_TTNN_CACHE means "no cache", not "cache in the cwd"."""
    from models.demos.llama_3p1_8b_d_p.tt.runners.adapters.llama_3p1_8b import Llama31PrefillAdapter

    monkeypatch.setenv("PREFILL_TTNN_CACHE", "")
    assert Llama31PrefillAdapter().weight_cache_path((4, 8)) is None


def test_bundled_hf_config_matches_the_dim_ssot():
    """The repo-bundled config.json the adapter defaults to really is Llama-3.1-8B.

    Guards the default path itself: if it is retargeted at another Llama, or the bundled file moves,
    this fails here rather than as a wrong-shape PCC several issues later.
    """
    import json
    from pathlib import Path

    from models.demos.llama_3p1_8b_d_p.tt.model_config import cross_check_hf_config
    from models.demos.llama_3p1_8b_d_p.tt.runners.adapters.llama_3p1_8b import Llama31PrefillAdapter

    config_json = Path(Llama31PrefillAdapter.hf_model_default) / "config.json"
    assert config_json.is_file(), f"bundled config missing: {config_json}"

    raw = json.loads(config_json.read_text())
    cross_check_hf_config(type("Cfg", (), raw))  # cross_check reads attributes, not keys

    # The RoPE frame is the one contract that survives every byte-level migration gate, so pin it
    # against the checkpoint rather than only against our own constants.
    assert raw["rope_scaling"]["rope_type"] == "llama3"
    assert raw["rope_scaling"]["factor"] == 8.0
    assert raw["rope_theta"] == 500000.0


def test_mesh_config_rejects_bad_axes_before_deriving(expect_error):
    """Axis validation must happen at the API boundary, not surface later as wrong sharding.

    `tp_axis=-1` is the case worth pinning: it indexes `mesh_shape` fine and satisfies the
    "TP spans the whole axis" check, so without an explicit guard it would construct a config whose
    SP axis is wrong rather than raising.
    """
    from models.demos.llama_3p1_8b_d_p.tt.config import MeshConfig

    with expect_error(ValueError, "tp_axis"):
        MeshConfig((4, 8), tp=8, tp_axis=-1)
    with expect_error(ValueError, "tp_axis"):
        MeshConfig((4, 8), tp=8, tp_axis=2)
    with expect_error(ValueError, "mesh_shape must be 2-D"):
        MeshConfig((4, 8, 2), tp=8)
    # TP must span the whole axis; a smaller TP would disagree with the mapper.
    with expect_error(ValueError, "must equal"):
        MeshConfig((4, 8), tp=4, tp_axis=1)


def test_mesh_config_axes_are_complementary():
    """SP and TP must never resolve to the same axis, on either orientation."""
    from models.demos.llama_3p1_8b_d_p.tt.config import MeshConfig

    cfg = MeshConfig((4, 8), tp=8, tp_axis=1)  # the target: TP on cols, SP on rows
    assert (cfg.tp_axis, cfg.sp_axis) == (1, 0)
    assert (cfg.tp, cfg.sp) == (8, 4)
    assert cfg.total_devices == 32
    assert cfg.shard_size(14336) == 1792 and cfg.shard_size(4096) == 512

    flipped = MeshConfig((8, 4), tp=8, tp_axis=0)  # TP on rows, SP on cols
    assert (flipped.tp_axis, flipped.sp_axis) == (0, 1)
    assert (flipped.tp, flipped.sp) == (8, 4)
