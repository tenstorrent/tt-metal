# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1: owner combine (k > 8, row-form partials) and compact gamma pages.

Pins the owner topology (member count forced through the descriptor's FORCE_K knob where the cost rule would pick a
symmetric k), incl. k not a multiple of 32 (zero-filled gather rows), several gather tiles (k > 32), multi-row blocks
(owners of several rows), rms / LN FPU / LN SFPU paths and both DEST widths.
"""

from __future__ import annotations

import pytest
import ttnn

import ttnn.operations.all_gather_rms_norm.all_gather_rms_norm_program_descriptor as pd

from .test_all_gather_rms_norm import _DEVICE_PARAMS, run_case, R, L, RING, LINE

CASES = [
    # (id, forced k or None, case kwargs)
    ("decode_rms_auto", None, dict(shape=(1, 1, 32, 2048), gamma_layout=R)),
    ("decode_ln_fpu_auto", None, dict(shape=(1, 1, 32, 2048), norm_type="layer_norm", gamma_layout=L, with_beta=True)),
    ("rms_k12_pad_rows", 12, dict(shape=(1, 1, 64, 640), gamma_layout=L)),
    ("ln_k20_pad_rows", 20, dict(shape=(1, 1, 64, 640), norm_type="layer_norm", gamma_layout=R, with_beta=True)),
    ("rms_k40_two_gather_tiles", 40, dict(shape=(1, 1, 32, 1280), gamma_layout=R)),
    ("ln_k40_two_gather_tiles", 40, dict(shape=(1, 1, 32, 1280), norm_type="layer_norm", gamma_layout=L)),
    ("rms_multirow_k16", 16, dict(shape=(1, 1, 320, 1024), gamma_layout=R)),
    ("ln_multirow_k16", 16, dict(shape=(1, 1, 320, 1024), norm_type="layer_norm", gamma_layout=R, with_beta=True)),
    (
        "ln_sfpu_fp32_big_offset_k16",
        16,
        dict(
            shape=(1, 1, 32, 1024),
            dtype=ttnn.float32,
            norm_type="layer_norm",
            gamma_layout=R,
            gamma_dtype=ttnn.float32,
            with_beta=True,
            offset=50.0,
        ),
    ),
    ("ln_bf16_big_offset_k16", 16, dict(shape=(1, 1, 64, 1024), norm_type="layer_norm", offset=50.0)),
    ("rms_acc16_k16", 16, dict(shape=(1, 1, 64, 1024), gamma_layout=L, fp32_dest_acc_en=False)),
    ("ln_nonaligned_k12", 12, dict(shape=(1, 1, 32, 1000), norm_type="layer_norm", gamma_layout=R, with_beta=True)),
    ("rms_nonaligned_k12", 12, dict(shape=(1, 1, 32, 1000), gamma_layout=L)),
]


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["mesh1x4"], indirect=True)
@pytest.mark.parametrize("force_k,case", [(c[1], c[2]) for c in CASES], ids=[c[0] for c in CASES])
def test_owner_combine(mesh_device, force_k, case, monkeypatch):
    monkeypatch.setattr(pd, "FORCE_K", force_k)
    run_case(mesh_device, cluster_axis=1, **case)
    info = pd.LAST_INFO
    assert info["owner"] and info["k"] > pd.K_SYM_MAX, f"owner path not taken: {info}"
    if force_k is not None:
        assert info["k"] == force_k
