# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device-free checks of the MINIMAX_H3_SDPA_MM_FIDELITY knob for the ring SDPA matmuls."""

from types import SimpleNamespace

import pytest

import ttnn

from ....models.transformers.minimax_h3 import attention_minimax_h3 as am


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, None), ("", None), ("lofi", "LoFi"), (" LoFi ", "LoFi"), ("hifi2", "HiFi2"), ("hifi4", "HiFi4")],
)
def test_sdpa_matmul_fidelity_parses(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(am.SDPA_MM_FIDELITY_ENV, raising=False)
    else:
        monkeypatch.setenv(am.SDPA_MM_FIDELITY_ENV, raw)
    got = am.sdpa_matmul_fidelity()
    assert got == (None if expected is None else getattr(ttnn.MathFidelity, expected))


def test_sdpa_matmul_fidelity_rejects_unknown(monkeypatch, expect_error):
    monkeypatch.setenv(am.SDPA_MM_FIDELITY_ENV, "lofi2")
    with expect_error(ValueError, "MINIMAX_H3_SDPA_MM_FIDELITY"):
        am.sdpa_matmul_fidelity()


@pytest.mark.parametrize("ring", [True, False])
def test_sdpa_program_config_carries_fidelity_on_ring_only(monkeypatch, ring):
    built = []
    monkeypatch.setattr(ttnn, "SDPAProgramConfig", lambda **kw: built.append(kw) or kw)
    attn = object.__new__(am.MiniMaxH3Attention)
    for name, value in dict(
        measured_sdpa_chunk_sizes={},
        sdpa_worker_grid=(11, 10),
        full_grid=SimpleNamespace(x=12, y=10),
        _sdpa_program_configs={},
        sdpa_matmul_fidelity=ttnn.MathFidelity.LoFi,
    ).items():
        object.__setattr__(attn, name, value)
    attn._sdpa_program_config(4768, ring=ring)
    assert built[0]["matmul_math_fidelity"] == (ttnn.MathFidelity.LoFi if ring else None)
