# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""F271: BRINGUP_CHUNK sets a measurement run's chunk (the chunk-size sweep) without tripping the owner's
serving.max_chunk check, which reads the chunk the spec writes."""

from models.demos.common.bringup.core.spec import CHUNK_ENV, Spec


def _spec():
    return Spec({"model": "m", "target": {"seq": 16384, "chunk": 4096}})


def test_chunk_env_overrides_target(monkeypatch):
    monkeypatch.delenv(CHUNK_ENV, raising=False)
    assert int(_spec().get("target.chunk")) == 4096
    monkeypatch.setenv(CHUNK_ENV, "8192")
    s = _spec()
    assert int(s.get("target.chunk")) == 8192
    assert s.raw["target"]["chunk"] == 4096  # the written spec is untouched
    assert int(s.get("target.seq")) == 16384
