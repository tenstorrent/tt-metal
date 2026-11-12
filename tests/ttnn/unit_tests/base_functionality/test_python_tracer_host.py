# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Host-only tracer tests: no device is opened."""

import json

import pytest

import ttnn
import ttnn.operation_tracer


class _StrRaisesWithDict:
    def __str__(self):
        raise TypeError("no caster")


class _StrRaisesNoDict:
    __slots__ = ()

    def __str__(self):
        raise TypeError("no caster")


def test_paged_cache_geometry_override_repr():
    assert str(ttnn.PagedCacheGeometryOverride()).startswith("PagedCacheGeometryOverride(")


def test_sdpa_program_config_repr():
    config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        q_chunk_size=32,
        k_chunk_size=32,
    )
    assert str(config).startswith("SDPAProgramConfig(")


@pytest.fixture
def warnings_seen(monkeypatch):
    """Isolate the warn-once set and record warning calls."""
    monkeypatch.setattr(ttnn.operation_tracer, "_STR_FAILURE_WARNED_TYPES", set())
    calls = []
    monkeypatch.setattr(ttnn.operation_tracer.logger, "warning", lambda msg, *a, **kw: calls.append(msg))
    return calls


@pytest.mark.parametrize(
    "value_type,key",
    [
        (_StrRaisesWithDict, "repr"),
        (_StrRaisesNoDict, "value"),
    ],
)
def test_unserializable_str_falls_back_to_placeholder(tmp_path, warnings_seen, value_type, key):
    for _ in range(2):
        ttnn.operation_tracer.serialize_operation_parameters(
            "ttnn.fake_op",
            (value_type(),),
            {"cfg": value_type()},
            tmp_path,
            serialize_tensor_values=False,
        )

    trace_files = sorted(tmp_path.glob("*.json"))
    assert len(trace_files) == 2
    placeholder = f"<unserializable {value_type.__name__}>"
    for trace_file in trace_files:
        data = json.loads(trace_file.read_text())
        assert data["args"][0]["value"] == {"type": value_type.__name__, key: placeholder}
        assert data["kwargs"]["cfg"] == {"type": value_type.__name__, key: placeholder}

    # Four failing str() calls on one type warn exactly once.
    assert len(warnings_seen) == 1
    assert value_type.__name__ in warnings_seen[0]
