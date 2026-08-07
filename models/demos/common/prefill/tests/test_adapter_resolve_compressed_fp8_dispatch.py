# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Host-only truth-table test for ``PrefillModelAdapter.resolve_compressed_fp8_dispatch``.

The resolver is the rollout gate for a default-on production feature (FP8 MoE dispatch),
so every cell of its decision table — env unset / "0" / "1" / unrecognized ×
``supports_compressed_fp8_dispatch`` × architecture — is pinned here. No device needed:
the arch check is patched.
"""

import pytest

import models.common.utility_functions as utility_functions
from models.demos.common.prefill.adapter import PrefillModelAdapter

ENV_VAR = "PREFILL_COMPRESSED_FP8_DISPATCH"


class _StubAdapter(PrefillModelAdapter):
    """Minimal concrete adapter — only the resolver under test is exercised."""

    name = "stub"

    def load_hf_config(self):
        raise NotImplementedError

    def weight_cache_path(self, mesh_shape):
        raise NotImplementedError

    def allocate_kv_cache(self, *, mesh_device, hf_config, params):
        raise NotImplementedError

    def build_runtime(self, *, mesh_device, hf_config, params):
        raise NotImplementedError


class _SupportedAdapter(_StubAdapter):
    name = "stub_supported"
    supports_compressed_fp8_dispatch = True


@pytest.fixture
def patch_arch(monkeypatch):
    """Patch the shared is_blackhole() helper (resolved lazily inside the resolver)."""

    def _set(blackhole: bool):
        monkeypatch.setattr(utility_functions, "is_blackhole", lambda: blackhole)

    return _set


# Truth table: (env value or None, supports, is_blackhole) -> expected.
# The env var is a kill switch only: default TRUE for validated models on Blackhole,
# "0" disables, and no value can ENABLE fp8 for an unvalidated model or non-BH hardware.
TRUTH_TABLE = [
    # env unset: the default — on for validated models on Blackhole only
    (None, True, True, True),
    (None, True, False, False),
    (None, False, True, False),
    (None, False, False, False),
    # "0": kill switch everywhere
    ("0", True, True, False),
    ("0", True, False, False),
    ("0", False, True, False),
    ("0", False, False, False),
    # "1": documented no-op (same as unset) — it can never enable, only warn where it can't
    ("1", True, True, True),
    ("1", True, False, False),
    ("1", False, True, False),
    ("1", False, False, False),
    # unrecognized value: warns and is ignored (same as unset)
    ("true", True, True, True),
    ("yes", True, False, False),
    ("ON", False, True, False),
]


@pytest.mark.parametrize(
    "env_value, supports, blackhole, expected",
    TRUTH_TABLE,
    ids=[f"env={e or 'unset'}-supports={s}-bh={b}" for e, s, b, _ in TRUTH_TABLE],
)
def test_resolve_compressed_fp8_dispatch(monkeypatch, patch_arch, env_value, supports, blackhole, expected):
    if env_value is None:
        monkeypatch.delenv(ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(ENV_VAR, env_value)
    patch_arch(blackhole)

    adapter = _SupportedAdapter() if supports else _StubAdapter()
    assert adapter.resolve_compressed_fp8_dispatch() is expected
