# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F131: F128's limits at run time: the contract's producer keys, the removed thresholds, the quick profile, the
models' runner contract tests. CPU only."""

import ast
import os
from pathlib import Path

import pytest

from models.demos.common.bringup.selftest.conftest import got
from models.demos.common.bringup.selftest.test_device_side import gspec, noise  # noqa: F401 (fixtures)
from models.demos.common.bringup.testing import contract, harness

ROOT = Path(contract.__file__).resolve().parents[1]
MODELS = ROOT.parents[1]
DRIFT = {
    f"contract_state_{n}_L{i:02d}": {"pcc": d, "pcc_res": 0.99999}
    for n, d in (("key", 0.999), ("value", 0.998), ("index_key", 0.99))
    for i in (0, 1)
}


class _Spec:
    def __init__(self, hooks=None):
        self._hooks = hooks or object()

    def hooks(self):
        return self._hooks


def test_producer_keys_map_to_golden_state_names(fx):
    """Gemma / MiMo producers return k / v, their goldens hold key / value: the limit comes from the mapped state."""
    fx()
    failed = []
    contract.producer_limits(_Spec(), object(), {"k": 0.997, "v": 0.99}, DRIFT, failed)
    m = got()
    assert m["pcc_producer_kv_k_limit"] == pytest.approx(1 - 4 * 0.001)  # F135: calib^2 on a PCC's miss
    assert m["pcc_producer_kv_v_limit"] == pytest.approx(1 - 4 * 0.002)
    assert len(failed) == 1 and "PCC v (value)" in failed[0]


def test_an_adapter_names_its_own_keys_and_an_unmapped_key_stops():
    class Adapter:
        @staticmethod
        def kv_state_names():
            return {"index": "index_key"}

    failed = []
    contract.producer_limits(_Spec(), Adapter(), {"index": 0.999, "mtp": 0.999}, DRIFT, failed)
    assert len(failed) == 1 and "'mtp'" in failed[0] and "kv_state_names" in failed[0]
    # a key that is a state name maps to itself; "index" alone is not one ("index_key" is)
    failed = []
    contract.producer_limits(_Spec(), object(), {"index_key": 0.999, "index": 0.999}, DRIFT, failed)
    assert len(failed) == 1 and "'index'" in failed[0]


def test_a_removed_threshold_names_its_replacement(fx):
    from models.demos.common.bringup.core.spec import Spec

    s = Spec.load(fx(thresholds={"state": 0.97}))
    for key in ("state", "layer", "final_hidden", "top5"):
        with pytest.raises(KeyError, match="F128"):
            harness.threshold(s, key)


def test_no_runner_contract_test_takes_a_fixed_threshold():
    """Every model's tests/bringup/contract/*.py: the KV limit is the drift's (e2e_limits), the server's own floors may
    stay as the server's rules; harness.threshold() is never called."""
    bad = []
    for f in sorted(MODELS.glob("*/tests/bringup/contract/*.py")):
        for n in ast.walk(ast.parse(f.read_text())):
            if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", "")) == "threshold":
                bad.append(f"{f.relative_to(MODELS)}:{n.lineno}")
    assert not bad, bad


def test_the_quick_profile_records_layer_and_state_limits(gspec, noise, monkeypatch):  # noqa: F811
    """F131: the quick A/B's profile records every layer's and every state's limit (the ladder's), so a `>= limit`
    gate can pass on it."""
    from models.demos.common.bringup.testing import profile

    s = gspec()
    noise(1e-3)
    monkeypatch.setenv("BRINGUP_PROFILE_WALL", "1")
    monkeypatch.setenv("BRINGUP_QUICK", "1")
    monkeypatch.delenv("TT_METAL_DEVICE_PROFILER", raising=False)
    monkeypatch.setattr(profile, "WALL_RUNS", 2)
    monkeypatch.setattr(profile, "WALL_WARM", 1)
    monkeypatch.setattr(profile.accuracy_guard, "check", lambda s: None)
    profile.run_profile(s, None)
    m = got()
    for i in range(3):
        assert m[f"err_layer_L{i:02d}"] <= m[f"err_layer_L{i:02d}_limit"] and f"pcc_layer_L{i:02d}" in m
        assert f"err_state_key_L{i:02d}_limit" in m and f"err_state_value_L{i:02d}" in m


def test_a_profile_at_a_position_records_timing_with_its_suffix_and_the_host_profile(
    gspec, noise, monkeypatch
):  # noqa: F811
    """F230 (owner 2026-10-30): one chunk at a token position (no golden there: zero prefix, random tokens, nothing
    compared), its warm wall and HOST_RUNS runs under cProfile, every metric named _p<pos> so one gate holds several
    positions; the raw stats beside the results."""
    from models.demos.common.bringup.core import metrics as M
    from models.demos.common.bringup.testing import profile

    s = gspec()
    noise(1e-3)
    from models.demos.common.bringup.selftest.fixture_model import V

    s.data.setdefault("checkpoint", {}).setdefault("config", {})["vocab_size"] = V  # the toy's vocabulary
    pos = 2 * int(s.get("target.chunk"))
    monkeypatch.setenv("BRINGUP_POS", str(pos))
    monkeypatch.setenv("BRINGUP_PROFILE_WALL", "1")
    monkeypatch.setenv("BRINGUP_PROFILE_HOST", "1")
    monkeypatch.delenv("BRINGUP_QUICK", raising=False)
    monkeypatch.delenv(M.SUFFIX_ENV, raising=False)
    monkeypatch.delenv("TT_METAL_DEVICE_PROFILER", raising=False)
    monkeypatch.setattr(profile, "WALL_RUNS", 2)
    monkeypatch.setattr(profile, "WALL_WARM", 1)
    monkeypatch.setattr(profile, "HOST_RUNS", 2)
    monkeypatch.setattr(profile.accuracy_guard, "check", lambda s: None)
    try:
        profile.run_profile(s, None)
    finally:
        os.environ.pop(M.SUFFIX_ENV, None)
    m = got()
    p = f"_p{pos}"
    assert m[f"chunk_wall_ms{p}"] > 0 and m[f"host_cprofile_ms{p}"] > 0 and f"host_launch_calls{p}" in m
    assert "chunk_wall_ms" not in m and not any(k.startswith("pcc_") for k in m), sorted(m)
    assert list(M.results_dir().glob(f"*_host{p}.pstats")) and list(M.results_dir().glob(f"*_host{p}.json"))
