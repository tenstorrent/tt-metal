# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): exp_approx_mode of the prefill SDPA (tt/tp_common.py prefill_sdpa_chunks).

* Without args.prefill_sdpa_config_fn the prefill SDPA uses the approximate exponential at every row count;
  QWEN36_SDPA_PREFILL_EXP_APPROX=0 keeps the accurate one.
* A prefill_sdpa_config_fn picks exp_approx_mode itself, whatever QWEN36_SDPA_PREFILL_EXP_APPROX says.
"""

from types import SimpleNamespace

import pytest

from models.demos.blackhole.qwen36.tt import tp_common as tpc


@pytest.mark.parametrize("S", [32, 128, 2048, 8192])
def test_the_default_prefill_sdpa_takes_the_approximate_exponential(monkeypatch, S):
    monkeypatch.delenv(tpc.SDPA_PREFILL_EXP_APPROX_ENV, raising=False)
    assert tpc.prefill_sdpa_chunks(SimpleNamespace(prefill_sdpa_config_fn=None), S, 128) == (128, 128, True)
    assert tpc.prefill_sdpa_chunks(SimpleNamespace(), S, 64) == (64, 64, True)


def test_the_environment_variable_keeps_the_accurate_exponential(monkeypatch):
    monkeypatch.setenv(tpc.SDPA_PREFILL_EXP_APPROX_ENV, "0")
    assert tpc.prefill_sdpa_chunks(SimpleNamespace(), 2048, 128) == (128, 128, False)


@pytest.mark.parametrize("env", [None, "0", "1"])
@pytest.mark.parametrize("fn_exp", [False, True])
def test_a_tuning_function_picks_the_exponential_itself(monkeypatch, env, fn_exp):
    if env is None:
        monkeypatch.delenv(tpc.SDPA_PREFILL_EXP_APPROX_ENV, raising=False)
    else:
        monkeypatch.setenv(tpc.SDPA_PREFILL_EXP_APPROX_ENV, env)
    args = SimpleNamespace(prefill_sdpa_config_fn=lambda S: (256, 512, fn_exp))
    assert tpc.prefill_sdpa_chunks(args, 2048, 128) == (256, 512, fn_exp)
