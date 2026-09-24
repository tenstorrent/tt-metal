# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of the Qwen Galaxy decode-path flags. No device is opened."""

from types import SimpleNamespace

import pytest

from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs

_PREFETCHER_ENV = ("QWEN_BH_PREFETCHER", "QWEN_BH_UNFUSED_CCL")


def _configure(monkeypatch, *, arch, num_devices, env):
    for name in _PREFETCHER_ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    args = SimpleNamespace(
        num_devices=num_devices,
        is_blackhole=arch == "blackhole",
        use_prefetcher=False,
        prepare_decode_before_prefill=TtQwenModelArgs.prepare_decode_before_prefill,
    )
    TtQwenModelArgs._configure_prefetcher_path(args)
    return args


@pytest.mark.parametrize(
    "arch,num_devices,env,use_prefetcher,use_unfused_ccl,decode_before_prefill",
    [
        # Blackhole Galaxy with the prefetcher on: decode is staged after the prefill warmup on
        # both CCL paths, because the global circular buffer is live in decode either way.
        ("blackhole", 32, {"QWEN_BH_PREFETCHER": "1"}, True, True, False),
        ("blackhole", 32, {"QWEN_BH_PREFETCHER": "1", "QWEN_BH_UNFUSED_CCL": "1"}, True, True, False),
        ("blackhole", 32, {"QWEN_BH_PREFETCHER": "1", "QWEN_BH_UNFUSED_CCL": "0"}, True, False, False),
        # Blackhole Galaxy without the prefetcher (the default, used by vLLM): decode-first stays.
        ("blackhole", 32, {}, False, False, True),
        ("blackhole", 32, {"QWEN_BH_PREFETCHER": "0"}, False, False, True),
        ("blackhole", 32, {"QWEN_BH_PREFETCHER": "0", "QWEN_BH_UNFUSED_CCL": "1"}, False, False, True),
        # Wormhole Galaxy always uses the prefetcher and keeps the decode-first order.
        ("wormhole_b0", 32, {}, True, False, True),
        ("wormhole_b0", 32, {"QWEN_BH_PREFETCHER": "1"}, True, False, True),
        # Not a Galaxy: no prefetcher at all.
        ("blackhole", 8, {"QWEN_BH_PREFETCHER": "1"}, False, False, True),
        ("wormhole_b0", 8, {}, False, False, True),
    ],
)
def test_prefetcher_path_flags(
    monkeypatch, arch, num_devices, env, use_prefetcher, use_unfused_ccl, decode_before_prefill
):
    args = _configure(monkeypatch, arch=arch, num_devices=num_devices, env=env)
    assert args.use_prefetcher is use_prefetcher
    assert args.use_unfused_ccl is use_unfused_ccl
    assert args.prepare_decode_before_prefill is decode_before_prefill


def test_blackhole_prefetcher_never_stages_decode_first(monkeypatch):
    """The ordering must follow the global-CB trigger, not the CCL flavour."""
    for unfused in ("0", "1"):
        args = _configure(
            monkeypatch,
            arch="blackhole",
            num_devices=32,
            env={"QWEN_BH_PREFETCHER": "1", "QWEN_BH_UNFUSED_CCL": unfused},
        )
        assert args.use_prefetcher and args.is_blackhole
        assert args.prepare_decode_before_prefill is False
