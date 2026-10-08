# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import pytest

from models.experimental.ops.quasar.qwen3_vl.tests.e2e.config import RunConfig


def pytest_addoption(parser):
    g = parser.getgroup("qwen3_vl_e2e")
    g.addoption("--qwen-size", default="tiny", choices=["tiny", "demo"])
    g.addoption("--qwen-vision-layers", type=int, default=2)
    g.addoption("--qwen-text-layers", type=int, default=2)
    g.addoption("--qwen-decode-steps", type=int, default=1)
    g.addoption("--qwen-deepstack-at", type=int, default=None)
    g.addoption(
        "--qwen-kv-blocks", type=int, default=None, help="Paged KV-cache blocks of 32 tokens (default: preset)."
    )
    g.addoption("--qwen-host-ops", default="")
    g.addoption("--qwen-disable-wa", default="")
    g.addoption(
        "--qwen-allow-uncertified",
        action="store_true",
        default=False,
        help="Allow host fallbacks not yet checked against the real op (bisecting only).",
    )
    g.addoption("--qwen-quasar-config", action="store_true", default=False)
    g.addoption("--qwen-expect-grid", default=None)
    g.addoption("--qwen-run-dir", default="generated/qwen3_vl_quasar/adhoc")
    g.addoption("--qwen-dump-stages", action="store_true", default=False, help="Save golden and TT stage tensors.")
    g.addoption(
        "--qwen-clear-program-cache-before-decode",
        action="store_true",
        default=False,
        help="Debug: clear the program cache between prefill and decode.",
    )
    g.addoption(
        "--qwen-probe-before-decode",
        default="",
        help="Debug: comma-separated probes (tests/e2e/probes.py) to run on dummy tensors right before decode.",
    )
    g.addoption(
        "--qwen-check-tensor-integrity",
        action="store_true",
        default=False,
        help="Debug: report model device tensors (KV cache excluded) whose contents change between model build and decode.",
    )
    g.addoption(
        "--qwen-dump-decode-ops",
        action="store_true",
        default=False,
        help="Debug: save every top-level op output of the decode steps to decode_ops.pt.",
    )
    g.addoption(
        "--qwen-resume-prefill",
        default=None,
        help="Skip vision and prefill: decode from the prefill_snapshot.pt of an earlier run folder (same config).",
    )


@pytest.fixture
def qwen_run_config(request):
    return RunConfig.from_options(request.config.getoption)
