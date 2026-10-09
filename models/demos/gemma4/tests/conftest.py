# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from .test_factory import configure_spec_decode_smoke_env, resolve_assistant_model_path, skip_if_config_only_checkpoint

_SPEC_DECODE_SMOKE_TESTS = frozenset(
    {
        "test_assistant_config_loads",
        "test_spec_decode_matches_greedy",
        "test_verify_batchsize_invariance",
    }
)

_MARKERS_REQUIRING_REAL_CHECKPOINT = frozenset(
    {
        "gemma4_prefill_trace",
        "gemma4_batched_prefill",
        "gemma4_hf_direct_parity",
    }
)


# Marker registration for these lives in models/demos/gemma4/conftest.py (the parent
# conftest), since demo/text_demo.py also uses gemma4_batched_prefill and is outside
# this tests/ directory's collection scope.


def pytest_addoption(parser):
    """Add custom command line options for pytest"""
    parser.addoption(
        "--test-modules",
        action="store",
        default="all",
        help="Comma-separated list of modules to test. Options: all, attention, rms_norm, router, experts, shared_mlp, moe, layer, model. Example: --test-modules=attention,shared_mlp",
    )


@pytest.fixture
def test_modules(request):
    """Fixture to get the test_modules value from command line or use default 'all'"""
    return request.config.getoption("--test-modules")


def pytest_sessionstart(session):
    """Pre-resolve assistant weights in CI before collection (spec-decode smokes only)."""
    if os.environ.get("CI") != "true":
        return
    args = getattr(session.config, "args", None) or []
    # Positional args only. A `-k 'not test_spec_decode_'` deselect must not match.
    arg_str = " ".join(str(a) for a in args)
    if "test_spec_decode.py" not in arg_str:
        return
    configure_spec_decode_smoke_env()


def _item_base_name(item):
    """Unparametrized test name. ``item.name`` is ``foo[blackhole-1x4]``."""
    return getattr(item, "originalname", None) or item.name.split("[", 1)[0]


def pytest_runtest_setup(item):
    """Skip PR integration tests when CI uses config-only HF_MODEL (no weights/tokenizer)."""
    if _item_base_name(item) in _SPEC_DECODE_SMOKE_TESTS:
        # Do not call configure_spec_decode_smoke_env() here: it rewrites HF_MODEL
        # for every later test in the process. The dedicated smoke pytest sets
        # that env in sessionstart; everyone else only resolves an existing dir.
        if os.environ.get("GEMMA4_SPEC_DECODE_ENV_READY") != "1":
            if not resolve_assistant_model_path(allow_download=False):
                pytest.skip("assistant weights not available (set GEMMA4_ASSISTANT_MODEL locally)")

    if _MARKERS_REQUIRING_REAL_CHECKPOINT.intersection(m.name for m in item.iter_markers()):
        skip_if_config_only_checkpoint()


@pytest.fixture(autouse=True)
def _enforce_max_prefill(request):
    """Auto-skip parametrized tests whose seq_len exceeds --max-prefill.

    Looks up the seq_len param on the test's callspec; tests without a seq_len
    parametrization are unaffected. Decode (seq_len=1) always runs.
    """
    callspec = getattr(request.node, "callspec", None)
    if callspec is None:
        return
    seq_len = callspec.params.get("seq_len")
    if not isinstance(seq_len, int):
        return
    max_prefill = request.config.getoption("--max-prefill")
    if seq_len > max_prefill:
        pytest.skip(f"seq_len={seq_len} > --max-prefill={max_prefill}")
