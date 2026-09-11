# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from models.demos.gemma4.tt.model_config import Gemma4ModelArgs

_DEFAULT_MAX_PREFILL = 8192


def pytest_addoption(parser):
    parser.addoption("--skip-model-load", action="store_true", default=False, help="Skip loading the model state dict")
    parser.addoption(
        "--speculative",
        action="store_true",
        default=False,
        help=(
            "Run the text demo in speculative-decoding mode (it-assistant drafter "
            "verified by the target). batch=1 only; the drafter defaults to "
            "<HF_MODEL>-assistant unless GEMMA4_ASSISTANT_MODEL is set."
        ),
    )
    parser.addoption(
        "--spec-draft-len",
        action="store",
        type=int,
        default=None,
        help="Speculative draft length K (drafts proposed per verify). Default: 3 (or GEMMA4_SPEC_DRAFT_LEN).",
    )
    parser.addoption(
        "--max-prefill",
        action="store",
        type=int,
        default=_DEFAULT_MAX_PREFILL,
        help=(
            "Maximum prefill seq_len for unit-test PREFILL_BUCKETS and short "
            f"demo buckets (test_demo / batch_prefill). Default: {_DEFAULT_MAX_PREFILL}. "
            "Does not apply to test_demo_long_context / text_demo_v2 long-context-* "
            "rows — select those with ``-k long-context-*``."
        ),
    )


def pytest_configure(config):
    """WH 256k metal: do not reserve the 192 MB trace region (device fixture).

    ``_device_params()`` is evaluated at collection, so this must run first.
    Tail-pool skip for 256k is also in ``Gemma4Attention`` (max_seq_len).
    """
    kw = str(getattr(config.option, "keyword", None) or "")
    args = " ".join(str(a) for a in (getattr(config, "args", None) or ()))
    if "long-context-256k" not in kw and "long-context-256k" not in args:
        return
    try:
        from models.common.utility_functions import is_blackhole

        if is_blackhole():
            return
    except Exception:
        pass
    os.environ.setdefault("GEMMA4_TRACE_REGION_SIZE", "0")
    os.environ.setdefault("GEMMA4_TAIL_POOL_SLOTS", "0")


@pytest.fixture(scope="session")
def state_dict(request):
    load_model = not request.config.getoption("--skip-model-load")
    model_path = os.getenv("HF_MODEL", None)
    if model_path is None or not load_model:
        return {}
    else:
        return Gemma4ModelArgs.load_state_dict(model_path, dummy_weights=False)
