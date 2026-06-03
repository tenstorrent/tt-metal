# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from models.demos.gemma4.tt.model_config import Gemma4ModelArgs

_DEFAULT_MAX_PREFILL = 8192


def pytest_addoption(parser):
    parser.addoption("--skip-model-load", action="store_true", default=False, help="Skip loading the model state dict")
    parser.addoption(
        "--max-prefill",
        action="store",
        type=int,
        default=_DEFAULT_MAX_PREFILL,
        help=(
            "Maximum prefill seq_len to run. Tests parametrized over "
            "PREFILL_BUCKETS skip lengths above this cap; the demo skips "
            "buckets above this cap in test_demo_prefill_lengths. Default: "
            f"{_DEFAULT_MAX_PREFILL}. Set higher (up to 262144) to exercise "
            "long-context kernels."
        ),
    )
    parser.addoption(
        "--page-block-size",
        action="store",
        type=int,
        default=None,
        help=(
            "Paged-attention block size for the demo's KV cache (issue #44946 "
            "page_block_size sweep). Sets GEMMA4_PAGE_BLOCK_SIZE so the demo "
            "helpers pick it up. One of {32, 64, 128, 256}; default 64."
        ),
    )


@pytest.fixture(autouse=True)
def _apply_page_block_size(request):
    """Propagate --page-block-size into GEMMA4_PAGE_BLOCK_SIZE for the run.

    The demo's page-block helpers read the env var (so a plain shell sweep works
    too); this fixture lets pytest -p/-k invocations set it via CLI. Restores the
    prior value afterwards so a single pytest process sweeping multiple values
    via parametrization doesn't leak state across tests.
    """
    value = request.config.getoption("--page-block-size")
    if value is None:
        yield
        return
    prev = os.environ.get("GEMMA4_PAGE_BLOCK_SIZE")
    os.environ["GEMMA4_PAGE_BLOCK_SIZE"] = str(value)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("GEMMA4_PAGE_BLOCK_SIZE", None)
        else:
            os.environ["GEMMA4_PAGE_BLOCK_SIZE"] = prev


@pytest.fixture(scope="session")
def state_dict(request):
    load_model = not request.config.getoption("--skip-model-load")
    model_path = os.getenv("HF_MODEL", None)
    if model_path is None or not load_model:
        return {}
    else:
        return Gemma4ModelArgs.load_state_dict(model_path, dummy_weights=False)
