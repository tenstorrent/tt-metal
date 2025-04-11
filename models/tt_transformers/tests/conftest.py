# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import pytest
import gc

from models.tt_transformers.tt.model_config import parse_optimizations


@pytest.fixture(autouse=True)
def ensure_gc():
    gc.collect()


def pytest_addoption(parser):
    parser.addoption(
        "--optimizations",
        action="store",
        default=None,
        type=parse_optimizations,
        help="Precision and fidelity configuration diffs over default (i.e., accuracy)",
    )
