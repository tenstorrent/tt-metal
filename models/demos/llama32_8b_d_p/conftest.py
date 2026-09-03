# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Package-local pytest configuration for `llama32_8b_d_p`.

Shape copied from `models/demos/minimax_m3/conftest.py` (`pytest_addoption` at :12,
`--skip-model-load` at :13, session-scoped `state_dict` fixture at :16-17).

Deliberately does **not** define `mesh_device` or `reset_seeds` — both come from the repo-root
`conftest.py` (`conftest.py:554` and `conftest.py:34` respectively, both verified). Redefining
them would shadow the fixtures every other model test in the tree relies on
(BRINGUP_RECIPE.md:329-330).
"""

from __future__ import annotations

import os

import pytest
from loguru import logger


def pytest_addoption(parser):
    parser.addoption(
        "--skip-model-load",
        action="store_true",
        default=False,
        help="Skip loading the model state dict; tests then run on random weights only.",
    )


@pytest.fixture(scope="session")
def state_dict(request):
    """The real HF state dict, or `{}` when no checkpoint is available.

    Returns `{}` — rather than raising — when `--skip-model-load` is passed or `HF_MODEL` is not a
    directory. Every module test in this package is built to run from *identical random weights*
    driving both the torch reference and the TT module (BRINGUP_RECIPE.md:305-308, :588), so an
    empty dict is a supported mode, not a degraded one. Tests that genuinely need real weights
    carry `tests.test_factory.requires_hf_reference` and skip.

    Loaded via `models/tt_transformers/tt/load_checkpoints.py:18` `load_hf_state_dict(ckpt_dir)`,
    which reads `model.safetensors.index.json` if present and `model.safetensors` otherwise. Keys
    stay in **HF layout** here; conversion to Meta layout (`convert_hf_to_meta`, :193) is the
    caller's decision, because the RoPE convention determines whether the Q/K permute is wanted
    (BRINGUP_RECIPE.md:626-641).
    """
    if request.config.getoption("--skip-model-load"):
        logger.info("[llama32_8b_d_p] --skip-model-load: returning an empty state_dict")
        return {}

    hf_model = os.getenv("HF_MODEL")
    if not (hf_model and os.path.isdir(hf_model)):
        logger.warning(
            "[llama32_8b_d_p] HF_MODEL is not a directory (got {!r}); returning an empty "
            "state_dict. Real-weight tests will skip. See bringup_log/07_RISKS.md R-003.".format(hf_model)
        )
        return {}

    from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict

    logger.info(f"[llama32_8b_d_p] loading state dict from {hf_model}")
    sd = load_hf_state_dict(hf_model)
    logger.info(f"[llama32_8b_d_p] loaded {len(sd)} tensors")
    return sd
