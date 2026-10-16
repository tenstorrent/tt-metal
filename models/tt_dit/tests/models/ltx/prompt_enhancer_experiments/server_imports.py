# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""pytest plugin: import what tt-media-server's LTX worker imports, before the test builds anything.

The media server's LTX runner lives in ``tt_model_runners/dit_runners.py``, which imports seven other
DiT pipelines at module level; some of their modules mutate shared state at import (MiniMax-H3 registers
its conv3d blockings into the global fallback table). This plugin replays those imports so a pipeline
test sees the same process state as the server. Imports run at session start, after the LTX conftest's
pre-torch re-exec, so the host pinning is unchanged.
Load with ``-p models.tt_dit.tests.models.ltx.prompt_enhancer_experiments.server_imports``.
"""

import importlib
import time

from loguru import logger

# tt-media-server dit_runners.py module-level tt-metal imports (tt-inference-server @ cb0b90f7).
SERVER_IMPORTS = (
    "models.common.utility_functions",
    "models.tt_dit.pipelines.flux1.pipeline_flux1",
    "models.tt_dit.pipelines.minimax_h3.pipeline_minimax_h3",
    "models.tt_dit.pipelines.motif.pipeline_motif",
    "models.tt_dit.pipelines.qwenimage.pipeline_qwenimage",
    "models.tt_dit.pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large",
    "models.tt_dit.pipelines.wan.pipeline_wan",
    "models.tt_dit.pipelines.wan.pipeline_wan_i2v",
)


def pytest_sessionstart(session):  # noqa: ARG001
    t0 = time.time()
    for name in SERVER_IMPORTS:
        importlib.import_module(name)
    logger.warning(f"[server-imports] imported {len(SERVER_IMPORTS)} media-server modules in {time.time() - t0:.1f}s")
