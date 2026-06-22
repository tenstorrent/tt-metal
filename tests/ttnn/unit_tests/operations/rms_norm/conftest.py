# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# TEMPORARY mode-2 correctness harness (Refinement 9 Part D, TRANSPORT_REDUCE_BCAST).
# Autouse fixture forces _FORCE_TRANSPORT=2 only when RMS_FORCE_TRANSPORT=2 is in the env,
# so the same suites can be run with mode 2 forced and with the production default.
# DELETE this conftest before committing — _FORCE_TRANSPORT must stay None in the module.
import os
import pytest

import ttnn.operations.rms_norm.rms_norm_program_descriptor as desc


@pytest.fixture(autouse=True)
def _force_transport_mode():
    forced = os.environ.get("RMS_FORCE_TRANSPORT")
    saved = desc._FORCE_TRANSPORT
    if forced is not None and forced != "":
        desc._FORCE_TRANSPORT = int(forced)
    yield
    desc._FORCE_TRANSPORT = saved
