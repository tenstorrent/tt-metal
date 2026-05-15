# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest


USE_GOLDEN_PARAMS = (
    pytest.param(True, id="use_golden"),
    pytest.param(
        False,
        id="use_pipeline",
        marks=pytest.mark.xfail(
            raises=NotImplementedError,
            strict=True,
            reason="DFlash production host/device pipeline is not wired yet.",
        ),
    ),
)


def require_golden_or_not_implemented(use_golden: bool, replacement_point: str) -> None:
    if use_golden:
        return
    raise NotImplementedError(
        f"{replacement_point} is not implemented yet; replace this branch with the real DFlash pipeline call."
    )
