# SPDX-License-Identifier: Apache-2.0
"""Pytest plugin: neutralize the `requires_grid_size` skip.

Diagnostic only. The MLA tests declare `requires_grid_size((13, 10))`, which hard-skips on
this 1x-harvested BH Galaxy (12x10). That marker is a blanket precondition, so it does not
tell us WHICH layout constant actually needs the 13th column. Stubbing the check lets the
test run and fail (or pass) on its real constraint instead of on the guard.

Do not use this to claim a test passes on 12x10 -- use it to find the true failure.
"""


def pytest_configure(config):
    import tt_metal_conftest

    tt_metal_conftest._check_requires_grid_size = lambda *a, **k: None
    print("[no_grid_guard] requires_grid_size check stubbed out (diagnostic)")
