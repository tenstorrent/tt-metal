# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""A golden defined outside tt-llk, registered through the harness decorator.

Consumers keep their goldens in a suite-local package like this one. Note it is
NOT under a ``helpers/`` directory: that name is reserved for the harness, and a
Python package there would shadow the harness's own ``helpers`` package.

``register_golden`` comes from the facade, like everything else a consumer
touches -- this module is part of the representative consumer, so it has to obey
the same rule as the tests.
"""

from tt_llk_harness import register_golden


@register_golden
class OutOfTreeGolden:
    """Identity reference; exists to prove out-of-tree registration works."""

    MARKER = "out-of-tree-fixture"

    def __call__(self, operand=None, **kwargs):
        if operand is None:
            return self.MARKER
        return operand
