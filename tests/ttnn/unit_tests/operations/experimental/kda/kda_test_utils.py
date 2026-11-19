# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Re-export shim. Implementation lives in models.demos.deepseek_v3_d_p.tests.kda.kda_test_utils."""

import sys

from models.demos.deepseek_v3_d_p.tests.kda import kda_test_utils as _impl

sys.modules[__name__] = _impl
