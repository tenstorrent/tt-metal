# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Re-export shim. Implementation lives in tt_py_test_utils_common.kda_test_utils."""

import sys

from tt_py_test_utils_common import kda_test_utils as _impl

sys.modules[__name__] = _impl
