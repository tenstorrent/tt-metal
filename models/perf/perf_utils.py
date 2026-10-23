# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Re-export shim. Implementation lives in tt_py_test_utils_common.perf.perf_utils."""

import sys

from tt_py_test_utils_common.perf import perf_utils as _impl

sys.modules[__name__] = _impl
