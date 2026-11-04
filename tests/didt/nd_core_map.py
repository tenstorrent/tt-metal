# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Re-export shim. Implementation lives in tt_py_test_utils_common.nd_core_map."""

import sys

from tt_py_test_utils_common import nd_core_map as _impl

sys.modules[__name__] = _impl
