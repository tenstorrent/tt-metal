# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Re-export shim. Implementation lives in tt_py_test_utils_common.op_test_base."""

import sys

from tt_py_test_utils_common import op_test_base as _impl

sys.modules[__name__] = _impl
