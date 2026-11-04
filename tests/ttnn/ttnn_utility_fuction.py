# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Re-export shim. Implementation lives in tt_py_test_utils_common.ttnn_utility_fuction."""

import sys

from tt_py_test_utils_common import ttnn_utility_fuction as _impl

sys.modules[__name__] = _impl
