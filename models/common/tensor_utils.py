# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Re-export shim. Implementation lives in tt_py_test_utils_common.tensor_utils."""

import sys

from tt_py_test_utils_common import tensor_utils as _impl

sys.modules[__name__] = _impl
