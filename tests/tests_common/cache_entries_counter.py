# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Re-export shim. Implementation lives in tt_py_test_utils_common.cache_entries_counter."""

import sys

from tt_py_test_utils_common import cache_entries_counter as _impl

sys.modules[__name__] = _impl
