# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Re-export shim. Implementation lives in tt_py_test_utils_common.perf.merge_device_perf_results."""

import sys

from tt_py_test_utils_common.perf import merge_device_perf_results as _impl

if __name__ == "__main__":
    raise SystemExit(_impl.main())

sys.modules[__name__] = _impl
