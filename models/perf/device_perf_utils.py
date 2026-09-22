# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Re-export shim. Implementation lives in tt_py_test_utils_common.perf.device_perf_utils."""

from tt_py_test_utils_common.perf import device_perf_utils as _impl

_KEEP = {"__name__", "__file__", "__package__", "__loader__", "__spec__", "__cached__"}
globals().update({k: v for k, v in _impl.__dict__.items() if k not in _KEEP})
del _KEEP
