# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""In-process device-kernel-time sweeps for GLM-4.7-Flash matmuls.

IMPORTANT: importing this package sets the profiler env vars *before* ttnn is
imported anywhere.  That is the only reason this import lives at the top of the
package __init__ -- the tt-metal device profiler reads these flags once, at the
first device open, and there is no Python API to flip them afterwards.  Keeping
them here means the sweep *interface* (targets, axes) stays pure Python with no
env vars for the caller to manage.
"""

from . import profiler_setup  # noqa: F401  (must run before any `import ttnn`)
