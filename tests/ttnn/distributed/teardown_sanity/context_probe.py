# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Level 1, Python: one ttnn rank on the DistributedContext alone (no device, no mesh), with one injected fault.
The scenarios, markers and knobs are probe_common.py's."""

from probe_common import run

run(mesh=False)
