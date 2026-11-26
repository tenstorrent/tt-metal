# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Level 2, Python: one ttnn rank that opens a mock MeshDevice (the control plane's distributed init) around the
collectives, with one injected fault. The scenarios, markers and knobs are probe_common.py's."""

from probe_common import run

run(mesh=True)
