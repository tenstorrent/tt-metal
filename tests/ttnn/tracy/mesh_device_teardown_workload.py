# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Standalone repro workload for ``test_mesh_device_teardown_no_segfault``.

Regression repro for https://github.com/tenstorrent/tt-metal/issues/44472:
a bare open_mesh_device()/close_mesh_device() pair, no workload and no
explicit profiling, segfaulted on Python process exit. Kept as a standalone
script (rather than inline in the test) so the pytest parent process never
opens a device itself and a crash here only takes down this subprocess.
"""

import sys

import ttnn

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
ttnn.close_mesh_device(mesh)
sys.stdout.flush()
print("mesh_device_teardown_workload: closed cleanly")
