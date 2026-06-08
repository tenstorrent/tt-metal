# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hardware-free mock open replaying the captured fingerprint -> print 'MKEY <build_key>'.
Mirrors run_safe_pytest.sh's _precompile_mockkey. Requires the slow-dispatch + mock-desc +
fingerprint env to be set by the caller."""
import ttnn

md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
try:
    print("MKEY", ttnn.cluster.get_build_key())
finally:
    ttnn.close_mesh_device(md)
