# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Open the REAL device, print 'RKEY <2erisc> <build_key>', capture the JIT build fingerprint.
Mirrors run_safe_pytest.sh's _precompile_realkey. Writes the fingerprint to $PRECOMPILE_FP."""
import os
import ttnn

md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
try:
    f = 1 if ttnn.cluster.get_enable_2_erisc_mode() else 0
    k = ttnn.cluster.get_build_key()
    ttnn.cluster.capture_jit_build_fingerprint(os.environ["PRECOMPILE_FP"])
finally:
    ttnn.close_mesh_device(md)
print(f"RKEY {f} {k}")
