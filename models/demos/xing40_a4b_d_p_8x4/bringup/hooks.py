# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up hooks for xing40_a4b_d_p_8x4: Xing4.0-29B-A4B on an 8x4 Blackhole Galaxy mesh (SP = 8 x TP = 4, 2 routed
experts per chip), the same checkpoint, goldens and model code as the prior bring-up xing40_a4b_d_p (4x2).

The xing40_a4b_d_p model code takes every per-chip size from mesh.shape, so both the CPU side and the device side are
the prior's hooks unchanged; only the spec (box.mesh) differs. Switch meshes with BRINGUP_SPEC:
  4x2: models/demos/xing40_a4b_d_p/bringup/spec.yaml
  8x4: models/demos/xing40_a4b_d_p_8x4/bringup/spec.yaml
"""

from models.demos.xing40_a4b_d_p.bringup import hooks as _prior

for _name in dir(_prior):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_prior, _name)
del _name
