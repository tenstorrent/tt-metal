# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up hooks for glm53_flash_d_p_lb: GLM-5.3-Flash, all 45 text layers, on the 8x Blackhole p150b LoudBox (mesh
2x4: KDA SP = 2 x TP = 4, routed experts EP = 8 at bfp4), the same checkpoint and model code as the prior bring-up
glm53_flash_d_p (layers 0-4, mesh 2x2).

The glm53_flash_d_p model code takes every per-chip size from mesh.shape and the routed-expert dtype from the spec
(device.experts_dtype), so both the CPU side and the device side are the prior's hooks unchanged; only the spec differs.
Switch with BRINGUP_SPEC:
  2x2, layers 0-4:  models/demos/glm53_flash_d_p/bringup/spec.yaml
  2x4, all layers:  models/demos/glm53_flash_d_p_lb/bringup/spec.yaml
"""

from models.demos.glm53_flash_d_p.bringup import hooks as _prior

for _name in dir(_prior):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_prior, _name)
del _name
