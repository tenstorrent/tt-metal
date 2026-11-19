#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
if fuser /dev/tenstorrent/*; then
    echo 'Target mesh is owned; reset refused' >&2
    exit 2
fi
timeout 60 tt-smi -ls --local
timeout 180 tt-smi -r
timeout 60 tt-smi -ls --local
timeout 120 python - <<'PY'
import ttnn

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
ttnn.close_mesh_device(mesh)
print("MESH_SMOKE_OK")
PY
