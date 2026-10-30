# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Print the chip ids of each 1x8 row of the Galaxy's 4x8 mesh, as ``ROW_DEVICES=<group> <group> ...``: the
TT_VISIBLE_DEVICES groups serve_rows.sh gives its engines. The ids depend on the machine, so serve_rows.sh runs this
before it starts the engines (unless ROW_DEVICES is set), as vllm-tt-plugin's standard DP discovers its ranks' groups.
Opens the parent mesh, so run it with nothing else on the machine."""

import ttnn

mesh = ttnn.open_mesh_device(ttnn.MeshShape(4, 8))
rows = mesh.create_submeshes(ttnn.MeshShape(1, 8))
groups = [",".join(str(i) for i in row.get_device_ids()) for row in rows]
for row in rows:
    ttnn.close_mesh_device(row)
ttnn.close_mesh_device(mesh)
print("ROW_DEVICES=" + " ".join(groups), flush=True)  # last, after the runtime's closing log lines
