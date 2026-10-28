# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Print the chip ids of each 1x8 row of the Galaxy's 4x8 mesh: the TT_VISIBLE_DEVICES groups serve_rows.sh gives its
engines (ROW_DEVICES). Opens the parent mesh, so run it with nothing else on the machine."""

import ttnn

mesh = ttnn.open_mesh_device(ttnn.MeshShape(4, 8))
rows = mesh.create_submeshes(ttnn.MeshShape(1, 8))
print(" ".join(",".join(str(i) for i in row.get_device_ids()) for row in rows))
for row in rows:
    ttnn.close_mesh_device(row)
ttnn.close_mesh_device(mesh)
