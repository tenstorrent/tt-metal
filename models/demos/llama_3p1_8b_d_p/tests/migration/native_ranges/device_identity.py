"""Live physical UMD/fabric/ASIC mapping; copied from accepted startup009."""
from runner_support import require


def device_namespace_receipt(ttnn, mesh, mapping):
    """Resolve UMD IDs from physical device accessors, never fabric/map columns."""
    rows = []
    require(tuple(mesh.shape) == (4, 8), "Expected full4x8 owner mesh")
    for r in range(4):
        for c in range(8):
            coord = ttnn.MeshCoordinate(r, c)
            physical = int(mesh.get_device_id(coord))
            fid = mesh.get_fabric_node_id(coord)
            mesh_id, fabric_id = int(fid.mesh_id), int(fid.chip_id)
            asic = int(ttnn.cluster.get_chip_unique_id_from_fabric_node_id(mesh_id, fabric_id))
            require(mapping.get(f"{mesh_id}:{fabric_id}") == asic, "Live mesh/table ASIC identity mismatch")
            rows.append(
                dict(
                    coord=[r, c],
                    umd_device_id=physical,
                    fabric_mesh_id=mesh_id,
                    fabric_chip_id=fabric_id,
                    asic_unique_id=asic,
                )
            )
    result = dict(rows=rows, umd_device_ids=sorted(int(x) for x in mesh.get_device_ids()))
    validate_device_namespace(result, mapping)
    return result


def validate_device_namespace(receipt, mapping):
    rows = receipt["rows"]
    ids = receipt["umd_device_ids"]
    require(
        len(rows) == 32 and {tuple(x["coord"]) for x in rows} == {(r, c) for r in range(4) for c in range(8)},
        "Incomplete device-coordinate inventory",
    )
    require(
        len(ids) == 32 and len(set(ids)) == 32 and all(type(x) is int and 0 <= x <= 0xFFFFFFFF for x in ids),
        "Invalid physical UMD IDs",
    )
    require(sorted(x["umd_device_id"] for x in rows) == ids, "Physical per-coordinate IDs disagree with mesh inventory")
    recovered = {f"{x['fabric_mesh_id']}:{x['fabric_chip_id']}": x["asic_unique_id"] for x in rows}
    require(
        len(recovered) == 32 and len(set(recovered.values())) == 32 and recovered == mapping,
        "Native physical/fabric/ASIC map disagreement",
    )
    return ids
