import ttnn

d = ttnn.open_device(device_id=0)
cores = ttnn.corerange_to_cores(
    ttnn.num_cores_to_corerangeset(110, d.compute_with_storage_grid_size(), True), None, True
)
import json

print(json.dumps({f"{c.x},{c.y}": list(d.worker_core_from_logical_core(c).__repr__() for _ in [0]) for c in cores}))
ttnn.close_device(d)
