import ttnn
from ttnn.operations.tilize import tilize_program_descriptor as pd

device = ttnn.open_device(device_id=0)
try:
    for r, c in [(1, 64), (64, 64), (1, 256), (2, 128), (512, 2)]:
        _, a, cc = pd._rect_assignment(device, r, c, 1)
        print("CORES", r, c, "cores", len(a), "cores_c", cc, "busiest", max(x[2] * x[4] for x in a))
finally:
    ttnn.close_device(device)
