import torch, ttnn
import ttnn.operations.tilize.tilize_program_descriptor as pd
import ttnn.operations.tilize.tilize as tm
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)
orig = pd.create_program_descriptor


def wrap(*a, **k):
    d = orig(*a, **k)
    r = [kd for kd in d.kernels if "reader" in kd.kernel_source][0]
    print(
        "CO",
        list(a[0].shape),
        "co_read",
        r.compile_time_args[28],
        "coalesce",
        r.compile_time_args[26],
        "sems",
        len(d.semaphores),
        "cores",
        r.core_ranges.num_cores(),
    )
    return d


tm.create_program_descriptor = wrap
for shape in [(1, 1, 32, 2048), (1, 1, 128, 64), (1, 1, 2048, 512)]:
    x = torch.randn(shape).to(torch.bfloat16)
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = tilize(t)
    print("CO ok", torch.equal(ttnn.to_torch(out), x))
ttnn.close_device(device)
