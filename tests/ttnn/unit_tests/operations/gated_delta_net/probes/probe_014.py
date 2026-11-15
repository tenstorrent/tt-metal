import torch, ttnn, itertools
from collections import defaultdict
from ttnn.operations.gated_delta_net import gated_delta_net_program_descriptor as pd
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import ORDER, compute_config, make_inputs

device = ttnn.open_device(device_id=0)
base = ttnn.device.get_allocator_base_address(device, ttnn.BufferType.L1)
print("L1 base", base, "unreserved size", ttnn.get_max_worker_l1_unreserved_size())
from ttnn.operations.gated_delta_net.gated_delta_net import default_compute_kernel_config

shapes0 = [
    ((1, 1, 2560), (1, 12, 128, 128)),
    ((1, 8, 2560), (8, 12, 128, 128)),
    ((1, 16, 2560), (16, 12, 128, 128)),
    ((1, 128, 2560), (1, 12, 128, 128)),
    ((1, 512, 2560), (1, 12, 128, 128)),
    ((1, 2048, 2560), (1, 12, 128, 128)),
    ((1, 4096, 2560), (1, 12, 128, 128)),
    ((1, 1, 10240), (1, 48, 128, 128)),
    ((1, 1, 4096), (1, 16, 128, 128)),
    ((1, 64, 4096), (1, 16, 128, 128)),
    ((1, 64, 2560), (1, 12, 128, 128)),
    ((1, 8192, 256), (1, 3, 32, 64)),
    ((1, 40, 256), (40, 3, 32, 64)),
    ((1, 5, 1760), (1, 33, 32, 32)),
    ((1, 512, 256), (1, 3, 32, 64)),
]
shapes = [
    ((1, 16, 2560), (16, 12, 128, 128), ttnn.float32),
    ((1, 512, 2560), (1, 12, 128, 128), ttnn.float32),
    ((1, 128, 2560), (1, 12, 128, 128), ttnn.float32),
    ((1, 2048, 2560), (1, 12, 128, 128), ttnn.bfloat16),
    ((1, 128, 2560), (1, 12, 128, 128), ttnn.bfloat16),
]
for q, s, dd in shapes:
    for dt in (dd,):
        t = make_inputs((q, s))
        ins = [
            ttnn.from_torch(
                t[n],
                dtype=ttnn.float32 if n == "recurrent_state" else dt,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for n in ORDER
        ]
        B, Hv, K, V = s
        rows, C = q[1], q[2]
        out = [
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, rows, Hv * V]), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
            ),
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 3 * B, C]), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
            ),
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([B, Hv, K, V]), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
            ),
        ]
        d = pd.create_program_descriptor(ins, out, epsilon=1e-6, compute_kernel_config=default_compute_kernel_config())
        per = defaultdict(int)
        for cb in d.cbs:
            for c in ttnn.corerange_to_cores(cb.core_ranges):
                per[(c.x, c.y)] += cb.total_size
        mx = max(per.values())
        sigs = defaultdict(list)
        for cb in d.cbs:
            cs = {(c.x, c.y) for c in ttnn.corerange_to_cores(cb.core_ranges)}
            sigs[frozenset(cs)].append((cb.format_descriptors[0].buffer_index, cb.total_size))
        roles = defaultdict(list)
        for k, (x, y) in enumerate(per):
            key = tuple(sorted((i, sz) for cs, l in sigs.items() if (x, y) in cs for i, sz in l))
            roles[key].append((x, y))
        for key, cores in roles.items():
            print(
                "  role cores", len(cores), "total", sum(sz for _, sz in key), [(i, sz) for i, sz in key if sz >= 8192]
            )
        print(
            q, s, dt, "cores", len(per), "max CB bytes", mx, "end", base + mx, "OK" if base + mx <= 1200000 else "OVER"
        )
ttnn.close_device(device)
