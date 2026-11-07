# Probe: do sub-64-B NOC writes into DRAM patch only their bytes? (kernels_probe/dram_partial_write.cpp)
import numpy as np
import torch

import ttnn


def test_dram_partial(device):
    t = ttnn.from_torch(torch.zeros(32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    c = ttnn.CoreCoord(0, 0)
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(c, c)})
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [t.buffer_address()]
    prog = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source="models/demos/blackhole/bonsai2/tests/kernels_probe/dram_partial_write.cpp",
                core_ranges=crs,
                compile_time_args=ttnn.TensorAccessorArgs(t).get_compile_time_args(),
                runtime_args=rt,
                config=ttnn.WriterConfigDescriptor(),
            )
        ],
        semaphores=[],
        cbs=[
            ttnn.CBDescriptor(
                total_size=2048,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2048)],
            )
        ],
    )
    ttnn.generic_op([t, t], prog)
    raw = ttnn.to_torch(t).view(torch.int16).numpy()
    # back to the tile's byte order: faces
    from_t = ttnn.to_torch(t)
    b = ttnn.from_device(t)
    data = np.frombuffer(bytes(b.to_torch().contiguous().view(torch.uint8).numpy()), dtype=np.uint8)
    # tilized byte image: recompute by tilizing the torch tensor
    x = b.to_torch().reshape(32, 32)
    faces = [x[:16, :16], x[:16, 16:], x[16:, :16], x[16:, 16:]]
    img = torch.cat([f.reshape(-1) for f in faces]).contiguous().view(torch.uint8).numpy()
    nz = np.nonzero(img)[0]
    exp = {}
    for o, n in [(5, 1), (17, 1), (80, 16), (200, 16), (320, 2), (1000, 32), (1501, 3)]:
        for i in range(o, o + n):
            exp[i] = (i * 7 + 1) & 0xFF
    got = {int(i): int(img[i]) for i in nz}
    bad = {k: v for k, v in got.items() if exp.get(k) != v}
    miss = {k: v for k, v in exp.items() if got.get(k) != v and v != 0}
    print(
        "PARTIAL nonzero",
        len(got),
        "expected",
        len(exp),
        "bad",
        list(bad.items())[:20],
        "missing",
        list(miss.items())[:20],
    )
