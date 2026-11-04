# Decode the element order the unpacker reads from a packed ternary tile: run 10 tiles where element at source
# index i holds +1 if bit b of i is set, else -1; the output position's sign over the 10 runs spells i.
import numpy as np
import torch

import ttnn
from models.demos.blackhole.bonsai2.tests.test_ternary_probe import _run
from models.demos.blackhole.bonsai2.tt.ternary.pack import pack_ternary_tiles

device = ttnn.open_device(device_id=0)
try:
    idx = np.arange(2048).reshape(2, 32, 32)
    src = np.zeros((64, 32), dtype=np.int64)
    for b in range(11):
        t = np.where((idx >> b) & 1, 1, -1).astype(np.int8)
        out = _run(device, torch.eye(32), pack_ternary_tiles(t), 127).numpy()
        assert set(np.unique(out)) <= {-1.0, 1.0}, np.unique(out)
        src |= (out > 0).astype(np.int64) << b
    # tile 1's first rows: from tile 1's own tail (rotation) or tile 0's tail (stream shift)?
    print(
        "tile1 row0 sources",
        [(int(v) // 1024, divmod(int(v) % 1024, 32)) for v in src[32, :2]],
        [(int(v) // 1024, divmod(int(v) % 1024, 32)) for v in src[32, 16:18]],
    )
    print("tile0 row0 sources", [(int(v) // 1024, divmod(int(v) % 1024, 32)) for v in src[0, :2]])
    src = src[:32]
    # src[r, c] = the (row*32+col) index my packer put at the element the hardware shows at (r, c)
    print("identity?", np.array_equal(src, idx))
    np.save("/tmp/ternary_perm.npy", src)
    for r in (0, 1, 2, 15, 16, 17, 31):
        print(r, [divmod(int(v), 32) for v in src[r, :8]], "...", [divmod(int(v), 32) for v in src[r, 16:20]])
finally:
    ttnn.close_device(device)
