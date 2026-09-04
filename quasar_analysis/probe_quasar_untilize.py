"""Narrow the quasar TILE -> ROW_MAJOR corruption that
models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe_bf16/test_to_layout_forge_bf16.py
reports on 11 of its 50 call-sites (cause C in quasar_analysis/forge_fe_bf16_runs/SUMMARY.txt).

Run:
  TT_METAL_SIMULATOR=<dir>/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1 ARCH_NAME=quasar \
  python quasar_analysis/probe_quasar_untilize.py

Columns:
  roundtrip  = from_torch(TILE, device) -> to_torch, no device op at all (is the upload sound?)
  to_layout  = quasar.to_layout(TILE -> ROW_MAJOR)
  untilize   = quasar.untilize
  unt_unpad  = quasar.untilize_with_unpadding
"""

import torch, ttnn
import numpy as np

DRAM = ttnn.DRAM_MEMORY_CONFIG
device = ttnn.open_device(device_id=0, l1_small_size=24576)
torch.manual_seed(0)

SHAPES = [
    (1, 1, 3136, 256),
    (1, 1, 3136, 64),
    (1, 1, 196, 1024),
    (1, 1, 196, 256),
    (1, 1, 50176, 3),
    (1, 1, 3136, 128),
    (1, 1, 784, 512),
]


def pcc(a, b):
    x = a.float().flatten()
    y = b.float().flatten()
    if x.shape != y.shape:
        return "shape!"
    c = np.corrcoef(x.numpy(), y.numpy())[0, 1]
    return "%.4f" % c


print("%-16s %-10s %-10s %-10s %-10s" % ("shape", "roundtrip", "to_layout", "untilize", "unt_unpad"))
for sh in SHAPES:
    host = torch.randn(sh, dtype=torch.bfloat16)
    res = []
    for route in ("roundtrip", "to_layout", "untilize", "untilize_with_unpadding"):
        tt = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
        try:
            if route == "roundtrip":
                out = tt
            elif route == "to_layout":
                out = ttnn.experimental.quasar.to_layout(tt, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
            elif route == "untilize":
                out = ttnn.experimental.quasar.untilize(tt, memory_config=DRAM)
            else:
                out = ttnn.experimental.quasar.untilize_with_unpadding(tt, [d - 1 for d in sh], memory_config=DRAM)
            got = ttnn.to_torch(ttnn.from_device(out))
            res.append(pcc(host, got))
        except Exception as e:
            res.append(type(e).__name__[:10])
    print("%-16s %-10s %-10s %-10s %-10s" % ("x".join(map(str, sh)), *res))
ttnn.close_device(device)
