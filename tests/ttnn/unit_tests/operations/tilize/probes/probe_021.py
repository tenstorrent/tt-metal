import os, torch, ttnn
from ttnn.operations.tilize import tilize
import ttnn.operations.tilize.tilize_program_descriptor as pd
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

mode = os.environ["MODE"]
dev = ttnn.open_device(device_id=0)
orig = pd.NumericConfig.compute_config


def cc(self, cbs):
    c = orig(self, cbs)
    if mode in ("precise", "both"):
        c.bfp8_pack_precise = True
    if mode in ("fp32dest", "both"):
        c.fp32_dest_acc_en = True
    return c


pd.NumericConfig.compute_config = cc
torch.manual_seed(0)
x = torch.randn(1, 1, 32, 64).bfloat16()
t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
for bf in [ttnn.bfloat8_b, ttnn.bfloat4_b]:
    for th in [16, 8]:
        y = tilize(t, dtype=bf, tile=ttnn.Tile([th, 32]))
        print("RES", mode, bf, th, comp_pcc(x, ttnn.to_torch(y), 0.98)[1][-25:])
ttnn.close_device(dev)
