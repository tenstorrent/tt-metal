import torch, ttnn
from ttnn.operations.tilize import tilize
import ttnn.operations.tilize.tilize_program_descriptor as pd
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

dev = ttnn.open_device(device_id=0)
orig = pd.NumericConfig.compute_config


def run(precise):
    def cc(self, cbs):
        c = orig(self, cbs)
        c.bfp8_pack_precise = precise
        return c

    pd.NumericConfig.compute_config = cc
    for ind in [ttnn.bfloat16, ttnn.float32]:
        for bf in [ttnn.bfloat8_b, ttnn.bfloat4_b]:
            torch.manual_seed(0)
            x = torch.randn(1, 1, 64, 128).to(torch.float32 if ind == ttnn.float32 else torch.bfloat16)
            t = ttnn.from_torch(x, dtype=ind, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
            dv = ttnn.to_torch(tilize(t, dtype=bf))
            h = ttnn.to_torch(ttnn.from_torch(x, dtype=bf, layout=ttnn.TILE_LAYOUT))
            print(
                "RES precise",
                precise,
                ind,
                bf,
                "dev",
                comp_pcc(x, dv, 0.98)[1][-22:],
                "host",
                comp_pcc(x, h, 0.98)[1][-22:],
                "equal",
                torch.equal(dv.float(), h.float()),
                (dv.float() - h.float()).abs().max().item(),
            )
            # rank-0 / rank-1
        for shape in [[], [64]]:
            torch.manual_seed(1)
            x = torch.randn(shape).to(torch.float32 if ind == ttnn.float32 else torch.bfloat16)
            t = ttnn.from_torch(x, dtype=ind, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
            y = tilize(t, dtype=ttnn.bfloat4_b, pad_value=0.0)
            print("RES precise", precise, ind, shape, comp_pcc(x, ttnn.to_torch(y).reshape(x.shape), 0.98))


run(False)
run(True)
ttnn.close_device(dev)
