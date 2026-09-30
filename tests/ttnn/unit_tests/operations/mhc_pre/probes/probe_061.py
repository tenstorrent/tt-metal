import torch, ttnn, importlib.util, pathlib, os, json
import ttnn.operations.mhc_pre.mhc_pre_program_descriptor as pd
from ttnn.operations.mhc_pre import mhc_pre

spec = importlib.util.spec_from_file_location("acc", "tests/ttnn/unit_tests/operations/mhc_pre/test_mhc_pre.py")
acc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acc)
dev = ttnn.open_device(device_id=0)
XD = getattr(ttnn, os.environ.get("XD", "float32"))
WD = getattr(ttnn, os.environ.get("WD", "bfloat16"))
xs = tuple(json.loads(os.environ.get("XS", "[1,1,640,7168]")))
for knobs in json.loads(os.environ["CFGS"]):
    saved = {k: getattr(pd, k) for k in knobs}
    for k, v in knobs.items():
        setattr(pd, k, v)
    for seed in (7, 8, 7, 8):
        x, w, b, scale = acc.make_inputs(xs, seed=seed)
        if WD == ttnn.bfloat16:
            w = w.to(torch.bfloat16).float()
        tx, tw, tb = acc.to_dev(x, dev, XD), acc.to_dev(w, dev, WD), acc.to_dev(b, dev, ttnn.float32)
        y, post, comb = mhc_pre(tx, tw, tb, scale=scale)
        yr, pr, cr = acc.torch_mhc_pre(x, w, b, scale, iters=20)
        g = ttnn.to_torch(y).float()
        err = (g - yr.float()).abs()
        rows = torch.nonzero((err.reshape(-1, g.shape[-1]) > 0.05).any(-1)).flatten()
        print("RES", knobs, seed, "y badrows", rows.numel(), sorted(set((rows // 32).tolist())))
    for k, v in saved.items():
        setattr(pd, k, v)
ttnn.close_device(dev)
