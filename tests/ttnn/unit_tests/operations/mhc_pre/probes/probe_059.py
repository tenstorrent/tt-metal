import torch, ttnn, importlib.util, pathlib
import ttnn.operations.mhc_pre.mhc_pre_program_descriptor as pd
from ttnn.operations.mhc_pre import mhc_pre

spec = importlib.util.spec_from_file_location("acc", "tests/ttnn/unit_tests/operations/mhc_pre/test_mhc_pre.py")
acc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acc)
dev = ttnn.open_device(device_id=0)
for knobs in [dict(READER_NOC_FLIP_ROWS=0), dict(), dict(READER_NOC_FLIP_ROWS=0, W_SHARE_ON_READER=False)]:
    saved = {k: getattr(pd, k) for k in knobs}
    for k, v in knobs.items():
        setattr(pd, k, v)
    for seed in (7, 8, 7):
        x, w, b, scale = acc.make_inputs((1, 1, 640, 4 * 1792), seed=seed)
        w = w.to(torch.bfloat16).float()
        tx, tw, tb = (
            acc.to_dev(x, dev, ttnn.float32),
            acc.to_dev(w, dev, ttnn.bfloat16),
            acc.to_dev(b, dev, ttnn.float32),
        )
        if seed == 7:
            p = pd.make_plan(dev, tx, tw, 4)
            print(
                "PLAN gw",
                p.group_w,
                "gh",
                p.group_h,
                "groups",
                p.num_groups,
                "cc",
                p.core_c_tiles,
                "bt",
                p.block_token_tiles,
            )
        y, post, comb = mhc_pre(tx, tw, tb, scale=scale)
        yr, pr, cr = acc.torch_mhc_pre(x, w, b, scale, iters=20)
        out = []
        for nm, t, r in (("y", y, yr), ("post", post, pr), ("comb", comb, cr)):
            g = ttnn.to_torch(t).float()
            bad = ~torch.isfinite(g)
            err = (g - r.float()).abs()
            rows = torch.nonzero((err.reshape(-1, g.shape[-1]) > 0.05).any(-1)).flatten()
            out.append(f"{nm}: nonfinite={bad.sum().item()} badrows={rows.numel()} first={rows[:6].tolist()}")
        print("RES", knobs, seed, out)
    for k, v in saved.items():
        setattr(pd, k, v)
ttnn.close_device(dev)
