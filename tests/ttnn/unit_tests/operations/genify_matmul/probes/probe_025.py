import math, torch, ttnn
import eval.golden_tests.genify_matmul.helpers as h
from eval.golden_tests.genify_matmul.feature_spec import LOOSE_CASES

captured = {}
orig = h.check_output


def spy(out, expected, **kw):
    got = ttnn.to_torch(out).float().reshape(expected.shape)
    e = expected.float()
    pcc = torch.corrcoef(torch.stack([got.flatten(), e.flatten()]))[0, 1].item()
    rms = (torch.sqrt(((got - e) ** 2).mean()) / torch.sqrt((e**2).mean())).item()
    captured["v"] = (pcc, rms, kw.get("tolerance"))
    return orig(out, expected, **kw)


h.check_output = spy
device = ttnn.open_device(device_id=0)
try:
    cases = [c for c in LOOSE_CASES if c["placement"] == "height_sharded"][:3] + [LOOSE_CASES[2]]
    for c in cases:
        h.run_genify_matmul(
            c["inputs"],
            dtype=c["dtype"],
            weight_dtype=c["weight_dtype"],
            layout=c["layout"],
            fp32_dest_acc_en=c["fp32_dest_acc_en"],
            placement=c["placement"],
            epilogue=c["epilogue"],
            device=device,
            extras=c["extras"],
        )
        print("CASE", c["inputs"], c["placement"], c["epilogue"], "pcc=%.6f rms=%.4f tol=%s" % captured["v"])
    for pl in ("height_sharded", "interleaved", "block_sharded"):
        for acc in (True, False):
            h.run_genify_matmul(
                ((1, 4, 128, 128), (1, 4, 128, 256)),
                dtype=ttnn.bfloat16,
                weight_dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                fp32_dest_acc_en=acc,
                placement=pl,
                epilogue="none",
                device=device,
            )
            print("CASE batched", pl, "acc", acc, "pcc=%.6f rms=%.4f tol=%s" % captured["v"])
    h.run_genify_matmul(
        ((1, 2048), (2048, 1536)),
        dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        fp32_dest_acc_en=False,
        placement="height_sharded",
        epilogue="none",
        device=device,
    )
    print("CASE hs M=1 K=2048 dest16", "pcc=%.6f rms=%.4f tol=%s" % captured["v"])
finally:
    ttnn.close_device(device)
