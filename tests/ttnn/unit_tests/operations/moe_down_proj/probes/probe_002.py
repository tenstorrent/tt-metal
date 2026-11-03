import sys, torch, ttnn

sys.path.insert(0, "tests/ttnn/unit_tests/operations/moe_down_proj")
import test_moe_down_proj as T

device = ttnn.open_device(device_id=0)
for name, tokens, counts in T.CASES[:5]:
    x_dev, w_dev, c_dev, xq, wq = T.make_operands(device, tokens, counts)
    prev = None
    for it in range(2):
        out = T.moe_down_proj(x_dev, w_dev, c_dev)
        rows = xq.shape[-2]
        exp, real = T.reference(xq, wq, counts)
        act = ttnn.to_torch(out).reshape(rows, 768).float()
        a, e = act[real], exp[real]
        print(
            "RES",
            name,
            it,
            "pcc",
            T.pcc(a, e),
            "rms",
            T.relative_rms(a, e),
            "nan",
            torch.isnan(a).sum().item(),
            "same_as_prev",
            None if prev is None else torch.equal(prev, a),
        )
        prev = a
ttnn.close_device(device)
