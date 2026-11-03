import sys, torch, ttnn

sys.path.insert(0, "tests/ttnn/unit_tests/operations/moe_down_proj")
import test_moe_down_proj as T

device = ttnn.open_device(device_id=0)
for name, tokens, counts in T.CASES[:3]:
    x_dev, w_dev, c_dev, xq, wq = T.make_operands(device, tokens, counts)
    out = T.moe_down_proj(x_dev, w_dev, c_dev)
    rows = xq.shape[-2]
    exp, real = T.reference(xq, wq, counts)
    act = ttnn.to_torch(out).reshape(rows, 768).float()
    print(
        name,
        "real rows",
        real.sum().item(),
        "act[real] abs max",
        act[real].abs().max().item(),
        "exp max",
        exp[real].abs().max().item(),
    )
    print(act[real][:3, :8])
    print(exp[real][:3, :8])
    d = (act[real].double() - exp[real]).abs()
    print("cols bad:", (d.max(0).values > 0.1).nonzero().flatten()[:40].tolist())
ttnn.close_device(device)
