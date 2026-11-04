import torch, ttnn, sys

sys.path.insert(0, ".")
from eval.golden_tests.nomic_residual_layer_norm.helpers import (
    make_inputs,
    reference_layer_norm,
    to_device,
    compute_config,
)
from ttnn.operations.nomic_residual_layer_norm import nomic_residual_layer_norm, _placement, _plan

device = ttnn.open_device(device_id=0)
try:
    for shape in [(2, 1, 512, 768), (8, 1, 512, 768)]:
        for in_place in (False, True):
            x, r, g, b = make_inputs(shape, 0)
            exp = reference_layer_norm(x, g, b, r).float()
            tx, tr, tg, tb = to_device(x, r, g, b, device, "model_layout")
            print("placement", _placement(tx), _plan(shape, device, "model_layout").block_rows)
            kw = dict(residual=tr, compute_kernel_config=compute_config())
            if in_place:
                kw["output_tensor"] = tx
            out = nomic_residual_layer_norm(tx, tg, tb, **kw)
            print("same buf", out.buffer_address() == tx.buffer_address())
            got = ttnn.to_torch(out).float()
            pcc = torch.corrcoef(torch.stack([got.flatten(), exp.flatten()]))[0, 1].item()
            rms = (((got - exp) ** 2).mean().sqrt() / exp.std()).item()
            print(shape, in_place, "pcc", pcc, "rms", rms)
finally:
    ttnn.close_device(device)
