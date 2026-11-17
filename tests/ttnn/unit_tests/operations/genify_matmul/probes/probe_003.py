import math, torch, ttnn
from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor, pytorch_genify_matmul, quantize
from eval.metrics import compute_metrics_torch
from ttnn.operations.genify_matmul import genify_matmul

device = ttnn.open_device(device_id=0)
try:
    for a_shape, b_shape in [((1, 384, 1536), (1536, 1024)), ((1, 64, 16384), (16384, 512))]:
        torch.manual_seed(0)
        A = torch.randn(a_shape)
        B = torch.randn(b_shape) / math.sqrt(b_shape[-2])
        ta = create_ttnn_input_tensor(A, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        tb = create_ttnn_input_tensor(B, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        tb_l1 = create_ttnn_input_tensor(
            B, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        exp = pytorch_genify_matmul(quantize(A, ttnn.bfloat16), quantize(B, ttnn.bfloat16))
        for label, kw, w in [
            ("out_fp32", dict(dtype=ttnn.float32), tb),
            ("out_bf8b", dict(dtype=ttnn.bfloat8_b), tb),
            ("w_l1", {}, tb_l1),
        ]:
            try:
                o = genify_matmul(ta, w, **kw)
                m = compute_metrics_torch(ttnn.to_torch(o).float(), exp, readback_dtype=torch.float32)
                print(
                    f"PROBE {a_shape} {label}: out dtype {o.dtype} pcc={m.pcc:.7f} rms={m.rms:.5f} max={m.max_abs_diff:.5f}"
                )
            except Exception as e:
                print(f"PROBE {a_shape} {label}: EXC {type(e).__name__}: {str(e)[:200]}")
finally:
    ttnn.close_device(device)
