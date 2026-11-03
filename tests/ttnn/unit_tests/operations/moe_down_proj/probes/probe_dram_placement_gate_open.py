import sys, torch, ttnn

sys.path.insert(0, "/localdev/kgrujcic/tt-metal-codegen")
import ttnn.operations.moe_down_proj

m = sys.modules["ttnn.operations.moe_down_proj.moe_down_proj"]
m.SUPPORTED["buffer_type"].append(ttnn.BufferType.DRAM)
m.SUPPORTED["out_buffer_type"].append(ttnn.BufferType.DRAM)
from tests.ttnn.unit_tests.operations.moe_down_proj.test_moe_down_proj import (
    make_operands,
    reference,
    pcc,
    relative_rms,
    LAYER0,
    CASES,
)

dev = ttnn.open_device(device_id=0)
try:
    for tokens, counts in ((37, CASES[2][2]), (4096, LAYER0)):
        for xm, om in (
            (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
            (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
            (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        ):
            x, w, c, xq, wq = make_operands(dev, tokens, counts, xm)
            out = m.moe_down_proj(x, w, c, memory_config=om)
            rows = xq.shape[-2]
            exp, real = reference(xq, wq, counts)
            act = ttnn.to_torch(out).reshape(rows, 768)[real]
            print("RES", tokens, xm.buffer_type, om.buffer_type, pcc(act, exp[real]), relative_rms(act, exp[real]))
finally:
    ttnn.close_device(dev)
