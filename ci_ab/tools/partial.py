# MAX REDUCE_ROW with a partial scaler tile (prepare_reduce_scaler(valid) layout: row 0 of each face, the first `valid`
# columns 1.0 and the rest 0): the zero columns must not take part in the max. usage: partial.py <side>
import sys
import torch
import ttnn
from ttnn_kernels import run_case

side = sys.argv[1]
device = ttnn.open_device(device_id=0)
g = torch.Generator().manual_seed(17)
for valid in (1, 15, 17, 31):
    n = 64
    x = -(torch.rand(n, 32, 32, generator=g) * 100 + 1)  # valid data: all negative
    x[n // 2 :] = torch.rand(n // 2, 32, 32, generator=g) * 100 + 1  # second half: all positive
    x[:, :, valid:] = torch.where(torch.arange(n)[:, None, None] % 4 == 0, 5.0, torch.where(torch.arange(n)[:, None, None] % 4 == 1, 0.0, torch.where(torch.arange(n)[:, None, None] % 4 == 2, float("-inf"), 3.0e38)))
    x = x.bfloat16().float()
    s = torch.zeros(n, 32, 32)
    s[:, 0, :valid] = 1.0
    s[:, 16, :valid] = 1.0
    for d in (False, True):
        y = run_case(device, x, s, "bf16", d).float()
        ref = x[:, :, :valid].amax(dim=2)
        got = y[:, :, 0]
        bad = (got != ref.to(y.dtype).float())
        print(f"PARTIAL {side} valid={valid} dest32={d}: {int(bad.sum())} of {bad.numel()} row maxima differ from the max over the valid columns"
              + (f"; e.g. tile {int(bad.nonzero()[0,0])} row {int(bad.nonzero()[0,1])}: got {float(got[bad][0]):.6g} want {float(ref.to(y.dtype).float()[bad][0]):.6g}" if int(bad.sum()) else ""), flush=True)
ttnn.close_device(device)
