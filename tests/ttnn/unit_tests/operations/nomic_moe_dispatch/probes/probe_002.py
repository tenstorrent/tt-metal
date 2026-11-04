import time, cProfile, pstats, io
import torch, ttnn
from eval.golden_tests.nomic_moe_dispatch.helpers import make_table, gather_rows, table_tensor, index_tensor
from ttnn.operations.nomic_moe_dispatch import nomic_moe_dispatch

device = ttnn.open_device(device_id=0)
T, R = 4096, 8416
tt_table = table_tensor(make_table(T), device, "l1")
tt_index = index_tensor(gather_rows(T, R), device)
out = nomic_moe_dispatch(tt_table, tt_index)
ttnn.synchronize_device(device)
ttnn.deallocate(out)
N = 50
t0 = time.perf_counter()
for _ in range(N):
    out = nomic_moe_dispatch(tt_table, tt_index)
    ttnn.deallocate(out)
ttnn.synchronize_device(device)
print("HOST us/call (incl sync at end):", (time.perf_counter() - t0) / N * 1e6)
pr = cProfile.Profile()
pr.enable()
for _ in range(N):
    out = nomic_moe_dispatch(tt_table, tt_index)
    ttnn.deallocate(out)
pr.disable()
ttnn.synchronize_device(device)
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(15)
print(s.getvalue()[:4000])
ttnn.close_device(device)
