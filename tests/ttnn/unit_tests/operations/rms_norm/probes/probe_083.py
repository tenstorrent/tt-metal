import importlib.util

p = "/localdev/mstaletovic/2026_08_11/1659_mstaletovic_agent_eval/clones/rms_norm_run1/tt-metal/ttnn/ttnn/operations/rms_norm/perf_experiments/colvalid_payload/bench.py"
spec = importlib.util.spec_from_file_location("colvalid_bench", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
tri = []
for _ in range(3):
    for c in ("wshard1024", "wshard7168"):
        tri += [(0, c, 0), (1, c, 0), (2, c, 0)]
m.sweep(tri, keep_log=False)
