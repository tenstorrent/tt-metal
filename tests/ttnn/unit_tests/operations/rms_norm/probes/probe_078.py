import importlib.util

p = "/localdev/mstaletovic/2026_08_11/1659_mstaletovic_agent_eval/clones/rms_norm_run1/tt-metal/ttnn/ttnn/operations/rms_norm/perf_experiments/colvalid_payload/bench.py"
spec = importlib.util.spec_from_file_location("colvalid_bench", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
# The focus-shape call sits in the ~2-3% noise band -> 3 repeats, take the median.
tri = []
for _ in range(3):
    tri += [(0, "decode7168", 0), (1, "decode7168", 0), (2, "decode7168", 0)]
tri += [(0, "bshard1024", 0), (1, "bshard1024", 0), (2, "bshard1024", 0)]
tri += [(0, "wshard1024", 0), (1, "wshard1024", 0), (2, "wshard1024", 0)]
m.sweep(tri, keep_log=False)
