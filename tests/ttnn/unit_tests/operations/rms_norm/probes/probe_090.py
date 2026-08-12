import importlib.util

p = "/localdev/mstaletovic/2026_08_11/1659_mstaletovic_agent_eval/clones/rms_norm_run1/tt-metal/ttnn/ttnn/operations/rms_norm/perf_experiments/colvalid_payload/bench.py"
spec = importlib.util.spec_from_file_location("colvalid_bench", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
tri = []
for _ in range(3):
    tri += [(0, "decode1024", 0), (1, "decode1024", 0), (2, "decode1024", 0)]
for _ in range(2):
    tri += [(0, "wtail", 0), (1, "wtail", 0), (0, "rm_interleaved", 0), (1, "rm_interleaved", 0)]
m.sweep(tri, keep_log=False)
