import importlib.util

p = "/localdev/mstaletovic/2026_08_11/1659_mstaletovic_agent_eval/clones/rms_norm_run1/tt-metal/ttnn/ttnn/operations/rms_norm/perf_experiments/colvalid_payload/bench.py"
spec = importlib.util.spec_from_file_location("colvalid_bench", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.sweep([(1, "decode7168", 0), (1, "bshard1024", 0), (2, "decode7168", 0), (1, "hshard512", 0)], keep_log=False)
