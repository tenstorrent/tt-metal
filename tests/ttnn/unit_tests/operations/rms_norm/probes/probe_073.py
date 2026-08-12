import importlib.util

p = "/localdev/mstaletovic/2026_08_11/1659_mstaletovic_agent_eval/clones/rms_norm_run1/tt-metal/ttnn/ttnn/operations/rms_norm/perf_experiments/colvalid_payload/bench.py"
spec = importlib.util.spec_from_file_location("colvalid_bench", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
tri = []
for c in ["decode1024", "wtail", "hshard512", "rm_interleaved", "prefill7168"]:
    tri += [(0, c, 0), (1, c, 0)]
# fp32_dest_acc_en=True: the stat tile is 4096 B, so a face is 1024 B.
tri += [(0, "decode7168", 0, True), (1, "decode7168", 0, True)]
m.sweep(tri)
