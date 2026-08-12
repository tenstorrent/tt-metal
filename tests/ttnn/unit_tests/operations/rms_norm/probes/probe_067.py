import importlib.util

p = "/localdev/mstaletovic/2026_08_11/1659_mstaletovic_agent_eval/clones/rms_norm_run1/tt-metal/ttnn/ttnn/operations/rms_norm/perf_experiments/colvalid_payload/bench.py"
spec = importlib.util.spec_from_file_location("colvalid_bench", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
# POISON: stamp NaN/-Inf into every face a short payload never writes.
# (0,poison) is the control: mode 0 overwrites all four faces, so it must pass.
m.sweep(
    [
        (0, "decode7168", 1),
        (1, "decode7168", 1),
        (2, "decode7168", 1),
        (1, "wshard1024", 1),
        (1, "bshard1024", 1),
    ]
)
