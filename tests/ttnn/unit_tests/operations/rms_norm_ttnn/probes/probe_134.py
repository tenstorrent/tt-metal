import sys

sys.path.insert(
    0,
    "/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/rms_norm_ttnn_run1/tt-metal/ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/gather_transport",
)
import gather_bench as B

V = [
    "col_2x1024",
    "sem_only",
    "col_2x1024_flush",
    "col_2x1024_dualnoc",
    "col_2x1024_dualnoc_flush",
    "row_2x64_flush",
    "row_flush_xb1",
    "row_2x64_dualnoc",
]
B.sweep([(8, 1), (3, 3), (7, 4), (8, 4), (8, 8)], V, trials=5)
