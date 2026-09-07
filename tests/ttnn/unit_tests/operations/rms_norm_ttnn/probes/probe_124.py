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
    "row_2x64",
    "row_2x64_flush",
    "col_2x1024_dualnoc",
    "row_2x64_dualnoc",
    "row_xpose_send",
    "row_xpose_full",
    "col_xpose_full",
]
B.sweep([(7, 4)], V, trials=5)
