import sys

sys.path.insert(
    0,
    "/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/rms_norm_ttnn_run1/tt-metal/ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/gather_transport",
)
import gather_bench as B

B.sweep(
    [(7, 4)], ["col_2x1024", "col_2x1024_flush", "col_2x1024_dualnoc_flush", "row_2x64_flush", "sem_only"], trials=5
)
