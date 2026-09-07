import sys

sys.path.insert(
    0,
    "/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/rms_norm_ttnn_run1/tt-metal/ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/gather_transport",
)
import gather_bench as B

V = ["col_2x1024", "row_2x64_flush", "xs_copy_col_flush", "xs_xpose_col_flush", "row_flush_xb_dest", "row_xdest_full"]
B.sweep([(7, 4)], V, trials=5)
