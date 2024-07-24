This folder consist the base pipeline on which optimization is applied.

FPS as on 24/07/2024:
FPS (MatMul/Conv Ops only): 5214.009
FPS (Other Device Ops): 4164.463
FPS (All Ops): 2767.346

Command to generate the perf sheet: `./tt_metal/tools/profiler/profile_this.py -n down1_exp1 -c "pytest models/experimental/yolov4/ttnn_experiment1/downsample1_exp1.py"`.

Please build profiler before running the command.
If assert len(deviceOps[device]) == len issue is encountered please comment that assert statement in process_ops_logs.py file.

Note: The FPS may change for each run.
