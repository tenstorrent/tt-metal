This folder consist the base pipeline on which optimization is applied.

FPS as on 25/07/2024:
FPS (MatMul/Conv Ops only): 907.277
FPS (Other Device Ops): 3234.435
FPS (All Ops): 794.978

Command to generate the perf sheet: `./tt_metal/tools/profiler/profile_this.py -n head_exp1 -c "pytest models/experimental/yolov4/ttnn_experiment1/head_exp1.py"`.

Please build profiler before running the command.
If assert len(deviceOps[device]) == len issue is encountered please comment that assert statement in process_ops_logs.py file.

Note: The FPS may change for each run.
