This folder consist the base pipeline on which optimization is applied.

FPS:
FPS (MatMul/Conv Ops only): 962.32
FPS (Other Device Ops): 1878.058
FPS (All Ops): 761.946

Command to generate the perf sheet: `./tt_metal/tools/profiler/profile_this.py -n down4_exp1 -c "pytest models/experimental/yolov4/ttnn_experiment1/downsample4_exp1.py"`.

Please build profiler before running the command.
If assert len(deviceOps[device]) == len issue is encountered please comment that assert statement in process_ops_logs.py file.

Note: The FPS may change for each run.
