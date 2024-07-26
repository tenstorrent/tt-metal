This folder consist of optimization of DS3 using enable_act_double_buffer = True and enable_split_reader = True wherever possible.

FPS:
FPS (MatMul/Conv Ops only): 3061.09
FPS (Other Device Ops): 2469.679
FPS (All Ops): 1770.861

Command to generate the perf sheet: `./tt_metal/tools/profiler/profile_this.py -n down3_exp2 -c "pytest models/experimental/yolov4/ttnn_experiment2/downsample3_exp2.py"`.

Please build profiler before running the command.
If assert len(deviceOps[device]) == len issue is encountered please comment that assert statement in process_ops_logs.py file.

Note: The FPS may change for each run.
