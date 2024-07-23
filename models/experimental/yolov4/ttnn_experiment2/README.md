This folder consist of optimization of DS5 using enable_act_double_buffer = True and enable_split_reader = True wherever possible.

FPS:
FPS (MatMul/Conv Ops only): 2521.807
FPS (Other Device Ops): 3496.125
FPS (All Ops): 1878.89

Command to generate the perf sheet: `./tt_metal/tools/profiler/profile_this.py -n down5_exp2 -c "pytest models/experimental/yolov4/ttnn_experiment2/downsample5_exp2.py"`.

Please build profiler before running the command.
If assert len(deviceOps[device]) == len issue is encountered please comment that assert statement in process_ops_logs.py file.

Note: The FPS may change for each run.
