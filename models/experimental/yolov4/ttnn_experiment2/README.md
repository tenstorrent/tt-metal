This folder consist of optimization of DS1 using enable_act_double_buffer = True and enable_split_reader = True wherever possible.

FPS as on 24/07/2024:
FPS (MatMul/Conv Ops only): 5342.28
FPS (Other Device Ops): 4094.233
FPS (All Ops): 2770.874

Command to generate the perf sheet: `./tt_metal/tools/profiler/profile_this.py -n down1_exp2 -c "pytest models/experimental/yolov4/ttnn_experiment2/downsample1_exp2.py"`.

Please build profiler before running the command.
If assert len(deviceOps[device]) == len issue is encountered please comment that assert statement in process_ops_logs.py file.

Note: The FPS may change for each run.
