This folder consist of optimization of head sub_module using enable_act_double_buffer = True and enable_split_reader = True wherever possible.

FPS as on 25/07/2024:
FPS (MatMul/Conv Ops only): 908.273
FPS (Other Device Ops): 3234.456
FPS (All Ops): 795.032

Command to generate the perf sheet: `./tt_metal/tools/profiler/profile_this.py -n head_exp2 -c "pytest models/experimental/yolov4/ttnn_experiment2/head_exp2.py"`.

Please build profiler before running the command.
If assert len(deviceOps[device]) == len issue is encountered please comment that assert statement in process_ops_logs.py file.

Note: The FPS may change for each run.
