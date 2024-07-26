This folder consist of optimization of Neck sub_module using enable_act_double_buffer = True and enable_split_reader = True wherever possible.

FPS as on 25/07/2024:
FPS (MatMul/Conv Ops only): 1275.204
FPS (Other Device Ops): 15.791
FPS (All Ops): 15.649

Command to generate the perf sheet: `./tt_metal/tools/profiler/profile_this.py -n neck_exp2 -c "pytest models/experimental/yolov4/ttnn_experiment2/neck_exp2.py"`.

Please build profiler before running the command.
If assert len(deviceOps[device]) == len issue is encountered please comment that assert statement in process_ops_logs.py file.

Note: The FPS may change for each run.
