This folder consist of optimization of Head sub_module using bfloat_8 for conv dtype wherever possible.

FPS as on 25/07/2024:
FPS (MatMul/Conv Ops only): 969.501
FPS (Other Device Ops): 3465.448
FPS (All Ops): 855.65

Command to generate the perf sheet: `./tt_metal/tools/profiler/profile_this.py -n head_exp5 -c "pytest models/experimental/yolov4/ttnn_experiment5/head_exp5.py"`.

Please build profiler before running the command.
If assert len(deviceOps[device]) == len issue is encountered please comment that assert statement in process_ops_logs.py file.

Note: The FPS may change for each run.
