This folder consist of optimization of Neck sub_module using reallocate_halo_output=True wherever possible.

FPS as on 25/07/2024:
FPS (MatMul/Conv Ops only): 1276.523
FPS (Other Device Ops): 15.787
FPS (All Ops): 15.646

Command to generate the perf sheet: `./tt_metal/tools/profiler/profile_this.py -n neck_exp4 -c "pytest models/experimental/yolov4/ttnn_experiment4/neck_exp4.py"`.

Please build profiler before running the command.
If assert len(deviceOps[device]) == len issue is encountered please comment that assert statement in process_ops_logs.py file.

Note: The FPS may change for each run.
