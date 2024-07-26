This folder consist of optimization of Head sub_module using reallocate_halo_output=True wherever possible.

FPS as on 25/07/2024:
FPS (MatMul/Conv Ops only): 907.986
FPS (Other Device Ops): 3128.392
FPS (All Ops): 787.222

Command to generate the perf sheet: `./tt_metal/tools/profiler/profile_this.py -n head_exp4 -c "pytest models/experimental/yolov4/ttnn_experiment4/head_exp4.py"`.

Please build profiler before running the command.
If assert len(deviceOps[device]) == len issue is encountered please comment that assert statement in process_ops_logs.py file.

Note: The FPS may change for each run.
