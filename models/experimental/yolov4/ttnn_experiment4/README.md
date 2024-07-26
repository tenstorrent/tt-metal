This folder consist of optimization of DS3 using reallocate_halo_output=True wherever possible.

FPS:
FPS (MatMul/Conv Ops only): 3273.74
FPS (Other Device Ops): 2123.44
FPS (All Ops): 1732.904

Command to generate the perf sheet: `./tt_metal/tools/profiler/profile_this.py -n down3_exp4 -c "pytest models/experimental/yolov4/ttnn_experiment4/downsample3_exp4.py"`.

Please build profiler before running the command.
If assert len(deviceOps[device]) == len issue is encountered please comment that assert statement in process_ops_logs.py file.

Note: The FPS may change for each run.
