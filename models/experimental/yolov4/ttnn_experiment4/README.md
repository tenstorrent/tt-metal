This folder consist of optimization of DS5 using reallocate_halo_output=True wherever possible.

FPS:
FPS (MatMul/Conv Ops only): 4693.778
FPS (Other Device Ops): 2452.014
FPS (All Ops): 2161.237

Command to generate the perf sheet: `./tt_metal/tools/profiler/profile_this.py -n down5_exp4 -c "pytest models/experimental/yolov4/ttnn_experiment4/downsample5_exp4.py"`.

Please build profiler before running the command.
If assert len(deviceOps[device]) == len issue is encountered please comment that assert statement in process_ops_logs.py file.

Note: The FPS may change for each run.
