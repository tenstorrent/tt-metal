This folder consist of optimization of DS2 using reallocate_halo_output=True wherever possible.

FPS as on 24/07/2024:
FPS (MatMul/Conv Ops only): 8309.789
FPS (Other Device Ops): 5334.386
FPS (All Ops): 4464.963

Command to generate the perf sheet: `./tt_metal/tools/profiler/profile_this.py -n down2_exp4 -c "pytest models/experimental/yolov4/ttnn_experiment4/downsample2_exp4.py"`.

Please build profiler before running the command.
If assert len(deviceOps[device]) == len issue is encountered please comment that assert statement in process_ops_logs.py file.

Note: The FPS may change for each run.
