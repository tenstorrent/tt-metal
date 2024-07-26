This folder consist of optimization of DS1 using reallocate_halo_output=True wherever possible.

FPS as on 24/07/2024:
FPS (MatMul/Conv Ops only): 5681.592
FPS (Other Device Ops): 3584.949
FPS (All Ops): 2650.565

Command to generate the perf sheet: `./tt_metal/tools/profiler/profile_this.py -n down1_exp4 -c "pytest models/experimental/yolov4/ttnn_experiment4/downsample1_exp4.py"`.

Please build profiler before running the command.
If assert len(deviceOps[device]) == len issue is encountered please comment that assert statement in process_ops_logs.py file.

Note: The FPS may change for each run.
