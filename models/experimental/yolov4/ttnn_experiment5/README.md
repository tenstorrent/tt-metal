This folder consist of optimization of DS4 using bfloat_8 for conv dtype wherever possible.

FPS:
FPS (MatMul/Conv Ops only): 1469.298
FPS (Other Device Ops): 1328.905
FPS (All Ops): 909.992

Command to generate the perf sheet: `./tt_metal/tools/profiler/profile_this.py -n down4_exp5 -c "pytest models/experimental/yolov4/ttnn_experiment5/downsample4_exp5.py"`.

Please build profiler before running the command.
If assert len(deviceOps[device]) == len issue is encountered please comment that assert statement in process_ops_logs.py file.

Note: The FPS may change for each run.
