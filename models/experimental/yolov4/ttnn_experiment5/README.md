This folder consist of optimization of DS2 using bfloat_8 for conv dtype wherever possible.

FPS as on 24/07/2024:
FPS (MatMul/Conv Ops only): 10573.06
FPS (Other Device Ops): 7729.469
FPS (All Ops): 5880.623

Command to generate the perf sheet: `./tt_metal/tools/profiler/profile_this.py -n down2_exp5 -c "pytest models/experimental/yolov4/ttnn_experiment5/downsample2_exp5.py"`.

Please build profiler before running the command.
If assert len(deviceOps[device]) == len issue is encountered please comment that assert statement in process_ops_logs.py file.

Note: The FPS may change for each run.
