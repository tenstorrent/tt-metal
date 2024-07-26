This folder consist optimization of DS5 by removing reshard=True wherever possible and using block sharding wherever needed (This is done using perf sheet analyze, If the height sharding conv core count is less then changed to block sharding)

FPS:
FPS (MatMul/Conv Ops only): 2645.328
FPS (Other Device Ops): 4079.268
FPS (All Ops): 2027.974

Command to generate the perf sheet: `./tt_metal/tools/profiler/profile_this.py -n down5_exp3 -c "pytest models/experimental/yolov4/ttnn_experiment3/downsample5_exp3.py"`.

Please build profiler before running the command.
If assert len(deviceOps[device]) == len issue is encountered please comment that assert statement in process_ops_logs.py file.

Note: The FPS may change for each run.
