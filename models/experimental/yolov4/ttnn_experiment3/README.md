This folder consist optimization of DS1 by removing reshard=True wherever possible and using block sharding wherever needed (This is done using perf sheet analyze, If the height sharding conv core count is less then changed to block sharding)

FPS as on 24/07/2024:
FPS (MatMul/Conv Ops only): 5216.511
FPS (Other Device Ops): 4156.898
FPS (All Ops): 2764.806

Command to generate the perf sheet: `./tt_metal/tools/profiler/profile_this.py -n down1_exp3 -c "pytest models/experimental/yolov4/ttnn_experiment3/downsample1_exp3.py"`.

Please build profiler before running the command.
If assert len(deviceOps[device]) == len issue is encountered please comment that assert statement in process_ops_logs.py file.

Note: The FPS may change for each run.
