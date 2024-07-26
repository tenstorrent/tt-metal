This folder consist optimization of Head sub_module by removing reshard=True wherever possible and using block sharding wherever needed (This is done using perf sheet analyze, If the height sharding conv core count is less then changed to block sharding)

FPS as on 25/07/2024:
FPS (MatMul/Conv Ops only): 1440.698
FPS (Other Device Ops): 3948.714
FPS (All Ops): 1259.406

Command to generate the perf sheet: `./tt_metal/tools/profiler/profile_this.py -n head_exp3 -c "pytest models/experimental/yolov4/ttnn_experiment3/head_exp3.py"`.

Please build profiler before running the command.
If assert len(deviceOps[device]) == len issue is encountered please comment that assert statement in process_ops_logs.py file.

Note: The FPS may change for each run.
