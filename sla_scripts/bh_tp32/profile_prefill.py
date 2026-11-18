import os

# Called by prefill_profile.sh with TAG set: profiles tests/test_prefill_layer_prof.py under the device profiler.
from tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

subdir = "bh-prefill-layer-" + os.environ.get("TAG", "x")
cmd = "pytest models/demos/llama3_70b_galaxy/tests/test_prefill_layer_prof.py -s --timeout=0"
run_device_profiler(cmd, subdir, check_test_return_code=False, device_analysis_types=["device_kernel_duration"])
print("OPS_CSV", get_latest_ops_log_filename(subdir))
