import os
import time

import tt_lib as ttl

def one_second_profile (name):
    ttl.profiler.start_profiling(name)

    ttl.profiler.append_meta_data(f"{name}_meta_1")
    ttl.profiler.append_meta_data(f"{name}_meta_2")

    time.sleep(1)

    ttl.profiler.start_profiling(f"{name[:-1]}2")
    ttl.profiler.append_meta_data(f"{name[:-1]}2_meta_1")
    time.sleep(1)
    ttl.profiler.append_meta_data(f"{name[:-1]}2_meta_2")
    ttl.profiler.stop_profiling(f"{name[:-1]}2")

    time.sleep(2)
    ttl.profiler.append_meta_data(f"{name}_meta_3")
    ttl.profiler.stop_profiling(name)

def test_nested_profiling():
    testLocation ="tt_metal/tools/profiler/logs/test_profiler/nested_profiling"
    os.system(f"rm -rf {testLocation}")
    ttl.profiler.set_profiler_flag(True)
    ttl.profiler.set_profiler_location(testLocation)

    one_second_profile ("first_level_1")
    time.sleep(0.1)
    one_second_profile ("first_level_1")
    time.sleep(0.1)
    one_second_profile ("second_level_1")
