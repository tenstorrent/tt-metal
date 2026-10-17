import subprocess, os
r = subprocess.run(["./build_Release/test/tt_metal/unit_tests_data_movement", "--gtest_filter=*NocEstimatorDRAMInterleavedOneFromAll*"], capture_output=True, text=True)
print(r.stdout[-3000:]); print(r.stderr[-2000:])
