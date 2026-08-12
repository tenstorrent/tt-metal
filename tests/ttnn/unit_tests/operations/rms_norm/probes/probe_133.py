import sys, ttnn

sys.path.insert(0, "tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/mcast_ack_elision")
import bench

dev = ttnn.open_device(device_id=0)
try:
    for _ in range(3):
        bench.main(dev, geo_names=("decode110_b1", "decode110_b4"))
finally:
    ttnn.close_device(dev)
