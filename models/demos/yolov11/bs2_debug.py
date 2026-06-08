import sys, traceback, torch, ttnn
from models.demos.yolov11.common import YOLOV11_L1_SMALL_SIZE
from models.demos.yolov11.runner.performant_runner import YOLOv11PerformantRunner
BS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
d=ttnn.CreateDevice(0,l1_small_size=YOLOV11_L1_SMALL_SIZE,trace_region_size=23887872,num_command_queues=2)
try:
    r=YOLOv11PerformantRunner(d,BS,ttnn.bfloat8_b,ttnn.bfloat8_b,resolution=(640,640),model_location_generator=None)
    print("RUNNER OK")
except Exception:
    traceback.print_exc()
finally:
    ttnn.close_device(d)
