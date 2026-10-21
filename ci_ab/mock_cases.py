# Runs every case of test_cls_ab.py once on the device (mock or card), one line per case.
import importlib.util, os, sys
import ttnn
spec = importlib.util.spec_from_file_location("test_cls_ab", os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_cls_ab.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
dev = ttnn.open_device(device_id=0, l1_small_size=32768)
ok = fail = 0
for c in m.CASES:
    try:
        m.run_case(dev, c); ok += 1
    except Exception as e:
        fail += 1; print("FAIL", c, str(e).splitlines()[0][:200] if str(e) else type(e).__name__, flush=True)
print(f"done ok={ok} fail={fail}", flush=True)
ttnn.close_device(dev)
