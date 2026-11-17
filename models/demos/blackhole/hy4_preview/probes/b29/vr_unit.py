"""B29: run tests/test_hy4_vrows_device.py (TEST=pcc VAR=cmp|xmc ...) on a 1x8 slot"""
import os
import runpy

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
runpy.run_path(os.path.join(HERE, "tests", "test_hy4_vrows_device.py"), run_name="__main__")
