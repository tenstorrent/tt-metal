"""B26: run the fuse-sparse-merge unit test (tests/test_hy4_fused_merge_device.py: TEST=xpcc|xlat|...) on a 1x8 slot; env FMTS/SS/CASES filter."""
import os
import runpy

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
runpy.run_path(os.path.join(HERE, "tests", "test_hy4_fused_merge_device.py"), run_name="__main__")
