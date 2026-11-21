"""Tests for compute-kernel API ordering static checker."""
import unittest
from tools.check_kernel_api_order import KernelApiOrderChecker

class TestKernelApiOrderChecker(unittest.TestCase):
    def setUp(self):
        self.checker = KernelApiOrderChecker()

    def test_valid_ordering_passes(self):
        valid_kernel = """
        void MAIN {
            mm_init();
            matmul_tiles(0, 1, 0, 0);
            mm_uninit();
        }
        """
        violations = self.checker.check_source(valid_kernel)
        self.assertEqual(len(violations), 0)

    def test_compute_before_init_detected(self):
        invalid_kernel = """
        void MAIN {
            matmul_tiles(0, 1, 0, 0);
        }
        """
        violations = self.checker.check_source(invalid_kernel)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["rule"], "COMPUTE_BEFORE_INIT")

    def test_double_init_without_uninit_detected(self):
        invalid_kernel = """
        void MAIN {
            mm_init();
            binary_init();
        }
        """
        violations = self.checker.check_source(invalid_kernel)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["rule"], "DOUBLE_INIT_WITHOUT_UNINIT")

if __name__ == "__main__":
    unittest.main()
