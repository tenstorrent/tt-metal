"""Host-only tests for the standalone benchmark_work.py patch proposal."""
import importlib.util
import pathlib
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve()
while not (ROOT / "models/autoports/google_gemma_4_31b_it/tools/benchmark_work.py").exists():
    ROOT = ROOT.parent
SOURCE = ROOT / "models/autoports/google_gemma_4_31b_it/tools/benchmark_work.py"
PATCH = pathlib.Path(__file__).with_name("benchmark_work_precision_groups.patch")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def policy(attention="bfp4", mlp="bfp4", down="bfp4", head="bfp4"):
    fidelities = {
        "bfp4": "LoFi",
        "bfp4_hifi2": "HiFi2",
        "bfp8": "HiFi2",
        "bfp8_lofi": "LoFi",
        "bf16": "HiFi4",
    }
    groups = {"attention": attention, "mlp": mlp, "down": down, "lm_head": head}
    return {
        "weight_groups": groups,
        "layer_exceptions": {},
        "kv_cache_dtype": "bfloat8_b",
    }


class PrecisionGroupsPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = load_module(SOURCE, "benchmark_work_original")
        cls.temp = tempfile.TemporaryDirectory()
        work = pathlib.Path(cls.temp.name)
        shutil.copy2(SOURCE, work / "benchmark_work.py")
        subprocess.run(["patch", "-p1", "-i", str(PATCH)], cwd=work, check=True, capture_output=True, text=True)
        cls.candidate = load_module(work / "benchmark_work.py", "benchmark_work_candidate")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_legacy_baseline_and_head_only_numbers_are_exact(self):
        positions = [0, 1, 17, 1023, 1024, 8191]
        for batch in (1, 7, 32):
            for precision in (
                None,
                policy(),
                policy(head="bfp8"),
                policy(head="bf16"),
            ):
                with self.subTest(batch=batch, precision=precision):
                    expected = self.original.decode_work(positions, batch, precision)
                    actual = self.candidate.decode_work(positions, batch, precision)
                    self.assertEqual(actual, expected)

    def test_group_element_split_reconstructs_existing_projection_total(self):
        for n, d, kv, _window, shared in self.candidate.LAYERS:
            grouped = self.candidate.projection_group_elements(d, kv, shared, physical=True)
            self.assertEqual(sum(grouped.values()), self.candidate.projection_elements(d, kv, shared, physical=True))

    def test_down_bfp4_to_bfp8_adds_analytical_physical_bytes(self):
        positions = [0, 1, 256, 1023]
        baseline = self.candidate.decode_work(positions, 1, policy())
        changed = self.candidate.decode_work(positions, 1, policy(down="bfp8"))
        elements = sum(
            n * self.candidate.projection_group_elements(d, kv, shared, physical=True)["down"]
            for n, d, kv, _window, shared in self.candidate.LAYERS
        )
        expected_delta = (self.candidate.BFP8_BYTES - self.candidate.BFP4_BYTES) * elements
        self.assertEqual(changed["weight_bytes"] - baseline["weight_bytes"], expected_delta)
        self.assertEqual(changed["dram_bytes"] - baseline["dram_bytes"], expected_delta)
        self.assertEqual(expected_delta, elements / 2)

    def test_precision_label_bytes_ignore_compute_fidelity_suffix(self):
        for a, b in (("bfp4", "bfp4_hifi2"), ("bfp8", "bfp8_lofi")):
            left = self.candidate.decode_work([0, 4], 1, policy(down=a))
            right = self.candidate.decode_work([0, 4], 1, policy(down=b))
            self.assertEqual(left["weight_bytes"], right["weight_bytes"])

    def test_unsupported_layer_exceptions_kv_and_labels_rejected(self):
        bad = policy(down="bfp8")
        bad["layer_exceptions"] = {"0": {"down": "bfp4"}}
        with self.assertRaises(ValueError):
            self.candidate.decode_work([0], 1, bad)
        bad = policy(down="bfp8")
        bad["kv_cache_dtype"] = "bfloat16"
        with self.assertRaises(ValueError):
            self.candidate.decode_work([0], 1, bad)
        bad = policy(down="bfp8")
        bad["weight_groups"].pop("mlp")
        with self.assertRaises(ValueError):
            self.candidate.decode_work([0], 1, bad)
        bad = policy(down="int4")
        with self.assertRaises(KeyError):
            self.candidate.decode_work([], 1, bad)


if __name__ == "__main__":
    unittest.main()
