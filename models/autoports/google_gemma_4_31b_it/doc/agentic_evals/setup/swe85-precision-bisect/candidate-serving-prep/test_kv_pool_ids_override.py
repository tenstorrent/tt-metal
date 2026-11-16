"""Host-only tests for the inert GEMMA4_KV_POOL_IDS patch proposal."""
import ast
import importlib.util
import pathlib
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve()
while not (ROOT / "models/autoports/google_gemma_4_31b_it/tt/kv_pool_budget.py").exists():
    ROOT = ROOT.parent
PATCH = pathlib.Path(__file__).with_name("kv_pool_ids_override.patch")
KV_REL = pathlib.Path("models/autoports/google_gemma_4_31b_it/tt/kv_pool_budget.py")
GEN_REL = pathlib.Path("models/autoports/google_gemma_4_31b_it/tt/generator_vllm.py")
PLUGIN_ROOT = pathlib.Path("bringup/artifacts/v4-serving-plugin/src")
PLUGIN_WORKER = PLUGIN_ROOT / "vllm_tt_plugin/worker.py"


def load_module(path):
    spec = importlib.util.spec_from_file_location("candidate_kv_pool_budget", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KVPoolOverridePatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        work = pathlib.Path(cls.temp.name)
        for rel in (KV_REL, GEN_REL):
            dest = work / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, dest)
        subprocess.run(["patch", "-p1", "-i", str(PATCH)], cwd=work, check=True, capture_output=True, text=True)
        cls.module = load_module(work / KV_REL)
        cls.generator_source = (work / GEN_REL).read_text()
        cls.original = load_module(ROOT / KV_REL)
        cls.plugin_worker_source = (ROOT / PLUGIN_WORKER).read_text()
        cls.plugin_pth = next(
            (ROOT / "python_env/lib/python3.10/site-packages").glob("__editable__.vllm_tt_plugin-*.pth")
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_defaults_and_legacy_b8_behavior_are_exact(self):
        for batch in (1, 8, 16, 32):
            with self.subTest(batch=batch):
                self.assertEqual(
                    self.module.get_pool_token_budget(max_model_len=262144, max_num_seqs=batch),
                    self.original.get_pool_token_budget(max_model_len=262144, max_num_seqs=batch),
                )
        self.assertEqual(
            self.module.get_pool_token_budget(max_model_len=262144, max_num_seqs=8, b8_pool_ids="20480"),
            self.original.get_pool_token_budget(max_model_len=262144, max_num_seqs=8, b8_pool_ids="20480"),
        )
        for batch in (1, 16, 32):
            self.assertEqual(
                self.module.get_pool_token_budget(max_model_len=262144, max_num_seqs=batch, b8_pool_ids="20480"),
                self.original.get_pool_token_budget(max_model_len=262144, max_num_seqs=batch, b8_pool_ids="20480"),
            )

    def test_explicit_override_works_at_each_supported_batch_and_logs_exact_pool(self):
        expected_min = {1: 8449, 8: 10248, 16: 12304, 32: 16416}
        for batch in (1, 8, 16, 32):
            with self.subTest(batch=batch):
                ids = expected_min[batch] + 37
                info = self.module.get_pool_budget_info(
                    max_model_len=262144, max_num_seqs=batch, pool_ids_override=str(ids)
                )
                reserves = 32 * batch + 8 * 1024 * batch
                self.assertEqual(info["synthetic_token_budget"], ids * 32 - reserves)
                self.assertEqual(info["pool_ids_override"], ids)
                self.assertEqual(info["pool_override_source"], "GEMMA4_KV_POOL_IDS")
                self.assertEqual(info["pool_total_tokens_including_worker_reserves"], ids * 32)
                self.assertEqual(info["pool_minimum_ids_for_context"], expected_min[batch])
                self.assertEqual(info["worker_output_reserve_tokens"], 32 * batch)
                self.assertEqual(info["worker_sliding_reserve_tokens"], 8 * 1024 * batch)
        self.assertIn("**pool_info", self.generator_source)
        self.assertIn('"pool_ids_override": effective_pool_ids', PATCH.read_text())

    def test_override_rejects_invalid_value_batch_context_and_minimum(self):
        for value in ("", "0", "01", "+10000", "10000.0", " 12000", "-1", 12000):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.module.get_pool_token_budget(max_model_len=262144, max_num_seqs=8, pool_ids_override=value)
        with self.assertRaisesRegex(ValueError, "context-preserving minimum"):
            self.module.get_pool_token_budget(max_model_len=262144, max_num_seqs=8, pool_ids_override="10247")
        with self.assertRaisesRegex(ValueError, "advertised"):
            self.module.get_pool_token_budget(max_model_len=131072, max_num_seqs=8, pool_ids_override="12000")
        with self.assertRaisesRegex(ValueError, "supports batches"):
            self.module.get_pool_token_budget(max_model_len=262144, max_num_seqs=4, pool_ids_override="12000")

    def test_conflicting_environment_overrides_rejected_and_legacy_errors_preserved(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.module.get_pool_token_budget(
                max_model_len=262144, max_num_seqs=8, b8_pool_ids="20480", pool_ids_override="20480"
            )
        with self.assertRaisesRegex(ValueError, "supports only 20480"):
            self.module.get_pool_token_budget(max_model_len=262144, max_num_seqs=8, b8_pool_ids="18000")
        with self.assertRaisesRegex(ValueError, "advertised"):
            self.module.get_pool_token_budget(max_model_len=1048576, max_num_seqs=8, b8_pool_ids="20480")

    def test_override_matches_active_plugin_heuristic_and_respects_admission_floor(self):
        # Confirm we inspect the package source selected by the environment's editable install.
        self.assertEqual(self.plugin_pth.read_text().strip(), str(ROOT / PLUGIN_ROOT))
        tree = ast.parse(self.plugin_worker_source)
        worker = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "get_num_available_blocks_tt"
        )
        statements = {
            ast.unparse(node) for node in ast.walk(worker) if isinstance(node, (ast.Assign, ast.AugAssign, ast.If))
        }
        self.assertIn("per_user_output_reservation = max(cache_config.block_size, output_tokens_per_step)", statements)
        self.assertIn("max_tokens_all_users += per_user_output_reservation * max_batch", statements)
        self.assertIn("max_tokens_all_users += sliding_window * max_batch * _MAX_SLIDING_GROUPS_HEURISTIC", statements)
        self.assertIn("num_tt_blocks = math.ceil(max_tokens_all_users / cache_config.block_size)", statements)
        self.assertTrue(any(item.startswith("if admission_floor > num_tt_blocks:") for item in statements))
        self.assertIn("num_tt_blocks = admission_floor", statements)
        self.assertIn("_HYBRID_KV_CACHE_GROUPS_ENABLED = True", self.generator_source)
        self.assertIn("block_size=32", self.generator_source)

        expected_min = {1: 8449, 8: 10248, 16: 12304, 32: 16416}
        known_single_request_floor = 4902  # vllm_integration/pool_plan.json, max context + 4096 in-flight tokens
        for batch, minimum in expected_min.items():
            with self.subTest(batch=batch):
                ids = minimum
                synthetic = self.module.get_pool_token_budget(
                    max_model_len=262144, max_num_seqs=batch, pool_ids_override=str(ids)
                )
                # AR output width is 1; worker reserves max(32-token block, output width).
                output_per_user = 32
                sliding_window = 1024
                heuristic_tokens = synthetic + (output_per_user + 8 * sliding_window) * batch
                heuristic_blocks = (heuristic_tokens + 31) // 32
                self.assertEqual(heuristic_blocks, ids)
                # Current measured/planned admission floor is lower. A larger
                # floor wins, so the requested count is not an unconditional cap.
                admission_floor = known_single_request_floor + batch
                self.assertGreater(ids, admission_floor)
                self.assertEqual(max(heuristic_blocks, admission_floor), ids)
                raised_floor = ids + 1
                self.assertEqual(max(heuristic_blocks, raised_floor), raised_floor)

    def test_existing_allocator_fit_guard_remains_untouched_in_generator(self):
        source = self.generator_source
        self.assertIn("memory_plan = validate_pool_memory(", source)
        self.assertIn('if not memory_plan["fits"]:', source)
        self.assertIn(
            'raise RuntimeError("Fixed serving KV pool exceeds launch free memory after recorded reserves")', source
        )


if __name__ == "__main__":
    unittest.main()
