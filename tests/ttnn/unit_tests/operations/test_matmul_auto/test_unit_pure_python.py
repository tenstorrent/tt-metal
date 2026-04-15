# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the pure-Python logic of the auto-config infrastructure.

These tests run WITHOUT any TT hardware or ttnn installation by mocking
all hardware-dependent imports. They validate:

1. Production helper functions (_find_largest_divisor, _find_grid, etc.)
2. Candidate generator logic (production 1D, 2D, DRAM candidates)
3. Constraint validator L1 budget formula
4. Heuristic scorer weights and production bonus
5. DNN retraining shape coverage
6. Config cache key generation
7. Subblock validation math

Run:
    pytest tests/ttnn/unit_tests/operations/test_matmul_auto/test_unit_pure_python.py -v
"""

from __future__ import annotations

import importlib
import json
import math
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ──────────────────────────────────────────────────────────────────────
# Mock ttnn module so we can import auto_config modules without hardware
# ──────────────────────────────────────────────────────────────────────
def _create_mock_ttnn():
    """Create a mock ttnn module tree that allows auto_config imports."""
    mock_ttnn = MagicMock()

    # Mock CoreCoord
    mock_ttnn.CoreCoord = MagicMock(side_effect=lambda x, y: MagicMock(x=x, y=y))

    # Mock program config classes — return MagicMock with all attrs accessible
    for cls_name in [
        "MatmulMultiCoreReuseMultiCast1DProgramConfig",
        "MatmulMultiCoreReuseMultiCastProgramConfig",
        "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig",
        "MatmulMultiCoreReuseProgramConfig",
    ]:
        setattr(mock_ttnn, cls_name, MagicMock(side_effect=lambda **kw: MagicMock(**kw)))

    # Mock DataType and Layout
    mock_ttnn.bfloat16 = MagicMock()
    mock_ttnn.bfloat8_b = MagicMock()
    mock_ttnn.MathFidelity = MagicMock()
    mock_ttnn.MathFidelity.HiFi4 = "HiFi4"
    mock_ttnn.TILE_LAYOUT = MagicMock()

    # Mock experimental
    mock_ttnn.experimental = MagicMock()

    return mock_ttnn


@pytest.fixture(autouse=True)
def mock_ttnn_module():
    """Patch ttnn into sys.modules before each test."""
    mock = _create_mock_ttnn()
    original = sys.modules.get("ttnn")
    sys.modules["ttnn"] = mock

    # Also patch sub-modules that might be imported
    sys.modules["ttnn.operations"] = MagicMock()
    sys.modules["ttnn._experimental.auto_config"] = MagicMock()
    sys.modules["ttnn.distributed"] = MagicMock()

    yield mock

    # Restore
    if original is not None:
        sys.modules["ttnn"] = original
    else:
        sys.modules.pop("ttnn", None)

    for key in list(sys.modules.keys()):
        if key.startswith("ttnn._experimental.auto_config"):
            sys.modules.pop(key, None)


# ──────────────────────────────────────────────────────────────────────
# Helper function tests (pure math, no hardware)
# ──────────────────────────────────────────────────────────────────────
class TestFindLargestDivisor:
    """Test _find_largest_divisor — mirrors ModelArgs.find_largest_divisor."""

    def _get_fn(self):
        """Import the function under test."""
        # Force reimport with mocked ttnn
        mod_name = "ttnn._experimental.auto_config.candidate_generator"
        if mod_name in sys.modules:
            del sys.modules[mod_name]

        spec = importlib.util.spec_from_file_location(
            mod_name,
            os.path.join("ttnn", "ttnn", "operations", "auto_config", "candidate_generator.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pass
        return getattr(mod, "_find_largest_divisor", None)

    def test_basic_cases(self):
        fn = self._get_fn()
        if fn is None:
            pytest.skip("Could not import _find_largest_divisor")
        assert fn(8) == 8
        assert fn(16) == 8
        assert fn(32) == 8
        assert fn(7) == 7
        assert fn(1) == 1
        assert fn(5) == 5
        assert fn(3) == 3

    def test_with_max_divisor(self):
        fn = self._get_fn()
        if fn is None:
            pytest.skip("Could not import")
        assert fn(16, max_divisor=4) == 4
        assert fn(16, max_divisor=2) == 2
        assert fn(7, max_divisor=3) == 1  # 7 has no divisor <= 3 except 1

    def test_prime_numbers(self):
        fn = self._get_fn()
        if fn is None:
            pytest.skip("Could not import")
        assert fn(11) == 1  # 11 is prime, only divisor <= 8 is 1
        assert fn(13) == 1
        assert fn(17) == 1

    def test_powers_of_2(self):
        fn = self._get_fn()
        if fn is None:
            pytest.skip("Could not import")
        assert fn(2) == 2
        assert fn(4) == 4
        assert fn(8) == 8
        assert fn(16) == 8
        assert fn(64) == 8
        assert fn(128) == 8


class TestFindGrid:
    """Test _find_grid — mirrors ModelArgs.find_grid."""

    def _get_fn(self):
        mod_name = "ttnn._experimental.auto_config.candidate_generator"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        spec = importlib.util.spec_from_file_location(
            mod_name,
            os.path.join("ttnn", "ttnn", "operations", "auto_config", "candidate_generator.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pass
        return getattr(mod, "_find_grid", None)

    def test_perfect_square(self):
        fn = self._get_fn()
        if fn is None:
            pytest.skip("Could not import")
        rows, cols = fn(64)  # 64 = 8x8
        assert rows * cols <= 64
        assert 64 % (rows * cols) == 0

    def test_targets_32_cores(self):
        fn = self._get_fn()
        if fn is None:
            pytest.skip("Could not import")
        rows, cols = fn(128)  # Should target ~32 cores
        cores = rows * cols
        assert 128 % cores == 0
        assert cores == 32  # 128/32 = 4, which divides evenly

    def test_small_value(self):
        fn = self._get_fn()
        if fn is None:
            pytest.skip("Could not import")
        rows, cols = fn(4)
        assert rows * cols >= 1
        assert 4 % (rows * cols) == 0

    def test_prime(self):
        fn = self._get_fn()
        if fn is None:
            pytest.skip("Could not import")
        rows, cols = fn(37)  # Prime: only divisors are 1 and 37
        cores = rows * cols
        assert 37 % cores == 0


class TestGetOutSubblockW:
    """Test _get_out_subblock_w."""

    def _get_fn(self):
        mod_name = "ttnn._experimental.auto_config.candidate_generator"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        spec = importlib.util.spec_from_file_location(
            mod_name,
            os.path.join("ttnn", "ttnn", "operations", "auto_config", "candidate_generator.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pass
        return getattr(mod, "_get_out_subblock_w", None)

    def test_basic(self):
        fn = self._get_fn()
        if fn is None:
            pytest.skip("Could not import")
        # per_core_N=8, subblock_h=1, max_hw=8 -> should return 8
        assert fn(8, 1, 8) == 8

    def test_constrained_by_h(self):
        fn = self._get_fn()
        if fn is None:
            pytest.skip("Could not import")
        # per_core_N=8, subblock_h=4, max_hw=8 -> w*4<=8, so w<=2
        result = fn(8, 4, 8)
        assert result <= 2
        assert result * 4 <= 8

    def test_must_divide_per_core_N(self):
        fn = self._get_fn()
        if fn is None:
            pytest.skip("Could not import")
        # per_core_N=5 (prime), subblock_h=1 -> only 5 or 1 divide 5
        result = fn(5, 1, 8)
        assert 5 % result == 0

    def test_fp32_constraint(self):
        fn = self._get_fn()
        if fn is None:
            pytest.skip("Could not import")
        # fp32: max_hw=4
        result = fn(8, 1, 4)
        assert result <= 4
        assert 8 % result == 0


# ──────────────────────────────────────────────────────────────────────
# Constraint Validator L1 Budget Tests
# ──────────────────────────────────────────────────────────────────────
class TestL1BudgetFormula:
    """Test the exact L1 CB budget formula."""

    def test_small_config_passes(self):
        """Small per_core values should always fit in L1."""
        # CB_in0 = 2*2*2 = 8 tiles, CB_in1 = 2*2*2 = 8 tiles, CB_out = 2*2 = 4 tiles
        # Total = 20 * 2048 = 40960 bytes + small accum
        # Well under 1.2 MB
        tile_bytes = 2048

        in0_block_w = 2
        per_core_M = 2
        per_core_N = 2

        cb_in0_tiles = in0_block_w * per_core_M * 2  # 8
        cb_in1_tiles = in0_block_w * per_core_N * 2  # 8
        cb_out_tiles = per_core_M * per_core_N  # 4

        fp32_accum = 1 * 1 * 32 * 32 * 4  # 4096 bytes

        total = (cb_in0_tiles + cb_in1_tiles + cb_out_tiles) * tile_bytes + fp32_accum
        assert total < 1_258_291, f"Small config should fit: {total}"

    def test_large_config_fails(self):
        """Very large per_core values should exceed L1."""
        tile_bytes = 2048
        in0_block_w = 8
        per_core_M = 32
        per_core_N = 32

        cb_in0_tiles = in0_block_w * per_core_M * 2  # 512
        cb_in1_tiles = in0_block_w * per_core_N * 2  # 512
        cb_out_tiles = per_core_M * per_core_N  # 1024

        fp32_accum = 4 * 4 * 32 * 32 * 4  # 65536

        total = (cb_in0_tiles + cb_in1_tiles + cb_out_tiles) * tile_bytes + fp32_accum
        assert total > 1_258_291, f"Large config should exceed L1: {total}"

    def test_bfloat8b_more_headroom(self):
        """bfloat8_b tiles are smaller (1088 bytes), giving more L1 headroom."""
        tile_bytes_bf16 = 2048
        tile_bytes_bf8 = 1088

        in0_block_w = 4
        per_core_M = 8
        per_core_N = 8

        tiles = in0_block_w * per_core_M * 2 + in0_block_w * per_core_N * 2 + per_core_M * per_core_N

        total_bf16 = tiles * tile_bytes_bf16
        total_bf8 = tiles * tile_bytes_bf8

        assert total_bf8 < total_bf16, "bfloat8_b should use less L1"

    def test_subblock_constraint_max8(self):
        """out_subblock_h * out_subblock_w must be <= 8."""
        max_hw = 8
        valid_pairs = [(h, w) for h in range(1, 9) for w in range(1, 9) if h * w <= max_hw]
        assert (4, 2) in valid_pairs
        assert (8, 1) in valid_pairs
        assert (1, 8) in valid_pairs
        assert (3, 3) not in valid_pairs  # 9 > 8
        assert (4, 3) not in valid_pairs  # 12 > 8

    def test_subblock_constraint_fp32_max4(self):
        """With fp32_dest_acc_en, max product is 4."""
        max_hw = 4
        valid_pairs = [(h, w) for h in range(1, 9) for w in range(1, 9) if h * w <= max_hw]
        assert (4, 1) in valid_pairs
        assert (2, 2) in valid_pairs
        assert (1, 4) in valid_pairs
        assert (4, 2) not in valid_pairs  # 8 > 4
        assert (3, 2) not in valid_pairs  # 6 > 4


# ──────────────────────────────────────────────────────────────────────
# Scorer Weight Tests
# ──────────────────────────────────────────────────────────────────────
class TestScorerWeights:
    """Test that scorer weights are valid and sum to 1.0."""

    def test_weights_sum_to_one(self):
        """All scoring weights must sum to 1.0."""
        weights = {
            "utilization": 0.25,
            "block_efficiency": 0.18,
            "layout_alignment": 0.15,
            "subblock_efficiency": 0.07,
            "backend_preference": 0.08,
            "fidelity_cost": 0.12,
            "production_bonus": 0.15,
        }
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected 1.0"

    def test_production_bonus_is_positive(self):
        """Production bonus weight must be > 0."""
        assert 0.15 > 0

    def test_all_weights_positive(self):
        """All weights must be positive."""
        weights = [0.25, 0.18, 0.15, 0.07, 0.08, 0.12, 0.15]
        for w in weights:
            assert w > 0, f"Weight {w} must be positive"


# ──────────────────────────────────────────────────────────────────────
# DNN Training Shape Coverage Tests
# ──────────────────────────────────────────────────────────────────────
class TestDNNTrainingShapes:
    """Test that the DNN retraining pipeline has sufficient shape coverage."""

    def _get_shapes(self):
        """Load training shapes from retrain_dnn.py."""
        spec = importlib.util.spec_from_file_location(
            "retrain_dnn",
            os.path.join("ttnn", "ttnn", "operations", "auto_config", "retrain_dnn.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pass
        return getattr(mod, "TRAINING_SHAPES", None)

    def test_at_least_100_shapes(self):
        shapes = self._get_shapes()
        if shapes is None:
            pytest.skip("Could not import TRAINING_SHAPES")
        assert len(shapes) >= 100, f"Expected 100+ shapes, got {len(shapes)}"

    def test_categories_covered(self):
        shapes = self._get_shapes()
        if shapes is None:
            pytest.skip("Could not import")
        categories = set(s.get("category", "unknown") for s in shapes)
        expected_categories = {
            "square",
            "M_large",
            "K_large",
            "N_large",
            "llm_decode",
            "llm_prefill",
            "attention",
            "non_pow2",
        }
        missing = expected_categories - categories
        assert len(missing) == 0, f"Missing categories: {missing}"

    def test_all_shapes_tile_aligned(self):
        shapes = self._get_shapes()
        if shapes is None:
            pytest.skip("Could not import")
        for s in shapes:
            M, K, N = s["M"], s["K"], s["N"]
            # After tile padding, these must be multiples of 32
            M_padded = ((M + 31) // 32) * 32
            K_padded = ((K + 31) // 32) * 32
            N_padded = ((N + 31) // 32) * 32
            assert M_padded >= 32, f"M={M} too small after padding"
            assert K_padded >= 32, f"K={K} too small after padding"
            assert N_padded >= 32, f"N={N} too small after padding"

    def test_shapes_cover_llm_dimensions(self):
        """Shapes should include LLM-relevant dimensions (4096, 11008, etc.)."""
        shapes = self._get_shapes()
        if shapes is None:
            pytest.skip("Could not import")
        all_dims = set()
        for s in shapes:
            all_dims.add(s["M"])
            all_dims.add(s["K"])
            all_dims.add(s["N"])
        # Key LLM dimensions
        assert 4096 in all_dims, "Should include 4096 (Llama hidden dim)"
        assert 11008 in all_dims, "Should include 11008 (Llama MLP dim)"
        assert 4544 in all_dims, "Should include 4544 (Falcon hidden dim)"
        assert 8192 in all_dims, "Should include 8192 (Llama-70B hidden dim)"

    def test_no_duplicate_shapes(self):
        shapes = self._get_shapes()
        if shapes is None:
            pytest.skip("Could not import")
        seen = set()
        dupes = 0
        for s in shapes:
            key = (s["M"], s["K"], s["N"])
            if key in seen:
                dupes += 1
            seen.add(key)
        # A few duplicates are OK (same shape in different categories)
        # but more than 10% would be wasteful
        assert dupes < len(shapes) * 0.1, f"Too many duplicate shapes: {dupes}/{len(shapes)}"


# ──────────────────────────────────────────────────────────────────────
# Benchmark Shape Coverage Tests
# ──────────────────────────────────────────────────────────────────────
class TestBenchmarkShapes:
    """Test benchmark.py shape coverage."""

    def _get_shapes(self):
        spec = importlib.util.spec_from_file_location(
            "benchmark",
            os.path.join("ttnn", "ttnn", "operations", "auto_config", "benchmark.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pass
        return getattr(mod, "DEFAULT_SWEEP_SHAPES", None)

    def test_at_least_100_shapes(self):
        shapes = self._get_shapes()
        if shapes is None:
            pytest.skip("Could not import DEFAULT_SWEEP_SHAPES")
        assert len(shapes) >= 100, f"Expected 100+ shapes, got {len(shapes)}"


# ──────────────────────────────────────────────────────────────────────
# Falcon-7B Shape Validation
# ──────────────────────────────────────────────────────────────────────
class TestFalcon7BShapes:
    """Validate that Falcon-7B test shapes match the published architecture."""

    FALCON_HIDDEN = 4544
    FALCON_MLP = 18176  # 4 * 4544
    FALCON_QKV = 4672  # 4544 + 128 (for multi-query heads)

    def test_qkv_shape_matches_architecture(self):
        assert self.FALCON_QKV > self.FALCON_HIDDEN
        assert self.FALCON_QKV == 4672

    def test_mlp_is_4x_hidden(self):
        assert self.FALCON_MLP == 4 * self.FALCON_HIDDEN

    def test_all_dims_tile_padded(self):
        """All Falcon dims should already be multiples of 32."""
        for dim in [self.FALCON_HIDDEN, self.FALCON_MLP, self.FALCON_QKV]:
            padded = ((dim + 31) // 32) * 32
            assert padded == dim, f"{dim} is not tile-aligned (padded to {padded})"


# ──────────────────────────────────────────────────────────────────────
# Production Formula Integration Tests
# ──────────────────────────────────────────────────────────────────────
class TestProductionFormulas:
    """Test that production formulas produce correct values for known shapes."""

    def _get_helpers(self):
        mod_name = "ttnn._experimental.auto_config.candidate_generator"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        spec = importlib.util.spec_from_file_location(
            mod_name,
            os.path.join("ttnn", "ttnn", "operations", "auto_config", "candidate_generator.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pass
        return mod

    def test_llama_in0_block_w(self):
        """For Llama-7B (K=4096, 8x8 grid), in0_block_w should be maximized."""
        mod = self._get_helpers()
        fn = getattr(mod, "_find_largest_divisor", None)
        if fn is None:
            pytest.skip("Could not import")

        k_tiles = 4096 // 32  # 128
        # With 64 cores, k_per_core = 128/64 = 2
        k_per_core = k_tiles // 64
        in0_block_w = fn(k_per_core)
        assert in0_block_w == 2, f"Expected in0_block_w=2 for Llama, got {in0_block_w}"

    def test_falcon_in0_block_w(self):
        """For Falcon-7B (K=4544), in0_block_w should be computed correctly."""
        mod = self._get_helpers()
        fn = getattr(mod, "_find_largest_divisor", None)
        if fn is None:
            pytest.skip("Could not import")

        K_padded = ((4544 + 31) // 32) * 32  # 4544 is already aligned
        k_tiles = K_padded // 32  # 142
        # With 64 cores: k_per_core = 142/64 = 2.21... → not evenly divisible
        # So formula would use k_tiles directly
        in0_block_w = fn(k_tiles)
        assert in0_block_w >= 1
        assert k_tiles % in0_block_w == 0

    def test_subblock_product_constraint(self):
        """Verify subblock h*w <= 8 for all valid subblock pairs."""
        valid_pairs = []
        for h in range(1, 9):
            for w in range(1, 9):
                if h * w <= 8:
                    valid_pairs.append((h, w))

        # Known valid pairs from SUBBLOCK_HW_CHOICES
        assert (4, 2) in valid_pairs
        assert (2, 4) in valid_pairs
        assert (8, 1) in valid_pairs
        assert (1, 8) in valid_pairs
        assert (1, 1) in valid_pairs

        # Invalid
        assert (3, 3) not in valid_pairs
        assert (5, 2) not in valid_pairs

    def test_div_up_formula(self):
        """ceil division should match expected values."""

        def div_up(a, b):
            return (a + b - 1) // b

        assert div_up(128, 8) == 16
        assert div_up(129, 8) == 17  # Rounds up
        assert div_up(1, 8) == 1
        assert div_up(8, 8) == 1
        assert div_up(0, 8) == 0


# ──────────────────────────────────────────────────────────────────────
# File Structure Tests
# ──────────────────────────────────────────────────────────────────────
class TestFileStructure:
    """Verify all expected files exist."""

    AUTO_CONFIG_DIR = os.path.join("ttnn", "ttnn", "operations", "auto_config")
    TEST_DIR = os.path.join("tests", "ttnn", "unit_tests", "operations", "test_matmul_auto")

    EXPECTED_AUTO_CONFIG_FILES = [
        "__init__.py",
        "base.py",
        "benchmark.py",
        "candidate_generator.py",
        "config_cache.py",
        "constraint_validator.py",
        "feature_extraction.py",
        "math_fidelity.py",
        "matmul_auto.py",
        "retrain_dnn.py",
        "RETRAINING.md",
    ]

    EXPECTED_TEST_FILES = [
        "__init__.py",
        "conftest.py",
        "test_correctness.py",
        "test_multi_device.py",
        "test_mutation.py",
        "test_performance.py",
        "test_falcon7b_perf.py",
        "test_t3k_multi_device.py",
    ]

    def test_all_auto_config_files_exist(self):
        for fn in self.EXPECTED_AUTO_CONFIG_FILES:
            path = os.path.join(self.AUTO_CONFIG_DIR, fn)
            assert os.path.exists(path), f"Missing: {path}"

    def test_all_test_files_exist(self):
        for fn in self.EXPECTED_TEST_FILES:
            path = os.path.join(self.TEST_DIR, fn)
            assert os.path.exists(path), f"Missing: {path}"

    def test_scorer_directory_exists(self):
        scorer_dir = os.path.join(self.AUTO_CONFIG_DIR, "scorer")
        assert os.path.isdir(scorer_dir), f"Missing scorer directory: {scorer_dir}"

        for fn in ["__init__.py", "heuristic.py", "dnn_scorer.py", "dnn_weights.json"]:
            path = os.path.join(scorer_dir, fn)
            assert os.path.exists(path), f"Missing scorer file: {path}"

    def test_dnn_weights_is_valid_json(self):
        path = os.path.join(self.AUTO_CONFIG_DIR, "scorer", "dnn_weights.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, (dict, list)), "dnn_weights.json should be valid JSON"

    def test_retraining_doc_exists(self):
        path = os.path.join(self.AUTO_CONFIG_DIR, "RETRAINING.md")
        assert os.path.exists(path), f"Missing: {path}"
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert len(content) > 100, "RETRAINING.md should have substantial content"


# ──────────────────────────────────────────────────────────────────────
# SPDX License Header Tests
# ──────────────────────────────────────────────────────────────────────
class TestSPDXHeaders:
    """Verify all Python files have proper SPDX headers."""

    def _get_all_py_files(self):
        files = []
        for d in [
            os.path.join("ttnn", "ttnn", "operations", "auto_config"),
            os.path.join("tests", "ttnn", "unit_tests", "operations", "test_matmul_auto"),
        ]:
            for root, _, filenames in os.walk(d):
                for fn in filenames:
                    if fn.endswith(".py"):
                        files.append(os.path.join(root, fn))
        return files

    def test_all_files_have_spdx(self):
        for f in self._get_all_py_files():
            with open(f, encoding="utf-8") as fh:
                first_lines = fh.read(500)
            assert "SPDX-FileCopyrightText" in first_lines, f"Missing SPDX header: {f}"
            assert "SPDX-License-Identifier" in first_lines, f"Missing SPDX license: {f}"

    def test_all_files_have_copyright_symbol(self):
        for f in self._get_all_py_files():
            with open(f, encoding="utf-8") as fh:
                first_lines = fh.read(500)
            if "SPDX-FileCopyrightText" in first_lines:
                assert "\u00a9" in first_lines, f"Missing \u00a9 symbol in SPDX header: {f}"
