# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0
#
# CPU-only package checks: the package imports without ttnn, configuration
# and device-option validation, precision policies, and the demo CSV
# reader/writer. No checkpoint or device needed.

import importlib
import sys

import numpy as np

from models.experimental.chronos2 import tt
from models.experimental.chronos2.demo import demo
from models.experimental.chronos2.tt.model_config import PRECISION_DTYPES

POLICY_KEYS = {"mode", "weights", "activations", "accumulation", "exceptions"}


def test_precision_policies_are_complete():
    assert tt.DEFAULT_PRECISION == "fp32"
    assert set(tt.SUPPORTED_PRECISIONS) == {"fp32", "bf16", "bfp8_b"}
    for p in tt.SUPPORTED_PRECISIONS:
        pol = tt.PRECISION_POLICIES[p]
        assert set(pol) == POLICY_KEYS
        assert pol["activations"] in tt.SUPPORTED_PRECISIONS
        assert p in PRECISION_DTYPES


def test_precision_dtypes():
    # bf16 / fp32 upload every tensor in one dtype; bfp8_b changes only linear weights
    assert PRECISION_DTYPES["bf16"] == ("bfloat16", "bfloat16")
    assert PRECISION_DTYPES["fp32"] == ("float32", "float32")
    assert PRECISION_DTYPES["bfp8_b"] == ("bfloat16", "bfloat8_b")
    pol = tt.PRECISION_POLICIES["bfp8_b"]
    assert (pol["weights"], pol["activations"], pol["accumulation"]) == ("bfp8_b", "bf16", "fp32")


def test_bf16_policy_declares_fp32_parts():
    # the bf16 graph keeps an fp32 residual stream and head under HiFi4 matmuls
    pol = tt.PRECISION_POLICIES["bf16"]
    assert (pol["mode"], pol["weights"], pol["activations"], pol["accumulation"]) == ("bf16", "bf16", "bf16", "fp32")
    text = " ".join(pol["exceptions"])
    assert "HiFi4" in text
    assert "fp32 residual stream" in text
    assert "quantile head in FP32" in text


def test_import_does_not_touch_ttnn():
    had = "ttnn" in sys.modules
    importlib.reload(tt)
    assert ("ttnn" in sys.modules) == had


def test_device_option_bounds(expect_error):
    assert tt.validate_device_options({"trace_region_size": 0}) == {"trace_region_size": 0}
    with expect_error(ValueError, "outside"):
        tt.validate_device_options({"trace_region_size": -1})
    with expect_error(ValueError, "unknown device option"):
        tt.validate_device_options({"bogus": 1})


def test_unsupported_precision_fails_loudly(expect_error):
    cfg = _tiny_cfg()
    with expect_error(ValueError, "not supported"):
        tt.Backend("/nonexistent", cfg, None, "fp16", None)


def test_resolve_config_accepts_dict_and_instance():
    cfg = _tiny_cfg()
    assert tt.resolve_config(cfg, "/nonexistent") is cfg


def test_demo_csv_roundtrip(tmp_path):
    src = tmp_path / "in.csv"
    src.write_text("ts,a,b\n0,1.0,\n1,2.0,5\n2,3.5,6\n")
    ids, v, m = demo.read_wide_csv(str(src), "ts")
    assert ids == ["a", "b"]
    np.testing.assert_array_equal(m, [[1, 1, 1], [0, 1, 1]])
    np.testing.assert_array_equal(v, np.array([[1, 2, 3.5], [0, 5, 6]], np.float32))
    q = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
    out = tmp_path / "out.csv"
    demo.write_quantile_csv(str(out), ids, (0.1, 0.9), q)
    lines = out.read_text().strip().splitlines()
    assert lines[0] == "series_id,step,q0.1,q0.9"
    assert len(lines) == 1 + 2 * 3
    assert lines[-1] == "b,3,10,11"


def _tiny_cfg():
    return tt.Chronos2Config.from_dict(
        {
            "d_model": 8,
            "d_ff": 16,
            "d_kv": 4,
            "num_heads": 2,
            "num_layers": 1,
            "layer_norm_epsilon": 1e-6,
            "dense_act_fn": "relu",
            "chronos_config": {
                "context_length": 64,
                "input_patch_size": 4,
                "input_patch_stride": 4,
                "output_patch_size": 4,
                "max_output_patches": 4,
                "quantiles": [0.1, 0.5, 0.9],
            },
        }
    )
