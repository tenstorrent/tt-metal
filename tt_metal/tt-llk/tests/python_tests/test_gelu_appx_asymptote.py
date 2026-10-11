# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    FastMode,
    MathOperation,
)
from helpers.stimuli_config import StimuliConfig
from test_sfpu_unary import eltwise_unary_sfpu


@pytest.mark.parametrize("fresh_cpp_impl", [0, 4], ids=["production", "licensed_cpp"])
def test_gelu_appx_licensed_bf16_stream(fresh_cpp_impl, monkeypatch):
    """Expose the timed BF16 path to the exhaustive raw-pattern streamer."""
    retain = os.environ.get("SFPU_RETAIN_STREAM")
    if retain:
        retained = Path(retain)
        retained.unlink(missing_ok=True)
        collect = StimuliConfig.collect_raw_result_bytes

        def collect_and_retain(self, location):
            raw = collect(self, location)
            with retained.open("ab") as output:
                output.write(raw)
            return raw

        monkeypatch.setattr(
            StimuliConfig, "collect_raw_result_bytes", collect_and_retain
        )

    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        DestAccumulation.No,
        ApproximationMode.No,
        MathOperation.GeluAppx,
        FastMode.No,
        [64, 64],
        custom_atol=0.13,
        custom_rtol=0.05,
        fresh_cpp_impl=fresh_cpp_impl,
    )


@pytest.mark.parametrize(
    "input_value, exact_expected",
    [
        (258.0, 258.0),
        (4.03125, 4.03125),
        (-3.0, None),
        (-5.0, None),
    ],
)
def test_gelu_appx_licensed_asymptote(
    input_value, exact_expected, tmp_path, monkeypatch
):
    """Gate the exact Float16_b tail values repaired by the zero intercept."""
    source = tmp_path / "source.pt"
    result = tmp_path / "result.pt"
    # Include ordinary non-tail values so the normal golden comparison remains
    # meaningful in addition to the exact witness checks below.
    inputs = torch.tensor(
        [input_value, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
        dtype=torch.bfloat16,
    )
    torch.save(inputs, source)
    monkeypatch.setenv("SFPU_SRC_OVERRIDE", str(source))
    monkeypatch.setenv("SFPU_DUMP", str(result))

    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        DestAccumulation.No,
        ApproximationMode.No,
        MathOperation.GeluAppx,
        FastMode.No,
        [64, 64],
        custom_atol=0.13,
        custom_rtol=0.05,
        fresh_cpp_impl=4,
    )

    dumped = torch.load(result, weights_only=True)
    actual = dumped["result"][: inputs.numel()]
    if exact_expected is not None:
        assert actual[0].item() == exact_expected
    else:
        golden = torch.nn.functional.gelu(
            torch.tensor(input_value, dtype=torch.float64)
        )
        # The existing license requires no worse maximum absolute error than
        # the unchanged hand kernel; it does not require an exact-zero tail.
        # This is the measured all-finite hand bar in stored-output versus
        # exact-double-GELU units (the rounded-golden bar is a different metric).
        assert abs(actual[0].to(torch.float64) - golden) <= 0.023886588700354136
