# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-point, non-perturbing-before-capture BH Welford differential."""

from dataclasses import dataclass

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, PerfRunType
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import TemplateParameter
from helpers.tilize_untilize import tilize_block, untilize_block


@dataclass
class TraceImplTemplate(TemplateParameter):
    trace_impl: int

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t TRACE_IMPL = {self.trace_impl}u;"


@dataclass
class TraceNTemplate(TemplateParameter):
    trace_n: int

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t TRACE_N = {self.trace_n}u;"


def _decode(tiles, tile):
    rows = tiles[tile * 32 : tile * 32 + 4, :16:2]
    return rows.reshape(2, 2, 8).transpose(1, 2).flatten().float()


def _run(impl, n):
    torch.manual_seed(20260814)
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    input_2d = torch.empty((32, 32), dtype=torch.bfloat16).uniform_(-4.0, 4.0)
    source = tilize_block(
        input_2d.flatten(), [32, 32], stimuli_format=formats.input_format
    ).flatten()
    config = TestConfig(
        "sources/sfpu_welford_prefix_snapshot.cpp",
        formats,
        templates=[TraceImplTemplate(impl), TraceNTemplate(n)],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            source,
            formats.input_format,
            torch.zeros_like(source),
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=9,
        ),
        dest_acc=DestAccumulation.No,
        unpack_to_dest=False,
        disable_format_inference=True,
        compile_time_formats=True,
    )
    # laneMO stratified-sampling hook (corpus/tools/lanemo_sample_sweep.py),
    # env-gated and inert otherwise. sem vs hand = TRACE_IMPL 2 (vfloat_direct)
    # vs 0 (handwritten_direct). The 9-tile register-snapshot Res is the output
    # compared. Streams then skips (this helper's caller does register analysis
    # the sampling deliberately bypasses). See test_sfpu_blaze.py for the contract.
    import os as _os

    _lanemo_sample = _os.environ.get("LANEMO_SAMPLE")
    if _lanemo_sample:
        import sys as _sys
        from pathlib import Path as _Path

        _sys.path.insert(
            0, str(_Path(__file__).resolve().parents[1] / "corpus" / "tools")
        )
        import lanemo_sample_gen as _G

        _G.stream_on_device(
            config,
            TestConfig.TENSIX_LOCATION,
            _lanemo_sample,
            _os.environ.get("LANEMO_OP", "op"),
            int(_os.environ.get("LANEMK_WAIT_TIMEOUT", "60")),
        )
        pytest.skip("laneMO sample stream complete")

    tiles = untilize_block(
        torch.tensor(config.run().result, dtype=torch.bfloat16),
        formats.output_format,
        [288, 32],
    )
    return input_2d, torch.stack([_decode(tiles, index) for index in range(9)])


IMPLEMENTATIONS = [
    (0, "handwritten_direct"),
    (1, "handwritten_replay"),
    (2, "vfloat_direct"),
    (3, "vfloat_rescue"),
    (4, "vfloat_manual_early_fold"),
]


@pytest.mark.parametrize("impl,label", IMPLEMENTATIONS)
@pytest.mark.parametrize("n", [1, 2, 32])
def test_sfpu_welford_prefix_snapshot(impl, label, n):
    input_2d, observed = _run(impl, n)
    expected_mean = input_2d[:n].float().mean(dim=0)
    expected_m2 = ((input_2d[:n].float() - expected_mean) ** 2).sum(dim=0)
    names = [
        "LREG0",
        "LREG1",
        "LREG2",
        "LREG3",
        "LREG4",
        "LREG5",
        "LREG6",
        "LREG7",
        "LREG11",
    ]
    for index, name in enumerate(names):
        print(
            f"PREFIX_ABI impl={label} n={n} reg={name} value0={observed[index,0].item():.7g}"
        )
    mean_error = (observed[4] - expected_mean).abs().max().item()
    m2_error = (observed[5] - expected_m2).abs().max().item()
    print(
        f"PREFIX_CORRECTNESS impl={label} n={n} mean_max_abs={mean_error:.7g} m2_max_abs={m2_error:.7g}"
    )
    assert torch.allclose(observed[4], expected_mean, rtol=2e-2, atol=2e-2)
    assert torch.allclose(observed[5], expected_m2, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("impl,label", IMPLEMENTATIONS)
def test_sfpu_welford_device_profile(perf_report, impl, label):
    """One on-device math-zone sample; pytest elapsed time is deliberately ignored.

    This is the LLK equivalent of the fitter's device-profiler collection: the
    profiler build records the MATH TRISC zone around WELFORD_BODY and PerfConfig
    retrieves those device timestamps after the ELF has completed.  The caller
    launches a fresh pytest process per sample, so no host-side timing or cached
    profiler state is carried between samples.
    """
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    torch.manual_seed(20260814)
    input_2d = torch.empty((32, 32), dtype=torch.bfloat16).uniform_(-4.0, 4.0)
    source = tilize_block(
        input_2d.flatten(), [32, 32], stimuli_format=formats.input_format
    ).flatten()
    config = PerfConfig(
        "sources/sfpu_welford_prefix_snapshot.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        # Do not fold diagnostic SFPSTORE captures into the timed math zone.
        templates=[TraceImplTemplate(impl), TraceNTemplate(0)],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            source,
            formats.input_format,
            torch.zeros_like(source),
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=9,
        ),
        dest_acc=DestAccumulation.No,
        unpack_to_dest=False,
        disable_format_inference=True,
        compile_time_formats=True,
    )
    # Use the module fixture, not a private report, so conftest persists both
    # raw and post-process CSV rows in addition to the greppable console line.
    config.run(perf_report, run_count=1)
    frame = perf_report.frame()
    rows = frame[frame["marker"] == "WELFORD_BODY"]
    assert len(rows) == 1, frame.to_string(index=False)
    value = rows.iloc[0]["mean(MATH_ISOLATE)"]
    assert value > 0
    # Stable, greppable device-only output for the external fitter-style reducer.
    print(f"WELFORD_DEVICE_PROFILE impl={label} math_cycles={int(value)}")
