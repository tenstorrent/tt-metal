# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from models.common.utility_functions import is_blackhole
from .sdpa_recipe_test_utils import VARIANTS, digest, make_inputs, metrics, reference
from .test_sdpa_joint_recipes import joint, joined, options, upload


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    "q_lengths,k_lengths,distribution",
    [
        ((1, 1), (1, 1), "normal"),
        ((15, 17), (15, 17), "normal"),
        ((31, 33), (31, 33), "uniform"),
        ((255, 1), (255, 1), "normal"),
        ((257, 31), (257, 31), "normal"),
        ((511, 1), (511, 1), "constant_v"),
        ((512, 17), (512, 17), "changed_max"),
        ((767, 33), (767, 33), "zero_v"),
        ((17, 33), (513, 31), "normal"),
        ((513, 31), (17, 33), "normal"),
    ],
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 16777216}], indirect=True)
def test_recipe_subtile_tails(device, variant, q_lengths, k_lengths, distribution, record_property):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    device.enable_program_cache()
    host = make_inputs(sum(k_lengths), distribution, q_length=sum(q_lengths), heads=2)
    expected = reference(*host)
    segments = [
        [x[..., : lengths[0], :].contiguous() for x, lengths in zip(host, (q_lengths, k_lengths, k_lengths))],
        [x[..., lengths[0] :, :].contiguous() for x, lengths in zip(host, (q_lengths, k_lengths, k_lengths))],
    ]
    grid = (4, 1)
    for layout in ("dense", "joint"):

        def invoke(inputs):
            if layout == "joint":
                return joint(inputs, variant, grid)
            return [
                ttnn.transformer.scaled_dot_product_attention(*inputs[0], is_causal=False, **options(variant, grid))
            ]

        sources = segments if layout == "joint" else [host]
        clean = [upload(device, segment, variant, pad_value=0) for segment in sources]
        poisoned = [upload(device, segment, variant, pad_value=float("nan")) for segment in sources]
        before = [digest(ttnn.to_torch(x)) for segment in poisoned for x in segment]
        actual = joined(invoke(poisoned))
        assert digest(actual) == digest(joined(invoke(clean)))
        observed = metrics(actual, expected)
        for key, value in observed.items():
            record_property(f"{layout}_{key}", value)
        if distribution == "zero_v":
            assert observed["max_abs"] <= 1e-6
        else:
            limits = {"A": 8, "B": 8, "C": 2, "D": 0.4, "E_bf16": 8, "E_bfp8": 8, "E_bfp4": 35}
            assert observed["l2_pct"] < limits[variant]
        assert before == [digest(ttnn.to_torch(x)) for segment in poisoned for x in segment]
        trace = ttnn.begin_trace_capture(device, cq_id=0)
        traced = invoke(poisoned)
        ttnn.end_trace_capture(device, trace, cq_id=0)
        try:
            for _ in range(2):
                ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                assert digest(joined(traced)) == digest(actual)
        finally:
            ttnn.release_trace(device, trace)


@pytest.mark.parametrize("variant", VARIANTS)
def test_recipe_tail_cache_identity(device, variant):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    device.enable_program_cache()
    host = make_inputs(33, "normal", q_length=33)
    inputs = []
    outputs = []
    for split in (17, 18):
        inputs.append(
            [
                upload(device, [x[..., start:end, :].contiguous() for x in host], variant)
                for start, end in ((0, split), (split, 33))
            ]
        )
        outputs.append(joined(joint(inputs[-1], variant, (1, 1))))
        if len(inputs) == 1:
            entries = device.num_program_cache_entries()
        else:
            assert device.num_program_cache_entries() > entries
    entries = device.num_program_cache_entries()
    for source, expected in zip(inputs, outputs):
        assert digest(joined(joint(source, variant, (1, 1)))) == digest(expected)
        assert device.num_program_cache_entries() == entries
