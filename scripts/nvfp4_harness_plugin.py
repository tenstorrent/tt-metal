# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""NVFP4 harness plugin: the shared candidate/timing plugin plus input bit permutation."""

import json
import os

import pytest

from scripts.quant_harness_plugin import (  # noqa: F401  (pytest discovers hooks/fixtures by name)
    pytest_collection_finish,
    pytest_runtest_logreport,
    pytest_sessionfinish,
    pytest_sessionstart,
    quant_candidate_kernel_directory,
)


def permute_packed_bits(packed, permutation):
    result = packed.clone().zero_()
    for output_bit, source_bit in enumerate(permutation):
        result |= ((packed >> source_bit) & 1) << output_bit
    return result


@pytest.fixture(autouse=True)
def nvfp4_input_bit_permutation(monkeypatch):
    permutation = json.loads(os.environ.get("NVFP4_INPUT_BIT_PERMUTATION", "null"))
    if permutation is not None:
        import ttnn

        original_pack = ttnn.experimental.nvfp4.pack

        def pack_with_encoding(*args, **kwargs):
            packed, scales = original_pack(*args, **kwargs)
            if kwargs.get("experimental_fast", False):
                packed = permute_packed_bits(packed, permutation)
            return packed, scales

        monkeypatch.setattr(ttnn.experimental.nvfp4, "pack", pack_with_encoding)
