# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Every registered long-context case fits its golden, and every registered golden opens and reads through its reader,
before any device run. Reads the goldens from the registry (Weka). CPU only."""

from __future__ import annotations

from collections import Counter

import pytest

from models.demos.common.prefill.long_context_accuracy.fp64_golden import open_golden
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.cases import ALL_CASES
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3.golden import open_k3_layer0_golden
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3_1.golden import open_k31_chain


def test_every_case_fits_its_golden() -> None:
    """Unique ids; each case's golden opens through the case's reader, and holds every context length the case runs,
    each a positive row count with bars."""
    repeated = [case_id for case_id, n in Counter(case.id for case in ALL_CASES).items() if n > 1]
    assert not repeated, f"case ids used twice: {repeated}"
    for case in ALL_CASES:
        golden = open_golden(case.golden_id, reader=case.golden_reader)
        assert case.chunk > 0, case.id
        assert case.context_lengths, f"{case.id} runs no context length"
        too_long = [n for n in case.context_lengths if not 0 < n <= golden.rows]
        assert (
            not too_long
        ), f"{case.id}: context lengths {too_long} are not inside the {golden.rows} rows of {golden.golden_id}"


@pytest.mark.parametrize(
    "open_registered, row_products",
    [
        (open_k3_layer0_golden, ["output"]),
        (open_k31_chain, ["moe_input_1", "routing_ids_2", "dsa_input_3", "latent", "topk_indices_3"]),
    ],
    ids=["k3-l0-kda-fp64-stored-fp32-1m", "k31-l0-3-fp64-1m"],
)
def test_registered_goldens_read_through_their_readers(open_registered, row_products) -> None:
    golden = open_registered()
    segment = golden.manifest["segment"]
    rows = golden.read_rows(row_products, segment - 1, segment + 1)  # (across the first segment boundary)
    for name in row_products:
        shape = golden.manifest["products"][name]["shape"]
        assert tuple(rows[name].shape) == (2, *shape), name
