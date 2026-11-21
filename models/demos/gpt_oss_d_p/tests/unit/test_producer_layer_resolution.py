# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-free check that the producer's GPT-OSS KV verify cannot pass on a partial device map."""

from types import SimpleNamespace

import pytest

from models.demos.common.prefill.runners import prefill_producer as producer
from models.demos.common.prefill.tests.test_prefill_producer_kv_decode import gqa_trace  # noqa: F401 (fixture)


# A single rank holds every GPT-OSS layer, so an unresolved layer there must fail; with several ranks it is
# another rank's layer and is skipped.
def test_producer_gpt_oss_single_rank_rejects_unresolved_layer(gqa_trace, monkeypatch, expect_error):
    table, devices, trace = gqa_trace()
    monkeypatch.setattr(
        producer, "ADAPTER", SimpleNamespace(name="gpt_oss_d_p", model_config=producer.ADAPTER.model_config)
    )
    table.foreign.add((31, 0))
    result = producer._read_slot_kv_and_check_pcc(table, devices, 1, 33, trace, num_ranks=2)
    assert result == pytest.approx({"k": 1.0, "v": 1.0}, abs=1e-6)
    with expect_error(RuntimeError, "31/32 layers"):
        producer._read_slot_kv_and_check_pcc(table, devices, 1, 33, trace, num_ranks=1)
