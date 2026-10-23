# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.demos.deepseek_v3.utils import config_helpers as ch
from models.demos.deepseek_v3.utils.config_dataclass import PrefillChunkSizes

pytestmark = pytest.mark.t3k_compat


@pytest.mark.parametrize(
    "env_value,expected",
    [
        (None, ttnn.FabricConfig.FABRIC_1D),
        ("", ttnn.FabricConfig.FABRIC_1D),
        ("0", ttnn.FabricConfig.FABRIC_1D),
        ("false", ttnn.FabricConfig.FABRIC_1D),
        ("False", ttnn.FabricConfig.FABRIC_1D),
        ("1", ttnn.FabricConfig.FABRIC_1D_RING),
        ("true", ttnn.FabricConfig.FABRIC_1D_RING),
    ],
)
def test_get_fabric_config_parses_use_torus_mode(monkeypatch: pytest.MonkeyPatch, env_value, expected) -> None:
    if env_value is None:
        monkeypatch.delenv("USE_TORUS_MODE", raising=False)
    else:
        monkeypatch.setenv("USE_TORUS_MODE", env_value)

    assert ch.get_fabric_config() == expected


@pytest.mark.parametrize(
    "max_seq_len,num_rows,expected",
    [
        (2048, 16, PrefillChunkSizes(model_chunk=2048, mla_chunk=2048, wkv_b2_chunk=2048)),
        (32768, 16, PrefillChunkSizes(model_chunk=32768, mla_chunk=32768, wkv_b2_chunk=8192)),
        (40000, 16, PrefillChunkSizes(model_chunk=32768, mla_chunk=32768, wkv_b2_chunk=8192)),
        (49152, 16, PrefillChunkSizes(model_chunk=49152, mla_chunk=49152, wkv_b2_chunk=4096)),
        (131072, 16, PrefillChunkSizes(model_chunk=1024, mla_chunk=256, wkv_b2_chunk=256)),
        (8192, 8, PrefillChunkSizes(model_chunk=8192, mla_chunk=8192, wkv_b2_chunk=2048)),
        (12000, 8, PrefillChunkSizes(model_chunk=8192, mla_chunk=8192, wkv_b2_chunk=2048)),
        (32768, 4, PrefillChunkSizes(model_chunk=32768, mla_chunk=32768, wkv_b2_chunk=2048)),
    ],
)
def test_get_prefill_chunk_sizes_uses_lower_bound_thresholds(max_seq_len, num_rows, expected) -> None:
    assert ch.get_prefill_chunk_sizes(max_seq_len, num_rows) == expected
