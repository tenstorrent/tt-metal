# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The prefill KV table against the Mistral Large 4 decode ring's table, without a device.

The ring publishes config "0" (KV, bfloat16 ROW_MAJOR, 32 x 576 per record, every layer, KV-shared layers holding a
copy of their source's rows) and config "1" (index keys, bfloat8_b TILE, 32 x 128 per record, the 44 index generator
layers). The KV manager copies records byte for byte, so names, record sizes and covered layers must match.
"""

from types import SimpleNamespace

import pytest

import ttnn
from models.demos.deepseek_v3_d_p.tt.mistral_large_4 import kv_table as KT

NUM_LAYERS = 72
KV_SHARED = set(range(39, 72, 4))  # read layer 35's KV
INDEX_SHARED = {j for p in range(17, 71, 4) for j in (p, p + 1)}  # reuse the previous generator's top-k


class _Cfg:
    kv_lora, qk_rope, index_dim = 512, 64, 128

    def owns_kv(self, i):
        return i not in KV_SHARED

    def kv_source(self, i):
        return 35 if i in KV_SHARED else i

    def is_full(self, i):
        return i not in INDEX_SHARED


def _geometry(monkeypatch, mirrors=True):
    monkeypatch.setenv("MISTRAL_LARGE_4_KV_MIRRORS", "1" if mirrors else "0")
    return KT.Geometry.of(_Cfg(), range(NUM_LAYERS), sp=8, chunk=5120, max_seq=56320, slots=2)


def test_config_names_match_ring():
    assert KT.CONFIGS == ("0", "1")
    assert KT.KINDS == {"0": "kvpe", "1": "index"}


def test_kv_covers_every_layer_with_mirrors(monkeypatch):
    geo = _geometry(monkeypatch)
    assert geo.owners(KT.KV) == tuple(range(NUM_LAYERS))
    assert len(geo.owners(KT.INDEX)) == 44
    assert not set(geo.owners(KT.INDEX)) & INDEX_SHARED


def test_without_mirrors_shared_layers_have_no_entry(monkeypatch):
    geo = _geometry(monkeypatch, mirrors=False)
    assert set(geo.owners(KT.KV)) == set(range(NUM_LAYERS)) - KV_SHARED


@pytest.mark.parametrize(
    "dtype, layout, width, size",
    [
        (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, 576, 36864),  # the ring's KV record
        (ttnn.bfloat8_b, ttnn.TILE_LAYOUT, 128, 4352),  # the ring's index record
        (ttnn.bfloat8_b, ttnn.TILE_LAYOUT, 576, 19584),
    ],
)
def test_record_bytes(dtype, layout, width, size):
    cache = SimpleNamespace(shape=[2, 1, 7040, width], dtype=dtype, layout=layout)
    assert KT.record_bytes(cache) == size
