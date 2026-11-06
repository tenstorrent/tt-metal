# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU only (no device): the routed experts' weight cache has one on-disk layout (owner 2026-11-06), the bank-major
<stem>.bm.bin (flat_routed_expert bank_major.py).

  - tools/convert_expert_cache.py (the migration of an old .tensorbin cache) writes the same file the miss path does
    (bank_major.write_from_host), and it reads back as the source after the inverse permutation (in a subprocess with
    tt-metal's mock cluster: a host tensor needs a tt-metal context, never the real cards)
  - _CacheMarker lists the .bm.bin files, never deletes a cache file, and keeps an old marker's stats
  - export_pinned_limit sets TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES when unset, keeps an explicit value
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
MOCK = REPO / "tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_P150.yaml"

CONVERT = r"""
import os, sys, torch, ttnn
from ttnn.bringup.flat_routed_expert_ttnn import bank_major as bm
from models.demos.deepseek_v41_flash_d_p.tools import convert_expert_cache as C
d = sys.argv[1]
grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM,
                       ttnn.ShardSpec(grid, (96, 32), ttnn.ShardOrientation.ROW_MAJOR))
host = ttnn.from_torch(torch.randn(1, 96, 256), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, memory_config=mc)
old = os.path.join(d, "L00.bfp4_mse.h.flat.bf4.layk.w_gu.tensorbin")
ttnn._ttnn.tensor.dump_tensor_flatbuffer(old, host)          # an old-layout cache file
out = C.convert(old, None, verify=True)                      # the migration (verify: inverse permutation == source)
assert out == bm.bm_path(old) and bm.is_complete(out)
mine = os.path.join(d, "miss", "L00.bfp4_mse.h.flat.bf4.layk.w_gu.bm.bin")
os.makedirs(os.path.dirname(mine))
bm.write_from_host(mine, host)                               # what a cache miss writes
a, b = open(out, "rb").read(), open(mine, "rb").read()
assert a == b, "the converter and the miss path write different files"
hdr = bm.read_header(out)
assert hdr["rows"] == 96 and hdr["banks"] == 8 and hdr["dtype"] == "BFLOAT4_B" and hdr["mesh"] == [1, 1], hdr
assert C.convert(old, None, verify=False) == out             # already converted: kept
print("CONVERT_OK", hdr["tile_bytes"])
"""


def test_converter_writes_what_the_miss_path_writes(tmp_path):
    if not MOCK.is_file():
        pytest.skip(f"no mock cluster descriptor {MOCK}")
    env = dict(os.environ, TT_METAL_MOCK_CLUSTER_DESC_PATH=str(MOCK), PYTHONPATH=str(REPO), TT_METAL_HOME=str(REPO))
    r = subprocess.run(
        [sys.executable, "-c", CONVERT, str(tmp_path)], env=env, cwd=REPO, capture_output=True, text=True, timeout=600
    )
    assert "CONVERT_OK 576" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
    assert "already converted" in r.stdout


def test_marker_lists_bm_files_and_never_deletes(tmp_path):
    from models.demos.deepseek_v41_flash_d_p.tt.experts import _CacheMarker

    prefix = str(tmp_path / "8x4" / "L03.bfp4_mse.abc")
    (tmp_path / "8x4").mkdir()
    keep = [Path(prefix + s) for s in (".flat.bf4.layk.w_gu.bm.bin", ".flat.bf4.layk.w_d.bm.bin")]
    old = Path(prefix + ".flat.bf4.layk.w_gu.tensorbin")
    for f in [*keep, old]:
        f.write_bytes(b"x" * 10)
    Path(prefix + ".done.json").write_text(json.dumps({"files": {old.name: 10}, "stats": {"changed": 1}}))
    m = _CacheMarker(prefix)  # an old marker (it lists the .tensorbin): its stats kept, nothing deleted
    assert m.stats == {"changed": 1} and all(f.exists() for f in [*keep, old])
    m.write({"changed": 2})
    rec = json.loads(Path(prefix + ".done.json").read_text())
    assert rec == {"files": {f.name: 10 for f in sorted(keep)}, "stats": {"changed": 2}}
    assert _CacheMarker(None).stats is None


def test_export_pinned_limit_sets_it_when_unset_and_keeps_an_explicit_value(monkeypatch):
    from models.demos.deepseek_v41_flash_d_p.tt import settings as S

    monkeypatch.delenv("DS41_EXPERTS_IMPL", raising=False)
    env = {}
    assert S.export_pinned_limit(env) == str(16 << 30) and env[S.PINNED_LIMIT_ENV] == str(16 << 30)
    env = {S.PINNED_LIMIT_ENV: "0"}  # e.g. a runner test's KMD 2.9 workaround: kept
    assert S.export_pinned_limit(env) is None and env[S.PINNED_LIMIT_ENV] == "0"
    monkeypatch.setenv("DS41_EXPERTS_IMPL", "bfp8")
    assert S.pinned_limit_bytes() == 32 << 30
