# k2_soak: indexer_score_dsa with consumed_rows=1 + valid_length_lane_rows (QWEN38_IDX_ROWWRITE=1) and a valid_length_tensor,
# KC 1 and 4, 20 calls per trace, 1000 replays; q and the gate are rewritten every 10 replays.  The comparison covers the
# consumed row of each lane over its live columns (the other rows are not written).  Temporary: delete with soak_common.py.
import importlib.util
import os

import pytest
import torch

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, *[".."] * 7))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = _load("indexer_rowwrite_for_soak", os.path.join(HERE, "test_indexer_score_lanes_kc_rowwrite.py"))
sc = _load("soak_common_for_soak", os.path.join(ROOT, "models/demos/blackhole/qwen38_flash_next/tests/soak_common.py"))

pytestmark = pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score is Blackhole-only")


@pytest.mark.parametrize("device_params", [{"trace_region_size": 128 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("kc", [1, 4], ids=lambda v: f"kc{v}")
@pytest.mark.parametrize("lanes,blocks", [(8, 8192), (4, 65536)], ids=lambda v: str(v))
def test_soak_indexer_rowwrite(device, lanes, blocks, kc):
    q, k, w, rows = base._inputs(lanes, blocks)
    q_d, k_d, w_d = base._dev(q, device), base._dev(k, device), base._dev(w, device)
    vals, len_t = base._lens(lanes, blocks, device)
    kwargs = dict(
        chunk_start_idx=lanes * rows - 32,
        consumed_rows=1,
        valid_length_lane_rows=rows,
        valid_length_tensor=len_t,
        **base._kc_cfg(kc),
    )
    hosts = []
    for v in range(sc.VARIANTS):
        qv, _, wv, _ = base._inputs(lanes, blocks, seed=7 + 100 * v) if v else (q, None, w, None)
        hosts.append(
            (
                sc.host_tensor(qv, ttnn.bfloat16, ttnn.TILE_LAYOUT),
                sc.host_tensor(wv, ttnn.bfloat16, ttnn.TILE_LAYOUT),
            )
        )
    box = {}

    def call(keep):
        out = ttnn.experimental.indexer_score_dsa(q_d, k_d, w_d, **kwargs)
        if keep:
            box["out"] = out
        else:
            ttnn.deallocate(out)

    def prepare(i):
        hq, hw = hosts[i % sc.VARIANTS]
        sc.copy_in(hq, q_d)
        sc.copy_in(hw, w_d)

    def observe():
        res = ttnn.to_torch(box["out"])
        out = []
        for u in range(lanes):
            lo, hi = base._live_cols(u, rows, blocks, vals)
            out.append(sc.raw(res[0, 0, u, lo:hi]))
        return out

    sc.soak(device, f"indexer_rowwrite_kc{kc}", lanes, call, prepare, observe)
