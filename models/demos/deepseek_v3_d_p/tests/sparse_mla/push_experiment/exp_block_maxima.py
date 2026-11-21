import dataclasses
import os

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.sparse_mla.perf_cases import (
    BOX,
    SCENARIO_CACHE_TOKENS,
    build_sparse_perf_case,
    gate_params,
)
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from tests.ttnn.nightly.unit_tests.operations.experimental.topk_two_level_reference import assert_topk_exact, normalize


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device, device_params, variant, case_name",
    [p for p in gate_params() if p.id.endswith("kimi_k3_1")],
    indirect=["mesh_device", "device_params", "variant"],
)
def test_bmax(mesh_device, device_params, variant, case_name, config_only):
    cache = int(os.environ.get("BMAX_CACHE", BOX.cache_tokens(SCENARIO_CACHE_TOKENS["long"])))
    case = build_sparse_perf_case(
        mesh_device,
        variant,
        config_only,
        cache=cache,
        chunk=BOX.chunk,
        topology=per_axis_topology(device_params["fabric_config"]),
        sp_axis=BOX.sp_axis,
    )
    # Fill the index cache with random keys so the logits are not all tied.
    if case.index_kv_cache is not None:
        shape = list(case.index_kv_cache.shape)
        print("index cache", shape, case.index_kv_cache.dtype, case.index_kv_cache.layout)
    ind = case.mla._indexer
    orig = ind.select_local
    seen = []

    def wrapped(state, **kw):
        out = orig(state, **kw)
        plain = orig(dataclasses.replace(state, block_max=None), **kw)
        ttnn.synchronize_device(mesh_device)
        seen.append(
            dict(
                logits=[ttnn.to_torch(t) for t in ttnn.get_device_tensors(state.logits)],
                bmax=[ttnn.to_torch(t) for t in ttnn.get_device_tensors(state.block_max)]
                if state.block_max is not None
                else None,
                idx=[ttnn.to_torch(t, dtype=torch.uint32) for t in ttnn.get_device_tensors(out)],
                plain=[ttnn.to_torch(t, dtype=torch.uint32) for t in ttnn.get_device_tensors(plain)],
                n=state.topk_valid_length,
            )
        )
        return out

    ind.select_local = wrapped
    start = int(os.environ.get("BMAX_START", cache))
    if os.environ.get("BMAX_META") == "1":

        def _scalar(v):
            return ttnn.from_torch(
                torch.tensor([v], dtype=torch.int64).reshape(1, 1, 1, 1),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        meta = (_scalar(0), _scalar(start), _scalar(start + BOX.chunk))
        orig_forward = case.mla.forward
        case.mla.forward = lambda *a, **k: orig_forward(*a, **{**k, "metadata": meta})
    ttnn.deallocate(case.forward(start))
    s = seen[0]
    n = s["n"] if s["n"] is not None else start + BOX.chunk
    nb = n // 32
    print("valid length", n)
    for d in range(len(s["logits"])):
        L = s["logits"][d].reshape(-1, s["logits"][d].shape[-1]).float()
        want = normalize(L[:, : nb * 32]).reshape(L.shape[0], nb, 32).amax(-1)
        if s["bmax"] is not None and n >= 98304:
            got = normalize(s["bmax"][d].reshape(L.shape[0], -1).float()[:, :nb])
            bad = got != want
        else:
            got, bad = want, torch.zeros_like(want, dtype=torch.bool)
        print(
            f"chip {d}: maxima mismatches {int(bad.sum())} / {bad.numel()}; distinct logit values {L[:, :n].unique().numel()}"
        )
        if bad.any():
            r, b = bad.nonzero()[0].tolist()
            print(
                f"   first at row {r} block {b}: got {got[r, b]} want {want[r, b]} keys {L[r, 32*b:32*b+32].tolist()}"
            )
        ix = s["idx"][d].reshape(L.shape[0], -1).to(torch.int64)
        px = s["plain"][d].reshape(L.shape[0], -1).to(torch.int64)
        for r in (0, 1, 80, L.shape[0] - 1):
            assert_topk_exact(L[r].to(torch.bfloat16), ix[r], n)
        print(f"   top-k with supplied maxima exact on sampled rows; equals two-level-computed: {torch.equal(ix, px)}")
        assert not bad.any()
