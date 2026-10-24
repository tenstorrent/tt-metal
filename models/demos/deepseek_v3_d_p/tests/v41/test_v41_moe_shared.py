# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The decode shared expert folded into the tail (``tt/v41/moe_shared.py``, ``V41_SHARED_IN_FFN=rs|ffn``)
against
today's path.

One MoE layer (``V41_SH_LAYER``, default 2) from the weight cache, built as the batched decoder builds it (the
decode routing and tail on, the shared expert as today), fed ``V41_SH_TRIALS`` random tiles (default 4): the
MoE
output on each, then :func:`install_shared_in_ffn` and the MoE output again on the same tiles, every row of
every
chip compared (bf16 tolerance: the shared expert's bfp8 input packing, its fp32 accumulation order and the
sum's
place before the reduce-scatter are the differences). Then both under trace (``V41_SH_REPLAYS``, default 20):
the
per-layer device time.

Run with ``V41_DECODE_ROUTING=1 V41_DECODE_TAIL=1 TT_V41_EXPERT_DTYPE=bfp4`` (the served expert dtype) and
``V41_SHARED_IN_FFN`` unset; ``V41_SH_MODE`` (``rs`` or ``ffn``, default ``rs``) is the mode the test installs
(``V41_SHARED_IN_FFN_DTYPE`` applies to ``ffn``)."""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tests.v41.weight_cache import weight_cache_dir
from models.demos.deepseek_v3_d_p.tt.v41.batch_decode import TtV41BatchDecoder
from models.demos.deepseek_v3_d_p.tt.v41.decode import ROWS, SHARED_IN_FFN_DTYPE
from models.demos.deepseek_v3_d_p.tt.v41.moe_shared import install_shared_in_ffn
from models.demos.deepseek_v3_d_p.tt.v41.transformer import TtV41Transformer
from models.demos.deepseek_v3_d_p.tt.v41.weights import load_layer, load_layer_dense, resolve_checkpoint

try:
    from tracy import signpost
except ModuleNotFoundError:  # not under the tracy wrapper

    def signpost(_name: str) -> None:
        pass


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _all(t) -> torch.Tensor:
    """Every chip's ``[ROWS, hidden/tp]`` tile -> ``[chips, ROWS, hidden/tp]`` fp32."""
    return torch.stack([ttnn.to_torch(c)[0, 0].float() for c in ttnn.get_device_tensors(t)])


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "mesh_device, device_params", galaxy_meshes(trace_region_size=256 * 1024 * 1024), indirect=True
)
def test_v41_moe_shared(mesh_device, device_params):
    assert os.environ.get("V41_SHARED_IN_FFN", "0") == "0", "run with V41_SHARED_IN_FFN unset; the test installs it"
    mode = os.environ.get("V41_SH_MODE", "rs")
    assert os.environ.get("V41_DECODE_TAIL") == "1", "run with V41_DECODE_ROUTING=1 V41_DECODE_TAIL=1"
    layer = int(os.environ.get("V41_SH_LAYER", 2))
    trials = int(os.environ.get("V41_SH_TRIALS", 4))
    replays = int(os.environ.get("V41_SH_REPLAYS", 20))
    users = 4
    ckpt = resolve_checkpoint()
    if ckpt is None:
        pytest.skip("V4.1 checkpoint shards not downloaded")
    sp, tp = tuple(mesh_device.shape)
    max_seq = 1024
    spec = orc.real_spec((layer,), max_seq, checkpoint=ckpt.root)
    root = weight_cache_dir(spec, mesh_device.shape)
    top = ckpt.read(["embed.weight", "norm.weight", "head.weight"])
    t0 = time.time()
    model = TtV41Transformer(
        mesh_device,
        C,
        [layer],
        lambda l, include_moe: (load_layer if include_moe else load_layer_dense)(ckpt, l),
        top["embed.weight"],
        top["norm.weight"],
        top["head.weight"],
        max_seq_len=max_seq,
        chunk=1024,
        weight_cache_path=root,
    )
    decoder = TtV41BatchDecoder(model, root, users_per_rank=users, trace=False)
    logger.info(f"shared moe: layer {layer} built in {time.time() - t0:.1f} s")
    # one prefill first, as every served or tested flow has (the decode tail's reduce-scatter
    # and the routed-expert FFN programs compiled after it)
    torch.manual_seed(3)
    prompt = torch.randint(1000, 20000, (C.SLIDING_WINDOW + 7,))
    _, state = model.prefill(prompt, logit_positions=1)
    del state
    moe = decoder.moes[0]
    hidden = C.EMB_SIZE
    torch.manual_seed(7)
    tiles = []
    for _ in range(trials):
        h = torch.randn(sp, ROWS, hidden)
        tiles.append(
            ttnn.from_torch(
                h.view(1, 1, sp * ROWS, hidden),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, (sp, tp), dims=(2, 3)),
            )
        )
    refs = [_all(decoder._ffn(moe, t)) for t in tiles]
    install_shared_in_ffn(moe, mode, SHARED_IN_FFN_DTYPE)
    result = {"layer": layer, "trials": [], "mesh": [sp, tp], "mode": mode, "dtype": str(SHARED_IN_FFN_DTYPE)}
    worst = {"max_abs": 0.0, "pcc": 1.0, "rel": 0.0}
    for trial, (t, ref) in enumerate(zip(tiles, refs)):
        out = _all(decoder._ffn(moe, t))
        diff = (ref - out).abs()
        max_abs = float(diff.max())
        rel = max_abs / max(float(ref.abs().max()), 1e-6)
        pcc = _pcc(ref, out)
        entry = {
            "trial": trial,
            "max_abs": round(max_abs, 5),
            "rel": round(rel, 6),
            "pcc": round(pcc, 7),
            "ref_absmax": round(float(ref.abs().max()), 4),
            "mean_abs": round(float(diff.mean()), 6),
        }
        result["trials"].append(entry)
        logger.info(f"shared moe trial {json.dumps(entry)}")
        worst = {
            "max_abs": max(worst["max_abs"], max_abs),
            "pcc": min(worst["pcc"], pcc),
            "rel": max(worst["rel"], rel),
        }
    result["worst"] = {k: round(v, 7) for k, v in worst.items()}
    # the per-layer device time of each path, under trace
    tt_h = tiles[0]
    stock, folded = moe.moe.shared_expert._shared, moe.moe.shared_expert

    def with_shared(shared):
        def fn():
            moe.moe.shared_expert = shared
            return decoder._ffn(moe, tt_h)

        return fn

    timing = {}
    paths = (("today", with_shared(stock)), ("folded", with_shared(folded)))
    if os.environ.get("V41_SH_PROFILE") == "1":
        # With device profiling enabled: one eager call of each path between signposts (the per-op kernel split)
        ttnn.synchronize_device(mesh_device)
        signpost("start")
        for name, fn in paths:
            ttnn.deallocate(fn())
            ttnn.synchronize_device(mesh_device)
            signpost(name)
        paths = ()
    for name, fn in paths:
        for _ in range(2):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        outs = [fn() for _ in range(4)]
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        best = None
        for _ in range(replays):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            dt = (time.perf_counter() - t0) / 4
            best = dt if best is None else min(best, dt)
        ttnn.release_trace(mesh_device, tid)
        for o in outs:
            ttnn.deallocate(o)
        timing[name] = round(best * 1e6, 1)
        logger.info(f"shared moe traced {name}: {timing[name]} us a layer (min of {replays} replays of 4 calls)")
    result["layer_us"] = timing
    logger.info(f"V41_SHARED_MOE_RESULT {json.dumps(result)}")
    out_path = os.environ.get("V41_SH_OUT")
    if out_path:
        with open(os.path.expanduser(out_path), "w") as f:
            json.dump(result, f, indent=1)
    decoder.release()
    bar = (0.99999, 0.01) if mode == "rs" else (0.999, 0.1)  # ffn: the op's bfp8 input packing
    assert worst["pcc"] > bar[0] and worst["rel"] < bar[1], (worst, bar)
