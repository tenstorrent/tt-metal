# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""V4.1 routed-expert placement (bead 8y7.9.8, ``tt/v41/expert_placement.py``) on device.

* ``test_v41_moe_placement_small``: ``TtV41Moe`` at SmallV41Config dims (synthetic weights) in the checkpoint
  order and in a random expert order (the placement hook patched): the relabelling is invisible to the caller,
  i.e. the gate logits and indices come back in checkpoint ids and the outputs match.
* ``test_v41_moe_placement_production``: ``TtV41Moe`` of every checkpoint layer (bfp8 routed experts, one
  5120-token chunk: the held-out prompts' reference MoE input ``ffn_in``, 2048 tokens tiled) with the placement the
  code selects. Per layer and prompt: the device routing's per-chip routed pairs (most loaded chip / mean), top-6
  recall against the reference gate, a bit-identical repeat, and the traced MoE replay time. Logged as
  ``V41_MOE_PLACEMENT`` JSON lines; under ``scripts/run_safe_pytest.sh --profile`` every measured replay is wrapped in
  ``B moe L<layer> <prompt>`` / ``E ...`` signposts for the per-chip op table.
"""

import json
import time

import pytest
import torch
import torch.nn.functional as F
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.v41 import expert_dtype_reference as R
from models.demos.deepseek_v3_d_p.tests.v41 import expert_load_profile as LP
from models.demos.deepseek_v3_d_p.tests.v41.small_config import SmallV41Config
from models.demos.deepseek_v3_d_p.tests.v41.test_block_v41 import _pcc
from models.demos.deepseek_v3_d_p.tests.v41.test_moe_v41 import _run
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_expert_dtype import _weights
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_trace import MESH, capture
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import get_sp_mesh_composer, get_tp_mesh_composer
from models.demos.deepseek_v3_d_p.tt.v41 import moe as v41_moe
from models.demos.deepseek_v3_d_p.tt.v41.moe import TtV41Moe
from models.demos.deepseek_v3_d_p.tt.v41.weights import resolve_checkpoint

SMALL_SEQ = 512
TIMED_REPLAYS = 5
RECALL_BAR = 0.95  # test_moe_v41 G1 top-6 recall bar
SMALL_LOGITS_PCC = 0.9999  # device gate logits (bf16) vs fp32
SMALL_MARGIN = 0.005  # the permuted build may not be less accurate than the checkpoint-order build


def _order(moe) -> list[int]:
    """The checkpoint expert of every device slot of ``moe`` (checkpoint order when it has no placement)."""
    order = getattr(moe, "expert_order", None)
    return list(range(C.NUM_ROUTED_EXPERTS)) if order is None else list(order)


def _chip_loads(indices: torch.Tensor, order: list[int], num_chips: int) -> list[int]:
    """Routed pairs per linear chip for checkpoint-id ``indices`` under the slot order ``order``."""
    per_chip = len(order) // num_chips
    chip = torch.empty(len(order), dtype=torch.long)
    chip[torch.tensor(order)] = torch.arange(len(order)) // per_chip
    return torch.bincount(chip[indices.long().flatten()], minlength=num_chips).tolist()


def _small_weights(seed: int = 0) -> dict:
    c, g = SmallV41Config, torch.Generator().manual_seed(seed)
    rand = lambda *shape, scale: (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)
    d, h = c.EMB_SIZE, c.MOE_INTERMEDIATE_SIZE
    expert = lambda: {
        "gate_proj": rand(h, d, scale=d**-0.5),
        "up_proj": rand(h, d, scale=d**-0.5),
        "down_proj": rand(d, h, scale=h**-0.5),
    }
    return {
        "gate_weights": {
            "weight": rand(c.NUM_ROUTED_EXPERTS, d, scale=d**-0.5),
            "e_score_correction_bias": torch.randn(c.NUM_ROUTED_EXPERTS, generator=g) * 0.05,
        },
        "routed_expert_weights": [expert() for _ in range(c.NUM_ROUTED_EXPERTS)],
        "shared_expert_weights": expert(),
    }


def _recall(indices: torch.Tensor, ref: torch.Tensor) -> float:
    return (indices.long()[:, :, None] == ref.long()[:, None, :]).any(-1).float().mean().item()


def _small_reference(weights: dict, x: torch.Tensor) -> dict:
    """fp32 reference ``Gate`` + ``Expert`` + shared expert (``model.py``) on the small weights (no activation QDQ)."""
    c, x = SmallV41Config, x.float()
    gate = weights["gate_weights"]
    logits = x @ gate["weight"].float().T
    scores = F.softplus(logits).sqrt()
    indices = (scores + gate["e_score_correction_bias"]).topk(c.NUM_EXPERTS_PER_TOKEN, dim=-1)[1]
    w = scores.gather(1, indices)
    w = w / (w.sum(-1, keepdim=True) + 1e-20) * c.ROUTE_SCALE

    def expert(e, rows):
        g = (rows @ e["gate_proj"].float().T).clamp(max=c.SWIGLU_LIMIT)
        u = (rows @ e["up_proj"].float().T).clamp(-c.SWIGLU_LIMIT, c.SWIGLU_LIMIT)
        return F.silu(g) * u

    final = expert(weights["shared_expert_weights"], x) @ weights["shared_expert_weights"]["down_proj"].float().T
    for e, ew in enumerate(weights["routed_expert_weights"]):
        token, slot = torch.where(indices == e)
        if len(token):
            final[token] += (w[token, slot, None] * expert(ew, x[token])) @ ew["down_proj"].float().T
    return {"logits": logits, "indices": indices, "final": final}


def _small_forward(moe, mesh_device, x) -> dict:
    shape = tuple(mesh_device.shape)
    tt_x = ttnn.from_torch(
        x[None, None],
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims=(2, 3)),
    )
    out, inter = moe(tt_x, return_intermediates=True)
    sp, tp = get_sp_mesh_composer(mesh_device), get_tp_mesh_composer(mesh_device)
    k, seq = C.NUM_EXPERTS_PER_TOKEN, x.shape[0]
    return {
        "logits": ttnn.to_torch(inter.gate_logits, mesh_composer=sp).float().reshape(seq, -1),
        "indices": ttnn.to_torch(inter.gate_indices, mesh_composer=sp, dtype=torch.int32).reshape(seq, k),
        "routed": ttnn.to_torch(inter.routed_output, mesh_composer=tp).float().reshape(seq, -1),
        "final": ttnn.to_torch(out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, shape, dims=(2, 3))).reshape(
            seq, -1
        ),
    }


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_v41_moe_placement_small(mesh_device, device_params, monkeypatch):
    c = SmallV41Config
    weights = _small_weights()
    x = torch.randn(SMALL_SEQ, c.EMB_SIZE, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    order = torch.randperm(c.NUM_ROUTED_EXPERTS, generator=torch.Generator().manual_seed(2)).tolist()
    runs = {}
    for name, placement in (("checkpoint", None), ("permuted", tuple(order))):
        monkeypatch.setattr(v41_moe, "expert_order", lambda layer, experts, group, groups, p=placement: p)
        start = time.perf_counter()
        moe = TtV41Moe(mesh_device, c, 2, weights, SMALL_SEQ)
        assert (getattr(moe, "expert_order", None) is None) == (placement is None)
        runs[name] = _small_forward(moe, mesh_device, x)
        repeat = _small_forward(moe, mesh_device, x)
        for key, value in runs[name].items():
            assert torch.equal(value, repeat[key]), f"{name}: {key} differs between repeated runs"
        logger.info(f"small MoE {name}: built + 2 forwards {time.perf_counter() - start:.1f}s")
        del moe
    # the permuted gate GEMM is not bit-identical (last-bit bf16 differences), so near-ties of the bf16 scores can
    # select differently: both builds are judged against the same fp32 reference, and a wrong expert or column
    # order would give unrelated logits (pcc ~ 0) and a collapsed recall
    ref = _small_reference(weights, x)
    report = {
        name: {
            "logits": _pcc(ref["logits"], run["logits"]),
            "final": _pcc(ref["final"], run["final"]),
            "recall": _recall(run["indices"], ref["indices"]),
        }
        for name, run in runs.items()
    }
    logger.info(f"small MoE placement vs fp32 reference: {report}")
    base, perm = report["checkpoint"], report["permuted"]
    assert perm["logits"] >= SMALL_LOGITS_PCC, report
    assert perm["recall"] >= base["recall"] - SMALL_MARGIN and perm["final"] >= base["final"] - SMALL_MARGIN, report


@pytest.mark.timeout(5400)
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_v41_moe_placement_production(mesh_device, device_params):
    ckpt = resolve_checkpoint()
    if ckpt is None:
        pytest.skip("V4.1 checkpoint shards not downloaded")
    shape, chips = tuple(mesh_device.shape), mesh_device.get_num_devices()
    logger.info(f"device ids (mesh row-major): {mesh_device.get_device_ids()}")
    spec = LP.spec()
    t0 = time.perf_counter()
    inputs = {name: orc.oracle(spec, LP.prompt_tokens(name))["blocks"] for name in LP.HELD_OUT}
    logger.info(f"held-out oracles loaded {time.perf_counter() - t0:.1f}s")
    failures = []
    for layer in LP.LAYERS:
        t0 = time.perf_counter()
        root, w, marker = _weights(ckpt, layer, "bfp8", shape, spec)
        gate = LP.gate_weights(layer)
        moe = TtV41Moe(
            mesh_device, C, layer, w, R.CHUNK, routed_expert_weights_dtype=ttnn.bfloat8_b, weight_cache_path=root
        )
        marker.touch()
        del w
        order = _order(moe)
        logger.info(f"MoE L{layer} built {time.perf_counter() - t0:.1f}s (placement: {order != sorted(order)})")
        for name, blocks in inputs.items():
            x = R.tile_rows(blocks[layer]["ffn_in"].to(torch.bfloat16))
            first = _run(moe, mesh_device, x)
            repeat = _run(moe, mesh_device, x)
            for key in ("final", "indices"):
                assert torch.equal(first[key], repeat[key]), f"L{layer} {name}: {key} differs between repeats"
            ref = LP.reference_indices(x, *gate)
            recall = (first["indices"].long()[:, :, None] == ref[:, None, :]).any(-1).float().mean().item()
            loads = _chip_loads(first["indices"], order, chips)
            identity = _chip_loads(first["indices"], list(range(C.NUM_ROUTED_EXPERTS)), chips)

            tt_x = ttnn.from_torch(
                x[None, None],
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims=(2, 3)),
            )
            trace, _ = capture(mesh_device, lambda: moe(tt_x), [moe])
            trace.replay()  # warm replay
            ttnn.synchronize_device(mesh_device)
            ttnn.ReadDeviceProfiler(mesh_device)
            times = []
            for _ in range(TIMED_REPLAYS):
                t1 = time.perf_counter()
                signpost(f"B moe L{layer} {name}")
                trace.replay()
                ttnn.synchronize_device(mesh_device)
                signpost(f"E moe L{layer} {name}")
                times.append((time.perf_counter() - t1) * 1e3)
                ttnn.ReadDeviceProfiler(mesh_device)
            trace.release()
            report = {
                "layer": layer,
                "prompt": name,
                "placement": order != sorted(order),
                "traced_ms": sorted(times)[len(times) // 2],
                "traced_ms_all": [round(t, 3) for t in times],
                "chip_pairs": loads,
                "max_over_mean": round(max(loads) / (sum(loads) / chips), 3),
                "checkpoint_order_max_over_mean": round(max(identity) / (sum(identity) / chips), 3),
                "recall": round(recall, 5),
            }
            logger.info(f"V41_MOE_PLACEMENT {json.dumps(report)}")
            if recall < RECALL_BAR:
                failures.append(report)
        del moe
    assert not failures, failures
