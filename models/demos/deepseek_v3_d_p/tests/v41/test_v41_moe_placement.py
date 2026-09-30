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
* ``test_v41_moe_placement_text_ab`` (bead 8y7.9.12): same-tree A/B of the production placement (``profile``) against
  the checkpoint order (``checkpoint``) on the G2 profile input, the first 5120 tokens of real text
  (``oracle.text_tokens``) through the real sharing schedule (``test_v41_perf.text_stack``). Per layer, rounds
  profile / checkpoint / checkpoint / profile (ABBA against drift) on the same inputs: the traced block replay (layer
  span; its MoE swapped between placements) on the block input the device stack gives it, and the traced MoE replay
  on the reference MoE input of that text (``text_spec``: 5120 distinct tokens, not tiled). Per placement: device
  routing per-chip pairs, top-6 recall. Logged as ``V41_MOE_PLACEMENT_AB`` JSON lines; signposts
  ``B block|moe L<layer> <placement> r<round>`` under ``--profile``. The checkpoint order is a construction-time
  choice of this test only (``expert_placement_choice``); its device weights live in their own weight-cache
  directory, prepared outside the device lock with ``prepare_text_ab`` on a mock mesh.
"""

import gc
import json
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from loguru import logger
from tracy import signpost

import ttnn
from models.common import timing_events
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.v41 import expert_dtype_reference as R
from models.demos.deepseek_v3_d_p.tests.v41 import expert_load_profile as LP
from models.demos.deepseek_v3_d_p.tests.v41 import weight_cache
from models.demos.deepseek_v3_d_p.tests.v41.small_config import SmallV41Config
from models.demos.deepseek_v3_d_p.tests.v41.test_block_v41 import _pcc
from models.demos.deepseek_v3_d_p.tests.v41.test_moe_v41 import _run
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_expert_dtype import _weights
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_perf import text_stack
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_trace import MESH, capture
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import get_sp_mesh_composer, get_tp_mesh_composer
from models.demos.deepseek_v3_d_p.tt.v41 import expert_placement
from models.demos.deepseek_v3_d_p.tt.v41 import moe as v41_moe
from models.demos.deepseek_v3_d_p.tt.v41.moe import TtV41Moe
from models.demos.deepseek_v3_d_p.tt.v41.weights import load_layer, load_layer_dense, resolve_checkpoint
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker

SMALL_SEQ = 512
TIMED_REPLAYS = 5
RECALL_BAR = 0.95  # test_moe_v41 G1 top-6 recall bar
SMALL_LOGITS_PCC = 0.9999  # device gate logits (bf16) vs fp32
SMALL_MARGIN = 0.005  # the permuted build may not be less accurate than the checkpoint-order build
PLACEMENTS = ("profile", "checkpoint")
AB_ROUNDS = ("profile", "checkpoint", "checkpoint", "profile")  # per layer, ABBA: each placement in both halves


def text_spec() -> orc.OracleSpec:
    """The reference of the G2 profile input: ``oracle.text_tokens(CHUNK)`` through the real sharing schedule."""
    return orc.real_spec(LP.LAYERS, R.CHUNK, checkpoint=orc.HF_SNAPSHOT)


@contextmanager
def expert_placement_choice(name: str):
    """Construction-time expert order of this test: ``profile`` is the production placement
    (``expert_placement.expert_order``); ``checkpoint`` keeps the checkpoint order (``expert_order`` patched to None,
    and the weight-cache key built without the load profile, which then does not decide the stored bytes)."""
    assert name in PLACEMENTS, name
    with pytest.MonkeyPatch.context() as mp:
        if name == "checkpoint":
            mp.setattr(v41_moe, "expert_order", lambda *args: None)
            mp.setattr(weight_cache, "CONVERSION_DATA", ())
        yield


def placement_cache_dir(name: str, mesh_shape) -> Path:
    """The routed-expert bfp8 weight-cache directory of placement ``name`` (``profile``: the production one)."""
    with expert_placement_choice(name):
        root = weight_cache.weight_cache_dir(text_spec(), mesh_shape, ttnn.bfloat8_b)
    return root if name == "profile" else root.with_name(root.name.replace("-mesh", "-checkpoint_order-mesh"))


def build_moe(mesh_device, ckpt, layer: int, name: str) -> TtV41Moe:
    """``TtV41Moe`` of real ``layer`` (bfp8 routed experts, one CHUNK) in placement ``name``, as ``TtV41Block``
    builds it; the routed experts come from the placement's weight cache (built from the checkpoint on a miss)."""
    root = placement_cache_dir(name, tuple(mesh_device.shape))
    root.mkdir(parents=True, exist_ok=True)
    init_checker(root)
    marker = root / f"layer_{layer}.complete"
    start = time.perf_counter()
    w = load_layer_dense(ckpt, layer) if marker.exists() else load_layer(ckpt, layer)
    with expert_placement_choice(name):
        moe = TtV41Moe(
            mesh_device, C, layer, w, R.CHUNK, routed_expert_weights_dtype=ttnn.bfloat8_b, weight_cache_path=root
        )
    marker.touch()
    assert (moe.expert_order is None) == (name == "checkpoint"), (name, layer)
    logger.info(f"MoE L{layer} {name} built {time.perf_counter() - start:.1f}s ({root.name})")
    return moe


def prepare_text_ab(mesh_device) -> None:
    """Fills both placements' weight caches of every layer (run on a mock mesh, outside the device lock)."""
    ckpt = resolve_checkpoint()
    for layer in LP.LAYERS:
        for name in PLACEMENTS:
            del_me = build_moe(mesh_device, ckpt, layer, name)
            del del_me
            gc.collect()


def _traced_ms(mesh_device, forward, moes, label: str) -> list[float]:
    """Synchronized wall ms of TIMED_REPLAYS trace replays of ``forward`` (after a warm replay), each between
    ``B <label>`` / ``E <label>`` signposts."""
    trace, _ = capture(mesh_device, forward, moes)
    trace.replay()  # warm replay
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)
    times = []
    for _ in range(TIMED_REPLAYS):
        t1 = time.perf_counter()
        signpost(f"B {label}")
        trace.replay()
        ttnn.synchronize_device(mesh_device)
        signpost(f"E {label}")
        times.append((time.perf_counter() - t1) * 1e3)
        ttnn.ReadDeviceProfiler(mesh_device)
    trace.release()
    return times


def _median(values):
    return sorted(values)[len(values) // 2]


def _chip_of(order: list[int], num_chips: int) -> torch.Tensor:
    per_chip = len(order) // num_chips
    chip = torch.empty(len(order), dtype=torch.long)
    chip[torch.tensor(order)] = torch.arange(len(order)) // per_chip
    return chip


def _order(moe) -> list[int]:
    """The checkpoint expert of every device slot of ``moe`` (checkpoint order when it has no placement)."""
    order = getattr(moe, "expert_order", None)
    return list(range(C.NUM_ROUTED_EXPERTS)) if order is None else list(order)


def _chip_loads(indices: torch.Tensor, order: list[int], num_chips: int) -> list[int]:
    """Routed pairs per linear chip for checkpoint-id ``indices`` under the slot order ``order``."""
    chip = _chip_of(order, num_chips)
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
            times = _traced_ms(mesh_device, lambda: moe(tt_x), [moe], f"moe L{layer} {name}")
            report = {
                "layer": layer,
                "prompt": name,
                "placement": order != sorted(order),
                "traced_ms": _median(times),
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


@pytest.mark.timeout(5400)
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_v41_moe_placement_text_ab(mesh_device, device_params):
    ckpt = resolve_checkpoint()
    if ckpt is None:
        pytest.skip("V4.1 checkpoint shards not downloaded")
    shape, chips = tuple(mesh_device.shape), mesh_device.get_num_devices()
    logger.info(f"device ids (mesh row-major): {mesh_device.get_device_ids()}")
    with timing_events.phase("oracle", key="text_ab"):
        start = time.perf_counter()
        ref_blocks = orc.oracle(text_spec(), orc.text_tokens(R.CHUNK))["blocks"]
        moe_in = {layer: ref_blocks[layer]["ffn_in"].to(torch.bfloat16) for layer in LP.LAYERS}
        del ref_blocks
        logger.info(f"text reference MoE inputs loaded {time.perf_counter() - start:.1f}s")
    with timing_events.phase("weights", key="text_stack"):
        blocks, x0, pre0, fresh = text_stack(mesh_device, "bfp8")
    for layer in LP.LAYERS:
        assert blocks[layer].ffn.expert_order is not None, f"L{layer}: production build has no placement"
    with timing_events.phase("compute", key="stack_inputs"):
        start = time.perf_counter()
        state, inputs, (x, pre) = fresh(), {}, (x0, pre0)
        for layer in LP.LAYERS:  # one untraced pass: compiles and gives every block its device input
            inputs[layer] = (x, pre)
            x, pre = blocks[layer](x, pre, state, R.CHUNK)
        ttnn.synchronize_device(mesh_device)
        logger.info(f"untraced stack pass {time.perf_counter() - start:.1f}s")

    failures, summary = [], []
    for layer in LP.LAYERS:
        block, (bx, bpre) = blocks[layer], inputs[layer]
        x = moe_in[layer]
        tt_x = ttnn.from_torch(
            x[None, None],
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims=(2, 3)),
        )
        ref = LP.reference_indices(x, *LP.gate_weights(layer))
        current, routing, rounds = "profile", {}, []
        for rnd, name in enumerate(AB_ROUNDS):
            if name != current:
                with timing_events.phase("weights", key=f"moe L{layer} {name}"):
                    block.ffn = None
                    gc.collect()
                    block.ffn = build_moe(mesh_device, ckpt, layer, name)
                current = name
            moe = block.ffn
            if name not in routing:
                run = _run(moe, mesh_device, x)
                order = _order(moe)
                pairs = _chip_loads(run["indices"], order, chips)
                counts = torch.bincount(run["indices"].long().flatten(), minlength=C.NUM_ROUTED_EXPERTS).double()
                modelled = expert_placement.chip_costs(counts.numpy(), _chip_of(order, chips).numpy(), chips)[0]
                routing[name] = {
                    "final": run["final"],
                    "chip_pairs": pairs,
                    "pairs_max_over_mean": round(max(pairs) / (sum(pairs) / chips), 3),
                    "modelled_chip_ms": [round(float(v) / 1e3, 3) for v in modelled],
                    "recall": round(
                        (run["indices"].long()[:, :, None] == ref[:, None, :]).any(-1).float().mean().item(), 5
                    ),
                }
                if routing[name]["recall"] < RECALL_BAR:
                    failures.append((layer, name, routing[name]["recall"]))
            with timing_events.phase("compute", key=f"L{layer} {name} r{rnd}"):
                block_times = _traced_ms(
                    mesh_device, lambda: block(bx, bpre, state, R.CHUNK), [moe], f"block L{layer} {name} r{rnd}"
                )
                moe_times = _traced_ms(mesh_device, lambda: moe(tt_x), [moe], f"moe L{layer} {name} r{rnd}")
            rounds.append({"placement": name, "block_ms": _median(block_times), "moe_ms": _median(moe_times)})
            logger.info(f"V41_MOE_PLACEMENT_AB_ROUND {json.dumps({'layer': layer, 'round': rnd, **rounds[-1]})}")
            del moe
        assert current == "profile"  # AB_ROUNDS ends in the production placement: later layers' stack is unchanged
        mean = lambda key, name: sum(r[key] for r in rounds if r["placement"] == name) / 2
        report = {
            "layer": layer,
            "block_ms": {n: round(mean("block_ms", n), 3) for n in PLACEMENTS},
            "moe_ms": {n: round(mean("moe_ms", n), 3) for n in PLACEMENTS},
            "rounds": rounds,
            "routing": {n: {k: v for k, v in r.items() if k != "final"} for n, r in routing.items()},
            "final_pcc_profile_vs_checkpoint": round(
                _pcc(routing["profile"]["final"], routing["checkpoint"]["final"]), 6
            ),
        }
        summary.append(report)
        logger.info(f"V41_MOE_PLACEMENT_AB {json.dumps(report)}")
    for r in summary:
        logger.info(
            f"L{r['layer']}: block {r['block_ms']['checkpoint']:.2f} -> {r['block_ms']['profile']:.2f} ms, MoE "
            f"{r['moe_ms']['checkpoint']:.2f} -> {r['moe_ms']['profile']:.2f} ms (checkpoint order -> profile)"
        )
    assert not failures, failures
