# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill by the prefill stack, decode by the SC1 modules, layers 0-5 with checkpoint weights on
the 8x4 Galaxy, against HF (tt/model.py, tt/prefill_bridge.py; tenstorrent/tt-metal#65893).

    token ids -> the stack's prefill -> bridge -> TtHy4KvCache -> TtHy4Model.decode_step x T -> logits

The golden (reference/hf_harness.py record_decode_golden) is the HF model cut to the same six
layers, run in BF16 over the prompt and its own greedy continuation. Step j feeds the token at
position N - 1 + j (step 0 re-runs the last prompt token: the prefill hands over no state) and is
compared with HF's rows of that position: every layer's attention input and output, MoE / FFN
input and output and streams, the head readout, the logits and the next token. The steps are
teacher-forced with HF's tokens, so every step is compared on HF's own context; a second pass
lets the device feed its own greedy tokens back and counts how many of HF's it reproduces.

* test_prefill_then_decode: one user, a prompt below the top-k (1022 tokens: every position is
  attended) and one above it (5118 tokens, two prefill chunks: the indexer's selection decides
  what is attended), with the caches handed over through the host or on the device. The order is
  the one 78 layers need: prefill, free the prefill weights, load the decode weights. Also the
  rows the steps appended, and DRAM and wall time along the way.
* test_users_and_latency: 8 users in one cache with both prompts interleaved, prefill and decode
  weights resident together, batched steps, and the per-layer time split at 1 user and at 8.

Every case builds the prefill runtime and six layers of decode weights from the checkpoint (some
minutes): one case per job, with a long timeout.
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.reference import hf_harness
from models.experimental.hy4_preview.reference.indexer import full_layers
from models.experimental.hy4_preview.tests.common import GALAXY_MESH, HY4_HF_MODEL_ENV, pcc
from models.experimental.hy4_preview.tt import frames
from models.experimental.hy4_preview.tt import prefill_bridge as pb
from models.experimental.hy4_preview.tt.common import MeshPlan
from models.experimental.hy4_preview.tt.decoder_layer import hidden_to_device
from models.experimental.hy4_preview.tt.kv_cache import INDEX_KEY, LATENT, KvCacheConfig, TtHy4KvCache
from models.experimental.hy4_preview.tt.model import LAYER_STAGES, RECORDED, Checkpoint, ModelConfig, TtHy4Model

LAYERS = hf_harness.DEFAULT_NUM_LAYERS
TYPES = tuple(full_layers(LAYERS))
FULL = tuple(n for n, kind in enumerate(TYPES) if kind == "full")
MAX_SEQ = 8192
ROPE = 64
TARGET_LAYER_MS = 1e3 / 75 / 78  # 8 users x 1M context at 75 tokens per second per user

# Floors per recorded stage of a decode step against HF (BF16), layers 0-5, BFP8 weights with BFP4
# routed experts, over a cache the prefill stack wrote (scaled-FP8 latents, its own BFP4 experts).
# Set after the first runs of this test, with room: the lowest measured over both prompts, every
# hand-over and the batch of 8 users are attn_in 0.9965, attn_out 0.9960, streams 0.9959,
# mlp_in 0.9960, head readout 0.9956, logits 0.9953, appended latent rows 0.9998, appended index
# keys 0.9990, 97.2% of HF's top-2048 picks. The device runs on its own streams (nothing is fed
# from the golden), so every number includes the layers below and the prefill's rows.
PCC_STAGE = {
    "attn_in": 0.99,
    "attn_out": 0.98,
    "streams_mid": 0.99,
    "mlp_in": 0.99,
    # Asserted where the device's router makes HF's eight picks for the token (and on the dense
    # layer): one token's MoE output moves with every expert picked differently, 0.946 to 0.979
    # measured with seven of HF's eight against 0.9876 or better with all eight. With other picks
    # the score is logged, the picks must share MIN_ROUTER_SHARED with HF's, and the streams after
    # the layer keep their own floor.
    "mlp_out": 0.98,
    "streams_out": 0.99,
}
PCC_READOUT = 0.99
PCC_LOGITS = 0.98
PCC_APPENDED = {"latent": 0.99, "index_key": 0.99}
MIN_PICKS_SHARED = 0.96  # of the 2048 picks, with HF's (bf16) of the same step
MIN_ROUTER_SHARED = 0.75  # of a token's 8 expert picks, with HF's
# How far below a host router's cut on the same input a device expert pick may score: the device
# router's own tolerance (tests/test_moe.py PICK_TOLERANCE).
ROUTER_PICK_TOLERANCE = 2e-3

galaxy = pytest.mark.parametrize(
    "device_params, mesh_device", [(pb.device_params(), GALAXY_MESH)], indirect=True, ids=["8x4-torus"]
)


def _golden(prompt_len):
    golden = hf_harness.find_decode_golden(LAYERS, prompt_len)
    if golden is None:
        pytest.skip(f"no decode golden for a {prompt_len}-token prompt under {hf_harness.GOLDEN_ROOT}")
    assert tuple(golden["meta"]["indexer_types"]) == TYPES
    return golden


def _checkpoint():
    root = os.environ.get(HY4_HF_MODEL_ENV)
    if not root or not os.path.isfile(os.path.join(root, "model.safetensors.index.json")):
        pytest.skip(f"{HY4_HF_MODEL_ENV} does not point at the checkpoint")
    return Checkpoint(root)


def _decode_cache(mesh_device, users, dtype="bfp8"):
    cfg = KvCacheConfig(num_layers=LAYERS, indexer_layers=FULL, max_users=users, max_context=MAX_SEQ, dtype=dtype)
    return TtHy4KvCache(mesh_device, cfg)


def _gb(view):
    return round(view["allocated_gb"], 4)


def _build_model(mesh_device, cache, numbers, label):
    """The decode model, with DRAM per chip recorded as it grows."""
    marks = {"start": pb.dram(mesh_device)["allocated_gb"]}
    plan = MeshPlan(mesh_device)
    assert (plan.hidden_tp, plan.head_tp) == (4, 8)
    t0 = time.perf_counter()
    model = TtHy4Model(
        plan,
        ModelConfig(num_layers=LAYERS),
        cache,
        _checkpoint(),
        progress=lambda name: marks.__setitem__(name, pb.dram(mesh_device)["allocated_gb"]),
    )
    ttnn.synchronize_device(mesh_device)
    numbers["decode_build_s"] = time.perf_counter() - t0
    names = list(marks)
    grown = {names[i]: marks[names[i]] - marks[names[i - 1]] for i in range(1, len(names))}
    numbers["decode_weights_gb"] = grown
    layers = [grown[f"layer {n}"] for n in range(LAYERS)]
    full = [layers[n] for n in FULL if n > 0]
    shared = [layers[n] for n in range(1, LAYERS) if n not in FULL]
    # 78 layers: the dense layer 0 (full), 20 more full MoE layers, 57 shared MoE layers, one embedding and head
    numbers["decode_weights_gb_78"] = (
        grown["embedding_and_head"] + layers[0] + 20 * sum(full) / len(full) + 57 * sum(shared) / len(shared)
    )
    logger.info(
        f"MEASURED {label} decode weights, DRAM per chip: embedding + head {grown['embedding_and_head']:.3f} GB, "
        f"layers {[round(x, 3) for x in layers]} GB (layer 0 dense; full indexer on {list(FULL)}); "
        f"78 layers extrapolate to {numbers['decode_weights_gb_78']:.2f} GB; built in {numbers['decode_build_s']:.0f} s "
        f"(per layer {[round(s) for s in model.seconds['layers']]} s, embedding + head {model.seconds['embedding_and_head']:.0f} s)"
    )
    return model


_ROUTERS = {}


def _host_router_margin(layer, mlp_in, picked):
    """How far below a host router's cut the device's worst pick scores, on the same MoE input
    [hidden] (HF's router: sigmoid scores plus the correction bias, top 8). Zero or less when the
    device made the host router's picks; a small positive number is a near tie that went the other
    way in the device's arithmetic."""
    if layer not in _ROUTERS:
        prefix = f"model.layers.{layer}.mlp.gate."
        _ROUTERS[layer] = _checkpoint().tensors([prefix + "weight", prefix + "e_score_correction_bias"])
    weight, bias = _ROUTERS[layer]
    biased = torch.sigmoid(torch.nn.functional.linear(mlp_in.float(), weight)) + bias
    cut = biased.topk(int(picked.sum())).values[-1]
    return (cut - biased[picked].min()).item()


def _router_picks(model, layer, mlp_in):
    """The experts the device's router picks for MoE inputs [U, hidden] (the rows a step recorded,
    as BF16 holds them): bool [U, E]."""
    mlp = model.layers[layer].mlp
    x = hidden_to_device(model.plan, mlp_in, ttnn.bfloat16)
    return model.plan.to_torch_replicated(mlp.route(mlp.whole(x))).float()[0, 0] != 0


def _compare_step(label, golden, j, record, logits, user_row=0, model=None):
    """One step's record (row ``user_row``) against HF's row j. Returns the lowest PCC per stage.
    With ``model``, also logs how many of HF's expert picks the device's router makes on its own
    MoE input: what the mlp_out score of a single token hangs on."""
    low = {}
    picks_shared = None
    for layer in range(LAYERS):
        hf, dev = golden["layers"][layer], record[layer]
        router_shared = 1.0  # the dense layer has no router
        if model is not None and "router_indices" in hf:
            picked = _router_picks(model, layer, dev["mlp_in"])[user_row]
            theirs = set(hf["router_indices"][j].tolist())
            mine = set(picked.nonzero().flatten().tolist())
            router_shared = len(mine & theirs) / len(theirs)
            margin = _host_router_margin(layer, dev["mlp_in"][user_row], picked)
            logger.info(
                f"{label} L{layer} router: {len(mine & theirs)} of {len(theirs)} expert picks are HF's; on the "
                f"device's own input its worst pick is {margin:.1e} below a host router's cut"
            )
            low["router_shared"] = min(low.get("router_shared", 1.0), router_shared)
            low["router_margin"] = max(low.get("router_margin", 0.0), margin)
        for key in RECORDED:
            score = pcc(hf[key][j].float(), dev[key][user_row])
            if key == "mlp_out" and router_shared < 1.0:
                logger.info(f"{label} L{layer} mlp_out with other expert picks than HF's: pcc={score:.6f}")
                low["mlp_out_other_picks"] = min(low.get("mlp_out_other_picks", 1.0), score)
                continue
            logger.info(f"{label} L{layer} {key}: pcc={score:.6f}")
            low[key] = min(low.get(key, 1.0), score)
        if "picks" in dev:
            position = golden["meta"]["positions"][j]
            picked, valid = dev["picks"][user_row]
            mine = set(picked[valid].tolist())
            hf_picks = hf["topk_indices"][j].long()
            theirs = set(hf_picks[hf_picks <= position].tolist())  # HF pads a short context with keys it masks
            assert (
                len(mine) == int(valid.sum()) == min(golden["meta"]["index_topk"], position + 1)
            ), f"{label} L{layer}: {int(valid.sum())} valid picks, {len(mine)} distinct, at position {position}"
            assert max(mine) <= position, f"{label} L{layer}: a pick is ahead of the query"
            shared = len(mine & theirs) / len(theirs)
            logger.info(f"{label} L{layer} picks: {len(mine & theirs)} of {len(theirs)} are HF's ({shared:.4f})")
            picks_shared = shared if picks_shared is None else min(picks_shared, shared)
    readout = pcc(golden["hc_head"][j].float(), record[LAYERS]["hc_head"][user_row])
    logger.info(f"{label} head readout: pcc={readout:.6f}")
    hf_logits = golden["logits"][j]
    score = pcc(hf_logits, logits)
    logger.info(f"{label} logits: pcc={score:.6f}")
    hf_top, top = hf_logits.topk(5).indices.tolist(), logits.topk(5).indices.tolist()
    logger.info(
        f"{label} next token: device {top[0]} HF {hf_top[0]} ({'same' if top[0] == hf_top[0] else 'DIFFERENT'}); "
        f"top-5 device {top} HF {hf_top}; HF's margin top1 - top2 {hf_logits.topk(2).values.diff().abs().item():.3f}"
    )
    low.update(readout=readout, logits=score, same_token=top[0] == hf_top[0])
    if picks_shared is not None:
        low["picks_shared"] = picks_shared
    return low


def _judge_steps(label, lows) -> list:
    """Log the lowest score of every stage over the steps; return what is under its floor (checked
    at the end of the test, so that one low stage does not hide the measurements after it)."""
    keys = list(PCC_STAGE) + ["readout", "logits"]
    worst = {key: min(low[key] for low in lows if key in low) for key in keys if any(key in low for low in lows)}
    logger.info(f"{label} lowest over {len(lows)} steps: " + ", ".join(f"{k} {v:.6f}" for k, v in worst.items()))
    other = [low["mlp_out_other_picks"] for low in lows if "mlp_out_other_picks" in low]
    router = [low["router_shared"] for low in lows if "router_shared" in low]
    if router:
        logger.info(
            f"{label} router: lowest share of HF's expert picks {min(router):.3f}; mlp_out where the picks differ: "
            + (f"lowest pcc {min(other):.6f} over {len(other)} steps" if other else "never")
        )
    same = sum(low["same_token"] for low in lows)
    logger.info(f"{label} next token equal to HF's on {same} of {len(lows)} steps")
    floors = {**PCC_STAGE, "readout": PCC_READOUT, "logits": PCC_LOGITS}
    failures = [f"{label}: {k} pcc={v:.6f} under {floors[k]}" for k, v in worst.items() if v < floors[k]]
    shared = [low["picks_shared"] for low in lows if "picks_shared" in low]
    if shared:
        logger.info(f"{label} lowest share of HF's top-k picks over the steps and full layers: {min(shared):.4f}")
    if shared and min(shared) < MIN_PICKS_SHARED:
        failures.append(f"{label}: picks shared with HF {min(shared):.4f} under {MIN_PICKS_SHARED}")
    if router and min(router) < MIN_ROUTER_SHARED:
        failures.append(f"{label}: expert picks shared with HF {min(router):.3f} under {MIN_ROUTER_SHARED}")
    margins = [low["router_margin"] for low in lows if "router_margin" in low]
    if margins:
        logger.info(
            f"{label} router: the device's worst pick below a host router's cut on its own input: {max(margins):.1e}"
        )
        if max(margins) > ROUTER_PICK_TOLERANCE:
            failures.append(f"{label}: a device expert pick is {max(margins):.1e} below a host router's cut")
    return failures


def _idle_sync_ms(mesh_device, repeats=20):
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        ttnn.synchronize_device(mesh_device)
    return 1e3 * (time.perf_counter() - t0) / repeats


KINDS = {
    "dense FFN + full indexer": [0],
    "MoE + full indexer": [n for n in FULL if n > 0],
    "MoE + shared indexer": [n for n in range(1, LAYERS) if n not in FULL],
}


def _log_profile(label, profiles, users, sync_ms):
    """The split of a layer's time over its pieces, by layer kind, from steps run with a synchronize
    after every piece. An idle synchronize takes ``sync_ms``; that much is taken off every piece
    for each synchronize it includes, so the pieces carry about that much uncertainty each."""
    for kind, layers in KINDS.items():
        total = {}
        for profile in profiles:
            for layer in layers:
                for stage, seconds in profile["layers"][layer].items():
                    total[stage] = total.get(stage, 0.0) + seconds
        count = len(profiles) * len(layers)
        mean = {stage: value / count for stage, value in total.items()}
        inner = {k.split(" / ")[1]: 1e3 * v - users * sync_ms for k, v in mean.items() if k.startswith("attention / ")}
        inner = {stage: ms for stage, ms in inner.items() if mean[f"attention / {stage}"] > 0}
        ms = {}
        for stage in LAYER_STAGES:
            syncs = mean[f"laps / {stage}"]
            if stage == "attention":  # every user's step: one synchronize at its start, one per stage it marks
                syncs += users * (len(inner) + 1)
            ms[stage] = 1e3 * mean[stage] - syncs * sync_ms
        logger.info(
            f"MEASURED {label} {users} user(s), {kind}, split of a layer (ms, staged, idle synchronize of "
            f"{sync_ms:.2f} ms taken off each piece): "
            + ", ".join(f"{stage} {ms[stage]:.2f}" for stage in LAYER_STAGES)
            + "; inside attention, summed over the users: "
            + ", ".join(f"{stage} {value:.2f}" for stage, value in inner.items())
        )


def _measure_latency(model, mesh_device, label, tokens, positions, users, context):
    """Untraced wall time of a decode step for ``users``: the whole step and every layer on its own
    with the ops issued back to back (no synchronize inside), then the split of a layer."""
    count = len(users)
    model.greedy(model.decode_step(tokens, positions, users=users))  # compiles this batch's programs
    sync_ms = _idle_sync_ms(mesh_device)
    repeats = 4
    t0 = time.perf_counter()
    for _ in range(repeats):
        model.greedy(model.decode_step(tokens, positions, users=users))
    ttnn.synchronize_device(mesh_device)
    step = (time.perf_counter() - t0) / repeats
    logger.info(
        f"MEASURED {label} decode step, {count} user(s), context {context}, untraced, issued back to back: "
        f"{step * 1e3:.1f} ms per step ({LAYERS} layers + embedding + head + greedy) = {count / step:.2f} tokens "
        f"per second over {LAYERS} layers; budget for 78 layers and 8 users: {1e3 / 75:.1f} ms per step"
    )
    times = model.layer_times(tokens, positions, users=users)
    per_layer = [1e3 * t for t in times["layers"]]
    by_kind = {kind: sum(per_layer[n] for n in layers) / len(layers) for kind, layers in KINDS.items()}
    logger.info(
        f"MEASURED {label} per layer, {count} user(s), context {context}, untraced, issued back to back (ms): "
        f"{[round(t, 2) for t in per_layer]}; by kind: "
        + ", ".join(f"{kind} {ms:.2f}" for kind, ms in by_kind.items())
        + f"; embedding {1e3 * times['embedding']:.2f}, head readout + norm + LM head + greedy {1e3 * times['head']:.2f}; "
        f"budget {TARGET_LAYER_MS:.3f} ms per layer for the whole batch: "
        + ", ".join(f"{kind} {ms / TARGET_LAYER_MS:.0f}x" for kind, ms in by_kind.items())
    )
    # 21 full layers of which one dense, 57 shared
    model_ms = by_kind["dense FFN + full indexer"] + 20 * by_kind["MoE + full indexer"]
    model_ms += 57 * by_kind["MoE + shared indexer"] + 1e3 * (times["embedding"] + times["head"])
    logger.info(
        f"MEASURED {label} extrapolated to 78 layers, {count} user(s), context {context}: {model_ms:.0f} ms per step, "
        f"{count * 1e3 / model_ms:.2f} tokens per second in all, {1e3 / model_ms:.2f} per user (target 75 per user)"
    )
    profiles = []
    for _ in range(2):
        profile = {}
        model.decode_step(tokens, positions, users=users, profile=profile)
        profiles.append(profile)
    _log_profile(f"{label} context {context}", profiles, count, sync_ms)
    return by_kind


# id -> (prompt tokens, hand-over, the prefill stack's KVPE format). "device" moves on the device what
# the bridge can: the index keys always, the latents too out of a BF16_RM prefill cache ("allondevice":
# no cache row passes through the host); the latents of the packed scaled-FP8 cache go through the host.
CASES = {
    "P1022-host": (1022, "host", "scaled_fp8"),
    "P5118-host": (5118, "host", "scaled_fp8"),
    "P5118-device": (5118, "device", "scaled_fp8"),
    "P5118-allondevice": (5118, "device", "bf16_rm"),
}


@galaxy
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.timeout(0)
def test_prefill_then_decode(mesh_device, case):
    prompt_len, handover, kv_format = CASES[case]
    golden = _golden(prompt_len)
    ids, steps = golden["input_ids"], golden["meta"]["steps"]
    user, users = 1, 2  # the prompt lives in the second slot of a two-user cache
    label = f"model {case}"
    numbers = {"dram_open": _gb(pb.dram(mesh_device))}

    bridge = pb.PrefillBridge(mesh_device, num_layers=LAYERS, num_users=users, max_seq_len=MAX_SEQ, kv_format=kv_format)
    try:
        # ---------------- prefill, hand-over, free the prefill weights
        bridge.build()
        numbers["dram_prefill_weights_and_caches"] = _gb(pb.dram(mesh_device))
        n = bridge.prefill(ids[:prompt_len], user=user)
        cache = _decode_cache(mesh_device, users)
        numbers["dram_with_decode_cache"] = _gb(pb.dram(mesh_device))
        t0 = time.perf_counter()
        n, took = bridge.hand_over(cache, user, on_device=handover == "device")
        numbers["handover_s"] = time.perf_counter() - t0
        assert ("host_latents" in took or "host" in took) == (case != "P5118-allondevice"), took
        logger.info(f"MEASURED {label} hand-over ({handover}): {numbers['handover_s']:.2f} s in all; {took}")
        bridge.release_model()
        numbers["dram_prefill_model_released"] = _gb(pb.dram(mesh_device))
        bridge.release_caches()
        numbers["dram_prefill_caches_released"] = _gb(pb.dram(mesh_device))
        numbers["prefill_s"] = {
            k: bridge.seconds.get(k) for k in ("build_runtime", "compile", "prefill", "release_model")
        }
        logger.info(f"MEASURED {label} prefill side: {numbers['prefill_s']}; DRAM allocated per chip, GB: {numbers}")

        # ---------------- decode weights
        model = _build_model(mesh_device, cache, numbers, label)
        numbers["dram_decode_weights_and_cache"] = _gb(pb.dram(mesh_device))

        # ---------------- T steps, teacher-forced with HF's tokens
        lows = []
        for j in range(steps):
            position = n - 1 + j
            record = []
            logits = model.decode_step([int(ids[position])], [position], users=[user], record=record)
            host_logits = model.lm_head.to_torch(logits)[0]
            token = int(model.tokens_to_torch(model.greedy(logits))[0])
            assert (
                token == int(host_logits.argmax()) or host_logits[token] == host_logits.max()
            ), f"{label} step {j}: greedy on the device gave {token}, the logits' argmax is {int(host_logits.argmax())}"
            lows.append(_compare_step(f"{label} step {j} P{position}", golden, j, record, host_logits, model=model))
        numbers["l1_small"] = pb.l1_small(mesh_device)
        failures = _judge_steps(label, lows)

        # ---------------- the rows the steps appended, against HF's cache rows of those positions
        last = n - 1 + steps
        for layer in range(LAYERS):
            cached = golden["cache"][layer]
            got = cache.to_torch(layer, user, last, LATENT)[n - 1 :]
            ref = frames.tail_to_device(cached["latent"][n - 1 : last].float(), ROPE)
            score = pcc(ref, got)
            logger.info(f"{label} L{layer} appended latent rows: pcc={score:.6f}")
            if score < PCC_APPENDED["latent"]:
                failures.append(f"{label} L{layer}: appended latent rows pcc={score:.6f}")
            if layer in FULL:
                got = cache.to_torch(layer, user, last, INDEX_KEY)[n - 1 :]
                ref = frames.index_to_device(cached["index_key"][n - 1 : last], ROPE)
                score = pcc(ref, got)
                logger.info(f"{label} L{layer} appended index-key rows: pcc={score:.6f}")
                if score < PCC_APPENDED["index_key"]:
                    failures.append(f"{label} L{layer}: appended index keys pcc={score:.6f}")

        # ---------------- again from the last prompt token, the device feeding its own tokens back
        generated = model.generate([int(ids[n - 1])], [n - 1], steps, users=[user])[:, 0]
        hf_tokens = ids[n : n + steps]
        agree = int((generated == hf_tokens).sum())
        logger.info(
            f"{label} free-running greedy: device {generated.tolist()} HF {hf_tokens.tolist()}: {agree} of {steps} equal"
        )

        # ---------------- time
        position = n - 1 + steps
        _measure_latency(model, mesh_device, label, [int(generated[-1])], [position], [user], context=position)
        logger.info(
            f"MEASURED {label} L1_SMALL per core after the steps: {numbers['l1_small']}; all numbers: {numbers}"
        )
        if agree < steps - 2:
            failures.append(f"{label}: the free-running device reproduced {agree} of HF's {steps} tokens")
        assert not failures, failures
    finally:
        bridge.close()


# The control of the tests above: the same first decode step with the prefilled rows taken away.
# A step that did not read the cache (or read it only where it does not matter) would score the
# same either way. Floors and ceilings set with the first run of this test; measured there:
# see the test's log lines "control ...".
CONTROL_MAX_ATTN_OUT = 0.9  # layer 0's attention output over an emptied cache, against HF's
CONTROL_MAX_LOGITS = 0.9


@galaxy
@pytest.mark.timeout(0)
def test_decode_needs_the_prefilled_rows(mesh_device):
    """Does the decode step read what the prefill stack wrote? Layers 0-5, the 1022-token prompt
    (every cached position is attended), the first decode step three times: over the rows the
    prefill stack wrote and the bridge handed over (must match HF), over the same cache with the
    rows of positions [0, N - 1) zeroed in every layer (must NOT match HF: the step then sees a
    context of one token), and over the rows put back (must give the first scores again)."""
    from models.experimental.hy4_preview.tt.model import WEIGHT_CACHE_ENV, Hy4Engine

    prompt_len = 1022
    golden = _golden(prompt_len)
    ids = golden["input_ids"]
    label = "control P1022"
    weight_cache = os.environ.get(WEIGHT_CACHE_ENV, "generated/hy4_preview/decode-cache/weights")
    engine = Hy4Engine(mesh_device, num_layers=LAYERS, max_context=MAX_SEQ, weight_cache=weight_cache)
    try:
        n = engine.prefill({0: ids[:prompt_len]})[0]
        model = engine.load()
        cache = engine.cache

        def first_step(what):
            engine.rewind(0, n, int(ids[n - 1]))
            record = []
            logits = model.lm_head.to_torch(engine.step(record=record))[0]
            layer0 = pcc(golden["layers"][0]["attn_out"][0].float(), record[0]["attn_out"][0])
            logger.info(f"{label} {what} L0 attn_out against HF: pcc={layer0:.6f}")
            low = _compare_step(f"{label} {what}", golden, 0, record, logits)
            return layer0, low

        real_l0, real = first_step("the prefilled rows")
        saved = {}
        for layer in range(LAYERS):
            key = cache.to_torch(layer, 0, n - 1, INDEX_KEY) if layer in FULL else None
            saved[layer] = (cache.to_torch(layer, 0, n - 1, LATENT), key)
            assert torch.count_nonzero(saved[layer][0]) > 0, f"L{layer}: the hand-over left no latent rows"
            cache.write_prefill_host(
                layer,
                0,
                latent=torch.zeros_like(saved[layer][0]),
                index_key=None if key is None else torch.zeros_like(key),
            )
        empty_l0, empty = first_step("the rows zeroed")
        for layer, (latent, key) in saved.items():
            cache.write_prefill_host(layer, 0, latent=latent, index_key=key)
        again_l0, again = first_step("the rows put back")
        logger.info(
            f"{label}: L0 attn_out / lowest attn_out / logits against HF: the prefilled rows {real_l0:.6f} / "
            f"{real['attn_out']:.6f} / {real['logits']:.6f} (next token {'same' if real['same_token'] else 'DIFFERENT'}); "
            f"the rows zeroed {empty_l0:.6f} / {empty['attn_out']:.6f} / {empty['logits']:.6f} (next token "
            f"{'same' if empty['same_token'] else 'DIFFERENT'}); the rows put back {again_l0:.6f} / "
            f"{again['attn_out']:.6f} / {again['logits']:.6f}"
        )
        assert real["attn_out"] >= PCC_STAGE["attn_out"] and real["logits"] >= PCC_LOGITS, real
        assert real["same_token"], f"{label}: the step over the prefilled rows does not give HF's token"
        assert (
            empty_l0 < CONTROL_MAX_ATTN_OUT
        ), f"{label}: L0 attn_out over an emptied cache still matches HF: {empty_l0}"
        assert empty["logits"] < CONTROL_MAX_LOGITS, f"{label}: logits over an emptied cache still match HF: {empty}"
        assert abs(again["attn_out"] - real["attn_out"]) < 1e-3 and abs(again["logits"] - real["logits"]) < 1e-3, (
            real,
            again,
        )
    finally:
        engine.close()


def _log_identical_users(label, record, logits, groups):
    """Users that hold the same prompt at the same position are the same computation in different
    rows of the batch and different user stripes of the cache. Logs, per group, the first recorded
    stage at which their rows stop being bit-equal and how far apart they are there and at the
    logits. Not asserted: it documents how position-dependent the device's arithmetic is."""
    for group in groups:
        first = None
        for layer in range(LAYERS):
            for key in RECORDED:
                rows = record[layer][key][group].float()
                spread = (rows - rows[0]).abs().max().item()
                if spread > 0 and first is None:
                    first = f"L{layer} {key}: max|Δ|={spread:.2e} on values up to {rows.abs().max().item():.2e}"
        rows = logits[group]
        spread = (rows - rows[0]).abs().max().item()
        logger.info(
            f"{label} users {group} (same prompt, same position): "
            + ("every recorded stage bit-equal" if first is None else f"first difference at {first}")
            + f"; logits max|Δ|={spread:.2e}, argmax {rows.argmax(-1).tolist()}"
        )


@galaxy
@pytest.mark.timeout(0)
def test_users_and_latency(mesh_device):
    """Eight users in one cache: users 0, 2, 4, 6 hold the 1022-token prompt and 1, 3, 5, 7 the
    5118-token one, so one batched step has users at different positions on different chips. The
    prefill and the decode weights are resident together (six layers of each fit), the caches go
    over on the device where they can, and every user's rows of every batched step are compared
    with its own golden. Then the time of a step for one user and for the batch of eight."""
    users = 8
    goldens = [_golden(1022), _golden(5118)]
    golden_of = [goldens[u % 2] for u in range(users)]
    lengths = [g["meta"]["prompt_len"] for g in golden_of]
    steps = 3
    label = "model 8 users"
    numbers = {"dram_open": _gb(pb.dram(mesh_device))}

    bridge = pb.PrefillBridge(mesh_device, num_layers=LAYERS, num_users=users, max_seq_len=MAX_SEQ)
    try:
        bridge.build()
        numbers["dram_prefill_weights_and_caches"] = _gb(pb.dram(mesh_device))
        cache = _decode_cache(mesh_device, users)
        numbers["dram_with_decode_cache"] = _gb(pb.dram(mesh_device))
        model = _build_model(mesh_device, cache, numbers, label)  # both weight sets on the device now
        numbers["dram_both_weight_sets"] = _gb(pb.dram(mesh_device))
        prefill_gb = numbers["dram_prefill_weights_and_caches"] - numbers["dram_open"]
        logger.info(
            f"MEASURED {label} both weight sets resident at {LAYERS} layers: prefill weights + its caches + compile "
            f"buffers {prefill_gb:.3f} GB per chip, decode weights {sum(numbers['decode_weights_gb'].values()):.3f} GB "
            f"per chip, all allocated {numbers['dram_both_weight_sets']:.3f} of {pb.dram(mesh_device)['total_gb']:.1f} GB"
        )
        for user in range(users):
            bridge.prefill(golden_of[user]["input_ids"][: lengths[user]], user=user)
        t0 = time.perf_counter()
        host = None
        for user in range(users):
            if host is None and not bridge.device_latents():
                host = bridge.read_host()
            bridge.hand_over(cache, user, host=host, on_device=True)
        ttnn.synchronize_device(mesh_device)
        numbers["handover_s"] = time.perf_counter() - t0
        logger.info(
            f"MEASURED {label} prefill of {users} users {[round(s, 2) for s in bridge.seconds['prefill']]} s; hand-over "
            f"of {users} users (index keys on the device, latents through the host) {numbers['handover_s']:.2f} s"
        )

        lows = []
        for j in range(steps):
            positions = [n - 1 + j for n in lengths]
            tokens = [int(golden_of[u]["input_ids"][positions[u]]) for u in range(users)]
            record = []
            logits = model.lm_head.to_torch(model.decode_step(tokens, positions, record=record))
            _log_identical_users(f"{label} step {j}", record, logits, [list(range(g, users, 2)) for g in (0, 1)])
            for user in range(users):
                what = f"{label} step {j} user {user} P{positions[user]}"
                lows.append(_compare_step(what, golden_of[user], j, record, logits[user], user_row=user, model=model))
        failures = _judge_steps(label, lows)

        # ---------------- time: one user, then the batch, at the positions after the checked steps
        positions = [n - 1 + steps for n in lengths]
        tokens = [int(golden_of[u]["input_ids"][positions[u]]) for u in range(users)]
        for batch in ([1], list(range(users))):  # user 1 holds the longer context
            where = [positions[u] for u in batch]
            context = where[0] if len(batch) == 1 else f"{min(where)} and {max(where)}"
            _measure_latency(model, mesh_device, label, [tokens[u] for u in batch], where, batch, context=context)
        numbers["l1_small"] = pb.l1_small(mesh_device)
        logger.info(f"MEASURED {label} L1_SMALL per core: {numbers['l1_small']}; all numbers: {numbers}")
        assert not failures, failures
    finally:
        bridge.close()
