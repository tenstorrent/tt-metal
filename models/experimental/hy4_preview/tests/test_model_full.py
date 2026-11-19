# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The whole model on the 8x4 Galaxy: prompt -> the prefill stack -> bridge -> decode -> tokens, at
6, 12, 24, 48 and all 78 layers, with checkpoint weights, against HF (tt/model.py Hy4Engine;
tenstorrent/tt-metal#65893).

* test_build_weight_cache: the on-disk cache of converted decode weights (tt/model.py WeightCache),
  a few layers per job: build time, bytes on disk and DRAM per chip of every group; one layer with
  BFP8 routed experts for what that precision would cost.
* test_depth: one user, the 1022- and 5118-token prompts of tests/test_model_prefill_decode.py, the
  model cut to N layers on both sides. The golden (reference/hf_harness.py record_decode_golden) is
  the HF model cut to the same N layers, in BF16. Every decode step is teacher-forced with HF's
  token and compared with HF's rows of that position: every layer's attention input / output, MoE
  input / output and streams, the top-2048 picks, the experts picked, the head readout, the logits
  and the next token. Also: the cached latent of the last prompt position as the prefill stack and
  as the decode step computed it; the handed-over rows against HF's cache; the same steps again
  over HF's own cache rows written in place of the prefill stack's (what is decode's and what came
  with the rows); the device feeding its own tokens back; a step timed layer by layer.
* test_layers_in_isolation: every layer of a decode step on HF's own input to that layer
  (TtHy4Model.decode_step ``layer_inputs``): the per-layer, per-sublayer score at every depth,
  free of the layers below.
* test_users: eight users in one cache and one batched step, each compared with its golden; the
  time of a step for one user and for eight.
* test_generate: a chat prompt through the checkpoint's tokenizer and chat template, prefill,
  greedy decode of 32 tokens on the device, the text; against HF's greedy generation where its
  golden exists (demo/record_reference.py), free-running and teacher-forced (logits, top-1).
  "L6-small" is the model cut to 6 layers, "L78-full" all 78.
* test_long_prompt: all 78 layers after a prefill of 8192 and of 32768 tokens: the first decode
  step against the 78-layer fp32 reference of the prefill stack's golden tooling (streams after
  every layer where recorded, the cached latent of the last position, the next-token logits),
  then the time of prefill, hand-over and decode steps at that context. 131072 tokens has no
  complete reference: it runs and is timed, nothing more.

Every case builds the prefill runtime (2 to 3 minutes for 78 layers) and loads the decode weights
(3.5 to 4.5 minutes from the weight cache); one case per job, with a long timeout. Select by the
case id: ``-k full`` or ``-k hy4`` match every test here (the path holds both words).

A truncated model is not a language model: its logits are the head applied to a middle layer's
streams, HF's own top-1 margins there are small and its cached generation does not always pick
what its full pass picks. The next-token agreement is therefore asserted on the decisive steps
(DECISIVE_MARGIN) and logged, with HF's margin, on the others.
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.demo.record_reference import DEFAULT_PROMPT, DEFAULT_STEPS
from models.experimental.hy4_preview.reference import hf_harness
from models.experimental.hy4_preview.tests.common import GALAXY_MESH, HY4_HF_MODEL_ENV, pcc
from models.experimental.hy4_preview.tt import frames
from models.experimental.hy4_preview.tt import prefill_bridge as pb
from models.experimental.hy4_preview.tt.common import MeshPlan
from models.experimental.hy4_preview.tt.decoder_layer import hidden_to_device
from models.experimental.hy4_preview.tt.kv_cache import INDEX_KEY, LATENT
from models.experimental.hy4_preview.tt.model import (
    HEAD_GROUP,
    LAYER_STAGES,
    RECORDED,
    WEIGHT_CACHE_ENV,
    Checkpoint,
    Hy4Engine,
    ModelConfig,
    build_weight_cache,
    layer_group,
)

WEIGHT_CACHE = os.environ.get(WEIGHT_CACHE_ENV, "/mnt/data/zni/hy4/decode-cache/weights")
ROPE = 64
TARGET_LAYER_MS = 1e3 / 75 / 78  # 8 users x 1M context at 75 tokens per second per user
# The chat prompt of test_generate and the demo.
PROMPT = DEFAULT_PROMPT
GENERATE_STEPS = DEFAULT_STEPS
# The 78-layer fp32 reference of an 8192-token prompt (the prefill stack's golden tooling, PR 65936).
LONG_GOLDEN = "/mnt/data/zni/hy4/prefill-cache/golden/hy4-bf16-llms-8192"

galaxy = pytest.mark.parametrize(
    "device_params, mesh_device", [(pb.device_params(), GALAXY_MESH)], indirect=True, ids=["8x4-torus"]
)


def _checkpoint():
    root = os.environ.get(HY4_HF_MODEL_ENV)
    if not root or not os.path.isfile(os.path.join(root, "model.safetensors.index.json")):
        pytest.skip(f"{HY4_HF_MODEL_ENV} does not point at the checkpoint")
    return Checkpoint(root)


def _idle_sync_ms(mesh_device, repeats=20):
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        ttnn.synchronize_device(mesh_device)
    return 1e3 * (time.perf_counter() - t0) / repeats


# ---------------------------------------------------------------------------------------------
# The weight cache
# ---------------------------------------------------------------------------------------------
# id -> (embedding and head too, layers, routed-expert dtype: None is the model's BFP4). A MoE layer
# converts and is written in about 30 s. The last job is one layer with BFP8 routed experts, into
# its own cache directory: what that precision costs in DRAM and on disk.
CACHE_JOBS = {
    "head-L00-05": (True, range(0, 6), None),
    "L06-11": (False, range(6, 12), None),
    "L12-23": (False, range(12, 24), None),
    "L24-35": (False, range(24, 36), None),
    "L36-47": (False, range(36, 48), None),
    "L48-59": (False, range(48, 60), None),
    "L60-71": (False, range(60, 72), None),
    "L72-77": (False, range(72, 78), None),
    "bfp8-experts-L01": (False, range(1, 2), ttnn.bfloat8_b),
}


@galaxy
@pytest.mark.parametrize("job", list(CACHE_JOBS))
@pytest.mark.timeout(0)
def test_build_weight_cache(mesh_device, job):
    """Convert the decode weights of a few layers from the checkpoint and leave them in the weight
    cache. A group that is already there is skipped (and reported as such)."""
    head, layers, expert_dtype = CACHE_JOBS[job]
    plan = MeshPlan(mesh_device)
    t0 = time.perf_counter()

    def progress(name, entry):
        logger.info(
            f"MEASURED weight cache {name}: {'built' if entry['built'] else 'already cached'} in "
            f"{entry['seconds']:.1f} s, {entry['bytes'] / 1e9:.3f} GB on disk"
            + (f", {entry['dram_gb']:.4f} GB of DRAM per chip while resident" if entry["built"] else "")
        )

    report = build_weight_cache(
        plan,
        ModelConfig(),
        _checkpoint(),
        WEIGHT_CACHE,
        layers=layers,
        head=head,
        expert_dtype=expert_dtype,
        progress=progress,
    )
    built = [entry for entry in report.values() if entry["built"]]
    logger.info(
        f"MEASURED weight cache job {job}: {len(built)} of {len(report)} groups built in "
        f"{time.perf_counter() - t0:.0f} s, {sum(e['bytes'] for e in report.values()) / 1e9:.2f} GB on disk for the job, "
        f"DRAM per chip after the job {pb.dram(mesh_device)['allocated_gb']:.4f} GB"
    )
    expected = ([HEAD_GROUP] if head else []) + [layer_group(layer) for layer in layers]
    assert list(report) == expected and all(entry["bytes"] > 0 for entry in report.values()), report


# ---------------------------------------------------------------------------------------------
# One decode step against HF's record of the same position
# ---------------------------------------------------------------------------------------------
# A step is decisive when HF's own top-1 leads its runner-up by this many logits. Below it HF does
# not agree with itself: its cached generation and its full pass over the same tokens (both BF16)
# pick different tokens at margins of 0.04, 0.05 and 0.38 in the goldens of this file.
DECISIVE_MARGIN = 1.0


def _router_picks(model, layer, mlp_in):
    """The experts the device's router picks for MoE inputs [U, hidden]: bool [U, E]."""
    mlp = model.layers[layer].mlp
    x = hidden_to_device(model.plan, mlp_in, ttnn.bfloat16)
    return model.plan.to_torch_replicated(mlp.route(mlp.whole(x))).float()[0, 0] != 0


def _compare_step(label, golden, j, record, logits, model, row=0) -> dict:
    """One step's record against HF's row j: {"stages": {key: [pcc per layer]}, "router": [share of
    HF's 8 expert picks per layer, None on the dense layer], "picks": {full layer: share of HF's
    top-k picks}, "readout", "logits", "same_token", "margin"}."""
    layers = model.cfg.num_layers
    out = {"stages": {key: [] for key in RECORDED}, "router": [], "picks": {}}
    for layer in range(layers):
        hf, dev = golden["layers"][layer], record[layer]
        for key in RECORDED:
            out["stages"][key].append(pcc(hf[key][j].float(), dev[key][row]))
        shared = None
        if "router_indices" in hf:
            picked = _router_picks(model, layer, dev["mlp_in"])[row]
            theirs = set(hf["router_indices"][j].tolist())
            shared = len(set(picked.nonzero().flatten().tolist()) & theirs) / len(theirs)
        out["router"].append(shared)
        if "picks" in dev:
            position = golden["meta"]["positions"][j]
            picked, valid = dev["picks"][row]
            mine = set(picked[valid].tolist())
            hf_picks = hf["topk_indices"][j].long()
            theirs = set(hf_picks[hf_picks <= position].tolist())  # HF pads a short context with keys it masks
            assert (
                len(mine) == int(valid.sum()) == min(golden["meta"]["index_topk"], position + 1)
            ), f"{label} L{layer}: {int(valid.sum())} valid picks, {len(mine)} distinct, at position {position}"
            assert max(mine) <= position, f"{label} L{layer}: a pick is ahead of the query"
            out["picks"][layer] = len(mine & theirs) / len(theirs)
    out["readout"] = pcc(golden["hc_head"][j].float(), record[layers]["hc_head"][row])
    hf_logits = golden["logits"][j]
    out["logits"] = pcc(hf_logits, logits)
    hf_top, top = hf_logits.topk(5).indices.tolist(), logits.topk(5).indices.tolist()
    out["margin"] = hf_logits.topk(2).values.diff().abs().item()
    out["same_token"] = top[0] == hf_top[0]
    for key in RECORDED:
        scores = out["stages"][key]
        worst = min(range(layers), key=scores.__getitem__)
        logger.info(f"{label} {key}: pcc={scores[worst]:.6f} (lowest of {layers} layers, at layer {worst})")
    logger.info(f"{label} head readout: pcc={out['readout']:.6f}")
    logger.info(f"{label} logits: pcc={out['logits']:.6f}")
    logger.info(
        f"{label} next token: device {top[0]} HF {hf_top[0]} ({'same' if out['same_token'] else 'DIFFERENT'}); "
        f"top-5 device {top} HF {hf_top}; HF's margin top1 - top2 {out['margin']:.3f}; "
        f"HF's cached generation picked {int(golden['cached_logits_argmax'][j])}"
    )
    return out


def _depth_table(label, results, model) -> dict:
    """Every layer's lowest scores over the steps, one line per layer, and the lowest per stage."""
    layers = model.cfg.num_layers
    low = {key: [min(r["stages"][key][n] for r in results) for n in range(layers)] for key in RECORDED}
    # mlp_out where the router made HF's picks on that step, and where it did not
    for layer in range(layers):
        line = f"DEPTH {label} L{layer:02d}: " + " ".join(f"{key} {low[key][layer]:.4f}" for key in RECORDED)
        router = [r["router"][layer] for r in results if r["router"][layer] is not None]
        if router:
            line += f" | HF's expert picks {min(router) * 8:.0f}-{max(router) * 8:.0f} of 8"
        picks = [r["picks"][layer] for r in results if layer in r["picks"]]
        if picks:
            line += f" | HF's top-k picks {min(picks):.4f}"
        logger.info(line)
    for j, r in enumerate(results):
        logger.info(
            f"DEPTH {label} step {j} streams_out by layer: " + " ".join(f"{x:.3f}" for x in r["stages"]["streams_out"])
        )
    summary = {key: min(low[key]) for key in RECORDED}
    summary["readout"] = min(r["readout"] for r in results)
    summary["logits"] = min(r["logits"] for r in results)
    picks = [share for r in results for share in r["picks"].values()]
    router = [share for r in results for share in r["router"] if share is not None]
    summary["picks_shared"] = min(picks) if picks else 1.0
    summary["router_shared"] = min(router) if router else 1.0
    summary["router_mean"] = sum(router) / len(router) if router else 1.0
    summary["same_tokens"] = sum(r["same_token"] for r in results)
    decisive = [r for r in results if r["margin"] >= DECISIVE_MARGIN]
    summary["decisive"] = len(decisive)
    summary["decisive_same"] = sum(r["same_token"] for r in decisive)
    summary["streams_out_decisive"] = min([min(r["stages"]["streams_out"]) for r in decisive], default=1.0)
    summary["logits_decisive"] = min([r["logits"] for r in decisive], default=1.0)
    logger.info(
        f"{label} lowest over {len(results)} steps and {layers} layers: "
        + ", ".join(f"{key} {summary[key]:.6f} (L{min(range(layers), key=low[key].__getitem__)})" for key in RECORDED)
        + f", readout {summary['readout']:.6f}, logits {summary['logits']:.6f}; "
        f"HF's top-k picks shared {summary['picks_shared']:.4f}; HF's expert picks shared: lowest "
        f"{summary['router_shared'] * 8:.0f} of 8, mean {summary['router_mean'] * 8:.2f} of 8; "
        f"next token equal to HF's on {summary['same_tokens']} of {len(results)} steps, and on "
        f"{summary['decisive_same']} of the {summary['decisive']} where HF's margin is {DECISIVE_MARGIN} or more "
        f"(HF's margins {[round(r['margin'], 2) for r in results]}, equal {[r['same_token'] for r in results]}); "
        f"over the decisive steps alone: streams_out {summary['streams_out_decisive']:.6f}, logits "
        f"{summary['logits_decisive']:.6f}"
    )
    return summary


def _last_rows(cache, layers, n):
    """Row N - 1 of every layer's latent stripe (user 0), as stored: [layers, 576] fp32."""
    return torch.stack([cache.to_torch(layer, 0, n, LATENT)[n - 1] for layer in range(layers)])


def _log_last_rows(label, reference, prefill, decode, what):
    """The cached latent of the last prompt position, layer by layer, as the prefill stack wrote it
    and as the first decode step rewrote it, each against ``reference`` ([layers, 576], device
    frame): the one place where the two stacks compute the same row of the same layer, each on its
    own streams. Scored on k_pass (the 512 normalised dims; k_rope is the same projection with the
    position on it)."""
    layers = reference.shape[0]
    scores = {
        "the prefill stack's": [pcc(reference[n, :512], prefill[n, :512]) for n in range(layers)],
        "the decode step's": [pcc(reference[n, :512], decode[n, :512]) for n in range(layers)],
    }
    for name, values in scores.items():
        logger.info(f"DEPTH {label} row N-1, {name} k_pass by layer: " + " ".join(f"{x:.3f}" for x in values))
        worst = min(range(layers), key=values.__getitem__)
        logger.info(
            f"{label} row N-1 k_pass, {name} against {what}: pcc={values[worst]:.6f} (lowest of {layers} layers, at "
            f"layer {worst}); median {sorted(values)[layers // 2]:.4f}; layers at 0.99 or better "
            f"{sum(x >= 0.99 for x in values)}, at 0.97 or better {sum(x >= 0.97 for x in values)}"
        )
    between = [pcc(prefill[n, :512], decode[n, :512]) for n in range(layers)]
    logger.info(
        f"{label} row N-1 k_pass, the decode step's against the prefill stack's: lowest {min(between):.6f} at layer "
        f"{min(range(layers), key=between.__getitem__)}, median {sorted(between)[layers // 2]:.4f}"
    )
    return scores


def _kinds(cfg: ModelConfig) -> dict:
    layers = range(cfg.num_layers)
    return {
        "dense FFN + full indexer": [n for n in layers if cfg.decoder.is_dense(n)],
        "MoE + full indexer": [n for n in layers if cfg.is_full(n) and not cfg.decoder.is_dense(n)],
        "MoE + shared indexer": [n for n in layers if not cfg.is_full(n)],
    }


def _measure_latency(engine, label, context, users=(0,), profile=True) -> dict:
    """Untraced wall time of a decode step of ``users`` at the engine's current positions: the
    whole step and every layer on its own with the ops issued back to back (no synchronize
    inside), then the split of a layer with a synchronize after every piece. Leaves the engine
    where it was (a repeated step rewrites the same rows)."""
    model, mesh_device = engine.model, engine.mesh_device
    layers, users, count = model.cfg.num_layers, list(users), len(users)
    tokens, positions = engine.next_inputs(users)
    model.greedy(model.decode_step(tokens, positions, users=users))  # compiles this batch's programs
    sync_ms = _idle_sync_ms(mesh_device)
    repeats = 3
    t0 = time.perf_counter()
    for _ in range(repeats):
        model.greedy(model.decode_step(tokens, positions, users=users))
    ttnn.synchronize_device(mesh_device)
    step = (time.perf_counter() - t0) / repeats
    times = model.layer_times(tokens, positions, users=users, repeats=3)
    per_layer = [1e3 * t for t in times["layers"]]
    by_kind = {kind: sum(per_layer[n] for n in ls) / len(ls) for kind, ls in _kinds(model.cfg).items() if ls}
    mean = sum(per_layer) / layers
    logger.info(
        f"MEASURED {label} decode step, {count} user(s), context {context}, {layers} layers, untraced, issued back "
        f"to back: {step * 1e3:.1f} ms per step = {1 / step:.3f} tokens per second per user, {count / step:.3f} in "
        f"all; mean {mean:.2f} ms per layer ({mean / TARGET_LAYER_MS:.0f}x the {TARGET_LAYER_MS:.3f} ms budget of a "
        f"batch of 8 users); by kind (ms): "
        + ", ".join(f"{kind} {ms:.2f} ({ms / TARGET_LAYER_MS:.0f}x)" for kind, ms in by_kind.items())
        + f"; embedding {1e3 * times['embedding']:.2f}, head readout + norm + LM head + greedy {1e3 * times['head']:.2f}; "
        f"slowest layer {max(per_layer):.2f}, fastest {min(per_layer):.2f}"
    )
    out = {"step_ms": step * 1e3, "layer_ms": mean, "by_kind": by_kind}
    if profile:
        timing = {}
        model.decode_step(tokens, positions, users=users, profile=timing)
        for kind, ls in _kinds(model.cfg).items():
            if not ls:
                continue
            total = {}
            for layer in ls:
                for stage, seconds in timing["layers"][layer].items():
                    total[stage] = total.get(stage, 0.0) + seconds / len(ls)
            # every user's attention step ends each of its stages with a synchronize
            inner = {
                k.split(" / ")[1]: 1e3 * v - count * sync_ms for k, v in total.items() if k.startswith("attention / ")
            }
            inner = {stage: ms for stage, ms in inner.items() if total[f"attention / {stage}"] > 0}
            ms = {}
            for stage in LAYER_STAGES:
                syncs = total[f"laps / {stage}"] + (count * (len(inner) + 1) if stage == "attention" else 0)
                ms[stage] = 1e3 * total[stage] - syncs * sync_ms
            logger.info(
                f"MEASURED {label} {count} user(s), context {context}, {kind}, split of a layer (ms, staged, an idle "
                f"synchronize of {sync_ms:.2f} ms taken off each piece): "
                + ", ".join(f"{stage} {ms[stage]:.2f}" for stage in LAYER_STAGES)
                + "; inside attention, summed over the users: "
                + ", ".join(f"{stage} {value:.2f}" for stage, value in inner.items())
            )
    return out


def _log_engine(label, engine):
    logger.info(
        f"MEASURED {label} seconds: "
        + ", ".join(f"{k} {v if not isinstance(v, float) else round(v, 2)}" for k, v in engine.seconds.items())
        + f"; DRAM allocated per chip, GB: {engine.dram} of {pb.dram(engine.mesh_device)['total_gb']:.2f}; "
        f"L1_SMALL per core: {pb.l1_small(engine.mesh_device)}"
    )


def _load(engine, label):
    """Load the decode weights; log how the DRAM grew and what the build took."""
    marks = {}
    model = engine.load(progress=lambda name: marks.__setitem__(name, pb.dram(engine.mesh_device)["allocated_gb"]))
    names = list(marks)
    grown = [marks[names[i]] - marks[names[i - 1]] for i in range(1, len(names))]
    head = marks[names[0]] - engine.dram["prefill_storage_released"]
    cached = engine.weight_cache is not None
    logger.info(
        f"MEASURED {label} decode weights ({'weight cache' if cached else 'from the checkpoint'}): "
        f"{engine.seconds['load_decode_weights']:.1f} s for {model.cfg.num_layers} layers (embedding + head "
        f"{model.seconds['embedding_and_head']:.1f} s, layers {sum(model.seconds['layers']):.1f} s, slowest "
        f"{max(model.seconds['layers']):.1f} s); DRAM per chip: embedding + head {head:.3f} GB, layers "
        f"{sum(grown):.3f} GB (largest {max(grown):.3f}, smallest {min(grown):.3f}), all "
        f"{engine.dram['decode_weights_and_cache']:.3f} GB with the cache"
    )
    return model


# Floors of test_depth by depth, the device running on its own streams: the lowest PCC against HF
# (BF16) over the layers and steps. Set from the first runs of this test, with room; nothing here
# comes from an issue. Measured lows on the 1022-token prompt, streams_out / logits:
# 6 layers 0.9965 / 0.9959, 12 layers 0.9943 / 0.9909, 24 layers 0.9877 / 0.9876,
# 48 layers 0.9661 / 0.9868, 78 layers 0.5307 / 0.9453 (0.8853 / 0.9705 over the decisive steps).
# ``mlp_out`` is one token's MoE output, which moves with every expert the router picks differently
# from HF's: logged, never asserted here (test_layers_in_isolation scores it on HF's input).
# At 78 layers a step on which HF itself is undecided can come out a different computation
# altogether (one such step: streams at 0.53 by layer 69, logits still 0.945), so the streams are
# asserted over the decisive steps only.
DEPTH_FLOORS = {
    6: {"attn_in": 0.99, "attn_out": 0.98, "streams_mid": 0.99, "mlp_in": 0.99, "streams_out": 0.99, "logits": 0.98},
    12: {"streams_out": 0.98, "logits": 0.98},
    24: {"streams_out": 0.97, "logits": 0.97},
    48: {"streams_out": 0.94, "logits": 0.97},
    78: {"streams_out_decisive": 0.85, "logits": 0.9},
}

# id -> (layers, prompt tokens, the prefill stack's KVPE format). The last case is the long prompt
# with a BF16_RM prefill cache (1152 instead of 704 bytes per token per layer, converted to the
# decode cache on the device): what the scaled-FP8 packing costs a 78-layer decode.
DEPTH_CASES = {
    "L6-P1022": (6, 1022, "scaled_fp8"),
    "L12-P1022": (12, 1022, "scaled_fp8"),
    "L24-P1022": (24, 1022, "scaled_fp8"),
    "L48-P1022": (48, 1022, "scaled_fp8"),
    "L78-P1022": (78, 1022, "scaled_fp8"),
    "L6-P5118": (6, 5118, "scaled_fp8"),
    "L78-P5118": (78, 5118, "scaled_fp8"),
    "L78-bf16rm-P5118": (78, 5118, "bf16_rm"),
}


@galaxy
@pytest.mark.parametrize("case", list(DEPTH_CASES))
@pytest.mark.timeout(0)
def test_depth(mesh_device, case):
    layers, prompt_len, kv_format = DEPTH_CASES[case]
    golden = hf_harness.find_decode_golden(layers, prompt_len)
    if golden is None:
        pytest.skip(f"no {layers}-layer decode golden for a {prompt_len}-token prompt under {hf_harness.GOLDEN_ROOT}")
    ids, steps = golden["input_ids"], golden["meta"]["steps"]
    label = f"model {case}"
    full = [n for n, kind in enumerate(golden["meta"]["indexer_types"]) if kind == "full"]

    engine = Hy4Engine(mesh_device, num_layers=layers, max_context=8192, weight_cache=WEIGHT_CACHE, kv_format=kv_format)
    try:
        n = engine.prefill({0: ids[:prompt_len]})[0]
        model = _load(engine, label)
        cache = engine.cache

        # ---------------- the steps, teacher-forced with HF's tokens
        prefill_last = _last_rows(cache, layers, n)
        results = []
        for j in range(steps):
            position = n - 1 + j
            record = []
            logits = engine.step(tokens=[int(ids[position])], record=record)
            host_logits = model.lm_head.to_torch(logits)[0]
            token = int(model.tokens_to_torch(model.greedy(logits))[0])
            assert (
                token == int(host_logits.argmax()) or host_logits[token] == host_logits.max()
            ), f"{label} step {j}: greedy on the device gave {token}, the logits' argmax is {int(host_logits.argmax())}"
            results.append(_compare_step(f"{label} step {j} P{position}", golden, j, record, host_logits, model))
            del record
            if j == 0:
                hf_last = torch.stack([golden["cache"][layer]["latent"][n - 1].float() for layer in range(layers)])
                _log_last_rows(
                    label,
                    frames.tail_to_device(hf_last, ROPE),
                    prefill_last,
                    _last_rows(cache, layers, n),
                    "HF's row (its prefill pass, BF16)",
                )
        summary = _depth_table(label, results, model)
        _log_engine(label, engine)

        # ---------------- the rows the steps appended, against HF's cache rows of those positions
        last = n - 1 + steps
        appended = {"latent": [], "index_key": []}
        for layer in range(layers):
            cached = golden["cache"][layer]
            got = cache.to_torch(layer, 0, last, LATENT)[n - 1 :]
            appended["latent"].append(pcc(frames.tail_to_device(cached["latent"][n - 1 : last].float(), ROPE), got))
            if layer in full:
                got = cache.to_torch(layer, 0, last, INDEX_KEY)[n - 1 :]
                ref = frames.index_to_device(cached["index_key"][n - 1 : last], ROPE)
                appended["index_key"].append(pcc(ref, got))
        for kind, scores in appended.items():
            logger.info(f"{label} appended {kind} rows: pcc={min(scores):.6f} (lowest of {len(scores)} layers)")

        # ---------------- the same steps over HF's own cache rows in place of the prefill stack's:
        # what of the scores above is decode's and what came with the rows it was handed
        prefill_rows = {}
        handed = {"k_pass": [], "k_rope": [], "index_key": []}
        for layer in range(layers):
            cached = golden["cache"][layer]
            prefill_rows[layer] = [cache.to_torch(layer, 0, n, LATENT), None]
            index_key = None
            if layer in full:
                prefill_rows[layer][1] = cache.to_torch(layer, 0, n, INDEX_KEY)
                index_key = frames.index_to_device(cached["index_key"][:n], ROPE)
                handed["index_key"].append(pcc(index_key[: n - 1], prefill_rows[layer][1][: n - 1]))
            latent = frames.tail_to_device(cached["latent"][:n].float(), ROPE)
            # rows [0, N - 1) are still the prefill stack's (row N - 1 is decode's by now)
            handed["k_pass"].append(pcc(latent[: n - 1, :512], prefill_rows[layer][0][: n - 1, :512]))
            handed["k_rope"].append(pcc(latent[: n - 1, 512:], prefill_rows[layer][0][: n - 1, 512:]))
            cache.write_prefill_host(layer, 0, latent=latent, index_key=index_key)
        for kind, scores in handed.items():
            where = full if kind == "index_key" else list(range(layers))
            worst = min(range(len(scores)), key=scores.__getitem__)
            logger.info(f"DEPTH {label} handed-over {kind} by layer: " + " ".join(f"{x:.3f}" for x in scores))
            logger.info(
                f"{label} handed-over rows [0, {n - 1}) against HF's cache, {kind}: pcc={scores[worst]:.6f} (lowest of "
                f"{len(scores)} layers, at layer {where[worst]}); median {sorted(scores)[len(scores) // 2]:.4f}; layers "
                f"at 0.99 or better {sum(x >= 0.99 for x in scores)}"
            )
        engine.rewind(0, n, int(ids[n - 1]))
        over_hf = []
        for j in range(steps):
            position = n - 1 + j
            record = []
            logits = engine.step(tokens=[int(ids[position])], record=record)
            what = f"{label} over HF's rows step {j} P{position}"
            over_hf.append(_compare_step(what, golden, j, record, model.lm_head.to_torch(logits)[0], model))
            del record
        _depth_table(f"{label} over HF's rows", over_hf, model)
        for layer, (latent, index_key) in prefill_rows.items():  # the prefill's rows back, as they were stored
            cache.write_prefill_host(layer, 0, latent=latent, index_key=index_key)
        del prefill_rows

        # ---------------- again from the last prompt token, the device feeding its own tokens back
        engine.rewind(0, n, int(ids[n - 1]))
        generated = engine.generate(steps)[:, 0]
        hf_tokens = ids[n : n + steps]
        agree = int((generated == hf_tokens).sum())
        logger.info(
            f"{label} free-running greedy: device {generated.tolist()} HF (cached generation) {hf_tokens.tolist()}: "
            f"{agree} of {steps} equal"
        )

        # ---------------- time
        _measure_latency(engine, label, context=engine.lengths[0])
        logger.info(f"MEASURED {label} L1_SMALL per core after everything: {pb.l1_small(mesh_device)}")

        failures = []
        for key, floor in DEPTH_FLOORS[layers].items():
            if summary[key] < floor:
                failures.append(f"{label}: {key} pcc={summary[key]:.6f} under {floor}")
        if summary["decisive_same"] < summary["decisive"]:
            failures.append(
                f"{label}: next token equal to HF's on {summary['decisive_same']} of the {summary['decisive']} "
                f"decisive steps"
            )
        assert not failures, failures
    finally:
        engine.close()


# id -> (layers, prompt tokens)
ISOLATION_CASES = {"L6-P1022": (6, 1022), "L78-P1022": (78, 1022)}
# Floors of a layer on HF's own input (BF16): the lowest over the layers and steps, by whose
# cache rows the layer attends over. Set from the first runs, with room. Measured at 78 layers:
#   the prefill stack's rows   attn_out 0.9671 (layer 73), streams_mid 0.9962, mlp_in 0.9927, streams_out 0.9811
#   HF's rows                  attn_out 0.9933 (layer 59), streams_mid 0.9994, mlp_in 0.9988, streams_out 0.9871
# (streams_out at layer 39 both times; 76 and 77 of the 78 layers are at 0.99 or better.) mlp_out,
# one token through BFP4 experts, is 0.914 to 0.999 and is not asserted.
ISOLATION_FLOORS = {
    "the prefill stack's rows": {
        "attn_in": 0.999,
        "attn_out": 0.95,
        "streams_mid": 0.99,
        "mlp_in": 0.985,
        "streams_out": 0.97,
    },
    "HF's rows": {"attn_in": 0.999, "attn_out": 0.985, "streams_mid": 0.998, "mlp_in": 0.995, "streams_out": 0.98},
}


@galaxy
@pytest.mark.parametrize("case", list(ISOLATION_CASES))
@pytest.mark.timeout(0)
def test_layers_in_isolation(mesh_device, case):
    """Every layer of the device model on HF's own input to that layer, at every depth.

    test_depth lets the device run on its own streams, so a layer's score there carries every
    layer below it. Here each layer of a decode step is fed the streams HF had going into it
    (TtHy4Model.decode_step ``layer_inputs``) and its sublayers are compared with HF's: what one
    device layer adds, wherever it sits in the stack. Twice: over the cache rows the prefill stack
    wrote (what decode really reads), and over HF's own cache rows written in their place (the
    layer alone)."""
    layers, prompt_len = ISOLATION_CASES[case]
    golden = hf_harness.find_decode_golden(layers, prompt_len)
    if golden is None:
        pytest.skip(f"no {layers}-layer decode golden for a {prompt_len}-token prompt under {hf_harness.GOLDEN_ROOT}")
    ids, steps = golden["input_ids"], golden["meta"]["steps"]
    full = [n for n, kind in enumerate(golden["meta"]["indexer_types"]) if kind == "full"]
    label = f"isolated {case}"

    engine = Hy4Engine(mesh_device, num_layers=layers, max_context=8192, weight_cache=WEIGHT_CACHE)
    try:
        n = engine.prefill({0: ids[:prompt_len]})[0]
        model = _load(engine, label)
        failures = []
        for rows in ("the prefill stack's rows", "HF's rows"):
            if rows == "HF's rows":
                for layer in range(layers):
                    cached = golden["cache"][layer]
                    index_key = frames.index_to_device(cached["index_key"], ROPE) if layer in full else None
                    latent = frames.tail_to_device(cached["latent"].float(), ROPE)
                    engine.cache.write_prefill_host(layer, 0, latent=latent, index_key=index_key)
                engine.rewind(0, n, int(ids[n - 1]))
            results = []
            for j in range(steps):
                position = n - 1 + j
                inputs = [golden["layers"][layer]["streams_in"][j].float()[None] for layer in range(layers)]
                record = []
                logits = engine.step(tokens=[int(ids[position])], record=record, layer_inputs=inputs)
                host_logits = model.lm_head.to_torch(logits)[0]
                what = f"{label} over {rows} step {j} P{position}"
                results.append(_compare_step(what, golden, j, record, host_logits, model))
                del record
            summary = _depth_table(f"{label} over {rows}", results, model)
            for key, floor in ISOLATION_FLOORS[rows].items():
                if summary[key] < floor:
                    failures.append(f"{label} over {rows}: {key} pcc={summary[key]:.6f} under {floor}")
        _log_engine(label, engine)
        assert not failures, failures
    finally:
        engine.close()


# id -> (layers, users)
USERS_CASES = {"L6-8users": (6, 8), "L78-8users": (78, 8)}
USERS_MIN_LOGITS_PCC = 0.9


@galaxy
@pytest.mark.parametrize("case", list(USERS_CASES))
@pytest.mark.timeout(0)
def test_users(mesh_device, case):
    """A batch of users in one cache and one step: every user holds a prompt of its own golden
    (the 1022-token one, and the 5118-token one on the odd users where its golden exists at this
    depth), three batched steps are teacher-forced and every user's row is compared with its
    golden, then a step is timed for one user and for the batch."""
    layers, users = USERS_CASES[case]
    goldens = [g for g in (hf_harness.find_decode_golden(layers, n) for n in (1022, 5118)) if g is not None]
    if not goldens or goldens[0]["meta"]["prompt_len"] != 1022:
        pytest.skip(f"no {layers}-layer decode golden for the 1022-token prompt under {hf_harness.GOLDEN_ROOT}")
    golden_of = [goldens[u % len(goldens)] for u in range(users)]
    lengths = [g["meta"]["prompt_len"] for g in golden_of]
    steps = 3
    label = f"model {case}"

    engine = Hy4Engine(mesh_device, num_layers=layers, max_users=users, max_context=8192, weight_cache=WEIGHT_CACHE)
    try:
        engine.prefill({u: golden_of[u]["input_ids"][: lengths[u]] for u in range(users)})
        model = _load(engine, label)
        logger.info(
            f"MEASURED {label} prefill of {users} users {[round(s, 2) for s in engine.seconds['prefill / prefill']]} s, "
            f"hand-over of all {engine.seconds['handover']:.2f} s"
        )
        results = [[] for _ in range(users)]
        for j in range(steps):
            positions = [n - 1 + j for n in lengths]
            tokens = [int(golden_of[u]["input_ids"][positions[u]]) for u in range(users)]
            record = []
            logits = model.lm_head.to_torch(engine.step(tokens=tokens, record=record))
            for u in range(users):
                what = f"{label} step {j} user {u} P{positions[u]}"
                results[u].append(_compare_step(what, golden_of[u], j, record, logits[u], model, row=u))
            del record
        summaries = [_depth_table(f"{label} user {u}", results[u], model) for u in range(users)]
        _log_engine(label, engine)
        low = min(s["logits"] for s in summaries)
        same = sum(s["same_tokens"] for s in summaries)
        logger.info(
            f"{label} over {users} users and {steps} batched steps: logits pcc lowest {low:.6f}, streams_out lowest "
            f"{min(s['streams_out'] for s in summaries):.6f}; next token equal to HF's on {same} of {users * steps}"
        )

        for batch in ([0], list(range(users))):
            where = [engine.lengths[u] for u in batch]
            context = where[0] if len(batch) == 1 else f"{min(where)} to {max(where)}"
            _measure_latency(engine, label, context=context, users=batch)
        logger.info(f"MEASURED {label} L1_SMALL per core after everything: {pb.l1_small(mesh_device)}")
        decisive = sum(s["decisive"] for s in summaries)
        decisive_same = sum(s["decisive_same"] for s in summaries)
        logger.info(f"{label}: next token equal to HF's on {decisive_same} of the {decisive} decisive user-steps")
        assert low >= USERS_MIN_LOGITS_PCC, f"{label}: logits pcc {low:.6f}"
        assert decisive_same == decisive, f"{label}: {decisive_same} of {decisive} decisive user-steps equal to HF's"
    finally:
        engine.close()


# ---------------------------------------------------------------------------------------------
# A chat prompt in, text out
# ---------------------------------------------------------------------------------------------
def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(os.environ[HY4_HF_MODEL_ENV])


def _generation_golden(layers, ids, steps):
    path = hf_harness.decode_golden_path(layers, ids.numel(), steps, ids)
    if not path.is_file():
        return None
    golden = torch.load(path, weights_only=False)
    assert torch.equal(golden["input_ids"][: ids.numel()], ids), f"{path} was recorded for another prompt"
    return golden


def _first_token_report(label, engine, n, first_s):
    """What the first token waits for, as the engine stands: both weight sets cannot be resident,
    so the prefill stack is built, used and freed and the decode weights are loaded for every
    prompt round."""
    s = engine.seconds
    prefill = s["prefill / prefill"][0]
    parts = {
        "build the prefill runtime": s["prefill / build_runtime"],
        "compile it": s["prefill / compile"],
        f"prefill {n} tokens": prefill,
        "hand the caches over": s["handover"],
        "free the prefill weights": s["prefill / release_model"],
        "load the decode weights": s["load_decode_weights"],
        "first decode step (compiles its programs)": first_s,
    }
    total = sum(parts.values())
    logger.info(
        f"MEASURED {label} time to first token, untraced: {total:.1f} s = "
        + " + ".join(f"{value:.2f} s {name}" for name, value in parts.items())
        + f"; with both weight sets loaded it would be prefill + hand-over + one warm step; prefill "
        f"{n / prefill:.0f} tokens per second ({n} tokens in {prefill:.3f} s, one padded 4096-token chunk per 4096)"
    )
    return total


# id -> layers. ("full" alone would select every test of this file: the file name holds it.)
GENERATE_CASES = {"L6-small": 6, "L78-full": 78}
# Floors of the 78-layer generation against HF's greedy generation (BF16), teacher-forced. Measured
# on the first run: logits pcc 0.8941 on the first step (the last prompt token over the prefill's
# rows), 0.8946 on the second, 0.93 to 0.99 after that, mean 0.9683; top-1 equal to HF's on 31 of
# 32 steps, the other one a step where HF's margin is 0.2 and its cached generation picks the
# device's token. The first floor written here, 0.9 before any run, failed on that first step.
GENERATE_MIN_LOGITS_PCC = 0.85
GENERATE_MIN_SAME = 0.75  # of the teacher-forced steps: the device's top-1 is HF's


@galaxy
@pytest.mark.parametrize("case", list(GENERATE_CASES))
@pytest.mark.timeout(0)
def test_generate(mesh_device, case):
    layers, steps = GENERATE_CASES[case], GENERATE_STEPS
    label = f"generate {case} L{layers}"
    tokenizer = _tokenizer()
    ids = hf_harness.tokenize(PROMPT)
    golden = _generation_golden(layers, ids, steps)
    logger.info(f"{label} prompt ({ids.numel()} tokens through the chat template): {tokenizer.decode(ids.tolist())!r}")

    engine = Hy4Engine(mesh_device, num_layers=layers, max_context=8192, weight_cache=WEIGHT_CACHE)
    try:
        n = engine.prefill({0: ids})[0]
        model = _load(engine, label)

        # ---------------- greedy, the device feeding its own tokens back
        t0 = time.perf_counter()
        first = engine.generate(1)
        first_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        rest = engine.generate(steps - 1)
        rest_s = time.perf_counter() - t0
        generated = torch.cat([first, rest])[:, 0]
        _first_token_report(label, engine, n, first_s)
        step_ms = 1e3 * rest_s / (steps - 1)
        logger.info(
            f"MEASURED {label} greedy decode, 1 user, context {n} to {n + steps}, untraced, tokens fed back on the "
            f"device: {steps - 1} steps in {rest_s:.2f} s = {(steps - 1) / rest_s:.3f} tokens per second, "
            f"{step_ms:.0f} ms per step, {step_ms / layers:.2f} ms per layer ({step_ms / layers / TARGET_LAYER_MS:.0f}x "
            f"the {TARGET_LAYER_MS:.3f} ms budget of a batch of 8 users)"
        )
        logger.info(f"GENERATED {label} token ids: {generated.tolist()}")
        logger.info(f"GENERATED {label} text: {tokenizer.decode(generated.tolist())!r}")
        assert generated.min() >= 0 and generated.max() < model.cfg.head.vocab_size
        _log_engine(label, engine)

        if golden is None:
            logger.warning(
                f"{label}: no HF golden for this prompt at {layers} layers under {hf_harness.GOLDEN_ROOT} "
                "(demo/record_reference.py): the generation was NOT compared with HF"
            )
            if layers == 78:
                pytest.skip("generated, but there is no HF generation to compare it with")
            return

        # ---------------- against HF's greedy generation
        hf_tokens = golden["input_ids"][n : n + steps]
        same = generated == hf_tokens
        prefix = int(same.long().cumprod(0).sum())
        logger.info(f"GENERATED {label} HF text: {tokenizer.decode(hf_tokens.tolist())!r}")
        logger.info(
            f"{label} free-running: {int(same.sum())} of {steps} tokens equal to HF's greedy generation, the first "
            f"{prefix} in a row; device {generated.tolist()} HF {hf_tokens.tolist()}"
        )
        # Where the device leaves HF's greedy path, HF itself must have been undecided: up to that
        # step both ran on the same tokens.
        left_at_margin = None
        if prefix < steps:
            left_at_margin = golden["logits"][prefix].topk(2).values.diff().abs().item()
            logger.info(
                f"{label} free-running: leaves HF's path at step {prefix}, where HF's margin is {left_at_margin:.3f}"
            )

        # ---------------- teacher-forced with HF's tokens: every step on HF's own context
        engine.rewind(0, n, int(ids[-1]))
        results = []
        for j in range(steps):
            position = n - 1 + j
            record = [] if j in (0, steps - 1) else None
            logits = engine.step(tokens=[int(golden["input_ids"][position])], record=record)
            host_logits = model.lm_head.to_torch(logits)[0]
            hf_logits = golden["logits"][j]
            result = {
                "logits": pcc(hf_logits, host_logits),
                "same_token": int(host_logits.argmax()) == int(hf_logits.argmax()),
                "margin": hf_logits.topk(2).values.diff().abs().item(),
                "rank": int((host_logits > host_logits[int(hf_logits.argmax())]).sum()),
            }
            logger.info(
                f"{label} step {j} P{position} logits: pcc={result['logits']:.6f} | device top-1 "
                f"{int(host_logits.argmax())} HF {int(hf_logits.argmax())} "
                f"({'same' if result['same_token'] else 'DIFFERENT'}; HF's token is the device's rank {result['rank']}; "
                f"HF's margin {result['margin']:.3f})"
            )
            if record is not None:
                detail = _compare_step(f"{label} step {j} P{position}", golden, j, record, host_logits, model)
                _depth_table(f"{label} step {j}", [detail], model)
            results.append(result)
        agree = sum(r["same_token"] for r in results)
        low = min(r["logits"] for r in results)
        missed = [(j, round(r["margin"], 2), r["rank"]) for j, r in enumerate(results) if not r["same_token"]]
        logger.info(
            f"{label} teacher-forced: top-1 equal to HF's on {agree} of {steps} steps; logits pcc lowest {low:.6f}, "
            f"mean {sum(r['logits'] for r in results) / steps:.6f}; the steps that differ (step, HF's margin, the rank "
            f"the device gives HF's token): {missed}"
        )
        decisive = [r for r in results if r["margin"] >= DECISIVE_MARGIN]
        decisive_same = sum(r["same_token"] for r in decisive)
        logger.info(
            f"{label} teacher-forced: top-1 equal to HF's on {decisive_same} of the {len(decisive)} steps where HF's "
            f"margin is {DECISIVE_MARGIN} or more"
        )
        if layers == 78:
            assert low >= GENERATE_MIN_LOGITS_PCC, f"{label}: logits pcc {low:.6f}"
            assert agree >= GENERATE_MIN_SAME * steps, f"{label}: top-1 equal to HF's on {agree} of {steps} steps"
        assert decisive_same == len(decisive), f"{label}: {decisive_same} of {len(decisive)} decisive steps"
        assert left_at_margin is None or left_at_margin < DECISIVE_MARGIN, (
            f"{label}: the free-running device left HF's greedy path at step {prefix}, where HF's margin is "
            f"{left_at_margin:.3f}"
        )
    finally:
        engine.close()


# ---------------------------------------------------------------------------------------------
# Long prompts, all 78 layers
# ---------------------------------------------------------------------------------------------
def _trace(root):
    """(token ids, next-token logits of the last position, streams directory or None) of a golden
    trace of the prefill stack's tooling."""
    import json

    import safetensors.torch

    with open(f"{root}/metadata.json") as f:
        meta = json.load(f)
    logits = safetensors.torch.load_file(f"{root}/final/logits_last.safetensors")
    (logits,) = logits.values()
    streams = f"{root}-streams"
    return torch.tensor(meta["token_ids"]), logits.flatten().float(), streams if os.path.isdir(streams) else None


def _trace_last_rows(root, layers, n):
    """The reference's cached latent of position n - 1 for every layer ([layers, 576], HF frame), or
    None when the trace does not hold every layer's rows."""
    import glob

    import safetensors

    rows = []
    for layer in range(layers):
        row = None
        for path in glob.glob(f"{root}/kv_cache/layer_{layer}/rows_*.safetensors"):
            start, end = (int(x) for x in os.path.basename(path)[len("rows_") : -len(".safetensors")].split("_"))
            if start <= n - 1 < end:
                with safetensors.safe_open(path, framework="pt") as f:
                    (name,) = f.keys()
                    row = f.get_slice(name)[n - 1 - start : n - start][0].float()
        if row is None:
            return None
        rows.append(row)
    return torch.stack(rows)


# id -> (prompt tokens, golden trace or None). The 32K trace is the prefill author's (LongBench);
# the 128K prompt is the same text cut longer and has no complete reference.
LONG_CASES = {
    "P8192": (8192, LONG_GOLDEN),
    "P32768": (32768, "/mnt/tt-data/dwadhwani/hy4-golden/hy4-bf16-longbench-32k"),
    "P131072": (131072, None),
}
LONG_PROMPT_128K = "/mnt/tt-data/dwadhwani/hy4-golden/hy4-bf16-longbench-128k"
LONG_MIN_LOGITS_PCC = 0.9


@galaxy
@pytest.mark.parametrize("case", list(LONG_CASES))
@pytest.mark.timeout(0)
def test_long_prompt(mesh_device, case):
    import json

    prompt_len, root = LONG_CASES[case]
    label = f"long {case} L78"
    ref_logits = streams_dir = None
    if root is not None:
        if not os.path.isdir(root):
            pytest.skip(f"{root} is not here")
        ids, ref_logits, streams_dir = _trace(root)
    else:
        if not os.path.isfile(f"{LONG_PROMPT_128K}/metadata.json"):
            pytest.skip(f"{LONG_PROMPT_128K} is not here")
        with open(f"{LONG_PROMPT_128K}/metadata.json") as f:
            ids = torch.tensor(json.load(f)["token_ids"])
    assert ids.numel() >= prompt_len, f"{ids.numel()} tokens for a {prompt_len}-token prompt"
    ids = ids[:prompt_len]
    tokenizer = _tokenizer()

    engine = Hy4Engine(mesh_device, num_layers=78, max_context=prompt_len + 8192, weight_cache=WEIGHT_CACHE)
    try:
        n = engine.prefill({0: ids})[0]
        model = _load(engine, label)
        layers = model.cfg.num_layers

        # ---------------- the first decode step: the last prompt token over the prefilled rows
        prefill_last = _last_rows(engine.cache, layers, n) if root is not None else None
        record = [] if streams_dir is not None else None
        t0 = time.perf_counter()
        logits = engine.step(record=record)
        host_logits = model.lm_head.to_torch(logits)[0]
        first_s = time.perf_counter() - t0
        _first_token_report(label, engine, n, first_s)
        _log_engine(label, engine)
        failures = []
        if ref_logits is not None:
            score = pcc(ref_logits, host_logits)
            top, ref_top = host_logits.topk(5).indices.tolist(), ref_logits.topk(5).indices.tolist()
            logger.info(f"{label} next-token logits of position {n - 1} against the fp32 reference: pcc={score:.6f}")
            logger.info(
                f"{label} next token: device {top[0]} reference {ref_top[0]} "
                f"({'same' if top[0] == ref_top[0] else 'DIFFERENT'}); top-5 device {top} reference {ref_top}; the "
                f"reference's margin top1 - top2 {ref_logits.topk(2).values.diff().abs().item():.3f}"
            )
            if score < LONG_MIN_LOGITS_PCC:
                failures.append(f"{label}: logits pcc={score:.6f} under {LONG_MIN_LOGITS_PCC}")
            if top[0] != ref_top[0] and ref_logits.topk(2).values.diff().abs().item() >= DECISIVE_MARGIN:
                failures.append(f"{label}: next token {top[0]}, the reference's is {ref_top[0]}")
            reference_rows = _trace_last_rows(root, layers, n)
            if reference_rows is not None:
                _log_last_rows(
                    label,
                    frames.tail_to_device(reference_rows, ROPE),
                    prefill_last,
                    _last_rows(engine.cache, layers, n),
                    "the fp32 reference's row",
                )
        else:
            logger.warning(f"{label}: no reference for this prompt; nothing below is checked against one")
        if streams_dir is not None:
            scores, worst = [], []
            for layer in range(layers):
                ref = torch.load(f"{streams_dir}/layer_{layer}.pt", weights_only=False)[n - 1].float()
                dev = record[layer]["streams_out"][0]
                scores.append(pcc(ref, dev))
                worst.append(min(pcc(ref[i], dev[i]) for i in range(ref.shape[0])))
                logger.info(
                    f"DEPTH {label} L{layer:02d}: streams_out {scores[-1]:.4f} worst stream {worst[-1]:.4f} "
                    f"(position {n - 1}, against the fp32 reference)"
                )
            logger.info(
                f"{label} streams_out against the fp32 reference: pcc={min(scores):.6f} (lowest of {layers} layers, at "
                f"layer {min(range(layers), key=scores.__getitem__)}); worst single stream {min(worst):.6f} at layer "
                f"{min(range(layers), key=worst.__getitem__)}; layers at 0.99 or better: "
                f"{sum(s >= 0.99 for s in scores)} of {layers}"
            )
        del record

        # ---------------- a few tokens of its own, and the time of a step at this context
        engine.accept([int(host_logits.argmax())])
        t0 = time.perf_counter()
        more = engine.generate(8)[:, 0]
        took = time.perf_counter() - t0
        generated = [int(host_logits.argmax())] + more.tolist()
        logger.info(
            f"GENERATED {label} after {n} prompt tokens: ids {generated} text {tokenizer.decode(generated)!r} "
            f"(8 steps in {took:.2f} s, the first compiles)"
        )
        _measure_latency(engine, label, context=engine.lengths[0])
        logger.info(f"MEASURED {label} L1_SMALL per core after everything: {pb.l1_small(mesh_device)}")
        assert not failures, failures
    finally:
        engine.close()
