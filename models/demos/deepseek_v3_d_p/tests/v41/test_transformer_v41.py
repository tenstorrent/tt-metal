# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 prefill transformer (bead F9): embed -> layers -> final collapse -> norm -> head, chunked.

The prompt is real text, so the logits of its last ``SCORED`` positions are teacher-forced next-token
predictions (the repo's accuracy convention: token agreement over many positions of real text, e.g. DeepSeek-V3
test_demo_teacher_forced). Device vs the reference's single-shot prefill of the same layer subset: top-1
agreement, top-5 recall of the reference's top-1, and logits PCC over those positions; repeats are bit-identical.

Bars (user decisions 2026-09-29): the reference's agreement with itself when every attention and MoE output
carries the error its component bar allows and every attention and MoE input the error that flips selection and
routing like the device's (``NOISE``, ``NOISE_SEEDS`` seeds), worst seed, minus ``MARGIN``:
the stack must compose its components' errors no worse than that (a state or composition bug fails it).
Free-running V4.1 stacks are chaotic (top-k selection and MoE routing flips: real layer 20's MoE keeps the same
experts on 77% of rows at input PCC 0.9987), so an absolute bar on one token's logits measures the flips.
Cases: small dims (one chunk, two chunks, a padded last chunk), then production shape (S=2048) with synthetic
and real weights.
"""

import os
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tests.v41.reference_weights import device_weights
from models.demos.deepseek_v3_d_p.tests.v41.small_config import SmallV41Config, small_spec
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.v41.engram import TtV41Engram, V41EngramHash, V41EngramTable
from models.demos.deepseek_v3_d_p.tt.v41.transformer import TtV41Transformer
from models.demos.deepseek_v3_d_p.tt.v41.weights import (
    dequant_fp8_block,
    load_layer,
    load_layer_dense,
    resolve_checkpoint,
)
from tests.ttnn.utils_for_testing import comp_pcc

SCHEDULES = {"sharing": (0, 2, 3, 20, 21, 24), "engram": (0, 1, 2, 3), "dspark": (20, 36, 37, 38, 39)}
SEQ = 512
SCORED = 256  # scored prompt positions (the last ones)
# floor noise (user decisions 2026-09-29; playbook ~/knowledge/wiki/playbooks/Acceptance bars.md; evidence
# evidence/F9-drift-diagnosis): on every attention and MoE output, the relative error a component PCC bar of
# 0.999 allows (sqrt(2 * (1 - 0.999)) = 0.045; device components measure 3-5%); on every attention and MoE input,
# the level at which the reference's component matches the device's on oracle inputs (selection and routing
# flips): layer-20 attention 0.993 text / 0.988 random at 0.4% (small), real layer-20 MoE 0.9977 vs 0.9974 at 0.5%
NOISE = (0.045, 4e-3)
NOISE_SEEDS = 5
# below the worst reference self-agreement: ~1 binomial SD of an agreement rate at 256 positions; PCC is smooth
MARGIN = {"top1": 0.02, "top5": 0.02, "pcc": 0.002}
PRODUCTION_SEQ = 2048
PRODUCTION_CANDIDATE_BLOCKS = 96  # of 128 visible blocks at S=2048 (2048 would make every block a candidate)
WEIGHT_CACHE = Path(os.environ.get("TT_V41_WEIGHT_CACHE", Path.home() / ".cache" / "tt-v41-weights"))
MESH = [
    pytest.param(
        (2, 4),
        fabric2d_device_params(),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
        id="fabric2d-mesh-2x4",
    )
]


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("schedule", list(SCHEDULES))
@pytest.mark.parametrize("case", ["one_chunk", "two_chunks", "padded"])
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_v41_transformer_small(mesh_device, device_params, case, schedule):
    layers = SCHEDULES[schedule]
    topology = per_axis_topology(device_params["fabric_config"])[1]
    chunk, total = {"one_chunk": (SEQ, SEQ), "two_chunks": (SEQ // 2, SEQ), "padded": (SEQ // 2, SEQ - 12)}[case]
    spec = small_spec(layers, SEQ, dspark=schedule == "dspark")
    tokens = orc.text_tokens(total)
    reference = orc.build_reference(spec)
    orc.load_engram_rows(reference, spec, tokens)  # synthetic Engram rows the prompt hashes to
    engram, engram_hash = {}, None
    if reference.engram_hash is not None:
        engram_hash = V41EngramHash(SmallV41Config, reference.engram_hash.token_map)
        for pos, layer in enumerate(layers):
            e = reference.layers[pos].engram
            if e is None:
                continue
            weights = {
                "wkv": dequant_fp8_block(e.wkv.weight.detach(), e.wkv.scale.detach()),
                "q_weight": e.q_weight.detach(),
                "k_weight": e.k_weight.detach(),
            }
            table = V41EngramTable(e.embed.weight.detach(), e.embed.scale.detach(), e.embed.oracle_rows)
            engram[layer] = TtV41Engram(mesh_device, SmallV41Config, layer, weights, table, topology)
    dspark = None
    if reference.mtp:
        fp8 = lambda linear: dequant_fp8_block(linear.weight.detach(), linear.scale.detach())
        dspark = {
            "main_proj": fp8(reference.mtp[0].main_proj),
            "main_norm": reference.mtp[0].main_norm.weight.detach(),
            "layers": [{"wkv": fp8(b.attn.wkv), "kv_norm": b.attn.kv_norm.weight.detach()} for b in reference.mtp],
        }
    # device MoE tensors converted once per weight identity (the model build dominated small-dims runs)
    identity = orc._digest(asdict(spec.args), spec.seed, "synthetic", orc._reference_digest(synthetic=True))
    with _stage(f"{schedule} {case} build"):
        model = TtV41Transformer(
            mesh_device,
            SmallV41Config,
            list(layers),
            lambda layer, include_moe: device_weights(reference, layers.index(layer), include_moe),
            reference.embed.weight.detach(),
            reference.norm.weight.detach(),
            reference.head.weight.detach(),
            max_seq_len=SEQ,
            chunk=chunk,
            dspark_weights=dspark,
            engram=engram,
            engram_hash=engram_hash,
            weight_cache_path=WEIGHT_CACHE / f"small-{identity}",
            topology=topology,
        )
    state = _check(model, spec, tokens, reference, f"{schedule} {case}")
    if dspark is not None:
        result = orc.oracle(spec, tokens, model=reference)
        # rings seeded from free-running device streams inherit their drift; reported only. The G1 ring bar
        # (>= 0.998) applies with teacher-forced taps (user decision 2026-09-29), gated in test_dspark.
        assert len(state.dspark_rings) == len(reference.mtp)
        for k, ring in enumerate(state.dspark_rings):
            device = ttnn.to_torch(ttnn.get_device_tensors(ring)[0])[0, 0]
            pcc = comp_pcc(result["state"]["dspark_window"][k].float(), device.float(), 0.0)[1]
            logger.info(f"transformer {schedule} {case}: DSpark ring {k} PCC {pcc:.5f} (free-running, reported)")


@contextmanager
def _stage(name: str):
    """Log a stage's start and elapsed time, so a run's progress is visible while it runs."""
    logger.info(f"stage {name}: start")
    t = time.perf_counter()
    yield
    logger.info(f"stage {name}: {time.perf_counter() - t:.1f}s")


def _agreement(expected: torch.Tensor, actual: torch.Tensor) -> dict:
    """[positions, vocab] logits: top-1 agreement, top-5 recall of the expected top-1, logits PCC."""
    top1 = expected.argmax(-1)
    return {
        "top1": (actual.argmax(-1) == top1).float().mean().item(),
        "top5": (actual.topk(5, dim=-1).indices == top1[:, None]).any(-1).float().mean().item(),
        "pcc": comp_pcc(expected, actual, 0.0)[1],
    }


def _check(model, spec, tokens, reference, name):
    """Two prefills of ``tokens`` [1, S] scoring the last SCORED positions: bit-identical, and each agreement
    metric vs the reference >= the reference's worst self-agreement under noise minus MARGIN. Returns the first
    prefill's state."""
    scored = min(SCORED, tokens.shape[1])
    with _stage(f"{name} prefill (compile + run)"):
        logits, state = model.prefill(tokens[0], scored)
    with _stage(f"{name} prefill (repeat)"):
        logits2, _ = model.prefill(tokens[0], scored)
    with _stage(f"{name} reference logits (cached unless precomputed)"):
        expected = orc.tail_logits(spec, tokens, scored, reference)
        floor = {k: 1.0 for k in MARGIN}
        for seed in range(NOISE_SEEDS):
            noisy = _agreement(expected, orc.tail_logits(spec, tokens, scored, reference, noise=(*NOISE, seed)))
            floor = {k: min(floor[k], noisy[k]) for k in floor}
    device = _agreement(expected, logits)
    logger.info(
        f"transformer {name}: device "
        + ", ".join(
            f"{k} {device[k]:.4f} (bar {floor[k] - MARGIN[k]:.4f}, reference self {floor[k]:.4f})" for k in MARGIN
        )
    )
    assert torch.equal(logits, logits2), "prefill is not bit-identical across repeats"
    for k in MARGIN:
        assert device[k] >= floor[k] - MARGIN[k], (k, device[k], floor[k])
    return state


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("chunks", [1, 2], ids=["one_chunk", "two_chunks"])
@pytest.mark.parametrize("weights", ["synthetic", "real"])
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_v41_transformer_production(mesh_device, device_params, weights, chunks):
    """Real dims, layers 0 2 3 20 21 24 (every sharing role and SWA-only; Engram layer 1 needs checkpoint
    tables, not downloaded). Precompute the reference logits outside the device lock first (``tail_logits``, clean
    and per noise seed; disk-cached)."""
    layers = SCHEDULES["sharing"]
    ckpt = resolve_checkpoint() if weights == "real" else None
    if weights == "real" and ckpt is None:
        pytest.skip("V4.1 checkpoint shards not downloaded")
    topology = per_axis_topology(device_params["fabric_config"])[1]
    spec = orc.real_spec(
        layers,
        PRODUCTION_SEQ,
        candidate_topk_blocks=PRODUCTION_CANDIDATE_BLOCKS,
        checkpoint=ckpt.root if ckpt else None,
    )
    cfg = type("V41TestConfig", (C,), {"CANDIDATE_TOPK_BLOCKS": PRODUCTION_CANDIDATE_BLOCKS})
    tokens = orc.text_tokens(PRODUCTION_SEQ)
    reference = orc.build_reference(spec) if ckpt is None else None
    # device MoE tensors converted once, keyed by weight identity (dims + layer set, seed, checkpoint, init)
    identity = orc._digest(
        asdict(spec.args), spec.seed, str(spec.checkpoint), orc._reference_digest(synthetic=ckpt is None)
    )
    if ckpt is None:
        layer_weights = lambda layer, include_moe: device_weights(reference, layers.index(layer), include_moe)
        embed, norm, head = (
            reference.embed.weight.detach(),
            reference.norm.weight.detach(),
            reference.head.weight.detach(),
        )
    else:
        layer_weights = lambda layer, include_moe: (load_layer if include_moe else load_layer_dense)(ckpt, layer)
        top = ckpt.read(["embed.weight", "norm.weight", "head.weight"])
        embed, norm, head = top["embed.weight"], top["norm.weight"], top["head.weight"]
    with _stage(f"production {weights} build (MoE weights cached after the first build)"):
        model = TtV41Transformer(
            mesh_device,
            cfg,
            list(layers),
            layer_weights,
            embed,
            norm,
            head,
            max_seq_len=PRODUCTION_SEQ,
            chunk=PRODUCTION_SEQ // chunks,
            weight_cache_path=WEIGHT_CACHE / f"{weights}-{identity}",
            topology=topology,
        )
    _check(model, spec, tokens, reference, f"production {weights} chunks={chunks}")
