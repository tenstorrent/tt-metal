# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 prefill transformer (bead F9): embed -> layers -> final collapse -> norm -> head, chunked.

The prompt is real text, so the logits of its last ``SCORED`` positions are teacher-forced next-token
predictions. Free-running V4.1 stacks are chaotic (top-k selection and MoE routing flips: real layer 20's MoE
keeps the same experts on 77% of rows at input PCC 0.9987), so absolute bars on one token's logits measure flips.

Gate (user decisions 2026-09-29; playbook ~/knowledge/wiki/playbooks/Acceptance bars.md): each layer's
free-running streams (all rows and the last SCORED rows) vs the reference's single-shot prefill must be no
further than the reference itself drifts when every attention and MoE output carries the error its component bar
allows and every attention and MoE input the error that flips selection and routing like the device's
(``NOISE``, ``NOISE_SEEDS`` seeds, worst seed), minus ``DRIFT_MARGIN``: the stack composes its components' errors
no worse than the components predict (a state or composition bug fails it). Top-1 agreement, top-5 recall and
logits PCC over the scored positions are reported against the same floor; repeats are bit-identical.
Cases: small dims (one chunk, two chunks, a padded last chunk), then production shape (S=2048) with synthetic
and real weights, then long context on real weights (bead 8y7.19.1): N real-text chunks of 5120 tokens, the last
one (e.g. at start 51200 for 11 chunks) attending over the caches the preceding chunks filled on device, with the
released candidate count (2048 blocks). There the gate also covers the last chunk's rows (``chunk``), and the
block acceptance of test_block_v41 runs at the last chunk: every chunk teacher-forced through every block on a
fresh state (oracle inputs), the last chunk's block outputs >= the real block bar, the KV sources' compressed-KV and
index-K rows and every window carry >= the cache bar. The single-shot oracle scores the indexer in query blocks
(``oracle.INDEXER_QUERY_BLOCK``, bit-identical) so that it fits in host memory at S=56320.
"""

import hashlib
import os
import time
from contextlib import contextmanager

import pytest
import torch
from loguru import logger

import ttnn
from models.common import timing_events
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tests.v41.reference_weights import MOE_KEYS, device_weights
from models.demos.deepseek_v3_d_p.tests.v41.small_config import SmallV41Config, small_spec
from models.demos.deepseek_v3_d_p.tests.v41.test_block_v41 import (
    BLOCK_PCC,
    CACHE_PCC,
    KV_FORMATS,
    _pack,
    _pcc,
    _unpack,
    unrounded_kv,
)
from models.demos.deepseek_v3_d_p.tests.v41.weight_cache import (  # noqa: F401 (WEIGHT_CACHE re-export)
    WEIGHT_CACHE,
    host_weights,
    weight_cache_dir,
)
from models.demos.deepseek_v3_d_p.tt.v41.cache import WINDOW_SLOT, V41PrefillState
from models.demos.deepseek_v3_d_p.tt.v41.engram import TtV41Engram, V41EngramHash, V41EngramTable
from models.demos.deepseek_v3_d_p.tt.v41.transformer import TtV41Transformer
from models.demos.deepseek_v3_d_p.tt.v41.weights import (
    dequant_fp8_block,
    load_layer,
    load_layer_dense,
    resolve_checkpoint,
)
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat
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
MARGIN = {"top1": 0.02, "top5": 0.02, "pcc": 0.002}  # token metrics: reported against the floor, not gated
# per-layer stream PCC below the reference's worst noisy drift (user decision 2026-09-29: gate per-layer drift;
# Gaussian noise matches the stream error's size but not its flip-shaped per-row distribution, which token
# agreement is sensitive to)
DRIFT_MARGIN = 0.003
PRODUCTION_SEQ = 2048
PRODUCTION_CANDIDATE_BLOCKS = 96  # of 128 visible blocks at S=2048 (2048 would make every block a candidate)
LONG_CHUNK = 5120
# case -> (prompt tokens, chunk, candidate blocks; None = released 2048). Long cases: real weights only.
PRODUCTION_CASES = {
    "one_chunk": (PRODUCTION_SEQ, PRODUCTION_SEQ, PRODUCTION_CANDIDATE_BLOCKS),
    "two_chunks": (PRODUCTION_SEQ, PRODUCTION_SEQ // 2, PRODUCTION_CANDIDATE_BLOCKS),
    **{f"{n}x{LONG_CHUNK}": (n * LONG_CHUNK, LONG_CHUNK, None) for n in (2, 4, 11)},
    "4x5120_scaled_fp8": (4 * LONG_CHUNK, LONG_CHUNK, None),
}
# cases whose compressed KV uses another format (test_block_v41 KV_FORMATS): the transformer's free-running prefill
# has no format choice, so these gate only the teacher-forced block acceptance at the last chunk, in that format
KV_FORMAT_CASES = {"4x5120_scaled_fp8": "scaled_fp8"}
MESH = [
    pytest.param(
        (2, 4),
        fabric2d_device_params(),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
        id="fabric2d-mesh-2x4",
    )
]
MESH_4X2 = [  # LoudBox 4x2 (SP4 x TP2): production gate only (beads 8y7.20.*.2)
    pytest.param(
        (4, 2),
        fabric2d_device_params(),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
        id="fabric2d-mesh-4x2",
    )
]


def setup_small(mesh_device, case, schedule):
    """Everything before the small prefill: prompt, reference, Engram / DSpark modules and the transformer (device
    MoE tensors from / into the weight cache). Shared by the test and the cache prepare step (mock mesh)."""
    layers = SCHEDULES[schedule]
    chunk, total = {"one_chunk": (SEQ, SEQ), "two_chunks": (SEQ // 2, SEQ), "padded": (SEQ // 2, SEQ - 12)}[case]
    spec = small_spec(layers, SEQ, dspark=schedule == "dspark")
    tokens = orc.text_tokens(total)
    reference = orc.LazyReference(spec)()  # always needed here (Engram tables / DSpark weights); a phase event
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
            engram[layer] = TtV41Engram(mesh_device, SmallV41Config, layer, weights, table)
    dspark = None
    if reference.mtp:
        fp8 = lambda linear: dequant_fp8_block(linear.weight.detach(), linear.scale.detach())
        dspark = {
            "main_proj": fp8(reference.mtp[0].main_proj),
            "main_norm": reference.mtp[0].main_norm.weight.detach(),
            "layers": [{"wkv": fp8(b.attn.wkv), "kv_norm": b.attn.kv_norm.weight.detach()} for b in reference.mtp],
        }
    # device MoE tensors converted once per weight identity (tests/v41/weight_cache.py)
    with _stage(f"{schedule} {case} build", "weights"):
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
            weight_cache_path=weight_cache_dir(spec, mesh_device.shape),
        )
    return model, spec, tokens, reference, dspark


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("schedule", list(SCHEDULES))
@pytest.mark.parametrize("case", ["one_chunk", "two_chunks", "padded"])
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_v41_transformer_small(mesh_device, device_params, case, schedule):
    model, spec, tokens, reference, dspark = setup_small(mesh_device, case, schedule)
    state, _ = _check(model, spec, tokens, reference, f"{schedule} {case}")
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
def _stage(name: str, kind: str):
    """Log a stage's start and elapsed time (visible while the run runs) and emit it as a ``kind`` timing phase
    (reference / oracle / weights / compute) for the lock-phase breakdown."""
    logger.info(f"stage {name}: start")
    t = time.perf_counter()
    with timing_events.phase(kind, name=name):
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


def reference_data(spec, tokens, reference, last_chunk=None):
    """What the gate compares against: the clean oracle, the scored tail logits, the per-layer noisy drifts (with
    ``last_chunk``, also over the last chunk's rows) and the token-agreement floor (all disk-cached; the prepare step
    fills them outside the device lock)."""
    scored = min(SCORED, tokens.shape[1])
    clean = orc.oracle(spec, tokens, reference)
    expected = orc.tail_logits(spec, tokens, scored, reference)
    drifts = [
        orc.noise_drift(spec, tokens, scored, (*NOISE, s), reference, chunk=last_chunk) for s in range(NOISE_SEEDS)
    ]
    token_floor = {k: 1.0 for k in MARGIN}
    for seed in range(NOISE_SEEDS):
        noisy = _agreement(expected, orc.tail_logits(spec, tokens, scored, reference, noise=(*NOISE, seed)))
        token_floor = {k: min(token_floor[k], noisy[k]) for k in token_floor}
    return clean, expected, drifts, token_floor


def kv_format_reference(spec, tokens, reference):
    """What a KV_FORMAT_CASES case compares against (disk-cached; the prepare step fills it): the clean oracle and
    the KV sources' unrounded compressed KV."""
    clean = orc.oracle(spec, tokens, reference)
    return clean, unrounded_kv(spec, tokens, reference, clean)


def _check(model, spec, tokens, reference, name, last_chunk=None):
    """Two prefills of ``tokens`` [1, S]: bit-identical; each layer's free-running streams (all rows, the last
    SCORED rows and, with ``last_chunk``, the last chunk's rows) no further from the reference than its drift under
    the floor noise (worst seed) minus DRIFT_MARGIN. Token agreement vs the same floor is reported. Returns the
    first prefill's state and the clean oracle result."""
    scored = min(SCORED, tokens.shape[1])
    mesh, tp = model.mesh_device, model.mesh_device.shape[1]
    concat = ttnn.ConcatMesh2dToTensor(mesh, tuple(mesh.shape), dims=(2, 3))
    streams = {}

    def observe(layer, x, start, length):
        rows = ttnn.to_torch(x, mesh_composer=concat)[0, 0, :length]
        streams.setdefault(layer, []).append(_unpack(rows, model.config.HC_MULT, tp))

    with _stage(f"{name} prefill (compile + run, observed)", "compute"):
        logits, state = model.prefill(tokens[0], scored, observe)
    with _stage(f"{name} prefill (repeat)", "compute"):
        logits2, _ = model.prefill(tokens[0], scored)
    with _stage(f"{name} reference (cached unless precomputed)", "oracle"):
        clean, expected, drifts, token_floor = reference_data(spec, tokens, reference, last_chunk)
    assert torch.equal(logits, logits2), "prefill is not bit-identical across repeats"
    dump = os.environ.get("TT_V41_TRANSFORMER_DUMP")
    if dump:  # cross-process determinism: compare two runs' dumps with torch.equal
        final = torch.cat(streams[list(streams)[-1]])
        digests = {
            l: hashlib.sha256(torch.cat(p).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for l, p in streams.items()
        }
        torch.save({"logits": logits, "final_stream": final, "stream_sha256": digests}, dump)
        logger.info(
            f"transformer {name}: dumped logits {tuple(logits.shape)}, final stream {tuple(final.shape)} to {dump}"
        )
    tokens_device = _agreement(expected, logits)
    logger.info(
        f"transformer {name} tokens (reported): "
        + ", ".join(f"{k} {tokens_device[k]:.4f} (reference self {token_floor[k]:.4f})" for k in MARGIN)
    )
    failures = []
    windows = [("all", slice(None)), ("tail", slice(-scored, None))]
    if last_chunk:
        windows.append(("chunk", slice(-last_chunk, None)))
    for layer, parts in streams.items():
        device, ref = torch.cat(parts), clean["blocks"][layer]["x_out"]
        for key, rows in windows:
            value = comp_pcc(ref[rows].float(), device[rows].float(), 0.0)[1]
            floor = min(d[layer][key] for d in drifts)
            logger.info(f"transformer {name} layer {layer} {key}: device {value:.5f} (bar {floor - DRIFT_MARGIN:.5f})")
            if value < floor - DRIFT_MARGIN:
                failures.append((layer, key, value, floor))
    assert not failures, failures
    return state, clean


def _teacher_forced_last_chunk(model, clean, total, name, kv_format=MlaKvCacheFormat.BF16_RM, unrounded=None):
    """The block acceptance of test_block_v41 at the last chunk of a chunked prefill of ``total`` tokens: every chunk
    through every block on the oracle's inputs (streams and pre-mix) over a fresh state, so each block reads caches
    it wrote itself from teacher-forced inputs. Gates the last chunk's block outputs (>= BLOCK_PCC real), the KV
    sources' compressed-KV and index-K rows (all rows) and every window carry (>= CACHE_PCC). With ``kv_format``
    SCALED_FP8 the compressed KV is compared with ``unrounded`` (test_block_v41.unrounded_kv; vs the FP4-QDQ rows
    reported)."""
    mesh, cfg, chunk = model.mesh_device, model.config, model.chunk
    shape, tp, n = tuple(mesh.shape), mesh.shape[1], cfg.HC_MULT
    down = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, shape, dims=(2, 3)))
    if kv_format == MlaKvCacheFormat.BF16_RM:
        state = model._new_state()
    else:
        state = V41PrefillState(mesh, cfg, model.max_seq_len, chunk, model.layers, kv_format=kv_format)
    last = {}
    for start in range(0, total, chunk):
        length = min(chunk, total - start)
        for layer, block in zip(model.layers, model.blocks):
            rec = clean["blocks"][layer]
            x_rows = torch.zeros(chunk, *rec["x_in"].shape[1:])
            x_rows[:length] = rec["x_in"][start : start + length].float()
            pre_rows = torch.zeros(chunk, rec["pre_in"].shape[-1])
            pre_rows[:length] = rec["pre_in"][start : start + length].float()
            x = ttnn.from_torch(
                _pack(x_rows, tp),
                device=mesh,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, shape, dims=(2, 3)),
            )
            pre = ttnn.from_torch(
                pre_rows[None, None],
                device=mesh,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, shape, dims=(2, None)),
            )
            x_out, pre_out = block(x, pre, state, length)
            if start + length == total:
                last[layer] = (_unpack(down(x_out)[0, 0], n, tp)[:length], down(pre_out)[0, 0, :length, :n])
            for t in (x, pre, x_out, pre_out):
                ttnn.deallocate(t)
        state.advance(length)
    first, failures = total - length, []
    for layer, (x_out, pre_out) in last.items():
        rec = clean["blocks"][layer]
        entry = {
            "block_out": (_pcc(rec["x_out"][first:total], x_out), BLOCK_PCC["real"]),
            "pre_mix": (_pcc(rec["pre_out"][first:total], pre_out), None),
        }
        tail = rec["window_kv"][max(0, total - WINDOW_SLOT) : total].float()
        carry = torch.zeros(WINDOW_SLOT, tail.shape[-1])
        carry[WINDOW_SLOT - tail.shape[0] :] = tail
        entry["window_kv"] = (_pcc(carry, state.to_host(state.window_carry[layer])), CACHE_PCC)
        if layer in C.KV_SOURCE_LAYERS:
            pub = clean["shared"][layer]
            rows, w0 = pub["compress_kv"].shape[0], state.geometry.window_rows
            stored = state.to_host(state.kv[layer])[w0 : w0 + rows]
            if kv_format == MlaKvCacheFormat.SCALED_FP8:
                entry["compressed_kv"] = (_pcc(unrounded[layer], stored), CACHE_PCC)
                entry["compressed_kv_vs_fp4"] = (_pcc(pub["compress_kv"], stored), None)
            else:
                entry["compressed_kv"] = (_pcc(pub["compress_kv"], stored), CACHE_PCC)
            entry["index_k"] = (_pcc(pub["index_k"], state.to_host(state.index_k[layer])[:rows]), CACHE_PCC)
        logger.info(
            f"transformer {name} teacher-forced last chunk [{first}, {total}) layer {layer}: "
            + ", ".join(f"{k} {v:.5f}" + (f" (bar {bar})" if bar else "") for k, (v, bar) in entry.items())
        )
        failures += [(layer, k, v, bar) for k, (v, bar) in entry.items() if bar is not None and v < bar]
    assert not failures, failures


def setup_production(mesh_device, weights, case):
    """Everything before the production prefill of ``case`` (PRODUCTION_CASES; see setup_small); None when the
    checkpoint is not downloaded."""
    layers = SCHEDULES["sharing"]
    seq, chunk, candidate_blocks = PRODUCTION_CASES[case]
    ckpt = resolve_checkpoint() if weights == "real" else None
    if weights == "real" and ckpt is None:
        return None
    spec = orc.real_spec(
        layers,
        seq,
        candidate_topk_blocks=candidate_blocks,
        checkpoint=ckpt.root if ckpt else None,
    )
    cfg = C if candidate_blocks is None else type("V41TestConfig", (C,), {"CANDIDATE_TOPK_BLOCKS": candidate_blocks})
    tokens = orc.text_tokens(seq)
    # built only on an oracle or weight-cache miss (warm runs construct no reference model)
    reference = orc.LazyReference(spec)
    root = weight_cache_dir(spec, mesh_device.shape)
    if ckpt is None:

        def layer_weights(layer, include_moe):
            pos, dense = layers.index(layer), f"layer_{layer}.dense"
            if not include_moe:
                return host_weights(root, dense, lambda: device_weights(reference(), pos, include_moe=False))
            weights = device_weights(reference(), pos)
            host_weights(root, dense, lambda: {k: v for k, v in weights.items() if k not in MOE_KEYS})
            return weights

        top = host_weights(
            root,
            "top",
            lambda: {k: getattr(reference(), k).weight.detach() for k in ("embed", "norm", "head")},
        )
        embed, norm, head = top["embed"], top["norm"], top["head"]
    else:
        layer_weights = lambda layer, include_moe: (load_layer if include_moe else load_layer_dense)(ckpt, layer)
        top = ckpt.read(["embed.weight", "norm.weight", "head.weight"])
        embed, norm, head = top["embed.weight"], top["norm.weight"], top["head.weight"]
    with _stage(f"production {weights} build (MoE weights cached after the first build)", "weights"):
        model = TtV41Transformer(
            mesh_device,
            cfg,
            list(layers),
            layer_weights,
            embed,
            norm,
            head,
            max_seq_len=seq,
            chunk=chunk,
            weight_cache_path=root,
        )
    return model, spec, tokens, reference


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(
    "weights, chunks",
    [(w, c) for c in ("one_chunk", "two_chunks") for w in ("synthetic", "real")]
    + [("real", c) for c in PRODUCTION_CASES if c not in ("one_chunk", "two_chunks")],
    ids=lambda v: v,
)
@pytest.mark.parametrize("mesh_device, device_params", MESH + MESH_4X2 + galaxy_meshes(), indirect=True)
def test_v41_transformer_production(mesh_device, device_params, weights, chunks):
    """Real dims, layers 0 2 3 20 21 24 (every sharing role and SWA-only; Engram layer 1 needs checkpoint
    tables, not downloaded). Precompute the reference outside the device lock first (tests/v41/prepare_caches.py:
    oracle, ``tail_logits`` clean and per noise seed; disk-cached). Long cases (``<n>x5120``) also gate the last
    chunk's rows and run the block acceptance at the last chunk (module docstring). Galaxy 8x4 / 4x8 (bead
    8y7.13.4): the same cases and bars; the oracle is mesh-independent (LoudBox caches are reused)."""
    setup = setup_production(mesh_device, weights, chunks)
    if setup is None:
        pytest.skip("V4.1 checkpoint shards not downloaded")
    model, spec, tokens, reference = setup
    seq, chunk, _ = PRODUCTION_CASES[chunks]
    long = chunk == LONG_CHUNK
    name = f"production {weights} chunks={chunks}"
    if chunks in KV_FORMAT_CASES:
        with _stage(f"{name} reference (cached unless precomputed)", "oracle"):
            clean, unrounded = kv_format_reference(spec, tokens, reference)
        with _stage(f"{name} teacher-forced blocks", "compute"):
            _teacher_forced_last_chunk(model, clean, seq, name, KV_FORMATS[KV_FORMAT_CASES[chunks]], unrounded)
        return
    _, clean = _check(model, spec, tokens, reference, name, last_chunk=chunk if long else None)
    if long:
        with _stage(f"{name} teacher-forced blocks", "compute"):
            _teacher_forced_last_chunk(model, clean, seq, name)
    logger.info(f"{name}: reference model built: {reference.built}")
