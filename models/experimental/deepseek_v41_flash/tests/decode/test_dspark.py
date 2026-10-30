# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""``DeepSeekV41DSpark`` (the checkpoint's ``mtp.0..2`` block drafter) against ``Transformer.forward_spec``.

The drafter runs TP4 on a 1x4 submesh of an 8x4 Galaxy, on its own: random target-layer hiddens stand
in for the decode model's stash. A case seeds the rings with ``prompt`` positions, then drafts at
later positions. Each draft compares the pre-Markov logits by PCC, the proposals and the confidence logits.

* ``test_dspark_draft``: eager, one ring row written before each draft, as the reference's ``forward_spec``
  does; the stages' attention and MoE are compared one by one.
* ``test_dspark_spec_rounds``: the drafter as the spec loop drives it. ``write_main`` and ``draft`` are
  captured as traces and replayed; each round verifies a block of ``SPEC_DRAFTS + 1`` rows and writes all of
  them, rejected ones included. The rings must change at the written slots and nowhere else (exact), the
  replays must allocate nothing, and the upload and replay times are logged.

The routed experts come from the cache ``tools/fill_expert_cache.py --dspark --dtype bfloat8_b`` fills (and
``--dtype bfloat4_b`` for ``prefetch_bf4_experts``).

    pytest models/experimental/deepseek_v41_flash/tests/decode/test_dspark.py
"""

import contextlib
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.deepseek_v41_flash.tests.reference import load_reference, reference_dspark
from models.experimental.deepseek_v41_flash.tt.config import CACHE_DIR, DEFAULT_MODEL_DIR, dequantized, load_config
from models.experimental.deepseek_v41_flash.tt.decode.dspark import DeepSeekV41DSpark, host_to_device
from models.experimental.deepseek_v4_flash.tt.common import _trace_capture_guard
from models.experimental.deepseek_v4_flash.tt.layers import Linear
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader, resolve_snapshot_dir
from tests.ttnn.unit_tests.operations.prefetcher_common import tensor_prefetcher_session

# Stage 0 attention sees only the reference's own inputs and caches, so it is held tight at every frontier.
ATTENTION_PCC = 0.999
# A near-tie in a stage's top-3-of-128 routing (the scores are bf16) can flip and send a row to other experts;
# random hidden states make such ties common, and bfloat4_b experts more so. The MoE is judged on the device's own
# input, by the median, and the end-to-end drift only loosely. 0.95 is the bfloat4_b bar of tests/decode/test_moe.py.
MOE_PCC = 0.95
MEAN_LOGITS_PCC = 0.93
MIN_FIRST_ID_MATCH = 0.5
TP_SIZE = 4
# The drafter's attention dtype: bfloat4_b costs stage 0's attention ~0.96 PCC against ~0.9994 here.
WEIGHT_DTYPE = ttnn.bfloat8_b
PARENT_MESH = (8, 4)
DRAFTS = 8
# Drafts verified per round, as tests/decode/test_dspark_spec.py runs the spec loop.
SPEC_DRAFTS = 4
# Drafts accepted in each round: every count from none to all, so a block is cut at every offset.
ACCEPTED = (0, 4, 2, 1, 3, 0)
# A replay runs the same programs on the same inputs as the eager draft it was captured from.
TRACED_PCC = 0.9999
TRACE_REGION = int(os.environ.get("DEEPSEEK_V4_TRACE_REGION_SIZE", 256 << 20))


def _snapshot():
    try:
        return resolve_snapshot_dir(DEFAULT_MODEL_DIR)
    except FileNotFoundError:
        return None


def _to_torch(t: ttnn.Tensor) -> torch.Tensor:
    """Rank 0's copy of a tensor replicated over the mesh."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def _keep_output(outputs: dict, key):
    """Forward hook storing the module's output in ``outputs[key]``."""

    def hook(module, args, out):
        outputs[key] = out

    return hook


def _spy(obj, method: str, outputs: dict, key) -> None:
    """Have ``obj.<method>`` also store DRAM copies of its first argument and its output (the first, for
    a tuple) in ``outputs[key]``, as ``(input, output)``."""
    call = getattr(obj, method)

    def copy(t):
        return ttnn.clone(ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG))

    def spied(*args, **kwargs):
        given = copy(args[0])
        out = call(*args, **kwargs)
        outputs[key] = (given, copy(out[0] if isinstance(out, tuple) else out))
        return out

    setattr(obj, method, spied)


@contextlib.contextmanager
def _drafters(
    mesh_device, max_seq: int, use_prefetcher: bool = True, weight_dtype=WEIGHT_DTYPE, expert_dtype=ttnn.bfloat8_b
):
    """``(config, reference, drafter, logits_seen)``: the CPU drafter (:func:`reference_dspark`) and the device one on
    the first 1x4 submesh, alone on it as on its own pipeline stage. The device ``lm_head`` keeps its last output in
    ``logits_seen["logits"]``: the pre-Markov logits, which :meth:`DeepSeekV41DSpark.draft` does not return.

    Under ``use_prefetcher`` the drafter prefetches through its own rings, inside a prefetcher session that spans
    the block."""
    device = mesh_device.create_submesh(ttnn.MeshShape(1, TP_SIZE), ttnn.MeshCoordinate(0, 0))
    snapshot = _snapshot()
    config = load_config(snapshot)
    loader = DeepseekV4WeightLoader(snapshot)
    reference = reference_dspark(load_reference(snapshot), snapshot, loader, max_seq)
    freqs = reference.mtp[0].attn.freqs_cis

    cache = WeightCache(CACHE_DIR)
    lm_head = Linear(
        dequantized(loader, "lm_head.weight", translate=True), device, cache.file("lm_head"), dtype=ttnn.bfloat16
    )
    drafter = DeepSeekV41DSpark(
        config,
        loader,
        device,
        lm_head,
        (freqs.real, freqs.imag),
        cache=cache,
        weight_dtype=weight_dtype,
        expert_dtype=expert_dtype,
        tp_size=TP_SIZE,
        use_prefetcher=use_prefetcher,
    )
    logits_seen = {}

    def lm_head_spy(x):
        logits_seen["logits"] = lm_head(x)
        return logits_seen["logits"]

    drafter.lm_head = lm_head_spy
    with tensor_prefetcher_session(device) if use_prefetcher else contextlib.nullcontext():
        yield config, reference, drafter, logits_seen


def _seed(config, reference, drafter, prompt: int, max_seq: int):
    """Random target-layer means and tokens for ``max_seq`` positions: every position's means go into the stash,
    and both drafters' rings take positions ``[0, prompt)``. Returns ``(main_hidden [1, max_seq, n * D],
    tokens [max_seq])``."""
    torch.manual_seed(0)
    hidden, n_targets = config.hidden_size, len(drafter.targets)
    assert max_seq <= drafter.stash_rows, f"{max_seq} positions do not fit the {drafter.stash_rows}-row stash"
    main_hidden = torch.randn(1, max_seq, n_targets * hidden).bfloat16().float()
    tokens = torch.randint(config.vocab_size, (max_seq,))
    stash = torch.zeros(n_targets, 1, 1, drafter.stash_rows, hidden)
    for p in range(max_seq):
        stash[:, 0, 0, p % drafter.stash_rows] = main_hidden[0, p].reshape(n_targets, hidden)
    for k, pieces in enumerate(drafter.stash):
        for j, dst in enumerate(pieces):
            w = dst.shape[-1]
            host_to_device(stash[k][..., j * w : (j + 1) * w], dst)

    reference.spec(tokens[:1], main_hidden[:, :prompt], 0)
    seed = list(range(max(0, prompt - drafter.window), prompt))
    for i in range(0, len(seed), drafter.main_rows):
        drafter.set_main_rows(seed[i : i + drafter.main_rows])
        drafter.write_main()
    return main_hidden, tokens


def _reference_draft(reference, tokens, main_hidden, frontier: int, block: int):
    """The CPU drafter at ``frontier``, which writes that position's ring row first: its proposals ``[block]``, its
    pre-Markov logits ``[block, V]`` (what the device's ``lm_head`` returns) and its confidence logits ``[block]``."""
    ids, logits, conf = reference.spec(
        tokens[frontier + 1 : frontier + 2], main_hidden[:, frontier : frontier + 1], frontier
    )
    markov = torch.stack([reference.mtp[-1].markov_head(ids[:, i])[0] for i in range(block)], dim=1)
    return ids[0, 1:], (logits - markov).reshape(block, -1), conf.flatten()


def _device_draft(out, logits, block: int):
    """``(ids [block], pre-Markov logits [block, V], confidence [block])`` of a device draft: ``out`` as
    :meth:`DeepSeekV41DSpark.draft` returns it, ``logits`` what its ``lm_head`` returned."""
    ids, conf, _ = out
    return (
        _to_torch(ids).flatten().long(),
        _to_torch(logits).float().reshape(block, -1),
        _to_torch(conf).float().flatten(),
    )


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}], indirect=True)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
@pytest.mark.parametrize("prompt", (10, 140), ids=("first_lap", "wrapped"))
@pytest.mark.parametrize(
    "use_prefetcher, expert_dtype",
    ((True, ttnn.bfloat8_b), (False, ttnn.bfloat8_b), (True, ttnn.bfloat4_b)),
    ids=("prefetch", "l1", "prefetch_bf4_experts"),
)
@torch.no_grad()
def test_dspark_draft(mesh_device, prompt, use_prefetcher, expert_dtype):
    if tuple(mesh_device.shape) != PARENT_MESH:
        pytest.skip(f"needs an {PARENT_MESH[0]}x{PARENT_MESH[1]} mesh, got {tuple(mesh_device.shape)}")
    max_seq = prompt + DRAFTS + 16
    with _drafters(mesh_device, max_seq, use_prefetcher, expert_dtype=expert_dtype) as (
        config,
        reference,
        drafter,
        logits_seen,
    ):
        # Each stage's attention and MoE outputs and output streams, device and reference, for the log.
        got_parts, want_parts = {}, {}
        for s, (stage, block) in enumerate(zip(drafter.stages, reference.mtp)):
            for name in ("attention", "moe", "forward"):
                _spy(stage, name, got_parts, (s, name))
            block.attn.register_forward_hook(_keep_output(want_parts, (s, "attention")))
            block.ffn.register_forward_hook(_keep_output(want_parts, (s, "moe")))
            block.register_forward_hook(_keep_output(want_parts, (s, "forward")))
        main_hidden, tokens = _seed(config, reference, drafter, prompt, max_seq)

        failures, first_matches, pccs, moe_alone = [], 0, [], []
        for f in range(prompt, prompt + DRAFTS):
            want_ids, want_logits, want_conf = _reference_draft(reference, tokens, main_hidden, f, drafter.block)
            drafter.set_main_rows([f])
            drafter.write_main()
            drafter.set_block(int(tokens[f + 1]), f)
            got_ids, got_logits, got_conf = _device_draft(drafter.draft(), logits_seen["logits"], drafter.block)

            for (s, name), want in sorted(want_parts.items()):
                want = want[0] if isinstance(want, tuple) else want
                given, got = (_to_torch(t).float() for t in got_parts[(s, name)])
                _, pcc = comp_pcc(want.float(), got.reshape(want.shape))
                report = f"frontier {f}: stage {s} {name}: PCC {pcc}"
                if (s, name) == (0, "attention") and pcc < ATTENTION_PCC:
                    failures.append(f"frontier {f}: stage 0 attention PCC {pcc} < {ATTENTION_PCC}")
                if name == "moe":
                    # The MoE alone: the reference on the device's own input.
                    _, alone = comp_pcc(reference.mtp[s].ffn(given.reshape(want.shape)), got.reshape(want.shape))
                    moe_alone.append(alone)
                    report += f" (given the device's input {alone})"
                logger.info(report)
            _, pcc = comp_pcc(want_logits, got_logits)
            pccs.append(pcc)
            first_matches += int(got_ids[0] == want_ids[0])
            logger.info(
                f"frontier {f}: logits PCC {pcc}, ids {got_ids.tolist()} vs {want_ids.tolist()}, "
                f"confidence {got_conf.tolist()} vs {want_conf.tolist()}"
            )
        mean_pcc = sum(pccs) / len(pccs)
        median_moe = sorted(moe_alone)[len(moe_alone) // 2]
        logger.info(
            f"mean logits PCC {mean_pcc}, first proposal matched {first_matches} / {DRAFTS}, "
            f"median MoE PCC on the device's input {median_moe}"
        )
        if median_moe < MOE_PCC:
            failures.append(f"median MoE PCC on the device's input {median_moe} < {MOE_PCC}")
        if mean_pcc < MEAN_LOGITS_PCC:
            failures.append(f"mean logits PCC {mean_pcc} < {MEAN_LOGITS_PCC}")
        if first_matches < MIN_FIRST_ID_MATCH * DRAFTS:
            failures.append(f"first proposal matched {first_matches} / {DRAFTS} drafts")
        assert not failures, "; ".join(failures)


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(7200)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "trace_region_size": TRACE_REGION}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
@pytest.mark.parametrize("prompt", (10, 140, 200), ids=("first_lap", "wrapped", "ring_wrapped"))
@torch.no_grad()
def test_dspark_spec_rounds(mesh_device, prompt):
    """Rounds of the spec loop on the traced drafter: draft at the frontier, then write the whole verified block
    ``[frontier + 1, frontier + 1 + SPEC_DRAFTS]``, of which only the first ``accepted + 1`` rows are accepted.

    The CPU drafter takes only the accepted rows, one position at a time, so a rejected row the draft's mask let
    through shows up in the next draft. ``wrapped`` puts the frontier past the window, where the mask must also
    hide positions that left it; ``ring_wrapped`` puts it past the ring, so the window and the rejected rows wrap
    around slot 0 and the ring's slots outside the window hold real, older rows."""
    if tuple(mesh_device.shape) != PARENT_MESH:
        pytest.skip(f"needs an {PARENT_MESH[0]}x{PARENT_MESH[1]} mesh, got {tuple(mesh_device.shape)}")
    k = SPEC_DRAFTS
    max_seq = prompt + sum(m + 1 for m in ACCEPTED) + k + 16
    with _drafters(mesh_device, max_seq) as (config, reference, drafter, logits_seen):
        assert k + 1 <= drafter.main_rows and k <= drafter.block
        device, ring_rows, block = drafter.device, drafter.ring_rows, drafter.block
        main_hidden, tokens = _seed(config, reference, drafter, prompt, max_seq)

        # The seeding compiled write_main; this eager draft compiles draft (every program runs before the first
        # capture) and is what the first replay, on the same state and inputs, must reproduce.
        frontier = prompt - 1
        drafter.set_block(int(tokens[frontier + 1]), frontier)
        eager_ids, eager_logits, _ = _device_draft(drafter.draft(), logits_seen["logits"], block)
        ttnn.synchronize_device(device)
        tids = {}
        tids["write_main"] = ttnn.begin_trace_capture(device, cq_id=0)
        with _trace_capture_guard():
            drafter.write_main()
        ttnn.end_trace_capture(device, tids["write_main"], cq_id=0)
        tids["draft"] = ttnn.begin_trace_capture(device, cq_id=0)
        with _trace_capture_guard():
            draft_out = drafter.draft()
        ttnn.end_trace_capture(device, tids["draft"], cq_id=0)
        draft_logits = logits_seen["logits"]
        allocated = len(ttnn._ttnn.reports.get_buffers(device))

        def replay(name: str) -> float:
            t0 = time.perf_counter()
            ttnn.execute_trace(device, tids[name], cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            return time.perf_counter() - t0

        failures, pccs, first_matches = [], [], 0
        # The position each ring slot last took: a round rewrites the rows the previous one rejected from the same
        # stash rows, so those slots keep their values.
        slot_position = {p % ring_rows: p for p in range(max(0, prompt - drafter.window), prompt)}
        seconds = {"set_block": [], "draft": [], "set_main_rows": [], "write_main": []}
        for r, accepted in enumerate(ACCEPTED):
            want_ids, want_logits, want_conf = _reference_draft(reference, tokens, main_hidden, frontier, block)
            t0 = time.perf_counter()
            drafter.set_block(int(tokens[frontier + 1]), frontier)
            seconds["set_block"].append(time.perf_counter() - t0)
            seconds["draft"].append(replay("draft"))
            got_ids, got_logits, got_conf = _device_draft(draft_out, draft_logits, block)
            if r == 0:
                _, pcc = comp_pcc(eager_logits, got_logits)
                if not torch.equal(eager_ids, got_ids) or pcc < TRACED_PCC:
                    failures.append(
                        f"first replay vs the eager draft: ids {got_ids.tolist()} vs {eager_ids.tolist()}, "
                        f"logits PCC {pcc} < {TRACED_PCC}"
                    )
            _, pcc = comp_pcc(want_logits, got_logits)
            pccs.append(pcc)
            first_matches += int(got_ids[0] == want_ids[0])
            logger.info(
                f"round {r} (frontier {frontier}): logits PCC {pcc}, ids {got_ids.tolist()} vs {want_ids.tolist()}, "
                f"confidence {got_conf.tolist()} vs {want_conf.tolist()}"
            )

            positions = list(range(frontier + 1, frontier + 2 + k))
            slots = [p % ring_rows for p in positions]
            untouched = torch.ones(ring_rows, dtype=torch.bool)
            untouched[slots] = False
            fresh = [s for s, p in zip(slots, positions) if slot_position.get(s) != p]
            slot_position.update(zip(slots, positions))
            before = [_to_torch(stage.ring) for stage in drafter.stages]
            t0 = time.perf_counter()
            drafter.set_main_rows(positions)
            seconds["set_main_rows"].append(time.perf_counter() - t0)
            seconds["write_main"].append(replay("write_main"))
            for s, (stage, old) in enumerate(zip(drafter.stages, before)):
                new = _to_torch(stage.ring)
                if not torch.equal(new[0, 0, untouched], old[0, 0, untouched]):
                    changed = (new[0, 0] != old[0, 0]).any(-1) & untouched
                    failures.append(
                        f"round {r}: stage {s} write_main changed slots {changed.nonzero().flatten().tolist()} "
                        f"it was not given (positions {positions})"
                    )
                if (new[0, 0, fresh] == old[0, 0, fresh]).all(-1).any():
                    failures.append(f"round {r}: stage {s} write_main left a newly written slot of {fresh} as it was")
            # The CPU rings take the accepted rows; the new frontier's row is written by the next round's draft.
            for q in positions[:accepted]:
                reference.spec(tokens[q + 1 : q + 2], main_hidden[:, q : q + 1], q)
            frontier = positions[accepted]

        after = len(ttnn._ttnn.reports.get_buffers(device))
        if after != allocated:
            failures.append(f"{after - allocated} device buffers allocated by the traced replays")
        for tid in tids.values():
            ttnn.release_trace(device, tid)

        def ms(values):
            return f"{1e3 * sorted(values)[len(values) // 2]:.2f} ms"

        logger.info(
            "median per round: "
            + ", ".join(f"{name} {ms(values)}" for name, values in seconds.items())
            + " (uploads are host time; a replay is its dispatch plus the device, on an otherwise idle submesh)"
        )
        mean_pcc = sum(pccs) / len(pccs)
        logger.info(f"mean logits PCC {mean_pcc}, first proposal matched {first_matches} / {len(ACCEPTED)}")
        if mean_pcc < MEAN_LOGITS_PCC:
            failures.append(f"mean logits PCC {mean_pcc} < {MEAN_LOGITS_PCC}")
        if first_matches < MIN_FIRST_ID_MATCH * len(ACCEPTED):
            failures.append(f"first proposal matched {first_matches} / {len(ACCEPTED)} drafts")
        assert not failures, "; ".join(failures)
