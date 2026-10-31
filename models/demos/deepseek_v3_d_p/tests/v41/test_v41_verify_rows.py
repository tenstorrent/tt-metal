# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The speculative round on the whole model at the production pool (speculative decode, device gate and the
first measured round): one user from a real prompt, the DSpark drafter (``tt/v41/dspark_drafter.py``)
proposing
``K`` tokens a round and the verify decoder (``tt/v41/spec_verify.py``) checking them as ``1 + K`` rows of one
step, greedy.

Phases, all on one model and one prefill of ``V41_VR_PROMPT`` tokens of real text (the rings seeded by the
prefill):

1. ``plain``: the slot joined, ``V41_VR_TOKENS`` greedy tokens one row a step (``K = 0``: the same graph), the
   text and the replayed step's wall and device time;
2. for each ``K`` in ``V41_VR_KS``: the slot re-joined from the prefill state, rounds of draft + verify +
accept +
   commit until ``V41_VR_TOKENS`` tokens are committed; the committed text must equal the plain text token for
   token (the serial-replay gate of ``spec_state``: a rejected row's writes must not reach a committed
   position),
   the accepted drafts per round, the round's wall split (draft, verify, host) and the time per committed
   token.

``V41_VR_SEQ`` is the pool's capacity (the served 1,054,720 by default), ``V41_VR_OUT`` takes the JSON record,
which
also carries the DRAM a chip (allocated and free) after the model, the decoder and the drafter.

``V41_VR_FUSED=1`` (default): the drafter runs inside the verify step's capture on every accept outcome (one
replay and one readback a round; the first round's drafts come from one standalone draft call after the join);
``0``: a standalone (traced) draft call before every verify step, the taps selected from the step's tile.
``V41_VR_TEMP=T`` (default 0: greedy), ``V41_VR_TOPP`` (default 1.0) and ``V41_VR_TOPK`` (default 0: the
device's
32) sample the plain steps and the rounds with that triple on every live row (the match rule
exact against sequential served sampling); the committed text then differs from the plain text by
construction (``matches_plain`` is recorded, not asserted) and the record carries the acceptance under
sampling.
``V41_VR_LIVE=n`` (default 1; first c32 measurement): ``n`` live slots in lockstep, all
joined from the one prefill (the same prompt, so every slot drafts and accepts the same tokens), ``n / sp`` a
mesh row at user indices ``0 .. n / sp - 1`` of the rank's ``V41_VR_USERS`` slots (``n < sp``: ``n`` slots of
rank 0 alone, the c2 .. c4 regime); the MoE tile's real rows are
``V41_VR_USERS x (1 + Kmax)`` a rank (``TtV41VerifyDecoder(real_rows=)``: decoupled from
the pool's slots, so the production 32-slot pool at ``V41_VR_SEQ`` serves 32 live slots; the earlier geometry
realised the
rows through ``V41_VR_USERS=12 V41_VR_SEQ=348160``, a 96-slot pool at a third of the capacity, with the live
slots at user indices ``0, 1 + Kmax, ..``) and a slot's ``1 + K`` rows are ``u (1 + Kmax) .. u (1 + Kmax) +
K``;
the drafter is not in the step's graph (``V41_VR_FUSED=0`` is forced: the fused call takes one live slot a
rank)
but has a ring per slot (``TtDSparkDrafter(users=)``) and one standalone call after the
step proposes for every live slot of a rank at once from its own ring (``run_blocks``; the taps of the
accepted
rows written first). The round's wall and device time are the step's over all the slots; the tokens a round
are
one slot's. ``V41_VR_REAL_ROWS`` overrides the real rows (0: the slots are the real rows).
"""

import json
import os
import time

import pytest
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_depth import _rss_gb, parse_layers
from models.demos.deepseek_v3_d_p.tests.v41.weight_cache import WEIGHT_CACHE, routed_expert_dtype, weight_cache_dir
from models.demos.deepseek_v3_d_p.tt.v41.dspark_drafter import TtDSparkDrafter
from models.demos.deepseek_v3_d_p.tt.v41.dspark_weights import seeding_weights
from models.demos.deepseek_v3_d_p.tt.v41.engram_checkpoint import build_engram_modules
from models.demos.deepseek_v3_d_p.tt.v41.spec_verify import TtV41VerifyDecoder
from models.demos.deepseek_v3_d_p.tt.v41.transformer import TtV41Transformer
from models.demos.deepseek_v3_d_p.tt.v41.weights import load_layer, load_layer_dense, resolve_checkpoint

TRACE_REGION = 2 * 1024**3


def _dram(mesh_device) -> dict:
    try:
        view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
        banks = int(view.num_banks)
        return {
            "allocated": int(view.total_bytes_allocated_per_bank) * banks,
            "free": int(view.total_bytes_free_per_bank) * banks,
            "largest_free_per_bank": int(view.largest_contiguous_bytes_free_per_bank),
        }
    except Exception as e:
        logger.warning(f"DRAM statistics unavailable: {e}")
        return {}


@pytest.mark.timeout(int(os.environ.get("V41_VR_TIMEOUT", 3 * 3600)))
@pytest.mark.parametrize("mesh_device, device_params", galaxy_meshes(trace_region_size=TRACE_REGION), indirect=True)
def test_v41_verify_rows(mesh_device, device_params):
    layers = parse_layers(os.environ.get("V41_VR_LAYERS", "all"))
    cap = int(os.environ.get("V41_VR_SEQ", 1054720))
    chunk = int(os.environ.get("V41_VR_CHUNK", 5120))
    users = int(os.environ.get("V41_VR_USERS", 4))
    prompt_len = int(os.environ.get("V41_VR_PROMPT", 300))
    n_tokens = int(os.environ.get("V41_VR_TOKENS", 24))
    ks = [int(k) for k in os.environ.get("V41_VR_KS", "1,2,3").split(",") if k]
    warmup = int(os.environ.get("V41_VR_WARMUP", 2))
    fused = os.environ.get("V41_VR_FUSED", "1") == "1"
    live = int(os.environ.get("V41_VR_LIVE", 1))
    if live > 1:
        fused = False
    # V41_VR_READ_RANKS=1: the standalone call read as the served round reads it (``run_blocks(read=False)``
    # then
    # ``read_ranks``: every rank's blocks)
    read_ranks = os.environ.get("V41_VR_READ_RANKS", "0") == "1" and not fused
    # V41_VR_PREPARE_NEXT=1 (with V41_VR_READ_RANKS=1): the next step's position-dependent inputs built while
    # the
    # draft replay runs, as the served round does
    prepare_next = read_ranks and os.environ.get("V41_VR_PREPARE_NEXT", "1") == "1"
    temperature = float(os.environ.get("V41_VR_TEMP", 0))
    sampling = (
        {"k": int(os.environ.get("V41_VR_TOPK", 0)), "p": float(os.environ.get("V41_VR_TOPP", 1.0)), "t": temperature}
        if temperature > 0
        else None
    )
    ckpt = resolve_checkpoint()
    if ckpt is None:
        pytest.skip("V4.1 checkpoint shards not downloaded")
    from transformers import AutoTokenizer

    sp, tp = tuple(mesh_device.shape)
    real_rows = int(os.environ.get("V41_VR_REAL_ROWS", users * (1 + max(ks)) if live > 1 else 0)) or None
    if live > 1 and real_rows is None:  # geometry: the slots are the real rows
        assert live % sp == 0 and (live // sp) * (1 + max(ks)) <= users, (live, sp, ks, users)
        slots = [r * users + u * (1 + max(ks)) for r in range(sp) for u in range(live // sp)]
    elif live > 1 and live < sp:  # the c2 .. c4 regime: the live slots on rank 0
        assert live <= users and real_rows <= 32, (live, users, real_rows)
        slots = list(range(live))
    elif live > 1:
        assert live % sp == 0 and live // sp <= users and real_rows <= 32, (live, sp, users, real_rows)
        slots = [r * users + u for r in range(sp) for u in range(live // sp)]
    else:
        slots = [0]
    rank0 = [slot for slot in slots if slot // users == 0]  # the live slots of rank 0 (every rank's are alike)
    draft_users = [slot % users for slot in rank0]  # the standalone call's blocks, in user order
    max_seq = -(-(prompt_len + n_tokens + max(ks) + 2) // chunk) * chunk  # the prefill state's capacity
    assert cap % chunk == 0 and cap >= max_seq, (cap, chunk, max_seq)
    tokenizer = AutoTokenizer.from_pretrained(orc.HF_SNAPSHOT)
    need = prompt_len + 8
    try:
        text = orc.text_tokens(need)[0]
    except ValueError as e:  # a prompt beyond the book (the 1M gate): the book repeated end to end
        have = int(str(e).split()[0])
        text = orc.text_tokens(have)[0].repeat(-(-need // have))[:need]
        logger.info(
            f"verify rows: {have} tokens of text repeated {-(-need // have)} times for a prompt of {prompt_len}"
        )
    prompt = text[:prompt_len]
    record = {
        "fused": fused,
        "taps": os.environ.get("V41_VR_TAPS", "step"),
        "layers": layers,
        "n_layers": len(layers),
        "capacity": cap,
        "chunk": chunk,
        "users_per_rank": users,
        "real_rows": real_rows,
        "live_slots": slots,
        "prompt": prompt_len,
        "tokens": n_tokens,
        "ks": ks,
        "sampling": sampling,
        "mesh": [sp, tp],
        "expert_dtype": routed_expert_dtype().name,
        "flags": {k: v for k, v in os.environ.items() if k.startswith(("V41_", "TT_V41_"))},
    }
    spec = orc.real_spec(tuple(layers), max_seq, checkpoint=ckpt.root)
    root = weight_cache_dir(spec, mesh_device.shape)

    t0 = time.time()
    engram, engram_hash = build_engram_modules(mesh_device, C, ckpt, layers, tokenizer)
    top = ckpt.read(["embed.weight", "norm.weight", "head.weight"])
    model = TtV41Transformer(
        mesh_device,
        C,
        layers,
        lambda layer, include_moe: (load_layer if include_moe else load_layer_dense)(ckpt, layer),
        top["embed.weight"],
        top["norm.weight"],
        top["head.weight"],
        max_seq_len=max_seq,
        chunk=chunk,
        engram=engram,
        engram_hash=engram_hash,
        weight_cache_path=root,
        dspark_weights=seeding_weights(ckpt, C),
    )
    record["model_build_s"] = round(time.time() - t0, 1)
    record["dram_after_model"] = _dram(mesh_device)
    t0 = time.time()
    decoder = TtV41VerifyDecoder(
        model,
        root,
        users_per_rank=users,
        trace=True,
        warmup=warmup,
        sample_on_device=True,
        max_seq_len=cap,
        real_rows=real_rows,
    )
    record["decoder_build_s"] = round(time.time() - t0, 1)
    record["dram_after_decoder"] = _dram(mesh_device)
    record["ring_slots"] = decoder.ring_slots
    t0 = time.time()
    drafter = TtDSparkDrafter(
        mesh_device,
        C,
        ckpt,
        model.embedding,
        None,
        WEIGHT_CACHE,
        routed_expert_dtype(),
        shared_head=model.head,
        trace=True,
        warmup=warmup,
        ring_slots=decoder.ring_slots if fused or live > 1 else 128,
        users=users if live > 1 else 1,
    )
    if fused:
        decoder.attach_drafter(drafter)
    elif live > 1:
        drafter.gather_sp = decoder.pool.gather_sp  # the served round's one readback (attach_drafter sets it there)
        # the merged index selection's index pages for the live rows' layouts (attach_drafter builds the
        # one-slot ones)
        decoder.prepare_merge(
            [
                tuple(row for slot in rank0 for row in decoder.block_rows(slot % users, k, one_slot=False))
                for k in [0] + ks
            ]
        )
    record["drafter_build_s"] = round(time.time() - t0, 1)
    record["drafter_build_detail_s"] = drafter.build_s
    record["dram_after_drafter"] = _dram(mesh_device)
    if record["dram_after_drafter"] and record["dram_after_decoder"]:
        record["drafter_dram_bytes"] = (
            record["dram_after_drafter"]["allocated"] - record["dram_after_decoder"]["allocated"]
        )
        record["dram_margin_before_drafter"] = record["dram_after_decoder"]["free"]
        record["dram_margin_after_drafter"] = record["dram_after_drafter"]["free"]

    # --- the prefill (rings seeded) ---
    t0 = time.time()
    logits, state = model.prefill(prompt, logit_positions=1)
    first = int(logits[-1].argmax())
    record["prefill_s"] = round(time.time() - t0, 1)
    record["first_token"] = first
    assert state.dspark_rings is not None, "the prefill did not seed the drafter's rings"
    # V41_VR_TAPS=streams: the drafter's tap from the step's block outputs (eight eager ops a round); step
    # (default): the step computes every row's tap in its graph and the drafter selects the accepted row's in
    # its own
    taps_mode = os.environ.get("V41_VR_TAPS", "step")
    tap_layers = list(C.DSPARK_TARGET_LAYER_IDS)
    stream_layers = [l - 1 for l in tap_layers]  # a tap is the block input of its layer: the previous block's output
    streams = {}
    if taps_mode == "streams":
        decoder.on_block = lambda layer, x: streams.__setitem__(layer, x) if layer in stream_layers else None

    def rejoin():
        for slot in slots:
            decoder.join(slot, prompt, state)
        for u in draft_users:  # every rank's ring of user u (the rings are replicated: the lockstep slots)
            drafter.load_rings_from_device(state.dspark_rings, start=prompt_len, user=u)

    def draft_entries(tok, frontier, taps):
        """The standalone call's entries, one block a live user of a rank (the same on every rank: lockstep);
        ``taps``: the ``(tap row, position)`` pairs of the rows just committed (slot 0's rows: alike on every
        slot), written to each slot's own ring before its block."""
        return [[{"user": u, "token": tok, "position": frontier - 1, "taps": taps} for u in draft_users]] * sp

    def step(block):
        """The step over every live slot with the same block -> slot 0's argmaxes (every slot's must be the
        same:
        the slots are in lockstep on one prompt; under sampling every slot draws its own and slot 0's are
        taken,
        the slots stay in lockstep because the block and the committed count are slot 0's for every slot)."""
        out = decoder.decode(
            {slot: list(block) for slot in slots},
            {slot: (sampling["k"], sampling["p"], sampling["t"]) for slot in slots} if sampling else None,
        )
        assert sampling or all(out[slot] == out[0] for slot in slots), {slot: out[slot] for slot in slots}
        return out[0]

    # --- plain: one row a step ---
    rejoin()
    plain, plain_steps = [first], []
    tok = first
    for i in range(n_tokens):
        t0 = time.time()
        out = step([tok])
        wall = time.time() - t0
        for slot in slots:
            decoder.commit(slot, 0)
        tok = out[0]
        plain.append(tok)
        entry = {"i": i, "wall_s": round(wall, 4), **decoder.last}
        plain_steps.append(entry)
        logger.info(f"plain step {json.dumps(entry)}")
    replays = [e for e in plain_steps if e["mode"] == "replay"]
    record["plain"] = {
        "text": tokenizer.decode(plain),
        "tokens": plain,
        "steps": plain_steps,
        "replay_wall_s_min": min(e["wall_s"] for e in replays) if replays else None,
        "replay_wall_s_median": sorted(e["wall_s"] for e in replays)[len(replays) // 2] if replays else None,
        "replay_device_s_min": min(e["device_s"] for e in replays) if replays else None,
    }
    logger.info(f"plain text: {record['plain']['text']!r}")

    # --- speculative rounds ---
    record["rounds"] = {}
    mismatches = []
    for k in ks:
        rejoin()
        committed, rounds = [first], []
        tok, frontier = first, prompt_len  # the next token to process and its position
        tap = None  # the frontier - 1 tap: the prefill's seeding wrote its slot
        tap_row = None
        taps = []  # (tap row, position) of the rows the last round committed: the standalone call writes them
        if fused:
            # the first drafts: one standalone draft call (no ring write: the prefill seeded the slot)
            t0 = time.time()
            drafter.stage(drafter.host_inputs(tok, None, frontier - 1, write=False))
            drafts = drafter.run(tok)[:k]
            first_draft_s = round(time.time() - t0, 4)
            logger.info(f"K={k} first drafts {drafts} in {first_draft_s} s ({drafter.last})")
        while len(committed) - 1 < n_tokens:
            t0 = time.time()
            if fused:
                t1 = t2 = t3 = t0
            else:
                if taps_mode == "streams":
                    inputs = drafter.host_inputs(tok, None, frontier - 1, write=tap is not None, tap_row=tap_row or 0)
                    tap_dev = (
                        drafter.tap_rows_from_streams([streams[l] for l in stream_layers], tap_row)
                        if tap is not None
                        else None
                    )
                    drafter.stage(inputs, tap_dev)
                    if tap_dev is not None:
                        ttnn.deallocate(tap_dev)
                else:
                    # the accepted rows' taps (all of them: the reference's ring) then the
                    # block of every live user of the rank, one call
                    inputs = drafter.host_inputs_blocks(draft_entries(tok, frontier, taps))
                    drafter.stage(inputs, tap32=decoder.taps if tap is not None else None)
                t1 = time.time()
                if read_ranks:  # the served round's readback: no read in the call, every rank read after it
                    if prepare_next and len(draft_users) in drafter.captures:
                        # the served round's overlap: the next step's position half while
                        # the draft replay runs
                        drafter.run_blocks([tok] * len(draft_users), read=False, wait=False)
                        decoder.prepare_next({slot: [0] * (1 + k) for slot in slots})
                        drafter.wait()
                    else:
                        drafter.run_blocks([tok] * len(draft_users), read=False)
                    by_block = drafter.read_ranks(drafter.out)[0]
                else:
                    by_block = drafter.run_blocks([tok] * len(draft_users))
                drafts = by_block[0][:k]
                if not sampling and any(b[:k] != drafts for b in by_block):
                    logger.warning(f"K={k} the blocks' drafts differ: {by_block}")
                t2 = time.time()
                t3 = time.time()
            block = [tok] + drafts
            argmax = step(block)
            t4 = time.time()
            a, emitted = decoder.accept(block, argmax)
            for slot in slots:
                decoder.commit(slot, a)
            committed += emitted
            tok = emitted[-1]
            tap, tap_row = True, decoder.rows[0][a]
            taps = [(decoder.rows[0][i], frontier + i) for i in range(a + 1)]
            frontier += a + 1
            if fused:
                drafts = decoder.drafts[0][a][:k]  # the drafts of the accepted outcome, out of the same replay
            t5 = time.time()
            entry = {
                "frontier": frontier - a - 1,
                "block": block,
                "argmax": argmax,
                "accepted": a,
                "emitted": emitted,
                "wall_s": round(t5 - t0, 4),
                "draft_inputs_s": round(t1 - t0, 4),
                "draft_s": round(t2 - t1, 4),
                "draft_mode": decoder.last.get("mode") if fused else drafter.last.get("mode"),
                "draft_blocks": None if fused else drafter.last.get("blocks"),
                "draft_device_s": None if fused else drafter.last.get("device_s"),
                "draft_read_s": None if fused else drafter.last.get("read_s"),
                "verify_s": round(t4 - t3, 4),
                "verify_mode": decoder.last.get("mode"),
                "verify_device_s": decoder.last.get("device_s"),
                "verify_inputs_s": decoder.last.get("inputs_s"),
                "verify_stage_s": decoder.last.get("stage_s"),
                "verify_read_s": decoder.last.get("read_s"),
                "commit_s": round(t5 - t4, 4),
            }
            if "items" in decoder.last:  # V41_DECODE_INPUT_TIMING=1: the host inputs' items (ms)
                entry["verify_items"] = decoder.last["items"]
            if not fused and "items" in drafter.last:  # the standalone draft call's items (ms)
                entry["draft_items"] = drafter.last["items"]
            if fused:
                entry["next_drafts"] = decoder.drafts[0]
            rounds.append(entry)
            logger.info(f"K={k} round {json.dumps(entry)}")
        committed = committed[: n_tokens + 1]
        steady = [r for r in rounds if r["draft_mode"] == "replay" and r["verify_mode"] == "replay"]
        n_steady = sum(len(r["emitted"]) for r in steady)
        summary = {
            "text": tokenizer.decode(committed),
            "tokens": committed,
            "rounds": rounds,
            "n_rounds": len(rounds),
            "accepted_per_round": round(sum(r["accepted"] for r in rounds) / len(rounds), 3),
            "accepted_hist": {a: sum(r["accepted"] == a for r in rounds) for a in range(k + 1)},
            "tokens_per_round": round(sum(len(r["emitted"]) for r in rounds) / len(rounds), 3),
            "steady_rounds": len(steady),
            "steady_wall_s_per_round": round(sum(r["wall_s"] for r in steady) / len(steady), 4) if steady else None,
            "steady_s_per_token": round(sum(r["wall_s"] for r in steady) / n_steady, 4) if n_steady else None,
            "steady_draft_s": round(sum(r["draft_s"] for r in steady) / len(steady), 4) if steady else None,
            "steady_verify_s": round(sum(r["verify_s"] for r in steady) / len(steady), 4) if steady else None,
            "steady_verify_device_s": (
                round(sum(r["verify_device_s"] for r in steady) / len(steady), 4) if steady else None
            ),
            "steady_host_s": (
                round(sum(r["wall_s"] - r["draft_s"] - r["verify_s"] for r in steady) / len(steady), 4)
                if steady
                else None
            ),
            "matches_plain": committed == plain,
        }
        if fused:
            summary["first_draft_s"] = first_draft_s
        if committed != plain and not sampling:
            n = next((i for i, (x, y) in enumerate(zip(committed, plain)) if x != y), min(len(committed), len(plain)))
            mismatches.append(f"K={k}: committed text differs from the plain text at token {n}")
        record["rounds"][str(k)] = summary
        logger.info(f"K={k}: {json.dumps({kk: v for kk, v in summary.items() if kk not in ('rounds', 'tokens')})}")
    record["dram_after_rounds"] = _dram(mesh_device)
    record["mismatches"] = mismatches
    decoder.release()
    drafter.release()
    record["rss_peak_gb"] = round(_rss_gb(), 1)
    logger.info("V41_VR_RESULT " + json.dumps(record))
    out_path = os.environ.get("V41_VR_OUT")
    if out_path:
        with open(out_path, "w") as f:
            json.dump(record, f, indent=1)
    assert not mismatches, "\n".join(mismatches)
