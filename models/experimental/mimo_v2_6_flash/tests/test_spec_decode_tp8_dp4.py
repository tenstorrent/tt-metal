# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MTP speculative decoding through the vLLM adapter (generator_vllm.py) on TP=8 x DP=4, without vLLM.

Lane 0 decodes a prompt greedily one token per step; lane 1 serves the same prompt the way the plugin drives a
speculating request: a plain step, then propose (MTP drafts) / verify (the request in every slot of its replica) /
greedy acceptance. The speculative tokens must equal the plain greedy tokens, and the drafts must be accepted at a rate
only a working drafter reaches (MIN_MEAN_ACCEPTED). Logs the mean accepted length and the time per token of both
loops. Needs vllm-tt-plugin (its spec_decode value types) and the MiMo checkpoint. The KV cache is laid out as vLLM
lays out the adapter's specs (``vllm_kv_layout``): one global and five SWA groups, each with its own block table.
Runs with the decode RoPE tables (the decode, verify and draft traces read their cos / sin from them): whole rows, and
compact ones as past 128K context (the 64 rotated dims plus the 1 / 0 tail); and without them (MIMO_DEVICE_ROPE=0: the
host uploads the cos / sin beside the packed inputs).

The first verifies also run as the plugin's other modes on the same block: logits (its argmax must be the ids), and
fused_sample (acceptance on device, tt/verify_sampler.py): with every row greedy it must commit what the ids walk
commits, and with the request sampled (temperature 1, top_p 0.95) and given uniforms it must commit what the plugin's
host walk (vllm_tt_plugin.spec_accept.accept_speculated_tokens) commits on the logits verify's logits with the same
uniforms, unless the device flagged the request's candidates truncated.

The drafts are chained behind the verify on device (tt/spec_chain.py): with the decode RoPE tables every propose after
a verify must return the drafts the verify's result page carried (only the first, after the plain step, is drafted
from the host plan), and the first ones are drafted again from the host plan (MIMO_SPEC_CHAIN_CHECK) and must be the
same tokens. Without the tables every draft is host-planned.

test_chained_spec_decode_tp8_dp4: the plugin's chained verifies (async scheduling, docs/SPEC_DECODE_CONTRACT.md 4j)
on lane 1: a deferred fused_sample verify, then each verify chained onto the one in flight (its block built on device
by the relay step from the pending step's result page, at positions the host only bounds), the older one read on
another thread and proposed after the newer one's submission. The chained greedy tokens must equal the plain greedy
ones, and each readout's p0 the request's real position. With both drafters (the relay reads pages the 3-layer program
wrote, anchor in slot 3) and the backfill on: only the first draft (host-planned, after the prompt) backfills.

Both drafters run: every MTP layer (MIMO_MTP_LAYERS=3, draft i + 1 from layer i, the passes planned on device from the
stash; the chain checks then also cover a draft planned again on the stash its chained twin left) and one layer
(MIMO_MTP_LAYERS=1, the drafter before it). Every-layer drafting needs the decode RoPE tables.

With the decode RoPE tables the target-hidden history and the ring backfill run (MIMO_MTP_BACKFILL, tt/mtp_backfill.py):
the first draft after the prompt backfills the rings from the prompt's last rows; the test also runs once without it
(the mean accepted length of both is logged), and when both ran in one session the drafts after the backfill must be
accepted at least as often as without it.
"""

import math
import threading
import time
from types import SimpleNamespace

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

from models.experimental.mimo_v2_6_flash.reference.config import MiMoConfig
from models.experimental.mimo_v2_6_flash.reference.spec_decode import (
    MAX_DRAFTS,
    SPEC_SLOTS,
    accept_greedy,
    planned_drafts,
)
from models.experimental.mimo_v2_6_flash.tests.common import DP, GALAXY_DEVICE_PARAMS, model_path
from models.experimental.mimo_v2_6_flash.tt import generator_vllm
from models.experimental.mimo_v2_6_flash.tt.generator_vllm import BLOCK_SIZE, MiMoV2ForCausalLM
from models.experimental.mimo_v2_6_flash.tt.attention import verify_grouped

pytest.importorskip("vllm_tt_plugin")

SPEC_DEVICE_PARAMS = {**GALAXY_DEVICE_PARAMS, "trace_region_size": 1_200_000_000}
PER_LANE = max(4, SPEC_SLOTS)  # a verify runs in SPEC_SLOTS decode slots of the lane
# The prefill warmup lays a lane's first-chunk batches (PER_LANE users of up to TtMiMoModel.PREFILL_BATCH_MAX_PADDED,
# 512 tokens) along one page-table row, so the row must hold PER_LANE * 512 positions.
MAX_SEQ_LEN = max(2048, PER_LANE * 512)
STEPS = 96
# The first verifies also run in logits and fused_sample mode and must agree with argmax_ids and the host walk.
LOGITS_CHECKS = 4
PROMPT = (
    "Write a Python function that returns the n-th Fibonacci number using memoization, then explain how the "
    "memoization changes its time complexity."
)
# Committed tokens per verify (the bonus token plus the accepted drafts) the drafter must average: drafts that never
# match commit 1.0, and greedy MTP commits 2.79 per verify on SPEED-Bench coding at c=1. One accepted draft per verify
# fails a broken drafter and leaves margin for this one short coding prompt.
MIN_MEAN_ACCEPTED = 2.0 if MAX_DRAFTS >= 3 else 1.5  # one draft (MIMO_SPEC_GROUPS=4): accepted half the time
# The mean accepted length of each (MTP layers, RoPE tables, backfill) case run in this session.
MEAN_ACCEPTED = {}


# The grouped verify attention (tt/attention.py _verify_grouped, on by default with groups of 2 slots) rounds
# differently from the plain decode's, so where the plain step's top logits tie to within bf16 rounding the verify can
# pick the other token. A divergence passes only there: the speculative token scores within TIE_STEPS bf16 steps of the
# plain step's best logit at that position (as common.near_top); every token before it matched.
# The grouped verify runs only under the KV pair split; without it (the per-slot verify) the tokens must match plain
# greedy exactly.
TIE_STEPS = 2


def near_tie(first_diff, seq, plain_top):
    if first_diff is None:
        return True
    if not (generator_vllm.KV_PAIR_SPLIT and verify_grouped()):
        return False
    if first_diff not in plain_top:  # the prefill's token
        return False
    top = plain_top[first_diff]
    best = float(top.values[0])
    step = 2.0 ** (math.floor(math.log2(abs(best))) - 7)
    scores = dict(zip(top.indices.tolist(), top.values.tolist()))
    logger.info(
        f"divergence at {first_diff}: speculative {seq[first_diff]}, plain top {list(scores.items())}, bf16 step {step}"
    )
    return seq[first_diff] in scores and scores[seq[first_diff]] >= best - TIE_STEPS * step


def vllm_kv_layout(cfg):
    """(KV cache group, buffer) of each layer as vLLM lays out get_kv_cache_spec's specs: its hybrid manager
    (vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_page_size) deals each attention kind's layers, layer 0's
    kind first, round-robin into groups of at most as many layers as the smaller kind has (9 global and 39 SWA layers:
    group 0 global, groups 1-5 SWA), and the i-th layer of every group shares buffer i. Every layer of a group is of
    one kind: the model makes one kv.PrefillTables per group for its first layer's kind (TtMiMoModel._prefill_tables),
    which gives a later chunk's SWA layers their window blocks only in a SWA group."""
    kinds = {}
    for i in range(cfg.num_hidden_layers):
        kinds.setdefault(cfg.is_swa(i), []).append(i)
    size = min(len(layers) for layers in kinds.values())
    groups, buffers = [None] * cfg.num_hidden_layers, [None] * cfg.num_hidden_layers
    first = 0
    for layers in kinds.values():
        n = -(-len(layers) // size)
        for k in range(n):
            for t, i in enumerate(layers[k::n]):
                groups[i], buffers[i] = first + k, t
        first += n
    return groups, buffers


def kv_cache_and_tables(model, cfg, groups, buffers, rows, width):
    """The KV cache (allocate_kv_cache_per_layer) and each group's block table [rows, width]. Block 0 is vLLM's null
    block; the groups share buffers, so their blocks are disjoint, and every row has blocks of its own, so a step that
    reads another slot's or lane's row reads blocks its request never wrote. Returns the cache, group 0's table (the
    plugin's page_table) and each layer's group table (its page_tables_per_layer). With the KV pair split
    (generator_vllm.KV_PAIR_SPLIT) the global group's blocks hold 128 positions: its rows use their first half and
    the rest is zeros, as the plugin pads every group's rows to one width."""
    num_groups = max(groups) + 1
    num_blocks = 1 + num_groups * rows * width
    specs = []
    pair_groups = set()
    for i in range(cfg.num_hidden_layers):
        dims = cfg.attention_dims(i)
        block = BLOCK_SIZE
        if generator_vllm.KV_PAIR_SPLIT and not cfg.is_swa(i):
            block = 2 * BLOCK_SIZE
            pair_groups.add(groups[i])
        specs.append(((num_blocks, 1, block, dims.head_dim), torch.bfloat16, buffers[i]))
    kv = model.allocate_kv_cache_per_layer(specs)
    group_tables = 1 + torch.arange(num_blocks - 1, dtype=torch.int32).reshape(num_groups, rows, width)
    for g in pair_groups:
        group_tables[g, :, -(-width // 2) :] = 0
    return kv, group_tables[0], [group_tables[g] for g in groups]


@pytest.mark.parametrize("backfill", [True, False], ids=["backfill", "no_backfill"])
@pytest.mark.parametrize("mtp_layers", [3, 1], ids=["mtp3", "mtp1"])
@pytest.mark.parametrize(
    "device_rope", ["whole", "compact", None], ids=["rope_tables", "compact_rope_tables", "host_rope"]
)
@pytest.mark.parametrize("device_params", [SPEC_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_spec_decode_tp8_dp4(mesh_device, device_rope, mtp_layers, backfill, monkeypatch):
    if mtp_layers not in (1, MAX_DRAFTS):
        pytest.skip(f"{mtp_layers} MTP layers draft {mtp_layers} tokens; the groups hold {MAX_DRAFTS}")
    if planned_drafts(mtp_layers) and device_rope is None:
        pytest.skip("the planned MTP program plans its passes on device: it needs the decode RoPE tables")
    if not backfill and (mtp_layers == 1 or device_rope != "whole"):
        pytest.skip("without the backfill: once (3 MTP layers, whole RoPE tables)")
    path = model_path()
    cfg = MiMoConfig.from_model_path(path)
    monkeypatch.setattr(generator_vllm, "SPEC_DECODE", "mtp")
    monkeypatch.setattr(generator_vllm, "MTP_LAYERS", mtp_layers)
    monkeypatch.setattr(generator_vllm, "MTP_BACKFILL", backfill)
    monkeypatch.setattr(generator_vllm, "DEVICE_ROPE", device_rope is not None)
    if device_rope == "compact":  # the 1M entry's tables at this test's short context
        monkeypatch.setattr(generator_vllm, "ROPE_TABLE_FULL_WIDTH_MAX_LEN", 0)
    groups, buffers = vllm_kv_layout(cfg)
    monkeypatch.setattr(MiMoV2ForCausalLM, "_layer_groups", groups)
    monkeypatch.setattr(MiMoV2ForCausalLM, "prefill_token_budget", 2048)

    rows = DP * PER_LANE
    width = MAX_SEQ_LEN // BLOCK_SIZE
    hf = SimpleNamespace(num_hidden_layers=cfg.num_hidden_layers, _name_or_path=path)
    model = MiMoV2ForCausalLM.initialize_vllm_model(hf, mesh_device, rows, MAX_SEQ_LEN, tt_data_parallel=DP)
    kv, page_table, layer_tables = kv_cache_and_tables(model, cfg, groups, buffers, rows, width)
    model.warmup_model_prefill(kv)
    # The plugin's two passes: eager, then trace capture.
    model.warmup_model_decode(kv, enable_trace=False)
    model.warmup_model_decode(kv, enable_trace=True)

    prompt = AutoTokenizer.from_pretrained(path)(PROMPT, return_tensors="pt")["input_ids"][0].to(torch.int32)
    plain_row, spec_row = 0, PER_LANE  # lane 0 slot 0, lane 1 slot 0
    plain_top = {}  # position -> the plain step's top logits there (near_tie)

    def prefill(row):
        # Row i of a prefill's tables is its i-th request's (prefill_forward): this one-request batch sends the slot's
        # rows, which its decode steps and verifies then read by slot.
        logits = model.prefill_forward(
            prompt[None],
            torch.zeros(1, dtype=torch.int32),
            torch.tensor([prompt.numel()]),
            page_table[row : row + 1],
            kv,
            empty_slots=[row],
            page_tables_per_layer=[t[row : row + 1] for t in layer_tables],
        )
        return int(logits[0, -1].argmax())

    def decode(row, token, pos):
        tokens = torch.zeros(rows, 1, dtype=torch.int32)
        positions = torch.full((rows,), -1, dtype=torch.int32)
        tokens[row, 0], positions[row] = token, pos
        logits = model.decode_forward(
            tokens, positions, page_table, kv, enable_trace=True, page_tables_per_layer=layer_tables
        )
        if row == plain_row:
            plain_top[pos + 1] = logits[row, 0].float().topk(4)
        return int(logits[row, 0].argmax())

    # Plain greedy on lane 0.
    plain = prompt.tolist() + [prefill(plain_row)]
    t0 = time.perf_counter()
    while len(plain) < prompt.numel() + STEPS:
        plain.append(decode(plain_row, plain[-1], len(plain) - 1))
    plain_s = time.perf_counter() - t0

    # Speculative greedy on lane 1: a plain step, then propose / verify / accept.
    seq = prompt.tolist() + [prefill(spec_row)]
    t0 = time.perf_counter()
    seq.append(decode(spec_row, seq[-1], len(seq) - 1))
    committed, accepted, counts, check_s = [seq[-1]], 1, [], 0.0

    def propose():
        tok = torch.zeros(rows, 1 + MAX_DRAFTS, dtype=torch.int32)
        pos = torch.full((rows, 1 + MAX_DRAFTS), -1, dtype=torch.int32)
        tok[spec_row, : len(committed)] = torch.tensor(committed, dtype=torch.int32)
        pos[spec_row, : len(committed)] = torch.arange(len(seq) - len(committed), len(seq), dtype=torch.int32)
        out = model.propose_draft_tokens(MAX_DRAFTS, tok, pos, torch.ones(rows, dtype=torch.int32) * accepted)
        assert int(out.num_valid[spec_row]) == MAX_DRAFTS and int(out.num_valid.sum()) == MAX_DRAFTS
        return out.draft_token_ids[spec_row].tolist()

    def fused_verify(tokens, positions, valid, spec_sampling):
        out = model.decode_forward(
            tokens,
            positions,
            page_table,
            kv,
            page_tables_per_layer=layer_tables,
            spec_mode="fused_sample",
            num_valid_drafts=valid,
            accepted_counts=torch.ones(rows, dtype=torch.int32),
            **({} if spec_sampling is None else {"spec_sampling": spec_sampling}),
        )
        assert out.spec_mode == "fused_sample" and out.accepted_counts.shape == (rows,)
        return out

    fused_mismatch = []

    def check_fused_sampled(tokens, positions, valid, full):
        from vllm_tt_plugin.spec_accept import accept_speculated_tokens

        vocab = model.gen.cfg.vocab_size
        # Uniforms on the Q24 grid, so the device's floor(u * 2^24) and the host's float see one value.
        uniforms = torch.zeros(rows, MAX_DRAFTS + 2)
        uniforms[spec_row] = torch.randint(0, 1 << 24, (MAX_DRAFTS + 2,)).float() / (1 << 24)
        sampled = torch.zeros(rows, dtype=torch.bool)
        sampled[spec_row] = True
        spec_sampling = {
            "temperature": sampled.float(),
            "top_k": torch.where(sampled, vocab, 1).to(torch.int32),
            "top_p": torch.where(sampled, 0.95, 1.0).float(),
            "greedy": ~sampled,
            "uniforms": uniforms,
        }
        lane = model.models[spec_row // PER_LANE].verify_sampler
        truncated = lane.truncated
        out = fused_verify(tokens, positions, valid, spec_sampling)
        ref = accept_speculated_tokens(
            target_logits=full[spec_row : spec_row + 1],
            draft_token_ids=tokens[spec_row : spec_row + 1, 1:],
            num_valid_drafts=valid[spec_row : spec_row + 1],
            temperature=torch.ones(1),
            top_p=torch.tensor([0.95]),
            uniforms=uniforms[spec_row : spec_row + 1].double(),
        )
        got = out.accepted_token_ids[spec_row, : int(out.accepted_counts[spec_row])].tolist()
        want = ref.committed_token_ids[0, : int(ref.accepted_counts[0])].tolist()
        if got != want:
            fused_mismatch.append((got, want))
            assert lane.truncated > truncated, f"sampled fused {got} != host walk {want} on untruncated candidates"

    while len(seq) < prompt.numel() + STEPS:
        drafts = propose()
        p = len(seq) - 1
        tokens = torch.zeros(rows, 1 + MAX_DRAFTS, dtype=torch.int32)
        positions = torch.full((rows, 1 + MAX_DRAFTS), -1, dtype=torch.int32)
        tokens[spec_row] = torch.tensor([seq[-1]] + drafts, dtype=torch.int32)
        positions[spec_row] = torch.arange(p, p + 1 + MAX_DRAFTS, dtype=torch.int32)
        valid = torch.zeros(rows, dtype=torch.int32)
        valid[spec_row] = MAX_DRAFTS
        if len(counts) < LOGITS_CHECKS:
            # A logits verify (sampled requests) answers the same cells: each one's argmax is the argmax_ids
            # verify's token. It runs first, so the state the argmax verify leaves for the next step is unchanged.
            t = time.perf_counter()
            full = model.decode_forward(
                tokens,
                positions,
                page_table,
                kv,
                page_tables_per_layer=layer_tables,
                spec_mode="logits",
                num_valid_drafts=valid,
                accepted_counts=torch.ones(rows, dtype=torch.int32),
            ).logits
            assert full.shape == (rows, 1 + MAX_DRAFTS, model.gen.cfg.vocab_size) and bool(full.isfinite().all())
            logits_argmax = full[spec_row].argmax(dim=-1)
            fused_greedy = fused_verify(tokens, positions, valid, None)
            check_fused_sampled(tokens, positions, valid, full)
            check_s += time.perf_counter() - t
        out = model.decode_forward(
            tokens,
            positions,
            page_table,
            kv,
            page_tables_per_layer=layer_tables,
            spec_mode="argmax_ids",
            num_valid_drafts=valid,
            accepted_counts=torch.ones(rows, dtype=torch.int32),
        )
        committed = accept_greedy(out.argmax_ids[spec_row].tolist(), drafts, MAX_DRAFTS)
        model.chain_check = len(counts) < LOGITS_CHECKS  # the next propose checks the chained drafts
        if len(counts) < LOGITS_CHECKS:
            assert logits_argmax.tolist() == out.argmax_ids[spec_row].tolist(), "logits verify disagrees with ids"
            count = int(fused_greedy.accepted_counts[spec_row])
            assert fused_greedy.accepted_token_ids[spec_row, :count].tolist() == committed, "greedy fused != ids walk"
        accepted = len(committed)
        counts.append(accepted)
        seq += committed
    spec_s = time.perf_counter() - t0 - check_s

    n = prompt.numel() + STEPS
    stats = model.chain_stats
    logger.info(f"MTP spec decode chain: {stats}")
    assert stats["mismatched"] == 0, "a chained draft differs from the host plan's"
    if device_rope:
        assert stats["host"] == 1 and stats["chained"] == len(counts) - 1, "a propose after a verify drafted again"
        assert stats["checked"] == LOGITS_CHECKS
    else:
        assert stats["chained"] == 0 and stats["host"] == len(counts)
    # The first draft after the prompt backfills; the chained ones continue the rings.
    assert stats["backfills"] == (1 if backfill and device_rope else 0)
    mean = sum(counts) / len(counts)
    logger.info(
        f"MTP spec decode ({mtp_layers} MTP layers, backfill {backfill}): mean accepted "
        f"{mean:.2f} over "
        f"{len(counts)} verifies "
        f"(histogram {[counts.count(m) for m in range(1, MAX_DRAFTS + 2)]}); "
        f"{1e3 * plain_s / STEPS:.1f} ms/token plain, {1e3 * spec_s / (len(seq) - prompt.numel() - 1):.1f} ms/token "
        "speculative (drafts chained behind the verify, one read per verify); sampled fused checks differing on "
        "truncated candidates: "
        f"{len(fused_mismatch)} of {LOGITS_CHECKS}"
    )
    first_diff = next((i for i in range(prompt.numel(), n) if seq[i] != plain[i]), None)
    assert near_tie(first_diff, seq, plain_top), f"speculative tokens diverge from greedy at position {first_diff}"
    assert mean >= MIN_MEAN_ACCEPTED, f"mean accepted {mean:.2f} < {MIN_MEAN_ACCEPTED}: the drafts rarely match"
    # Without the backfill the rings hold zeros at the prompt's positions, which every draft of this run attends (the
    # prompt and STEPS tokens fit in the MTP layers' 128-position window): with the backfilled rings the drafts must be
    # accepted at least as often.
    MEAN_ACCEPTED[(mtp_layers, device_rope, backfill)] = mean
    twin = MEAN_ACCEPTED.get((mtp_layers, device_rope, not backfill))
    if twin is not None:
        with_backfill, without = (mean, twin) if backfill else (twin, mean)
        assert with_backfill >= without, f"mean accepted {with_backfill:.2f} with the backfill < {without:.2f} without"


@pytest.mark.parametrize("mtp_layers", [3, 1], ids=["mtp3", "mtp1"])
@pytest.mark.parametrize("device_params", [SPEC_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_chained_spec_decode_tp8_dp4(mesh_device, mtp_layers, monkeypatch):
    if mtp_layers not in (1, MAX_DRAFTS):
        pytest.skip(f"{mtp_layers} MTP layers draft {mtp_layers} tokens; the groups hold {MAX_DRAFTS}")
    path = model_path()
    cfg = MiMoConfig.from_model_path(path)
    monkeypatch.setattr(generator_vllm, "SPEC_DECODE", "mtp")
    monkeypatch.setattr(generator_vllm, "MTP_LAYERS", mtp_layers)
    monkeypatch.setattr(generator_vllm, "MTP_BACKFILL", True)
    monkeypatch.setattr(generator_vllm, "DEVICE_ROPE", True)
    monkeypatch.setattr(generator_vllm, "SPEC_DEFER", True)
    groups, buffers = vllm_kv_layout(cfg)
    monkeypatch.setattr(MiMoV2ForCausalLM, "_layer_groups", groups)
    monkeypatch.setattr(MiMoV2ForCausalLM, "prefill_token_budget", 2048)

    rows = DP * PER_LANE
    width = MAX_SEQ_LEN // BLOCK_SIZE
    hf = SimpleNamespace(num_hidden_layers=cfg.num_hidden_layers, _name_or_path=path)
    model = MiMoV2ForCausalLM.initialize_vllm_model(hf, mesh_device, rows, MAX_SEQ_LEN, tt_data_parallel=DP)
    kv, page_table, layer_tables = kv_cache_and_tables(model, cfg, groups, buffers, rows, width)
    model.warmup_model_prefill(kv)
    model.warmup_model_decode(kv, enable_trace=False)
    model.warmup_model_decode(kv, enable_trace=True)
    prompt = AutoTokenizer.from_pretrained(path)(PROMPT, return_tensors="pt")["input_ids"][0].to(torch.int32)
    plain_row, spec_row = 0, PER_LANE
    plain_top = {}  # position -> the plain step's top logits there (near_tie)
    k1 = 1 + MAX_DRAFTS

    def prefill(row):
        # Row i of a prefill's tables is its i-th request's (prefill_forward): this one-request batch sends the slot's
        # rows, which its decode steps and verifies then read by slot.
        logits = model.prefill_forward(
            prompt[None],
            torch.zeros(1, dtype=torch.int32),
            torch.tensor([prompt.numel()]),
            page_table[row : row + 1],
            kv,
            empty_slots=[row],
            page_tables_per_layer=[t[row : row + 1] for t in layer_tables],
        )
        return int(logits[0, -1].argmax())

    def decode(row, token, pos):
        tokens = torch.zeros(rows, 1, dtype=torch.int32)
        positions = torch.full((rows,), -1, dtype=torch.int32)
        tokens[row, 0], positions[row] = token, pos
        logits = model.decode_forward(
            tokens, positions, page_table, kv, enable_trace=True, page_tables_per_layer=layer_tables
        )
        if row == plain_row:
            plain_top[pos + 1] = logits[row, 0].float().topk(4)
        return int(logits[row, 0].argmax())

    plain = prompt.tolist() + [prefill(plain_row)]
    while len(plain) < prompt.numel() + STEPS:
        plain.append(decode(plain_row, plain[-1], len(plain) - 1))

    def verify(tokens, positions, valid, counts, **chain):
        return model.decode_forward(
            tokens,
            positions,
            page_table,
            kv,
            page_tables_per_layer=layer_tables,
            spec_mode="fused_sample",
            num_valid_drafts=valid,
            accepted_counts=counts,
            defer_read=True,
            **chain,
        ).handle

    def read(handle):
        box = []
        reader = threading.Thread(target=lambda: box.append(model.read_decode_output(handle)))
        reader.start()
        reader.join(timeout=120)
        assert box, "the deferred read did not finish"
        return box[0]

    # A plain step and its propose (host-planned draft), then a deferred verify built from host state.
    seq = prompt.tolist() + [prefill(spec_row)]
    seq.append(decode(spec_row, seq[-1], len(seq) - 1))
    tok = torch.full((rows, k1), -1, dtype=torch.int32)
    pos = torch.full((rows, k1), -1, dtype=torch.int32)
    tok[spec_row, 0], pos[spec_row, 0] = seq[-1], len(seq) - 1
    drafts = model.propose_draft_tokens(MAX_DRAFTS, tok, pos, torch.ones(rows, dtype=torch.int32))
    tokens = torch.zeros(rows, k1, dtype=torch.int32)
    positions = torch.full((rows, k1), -1, dtype=torch.int32)
    tokens[spec_row] = torch.tensor([seq[-1]] + drafts.draft_token_ids[spec_row].tolist(), dtype=torch.int32)
    positions[spec_row] = torch.arange(len(seq) - 1, len(seq) - 1 + k1, dtype=torch.int32)
    valid = torch.zeros(rows, dtype=torch.int32)
    valid[spec_row] = MAX_DRAFTS
    pending = verify(tokens, positions, valid, torch.ones(rows, dtype=torch.int32))
    counts, t0 = [], time.perf_counter()
    chain_rows = torch.zeros(rows, dtype=torch.bool)
    chain_rows[spec_row] = True
    accepted = torch.ones(rows, dtype=torch.int32)
    accepted[spec_row] = -1
    while True:
        newer = None
        if len(seq) < prompt.numel() + STEPS:
            # The plugin's provisional column 0: the applied length plus 1 + K for the verify in flight, minus one.
            bound = len(seq) - 1 + k1
            tokens = torch.full((rows, k1), -1, dtype=torch.int32)
            positions = torch.full((rows, k1), -1, dtype=torch.int32)
            positions[spec_row] = torch.arange(bound, bound + k1, dtype=torch.int32)
            newer = verify(tokens, positions, valid, accepted, chain_rows=chain_rows, chain_tip=pending.tip_id)
        readout = read(pending)
        m = int(readout.accepted_counts[spec_row])
        assert int(readout.p0[spec_row]) == len(seq) - 1, "the device's column 0 is the request's real position"
        committed = readout.accepted_token_ids
        seq += committed[spec_row, :m].tolist()
        counts.append(m)
        first = readout.p0.to(torch.int64).unsqueeze(1)
        committed_positions = (first + torch.arange(1, k1 + 1)).masked_fill(first < 0, -1).to(torch.int32)
        out = model.propose_draft_tokens(
            MAX_DRAFTS, committed, committed_positions, readout.accepted_counts, hidden=pending
        )
        assert int(out.num_valid[spec_row]) == MAX_DRAFTS, "the step's chained drafts come back with its propose"
        if newer is None:
            break
        pending = newer
    spec_s = time.perf_counter() - t0
    n = min(len(seq), len(plain))
    mean = sum(counts) / len(counts)
    logger.info(
        f"MTP chained spec decode: mean accepted {mean:.2f} over {len(counts)} verifies, "
        f"{1e3 * spec_s / (len(seq) - prompt.numel() - 2):.1f} ms/token; {model.chain_stats}"
    )
    assert model.chain_stats["uncovered"] == 0
    # The only host-planned draft (the first, after the prompt) backfilled the rings; every chained one continued them.
    assert model.chain_stats["backfills"] == 1
    first_diff = next((i for i in range(prompt.numel(), n) if seq[i] != plain[i]), None)
    assert near_tie(first_diff, seq, plain_top), f"chained speculative tokens diverge from greedy at {first_diff}"
    assert mean >= MIN_MEAN_ACCEPTED, f"mean accepted {mean:.2f} < {MIN_MEAN_ACCEPTED}: the drafts rarely match"
