# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""APC_CHUNKED_PREFILL_PLAN.md milestone M1, device test of the model side: three prompts of 5 chunks each are prefilled over several steps through the vLLM adapter,
interleaved the way the plugin's scheduler (``prefill_chunk_alignment`` = the chunk, a step budget of 3 chunks) hands them out: every non-final piece ends on a multiple of the chunk,
the continuation of a prompt lands on a different logical slot every step, new prompts start while others are part-way. Checks:

  (1) zero "recomputing from 0" (model) / "without carried state" (adapter) log lines: no prompt restarts,
  (2) the first-token logits of every prompt are bit-identical to the same prompt prefilled alone in ONE call, and the first greedy tokens of a joint decode agree,
  (3) negative control: a piece that ends off the chunk grid DOES make the next piece recompute from 0 (what the scheduler alignment prevents).

    DSV41_LAYERS=0-3 pytest models/demos/blackhole/deepseek_v41_flash/tests/test_apc_m1_chunked_continuation.py -s
Env: DSV41_M1_U (users per mesh row, 4), DSV41_M1_LEN (prompt lengths, comma list, default 2500,2400,2600), DSV41_M1_BUDGET (tokens per step, 1536).
"""

import os
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
from models.tt_transformers.tt.common import PagedAttentionConfig

U = int(os.environ.get("DSV41_M1_U", "4"))
CHUNK = 512
LENS = [int(x) for x in os.environ.get("DSV41_M1_LEN", "2500,2400,2600").split(",")]
BUDGET = int(os.environ.get("DSV41_M1_BUDGET", "1536"))
LAYERS = os.environ.get("DSV41_LAYERS", "0-3")
DECODE_STEPS = 4


def schedule(lens, budget, chunk):
    """What the plugin's TTScheduler does with ``prefill_chunk_alignment = chunk`` (continuations first, then new prompts in FIFO order, a non-final piece ends on a multiple of the chunk,
    a piece of 0 tokens is skipped): -> a list of steps, each a list of (prompt index, start, end)."""
    done = [0] * len(lens)
    steps = []
    while any(d < n for d, n in zip(done, lens)):
        left, step = budget, []
        order = [i for i in range(len(lens)) if 0 < done[i] < lens[i]] + [i for i in range(len(lens)) if done[i] == 0]
        for i in order:
            n = min(lens[i] - done[i], left)
            if done[i] + n < lens[i]:
                n = (done[i] + n) // chunk * chunk - done[i]
            if n <= 0:
                continue
            step.append((i, done[i], done[i] + n))
            done[i] += n
            left -= n
        steps.append(step)
    return steps


def test_scheduler_model_matches_the_plugin_rule():
    steps = schedule([2500, 2400, 2600], 1536, 512)
    assert steps == [
        [(0, 0, 1536)],
        [(0, 1536, 2500), (1, 0, 512)],
        [(1, 512, 2048)],
        [(1, 2048, 2400), (2, 0, 1024)],
        [(2, 1024, 2560)],
        [(2, 2560, 2600)],
    ]


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": int(os.environ.get("DSV41_IL_TRACE_REGION", "1600000000")),
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_chunked_continuation_through_the_adapter(mesh_device):
    os.environ.update(
        {
            "DSV41_PF_UMASK": "1",
            "DSV41_MO_OVERLAP": "0",  # (as initialize_vllm_model: the sub-device MoE overlap is not capturable inside the chunk trace)
            "DSV41_PREFILL_UP": "1",
            "DSV41_VLLM_INTERLEAVE": "1",
            "DSV41_VLLM_CHUNK": str(CHUNK),
        }
    )
    from models.demos.blackhole.deepseek_v41_flash.tt.generator import Generator
    from models.demos.blackhole.deepseek_v41_flash.tt.generator_vllm import DeepseekV41ForCausalLM

    md = mesh_device
    log = lambda m: print(m, flush=True)
    a, _, b = LAYERS.partition("-")
    layer_ids = list(range(int(a), int(b or a) + 1))
    B = 4 * U
    S_pad = -(-max(LENS) // CHUNK) * CHUNK
    max_ctx = S_pad + 256
    os.environ["DSV41_VLLM_S_PAD"] = str(S_pad)
    pp = default_page_params(max_ctx, U)
    args, m, _, _ = create_tt_model(
        md,
        B,
        max_ctx,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=layer_ids,
        log=log,
    )
    assert m.Up == 1, f"Up={m.Up}: the test needs DSV41_PREFILL_UP=1"
    ad = DeepseekV41ForCausalLM(Generator([m], [args], md), B, max_ctx)

    recompute = []
    orig_log = m.log

    def log_spy(msg):
        if "recomputing from" in str(msg):
            recompute.append(str(msg))
        return orig_log(msg)

    m.log = log_spy
    sink = logger.add(
        lambda msg: recompute.append(msg.record["message"])
        if "without carried state" in msg.record["message"]
        else None
    )

    g = torch.Generator().manual_seed(4321)
    prompts = [torch.randint(1000, 100000, (n,), generator=g, dtype=torch.int64) for n in LENS]
    sp = SimpleNamespace(temperature=[0.0] * B, top_k=[1] * B, enable_log_probs=[False] * B)

    def call(rows, slots):
        """one prefill_forward call: rows [(prompt idx, start, end)] on logical slots ``slots`` -> logits [N, 1, V]"""
        L = max(e for _, _, e in rows)
        toks = torch.zeros(len(rows), L, dtype=torch.int32)
        for r, (i, st, e) in enumerate(rows):
            toks[r, :e] = prompts[i][:e].to(torch.int32)
        return ad.prefill_forward(
            toks,
            prompt_lens=[e for _, _, e in rows],
            start_pos=[st for _, st, _ in rows],
            empty_slots=slots,
            sampling_params=None,
        )

    def joint_decode(slot_of, first):
        cur, pos, out_tokens = dict(first), {i: LENS[i] for i in slot_of}, {i: [first[i]] for i in slot_of}
        for _ in range(DECODE_STEPS):
            tk = torch.zeros(B, 1, dtype=torch.int32)
            ps = torch.full((B,), -1, dtype=torch.long)
            for i, s in slot_of.items():
                tk[s, 0], ps[s] = cur[i], pos[i]
            out = ad.decode_forward(tk, ps, sampling_params=sp)
            for i, s in slot_of.items():
                cur[i], pos[i] = int(out[s, 0]), pos[i] + 1
                out_tokens[i].append(cur[i])
        return out_tokens

    # ---- (1) interleaved, aligned pieces -----------------------------------------------------------------------------------------------------
    steps = schedule(LENS, BUDGET, CHUNK)
    log(f"schedule (budget {BUDGET}): {steps}")
    assert all(e % CHUNK == 0 or e == LENS[i] for step in steps for i, _, e in step)
    free = list(range(B - 1, -1, -1))  # logical slots handed out from the top
    last_slot, first_logits = {}, {}
    n_windows = 0
    for k, step in enumerate(steps):
        slots = []
        for i, st, e in step:
            slots.append(
                free.pop() if st == 0 else free.pop(0 if len(free) > 1 else 0)
            )  # a different free slot than before
        lg = call(step, slots)
        n_windows += sum(-(-(e - st) // CHUNK) for _, st, e in step)
        for r, (i, st, e) in enumerate(step):
            last_slot[i] = slots[r]
            if e == LENS[i]:
                first_logits[i] = lg[r, 0].float().clone()
        log(f"step {k}: {step} slots {slots} replays {m.timing.get('replays')} total {m.timing.get('total', 0):.2f} s")
    assert len(first_logits) == len(LENS)
    chunk_first = {i: int(first_logits[i].argmax()) for i in first_logits}
    chunk_tokens = joint_decode(last_slot, chunk_first)
    assert not recompute, f"prompts restarted from position 0: {recompute}"
    log(
        f"(1) OK: no recompute over {len(steps)} steps, {n_windows} windows (5 chunks x 3 prompts = {sum(-(-n // CHUNK) for n in LENS)})"
    )

    # ---- (2) the same prompts alone, one call each ---------------------------------------------------------------------------------------------
    for i in range(len(LENS)):
        ad.release_request(last_slot[i])
    solo_logits, solo_slot = {}, {}
    for i, n in enumerate(LENS):
        lg = call([(i, 0, n)], [i])
        solo_logits[i], solo_slot[i] = lg[0, 0].float().clone(), i
    solo_first = {i: int(solo_logits[i].argmax()) for i in solo_logits}
    solo_tokens = joint_decode(solo_slot, solo_first)
    for i in range(len(LENS)):
        same = torch.equal(first_logits[i], solo_logits[i])
        d = (first_logits[i] - solo_logits[i]).abs().max().item()
        log(
            f"(2) prompt {i} ({LENS[i]} tokens): first-token logits bit-identical to the one-call prefill {same} (max abs diff {d:.3g}); first token {chunk_first[i]} vs {solo_first[i]}"
        )
        log(f"    greedy tokens chunked {chunk_tokens[i]} vs solo {solo_tokens[i]}")
    assert all(
        torch.equal(first_logits[i], solo_logits[i]) for i in range(len(LENS))
    ), "chunked first-token logits differ from the one-call prefill"
    assert (
        chunk_tokens == solo_tokens
    ), "decode after the chunked prefill differs from decode after the one-call prefill"

    # ---- (3) negative control: a piece that ends off the chunk grid --------------------------------------------------------------------------
    for i in range(len(LENS)):
        ad.release_request(solo_slot[i])
    recompute.clear()
    n = LENS[0]
    call([(0, 0, 1300)], [5])  # ends off the grid: the carried state is not resumable
    call([(0, 1300, n)], [6])
    assert recompute, "the control did not show the restart it should"
    log(f"(3) OK: an off-grid piece restarts from 0 as expected: {recompute[:1]}")
    logger.remove(sink)
