# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DEVICE tests of the serving decode contract (change 1.6S): Qwen36ForCausalLM driven like the vLLM TT plugin.

``PluginDriver`` vendors the two pieces of ``vllm_tt_plugin.async_decode.TTAsyncDecodeController`` that decide what the
model sees (``plan_decode_reload`` and the decode kwargs assembly; no vllm import) and drives
``Qwen36ForCausalLM.decode_forward`` in three modes:

* ``host_reload``: every step is a full authoritative reload, synchronous (the reference);
* ``sync``: resident plan (reload only on transitions), result read right after the submit;
* ``async``: resident plan, ``read_from_device=False`` + ``read_decode_output(async_read=True)``, the events of step k
  are waited for only AFTER step k+1 was submitted; on steady steps the host token / position arguments are POISONED
  (the contract says they are stale then), so a model that reads them corrupts the stream and the GDN state.

Scenarios: B=1 greedy 64 steps; 3 active users in a width-4 batch with an idle row; a user finishing (layout change +
slot remap + reload, bucket 4 -> 2); KV page-table growth (reload_page_table only); top-k / top-p with a fixed seed.
Asserts identical token streams across the modes, equal GDN recurrent / conv state after the run, device positions ==
start + steps with idle rows at -1, and (separate test) measures TPOT with an injected 2 ms host delay.

Run (needs the 4-chip TP mesh and the weights, see HF_MODEL):
    pytest models/demos/blackhole/qwen36/tests/test_vllm_decode_contract.py -x -s
    QWEN36_SERVE_DEVICE_DECODE=0 QWEN36_CONTRACT_DUMP=/tmp/ref.json pytest ... -k host_reload   # pre-1.6S reference
    QWEN36_SERVE_DEVICE_DECODE=1 QWEN36_CONTRACT_REF=/tmp/ref.json pytest ...                  # compare to it
"""

import json
import os
import statistics
import time
from dataclasses import dataclass, field

import pytest
import torch
from loguru import logger

import ttnn
from models.common.sampling.sampling_params import SamplingParams
from models.demos.blackhole.qwen36.tests.test_decode_bucketing import _parametrize_traced
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM

N_LAYERS = int(os.environ.get("QWEN36_CONTRACT_LAYERS", "8"))
BLOCK = 64
CTX = 2048
BPU = CTX // BLOCK  # blocks per user
WIDTH = 4  # serving batch width (max_num_seqs)
PROMPT_LEN = 70
POISON_TOKEN = 4242
POISON_POS_SHIFT = 977
HOST_DELAY_S = 0.002

# ---------------------------------------------------------------------------------------------------------------------
# vendored plugin logic
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class ReloadPlan:
    reload_inputs: bool
    reload_page_table: bool
    reload_sampling_params: bool
    reset_sampling_state: bool

    def __post_init__(self):
        assert not (self.reset_sampling_state and not self.reload_inputs)
        assert not (self.reload_page_table and self.reload_inputs)


@dataclass
class User:
    uid: int
    max_new: int  # decode steps this user takes
    sampling: dict
    prompt: torch.Tensor = None
    first_token: int = 0
    out: list = field(default_factory=list)  # sampled tokens produced by decode steps (applied on the host)
    submitted: int = 0  # decode steps submitted so far (the scheduler's count; the host token list lags in async)


class PluginDriver:
    def __init__(self, gen, kv_cache, mode, vocab, host_delay_s=0.0):
        assert mode in ("host_reload", "sync", "async")
        self.gen, self.kv_cache, self.mode, self.vocab, self.delay = gen, kv_cache, mode, vocab, host_delay_s
        self.supports_resident = (
            bool(gen.model_capabilities.get("supports_async_decode", False)) and mode != "host_reload"
        )
        self.chain_valid = False
        self.prev_device_sampling = None
        self.submitted_pt = None
        self.prev_layout = None
        self.pending = None  # (host_out, events, row_uids) of the in-flight async step
        self.step_times = []
        self.last_layout = None

    # -- vendored: TTAsyncDecodeController.plan_decode_reload ------------------------------------------------------
    def plan_decode_reload(self, device_sampling, layout_changed, page_tables_changed):
        sampling_mode_changed = self.prev_device_sampling is not None and self.prev_device_sampling != device_sampling
        transition = (not self.chain_valid) or layout_changed or sampling_mode_changed
        reload_inputs = (not device_sampling) or transition or (not self.supports_resident)
        sampling_reset = device_sampling and transition
        return ReloadPlan(
            reload_inputs=reload_inputs,
            reload_page_table=(not reload_inputs and page_tables_changed),
            reload_sampling_params=sampling_reset,
            reset_sampling_state=sampling_reset,
        )

    # -- helpers ---------------------------------------------------------------------------------------------------
    @staticmethod
    def page_row(uid, pos):
        """Allocated blocks of user uid when its next KV position is pos (unallocated entries are 0)."""
        row = torch.zeros(BPU, dtype=torch.int32)
        n = pos // BLOCK + 1
        row[:n] = torch.arange(uid * BPU, uid * BPU + n, dtype=torch.int32)
        return row

    def sampling_params(self, rows):
        pad = dict(temperature=0.0, top_k=1, top_p=1.0, seed=None)
        full = [rows[i].sampling if i < len(rows) else pad for i in range(WIDTH)]
        return SamplingParams(
            temperature=[r["temperature"] for r in full],
            top_k=[r["top_k"] for r in full],
            top_p=[r["top_p"] for r in full],
            seed=[r["seed"] for r in full],
        )

    def finalize(self):
        """Wait for the in-flight async step and apply its tokens to the host request state."""
        if self.pending is None:
            return
        host_out, events, rows = self.pending
        self.pending = None
        for ev in events:
            ttnn.event_synchronize(ev)
        toks = self.gen.process_decode_output_host(host_out, is_tokens=True)
        toks = toks[0] if isinstance(toks, tuple) else toks
        self._apply(rows, toks)

    @staticmethod
    def _apply(rows, toks):
        toks = torch.as_tensor(toks).reshape(-1)
        for i, u in enumerate(rows):
            u.out.append(int(toks[i]))

    # -- one decode submission ---------------------------------------------------------------------------------------
    def step(self, users, step_idx):
        rows = [u for u in users if u.max_new > step_idx]  # front-packed (condensed) active rows
        layout = tuple(u.uid for u in rows)
        layout_changed = layout != self.prev_layout
        remap = None
        if layout_changed and self.prev_layout is not None:
            # Row i of the new layout takes the state at the slot its request had in the previous layout.
            old_slot = {uid: s for s, uid in enumerate(self.prev_layout)}
            order = [old_slot[uid] for uid in layout if uid in old_slot]
            if len(order) == len(layout) and order != list(range(len(layout))):
                rest = [s for s in range(WIDTH) if s not in order]
                remap = order + rest
        if layout_changed:
            self.finalize()  # drain: apply every pending token before the layout moves

        tokens = torch.zeros(WIDTH, 1, dtype=torch.int32)
        start_pos = torch.full((WIDTH,), -1, dtype=torch.int32)
        pt = torch.zeros(WIDTH, BPU, dtype=torch.int32)
        for i, u in enumerate(rows):
            pos = PROMPT_LEN + u.submitted  # KV position this step writes (scheduler count, always exact)
            pt[i] = self.page_row(u.uid, pos)
            start_pos[i] = pos
            tokens[i, 0] = u.out[-1] if u.out else u.first_token
        pt_changed = self.submitted_pt is None or not torch.equal(pt, self.submitted_pt)
        plan = self.plan_decode_reload(True, layout_changed, pt_changed)
        if not plan.reload_inputs:
            # Contract: tokens / start_pos are stale (one step behind). Poison them; page_table stays current.
            tokens[: len(rows), 0] = POISON_TOKEN
            start_pos[: len(rows)] += POISON_POS_SHIFT
        elif self.mode == "async":
            assert self.pending is None, "a reload must be preceded by a drain"

        kwargs = dict(
            tokens=tokens,
            page_table=pt,
            kv_cache=self.kv_cache,
            start_pos=start_pos,
            sampling_params=self.sampling_params(rows),
            reload_inputs=plan.reload_inputs,
            reload_page_table=plan.reload_page_table,
            reload_sampling_params=plan.reload_sampling_params,
            reset_sampling_state=plan.reset_sampling_state,
        )
        if remap is not None:
            kwargs["slot_remap"] = remap
        t0 = time.perf_counter()
        if self.mode == "async":
            tt_out = self.gen.decode_forward(**kwargs, enable_trace=True, read_from_device=False)
        else:
            out = self.gen.decode_forward(**kwargs, enable_trace=True, read_from_device=True)
        # accepted: commit the plugin-side residency / ownership state
        self.chain_valid = True
        self.prev_device_sampling = True
        self.prev_layout = layout
        self.last_layout = rows
        if plan.reload_inputs or plan.reload_page_table:
            self.submitted_pt = pt.clone()
        for u in rows:
            u.submitted += 1

        if self.mode == "async":
            host_out, events = self.gen.read_decode_output(tt_out, async_read=True)
            prev, self.pending = self.pending, (host_out, events, rows)
            time.sleep(self.delay)  # host work (scheduling, detokenize...) overlaps this step's device time
            if prev is not None:  # step k-1 is finalized only after step k is on the device
                for ev in prev[1]:
                    ttnn.event_synchronize(ev)
                toks = self.gen.process_decode_output_host(prev[0], is_tokens=True)
                self._apply(prev[2], toks[0] if isinstance(toks, tuple) else toks)
        else:
            toks = out[0] if isinstance(out, tuple) else out
            self._apply(rows, toks)
            time.sleep(self.delay)
        self.step_times.append(time.perf_counter() - t0)


# ---------------------------------------------------------------------------------------------------------------------
# model / scenario plumbing
# ---------------------------------------------------------------------------------------------------------------------

GREEDY = dict(temperature=0.0, top_k=1, top_p=1.0, seed=None)


def _sampled(uid):
    return dict(temperature=0.8, top_k=20, top_p=0.9, seed=1000 + uid)


def _build(mesh_device):
    model = Qwen36Model.from_pretrained(
        mesh_device,
        max_batch_size=WIDTH,
        max_seq_len=CTX,
        n_layers=N_LAYERS,
        hf_model=os.environ.get("HF_MODEL", "Qwen/Qwen3.6-27B"),
    )
    gen = Qwen36ForCausalLM([model], [model.args], mesh_device)
    num_blocks = WIDTH * BPU
    kv_shape = (num_blocks, model.args.n_local_kv_heads, BLOCK, model.args.head_dim)
    kv_cache = gen.allocate_kv_cache(kv_shape, ttnn.bfloat16, N_LAYERS)
    # Plugin order (model_runner.py): Phase 1 eager warmup, reset flag, Phase 2 traced prefill then traced decode.
    dkw = dict(kv_cache=kv_cache, max_batch_size=WIDTH, num_blocks=BPU, can_sample_on_device=True)
    gen.warmup_model_prefill(kv_cache=kv_cache, enable_trace=False)
    gen.warmup_model_decode(enable_trace=False, **dkw)
    gen.already_warmed_up_prefill = False
    gen.warmup_model_prefill(kv_cache=kv_cache, enable_trace=True)
    gen.warmup_model_decode(enable_trace=True, **dkw)
    return gen, kv_cache


def _zero_decode_gdn_state(model):
    """Zero the batched (decode-binding) GDN state in place: recurrent, conv_states and the RM conv history."""
    for layer in model.layers:
        if layer.is_full_attention:
            continue
        dn = layer.attention
        targets = [dn.rec_state] + list(dn.conv_states)
        if getattr(dn, "_conv_hist_rm", None) is not None:
            targets.append(dn._conv_hist_rm)
        for t in targets:
            z = ttnn.zeros_like(t)
            ttnn.copy(z, t)
            ttnn.deallocate(z)
        if getattr(dn, "_fused_decode", False):
            dn._conv_fmt = "both"  # every conv format zeroed


def _prefill(gen, kv_cache, users):
    """Prefill every user into slot == index and take the greedy first token from the prefill logits."""
    model = gen.model[0]
    _zero_decode_gdn_state(model)
    g = torch.Generator().manual_seed(7)
    prompts = torch.randint(1000, 20000, (len(users), PROMPT_LEN), generator=g, dtype=torch.int32)
    page = torch.stack([torch.arange(u.uid * BPU, (u.uid + 1) * BPU, dtype=torch.int32) for u in users])
    logits, _ = gen.prefill_forward(
        prompts, page, kv_cache, [PROMPT_LEN] * len(users), empty_slots=list(range(len(users))), enable_trace=True
    )
    for i, u in enumerate(users):
        u.prompt = prompts[i]
        u.first_token = int(logits[i].reshape(-1)[: model.args.vocab_size].argmax())
        u.out, u.submitted = [], 0


def _gdn_state(model):
    """Per-tensor entries (name, batch_axis, [per-device shards])."""
    snap = []
    for li, layer in enumerate(model.layers):
        if layer.is_full_attention:
            continue
        dn = layer.attention
        named = [("rec_state", 0, dn.rec_state)]
        named += [(f"conv_states[{j}]", 1, c) for j, c in enumerate(dn.conv_states)]
        hist = getattr(dn, "_conv_hist_rm", None)
        if hist is not None:
            named.append(("_conv_hist_rm", 0, hist))
        for nm, axis, t in named:
            shards = [ttnn.to_torch(x).float() for x in ttnn.get_device_tensors(t)]
            if not all(s.shape[axis] == WIDTH for s in shards):
                print(f"GDN state layout: layer {li} {nm} shard shapes {[tuple(s.shape) for s in shards]}")
                pytest.fail(f"layer {li} {nm}: dim {axis} of a device shard != WIDTH={WIDTH}")
            snap.append((f"layer{li}.{nm}", axis, shards))
    return snap


def _device_positions(gen, bucket):
    inputs = gen._bucket_trace_store[bucket][1][True][0]  # [tokens, cur_pos, rope_idx, page_table]
    return ttnn.to_torch(ttnn.get_device_tensors(inputs[1])[0]).reshape(-1).tolist()


def _run(gen, kv_cache, mode, users, n_steps, delay=0.0):
    _prefill(gen, kv_cache, users)
    drv = PluginDriver(gen, kv_cache, mode, gen.model[0].args.vocab_size, host_delay_s=delay)
    for s in range(n_steps):
        drv.step(users, s)
    drv.finalize()
    ttnn.synchronize_device(gen.mesh_device)
    result = dict(
        tokens={u.uid: list(u.out) for u in users},
        state=_gdn_state(gen.model[0]),
        step_times=drv.step_times,
        last_layout=[u.uid for u in drv.last_layout],
        users=users,
        positions=_device_positions(gen, drv_bucket(drv)) if mode != "host_reload" else None,
    )
    return result


def drv_bucket(drv):
    n = max(1, len(drv.last_layout))
    return min(WIDTH, 1 << max(0, (n - 1).bit_length()))


def _scenario(name):
    if name == "b1_greedy_64":
        return [User(0, 64, GREEDY)], 64
    if name == "three_users_idle_row":  # 3 active rows in a width-4 batch: bucket 4 with one idle (-1) row
        return [User(i, 40, GREEDY) for i in range(3)], 40
    if name == "user_finishes_remap":  # user 1 stops after 20 steps: condense (remap) + reload, bucket 4 -> 2
        return [User(0, 48, GREEDY), User(1, 20, GREEDY), User(2, 48, GREEDY)], 48
    if name == "page_table_growth":  # prompt 70 -> position 128 crosses a block boundary (64): reload_page_table only
        return [User(0, 70, GREEDY), User(1, 70, GREEDY)], 70
    if name == "topk_topp_seeded":
        return [User(i, 40, _sampled(i)) for i in range(3)], 40
    raise KeyError(name)


SCENARIOS = ["b1_greedy_64", "three_users_idle_row", "user_finishes_remap", "page_table_growth", "topk_topp_seeded"]


def _dump_or_compare(name, tokens):
    dump, ref = os.environ.get("QWEN36_CONTRACT_DUMP"), os.environ.get("QWEN36_CONTRACT_REF")
    if dump:
        data = json.load(open(dump)) if os.path.exists(dump) else {}
        data[name] = {str(k): v for k, v in tokens.items()}
        json.dump(data, open(dump, "w"))
    if ref and os.path.exists(ref):
        want = json.load(open(ref)).get(name)
        if want is not None:
            assert {str(k): v for k, v in tokens.items()} == want, f"{name}: differs from the pre-1.6S reference"


@torch.no_grad()
@_parametrize_traced()
@pytest.mark.parametrize("name", SCENARIOS)
def test_serving_decode_modes_agree(mesh_device, reset_seeds, ensure_gc, name):
    """host_reload == sync == async: tokens, GDN state, device positions (idle rows at -1)."""
    gen, kv = _build(mesh_device)
    if not gen.model_capabilities.get("supports_async_decode"):
        pytest.skip("QWEN36_SERVE_DEVICE_DECODE=0: no resident decode to verify (use -k host_reload for the reference)")
    results = {}
    for mode in ("host_reload", "sync", "async"):
        users, n_steps = _scenario(name)
        results[mode] = _run(gen, kv, mode, users, n_steps)
        logger.info(f"{name}/{mode}: {len(results[mode]['tokens'][0])} tokens for user 0")
    _dump_or_compare(name, results["host_reload"]["tokens"])

    ref = results["host_reload"]
    for mode in ("sync", "async"):
        got = results[mode]
        assert got["tokens"] == ref["tokens"], f"{name}: {mode} token stream differs from the host-reload reference"
        assert len(got["state"]) == len(ref["state"])
        assert (
            ref["last_layout"] == got["last_layout"]
        ), f"{name}/{mode}: layout {got['last_layout']} != ref {ref['last_layout']}"
        n_live = len(got["last_layout"])
        n_idle_diff, idle_max = 0, 0.0
        for i, ((nm, ax, ta), (_, _, tb)) in enumerate(zip(got["state"], ref["state"])):
            idle_differs = False
            for d, (a_d, b_d) in enumerate(zip(ta, tb)):
                a_l, b_l = a_d.narrow(ax, 0, n_live), b_d.narrow(ax, 0, n_live)
                assert torch.equal(
                    a_l, b_l
                ), f"{name}/{mode}: GDN state tensor {i} ({nm}) device {d} live rows differ (max abs {(a_l - b_l).abs().max()})"
                if n_live < WIDTH:
                    m = float(
                        (a_d.narrow(ax, n_live, WIDTH - n_live) - b_d.narrow(ax, n_live, WIDTH - n_live)).abs().max()
                    )
                    if m > 0:
                        idle_differs = True
                        idle_max = max(idle_max, m)
            n_idle_diff += idle_differs
        logger.info(
            f"{name}/{mode}: idle-row GDN state (rows {n_live}:{WIDTH}): {n_idle_diff}/{len(got['state'])} tensors differ, max abs diff {idle_max}"
        )
        # Device-resident position after the last step: start + steps for the live rows, -1 for the idle rows.
        users = got["users"]
        live = [u for u in users if u.uid in got["last_layout"]]
        pos = got["positions"]
        for row, uid in enumerate(got["last_layout"]):
            u = next(x for x in live if x.uid == uid)
            assert pos[row] == PROMPT_LEN + u.submitted, f"{name}/{mode}: row {row} position {pos[row]}"
        assert all(p == -1 for p in pos[len(got["last_layout"]) :]), f"{name}/{mode}: idle rows advanced: {pos}"


@torch.no_grad()
@_parametrize_traced()
def test_host_reload_reference_only(mesh_device, reset_seeds, ensure_gc):
    """Runs under QWEN36_SERVE_DEVICE_DECODE=0 too: produces / checks the QWEN36_CONTRACT_DUMP / _REF streams."""
    gen, kv = _build(mesh_device)
    for name in SCENARIOS:
        users, n_steps = _scenario(name)
        res = _run(gen, kv, "host_reload", users, n_steps)
        _dump_or_compare(name, res["tokens"])
        assert all(len(t) == u.max_new for u, t in zip(users, res["tokens"].values()))


@torch.no_grad()
@_parametrize_traced()
def test_async_tpot_with_host_delay(mesh_device, reset_seeds, ensure_gc):
    """B=1 greedy, 64 steps, 2 ms of injected host work per step: async must hide it (TPOT async <= sync)."""
    gen, kv = _build(mesh_device)
    if not gen.model_capabilities.get("supports_async_decode"):
        pytest.skip("QWEN36_SERVE_DEVICE_DECODE=0")
    tpot = {}
    for mode in ("sync", "async"):
        users, n_steps = _scenario("b1_greedy_64")
        res = _run(gen, kv, mode, users, n_steps, delay=HOST_DELAY_S)
        steady = res["step_times"][4:]
        tpot[mode] = (statistics.median(steady) * 1e3, statistics.mean(steady) * 1e3)
        logger.info(f"TPOT {mode}: p50 {tpot[mode][0]:.2f} ms  mean {tpot[mode][1]:.2f} ms (host delay 2 ms)")
    logger.info(
        f"async - sync: p50 {tpot['async'][0] - tpot['sync'][0]:+.2f} ms, mean {tpot['async'][1] - tpot['sync'][1]:+.2f} ms"
    )
    assert tpot["async"][1] <= tpot["sync"][1] * 1.02, tpot
