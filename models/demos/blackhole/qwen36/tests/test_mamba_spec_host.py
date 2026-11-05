# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the MTP spec class's GDN state in vLLM MambaSpec spec blocks (QWEN36_MAMBA_SPEC=1, K = 3).

A real TTScheduler (vllm-tt-plugin) over a real KVCacheManager schedules requests with 3 drafts per step. The KV cache
groups come from the model's own specs (tt/kv_cache_spec.qwen36_kv_cache_spec, num_speculative_blocks = 3). Each step
the plugin's MambaCacheGroups gives the copies before the forward (begin_step), each row's 1 + K window and start count
(step_inputs), and the copies after acceptance (after_acceptance), the calls TTModelRunner makes.

The model is played at page level: tt/spec_block_state.SpecBlockBook builds the ctrl pages and ring_page_idx rows,
MTPSpecEngine._stage_gdn_windows scatters the plugin's rows to engine rows, and a fake device runs gdn_spec_step's page
contract on them (start state from the ctrl page's start pages, the state after token t to ring_page_idx column t, HOLD
rows untouched, the conv window in the layer's private window). Block copies, conv writes into blocks and prefill
resumes run as the model runs them (copy_plan, take_pending, MTPSpecEngine._paged_plan). A state is a label of the
tokens it has read; every read of a recurrent state page and of a conv history must find exactly the tokens the row has
computed.

Needs vLLM 0.31 and vllm-tt-plugin (src/ on PYTHONPATH), and the ttnn host stub for the model import:
    PYTHONPATH=<worktree>:<plugin>/src:<tools>/ttnn-host-stub pytest --noconftest \
        models/demos/blackhole/qwen36/tests/test_mamba_spec_host.py
"""

import json
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm.v1.core.sched.scheduler")
pytest.importorskip("vllm_tt_plugin.mamba_state")

from vllm.config import (  # noqa: E402
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.utils.hashing import sha256  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
    get_request_block_hasher,
    init_none_hash,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.kv_cache_interface import MambaSpec  # noqa: E402
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput  # noqa: E402
from vllm.v1.request import Request  # noqa: E402
from vllm.v1.structured_output import StructuredOutputManager  # noqa: E402
from vllm_tt_plugin.config import store_tt_spec_plan  # noqa: E402
from vllm_tt_plugin.mamba_state import mamba_cache_groups  # noqa: E402
from vllm_tt_plugin.scheduler import TTScheduler  # noqa: E402
from vllm_tt_plugin.spec_admission import MODEL_OWNED_DRAFT_METHOD, MODEL_OWNED_DRAFT_SENTINEL  # noqa: E402
from vllm_tt_plugin.spec_decode import ACCEPT_MODE_ARGMAX_IDS, DRAFTER_STATE_INTERNAL, SpecPlan  # noqa: E402

from models.demos.blackhole.qwen36.tests.test_kv_cache_spec_host import TEXT_CONFIG  # noqa: E402
from models.demos.blackhole.qwen36.tt import spec_block_state as sbs  # noqa: E402
from models.demos.blackhole.qwen36.tt.gdn.tp import spec_ctrl_words  # noqa: E402
from models.demos.blackhole.qwen36.tt.kv_cache_spec import qwen36_kv_cache_spec  # noqa: E402
from models.demos.blackhole.qwen36.tt.spec_engine import MTPSpecEngine  # noqa: E402

K = 3
T = K + 1
B = 4  # engine rows (max_num_seqs)
HOLD = 0xFFFFFFFF
_QWEN2 = {
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 16,
    "intermediate_size": 32,
    "max_position_embeddings": 65536,
    "model_type": "qwen2",
    "num_attention_heads": 2,
    "num_hidden_layers": 1,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06,
    "rope_theta": 10000.0,
    "torch_dtype": "float16",
    "vocab_size": 128,
    "bos_token_id": 1,
    "eos_token_id": 2,
}


def _label(tokens):
    return hash(tuple(tokens))


JUNK = "junk"  # a state written by a padding verify row (a draft the step did not carry)


# --------------------------------------------------------------------------------------------------- page layouts
# (TP, block, n_banks, layout): the Galaxy primary (BF8 1024, 8 banks, two pools), its 7-bank Bin6 variant (896), the
# k_only variant (2048), and the P150x4 TP=4 test layouts.
GEOMETRIES = [
    (8, 1024, 8, "two_pool"),
    (8, 896, 7, "two_pool"),
    (8, 2048, 8, "k_only"),
    (4, 2048, 8, "two_pool"),
    (4, 4096, 8, "k_only"),
]


def _layout(tp, block, n_banks, name, num_blocks=200):
    nv, C = 48 // tp, (2 * 16 * 128 + 48 * 128) // tp
    S = sbs.kv_block_bytes_per_bank([num_blocks, 1, block, 256], "bfloat8_b", n_banks)
    return sbs.SpecBlockLayout(S, n_banks, num_blocks, nv, sbs.conv_carry_pages(4, C), name)


@pytest.mark.parametrize("geo", GEOMETRIES, ids=lambda g: f"tp{g[0]}_b{g[1]}_{g[2]}banks_{g[3]}")
def test_layout_pages_lie_inside_their_block_and_never_overlap(geo):
    lay = _layout(*geo)
    for b in range(lay.num_blocks):
        lay.check_block(b)  # inside the block's bytes on every bank, heads and history disjoint
    heads = lay.head_pages(torch.arange(lay.num_blocks))
    assert bool(((heads % sbs.HALF_PAGES) == 0).all()), "head pages must start on a [Dk / 2, Dv] view block"
    k_side = heads < lay.pool1_pages
    assert bool((k_side.sum(-1) == lay.heads_in_k).all())
    if geo[3] == "k_only":
        assert lay.heads_in_k == lay.nv


@pytest.mark.parametrize(
    "geo",
    [(8, 1024, 8, "k_only"), (4, 1024, 8, "two_pool"), (8, 1024, 7, "two_pool")],
    ids=["tp8_1024_k_only", "tp4_1024", "tp8_1024_7banks"],
)
def test_layout_refuses_a_block_that_cannot_hold_the_state(geo):
    with pytest.raises(ValueError):  # allow-pytest.raises: host test, runs with --noconftest (no expect_error)
        _layout(*geo)


def test_ctrl_page_words_follow_the_op_contract():
    lay = _layout(8, 1024, 8, "two_pool")
    blocks = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24], [0, 0, 0, 0], [41, 42, 43, 44]])
    start = torch.tensor([2, 0, 0, 3])
    sw = sbs.StepWindows([blocks], start)
    words = spec_ctrl_words(B, lay.nv)
    page = sbs.ctrl_page_paged([2, 0, 0, 3], sw.init_pages(lay, 0), 1, words, hold=[2]).reshape(-1)
    u32 = page.to(torch.int64) & 0xFFFFFFFF
    assert int(u32[0]) == 1
    assert u32[1 : 1 + B].tolist() == [2, 0, 0, 3]
    init = u32[1 + B : 1 + B + B * lay.nv].reshape(B, lay.nv)
    assert init[0].tolist() == lay.head_pages(torch.tensor([13]))[0].tolist()
    assert init[3].tolist() == lay.head_pages(torch.tensor([44]))[0].tolist()
    assert init[2].tolist() == [HOLD] * lay.nv
    idx = sw.ring_page_idx(lay, 0)
    assert tuple(idx.shape) == (B, T * lay.nv)
    for t in range(T):
        assert idx[1, t * lay.nv : (t + 1) * lay.nv].tolist() == lay.head_pages(torch.tensor([21 + t]))[0].tolist()


def test_chain_body_twin_builds_the_host_ctrl_pages():
    # The chain body (QWEN36_SPEC_CHAIN) gathers each group's start pages from its ring_page_idx at column mi'; for a
    # chained step that is the host's page with start column mi and no held row.
    from models.demos.blackhole.qwen36.tt.spec_chain import chain_ctrl_rows

    lay = _layout(4, 2048, 8, "two_pool")
    book = sbs.SpecBlockBook(lay, 3, B, T, "none")
    g = torch.Generator().manual_seed(0)
    blocks = [torch.randperm(60, generator=g)[: B * T].reshape(B, T) + 1 for _ in range(3)]
    book.set_step(blocks, torch.zeros(B, dtype=torch.int64), range(B))
    words = spec_ctrl_words(B, lay.nv)
    for mi in ([0, 1, 2, 3], [3, 3, 0, 1], [2, 0, 0, 0]):
        host = book.ctrl_rows(mi, 1, words, hold=(), start_col=mi).to(torch.int64) & 0xFFFFFFFF
        twin = chain_ctrl_rows(mi, 1, B=B, K=K, Nv=lay.nv, ctrl_words=words, ring_idx=book.idx_host)
        assert [r.tolist() for r in host] == [r.tolist() for r in twin]
    assert max(int(r.max()) for r in book.idx_host) < 1 << 24  # exact in the body's fp32 math


# --------------------------------------------------------------------------------------------------- the fake device
class _FakeDevice:
    """Pages of every GDN group as labels: the head pages (first page of each head) and the conv history page of a
    block; the private conv window per engine row (the label after each verify token); the decode taps per row."""

    def __init__(self, lay, G):
        self.lay, self.G = lay, G
        self.rec = {}  # (group, first page of a head) -> label
        self.conv = {}  # (group, first conv page) -> label
        self.win = {}  # engine row -> [label after token t] of its latest verify
        self.taps = {}  # engine row -> label the prefill left in the decode conv taps
        self.reads = 0

    def _heads(self, b):
        return self.lay.head_pages(torch.tensor([b]))[0].tolist()

    def write_block(self, g, b, lab):
        for p in self._heads(b):
            self.rec[(g, p)] = lab
        self.conv[(g, self.lay.conv_page(b))] = lab

    def read_block(self, g, b):
        heads = {self.rec.get((g, p)) for p in self._heads(b)}
        assert len(heads) == 1, f"group {g} block {b}: heads disagree {heads}"
        return heads.pop(), self.conv.get((g, self.lay.conv_page(b)))

    def copy(self, src, dst):
        for g in range(self.G):
            for ps, pd in zip(self._heads(src), self._heads(dst)):
                self.rec[(g, pd)] = self.rec.get((g, ps))
            self.conv[(g, self.lay.conv_page(dst))] = self.conv.get((g, self.lay.conv_page(src)))

    def zero(self, b):
        for g in range(self.G):
            self.write_block(g, b, None)

    def run_ops(self, ops):
        for op in ops:
            if op[0] == "copy":
                self.copy(op[1], op[2])
            else:
                _, u, t, g, dst = op
                self.conv[(g, self.lay.conv_page(dst))] = self.win[u][t]

    def verify(self, book, mi, rows, expect, labels):
        """gdn_spec_step's page contract over every group: per (u, h) the start state at ctrl word 1 + B + u * Nv + h
        (HOLD = untouched row), the state after token t to ring_page_idx[u, t * Nv + h]; the conv carry from the
        private window at mi[u]. ``expect[u]`` the label the start state must hold, ``labels[u]`` the T written."""
        words = spec_ctrl_words(B, self.lay.nv)
        nv = self.lay.nv
        for g in range(self.G):
            page = book.ctrl_rows(mi, 0, words, hold=[u for u in range(B) if u not in rows])[g].to(torch.int64)
            page &= 0xFFFFFFFF
            init = page[1 + B : 1 + B + B * nv].reshape(B, nv)
            for u in range(B):
                if u not in rows:
                    assert init[u].tolist() == [HOLD] * nv
                    continue
                got = {self.rec.get((g, int(p))) for p in init[u]}
                assert got == {expect[u]}, (g, u, got, expect[u])
                assert int(page[1 + u]) == mi[u]
                self.reads += 1
            for u in rows:
                for t in range(T):
                    for h in range(nv):
                        self.rec[(g, int(book.idx_host[g][u, t * nv + h]))] = labels[u][t]
        for u in rows:
            assert self.win[u][mi[u]] == expect[u], (u, mi[u], self.win[u], expect[u])
            self.win[u] = list(labels[u])


# --------------------------------------------------------------------------------------------------- the harness
class _EngineStub:
    """Just what MTPSpecEngine._stage_gdn_windows / _paged_plan read: the slot map, B, T and the model's block state."""

    _to_rows = MTPSpecEngine._to_rows
    _set_slot_map = MTPSpecEngine._set_slot_map
    _stage_gdn_windows = MTPSpecEngine._stage_gdn_windows
    _paged_plan = MTPSpecEngine._paged_plan

    def __init__(self, book, block_size):
        self.B, self.T = B, T
        self.block_size = block_size
        self.book = book
        self.model = SimpleNamespace(_spec_blocks=None)
        self._paged_step_rows = None
        self._set_slot_map(range(B))

    def _paged_state(self):
        return self.book


class _BookAsState(sbs.SpecBlockBook):
    """SpecGdnBlockState's host surface over a SpecBlockBook (the engine calls these on the state)."""

    def set_step(self, blocks, start_col, live_rows, boundary=()):
        super().set_step(blocks, start_col, live_rows, boundary)


class _Harness:
    def __init__(
        self,
        tmp_path,
        mode,
        tp=8,
        block=1024,
        n_banks=8,
        layout="two_pool",
        num_blocks=96,
        retention=0,
        max_model_len=8192,
        budget=2048,
    ):
        (tmp_path / "config.json").write_text(json.dumps(_QWEN2))
        model_config = ModelConfig(model=str(tmp_path), dtype="float16", seed=0, skip_tokenizer_init=True)
        model_config.max_model_len = max_model_len
        cfg = VllmConfig(
            model_config=model_config,
            scheduler_config=SchedulerConfig(
                max_num_seqs=B,
                max_num_batched_tokens=budget,
                max_model_len=max_model_len,
                enable_chunked_prefill=True,
                async_scheduling=False,
                is_encoder_decoder=False,
            ),
            cache_config=CacheConfig(block_size=block, cache_dtype="auto"),
            device_config=DeviceConfig(device="cpu"),
        )
        cfg.scheduler_config.disable_hybrid_kv_cache_manager = False
        cfg.cache_config.kv_cache_layout = "LBNHC"
        cfg.cache_config.enable_prefix_caching = mode == "align"
        cfg.cache_config.mamba_cache_mode = mode
        cfg.cache_config.prefix_cache_retention_interval = retention
        mamba_bs = -(-max_model_len // block) * block if mode == "none" else block
        specs = qwen36_kv_cache_spec(
            TEXT_CONFIG,
            tp,
            block,
            kv_bf8=True,
            num_speculative_blocks=K,
            mamba_cache_mode=mode,
            mamba_block_size=mamba_bs,
        )
        groups = get_kv_cache_groups(cfg, specs)
        page = next(iter(specs.values())).page_size_bytes
        self.kv = get_kv_cache_config_from_groups(cfg, groups, 16 * page * num_blocks + 1)
        assert self.kv.num_blocks == num_blocks
        cfg.cache_config.num_gpu_blocks = num_blocks
        cfg.speculative_config = SpeculativeConfig(
            method=MODEL_OWNED_DRAFT_METHOD,
            model=MODEL_OWNED_DRAFT_SENTINEL,
            num_speculative_tokens=K,
            target_model_config=cfg.model_config,
            target_parallel_config=cfg.parallel_config,
        )
        store_tt_spec_plan(
            cfg,
            SpecPlan(
                effective_k=K,
                lanes_per_request=1,
                extra_bytes_per_seq=0,
                extra_bytes_per_token=0,
                accept_modes=(ACCEPT_MODE_ARGMAX_IDS,),
                drafter_state=DRAFTER_STATE_INTERNAL,
                supports_narrow_decode=False,
            ),
        )
        bs, hash_bs = resolve_kv_cache_block_sizes(self.kv, cfg)
        cfg.cache_config.block_size = min(g.kv_cache_spec.block_size for g in self.kv.kv_cache_groups)
        self.block = block
        self.mode = mode
        self.scheduler = TTScheduler(
            vllm_config=cfg,
            kv_cache_config=self.kv,
            block_size=bs,
            hash_block_size=hash_bs,
            log_stats=True,
            structured_output_manager=StructuredOutputManager(cfg),
        )
        self.groups = mamba_cache_groups(self.kv.kv_cache_groups, cfg.cache_config, max_model_len)
        self.mamba = [i for i, g in enumerate(self.kv.kv_cache_groups) if isinstance(g.kv_cache_spec, MambaSpec)]
        self.max_model_len = max_model_len
        lay = _layout(tp, block, n_banks, layout, num_blocks)
        self.book = _BookAsState(lay, len(self.mamba), B, T, mode, block if mode == "align" else 0)
        self.dev = _FakeDevice(lay, len(self.mamba))
        self.eng = _EngineStub(self.book, block)
        self.req_states = {}
        self.row_of = {}  # request id -> engine row
        self.contract_counts = (
            {}
        )  # request id -> tokens its last verify committed (the spec contract's accepted_counts)
        self.plugin_counts = {}  # TTModelRunner._req_accepted_counts (MambaCacheGroups.begin_step input)
        self.fresh = set()  # engine rows prefilled and not verified yet
        self.hits = {}
        self.resumes = []  # (request id, start, end) of prefill rows that resumed from their state block
        self.verifies = 0
        self.held = 0
        self.n_out = 0
        self.step_no = 0
        self.accept = lambda req_id, step, drafts: len(drafts)
        self.tokens = {}  # request id -> its tokens after its last step
        self.conv_writes = 0  # conv histories written from a window (copy sources and align boundaries)
        self.boundary_writes = 0

    def add(self, req_id, prompt, max_tokens=24):
        init_none_hash(sha256)
        params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
        params.update_from_generation_config({}, eos_token_id=2)
        self.scheduler.add_request(
            Request(
                request_id=req_id,
                prompt_token_ids=list(prompt),
                sampling_params=params,
                pooling_params=None,
                block_hasher=get_request_block_hasher(self.scheduler.hash_block_size, sha256),
            )
        )

    def run(self, max_steps=600):
        for _ in range(max_steps):
            if not self.step():
                return
        raise AssertionError("scenario did not finish")

    def _free_row(self, req_id):
        u = self.row_of.pop(req_id, None)
        if u is not None:
            self.book.forget_row(u)
            self.fresh.discard(u)
            self.dev.win.pop(u, None)

    def _claim_row(self, req_id):
        if req_id not in self.row_of:
            used = set(self.row_of.values())
            free = [u for u in range(B) if u not in used]
            # the highest free row first, so engine rows differ from the scheduler's row order
            self.row_of[req_id] = free[-1]
        return self.row_of[req_id]

    def _copies(self, pairs):
        ops = self.book.copy_plan(pairs)
        self.conv_writes += sum(1 for op in ops if op[0] == "conv")
        self.dev.run_ops(ops)

    def step(self):
        sched = self.scheduler
        out = sched.schedule()
        for r in list(out.finished_req_ids) + list(out.preempted_req_ids or ()):
            self.req_states.pop(r, None)
            self._free_row(r)
            self.contract_counts.pop(r, None)
            self.plugin_counts.pop(r, None)
        if not out.num_scheduled_tokens:
            return False
        cached = out.scheduled_cached_reqs
        for new in out.scheduled_new_reqs:
            self.req_states[new.req_id] = SimpleNamespace(
                num_computed_tokens=new.num_computed_tokens, block_ids=[list(ids) for ids in new.block_ids]
            )
            self.hits[new.req_id] = new.num_computed_tokens
        for i, r in enumerate(cached.req_ids):
            if r in cached.resumed_req_ids:  # a preempted request replays from its computed tokens, new blocks
                self.req_states[r] = SimpleNamespace(
                    num_computed_tokens=cached.num_computed_tokens[i],
                    block_ids=[list(ids) for ids in cached.new_block_ids[i]],
                )
                continue
            st = self.req_states[r]
            st.num_computed_tokens = cached.num_computed_tokens[i]
            if cached.new_block_ids[i] is not None:
                for row, ids in zip(st.block_ids, cached.new_block_ids[i]):
                    row.extend(ids)
        for b in out.new_block_ids_to_zero or ():  # TTModelRunner order: zeroing, vLLM's copies, begin_step's copies
            self.book.forget_block(b)
            self.dev.zero(b)
        self._copies([(c.src_block_id, c.dst_block_id) for c in out.kv_cache_block_copies or ()])
        self._copies(self.groups.begin_step(out, self.req_states, self.plugin_counts))
        req_ids = list(out.num_scheduled_tokens)
        width = self.groups.max_num_blocks(self.max_model_len)
        tables = []
        for g in range(len(self.kv.kv_cache_groups)):
            t = torch.zeros((len(req_ids), width), dtype=torch.int32)
            for row, r in enumerate(req_ids):
                ids = self.req_states[r].block_ids[g]
                t[row, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
            tables.append(t)
        inputs = self.groups.step_inputs(
            tables, req_ids, self.req_states, out.num_scheduled_tokens, list(range(len(tables))), len(req_ids)
        )
        rows_prefill, rows_verify = [], []
        for row, r in enumerate(req_ids):
            req = sched.requests[r]
            nc = self.req_states[r].num_computed_tokens
            (rows_prefill if nc < req.num_prompt_tokens else rows_verify).append((row, r))

        sampled = {}
        verified = {}
        # ---- prefill rows (prefill_forward: one request at a time, gdn_req = its row in the step's state blocks)
        if rows_prefill:
            self.book.set_prefill_blocks(
                [inputs.state_block_ids_per_layer[g][[row for row, _ in rows_prefill], 0] for g in self.mamba]
            )
            for i, (row, r) in enumerate(rows_prefill):
                req = sched.requests[r]
                nc, ns = self.req_states[r].num_computed_tokens, out.num_scheduled_tokens[r]
                tokens = list(req.all_token_ids)
                L = nc + ns
                final = L >= req.num_prompt_tokens
                u = self._claim_row(r)
                self.book.forget_row(u)
                plan = self.eng._paged_plan(nc, L, None, final, i)
                for g in range(len(self.mamba)):
                    b = self.book.prefill_block(g, i)
                    if plan is not None and plan[0] > 0:
                        assert self.dev.read_block(g, b) == (_label(tokens[:nc]),) * 2, ("restore", r, nc, g, b)
                    self.book.forget_block(b)
                    self.dev.write_block(g, b, _label(tokens[:L]))
                if plan is not None and plan[0] > 0:
                    self.resumes.append((r, nc, L))
                if final:
                    self.dev.taps[u] = _label(tokens[:L])
                    self.fresh.add(u)
                    self.contract_counts[r] = 1
                    self.n_out += 1
                    sampled[r] = [1000 + self.n_out]
                else:
                    sampled[r] = []
        # ---- the verify (decode_forward): every live row not in this step is held
        if rows_verify:
            n = len(rows_verify)
            blocks = []
            for g in self.mamba:
                w = torch.zeros(B, T, dtype=torch.int64)
                w[:n] = inputs.state_block_windows_per_layer[g][[row for row, _ in rows_verify]].to(torch.int64)
                blocks.append(w)
            counts = torch.ones(B, dtype=torch.int64)
            counts[:n] = inputs.state_accepted_counts[[row for row, _ in rows_verify]].to(torch.int64)
            start0 = torch.zeros(B, dtype=torch.int64)
            active = torch.zeros(B, dtype=torch.bool)
            for slot, (row, r) in enumerate(rows_verify):
                start0[slot] = self.req_states[r].num_computed_tokens
                active[slot] = True
            # runner slot -> engine row (the host slot map): slot i holds request rows_verify[i]
            phys = [None] * B
            for slot, (_, r) in enumerate(rows_verify):
                phys[slot] = self._claim_row(r)
            rest = [u for u in range(B) if u not in phys]
            phys = [p if p is not None else rest.pop(0) for p in phys]
            self.eng._set_slot_map(phys)
            (start_rows, active_rows) = self.eng._to_rows(start0, active)
            self.eng._stage_gdn_windows((blocks, counts), start_rows, active_rows)
            live = self.eng._paged_step_rows
            self.held += sum(1 for u in self.row_of.values() if u not in live)
            mi = [0] * B
            expect, labels = {}, {}
            for slot, (row, r) in enumerate(rows_verify):
                u = phys[slot]
                req = sched.requests[r]
                nc, ns = self.req_states[r].num_computed_tokens, out.num_scheduled_tokens[r]
                drafts = list(out.scheduled_spec_decode_tokens.get(r, ()))
                tokens = list(req.all_token_ids)
                query = tokens[nc : nc + ns - len(drafts)] + drafts
                if u in self.fresh:  # fast join: E_prev rows 1 .. kc - 1 from the prefill taps, mi = 0
                    self.dev.win[u] = [self.dev.taps[u]] + [JUNK] * K
                    self.fresh.discard(u)
                    mi[u] = 0
                else:
                    mi[u] = self.contract_counts.get(r, 1) - 1
                expect[u] = _label(tokens[:nc])
                labels[u] = [_label(tokens[:nc] + query[: t + 1]) if t < ns else JUNK for t in range(T)]
            self.dev.verify(self.book, mi, live, expect, labels)
            due = [("conv", u, t, g, b) for u, t, g, b in self.book.take_pending()]
            self.boundary_writes += len(due)
            self.dev.run_ops(due)
            self.verifies += 1
            for slot, (row, r) in enumerate(rows_verify):
                drafts = list(out.scheduled_spec_decode_tokens.get(r, ()))
                a = max(0, min(self.accept(r, self.step_no, drafts), len(drafts)))
                self.n_out += 1
                sampled[r] = drafts[:a] + [5000 + self.n_out]
                verified[r] = a + 1
                self.contract_counts[r] = a + 1
        for r in req_ids:
            self.plugin_counts.pop(r, None)
        self.plugin_counts.update(verified)
        self._copies(
            self.groups.after_acceptance(
                {r: (self.req_states[r].num_computed_tokens, c) for r, c in verified.items()}, self.req_states
            )
        )
        sched.update_from_output(
            out,
            ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={r: i for i, r in enumerate(req_ids)},
                sampled_token_ids=[sampled.get(r, []) for r in req_ids],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
        # The TT rail's first verify after a prefill has no drafts (the drafter has no hidden row yet): drafts only for rows
        # this step verified.
        running = [r for r in verified if r in sched.requests]
        if running:
            sched.update_draft_token_ids(
                DraftTokenIds(
                    running,
                    [[90_000 + 100 * self.step_no + 10 * i + j for j in range(K)] for i in range(len(running))],
                )
            )
        for r in req_ids:
            if r in sched.requests:
                self.tokens[r] = list(sched.requests[r].all_token_ids)
        self.step_no += 1
        return True


def _prompt(seed, n):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(10, 30_000, (n,), generator=g).tolist()


def _rotate(req_id, step, drafts):
    """Accept 0 .. K drafts in a rotation, so every count appears."""
    return (step + len(req_id)) % (len(drafts) + 1) if drafts else 0


@pytest.mark.parametrize("mode", ["none", "align"])
def test_decode_reads_the_accepted_column_after_every_count(tmp_path, mode):
    h = _Harness(tmp_path, mode)
    h.accept = _rotate
    for k in range(3):
        h.add(f"r{k}", _prompt(k, 300 + 97 * k), max_tokens=40)
    h.run()
    assert h.verifies > 10 and h.dev.reads > 0


def _decode_boundary_scenario(tmp_path):
    """Requests whose decode crosses the 1024 boundary with every accept count, then follow-ups that hit the boundary
    states decode left. "t1" (1023 tokens): its first verify has no drafts and its state after token 0 is the boundary
    state, in the window column vLLM keeps without a copy. "t2" / "t3" (1022 / 1021): verifies with drafts straddle the
    boundary; begin_step and after_acceptance copy from the accepted / boundary column. Dense retention keeps them."""
    h = _Harness(tmp_path, "align", retention=None)
    h.accept = _rotate
    for name, n in (("t1", 1023), ("t2", 1022), ("t3", 1021)):
        h.add(name, _prompt(len(name) * 7 + n, n), max_tokens=24)
    h.run()
    for name in ("t1", "t2", "t3"):
        h.add("f" + name, h.tokens[name][: 1024 + 5] + _prompt(40 + len(name), 300), max_tokens=6)
    h.run()
    return h


def test_rejections_at_a_block_boundary_and_hits_on_the_states_decode_left(tmp_path):
    h = _decode_boundary_scenario(tmp_path)
    for name in ("t1", "t2", "t3"):
        assert h.hits["f" + name] == 1024, h.hits
        assert any(r == "f" + name and s == 1024 for r, s, _ in h.resumes), h.resumes
    assert h.boundary_writes > 0 and h.conv_writes > 0


def _broken(name):
    """A SpecBlockBook with one mechanism removed."""

    class Broken(_BookAsState):
        def copy_plan(self, copies):
            ops = super().copy_plan(copies)
            if name == "no_conv_on_copy":
                ops = [op for op in ops if op[0] == "copy"]
            return ops

        def take_pending(self):
            out = super().take_pending()
            return [] if name == "no_boundary_write" else out

        def set_step(self, blocks, start_col, live_rows, boundary=()):
            if name == "start_at_column_0":
                start_col = torch.zeros_like(start_col)
            super().set_step(blocks, start_col, live_rows, boundary)

    return Broken


@pytest.mark.parametrize("name", ["no_conv_on_copy", "no_boundary_write", "start_at_column_0"])
def test_each_mechanism_is_needed(tmp_path, monkeypatch, name):
    monkeypatch.setitem(globals(), "_BookAsState", _broken(name))
    with pytest.raises(AssertionError):  # allow-pytest.raises: host test, runs with --noconftest (no expect_error)
        _decode_boundary_scenario(tmp_path)


def test_a_copied_block_names_the_state_of_its_source_column():
    # mode align, all drafts accepted at a step ending on a boundary: after_acceptance copies window column K into
    # column 0 (src == dst column), and the next begin_step copies column 0 on. The second copy must carry column K's
    # conv rows, not column 0's.
    lay = _layout(8, 1024, 8, "two_pool")
    book = sbs.SpecBlockBook(lay, 1, B, T, "align", 1024)
    w = torch.zeros(B, T, dtype=torch.int64)
    w[2] = torch.tensor([30, 31, 32, 33])
    book.set_step([w], torch.zeros(B, dtype=torch.int64), [2])
    assert book.copy_plan([(33, 30)]) == [("copy", 33, 30), ("conv", 2, K, 0, 30)]
    assert book.copy_plan([(30, 40)]) == [("copy", 30, 40), ("conv", 2, K, 0, 40)]
    assert book.copy_plan([(77, 31)]) == [("copy", 77, 31)]  # an unmapped source: the block holds its full state
    assert book.copy_plan([(31, 41)]) == [("copy", 31, 41)]  # ... and so does its copy


@pytest.mark.parametrize("retention", [0, None])
def test_repeated_prompt_resumes_from_the_cached_boundary(tmp_path, retention):
    h = _Harness(tmp_path, "align", retention=retention)
    h.accept = _rotate
    a = _prompt(5, 2 * 1024 + 300)
    h.add("a", a, max_tokens=16)
    h.run()
    h.add("a2", a, max_tokens=16)
    h.run()
    assert h.hits["a2"] == 2 * 1024
    assert ("a2", 2 * 1024, len(a)) in h.resumes


def test_chunked_prefill_resumes_from_its_own_block(tmp_path):
    h = _Harness(tmp_path, "align", budget=1024)
    h.accept = _rotate
    a = _prompt(6, 3 * 1024 + 200)
    h.add("a", a, max_tokens=8)
    h.run()
    starts = [s for r, s, _ in h.resumes if r == "a"]
    assert starts == [1024, 2048, 3072], starts


def test_chunked_prefill_mode_none(tmp_path):
    h = _Harness(tmp_path, "none", budget=1024)
    h.accept = _rotate
    h.add("a", _prompt(7, 2 * 1024 + 500), max_tokens=12)
    h.add("b", _prompt(8, 700), max_tokens=20)
    h.run()
    assert [s for r, s, _ in h.resumes if r == "a"] == [1024, 2048]


def test_a_chunk_that_ends_inside_a_block_is_recomputed(tmp_path):
    # mode none with a 768-token budget: chunk ends fall inside 1024-token blocks; paged_fill_cache would overwrite the
    # block's earlier rows, so only block-aligned starts resume, the others recompute from 0
    h = _Harness(tmp_path, "none", budget=768)
    h.accept = _rotate
    h.add("a", _prompt(12, 2600), max_tokens=8)
    h.run()
    assert all(s % 1024 == 0 for r, s, _ in h.resumes if r == "a"), h.resumes


@pytest.mark.parametrize("mode", ["none", "align"])
def test_rows_left_out_of_a_step_are_held_and_resume(tmp_path, mode):
    # A long prompt arriving mid-decode takes the step budget: running rows sit out steps (the TT runner runs prefill
    # and verify as separate forwards, and vLLM schedules the partial prefill first) and must find their state after.
    h = _Harness(tmp_path, mode, budget=1024)
    h.accept = _rotate
    h.add("r0", _prompt(9, 200), max_tokens=60)
    for _ in range(4):
        h.step()
    h.add("big", _prompt(10, 3 * 1024), max_tokens=8)
    h.add("r1", _prompt(11, 150), max_tokens=40)
    h.run()
    assert h.held > 0


def test_preemption_replays_from_a_prefix_hit_or_from_zero(tmp_path):
    # 24 blocks for 4 requests that each want 4..6 blocks: the scheduler preempts and resumes.
    h = _Harness(tmp_path, "align", num_blocks=24, retention=None)
    h.accept = _rotate
    for k in range(4):
        h.add(f"p{k}", _prompt(20 + k, 1024 + 900), max_tokens=200)
    h.run(max_steps=3000)
    assert h.verifies > 50


@pytest.mark.parametrize("geo", [(4, 2048, 8, "two_pool"), (8, 2048, 8, "k_only")], ids=["tp4_two_pool", "tp8_k_only"])
def test_other_layouts(tmp_path, geo):
    tp, block, nb, name = geo
    h = _Harness(tmp_path, "align", tp=tp, block=block, n_banks=nb, layout=name, retention=None)
    h.accept = _rotate
    a = _prompt(30, block - 2)
    h.add("a", a, max_tokens=12)
    h.add("b", _prompt(31, block + 100), max_tokens=12)
    h.run()
    h.add("a2", a[: block - 2], max_tokens=4)
    h.run()
    assert h.verifies > 5
