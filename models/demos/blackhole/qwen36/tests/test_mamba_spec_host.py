# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the MTP spec class's GDN state in vLLM MambaSpec spec blocks (QWEN36_MAMBA_SPEC=1, K = 3).

A real TTScheduler (vllm-tt-plugin) over a real KVCacheManager schedules requests with 3 drafts per step. The KV cache
groups come from the model's own specs (tt/kv_cache_spec.qwen36_kv_cache_spec, num_speculative_blocks = 3). Each step
the plugin's MambaCacheGroups gives the copies before the forward (begin_step), each row's 1 + K window and start count
(step_inputs), and the copies after acceptance (after_acceptance), the calls TTModelRunner makes. The model's
mamba_state keyword built from them goes through Qwen36MTPContractForCausalLM's conversion (qwen36_vllm), which maps
the groups to the bound groups by layer name.

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

import contextlib
import inspect
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
from vllm_tt_plugin.scheduler import TTScheduler, get_tt_mamba_checkpoints  # noqa: E402
from vllm_tt_plugin.spec_admission import MODEL_OWNED_DRAFT_METHOD, MODEL_OWNED_DRAFT_SENTINEL  # noqa: E402
from vllm_tt_plugin.spec_decode import ACCEPT_MODE_ARGMAX_IDS, DRAFTER_STATE_INTERNAL, SpecPlan  # noqa: E402

from models.demos.blackhole.qwen36.tests.test_kv_cache_spec_host import TEXT_CONFIG  # noqa: E402
from models.demos.blackhole.qwen36.tests.test_mamba_align_host import (  # noqa: E402
    mamba_state,
    plugin_checkpoints,
    plugin_windows,
)
from models.demos.blackhole.qwen36.tt import spec_block_state as sbs  # noqa: E402
from models.demos.blackhole.qwen36.tt.gdn.tp import spec_ctrl_words  # noqa: E402
from models.demos.blackhole.qwen36.tt.kv_cache_spec import gdn_layer_name, qwen36_kv_cache_spec  # noqa: E402
from models.demos.blackhole.qwen36.tt.qwen36_vllm import (  # noqa: E402
    Qwen36MTPContractForCausalLM,
    _spec_prefill_blocks,
)
from models.demos.blackhole.qwen36.tt.spec_engine import MTPSpecEngine, _CheckpointSnapRun  # noqa: E402

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
# (TP, block, n_banks): the Galaxy primary (BF8 1024, 8 banks, heads in K and V), its 7-bank Bin6 variant (896), BF8
# 2048 at TP=8, and the P150x4 TP=4 test geometries (2048, 4096).
GEOMETRIES = [
    (8, 1024, 8),
    (8, 896, 7),
    (8, 2048, 8),
    (4, 2048, 8),
    (4, 4096, 8),
]
EVERY_HEAD_IN_K = [(8, 2048, 8), (4, 4096, 8)]  # K's block holds every head: V's block holds only the conv history


def _layout(tp, block, n_banks, num_blocks=200):
    nv, C = 48 // tp, (2 * 16 * 128 + 48 * 128) // tp
    S = sbs.kv_block_bytes_per_bank([num_blocks, 1, block, 256], "bfloat8_b", n_banks)
    return sbs.SpecBlockLayout(S, n_banks, num_blocks, nv, sbs.conv_carry_pages(4, C))


@pytest.mark.parametrize("geo", GEOMETRIES, ids=lambda g: f"tp{g[0]}_b{g[1]}_{g[2]}banks")
def test_layout_pages_lie_inside_their_block_and_never_overlap(geo):
    lay = _layout(*geo)
    for b in range(lay.num_blocks):
        lay.check_block(b)  # inside the block's bytes on every bank, heads and history disjoint
    heads = lay.head_pages(torch.arange(lay.num_blocks))
    assert bool(((heads % sbs.HALF_PAGES) == 0).all()), "head pages must start on a [Dk / 2, Dv] view block"
    k_side = heads < lay.pool1_pages
    assert bool((k_side.sum(-1) == lay.heads_in_k).all())
    assert (lay.heads_in_k == lay.nv) == (geo in EVERY_HEAD_IN_K)


@pytest.mark.parametrize("geo", EVERY_HEAD_IN_K, ids=lambda g: f"tp{g[0]}_b{g[1]}")
def test_every_head_in_k_puts_the_conv_history_at_the_start_of_v(geo):
    """K's block holds every head: the heads are K's, consecutive from the block's first whole fp32 page (rounded up
    to HALF_PAGES), and each block's conv history starts at the first bf16 page of V's block."""
    lay = _layout(*geo)
    S, nb = lay.block_bytes, lay.n_banks
    assert lay.heads_in_k == lay.nv
    blocks = torch.arange(lay.num_blocks)
    row0 = -(-blocks * S // sbs.FP32_PAGE)  # each block's first whole fp32 row on a bank
    first = -(-row0 * nb // sbs.HALF_PAGES) * sbs.HALF_PAGES
    want = first.unsqueeze(-1) + sbs.HEAD_PAGES * torch.arange(lay.nv)
    assert torch.equal(lay.head_pages(blocks), want) and bool((want < lay.pool1_pages).all())
    for b in range(lay.num_blocks):
        assert lay.conv_page(b) == -(-b * S // sbs.BF16_PAGE) * nb < lay.conv_pages_total, f"block {b}"


@pytest.mark.parametrize("geo", [(4, 1024, 8), (8, 1024, 7)], ids=["tp4_1024", "tp8_1024_7banks"])
def test_layout_refuses_a_block_that_cannot_hold_the_state(geo):
    with pytest.raises(ValueError):  # allow-pytest.raises: host test, runs with --noconftest (no expect_error)
        _layout(*geo)


def test_ctrl_page_words_follow_the_op_contract():
    lay = _layout(8, 1024, 8)
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

    lay = _layout(4, 2048, 8)
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
    # the body's table (SpecGdnBlockState.pages_dev): column t * G * Nv + g * Nv + h = ring_page_idx[g][u, t * Nv + h]
    cat = book.pages_cat().to(torch.int64).reshape(B, T, 3, lay.nv)
    for g in range(3):
        assert torch.equal(cat[:, :, g].reshape(B, -1), book.idx_host[g].to(torch.int64))


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
        """copy_kv_blocks: every byte of block src to the same offset in block dst. A state page moves with its bytes:
        fp32 page p (pool-relative) of row p // nb lands on row p // nb + (dst - src) * S / 4096, a whole page only
        when that is an integer (the blocks have the same phase); otherwise its bytes straddle two pages and no state
        page of dst holds it. bf16 conv pages move by (dst - src) * S / 2048 rows (always whole)."""
        lay, nb, S, P = self.lay, self.lay.n_banks, self.lay.block_bytes, self.lay.pool1_pages
        shift32, shift16 = (dst - src) * S / 4096, (dst - src) * S // 2048
        moves = []
        for (g, p), lab in list(self.rec.items()):
            off, q = (0, p) if p < P else (P, p - P)
            if q // nb >= -(-src * S // 4096) and q // nb < ((src + 1) * S) // 4096:
                moves.append(((g, p), lab, off, q))
        for g in range(self.G):  # dst's bytes are src's now: whatever dst's layout pages held is gone
            self.write_block(g, dst, "overwritten")
        for (g, p), lab, off, q in moves:
            if shift32 == int(shift32):
                self.rec[(g, off + q + int(shift32) * nb)] = lab
        cmoves = []
        T16 = lay.conv_pages_total
        for (g, c), lab in list(self.conv.items()):
            off, q = (0, c) if c < T16 else (T16, c - T16)
            if q // nb >= -(-src * S // 2048) and q // nb < ((src + 1) * S) // 2048:
                cmoves.append((g, off + q + shift16 * nb, lab))
        for g, c, lab in cmoves:
            self.conv[(g, c)] = lab

    def relayout(self, src, dst):
        """SpecGdnBlockState._relayout: the state copied page by page with each block's own layout."""
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
            elif op[0] == "relayout":
                self.relayout(op[1], op[2])
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

    def write_checkpoint(self, blocks):
        """SpecGdnBlockState.write_checkpoint on the fake device: the bound scratch (``scratch``, the label the
        harness's segment loop left there) into one block per group."""
        for g, b in enumerate(blocks):
            self.forget_block(b)
            self.dev.write_block(g, b, self.scratch)

    def write_checkpoint_snapshot(self, blocks, k):
        """SpecGdnBlockState.write_checkpoint_snapshot: in-replay snapshot k holds the state a segment ending at the
        checkpoint would leave in the scratch (the label the harness sets before each save)."""
        self.write_checkpoint(blocks)


class _Harness:
    def __init__(
        self,
        tmp_path,
        mode,
        tp=8,
        block=1024,
        n_banks=8,
        num_blocks=96,
        retention=0,
        max_model_len=8192,
        budget=2048,
        checkpoint=False,
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
            prefill_checkpoint_alignment=block if checkpoint else None,
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
        lay = _layout(tp, block, n_banks, num_blocks)
        self.book = _BookAsState(lay, len(self.mamba), B, T, mode, block if mode == "align" else 0)
        layer_of = {gdn_layer_name(i): i for i in range(len(specs))}
        self.book.layer_groups = {
            layer_of[name]: k for k, g in enumerate(self.mamba) for name in self.kv.kv_cache_groups[g].layer_names
        }
        self.dev = _FakeDevice(lay, len(self.mamba))
        self.book.dev = self.dev
        self.prefill_steps = {}  # request id -> prefill steps of its prompt
        self.checkpoint_writes = []  # (request id, n) of every prefill checkpoint state written
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
        # Synchronous scheduling: each row's first query position is the scheduler's num_computed_tokens. The plugin of
        # vllm-tt-plugin#280 takes these positions; the older one reads the request states.
        positions = {r: self.req_states[r].num_computed_tokens for r in out.num_scheduled_tokens}
        new_api = "num_computed" in inspect.signature(self.groups.begin_step).parameters
        self._copies(
            self.groups.begin_step(out, self.req_states, positions, self.plugin_counts)
            if new_api
            else self.groups.begin_step(out, self.req_states, self.plugin_counts)
        )
        req_ids = list(out.num_scheduled_tokens)
        width = self.groups.max_num_blocks(self.max_model_len)
        tables = []
        for g in range(len(self.kv.kv_cache_groups)):
            t = torch.zeros((len(req_ids), width), dtype=torch.int32)
            for row, r in enumerate(req_ids):
                ids = self.req_states[r].block_ids[g]
                t[row, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
            tables.append(t)
        windows, counts = plugin_windows(
            self.groups,
            tables,
            req_ids,
            positions if new_api else self.req_states,
            out.num_scheduled_tokens,
            len(req_ids),
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
            # TTModelRunner's prefill checkpoint inputs: the rows of this prefill step, in its order
            ckpts = plugin_checkpoints(
                self.groups,
                get_tt_mamba_checkpoints(out),
                [r for _, r in rows_prefill],
                self.req_states,
                len(rows_prefill),
            )
            state = mamba_state(self.groups, windows, [row for row, _ in rows_prefill], checkpoints=ckpts)
            ids, ckpt = _spec_prefill_blocks(state, self.book)
            self.book.set_prefill_blocks(ids)
            self.book.set_prefill_checkpoints(ckpt, state.checkpoint_positions)
            for i, (row, r) in enumerate(rows_prefill):
                req = sched.requests[r]
                nc, ns = self.req_states[r].num_computed_tokens, out.num_scheduled_tokens[r]
                tokens = list(req.all_token_ids)
                L = nc + ns
                final = L >= req.num_prompt_tokens
                u = self._claim_row(r)
                self.book.forget_row(u)
                self.prefill_steps[r] = self.prefill_steps.get(r, 0) + 1
                plan = self.eng._paged_plan(nc, L, None, final, i)
                for g in range(len(self.mamba)):
                    b = self.book.prefill_block(g, i)
                    if plan is not None and plan[0] > 0:
                        assert self.dev.read_block(g, b) == (_label(tokens[:nc]),) * 2, ("restore", r, nc, g, b)
                # _prefill_fast / _prefill_for_spec_resumable: a segment ends at each save point, and the bound scratch
                # then holds the state after tokens[:n]
                points = plan[2] if plan is not None else []
                for n in points:
                    self.book.scratch = _label(tokens[:n])
                    points.save(n)
                    self.checkpoint_writes.append((r, n))
                for g in range(len(self.mamba)):
                    b = self.book.prefill_block(g, i)
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
            state = mamba_state(self.groups, windows, [row for row, _ in rows_verify], counts=counts)
            gdn_windows = Qwen36MTPContractForCausalLM._gdn_windows(state, self.book, B)
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
            self.eng._stage_gdn_windows(gdn_windows, start_rows, active_rows)
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
                ops = [op for op in ops if op[0] != "conv"]
            if name == "byte_copy_across_phases":
                ops = [("copy", *op[1:]) if op[0] == "relayout" else op for op in ops]
            return ops

        def take_pending(self):
            out = super().take_pending()
            return [] if name == "no_boundary_write" else out

        def set_step(self, blocks, start_col, live_rows, boundary=()):
            if name == "start_at_column_0":
                start_col = torch.zeros_like(start_col)
            super().set_step(blocks, start_col, live_rows, boundary)

    return Broken


@pytest.mark.parametrize(
    "name", ["no_conv_on_copy", "no_boundary_write", "start_at_column_0", "byte_copy_across_phases"]
)
def test_each_mechanism_is_needed(tmp_path, monkeypatch, name):
    monkeypatch.setitem(globals(), "_BookAsState", _broken(name))
    with pytest.raises(AssertionError):  # allow-pytest.raises: host test, runs with --noconftest (no expect_error)
        _decode_boundary_scenario(tmp_path)


def test_a_copied_block_names_the_state_of_its_source_column():
    # mode align, all drafts accepted at a step ending on a boundary: after_acceptance copies window column K into
    # column 0 (src == dst column), and the next begin_step copies column 0 on. The second copy must carry column K's
    # conv rows, not column 0's.
    lay = _layout(8, 1024, 8)
    book = sbs.SpecBlockBook(lay, 1, B, T, "align", 1024)
    w = torch.zeros(B, T, dtype=torch.int64)
    w[2] = torch.tensor([30, 31, 32, 33])
    book.set_step([w], torch.zeros(B, dtype=torch.int64), [2])
    # at BF8 1024 odd and even blocks hold their fp32 pages at different offsets: those copies move page by page
    assert book.copy_plan([(33, 30)]) == [("relayout", 33, 30), ("conv", 2, K, 0, 30)]
    assert book.copy_plan([(30, 40)]) == [("copy", 30, 40), ("conv", 2, K, 0, 40)]
    assert book.copy_plan([(77, 31)]) == [("copy", 77, 31)]  # an unmapped source: the block holds its full state
    assert book.copy_plan([(31, 42)]) == [("relayout", 31, 42)]  # ... and so does its copy


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


CKPT_LENGTHS = (900, 1500, 2048, 2049, 2050, 3000, 3100)


def _checkpoint_run(tmp_path, retention, checkpoint):
    """Each prompt of CKPT_LENGTHS alone (they fit one 4096-token chunk), then its follow-up (the prompt + 300
    tokens)."""
    tmp_path.mkdir()
    h = _Harness(tmp_path, "align", retention=retention, budget=4096, checkpoint=checkpoint)
    h.accept = _rotate
    for k, n in enumerate(CKPT_LENGTHS):
        a = _prompt(40 + k, n)
        h.add(f"a{n}", a, max_tokens=6)
        h.run()
        h.add(f"b{n}", a + _prompt(60 + k, 300), max_tokens=6)
        h.run()
    return h


@pytest.mark.parametrize("retention", [0, None])
def test_prefill_checkpoints_run_each_prompt_in_one_step(tmp_path, retention):
    """vLLM's prefill checkpoints (QWEN36_MAMBA_SPEC=1, mode align): vLLM no longer ends a prompt's chunk at its last
    full block; the prefill writes the state there into vLLM's checkpoint block (a segment ends there,
    CheckpointPoints). Against the same requests without checkpoints: every prompt in one prefill step, fewer steps
    overall, every follow-up hit at least as long, and every resume reads exactly the state of its tokens (the harness
    asserts each read)."""
    split = _checkpoint_run(tmp_path / "split", retention, False)
    h = _checkpoint_run(tmp_path / "ckpt", retention, True)
    firsts = [f"a{n}" for n in CKPT_LENGTHS]
    assert all(h.prefill_steps[r] == 1 for r in firsts), h.prefill_steps
    assert sum(h.prefill_steps.values()) < sum(split.prefill_steps.values())
    assert not split.checkpoint_writes
    for n in CKPT_LENGTHS:
        assert h.hits[f"b{n}"] >= split.hits[f"b{n}"], (n, h.hits, split.hits)
        last = (n // 1024) * 1024
        if n > 1024 and n % 1024:
            assert (f"a{n}", last) in h.checkpoint_writes, h.checkpoint_writes
            assert h.hits[f"b{n}"] == last
            assert (f"b{n}", last, n + 300) in h.resumes, h.resumes


def test_a_prefill_that_skips_its_checkpoints_serves_a_stale_state(tmp_path, monkeypatch):
    """Negative control: without the checkpoint writes a follow-up resumes from a block that holds no state of its
    tokens."""
    monkeypatch.setattr(_BookAsState, "write_checkpoint", lambda self, blocks: None)
    with pytest.raises(AssertionError, match="restore"):  # allow-pytest.raises: --noconftest host test
        _checkpoint_run(tmp_path / "ckpt", 0, True)


def test_in_replay_checkpoints_come_from_the_snapshot_and_replay_ends_from_the_scratch():
    """QWEN36_PREFILL_INTRACE_SNAP with vLLM's checkpoints (one prefill call, one segment): a checkpoint strictly inside
    a replay is written from that replay's snapshot slot, one at a replay end from the bound state, in position order.
    """
    calls = []
    points = sbs.CheckpointPoints(
        [(1024, [5]), (2048, [6]), (3072, [7]), (3584, [8])],
        lambda blocks: calls.append(("scratch", blocks)),
        lambda blocks, k: calls.append(("snapshot", blocks, k)),
    )
    run = _CheckpointSnapRun(None, points)
    for cs in (0, 2048):
        run.after(cs, 2048, run.inside(cs, 2048))
    assert calls == [("snapshot", [5], 0), ("scratch", [6]), ("snapshot", [7], 0), ("snapshot", [8], 1)]
    assert run.saved == [1024, 2048, 3072, 3584] and run.n_inside == 3


@pytest.mark.parametrize(
    "env, on",
    [(dict(QWEN36_MAMBA_SPEC="1"), True), ({}, False), (dict(QWEN36_PREFILL_INTRACE_SNAP="1"), True)],
)
def test_the_spec_flag_turns_on_the_in_replay_snapshots(monkeypatch, env, on):
    """QWEN36_MAMBA_SPEC=1 turns on the in-replay snapshot traces (for vLLM's prefill checkpoints) without
    QWEN36_PREFILL_INTRACE_SNAP; without it that variable keeps its own meaning."""
    from models.demos.blackhole.qwen36.tt import tp_common as tpc

    for k in ("QWEN36_MAMBA_SPEC", "QWEN36_PREFILL_INTRACE_SNAP"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(tpc, "_PREFILL_INTRACE_SNAP", None)
    assert tpc.prefill_intrace_snap_enabled() is on


def test_a_checkpoint_off_the_block_grid_or_outside_the_prefill_is_refused():
    book = _BookAsState(_layout(8, 1024, 8), 3, B, T, "align", 1024)
    eng = _EngineStub(book, 1024)
    book.set_prefill_blocks([torch.tensor([5])] * 3)
    for n, (s, L) in [(1500, (0, 3000)), (3072, (0, 3000)), (1024, (1024, 3000))]:
        book.set_prefill_checkpoints([torch.tensor([[7]])] * 3, torch.tensor([[n]]))
        with pytest.raises(ValueError, match="checkpoint"):  # allow-pytest.raises: --noconftest host test
            eng._paged_plan(s, L, None, True, 0)
    book.set_prefill_checkpoints([torch.tensor([[7]])] * 3, torch.tensor([[2048]]))
    s, src, points = eng._paged_plan(1024, 3000, None, True, 0)
    assert (s, list(points)) == (1024, [2048]) and src == (book, 0)
    book.set_prefill_checkpoints(None, torch.zeros(1, 1, dtype=torch.int32))
    assert eng._paged_plan(0, 3000, None, True, 0) is None


def test_a_multimodal_row_writes_its_checkpoints_from_position_0():
    """A multimodal prompt never resumes from a block, but vLLM's checkpoints inside it become segment ends from 0
    (a block before the first image hashes like text, so it must hold the state); without checkpoints the plan stays
    None (one segment, as before)."""
    book = _BookAsState(_layout(8, 1024, 8), 3, B, T, "align", 1024)
    eng = _EngineStub(book, 1024)
    book.set_prefill_blocks([torch.tensor([5])] * 3)
    book.set_prefill_checkpoints([torch.tensor([[7]])] * 3, torch.tensor([[2048]]))
    s, src, points = eng._paged_plan(1024, 3000, "VISION", True, 0)
    assert (s, src, list(points)) == (0, None, [2048])
    book.set_prefill_checkpoints(None, torch.zeros(1, 1, dtype=torch.int32))
    assert eng._paged_plan(1024, 3000, "VISION", True, 0) is None


@pytest.mark.parametrize("mode", ["none", "align"])
def test_only_the_spec_class_declares_prefill_checkpoints_and_only_in_mode_align(monkeypatch, mode):
    """QWEN36_MAMBA_SPEC=1: the spec class's MambaSpec declares block-aligned prefill checkpoints in mode align; mode
    none and the non-spec class declare none."""
    from models.demos.blackhole.qwen36.tt import kv_cache_spec as kcs
    from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM, Qwen36MTPContractForCausalLM

    seen = []
    monkeypatch.setenv("QWEN36_MAMBA_SPEC", "1")
    monkeypatch.setattr(kcs, "qwen36_kv_cache_spec", lambda *a, **k: seen.append(k["prefill_checkpoint_alignment"]))
    vc = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(), max_model_len=8192),
        cache_config=SimpleNamespace(mamba_cache_mode=mode, block_size=2048),
    )
    Qwen36MTPContractForCausalLM.get_kv_cache_spec(vc)
    Qwen36ForCausalLM.get_kv_cache_spec(vc)
    assert seen == [2048 if mode == "align" else None, None]


def test_the_block_state_allocates_both_trace_sets_in_warmup(monkeypatch):
    """_prepare_block_traces allocates the write traces' (the scratch's and one per in-replay snapshot slot) and the
    copy traces' control tensors once (the eager warmup calls it before any trace exists)."""
    from models.demos.blackhole.qwen36.tt.tp_common import GDN_SNAP_SLOTS

    made = []
    monkeypatch.setattr(sbs, "_WriteTrace", lambda st, snap=None: made.append(("write", snap)) or ("W", snap))
    monkeypatch.setattr(sbs, "_CopyTraces", lambda st: made.append("copy") or "C")
    st = sbs.SpecGdnBlockState.__new__(sbs.SpecGdnBlockState)
    st.layers = [(0, SimpleNamespace(_snap_rec=object()), 0), (4, SimpleNamespace(_snap_rec=object()), 1)]
    st._prepare_block_traces()
    st._prepare_block_traces()
    snaps = list(range(GDN_SNAP_SLOTS))
    assert made == [("write", None)] + [("write", k) for k in snaps] + ["copy"]
    assert (st._wt_pending, st._ct_pending) == (("W", None), "C")
    assert st._swt_pending == {k: ("W", k) for k in snaps}


def test_no_snapshot_write_traces_without_snapshot_buffers(monkeypatch):
    # Without QWEN36_PREFILL_INTRACE_SNAP's buffers no checkpoint comes from a snapshot, so no trace is allocated.
    monkeypatch.setattr(sbs, "_WriteTrace", lambda st, snap=None: ("W", snap))
    monkeypatch.setattr(sbs, "_CopyTraces", lambda st: "C")
    st = sbs.SpecGdnBlockState.__new__(sbs.SpecGdnBlockState)
    st.layers = [(0, SimpleNamespace(_snap_rec=None), 0)]
    st._prepare_block_traces()
    assert getattr(st, "_swt_pending", None) is None


def test_a_checkpoint_from_a_snapshot_replays_that_slots_write_trace(monkeypatch):
    """write_checkpoint_snapshot replays the captured trace of snapshot slot k (no eager op runs) and forgets the
    target blocks, as the eager write does; without a trace for k it runs the eager write."""
    replayed, forgotten, eager = [], [], []

    class Trace:
        def __init__(self, ok):
            self.ok = ok

        def replay(self, blocks):
            replayed.append(list(blocks))
            return self.ok

    st = sbs.SpecGdnBlockState.__new__(sbs.SpecGdnBlockState)
    st.layers = []
    st.book = SimpleNamespace(forget_block=forgotten.append)
    st._snap_write_traces = {2: Trace(True), 3: Trace(False)}
    monkeypatch.setattr(sbs, "_PageTables", lambda s: eager.append("eager") or contextlib.nullcontext())
    st.write_checkpoint_snapshot([5, 6, 7], 2)
    assert (replayed, forgotten, eager) == ([[5, 6, 7]], [5, 6, 7], [])
    st.write_checkpoint_snapshot([8, 9, 10], 3)  # the trace does not apply: the eager write runs
    st.write_checkpoint_snapshot([8, 9, 10], 1)  # no trace for slot 1
    assert eager == ["eager", "eager"]


# ------------------------------------------------------------------- block I/O programs and the post-capture guard
class _ProgramCache:
    """GdnBlockIO._run's programs by what the generic_op program hash reads (mode and compile-time args with the core
    grid, accessor args, runtime-arg count, circular buffer size). Once ``forbid`` is set a program not seen before
    raises, as a program-cache miss does after QWEN36_FORBID_POST_CAPTURE_COMPILE=1 arms the guard."""

    def __init__(self):
        self.seen, self.forbid = set(), False

    def run(self, io, mode, cta_extra, accessors, common, tensors, items, l1):
        n, nx, _ = io._cores(items)
        key = (mode, tuple(cta_extra), n, nx, tuple(t.acc for t in accessors), len(common), l1)
        if self.forbid and key not in self.seen:
            raise RuntimeError(f"program cache miss after capture: mode {mode} on {n} cores")
        self.seen.add(key)


class _Buf:
    """A fake DRAM-interleaved device buffer; ``acc`` stands for its TensorAccessorArgs compile-time args."""

    def __init__(self, acc, addr, page, shape=(), layout=None, dtype=None):
        self.acc, self.addr, self.page, self.shape, self.layout, self.dtype = acc, addr, page, shape, layout, dtype

    def buffer_address(self):
        return self.addr

    def buffer_aligned_page_size(self):
        return self.page

    def memory_config(self):
        import ttnn

        return SimpleNamespace(memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED)


def _block_io_state(monkeypatch, cache):
    """SpecGdnBlockState with 12 GDN layers in 3 groups (TP=8, BF8 block 1024) on a 13 x 10 core grid: its GdnBlockIO
    programs go to ``cache``, the eager ops of warmup are stubbed out."""
    import ttnn
    from models.demos.blackhole.qwen36.tt import gdn_block_io as gbio
    from models.demos.blackhole.qwen36.tt import gdn_block_state as gbs

    monkeypatch.setattr(gbio, "_accessor", lambda t: [t.acc])
    monkeypatch.setattr(gbio.GdnBlockIO, "_run", lambda io, *a: cache.run(io, *a))
    monkeypatch.setattr(gbs, "zero_kv_blocks", lambda model, ids: None)
    monkeypatch.setattr(sbs, "_PageTables", lambda s: contextlib.nullcontext())
    monkeypatch.setattr(ttnn, "from_torch", lambda t, **k: t)
    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: None)
    lay = _layout(8, 1024, 8, num_blocks=96)
    C = (2 * 16 * 128 + 48 * 128) // 8
    mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=13, y=10))
    layers = []
    for i in range(12):
        k, v = 1000 * (2 * i + 1), 1000 * (2 * i + 2)
        views = sbs._LayerViews(
            rec1=_Buf("fp32", k, sbs.FP32_PAGE),
            rec2=_Buf("fp32", v, sbs.FP32_PAGE),
            conv=_Buf("bf16", v, sbs.BF16_PAGE),
            conv_k=_Buf("bf16", k, sbs.BF16_PAGE),
            group=i % 3,
            two_pool=True,
            k=_Buf("bf8", k, sbs.BF8_TILE),
            v=_Buf("bf8", v, sbs.BF8_TILE),
        )
        win = _Buf("window", 100_000 + i, sbs.BF16_PAGE, (B, 3 + T, C), ttnn.TILE_LAYOUT, ttnn.bfloat16)
        dn = SimpleNamespace(
            Nv=lay.nv, Dk=128, Dv=128, K=4, qkv_dim_tp=C, _spec_paged=views, verify_win_cur=lambda w=win: w
        )
        layers.append((i, dn, i % 3))
    st = sbs.SpecGdnBlockState.__new__(sbs.SpecGdnBlockState)
    st.model = SimpleNamespace(
        mesh_device=mesh,
        _paged_kv_caches=[],
        _bind_gdn_prefill_scratch=lambda: None,
        _unbind_gdn_prefill_scratch=lambda prev: None,
    )
    st.mesh, st.T, st.G, st.layout, st.layers = mesh, T, 3, lay, layers
    st._io = st._io_zero = None
    # the fused steps' int32 ROW_MAJOR result buffers, set by MTPSpecEngine.prepare_decode before the warmup
    st.chain_results = [
        _Buf("result", 200_000 + r, 64, (2 + r, 64), ttnn.ROW_MAJOR_LAYOUT, ttnn.int32) for r in range(2)
    ]
    st.write_prefill = st.restore = lambda u: None
    st.materialize = lambda *a, **k: None
    st._zero_now = lambda ids: None
    st._prepare_block_traces = lambda: None
    return st


def test_boundary_conv_writes_after_the_warmup_compile_nothing(tmp_path, monkeypatch):
    """SpecGdnBlockState.warmup compiles every WINDOW program after_verify can build (one per core grid): decodes that
    cross the 1024 boundary, then every count of boundary conv writes in a step, run with program-cache misses
    forbidden."""
    cache = _ProgramCache()
    st = _block_io_state(monkeypatch, cache)
    st.warmup()
    cache.forbid = True
    served = []

    def serve(due):
        st.book = SimpleNamespace(take_pending=lambda: list(due))
        st.after_verify()
        served.append(len(due))

    class Served(_BookAsState):
        def take_pending(self):
            due = super().take_pending()
            serve(due)
            return due

    monkeypatch.setitem(globals(), "_BookAsState", Served)
    _decode_boundary_scenario(tmp_path)
    assert sum(served) > 0
    for n in range(1, 2 * B * st.G + 1):  # up to every row's three groups twice
        serve([(u % B, 0, u % st.G, 5) for u in range(n)])


@pytest.mark.parametrize("on_device", [False, True], ids=["counts_on_host", "result_buffer"])
def test_chained_copies_after_the_warmup_compile_nothing(monkeypatch, on_device):
    """SpecGdnBlockState.warmup compiles every CHAIN program run_chain_copies can build: host-selected (counts) and
    device-selected (each fused step's result buffer), for every count of copies in a step."""
    cache = _ProgramCache()
    st = _block_io_state(monkeypatch, cache)
    st.warmup()
    cache.forbid = True
    st.book = SimpleNamespace(forget_block=lambda b: None)
    for n in range(1, 2 * st._io._ch_max() + 2):
        for res in st.chain_results if on_device else [None]:
            st.chain_copies = [(u % B, u % st.G, 5 + u, 40 + u, None if u % 2 else 1, (1, 2)) for u in range(n)]
            if res is None:
                st.run_chain_copies(counts={u: 1 for u in range(B)})
            else:
                st.run_chain_copies(result=res)
            assert st.chain_copies == []


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


@pytest.mark.parametrize("geo", [(4, 2048, 8), (8, 2048, 8)], ids=["tp4_b2048", "tp8_b2048"])
def test_other_geometries(tmp_path, geo):
    tp, block, nb = geo
    h = _Harness(tmp_path, "align", tp=tp, block=block, n_banks=nb, retention=None)
    h.accept = _rotate
    a = _prompt(30, block - 2)
    h.add("a", a, max_tokens=12)
    h.add("b", _prompt(31, block + 100), max_tokens=12)
    h.run()
    h.add("a2", a[: block - 2], max_tokens=4)
    h.run()
    assert h.verifies > 5


# --------------------------------------------------------------------------------------------------- one layout
# Inside Qwen36MTPContractForCausalLM every block writer and reader uses SpecBlockLayout (spec_block_state.py), never
# the non-spec prototype's GdnBlockState layout (gdn_block_state.py): the class binds only SpecGdnBlockState, its block
# hooks and prefill go through it, and SpecGdnBlockState's own device writes land on the pages the verify reads.


def _per_layer_specs():
    """(shapes, dtypes, tensor_idx) of Qwen3.8's 64 layers: attention every 4th, slot = rank within its type."""
    specs, n_attn, n_gdn = [], 0, 0
    for i in range(64):
        if i % 4 == 3:
            specs.append(([1, 1, 2048, 256], "bf8", n_attn))
            n_attn += 1
        else:
            specs.append((((3, 2560), (12, 128, 128)), ("bf16", "fp32"), n_gdn % 16))
            n_gdn += 1
    return specs


def test_spec_class_binds_and_routes_only_through_the_spec_layout(monkeypatch):
    from models.demos.blackhole.qwen36.tt import gdn_block_state as gbs
    from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36MTPContractForCausalLM

    def non_spec(*a, **k):
        raise AssertionError("the spec class touched the non-spec GdnBlockState layout")

    monkeypatch.setattr(gbs, "GdnBlockState", non_spec)
    monkeypatch.setattr(gbs, "copy_kv_blocks", non_spec)
    monkeypatch.setattr(gbs, "zero_kv_blocks", non_spec)
    made = []

    class Recorder:
        def __init__(self, model, specs, kv, T, cache_mode, mamba_block_size, partial_prefix_hits):
            made.append((T, cache_mode, mamba_block_size, partial_prefix_hits))
            self.layout = SimpleNamespace(heads_in_k=8, nv=12)
            self.G, self.calls = 3, []

        def copy_blocks(self, copies):
            self.calls.append(("copy", copies))

        def zero_blocks(self, ids):
            self.calls.append(("zero", ids))

    monkeypatch.setattr(sbs, "SpecGdnBlockState", Recorder)
    model = SimpleNamespace(num_devices=4, args=SimpleNamespace(max_batch_size=8))
    me = SimpleNamespace(
        model=[model],
        _SPEC_K=K,
        allocate_kv_cache=lambda shape, dtype, n: [("k", "v")] * n,
        _vllm_config_or_none=lambda: SimpleNamespace(
            cache_config=SimpleNamespace(mamba_cache_mode="align", block_size=2048)
        ),
    )
    cls = Qwen36MTPContractForCausalLM
    kv = cls.allocate_kv_cache_per_layer(me, _per_layer_specs())
    assert len(kv) == 16 and made == [(T, "align", 2048, False)]
    assert isinstance(model._spec_blocks, Recorder) and not hasattr(model, "_paged_state")
    cls.copy_kv_cache_blocks(me, None, [(3, 4)])
    cls.zero_kv_cache_blocks(me, None, [5])
    assert model._spec_blocks.calls == [("copy", [(3, 4)]), ("zero", [5])]


def test_zeroing_runs_at_once_and_skips_copied_blocks_never(monkeypatch):
    """zero_blocks zeroes every listed block before the hook returns (vLLM applies its block copies after the zeroing);
    an empty list runs no device work."""
    st = sbs.SpecGdnBlockState.__new__(sbs.SpecGdnBlockState)
    forgotten, zeroed = [], []
    st.book = SimpleNamespace(forget_block=forgotten.append)
    st._zero_now = lambda ids: zeroed.append(list(ids))
    monkeypatch.setattr(sbs.SpecGdnBlockState, "_timed", lambda self, kind, n: contextlib.nullcontext())
    st.zero_blocks([5, 6, 7])
    assert zeroed == [[5, 6, 7]] and forgotten == [5, 6, 7]
    st.zero_blocks([])
    assert zeroed == [[5, 6, 7]]


class _FT:
    """A fake device tensor: a label, or per-user rows (the conv window). Its head count never matches a pool's, so a
    head slice is always taken."""

    shape = (1, -1, 1, 1)

    def __init__(self, label=None, users=None, rows=None):
        self.label, self.users, self.rows = label, users, rows


class _View:
    """A flat alias view: page -> label; ``gran`` pages per dim-0 slice."""

    def __init__(self, gran):
        self.gran, self.pages = gran, {}


def _fake_ttnn(monkeypatch):
    import ttnn

    def sl(t, start, end, memory_config=None):
        if isinstance(t, _View):
            labs = {t.pages.get(p) for p in range(start[0] * t.gran, end[0] * t.gran)}
            assert len(labs) == 1, f"a read spans pages of different states: {labs}"
            return _FT(labs.pop())
        if t.users is not None:
            return _FT(rows=t.users[start[0]])
        if t.rows is not None:
            return _FT(label=tuple(t.rows[start[1] : end[1]]))
        return _FT(t.label)

    def fill(view, src, pt, batch_idx=0):
        for e in torch.as_tensor(pt).reshape(-1).tolist():
            for p in range(e * view.gran, (e + 1) * view.gran):
                view.pages[p] = src.label

    def concat(parts, dim=0, memory_config=None):
        labs = {p.label for p in parts}
        assert len(labs) == 1, labs
        return _FT(labs.pop())

    same = lambda t, *a, **k: t if isinstance(t, _FT) and (t.rows is not None or t.users is not None) else _FT(t.label)
    monkeypatch.setattr(ttnn, "slice", sl)
    monkeypatch.setattr(ttnn, "reshape", same)
    monkeypatch.setattr(ttnn, "to_layout", same)
    monkeypatch.setattr(ttnn, "pad", same)
    monkeypatch.setattr(ttnn, "concat", concat)
    monkeypatch.setattr(ttnn, "copy", lambda src, dst: setattr(dst, "label", src.label))
    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    monkeypatch.setattr(ttnn, "from_torch", lambda t, **k: t)
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: None)
    monkeypatch.setattr(ttnn.experimental, "paged_fill_cache", fill)


@pytest.mark.parametrize("geo", [(4, 2048, 8), (8, 1024, 8), (8, 2048, 8)])
def test_spec_state_device_writes_land_where_the_verify_reads(monkeypatch, geo):
    """SpecGdnBlockState's eager writes (prefill, copy materialize) and reads (restore) run with a fake ttnn that keeps
    each view page's label; the verify's ctrl start pages (SpecBlockLayout via SpecBlockBook) must find the prefill's
    state, a restore must return it, and a copy from a window column must carry that column's conv rows."""
    from models.demos.blackhole.qwen36.tt import gdn_block_state as gbs

    _fake_ttnn(monkeypatch)
    tp = geo[0]
    lay = _layout(*geo)
    C = (2 * 16 * 128 + 48 * 128) // tp
    views = sbs._LayerViews(
        rec1=_View(sbs.HALF_PAGES),
        rec2=_View(sbs.HALF_PAGES),
        conv=_View(1),
        conv_k=_View(1),
        group=0,
        two_pool=lay.heads_in_k < lay.nv,
    )
    win_rows = [f"x{r}" for r in range(3 + T)]
    dn = SimpleNamespace(
        Nv=lay.nv,
        Dk=128,
        Dv=128,
        K=4,
        qkv_dim_tp=C,
        rec_state=_FT("rec:prompt"),
        conv_carry=_FT("conv:prompt"),
        _spec_paged=views,
        verify_win_cur=lambda: _FT(users={2: win_rows}),
    )
    st = sbs.SpecGdnBlockState.__new__(sbs.SpecGdnBlockState)
    st.model, st.mesh, st.T, st.G, st.layout = None, None, T, 1, lay
    st.layers, st.layer_groups = [(0, dn, 0)], {0: 0}
    st.book = sbs.SpecBlockBook(lay, 1, B, T, "align", 1024)

    def heads(b):  # what the verify reads: the start pages its ctrl page names
        w = torch.zeros(B, T, dtype=torch.int64)
        w[2] = torch.tensor([b, b + 1, b + 2, b + 3])
        st.book.set_step([w], torch.zeros(B, dtype=torch.int64), [2])
        page = st.book.ctrl_rows([0] * B, 0, spec_ctrl_words(B, lay.nv))[0].to(torch.int64)
        init = page[1 + B : 1 + B + B * lay.nv].reshape(B, lay.nv)[2].tolist()
        return [(views.rec1 if p < lay.pool1_pages else views.rec2).pages.get(p % lay.pool1_pages) for p in init]

    st.book.set_prefill_blocks([torch.tensor([7])])
    st.write_prefill(0)  # the prefill's state into block 7
    assert heads(7) == ["rec:prompt"] * lay.nv
    dn.rec_state, dn.conv_carry = _FT(), _FT()
    st.restore(0)  # a prefix hit on block 7
    assert (dn.rec_state.label, dn.conv_carry.label) == ("rec:prompt", "conv:prompt")

    def whole_block(model, pairs):  # copy_kv_blocks moves every byte of the block; here every page of the layout
        for src, dst in pairs:
            ps, pd = lay.head_pages(torch.tensor([src]))[0].tolist(), lay.head_pages(torch.tensor([dst]))[0].tolist()
            for a, b in zip(ps, pd):
                for i in range(sbs.HEAD_PAGES):
                    va = views.rec1 if a < lay.pool1_pages else views.rec2
                    va.pages[b % lay.pool1_pages + i] = va.pages.get(a % lay.pool1_pages + i)
            cs, cd = lay.conv_page(src), lay.conv_page(dst)
            for i in range(lay.conv_pages):
                vs, ps_ = views.conv_at(cs + i, lay.conv_pages_total)
                vd, pd_ = views.conv_at(cd + i, lay.conv_pages_total)
                vd.pages[pd_] = vs.pages.get(ps_)

    monkeypatch.setattr(gbs, "copy_kv_blocks", whole_block)
    heads(20)  # row 2's latest verify: window blocks 20 .. 23
    st.copy_blocks([(22, 40)])  # vLLM copies window column 2 on
    view, q = views.conv_at(lay.conv_page(40), lay.conv_pages_total)
    assert view.pages.get(q) == ("x3", "x4", "x5")  # the conv rows after token 2: window rows 3 .. 5
    st.book.set_prefill_blocks([torch.tensor([40])])
    st.restore(0)
    assert dn.conv_carry.label == ("x3", "x4", "x5")


# ------------------------------------------------------------------------------------- chained state moves
# Chained state moves: on a chained verify in mode align the plugin sends (the chain of each mamba_state group), per
# accepted count c of the pending verify, each row's start block and state window. The chain body gathers both tables
# (SpecBlockBook.pages_cat / windows_cat) at column c - 1; the host adopts the window of the actual count once it has
# the count.


def _moves_case(G=2):
    lay = _layout(4, 2048, 8)
    book = sbs.SpecBlockBook(lay, G, B, T, "align", 2048)
    stay = torch.arange(1, B * T + 1).reshape(B, T) * 2 + 10  # every row's staged (optimistic) window
    blocks = [stay + 100 * g for g in range(G)]
    book.set_step(blocks, torch.full((B,), T - 1), range(B))
    # row 1 chained: counts 1..2 stay in the staged window's previous column set, counts 3..4 move one column on
    starts = [torch.zeros(B, T, dtype=torch.int64) for _ in range(G)]
    wins = [torch.zeros(B, T, T, dtype=torch.int64) for _ in range(G)]
    for g in range(G):
        old = torch.tensor([60, 61, 62, 63]) + 100 * g
        new = blocks[g][1]
        starts[g][1] = torch.tensor([old[0], old[1], old[2], old[3]])
        wins[g][1, :2] = old
        wins[g][1, 2:] = new
    return lay, book, blocks, starts, wins


def test_chain_tables_select_start_block_and_window_per_count():
    lay, book, blocks, starts, wins = _moves_case()
    G, nv = 2, lay.nv
    plain_pages, plain_wins = book.pages_cat(), book.windows_cat()
    book.set_moves(starts, wins, [1])
    pages, wtab = book.pages_cat(), book.windows_cat()
    for u in range(B):
        for c in range(1, T + 1):
            for g in range(G):
                got_start = pages[u, (c - 1) * G * nv + g * nv : (c - 1) * G * nv + (g + 1) * nv].to(torch.int64)
                got_win = wtab[u, ((c - 1) * G + g) * T * nv : ((c - 1) * G + g + 1) * T * nv].to(torch.int64)
                if u == 1:
                    want_start = lay.head_pages(starts[g][1, c - 1])
                    want_win = lay.head_pages(wins[g][1, c - 1]).reshape(-1)
                else:  # rows without moves: start column c - 1 of the staged window, the staged window at every count
                    want_start = lay.head_pages(blocks[g][u, c - 1])
                    want_win = lay.head_pages(blocks[g][u]).reshape(-1)
                assert torch.equal(got_start, want_start), (u, c, g)
                assert torch.equal(got_win, want_win), (u, c, g)
    others = [u for u in range(B) if u != 1]
    assert torch.equal(pages[others], plain_pages[others]) and torch.equal(wtab[others], plain_wins[others])


@pytest.mark.parametrize("c", [1, 2, 3, 4])
def test_chained_row_adopts_the_window_of_its_count(c):
    """Once the host has the count, a copy from the window column of the actual count carries that column's conv rows;
    the optimistic window's blocks no longer map to the row."""
    lay, book, blocks, starts, wins = _moves_case()
    book.set_moves(starts, wins, [1])
    book.resolve_moves(torch.tensor([T, c, T, T]))
    assert not book.moves
    for g in range(2):
        for t in range(T):
            b = int(wins[g][1, c - 1, t])
            assert book.window_map[b] == (1, t, g)
            assert int(book.idx_host[g][1, t * lay.nv]) == int(lay.head_pages(b)[0])
    ops = book.copy_plan([(int(wins[0][1, c - 1, 1]), 199)])
    assert ops[-1] == ("conv", 1, 1, 0, 199)
    if c <= 2:  # the optimistic window is not the actual one: its blocks are not the row's any more
        assert int(blocks[0][1, 3]) not in book.window_map


def test_an_unresolved_chained_row_forces_the_next_upload():
    lay, book, blocks, starts, wins = _moves_case()
    book.set_moves(starts, wins, [1])
    changed = book.set_step(blocks, torch.zeros(B, dtype=torch.int64), range(B))
    assert changed == [0, 1] and not book.moves


def test_a_released_row_drops_its_unresolved_moves():
    """A request ends (or prefills again) on a row whose last chained step's count never reached the host. Its moves go
    with its window map: resolving them afterwards must not map the old window to the row, so a copy from one of those
    block ids (vLLM reuses them, e.g. as the next request's prefill state block) copies the block's own conv history
    instead of the old verify window's rows, and the next step uploads every group's window rows again."""
    lay, book, blocks, starts, wins = _moves_case()
    book.set_moves(starts, wins, [1])
    book.forget_row(1)
    book.resolve_moves(torch.tensor([T, 3, T, T]))
    reused = int(wins[0][1, 2, 1])
    assert reused not in book.window_map
    ops = book.copy_plan([(reused, 199)])
    assert [op[0] for op in ops] in (["copy"], ["relayout"]) and ops[0][1:] == (reused, 199)
    assert book.set_step(blocks, torch.zeros(B, dtype=torch.int64), range(B)) == [0, 1]


def test_a_chained_row_the_host_stages_starts_from_its_start_block():
    """A chain that falls back to the host (count known): set_start adopts that count's window and start block, and in
    mode align a start at a block's last position leaves the conv write of window column 0 due."""
    lay, book, blocks, starts, wins = _moves_case()
    book.set_moves(starts, wins, [1])
    changed = book.set_start({1: 1}, start_pos=torch.tensor([0, 2047, 0, 0]))
    assert changed == [0, 1]
    for g in range(2):
        assert torch.equal(book.init[g][1], lay.head_pages(starts[g][1, 1]))
        assert torch.equal(book.step_blocks[g][1], wins[g][1, 1])
    assert book.take_pending() == [(1, 0, 0, int(wins[0][1, 1, 0])), (1, 0, 1, int(wins[1][1, 1, 0]))]


def test_chain_boundary_conv_write_targets_the_count_that_starts_there():
    lay, book, blocks, starts, wins = _moves_case()
    book.pending = [(1, 0)]  # set_step's guess from a provisional position: dropped for a chained row
    book.set_moves(starts, wins, [1])
    assert book.pending == []
    book.chain_boundaries(torch.tensor([0, 2046, 0, 0]))  # count 2 starts at position 2047
    assert book.take_pending() == [(1, 0, 0, int(wins[0][1, 1, 0])), (1, 0, 1, int(wins[1][1, 1, 0]))]


def test_chain_moves_are_grouped_with_their_copies():
    groups = sbs.gdn_groups_by_slot(_per_layer_specs(), lambda e: isinstance(e[0], tuple))
    blocks = SimpleNamespace(layer_groups=groups, G=3)

    def state(copy_src=0):
        def group(g):
            return SimpleNamespace(
                layer_names=tuple(gdn_layer_name(i) for i, h in groups.items() if h == g),
                windows=torch.full((2, T), 10 + g),
                checkpoint_blocks=None,
                chain=SimpleNamespace(
                    copies=torch.full((2, T, 2), 0).index_fill_(2, torch.tensor([0]), copy_src),
                    start_blocks=torch.full((2, T), 20 + g),
                    windows=torch.full((2, T, T), 30 + g),
                ),
            )

        return SimpleNamespace(
            groups=tuple(group(g) for g in range(3)), accepted_counts=torch.tensor([-1, -1]), checkpoint_positions=None
        )

    out, counts, moves = Qwen36MTPContractForCausalLM._gdn_windows(state(), blocks, 4)
    assert [int(w[0, 0]) for w in out] == [10, 11, 12] and counts.tolist() == [-1, -1, 1, 1]
    assert [int(s[1, 3]) for s in moves[0]] == [20, 21, 22] and [int(s[3, 0]) for s in moves[0]] == [0, 0, 0]
    assert [tuple(w.shape) for w in moves[1]] == [(4, T, T)] * 3 and int(moves[1][2][0, 1, 2]) == 32
    assert moves[2] is None  # every pair (0, 0): nothing to copy
    _, _, moves = Qwen36MTPContractForCausalLM._gdn_windows(state(copy_src=7), blocks, 4)
    assert [tuple(c.shape) for c in moves[2]] == [(4, T, 2)] * 3
    assert int(moves[2][1][1, 2, 0]) == 7 and int(moves[2][1][3, 0, 0]) == 0  # padding rows: (0, 0)


def _torch_ops_over_fake(monkeypatch):
    """The fake ttnn of _fake_ttnn plus plain torch semantics for real tensors (the device-selected copy's count math)."""
    import ttnn

    fake_slice, fake_same = ttnn.slice, ttnn.reshape
    dt = {ttnn.float32: torch.float32, ttnn.int32: torch.int32}

    def sl(t, start, end, memory_config=None):
        if isinstance(t, torch.Tensor):
            return t[tuple(slice(a, b) for a, b in zip(start, end))]
        return fake_slice(t, start, end, memory_config)

    keep = lambda f: lambda t, *a, **k: t if isinstance(t, torch.Tensor) else f(t, *a, **k)
    monkeypatch.setattr(ttnn, "slice", sl)
    monkeypatch.setattr(ttnn, "to_layout", keep(fake_same))
    monkeypatch.setattr(ttnn, "typecast", lambda t, d, **k: t.to(dt[d]))
    monkeypatch.setattr(ttnn, "eq", lambda t, c, **k: (t == c).to(torch.float32))
    monkeypatch.setattr(ttnn, "add", lambda a, b, **k: a + b)
    monkeypatch.setattr(ttnn, "multiply", lambda a, b, **k: a * b)


def _chain_copy_state(monkeypatch):
    """One GDN layer (group 0) on fake views; row 2's latest verify wrote window blocks 20 .. 23, block 21 holds the
    state after token 1 ("rec:21") and the conv window holds rows x0 .. x6."""
    _fake_ttnn(monkeypatch)
    _torch_ops_over_fake(monkeypatch)
    lay = _layout(4, 2048, 8)
    views = sbs._LayerViews(
        rec1=_View(sbs.HALF_PAGES), rec2=_View(sbs.HALF_PAGES), conv=_View(1), conv_k=_View(1), group=0, two_pool=True
    )
    rows = [f"x{r}" for r in range(3 + T)]
    dn = SimpleNamespace(
        Nv=lay.nv,
        Dk=128,
        Dv=128,
        K=4,
        qkv_dim_tp=(2 * 16 * 128 + 48 * 128) // 4,
        _spec_paged=views,
        verify_win_cur=lambda: _FT(users={2: rows}),
    )
    st = sbs.SpecGdnBlockState.__new__(sbs.SpecGdnBlockState)
    st.model, st.mesh, st.T, st.G, st.layout = None, None, T, 1, lay
    st.layers, st.layer_groups, st._slot_layers = [(0, dn, 0)], {0: 0}, {0: dn}
    st.book = sbs.SpecBlockBook(lay, 1, B, T, "align", 2048)
    st.moves_counts, st.chain_copies = None, []
    w = torch.zeros(B, T, dtype=torch.int64)
    w[2] = torch.tensor([20, 21, 22, 23])
    st.book.set_step([w], torch.zeros(B, dtype=torch.int64), [2])
    with sbs._PageTables(st) as pts:
        for b in (21, 40):
            st._write_rec(dn, b, _FT(f"rec:{b}"), pts)
    rec = lambda b: {
        (views.rec1 if p < lay.pool1_pages else views.rec2).pages.get(int(p) % lay.pool1_pages)
        for p in lay.head_pages(torch.tensor([b]))[0].tolist()
    }
    conv = lambda b: views.conv_at(lay.conv_page(b), lay.conv_pages_total)[0].pages.get(
        views.conv_at(lay.conv_page(b), lay.conv_pages_total)[1]
    )
    # counts 2 .. 4 of row 2 accept the token whose state ends a block: copy window column 1 (block 21) into block 40
    cp = torch.zeros(B, T, 2, dtype=torch.int64)
    cp[2, 1:] = torch.tensor([21, 40])
    return st, rec, conv, cp


@pytest.mark.parametrize("count", [1, 2, 4])
@pytest.mark.parametrize("on_device", [False, True])
def test_chained_copy_runs_for_the_counts_that_select_it(monkeypatch, count, on_device):
    """The chain copies of a chained step (mamba_state): the copy of the pending verify's window column runs only when the
    row's accepted count selects it, from the host count (fallback) or the device count (result buffer row RES_COUNTS;
    an unselected copy writes into the null block 0 instead), and carries that column's conv rows."""
    from models.demos.blackhole.qwen36.tt import gdn_block_state as gbs
    from models.demos.blackhole.qwen36.tt.spec_chain import RES_COUNTS

    st, rec, conv, cp = _chain_copy_state(monkeypatch)
    monkeypatch.setattr(
        gbs,
        "copy_kv_blocks",
        lambda model, pairs: [st._relayout(s, d, lambda f, n: torch.arange(f, f + n)) for s, d in pairs],
    )
    st.capture_chain_copies([cp], [2])
    assert st.chain_copies == [(2, 0, 21, 40, 1, (2, 3, 4))]
    if on_device:
        res = torch.zeros(3, 32, dtype=torch.int32)
        res[RES_COUNTS, 2] = count
        st.run_chain_copies(result=res)
    else:
        st.run_chain_copies(counts={2: count})
    assert st.chain_copies == []
    if count >= 2:
        assert rec(40) == {"rec:21"} and conv(40) == ("x2", "x3", "x4")  # conv rows after token 1
    else:
        assert rec(40) == {"rec:40"} and conv(40) is None
        if on_device:
            assert rec(0) == {"rec:21"} and conv(0) == ("x2", "x3", "x4")


def test_a_chained_copy_source_the_older_count_decides_is_looked_up_after_reading_that_count():
    """A row whose own chained moves are unresolved: when its copy source sits at different window columns by the older
    pending count, capture_chain_copies reads that count (moves_counts(wait=True)) and uses the adopted window."""
    lay, book, blocks, starts, wins = _moves_case(G=1)
    st = sbs.SpecGdnBlockState.__new__(sbs.SpecGdnBlockState)
    st.book, st.T, st.chain_copies = book, T, []
    wins[0][1, 0] = torch.tensor([60, 61, 62, 63])
    wins[0][1, 3] = torch.tensor([61, 70, 71, 72])  # block 61: column 1 at count 1, column 0 at count 4
    book.set_moves(starts, wins, [1])
    calls = []

    def counts(wait=False):
        calls.append(wait)
        return torch.tensor([1, 4, 1, 1]) if wait else None

    st.moves_counts = counts
    cp = torch.zeros(B, T, 2, dtype=torch.int64)
    cp[1, 3] = torch.tensor([61, 90])
    st.capture_chain_copies([cp], [1])
    assert calls == [False, True] and st.chain_copies == [(1, 0, 61, 90, 0, (4,))]


def test_chain_audit_twin_reads_the_pages_of_its_own_staging():
    """The chain audit runs when the reader thread reads the step, after the engine staged later steps. Its expected
    ctrl words must come from the start pages the step was staged with (start_pages_snapshot), not from the live book:
    a row that finished since then holds the null block in the live book."""
    lay = _layout(8, 1024, 8)
    book = sbs.SpecBlockBook(lay, 1, B, T, "none")
    words = spec_ctrl_words(B, lay.nv)
    w = torch.arange(1, B * T + 1).reshape(B, T) + 20
    book.set_step([w], torch.zeros(B, dtype=torch.int64), range(B))
    snap = book.start_pages_snapshot()
    staged = book.ctrl_rows([1] * B, 0, words, start_col=[1] * B)
    later = w.clone()
    later[3] = 0  # row 3 finished: padding holds block 0
    book.set_step([later], torch.zeros(B, dtype=torch.int64), [u for u in range(B) if u != 3])
    assert torch.equal(book.ctrl_rows([1] * B, 0, words, start_col=[1] * B, pages=snap), staged)
    assert not torch.equal(book.ctrl_rows([1] * B, 0, words, start_col=[1] * B), staged)


def test_model_aux_builder_passes_the_snapshot_through_the_block_state():
    """Qwen36Model._vfy_aux_host with a SpecGdnBlockState (not the bare book): the chain-audit snapshot reaches
    SpecBlockBook.ctrl_rows."""
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    lay = _layout(4, 2048, 8)
    st = sbs.SpecGdnBlockState.__new__(sbs.SpecGdnBlockState)
    st.book = sbs.SpecBlockBook(lay, 1, B, T, "none")
    w = torch.arange(1, B * T + 1).reshape(B, T) + 20
    st.book.set_step([w], torch.zeros(B, dtype=torch.int64), range(B))
    snap = st.book.start_pages_snapshot()
    words = spec_ctrl_words(B, lay.nv)
    model = SimpleNamespace(
        _vfy_fused=True,
        _vfy_gdn=[SimpleNamespace(_spec_win_par=0)],
        _spec_blocks=st,
        _vfy_ctrl_words=words,
        _vfy_B=B,
        _vfy_T=T,
        rope=SimpleNamespace(max_seq_len=4096),
    )
    out = Qwen36Model._vfy_aux_host(
        model, list(range(B * T)), [1] * B, rope_delta=0, predict=True, hold=(), gdn_pages=snap
    )
    assert int(out.shape[0]) == 3


class _AddrFT(_FT):
    def __init__(self, label, addr):
        super().__init__(label)
        self.addr = addr

    def buffer_address(self):
        return self.addr


def test_write_trace_replays_the_scratch_into_each_target_block(monkeypatch):
    """_WriteTrace: the captured prefill block write restages its page tables for the target blocks and
    lands the bound scratch's state and conv history there; a moved scratch address makes the caller write eagerly."""
    import ttnn

    st, rec, conv, _ = _chain_copy_state(monkeypatch)
    dn = st.layers[0][1]
    dn.rec_state, dn.conv_carry = _AddrFT("rec:scratch", 100), _AddrFT("conv:scratch", 200)
    replays = []

    def copy_h2d(host, dev):
        dev.copy_(host)

    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", copy_h2d)
    monkeypatch.setattr(ttnn, "synchronize_device", lambda mesh: None)
    monkeypatch.setattr(ttnn, "begin_trace_capture", lambda mesh, cq_id=0: 7)
    monkeypatch.setattr(ttnn, "end_trace_capture", lambda mesh, tid, cq_id=0: None)
    wt = sbs._WriteTrace(st)
    wt.capture()
    monkeypatch.setattr(
        ttnn, "execute_trace", lambda mesh, tid, cq_id=0, blocking=False: replays.append(tid) or wt.body()
    )
    st._write_trace = wt
    st.book.set_prefill_blocks([torch.tensor([40])])
    st.write_prefill(0)
    assert replays == [7] and rec(40) == {"rec:scratch"} and conv(40) == "conv:scratch"
    dn.rec_state = _AddrFT("rec:other", 101)  # a different binding: not the captured addresses
    assert not wt.replay([41])
    st.book.set_prefill_blocks([torch.tensor([41])])
    st.write_prefill(0)  # eager
    assert replays == [7] and rec(41) == {"rec:other"}
