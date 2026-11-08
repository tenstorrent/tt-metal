# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: QWEN36_MAMBA_SPEC with prefix caching (mamba_cache_mode "align") under vLLM 0.31's scheduler.

A real TTScheduler (vllm-tt-plugin) over a real KVCacheManager schedules requests. The KV cache groups come from the
model's own specs (tt/kv_cache_spec.qwen36_kv_cache_spec, 1 attention and 3 GDN groups of 16 layers). Each step, the
plugin's MambaCacheGroups gives the state copies before the forward (begin_step) and each row's state window per group
(step_inputs), the calls TTModelRunner makes. The model's mamba_state keyword built from them goes through
Qwen36ForCausalLM's conversion (qwen36_vllm._gdn_step_blocks), which maps the groups to the GDN layers by layer name.
A ledger then plays the model:

* block zeroing and (src, dst) copies as Qwen36ForCausalLM.zero_kv_cache_blocks / copy_kv_cache_blocks do them: whole
  blocks, in list order;
* a prefill row resumes at start_pos from its state block when Qwen36Model._plan_block_resume plans a resume, and
  recomputes from 0 otherwise; a decode row reads its state block;
* every row writes the state after its last scheduled token into its state block.

Every read must find the state after exactly the tokens the request has computed. A state is a label of those tokens.

Needs vLLM 0.31 and vllm-tt-plugin (src/ on PYTHONPATH), and the ttnn host stub for the model import:
    PYTHONPATH=<worktree>:<plugin>/src:<tools>/ttnn-host-stub pytest --noconftest \
        models/demos/blackhole/qwen36/tests/test_mamba_align_host.py
"""

import json
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm.v1.core.sched.scheduler")
pytest.importorskip("vllm_tt_plugin.mamba_state")

from vllm.config import CacheConfig, DeviceConfig, ModelConfig, SchedulerConfig, VllmConfig  # noqa: E402
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
from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402
from vllm.v1.request import Request  # noqa: E402
from vllm.v1.structured_output import StructuredOutputManager  # noqa: E402
from vllm_tt_plugin.mamba_state import mamba_cache_groups  # noqa: E402
from vllm_tt_plugin.scheduler import TTScheduler  # noqa: E402

from models.demos.blackhole.qwen36.tests.test_kv_cache_spec_host import TEXT_CONFIG  # noqa: E402
from models.demos.blackhole.qwen36.tt.kv_cache_spec import gdn_layer_name, qwen36_kv_cache_spec  # noqa: E402
from models.demos.blackhole.qwen36.tt.model import Qwen36Model  # noqa: E402
from models.demos.blackhole.qwen36.tt.qwen36_vllm import _gdn_step_blocks  # noqa: E402

MAX_NUM_SEQS = 4
NUM_LAYERS = 64
# A tiny HF config for ModelConfig; the scheduler reads no model weights.
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


def plugin_windows(groups, tables, req_ids, num_computed, num_scheduled, num_rows):
    """vllm-tt-plugin's MambaCacheGroups.step_inputs of one step: (each MambaSpec group's windows [num_rows, 1 + K] in
    group_indices order, the start-state counts [num_rows] or None)."""
    return groups.step_inputs(tables, req_ids, num_computed, num_scheduled, num_rows)


def plugin_checkpoints(groups, checkpoints, req_ids, requests, num_rows):
    """vllm-tt-plugin's MambaCacheGroups.checkpoint_inputs: (each MambaSpec group's checkpoint blocks [num_rows, W] in
    group_indices order, the positions [num_rows, W])."""
    return groups.checkpoint_inputs(checkpoints, req_ids, requests, num_rows)


def mamba_state(groups, windows, rows, counts=None, checkpoints=None):
    """The model's mamba_state keyword for ``rows`` of a step, from plugin_windows (``windows``, ``counts``) and, for a
    prefill, plugin_checkpoints of those rows: vllm-tt-plugin's MambaCacheGroups.step_state."""
    windows = [w[rows] for w in windows]
    counts = None if counts is None else counts[rows]
    blocks, positions = (None, None) if checkpoints is None else checkpoints
    return groups.step_state(windows, counts, blocks, positions)


class _Ledger:
    """Device state per block id: the label of the tokens a GDN state block holds the state after (None: zeros)."""

    def __init__(self):
        self.blocks = {}
        self.copies = []

    def zero(self, block_ids):
        for b in block_ids:
            self.blocks[b] = None

    def copy(self, pairs):
        for src, dst in pairs:
            self.copies.append((src, dst))
            if src != dst:
                self.blocks[dst] = self.blocks.get(src)


def _plan(L, start, block):
    """Qwen36Model._plan_block_resume of a model with the chunk trace captured (chunk 2048) and a paged KV cache of
    `block`-token blocks."""
    me = SimpleNamespace(
        _prefix_cache=None,
        _chunk_state_cache=None,
        _PREFIX_ALIGN=128,
        _chunked_trace_id=1,
        _chunked_chunk_size=2048,
        _paged_kv_caches=[[SimpleNamespace(shape=(8, 1, block, 256))]],
    )
    return Qwen36Model._plan_block_resume(me, L, start, ("blocks", 0))


class _Harness:
    def __init__(self, tmp_path, tp, block, num_blocks, max_model_len, budget, retention=0):
        (tmp_path / "config.json").write_text(json.dumps(_QWEN2))
        model_config = ModelConfig(model=str(tmp_path), dtype="float16", seed=0, skip_tokenizer_init=True)
        model_config.max_model_len = max_model_len
        cfg = VllmConfig(
            model_config=model_config,
            scheduler_config=SchedulerConfig(
                max_num_seqs=MAX_NUM_SEQS,
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
        cfg.scheduler_config.max_num_batched_tokens = budget
        cfg.scheduler_config.enable_chunked_prefill = True
        cfg.cache_config.kv_cache_layout = "LBNHC"
        cfg.cache_config.enable_prefix_caching = True
        cfg.cache_config.mamba_cache_mode = "align"
        # vLLM's default 0 keeps only semantic boundary states; None keeps every block boundary.
        cfg.cache_config.prefix_cache_retention_interval = retention
        specs = qwen36_kv_cache_spec(TEXT_CONFIG, tp, block, kv_bf8=True, mamba_cache_mode="align")
        groups = get_kv_cache_groups(cfg, specs)
        page = next(iter(specs.values())).page_size_bytes
        self.kv = get_kv_cache_config_from_groups(cfg, groups, 16 * page * num_blocks + 1)
        assert self.kv.num_blocks == num_blocks
        cfg.cache_config.num_gpu_blocks = num_blocks
        bs, hash_bs = resolve_kv_cache_block_sizes(self.kv, cfg)
        cfg.cache_config.block_size = block
        self.block = block
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
        self.gdn_layers = [i for i in range(NUM_LAYERS) if gdn_layer_name(i) in specs]
        self.max_model_len = max_model_len
        self.ledger = _Ledger()
        self.req_states = {}
        self.hits = {}  # request id -> computed tokens it arrived with (a prefix hit)
        self.resumes = []  # (request id, start, end) of prefill rows that resumed from their state block
        self.recomputes = []  # (request id, start, end) of prefill rows with start > 0 recomputed from 0
        self.n_out = 0

    def add(self, req_id, prompt, max_tokens=4):
        init_none_hash(sha256)
        params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
        params.update_from_generation_config({}, eos_token_id=2)
        self.scheduler.add_request(
            Request(
                request_id=req_id,
                prompt_token_ids=list(prompt),
                sampling_params=params,
                pooling_params=None,
                block_hasher=get_request_block_hasher(self.block, sha256),
            )
        )

    def run(self, max_steps=400):
        for _ in range(max_steps):
            if not self.step():
                return
        raise AssertionError("scenario did not finish")

    def step(self):
        out = self.scheduler.schedule()
        if not out.num_scheduled_tokens:
            return False
        for r in out.finished_req_ids:
            self.req_states.pop(r, None)
        for new in out.scheduled_new_reqs:
            self.req_states[new.req_id] = SimpleNamespace(
                num_computed_tokens=new.num_computed_tokens, block_ids=[list(ids) for ids in new.block_ids]
            )
            self.hits[new.req_id] = new.num_computed_tokens
        cached = out.scheduled_cached_reqs
        for i, r in enumerate(cached.req_ids):
            st = self.req_states[r]
            st.num_computed_tokens = cached.num_computed_tokens[i]
            if cached.new_block_ids[i] is not None:
                for row, ids in zip(st.block_ids, cached.new_block_ids[i]):
                    row.extend(ids)
        # TTModelRunner order: zeroing, vLLM's copies, then the Mamba state copies of begin_step.
        self.ledger.zero(out.new_block_ids_to_zero or ())
        self.ledger.copy([(c.src_block_id, c.dst_block_id) for c in out.kv_cache_block_copies or ()])
        # the position of each scheduled row's first query token (synchronous scheduling: vLLM's computed tokens)
        positions = {r: st.num_computed_tokens for r, st in self.req_states.items()}
        self.ledger.copy(self.groups.begin_step(out, self.req_states, positions, {}))
        req_ids = list(out.num_scheduled_tokens)
        width = self.groups.max_num_blocks(self.max_model_len)
        tables = []
        for g in range(len(self.kv.kv_cache_groups)):
            t = torch.zeros((len(req_ids), width), dtype=torch.int32)
            for row, r in enumerate(req_ids):
                ids = self.req_states[r].block_ids[g]
                t[row, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
            tables.append(t)
        windows, _ = plugin_windows(self.groups, tables, req_ids, positions, out.num_scheduled_tokens, MAX_NUM_SEQS)
        state = mamba_state(self.groups, windows, list(range(len(req_ids))))
        state_blocks, _ = _gdn_step_blocks(state, self.gdn_layers)
        sampled = []
        for row, r in enumerate(req_ids):
            request = self.scheduler.requests[r]
            nc, ns = self.req_states[r].num_computed_tokens, out.num_scheduled_tokens[r]
            tokens = list(request.all_token_ids)
            prefill = nc < request.num_prompt_tokens
            blocks = [int(b[row]) for b in state_blocks]
            assert 0 not in blocks, (r, nc, blocks)
            read = nc > 0
            if prefill and nc > 0:
                if _plan(nc + ns, nc, self.block) is None:
                    read = False
                    self.recomputes.append((r, nc, nc + ns))
                else:
                    self.resumes.append((r, nc, nc + ns))
            for b in blocks:
                if read:
                    assert self.ledger.blocks.get(b) == _label(tokens[:nc]), (r, nc, ns, b)
                self.ledger.blocks[b] = _label(tokens[: nc + ns])
            if nc + ns >= request.num_tokens:
                self.n_out += 1
                sampled.append([1000 + self.n_out])
            else:
                sampled.append([])
        self.scheduler.update_from_output(
            out,
            ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={r: i for i, r in enumerate(req_ids)},
                sampled_token_ids=sampled,
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
        return True


def _prompt(seed, n):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(10, 30_000, (n,), generator=g).tolist()


# (TP, block): Galaxy / 8-chip BH at BF8 2048, P150x4 at BF8 4096.
LAYOUTS = [(8, 2048), (4, 4096)]


@pytest.fixture(params=LAYOUTS, ids=lambda p: f"tp{p[0]}_b{p[1]}")
def layout(request):
    return request.param


def _harness(tmp_path, layout, num_blocks=64, max_model_len=32768, budget=None, retention=0):
    tp, block = layout
    return _Harness(tmp_path, tp, block, num_blocks, max_model_len, budget or max_model_len, retention)


def test_repeated_prompt_resumes_from_the_cached_boundary_state(tmp_path, layout):
    h = _harness(tmp_path, layout)
    bs = h.block
    a = _prompt(1, 2 * bs + bs // 2)
    h.add("a", a)
    h.run()
    h.add("a2", a)
    h.run()
    assert h.hits["a2"] == 2 * bs
    assert ("a2", 2 * bs, len(a)) in h.resumes


def test_shared_two_block_prefix_resumes_there(tmp_path, layout):
    h = _harness(tmp_path, layout)
    bs = h.block
    a = _prompt(2, 2 * bs + 300)
    b = a[: 2 * bs] + _prompt(3, 700)
    h.add("a", a)
    h.run()
    h.add("b", b)
    h.run()
    assert h.hits["b"] == 2 * bs and ("b", 2 * bs, len(b)) in h.resumes


def test_multi_turn_follow_up_resumes_from_the_state_decode_left(tmp_path, layout):
    # The first turn ends 2 tokens short of a block boundary and decodes past it, so the boundary state the follow-up
    # hits was written by decode steps and moved into its own block by begin_step's copy. Dense retention keeps that
    # boundary (vLLM's default keeps only the prompt's chunk-end boundary, one block lower).
    h = _harness(tmp_path, layout, retention=None)
    bs = h.block
    a = _prompt(4, 2 * bs - 2)
    h.add("t1", a, max_tokens=6)
    h.run()
    first_turn = a + [1000 + i for i in range(1, 7)]
    h.add("t2", first_turn[: 2 * bs + 4] + _prompt(5, 500))
    h.run()
    assert h.hits["t2"] == 2 * bs and ("t2", 2 * bs, 2 * bs + 504) in h.resumes


def test_evicted_boundary_state_is_not_hit(tmp_path, layout):
    h = _harness(tmp_path, layout, num_blocks=24)
    bs = h.block
    a = _prompt(6, 2 * bs + 100)
    h.add("a", a)
    h.run()
    for k in range(4):  # unrelated prompts that need every free block
        h.add(f"x{k}", _prompt(10 + k, 3 * bs))
        h.run()
    h.add("a2", a)
    h.run()
    # No hit: a2 recomputes from 0, and its only resume is at the end of its own first chunk (the prompt splits at the
    # last block boundary, 2 * bs).
    assert h.hits["a2"] == 0
    assert [s for r, s, _ in h.resumes if r == "a2"] == [2 * bs]


def test_chunked_prefill_resumes_every_chunk(tmp_path, layout):
    # A per-step budget of one 2048-token chunk: at block 2048 the chunks end on block boundaries and every chunk
    # resumes; at block 4096 every second chunk starts inside a block and recomputes from 0 (its K/V fill would start
    # at the block's first row).
    h = _harness(tmp_path, layout, budget=2048)
    bs = h.block
    a = _prompt(7, 3 * bs + 200)
    h.add("a", a)
    h.run()
    ends = list(range(2048, len(a), 2048))
    assert [s for r, s, _ in h.resumes if r == "a"] == [s for s in ends if s % bs == 0]
    assert [s for r, s, _ in h.recomputes if r == "a"] == [s for s in ends if s % bs]


@pytest.mark.parametrize(
    "L, start, block, plan_start, split",
    [
        (5000, 0, 2048, None, None),  # nothing to resume
        (5000, 4096, 2048, 4096, None),
        (4098, 4096, 2048, None, None),  # 2 tokens after the hit: recompute
        (4096 + 2049, 4096, 64, 4096, 4096 + 2048 - 128),  # a 1-token tail after a full chunk: split 128 earlier
        (4096 + 2049, 4096, 1024, 4096, 4096 + 2048 - 1024),  # ... one block earlier
        (4096 + 2049, 4096, 2048, 4096, None),  # block = chunk: no earlier block boundary, the tail runs eagerly
        (5000, 4000, 64, None, None),  # not 128-aligned
        (5000, 3072, 2048, None, None),  # not on a block boundary
    ],
)
def test_plan_block_resume(L, start, block, plan_start, split):
    plan = _plan(L, start, block)
    if plan_start is None:
        assert plan is None
        return
    s, src, points, _, vllm_start, chunk_pts, cont = plan
    assert (s, vllm_start, src) == (plan_start, start, ("blocks", 0))
    assert points == ([] if split is None else [split]) and not chunk_pts and not cont


def test_plan_block_resume_refuses_snapshot_stores():
    me = SimpleNamespace(_prefix_cache=object(), _chunk_state_cache=None)
    with pytest.raises(RuntimeError):
        Qwen36Model._plan_block_resume(me, 5000, 4096, None)
