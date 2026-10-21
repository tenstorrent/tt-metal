# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Adapter checks of ``tt/generator_vllm.py`` without the vLLM engine: the test drives the plugin's call sequence
(KV spec -> per-layer allocation -> two-phase warmup -> prefill / decode with the reload-contract commands -> split
readback) itself, with a vLLM-style hybrid block manager:

* one block pool shared by vLLM's KV groups (here: the groups vLLM computes for the 48-layer spec), block 0 the null
  block, freed blocks reused last-in-first-out, so a block a sliding-window group frees below its window is
  immediately rewritten by another group or request;
* sliding-window groups free every block below ``processed_tokens - window + 1`` before allocating (vLLM
  ``SlidingWindowManager.remove_skipped_blocks``), so their page tables hold block 0 below each row's window;
* page tables as the plugin sends them: one per layer (``page_tables_per_layer``), padded to max_num_seqs rows.

Checks (the model is ``GEMMA4_VLLM_TEST_LAYERS`` real layers, default ``0,5``: one sliding and one full layer):

* ``test_single_user_matches_generator``: a non-aligned 1000-token prompt (traced prefill bucket 1024) and 300 greedy
  decode steps through the async resident path (stale host tokens / positions, page-table-only refreshes, one-step
  readback lag) equal the standalone generator's tokens bitwise; the decode crosses the sliding window, so freed
  blocks are reused while the window advances;
* ``test_chunked_prefill_matches_generator``: a 9000-token prompt prefilled as two scheduler chunks (0..8192 with
  host logits for the intermediate chunk, 8192..9000 sampled on device) equals the standalone single-call prefill;
* ``test_batch_resident_vs_reload``: four requests with staggered arrivals, a finishing request (layout change with a
  slot remap) and page-table growth: resident async decode equals a host-authoritative control that reloads every
  input on every step;
* ``test_seeded_rows_follow_request``: a seeded request samples the same tokens alone and inside a batch at another
  slot (seeds aligned to the input position on every reset).

  pytest models/autoports/google_gemma_4_12b_it/tests/test_vllm_adapter.py
"""

import copy
import os
from types import SimpleNamespace

import pytest
import torch

MAX_MODEL_LEN = 16384
MAX_SEQS = 32
POOL_BLOCKS = 2000
LAYERS = os.environ.get("GEMMA4_VLLM_TEST_LAYERS", "0,5")
ARCH = "Gemma4UnifiedForConditionalGeneration"
TARGET = "models.autoports.google_gemma_4_12b_it.tt.generator_vllm:Gemma4AutoportForCausalLM"


def vllm_config():
    from vllm.engine.arg_utils import EngineArgs

    os.environ.setdefault("TT_MODEL_CLASS_OVERRIDES", f"{ARCH}={TARGET}")
    os.environ.setdefault("MESH_DEVICE", "P300x2")
    args = EngineArgs(
        model="google/gemma-4-12B-it",
        block_size=64,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_SEQS,
        max_num_batched_tokens=8192,
        enable_chunked_prefill=True,
        additional_config={"tt": {"sample_on_device_mode": "all", "prefill_chunk_alignment": 8192}},
    )
    cfg = args.create_engine_config()
    cfg.cache_config.kv_cache_layout = "LBNHC"
    return cfg


@pytest.fixture(scope="module")
def setup():
    from vllm.model_executor.model_loader.utils import get_model_architecture
    from vllm.v1.core.kv_cache_utils import get_kv_cache_groups

    from models.autoports.google_gemma_4_12b_it.tt.generator import Gemma4Generator
    from models.autoports.google_gemma_4_12b_it.tt.model import Gemma4Model
    from models.autoports.google_gemma_4_12b_it.tt.multichip_decoder import close_qb2_mesh, open_qb2_mesh

    torch.set_num_threads(16)
    os.environ["GEMMA4_12B_VLLM_LAYERS"] = LAYERS
    cfg = vllm_config()
    cls, arch = get_model_architecture(cfg.model_config)
    assert cls.__module__ + ":" + cls.__name__ == TARGET, (arch, cls)
    specs = cls.get_kv_cache_spec(cfg)
    groups = get_kv_cache_groups(cfg, specs)
    group_of = {int(n.split(".")[2]): g for g, grp in enumerate(groups) for n in grp.layer_names}
    windows = [getattr(grp.kv_cache_spec, "sliding_window", None) for grp in groups]
    mesh = open_qb2_mesh(trace_region_size=0)
    adapter = cls.initialize_vllm_model(cfg.model_config.hf_config, mesh, MAX_SEQS, max_seq_len=MAX_MODEL_LEN)
    # the plugin's per-layer specs: (num_blocks, heads per device, block, head size), tensor_idx = position in group
    per_layer = [None] * len(group_of)
    for grp in groups:
        for pos, name in enumerate(grp.layer_names):
            s = grp.kv_cache_spec
            per_layer[int(name.split(".")[2])] = (
                (POOL_BLOCKS, s.num_kv_heads // 4, s.block_size, s.head_size),
                s.dtype,
                pos,
            )
    kv = adapter.allocate_kv_cache_per_layer(per_layer)
    # the standalone reference on the same mesh: every persistent tensor of both generators exists before the first
    # trace capture (a tensor allocated after a capture can overlap that trace's intermediates), compile both, then
    # capture both
    ref_model = Gemma4Model(
        mesh, layer_indices=[int(v) for v in LAYERS.split(",")], max_context=MAX_MODEL_LEN, num_blocks=600
    )
    ref = Gemma4Generator(ref_model, prepare=False)
    adapter.warmup_model_prefill(kv_cache=kv, enable_trace=False, can_sample_on_device=True)
    adapter.warmup_model_decode(kv_cache=kv, enable_trace=False, max_batch_size=MAX_SEQS, num_blocks=256)
    ref.prepare(capture=False)
    adapter.warmup_model_prefill(kv_cache=kv, enable_trace=True, can_sample_on_device=True)
    adapter.warmup_model_decode(kv_cache=kv, enable_trace=True, max_batch_size=MAX_SEQS, num_blocks=256)
    ref.prepare()
    yield SimpleNamespace(adapter=adapter, kv=kv, group_of=group_of, windows=windows, ref=ref, num_layers=len(group_of))
    ref.teardown()
    adapter.release_persistent_capture()
    close_qb2_mesh(mesh)


class BlockManager:
    """vLLM-style hybrid block bookkeeping: one pool, one block list per (request, group), LIFO reuse."""

    def __init__(self, num_blocks, windows, block=64):
        self.free = list(range(1, num_blocks))[::-1]
        self.windows = windows
        self.block = block
        self.reqs = {}

    def add(self, rid):
        self.reqs[rid] = [[] for _ in self.windows]

    def allocate(self, rid, processed, total):
        """Free the window-skipped blocks for ``processed`` computed tokens, then cover positions ``[0, total)``."""
        for g, w in enumerate(self.windows):
            blocks = self.reqs[rid][g]
            if w:
                skip = max(0, processed - w + 1) // self.block
                for i in range(min(skip, len(blocks))):
                    if blocks[i]:
                        self.free.append(blocks[i])
                        blocks[i] = 0
            while len(blocks) * self.block < total:
                blocks.append(self.free.pop())

    def release(self, rid):
        for blocks in self.reqs.pop(rid):
            self.free += [b for b in blocks if b]

    def tables(self, rids, num_layers, group_of, width=MAX_MODEL_LEN // 64):
        per_group = []
        for g in range(len(self.windows)):
            t = torch.zeros(MAX_SEQS, width, dtype=torch.int32)
            for r, rid in enumerate(rids):
                blocks = self.reqs[rid][g]
                t[r, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
            per_group.append(t)
        # the plugin pads each layer's table separately: distinct tensors per layer
        return per_group[0], [per_group[group_of[i]].clone() for i in range(num_layers)]


def params(rows, temperature=0.0, top_k=1, top_p=1.0, seeds=None):
    return SimpleNamespace(
        temperature=[temperature] * rows,
        top_k=[top_k] * rows,
        top_p=[top_p] * rows,
        seed=list(seeds) if seeds is not None else [None] * rows,
    )


class Driver:
    """The plugin's runner, reduced to the call sequence the adapter sees (front-packed rows, DP=1)."""

    def __init__(self, s, resident=True, sync=False):
        self.s = s
        self.a = s.adapter
        self.bm = BlockManager(POOL_BLOCKS, s.windows)
        self.resident = resident  # False: host-authoritative control (reload_inputs on every step)
        self.sync = sync  # synchronous scheduling: finalize without read_decode_output, no overlap
        self.reqs = {}  # rid -> dict(tokens, computed, out, params)
        self.order = []  # front-packed rows: a request's slot is its index
        self.layout = None
        self.pending = None
        self.prev_tables = None

    def prefill(self, rid, prompt, chunks=None, sampling=None):
        self.drain()
        self.reqs[rid] = dict(tokens=list(prompt), computed=0, out=[], sampling=sampling or params(1))
        self.bm.add(rid)
        self.order.append(rid)
        slot = self.order.index(rid)
        chunks = chunks or [len(prompt)]
        r = self.reqs[rid]
        for end in chunks:
            start = r["computed"]
            self.bm.allocate(rid, start, end)
            pt, per_layer = self.bm.tables([rid], self.s.num_layers, self.s.group_of)
            tokens = torch.tensor([prompt[:end]], dtype=torch.int64)
            final = end == len(prompt)
            sp = r["sampling"] if final else None  # intermediate chunks: host sampling (logits, unused)
            out = self.a.prefill_forward(
                tokens,
                [start],
                [end],
                pt,
                kv_cache=self.s.kv,
                enable_trace=True,
                empty_slots=[slot],
                page_tables_per_layer=per_layer,
                sampling_params=copy.deepcopy(sp) if sp is not None else None,
            )
            r["computed"] = end
            if final:
                tok = int(out.reshape(-1)[0]) if sp is not None else int(out[0, -1].argmax())
                r["tokens"].append(tok)
                r["out"].append(tok)
            else:
                assert out.shape == (1, 1, self.a.model.vocab_size)
        self.layout = None  # prefill -> decode is a transition

    def drain(self):
        if self.pending is not None:
            self._finish(self.pending)
            self.pending = None

    def _finish(self, pending):
        handle, rids = pending
        toks = self.a.process_decode_output_host(handle, is_tokens=True).reshape(-1)
        for i, rid in enumerate(rids):
            r = self.reqs[rid]
            r["tokens"].append(int(toks[i]))
            r["out"].append(int(toks[i]))

    def finish(self, rid):
        self.drain()
        self.bm.release(rid)
        self.order.remove(rid)
        return self.reqs.pop(rid)["out"]

    def decode(self, rids, steps):
        a = self.a
        for _ in range(steps):
            assert list(rids) == self.order
            layout = tuple(rids)
            transition = self.layout != layout
            if transition:
                self.drain()  # the plugin drains pending work before a reload
            n = len(rids)
            bucket = next(b for b in a.tt_supported_decode_batch_sizes if b >= n)
            # vLLM schedules with optimistic counts: the placeholder of the in-flight step counts as computed
            positions = []
            for rid in rids:
                r = self.reqs[rid]
                pos = len(r["tokens"]) - 1 + (1 if self.pending is not None else 0)
                processed = len(r["tokens"]) - 1
                self.bm.allocate(rid, processed, pos + 1)
                positions.append(pos)
            pt, per_layer = self.bm.tables(rids, self.s.num_layers, self.s.group_of)
            stacked = torch.stack(per_layer)
            changed = self.prev_tables is None or not torch.equal(self.prev_tables, stacked)
            reload = transition or not self.resident
            if reload:
                self.drain()  # a host-authoritative reload first applies the in-flight step's token
            tokens = torch.zeros(bucket, 1, dtype=torch.int64)
            start = torch.full((bucket,), -1, dtype=torch.int64)
            for i, rid in enumerate(rids):
                r = self.reqs[rid]
                if reload:
                    tokens[i, 0] = r["tokens"][-1]
                    start[i] = len(r["tokens"]) - 1
                else:  # stale: the plugin's host state trails the device by the in-flight step
                    tokens[i, 0] = 7
                    start[i] = 0
            sp = [self.reqs[rid]["sampling"] for rid in rids]
            dev = SimpleNamespace(
                temperature=sum((p.temperature for p in sp), []) + [0.0] * (bucket - n),
                top_k=sum((p.top_k for p in sp), []) + [1] * (bucket - n),
                top_p=sum((p.top_p for p in sp), []) + [1.0] * (bucket - n),
                seed=sum((p.seed for p in sp), []) + [None] * (bucket - n),
            )
            handle = a.decode_forward(
                tokens,
                start,
                pt[:bucket],
                kv_cache=self.s.kv,
                enable_trace=True,
                read_from_device=False,
                page_tables_per_layer=per_layer,
                slot_remap=list(range(bucket)) if transition else None,
                sampling_params=dev,
                reload_inputs=reload,
                reload_page_table=(not reload) and changed,
                reload_sampling_params=reload,
                reset_sampling_state=reload,
            )
            if not self.sync:
                handle, events = a.read_decode_output(handle, async_read=True)
            self.layout = layout
            self.prev_tables = stacked
            if self.pending is not None:
                self._finish(self.pending)
            self.pending = (handle, list(rids))
            if self.sync:
                self.drain()
        self.drain()


def tokens_of(s, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(10, 200000, (n,), generator=g).tolist()


def test_single_user_matches_generator(setup):
    s = setup
    prompt = tokens_of(s, 1000)
    want = s.ref.generate(prompt, 301, stop_on_eos=False)
    d = Driver(s)
    d.prefill("a", prompt)
    d.decode(["a"], 300)
    got = d.finish("a")
    assert got == want, f"first mismatch at {next(i for i, (x, y) in enumerate(zip(got, want)) if x != y)}"
    c = s.adapter.generator.counters
    print("counters", c)
    assert c["page_table_resident_steps"] > 200 and c["eager_decode_steps"] == 0


def test_chunked_prefill_matches_generator(setup):
    s = setup
    prompt = tokens_of(s, 9000, seed=1)
    want = s.ref.generate(prompt, 41, stop_on_eos=False)
    d = Driver(s)
    d.prefill("b", prompt, chunks=[8192, 9000])
    d.decode(["b"], 40)
    assert d.finish("b") == want


def run_batch(s, resident):
    d = Driver(s, resident=resident)
    prompts = {
        rid: tokens_of(s, n, seed=10 + i) for i, (rid, n) in enumerate([("p", 130), ("q", 1100), ("r", 64), ("t", 7)])
    }
    outs = {}
    d.prefill("p", prompts["p"])
    d.decode(["p"], 20)
    d.prefill("q", prompts["q"])
    d.decode(["p", "q"], 70)  # crosses block boundaries of both rows
    d.prefill("r", prompts["r"])
    d.prefill("t", prompts["t"])
    d.decode(["p", "q", "r", "t"], 30)
    outs["p"] = d.finish("p")
    d.decode(["q", "r", "t"], 40)  # layout change: rows move up one slot
    for rid in ("q", "r", "t"):
        outs[rid] = d.finish(rid)
    return outs


def test_sync_scheduling_finalize(setup):
    """Synchronous scheduling (--no-async-scheduling): decode_forward(read_from_device=False) is finalized by
    process_decode_output_host without read_decode_output; the tokens equal the overlapped path's."""
    s = setup
    prompt = tokens_of(s, 300, seed=3)
    outs = []
    for sync in (False, True):
        d = Driver(s, resident=not sync, sync=sync)
        d.prefill("s", prompt)
        d.decode(["s"], 80)
        outs.append(d.finish("s"))
    assert outs[0] == outs[1]


def test_batch_resident_vs_reload(setup):
    s = setup
    control = run_batch(s, resident=False)
    resident = run_batch(s, resident=True)
    assert resident == control


def test_seeded_rows_follow_request(setup):
    s = setup
    sp = params(1, temperature=1.0, top_k=20, top_p=0.9, seeds=[1234])
    prompt = tokens_of(s, 40, seed=5)
    d = Driver(s)
    d.prefill("x", prompt, sampling=copy.deepcopy(sp))
    d.decode(["x"], 20)
    alone = d.finish("x")
    d = Driver(s)
    d.prefill("y", tokens_of(s, 70, seed=6), sampling=params(1, temperature=1.0, top_k=20, top_p=0.9))
    d.decode(["y"], 3)
    d.prefill("x", prompt, sampling=copy.deepcopy(sp))
    d.decode(["y", "x"], 20)  # x now sits in slot 1 next to an unseeded request
    batched = d.finish("x")
    d.finish("y")
    assert batched == alone
    assert len(set(alone)) > 3


def test_chunked_prefill_logits(setup):
    """Host-sampling logits of the 9000-token prompt: one call (two chunks inside the generator) and two scheduler
    chunks (8192 + 808, the continuation reading hybrid page tables with freed sliding blocks) equal the standalone
    generator's single call bitwise."""
    s = setup
    a = s.adapter
    prompt = tokens_of(s, 9000, seed=1)
    row = s.ref._alloc_row(9000)
    want = s.ref.prefill_forward(torch.tensor([prompt]), page_table=row[None], prompt_lens=[9000])[0, 0]
    s.ref._release(row)
    for chunks in ([9000], [8192, 9000]):
        bm = BlockManager(POOL_BLOCKS, s.windows)
        bm.add("z")
        start = 0
        for end in chunks:
            bm.allocate("z", start, end)
            pt, per_layer = bm.tables(["z"], s.num_layers, s.group_of)
            out = a.prefill_forward(
                torch.tensor([prompt[:end]]),
                [start],
                [end],
                pt,
                kv_cache=s.kv,
                empty_slots=[0],
                page_tables_per_layer=per_layer,
            )
            start = end
        assert out.shape == (1, 1, a.model.vocab_size)
        assert torch.equal(out[0, 0], want), (chunks, float((out[0, 0] - want).abs().max()))
