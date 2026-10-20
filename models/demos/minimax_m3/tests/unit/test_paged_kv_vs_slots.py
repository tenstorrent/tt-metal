# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Paged KV cache through the MSA attention layer, SP=8 x TP=4. Each case first fills the caches with single-chunk
forwards (not compared), then runs one compared forward on both: its chunks' real output rows and the K / V / index_k
the slots hold afterwards (read back through the page table) must match the slot cache bit for bit, the compared
paged forward must compile no program, and every page of the pool no written row maps, and every cache layer but the
one written, must stay zero.

Cases: an ordered table; a shuffled one ending on a partial chunk; two requests sharing whole prefix pages (the second
writes only its suffix); a packed forward of 4 chunks; a resume that starts mid-page; a row mapping only the request's
own pages (the pad rows past its last page fall on unmapped pages). The layer writes cache layer 1 of 3, so the
pool's page x layer entries are exercised. Random weights, the runtime's whole-cache indexed RoPE.

test_prefill_runtime_paged runs a packed schedule through TtPrefillRuntime with real weights (two MSA + MoE layers):
compile() on the all-unmapped table must leave the pool zero, the slots' K / V / index_k must then match the slot
cache's bit for bit, and the schedule must compile no program after compile().
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.minimax_m3.tt.attention import allocate_kv_caches
from models.demos.minimax_m3.tt.attention.kv_cache import (
    PAGE_SIZE,
    PAGE_UNMAPPED,
    fixed_page_table,
    page_owner,
    read_paged_slot_block,
    set_page_table,
)
from models.demos.minimax_m3.tt.prefill_chunk import PrefillChunk, pack_forward_tokens, validate_chunks
from models.demos.minimax_m3.tt.runners.prefill_kv_validation import naturalize_kv_block

from ..test_factory import M3_HIDDEN, packed_attention_forward, parametrize_mesh_with_fabric, random_sp_attention

SP, C, USERS = 8, 1024, 4
CACHE_LAYERS, WRITTEN_LAYER = 3, 1  # the attention layer writes cache layer 1 of 3
CAP = 12 * C
SPARE_PAGES = 8  # pages no table maps, so every case has pool pages that must stay zero
NUM_PAGES = USERS * CAP // SP // PAGE_SIZE + SPARE_PAGES
SHARED_PREFIX = 2 * C  # shared_prefix: row 1 maps row 0's pages under these positions, a whole number of pages


def fills(slot, start, end):
    """Single-chunk forwards covering [start, end) of ``slot``."""
    return [PrefillChunk(slot, a, min(a + C, end)) for a in range(start, end, C)]


# name -> (table, fills on the slot cache only, fills on both caches, the compared forward)
CASES = {
    "ordered": ("ordered", [], fills(0, 0, 2 * C), [PrefillChunk(0, 2 * C, 3 * C)]),
    "shuffled_partial": ("shuffled", [], fills(1, 0, 3 * C), [PrefillChunk(1, 3 * C, 3 * C + 700)]),
    # The slot cache fills slot 1's prefix itself; the paged row 1 maps slot 0's pages there.
    "shared_prefix": (
        "shared",
        fills(1, 0, SHARED_PREFIX),
        fills(0, 0, SHARED_PREFIX),
        [PrefillChunk(1, SHARED_PREFIX, SHARED_PREFIX + C)],
    ),
    "packed": (
        "shuffled",
        [],
        fills(1, 0, 2 * C) + fills(2, 0, 8 * C),
        [PrefillChunk(0, 0, C), PrefillChunk(1, 2 * C, 3 * C), PrefillChunk(2, 8 * C, 9 * C), PrefillChunk(3, 0, 600)],
    ),
    "mid_page_resume": ("shuffled", [], fills(0, 0, 2 * C + 96), [PrefillChunk(0, 2 * C + 96, 3 * C + 96)]),
    "request_pages_only": ("request", [], fills(2, 0, C), [PrefillChunk(2, C, C + 300)]),
}


def build_table(kind, *, width, num_pages, seed):
    """Host page table. ``ordered``: the runner's fixed page table, row r taking physical pages [r*n, (r+1)*n) of every
    rank in position order. ``shuffled``: every row's pages drawn at random from each rank's pool. ``shared``: shuffled, and
    row 1 maps row 0's pages under SHARED_PREFIX. ``request``: shuffled, and row 2 maps only the pages under its
    C + 300 tokens."""
    owner = [page_owner(q, chunk_local=C // SP, sp=SP) for q in range(width)]
    if kind == "ordered":
        table = fixed_page_table(num_users=USERS, max_seq_len=width * PAGE_SIZE, sp=SP, chunk_size=C)
        assert int(table.max()) < num_pages
        return table
    gen = torch.Generator().manual_seed(seed)
    free = [torch.randperm(num_pages, generator=gen).tolist() for _ in range(SP)]
    table = torch.tensor([[free[owner[q]].pop() for q in range(width)] for _ in range(USERS)])
    if kind == "shared":
        table[1, : SHARED_PREFIX // PAGE_SIZE] = table[0, : SHARED_PREFIX // PAGE_SIZE]
    elif kind == "request":
        table[2, -(-(C + 300) // PAGE_SIZE) :] = PAGE_UNMAPPED
    return table


@pytest.mark.timeout(600)
@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
def test_paged_kv_vs_slots(mesh_device, device_params, reset_seeds):
    rows, cols = tuple(mesh_device.shape)
    assert (rows, cols) == (SP, 4)
    layer = random_sp_attention(
        mesh_device, chunk_size=C, max_seq_len=CAP, is_sparse=True, global_layer_idx=3, local_layer_idx=WRITTEN_LAYER
    )
    donors = {}  # slot -> the slot whose inputs it repeats under SHARED_PREFIX (shared_prefix)

    def rows_at(slot, start):
        """The input rows of ``slot``'s chunk starting at ``start``, distinct per slot, so a read of the wrong row
        shows. A slot sharing a donor's prefix pages takes the donor's rows there, so its slot cache holds the
        same KV."""
        if start < SHARED_PREFIX:
            slot = donors.get(slot, slot)
        seed = slot * CAP + start + 1
        return torch.randn(C, M3_HIDDEN, generator=torch.Generator().manual_seed(seed)) * 0.1

    def poison_gather_buffers():
        """Fill the persistent gather outputs with a sentinel, so a gather that leaves rows unwritten cannot pass on
        what the forward before it left there. Finite: the consumers read rows past the prefix as masked future
        positions (CCLManager.get_high_bw_gather_buffer)."""
        for buf in layer.ccl._high_bw_gather_buffers.values():
            sentinel = ttnn.from_torch(
                torch.full(list(buf.shape), 64.0),
                dtype=buf.dtype,
                layout=buf.layout,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            ttnn.copy_host_to_device_tensor(sentinel, buf)

    def run(chunks, kvc):
        """One attention forward over ``chunks``; returns each chunk's output rows in natural order."""
        poison_gather_buffers()
        n, s = len(chunks), C // rows
        out = packed_attention_forward(layer, [rows_at(c.slot_id, c.actual_start) for c in chunks], chunks, kvc)
        order = pack_forward_tokens(
            [[k * C + i for i in range(C)] for k in range(n)], [c.actual_start for c in chunks], rows
        )
        natural = []
        for k in range(n):
            # pack_forward_tokens is rank-major, then chunk, then row: rank r's s rows of chunk k start at r*n*s + k*s.
            positions = torch.tensor([order[r * n * s + k * s + j] - k * C for r in range(rows) for j in range(s)])
            rows_k = torch.empty(C, M3_HIDDEN)
            rows_k[positions] = out[k]
            natural.append(rows_k)
        return natural

    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape)

    def natural(block):
        return naturalize_kv_block(block, n_tokens=CAP, sp_factor=rows, chunk_size=C, max_seq_len=CAP)

    def slot_cache_kv(kvc, slot):
        """[k, v, index_k] of ``slot``'s cache layer WRITTEN_LAYER, natural order, from a slot cache. index_k is
        replicated across the TP cols, so one col is kept."""
        row = slot * CACHE_LAYERS + WRITTEN_LAYER
        out = []
        for t in (kvc.k, kvc.v, kvc.index_k):
            shape = list(t.shape)
            sl = ttnn.slice(t, [row, 0, 0, 0], [row + 1, *shape[1:]], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            out.append(natural(ttnn.to_torch(sl, mesh_composer=composer).float()[0]))
            ttnn.deallocate(sl)
        out[2] = out[2][:1]
        return out

    def paged_cache_kv(kvc, slot):
        """slot_cache_kv of a paged cache, read through its page table."""
        out = [
            natural(read_paged_slot_block(kvc, t, slot, rows=CAP // rows, chunk_local=C // rows)[WRITTEN_LAYER])
            for t in (kvc.k, kvc.v, kvc.index_k)
        ]
        out[2] = out[2][:1]
        return out

    def written_outside_rows(kvc, table, written_slots):
        """Names of the pool tensors holding a nonzero value in a page no written row maps, or in a cache layer the
        attention layer does not write."""
        other_layers = [l for l in range(CACHE_LAYERS) if l != WRITTEN_LAYER]
        dirty = []
        for name, t in (("k", kvc.k), ("v", kvc.v), ("index_k", kvc.index_k)):
            pool = ttnn.to_torch(t, mesh_composer=composer)  # [pages * layers, tp, sp * page, hd]
            pool = pool.reshape(NUM_PAGES, CACHE_LAYERS, cols, rows, PAGE_SIZE, -1)
            for rank in range(rows):
                owned = [q for q in range(table.shape[1]) if page_owner(q, chunk_local=C // rows, sp=rows) == rank]
                used = set(table[sorted(written_slots)][:, owned].flatten().tolist())
                unused = [p for p in range(NUM_PAGES) if p not in used]
                if pool[unused, :, :, rank].any() or pool[:, other_layers, :, rank].any():
                    dirty.append(f"{name} rank {rank}")
        return dirty

    failures = []
    for seed, (name, (table_kind, slot_only, both, forward)) in enumerate(CASES.items()):
        validate_chunks(forward, chunk_size=C, forward_chunks=len(forward), num_users=USERS, max_seq_len=CAP)
        donors.clear()
        if table_kind == "shared":
            donors[1] = 0
        slots_kv = allocate_kv_caches(mesh_device, num_layers=CACHE_LAYERS, max_seq_len=CAP, num_users=USERS)
        paged_kv = allocate_kv_caches(
            mesh_device, num_layers=CACHE_LAYERS, max_seq_len=CAP, num_users=USERS, num_pages=NUM_PAGES
        )
        table = build_table(table_kind, width=paged_kv.page_table.shape[-1], num_pages=NUM_PAGES, seed=seed)
        set_page_table(paged_kv, table)
        for chunk in slot_only:
            run([chunk], slots_kv)
        for chunk in both:
            for kvc in (slots_kv, paged_kv):
                run([chunk], kvc)
        slot_out = run(forward, slots_kv)
        programs = mesh_device.num_program_cache_entries()
        paged_out = run(forward, paged_kv)
        ttnn.synchronize_device(mesh_device)
        # Every per-chunk offset, row, layer and table address is a runtime arg of the paged ops: the forward reuses
        # the programs the fills and earlier cases compiled.
        new_programs = mesh_device.num_program_cache_entries() - programs
        if new_programs:
            failures.append(f"{name}: the paged forward compiled {new_programs} programs")

        for k, chunk in enumerate(forward):
            real = chunk.actual_end - chunk.actual_start
            same = torch.equal(slot_out[k][:real], paged_out[k][:real])
            logger.info(f"[{name}] chunk {k} {chunk}: real rows bit-exact {same}")
            if not same:
                diff = (slot_out[k][:real] - paged_out[k][:real]).abs().max().item()
                failures.append(f"{name} chunk {k}: out differs (max |diff| {diff:.4g})")
        for slot, end in {c.slot_id: c.actual_end for c in forward if not c.is_empty}.items():
            for cache, want, got in zip(
                ("k", "v", "index_k"), slot_cache_kv(slots_kv, slot), paged_cache_kv(paged_kv, slot)
            ):
                same = torch.equal(want[:, :end], got[:, :end])
                logger.info(f"[{name}] slot {slot} {cache} [0, {end}): bit-exact {same}")
                if not same:
                    failures.append(f"{name} slot {slot} {cache}: cache differs")
        written_slots = {c.slot_id for c in both + forward if not c.is_empty}
        failures += [
            f"{name}: {where} written outside the rows' pages"
            for where in written_outside_rows(paged_kv, table, written_slots)
        ]
        slots_kv.deallocate()
        paged_kv.deallocate()

    assert not failures, f"paged vs slots: {failures}"


RUNTIME_LAYERS = [3, 4]  # two MSA-attention + MoE layers: a paged cache serves no dense ring_joint layer
RUNTIME_CAP = 8 * C
EMPTY = PrefillChunk(0, 0, 0)  # pads a forward to its fixed chunk count
# Slots 1 and 3 end their second forward mid-tile (C + 700, 300) and resume at the 32-aligned start below it
# (C + 672, 288), rewriting the partial tile, as a multi-turn resume does.
RUNTIME_SCHEDULE = [
    [PrefillChunk(0, 0, C), PrefillChunk(0, C, 2 * C), PrefillChunk(1, 0, C), PrefillChunk(2, 0, C)],
    [PrefillChunk(0, 2 * C, 3 * C), PrefillChunk(1, C, C + 700), PrefillChunk(2, C, 2 * C), PrefillChunk(3, 0, 300)],
    [PrefillChunk(1, C + 672, 2 * C + 672), PrefillChunk(3, 288, C + 288), EMPTY, EMPTY],
]


@pytest.mark.timeout(600)
@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
def test_prefill_runtime_paged(mesh_device, device_params, reset_seeds, monkeypatch):
    if not os.getenv("HF_MODEL"):
        pytest.skip("needs real MiniMax-M3 weights (HF_MODEL + a warm TT_CACHE_PATH tilized cache)")
    from models.demos.minimax_m3.tt.model_config import ModelArgs
    from models.demos.minimax_m3.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig
    from models.demos.minimax_m3.tt.weight_cache import weight_cache_is_complete

    built = max(RUNTIME_LAYERS) + 1
    model_args = ModelArgs(mesh_device=mesh_device)
    hf_config = model_args.hf_config
    hf_config.num_hidden_layers = built
    monkeypatch.setenv("M3_LOAD_NLAYERS", str(built))
    cache_path = model_args.weight_cache_path(ttnn.bfloat8_b)
    if not weight_cache_is_complete(cache_path, hf_config, built, ttnn.bfloat4_b):
        pytest.skip(f"tilized weight cache for the first {built} layers is not complete at {cache_path}")
    sp = mesh_device.shape[0]
    assert sp == SP
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape)
    runtime = TtPrefillRuntime(
        mesh_device,
        hf_config,
        {},
        TtPrefillRuntimeConfig(
            num_layers=len(RUNTIME_LAYERS),
            layer_indices=RUNTIME_LAYERS,
            max_seq_len=RUNTIME_CAP,
            mesh_shape=tuple(mesh_device.shape),
            chunk_size=C,
            forward_tokens_budget=4 * C,
            num_users=USERS,
            expert_weight_dtype=ttnn.bfloat4_b,
            weight_cache_path=cache_path,
            topology=ttnn.Topology.Linear,
        ),
    )
    gen = torch.Generator().manual_seed(0)
    tokens = [torch.randint(0, hf_config.vocab_size, (RUNTIME_CAP,), generator=gen).tolist() for _ in range(USERS)]

    def run(kv_cache):
        for forward in RUNTIME_SCHEDULE:
            chunk_tokens = [
                tokens[c.slot_id][c.actual_start : c.actual_end] + [0] * (C - (c.actual_end - c.actual_start))
                for c in forward
            ]
            runtime.prefill_forward(
                runtime.make_forward_input(chunk_tokens, [c.actual_start for c in forward]), kv_cache, forward
            )
        ttnn.synchronize_device(mesh_device)

    def allocate(**paging):
        return allocate_kv_caches(
            mesh_device,
            num_layers=len(RUNTIME_LAYERS),
            max_seq_len=RUNTIME_CAP,
            num_users=USERS,
            head_dim=hf_config.head_dim,
            **paging,
        )

    slots_kv = allocate()
    runtime.compile(slots_kv)
    run(slots_kv)

    num_pages = USERS * RUNTIME_CAP // sp // PAGE_SIZE + SPARE_PAGES
    paged_kv = allocate(num_pages=num_pages)
    # compile() on the all-unmapped table allocate_kv_caches installs drops every warm-up write.
    runtime.compile(paged_kv)
    for name in ("k", "v", "index_k"):
        assert not ttnn.to_torch(
            getattr(paged_kv, name), mesh_composer=composer
        ).any(), f"compile() wrote the {name} pool"
    width = paged_kv.page_table.shape[-1]
    set_page_table(paged_kv, build_table("shuffled", width=width, num_pages=num_pages, seed=7))
    programs = mesh_device.num_program_cache_entries()
    run(paged_kv)
    new_programs = mesh_device.num_program_cache_entries() - programs

    ends = {}
    for forward in RUNTIME_SCHEDULE:
        for c in forward:
            if not c.is_empty:
                ends[c.slot_id] = c.actual_end
    failures = []
    for slot, end in sorted(ends.items()):
        for name, want, got in zip(
            ("k", "v", "index_k"), runtime.read_slot_kv(slots_kv, slot), runtime.read_slot_kv(paged_kv, slot)
        ):
            for layer in range(len(RUNTIME_LAYERS)):
                same = torch.equal(
                    naturalize_kv_block(want[layer], n_tokens=end, sp_factor=sp, chunk_size=C, max_seq_len=RUNTIME_CAP),
                    naturalize_kv_block(got[layer], n_tokens=end, sp_factor=sp, chunk_size=C, max_seq_len=RUNTIME_CAP),
                )
                logger.info(f"[runtime] slot {slot} [0, {end}) layer {RUNTIME_LAYERS[layer]} {name}: bit-exact {same}")
                if not same:
                    failures.append(f"slot {slot} layer {RUNTIME_LAYERS[layer]} {name}")
    runtime.release_sub_device_managers()
    assert not failures, f"paged runtime KV differs from slots: {failures}"
    assert new_programs == 0, f"the paged schedule compiled {new_programs} programs after compile()"
