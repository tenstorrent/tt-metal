# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A host-only stand-in for the device under the V4.1 prefill runtime (prefix reuse H1 tests; no chips, no real ttnn).

Importing this module installs, BEFORE anything imports them:
  * ``ttnn``: a torch-backed fake with the ops ``kv_export.py`` / ``tt/v41/prefix_reuse.py`` / ``runtime.py`` use, every call
    recorded in ``SYS.trace`` (tensors named by creation order) -- two runs that issue the same device work give equal traces;
  * ``models.demos.deepseek_v3_d_p.utils.kv_cache_utils`` (``DRAMZeroFill``) and ``models.demos.deepseek_v3_d_p.tt.v41.prefill``
    (a placeholder ``V41Prefill``: ``FakePrefill`` below is passed in instead).

``FakePrefill`` models what matters for a resume, with the real V4.1 layer roles (``V41Config``, the checkpoint's config.json):
every attention state's counters, window carry, the KV sources' entries / index keys (written over the chunk's full width, as the
compressor does), the A3 islands' SHARED score mask opened by a glue that copies ``TtCSA.glue_chunk`` (one
``_zeros_block_tile(S_l, entry_count - mask_zeroed_upto)`` -- a new key after the capture is recorded in ``SYS.late_allocs``, the
DS4F-0262 hazard of design risk R2), and the Engram n-gram history. A chunk's outputs hash its ids (causally), its start, the
carry it starts from, its counters, the source entries it may see (never the stale rows past them), the mask and the Engram
history: resuming from anything but a cold prefill's state changes every later row.
"""

from __future__ import annotations

import sys
import types

import torch

P = (1 << 31) - 1
NEG = float("-inf")


# ---------------------------------------------------------------------------------------------------------------- hashing
def H(*xs):
    """Fold ints / int64 tensors (broadcast) into values < P."""
    acc = 1469598103
    for x in xs:
        x = torch.remainder(x.to(torch.int64), P) if isinstance(x, torch.Tensor) else int(x) % P
        if isinstance(acc, torch.Tensor) or isinstance(x, torch.Tensor):
            acc = torch.remainder(acc * 1000003 + x, P)
        else:
            acc = (acc * 1000003 + x) % P
    return acc


def digest_rows(data: torch.Tensor) -> int:
    """Order-sensitive digest of a [..., rows, width] tensor's rows (all of every row: int64 or float with +-inf)."""
    x = data.reshape(-1, data.shape[-1])
    if x.is_floating_point():
        x = torch.where(torch.isinf(x), torch.where(x > 0, 7.0, 3.0), x * 16 + 1).to(torch.int64)
    rowv = torch.remainder((torch.remainder(x, P) * (torch.arange(x.shape[1]) + 1)).sum(dim=1), P)
    return int(torch.remainder((rowv * (torch.arange(rowv.shape[0]) + 1)).sum(), P))


# ---------------------------------------------------------------------------------------------------------------- fake ttnn
class _Sys:
    def __init__(self):
        self.reset()

    def reset(self):
        self.trace = []
        self.next_id = 0
        self.captured = False
        self.late_allocs = []
        self.alive = 0


SYS = _Sys()


class FakeT:
    def __init__(self, data: torch.Tensor, dtype="bf16", layout="TILE", device=None):
        self.data = data
        self.dtype = dtype
        self.layout = layout
        self._device = device
        self.sid = SYS.next_id
        SYS.next_id += 1
        self.dead = False

    @property
    def shape(self):
        return tuple(self.data.shape)

    def device(self):
        return self._device

    def __repr__(self):
        return f"T{self.sid}{list(self.shape)}{self.dtype}"


class FakeMesh:
    shape = (8, 4)

    def num_program_cache_entries(self):
        return 0

    def arch(self):
        return "bh"


MESH = FakeMesh()


def _norm(v):
    if isinstance(v, FakeT):
        return f"T{v.sid}{list(v.shape)}{v.dtype}"
    if isinstance(v, (list, tuple)):
        return type(v)(_norm(x) for x in v)
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return type(v).__name__


def rec(name, *args, **kw):
    SYS.trace.append((name, _norm(args), _norm(kw)))


def _alive(t: FakeT):
    assert not t.dead, f"use of a deallocated tensor {t}"
    return t


def _mk_ttnn():
    m = types.ModuleType("ttnn")
    m.__fake__ = True
    m.bfloat16, m.bfloat8_b, m.float32, m.uint32 = "bf16", "bfp8", "f32", "u32"
    m.TILE_LAYOUT, m.ROW_MAJOR_LAYOUT = "TILE", "RM"
    m.DRAM_MEMORY_CONFIG = "DRAM"
    m.TILE_SIZE = 32

    class ReplicateTensorToMesh:
        def __init__(self, dev):
            self.dev = dev

    m.ReplicateTensorToMesh = ReplicateTensorToMesh

    class BufferType:
        DRAM = "DRAM"

    m.BufferType = BufferType

    def get_memory_view(mesh, kind):
        raise RuntimeError("no memory view on the fake device")

    def from_torch(x, dtype=None, layout=None, device=None, memory_config=None, mesh_mapper=None, **_):
        t = FakeT(
            x.clone().to(torch.int64) if not x.is_floating_point() or not torch.isinf(x).any() else x.clone(),
            dtype or "bf16",
            layout or "RM",
            device,
        )
        rec("from_torch", t, dtype=dtype, layout=layout)
        return t

    def slice(t, start, end, *a, **k):
        _alive(t)
        idx = tuple(builtins_slice(s, e) for s, e in zip(start, end))
        out = FakeT(t.data[idx].clone(), t.dtype, t.layout, t._device)
        rec("slice", t, list(start), list(end), out)
        return out

    def typecast(t, dtype, **_):
        _alive(t)
        out = FakeT(t.data.clone(), dtype, t.layout, t._device)
        rec("typecast", t, dtype, out)
        return out

    def deallocate(t, *a, **k):
        if isinstance(t, FakeT):
            rec("deallocate", t)
            t.dead = True

    def reshape(t, shape, **_):
        _alive(t)
        out = FakeT(t.data.reshape(tuple(shape)).clone(), t.dtype, t.layout, t._device)
        rec("reshape", t, list(shape), out)
        return out

    def permute(t, dims, **_):
        _alive(t)
        out = FakeT(t.data.permute(*dims).contiguous(), t.dtype, t.layout, t._device)
        rec("permute", t, list(dims), out)
        return out

    def clone(t, **_):
        _alive(t)
        out = FakeT(t.data.clone(), t.dtype, t.layout, t._device)
        rec("clone", t, out)
        return out

    def copy(src, dst, **_):
        _alive(src), _alive(dst)
        assert src.data.shape == dst.data.shape, (src, dst)
        dst.data.copy_(src.data)
        rec("copy", src, dst)

    def synchronize_device(mesh, *a, **k):
        rec("synchronize_device")

    m.get_memory_view = get_memory_view
    m.from_torch, m.slice, m.typecast, m.deallocate = from_torch, slice, typecast, deallocate
    m.reshape, m.permute, m.clone, m.copy, m.synchronize_device = reshape, permute, clone, copy, synchronize_device

    kv = types.SimpleNamespace()

    def fill_cache_for_user_(cache, block, b, update_idx=0):
        _alive(cache), _alive(block)
        n = block.data.shape[2]
        assert block.data.shape[3] == cache.data.shape[3], (block, cache)
        cache.data[b, 0, update_idx : update_idx + n] = block.data[0, 0]
        rec("fill_cache_for_user_", cache, block, b, update_idx=update_idx)

    kv.fill_cache_for_user_ = fill_cache_for_user_
    m.kv_cache = kv

    exp = types.SimpleNamespace()

    def slice_write(block, out, start, end, step):
        _alive(block), _alive(out)
        c0, c1 = start[3], end[3]
        assert block.data.shape[3] == c1 - c0, (block, start, end)
        out.data[..., c0:c1] = block.data
        rec("slice_write", block, out, list(start), list(end))

    def update_padded_kv_cache(cache, part, slot_idx, layer_idx, num_layers, kv_actual_global, cluster_axis):
        _alive(cache), _alive(part)
        sp = cache._device.shape[0]
        r = kv_actual_global // sp
        n = part.data.shape[2]
        cache.data[slot_idx, 0, r : r + n] = part.data[0, 0]
        rec("update_padded_kv_cache", cache, part, slot_idx=slot_idx, kv_actual_global=kv_actual_global)

    exp.slice_write = slice_write
    exp.deepseek_prefill = types.SimpleNamespace(update_padded_kv_cache=update_padded_kv_cache)
    m.experimental = exp
    return m


import builtins as _b  # noqa: E402

builtins_slice = _b.slice
assert "ttnn" not in sys.modules or getattr(sys.modules["ttnn"], "__fake__", False), "the REAL ttnn is already imported"
TTNN = sys.modules.setdefault("ttnn", _mk_ttnn())


class DRAMZeroFill:
    @staticmethod
    def op(t):
        _alive(t)
        t.data.zero_()
        rec("DRAMZeroFill", t)
        return t


_kvu = types.ModuleType("models.demos.deepseek_v3_d_p.utils.kv_cache_utils")
_kvu.DRAMZeroFill = DRAMZeroFill
sys.modules.setdefault("models.demos.deepseek_v3_d_p.utils.kv_cache_utils", _kvu)
_pf_stub = types.ModuleType("models.demos.deepseek_v3_d_p.tt.v41.prefill")


class _V41PrefillPlaceholder:  # the runtime is built with object.__new__ and given a FakePrefill
    pass


_pf_stub.V41Prefill = _V41PrefillPlaceholder
sys.modules.setdefault("models.demos.deepseek_v3_d_p.tt.v41.prefill", _pf_stub)


# ----------------------------------------------------------------------------------------------------------- fake model
WIDTH, KEYW, WINDOW = 512, 128, 128


class FakeIndexer:
    def __init__(self):
        self._cut_masks = {}

    def _zeros_block_tile(self, S_l, n):
        key = ("zeros", int(S_l), int(n))
        if key not in self._cut_masks:
            if SYS.captured:
                SYS.late_allocs.append(key)
            self._cut_masks[key] = FakeT(torch.zeros(1, 1, int(S_l), int(n), dtype=torch.float64), "bf16", "TILE", MESH)
        return self._cut_masks[key]


class FakeAttn:
    def __init__(self, layer, role, topk):
        self.layer, self.role = layer, role
        self.sliding_window = WINDOW
        self.indexer = FakeIndexer() if role.mode == "full" else None
        self.topk = topk

    def _write_rm(self, cache, block, batch, row):
        _alive(cache), _alive(block)
        H_ = block.data.shape[2]
        assert row + H_ <= cache.data.shape[2], (row, H_, cache)
        cache.data[batch, 0, row : row + H_] = block.data[0, 0]
        rec("write_rm", cache, block, batch, row)

    def reset_score_mask(self, state):
        if state.score_mask is None:
            return
        state.score_mask.data.fill_(NEG)
        state.mask_zeroed_upto = 0
        rec("reset_score_mask", state.score_mask)


class FakeState:
    def __init__(self, slot, cap_entries=0, mask=None):
        self.slot = slot
        self.kv_actual = 0
        self.entry_count = 0
        self.sliding_carry = FakeT(torch.zeros(1, 1, WINDOW, WIDTH, dtype=torch.int64), "bf16", "TILE", MESH)
        self.compressed_kv = (
            FakeT(torch.zeros(1, 1, cap_entries, WIDTH, dtype=torch.int64), "bf16", "TILE", MESH)
            if cap_entries
            else None
        )
        self.index_k = (
            FakeT(torch.zeros(1, 1, cap_entries, KEYW, dtype=torch.int64), "bf16", "TILE", MESH)
            if cap_entries
            else None
        )
        self.score_mask = mask
        self.mask_zeroed_upto = 0


class FakeBlock:
    def __init__(self, layer, attn, states):
        self.layer, self.attn, self.states = layer, attn, states
        self.state = states[0]
        self._attn_islands = None

    def use_slot(self, s):
        self.state = self.states[s]

    def _attn_islands_enabled(self):
        return bool(self._attn_islands)


class FakeEngram:
    """EngramHost's per-slot n-gram history (positions -> ids) and rows hashed over the 4-gram ending at each position."""

    def __init__(self, n_slots, max_seq):
        self.cache = [torch.full((max_seq + 64,), -5, dtype=torch.int64) for _ in range(n_slots)]

    def rows_for(self, slot, part, start):
        part = torch.tensor([int(t) for t in part], dtype=torch.int64)
        c = self.cache[slot]
        c[start : start + len(part)] = part
        pos = torch.arange(start, start + len(part))
        grams = [torch.where(pos - k >= 0, c[(pos - k).clamp(min=0)], torch.full_like(pos, 99991)) for k in range(4)]
        return {1: H(11, *grams), 14: H(14, *grams)}


class FakePrefill:
    """The parts of ``V41Prefill`` the runtime, the export and prefix reuse touch (see the module docstring)."""

    def __init__(self, cfg, *, chunk, max_seq_len, num_users=1, sp=8, topk=64):
        self.cfg, self.chunk, self.max_seq_len, self.num_users, self.sp = (
            cfg,
            int(chunk),
            int(max_seq_len),
            num_users,
            sp,
        )
        self.slot = 0
        self.islands = False
        self._ids_dev = None
        self.n_layers = cfg.first_decoder_layer
        self.kv_only_layer = cfg.first_decoder_layer
        self.blocks = []
        S_l = self.chunk // sp
        masks = {}
        for L in range(self.n_layers):
            role = cfg.role(L)
            attn = FakeAttn(L, role, topk)
            states = {}
            for s in range(num_users):
                if role.mode == "full":
                    r = role.compress_ratio
                    cap = -(-max_seq_len // r) + self.chunk  # + one chunk's padded width
                    cap = -(-cap // 128) * 128
                    if s not in masks:
                        masks[s] = FakeT(torch.full((1, 1, S_l, cap), NEG, dtype=torch.float64), "bf16", "TILE", MESH)
                    assert masks[s].shape[3] == cap
                    states[s] = FakeState(s, cap, masks[s])
                    # _source_alloc_state: the A3 zero block of the FIRST traced chunk's open, built before any capture
                    need = max(WINDOW, topk * r)
                    attn.indexer._zeros_block_tile(S_l, (-(-need // self.chunk) * self.chunk) // r)
                else:
                    states[s] = FakeState(s)
            self.blocks.append(FakeBlock(L, attn, states))
        self.by_layer = {b.layer: b for b in self.blocks}
        r20 = cfg.role(self.kv_only_layer).compress_ratio
        self.kv_attn = FakeAttn(self.kv_only_layer, cfg.role(self.kv_only_layer), topk)
        self.kv_states = {s: FakeState(s, -(-max_seq_len // r20) + self.chunk) for s in range(num_users)}
        self.kv_actuals = {s: 0 for s in range(num_users)}
        self.calls = []

    @property
    def kv_state(self):
        return self.kv_states[self.slot]

    @property
    def kv_actual(self):
        return self.kv_actuals[self.slot]

    @kv_actual.setter
    def kv_actual(self, v):
        self.kv_actuals[self.slot] = int(v)

    def use_slot(self, slot):
        self.slot = int(slot)
        for b in self.blocks:
            b.use_slot(self.slot)

    def reset(self):
        """== V41Prefill.reset (prefill.py:236-251)."""
        rec("pf.reset", self.slot)
        states = [b.state for b in self.blocks] + ([self.kv_state] if self.kv_attn is not None else [])
        for st in states:
            st.kv_actual, st.entry_count = 0, 0
        self.kv_actual = 0
        seen = set()
        for b in self.blocks:
            m = getattr(b.state, "score_mask", None)
            if m is not None and b._attn_islands_enabled() and id(m) not in seen:
                seen.add(id(m))
                b.attn.reset_score_mask(b.state)

    def enable_islands(self):
        rec("pf.enable_islands")
        assert not self.islands
        for b in self.blocks:
            if b.attn.role.mode == "full":
                b._attn_islands = True
        self.islands = True
        SYS.captured = True

    # ---- the compute ------------------------------------------------------------------------------------------------
    def _chunk(self, part, start, engram_host, on_hidden):
        real, C = len(part), self.chunk
        rec("pf._chunk", self.slot, start, real, H(*[int(t) for t in part]) if part else 0)
        assert 0 < real <= C and (real == C or real % 128 == 0), real
        ids = torch.zeros(C, dtype=torch.int64)
        ids[:real] = torch.tensor([int(t) for t in part], dtype=torch.int64)
        h = H(3, ids, torch.arange(C) + start)
        rows = engram_host.rows_for(self.slot, part, start) if engram_host is not None else {}
        for blk in self.blocks:
            if blk.layer in rows:
                er = torch.zeros(C, dtype=torch.int64)
                er[:real] = rows[blk.layer]
                h = H(h, er)
            h = self._attn(blk, h, start, real)
        st = self.kv_state
        assert st.kv_actual == start, (st.kv_actual, start)
        E0 = st.entry_count
        st.compressed_kv.data[0, 0, E0 : E0 + C] = H(20, h).unsqueeze(-1)
        st.index_k.data[0, 0, E0 : E0 + C] = H(21, h).unsqueeze(-1)
        st.sliding_carry.data[0, 0] = H(22, h[real - WINDOW : real]).unsqueeze(-1)
        st.entry_count += real
        st.kv_actual += real
        self.kv_actual += real

    def _attn(self, blk, h, start, real):
        st, L, C = blk.state, blk.layer, self.chunk
        role = self.cfg.role(L)
        assert st.kv_actual == start, (L, st.kv_actual, start)  # RoPE positions / masks key on the counter
        # chunk 0 masks the carry columns (V41Prefill.reset: "chunk 0's carry columns are masked"): a stale carry is not read
        carry_d = digest_rows(st.sliding_carry.data) if st.kv_actual else 0
        if role.mode == "swa":
            ctx = H(L, carry_d, st.kv_actual)
        elif role.mode == "full":
            r = role.compress_ratio
            E0 = st.entry_count
            mask_d = self._glue(blk, st, r) if self._traced(blk, st, r) else H(E0, 5)
            ent_d = digest_rows(st.compressed_kv.data[0, 0, :E0]) if E0 else 0
            key_d = digest_rows(st.index_k.data[0, 0, :E0]) if E0 else 0
            ctx = H(L, carry_d, st.kv_actual, E0, ent_d, key_d, mask_d)
        else:
            src = self.by_layer[role.kv_source].state
            E = src.entry_count  # the source already wrote this chunk's entries
            ent_d = digest_rows(src.compressed_kv.data[0, 0, :E])
            ctx = H(L, carry_d, st.kv_actual, st.entry_count, E, ent_d)
        out = H(h, ctx)
        out = torch.remainder(torch.cumsum(out, 0), P)  # causal: row p sees rows <= p of the chunk
        if role.mode == "full":
            r = role.compress_ratio
            E0 = st.entry_count
            grp = torch.remainder(out.view(C // r, r).sum(dim=1), P)
            st.compressed_kv.data[0, 0, E0 : E0 + C // r] = H(L, grp, 7).unsqueeze(-1)
            st.index_k.data[0, 0, E0 : E0 + C // r] = H(L, grp, 11).unsqueeze(-1)
            st.entry_count += real // r
        elif role.mode != "swa":
            st.entry_count = self.by_layer[role.kv_source].state.entry_count
        st.sliding_carry.data[0, 0] = H(L, out[real - WINDOW : real], 13).unsqueeze(-1)
        st.kv_actual += real
        return out

    def _traced(self, blk, st, r):
        return blk._attn_islands_enabled() and st.kv_actual >= max(WINDOW, blk.attn.topk * r)

    def _glue(self, blk, st, r):
        """TtCSA.glue_chunk's mask writes (csa.py:1116-1146): open [mask_zeroed_upto, entry_count) in ONE block, then the
        full-width cut of this chunk's entries."""
        mask, S_l = st.score_mask, self.chunk // self.sp
        if st.entry_count > st.mask_zeroed_upto:
            c0, n = st.mask_zeroed_upto, st.entry_count - st.mask_zeroed_upto
            z = blk.attn.indexer._zeros_block_tile(S_l, n)
            mask.data[..., c0 : c0 + n] = z.data
            st.mask_zeroed_upto = st.entry_count
        n_cut = self.chunk // r
        mask.data[..., st.entry_count : st.entry_count + n_cut] = 2.0  # the cut block (a marker)
        return digest_rows(mask.data[0, 0, :1])


# ----------------------------------------------------------------------------------------------------------- the export
def make_export(kv_export_mod, cfg, *, max_seq_len, num_users=1):
    """A ``V41KvExport`` of FakeT tensors in the real shapes / layouts (allocate_v41_kv_export, decoder off)."""
    RING = 256
    U = num_users

    def t(b, rows, w, dtype, layout):
        return FakeT(torch.zeros(b, 1, rows, w, dtype=torch.int64), dtype, layout, MESH)

    n_win = cfg.first_decoder_layer + 1 - 2
    return kv_export_mod.V41KvExport(
        swa=t(U * 2, RING, WIDTH, "bfp8", "TILE"),
        win=t(U * n_win, RING, WIDTH, "bf16", "RM"),
        zero=t(1, RING, WIDTH, "bf16", "RM"),
        ent2=t(U * 3, max_seq_len // 2, WIDTH, "bf16", "RM"),
        ent1=t(U, max_seq_len, WIDTH, "bf16", "RM"),
        idx2=t(U * 3, max_seq_len // 2, KEYW, "bfp8", "TILE"),
        idx1=t(U, max_seq_len, KEYW, "bfp8", "TILE"),
        num_users=U,
        max_seq_len=max_seq_len,
        sources2=(2, 8, 14),
        n_win=n_win,
    )


def export_bytes(exp) -> dict:
    return {k: v.data.clone() for k, v in exp.tensors().items()}


def model_state(pf, slot, S, live: bool = True) -> dict:
    """The model's own rows of ``slot``: entries / keys [0, S / ratio) always; with ``live`` also the counters and carries
    (they are the device's position, which a prompt whose LAST chunk was skipped does not move -- by design: the retention
    then holds no live state for it)."""
    out = {"kv_actual": pf.kv_actuals[slot]} if live else {}
    for b in pf.blocks:
        st = b.states[slot]
        r = pf.cfg.role(b.layer).compress_ratio
        out[b.layer] = [st.kv_actual, st.entry_count, st.sliding_carry.data.clone()] if live else []
        if st.compressed_kv is not None:
            n = S // r
            out[b.layer] += [st.compressed_kv.data[0, 0, :n].clone(), st.index_k.data[0, 0, :n].clone()]
    st = pf.kv_states[slot]
    out[20] = ([st.kv_actual, st.entry_count, st.sliding_carry.data.clone()] if live else []) + [
        st.compressed_kv.data[0, 0, :S].clone(),
        st.index_k.data[0, 0, :S].clone(),
    ]
    return out


def same(a, b) -> bool:
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, torch.Tensor):
        return a.shape == b.shape and bool(torch.equal(a, b))
    return a == b
