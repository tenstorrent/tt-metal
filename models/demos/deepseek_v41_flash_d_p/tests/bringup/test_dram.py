# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.27 (owner 2026-11-02): DRAM per chip, measured from the allocator, for ONE serving slot at the deployment context.

The model as the serving runtime loads it (tt/runners/adapter.py: TtDS41Model over every layer, lm_head off, the
routed experts' weight cache) with the engine's caches for one slot (tt/runners/kv_contract.DS41ContractKV, num_users
1) at CONTEXT = 1,048,576 tokens (max_seq = the smallest whole number of chunks holding it: 1,049,600), the slot's
state bound to its slabs (keep_history off, as served). Then one chunk runs at START = 1,039,360 (the last whole chunk
before 1M) exactly as DS41PrefillRuntime._run does (note_tokens, begin_chunk, embed, every layer), the token ids random.

Read from the allocator (ttnn.get_memory_view(<device>, DRAM): allocated bytes per bank x banks), per chip of the mesh
(each chip's own view where the device exposes one; a mesh's lockstep allocator gives every chip the same addresses, so
the mesh view stands for each chip otherwise; `per_chip_source` says which):

    dram_capacity_gib     the allocator's DRAM per chip
    dram_load_gib_max/min  allocated after the load (model + served caches + the slot's state), over chips
    dram_peak_gib_max      the max over chips and samples while the chunk runs: after every layer, and inside the MoE
                           and the attention (before every free there: allocation only grows between frees, so the
                           sample before each free is the high-water mark; ccl.dealloc's binding and ttnn.deallocate
                           are wrapped for the chunk only)
    dram_gib_<group>       bytes per group from the buffers the model allocates (each DRAM tensor once, by address):
                           routed_experts (the flat op's w_gu / w_d / w_rd of every layer), other_weights (every other
                           tensor the layers / embedding / norm hold), served_caches (the engine-owned slabs),
                           tables (load-time tables: rope, window / sparse tables, masks, index tables), state (the
                           slot's other device state), buffers (load-time persistent outputs: host_alloc / pinned),
                           unattributed (allocator after load minus the groups)

Recorded, never gated by a fixed number (the gate checks only that the three totals are > 0). EXPERTS_IMPL decides
the routed experts' bytes (the A/B runs this test on both sides). Results also in <results>/P.27_dram.json.
"""

from __future__ import annotations

import time
import types

import pytest
import torch

import ttnn
from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing.harness import device_params, impl_mode, spec

S = spec()
CONTEXT = 1 << 20  # tokens a slot holds (1,048,576)
START = 1_039_360  # the last whole 5120-token chunk before 1M (203 chunks)
GIB = float(1 << 30)
TIMEOUT = max(int(S.get("box.test_timeout_s")), 4 * 3600)  # (a cold routed-expert cache packs every layer: ~2 min each)
_TABLE_WORDS = ("table", "rope", "cos", "sin", "mask", "page", "idx", "index_map", "perm")


def _view(dev):
    return ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)


def _alloc_bytes(v) -> int:
    return int(v.total_bytes_allocated_per_bank) * int(v.num_banks)


class Dram:
    """Per-chip allocated DRAM: each chip's own view where it differs from nothing, else the mesh's (lockstep)."""

    def __init__(self, mesh):
        self.mesh = mesh
        self.devs = []
        try:
            self.devs = list(mesh.get_devices())
        except Exception:  # noqa: BLE001
            self.devs = []
        self.n = int(mesh.get_num_devices())
        v = _view(mesh)
        self.capacity = int(v.total_bytes_per_bank) * int(v.num_banks)
        self.source = "mesh (lockstep: every chip the same)"
        if self.devs:
            try:
                per = [_alloc_bytes(_view(d)) for d in self.devs]
                if any(per) and max(per) >= _alloc_bytes(v):
                    self.source = "per chip"
            except Exception:  # noqa: BLE001
                pass

    def per_chip(self) -> list[int]:
        if self.source == "per chip":
            return [_alloc_bytes(_view(d)) for d in self.devs]
        return [_alloc_bytes(_view(self.mesh))] * self.n

    def max(self) -> int:
        return max(self.per_chip()) if self.source == "per chip" else _alloc_bytes(_view(self.mesh))


def _dram_tensors(roots, skip: set | None = None) -> dict:
    """{buffer address: (bytes per chip, path)} of every allocated DRAM device tensor reachable from roots (name ->
    object) through attributes, dicts, lists and tuples."""
    skip = skip if skip is not None else set()
    out, seen = {}, set(skip)
    stack = [(o, name) for name, o in roots.items()]
    leaf = (str, bytes, int, float, bool, type(None), torch.Tensor, types.ModuleType, types.FunctionType, type)
    while stack:
        o, path = stack.pop()
        if id(o) in seen or isinstance(o, leaf):
            continue
        seen.add(id(o))
        if isinstance(o, ttnn.Tensor):
            try:
                if (
                    o.storage_type() == ttnn.StorageType.DEVICE
                    and o.is_allocated()
                    and o.memory_config().buffer_type == ttnn.BufferType.DRAM
                ):
                    out.setdefault(
                        int(o.buffer_address()), (int(o.buffer_num_pages()) * int(o.buffer_aligned_page_size()), path)
                    )
            except Exception:  # noqa: BLE001
                pass
            continue
        mod = type(o).__module__ or ""
        if mod.startswith(("ttnn._ttnn", "numpy", "threading", "concurrent", "loguru", "logging", "queue")):
            continue
        if path.count(".") > 14:
            continue
        if isinstance(o, dict):
            stack += [(v, f"{path}[{k!r}]") for k, v in list(o.items())]
        elif isinstance(o, (list, tuple, set, frozenset)):
            stack += [(v, f"{path}[{i}]") for i, v in enumerate(o)]
        else:
            d = getattr(o, "__dict__", None)
            if isinstance(d, dict):
                stack += [(v, f"{path}.{k}") for k, v in list(d.items())]
            for k in getattr(type(o), "__slots__", ()) or ():
                if hasattr(o, k):
                    stack.append((getattr(o, k), f"{path}.{k}"))
    return out


def _routed(model) -> dict:
    out = {}
    for i, blk in model.blocks.items():
        ops = []
        mb = getattr(blk, "moe_block", None)
        if mb is not None:
            ops.append(mb.weights.experts.op)
        ex = getattr(blk, "experts", None)
        if ex is not None and getattr(ex, "fe", None) is not None:
            ops.append(ex.fe)
        for op in ops:
            for n in ("w_gu", "w_d", "w_rd"):
                t = getattr(op, n, None)
                if t is not None and t.is_allocated():
                    out.setdefault(
                        int(t.buffer_address()),
                        (int(t.buffer_num_pages()) * int(t.buffer_aligned_page_size()), f"L{i}.{n}"),
                    )
    return out


def _groups(model, state, contract, load_bytes: int) -> dict:
    """Bytes per chip per group (module docstring), each DRAM buffer counted once."""
    from models.demos.deepseek_v41_flash_d_p.tt import ccl

    taken = {}

    def put(group, found):
        for a, (b, p) in found.items():
            if a not in taken:
                taken[a] = (group, b, p)

    put("routed_experts", _routed(model))
    put("served_caches", _dram_tensors({f"served.{k}": v for k, v in contract.by_config.items()}))
    allt = _dram_tensors(
        {
            "model.tables": model.tables,
            "model._rope": model._rope,
            "model._crope": model._crope,
            "model._sparse": model._sparse,
        }
    )
    put("tables", allt)
    from models.demos.deepseek_v41_flash_d_p.tt import host_alloc, moe_block_adapter

    regs = {
        "ccl._SITES": ccl._SITES,
        "ccl._PINS": ccl._PINS,
        "host_alloc._TAIL_OUT": host_alloc._TAIL_OUT,
        "host_alloc._FORKS": host_alloc._FORKS,
        "moe_block._BLOCKS": moe_block_adapter._BLOCKS,
        "moe_block._DONE": moe_block_adapter._DONE,
    }
    put("buffers", _dram_tensors(regs))
    # (before the model's walk: model.state may be this slot's state; the state's walk skips the model it points back
    # to, DS41DeviceState.model, else every weight would count as state)
    put("state", _dram_tensors({"state": state}, skip={id(model), id(model.mesh)}))
    for a, (b, p) in _dram_tensors({"model": model}).items():
        if a in taken:
            continue
        last = p.rsplit(".", 1)[-1].lower()
        taken[a] = ("tables" if any(w in last for w in _TABLE_WORDS) else "other_weights", b, p)
    sums = {}
    for g, b, _ in taken.values():
        sums[g] = sums.get(g, 0) + b
    sums["unattributed"] = load_bytes - sum(sums.values())
    top = sorted(((b, g, p) for g, b, p in taken.values() if g == "other_weights"), reverse=True)[:8]
    print("[dram] largest other_weights buffers: " + "; ".join(f"{p} {b / GIB:.3f} GiB" for b, _, p in top))
    return sums


class Sampler:
    """Allocated DRAM before every free (and at the marks) while the chunk runs, labelled by layer and module."""

    def __init__(self, dram: Dram):
        self.dram, self.label, self.layer = dram, "embed", -1
        self.samples = []  # (layer, label, max over chips)
        self.n = 0

    def mark(self, label=None):
        v = self.dram.max()
        self.samples.append((self.layer, label or self.label, v))
        return v

    def wrap(self, fn):
        def dealloc(t, *a, **k):
            try:
                if t.is_allocated() and t.memory_config().buffer_type == ttnn.BufferType.DRAM:
                    self.n += 1
                    self.mark()
            except Exception:  # noqa: BLE001
                pass
            return fn(t, *a, **k)

        return dealloc

    def in_module(self, cls, name):
        orig = cls.__call__
        sampler = self

        def call(obj, *a, **k):
            prev, sampler.label = sampler.label, name
            sampler.mark(f"{name}:in")
            try:
                return orig(obj, *a, **k)
            finally:
                sampler.mark(f"{name}:out")
                sampler.label = prev

        cls.__call__ = call
        return cls, orig


@pytest.mark.timeout(TIMEOUT)
@pytest.mark.parametrize("device_params", [device_params(S)], indirect=True, ids=["box"])
@pytest.mark.parametrize("mesh_device", [tuple(S.mesh)], indirect=True, ids=["x".join(map(str, S.mesh))])
def test_dram(mesh_device):
    if metrics.in_precompile_collect_pass() or impl_mode() != "device":
        pytest.skip("device only (no precompile pass)")
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import Config
    from models.demos.deepseek_v41_flash_d_p.tt import ccl
    from models.demos.deepseek_v41_flash_d_p.tt.experts import expert_cache_dir
    from models.demos.deepseek_v41_flash_d_p.tt.model import TtDS41Model
    from models.demos.deepseek_v41_flash_d_p.tt.runners.kv_contract import DS41ContractKV
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    mesh = mesh_device
    chunk = int(S.get("deploy.chunk") or 5120)
    max_seq = -(-CONTEXT // chunk) * chunk
    assert START % chunk == 0 and START + chunk <= max_seq, (START, chunk, max_seq)
    layers = list(range(S.num_layers))
    path = hf_path(S)
    dram = Dram(mesh)
    rec = {
        "context": CONTEXT,
        "max_seq": max_seq,
        "chunk": chunk,
        "start": START,
        "slots": 1,
        "layers": len(layers),
        "mesh": list(mesh.shape),
        "per_chip_source": dram.source,
        "settings": {k: settings.get(k) for k in ("EXPERTS_IMPL", "MOE_IMPL", "ENGRAM_TABLE", "WINDOW_KV_FORMAT")},
    }
    print(f"[dram] {rec}")
    b0 = dram.max()
    t0 = time.time()
    contract = DS41ContractKV(mesh, Config.from_dir(path), layers, max_seq=max_seq, chunk=chunk, num_users=1)
    b1 = dram.max()
    model = TtDS41Model(
        mesh, path, layers, max_seq=max_seq, chunk=chunk, lm_head=False, weight_cache=expert_cache_dir(S)
    )
    b2 = dram.max()
    state = model.new_state(max_seq, slabs=contract.slabs(0), keep_history=False)
    b3 = dram.max()
    rec["load_s"] = round(time.time() - t0, 1)
    load = dram.per_chip()
    rec["alloc_gib"] = {
        "open": b0 / GIB,
        "served_caches_alloc": (b1 - b0) / GIB,
        "model_load": (b2 - b1) / GIB,
        "slot_state": (b3 - b2) / GIB,
    }
    groups = _groups(model, state, contract, max(load))
    rec["groups_gib"] = {k: round(v / GIB, 4) for k, v in groups.items()}

    # ---- one chunk at START, as DS41PrefillRuntime._run (no acks)
    sampler = Sampler(dram)
    patched = []
    for blk in model.blocks.values():
        for attr, name in (("attention", "attention"), ("moe_block", "moe")):
            obj = getattr(blk, attr, None)
            if obj is not None and not any(type(obj) is c for c, _ in patched):
                patched.append(sampler.in_module(type(obj), name))
    d0, td = ccl._DEALLOC[0], ttnn.deallocate
    ccl._DEALLOC[0] = sampler.wrap(d0)
    ttnn.deallocate = sampler.wrap(td)
    g = torch.Generator().manual_seed(27)
    ids = torch.randint(0, model.vocab, (chunk,), generator=g, dtype=torch.int64)
    end = START + chunk
    t1 = time.time()
    try:
        model.state = state
        state.note_tokens(ids, START, end)
        model.begin_chunk(ids, START)
        sampler.mark("begin_chunk")
        h = model.embed(ids, START)
        sampler.mark("embed")
        for i in layers:
            sampler.layer, sampler.label = i, "layer"
            h2 = model.layer(i, h, START, state, end=end)
            model.free(h)
            h = h2
            sampler.mark("layer_end")
        model.free(h)
        ttnn.synchronize_device(mesh)
        sampler.mark("done")
    finally:
        ccl._DEALLOC[0], ttnn.deallocate = d0, td
        for cls, orig in patched:
            cls.__call__ = orig
    rec["chunk_s"] = round(time.time() - t1, 1)
    peak = max(sampler.samples, key=lambda s: s[2])
    by_label = {}
    for layer, label, v in sampler.samples:
        k = label.split(":")[0]
        by_label[k] = max(by_label.get(k, 0), v)
    rec["peak"] = {"layer": peak[0], "where": peak[1], "gib": peak[2] / GIB, "samples": len(sampler.samples)}
    rec["peak_by_module_gib"] = {k: round(v / GIB, 4) for k, v in by_label.items()}
    rec["after_layer_gib"] = {
        int(layer): round(v / GIB, 4) for layer, label, v in sampler.samples if label == "layer_end"
    }

    cap, lmax, lmin = dram.capacity / GIB, max(load) / GIB, min(load) / GIB
    print(
        f"[dram] EXPERTS_IMPL={rec['settings']['EXPERTS_IMPL']} capacity {cap:.3f} GiB/chip; after load max {lmax:.3f} "
        f"min {lmin:.3f} GiB ({dram.source}); peak {peak[2] / GIB:.3f} GiB at layer {peak[0]} {peak[1]} "
        f"({len(sampler.samples)} samples); groups {rec['groups_gib']}; allocs {rec['alloc_gib']}; "
        f"load {rec['load_s']} s, chunk {rec['chunk_s']} s"
    )
    metrics.record("dram_capacity_gib", round(cap, 4))
    metrics.record("dram_load_gib_max", round(lmax, 4))
    metrics.record("dram_load_gib_min", round(lmin, 4))
    metrics.record("dram_peak_gib_max", round(peak[2] / GIB, 4))
    for k, v in rec["groups_gib"].items():
        metrics.record(f"dram_gib_{k}", v)
    metrics.record_json("dram", rec)
    state.free()
    del model
    assert lmax > 0 and peak[2] > 0 and cap > 0
