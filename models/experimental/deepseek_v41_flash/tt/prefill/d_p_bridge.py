# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""metal_2's whole-mesh DeepSeek-V4.1-Flash prefill (``models/demos/deepseek_v41_flash_d_p``) as a prefill source for
this model's decode (see ``PREFILL_D_P_INTEGRATION.md``).

:class:`DPPrefill` runs that prefill (SP over mesh rows, TP over mesh columns, one 8 x 4 mesh) through its own serving
runtime, layers 0-20 only: ``DS41_DECODER_SKIP=decode`` runs layers 0-19 and layer 20's KV / index head, which is what
this model's ``kv_only`` prefill produces. :meth:`DPPrefill.host_states` reads the result back as host tensors, and
:func:`to_prefill_states` turns them into the :class:`PrefillAttentionState` list
:meth:`~..model.DeepSeekV41Model.commit_prefill_state` takes.

Prefix caching: the runtime's KV cache holds several slots, each a one-user cache of ``max_seq`` tokens that keeps its
last prompt's state on the device until it is reused. A prompt continuing a slot's prompt resumes there
(:meth:`DPPrefill.resume_start`, :meth:`DPPrefill.prefill` ``start``), exactly: every chunk reads the slot's compressed
KV and index keys of the earlier positions, and its first chunk the window carry the slot's rings still hold (metal_2
keeps the last 192 rows a layer, so a slot continues only from its latest prompt's end, or up to 64 positions before
it). ``PREFILL_D_P_PREFIX_CACHE.md`` has the server and decode sides.

The hand-off goes through the host: metal_2's caches are striped over the whole mesh, decode's live on ``1 x TP``
stage submeshes, and a parent mesh and its submeshes allocate independently, so the two models must not hold memory
on the same chips at once (``PREFILL_D_P_INTEGRATION.md`` section 5).

Host state format, ``{"seq_len": T, "base": B, "layers": {i: {...}}}`` with per layer

* ``kv_tail``       bf16 ``[sliding_window, head_dim]``: the K=V rows of positions ``[T - window, T)`` in order,
* ``compressed_kv`` bf16 ``[(T - B) / ratio, head_dim]`` and ``idx_keys`` bf16 ``[(T - B) / ratio, index_head_dim]``
  (KV sources): the entries of positions ``[B, T)``, the ones a decode session holding ``[0, B)`` lacks (B 0: all).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
from loguru import logger

# metal_2 settings the bridge depends on (read by its model at load; DS41_ is metal_2's settings prefix).
# DECODER_SKIP=decode: layers 21-39 never run in prefill, as kv_only here. INDEX_K_SERVED=bf16: plain bf16 index keys
# (the default fp8_row192 is Blaze decode's row format; decoding it back loses precision this decode does not want).
_D_P_ENV = {
    "DS41_DECODER_SKIP": "decode",
    "DS41_INDEX_K_SERVED": "bf16",
}
D_P_CHUNK = 5120


def _apply_env() -> None:
    for name, value in _D_P_ENV.items():
        got = os.environ.setdefault(name, value)
        if got != value:
            raise ValueError(f"{name}={got}: the d_p bridge needs {name}={value}")


def kv_only_layers(cfg) -> list[int]:
    """Layers 0 .. the last KV source (layer 20): the layers whose state decode needs from a prefill."""
    return list(range(max(cfg.kv_sources) + 1))


class DPPrefill:
    """metal_2's prefill runtime over ``mesh_device`` (the whole 8 x 4 mesh, opened with metal_2's device params),
    prompts of up to ``max_seq`` tokens in chunks of ``chunk``, into one of ``len(slots)`` cache slots.

    ``slots`` cache slots, or with ``dram_reserve`` (bytes a chip) up to ``slots``, stopping before the next would leave
    less than that much DRAM free for the chunks' working buffers and the programs later starts compile. Each slot is
    a one-user ``KvCaches`` with its window rings, allocated before the cold-chunk compile, so the warm-up runs with
    the memory serving will have.

    ``prefill(ids, slot, start)`` runs a prompt chunk by chunk from ``start``; ``host_states(T, slot=, base=)`` reads
    layers 0-20 back."""

    def __init__(
        self,
        mesh_device,
        max_seq: int,
        chunk: int = D_P_CHUNK,
        progress: Optional[Callable] = None,
        slots: int = 1,
        dram_reserve: Optional[int] = None,
    ):
        _apply_env()
        from models.demos.common.prefill.adapter import PrefillRunParams
        from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import Config
        from models.demos.deepseek_v41_flash_d_p.tt.runners.adapter import DS41PrefillAdapter, model_path

        note = progress or (lambda message: logger.info(f"[d_p] {message}"))
        self.mesh_device, self.chunk = mesh_device, int(chunk)
        self.max_seq = -(-int(max_seq) // self.chunk) * self.chunk
        self.cfg = Config.from_dir(model_path())
        self.layers = kv_only_layers(self.cfg)
        sp, tp = tuple(mesh_device.shape)
        adapter = DS41PrefillAdapter()
        hf = adapter.load_hf_config()
        hf.max_seq_len = self.max_seq
        params = PrefillRunParams(
            mesh_shape=(sp, tp),
            num_layers=len(self.layers),
            first_layer_idx=0,
            is_first_rank=True,
            is_last_rank=True,
            max_seq_len=self.max_seq,
            chunk_size=self.chunk,
            num_users=1,
            capacity_factor=8,
            num_links=2,
            gate_mode_name=adapter.default_gate_mode,
            kv_only_last_layer=True,
            weight_cache_path=adapter.weight_cache_path((sp, tp)),
        )
        t0 = time.perf_counter()
        note(f"loading layers {self.layers[0]}-{self.layers[-1]} on the {sp} x {tp} mesh (SP {sp}, TP {tp})")
        self.runtime = adapter.build_runtime(mesh_device=mesh_device, hf_config=hf, params=params)
        assert self.runtime.layer_ids == self.layers, (self.runtime.layer_ids, self.layers)
        assert self.runtime.model.skip_mode == "decode", self.runtime.model.skip_mode
        self.slots: list = []
        slot_bytes = 0
        while len(self.slots) < max(1, int(slots)):
            free = self._dram_free()
            if dram_reserve is not None and self.slots and free - slot_bytes < dram_reserve:
                break
            kv = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf, params=params)
            self.runtime.state(kv, 0)  # (the slot's window rings and stores, allocated now)
            # (the first slot's figure also counts the history buffers every slot shares: an overestimate)
            slot_bytes = free - self._dram_free()
            self.slots.append(kv)
        self.kv = self.slots[0]
        note(
            f"{len(self.slots)} cache slot(s) of {self.max_seq} tokens, {slot_bytes / 2**20:.0f} MiB a chip each, "
            f"{self._dram_free() / 2**30:.1f} GiB a chip of DRAM left; runtime ready in {time.perf_counter() - t0:.0f} s; "
            "compiling one cold chunk"
        )
        t0 = time.perf_counter()
        self.runtime.compile(self.kv)
        note(f"compiled in {time.perf_counter() - t0:.0f} s")

    def _dram_free(self) -> int:
        """Free DRAM bytes a chip (the mesh allocates in lockstep: every chip the same)."""
        import ttnn

        view = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.DRAM)
        return int(view.total_bytes_free_per_bank) * int(view.num_banks)

    def resume_start(self, slot: int, held: int, end: int) -> int:
        """Where a prompt of ``end`` tokens whose first ``held`` tokens are slot ``slot``'s can continue from exactly:
        ``held`` (32-aligned down, at most ``end``; ``end`` itself: the slot holds the whole prompt, nothing to run),
        pulled back to ``max_seq - chunk`` when the first chunk would overrun the slot; 0 when that position's window
        carry (the 128 rows before it of every layer this prefill writes) is no longer in the slot's rings."""
        r = min(int(held), int(end)) // 32 * 32
        if r < end and r + self.chunk > self.max_seq:
            r = self.max_seq - self.chunk
        if r <= 0:
            return 0
        st = self.runtime.state(self.slots[slot], 0)
        skipped = set(self.runtime.model.decoder)
        return r if all(st.window[i].carry_ready(r) for i in self.layers if i not in skipped) else 0

    def prefill(
        self,
        ids: Sequence[int],
        on_chunk: Optional[Callable[[int, int, float], None]] = None,
        slot: int = 0,
        start: int = 0,
    ) -> int:
        """Prefill ``ids`` into cache slot ``slot`` from position ``start``; returns the prompt length. ``start`` > 0
        continues the slot's prompt, which holds ``ids[:start]`` (:meth:`resume_start`). The chunks keep to the
        ``chunk`` grid: a start off it runs a partial first chunk up to the grid. ``on_chunk(start, end, seconds)``
        after each chunk (host enqueue time; the device runs behind)."""
        import ttnn

        ids = [int(t) for t in ids]
        n = len(ids)
        if not 0 < n <= self.max_seq:
            raise ValueError(f"prompt of {n} tokens: the runtime holds 1 to {self.max_seq}")
        if not 0 <= start < n or start % 32 or start + self.chunk > self.max_seq:
            raise ValueError(
                f"start {start} for a {n}-token prompt: 32-aligned, before its end, at most {self.max_seq - self.chunk}"
            )
        kv = self.slots[slot]
        pos = int(start)
        while pos < n:
            end = min(n, (pos // self.chunk + 1) * self.chunk)
            t0 = time.perf_counter()
            inp = self.runtime.make_chunk_input(ids[pos:end], pos)
            self.runtime.prefill_chunk(inp, kv, slot_id=0, actual_start=pos, actual_end=end)
            ttnn.deallocate(inp)
            if on_chunk is not None:
                on_chunk(pos, end, time.perf_counter() - t0)
            pos = end
        ttnn.synchronize_device(self.mesh_device)
        return n

    def host_states(self, seq_len: int, times: Optional[dict] = None, slot: int = 0, base: int = 0) -> dict:
        """Layers 0-20 of cache slot ``slot`` after a prompt of ``seq_len`` tokens, in the host format (module
        docstring), the compressed entries from position ``base`` on (a multiple of 32, at most ``seq_len``).
        Layer 20's window ring is zeros: DECODER_SKIP=decode never writes it (decode's tail replay rebuilds it).

        Reads only what decode needs, in bf16: the compressed rows of ``[base, seq_len)`` (a device slice of each
        owner's slab, not the whole ``max_seq`` slab of every chip), one copy of TP-replicated stripes, and the window
        blocks of the last ``window`` positions from their owners. ``times``: seconds per part, added up."""
        cfg, T, B = self.cfg, int(seq_len), int(base)
        if not 0 <= B <= T or B % 32:
            raise ValueError(f"base {B} for {T} tokens: a multiple of 32 up to {T}")
        st = self.runtime.state(self.slots[slot], 0)
        skipped = set(self.runtime.model.decoder)
        window = cfg.window
        p0 = max(0, T - window)
        times = times if times is not None else {}
        out = {}
        for i in self.layers:
            t0 = time.perf_counter()
            tail = torch.zeros(window, cfg.head_dim, dtype=torch.bfloat16)
            if i not in skipped:
                tail[window - (T - p0) :] = _window_rows(st.window[i], p0, T)
            layer = {"kv_tail": tail}
            t1 = time.perf_counter()
            times["window"] = times.get("window", 0.0) + t1 - t0
            if cfg.is_kv_source(i):
                c = st.compress[i]
                layer["compressed_kv"] = _compressed_rows(c.ckv, T, B)
                layer["idx_keys"] = _compressed_rows(c.index_k, T, B)
                times["compressed"] = times.get("compressed", 0.0) + time.perf_counter() - t1
            out[i] = layer
        return {"seq_len": T, "base": B, "layers": out}


def _compressed_rows(c, length: int, base: int = 0) -> torch.Tensor:
    """Natural-order rows ``[(length - base) / ratio, width]`` bf16 (positions ``[base, length)``) of metal_2's served
    compressed cache ``c`` (``tt/compressor.py`` ``TtServedRows``), read from the owners' slabs over the blocks those
    positions reach."""
    import ttnn

    from models.demos.deepseek_v41_flash_d_p.tt.served import dealloc

    first, need = base // c.ratio, length // c.ratio
    if c.fp8:  # (INDEX_K_SERVED=fp8_row192: decoded on the host by metal_2)
        return c.to_host(length)[first:need].to(torch.bfloat16)
    slab = c.slab if c.slab is not None else c.served
    if need <= first:
        return torch.zeros(0, slab.width, dtype=torch.bfloat16)
    hs = c.stripe // c.ratio  # compressed rows per owner stripe
    period = c.owners * hs  # natural rows per block: one stripe of every owner
    b0 = first // period
    blocks = min(-(-need // period), c.crows // hs) - b0
    rows = blocks * hs
    sl = slab.slice_rows(b0 * hs, b0 * hs + rows)
    try:
        devs = ttnn.get_device_tensors(sl)
        if c.tp_axis is None:  # replicated over TP: mesh column 0's copy of every SP row
            devs = devs[:: c.geo.tp]
        shards = [ttnn.to_torch(t).reshape(rows, slab.width) for t in devs]
    finally:
        dealloc(sl)
    nat = torch.stack([s.reshape(blocks, hs, slab.width) for s in shards], dim=1).reshape(-1, slab.width)
    return nat[first - b0 * period : need - b0 * period].to(torch.bfloat16)


def _window_rows(win, p0: int, p1: int) -> torch.Tensor:
    """Positions ``[p0, p1)`` of metal_2's window cache ``win`` (``tt/attention.py`` ``TtWindowCache``) as bf16 rows,
    reading only the chips that own those ring blocks (``_ring_natural`` reads every chip's ring)."""
    import ttnn

    from models.demos.deepseek_v41_flash_d_p.tt.attention import RING

    ring = win.ring
    if ring is None:  # WINDOW_KV_KEEP=full: the served slab
        return win.to_host(p1)[p0:p1].to(torch.bfloat16)
    out = torch.zeros(max(0, p1 - p0), win.WIDTH, dtype=torch.bfloat16)
    if p1 <= p0 or ring.writer is not win and win.keep == "none" or not win.history:
        return out
    devs, shards = ttnn.get_device_tensors(ring.t), {}
    for j in range(ring.BLOCKS):
        q = ring.pos[j]
        if q is None:
            continue
        lo, hi = max(q, p0), min(ring.vend[j], p1)
        if lo < hi:
            k = ring.own[j] or 0
            if k not in shards:
                shards[k] = ttnn.to_torch(devs[k]).reshape(RING, win.WIDTH)
            out[lo - p0 : hi - p0] = shards[k][q % RING + lo - q : q % RING + hi - q]
    return out


def save_host_states(host: dict, path: str | Path, **extra) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**host, **extra}, path)
    return path


def load_host_states(path: str | Path) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def to_prefill_states(host: dict, layer_devices: Sequence, config) -> list:
    """Host states -> one :class:`PrefillAttentionState` a layer, ROW_MAJOR bf16 replicated on ``layer_devices[i]``
    (the layer's decode stage, so :meth:`~..model.DeepSeekV41Model.commit_prefill_state` writes its ring in place and
    migrates its source's entries from there). ``config`` is this model's checkpoint config (shapes checked)."""
    import ttnn
    from models.experimental.deepseek_v4_flash.tt.prefill.attention import PrefillAttentionState

    T = int(host["seq_len"])
    if host.get("base", 0):
        raise ValueError(
            f"host states hold the entries from position {host['base']} only: commit_prefill_state needs an entry "
            "offset for them first (PREFILL_D_P_PREFIX_CACHE.md)"
        )
    layers = host["layers"]
    sources = [i for i in config.kv_source_layer_ids if i < len(layer_devices)]
    missing = [i for i in range(max(sources) + 1) if i not in layers]
    if missing:
        raise ValueError(f"host states lack layers {missing} (decode needs 0 .. {max(sources)})")

    def upload(rows: torch.Tensor, device) -> ttnn.Tensor:
        return ttnn.from_torch(
            rows.to(torch.bfloat16).reshape(1, 1, *rows.shape[-2:]),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

    states = []
    for i in sorted(layers):
        layer, device = layers[i], layer_devices[i]
        tail = layer["kv_tail"]
        if tuple(tail.shape) != (config.sliding_window, config.head_dim):
            raise ValueError(
                f"layer {i}: kv_tail {tuple(tail.shape)}, decode wants ({config.sliding_window}, {config.head_dim})"
            )
        state = PrefillAttentionState(seq_len=T, kv_tail=upload(tail, device))
        if i in sources:
            ratio = int(config.compress_ratios[i])
            entries = T // ratio
            ckv, keys = layer["compressed_kv"], layer["idx_keys"]
            if ckv.shape[0] != entries or keys.shape[0] != entries:
                raise ValueError(
                    f"layer {i}: {ckv.shape[0]} entries / {keys.shape[0]} keys for {T} tokens at ratio {ratio}"
                )
            state.compressed_kv = upload(ckv, device)
            state.idx_keys = upload(keys, device)
        states.append(state)
    return states


def free_prefill_states(states: Sequence) -> None:
    import ttnn

    for state in states:
        for t in (state.kv_tail, state.compressed_kv, state.idx_keys):
            if t is not None:
                ttnn.deallocate(t)
