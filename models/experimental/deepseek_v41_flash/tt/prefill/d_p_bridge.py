# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""metal_2's whole-mesh DeepSeek-V4.1-Flash prefill (``models/demos/deepseek_v41_flash_d_p``) as a prefill source for
this model's decode (see ``PREFILL_D_P_INTEGRATION.md``).

:class:`DPPrefill` runs that prefill (SP over mesh rows, TP over mesh columns, one 8 x 4 mesh) through its own serving
runtime with one slot, layers 0-20 only: ``DS41_DECODER_SKIP=decode`` runs layers 0-19 and layer 20's KV / index head,
which is what this model's ``kv_only`` prefill produces. :meth:`DPPrefill.host_states` reads the result back as host
tensors, and :func:`to_prefill_states` turns them into the :class:`PrefillAttentionState` list
:meth:`~..model.DeepSeekV41Model.commit_prefill_state` takes.

The hand-off goes through the host: metal_2's caches are striped over the whole mesh, decode's live on ``1 x TP``
stage submeshes, and a parent mesh and its submeshes allocate independently, so the two models must not hold memory
on the same chips at once (``PREFILL_D_P_INTEGRATION.md`` section 5).

Host state format, ``{"seq_len": T, "layers": {i: {...}}}`` with per layer

* ``kv_tail``       bf16 ``[sliding_window, head_dim]``: the K=V rows of positions ``[T - window, T)`` in order,
* ``compressed_kv`` bf16 ``[T / ratio, head_dim]`` and ``idx_keys`` bf16 ``[T / ratio, index_head_dim]`` (KV sources).
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
    """metal_2's prefill runtime over ``mesh_device`` (the whole 8 x 4 mesh, opened with metal_2's device params), one
    slot, prompts of up to ``max_seq`` tokens in chunks of ``chunk``.

    ``prefill(ids)`` runs a prompt from position 0, chunk by chunk; ``host_states(T)`` reads layers 0-20 back."""

    def __init__(self, mesh_device, max_seq: int, chunk: int = D_P_CHUNK, progress: Optional[Callable] = None):
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
        self.kv = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf, params=params)
        note(f"runtime ready in {time.perf_counter() - t0:.0f} s; compiling one cold chunk")
        t0 = time.perf_counter()
        self.runtime.compile(self.kv)
        note(f"compiled in {time.perf_counter() - t0:.0f} s")

    def prefill(self, ids: Sequence[int], on_chunk: Optional[Callable[[int, int, float], None]] = None) -> int:
        """Prefill ``ids`` into the slot from position 0; returns the prompt length. ``on_chunk(start, end, seconds)``
        after each chunk (host enqueue time; the device runs behind)."""
        import ttnn

        ids = [int(t) for t in ids]
        n = len(ids)
        if not 0 < n <= self.max_seq:
            raise ValueError(f"prompt of {n} tokens: the runtime holds 1 to {self.max_seq}")
        for start in range(0, n, self.chunk):
            end = min(start + self.chunk, n)
            t0 = time.perf_counter()
            inp = self.runtime.make_chunk_input(ids[start:end], start)
            self.runtime.prefill_chunk(inp, self.kv, slot_id=0, actual_start=start, actual_end=end)
            ttnn.deallocate(inp)
            if on_chunk is not None:
                on_chunk(start, end, time.perf_counter() - t0)
        ttnn.synchronize_device(self.mesh_device)
        return n

    def host_states(self, seq_len: int, times: Optional[dict] = None) -> dict:
        """Layers 0-20 of the slot after a prompt of ``seq_len`` tokens, in the host format (module docstring).
        Layer 20's window ring is zeros: DECODER_SKIP=decode never writes it (decode's tail replay rebuilds it).

        Reads only what decode needs, in bf16: the compressed rows the prompt covers (a device slice of each owner's
        slab, not the whole ``max_seq`` slab of every chip), one copy of TP-replicated stripes, and the window blocks
        of the last ``window`` positions from their owners. ``times``: seconds per part, added up."""
        cfg, T = self.cfg, int(seq_len)
        st = self.runtime.state(self.kv, 0)
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
                layer["compressed_kv"] = _compressed_rows(c.ckv, T)
                layer["idx_keys"] = _compressed_rows(c.index_k, T)
                times["compressed"] = times.get("compressed", 0.0) + time.perf_counter() - t1
            out[i] = layer
        return {"seq_len": T, "layers": out}


def _compressed_rows(c, length: int) -> torch.Tensor:
    """Natural-order rows ``[length / ratio, width]`` bf16 of metal_2's served compressed cache ``c``
    (``tt/compressor.py`` ``TtServedRows``), read from the owners' slabs up to the last block the prompt reaches."""
    import ttnn

    from models.demos.deepseek_v41_flash_d_p.tt.served import dealloc

    if c.fp8:  # (INDEX_K_SERVED=fp8_row192: decoded on the host by metal_2)
        return c.to_host(length)[: length // c.ratio].to(torch.bfloat16)
    slab = c.slab if c.slab is not None else c.served
    hs = c.stripe // c.ratio  # compressed rows per owner stripe
    need = length // c.ratio
    blocks = min(-(-need // (c.owners * hs)), c.crows // hs)
    rows = blocks * hs
    sl = slab.slice_rows(0, rows)
    try:
        devs = ttnn.get_device_tensors(sl)
        if c.tp_axis is None:  # replicated over TP: mesh column 0's copy of every SP row
            devs = devs[:: c.geo.tp]
        shards = [ttnn.to_torch(t).reshape(rows, slab.width) for t in devs]
    finally:
        dealloc(sl)
    nat = torch.stack([s.reshape(blocks, hs, slab.width) for s in shards], dim=1).reshape(-1, slab.width)
    return nat[:need].to(torch.bfloat16)


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
