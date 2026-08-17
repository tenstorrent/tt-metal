# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Shared test harness: builds a matched (HF layer, TTNN layer) pair and compares them.

torch lives here and in ``tt/reference.py`` only — it is the explicit test boundary
(input construction, weight loading, golden computation, PCC). The TTNN layer's
``prefill_forward`` / ``decode_forward`` never touch it.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tt import reference as ref
from models.autoports.qwen_qwen3_6_35b_a3b.tt.functional_decoder import DecoderConfig, FunctionalDecoder

KIND_TO_LAYER_IDX = {
    "linear": ref.LINEAR_ATTENTION_LAYER_IDX,
    "full": ref.FULL_ATTENTION_LAYER_IDX,
}

ARTIFACT_DIR = ref.AUTOPORT_DIR / "doc" / "functional_decoder"


# =======================================================================================
# pair construction
# =======================================================================================
@dataclass
class LayerPair:
    device: Any
    kind: str
    layer_idx: int
    hf_config: Any
    tt: FunctionalDecoder
    hf: Any
    state_dict: dict
    page_table: Any = None
    page_table_torch: torch.Tensor = None
    weights_source: str = "synthetic"

    @property
    def cfg(self) -> DecoderConfig:
        return self.tt.cfg

    def close(self):
        self.tt = None
        self.hf = None


def build_layer_pair(
    device,
    *,
    kind: str,
    max_batch_size: int = 1,
    supported_context: int = 4096,
    real_weights: bool = False,
    seed: int = 0,
    build_hf: bool = True,
    **cfg_kwargs,
) -> LayerPair:
    """Construct the TTNN layer and (optionally) its HF twin from identical weights."""
    layer_idx = KIND_TO_LAYER_IDX[kind]
    hf_config = ref.load_hf_text_config()
    if real_weights:
        state_dict = ref.real_layer_state_dict(layer_idx, dtype=torch.float32)
    else:
        state_dict = ref.synthetic_layer_state_dict(layer_idx, seed=seed, dtype=torch.float32)

    tt_layer = FunctionalDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=layer_idx,
        mesh_device=device,
        max_batch_size=max_batch_size,
        supported_context=supported_context,
        **cfg_kwargs,
    )
    hf_layer = ref.build_hf_layer(hf_config, layer_idx, state_dict) if build_hf else None

    pair = LayerPair(
        device=device,
        kind=kind,
        layer_idx=layer_idx,
        hf_config=hf_config,
        tt=tt_layer,
        hf=hf_layer,
        state_dict=state_dict,
        weights_source="real" if real_weights else "synthetic",
    )
    if kind == "full":
        pair.page_table_torch, pair.page_table = make_page_table(
            device, max_batch_size, supported_context, tt_layer.cfg.block_size, seed=seed
        )
    return pair


def make_page_table(device, batch: int, context: int, block_size: int, seed: int = 0):
    """A deliberately shuffled page table so page indirection is really exercised.

    Every sequence's logical blocks map to a random permutation of the physical block pool,
    so an implementation that ignores the page table (or assumes identity mapping) fails.
    """
    blocks_per_seq = context // block_size
    total = batch * blocks_per_seq
    gen = torch.Generator().manual_seed(9_000 + seed)
    perm = torch.randperm(total, generator=gen).reshape(batch, blocks_per_seq)
    tt_pt = ttnn.from_torch(
        perm.to(torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(device),
    )
    return perm, tt_pt


# =======================================================================================
# tensor plumbing (explicit host boundary)
# =======================================================================================
def to_tt_prefill(device, x: torch.Tensor) -> ttnn.Tensor:
    """``[1, seq, hidden]`` torch -> ``[1, 1, seq, hidden]`` TTNN."""
    return ttnn.from_torch(
        x.reshape(1, 1, x.shape[-2], x.shape[-1]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(device),
    )


def to_tt_decode(device, x: torch.Tensor) -> ttnn.Tensor:
    """``[batch, 1, hidden]`` torch -> ``[1, 1, batch, hidden]`` TTNN."""
    return ttnn.from_torch(
        x.reshape(1, 1, x.shape[0], x.shape[-1]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(device),
    )


def to_tt_positions(device, positions: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        positions.to(torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(device),
    )


def from_tt(x: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(x, mesh_composer=None).float()


# =======================================================================================
# comparison
# =======================================================================================
@dataclass
class CompareResult:
    label: str
    pcc: float
    maxabs: float
    rel_rms: float
    detail: dict = field(default_factory=dict)

    def __str__(self):
        return f"{self.label}: pcc={self.pcc:.6f} maxabs={self.maxabs:.3e} rel_rms={self.rel_rms:.3e}"


def compare(label: str, got: torch.Tensor, want: torch.Tensor, **detail) -> CompareResult:
    got = got.float().reshape(want.shape)
    want = want.float()
    diff = (got - want).abs()
    denom = want.norm()
    return CompareResult(
        label=label,
        pcc=ref.pcc(got, want),
        maxabs=float(diff.max()),
        rel_rms=float((got - want).norm() / denom) if denom > 0 else 0.0,
        detail=detail,
    )


# =======================================================================================
# linear-attention state translation (HF layout <-> TTNN layout)
# =======================================================================================
def hf_conv_state_to_tt(pair: LayerPair, conv_state_hf: torch.Tensor) -> torch.Tensor:
    """HF ``[batch, 8192, 4]`` conv state -> TTNN ``[kernel-1, batch, 16384]``.

    Two layout differences, both by construction (see ``tt/functional_decoder.py``
    ``_prepare_weights``):

    * TTNN duplicates the 16 q/k heads up to 32 (HF's ``repeat_interleave`` folded into the
      projection and conv taps), so its conv channels are
      ``[q x2 (4096) | k x2 (4096) | v (4096) | z (4096)]`` vs HF's ``[q | k | v]`` (8192).
    * TTNN keeps ``kernel-1 = 3`` taps; HF keeps 4 whose oldest is provably dead
      (``tests/test_reference_math.py::test_hf_conv_state_oldest_column_is_dead``), so HF
      columns 1..3 map onto TTNN taps 0..2.

    The z block never reads its stored taps (its conv tap is the identity ``[0,0,0,1]``), so
    it is filled with zeros.
    """
    cfg = pair.cfg
    hk, dk = cfg.linear_num_key_heads, cfg.linear_key_head_dim
    hv, dv = cfg.linear_num_value_heads, cfg.linear_value_head_dim
    rep = cfg.num_v_head_groups
    batch = conv_state_hf.shape[0]
    taps = cfg.conv_kernel - 1

    hf = conv_state_hf[..., -taps:].permute(2, 0, 1)  # [taps, batch, 8192]
    q = hf[..., : hk * dk].reshape(taps, batch, hk, dk).repeat_interleave(rep, dim=2)
    k = hf[..., hk * dk : 2 * hk * dk].reshape(taps, batch, hk, dk).repeat_interleave(rep, dim=2)
    v = hf[..., 2 * hk * dk :]
    z = torch.zeros(taps, batch, hv * dv)
    return torch.cat([q.reshape(taps, batch, -1), k.reshape(taps, batch, -1), v, z], dim=-1)


def tt_conv_state_to_hf(pair: LayerPair, taps: list[torch.Tensor]) -> torch.Tensor:
    """TTNN taps (each ``[batch, 16384]``) -> HF ``[batch, 8192, 4]`` (oldest column zeroed)."""
    cfg = pair.cfg
    hk, dk = cfg.linear_num_key_heads, cfg.linear_key_head_dim
    rep = cfg.num_v_head_groups
    batch = taps[0].shape[0]
    cols = []
    for tap in taps:
        q = tap[:, : hk * dk * rep].reshape(batch, hk, rep, dk)[:, :, 0].reshape(batch, -1)
        k = tap[:, hk * dk * rep : 2 * hk * dk * rep].reshape(batch, hk, rep, dk)[:, :, 0].reshape(batch, -1)
        v = tap[:, 2 * hk * dk * rep : 3 * hk * dk * rep]
        cols.append(torch.cat([q, k, v], dim=-1))
    dead = torch.zeros_like(cols[0])
    return torch.stack([dead] + cols, dim=-1)


def read_tt_linear_state(pair: LayerPair) -> tuple[list[torch.Tensor], torch.Tensor]:
    taps = [from_tt(tap).reshape(-1, pair.cfg.conv_dim) for tap in pair.tt.conv_state]
    return taps, from_tt(pair.tt.recurrent_state)


def seed_tt_linear_state(pair: LayerPair, conv_state_hf: torch.Tensor, recurrent: torch.Tensor) -> None:
    """Write a given (HF-layout) linear-attention state into the TTNN buffers in place."""
    device = pair.device
    tt_taps = hf_conv_state_to_tt(pair, conv_state_hf)
    for j, buf in enumerate(pair.tt.conv_state):
        src = ttnn.from_torch(
            tt_taps[j].reshape(1, 1, -1, pair.cfg.conv_dim),
            dtype=buf.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(device),
        )
        ttnn.copy(src, buf)
        ttnn.deallocate(src)
    src = ttnn.from_torch(
        recurrent,
        dtype=pair.tt.recurrent_state.dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(device),
    )
    ttnn.copy(src, pair.tt.recurrent_state)
    ttnn.deallocate(src)


# =======================================================================================
# paged KV cache seeding
# =======================================================================================
def seed_kv_cache(pair: LayerPair, k: torch.Tensor, v: torch.Tensor, *, user_id: int) -> None:
    """Write ``k``/``v`` ``[1, n_kv, seq, head_dim]`` into the paged cache for ``user_id``.

    Goes through the page table exactly like the device op does, so a long-context decode
    test can be set up without paying for a full prefill.
    """
    cfg = pair.cfg
    block = cfg.block_size
    seq = k.shape[2]
    if seq % block:
        raise ValueError(f"seed_kv_cache needs a multiple of block_size={block}, got {seq}")
    rows = pair.page_table_torch[user_id]
    for name, src in (("keys", k), ("values", v)):
        cache = pair.tt.kv_cache[0 if name == "keys" else 1]
        host = ttnn.to_torch(cache).float()
        for logical in range(seq // block):
            host[int(rows[logical])] = src[0, :, logical * block : (logical + 1) * block, :]
        staged = ttnn.from_torch(
            host,
            dtype=cache.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=pair.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(pair.device),
        )
        ttnn.copy(staged, cache)
        ttnn.deallocate(staged)


def read_kv_cache(pair: LayerPair, *, user_id: int, seq: int) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = pair.cfg
    block = cfg.block_size
    rows = pair.page_table_torch[user_id]
    out = []
    for cache in pair.tt.kv_cache:
        host = ttnn.to_torch(cache).float()
        blocks = [host[int(rows[i])] for i in range(math.ceil(seq / block))]
        out.append(torch.cat(blocks, dim=1)[:, :seq, :].unsqueeze(0))
    return out[0], out[1]


# =======================================================================================
# high-level compare flows
# =======================================================================================
def snapshot_state(pair: LayerPair) -> dict:
    """Host copy of the persistent state, so a traced replay can be rewound."""
    if pair.cfg.is_linear:
        return {
            "conv": [ttnn.to_torch(tap).clone() for tap in pair.tt.conv_state],
            "recurrent": ttnn.to_torch(pair.tt.recurrent_state).clone(),
        }
    return {"kv": [ttnn.to_torch(c).clone() for c in pair.tt.kv_cache]}


def restore_state(pair: LayerPair, snap: dict) -> None:
    """Write a snapshot back in place (buffer addresses preserved -> trace stays valid)."""

    def put(dst, src):
        staged = ttnn.from_torch(
            src,
            dtype=dst.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=pair.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(pair.device),
        )
        ttnn.copy(staged, dst)
        ttnn.deallocate(staged)

    if pair.cfg.is_linear:
        for dst, src in zip(pair.tt.conv_state, snap["conv"]):
            put(dst, src)
        put(pair.tt.recurrent_state, snap["recurrent"])
    else:
        for dst, src in zip(pair.tt.kv_cache, snap["kv"]):
            put(dst, src)


class TracedDecode:
    """Capture ``decode_forward`` into a TTNN trace and replay it.

    The only tensors that change between replays are the pre-allocated activation and
    ``current_pos`` buffers, which is exactly why ``decode_forward`` takes ``current_pos`` as
    a device tensor. Everything else (weights, paged caches, conv/recurrent state, page
    table) keeps its address across replays, so the captured command stream stays valid.
    """

    def __init__(self, pair: LayerPair):
        self.pair = pair
        cfg = pair.cfg
        batch = cfg.max_batch_size
        self.x_dev = to_tt_decode(pair.device, torch.zeros(batch, 1, cfg.hidden_size))
        self.pos_dev = to_tt_positions(pair.device, torch.zeros(batch, dtype=torch.int32))

        # compile every program first: trace capture must not trigger a JIT build
        warm = pair.tt.decode_forward(self.x_dev, current_pos=self.pos_dev, page_table=pair.page_table)
        ttnn.deallocate(warm)
        ttnn.synchronize_device(pair.device)

        self.trace_id = ttnn.begin_trace_capture(pair.device, cq_id=0)
        self.out_dev = pair.tt.decode_forward(self.x_dev, current_pos=self.pos_dev, page_table=pair.page_table)
        ttnn.end_trace_capture(pair.device, self.trace_id, cq_id=0)
        ttnn.synchronize_device(pair.device)

    def run(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cfg = self.pair.cfg
        host_x = ttnn.from_torch(
            tokens.reshape(1, 1, cfg.max_batch_size, cfg.hidden_size),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(self.pair.device),
        )
        host_pos = ttnn.from_torch(
            positions.to(torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(self.pair.device),
        )
        ttnn.copy_host_to_device_tensor(host_x, self.x_dev)
        ttnn.copy_host_to_device_tensor(host_pos, self.pos_dev)
        ttnn.execute_trace(self.pair.device, self.trace_id, cq_id=0, blocking=True)
        return from_tt(self.out_dev).reshape(cfg.max_batch_size, 1, cfg.hidden_size)

    def release(self):
        ttnn.release_trace(self.pair.device, self.trace_id)
        ttnn.deallocate(self.x_dev)
        ttnn.deallocate(self.pos_dev)


def prefill_and_compare(
    pair: LayerPair,
    *,
    seq_len: int,
    user_id: int = 0,
    start_pos: int = 0,
    seed: int = 0,
    hf_cache=None,
    tail: int | None = None,
    hidden_states: torch.Tensor | None = None,
) -> CompareResult:
    """Run prefill on both sides and compare.

    ``tail`` uses the cheap exact long-context reference (``ref.hf_prefill_tail``): the HF
    K/V cache is filled in O(seq) and only the last ``tail`` query positions are run through
    the layer, so those positions still attend to the full context. Only valid for
    ``full_attention`` layers and ``start_pos == 0``.
    """
    cfg = pair.cfg
    if start_pos == 0 and hf_cache is None:
        pair.tt.reset_state()
    x = (
        hidden_states
        if hidden_states is not None
        else ref.synthetic_hidden_states(pair.hf_config, 1, seq_len, seed=seed)
    )
    tt_x = to_tt_prefill(pair.device, x)
    tt_out = pair.tt.prefill_forward(tt_x, user_id=user_id, page_table=pair.page_table, start_pos=start_pos)
    got = from_tt(tt_out)
    ttnn.deallocate(tt_x)
    ttnn.deallocate(tt_out)

    if tail is not None:
        want = ref.hf_prefill_tail(pair.hf, pair.hf_config, x, tail=tail)
        got = got.reshape(1, seq_len, cfg.hidden_size)[:, -tail:]
        label = f"prefill[{pair.kind}] seq={seq_len} tail={tail}"
    else:
        want = ref.hf_prefill(pair.hf, pair.hf_config, x, start_pos=start_pos, cache=hf_cache).output
        label = f"prefill[{pair.kind}] seq={seq_len} start={start_pos} user={user_id}"
    return compare(label, got, want, seq_len=seq_len, start_pos=start_pos, user_id=user_id)


def decode_and_compare(
    pair: LayerPair,
    *,
    prefill_len: int,
    steps: int = 1,
    seed: int = 0,
    user_ids: list[int] | None = None,
) -> list[CompareResult]:
    """Prefill every slot, then run ``steps`` batched decode steps and compare each.

    HF keeps one cache per layer with a single shared length, so the HF twin is run once per
    slot with its own cache; the TTNN side decodes all slots in one batched call. That is the
    point of the test: identical per-slot results out of a batched, paged device call.
    """
    cfg = pair.cfg
    batch = cfg.max_batch_size
    slots = user_ids if user_ids is not None else list(range(batch))
    if len(slots) != batch:
        raise ValueError("decode_and_compare needs one user_id per batch slot")

    pair.tt.reset_state()
    hf_caches = []
    for i, user_id in enumerate(slots):
        x = ref.synthetic_hidden_states(pair.hf_config, 1, prefill_len, seed=seed + 100 * i)
        tt_x = to_tt_prefill(pair.device, x)
        tt_out = pair.tt.prefill_forward(tt_x, user_id=user_id, page_table=pair.page_table)
        ttnn.deallocate(tt_x)
        ttnn.deallocate(tt_out)
        hf_caches.append(ref.hf_prefill(pair.hf, pair.hf_config, x).cache)

    results = []
    for step in range(steps):
        pos = prefill_len + step
        tokens = torch.stack(
            [
                ref.synthetic_hidden_states(pair.hf_config, 1, 1, seed=7000 + 13 * step + i).reshape(-1)
                for i in range(batch)
            ]
        ).reshape(batch, 1, cfg.hidden_size)
        tt_x = to_tt_decode(pair.device, tokens)
        tt_pos = to_tt_positions(pair.device, torch.full((batch,), pos))
        tt_out = pair.tt.decode_forward(tt_x, current_pos=tt_pos, page_table=pair.page_table)
        got = from_tt(tt_out).reshape(batch, 1, cfg.hidden_size)
        ttnn.deallocate(tt_x)
        ttnn.deallocate(tt_pos)
        ttnn.deallocate(tt_out)

        wants = []
        for i in range(batch):
            wants.append(
                ref.hf_decode(
                    pair.hf,
                    pair.hf_config,
                    tokens[i : i + 1],
                    positions=torch.tensor([pos]),
                    cache=hf_caches[i],
                )
            )
        want = torch.cat(wants, dim=0)
        results.append(compare(f"decode[{pair.kind}] pos={pos} batch={batch}", got, want, pos=pos, batch=batch))
    return results


# =======================================================================================
# evidence logging
# =======================================================================================
#: Log names to divert to ``<name>_partial.jsonl``. ``conftest.py`` fills this with the logs the
#: main suite owns when the session is a filtered (``-k`` / ``-m``) run, so a subset run cannot
#: overwrite committed evidence. Deliberately a *set of names* rather than a global suffix:
#: ``long_context.jsonl`` is produced by five filtered runs on purpose
#: (``tests/run_long_context.sh``, one case per process), so it must never be diverted.
PARTIAL_LOGS: set[str] = set()


def log_path(name: str) -> Path:
    suffix = "_partial" if name in PARTIAL_LOGS else ""
    return ARTIFACT_DIR / f"{name}{suffix}.jsonl"


def reset_log(name: str) -> None:
    """Start a fresh provenance log, so one run == one file.

    Two things this has to get right, both learned the hard way:

    * **Per run, not per process.** ``record`` stays append-only because the advertised-context
      evidence comes from five separate pytest processes (``tests/run_long_context.sh``) that all
      accumulate into ``long_context.jsonl``; truncating per process would leave only the last case.
      The main suite resets from the session fixture in ``conftest.py``, the runner scripts delete
      theirs before looping.
    * **Only for a whole-file run of the suite that owns the log.** A filtered run
      (``pytest -k context_contract``) collects a couple of tests that write no PCC rows, so
      truncating would replace 274 rows of committed evidence with nothing. ``conftest.py`` puts the
      main suite's logs in ``PARTIAL_LOGS`` for filtered sessions and those rows land in
      ``*_partial.jsonl``, which ``doc/.gitignore`` keeps out of the commit. Only those names are
      diverted -- ``long_context.jsonl`` is *always* written by filtered runs and must not be.
    """
    path = log_path(name)
    if path.exists():
        path.unlink()


def record(results, name: str, extra: dict | None = None) -> Path:
    """Append PCC/perf rows to a JSONL provenance log under doc/functional_decoder/."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = log_path(name)
    rows = results if isinstance(results, list) else [results]
    with path.open("a") as fh:
        for row in rows:
            payload = (
                {"label": row.label, "pcc": row.pcc, "maxabs": row.maxabs, "rel_rms": row.rel_rms, **row.detail}
                if isinstance(row, CompareResult)
                else dict(row)
            )
            if extra:
                payload.update(extra)
            fh.write(json.dumps(payload) + "\n")
    return path


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))
