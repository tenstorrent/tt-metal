# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Device-logits AR sampling for Hunyuan (``HY_DEVICE_SAMPLING``, default **on**).
#
# Default under device sampling (Instruct ``top_k`` e.g. 1024, or any k ≠ 32):
#   device logits → D2H full V → host ``torch.topk`` → host shortlist multinomial
#   → host token ids. Logs as ``[device_sampling]``.
#
# On-device top-k + sample when the user **explicitly** sets ``HY_TOP_K=32``
# (alias ``HY_TOPK=32``), or ``HY_TTNN_SAMPLING_OP=1``:
#   device logits → local ``ttnn.topk`` → (vocab-parallel) all_gather + offsets
#   → pad W≥64 → ``ttnn.sampling`` → device token ids.
#   ``ttnn.sampling`` hard-caps k to (0, 32]; BH hangs on W=32 (pad to ≥64).
#
# Full-vocab ``ttnn.sampling`` (W≈262144) OOMs with the resident backbone.
# Repetition penalty / stage-force / ratio processors still force host generate.
# Disable device-logits path with ``HY_DEVICE_SAMPLING=0`` (alias ``HY_SAMPLE_DEVICE=0``).

from __future__ import annotations

import os
from typing import Optional

import torch

import ttnn

_TTNN_SAMPLING_MAX_K = 32


def resolve_hy_top_k() -> int | None:
    """Return explicit ``HY_TOP_K`` / ``HY_TOPK`` override, or None if unset."""
    for key in ("HY_TOP_K", "HY_TOPK"):
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip() != "":
            return int(raw)
    return None


def device_sampling_enabled() -> bool:
    """Device-logits AR path — **on by default**.

    ``HY_DEVICE_SAMPLING=0`` (or alias ``HY_SAMPLE_DEVICE=0``) disables it.
    """
    for key in ("HY_DEVICE_SAMPLING", "HY_SAMPLE_DEVICE"):
        if key in os.environ:
            return os.environ.get(key, "1") != "0"
    return True


def ttnn_sampling_op_enabled() -> bool:
    """Use pure ``ttnn.topk`` + ``ttnn.sampling`` (device top-k).

    Enabled when:
      * ``HY_TTNN_SAMPLING_OP=1``, or
      * user explicitly sets ``HY_TOP_K=32`` / ``HY_TOPK=32``.

    Any other explicit top-k (e.g. 1024) or the Instruct default keeps the host
    torch shortlist path under device logits.
    """
    if os.environ.get("HY_TTNN_SAMPLING_OP", "0") == "1":
        return True
    k = resolve_hy_top_k()
    return k is not None and int(k) == _TTNN_SAMPLING_MAX_K


def sampling_padded_vocab(vocab_size: int) -> int:
    """Smallest W ≥ vocab_size with ``W = 32 * 2^n`` (legal for *full-vocab* sampling).

    Documented for the OOM we hit on V=133120→262144. This path does not feed that
    width into ``ttnn.sampling``; it top-k reduces first.
    """
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    tiles = (vocab_size + 31) // 32
    wt = 1 if tiles <= 1 else 1 << (tiles - 1).bit_length()
    return wt * 32


def _sampling_shortlist_width(width: int) -> int:
    """Legal shortlist W for ``ttnn.sampling``: ``32 * 2^n``, and at least 64.

    Wt=1 (W=32) is mathematically allowed by the op docs but hangs on Blackhole;
    pad with ``-inf`` / dummy indices up to 64.
    """
    target = max(int(width), 64)
    tiles = (target + 31) // 32
    wt = 1 << (tiles - 1).bit_length() if tiles > 1 else 1
    if wt < 2:
        wt = 2
    return wt * 32


def _pad_shortlist(vals: ttnn.Tensor, idxs_rm: ttnn.Tensor, target_w: int):
    """Pad values (``-inf``) and ROW_MAJOR indices (0) along W to ``target_w``."""
    cur = int(vals.shape[-1])
    if cur == target_w:
        return vals, idxs_rm
    if cur > target_w:
        raise ValueError(f"shortlist W={cur} exceeds target {target_w}")
    pad = target_w - cur
    vals_p = ttnn.pad(vals, [(0, 0), (0, 0), (0, 0), (0, pad)], value=float("-inf"))
    idxs_p = ttnn.pad(idxs_rm, [(0, 0), (0, 0), (0, 0), (0, pad)], value=0)
    return vals_p, idxs_p


def _clamp_top_k(top_k: int) -> int:
    """Clamp k for ``ttnn.sampling`` only (op hard limit (0, 32])."""
    if top_k is None or top_k < 1:
        return _TTNN_SAMPLING_MAX_K
    return min(int(top_k), _TTNN_SAMPLING_MAX_K)


def _host_top_k(top_k: int, vocab_size: int) -> int:
    """Host shortlist width: use Instruct ``top_k`` as-is (e.g. 1024), not the ttnn cap.

    ``top_k < 1`` means unrestricted → use full vocab (same as HF warper disable).
    """
    v = int(vocab_size)
    if top_k is None or int(top_k) < 1:
        return v
    return min(int(top_k), v)


def _temp_reciprocal(temperature: float) -> float:
    t = float(temperature)
    if t <= 0.0:
        return 1.0
    return 1.0 / t


def _num_devices(device) -> int:
    return device.get_num_devices() if hasattr(device, "get_num_devices") else 1


def _replicate_mapper(device):
    n = _num_devices(device)
    if n <= 1:
        return None
    shape = tuple(device.shape) if hasattr(device, "shape") else (n,)
    if len(shape) == 2:
        return ttnn.ShardTensor2dMesh(device, dims=(None, None), mesh_shape=shape)
    return ttnn.ReplicateTensorToMesh(device)


def _to_sampling_layout(logits_tt: ttnn.Tensor, batch_size: int) -> ttnn.Tensor:
    """``[B, 1, V]`` / ``[B, V]`` / ``[1, 1, B, V]`` → ``[1, 1, B, V]``."""
    shape = list(logits_tt.shape)
    if len(shape) == 4 and shape[0] == 1 and shape[1] == 1 and shape[2] == batch_size:
        return logits_tt
    if len(shape) == 3 and shape[0] == batch_size and shape[1] == 1:
        return ttnn.reshape(logits_tt, (1, 1, batch_size, shape[2]))
    if len(shape) == 2 and shape[0] == batch_size:
        return ttnn.reshape(logits_tt, (1, 1, batch_size, shape[1]))
    raise ValueError(f"unexpected logits shape for sampling layout: {shape} (B={batch_size})")


def _upload_rm(device, host: torch.Tensor, *, dtype, mesh_mapper):
    return ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )


def _upload_tile(device, host: torch.Tensor, *, dtype, mesh_mapper):
    return ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )


def _local_indices_tt(device, batch_size: int, local_v: int, mesh_mapper):
    """Replicated ``[1,1,B,local_V]`` local vocab indices for ``ttnn.topk``."""
    # uint16 is enough when local_V ≤ 65535 (Hunyuan V/4 = 33280).
    idx_dtype = ttnn.uint16 if local_v <= 65535 else ttnn.uint32
    host = (
        torch.arange(local_v, dtype=torch.int32).view(1, 1, 1, local_v).expand(1, 1, batch_size, local_v).contiguous()
    )
    return _upload_tile(device, host, dtype=idx_dtype, mesh_mapper=mesh_mapper)


def _device_index_offsets_tt(device, batch_size: int, k: int, num_shards: int, per_device_v: int, mesh_mapper):
    """Replicated offsets ``[1,1,B,k*num_shards]``: shard ``d`` → ``d * per_device_v``."""
    host = torch.zeros(1, 1, batch_size, k * num_shards, dtype=torch.int32)
    for d in range(num_shards):
        host[:, :, :, d * k : (d + 1) * k] = d * per_device_v
    return _upload_tile(device, host, dtype=ttnn.int32, mesh_mapper=mesh_mapper)


def _all_gather_shortlist(tensor: ttnn.Tensor, device, *, dim: int = 3) -> ttnn.Tensor:
    """Mesh-wide gather of top-k values/indices along the vocab dim."""
    cluster_shape = tuple(device.shape) if hasattr(device, "shape") else None
    # 1D meshes: gather all devices (cluster_axis=None). 2D: gather the full mesh
    # when vocab is sharded with ShardTensorToMesh across all devices (Hunyuan).
    cluster_axis = None
    if cluster_shape is not None and 1 in cluster_shape:
        cluster_axis = None  # 1×N / N×1 — all devices already on the non-unit axis
    kwargs = dict(
        dim=dim,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cluster_axis=cluster_axis,
    )
    try:
        return ttnn.all_gather(tensor, num_links=1, topology=ttnn.Topology.Linear, **kwargs)
    except TypeError:
        return ttnn.all_gather(tensor, **kwargs)


def _read_token_ids(out_tt: ttnn.Tensor, device, batch_size: int) -> torch.Tensor:
    """D2H only the sampled ids ``[B]`` (not full vocab)."""
    n = _num_devices(device)
    if n > 1:
        out = ttnn.to_torch(out_tt, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
        out = out.view(-1)[:batch_size]
    else:
        out = ttnn.to_torch(out_tt).view(-1)[:batch_size]
    return out.long()


def sampling_out_to_token_ids_tt(out_tt: ttnn.Tensor, batch_size: int) -> ttnn.Tensor:
    """``ttnn.sampling`` output ``[1,1,1,B]`` → embed-ready ``[B, 1]`` ROW_MAJOR ids."""
    # Keep the sampling buffer alive until reshape owns a view / new tensor.
    return ttnn.reshape(out_tt, (batch_size, 1))


def _logits_tt_to_host(logits_tt, device, batch_size: int, *, vocab_size: int, vocab_parallel: bool):
    """D2H device logits → host ``[B, V]`` (concat vocab shards when parallel)."""
    n = _num_devices(device)
    if n > 1:
        if vocab_parallel:
            logits = ttnn.to_torch(logits_tt, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=-1))
        else:
            logits = ttnn.to_torch(logits_tt, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
            logits = logits[:batch_size]
    else:
        logits = ttnn.to_torch(logits_tt)
    logits = logits.float().reshape(batch_size, -1)
    if logits.shape[-1] < vocab_size:
        raise ValueError(f"D2H logits width {logits.shape[-1]} < vocab_size {vocab_size}")
    return logits[:, :vocab_size]


def _sample_host_shortlist(
    logits_tt: ttnn.Tensor,
    device,
    *,
    vocab_size: int,
    batch_size: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    seed: Optional[int] = None,
    vocab_parallel: bool = False,
    deallocate_input: bool = True,
) -> torch.Tensor:
    """D2H → host topk (Instruct k, e.g. 1024) → host multinomial on shortlist.

    Matches log lines:
      ``[device_sampling] D2H logits for top-k reduce ...``
      ``[device_sampling] host topk k=...``
      ``[device_sampling] host shortlist sample ...``
      ``[device_sampling] sampled ids=...``
    """
    print(
        f"[device_sampling] D2H logits for top-k reduce B={batch_size} V={vocab_size} "
        f"vocab_parallel={vocab_parallel} ...",
        flush=True,
    )
    logits = _logits_tt_to_host(logits_tt, device, batch_size, vocab_size=vocab_size, vocab_parallel=vocab_parallel)
    if deallocate_input:
        ttnn.deallocate(logits_tt)

    k_val = _host_top_k(top_k, int(logits.shape[-1]))
    print(f"[device_sampling] host topk k={k_val} ...", flush=True)
    top_vals, top_idx = torch.topk(logits, k=k_val, dim=-1)

    temp = float(temperature) if temperature is not None else 1.0
    p_val = float(top_p) if top_p is not None else 1.0
    p_val = min(max(p_val, 0.0), 1.0)
    print(
        f"[device_sampling] host shortlist sample k={k_val} temp={temp} p={p_val} "
        f"(set HY_TOP_K=32 for ttnn.topk+sampling)",
        flush=True,
    )

    if temp <= 0.0:
        # Greedy on the shortlist.
        local = top_vals.argmax(dim=-1, keepdim=True)
    else:
        scores = top_vals / temp
        if p_val < 1.0:
            # Nucleus filter on the shortlist (same shape as HF warper, shortlist-local).
            sorted_logits, sorted_order = torch.sort(scores, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = probs.cumsum(dim=-1)
            remove = cum > p_val
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, torch.finfo(sorted_logits.dtype).min)
            # Unsort back to topk order.
            unsort = torch.empty_like(sorted_order)
            unsort.scatter_(-1, sorted_order, torch.arange(k_val, device=scores.device).expand_as(sorted_order))
            scores = sorted_logits.gather(-1, unsort)
        probs = torch.softmax(scores, dim=-1)
        g = None
        if seed is not None:
            g = torch.Generator(device=probs.device)
            g.manual_seed(int(seed))
        local = torch.multinomial(probs, num_samples=1, generator=g)

    ids = top_idx.gather(-1, local).view(batch_size).long()
    if torch.any(ids >= vocab_size) or torch.any(ids < 0):
        raise RuntimeError(
            f"sampled id outside vocab: min={int(ids.min())} max={int(ids.max())} vocab_size={vocab_size}"
        )
    print(f"[device_sampling] sampled ids={ids.tolist()}", flush=True)
    return ids


def sample_logits_ttnn(
    logits_tt: ttnn.Tensor,
    device,
    *,
    vocab_size: int,
    batch_size: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    seed: Optional[int] = None,
    vocab_parallel: bool = False,
    deallocate_input: bool = True,
    return_device_ids: bool = True,
) -> ttnn.Tensor | torch.Tensor:
    """Sample next-token ids from device logits.

    Default: D2H + host topk/shortlist.
    With ``HY_TOP_K=32`` / ``HY_TOPK=32`` or ``HY_TTNN_SAMPLING_OP=1``: pure
    ``ttnn.topk`` + ``ttnn.sampling``.

    When the ttnn op path is active and ``return_device_ids=True``, returns embed-ready
    ``[B, 1]`` uint32 on device. Host shortlist always returns host ``[B]`` long ids.
    """
    if not ttnn_sampling_op_enabled():
        return _sample_host_shortlist(
            logits_tt,
            device,
            vocab_size=vocab_size,
            batch_size=batch_size,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            vocab_parallel=vocab_parallel,
            deallocate_input=deallocate_input,
        )

    if batch_size < 1 or batch_size > _TTNN_SAMPLING_MAX_K:
        raise ValueError(f"ttnn.sampling supports batch_size in [1, 32], got {batch_size}")

    n = _num_devices(device)
    mesh_mapper = _replicate_mapper(device)
    k_val = _clamp_top_k(top_k)
    temp_val = _temp_reciprocal(temperature)
    p_val = float(top_p)
    if temperature is not None and float(temperature) <= 0.0:
        k_val = 1
        p_val = 0.0
        temp_val = 1.0
    p_val = min(max(p_val, 0.0), 1.0)
    # Always reduce to 32 so shortlist W is sampling-legal (W=32·2^n after gather).
    # User k is applied inside ttnn.sampling via the k tensor.
    topk_k = _TTNN_SAMPLING_MAX_K

    logits_4d = _to_sampling_layout(logits_tt, batch_size)
    local_v = int(logits_4d.shape[-1])
    if vocab_parallel and n > 1:
        if local_v * n != vocab_size:
            raise ValueError(f"vocab-parallel logits width {local_v} * num_devices {n} != vocab_size {vocab_size}")
        num_shards = n
        per_device_v = local_v
    else:
        if local_v != vocab_size:
            raise ValueError(f"logits last dim {local_v} != vocab_size {vocab_size}")
        num_shards = 1
        per_device_v = vocab_size

    # Prefer DRAM for topk/sampling while the backbone is resident (L1 CB clash).
    if logits_4d.memory_config().buffer_type != ttnn.BufferType.DRAM:
        moved = ttnn.to_memory_config(logits_4d, ttnn.DRAM_MEMORY_CONFIG)
        if logits_4d is not logits_tt:
            ttnn.deallocate(logits_4d)
        logits_4d = moved

    print(
        f"[device_sampling] on-device topk k={topk_k} (sample_k={k_val}) local_V={local_v} "
        f"vocab_parallel={vocab_parallel} shards={num_shards} ...",
        flush=True,
    )
    local_idx = _local_indices_tt(device, batch_size, local_v, mesh_mapper)
    top_vals, top_idx = ttnn.topk(
        logits_4d,
        k=topk_k,
        dim=-1,
        indices_tensor=local_idx,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(local_idx)
    # Drop full-vocab buffer once the shortlist exists (DRAM move / reshape).
    if logits_4d is not logits_tt:
        ttnn.deallocate(logits_4d)
    if deallocate_input:
        ttnn.deallocate(logits_tt)
    if num_shards > 1:
        print(f"[device_sampling] all_gather shortlist dim=-1 shards={num_shards} ...", flush=True)
        gathered_vals = _all_gather_shortlist(top_vals, device)
        ttnn.deallocate(top_vals)
        gathered_idx = _all_gather_shortlist(top_idx, device)
        ttnn.deallocate(top_idx)
        top_vals, top_idx = gathered_vals, gathered_idx

        offsets = _device_index_offsets_tt(device, batch_size, topk_k, num_shards, per_device_v, mesh_mapper)
        idx_i32 = ttnn.typecast(top_idx, dtype=ttnn.int32)
        ttnn.deallocate(top_idx)
        top_idx = ttnn.add(offsets, idx_i32, dtype=ttnn.int32)
        ttnn.deallocate(offsets)
        ttnn.deallocate(idx_i32)

    if top_idx.dtype != ttnn.int32 and top_idx.dtype != ttnn.uint32:
        cast = ttnn.typecast(top_idx, dtype=ttnn.int32)
        ttnn.deallocate(top_idx)
        top_idx = cast

    idx_rm = ttnn.untilize(top_idx, use_multicore=True)
    ttnn.deallocate(top_idx)

    sample_w = _sampling_shortlist_width(int(top_vals.shape[-1]))
    if sample_w != int(top_vals.shape[-1]):
        print(
            f"[device_sampling] pad shortlist {int(top_vals.shape[-1])} → {sample_w} " f"(BH Wt=1 hang workaround)",
            flush=True,
        )
        padded_vals, padded_idx = _pad_shortlist(top_vals, idx_rm, sample_w)
        if padded_vals is not top_vals:
            ttnn.deallocate(top_vals)
        if padded_idx is not idx_rm:
            ttnn.deallocate(idx_rm)
        top_vals, idx_rm = padded_vals, padded_idx

    k_tt = _upload_rm(
        device,
        torch.tensor([k_val] * batch_size, dtype=torch.int32),
        dtype=ttnn.uint32,
        mesh_mapper=mesh_mapper,
    )
    p_tt = _upload_rm(
        device,
        torch.tensor([p_val] * batch_size, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        mesh_mapper=mesh_mapper,
    )
    temp_tt = _upload_rm(
        device,
        torch.tensor([temp_val] * batch_size, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        mesh_mapper=mesh_mapper,
    )

    print(
        f"[device_sampling] ttnn.sampling W={sample_w} k={k_val} temp={temp_val:.4f} p={p_val} ...",
        flush=True,
    )
    sample_kwargs = dict(
        input_values_tensor=top_vals,
        input_indices_tensor=idx_rm,
        k=k_tt,
        p=p_tt,
        temp=temp_tt,
    )
    if seed is not None:
        sample_kwargs["seed"] = int(seed)

    out_tt = ttnn.sampling(**sample_kwargs)
    for t in (top_vals, idx_rm, k_tt, p_tt, temp_tt):
        ttnn.deallocate(t)

    token_tt = sampling_out_to_token_ids_tt(out_tt, batch_size)
    if token_tt is not out_tt:
        ttnn.deallocate(out_tt)

    if not return_device_ids:
        ids = _read_token_ids(token_tt, device, batch_size)
        ttnn.deallocate(token_tt)
        if torch.any(ids >= vocab_size) or torch.any(ids < 0):
            raise RuntimeError(
                f"sampled id outside vocab: min={int(ids.min())} max={int(ids.max())} vocab_size={vocab_size}"
            )
        print(f"[device_sampling] sampled ids={ids.tolist()}", flush=True)
        return ids

    print(f"[device_sampling] device token ids shape={list(token_tt.shape)} (no D2H)", flush=True)
    return token_tt


def can_use_device_sampling(
    config,
    *,
    stage_transitions=None,
    logits_processors=None,
) -> bool:
    """True when host-only processors are inactive so device-logits sampling is safe."""
    if not device_sampling_enabled():
        return False
    if not getattr(config, "do_sample", True):
        return False
    if getattr(config, "repetition_penalty", 1.0) != 1.0:
        return False
    if stage_transitions:
        return False
    if logits_processors:
        return False
    return True


def append_token_ids_tt(seq_tt: ttnn.Tensor | None, token_tt: ttnn.Tensor) -> ttnn.Tensor:
    """Concatenate ``token_tt`` ``[B,1]`` onto on-device generated ids ``[B,T]``.

    Always returns a buffer owned by the caller (clones the first token so the
    live ``token_tt`` can still be passed to embed / freed independently).
    """
    if seq_tt is None:
        return ttnn.clone(token_tt)
    out = ttnn.concat([seq_tt, token_tt], dim=1)
    ttnn.deallocate(seq_tt)
    return out


def upload_stop_ids_tt(device, stop_ids: list[int], batch_size: int, mesh_mapper=None):
    """Replicated stop-id table ``[B, n_stops]`` for on-device membership checks."""
    if not stop_ids:
        return None
    mapper = mesh_mapper if mesh_mapper is not None else _replicate_mapper(device)
    host = torch.tensor(list(stop_ids), dtype=torch.int32).view(1, -1).expand(batch_size, -1).contiguous()
    return _upload_rm(device, host, dtype=ttnn.uint32, mesh_mapper=mapper)


def token_hits_stop_tt(token_tt: ttnn.Tensor, stop_ids_tt: ttnn.Tensor, device, batch_size: int) -> list[bool]:
    """On-device ``token in stops`` per batch row; returns host bools (tiny D2H of flags only).

    ``token_tt`` is ``[B, 1]``, ``stop_ids_tt`` is ``[B, n_stops]``.
    """
    n_stops = int(stop_ids_tt.shape[-1])
    tok = token_tt
    if len(list(tok.shape)) != 2 or int(tok.shape[-1]) != 1:
        tok = ttnn.reshape(tok, (batch_size, 1))
    if n_stops == 1:
        tok_rep = tok
        owns_rep = False
    else:
        tok_rep = ttnn.concat([tok] * n_stops, dim=1)
        owns_rep = True
    if tok_rep.dtype != stop_ids_tt.dtype:
        stop_cmp = ttnn.typecast(stop_ids_tt, dtype=tok_rep.dtype)
        owns_stop = True
    else:
        stop_cmp = stop_ids_tt
        owns_stop = False
    hits = ttnn.eq(tok_rep, stop_cmp)
    if owns_rep:
        ttnn.deallocate(tok_rep)
    if owns_stop:
        ttnn.deallocate(stop_cmp)
    n = _num_devices(device)
    if n > 1:
        flags = ttnn.to_torch(hits, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
        flags = flags[:batch_size]
    else:
        flags = ttnn.to_torch(hits)
    flags = flags.view(batch_size, n_stops)
    ttnn.deallocate(hits)
    return [bool(flags[i].any().item()) for i in range(batch_size)]


def materialize_generated_ids(
    generated_tt: ttnn.Tensor | None,
    device,
    batch_size: int,
    *,
    prefix_ids: torch.Tensor,
    stop_set: set[int] | None = None,
) -> tuple[torch.Tensor, list[list[int]]]:
    """One-shot D2H of on-device generated ids → host ``sequences`` + ``new_tokens``.

    Trims each row at the first stop token (inclusive) when ``stop_set`` is given.
    """
    if generated_tt is None:
        return prefix_ids.long(), [[] for _ in range(batch_size)]
    n = _num_devices(device)
    if n > 1:
        gen = ttnn.to_torch(generated_tt, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
        gen = gen[:batch_size]
    else:
        gen = ttnn.to_torch(generated_tt)
    gen = gen.view(batch_size, -1).long()
    new_tokens: list[list[int]] = []
    trimmed_rows = []
    for i in range(batch_size):
        row = gen[i].tolist()
        if stop_set:
            cut = len(row)
            for j, tid in enumerate(row):
                if int(tid) in stop_set:
                    cut = j + 1
                    break
            row = row[:cut]
        new_tokens.append([int(x) for x in row])
        trimmed_rows.append(row)
    max_t = max((len(r) for r in trimmed_rows), default=0)
    if max_t == 0:
        return prefix_ids.long(), new_tokens
    padded = torch.zeros(batch_size, max_t, dtype=torch.long)
    for i, row in enumerate(trimmed_rows):
        if row:
            padded[i, : len(row)] = torch.tensor(row, dtype=torch.long)
    sequences = torch.cat([prefix_ids.long(), padded], dim=1)
    return sequences, new_tokens
