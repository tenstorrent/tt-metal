# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Interactive chatbot for HF models with TTNN backend."""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Fix import path: ensure project root comes before script directory in sys.path
# This prevents importing the local 'models/' subdirectory instead of the project 'models/' package
script_dir = str(Path(__file__).resolve().parent)
project_root = str(Path(__file__).resolve().parents[3])

# Remove script directory from the beginning of sys.path if present
if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)

# Ensure project root is in sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.logits_process import (
    LogitsProcessorList,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.experimental.tt_symbiote.core.run_config import DispatchManager, TracedRun
from models.experimental.tt_symbiote.modules.activation import TTNNSilu
from models.experimental.tt_symbiote.modules.linear import (
    TTNNLinearIColShardedWRowSharded,
    TTNNLinearIColShardedWAllReduced,
)
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict
from models.experimental.tt_symbiote.modules.attention import (
    PagedAttentionConfig,
    TTNNPagedAttentionKVCache,
)
from models.experimental.tt_symbiote.modules.decoder_layer import TTNNBailingMoEDecoderLayerPadded
from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedRMSNorm
from models.experimental.tt_symbiote.modules.embedding import TTNNBailingPaddedEmbedding, TTNNBailingRotaryEmbedding
from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.models.bailing_moe_v2 import TTNNBailingMoeV2Model

if TYPE_CHECKING:
    from torch import Tensor
else:
    # Runtime: ``Tensor`` is not imported (avoid torch in this module's namespace for tools
    # that evaluate annotations); type checkers still see ``torch.Tensor`` above.
    Tensor = Any  # type: ignore[misc]

# Stop decode on degenerate output (common in traced greedy if logits stall).
_DECODE_MAX_SAME_TOKEN_STREAK = 8
# If the last N generated ids repeat once (e.g. "tokA tokB tokA tokB…"), stop; streak counter misses alternation.
_DECODE_CYCLE_WINDOW = 8

MESH_DEVICE_MAP = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "T3K": (1, 8),
    "TG": (8, 4),
    "P150": (1, 1),
    "P300": (1, 2),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
    "BHGLX": (8, 4),
}


def get_mesh_shape():
    env = os.environ.get("MESH_DEVICE")
    if env and env in MESH_DEVICE_MAP:
        return MESH_DEVICE_MAP[env]
    num_devices = len(ttnn.get_device_ids())
    return (1, num_devices)


def setup_mesh_device():
    mesh_shape = get_mesh_shape()
    fabric_config = ttnn.FabricConfig.FABRIC_1D_RING
    ttnn.set_fabric_config(
        fabric_config,
        ttnn.FabricReliabilityMode.STRICT_INIT,
    )
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*mesh_shape),
        num_command_queues=1,
        trace_region_size=200_000_000,
    )
    print(f"Opened mesh device with {mesh_device.get_num_devices()} devices (shape={mesh_shape})")
    return mesh_device


def cleanup(mesh_device):
    TracedRun.release_all()
    for submesh in mesh_device.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def create_paged_kv_cache(model_config, device, batch_size=1):
    config = PagedAttentionConfig(
        block_size=64,
        max_num_blocks=32,
        batch_size=batch_size,
    )
    return TTNNPagedAttentionKVCache(
        num_layers=model_config.num_hidden_layers,
        num_kv_heads=model_config.num_key_value_heads,
        head_dim=model_config.head_dim,
        config=config,
        device=None,
    ).to_device(device)


def preprocess_generation_inputs(inputs, model_config, paged_cache, max_new_tokens, device):
    """Strip unused fields, enforce prompt length vs model/KV budget, then move tensors to device."""
    out = {k: v for k, v in inputs.items() if k != "token_type_ids"}

    kv_max = paged_cache.config.max_seq_length
    model_max = getattr(model_config, "max_position_embeddings", kv_max)
    max_total = min(model_max, kv_max)
    reserve = max(1, max_new_tokens)
    max_prompt_len = max(1, max_total - reserve)

    input_ids = out["input_ids"]
    seq_len = input_ids.shape[-1]
    if seq_len > max_prompt_len:
        print(
            f"Warning: prompt truncated from {seq_len} to {max_prompt_len} tokens "
            f"(context {max_total}, reserving {reserve} for generation)."
        )
        for key, value in list(out.items()):
            if isinstance(value, torch.Tensor) and value.shape[-1] == seq_len:
                out[key] = value[..., -max_prompt_len:]

    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in out.items()}


@dataclass
class DecodeParams:
    """Decode-time logits controls for ``decode_with_logit_postprocess``.

    With ``temperature > 0``, sampling uses ``ttnn.sampling`` on the last-row logits (device),
    with ``top_k`` (cap 32) and ``top_p`` as supported by that op.  Greedy decode uses
    ``ttnn.argmax``.  Repetition / no-repeat-ngram processors run on a CPU logits row from capture
    then re-upload for ``ttnn.sampling`` when sampling is enabled.
    """

    temperature: float = 0.0
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0


def build_logits_postprocess_processors(params: DecodeParams) -> LogitsProcessorList:
    procs = LogitsProcessorList()
    if params.repetition_penalty != 1.0:
        procs.append(RepetitionPenaltyLogitsProcessor(penalty=params.repetition_penalty))
    if params.no_repeat_ngram_size > 0:
        procs.append(NoRepeatNGramLogitsProcessor(params.no_repeat_ngram_size))
    return procs


def build_logits_postprocess_warpers(params: DecodeParams) -> LogitsProcessorList:
    if params.temperature <= 0:
        return LogitsProcessorList()
    warp = LogitsProcessorList()
    warp.append(TemperatureLogitsWarper(params.temperature))
    if params.top_k > 0:
        warp.append(TopKLogitsWarper(top_k=params.top_k))
    if params.top_p < 1.0:
        warp.append(TopPLogitsWarper(top_p=params.top_p))
    return warp


def _token_is_eos(token_id: int, eos_token_id) -> bool:
    if eos_token_id is None:
        return False
    if isinstance(eos_token_id, (list, tuple)):
        return token_id in eos_token_id
    return token_id == eos_token_id


def _generation_torch_device(model) -> torch.device:
    """HF bookkeeping tensors during symbiote decode; resilient after TTNN replaces many submodules."""
    try:
        return model.device
    except (RuntimeError, ValueError, StopIteration, AttributeError):
        pass
    p = next(model.parameters(), None)
    if p is not None:
        return p.device
    b = next(model.buffers(), None)
    if b is not None:
        return b.device
    return torch.device("cpu")


def _mesh_replicate_mapper(mesh_device):
    if mesh_device is not None and mesh_device.get_num_devices() > 1:
        return ttnn.ReplicateTensorToMesh(mesh_device)
    return None


def _ttnn_to_torch_mesh(tt_tensor, mesh_device) -> torch.Tensor:
    return to_torch_auto_compose(tt_tensor, device=mesh_device)


def _clamp_vocab_id(token_id: int, vocab_size: int) -> int:
    return max(0, min(int(token_id), vocab_size - 1))


def _trim_logits_to_vocab(logits_2d: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Slice padded lm_head output (V_padded ≥ V_real) to the true vocab size.

    TTNNLinearIColShardedWAllReduced pads columns to a tile multiple.  Positions
    [V_real, V_padded) contain uninitialised values that win argmax and
    corrupt the generated sequence.  Slice them away before any pick.
    """
    if vocab_size <= 0 or logits_2d.shape[-1] <= vocab_size:
        return logits_2d
    return logits_2d[..., :vocab_size]


def _resolve_captured_lm_logits_ttnn(captured):
    """Return ``(ttnn.Tensor, mesh_composer|None)`` for lm_head capture.

    Supports :class:`~models.experimental.tt_symbiote.core.tensor.TorchTTNNTensor` (``ttnn_tensor`` set)
    or a raw :class:`ttnn.Tensor` when ``lm_head`` runs with ``_bypass_tensor_wrapping``."""
    if isinstance(captured, ttnn.Tensor):
        return captured, None
    tt = getattr(captured, "ttnn_tensor", None)
    if tt is not None:
        cfg = getattr(captured, "ttnn_distributed_tensor_config", None)
        mc = cfg.mesh_composer if cfg is not None else None
        return tt, mc
    return None, None


def _captured_logits_last_row_to_torch(captured, mesh_device) -> torch.Tensor:
    """Last sequence row of lm_head logits on device, then composed ``to_torch``.

    When ``seq_len > 1``, ``ttnn.slice`` keeps only the last token row before the
    host read, so we transfer ``O(vocab)`` instead of ``O(seq * vocab)`` for logits.
    """
    tt, mesh_composer = _resolve_captured_lm_logits_ttnn(captured)
    if tt is None:
        x = captured.to_torch.float()
        return x.reshape(-1, x.shape[-1])[-1:, :]

    sh = list(tt.shape)
    rank = len(sh)
    tt_row = tt
    if rank >= 2:
        seq_dim = rank - 2
        s = int(sh[seq_dim])
        if s > 1:
            starts = [0] * rank
            ends = list(sh)
            starts[seq_dim] = s - 1
            ends[seq_dim] = s
            tt_row = ttnn.slice(tt, starts, ends)
    try:
        if mesh_composer is not None:
            row = ttnn.to_torch(tt_row, mesh_composer=mesh_composer).float()
        else:
            row = ttnn.to_torch(tt_row).float()
        return row.reshape(-1, row.shape[-1])
    finally:
        if tt_row is not tt:
            ttnn.deallocate(tt_row)


def _lm_head_capture_last_row_bf16_1xv(captured, mesh_device, vocab_sz: int) -> tuple:
    """Shared: last-token logits as ``[1, V]`` bf16 ROW_MAJOR on device (slice, trim, reshape, cast).

    Returns ``(work, to_free, orig_v, tt)`` where ``to_free`` lists tensors to deallocate (not ``tt``).
    """
    tt, _ = _resolve_captured_lm_logits_ttnn(captured)
    if tt is None:
        raise ValueError(
            "lm_head capture must be a ttnn.Tensor or TorchTTNNTensor with ttnn_tensor set (no CPU fallback)"
        )

    to_free: list = []
    cur = tt
    sh = list(cur.shape)
    rank = len(sh)
    if rank >= 2:
        seq_dim = rank - 2
        s = int(sh[seq_dim])
        if s > 1:
            starts = [0] * rank
            ends = list(sh)
            starts[seq_dim] = s - 1
            ends[seq_dim] = s
            cur = ttnn.slice(tt, starts, ends)
            to_free.append(cur)
    rsh = list(cur.shape)
    if vocab_sz > 0 and rsh[-1] > vocab_sz:
        starts = [0] * len(rsh)
        ends = list(rsh)
        ends[-1] = vocab_sz
        cur2 = ttnn.slice(cur, starts, ends)
        if cur is not tt:
            ttnn.deallocate(cur)
            to_free.pop()
        cur = cur2
        to_free.append(cur)
    work = cur
    if work.layout != ttnn.ROW_MAJOR_LAYOUT:
        lay = ttnn.to_layout(work, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if work is not tt:
            ttnn.deallocate(work)
            to_free.pop()
        work = lay
        to_free.append(work)
    v = int(work.shape[-1])
    if len(work.shape) != 2 or int(work.shape[0]) != 1:
        r2 = ttnn.reshape(work, (1, v))
        if work is not tt:
            ttnn.deallocate(work)
            to_free.pop()
        work = r2
        to_free.append(work)
    if work.dtype != ttnn.bfloat16:
        tc = ttnn.typecast(work, ttnn.bfloat16)
        if work is not tt:
            ttnn.deallocate(work)
            to_free.pop()
        work = tc
        to_free.append(work)
    orig_v = int(work.shape[-1])
    return work, to_free, orig_v, tt


def _device_greedy_token_id_from_lm_head_capture(captured, mesh_device, vocab_sz: int) -> int:
    """Greedy next token id with lm_head output kept on TTNN until ``argmax``.

    **Why this exists (vs. :func:`_captured_logits_last_row_to_torch`):** greedy decode only needs
    ``argmax(logits_last_row)``.  Pulling logits via ``ttnn.to_torch`` moves the **entire** vocab row
    to the host each step.  This path slices the last sequence position and trims padded vocab on
    device, runs ``ttnn.argmax``, then reads back **only** the small index tensor (no full-vocab
    logits round-trip).

    Use when HF logits processors are disabled.  Processors need CPU float rows, so they must keep
    using :func:`_captured_logits_last_row_to_torch` — there is **no** silent CPU fallback from this
    function; missing device logits (no ``ttnn.Tensor`` / ``ttnn_tensor``) raises.
    """
    work, to_free, orig_v, tt = _lm_head_capture_last_row_bf16_1xv(captured, mesh_device, vocab_sz)
    tt_idx = None
    try:
        tt_idx = ttnn.argmax(work, dim=-1, keepdim=True)
        next_id = _int_flat0_from_ttnn_tt(tt_idx, mesh_device)
        return _clamp_vocab_id(next_id, orig_v)
    finally:
        if tt_idx is not None:
            ttnn.deallocate(tt_idx)
        for t in reversed(to_free):
            if t is not tt:
                ttnn.deallocate(t)


def _device_sample_token_id_from_lm_head_capture(captured, mesh_device, vocab_sz: int, params: DecodeParams) -> int:
    """Sample next token via ``ttnn.sampling`` on captured lm_head logits (no full-vocab ``to_torch``)."""
    work, to_free, orig_v, tt = _lm_head_capture_last_row_bf16_1xv(captured, mesh_device, vocab_sz)
    try:
        return _ttnn_sampling_from_row_bf16(mesh_device, work, orig_v, params)
    finally:
        for t in reversed(to_free):
            if t is not tt:
                ttnn.deallocate(t)


def _upload_logits_row_to_mesh(mesh_device, scores_2d: torch.Tensor, mesh_mapper):
    """ROW_MAJOR bf16 logits on mesh; vocab padded to 32 with ``-inf`` on device."""
    orig_v = int(scores_2d.shape[-1])
    t = scores_2d.to(dtype=torch.bfloat16).contiguous()
    pad = (32 - (orig_v % 32)) % 32
    tt_scores = ttnn.from_torch(
        t,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=mesh_mapper,
    )
    if pad:
        tt_scores = ttnn.pad(tt_scores, padding=((0, 0), (0, pad)), value=float("-inf"))
    return tt_scores, orig_v


def _sample_next_id_gumbel_argmax_uploaded(mesh_device, tt_scores_row, orig_v: int) -> int:
    """Sample ~``multinomial(softmax(logits))`` using Gumbel–max on device (bf16 ROW logits row on mesh)."""
    seed = secrets.randbelow(1 << 31)
    shp = tuple(int(i) for i in tt_scores_row.shape)
    u = ttnn.rand(
        shp,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        low=1e-6,
        high=1.0 - 1e-6,
        seed=seed,
    )
    lu = ttnn.log(u)
    ttnn.deallocate(u)
    nlogu = ttnn.neg(lu)
    ttnn.deallocate(lu)
    lnl = ttnn.log(nlogu)
    ttnn.deallocate(nlogu)
    gumbel = ttnn.neg(lnl)
    ttnn.deallocate(lnl)
    perturbed = ttnn.add(tt_scores_row, gumbel)
    ttnn.deallocate(gumbel)
    tt_idx = ttnn.argmax(perturbed, dim=-1, keepdim=True)
    idx = int(_ttnn_to_torch_mesh(tt_idx, mesh_device).reshape(-1)[0].item())
    ttnn.deallocate(tt_idx)
    ttnn.deallocate(perturbed)
    return _clamp_vocab_id(idx, orig_v)


def _ttnn_sampling_from_row_bf16(mesh_device, tt_2d, orig_v: int, params: DecodeParams) -> int:
    """Run ``ttnn.sampling`` from ``[1, orig_v]`` bf16 ROW_MAJOR logits **already on device** (no host vocab upload)."""
    pad_w = (-orig_v) % 32
    vp = orig_v + pad_w

    if params.top_k > 0:
        k_val = max(1, min(int(params.top_k), 32))
    else:
        k_val = 32
    p_host = float(params.top_p) if 0.0 < float(params.top_p) < 1.0 else 0.0
    inv_temp = 1.0 / max(float(params.temperature), 1e-6)
    seed = secrets.randbelow(1 << 31)

    if pad_w:
        tt_2d = ttnn.pad(tt_2d, padding=((0, 0), (0, pad_w)), value=float("-inf"))
    tt_111v = ttnn.reshape(tt_2d, (1, 1, 1, vp))
    tt_user_rm = ttnn.repeat(
        tt_111v,
        repeat_dims=(
            1,
            1,
            32,
            1,
        ),
    )
    ttnn.deallocate(tt_111v)
    tt_vals = ttnn.to_layout(tt_user_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(tt_user_rm)

    tt_idx_1d = ttnn.arange(
        0,
        vp,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
    )
    tt_idx111v = ttnn.reshape(tt_idx_1d, (1, 1, 1, vp))
    ttnn.deallocate(tt_idx_1d)
    tt_ind = ttnn.repeat(tt_idx111v, repeat_dims=(1, 1, 32, 1))
    ttnn.deallocate(tt_idx111v)

    tt_k = ttnn.full(
        [32],
        k_val,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
    )
    tt_p = ttnn.full(
        [32],
        p_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
    )
    tt_temp = ttnn.full(
        [32],
        inv_temp,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
    )

    out_tt = None
    try:
        out_tt = ttnn.sampling(tt_vals, tt_ind, k=tt_k, p=tt_p, temp=tt_temp, seed=seed)
        out = _ttnn_to_torch_mesh(out_tt, mesh_device)
        tok = int(out.reshape(-1)[0].item())
    finally:
        for t in (tt_vals, tt_ind, tt_k, tt_p, tt_temp):
            ttnn.deallocate(t)
        if out_tt is not None:
            ttnn.deallocate(out_tt)

    return _clamp_vocab_id(tok, orig_v)


def _sample_next_id_from_tt_logits(mesh_device, tt_logits, params: DecodeParams, vocab_size: int = 0) -> int:
    """``lm_head`` logits on device (e.g. ``[1,1,V_padded]`` after AllReduced); trim padding then sample."""
    logits_cpu = _ttnn_to_torch_mesh(tt_logits, mesh_device).float()
    row = logits_cpu.reshape(-1, logits_cpu.shape[-1])[-1:, :]
    row = _trim_logits_to_vocab(row, vocab_size)
    orig_v = int(row.shape[-1])
    return _sample_next_id_ttnn_sampling(mesh_device, row, params, _mesh_replicate_mapper(mesh_device))


def _sample_next_id_ttnn_sampling(mesh_device, scores_1xV: torch.Tensor, params: DecodeParams, mesh_mapper) -> int:
    """Host ``[1,V]`` logits → device row → ``ttnn.sampling`` (fallback when no on-device logits tensor)."""
    if scores_1xV.dim() != 2 or scores_1xV.shape[0] != 1:
        raise ValueError("scores must be shape [1, vocab]")
    orig_v = int(scores_1xV.shape[-1])
    tt_2d = ttnn.from_torch(
        scores_1xV.to(dtype=torch.bfloat16).contiguous(),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=mesh_mapper,
    )
    return _ttnn_sampling_from_row_bf16(mesh_device, tt_2d, orig_v, params)


def _tt_tensor_replicated_token_id(mesh_device, token_id: int) -> ttnn.Tensor:
    """Single replicated token id on mesh without ``torch.tensor`` (``ttnn.full``)."""
    return ttnn.full(
        (1, 1),
        fill_value=token_id,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _hf_long_tensor_ids_batch1(ids_seq: list[int], device):
    """HF ``prepare_inputs_for_generation`` expects ``torch.LongTensor`` ``[1, seq]`` (batch size 1)."""
    import torch

    return torch.as_tensor([ids_seq], dtype=torch.long, device=device)


def _prompt_ids_to_mesh_tt(input_ids, mesh_device, mesh_mapper) -> ttnn.Tensor:
    import torch

    return ttnn.from_torch(
        input_ids.detach().cpu().to(torch.int32).contiguous(),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=mesh_mapper,
    )


def _greedy_argmax_logits_row(row) -> int:
    """Argmax over vocab from a [1, V] row (Tensor or tensor-like); no ``torch.argmax``."""
    v = row.reshape(-1).tolist()
    return _clamp_vocab_id(int(max(range(len(v)), key=lambda i: v[i])), len(v))


def _hf_outputs_last_token_logits_32(outputs, device):
    """Last-token logits row as float32 on ``device`` (HF fallback path)."""
    import torch

    return outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=device)


def _int_flat0_from_ttnn_tt(tt, mesh_device) -> int:
    """First element of a small TTNN tensor as Python ``int`` (e.g. sampled token id)."""
    x = _ttnn_to_torch_mesh(tt, mesh_device)
    return int(x.reshape(-1)[0].item())


def _greedy_id_ttnn_upload(mesh_device, scores: torch.Tensor, mesh_mapper) -> int:
    tt_scores, orig_v = _upload_logits_row_to_mesh(mesh_device, scores, mesh_mapper)
    try:
        tt_idx = ttnn.argmax(tt_scores, dim=-1, keepdim=True)
        idx = int(_ttnn_to_torch_mesh(tt_idx, mesh_device).reshape(-1)[0].item())
        ttnn.deallocate(tt_idx)
        return _clamp_vocab_id(idx, orig_v)
    finally:
        ttnn.deallocate(tt_scores)


def _greedy_id_torch(scores: torch.Tensor) -> int:
    orig_v = int(scores.shape[-1])
    idx = int(scores.argmax(dim=-1).reshape(-1)[0].item())
    return _clamp_vocab_id(idx, orig_v)


def _pick_next_token_device_logits(tt_logits, logits_processor, input_ids, mesh_device, vocab_size: int = 0):
    """Greedy from on-device full-vocab logits (AllReduced lm_head).

    ``vocab_size`` must be the *real* vocab size so that padded positions produced by the
    column-sharded linear are trimmed before argmax/sampling.
    """
    # Always pull to CPU so we can trim padding and apply processors uniformly.
    logits_cpu = _ttnn_to_torch_mesh(tt_logits, mesh_device).float()
    # Shape may be [1, 1, V] or [1, V]; normalise to [batch, V].
    row = logits_cpu.reshape(-1, logits_cpu.shape[-1])[-1:, :]
    row = _trim_logits_to_vocab(row, vocab_size)

    if not logits_processor:
        next_id = _greedy_id_torch(row)
    else:
        scores = logits_processor(input_ids, row)
        next_id = _greedy_id_torch(scores)

    tt_next = _tt_tensor_replicated_token_id(mesh_device, next_id)
    return tt_next, next_id


def _pick_next_token_ttnn_after_torch_processors(
    mesh_device,
    input_ids,
    next_token_logits_torch: torch.Tensor,
    logits_processor,
    logits_warper,
    do_sample: bool,
    decode_params: DecodeParams,
) -> ttnn.Tensor:
    """Torch logits path: HF processors/warpers run on host.

    When logits stay on host, warped rows are uploaded for sampling: ``ttnn.sampling`` from
    ``DecodeParams`` where supported, otherwise Gumbel–max on mesh (equivalent to host softmax +
    multinomial for warped logits). Decode uses on-device logits when ``outputs.logits.ttnn_tensor``
    exists and ``LogitsProcessorList`` is empty.
    """
    scores = logits_processor(input_ids, next_token_logits_torch)
    out_mapper = _mesh_replicate_mapper(mesh_device)
    multi = mesh_device.get_num_devices() > 1

    if multi:
        if not do_sample:
            next_id = _greedy_id_torch(scores)
        else:
            scores_w = logits_warper(input_ids, scores)
            tt_scores, orig_v = _upload_logits_row_to_mesh(mesh_device, scores_w, out_mapper)
            try:
                next_id = _sample_next_id_gumbel_argmax_uploaded(mesh_device, tt_scores, orig_v)
            finally:
                ttnn.deallocate(tt_scores)
    elif not do_sample:
        try:
            next_id = _greedy_id_ttnn_upload(mesh_device, scores, None)
        except Exception as exc:
            print(f"Warning: ttnn.argmax failed ({type(exc).__name__}: {exc}); using torch.argmax.")
            next_id = _greedy_id_torch(scores)
    else:
        try:
            next_id = _sample_next_id_ttnn_sampling(mesh_device, scores, decode_params, out_mapper)
        except Exception as exc:
            print(
                f"Warning: ttnn.sampling failed ({type(exc).__name__}: {exc}); "
                "using HF warpers + Gumbel–max on mesh."
            )
            scores_w = logits_warper(input_ids, scores)
            tt_scores, orig_v = _upload_logits_row_to_mesh(mesh_device, scores_w, out_mapper)
            try:
                next_id = _sample_next_id_gumbel_argmax_uploaded(mesh_device, tt_scores, orig_v)
            finally:
                ttnn.deallocate(tt_scores)

    return _tt_tensor_replicated_token_id(mesh_device, next_id)


class _LmHeadCapture:
    """Wraps lm_head ``call`` to capture on-device logits and return a dummy.

    The HF CausalLM forward calls ``logits = logits.float()`` right after
    ``self.lm_head(hidden_states)``.  That ``.float()`` triggers
    ``__torch_dispatch__`` → ``_unwrap_to_torch`` → ``ttnn.to_torch()``
    (expensive full-vocab device→host transfer, ~154ms × 128 steps ≈ 19s).

    This wrapper:
    1. Captures on-device logits as ``TorchTTNNTensor`` (``ttnn_tensor`` set) or a raw ``ttnn.Tensor``
       when ``lm_head`` uses ``_bypass_tensor_wrapping``.
    2. Returns a tiny CPU scalar instead so ``.float()`` becomes free.

    ``decode_with_logit_postprocess`` reads ``cap.captured_result``;
    ``outputs.logits`` (the dummy) is never used.
    ``_update_model_kwargs_for_generation`` only touches ``past_key_values``.
    """

    def __init__(self):
        self.captured_result = None
        self._orig_call = None

    def install(self, lm_head):
        self._orig_call = lm_head.call
        cap = self

        def _capturing_call(*args, **kwargs):
            result = cap._orig_call(*args, **kwargs)
            if cap.captured_result is None:
                tt = getattr(result, "ttnn_tensor", None)
                if isinstance(result, ttnn.Tensor):
                    cap.captured_result = result
                    return torch.tensor(0.0)
                if tt is not None:
                    cap.captured_result = result
                    return torch.tensor(0.0)
            return result

        lm_head.call = _capturing_call

    def uninstall(self, lm_head):
        if self._orig_call is not None:
            lm_head.call = self._orig_call


def _causal_lm_forward_pure_ttnn(model, model_inputs: dict, mesh_device) -> CausalLMOutputWithPast:
    """Run Ling/Bailing MoE V2 without HuggingFace ``forward`` on the transformer body.

    The base model uses :meth:`TTNNBailingMoeV2Model.forward_ttnn` (raw ``ttnn.Tensor`` via
    ``_bypass_tensor_wrapping``). ``lm_head`` also uses bypass so logits are raw ``ttnn.Tensor``;
    :class:`_LmHeadCapture` and :func:`_resolve_captured_lm_logits_ttnn` accept that shape.
    """
    base = model.model
    if not isinstance(base, TTNNBailingMoeV2Model):
        raise TypeError(f"_causal_lm_forward_pure_ttnn requires TTNNBailingMoeV2Model, got {type(base).__name__}")

    ids = model_inputs["input_ids"]
    mesh_mapper = _mesh_replicate_mapper(mesh_device)
    input_ids_tt = ttnn.from_torch(
        ids.cpu().to(torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=mesh_mapper,
    )

    attention_mask = model_inputs.get("attention_mask")
    past_key_values = model_inputs.get("past_key_values")
    cache_position = model_inputs.get("cache_position")
    position_ids = model_inputs.get("position_ids")

    cp_ttnn = None
    if cache_position is not None and isinstance(cache_position, torch.Tensor):
        cp_ttnn = ttnn.from_torch(
            cache_position.cpu().to(torch.int32),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mesh_mapper,
        )

    pos_ttnn = None
    if position_ids is not None and isinstance(position_ids, torch.Tensor):
        pos_ttnn = ttnn.from_torch(
            position_ids.cpu().to(torch.int32),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mesh_mapper,
        )

    lm = model.lm_head
    prev_lm_bypass = getattr(lm, "_bypass_tensor_wrapping", False)
    try:
        if isinstance(lm, TTNNModule):
            lm._bypass_tensor_wrapping = True
        base_out = base.forward_ttnn(
            input_ids_tt,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=pos_ttnn,
            cache_position=cp_ttnn,
            use_cache=True,
            return_dict=True,
        )
        logits = lm(base_out.last_hidden_state)
    finally:
        if isinstance(lm, TTNNModule):
            lm._bypass_tensor_wrapping = prev_lm_bypass
        ttnn.deallocate(input_ids_tt)
        if cp_ttnn is not None:
            ttnn.deallocate(cp_ttnn)
        if pos_ttnn is not None:
            ttnn.deallocate(pos_ttnn)

    return CausalLMOutputWithPast(logits=logits, past_key_values=base_out.past_key_values)


def decode_with_logit_postprocess(
    model,
    input_ids: Tensor,
    attention_mask: Tensor,
    past_key_values,
    max_new_tokens: int,
    decode_params: DecodeParams,
    mesh_device,
    *,
    prompt_ids_tt: ttnn.Tensor | None = None,
    decode_pure_ttnn: bool | None = None,
) -> ttnn.Tensor:
    """Autoregressive decode; state is ``input_ids_tt`` on mesh (HF only for ``prepare_inputs_for_generation`` / ``model()``).

    A wrapper on ``lm_head.call`` captures the raw on-device ``ttnn_tensor``
    before HF's ``logits.float()`` triggers the expensive ``_unwrap_to_torch``
    dispatch path.  The wrapper returns a tiny scalar so ``.float()`` is free.
    Each forward must capture on-device logits from ``lm_head`` (``TorchTTNNTensor`` with
    ``ttnn_tensor``, or raw ``ttnn.Tensor`` when ``lm_head`` uses bypass).  If capture fails, this raises — there is no fallback through HF
    ``outputs.logits`` (TTNN-only decode path; see CLAUDE.md symbiote CPU–TTNN roundtrip removal).

    If ``prompt_ids_tt`` is passed, it must match ``input_ids`` shape; the decode loop takes ownership and
    will deallocate it (avoid passing a tensor you still need elsewhere).

    Decode keeps token ids in a Python ``list`` (``ids_seq``) and builds HF ``input_ids`` via
    :func:`_hf_long_tensor_ids_batch1` instead of ``torch.cat`` / mesh round-trips.
    Greedy: ``ttnn.argmax`` on the captured last row (via :func:`_lm_head_capture_last_row_bf16_1xv`).
    ``temperature > 0``: ``ttnn.sampling`` on that same device row (via
    :func:`_device_sample_token_id_from_lm_head_capture`).  With logits processors, logits are
    ``to_torch``'d once for HF processors, then ``ttnn.sampling`` if ``temperature > 0``.

    PyTorch is only used inside small helpers (HF ``prepare_inputs_for_generation`` expects
    integer token tensors); this function body has no ``torch.`` calls.

    ``_TRACE_RUNNING`` is forced True for the whole decode loop: TracedRun replay is unsafe here
    because paged KV and cache state change every step while a recorded trace replays fixed buffer
    traffic — that mismatch yields wrong logits (e.g. repeated punctuation). Normal forward keeps
    output correct and matches the intended perf profile for this path.

    When ``decode_pure_ttnn`` is True (default), Ling-mini uses :func:`_causal_lm_forward_pure_ttnn` so the
    transformer stack runs as raw ``ttnn.Tensor`` (no ``TorchTTNNTensor`` in the body). Set env
    ``SYMBIOTE_DECODE_PURE_TTNN=0`` or pass ``decode_pure_ttnn=False`` to use ``model(**model_inputs)``.
    """
    import models.experimental.tt_symbiote.core.run_config as _run_config

    if decode_pure_ttnn is None:
        decode_pure_ttnn = os.environ.get("SYMBIOTE_DECODE_PURE_TTNN", "1") != "0"

    if int(input_ids.shape[0]) != 1:
        raise ValueError("decode_with_logit_postprocess supports batch size 1 only")
    bookkeeping_device = input_ids.device
    mesh_mapper = _mesh_replicate_mapper(mesh_device)
    if prompt_ids_tt is not None:
        if tuple(int(i) for i in prompt_ids_tt.shape) != tuple(input_ids.shape):
            raise ValueError("prompt_ids_tt shape must match input_ids")
        input_ids_tt = prompt_ids_tt
    else:
        input_ids_tt = _prompt_ids_to_mesh_tt(input_ids, mesh_device, mesh_mapper)

    if max_new_tokens <= 0:
        ttnn.synchronize_device(mesh_device)
        return input_ids_tt

    logits_processor = build_logits_postprocess_processors(decode_params)

    model_kwargs: dict = {
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": True,
    }
    cur_len = input_ids.shape[-1]
    model_kwargs = model._get_initial_cache_position(cur_len, bookkeeping_device, model_kwargs)

    _eos_cfg = model.config.eos_token_id
    _eos_gen = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    _eos_set: set[int] = set()
    for _src in (_eos_cfg, _eos_gen):
        if _src is None:
            continue
        if isinstance(_src, (list, tuple)):
            _eos_set.update(int(x) for x in _src)
        else:
            _eos_set.add(int(_src))
    eos_token_id = sorted(_eos_set) if _eos_set else None
    same_tok_run = 0
    prev_tok: int | None = None
    gen_hist: list[int] = []

    vocab_sz = int(getattr(model.config, "vocab_size", None) or 0)

    _prev_trace_running = _run_config._TRACE_RUNNING
    _run_config._TRACE_RUNNING = True

    cap = _LmHeadCapture()
    cap.install(model.lm_head)

    ids_seq = [int(x) for x in input_ids[0].tolist()]

    try:
        for _ in range(max_new_tokens):
            ids_for_hf = _hf_long_tensor_ids_batch1(ids_seq, bookkeeping_device)
            model_inputs = model.prepare_inputs_for_generation(ids_for_hf, **model_kwargs)
            if decode_pure_ttnn and isinstance(model.model, TTNNBailingMoeV2Model):
                outputs = _causal_lm_forward_pure_ttnn(model, model_inputs, mesh_device)
            else:
                outputs = model(**model_inputs, return_dict=True)

            captured = cap.captured_result
            cap.captured_result = None

            if captured is None:
                raise RuntimeError(
                    "lm_head did not yield a TTNN-backed logits tensor this step (capture empty). "
                    "Decode expects TTNN logits only — fix lm_head / symbiote wiring so "
                    "`TorchTTNNTensor.ttnn_tensor` is set (no HF outputs.logits fallback)."
                )
            if not logits_processor:
                if decode_params.temperature > 0:
                    next_id = _device_sample_token_id_from_lm_head_capture(
                        captured, mesh_device, vocab_sz, decode_params
                    )
                else:
                    next_id = _device_greedy_token_id_from_lm_head_capture(captured, mesh_device, vocab_sz)
            else:
                logits_cpu = _captured_logits_last_row_to_torch(captured, mesh_device)
                row = _trim_logits_to_vocab(logits_cpu, vocab_sz)
                scores = logits_processor(ids_for_hf, row)
                if decode_params.temperature > 0:
                    next_id = _sample_next_id_ttnn_sampling(
                        mesh_device,
                        scores,
                        decode_params,
                        _mesh_replicate_mapper(mesh_device),
                    )
                else:
                    next_id = _greedy_argmax_logits_row(scores)
            tt_next = _tt_tensor_replicated_token_id(mesh_device, next_id)

            model_kwargs = model._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=getattr(model.config, "is_encoder_decoder", False),
            )
            del outputs

            new_ids_tt = ttnn.concat([input_ids_tt, tt_next], dim=-1)
            ttnn.deallocate(input_ids_tt)
            ttnn.deallocate(tt_next)
            input_ids_tt = new_ids_tt

            ids_seq.append(next_id)

            if _token_is_eos(next_id, eos_token_id):
                break

            gen_hist.append(next_id)
            w = _DECODE_CYCLE_WINDOW
            if len(gen_hist) >= 2 * w and gen_hist[-w:] == gen_hist[-2 * w : -w]:
                break

            if prev_tok is not None and next_id == prev_tok:
                same_tok_run += 1
            else:
                same_tok_run = 1
                prev_tok = next_id
            if same_tok_run >= _DECODE_MAX_SAME_TOKEN_STREAK:
                break

    finally:
        _run_config._TRACE_RUNNING = _prev_trace_running
        cap.uninstall(model.lm_head)

    ttnn.synchronize_device(mesh_device)
    return input_ids_tt


def load_model(mesh_device, model_name="inclusionAI/Ling-mini-2.0"):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    nn_to_ttnn = {
        model.model.layers[0].__class__: TTNNBailingMoEDecoderLayerPadded,
        model.model.norm.__class__: TTNNDistributedRMSNorm,
        nn.Embedding: TTNNBailingPaddedEmbedding,
        model.model.rotary_emb.__class__: TTNNBailingRotaryEmbedding,
    }
    nn_to_ttnn2 = {
        nn.Linear: TTNNLinearIColShardedWRowSharded,
        nn.SiLU: TTNNSilu,
    }

    nn_to_ttnn3 = {
        model.model.__class__: TTNNBailingMoeV2Model,
    }

    modules1 = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
    modules2 = register_module_replacement_dict(model, nn_to_ttnn2, model_config=None)
    modules3 = register_module_replacement_dict(model, nn_to_ttnn3, model_config=None)
    type(model).device = property(lambda self: torch.device("cpu"))
    set_device(model, mesh_device)

    if mesh_device.get_num_devices() > 1 and isinstance(model.lm_head, TTNNLinearIColShardedWRowSharded):
        model.lm_head.__class__ = TTNNLinearIColShardedWAllReduced
        print("lm_head: TTNNLinearIColShardedWAllReduced (full vocab on each device after lm_head).")

    all_modules = {**modules1, **modules2}
    print(f"Preprocessing {len(all_modules)} TTNN module weights...")
    for k, v in tqdm(all_modules.items()):
        v.preprocess_weights()
        v.move_weights_to_device()

    model.eval()
    torch.set_grad_enabled(False)
    paged_cache = create_paged_kv_cache(model.config, mesh_device, batch_size=1)
    return model, tokenizer, paged_cache


def warmup(model, _tokenizer, mesh_device, paged_cache, decode_params=None):
    decode_params = decode_params or DecodeParams()
    print("Warming up with zero inputs at seq_len = 256 ...")
    for seq_len in [256, 1024]:
        prompt_tt = ttnn.zeros(
            (1, seq_len),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        mask_tt = ttnn.ones(
            (1, seq_len),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        input_ids = _ttnn_to_torch_mesh(prompt_tt, mesh_device).long()
        attention_mask = _ttnn_to_torch_mesh(mask_tt, mesh_device).long()
        ttnn.deallocate(mask_tt)
        out_tt = decode_with_logit_postprocess(
            model,
            input_ids,
            attention_mask,
            paged_cache,
            max_new_tokens=2,
            decode_params=decode_params,
            mesh_device=mesh_device,
            prompt_ids_tt=prompt_tt,
        )
        ttnn.deallocate(out_tt)
        paged_cache.reset()
        print(f"  seq_len={seq_len} done")
    TracedRun.release_all()
    print("Warmup complete.")


def chat_loop(
    model,
    tokenizer,
    paged_cache,
    mesh_device,
    max_new_tokens=256,
    decode_params=None,
):
    decode_params = decode_params or DecodeParams()
    messages = []
    print("\n--- Ling-mini-2.0 Chatbot ---")
    print("Type 'quit' or 'exit' to stop, '/clear' to reset history.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "/clear":
            messages = []
            paged_cache.reset()
            print("History cleared.\n")
            continue
        if user_input.lower() == "/clear_trace":
            TracedRun.release_all()
            print("Traces cleared.\n")
            continue

        messages.append({"role": "user", "content": user_input})

        torch_dev = _generation_torch_device(model)
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = preprocess_generation_inputs(
            inputs,
            model.config,
            paged_cache,
            max_new_tokens,
            torch_dev,
        )

        # Reset KV cache values in-place (preserves device buffer addresses so
        # decode traces remain valid) and release only prefill traces (different
        # prompt lengths require new prefill captures each turn).
        paged_cache.reset()

        prompt_len = inputs["input_ids"].shape[-1]
        outputs_tt = decode_with_logit_postprocess(
            model,
            inputs["input_ids"],
            inputs["attention_mask"],
            paged_cache,
            max_new_tokens=max_new_tokens,
            decode_params=decode_params,
            mesh_device=mesh_device,
        )
        try:
            outputs = _ttnn_to_torch_mesh(outputs_tt, mesh_device).long().to(torch_dev)
            gen_ids = outputs[0, prompt_len:].tolist()
            response = tokenizer.decode(gen_ids, skip_special_tokens=True)
        finally:
            ttnn.deallocate(outputs_tt)
        print(f"\nAssistant: {response}\n")

        messages.append({"role": "assistant", "content": response})


def main():
    parser = argparse.ArgumentParser(description="HF Chatbot with TTNN acceleration")
    parser.add_argument("--model", default="inclusionAI/Ling-mini-2.0", help="HuggingFace model name")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate per turn")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Logits temperature; 0=greedy, >0 enables sampling (top-p/top-k apply)",
    )
    parser.add_argument("--top-p", type=float, default=0.95, dest="top_p")
    parser.add_argument("--top-k", type=int, default=50, dest="top_k")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        dest="repetition_penalty",
        help=">1.0 discourages repeating tokens (1.0 disables)",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=0,
        dest="no_repeat_ngram_size",
        help="If >0, blocks repeating n-grams of this size",
    )
    args = parser.parse_args()
    decode_params = DecodeParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    DispatchManager.DisableTiming()  # Disable timing during interactive chat
    mesh_device = setup_mesh_device()
    try:
        model, tokenizer, paged_cache = load_model(mesh_device, args.model)
        warmup(model, tokenizer, mesh_device, paged_cache, decode_params)
        chat_loop(model, tokenizer, paged_cache, mesh_device, args.max_new_tokens, decode_params)
    finally:
        cleanup(mesh_device)


if __name__ == "__main__":
    main()
