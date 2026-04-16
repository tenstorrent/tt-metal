# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Ling-mini (Bailing MoE V2) helpers: autoregressive decode with on-device logits capture and mesh→torch."""

from __future__ import annotations

import os
import secrets
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.logits_process import (
    LogitsProcessorList,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
)

import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.run_config import traced_kv_full_forward_scope
from models.experimental.tt_symbiote.models.bailing_moe_v2 import TTNNBailingMoeV2Model
from models.experimental.tt_symbiote.modules.attention import (
    PagedAttentionConfig,
    TTNNPagedAttentionKVCache,
)

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any  # type: ignore[misc]

_DECODE_MAX_SAME_TOKEN_STREAK = 8
_DECODE_CYCLE_WINDOW = 8

# mesh_composer cache: (mesh_id, num_dev, mesh_shape), (dist_shape, placement_key) -> CppMeshToTensor | None
_LING_MESH_TO_TORCH_COMPOSER_CACHE: "OrderedDict[tuple, object | None]" = OrderedDict()
_LING_MESH_TO_TORCH_COMPOSER_CACHE_MAX = 64


def clear_ling_mesh_to_torch_composer_cache() -> None:
    """Drop cached mesh composers (e.g. after mesh teardown or in tests)."""
    _LING_MESH_TO_TORCH_COMPOSER_CACHE.clear()


def token_ids_list_for_hf_decode(token_ids, tokenizer) -> list[int]:
    """Clamp token ids to tokenizer range so Rust decode does not overflow on bad or out-of-range values."""
    lim = max(0, len(tokenizer) - 1)
    lim = min(lim, 2_147_483_647)
    unk = getattr(tokenizer, "unk_token_id", None)
    fallback = unk if unk is not None and 0 <= int(unk) <= lim else 0
    out: list[int] = []
    for x in token_ids:
        try:
            t = int(x)
        except (TypeError, ValueError, OverflowError):
            out.append(fallback)
            continue
        if t < 0 or t > lim:
            t = fallback
        out.append(t)
    return out


def _ling_extract_topology(tt_tensor: ttnn.Tensor) -> tuple[list[object], list[int]]:
    topology = tt_tensor.tensor_topology()
    return topology.placements(), list(topology.distribution_shape())


def _ling_mesh_cache_key(mesh_device) -> tuple:
    return (id(mesh_device), mesh_device.get_num_devices(), tuple(int(x) for x in mesh_device.shape))


def _ling_placement_key(placements: list) -> tuple:
    keys = []
    for p in placements:
        if isinstance(p, ttnn.PlacementShard):
            keys.append(("shard", int(p.dim)))
        elif isinstance(p, ttnn.PlacementReplicate):
            keys.append(("rep",))
        else:
            keys.append(("other", repr(p)))
    return tuple(keys)


def _ling_infer_mesh_composer(
    tt_tensor: ttnn.Tensor,
    mesh_device,
    placements: list,
    dist_shape: list[int],
) -> object | None:
    """Build a mesh composer for distributed tensors (topology-driven, aligned with auto_compose)."""
    if len(dist_shape) == 0 or (len(dist_shape) == 1 and dist_shape[0] == 1):
        return None

    tensor_device = tt_tensor.device()
    mesh = tensor_device or mesh_device
    if mesh is None:
        mesh = ttnn.GetDefaultDevice()
        if mesh is None:
            raise RuntimeError(
                "Ling mesh→torch: tensor is on host and no mesh_device was passed and no default device is set."
            )

    assert len(dist_shape) == len(placements)

    if len(dist_shape) == 1 and mesh.shape.dims() == 1:
        p0 = placements[0]
        if isinstance(p0, ttnn.PlacementShard):
            composer_cfg = ttnn.MeshComposerConfig(
                dims=[p0.dim],
                mesh_shape_override=ttnn.MeshShape(dist_shape),
            )
            return ttnn.create_mesh_composer(mesh, composer_cfg)
        return None

    dims: list[int] = []
    shape_override: list[int] = []
    for i, p in enumerate(placements):
        if isinstance(p, ttnn.PlacementShard):
            dims.append(p.dim)
            shape_override.append(dist_shape[i])
        else:
            assert isinstance(p, ttnn.PlacementReplicate)
            dims.append(0)
            shape_override.append(1)

    composer_cfg = ttnn.MeshComposerConfig(
        dims=dims,
        mesh_shape_override=ttnn.MeshShape(shape_override),
    )
    return ttnn.create_mesh_composer(mesh, composer_cfg)


def _ling_cached_mesh_composer(tt_tensor: ttnn.Tensor, mesh_device) -> object | None:
    placements, dist_shape = _ling_extract_topology(tt_tensor)
    topo_key = (tuple(int(x) for x in dist_shape), _ling_placement_key(placements))
    cache_key = (_ling_mesh_cache_key(mesh_device), topo_key)
    if cache_key in _LING_MESH_TO_TORCH_COMPOSER_CACHE:
        composer = _LING_MESH_TO_TORCH_COMPOSER_CACHE[cache_key]
        _LING_MESH_TO_TORCH_COMPOSER_CACHE.move_to_end(cache_key)
        return composer
    composer = _ling_infer_mesh_composer(tt_tensor, mesh_device, placements, dist_shape)
    _LING_MESH_TO_TORCH_COMPOSER_CACHE[cache_key] = composer
    _LING_MESH_TO_TORCH_COMPOSER_CACHE.move_to_end(cache_key)
    while len(_LING_MESH_TO_TORCH_COMPOSER_CACHE) > _LING_MESH_TO_TORCH_COMPOSER_CACHE_MAX:
        _LING_MESH_TO_TORCH_COMPOSER_CACHE.popitem(last=False)
    return composer


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
    """Trim prompts to fit context/KV budget and move tensors to the target device."""
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
    """Sampling and logits-processor settings used by ``decode_with_logit_postprocess`` (greedy vs ttnn sampling)."""

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


def _token_is_eos(token_id: int, eos_token_id) -> bool:
    if eos_token_id is None:
        return False
    if isinstance(eos_token_id, (list, tuple)):
        return token_id in eos_token_id
    return token_id == eos_token_id


def generation_torch_device(model) -> torch.device:
    """Pick a CPU-side torch device for HF bookkeeping tensors when the wrapped model has no clear ``.device``."""
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


def replicated_mesh_tt_to_torch(tt_tensor: ttnn.Tensor, mesh_device) -> torch.Tensor:
    """Mesh ``ttnn.Tensor`` to host ``torch.Tensor``, inferring and caching a mesh composer from topology."""
    if mesh_device is None or mesh_device.get_num_devices() <= 1:
        return ttnn.to_torch(tt_tensor)
    mesh_composer = _ling_cached_mesh_composer(tt_tensor, mesh_device)
    if mesh_composer is None:
        return ttnn.to_torch(tt_tensor)
    return ttnn.to_torch(tt_tensor, mesh_composer=mesh_composer)


def _clamp_vocab_id(token_id: int, vocab_size: int) -> int:
    return max(0, min(int(token_id), vocab_size - 1))


def _trim_logits_to_vocab(logits_2d: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Drop padded vocab columns from lm_head output so tile padding cannot win argmax."""
    if vocab_size <= 0 or logits_2d.shape[-1] <= vocab_size:
        return logits_2d
    return logits_2d[..., :vocab_size]


def _resolve_captured_lm_logits_ttnn(captured):
    """Resolve captured lm_head output to ``(ttnn.Tensor, mesh_composer|None)`` for TorchTTNNTensor or raw tensor."""
    if isinstance(captured, ttnn.Tensor):
        return captured, None
    tt = getattr(captured, "ttnn_tensor", None)
    if tt is not None:
        cfg = getattr(captured, "ttnn_distributed_tensor_config", None)
        mc = cfg.mesh_composer if cfg is not None else None
        return tt, mc
    return None, None


def _captured_logits_last_row_to_torch(captured, mesh_device) -> torch.Tensor:
    """Slice last sequence position of captured logits on device, then ``to_torch`` (one vocab row)."""
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


def _lm_head_capture_last_row_bf16_1xv(captured, vocab_sz: int) -> tuple:
    """Prepare last-token logits as ``[1,V]`` bf16 ROW_MAJOR on device; returns ``(work, to_free, orig_v, tt)``."""
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
    """Greedy next token via ``ttnn.argmax`` on device logits (avoids full-vocab host transfer)."""
    work, to_free, orig_v, tt = _lm_head_capture_last_row_bf16_1xv(captured, vocab_sz)
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
    """Sample next token with ``ttnn.sampling`` on captured on-device logits."""
    work, to_free, orig_v, tt = _lm_head_capture_last_row_bf16_1xv(captured, vocab_sz)
    try:
        return _ttnn_sampling_from_row_bf16(mesh_device, work, orig_v, params)
    finally:
        for t in reversed(to_free):
            if t is not tt:
                ttnn.deallocate(t)


def _ttnn_sampling_from_row_bf16(mesh_device, tt_2d, orig_v: int, params: DecodeParams) -> int:
    """``ttnn.sampling`` on a bf16 ROW_MAJOR logits row already resident on device."""
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
        out = replicated_mesh_tt_to_torch(out_tt, mesh_device)
        tok = int(out.reshape(-1)[0].item())
    finally:
        for t in (tt_vals, tt_ind, tt_k, tt_p, tt_temp):
            ttnn.deallocate(t)
        if out_tt is not None:
            ttnn.deallocate(out_tt)

    return _clamp_vocab_id(tok, orig_v)


def _sample_next_id_ttnn_sampling(mesh_device, scores_1xV: torch.Tensor, params: DecodeParams, mesh_mapper) -> int:
    """Upload host ``[1,V]`` logits to device then run ``ttnn.sampling`` (fallback path)."""
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
    """One new token id as a replicated ``ttnn.full`` tensor on the mesh."""
    return ttnn.full(
        (1, 1),
        fill_value=token_id,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _hf_long_tensor_ids_batch1(ids_seq: list[int], device):
    """Build ``[1, seq]`` long tensor for HF ``prepare_inputs_for_generation``."""
    return torch.as_tensor([ids_seq], dtype=torch.long, device=device)


def _prompt_ids_to_mesh_tt(input_ids, mesh_device, mesh_mapper) -> ttnn.Tensor:
    return ttnn.from_torch(
        input_ids.detach().cpu().to(torch.int32).contiguous(),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=mesh_mapper,
    )


def _greedy_argmax_logits_row(row) -> int:
    """Greedy argmax on a small logits row without ``torch.argmax``."""
    v = row.reshape(-1).tolist()
    return _clamp_vocab_id(int(max(range(len(v)), key=lambda i: v[i])), len(v))


def _int_flat0_from_ttnn_tt(tt, mesh_device) -> int:
    """Read scalar int from a small TTNN tensor via mesh ``to_torch``."""
    x = replicated_mesh_tt_to_torch(tt, mesh_device)
    return int(x.reshape(-1)[0].item())


class _LmHeadCapture:
    """Intercept ``lm_head.call`` to stash on-device logits and return a cheap scalar so HF ``.float()`` avoids full-vocab ``to_torch``."""

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
    """Causal LM forward using ``forward_ttnn`` with device-side ids and cache positions for paged KV."""
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
        if past_key_values is not None:
            with traced_kv_full_forward_scope():
                logits = lm(base_out.last_hidden_state)
        else:
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


def prefill_forward_pure_ttnn(model, model_inputs: dict, mesh_device) -> CausalLMOutputWithPast:
    return _causal_lm_forward_pure_ttnn(model, model_inputs, mesh_device)


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
    """Autoregressive decode: grows ``input_ids_tt`` on mesh using captured lm_head logits (greedy or sampling) and paged KV.

    Uses HF only for ``prepare_inputs_for_generation`` / kwargs updates; optional ``_causal_lm_forward_pure_ttnn`` avoids generic ``model()`` when enabled.
    """
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

    cap = _LmHeadCapture()
    cap.install(model.lm_head)

    ids_seq = [int(x) for x in input_ids[0].tolist()]

    try:
        for _ in range(max_new_tokens):
            ids_for_hf = _hf_long_tensor_ids_batch1(ids_seq, bookkeeping_device)
            past = model_kwargs.get("past_key_values")
            next_sequence_length = None
            if model_kwargs.get("use_cache") and past is not None:
                try:
                    sl = past.get_seq_length()
                except TypeError:
                    sl = past.get_seq_length(0)
                if sl > 0:
                    next_sequence_length = 1
            model_inputs = model.prepare_inputs_for_generation(
                ids_for_hf, next_sequence_length=next_sequence_length, **model_kwargs
            )
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
        cap.uninstall(model.lm_head)

    ttnn.synchronize_device(mesh_device)
    return input_ids_tt
