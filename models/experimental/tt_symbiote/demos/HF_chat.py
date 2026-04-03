# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Interactive chatbot for HF models with TTNN backend."""

import argparse
import os
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path

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
from models.experimental.tt_symbiote.models.bailing_moe_v2 import TTNNBailingMoeV2Model

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
    """Decode-time logits controls (HF processors / warpers / ``ttnn.sampling`` top_k cap ≤ 32)."""

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


def _sample_next_id_from_tt_logits(mesh_device, tt_logits, params: DecodeParams) -> int:
    """``lm_head`` logits on device (e.g. ``[1,1,V]`` TILE after AllReduced); sampling without uploading the vocab row."""
    orig_v = int(tt_logits.shape[-1])
    tt_rm = ttnn.to_layout(tt_logits, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_2d = ttnn.reshape(tt_rm, (1, orig_v))
    return _ttnn_sampling_from_row_bf16(mesh_device, tt_2d, orig_v, params)


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
    return ttnn.from_torch(
        torch.tensor([[token_id]], dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=_mesh_replicate_mapper(mesh_device),
    )


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


def _pick_next_token_device_logits(tt_logits, logits_processor, input_ids, mesh_device):
    """Greedy from on-device full-vocab logits (AllReduced lm_head); processors optional (CPU path if set)."""
    out_mapper = _mesh_replicate_mapper(mesh_device)

    if not logits_processor:
        tt_idx = ttnn.argmax(tt_logits, dim=-1, keepdim=False)
        tt_next = ttnn.typecast(tt_idx, ttnn.int32)
        ttnn.deallocate(tt_idx)
        next_id = int(_ttnn_to_torch_mesh(tt_next, mesh_device).reshape(-1)[0].item())
        return tt_next, next_id

    logits_cpu = _ttnn_to_torch_mesh(tt_logits, mesh_device).float()
    row = logits_cpu[:, -1, :]
    scores = logits_processor(input_ids, row)
    next_id = _greedy_id_torch(scores)
    tt_next = ttnn.from_torch(
        torch.tensor([[next_id]], dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=out_mapper,
    )
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
        next_id = None
        try:
            next_id = _sample_next_id_ttnn_sampling(mesh_device, scores, decode_params, out_mapper)
        except Exception as exc:
            print(
                f"Warning: ttnn.sampling failed ({type(exc).__name__}: {exc}); "
                "using HF warpers + Gumbel–max on mesh."
            )
            scores_w = logits_warper(input_ids, scores)
            try:
                tt_scores, orig_v = _upload_logits_row_to_mesh(mesh_device, scores_w, out_mapper)
                try:
                    next_id = _sample_next_id_gumbel_argmax_uploaded(mesh_device, tt_scores, orig_v)
                finally:
                    ttnn.deallocate(tt_scores)
            except Exception as exc2:
                print(
                    f"Warning: device Gumbel sampling failed ({type(exc2).__name__}: {exc2}); "
                    "using torch.softmax + multinomial."
                )
                probs = torch.softmax(scores_w.float(), dim=-1).clamp(min=1e-12)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                next_id = int(torch.multinomial(probs[0], num_samples=1).item())
        if next_id is None:
            raise RuntimeError("sampling failed without setting next_id")

    return ttnn.from_torch(
        torch.tensor([[next_id]], dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=out_mapper,
    )


def decode_with_logit_postprocess(
    model,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    past_key_values,
    max_new_tokens: int,
    decode_params: DecodeParams,
    mesh_device,
    *,
    prompt_ids_tt: ttnn.Tensor | None = None,
) -> ttnn.Tensor:
    """Autoregressive decode; state is ``input_ids_tt`` on mesh (HF only for ``prepare_inputs_for_generation`` / ``model()``).

    If ``prompt_ids_tt`` is passed, it must match ``input_ids`` shape; the decode loop takes ownership and
    will deallocate it (avoid passing a tensor you still need elsewhere).
    """
    torch_device = input_ids.device
    mesh_mapper = _mesh_replicate_mapper(mesh_device)
    if prompt_ids_tt is not None:
        if tuple(int(i) for i in prompt_ids_tt.shape) != tuple(input_ids.shape):
            raise ValueError("prompt_ids_tt shape must match input_ids")
        input_ids_tt = prompt_ids_tt
    else:
        input_ids_tt = ttnn.from_torch(
            input_ids.detach().cpu().to(torch.int32).contiguous(),
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mesh_mapper,
        )

    if max_new_tokens <= 0:
        return input_ids_tt

    logits_processor = build_logits_postprocess_processors(decode_params)
    logits_warper = build_logits_postprocess_warpers(decode_params)
    do_sample = decode_params.temperature > 0

    model_kwargs: dict = {
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": True,
    }
    cur_len = input_ids.shape[-1]
    model_kwargs = model._get_initial_cache_position(cur_len, torch_device, model_kwargs)

    eos_token_id = model.config.eos_token_id

    for _ in range(max_new_tokens):
        ids_torch = _ttnn_to_torch_mesh(input_ids_tt, mesh_device).long().to(torch_device)
        model_inputs = model.prepare_inputs_for_generation(ids_torch, **model_kwargs)
        outputs = model(**model_inputs, return_dict=True)

        tt_logits_raw = getattr(outputs.logits, "ttnn_tensor", None)
        if tt_logits_raw is not None and not do_sample:
            tt_next, next_id = _pick_next_token_device_logits(tt_logits_raw, logits_processor, ids_torch, mesh_device)
        elif tt_logits_raw is not None and do_sample and not logits_processor:
            next_id = _sample_next_id_from_tt_logits(mesh_device, tt_logits_raw, decode_params)
            tt_next = _tt_tensor_replicated_token_id(mesh_device, next_id)
        else:
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=torch_device)
            tt_next = _pick_next_token_ttnn_after_torch_processors(
                mesh_device,
                ids_torch,
                next_token_logits,
                logits_processor,
                logits_warper,
                do_sample,
                decode_params,
            )
            next_id = int(_ttnn_to_torch_mesh(tt_next, mesh_device).reshape(-1)[0].item())

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

        if _token_is_eos(next_id, eos_token_id):
            break

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
            response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
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
