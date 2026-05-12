# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from transformers.models.mistral3.modeling_mistral3 import Mistral3Model

import ttnn
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.lm_head import LMHead
from models.tt_transformers.tt.model_config import ModelArgs

try:
    from tests.scripts.common import get_updated_device_params
except ImportError:  # minimal fallback if tests package not on PYTHONPATH
    get_updated_device_params = lambda p: p  # type: ignore[assignment]

DEFAULT_MODEL_ID = "mistralai/Devstral-Small-2-24B-Instruct-2512"


def text_model_root(multimodal_inner: Mistral3Model):
    lm = multimodal_inner.language_model
    return lm.model if hasattr(lm, "model") else lm


def apply_devstral_hf_trust_patches():
    from models.tt_transformers.tt import model_config as mc

    orig_set = mc.ModelArgs._set_hf_params

    def _set_hf_params_trust(self, checkpoint_dir: str):
        self.trust_remote_code_hf = True
        return orig_set(self, checkpoint_dir)

    mc.ModelArgs._set_hf_params = _set_hf_params_trust  # type: ignore[method-assign]

    def _get_hf_model_cls_devstral_safe(self):
        from transformers import AutoModelForCausalLM
        from transformers.models.auto.modeling_auto import AutoModelForImageTextToText

        if not self.is_multimodal:
            return AutoModelForCausalLM
        if type(self.hf_config) in AutoModelForImageTextToText._model_mapping:
            return AutoModelForImageTextToText
        raise ValueError(
            f"Demo supports multimodal configs in AutoModelForImageTextToText only; got {type(self.hf_config)}"
        )

    mc.ModelArgs.get_hf_model_cls = _get_hf_model_cls_devstral_safe  # type: ignore[method-assign]


def tt_prefill_target_seqlen(seq_len: int, n_kv_heads: int, mesh_cluster_cols: int) -> int:
    """
    Prefill constraints (see ``models/tt_transformers/tt/attention.py`` ``forward_prefill``):

    - ``seq_len % 128 == 0`` (hard assert before QKV).
    - KV fill path: ``(n_kv // mesh_cols) * L // 64`` must be a multiple of the tile size (32), or
      ``interleaved_to_sharded`` fails.
    - When ``L > 1024``, ``wo`` reuses ``[1, L // 1024, 1024, H]``; **L must be a multiple of 1024** or
      the reshape is invalid and the next ``ttnn.linear`` sees the wrong inner dim (e.g. 5120 vs 4096).
    """
    k = n_kv_heads // mesh_cluster_cols
    assert k > 0
    target = seq_len if seq_len % 128 == 0 else seq_len + (128 - (seq_len % 128)) % 128
    for _ in range(32768):
        kv_ok = (k * target // 64) % 32 == 0
        wo_ok = target <= 1024 or (target % 1024 == 0)
        if kv_ok and wo_ok:
            return target
        target += 128
    raise RuntimeError("Could not find L satisfying TT prefill KV + WO chunking constraints.")


def pad_input_ids_and_positions_for_tt_prefill(
    input_ids: torch.LongTensor,
    position_ids: torch.LongTensor,
    pad_token_id: int,
    n_kv_heads: int,
    mesh_cluster_cols: int,
) -> tuple[torch.LongTensor, torch.LongTensor, int]:
    """Pad token ids and 2D position ids to a TT-valid prefill length (KV tile + optional WO chunk)."""
    seq_len = int(input_ids.shape[1])
    target = tt_prefill_target_seqlen(seq_len, n_kv_heads, mesh_cluster_cols)
    if target == seq_len:
        return input_ids, position_ids, seq_len
    pad = target - seq_len
    input_ids_pad = F.pad(input_ids, (0, pad), value=pad_token_id)
    extra = torch.arange(seq_len, target, dtype=position_ids.dtype, device=position_ids.device).unsqueeze(0)
    position_ids_pad = torch.cat([position_ids, extra], dim=1)
    return input_ids_pad, position_ids_pad, seq_len


def open_devstral_demo_mesh(mesh_width: int):
    device_params = {
        "trace_region_size": 30000000,
        "num_command_queues": 1,
    }
    mesh_shape = ttnn.MeshShape(1, mesh_width)
    return ttnn.open_mesh_device(mesh_shape=mesh_shape, **get_updated_device_params(device_params))


def squeeze_tt_hidden_to_bsh(tt_lm_out: ttnn.Tensor, mesh_device, seq_len_keep: int) -> torch.Tensor:
    tt_h = ttnn.to_torch(tt_lm_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
    while tt_h.dim() > 3:
        tt_h = tt_h.squeeze(0)
    if tt_h.dim() == 2:
        tt_h = tt_h.unsqueeze(0)
    return tt_h[:, :seq_len_keep, :]


def eos_token_ids(config, tokenizer=None) -> set[int]:
    """
    Collect EOS ids for stopping TT/HF loops.

    ``Mistral3ForConditionalGeneration`` often leaves ``config.eos_token_id`` unset while
    ``config.text_config.eos_token_id`` (and the tokenizer) still define EOS (e.g. 2). Missing this
    yields an empty set and generation never stops on end-of-sequence, which causes long repetition.
    """
    ids: set[int] = set()

    def _add(eos):
        if eos is None:
            return
        if isinstance(eos, (list, tuple)):
            ids.update(int(x) for x in eos)
        else:
            ids.add(int(eos))

    _add(getattr(config, "eos_token_id", None))
    tc = getattr(config, "text_config", None)
    if tc is not None:
        _add(getattr(tc, "eos_token_id", None))
    gen = getattr(config, "generation_config", None)
    if gen is not None:
        _add(getattr(gen, "eos_token_id", None))
    if tokenizer is not None:
        _add(getattr(tokenizer, "eos_token_id", None))
    return ids


def host_input_ids_to_tt_replicated(mesh_device, input_ids: torch.LongTensor) -> ttnn.Tensor:
    """Upload ``input_ids`` shaped ``[1, L]`` to a replicated ``[1, 1, 1, L]`` uint32 tensor on device."""
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise ValueError(f"Expected input_ids shape [1, L], got {tuple(input_ids.shape)}")
    return ttnn.from_torch(
        input_ids.reshape(1, 1, 1, -1).to(torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def tt_append_uint32_token(ids_tt: ttnn.Tensor, token_id: int, mesh_device) -> ttnn.Tensor:
    """Append one token on device: ``[1,1,1,L]`` + scalar → ``[1,1,1,L+1]``; deallocates ``ids_tt``."""
    single = ttnn.from_torch(
        torch.tensor([[[[int(token_id)]]]], dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out = ttnn.concat([ids_tt, single], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(ids_tt)
    ttnn.deallocate(single)
    return out


def tt_forward_prefill_from_device_ids(
    ids_tt: ttnn.Tensor,
    seq_len_keep: int,
    pad_token_id: int,
    mesh_device,
    tt_model: TtMinistral3Model,
    model_args: ModelArgs,
) -> ttnn.Tensor:
    """Run ``forward_prefill`` from an **unpadded** ``[1,1,1,seq_len]`` id tensor (replicated). Pads on device when required."""
    target = tt_prefill_target_seqlen(seq_len_keep, int(model_args.n_kv_heads), int(model_args.cluster_shape[1]))
    if target > seq_len_keep:
        pad_amt = target - seq_len_keep
        ids_work = ttnn.pad(
            ids_tt,
            ((0, 0), (0, 0), (0, 0), (0, pad_amt)),
            value=int(pad_token_id),
        )
        own_ids_work = True
    else:
        ids_work = ids_tt
        own_ids_work = False

    pos_tt = ttnn.reshape(
        ttnn.arange(
            0,
            target,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
        ),
        (1, 1, 1, target),
    )

    tt_out = tt_model.forward_prefill(ids_work, pos_tt)
    ttnn.deallocate(pos_tt)
    if own_ids_work:
        ttnn.deallocate(ids_work)
    return tt_out


def tt_replicated_ids_to_torch_long(_mesh_device, ids_tt: ttnn.Tensor, length: int) -> torch.LongTensor:
    """Read first ``length`` token ids from replicated ``[1,1,1,>=L]`` tensor to host ``[length]`` int64."""
    if length <= 0:
        return torch.empty(0, dtype=torch.long)
    sub = ttnn.slice(ids_tt, (0, 0, 0, 0), (1, 1, 1, length))
    th = ttnn.to_torch(ttnn.get_device_tensors(sub)[0]).reshape(-1)[:length].to(torch.long)
    ttnn.deallocate(sub)
    return th


def tt_forward_prefill_from_ids(
    input_ids: torch.LongTensor,
    pad_token_id: int,
    mesh_device,
    tt_model: TtMinistral3Model,
    seq_len_keep: int,
    model_args: ModelArgs,
) -> ttnn.Tensor:
    """TT prefill with a **short-lived** host → device copy of ``input_ids`` ``[1,L]`` (unpadded on host)."""
    ids_tt = host_input_ids_to_tt_replicated(mesh_device, input_ids)
    try:
        return tt_forward_prefill_from_device_ids(ids_tt, seq_len_keep, pad_token_id, mesh_device, tt_model, model_args)
    finally:
        ttnn.deallocate(ids_tt)


def tt_prefill_hidden_states_from_ids(
    input_ids: torch.LongTensor,
    pad_token_id: int,
    mesh_device,
    tt_model: TtMinistral3Model,
    seq_len_keep: int,
    model_args: ModelArgs,
) -> torch.Tensor:
    tt_out = tt_forward_prefill_from_ids(input_ids, pad_token_id, mesh_device, tt_model, seq_len_keep, model_args)
    return squeeze_tt_hidden_to_bsh(tt_out, mesh_device, seq_len_keep)


def demo_lm_head_max_columns_per_device(model_args: ModelArgs, cli_cap: int | None = None) -> int:
    """
    ``ModelArgs.max_columns_per_device_lm_head`` can still exceed Wormhole L1 for dram-sharded
    ``ttnn.linear`` (static circular buffers). Grids that already use ~16k caps need a **lower**
    ceiling than 24k—otherwise ``min(default, 24576)`` is a no-op.

    Defaults to **4096** columns per shard unless overridden (more matmul slices, smaller L1).
    """
    default = int(model_args.max_columns_per_device_lm_head)
    if cli_cap is not None:
        cap = max(1024, int(cli_cap))
    else:
        env = os.environ.get("DEVSTRAL2_LM_HEAD_MAX_COLUMNS_PER_DEVICE")
        if env is not None:
            cap = max(1024, int(env.strip()))
        else:
            cap = 4096
    return min(default, cap)


def cpu_lm_head_logits_last_token(
    tt_hidden_prefill_out: ttnn.Tensor,
    last_token_index: int,
    mesh_device,
    weight_vd: torch.Tensor,
    vocab_size: int,
    chunk_v: int = 4096,
) -> torch.Tensor:
    """Chunked ``h @ W.T`` on host; ``weight_vd`` is ``[vocab, dim]`` like HF ``output.weight``."""
    get_last = (last_token_index // 32) * 32
    h_block = ttnn.slice(
        tt_hidden_prefill_out,
        (0, 0, get_last, 0),
        (1, 1, get_last + 32, tt_hidden_prefill_out.shape[-1]),
    )
    h_block = ttnn.to_memory_config(h_block, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    h_t = ttnn.to_torch(h_block, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
    ttnn.deallocate(h_block)
    while h_t.dim() > 3:
        h_t = h_t.squeeze(0)
    r = last_token_index % 32
    if h_t.dim() == 2:
        h_row = h_t[r].contiguous()
    else:
        h_row = h_t[0, r].contiguous()
    h_row = h_row.to(torch.float32)
    W = weight_vd.to(torch.float32)
    if W.ndim != 2:
        raise RuntimeError(f"LM head weight must be 2D, got {tuple(W.shape)}")
    d = int(h_row.shape[0])
    if W.shape[1] == d:
        pass
    elif W.shape[0] == d:
        W = W.T
    else:
        raise RuntimeError(f"LM head weight {tuple(W.shape)} incompatible with hidden dim {d}")
    vs = min(int(vocab_size), int(W.shape[0]))
    parts: list[torch.Tensor] = []
    for v0 in range(0, vs, chunk_v):
        v1 = min(v0 + chunk_v, vs)
        parts.append(h_row @ W[v0:v1].T)
    return torch.cat(parts, dim=-1).unsqueeze(0).contiguous()


def tt_lm_head_logits_block(
    tt_hidden_prefill_out: ttnn.Tensor,
    last_token_index: int,
    model_args: ModelArgs,
    tt_lm_head: LMHead,
) -> ttnn.Tensor:
    """TT LM head on the 32-row block containing ``last_token_index``; returns sharded logits (``SamplingGenerator`` input)."""
    get_last = (last_token_index // 32) * 32
    h_block = ttnn.slice(
        tt_hidden_prefill_out,
        (0, 0, get_last, 0),
        (1, 1, get_last + 32, tt_hidden_prefill_out.shape[-1]),
    )
    lm_head_input_mem_cfg = model_args.get_lm_head_input_mem_config(Mode.PREFILL, None)
    if lm_head_input_mem_cfg.is_sharded():
        h_block = ttnn.interleaved_to_sharded(h_block, lm_head_input_mem_cfg)
    logits = tt_lm_head(h_block)
    return ttnn.to_memory_config(logits, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def tt_lm_head_logits_last_token(
    tt_hidden_prefill_out: ttnn.Tensor,
    last_token_index: int,
    mesh_device,
    model_args: ModelArgs,
    tt_lm_head: LMHead,
) -> torch.Tensor:
    """Run TT LM head on the 32-row prefill block that contains ``last_token_index``; return ``[1, vocab_size]``."""
    logits = tt_lm_head_logits_block(tt_hidden_prefill_out, last_token_index, model_args, tt_lm_head)
    try:
        logits_torch = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
        while logits_torch.dim() > 3:
            logits_torch = logits_torch.squeeze(0)
        r = last_token_index % 32
        if logits_torch.dim() == 2:
            row = logits_torch[r : r + 1, :]
        else:
            row = logits_torch[0, r : r + 1, :]
        vs = int(model_args.vocab_size)
        if row.shape[-1] > vs:
            row = row[..., :vs]
        return row.contiguous()
    finally:
        ttnn.deallocate(logits)


def devstral_supports_on_device_sampling(model_args: ModelArgs, mesh_device) -> bool:
    """Same gate as ``Transformer`` on-device sampling: vocab shard size must be ≤ 64k per split."""
    sampling_splits = model_args.num_devices if list(mesh_device.shape) != [1, 1] else 2
    return model_args.vocab_size // sampling_splits <= 64 * 1024


def tt_sampling_output_token_id(tt_tokens: ttnn.Tensor, batch_slot: int) -> int:
    """Read global token id for one batch row from ``SamplingGenerator.sample`` output (mirrors ``Generator`` prefill path)."""
    flat = ttnn.to_torch(ttnn.get_device_tensors(tt_tokens)[0]).reshape(-1)
    return int(flat[batch_slot].item())


def image_token_placeholder_positions(input_ids_1row: torch.Tensor, image_token_id: int) -> torch.LongTensor:
    """Return indices ``[N]`` along the prompt where ``input_ids == image_token_id`` (single batch row)."""
    if input_ids_1row.ndim != 1:
        raise ValueError(f"Expected 1-D input_ids row, got {tuple(input_ids_1row.shape)}")
    m = input_ids_1row == int(image_token_id)
    return torch.nonzero(m, as_tuple=False).squeeze(-1).to(torch.long)


def tt_multimodal_scatter_index_tt(mesh_device, positions_1d: torch.LongTensor, hidden_dim: int) -> ttnn.Tensor:
    """Expand ``positions_1d [N]`` for ``ttnn.scatter`` along sequence dim: shape ``[1, 1, N, hidden_dim]``."""
    n = int(positions_1d.numel())
    if n < 1:
        raise ValueError("No image_token_id placeholders in prompt input_ids.")
    idx_host = positions_1d.reshape(1, 1, n, 1).expand(1, 1, n, hidden_dim).to(torch.int32).contiguous()
    return ttnn.from_torch(
        idx_host,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def tt_projected_image_rows_to_device(mesh_device, img_rows_bf16: torch.Tensor) -> ttnn.Tensor:
    """``[N, H]`` bf16 → replicated ``[1, 1, N, H]`` TILE."""
    if img_rows_bf16.ndim != 2:
        raise ValueError(f"Expected img_rows [N, H], got {tuple(img_rows_bf16.shape)}")
    tile = img_rows_bf16.unsqueeze(0).unsqueeze(0).contiguous()
    return ttnn.from_torch(
        tile,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def tt_pad_prompt_embeddings_suffix_tt(
    mesh_device, pad_row_1d_bf16: torch.Tensor, pad_n: int, hidden_dim: int
) -> ttnn.Tensor | None:
    """Replicate ``pad_row_1d_bf16 [H]`` into ``[1, 1, pad_n, H]`` (TT concat suffix)."""
    if pad_n <= 0:
        return None
    if pad_row_1d_bf16.ndim != 1 or int(pad_row_1d_bf16.numel()) != hidden_dim:
        raise ValueError(f"pad_row must be [hidden_dim], got {tuple(pad_row_1d_bf16.shape)} vs dim {hidden_dim}")
    blk = pad_row_1d_bf16.unsqueeze(0).expand(pad_n, hidden_dim).contiguous().reshape(1, 1, pad_n, hidden_dim)
    return ttnn.from_torch(
        blk,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def tt_forward_prefill_multimodal_scatter_merge_from_device_ids(
    tt_lm: TtMinistral3Model,
    ids_tt: ttnn.Tensor,
    seq_len_keep: int,
    img_patch_rows_tt: ttnn.Tensor,
    scatter_idx_tt: ttnn.Tensor,
    pad_row_bf16_cpu: torch.Tensor,
    mesh_device,
    model_args: ModelArgs,
) -> ttnn.Tensor:
    """
    TT multimodal prompt: embed ids on device → ``ttnn.scatter`` image rows onto placeholders → concat pad embeddings
    to TT-valid sequence length → ``ttnn.arange`` positions → ``forward_prefill_from_embeddings``.
    """
    hid = tt_lm.embed_tokens(ids_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    hid4 = ttnn.unsqueeze_to_4D(hid)
    hid_work = ttnn.clone(hid4)
    merged = ttnn.scatter(hid_work, dim=2, index=scatter_idx_tt, src=img_patch_rows_tt)
    ttnn.deallocate(hid4)

    target = tt_prefill_target_seqlen(seq_len_keep, int(model_args.n_kv_heads), int(model_args.cluster_shape[1]))
    pad_n = target - seq_len_keep
    H = int(model_args.dim)
    pad_bf = pad_row_bf16_cpu.to(dtype=torch.bfloat16, device="cpu").contiguous().view(H)
    pad_tail = tt_pad_prompt_embeddings_suffix_tt(mesh_device, pad_bf, pad_n, H)

    full_emb = merged
    if pad_tail is not None:
        full_emb = ttnn.concat([merged, pad_tail], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(merged)
        ttnn.deallocate(pad_tail)
    try:
        pos_tt = ttnn.reshape(
            ttnn.arange(
                0,
                target,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
            ),
            (1, 1, 1, target),
        )
        try:
            return tt_lm.forward_prefill_from_embeddings(full_emb, None, pos_tt)
        finally:
            ttnn.deallocate(pos_tt)
    finally:
        ttnn.deallocate(full_emb)


def tt_sequence_last_uint32_token_id(mesh_device, ids_tt: ttnn.Tensor, seq_len: int) -> int:
    """Token id at index ``seq_len - 1`` from replicated ids ``[1,1,1,*]``."""
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    tail = ttnn.slice(ids_tt, (0, 0, 0, seq_len - 1), (1, 1, 1, seq_len))
    v = int(ttnn.to_torch(ttnn.get_device_tensors(tail)[0]).reshape(-1)[0].item())
    ttnn.deallocate(tail)
    return v
