# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from transformers.models.mistral3.modeling_mistral3 import Mistral3Model

import ttnn
from models.common.sampling import SamplingGenerator
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


_tt_prefill_target_seqlen_cache: dict = {}


def tt_prefill_target_seqlen(seq_len: int, n_kv_heads: int, mesh_cluster_cols: int) -> int:
    """Prefill constraints (see ``models/tt_transformers/tt/attention.py`` ``forward_prefill``): - ``seq_len % 128 == 0`` (hard assert before QKV). - KV fill path: ``(n_kv // mesh_cols) * L // 64`` must be a multiple of the tile size (32), or ``interleaved_to_sharded`` fails. - When ``L > 1024``, ``wo`` reuses ``[1, L // 1024, 1024, H]``; **L must be a multiple of 1024** or the reshape is invalid and the next ``ttnn.linear`` sees the wrong inner dim (e.g. 5120 vs 4096)."""
    cache_key = (seq_len, n_kv_heads, mesh_cluster_cols)
    if cache_key in _tt_prefill_target_seqlen_cache:
        return _tt_prefill_target_seqlen_cache[cache_key]
    k = n_kv_heads // mesh_cluster_cols
    assert k > 0
    target = seq_len if seq_len % 128 == 0 else seq_len + (128 - (seq_len % 128)) % 128
    for _ in range(32768):
        kv_ok = (k * target // 64) % 32 == 0
        wo_ok = target <= 1024 or (target % 1024 == 0)
        if kv_ok and wo_ok:
            _tt_prefill_target_seqlen_cache[cache_key] = target
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
    # Multi-chip decode/prefill issue all_gather / reduce_scatter ops (see
    # ``tt_pixtralattn.py`` and ``tt_ministralattn.py``). Those CCL kernels
    # require the fabric to be enabled BEFORE the mesh is opened — otherwise
    # the first call fails with ``Trying to get un-initialized fabric context``.
    # The fabric is a global runtime state, not a per-mesh parameter, so it
    # must be set via ``ttnn.set_fabric_config(...)``; passing it as a kwarg
    # to ``open_mesh_device`` raises ``TypeError`` on this API.
    # Two command queues let CCL and compute be dispatched independently,
    # which is worth ~5-15% in steady-state decode on BH multi-chip.
    if mesh_width > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)

    device_params = {
        "trace_region_size": 90_000_000,
        "num_command_queues": 2 if mesh_width > 1 else 1,
    }
    mesh_shape = ttnn.MeshShape(1, mesh_width)
    return ttnn.open_mesh_device(mesh_shape=mesh_shape, **get_updated_device_params(device_params))


def close_devstral_demo_mesh(mesh_device) -> None:
    """Close ``mesh_device`` and tear down the fabric if it was enabled. The fabric is global runtime state; leaving it enabled across processes / repeated mesh opens leaks resources and breaks subsequent runs. Always pair :func:`open_devstral_demo_mesh` with this helper."""
    try:
        ttnn.close_mesh_device(mesh_device)
    finally:
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            pass


def squeeze_tt_hidden_to_bsh(tt_lm_out: ttnn.Tensor, mesh_device, seq_len_keep: int) -> torch.Tensor:
    tt_h = ttnn.to_torch(tt_lm_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
    while tt_h.dim() > 3:
        tt_h = tt_h.squeeze(0)
    if tt_h.dim() == 2:
        tt_h = tt_h.unsqueeze(0)
    return tt_h[:, :seq_len_keep, :]


def eos_token_ids(config, tokenizer=None) -> set[int]:
    """Collect EOS ids for stopping TT/HF loops. ``Mistral3ForConditionalGeneration`` often leaves ``config.eos_token_id`` unset while ``config.text_config.eos_token_id`` (and the tokenizer) still define EOS (e.g. 2). Missing this yields an empty set and generation never stops on end-of-sequence, which causes long repetition."""
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
    """``ModelArgs.max_columns_per_device_lm_head`` can still exceed Wormhole L1 for dram-sharded ``ttnn.linear`` (static circular buffers). Grids that already use ~16k caps need a **lower** ceiling than 24k—otherwise ``min(default, 24576)`` is a no-op. Defaults to **4096** columns per shard unless overridden (more matmul slices, smaller L1)."""
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
    """Same gate as ``Transformer`` on-device sampling: vocab shard size must be ≤ 64k per split.

    Single-device [1,1] mesh is excluded: the shared TTSampling multi_step_reduction path
    incorrectly splits the indices_tensor to half the input width, causing out-of-bounds reads
    that produce garbage local indices and global token IDs >= vocab_size.  Use host logits
    sampling instead, which clips logits to vocab_size before argmax/multinomial.
    """
    if list(mesh_device.shape) == [1, 1]:
        return False
    sampling_splits = model_args.num_devices
    return model_args.vocab_size // sampling_splits <= 64 * 1024


def tt_sampling_output_token_id(tt_tokens: ttnn.Tensor, batch_slot: int) -> int:
    """Read global token id for one batch row from ``SamplingGenerator.sample`` output (mirrors ``Generator`` prefill path)."""
    flat = ttnn.to_torch(ttnn.get_device_tensors(tt_tokens)[0]).reshape(-1)
    return int(flat[batch_slot].item())


# --------------------------------------------------------------------------- #
# Traced decode helpers
# --------------------------------------------------------------------------- #
#
# These helpers turn the per-step decode work (forward_decode + LM head + sampling)
# into a single ``ttnn`` trace so the device replays one captured set of kernels
# every iteration. The Python loop only updates three small device input buffers
# (token id, position uint32, position int32) via ``ttnn.copy_host_to_device_tensor``.
#
# Layout choice: the captured region intentionally has **no torch fallbacks**.
# Token sampling defaults to :class:`SamplingGenerator` on device. When that is
# unavailable (vocab > 64k per shard, or ``--lm-head-cpu``) the trace captures
# only the parts that remain TT-native, and the host finishes the remainder.


@dataclass
class TtDecodeInputBuffers:
    """Persistent device buffers used as trace inputs for one decode step. All three live in DRAM so trace updates via ``copy_host_to_device_tensor`` are address-stable across iterations."""

    token_ids: ttnn.Tensor  # uint32 [1, 1]
    pos_uint32: ttnn.Tensor  # uint32 [1, 1]   - RoPE table lookup
    pos_int32: ttnn.Tensor  # int32  [1, 1]   - paged_update_cache index


@dataclass
class TtDecodeTraceContext:
    """Captured decode trace plus references to input/output tensors. Depending on what we could keep on device, ``output_tokens`` and / or ``output_logits`` are populated: - On-device sampling: ``output_tokens`` is the sampled-id tensor (populated by a *separate* ``SamplingGenerator`` trace that runs after the decode trace each step) and ``output_logits`` is the persistent logits tensor that both the decode trace and the sampling trace share. - TT LM head + host sampling: ``output_tokens`` is None, ``output_logits`` is the persistent decode-trace logits tensor. - CPU LM head: ``output_tokens`` is None, ``output_logits`` is None, ``output_hidden`` is the post-norm hidden state ``[1, 1, 32, dim]`` (callers run the LM head on host). ``sampling`` is held so :func:`tt_execute_decode_trace` can replay the sampling pipeline immediately after the decode trace when on-device sampling is enabled, and so :func:`tt_release_decode_trace` can tear down the sampling trace too."""

    trace_id: int
    buffers: TtDecodeInputBuffers
    output_tokens: Optional[ttnn.Tensor] = None
    output_logits: Optional[ttnn.Tensor] = None
    output_hidden: Optional[ttnn.Tensor] = None
    sampling: Optional[SamplingGenerator] = None


def tt_alloc_decode_input_buffers(mesh_device) -> TtDecodeInputBuffers:
    """Allocate the three small DRAM buffers consumed by traced decode steps."""
    zero = torch.tensor([[0]], dtype=torch.int32)
    common = dict(
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return TtDecodeInputBuffers(
        token_ids=ttnn.from_torch(zero, dtype=ttnn.uint32, **common),
        pos_uint32=ttnn.from_torch(zero, dtype=ttnn.uint32, **common),
        pos_int32=ttnn.from_torch(zero, dtype=ttnn.int32, **common),
    )


def tt_update_decode_input_buffers(
    mesh_device,
    buffers: TtDecodeInputBuffers,
    token_id: int,
    decode_pos: int,
) -> None:
    """Copy ``(token_id, decode_pos)`` into the pre-allocated decode buffers. Uses ``ttnn.copy_host_to_device_tensor`` so the device-side buffer addresses captured by the trace remain valid (no re-allocation, no torch CPU ops on the trace fast path)."""
    tok_host = ttnn.from_torch(
        torch.tensor([[int(token_id)]], dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    pos_u32_host = ttnn.from_torch(
        torch.tensor([[int(decode_pos)]], dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    pos_i32_host = ttnn.from_torch(
        torch.tensor([[int(decode_pos)]], dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.copy_host_to_device_tensor(tok_host, buffers.token_ids)
    ttnn.copy_host_to_device_tensor(pos_u32_host, buffers.pos_uint32)
    ttnn.copy_host_to_device_tensor(pos_i32_host, buffers.pos_int32)


def _tt_decode_lm_head_logits(
    tt_hidden: ttnn.Tensor,
    model_args: ModelArgs,
    tt_lm_head: LMHead,
) -> ttnn.Tensor:
    """Trace-safe LM head over the decode hidden block ``[1,1,32,dim]`` -> logits ``[1,1,32,V/shard]``. Mirrors :func:`tt_lm_head_logits_block` but skips the prefill-only ``ttnn.slice`` because decode output is already ``[1, 1, 32, dim]``. Uses ``Mode.PREFILL`` for the input mem config: this is what the non-traced decode path in ``demo_model_loading_prompt.py`` uses (the default non-prefetcher PREFILL / DECODE configs are identical in ``ModelArgs``)."""
    h = tt_hidden
    lm_head_input_mem_cfg = model_args.get_lm_head_input_mem_config(Mode.PREFILL, None)
    if lm_head_input_mem_cfg is not None and lm_head_input_mem_cfg.is_sharded():
        h = ttnn.interleaved_to_sharded(h, lm_head_input_mem_cfg)
    logits = tt_lm_head(h)
    return ttnn.to_memory_config(logits, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def tt_capture_decode_trace(
    mesh_device,
    tt_lm,
    model_args: ModelArgs,
    buffers: TtDecodeInputBuffers,
    *,
    tt_lm_head: Optional[LMHead] = None,
    sampling: Optional[SamplingGenerator] = None,
) -> TtDecodeTraceContext:
    """Compile + capture a decode trace. Capture mode is selected from ``tt_lm_head`` / ``sampling``: - ``tt_lm_head`` set, ``sampling`` set: main trace captures ``forward_decode -> lm_head`` (output: persistent DRAM logits) and a *second* trace (captured by :meth:`SamplingGenerator.capture_trace`) consumes those logits to produce sampled token ids. Both traces are replayed back-to-back in :func:`tt_execute_decode_trace`. - ``tt_lm_head`` set, ``sampling`` None: traced region is ``forward_decode -> lm_head`` (output: full DRAM logits, host samples). - Both None: traced region is ``forward_decode`` only (output: hidden states, host runs the CPU LM head and sampling). The caller MUST first populate the three buffers with valid values via :func:`tt_update_decode_input_buffers` (token id 0 + decode pos 0 is fine for warmup), so the warmup compile run sees realistic shapes. Sampling is intentionally captured as a SEPARATE trace (instead of being inlined into the main decode trace). This mirrors how ``tt_transformers`` handles ``enable_internal_trace=True`` sampling-in-trace and avoids the ``TT_FATAL: Writes are not supported during trace capture`` that fires if the in-trace ``ttnn.sampling`` op is the first instantiation of its program (the first call lazily uploads kernel args / runtime tables through the FD command queue, which counts as a host write). :meth:`SamplingGenerator.capture_trace` runs its own pre-trace warmup with the matching ``tt_out_tok``, so by the time the captured sampling trace runs, every program it needs is already in the cache."""
    # ── Warmup (compile decoder + lm_head kernels, allocate persistent CBs). Not part of the trace. ──
    # Sampling is intentionally NOT warmed up here: ``SamplingGenerator.capture_trace``
    # below does its own warmup with the matching ``tt_out_tok`` so the program
    # cache hits the right variant for the in-trace ``ttnn.sampling`` call.
    _warm_hidden = tt_lm.forward_decode_from_device_tensors(buffers.token_ids, buffers.pos_uint32, buffers.pos_int32)
    if tt_lm_head is not None:
        _warm_logits = _tt_decode_lm_head_logits(_warm_hidden, model_args, tt_lm_head)
        ttnn.deallocate(_warm_logits)
    ttnn.deallocate(_warm_hidden)
    ttnn.synchronize_device(mesh_device)

    # ── Capture the decode-only (and lm-head) graph onto the trace stream. ──
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    hidden = tt_lm.forward_decode_from_device_tensors(buffers.token_ids, buffers.pos_uint32, buffers.pos_int32)
    if tt_lm_head is None:
        # CPU LM head: trace only the decoder body. Hidden states stay live for replay.
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        return TtDecodeTraceContext(trace_id=trace_id, buffers=buffers, output_hidden=hidden)

    logits = _tt_decode_lm_head_logits(hidden, model_args, tt_lm_head)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    if sampling is None:
        # Host sampling: keep the logits tensor as the trace output.
        return TtDecodeTraceContext(trace_id=trace_id, buffers=buffers, output_logits=logits)

    # ── On-device sampling: capture a second trace via SamplingGenerator. ──
    # The decode trace above leaves results in the persistent ``logits`` tensor
    # each replay; the sampling trace below consumes that same tensor and writes
    # sampled token ids into ``tokens_out_prealloc``. Pre-allocating the output
    # outside any trace is required because the captured sampling trace must
    # already know the target buffer (passing ``tt_out_tok`` keeps the in-trace
    # ``ttnn.sampling`` op from allocating an INTERLEAVED output through the
    # host-data-uploading path inside trace capture).
    sampling_batch = int(sampling.tt_sampling.max_batch_size)
    tokens_out_prealloc = ttnn.from_torch(
        torch.zeros((1, 1, 1, sampling_batch), dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # ``SamplingGenerator.sample(enable_trace=True)`` only replays the captured
    # trace when ``self.enable_internal_trace`` is True. The caller may have
    # constructed the generator with ``enable_internal_trace=False`` (the
    # default for the devstral demos), so flip it on here so subsequent
    # ``tt_execute_decode_trace`` calls actually hit the trace fast path.
    sampling.enable_internal_trace = True
    sampling.capture_trace(logits, tt_out_tok=tokens_out_prealloc)

    return TtDecodeTraceContext(
        trace_id=trace_id,
        buffers=buffers,
        output_tokens=tokens_out_prealloc,
        output_logits=logits,
        sampling=sampling,
    )


def tt_execute_decode_trace(mesh_device, ctx: TtDecodeTraceContext) -> None:
    """Replay the previously captured decode trace (and, if present, the sampling trace). Non-blocking submission: the device pipelines this iteration's kernels while the host gets to start work for the next iteration. Reads of ``ctx.output_*`` go through ``ttnn.to_torch`` / ``ttnn.get_device_tensors`` which synchronize implicitly when callers actually need the data. When on-device sampling is enabled the sampling pipeline lives in a second trace captured by :class:`SamplingGenerator` (see :func:`tt_capture_decode_trace`). We replay it here right after the main decode trace via ``sampling.sample`` so callers see a single "execute trace" API."""
    ttnn.execute_trace(mesh_device, ctx.trace_id, cq_id=0, blocking=False)
    if ctx.sampling is not None and ctx.output_tokens is not None and ctx.output_logits is not None:
        ctx.sampling.sample(ctx.output_logits, enable_trace=True, tt_out_tok=ctx.output_tokens)


def tt_release_decode_trace(mesh_device, ctx: TtDecodeTraceContext) -> None:
    """Release the trace handle and deallocate the trace-managed output tensors."""
    ttnn.release_trace(mesh_device, ctx.trace_id)
    if ctx.sampling is not None:
        ctx.sampling.reset_trace()
    if ctx.output_tokens is not None:
        ttnn.deallocate(ctx.output_tokens)
    if ctx.output_logits is not None:
        ttnn.deallocate(ctx.output_logits)
    if ctx.output_hidden is not None:
        ttnn.deallocate(ctx.output_hidden)
    ttnn.deallocate(ctx.buffers.token_ids)
    ttnn.deallocate(ctx.buffers.pos_uint32)
    ttnn.deallocate(ctx.buffers.pos_int32)


def tt_read_decode_traced_token(ctx: TtDecodeTraceContext, batch_slot: int = 0) -> int:
    """Read the sampled token id for ``batch_slot`` from a sampling-enabled decode trace."""
    if ctx.output_tokens is None:
        raise RuntimeError(
            "Decode trace was not captured with on-device sampling; use "
            "tt_read_decode_traced_logits / tt_read_decode_traced_hidden instead."
        )
    return tt_sampling_output_token_id(ctx.output_tokens, batch_slot)


def tt_read_decode_traced_logits(
    ctx: TtDecodeTraceContext,
    mesh_device,
    model_args: ModelArgs,
    batch_slot: int = 0,
) -> torch.Tensor:
    """Read full logits row ``[1, vocab_size]`` for ``batch_slot`` from a TT-LM-head trace."""
    if ctx.output_logits is None:
        raise RuntimeError("Decode trace was not captured with TT LM head; cannot read logits.")
    logits_torch = ttnn.to_torch(ctx.output_logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
    while logits_torch.dim() > 3:
        logits_torch = logits_torch.squeeze(0)
    if logits_torch.dim() == 2:
        row = logits_torch[batch_slot : batch_slot + 1, :]
    else:
        row = logits_torch[0, batch_slot : batch_slot + 1, :]
    vs = int(model_args.vocab_size)
    if row.shape[-1] > vs:
        row = row[..., :vs]
    return row.contiguous()


def tt_read_decode_traced_hidden(
    ctx: TtDecodeTraceContext,
    mesh_device,
    batch_slot: int = 0,
) -> ttnn.Tensor:
    """Return a 32-row hidden block clone for callers that want to drive a CPU LM head. The clone is needed because ``output_hidden`` is owned by the trace and gets overwritten on the next ``execute_trace`` call. Caller is responsible for deallocating."""
    if ctx.output_hidden is None:
        raise RuntimeError("Decode trace did not retain hidden states.")
    return ttnn.clone(ctx.output_hidden, memory_config=ttnn.DRAM_MEMORY_CONFIG)


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
    """TT multimodal prompt: embed ids on device → ``ttnn.scatter`` image rows onto placeholders → concat pad embeddings to TT-valid sequence length → ``ttnn.arange`` positions → ``forward_prefill_from_embeddings``."""
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
