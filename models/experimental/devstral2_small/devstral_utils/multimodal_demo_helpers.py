# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
from transformers.models.mistral3.modeling_mistral3 import Mistral3Model

import ttnn
from models.common.modules.sampling.sampling_1d import Sampling1D
from models.common.utility_functions import nearest_32
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.lm_head import LMHead
from models.tt_transformers.tt.model_config import ModelArgs

try:
    from tests.scripts.common import get_updated_device_params
except ImportError:  # minimal fallback if tests package not on PYTHONPATH

    def get_updated_device_params(p):
        return p


if TYPE_CHECKING:
    from models.experimental.devstral2_small.tt.pipeline.tt_ministral3_model import TtMinistral3Model

DEFAULT_MODEL_ID = "mistralai/Devstral-Small-2-24B-Instruct-2512"
_DEFAULT_ROPE_THETA = 10000.0


def resolve_rope_parameters(config, *, default_theta: float = _DEFAULT_ROPE_THETA) -> dict:
    """Normalize RoPE params for transformers 4.x (flat ``rope_theta``) and 5.x (``rope_parameters`` dict)."""
    if hasattr(config, "standardize_rope_params"):
        config.standardize_rope_params()

    rp = getattr(config, "rope_parameters", None)
    if rp is None and isinstance(config, dict):
        rp = config.get("rope_parameters")
    if not rp:
        rp = {}
    if not isinstance(rp, dict):
        rp = dict(rp)

    if "rope_theta" not in rp:
        legacy = getattr(config, "rope_theta", None)
        if legacy is None and isinstance(config, dict):
            legacy = config.get("rope_theta")
        rp["rope_theta"] = float(legacy if legacy is not None else default_theta)
    if "rope_type" not in rp:
        rp["rope_type"] = rp.get("type", "default")
    return rp


def _inject_flat_rope_params(config_dict: dict) -> None:
    """In-place restore the flat RoPE keys the shared ``ModelArgs`` reads from a transformers-5.x
    config dict. 5.x nests RoPE under ``rope_parameters`` and drops the deprecated ``rope_scaling``
    on serialization, so we re-derive ``rope_theta`` / ``rope_scaling`` / ``original_max_position_embeddings``
    on the text config and ``rope_theta`` on the vision config. Operates on the ``to_dict()`` output
    (not config attributes) so the values survive serialization. Lets the unmodified tt_transformers
    ``ModelArgs`` consume a 5.x Devstral config without editing ``model_config.py``."""
    if not isinstance(config_dict, dict):
        return
    text = config_dict.get("text_config")
    text = text if isinstance(text, dict) else config_dict
    trp = text.get("rope_parameters")
    if isinstance(trp, dict) and trp:
        if trp.get("rope_theta") is not None and text.get("rope_theta") is None:
            text["rope_theta"] = float(trp["rope_theta"])
        if not text.get("rope_scaling"):
            text["rope_scaling"] = dict(trp)
        if (
            trp.get("original_max_position_embeddings") is not None
            and text.get("original_max_position_embeddings") is None
        ):
            text["original_max_position_embeddings"] = trp["original_max_position_embeddings"]
    vis = config_dict.get("vision_config")
    if isinstance(vis, dict):
        vrp = vis.get("rope_parameters")
        if isinstance(vrp, dict) and vrp.get("rope_theta") is not None and vis.get("rope_theta") is None:
            vis["rope_theta"] = float(vrp["rope_theta"])


def text_model_root(multimodal_inner: Mistral3Model):
    lm = multimodal_inner.language_model
    return lm.model if hasattr(lm, "model") else lm


def apply_devstral_hf_trust_patches():
    import transformers

    from models.tt_transformers.tt import model_config as mc

    # transformers 5.x compat: back-fill flat RoPE keys from the nested ``rope_parameters`` dict
    # so the unmodified shared ModelArgs (which reads flat keys) loads a 5.x Devstral config.
    # ModelArgs reads RoPE via ``self.hf_config.to_dict()``, so we wrap the returned config's
    # ``to_dict`` and inject into its output (attribute-level back-fill does not survive 5.x
    # serialization). Keeps this compat shim in the experimental package, not model_config.py.
    if not getattr(transformers.AutoConfig, "_devstral_rope_normalized", False):
        _orig_autoconfig_from_pretrained = transformers.AutoConfig.from_pretrained

        def _from_pretrained_rope_norm(*args, **kwargs):
            cfg = _orig_autoconfig_from_pretrained(*args, **kwargs)
            try:
                _orig_to_dict = cfg.to_dict

                def _to_dict_rope_norm(*a, **k):
                    d = _orig_to_dict(*a, **k)
                    _inject_flat_rope_params(d)
                    return d

                cfg.to_dict = _to_dict_rope_norm
            except Exception:  # never let normalization break config loading
                pass
            return cfg

        transformers.AutoConfig.from_pretrained = _from_pretrained_rope_norm
        transformers.AutoConfig._devstral_rope_normalized = True

    if hasattr(mc.ModelArgs, "_devstral_orig_set_hf_params") and hasattr(
        mc.ModelArgs, "_devstral_orig_get_hf_model_cls"
    ):
        return

    orig_set = mc.ModelArgs._set_hf_params
    orig_get_hf_model_cls = mc.ModelArgs.get_hf_model_cls

    def _set_hf_params_trust(self, checkpoint_dir: str):
        self.trust_remote_code_hf = True
        return orig_set(self, checkpoint_dir)

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

    mc.ModelArgs._set_hf_params = _set_hf_params_trust  # type: ignore[method-assign]
    mc.ModelArgs.get_hf_model_cls = _get_hf_model_cls_devstral_safe  # type: ignore[method-assign]
    mc.ModelArgs._devstral_orig_set_hf_params = orig_set  # type: ignore[attr-defined]
    mc.ModelArgs._devstral_orig_get_hf_model_cls = orig_get_hf_model_cls  # type: ignore[attr-defined]


_tt_prefill_target_seqlen_cache: dict = {}


def tt_prefill_target_seqlen(seq_len: int, n_kv_heads: int, mesh_cluster_cols: int) -> int:
    """Smallest L>=seq_len for TT prefill (128-align, KV tiles, WO chunks when L>1024)."""
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
    # Validate the requested topology up front rather than silently clamping: a clamped run looks
    # successful (sharding/cache/perf all report normally) while quietly executing on fewer devices.
    if mesh_width < 1:
        raise ValueError(f"--mesh-width must be >= 1, got {mesh_width}.")
    num_devices = ttnn.get_num_devices()
    if mesh_width > num_devices:
        raise ValueError(
            f"--mesh-width {mesh_width} requested but only {num_devices} device(s) are visible. "
            f"Reduce --mesh-width or expose more devices."
        )

    # Multi-chip: enable fabric before open; 2 CQs on BH multi-chip decode.
    if mesh_width > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)

    device_params = {
        "trace_region_size": 90_000_000,
        "num_command_queues": 2 if mesh_width > 1 else 1,
    }
    mesh_shape = ttnn.MeshShape(1, mesh_width)
    return ttnn.open_mesh_device(mesh_shape=mesh_shape, **get_updated_device_params(device_params))


def close_devstral_demo_mesh(mesh_device) -> None:
    """Close mesh and disable fabric (pair with :func:`open_devstral_demo_mesh`)."""
    try:
        ttnn.close_mesh_device(mesh_device)
    finally:
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            # Best-effort cleanup: fabric disable may fail in teardown; do not mask close errors.
            pass


def squeeze_tt_hidden_to_bsh(tt_lm_out: ttnn.Tensor, mesh_device, seq_len_keep: int) -> torch.Tensor:
    tt_h = ttnn.to_torch(tt_lm_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
    while tt_h.dim() > 3:
        tt_h = tt_h.squeeze(0)
    if tt_h.dim() == 2:
        tt_h = tt_h.unsqueeze(0)
    return tt_h[:, :seq_len_keep, :]


def eos_token_ids(config, tokenizer=None) -> set[int]:
    """Collect EOS ids from config, nested text/generation configs, and tokenizer."""
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
    sub = ttnn.slice(ids_tt, (0, 0, 0, 0), (1, 1, 1, length), memory_config=ttnn.L1_MEMORY_CONFIG)
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
    """Cap LM-head columns/device (default 4096; env ``DEVSTRAL2_LM_HEAD_MAX_COLUMNS_PER_DEVICE`` or ``cli_cap``)."""
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
        memory_config=ttnn.L1_MEMORY_CONFIG,
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
    """TT LM head on the 32-row block containing ``last_token_index``; returns sharded logits for Sampling1D."""
    get_last = (last_token_index // 32) * 32
    h_block = ttnn.slice(
        tt_hidden_prefill_out,
        (0, 0, get_last, 0),
        (1, 1, get_last + 32, tt_hidden_prefill_out.shape[-1]),
        memory_config=ttnn.L1_MEMORY_CONFIG,
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
    mesh_shape = list(mesh_device.shape)
    if min(mesh_shape) > 1:
        return False
    vocab_size = int(
        model_args.vocab_size
        if mesh_shape == [1, 1]
        else getattr(model_args, "padded_vocab_size", None) or model_args.vocab_size
    )
    sampling_splits = 2 if mesh_shape == [1, 1] else max(mesh_shape)
    return vocab_size % sampling_splits == 0 and vocab_size // sampling_splits <= 64 * 1024


class DevstralSampling1DAdapter:
    """Tiny trace/sample wrapper around TTTv2 Sampling1D for the Devstral demos."""

    def __init__(self, *, args: ModelArgs, mesh_device, tt_ccl, cq_id: int = 0):
        self.mesh_device = mesh_device
        self.cq_id = cq_id
        self.max_batch_size = 32
        mesh_shape = list(mesh_device.shape)
        self.vocab_size = int(
            args.vocab_size if mesh_shape == [1, 1] else getattr(args, "padded_vocab_size", None) or args.vocab_size
        )
        self.sampler = Sampling1D(
            vocab_size=self.vocab_size,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            max_batch_size=self.max_batch_size,
            max_top_k=getattr(args, "max_top_k", 32),
            sub_core_grids=getattr(args, "sub_core_grids", None),
            sub_core_grid_topk=getattr(args, "sub_core_grid_topk", None),
            start_core=getattr(args, "start_core", ttnn.CoreCoord(0, 0)),
            sampling_memory_config=getattr(args, "model_config", {}).get(
                "DECODE_SAMPLING_INPUT_MEMCFG", ttnn.DRAM_MEMORY_CONFIG
            ),
            pad_to_power_of_2=getattr(args, "pad_logits_to_power_of_2", False),
        )
        self._kpt = None
        self._trace_id = None
        self._trace_input = None
        self._trace_output = None
        self._trace_out_tok = None

    def reset_sampling_params(self, sampling_params):
        self.reset_trace()
        self._release_kpt()
        mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        self._kpt = (
            ttnn.from_torch(
                torch.tensor(list(sampling_params.top_k)[: self.max_batch_size], dtype=torch.int32),
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            ),
            ttnn.from_torch(
                torch.tensor(list(sampling_params.top_p)[: self.max_batch_size], dtype=torch.float32),
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            ),
            ttnn.from_torch(
                torch.tensor(list(sampling_params.temperature)[: self.max_batch_size], dtype=torch.float32),
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            ),
        )

    def _release_kpt(self):
        if self._kpt is None:
            return
        for tensor in self._kpt:
            ttnn.deallocate(tensor)
        self._kpt = None

    def _forward(self, logits, tt_out_tok=None):
        if self._kpt is None:
            raise RuntimeError("Call reset_sampling_params before sample().")
        k, p, temp = self._kpt
        logits_for_sampling = logits
        owns_logits_for_sampling = False
        if int(logits.shape[-1]) > self.vocab_size:
            # 1x1 LM head may produce padded logits; slice to the real vocab so Sampling1D stays on device.
            end = list(logits.shape)
            end[-1] = self.vocab_size
            logits_for_sampling = ttnn.slice(
                logits,
                tuple(0 for _ in end),
                tuple(end),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            owns_logits_for_sampling = True
        try:
            return self.sampler.decode_forward(logits_for_sampling, k=k, p=p, temp=temp, tt_out_tok=tt_out_tok)
        finally:
            if owns_logits_for_sampling:
                ttnn.deallocate(logits_for_sampling)

    def capture_trace(self, logits, *, tt_out_tok=None, skip_precompile=False):
        if not skip_precompile:
            self._forward(logits, tt_out_tok=tt_out_tok)
            ttnn.synchronize_device(self.mesh_device)
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=self.cq_id)
        output = self._forward(logits, tt_out_tok=tt_out_tok)
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=self.cq_id)
        ttnn.synchronize_device(self.mesh_device)
        self._trace_id = trace_id
        self._trace_input = logits
        self._trace_output = output
        self._trace_out_tok = tt_out_tok
        return output

    def sample(self, logits, *, enable_trace=True, tt_out_tok=None, skip_precompile=False):
        if not enable_trace:
            return self._forward(logits, tt_out_tok=tt_out_tok)
        if self._trace_id is None:
            return self.capture_trace(logits, tt_out_tok=tt_out_tok, skip_precompile=skip_precompile)
        if logits is not self._trace_input or tt_out_tok is not self._trace_out_tok:
            raise ValueError("Sampling1D trace input/output tensors changed; call reset_trace() before recapturing.")
        ttnn.execute_trace(self.mesh_device, self._trace_id, cq_id=self.cq_id, blocking=False)
        return self._trace_output

    def reset_trace(self):
        if self._trace_id is not None:
            ttnn.release_trace(self.mesh_device, self._trace_id)
        self._trace_id = None
        self._trace_input = None
        self._trace_output = None
        self._trace_out_tok = None


def tt_sampling_output_token_id(tt_tokens: ttnn.Tensor, batch_slot: int) -> int:
    """Read global token id for one batch row from Sampling1D output."""
    flat = ttnn.to_torch(ttnn.get_device_tensors(tt_tokens)[0]).reshape(-1)
    return int(flat[batch_slot].item())


def tt_warmup_prefill_lm_head_sampling(
    tt_out: ttnn.Tensor,
    last_token_index: int,
    model_args: ModelArgs,
    tt_lm_head: Optional[LMHead],
    sampling: Optional[DevstralSampling1DAdapter] = None,
) -> None:
    """JIT prefill LM-head/sampling kernels from a prefill output, then release warmup tensors."""
    try:
        if tt_lm_head is None:
            return
        logits = tt_lm_head_logits_block(tt_out, last_token_index, model_args, tt_lm_head)
        try:
            if sampling is not None:
                sample_result = sampling.sample(logits, enable_trace=False)
                sample_tensors = sample_result if isinstance(sample_result, tuple) else (sample_result,)
                for tensor in sample_tensors:
                    if tensor is not None:
                        ttnn.deallocate(tensor)
        finally:
            ttnn.deallocate(logits)
    finally:
        ttnn.deallocate(tt_out)


@dataclass
class TtDecodeInputBuffers:
    """DRAM decode trace inputs (token + positions); stable addresses for ``copy_host_to_device_tensor``."""

    token_ids: ttnn.Tensor  # uint32 [1, 1]
    pos_uint32: ttnn.Tensor  # uint32 [1, W] W=nearest_32(batch) - RoPE table lookup (trace-safe padding)
    pos_int32: ttnn.Tensor  # int32  [1, 1]   - paged_update_cache index


@dataclass
class TtDecodeTraceContext:
    """Decode trace handle, buffers, and outputs (device sampling, TT logits, or CPU LM hidden)."""

    trace_id: int
    buffers: TtDecodeInputBuffers
    output_tokens: Optional[ttnn.Tensor] = None
    output_logits: Optional[ttnn.Tensor] = None
    output_hidden: Optional[ttnn.Tensor] = None
    sampling: Optional[DevstralSampling1DAdapter] = None
    enable_trace: bool = True
    tt_lm: Optional["TtMinistral3Model"] = None
    model_args: Optional[ModelArgs] = None
    tt_lm_head: Optional[LMHead] = None
    page_table: Optional[ttnn.Tensor] = None


# Reused host staging for decode buffer updates (avoids per-step torch alloc).
_DECODE_STAGING_TOK = torch.zeros((1, 1), dtype=torch.int32)
# Width 32 = nearest_32(1): matches RoPE embedding tile padding without device alloc during trace.
_DECODE_POS_WIDTH = nearest_32(1)
_DECODE_STAGING_POS = torch.zeros((1, _DECODE_POS_WIDTH), dtype=torch.int32)


def tt_alloc_decode_input_buffers(mesh_device) -> TtDecodeInputBuffers:
    """Allocate the three small DRAM buffers consumed by traced decode steps."""
    zero_tok = torch.tensor([[0]], dtype=torch.int32)
    zero_pos = torch.zeros((1, _DECODE_POS_WIDTH), dtype=torch.int32)
    common = dict(
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return TtDecodeInputBuffers(
        token_ids=ttnn.from_torch(zero_tok, dtype=ttnn.uint32, **common),
        pos_uint32=ttnn.from_torch(zero_pos, dtype=ttnn.uint32, **common),
        pos_int32=ttnn.from_torch(zero_tok, dtype=ttnn.int32, **common),
    )


def tt_update_decode_input_buffers(
    mesh_device,
    buffers: TtDecodeInputBuffers,
    token_id: int,
    decode_pos: int,
) -> None:
    """Update decode buffers via ``copy_host_to_device_tensor`` (trace-stable addresses)."""
    _DECODE_STAGING_TOK[0, 0] = int(token_id)
    _DECODE_STAGING_POS.zero_()
    _DECODE_STAGING_POS[0, 0] = int(decode_pos)
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    tok_host = ttnn.from_torch(
        _DECODE_STAGING_TOK,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=mesh_mapper,
    )
    pos_u32_host = ttnn.from_torch(
        _DECODE_STAGING_POS,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=mesh_mapper,
    )
    pos_i32_host = ttnn.from_torch(
        _DECODE_STAGING_POS[:, :1],
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=mesh_mapper,
    )
    ttnn.copy_host_to_device_tensor(tok_host, buffers.token_ids)
    ttnn.copy_host_to_device_tensor(pos_u32_host, buffers.pos_uint32)
    ttnn.copy_host_to_device_tensor(pos_i32_host, buffers.pos_int32)


def _tt_decode_lm_head_logits(
    tt_hidden: ttnn.Tensor,
    model_args: ModelArgs,
    tt_lm_head: LMHead,
) -> ttnn.Tensor:
    """LM head on decode hidden ``[1,1,32,dim]`` (``Mode.PREFILL`` input memcfg)."""
    h = tt_hidden
    lm_head_input_mem_cfg = model_args.get_lm_head_input_mem_config(Mode.PREFILL, None)
    if lm_head_input_mem_cfg is not None and lm_head_input_mem_cfg.is_sharded():
        h = ttnn.interleaved_to_sharded(h, lm_head_input_mem_cfg)
    logits = tt_lm_head(h)
    return ttnn.to_memory_config(logits, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def tt_warmup_decode_trace_path(
    mesh_device,
    tt_lm,
    model_args: ModelArgs,
    buffers: TtDecodeInputBuffers,
    *,
    tt_lm_head: Optional[LMHead] = None,
    sampling: Optional[DevstralSampling1DAdapter] = None,
    sampling_output: Optional[ttnn.Tensor] = None,
    page_table: Optional[ttnn.Tensor] = None,
) -> None:
    """Run the exact decode/LM-head/sampling path once outside trace capture to JIT compile kernels."""
    warm_sampling_output = sampling_output
    owns_sampling_output = False
    if sampling is not None and tt_lm_head is not None and warm_sampling_output is None:
        warm_sampling_output = ttnn.from_torch(
            torch.zeros((1, 1, 1, int(sampling.max_batch_size)), dtype=torch.int32),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        owns_sampling_output = True

    warm_hidden = tt_lm.forward_decode_from_device_tensors(
        buffers.token_ids, buffers.pos_uint32, buffers.pos_int32, page_table=page_table
    )
    if tt_lm_head is not None:
        warm_logits = _tt_decode_lm_head_logits(warm_hidden, model_args, tt_lm_head)
        if sampling is not None:
            warm_sample = sampling.sample(warm_logits, enable_trace=False, tt_out_tok=warm_sampling_output)
            if isinstance(warm_sample, tuple):
                _, warm_log_probs = warm_sample
                if warm_log_probs is not None:
                    ttnn.deallocate(warm_log_probs)
        ttnn.deallocate(warm_logits)
    ttnn.deallocate(warm_hidden)
    if owns_sampling_output:
        ttnn.deallocate(warm_sampling_output)
    ttnn.synchronize_device(mesh_device)


def tt_capture_decode_trace(
    mesh_device,
    tt_lm,
    model_args: ModelArgs,
    buffers: TtDecodeInputBuffers,
    *,
    tt_lm_head: Optional[LMHead] = None,
    sampling: Optional[DevstralSampling1DAdapter] = None,
    page_table: Optional[ttnn.Tensor] = None,
    prewarmed: bool = False,
    enable_trace: bool = True,
) -> TtDecodeTraceContext:
    """Capture decode (+ optional LM head / sampling trace); prime buffers first.

    ``page_table`` (constant for the run) routes decode through the paged KV cache; it is captured
    into the trace as a fixed input (only the per-step token/position buffers are updated).
    """
    tokens_out_prealloc = None
    if enable_trace and sampling is not None and tt_lm_head is not None:
        sampling_batch = int(sampling.max_batch_size)
        tokens_out_prealloc = ttnn.from_torch(
            torch.zeros((1, 1, 1, sampling_batch), dtype=torch.int32),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    if not prewarmed:
        tt_warmup_decode_trace_path(
            mesh_device,
            tt_lm,
            model_args,
            buffers,
            tt_lm_head=tt_lm_head,
            sampling=sampling,
            sampling_output=tokens_out_prealloc,
            page_table=page_table,
        )

    if not enable_trace:
        return TtDecodeTraceContext(
            trace_id=-1,
            buffers=buffers,
            enable_trace=False,
            tt_lm=tt_lm,
            model_args=model_args,
            tt_lm_head=tt_lm_head,
            page_table=page_table,
        )

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    hidden = tt_lm.forward_decode_from_device_tensors(
        buffers.token_ids, buffers.pos_uint32, buffers.pos_int32, page_table=page_table
    )
    if tt_lm_head is None:
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        return TtDecodeTraceContext(trace_id=trace_id, buffers=buffers, output_hidden=hidden)

    logits = _tt_decode_lm_head_logits(hidden, model_args, tt_lm_head)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    if sampling is None:
        return TtDecodeTraceContext(trace_id=trace_id, buffers=buffers, output_logits=logits)

    sampling.capture_trace(logits, tt_out_tok=tokens_out_prealloc, skip_precompile=True)

    return TtDecodeTraceContext(
        trace_id=trace_id,
        buffers=buffers,
        output_tokens=tokens_out_prealloc,
        output_logits=logits,
        sampling=sampling,
    )


def tt_execute_decode_trace(mesh_device, ctx: TtDecodeTraceContext) -> None:
    """Replay decode trace (cq 0); run sampling trace when configured."""
    if not ctx.enable_trace:
        if ctx.tt_lm is None or ctx.model_args is None:
            raise RuntimeError("Untraced decode context is missing model handles.")
        hidden = ctx.tt_lm.forward_decode_from_device_tensors(
            ctx.buffers.token_ids,
            ctx.buffers.pos_uint32,
            ctx.buffers.pos_int32,
            page_table=ctx.page_table,
        )
        if ctx.tt_lm_head is None:
            if ctx.output_hidden is not None:
                ttnn.deallocate(ctx.output_hidden)
            ctx.output_hidden = hidden
            return

        logits = _tt_decode_lm_head_logits(hidden, ctx.model_args, ctx.tt_lm_head)
        ttnn.deallocate(hidden)
        if ctx.output_logits is not None:
            ttnn.deallocate(ctx.output_logits)
        ctx.output_logits = logits
        return

    ttnn.execute_trace(mesh_device, ctx.trace_id, cq_id=0, blocking=False)
    if ctx.sampling is not None and ctx.output_tokens is not None and ctx.output_logits is not None:
        ctx.sampling.sample(ctx.output_logits, enable_trace=True, tt_out_tok=ctx.output_tokens)


def tt_release_decode_trace(mesh_device, ctx: TtDecodeTraceContext) -> None:
    """Release the trace handle and deallocate the trace-managed output tensors."""
    if ctx.enable_trace:
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
    """Owned copy for CPU LM head (caller deallocates).

    ``to_memory_config`` returns the input tensor unchanged when it is already in DRAM, which would
    hand out the trace-owned ``output_hidden`` itself; the caller then deallocates it, freeing a tensor
    the trace still writes on every replay (and ``tt_release_decode_trace`` would double-free). Clone so
    the returned tensor is genuinely owned by the caller.
    """
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
    """Embed ids, scatter vision patches, pad, and ``forward_prefill_from_embeddings``."""
    hid = tt_lm.embed_tokens(ids_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    hid4 = ttnn.unsqueeze_to_4D(hid)
    hid_work = ttnn.to_memory_config(hid4, ttnn.DRAM_MEMORY_CONFIG)
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
    tail = ttnn.slice(ids_tt, (0, 0, 0, seq_len - 1), (1, 1, 1, seq_len), memory_config=ttnn.L1_MEMORY_CONFIG)
    v = int(ttnn.to_torch(ttnn.get_device_tensors(tail)[0]).reshape(-1)[0].item())
    ttnn.deallocate(tail)
    return v
