# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Full Hugging Face **NextN** model as draft (MoE / MLA / full decoder stack on Hub).

This is separate from :class:`NextNMTPHeadDraftAdapter`, which runs only the exported
MTP fusion head on **record hidden states**.

Here the draft runs ``AutoModelForCausalLM`` (``trust_remote_code``) with **KV cache**
on **token ids** only — the same pattern as :class:`BaseAsDraftAdapter`. The record’s
``last_hidden_state`` is **not** fed into NextN; only ``committed`` token ids drive the draft.

Use the case-study script under ``scripts/case_study_nextn_full_layer_draft_from_record_cpu.py``.

On **CPU** (and **MPS**), Hugging Face’s FP8 path for DeepSeek-style configs requires **GPU/XPU**;
:class:`NextNFullHuggingfaceDraftAdapter` therefore clears ``quantization_config`` when it is FP8 and
loads with ``torch_dtype`` (``float32`` / ``bfloat16`` / …) instead. Checkpoint tensors may still be
stored as FP8 with ``weight_scale_inv``; those are **block-dequantized** after load so matmul dtypes
match activations.

**Checkpoint vs config:** ``lmsys/DeepSeek-R1-NextN`` uses ``num_hidden_layers=1`` but often
``first_k_dense_replace=3``, which would build layer 0 as a **dense** MLP while the shard stores
**MoE** keys (``mlp.experts.*``). The adapter sets ``first_k_dense_replace=0`` for that case so
weights load. Embeddings and ``lm_head`` are usually absent from the NextN shard; use
``embed_head_aux_safetensors`` or rely on ``default_paths.DEFAULT_EMBED_HEAD_AUX_PATH`` when that file exists.
Hugging Face's "newly initialized" line for those keys is suppressed during ``from_pretrained``; this adapter logs
what was filled afterward.

**Not the MTP fusion graph:** The same ``nextn_layer_parameters.safetensors`` also stores **MTP fusion**
mats (``eh_proj``, ``enorm``, ``hnorm``, ``shared_head.head``) used by :class:`NextNMTPHeadDraftAdapter` with
**record ``last_hidden_state``**. The Hugging Face ``DeepseekV3Model`` code path does **not** reference those
modules — they appear as **unused checkpoint keys** and are **never run** here. So this adapter is **not**
a bug-for-bug substitute for the fusion draft; low acceptance vs a **full R1** trace is expected even when
MoE/MLA weights load correctly.

**Optional:** ``decoder_layer0_override_safetensors`` — after load, replace ``model.layers.0.*`` from a file
built by ``scripts/materialize_r1_decoder_layer_as_nextn_layer0.py`` (main R1 decoder layer remapped to
layer 0). Same CPU/RAM cost as the NextN MoE block; see ``SGLANG_NEXTN_AND_CPU.md``.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from collections.abc import Generator, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from pathlib import Path

from models.demos.speculative_deepseek_r1_broad.config import EagleConfig, PathProposal
from models.demos.speculative_deepseek_r1_broad.default_paths import DEFAULT_EMBED_HEAD_AUX_PATH
from models.demos.speculative_deepseek_r1_broad.hf_cache import ensure_sharded_safetensors_index_has_metadata
from models.demos.speculative_deepseek_r1_broad.local_hf_snapshot import load_nextn_mtp_auxiliary_safetensors
from models.demos.speculative_deepseek_r1_broad.models_draft import (
    _clone_hf_kv,
    draft_branch_token_ids_from_logits,
    draft_requires_positive_top_k,
    truncate_beams_by_draft_confidence,
)
from models.demos.deepseek_v3.utils.dequantize import dequantize_tensor

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _suppress_hf_checkpoint_load_console_noise() -> Generator[None, None, None]:
    """Mute HF/Accelerate noise during ``from_pretrained`` (sharded NextN on CPU).

    Accelerate may print a long line like "The following layers were not sharded: ..." when
    ``low_cpu_mem_usage=True`` loads a weight map into a meta model. That describes checkpoint
    layout (whole layers vs per-tensor shards), not a broken load.
    """

    log_names = (
        "transformers.modeling_utils",
        "transformers.modeling",
        "transformers",
        "accelerate",
        "accelerate.utils",
        "accelerate.big_modeling",
        "accelerate.utils.modeling",
    )
    prev_levels: dict[str, int] = {}
    for name in log_names:
        lg = logging.getLogger(name)
        prev_levels[name] = lg.level
        lg.setLevel(logging.ERROR)

    class _DropNotShardedBanner:
        __slots__ = ("_real",)

        def __init__(self, real: Any) -> None:
            self._real = real

        def write(self, s: str) -> int:
            if s and "were not sharded" in s.lower():
                return len(s)
            return self._real.write(s)

        def flush(self) -> None:
            self._real.flush()

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real, name)

    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _DropNotShardedBanner(old_out)
    sys.stderr = _DropNotShardedBanner(old_err)
    try:
        yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        for name, lvl in prev_levels.items():
            logging.getLogger(name).setLevel(lvl)


# HF logs these as "newly initialized" during from_pretrained; we hydrate right after load.
_NEXTN_POST_LOAD_GLOBAL_KEYS = frozenset(
    {"lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"}
)

# Fusion MTP tensors in the NextN shard — not wired into DeepseekV3Model / DeepseekV3DecoderLayer.
_MTP_FUSION_UNUSED_PREFIXES: tuple[str, ...] = (
    "model.layers.0.eh_proj",
    "model.layers.0.enorm",
    "model.layers.0.hnorm",
    "model.layers.0.shared_head.head",
)


def _warn_mtp_fusion_weights_not_in_hf_forward(unexpected_keys: Sequence[str]) -> None:
    if not unexpected_keys:
        return
    hit = [k for k in unexpected_keys if k.startswith(_MTP_FUSION_UNUSED_PREFIXES)]
    if not hit:
        return
    logger.warning(
        "NextN shard MTP fusion weights are **not used** by this HF model class (%d unused keys, e.g. %s). "
        "Forward is embed → MoE+MLA decoder → norm → lm_head on **token ids only** — not eh_proj(embed⊕hidden). "
        "For acceptance vs a full-R1 record similar to the fusion demo, use NextNMTPHeadDraftAdapter "
        "(record hidden + same fusion mats). MoE/MLA init here is not the same computation graph.",
        len(hit),
        hit[:6],
    )


def _resolve_embed_head_aux_path(explicit: str | Path | None) -> Path | None:
    if explicit is not None:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"embed_head_aux_safetensors not found: {p}")
        return p
    if DEFAULT_EMBED_HEAD_AUX_PATH.is_file():
        logger.info("Using default embed/head aux: %s", DEFAULT_EMBED_HEAD_AUX_PATH)
        return DEFAULT_EMBED_HEAD_AUX_PATH
    return None


def _resolve_torch_dtype(name: str) -> torch.dtype:
    n = (name or "float32").lower()
    if n == "float16":
        return torch.float16
    if n == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _config_uses_fp8_quantization(config: object) -> bool:
    qc = getattr(config, "quantization_config", None)
    if qc is None:
        return False
    if isinstance(qc, dict):
        method = qc.get("quant_method") or qc.get("quantization_method") or ""
        return str(method).lower() == "fp8"
    method = getattr(qc, "quant_method", None) or getattr(qc, "quantization_method", None)
    return str(method or "").lower() == "fp8"


def _strip_fp8_quantization_config_if_needed(config: object, *, device: torch.device) -> bool:
    """Remove FP8 ``quantization_config`` on CPU/MPS (HF FP8 kernels need GPU/XPU).

    Must **delete** the attribute, not set it to ``None``: ``transformers`` uses
    ``hasattr(config, "quantization_config")`` for ``pre_quantized`` and would call
    ``AutoHfQuantizer.supports_quant_method(None)``, which expects a dict.
    """
    if device.type not in ("cpu", "mps"):
        return False
    if not _config_uses_fp8_quantization(config):
        return False
    try:
        delattr(config, "quantization_config")
    except AttributeError:
        d = getattr(config, "__dict__", None)
        if isinstance(d, dict) and "quantization_config" in d:
            del d["quantization_config"]
        else:
            logger.warning(
                "Could not remove quantization_config from config; full NextN load on CPU may still fail."
            )
            return False
    return True


def _fp8_weight_dtypes() -> tuple[torch.dtype, ...]:
    out: list[torch.dtype] = []
    for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz"):
        dt = getattr(torch, name, None)
        if dt is not None:
            out.append(dt)
    return tuple(out)


def _fp8_weight_block_shape_from_config(config: object) -> tuple[int, int]:
    qc = getattr(config, "quantization_config", None)
    if isinstance(qc, dict):
        wb = qc.get("weight_block_size")
        if isinstance(wb, (list, tuple)) and len(wb) == 2:
            return int(wb[0]), int(wb[1])
    return (128, 128)


def _patch_nextn_single_layer_moe_alignment(config: object) -> int | None:
    """Align decoder MLP type with NextN shard (MoE tensors, not dense ``mlp.gate_proj``)."""
    nl = getattr(config, "num_hidden_layers", None)
    nre = getattr(config, "n_routed_experts", None)
    fkdr = getattr(config, "first_k_dense_replace", None)
    if nl == 1 and nre is not None and fkdr is not None and int(fkdr) > 0:
        prev = int(fkdr)
        config.first_k_dense_replace = 0
        return prev
    return None


@torch.no_grad()
def _dequantize_fp8_linear_weights(
    model: torch.nn.Module,
    *,
    block_shape: tuple[int, int],
    target_dtype: torch.dtype,
) -> int:
    """Convert FP8 ``*.weight`` tensors using paired ``*.weight_scale_inv`` (DeepSeek Hub layout)."""
    fp8_dtypes = _fp8_weight_dtypes()
    if not fp8_dtypes:
        return 0
    sd = model.state_dict()
    n = 0
    for name, param in model.named_parameters():
        if not name.endswith(".weight") or param.dtype not in fp8_dtypes:
            continue
        scale_key = name.replace(".weight", ".weight_scale_inv")
        if scale_key in sd:
            inv = sd[scale_key]
            dq = dequantize_tensor(param.data, inv, block_shape).to(target_dtype)
            param.data = dq
            n += 1
        else:
            param.data = param.data.to(target_dtype)
            n += 1
    return n


@torch.no_grad()
def _try_copy_shared_head_norm_to_model_norm(model: torch.nn.Module, snapshot_dir: Path) -> bool:
    """Map ``model.layers.0.shared_head.norm.weight`` from the NextN shard into ``model.norm``."""
    try:
        from safetensors import safe_open
    except ImportError:
        return False
    shard = snapshot_dir / "nextn_layer_parameters.safetensors"
    key = "model.layers.0.shared_head.norm.weight"
    if not shard.is_file():
        return False
    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "norm"):
        return False
    with safe_open(str(shard), framework="pt", device="cpu") as sf:
        if key not in sf.keys():
            return False
        w = sf.get_tensor(key)
    tgt = inner.norm.weight
    tgt.data.copy_(w.to(device=tgt.device, dtype=tgt.dtype))
    return True


@torch.no_grad()
def _apply_decoder_layer0_override_from_safetensors(
    model: torch.nn.Module,
    path: Path,
    *,
    target_dtype: torch.dtype,
) -> int:
    """Replace ``model.layers.0.*`` weights from a single ``.safetensors`` file.

    Use a file produced by ``materialize_r1_decoder_layer_as_nextn_layer0.py`` (R1 layer *K*
    keys remapped to ``model.layers.0.*``) to swap the NextN-trained decoder block for a main
    checkpoint layer while keeping the same HF module graph. Call **after** FP8 dequantization
    on the loaded NextN model so dtypes match ``target_dtype``.
    """
    try:
        from safetensors import safe_open
    except ImportError as e:
        raise ImportError("decoder layer override requires `safetensors`") from e
    sd = model.state_dict()
    to_load: dict[str, torch.Tensor] = {}
    ignored: list[str] = []
    with safe_open(str(path.expanduser().resolve()), framework="pt", device="cpu") as sf:
        for k in sf.keys():
            if not k.startswith("model.layers.0."):
                ignored.append(k)
                continue
            if k not in sd:
                ignored.append(k)
                continue
            t = sf.get_tensor(k)
            if sd[k].shape != t.shape:
                raise ValueError(
                    f"Override {k}: checkpoint shape {tuple(t.shape)} != model {tuple(sd[k].shape)}"
                )
            if torch.is_floating_point(t) or str(t.dtype).startswith("torch.float8"):
                to_load[k] = t.to(dtype=target_dtype)
            else:
                to_load[k] = t.clone()
    if not to_load:
        raise ValueError(
            f"No usable ``model.layers.0.*`` tensors in {path} (ignored {len(ignored)} keys)."
        )
    model.load_state_dict(to_load, strict=False)
    if ignored:
        logger.info(
            "decoder_layer0 override: ignored %d keys (wrong prefix or not in model), e.g. %s",
            len(ignored),
            ignored[:8],
        )
    logger.info("decoder_layer0 override: loaded %d tensors from %s", len(to_load), path)
    return len(to_load)


@torch.no_grad()
def _apply_embed_head_aux(
    model: torch.nn.Module,
    aux_path: Path,
    *,
    target_dtype: torch.dtype,
    device: torch.device,
) -> None:
    embed, head = load_nextn_mtp_auxiliary_safetensors(aux_path)
    embed = embed.to(device=device, dtype=target_dtype)
    head = head.to(device=device, dtype=target_dtype)
    m = getattr(model, "model", model)
    if not hasattr(m, "embed_tokens") or not hasattr(model, "lm_head"):
        raise RuntimeError("Expected CausalLM with model.embed_tokens and lm_head.")
    if embed.shape != m.embed_tokens.weight.shape:
        raise ValueError(f"embed shape {embed.shape} != embed_tokens {m.embed_tokens.weight.shape}")
    if head.shape != model.lm_head.weight.shape:
        raise ValueError(f"head shape {head.shape} != lm_head {model.lm_head.weight.shape}")
    m.embed_tokens.weight.data.copy_(embed)
    model.lm_head.weight.data.copy_(head)


def _hf_past_kv_seq_len(past_key_values: object) -> int:
    """Sequence length stored in ``past_key_values`` (``Cache``, full-model cache, or one-layer ``(k, v)``).

    A **single** ``DeepseekV3DecoderLayer`` returns ``past_key_value`` as ``(key_states, value_states)``
    where ``key_states`` is a 4D tensor ``[batch, heads_or_mqa, seq_len, dim]``. In that case
    ``past_key_values[0]`` is **already** the key tensor. The old logic treated ``past_key_values`` as
    *tuple-of-layers* and used ``past_key_values[0][0]`` as the key, which actually indexed **batch 0**
    of the key tensor and read ``shape[2]`` as a **head dimension** — wrong ``seq_len`` for masks / RoPE
    and collapsed speculative acceptance after the first draft step.
    """
    if past_key_values is None:
        return 0
    try:
        from transformers.cache_utils import Cache

        if isinstance(past_key_values, Cache):
            if hasattr(past_key_values, "get_seq_length"):
                return int(past_key_values.get_seq_length(0))
    except Exception:
        pass
    if not isinstance(past_key_values, tuple) or len(past_key_values) == 0:
        return 0
    first = past_key_values[0]
    # One layer's (k, v): first slot is key tensor [B, H, S, D] (incl. MLA ``H==1``).
    if isinstance(first, torch.Tensor):
        if first.dim() == 4:
            return int(first.shape[2])
        return 0
    # Full model legacy cache: tuple of layers, each layer is (k, v).
    if isinstance(first, (tuple, list)) and len(first) > 0 and isinstance(first[0], torch.Tensor):
        k = first[0]
        if k.dim() == 4:
            return int(k.shape[2])
    return 0


def _causal_attention_mask_2d(
    batch: int,
    input_seq_len: int,
    past_seq_len: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """All-valid 2D mask of shape ``(batch, past_seq_len + input_seq_len)``.

    Transformers may return no 4D causal mask when ``query_length == 1`` (decode step), but
    DeepSeek's eager ``DeepseekV3Attention`` requires a non-None ``attention_mask``. Passing
    this tensor forces the ``to_4d`` path in ``_prepare_4d_causal_attention_mask``.
    """
    return torch.ones((batch, past_seq_len + input_seq_len), dtype=torch.long, device=device)


class NextNFullHuggingfaceDraftAdapter:
    """Speculative draft via full NextN HF causal LM (all layers / MoE / MLP in that checkpoint)."""

    def __init__(
        self,
        *,
        model_id_or_path: str,
        device: str = "cpu",
        torch_dtype: str = "float32",
        trust_remote_code: bool = True,
        local_files_only: bool = False,
        keep_fp8_quantization_config: bool = False,
        embed_head_aux_safetensors: str | Path | None = None,
        decoder_layer0_override_safetensors: str | Path | None = None,
    ) -> None:
        self.device = torch.device(device)
        dt = _resolve_torch_dtype(torch_dtype)
        aux_embed_path = _resolve_embed_head_aux_path(embed_head_aux_safetensors)
        layer0_override: Path | None = None
        if decoder_layer0_override_safetensors is not None:
            layer0_override = Path(decoder_layer0_override_safetensors).expanduser().resolve()
            if not layer0_override.is_file():
                raise FileNotFoundError(f"decoder_layer0_override_safetensors not found: {layer0_override}")
        logger.info(
            "Loading full NextN draft model %r device=%s dtype=%s local_only=%s",
            model_id_or_path,
            device,
            torch_dtype,
            local_files_only,
        )
        local_dir = Path(model_id_or_path)
        snapshot_dir_for_shards = local_dir if local_dir.is_dir() else None
        if local_dir.is_dir():
            n = ensure_sharded_safetensors_index_has_metadata(local_dir)
            if n:
                logger.info(
                    "Patched %d sharded checkpoint index file(s) under %s (added empty metadata for transformers).",
                    n,
                    local_dir,
                )
        # Lazy import so importing eagle3 modules does not require transformers until draft is built.
        from transformers import AutoConfig, AutoModelForCausalLM

        cfg_kwargs: dict[str, Any] = dict(trust_remote_code=trust_remote_code)
        if local_files_only:
            cfg_kwargs["local_files_only"] = True
        config = AutoConfig.from_pretrained(model_id_or_path, **cfg_kwargs)

        fp8_block = _fp8_weight_block_shape_from_config(config)
        prev_dense = _patch_nextn_single_layer_moe_alignment(config)
        if prev_dense is not None:
            logger.info(
                "Set first_k_dense_replace=0 so single-layer NextN loads MoE shards (was %s).",
                prev_dense,
            )

        if not keep_fp8_quantization_config and _strip_fp8_quantization_config_if_needed(
            config, device=self.device
        ):
            logger.info(
                "Cleared config.quantization_config (FP8) for device=%s — HF requires GPU/XPU for FP8 kernels; "
                "loading weights as torch_dtype=%s; FP8 tensors will be dequantized after load (block=%s).",
                self.device.type,
                dt,
                fp8_block,
            )

        if hasattr(config, "torch_dtype"):
            try:
                config.torch_dtype = dt
            except (TypeError, ValueError):
                pass

        load_kwargs: dict[str, Any] = dict(
            trust_remote_code=trust_remote_code,
            torch_dtype=dt,
            config=config,
        )
        if local_files_only:
            load_kwargs["local_files_only"] = True
        if self.device.type == "cpu":
            load_kwargs["low_cpu_mem_usage"] = True
            load_kwargs["attn_implementation"] = "eager"

        import inspect

        try:
            load_with_info = "output_loading_info" in inspect.signature(
                AutoModelForCausalLM.from_pretrained
            ).parameters
        except (TypeError, ValueError):
            load_with_info = False

        def _load() -> Any:
            try:
                if load_with_info:
                    return AutoModelForCausalLM.from_pretrained(
                        model_id_or_path, **load_kwargs, output_loading_info=True
                    )
                return AutoModelForCausalLM.from_pretrained(model_id_or_path, **load_kwargs)
            except TypeError:
                load_kwargs.pop("attn_implementation", None)
                if load_with_info:
                    return AutoModelForCausalLM.from_pretrained(
                        model_id_or_path, **load_kwargs, output_loading_info=True
                    )
                return AutoModelForCausalLM.from_pretrained(model_id_or_path, **load_kwargs)

        # HF/Accelerate are chatty on sharded + low_cpu_mem_usage; see _suppress_hf_checkpoint_load_console_noise.
        with _suppress_hf_checkpoint_load_console_noise():
            loaded = _load()

        if isinstance(loaded, tuple) and len(loaded) == 2:
            self.model, loading_info = loaded
            if isinstance(loading_info, dict):
                missing = loading_info.get("missing_keys") or []
                unexpected = loading_info.get("unexpected_keys") or []
            else:
                missing = getattr(loading_info, "missing_keys", None) or []
                unexpected = getattr(loading_info, "unexpected_keys", None) or []
                if not missing and isinstance(loading_info, (list, tuple)) and len(loading_info) > 0:
                    missing = loading_info[0]
                    unexpected = loading_info[1] if len(loading_info) > 1 else []
            expected_missing = {k for k in missing if k in _NEXTN_POST_LOAD_GLOBAL_KEYS}
            other_missing = [k for k in missing if k not in _NEXTN_POST_LOAD_GLOBAL_KEYS]
            if expected_missing:
                logger.info(
                    "Checkpoint omits NextN globals (filled after load if aux/snapshot available): %s",
                    sorted(expected_missing),
                )
            if other_missing:
                logger.warning("from_pretrained missing_keys (%d): %s", len(other_missing), other_missing[:24])
            if unexpected:
                logger.info("from_pretrained unexpected_keys (%d): %s", len(unexpected), unexpected[:24])
            _warn_mtp_fusion_weights_not_in_hf_forward(unexpected)
        else:
            self.model = loaded

        # Cheap CPU fixes before FP8 dequant (which peaks memory on full MoE).
        norm_ok = False
        if snapshot_dir_for_shards is not None and _try_copy_shared_head_norm_to_model_norm(
            self.model, snapshot_dir_for_shards
        ):
            norm_ok = True
            logger.info("Copied shared_head.norm into model.norm from nextn_layer_parameters.safetensors.")

        if aux_embed_path is not None:
            _apply_embed_head_aux(self.model, aux_embed_path, target_dtype=dt, device=torch.device("cpu"))
            logger.info("Loaded embed_tokens + lm_head from auxiliary safetensors: %s", aux_embed_path)
        else:
            logger.warning(
                "No embed/head aux file (build with materialize_nextn_embed_head_aux_from_r1_shards.py or set "
                "embed_head_aux_safetensors). Default tried: %s — lm_head and embed_tokens stay randomly initialized.",
                DEFAULT_EMBED_HEAD_AUX_PATH,
            )

        if not norm_ok:
            if snapshot_dir_for_shards is None:
                logger.warning(
                    "model.norm was not patched: use a local snapshot directory as model_id_or_path "
                    "(not a Hub repo id) so nextn_layer_parameters.safetensors can be read for shared_head.norm."
                )
            else:
                logger.warning(
                    "model.norm was not patched: missing key model.layers.0.shared_head.norm.weight under %s",
                    snapshot_dir_for_shards,
                )

        n_dq = _dequantize_fp8_linear_weights(self.model, block_shape=fp8_block, target_dtype=dt)
        if n_dq:
            logger.info("Dequantized %d FP8 weight tensors to %s.", n_dq, dt)

        if layer0_override is not None:
            _apply_decoder_layer0_override_from_safetensors(self.model, layer0_override, target_dtype=dt)

        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def forward_draft(self, prefix_token_ids: Sequence[int], cfg: EagleConfig, **kwargs: object) -> PathProposal:
        if len(prefix_token_ids) == 0:
            raise ValueError("prefix_token_ids must be non-empty.")
        draft_mtp_greedy = getattr(cfg, "draft_mtp_greedy", False)
        if cfg.depth <= 0 or draft_requires_positive_top_k(cfg):
            return PathProposal(paths=[], draft_probs_per_path=None)

        top_k = 1 if draft_mtp_greedy else int(cfg.top_k)

        gen = kwargs.get("draft_torch_generator")
        gen_t = gen if isinstance(gen, torch.Generator) else None

        prefix = torch.tensor([list(prefix_token_ids)], dtype=torch.long, device=self.device)
        b, sq = prefix.shape
        attn = _causal_attention_mask_2d(b, sq, 0, device=self.device)
        out = self.model(input_ids=prefix, attention_mask=attn, use_cache=True)
        prefix_logits = out.logits[:, -1, :]
        prefix_probs = F.softmax(prefix_logits.float(), dim=-1)
        prefix_kv = out.past_key_values

        topk_ids = draft_branch_token_ids_from_logits(prefix_logits[0], cfg, top_k, gen_t)
        if not topk_ids:
            return PathProposal(paths=[], draft_probs_per_path=None)
        Beam = tuple[list[int], list[float], tuple]
        beams: list[Beam] = []
        for tok_id in topk_ids:
            q = float(prefix_probs[0, tok_id].item())
            beams.append(([tok_id], [q], _clone_hf_kv(prefix_kv)))

        beams = truncate_beams_by_draft_confidence(beams, cfg.max_paths, lambda b: b[1])

        for _depth_step in range(1, cfg.depth):
            if not beams:
                break
            next_beams: list[Beam] = []
            for appended, probs, kv in beams:
                new_token = torch.tensor([[appended[-1]]], dtype=torch.long, device=self.device)
                nb, nq = new_token.shape
                past_len = _hf_past_kv_seq_len(kv)
                attn_step = _causal_attention_mask_2d(nb, nq, past_len, device=self.device)
                out = self.model(
                    input_ids=new_token,
                    past_key_values=kv,
                    attention_mask=attn_step,
                    use_cache=True,
                )
                logits = out.logits[:, -1, :]
                logits_probs = F.softmax(logits.float(), dim=-1)
                new_kv = out.past_key_values
                topk_ids = draft_branch_token_ids_from_logits(logits[0], cfg, top_k, gen_t)
                if not topk_ids:
                    continue
                for tok_id in topk_ids:
                    q = float(logits_probs[0, tok_id].item())
                    next_beams.append((appended + [tok_id], probs + [q], _clone_hf_kv(new_kv)))
            next_beams = truncate_beams_by_draft_confidence(next_beams, cfg.max_paths, lambda b: b[1])
            beams = next_beams

        paths = [appended for appended, _, _ in beams]
        draft_probs_per_path = [probs for _, probs, _ in beams]
        return PathProposal(paths=paths, draft_probs_per_path=draft_probs_per_path)

    def propose_paths(self, prefix_token_ids: Sequence[int], cfg: EagleConfig, **kwargs: object) -> PathProposal:
        return self.forward_draft(prefix_token_ids, cfg, **kwargs)
