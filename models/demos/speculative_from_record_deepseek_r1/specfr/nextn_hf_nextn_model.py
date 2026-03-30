# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face **NextN** causal LM load for :class:`NextNSglangStructureDraftAdapter`.

Loads ``AutoModelForCausalLM`` (``trust_remote_code``) with the same FP8 / MoE / embed-head
handling as the former standalone ``nextn_full_layer_draft`` module. Structure draft only
needs ``.model`` and calls ``inner.layers[0]``, ``inner.norm``, ``lm_head`` itself — it does
**not** run the full-model ``forward`` on token ids only.

This bundle intentionally omits the legacy **ids-only** speculative ``forward_draft`` API; use
``NextNSglangCPUDraftAdapter`` (default) or ``--sglang-draft-structure`` for MTP fusion drafts.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Any

import torch

from specfr.default_paths import DEFAULT_EMBED_HEAD_AUX_PATH
from specfr.dequantize import dequantize_tensor
from specfr.hf_cache import ensure_sharded_safetensors_index_has_metadata
from specfr.local_hf_snapshot import load_nextn_mtp_auxiliary_safetensors

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _suppress_hf_checkpoint_load_console_noise() -> Generator[None, None, None]:
    """Mute HF/Accelerate noise during ``from_pretrained`` (sharded NextN on CPU)."""

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


_NEXTN_POST_LOAD_GLOBAL_KEYS = frozenset(
    {"lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"}
)

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
        "Structure draft applies fusion mats separately; HF forward does not use eh_proj(embed⊕hidden).",
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
    """Sequence length in ``past_key_values`` (Cache, full-model cache, or one-layer ``(k, v)``)."""
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
    if isinstance(first, torch.Tensor):
        if first.dim() == 4:
            return int(first.shape[2])
        return 0
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
    return torch.ones((batch, past_seq_len + input_seq_len), dtype=torch.long, device=device)


class NextNFullHuggingfaceDraftAdapter:
    """Load HF NextN causal LM; exposes ``.model`` for :class:`NextNSglangStructureDraftAdapter`."""

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
            "Loading HF NextN model %r device=%s dtype=%s local_only=%s",
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
                "No embed/head aux file (build with scripts/materialize_nextn_embed_head_aux_from_r1_shards.py "
                "in speculative_from_record_deepseek_r1 or set "
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
