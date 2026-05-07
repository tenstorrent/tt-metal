# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full MoE FFN (router + routed experts + shared expert MLP), HF-compatible.

Layout mirrors DeepSeek-V3 ``models/demos/deepseek_v3/tt/moe.py`` (class ``MoE``): composition of
gate, experts, and shared path — here implemented by subclassing HF ``Mistral4MoE`` so forward /
parameter names match checkpoints.

``TtMistral4MoE`` is both an HF ``nn.Module`` and exposes DeepSeek-style ``ttnn`` entrypoints
(:meth:`convert_weights`, :meth:`prefill_model_config`, :meth:`forward_prefill`, …). Until native
``ttnn`` MoE kernels exist, :meth:`forward_prefill` / :meth:`forward_decode` run HF ``Mistral4MoE``
on host and bridge activations through mesh ``from_torch`` / ``to_torch`` (parity tests).
"""

from __future__ import annotations

import inspect
import json
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.models.mistral4.configuration_mistral4 import Mistral4Config
from transformers.models.mistral4.modeling_mistral4 import Mistral4MoE

import ttnn
from models.demos.mistral_small_4_119B.tt.moe.experts import TtMistral4Experts
from models.demos.mistral_small_4_119B.tt.moe.moe_gate import TtMistral4MoEGate
from models.demos.mistral_small_4_119B.tt.moe.shared_expert import TtMistral4SharedExpert


def _experts_impl_config(config: Mistral4Config) -> None:
    if getattr(config, "_experts_implementation", None) in (None, ""):
        try:
            config._experts_implementation = "grouped_mm"
        except (AttributeError, TypeError):
            pass


# Matches ``models.demos.deepseek_v3.utils.run_config.MESH_DEVICE_STATE_DICT_KEY`` /
# ``MeshDeviceStub`` without importing DeepSeek (stub ``ttnn`` may omit ``Topology``, etc.).
MESH_DEVICE_STATE_DICT_KEY = "mesh_device"


class MeshDeviceStub:
    """Placeholder mesh shape; merged into real ``ttnn.MeshDevice`` by ``create_run_config``."""

    __slots__ = ("mesh_shape",)

    def __init__(self, mesh_shape: tuple[int, int]):
        self.mesh_shape = tuple(mesh_shape)


# HF Mistral4MoE state dict keys (relative to ``...mlp.`` prefix in the checkpoint).
_MOE_SUFFIXES: tuple[str, ...] = (
    "gate.weight",
    "experts.gate_up_proj",
    "experts.down_proj",
    "shared_experts.gate_proj.weight",
    "shared_experts.up_proj.weight",
    "shared_experts.down_proj.weight",
)


def mistral4_text_config_from_snapshot(model_dir: str | Path) -> Mistral4Config:
    """Build ``Mistral4Config`` from ``config.json`` ``text_config`` (Mistral3 multimodal layout)."""
    model_dir = Path(model_dir).resolve()
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json under {model_dir}")

    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)
    text_cfg = (
        raw.get("text_config")
        or raw.get("language_model", {}).get("text_config")
        or raw.get("language_model", {}).get("config", {}).get("text_config")
    )
    if not isinstance(text_cfg, dict):
        core_keys = {"hidden_size", "num_hidden_layers", "num_attention_heads"}
        if core_keys.issubset(set(raw.keys())):
            text_cfg = raw
        else:
            raise ValueError(
                f"{config_path} has no usable text_config (expected Mistral3 multimodal snapshot or plain Mistral4 config)."
            )
    sig = inspect.signature(Mistral4Config.__init__)
    allowed = set(sig.parameters) - {"self", "kwargs"}
    kwargs = {k: v for k, v in text_cfg.items() if k in allowed}
    return Mistral4Config(**kwargs)


def _maybe_cast_loaded_weight(tensor: torch.Tensor) -> torch.Tensor:
    """Cast FP8 dtypes to bf16 for ``nn.Parameter`` / bridge compatibility."""
    for dt_name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
        dt = getattr(torch, dt_name, None)
        if dt is not None and tensor.dtype == dt:
            return tensor.to(torch.float32).to(torch.bfloat16)

    if tensor.dtype == torch.uint8:
        raise RuntimeError(
            "Expert weights are UINT8 (blocked FP8 layout); load the full model with "
            "`transformers` + FP8 dequantization (see demo `materialize_mistral_moe_fp8_experts_from_disk`) "
            "or use bf16 shards."
        )

    if tensor.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return tensor

    return tensor.to(torch.bfloat16)


def _load_index(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Missing sharded index {index_path}. Expected model.safetensors.index.json next to shards."
        )
    with open(index_path, encoding="utf-8") as f:
        meta = json.load(f)
    return meta["weight_map"]


def _pick_prefix(weight_map: dict[str, str], layer_idx: int) -> str:
    candidates = (
        f"language_model.model.layers.{layer_idx}.mlp.",
        f"model.layers.{layer_idx}.mlp.",
    )
    for prefix in candidates:
        gate_key = prefix + "gate.weight"
        if gate_key in weight_map:
            return prefix
    raise KeyError(
        f"No MoE prefix found for layer {layer_idx}. "
        f"Tried language_model.model.layers.* and model.layers.* gate.weight keys."
    )


def _shard_groups(full_keys: Iterable[str], weight_map: dict[str, str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for fk in full_keys:
        if fk not in weight_map:
            raise KeyError(f"Tensor missing from checkpoint index: {fk}")
        out[weight_map[fk]].append(fk)
    return dict(out)


def _read_tensors_from_shard(shard_path: Path, keys: list[str]) -> dict[str, torch.Tensor]:
    if not shard_path.is_file():
        raise FileNotFoundError(f"Shard file missing: {shard_path}")
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(shard_path, framework="pt", device="cpu") as sf:
        available = set(sf.keys())
        for k in keys:
            if k not in available:
                raise KeyError(f"Key {k} not present in {shard_path}")
            tensors[k] = sf.get_tensor(k)
    return tensors


def load_ttmistral4_moe_from_sharded_safetensors(
    moe_module: torch.nn.Module,
    model_dir: str | Path,
    layer_idx: int,
    *,
    strict: bool = False,
) -> Any:
    """Load MoE weights for ``layer_idx`` from sharded ``safetensors`` into ``moe_module``."""
    model_dir = Path(model_dir).resolve()
    weight_map = _load_index(model_dir)
    prefix = _pick_prefix(weight_map, layer_idx)

    full_keys = [prefix + suffix for suffix in _MOE_SUFFIXES]
    shard_groups = _shard_groups(full_keys, weight_map)

    state_dict: dict[str, torch.Tensor] = {}
    for shard_name, keys in shard_groups.items():
        shard_tensors = _read_tensors_from_shard(model_dir / shard_name, keys)
        for fk, tensor in shard_tensors.items():
            short_key = fk[len(prefix) :]
            state_dict[short_key] = _maybe_cast_loaded_weight(tensor)

    expected_gate_shape = (int(moe_module.n_routed_experts), int(moe_module.config.hidden_size))
    if "gate.weight" in state_dict and tuple(state_dict["gate.weight"].shape) != expected_gate_shape:
        raise ValueError(
            f"Unexpected gate.weight shape at layer {layer_idx}: got {tuple(state_dict['gate.weight'].shape)}, "
            f"expected {expected_gate_shape}"
        )

    expected_experts_shape = (
        int(moe_module.n_routed_experts),
        int(moe_module.config.moe_intermediate_size * 2),
        int(moe_module.config.hidden_size),
    )
    if (
        "experts.gate_up_proj" in state_dict
        and tuple(state_dict["experts.gate_up_proj"].shape) != expected_experts_shape
    ):
        raise ValueError(
            f"Unexpected experts.gate_up_proj shape at layer {layer_idx}: got {tuple(state_dict['experts.gate_up_proj'].shape)}, "
            f"expected {expected_experts_shape}"
        )
    expected_down_shape = (
        int(moe_module.n_routed_experts),
        int(moe_module.config.hidden_size),
        int(moe_module.config.moe_intermediate_size),
    )
    if "experts.down_proj" in state_dict and tuple(state_dict["experts.down_proj"].shape) != expected_down_shape:
        raise ValueError(
            f"Unexpected experts.down_proj shape at layer {layer_idx}: got {tuple(state_dict['experts.down_proj'].shape)}, "
            f"expected {expected_down_shape}"
        )

    return moe_module.load_state_dict(state_dict, strict=strict)


def bridge_torch_state_dict_from_snapshot(model_dir: str | Path, layer_idx: int) -> dict[str, torch.Tensor]:
    """Load MoE for ``layer_idx`` and return a CPU state dict for ``bridge_torch_state_dict``."""
    cfg = mistral4_text_config_from_snapshot(model_dir)
    moe = TtMistral4MoE(cfg)
    load_ttmistral4_moe_from_sharded_safetensors(moe, model_dir, layer_idx, strict=False)
    return {k: v.detach().cpu() for k, v in moe.state_dict().items()}


class TtMistral4MoE(Mistral4MoE):
    """Mistral-4 MoE block with DeepSeek-style ``ttnn`` entrypoints and HF-compatible children.

    Uses :class:`TtMistral4MoEGate`, :class:`TtMistral4Experts`, and :class:`TtMistral4SharedExpert`
    so checkpoints load the same keys while TT-specific hooks live on those subclasses.
    """

    def __init__(self, config: Mistral4Config) -> None:
        # Build like ``Mistral4MoE`` but swap in TT skeleton children (HF-compatible forwards).
        nn.Module.__init__(self)
        self.config = config
        self.experts = TtMistral4Experts(config)
        self.gate = TtMistral4MoEGate(config)
        self.shared_experts = TtMistral4SharedExpert(config)
        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok

    # --- DeepSeek-compatible TT hooks (host-accurate bridge until native MoE kernels land) ---

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> dict[str, Any]:
        del hf_config
        assert len(state_dicts) == 1 and state_dicts[0] is not None, "Mistral MoE expects one state dict"
        (state_dict,) = state_dicts
        assert state_dict is not None
        output_path.mkdir(parents=True, exist_ok=True)

        bridge: dict[str, torch.Tensor] = {k: v.detach().cpu() for k, v in state_dict.items()}

        out: dict[str, Any] = {"bridge_torch_state_dict": bridge}
        if "gate.weight" in state_dict:
            w = state_dict["gate.weight"].to(torch.bfloat16)
            out["gate_weight_ttnn"] = ttnn.from_torch(
                w,
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
        return out

    @classmethod
    def create_shared_state(cls, hf_config: PretrainedConfig, mesh_device: Any) -> dict[str, Any]:
        del hf_config
        return {MESH_DEVICE_STATE_DICT_KEY: mesh_device}

    @classmethod
    def create_state(cls, hf_config: PretrainedConfig, mesh_device: Any, ccl: Any) -> dict[str, Any]:
        del hf_config, mesh_device, ccl
        return {}

    @classmethod
    def model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        fabric_config: ttnn.FabricConfig,
        mode: str,
        batch_size_per_row: int,
        topk_fallback: bool = False,
    ) -> dict[str, Any]:
        del topk_fallback
        return {
            "mesh_device": MeshDeviceStub(mesh_device.shape),
            "num_devices": mesh_device.get_num_devices(),
            "fabric_config": fabric_config,
            "hidden_size": hf_config.hidden_size,
            "mistral_hf_config": hf_config,
            "input_memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "output_memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "num_dispatch_devices": mesh_device.shape[0],
            "batch_size_per_row": batch_size_per_row,
            "mode": mode,
        }

    @classmethod
    def decode_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        fabric_config: ttnn.FabricConfig,
        batch_size_per_row: int,
        topk_fallback: bool = False,
    ) -> dict[str, Any]:
        return cls.model_config(
            hf_config,
            mesh_device,
            fabric_config,
            "decode",
            batch_size_per_row=batch_size_per_row,
            topk_fallback=topk_fallback,
        )

    @classmethod
    def prefill_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        fabric_config: ttnn.FabricConfig,
        topk_fallback: bool = False,
    ) -> dict[str, Any]:
        return cls.model_config(
            hf_config,
            mesh_device,
            fabric_config,
            "prefill",
            batch_size_per_row=1,
            topk_fallback=topk_fallback,
        )

    @classmethod
    def _bridge_host_forward(cls, x: ttnn.Tensor, cfg: dict[str, Any]) -> ttnn.Tensor:
        mesh_device = cfg["mesh_device"]
        hf_cfg = cfg["mistral_hf_config"]
        assert isinstance(hf_cfg, Mistral4Config)
        state_dict = cfg["bridge_torch_state_dict"]

        composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape))
        x_torch = ttnn.to_torch(x, mesh_composer=composer)

        while x_torch.dim() > 3 and x_torch.shape[1] == 1:
            x_torch = x_torch.squeeze(1)

        mcfg = deepcopy(hf_cfg)
        _experts_impl_config(mcfg)
        ref = Mistral4MoE(mcfg).eval()
        ref.load_state_dict(state_dict, strict=False)
        # Host MoE with ``grouped_mm`` must not mix bf16 weights with fp32 activations (router softmax /
        # top-k weights are often fp32). Run the reference in fp32, then cast back to bf16 for mesh I/O.
        ref = ref.to(dtype=torch.float32)
        x_in = x_torch.to(dtype=torch.float32)
        with torch.no_grad():
            y = ref(x_in)
        y = y.to(torch.bfloat16)

        return ttnn.from_torch(
            y.unsqueeze(1),
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
            dtype=ttnn.bfloat16,
            memory_config=cfg["output_memory_config"],
            layout=ttnn.TILE_LAYOUT,
        )

    @classmethod
    def forward_prefill(
        cls,
        x: ttnn.Tensor,
        cfg: dict[str, Any],
        handle_tensor_parallel: bool = False,
    ) -> ttnn.Tensor:
        del handle_tensor_parallel
        return cls._bridge_host_forward(x, cfg)

    @classmethod
    def forward_decode(
        cls,
        x: ttnn.Tensor,
        cfg: dict[str, Any],
        handle_tensor_parallel: bool = False,
    ) -> ttnn.Tensor:
        del handle_tensor_parallel
        return cls._bridge_host_forward(x, cfg)

    def load_from_model_dir(
        self,
        model_dir: str | Path,
        layer_idx: int,
        *,
        strict: bool = False,
    ):
        """Load MoE tensors for ``layer_idx`` from sharded ``safetensors``."""
        return load_ttmistral4_moe_from_sharded_safetensors(self, model_dir, layer_idx, strict=strict)
