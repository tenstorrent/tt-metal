# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""LoRA on the ttml MiniMax-H3 transformer: injection, PEFT-style init, diffusers-layout adapter files,
gradient norms over TP shards, EMA shadows and the masked flow-matching loss."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import ttnn
import ttml
from safetensors.numpy import load_file, save_file
from ttml.modules import LoraConfig, LoraModel

from ttml.models.minimax_h3 import H3Config, H3Inputs, upload
from ttml.models.minimax_h3.weights import interleave_halves_per_shard, permute_qk_rows, unpermute_qk_rows

LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"
# The 50 transformer blocks only: attention projections and the SwiGLU feed-forward.
TARGETS = [r"(?:^|\.)blocks\.\d+\.attn\.(to_q|to_k|to_v|to_out)$", r"(?:^|\.)blocks\.\d+\.ff\.(proj|out)$"]
_LEAF_RENAMES = [
    (re.compile(r"attn\.to_out$"), "attn.to_out.0"),
    (re.compile(r"ff\.proj$"), "ff.net.0.proj"),
    (re.compile(r"ff\.out$"), "ff.net.2"),
]
_SLOT_RE = re.compile(r"^(?P<base>.+)\.(?P<slot>lora_A|lora_B)$")
_NATIVE = ttml.autograd.PreferredPrecision.NATIVE


def inject(model, rank: int, alpha: int, dropout: float = 0.0, verbose: bool = False) -> LoraModel:
    return LoraModel(
        model, LoraConfig(rank=rank, alpha=alpha, target_modules=TARGETS, lora_dropout=dropout, verbose=verbose)
    )


def lora_params(lora: LoraModel) -> dict[str, ttml.autograd.Tensor]:
    """C++-style names -> tensors, the subset that trains (what the optimizer takes)."""
    return {n: t for n, t in lora.parameters().items() if t.get_requires_grad()}


def named_lora(lora: LoraModel):
    for name, tensor in lora.named_parameters():
        if name.endswith(("lora_A", "lora_B")):
            yield name[len("model.") :] if name.startswith("model.") else name, tensor


def to_diffusers_key(name: str) -> str:
    m = _SLOT_RE.match(name)
    if m is None:
        raise ValueError(f"not a LoRA parameter: {name!r}")
    base, slot = m.group("base"), m.group("slot")
    for pattern, repl in _LEAF_RENAMES:
        base, n = pattern.subn(repl, base)
        if n:
            break
    return f"transformer_{base}.{slot}.weight"  # blocks.N... -> transformer_blocks.N...


def _tp() -> int:
    mesh = ttml.maybe_mesh()
    return mesh.axis_size("tp") if mesh is not None else 1


def gather(tensor: ttml.autograd.Tensor) -> np.ndarray:
    s = ttml.Sharding.from_tensor(tensor)
    return np.asarray(s.gather(tensor)).astype(np.float32)


def gather_grad(tensor: ttml.autograd.Tensor) -> np.ndarray:
    tmp = ttml.autograd.create_tensor(tensor.get_grad(), False)
    return gather(tmp)


def assign_like(param: ttml.autograd.Tensor, value: np.ndarray) -> None:
    """Upload `value` (global 4-D) with the parameter's own mesh placement."""
    mapper = ttml.Sharding.from_tensor(param).derive_mapper()
    param.set_value(
        ttml.autograd.Tensor.from_numpy(
            np.ascontiguousarray(value, dtype=np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper
        ).get_value()
    )


def device_to_hf(name: str, arr: np.ndarray, config: H3Config) -> np.ndarray:
    """Device row order -> diffusers row order for the B matrices whose base rows were re-laid at load."""
    if name.endswith("lora_B") and (".to_q." in name or ".to_k." in name):
        return unpermute_qk_rows(arr, config.num_attention_heads, config.attention_head_dim, config.rotary_dim)
    if name.endswith("ff.proj.lora_B"):
        tp = _tp()
        per = arr.shape[0] // (2 * tp)
        blocks = arr.reshape(tp, 2, per, arr.shape[1])
        return np.concatenate([blocks[:, 0].reshape(-1, arr.shape[1]), blocks[:, 1].reshape(-1, arr.shape[1])])
    return arr


def hf_to_device(name: str, arr: np.ndarray, config: H3Config) -> np.ndarray:
    if name.endswith("lora_B") and (".to_q." in name or ".to_k." in name):
        return permute_qk_rows(arr, config.num_attention_heads, config.attention_head_dim, config.rotary_dim)
    if name.endswith("ff.proj.lora_B"):
        return interleave_halves_per_shard(arr, _tp())
    return arr


def init_lora_A_gaussian(lora: LoraModel, rank: int, seed: int) -> int:
    """PEFT `init_lora_weights="gaussian"`: A ~ N(0, 1/rank), B = 0 (ttml's B is already zero)."""
    rng = np.random.default_rng(seed)
    n = 0
    for name, tensor in named_lora(lora):
        if name.endswith("lora_A"):
            shape = gather(tensor).shape
            assign_like(tensor, rng.normal(0.0, 1.0 / rank, size=shape).astype(np.float32))
            n += 1
    return n


def state_dict(lora: LoraModel, config: H3Config) -> dict[str, np.ndarray]:
    out = {}
    for name, tensor in named_lora(lora):
        arr = gather(tensor)
        arr = arr.reshape(arr.shape[-2], arr.shape[-1])
        out[to_diffusers_key(name)] = np.ascontiguousarray(device_to_hf(name, arr, config))
    return out


def save_adapter(lora: LoraModel, config: H3Config, folder: Path, shadows: dict[str, ttnn.Tensor] | None = None) -> int:
    """Diffusers-layout adapter at `folder/pytorch_lora_weights.safetensors`; with `shadows`, the EMA weights."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if shadows is None:
        state = state_dict(lora, config)
    else:
        state = {}
        for name, tensor in named_lora(lora):
            tmp = ttml.autograd.create_tensor(shadows[name], False)
            arr = gather(tmp)
            arr = arr.reshape(arr.shape[-2], arr.shape[-1])
            state[to_diffusers_key(name)] = np.ascontiguousarray(device_to_hf(name, arr, config))
    save_file(state, str(folder / LORA_WEIGHT_NAME))
    return len(state)


def load_adapter(lora: LoraModel, config: H3Config, folder: Path) -> int:
    state = load_file(str(Path(folder) / LORA_WEIGHT_NAME))
    n = 0
    for name, tensor in named_lora(lora):
        key = to_diffusers_key(name)
        if key not in state:
            raise KeyError(f"{key} missing (ttml {name})")
        assign_like(tensor, hf_to_device(name, state[key].astype(np.float32), config)[None, None])
        n += 1
    return n


def grad_norms(lora: LoraModel) -> tuple[float, dict[str, float]]:
    """Global L2 norm over every LoRA gradient (TP shards gathered) and the per-tensor norms."""
    per = {}
    for name, tensor in named_lora(lora):
        if not tensor.is_grad_initialized():
            per[name] = 0.0
            continue
        g = gather_grad(tensor)
        per[name] = float(np.sqrt((g.astype(np.float64) ** 2).sum()))
    return float(np.sqrt(sum(v * v for v in per.values()))), per


def _is_sharded_grad(name: str) -> bool:
    """Row-parallel layers shard lora_A, column-parallel ones shard lora_B; the other factor is replicated."""
    row = ".attn.to_out." in name or ".ff.out." in name
    return name.endswith("lora_A") if row else name.endswith("lora_B")


def clip_grad_norm(lora: LoraModel, max_norm: float) -> float:
    """Global L2 norm over all LoRA grads, computed on device, then scaled to `max_norm` like torch's
    `clip_grad_norm_`. Sharded grads are summed over TP devices, replicated ones counted once. Returns the pre-clip
    norm. `max_norm <= 0` only measures."""
    device = ttml.autograd.AutoContext.get_instance().get_device()
    tp = _tp()
    acc = {True: None, False: None}
    params = []
    for name, tensor in named_lora(lora):
        if not tensor.is_grad_initialized():
            continue
        g = tensor.get_grad()
        sq = ttnn.sum(ttnn.multiply(g, g), dim=(0, 1, 2, 3), keepdim=True)  # (1, 1, 1, 1) per device
        key = _is_sharded_grad(name) and tp > 1
        acc[key] = sq if acc[key] is None else ttnn.add(acc[key], sq)
        params.append(tensor)
    total = 0.0
    for sharded, value in acc.items():
        if value is None:
            continue
        host = (
            np.asarray(
                ttml.autograd.create_tensor(value, False).to_numpy(
                    composer=ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0), precision=_NATIVE
                )
            )
            .astype(np.float64)
            .reshape(-1)
        )
        total += host.sum() if sharded else host[0]
    norm = float(np.sqrt(total))
    if max_norm > 0 and norm > max_norm:
        scale = max_norm / (norm + 1e-6)
        for tensor in params:
            tensor.set_grad(ttnn.multiply(tensor.get_grad(), scale))
    return norm


class Ema:
    """fp32 shadow copies of the LoRA weights on device; `decay = min(max_decay, (1 + step) / (10 + step))`.

    fp32 because bf16 shadows stop moving near decay 0.999; the lerp weight is a device tensor because a new
    Python float every step would recompile the kernel."""

    def __init__(self, lora: LoraModel, max_decay: float) -> None:
        self.max_decay = max_decay
        self.shadows = {
            name: ttnn.typecast(tensor.get_value(), ttnn.DataType.FLOAT32) for name, tensor in named_lora(lora)
        }
        self._lora = lora

    def update(self, step: int) -> float:
        decay = min(self.max_decay, (1 + step) / (10 + step))
        weight = upload(np.full((1, 1, 1, 1), 1.0 - decay), ttnn.DataType.FLOAT32).get_value()
        for name, tensor in named_lora(self._lora):
            s = self.shadows[name]
            ttnn.lerp(s, ttnn.typecast(tensor.get_value(), ttnn.DataType.FLOAT32), weight, output_tensor=s)
        return decay


def masked_mse(
    pred: ttml.autograd.Tensor, target_full: ttml.autograd.Tensor, row_mask: ttml.autograd.Tensor, inputs: H3Inputs
) -> ttml.autograd.Tensor:
    """MSE over the video rows only, normalised like the GPU's `mse_loss(pred[video], target)`.

    `pred * row_mask` zeroes the non-video rows and `target_full` is zero there, so the mean over all
    S_pad rows equals the video-row mean times V / S_pad."""
    loss = ttml.ops.loss.mse_loss(pred * row_mask, target_full, ttml.ops.ReduceType.MEAN)
    return loss * (inputs.S_pad / inputs.V)
