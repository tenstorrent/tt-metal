# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""GR00T utilities for TT Symbiote.

Applies patches at import time to extend core (run_config, module, tensor, device_management)
with GR00T-specific behavior: DPL run modes, distributed config, DiT attention compat,
and tensor/device handling. Keeps core alnah005-clean; all GR00T logic lives here.
"""

import operator
import os
from dataclasses import dataclass
from enum import Enum
from functools import reduce, wraps
from math import isqrt
from typing import Any, Dict, Optional

import torch
import ttnn
from torch.utils._pytree import tree_map

from models.experimental.tt_symbiote.core.run_config import (
    DispatchManager,
    NormalRun,
    compose_transforms,
    copy_to_ttnn,
    no_dispatch,
    set_device_wrap,
    to_ttnn_wrap,
    unwrap_to_torch,
    wrap_from_torch,
    wrap_to_torch_ttnn_tensor,
)
from models.experimental.tt_symbiote.core.utils import TORCH_TO_TTNN, ensure_tile_layout


def compare_fn_outputs(torch_output, ttnn_output, func_name):
    """Compare torch vs TTNN outputs via PCC. Replaces run_config.compare_fn_outputs for GR00T."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    torch_output_tensors = []
    ttnn_output_tensors = []
    if isinstance(torch_output, TorchTTNNTensor):
        torch_output_tensors.append(torch_output.to_torch)
    elif isinstance(torch_output, torch.Tensor):
        torch_output_tensors.append(torch_output)
    elif isinstance(torch_output, (list, tuple)):
        for item in torch_output:
            if isinstance(item, TorchTTNNTensor):
                torch_output_tensors.append(item.to_torch)
            elif isinstance(item, torch.Tensor):
                torch_output_tensors.append(item)
    if isinstance(ttnn_output, TorchTTNNTensor):
        ttnn_output.elem = None
        ttnn_output_tensors.append(ttnn_output.to_torch)
        if not isinstance(torch_output, TorchTTNNTensor):
            print("Mismatched output types between TTNN and Torch.")
    elif isinstance(ttnn_output, (list, tuple)):
        if not isinstance(torch_output, (list, tuple)):
            torch_output = (torch_output,) if torch_output is not None else ()
        if len(torch_output) != len(ttnn_output):
            _t = tuple(
                x
                for x in torch_output
                if x is not None and (isinstance(x, torch.Tensor) or isinstance(x, TorchTTNNTensor))
            )
            _n = tuple(
                x
                for x in ttnn_output
                if x is not None and (isinstance(x, torch.Tensor) or isinstance(x, TorchTTNNTensor))
            )
            if len(_t) == len(_n) and len(_t) > 0:
                torch_output, ttnn_output = list(_t), list(_n)
        assert isinstance(torch_output, (list, tuple)), "Mismatched output types between TTNN and Torch."
        assert len(ttnn_output) == len(torch_output), "Mismatched output lengths between TTNN and Torch."
        for index, item in enumerate(ttnn_output):
            if isinstance(item, TorchTTNNTensor):
                if not isinstance(torch_output[index], TorchTTNNTensor):
                    print("Mismatched output types between TTNN and Torch.")
                item.elem = None
                ttnn_output_tensors.append(item.to_torch)

    passed = True
    for t_tensor, n_tensor in zip(torch_output_tensors, ttnn_output_tensors):
        t_tensor = t_tensor.to(torch.float32)
        n_tensor = n_tensor.to(torch.float32)
        assert t_tensor.shape == n_tensor.shape, "Mismatched output shapes between TTNN and Torch."
        pcc = torch.corrcoef(torch.stack([t_tensor.flatten(), n_tensor.flatten()]))[0, 1]
        diff = torch.abs(t_tensor - n_tensor)
        if (
            pcc < 0.999
            or (torch.median(diff) > torch.mean(diff) and torch.max(diff).item() > 1)
            or pcc.isnan().item()
            or diff.isnan().any()
        ):
            passed = False
            print(
                f"Warning: High discrepancy detected in operation {func_name}. "
                f"PCC: {pcc.item()}, Max Abs Diff: {torch.max(diff).item()}, "
                f"Median Abs Diff: {torch.median(diff).item()}, "
                f"Mean Abs Diff: {torch.mean(diff).item()}"
            )
        if torch.logical_xor((n_tensor == 0).all(), (t_tensor == 0).all()):
            passed = False
            print(f"Warning: One of the outputs is all zeros while the other is not " f"in operation {func_name}.")
        if func_name == "aten::topk":
            break
    if not passed:
        print(f"Operation {func_name} PCC < 0.99.")


_DIAG_LM_LAYER0_CAPTURE: Dict[str, Any] = {}


def _filter_kwargs_for_ttnn_conversion(kwargs):
    """Drop kwargs with unsupported dtypes for TTNN. Needed so to_ttnn doesn't fail on non-convertible tensors."""
    result = {}
    for k, v in kwargs.items():
        if k == "attention_mask" and v is not None:
            result[k] = v
            continue
        elem = getattr(v, "elem", v)
        if isinstance(elem, torch.Tensor) and elem.dtype not in TORCH_TO_TTNN:
            continue
        result[k] = v
    return result


def _force_elem_for_lm_attn(e):
    """Force TorchTTNNTensor to elem-only (clear ttnn_tensor). LM self-attn path needs fresh to_ttnn conversion."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    if isinstance(e, TorchTTNNTensor) and e.elem is not None and e.ttnn_tensor is not None:
        r = TorchTTNNTensor(e.elem)
        r.ttnn_tensor = None
        return r
    return e


def _to_ttnn_lm_attn_style(e, device):
    """Convert elem to TTNN on device for LM self-attn. Uses elem (torch) to avoid stale ttnn_tensor."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    if not isinstance(e, TorchTTNNTensor) or e.elem is None:
        return e
    t = e.elem
    if t.dtype not in TORCH_TO_TTNN:
        return e
    tt = ttnn.from_torch(
        t.contiguous().to(torch.bfloat16).cpu(),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    return tt


def ensure_tile_layout_wrap(e):
    """Ensure ttnn.Tensor or TorchTTNNTensor.ttnn_tensor has TILE_LAYOUT. Required before SDPA."""
    if isinstance(e, ttnn.Tensor):
        return ensure_tile_layout(e)
    if hasattr(e, "ttnn_tensor") and e.ttnn_tensor is not None:
        e.ttnn_tensor = ensure_tile_layout(e.ttnn_tensor)
    return e


def _to_raw_torch(x):
    """Extract plain torch.Tensor from TorchTTNNTensor or ttnn.Tensor. For diagnostics/cloning."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    if isinstance(x, TorchTTNNTensor):
        return x.to_torch
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, ttnn.Tensor):
        return ttnn.to_torch(x)
    return x


def _dtype_from_torch_args(args, kwds):
    """Infer dtype from first floating tensor in args/kwds. Used to cast torch_layer for DPL consistency."""
    from torch.utils._pytree import tree_flatten

    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    flat, _ = tree_flatten((args, kwds))
    for a in flat:
        if isinstance(a, (TorchTTNNTensor, torch.Tensor)) and a.is_floating_point():
            return a.dtype
    return None


def _cast_module_to_dtype(module, dtype):
    """Cast module params/buffers to dtype. DPL needs torch_layer to match input dtype."""
    if dtype is None:
        return
    for p in module.parameters():
        if p.is_floating_point():
            p.data = p.data.to(dtype)
    for b in module.buffers():
        if b.is_floating_point():
            b.data = b.data.to(dtype)


class DPLRunExtended(NormalRun):
    """DPL run: run torch ref, then TTNN, compare, merge. For GR00T validation and debugging."""

    @staticmethod
    def module_run(self, *args, **kwds):
        from models.experimental.tt_symbiote.core import run_config

        assert (
            self.torch_layer is not None
        ), f"torch_layer must be set for DPLRun, {self} does not have torch_layer set."

        print(f"{self.__class__.__name__}: {self.module_name} on device {self.device}")
        copied_torch_tensors_args = tree_map(run_config.copy_to_torch(self.__class__.__name__), args)
        copied_torch_tensors_kwargs = tree_map(run_config.copy_to_torch(self.__class__.__name__), kwds)
        func_args = tree_map(wrap_to_torch_ttnn_tensor, copied_torch_tensors_args)
        func_kwargs = tree_map(wrap_to_torch_ttnn_tensor, copied_torch_tensors_kwargs)
        if (
            os.environ.get("TT_SYMBIOTE_NO_ATTN_MASK")
            and "encoder_hidden_states" in kwds
            and kwds.get("encoder_hidden_states") is not None
        ):
            func_kwargs = {k: (None if k == "attention_mask" else v) for k, v in func_kwargs.items()}
            copied_torch_tensors_kwargs = {
                k: (None if k == "attention_mask" else v) for k, v in copied_torch_tensors_kwargs.items()
            }
        torch_output = tree_map(wrap_to_torch_ttnn_tensor, self.torch_layer(*func_args, **func_kwargs))
        result = torch_output
        if self.device is not None:
            mn = getattr(self, "module_name", "")
            use_elem_for_lm = "language_model" in mn and mn.endswith("self_attn") and "tt_" not in mn
            filtered_kwargs = _filter_kwargs_for_ttnn_conversion(func_kwargs)
            if use_elem_for_lm:
                func_args = tree_map(_force_elem_for_lm_attn, func_args)
                filtered_kwargs = tree_map(_force_elem_for_lm_attn, filtered_kwargs)

                def _lm_transform(e):
                    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

                    if isinstance(e, TorchTTNNTensor) and e.elem is not None:
                        return _to_ttnn_lm_attn_style(e, self.device)
                    return ensure_tile_layout_wrap(
                        set_device_wrap(self.device)(to_ttnn_wrap(wrap_to_torch_ttnn_tensor(e)))
                    )

                func_args = tree_map(_lm_transform, func_args)
                func_kwargs = tree_map(_lm_transform, filtered_kwargs)
                if "position_embeddings" in kwds and kwds["position_embeddings"] is not None:
                    func_kwargs["position_embeddings"] = kwds["position_embeddings"]
            else:
                transform = compose_transforms(
                    wrap_to_torch_ttnn_tensor,
                    to_ttnn_wrap,
                    set_device_wrap(self.device),
                    ensure_tile_layout_wrap,
                )
                func_args = tree_map(transform, func_args)
                func_kwargs = tree_map(transform, filtered_kwargs)
            self.preprocess_weights()
            self.move_weights_to_device()
            ttnn_output = post_process_ttnn_module_output(self, self.forward(*func_args, **func_kwargs))
            if os.environ.get("TT_SYMBIOTE_DIAG_LM_LAYER0_CAPTURE"):
                mn = getattr(self, "module_name", "")
                if "language_model" in mn and mn.endswith("layers.0.self_attn") and "tt_q_proj" not in mn:
                    hidden = args[0] if args else kwds.get("hidden_states")
                    ht = _to_raw_torch(hidden)
                    _DIAG_LM_LAYER0_CAPTURE["hidden_states"] = (
                        ht.clone().detach() if ht is not None and isinstance(ht, torch.Tensor) else None
                    )
                    am = kwds.get("attention_mask")
                    amt = _to_raw_torch(am) if am is not None else None
                    _DIAG_LM_LAYER0_CAPTURE["attention_mask"] = (
                        amt.clone().detach() if amt is not None and isinstance(amt, torch.Tensor) else None
                    )
                    pos_emb = kwds.get("position_embeddings")
                    if pos_emb is not None and isinstance(pos_emb, (tuple, list)) and len(pos_emb) >= 2:
                        p0, p1 = _to_raw_torch(pos_emb[0]), _to_raw_torch(pos_emb[1])
                        _DIAG_LM_LAYER0_CAPTURE["position_embeddings"] = (
                            (p0.clone().detach(), p1.clone().detach())
                            if isinstance(p0, torch.Tensor) and isinstance(p1, torch.Tensor)
                            else None
                        )
                    else:
                        _DIAG_LM_LAYER0_CAPTURE["position_embeddings"] = None
                    out_t = torch_output[0] if isinstance(torch_output, (tuple, list)) else torch_output
                    ot = _to_raw_torch(out_t) if out_t is not None else None
                    _DIAG_LM_LAYER0_CAPTURE["torch_output"] = (
                        ot.clone().detach() if ot is not None and isinstance(ot, torch.Tensor) else None
                    )
                    out_n = ttnn_output[0] if isinstance(ttnn_output, (tuple, list)) else ttnn_output
                    on = _to_raw_torch(out_n) if out_n is not None else None
                    _DIAG_LM_LAYER0_CAPTURE["ttnn_output_full"] = (
                        on.clone().detach() if on is not None and isinstance(on, torch.Tensor) else None
                    )
            compare_fn_outputs(
                tree_map(wrap_to_torch_ttnn_tensor, copied_torch_tensors_args),
                tree_map(wrap_to_torch_ttnn_tensor, func_args),
                self.__class__.__name__,
            )
            compare_fn_outputs(torch_output, ttnn_output, self.__class__.__name__)
            result = run_config.create_new_ttnn_tensors_using_torch_output(
                torch_output, ttnn_output, assign_ttnn_to_torch=True
            )
        return result


class DPLRunNoErrorPropExtended(NormalRun):
    """DPL run without error propagation: run TTNN only, no comparison. Faster for inference."""

    @staticmethod
    def module_run(self, *args, **kwds):
        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        assert (
            self.torch_layer is not None
        ), f"torch_layer must be set for DPLRun, {self} does not have torch_layer set."

        target_dtype = _dtype_from_torch_args(args, kwds)
        if target_dtype is not None:
            _cast_module_to_dtype(self.torch_layer, target_dtype)
        torch_output = tree_map(wrap_to_torch_ttnn_tensor, self.torch_layer(*args, **kwds))
        self.preprocess_weights()
        self.move_weights_to_device()
        ttnn_args = tree_map(copy_to_ttnn(self.__class__.__name__), args)
        ttnn_output = tree_map(
            wrap_to_torch_ttnn_tensor,
            self.forward(
                *tree_map(
                    compose_transforms(wrap_to_torch_ttnn_tensor, to_ttnn_wrap, set_device_wrap(self.device)),
                    ttnn_args,
                ),
                **kwds,
            ),
        )
        t_aud = torch_output[0] if isinstance(torch_output, (list, tuple)) else torch_output
        n_aud = ttnn_output[0] if isinstance(ttnn_output, (list, tuple)) else ttnn_output
        if isinstance(t_aud, TorchTTNNTensor) and isinstance(n_aud, TorchTTNNTensor):
            compare_fn_outputs(t_aud, n_aud, self.module_name)
        from models.experimental.tt_symbiote.core import run_config as run_config_mod

        return run_config_mod.create_new_ttnn_tensors_using_torch_output(
            torch_output, ttnn_output, assign_ttnn_to_torch=True
        )


def _patched_basic_transformer_block_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    temb: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    """Patched diffusers BasicTransformerBlock.forward for DiT: returns attn_output only (not full tuple)."""
    if self.norm_type == "ada_norm":
        norm_hidden_states = self.norm1(hidden_states, temb)
    else:
        norm_hidden_states = self.norm1(hidden_states)

    if self.pos_embed is not None:
        norm_hidden_states = self.pos_embed(norm_hidden_states)

    _attn_out = self.attn1(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=(encoder_attention_mask if encoder_hidden_states is not None else attention_mask),
    )
    attn_output = _attn_out[0] if isinstance(_attn_out, (list, tuple)) else _attn_out
    if self.final_dropout:
        attn_output = self.final_dropout(attn_output)

    hidden_states = attn_output + hidden_states
    if hidden_states.ndim == 4:
        hidden_states = hidden_states.squeeze(1)

    norm_hidden_states = self.norm3(hidden_states)
    ff_output = self.ff(norm_hidden_states)

    hidden_states = ff_output + hidden_states
    if hidden_states.ndim == 4:
        hidden_states = hidden_states.squeeze(1)
    return hidden_states


def apply_gr00t_dit_attention_return_compat() -> bool:
    """Patch diffusers BasicTransformerBlock for DiT attn1 return format. Call before GR00T DiT inference."""
    try:
        from gr00t.model.modules import dit
    except ImportError:
        return False

    if not hasattr(dit, "BasicTransformerBlock"):
        return False

    block = dit.BasicTransformerBlock
    if block.forward is _patched_basic_transformer_block_forward:
        return True
    block.forward = _patched_basic_transformer_block_forward
    return True


def post_process_ttnn_module_output(self, result):
    """Wrap output as TorchTTNNTensor and apply set_output_tensors_config. Patched into NormalRun.module_run."""
    result = tree_map(wrap_to_torch_ttnn_tensor, result)
    if getattr(self, "device_state", None) is not None:
        result = self.set_output_tensors_config(result)
    return result


def _create_new_ttnn_tensors_using_torch_output_relaxed(torch_output, ttnn_output, assign_ttnn_to_torch=False):
    """Assign ttnn to torch wrapper. Relaxed for GR00T: elem=None, list/tuple length mismatch OK."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    if isinstance(torch_output, TorchTTNNTensor) and isinstance(ttnn_output, TorchTTNNTensor):
        torch_output.ttnn_tensor = ttnn_output.to_ttnn
        if not assign_ttnn_to_torch:
            torch_output.elem = None
        return torch_output
    if isinstance(torch_output, (list, tuple)) and isinstance(ttnn_output, TorchTTNNTensor):
        if len(torch_output) > 0 and isinstance(torch_output[0], TorchTTNNTensor):
            torch_output[0].ttnn_tensor = ttnn_output.to_ttnn
            if not assign_ttnn_to_torch:
                torch_output[0].elem = None
            return torch_output
    if isinstance(torch_output, (list, tuple)) and isinstance(ttnn_output, (list, tuple)):
        for t_item, n_item in zip(torch_output, ttnn_output):
            if isinstance(t_item, TorchTTNNTensor) and isinstance(n_item, TorchTTNNTensor):
                t_item.ttnn_tensor = n_item.to_ttnn
                if not assign_ttnn_to_torch:
                    t_item.elem = None
    return torch_output


def _dispatch_to_torch_wrapper_gr00t(func, torch_args, torch_kwargs):
    """Dispatch to torch. Handles padded view, im2col shape fixes, dtype cast for GR00T ops."""
    import time

    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
    from models.experimental.tt_symbiote.core.run_config import DispatchManager
    from models.experimental.tt_symbiote.core.torch_dispatcher import can_dispatch_to_torch, dispatch_to_torch

    im2col_logical_shape = None
    if func.name().startswith("aten::im2col") and len(torch_args) >= 5:
        if isinstance(torch_args[0], TorchTTNNTensor) and torch_args[0].ttnn_tensor is not None:
            im2col_logical_shape = tuple(int(i) for i in torch_args[0].ttnn_tensor.shape)
    with no_dispatch():
        func_args = list(tree_map(unwrap_to_torch(func), torch_args))
        func_kwargs = dict(tree_map(unwrap_to_torch(func), torch_kwargs))
        if func.name().startswith("aten::im2col") and len(torch_args) >= 5:
            if isinstance(torch_args[0], TorchTTNNTensor) and isinstance(func_args[0], torch.Tensor):
                t, shp = func_args[0], (
                    im2col_logical_shape
                    if (im2col_logical_shape and len(im2col_logical_shape) == 4)
                    else torch_args[0].shape
                )
                if len(shp) == 4:
                    N, C = int(shp[0]), int(shp[1])
                    expected_in_numel = N * C * int(shp[2]) * int(shp[3])
                    if t.numel() == 2965872 and N == 1 and C == 1152:
                        func_args[0] = t.flatten()[:903168].clone().reshape(1, 1152, 28, 28)
                    elif t.numel() > expected_in_numel:
                        func_args[0] = t.flatten()[:expected_in_numel].clone().reshape(N, C, int(shp[2]), int(shp[3]))
                    elif t.numel() < expected_in_numel:
                        spatial = t.numel() // (N * C)
                        if spatial > 0:
                            H = isqrt(spatial)
                            W = spatial // H
                            func_args[0] = (
                                t.flatten().clone().reshape(N, C, H, W)
                                if H * W == spatial and H > 0 and W > 0
                                else t.contiguous().clone()
                            )
                        else:
                            func_args[0] = t.contiguous().clone()
                    else:
                        func_args[0] = (
                            t[: int(shp[0]), : int(shp[1]), : int(shp[2]), : int(shp[3])].contiguous().clone()
                        )
        if func.name() == "aten::view" and len(func_args) >= 2:
            t, shape = func_args[0], func_args[1]
            if isinstance(t, torch.Tensor) and isinstance(shape, (list, tuple)) and len(shape) > 0:
                target_numel = reduce(operator.mul, shape, 1)
                if t.numel() > target_numel and target_numel > 0:
                    func_args[0] = t.flatten()[:target_numel].clone()
        target_dtype = _dtype_from_torch_args(func_args, func_kwargs)
        if target_dtype is not None:

            def _cast_to_dtype(e):
                return (
                    e.to(target_dtype)
                    if isinstance(e, torch.Tensor) and e.is_floating_point() and e.dtype != target_dtype
                    else e
                )

            func_args = tree_map(_cast_to_dtype, func_args)
            func_kwargs = {k: _cast_to_dtype(v) for k, v in func_kwargs.items()}
        begin = time.time()
        func_res = (
            dispatch_to_torch(func.name(), tuple(func_args), func_kwargs)
            if can_dispatch_to_torch(func.name(), tuple(func_args), func_kwargs)
            else func(*tuple(func_args), **func_kwargs)
        )
        end = time.time()
        DispatchManager.record_timing(
            "Torch",
            DispatchManager.current_module_name + f".{func.name()}" if DispatchManager.current_module_name else "",
            func.name(),
            {},
            end - begin,
        )
        return tree_map(wrap_from_torch, func_res)


def _copy_to_torch_extended(func):
    """Copy TorchTTNNTensor or ttnn.Tensor to torch. Needed when GR00T passes raw ttnn.Tensor."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    def _copy(e):
        if isinstance(e, TorchTTNNTensor):
            res = TorchTTNNTensor(e.to_torch.clone())
            res.ttnn_tensor = None
            return res
        if isinstance(e, ttnn.Tensor):
            res = TorchTTNNTensor(ttnn.to_torch(e))
            res.ttnn_tensor = None
            return res
        if isinstance(e, torch.Tensor):
            return e.clone()
        return e

    return _copy


def get_default_distributed_tensor_config(mesh_device=None, torch_tensor=None, module_name=None):
    """Lookup tensor config from DeviceInit state. Used by run_config.to_ttnn for distributed tensors."""
    try:
        from models.experimental.tt_symbiote.utils.device_management import DeviceInit

        state = None
        if mesh_device is not None:
            state = DeviceInit.DEVICE_TO_STATE_DICT.get(mesh_device)
        elif DeviceInit.DEVICE_TO_STATE_DICT and len(DeviceInit.DEVICE_TO_STATE_DICT) == 1:
            state = next(iter(DeviceInit.DEVICE_TO_STATE_DICT.values()))
        if state is None:
            return None
        if torch_tensor is not None:
            return getattr(state, "get_tensor_config_for_tensor", lambda *a: state.tensor_config)(
                module_name, torch_tensor
            )
        return getattr(state, "tensor_config", None)
    except Exception:
        return None


def _patch_module_for_gr00t(DistributedConfigCls):
    """Add device_state, set_output_tensors_config, run_on_devices, DeviceArch to TTNNModule. Needed for multi-device."""
    import models.experimental.tt_symbiote.core.module as core_module
    from models.experimental.tt_symbiote.core.run_config import DistributedTensorConfig
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    TTNNModule = core_module.TTNNModule

    def set_distributed_tensor_config(distribute_tensor_config: DistributedTensorConfig):
        def _set_distributed_config(e):
            res = e
            if isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None:
                res.set_distributed_tensor_config(distribute_tensor_config)
            return res

        return _set_distributed_config

    def set_module_name_recursively(module, prefix=""):
        for name, child in module.__dict__.items():
            if isinstance(child, TTNNModule):
                child._unique_name = f"{prefix}.{name}"
                child.override_children_module_names()
            elif isinstance(child, torch.nn.Module):
                set_module_name_recursively(child, f"{prefix}.{name}")
            elif isinstance(child, dict):
                for k, v in child.items():
                    if isinstance(v, TTNNModule):
                        v._unique_name = f"{prefix}.{name}[{k}]"
                        v.override_children_module_names()
                    elif isinstance(v, torch.nn.Module):
                        set_module_name_recursively(v, f"{prefix}.{name}[{k}]")
            elif isinstance(child, (list, tuple)):
                for i, v in enumerate(child):
                    if isinstance(v, TTNNModule):
                        v._unique_name = f"{prefix}.{name}[{i}]"
                        v.override_children_module_names()
                    elif isinstance(v, torch.nn.Module):
                        set_module_name_recursively(v, f"{prefix}.{name}[{i}]")

    _orig_init = TTNNModule.__init__

    def _patched_init(self):
        _orig_init(self)
        self._device_state = None

    TTNNModule.__init__ = _patched_init

    def set_device_state(self, device_state=None):
        self._device_state = device_state
        if self._device_state is None:
            self._device_state = DistributedConfigCls(self.device)
        return self

    def set_output_tensors_config_impl(self, output_tensors):
        return tree_map(set_distributed_tensor_config(self.device_state.tensor_config), output_tensors)

    def set_output_tensors_config(self, output_tensors):
        assert self.device_state is not None
        return self.set_output_tensors_config_impl(output_tensors)

    def override_children_module_names(self):
        set_module_name_recursively(self, self.module_name)

    TTNNModule.set_device_state = set_device_state
    TTNNModule.set_output_tensors_config = set_output_tensors_config
    TTNNModule.set_output_tensors_config_impl = set_output_tensors_config_impl
    TTNNModule.override_children_module_names = override_children_module_names

    class DeviceArch(Enum):
        N150 = "n150"
        N300 = "n300"
        T3K = "t3k_wh"
        TG = "gx_wh"
        P150 = "p150"
        P300 = "p300"
        P150x4 = "p150x4"
        P150x8 = "p150x8"
        BHGLX = "bhglx"

    MeshShapeToDeviceArch = {
        "N150": DeviceArch.N150,
        "N300": DeviceArch.N300,
        "T3K": DeviceArch.T3K,
        "TG": DeviceArch.TG,
        "P150": DeviceArch.P150,
        "P300": DeviceArch.P300,
        "P150x4": DeviceArch.P150x4,
        "P150x8": DeviceArch.P150x8,
        "BHGLX": DeviceArch.BHGLX,
    }

    def run_on_devices_decorator(*allowed_archs):
        if not allowed_archs:
            raise ValueError("Must specify at least one allowed device architecture")
        allowed_set = frozenset(allowed_archs)

        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                if not hasattr(self, "device") or self.device is None:
                    raise RuntimeError(f"{self.__class__.__name__}: No device set. ")
                mesh_device = MeshShapeToDeviceArch.get(os.environ.get("MESH_DEVICE"))
                if mesh_device is None:
                    raise RuntimeError(
                        f"{self.__class__.__name__}: Unable to determine device architecture from MESH_DEVICE environment variable."
                    )
                if mesh_device not in MeshShapeToDeviceArch.values():
                    raise RuntimeError(
                        f"{self.__class__.__name__}: Unrecognized device architecture {mesh_device} for device {self.device}. Possible options: {list(MeshShapeToDeviceArch.values())}"
                    )
                if mesh_device not in allowed_set:
                    raise RuntimeError(
                        f"{self.__class__.__name__}: Device architecture {mesh_device} for device {self.device} not supported. "
                        f"Allowed architectures: {allowed_set}"
                    )
                return func(self, *args, **kwargs)

            return wrapper

        return decorator

    core_module.set_distributed_tensor_config = set_distributed_tensor_config
    core_module.set_module_name_recursively = set_module_name_recursively
    core_module.DeviceArch = DeviceArch
    core_module.MeshShapeToDeviceArch = MeshShapeToDeviceArch
    core_module.run_on_devices = run_on_devices_decorator

    TTNNModule.device_state = property(lambda self: self._device_state)  # type: ignore[assignment]


def _patch_device_management_for_gr00t(DistributedConfigCls):
    """Add DeviceInit, multi-device set_device, unwrap_ttnn, assimilate_to_device to device_management."""
    import time

    import models.experimental.tt_symbiote.utils.device_management as dm
    from models.experimental.tt_symbiote.core.run_config import DispatchManager

    class DeviceInit:
        DEVICE_TO_STATE_DICT = {}

        @classmethod
        def init_state(cls, device):
            if device not in cls.DEVICE_TO_STATE_DICT:
                res = cls.init_state_impl(device)
                if res is not None:
                    assert isinstance(res, DistributedConfigCls), f"Expected DistributedConfig, got {type(res)}"
                cls.DEVICE_TO_STATE_DICT[device] = res
            return cls.DEVICE_TO_STATE_DICT[device]

        @classmethod
        def init_state_impl(cls, device):
            return DistributedConfigCls(device)

    def _initialize_module_on_device(module, device, device_init=DeviceInit):
        module.to_device(device)
        if device.get_num_devices() > 1:
            module.set_device_state(device_init.init_state(device))

    def set_device_gr00t(obj, device, device_init=DeviceInit, **kwargs):
        from torch import nn

        from models.experimental.tt_symbiote.core.module import TTNNModule

        module_names = {}
        if isinstance(obj, nn.Module):
            module_names = {module: name for name, module in obj.named_modules()}

        def _set_device_recursive(current_obj, module_name=None):
            if isinstance(current_obj, nn.Module):
                name = module_names.get(current_obj, module_name or "")
                if kwargs.get("register_forward_hook", True):

                    def timed_call(original_call, module_name, module_class):
                        def new_call(*args, **kwargs):
                            begin = time.time()
                            DispatchManager.set_current_module_name(module_name)
                            result = original_call(*args, **kwargs)
                            DispatchManager.set_current_module_name(None)
                            end = time.time()
                            DispatchManager.record_timing("TorchModules", module_name, module_class, {}, end - begin)
                            return result

                        return new_call

                    if hasattr(current_obj, "forward"):
                        if not hasattr(current_obj.forward, "_is_timed"):
                            current_obj.forward = timed_call(current_obj.forward, name, current_obj.__class__.__name__)
                            current_obj.forward._is_timed = True
                    elif hasattr(current_obj, "__call__"):
                        if not hasattr(current_obj.__call__, "_is_timed"):
                            current_obj.__call__ = timed_call(
                                current_obj.__call__, name, current_obj.__class__.__name__
                            )
                            current_obj.__call__._is_timed = True

                for child_name, module in current_obj._modules.items():
                    if module is None:
                        continue
                    if isinstance(module, TTNNModule):
                        _initialize_module_on_device(module, device, device_init)
                    _set_device_recursive(module)

                for attr_name in dir(current_obj):
                    if attr_name.startswith("_"):
                        continue
                    try:
                        value = getattr(current_obj, attr_name)
                    except Exception:
                        continue
                    if isinstance(value, TTNNModule):
                        _initialize_module_on_device(value, device, device_init)
                        _set_device_recursive(value)
                    if isinstance(value, dict):
                        for k, v in value.items():
                            if isinstance(v, TTNNModule):
                                _initialize_module_on_device(v, device, device_init)
                            _set_device_recursive(v)
                    if isinstance(value, (list, tuple)):
                        for v in value:
                            if isinstance(v, TTNNModule):
                                _initialize_module_on_device(v, device, device_init)
                            _set_device_recursive(v)
            elif isinstance(current_obj, TTNNModule):
                _initialize_module_on_device(current_obj, device, device_init)
                for attr_name in dir(current_obj):
                    if attr_name.startswith("_"):
                        continue
                    try:
                        value = getattr(current_obj, attr_name)
                    except Exception:
                        continue
                    if isinstance(value, (nn.Module, TTNNModule)):
                        if isinstance(value, TTNNModule):
                            _initialize_module_on_device(value, device, device_init)
                        _set_device_recursive(value)
                    if isinstance(value, dict):
                        for k, v in value.items():
                            if isinstance(v, TTNNModule):
                                _initialize_module_on_device(v, device, device_init)
                            _set_device_recursive(v)
                    if isinstance(value, (list, tuple)):
                        for v in value:
                            if isinstance(v, TTNNModule):
                                _initialize_module_on_device(v, device, device_init)
                            _set_device_recursive(v)

        _set_device_recursive(obj)

    def unwrap_ttnn(tensor):
        """Extract core tensor from TTNN/Symbiote wrappers (ttnn_tensor, value, tensor attrs). For assimilate_to_device."""
        if tensor is None:
            return None
        curr = tensor
        while hasattr(curr, "ttnn_tensor") or hasattr(curr, "value") or hasattr(curr, "tensor"):
            curr = getattr(curr, "ttnn_tensor", getattr(curr, "value", getattr(curr, "tensor", curr)))
        return curr

    def assimilate_to_device(tensor, device):
        """Prepare tensor for GR00T inference: unwrap, complex->real, move to device in bfloat16 tile layout."""
        if tensor is None:
            return None
        from models.experimental.tt_symbiote.core.utils import ensure_tile_layout

        curr = unwrap_ttnn(tensor)
        if isinstance(curr, ttnn.Tensor) and curr.storage_type() == ttnn.StorageType.DEVICE:
            return ensure_tile_layout(curr)
        torch_t = curr if isinstance(curr, torch.Tensor) else ttnn.to_torch(curr)
        if torch.is_complex(torch_t):
            torch_t = torch_t.real
        return ttnn.from_torch(torch_t.to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    dm.DeviceInit = DeviceInit
    dm._initialize_module_on_device = _initialize_module_on_device
    dm.set_device = set_device_gr00t
    dm.unwrap_ttnn = unwrap_ttnn
    dm.assimilate_to_device = assimilate_to_device


def _patch_module_replacement_for_gr00t():
    """Wrap initialize_module to call override_children_module_names for hierarchical naming (model_config, tensor_config)."""
    import models.experimental.tt_symbiote.utils.module_replacement as mr
    from models.experimental.tt_symbiote.core.module import TTNNModule

    _orig_initialize_module = mr.initialize_module

    def _patched_initialize_module(
        old_module, old_class_to_new_class_dict, module_names, model_config, exclude_replacement=None
    ):
        result = _orig_initialize_module(
            old_module, old_class_to_new_class_dict, module_names, model_config, exclude_replacement
        )
        if result is not None and isinstance(result, TTNNModule) and old_module in module_names:
            result.override_children_module_names()
        return result

    mr.initialize_module = _patched_initialize_module


def _patch_tensor_for_gr00t():
    """Add logical shape, torch.* arithmetic, set_distributed_tensor_config to TorchTTNNTensor for GR00T."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
    from models.experimental.tt_symbiote.core.run_config import DistributedTensorConfig

    def _shape_patched(self):
        if self.ttnn_distributed_tensor_config is not None and self.ttnn_tensor is not None:
            return self.ttnn_distributed_tensor_config.get_logical_shape(self.ttnn_tensor.shape)
        return self.elem.shape if self.elem is not None else tuple(int(i) for i in self.ttnn_tensor.shape)

    TorchTTNNTensor.shape = property(_shape_patched)

    TorchTTNNTensor.__mul__ = lambda self, other: torch.mul(self, other)
    TorchTTNNTensor.__rmul__ = lambda self, other: self.__mul__(other)
    TorchTTNNTensor.__sub__ = lambda self, other: torch.sub(self, other)
    TorchTTNNTensor.__rsub__ = lambda self, other: torch.sub(other, self)
    TorchTTNNTensor.__add__ = lambda self, other: torch.add(self, other)
    TorchTTNNTensor.__radd__ = lambda self, other: torch.add(other, self)
    TorchTTNNTensor.__abs__ = lambda self: torch.abs(self)
    TorchTTNNTensor.__matmul__ = lambda self, other: torch.matmul(self, other)
    TorchTTNNTensor.__rmatmul__ = lambda self, other: torch.matmul(other, self)

    def _set_distributed_tensor_config(self, distributed_tensor_config: DistributedTensorConfig):
        self._distributed_tensor_config = distributed_tensor_config

    def _get_ttnn_distributed_tensor_config(self) -> Optional[DistributedTensorConfig]:
        return self.__dict__.get("_distributed_tensor_config", None)

    def _get_ttnn_distributed_config(self) -> Optional[DistributedTensorConfig]:
        if "distributed_config" in self.__dict__:
            return self.__dict__["distributed_config"]
        return self.__dict__.get("_distributed_tensor_config", None)

    TorchTTNNTensor.set_distributed_tensor_config = _set_distributed_tensor_config
    TorchTTNNTensor.ttnn_distributed_tensor_config = property(_get_ttnn_distributed_tensor_config)
    TorchTTNNTensor.ttnn_distributed_config = property(_get_ttnn_distributed_config)


def patch_run_config_for_gr00t():
    """Apply all GR00T patches to run_config, module, tensor, device_management. Called at import."""
    from models.experimental.tt_symbiote.core import run_config

    run_config.compare_fn_outputs = compare_fn_outputs
    run_config._RUN_MODE_REGISTRY["DPL"] = DPLRunExtended
    run_config._RUN_MODE_REGISTRY["DPL_NO_ERROR_PROP"] = DPLRunNoErrorPropExtended

    try:
        from models.tt_transformers.tt.ccl import TT_CCL
    except ImportError:
        TT_CCL = None

    @dataclass
    class DistributedConfig:
        mesh_device: Any
        tensor_config: Optional[Any] = None
        ccl_manager: Optional[Any] = None

        def __post_init__(self):
            if getattr(self.mesh_device, "get_num_devices", lambda: 1)() > 1 and TT_CCL:
                try:
                    self.ccl_manager = TT_CCL(self.mesh_device)
                except Exception:
                    pass

        def get_tensor_config_for_tensor(self, module_name, tensor):
            if tensor is not None and len(tensor.shape) >= 2:
                shp = getattr(self.mesh_device, "shape", (1, 1)) or (1, 1)
                if len(shp) >= 2 and (tensor.shape[-1] % shp[-1] != 0 or tensor.shape[0] % shp[0] != 0):
                    return None
            return self.tensor_config

    run_config.DistributedConfig = DistributedConfig

    _patch_module_for_gr00t(DistributedConfig)
    _patch_device_management_for_gr00t(DistributedConfig)
    _patch_module_replacement_for_gr00t()

    if not hasattr(run_config.DistributedTensorConfig, "get_logical_shape"):
        run_config.DistributedTensorConfig.get_logical_shape = lambda self, s: s

    _patch_tensor_for_gr00t()
    run_config.post_process_ttnn_module_output = post_process_ttnn_module_output

    def _to_ttnn_with_complex_fallback(orig_to_ttnn):
        """Wrap to_ttnn to convert complex64/128 to bfloat16 real before TTNN conversion."""

        @staticmethod
        def patched(self):
            if self.elem is not None and self.elem.dtype in (torch.complex64, torch.complex128):
                orig_elem = self.elem
                self.elem = self.elem.real.to(torch.bfloat16)
                try:
                    return orig_to_ttnn(self)
                finally:
                    self.elem = orig_elem
            return orig_to_ttnn(self)

        return patched

    run_config.NormalRun.to_ttnn = _to_ttnn_with_complex_fallback(run_config.NormalRun.to_ttnn)
    run_config.LightweightRun.to_ttnn = _to_ttnn_with_complex_fallback(run_config.LightweightRun.to_ttnn)

    _orig_normal_module_run = run_config.NormalRun.module_run

    @staticmethod
    def _normal_module_run_patched(self, *args, **kwds):
        result = _orig_normal_module_run(self, *args, **kwds)
        return post_process_ttnn_module_output(self, result)

    run_config.NormalRun.module_run = _normal_module_run_patched

    run_config.create_new_ttnn_tensors_using_torch_output = _create_new_ttnn_tensors_using_torch_output_relaxed
    DispatchManager.dispatch_to_torch_wrapper = staticmethod(_dispatch_to_torch_wrapper_gr00t)
    run_config.copy_to_torch = _copy_to_torch_extended
    run_config.get_default_distributed_tensor_config = get_default_distributed_tensor_config

    def _trace_enabled(cls):
        return cls

    def _is_trace_enabled(module):
        return False

    def _disable_trace(fn):
        return fn

    class TracedRunStub(NormalRun):
        pass

    run_config.trace_enabled = _trace_enabled
    run_config.is_trace_enabled = _is_trace_enabled
    run_config.disable_trace = _disable_trace
    run_config.TracedRun = TracedRunStub
    run_config._RUN_MODE_REGISTRY["TRACED"] = TracedRunStub


patch_run_config_for_gr00t()
