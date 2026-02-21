# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""GR00T utilities: output comparison, DPL run extensions, and DiT attention compat."""

import os
from typing import Any, Dict, Optional

import torch
import ttnn

from models.experimental.tt_symbiote.core.run_config import (
    NormalRun,
    compose_transforms,
    copy_to_torch,
    copy_to_ttnn,
    create_new_ttnn_tensors_using_torch_output,
    post_process_ttnn_module_output,
    set_device_wrap,
    to_ttnn_wrap,
    tree_map,
    wrap_to_torch_ttnn_tensor,
)
from models.experimental.tt_symbiote.core.utils import TORCH_TO_TTNN, ensure_tile_layout


def compare_fn_outputs(torch_output, ttnn_output, func_name):
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
            or pcc.isnan().any()
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


def get_diag_lm_layer0_capture():
    return _DIAG_LM_LAYER0_CAPTURE


def _filter_kwargs_for_ttnn_conversion(kwargs):
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
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    if isinstance(e, TorchTTNNTensor) and e.elem is not None and e.ttnn_tensor is not None:
        r = TorchTTNNTensor(e.elem)
        r.ttnn_tensor = None
        return r
    return e


def _to_ttnn_lm_attn_style(e, device):
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
    if isinstance(e, ttnn.Tensor):
        return ensure_tile_layout(e)
    if hasattr(e, "ttnn_tensor") and e.ttnn_tensor is not None:
        e.ttnn_tensor = ensure_tile_layout(e.ttnn_tensor)
    return e


def _to_raw_torch(x):
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    if isinstance(x, TorchTTNNTensor):
        return x.to_torch
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, ttnn.Tensor):
        return ttnn.to_torch(x)
    return x


def _dtype_from_torch_args(args, kwds):
    from torch.utils._pytree import tree_flatten

    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    flat, _ = tree_flatten((args, kwds))
    for a in flat:
        if isinstance(a, (TorchTTNNTensor, torch.Tensor)) and a.is_floating_point():
            return a.dtype
    return None


def _cast_module_to_dtype(module, dtype):
    if dtype is None:
        return
    for p in module.parameters():
        if p.is_floating_point():
            p.data = p.data.to(dtype)
    for b in module.buffers():
        if b.is_floating_point():
            b.data = b.data.to(dtype)


def create_new_ttnn_tensors_using_torch_output_dpl(torch_output, ttnn_output, assign_ttnn_to_torch=False):
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    if isinstance(torch_output, (list, tuple)) and isinstance(ttnn_output, TorchTTNNTensor):
        if len(torch_output) > 0 and isinstance(torch_output[0], TorchTTNNTensor):
            torch_output[0].ttnn_tensor = ttnn_output.to_ttnn
            if not assign_ttnn_to_torch:
                torch_output[0].elem = None
            return torch_output

    return create_new_ttnn_tensors_using_torch_output(
        torch_output, ttnn_output, assign_ttnn_to_torch=assign_ttnn_to_torch
    )


class DPLRunExtended(NormalRun):
    @staticmethod
    def module_run(self, *args, **kwds):
        assert (
            self.torch_layer is not None
        ), f"torch_layer must be set for DPLRun, {self} does not have torch_layer set."

        print(f"{self.__class__.__name__}: {self.module_name} on device {self.device}")
        copied_torch_tensors_args = tree_map(copy_to_torch(self.__class__.__name__), args)
        copied_torch_tensors_kwargs = tree_map(copy_to_torch(self.__class__.__name__), kwds)
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
            result = create_new_ttnn_tensors_using_torch_output_dpl(
                torch_output, ttnn_output, assign_ttnn_to_torch=True
            )
        return result


class DPLRunNoErrorPropExtended(NormalRun):
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
        return create_new_ttnn_tensors_using_torch_output_dpl(torch_output, ttnn_output, assign_ttnn_to_torch=True)


def _patched_basic_transformer_block_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    temb: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
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
