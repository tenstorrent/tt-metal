"""Deterministic ttnn code emitter for op-REUSE / op-ADAPT leaves.

Pairs with `op_classifier.py`. Given a component name + the list of
OpDescriptors produced by the classifier, this module emits a partial
native TTNN stub that:

  * Loads every parameter for every op-REUSE / op-ADAPT leaf into a ttnn
    tensor inside `__init__`, named `self.w_<safe_op_name>_<param>`.
  * Exposes one `_apply_<safe_op_name>(self, x, ...)` instance method per
    deterministic leaf, calling the right `ttnn.*` primitive with the
    correct args.
  * Implements `__call__` as a torch fallback against the HF submodule —
    so the smoke test passes immediately and the bring-up loop has a
    component that runs end-to-end on day one.
  * Lists the op-NEW gaps (the things the classifier could NOT template)
    inside a `_LLM_GAPS` constant at the top of the file. The LLM is
    asked to rewrite ONLY `__call__`, using the pre-bound helpers and
    filling in the op-NEW gaps inline.

This shrinks the LLM's unit of work from "port this 200-line transformer
block" to "wire these pre-bound helpers together correctly", which is
the real lever for fast bring-up.
"""

from __future__ import annotations

import json
import re
from textwrap import indent
from typing import Any, Dict, List, Optional, Tuple

from .op_classifier import OpDescriptor, summarize_ops


_SPDX_HEADER = """# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""


_MESH_KW_HELPER = '''def _mesh_kw(device, **kw):
    """Return ``ttnn.from_torch`` kwargs that are mesh-aware.

    On a plain ``ttnn.Device`` this returns the caller's kwargs unchanged plus
    ``device=device`` — byte-identical to the pre-existing single-chip call
    pattern. On a ``ttnn.MeshDevice`` this also inserts
    ``mesh_mapper=ttnn.ReplicateTensorToMesh(device)`` so ``from_torch`` knows
    how to fan the tensor across chips (without this on a MeshDevice, the C++
    upload path stalls or produces incorrect placement).
    """
    kw["device"] = device
    try:
        if isinstance(device, ttnn.MeshDevice):
            kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
    except AttributeError:
        pass
    return kw


'''


_RUNTIME_FALLBACK_HELPER = '''def _log_runtime_fallback(helper, kind, reason):
    """Append a structured CPU-fallback event for the planner reporter.

    Best-effort; never raises and never blocks the test. Writes to
    `<demo_dir>/_runtime_fallbacks.jsonl` (or the path in env
    TT_HW_PLANNER_RUNTIME_FALLBACK_LOG). The planner truncates this file
    before each pytest invocation and consumes it afterwards.
    """
    try:
        import sys as _sys, json as _json, os as _os, time as _time
        from pathlib import Path as _Path
        _sys.stderr.write("[%s_CPU_FALLBACK] %s: %s\\n" % (kind.upper(), helper, reason))
        log_env = _os.environ.get("TT_HW_PLANNER_RUNTIME_FALLBACK_LOG")
        if log_env:
            log_path = _Path(log_env)
        else:
            # _stubs/<safe>.py  ->  demo_dir = parents[1]
            log_path = _Path(__file__).resolve().parents[1] / "_runtime_fallbacks.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        ev = {
            "component": _Path(__file__).stem,
            "helper": helper,
            "kind": kind,
            "reason": reason,
            "ts": _time.time(),
        }
        with log_path.open("a") as f:
            f.write(_json.dumps(ev) + "\\n")
    except Exception:
        pass


'''


def _safe_attr(dotted: str) -> str:
    """Turn a dotted op name (e.g. `blocks.0.attn.q_proj`) into a valid
    python identifier suffix (`blocks_0_attn_q_proj`)."""
    out = re.sub(r"[^0-9A-Za-z_]", "_", dotted)
    if out and out[0].isdigit():
        out = "_" + out
    return out or "root"


def _safe_class_name(component_name: str) -> str:
    parts = re.split(r"[^0-9A-Za-z]+", component_name)
    cls = "".join(p[:1].upper() + p[1:] for p in parts if p)
    return cls or "Component"


def _safe_module_name(component_name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]", "_", component_name) or "component"


def _emit_linear(op: OpDescriptor) -> Tuple[List[str], List[str], str]:
    attr = _safe_attr(op.name)
    in_f = op.params.get("in_features", 0)
    out_f = op.params.get("out_features", 0)
    has_bias = bool(op.params.get("has_bias", False))
    weight_key = f"{op.name}.weight"
    bias_key = f"{op.name}.bias"

    init: List[str] = []
    init.append(f"# op-REUSE: {op.name}  (Linear {in_f} -> {out_f}, bias={has_bias})")
    init.append(
        f"self.w_{attr}_weight = ttnn.from_torch("
        f"sd[{weight_key!r}].T.contiguous(), "
        f"**_mesh_kw(device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))"
    )
    if has_bias:
        init.append(
            f"self.w_{attr}_bias = ttnn.from_torch("
            f"sd[{bias_key!r}].reshape(1, -1), "
            f"**_mesh_kw(device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))"
        )

    apply: List[str] = []
    apply.append(f"def _apply_{attr}(self, x):")
    if has_bias:
        apply.append(f"    return ttnn.linear(x, self.w_{attr}_weight, " f"bias=self.w_{attr}_bias)")
    else:
        apply.append(f"    return ttnn.linear(x, self.w_{attr}_weight)")

    sig = f"self._apply_{attr}(x) -> ttnn.linear  (in={in_f}, out={out_f}, bias={has_bias})"
    return init, apply, sig


def _emit_layer_norm(op: OpDescriptor) -> Tuple[List[str], List[str], str]:
    attr = _safe_attr(op.name)
    eps = op.params.get("eps", 1e-5)
    has_weight = bool(op.params.get("has_weight", True))
    has_bias = bool(op.params.get("has_bias", True))
    norm_shape = op.params.get("normalized_shape") or []

    init: List[str] = []
    init.append(f"# op-REUSE: {op.name}  (LayerNorm {norm_shape}, eps={eps})")
    if has_weight:
        init.append(
            f"self.w_{attr}_weight = ttnn.from_torch("
            f"sd[{op.name + '.weight'!r}], "
            f"**_mesh_kw(device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))"
        )
    if has_bias:
        init.append(
            f"self.w_{attr}_bias = ttnn.from_torch("
            f"sd[{op.name + '.bias'!r}], "
            f"**_mesh_kw(device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))"
        )
    init.append(f"self._eps_{attr} = {float(eps)!r}")

    apply: List[str] = []
    apply.append(f"def _apply_{attr}(self, x):")
    kw = [f"epsilon=self._eps_{attr}"]
    if has_weight:
        kw.append(f"weight=self.w_{attr}_weight")
    if has_bias:
        kw.append(f"bias=self.w_{attr}_bias")
    apply.append(f"    return ttnn.layer_norm(x, {', '.join(kw)})")

    sig = f"self._apply_{attr}(x) -> ttnn.layer_norm  " f"(shape={norm_shape}, eps={eps})"
    return init, apply, sig


def _emit_rms_norm(op: OpDescriptor) -> Tuple[List[str], List[str], str]:
    attr = _safe_attr(op.name)
    eps = op.params.get("eps", 1e-6)
    has_weight = bool(op.params.get("has_weight", True))
    norm_shape = op.params.get("normalized_shape") or []

    init: List[str] = []
    init.append(f"# op-ADAPT: {op.name}  (RMSNorm {norm_shape}, eps={eps})")
    if has_weight:
        init.append(
            f"self.w_{attr}_weight = ttnn.from_torch("
            f"sd[{op.name + '.weight'!r}], "
            f"**_mesh_kw(device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))"
        )
    init.append(f"self._eps_{attr} = {float(eps)!r}")

    apply: List[str] = []
    apply.append(f"def _apply_{attr}(self, x):")
    if has_weight:
        apply.append(f"    return ttnn.rms_norm(x, epsilon=self._eps_{attr}, " f"weight=self.w_{attr}_weight)")
    else:
        apply.append(f"    return ttnn.rms_norm(x, epsilon=self._eps_{attr})")

    sig = f"self._apply_{attr}(x) -> ttnn.rms_norm  (shape={norm_shape}, eps={eps})"
    return init, apply, sig


def _emit_embedding(op: OpDescriptor) -> Tuple[List[str], List[str], str]:
    attr = _safe_attr(op.name)
    n = op.params.get("num_embeddings", 0)
    d = op.params.get("embedding_dim", 0)
    init: List[str] = []
    init.append(f"# op-REUSE: {op.name}  (Embedding {n} x {d})")
    init.append(
        f"self.w_{attr}_weight = ttnn.from_torch("
        f"sd[{op.name + '.weight'!r}], "
        f"**_mesh_kw(device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT))"
    )
    apply: List[str] = []
    apply.append(f"def _apply_{attr}(self, indices):")
    apply.append(f"    return ttnn.embedding(indices, self.w_{attr}_weight)")
    sig = f"self._apply_{attr}(indices) -> ttnn.embedding  (n={n}, dim={d})"
    return init, apply, sig


def _emit_activation(op: OpDescriptor) -> Tuple[List[str], List[str], str]:
    """Emit an activation `_apply_*` helper.

    For variants with a direct ttnn primitive (gelu/silu/relu/sigmoid/tanh)
    we emit the native call. For everything else (GLU, ReLU6, Mish, SwiGLU,
    custom modules, etc.) we ROUND-TRIP through the live torch reference
    submodule by calling the matching submodule on a fp32 CPU tensor and
    converting back. This keeps PCC honest and the bring-up loop unblocked;
    the LLM is told (via palette + contract) that it MAY specialize the
    helper with raw ttnn ops once the rest of the forward path is wired.
    """
    attr = _safe_attr(op.name)
    variant = op.params.get("variant", "")
    init: List[str] = [f"# op-REUSE: {op.name}  ({variant.upper()})"]
    apply: List[str] = []
    if variant in ("gelu", "silu", "relu", "sigmoid", "tanh"):
        apply.append(f"def _apply_{attr}(self, x):")
        apply.append(f"    return ttnn.{variant}(x)")
        sig = f"self._apply_{attr}(x) -> ttnn.{variant}"
    else:
        apply.append(f"def _apply_{attr}(self, x):")
        apply.append(f"    # ACTIVATION_CPU_FALLBACK: variant={variant}; LLM may specialize.")
        apply.append(f"    _log_runtime_fallback('_apply_{attr}', 'activation', 'variant={variant}')")
        apply.append(f"    sub = self._torch_module")
        apply.append(f"    for tok in {op.name!r}.split('.'):")
        apply.append(f"        sub = sub[int(tok)] if tok.isdigit() else getattr(sub, tok)")
        apply.append(f"    t = ttnn.to_torch(x).to(torch.float32)")
        apply.append(f"    with torch.no_grad():")
        apply.append(f"        out_t = sub(t)")
        apply.append(
            f"    return ttnn.from_torch(out_t.to(torch.bfloat16), **_mesh_kw(self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))"
        )
        sig = f"self._apply_{attr}(x)  ({variant})  [CPU fallback; optional to specialize]"
    return init, apply, sig


def _emit_conv2d(op: OpDescriptor) -> Tuple[List[str], List[str], str]:
    """Generate `_apply_<conv2d>` that wraps `ttnn.conv2d` with NCHW autodetect
    and a lazy weight-prepare step (modeled after segformer / sam2 demos).

    Contract: NCHW in, NCHW out. The first call lazily reformats weights via
    `ttnn.prepare_conv_weights` / `prepare_conv_bias` and moves them to
    device; subsequent calls reuse the prepared tensors.

    If the native call fails for any reason, the helper falls back to
    `torch.nn.functional.conv2d` so the bring-up loop is never blocked by a
    ttnn layout edge case (the PCC test still passes, and the LLM can
    specialize the helper later).
    """
    attr = _safe_attr(op.name)
    p = op.params
    in_ch = p.get("in_channels")
    out_ch = p.get("out_channels")
    has_bias = bool(p.get("has_bias"))

    init: List[str] = []
    init.append(
        f"# op-ADAPT: {op.name}  (Conv2d "
        f"{in_ch} -> {out_ch}, "
        f"k={p.get('kernel_size')}, s={p.get('stride')}, p={p.get('padding')}, "
        f"groups={p.get('groups')}, bias={has_bias})"
    )
    init.append(f"# Host-side ttnn tensors; lazily prepared + moved to device on first conv call.")
    init.append(
        f"self.w_{attr}_weight = ttnn.from_torch("
        f"sd[{op.name + '.weight'!r}], "
        f"dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)"
    )
    if has_bias:
        init.append(
            f"self.w_{attr}_bias = ttnn.from_torch("
            f"sd[{op.name + '.bias'!r}].reshape(1, 1, 1, -1), "
            f"dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)"
        )
    init.append(
        f"self._conv2d_{attr}_params = {json.dumps({k: list(v) if isinstance(v, tuple) else v for k, v in p.items() if k != 'has_bias'})}"
    )
    init.append(f"self._conv2d_{attr}_prepared = False")
    init.append(f"# Cached torch weight/bias for the CPU-fallback path (kept in fp32).")
    init.append(f"self._torch_w_{attr}_weight = sd[{op.name + '.weight'!r}].detach().to(torch.float32).contiguous()")
    if has_bias:
        init.append(f"self._torch_w_{attr}_bias = sd[{op.name + '.bias'!r}].detach().to(torch.float32).contiguous()")

    apply: List[str] = []
    apply.append(f"def _apply_{attr}(self, x, *, input_height, input_width):")
    apply.append(f"    p = self._conv2d_{attr}_params")
    apply.append(f"    in_ch = p['in_channels']")
    apply.append(f"    out_ch = p['out_channels']")
    apply.append(f"    batch_size = int(x.shape[0])")
    apply.append(f"    try:")
    apply.append(f"        # NCHW autodetect: ttnn.conv2d wants NHWC.")
    apply.append(f"        x_in = x")
    apply.append(f"        nchw_in = (")
    apply.append(f"            len(x_in.shape) == 4")
    apply.append(f"            and int(x_in.shape[1]) == in_ch")
    apply.append(f"            and int(x_in.shape[-1]) != in_ch")
    apply.append(f"        )")
    apply.append(f"        if nchw_in:")
    apply.append(f"            x_in = ttnn.to_layout(x_in, ttnn.ROW_MAJOR_LAYOUT)")
    apply.append(f"            x_in = ttnn.permute(x_in, (0, 2, 3, 1))")
    apply.append(f"        elif len(x_in.shape) == 4 and int(x_in.shape[-1]) == in_ch:")
    apply.append(f"            x_in = ttnn.to_layout(x_in, ttnn.ROW_MAJOR_LAYOUT)")
    apply.append(f"        else:")
    apply.append(f"            raise RuntimeError(")
    apply.append(f"                '_apply_{attr}: input shape ' + str(tuple(x_in.shape)) +")
    apply.append(f"                ' has no axis matching in_channels=' + str(in_ch))")
    apply.append(f"        conv_config = ttnn.Conv2dConfig(")
    apply.append(f"            weights_dtype=ttnn.bfloat16,")
    apply.append(f"            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,")
    apply.append(f"            deallocate_activation=False,")
    apply.append(f"        )")
    apply.append(f"        compute_config = ttnn.init_device_compute_kernel_config(")
    apply.append(f"            self.device.arch(),")
    apply.append(f"            math_fidelity=ttnn.MathFidelity.LoFi,")
    apply.append(f"            math_approx_mode=True,")
    apply.append(f"            fp32_dest_acc_en=False,")
    apply.append(f"            packer_l1_acc=False,")
    apply.append(f"        )")
    apply.append(f"        conv_kwargs = dict(")
    apply.append(f"            in_channels=in_ch,")
    apply.append(f"            out_channels=out_ch,")
    apply.append(f"            batch_size=batch_size,")
    apply.append(f"            input_height=input_height,")
    apply.append(f"            input_width=input_width,")
    apply.append(f"            kernel_size=tuple(p['kernel_size']),")
    apply.append(f"            stride=tuple(p['stride']),")
    apply.append(f"            padding=tuple(p['padding']),")
    apply.append(f"            dilation=tuple(p.get('dilation') or (1, 1)),")
    apply.append(f"            groups=p.get('groups', 1),")
    apply.append(f"            device=self.device,")
    apply.append(f"            conv_config=conv_config,")
    apply.append(f"        )")
    apply.append(f"        bias = getattr(self, 'w_{attr}_bias', None)")
    apply.append(f"        if not self._conv2d_{attr}_prepared:")
    apply.append(f"            self.w_{attr}_weight = ttnn.prepare_conv_weights(")
    apply.append(f"                weight_tensor=self.w_{attr}_weight,")
    apply.append(f"                weights_format='OIHW',")
    apply.append(f"                input_memory_config=x_in.memory_config(),")
    apply.append(f"                input_layout=x_in.get_layout(),")
    apply.append(f"                has_bias=(bias is not None),")
    apply.append(f"                **conv_kwargs,")
    apply.append(f"                input_dtype=ttnn.bfloat16,")
    apply.append(f"            )")
    apply.append(f"            self.w_{attr}_weight = ttnn.to_device(self.w_{attr}_weight, self.device)")
    apply.append(f"            if bias is not None:")
    apply.append(f"                self.w_{attr}_bias = ttnn.prepare_conv_bias(")
    apply.append(f"                    bias_tensor=bias,")
    apply.append(f"                    input_memory_config=x_in.memory_config(),")
    apply.append(f"                    input_layout=x_in.get_layout(),")
    apply.append(f"                    **conv_kwargs,")
    apply.append(f"                    input_dtype=ttnn.bfloat16,")
    apply.append(f"                )")
    apply.append(f"                self.w_{attr}_bias = ttnn.to_device(self.w_{attr}_bias, self.device)")
    apply.append(f"                bias = self.w_{attr}_bias")
    apply.append(f"            self._conv2d_{attr}_prepared = True")
    apply.append(f"        [out, [out_h, out_w]] = ttnn.conv2d(")
    apply.append(f"            input_tensor=x_in,")
    apply.append(f"            weight_tensor=self.w_{attr}_weight,")
    apply.append(f"            bias_tensor=bias,")
    apply.append(f"            **conv_kwargs,")
    apply.append(f"            compute_config=compute_config,")
    apply.append(f"            return_output_dim=True,")
    apply.append(f"            return_weights_and_bias=False,")
    apply.append(f"            dtype=ttnn.bfloat16,")
    apply.append(f"        )")
    apply.append(f"        if out.memory_config().is_sharded():")
    apply.append(f"            out = ttnn.sharded_to_interleaved(out, ttnn.L1_MEMORY_CONFIG)")
    apply.append(f"        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)")
    apply.append(f"        out = ttnn.reshape(out, (batch_size, out_h, out_w, out_ch))")
    apply.append(f"        if nchw_in:")
    apply.append(f"            out = ttnn.permute(out, (0, 3, 1, 2))")
    apply.append(f"        return out")
    apply.append(f"    except Exception as exc:")
    apply.append(f"        # CONV2D_CPU_FALLBACK: native ttnn.conv2d hit an edge case (sharding /")
    apply.append(f"        # tile alignment / dtype). Drop to torch CPU so bring-up does not stall;")
    apply.append(f"        # the LLM should specialize this helper for native ttnn execution.")
    apply.append(f"        _log_runtime_fallback('_apply_{attr}', 'conv2d', type(exc).__name__ + ': ' + str(exc))")
    apply.append(f"        t = ttnn.to_torch(x).to(torch.float32)")
    apply.append(f"        if t.dim() == 4 and t.shape[-1] == in_ch and t.shape[1] != in_ch:")
    apply.append(f"            t = t.permute(0, 3, 1, 2).contiguous()")
    apply.append(f"        bias_torch = getattr(self, '_torch_w_{attr}_bias', None)")
    apply.append(f"        if bias_torch is not None:")
    apply.append(f"            bias_torch = bias_torch.view(-1)")
    apply.append(f"        out_t = torch.nn.functional.conv2d(")
    apply.append(f"            t, self._torch_w_{attr}_weight,")
    apply.append(f"            bias=bias_torch,")
    apply.append(f"            stride=tuple(p['stride']),")
    apply.append(f"            padding=tuple(p['padding']),")
    apply.append(f"            dilation=tuple(p.get('dilation') or (1, 1)),")
    apply.append(f"            groups=p.get('groups', 1),")
    apply.append(f"        )")
    apply.append(
        f"        return ttnn.from_torch(out_t.to(torch.bfloat16), **_mesh_kw(self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))"
    )

    sig = (
        f"self._apply_{attr}(x, input_height, input_width) -> ttnn.conv2d  "
        f"({in_ch}->{out_ch}, k={p.get('kernel_size')})  [NCHW in/out; auto CPU fallback]"
    )
    return init, apply, sig


_EMITTERS = {
    "linear": _emit_linear,
    "layer_norm": _emit_layer_norm,
    "rms_norm": _emit_rms_norm,
    "embedding": _emit_embedding,
    "activation": _emit_activation,
    "conv2d": _emit_conv2d,
}


def emit_partial_stub(
    *,
    component_name: str,
    model_id: str,
    hf_reference: str,
    submodule_candidates: List[str],
    ops: List[OpDescriptor],
) -> Tuple[str, Dict[str, Any]]:
    """Produce the source of a partial native TTNN stub for `component_name`.

    Returns (source_text, manifest) where `manifest` summarizes what was
    pre-bound (so the loop / prompt can reference it).
    """
    cls_name = _safe_class_name(component_name)
    mod_name = _safe_module_name(component_name)

    init_blocks: List[str] = []
    apply_blocks: List[str] = []
    palette: List[str] = []
    pre_bound: List[Dict[str, Any]] = []
    gaps: List[Dict[str, Any]] = []

    for op in ops:
        if op.status in ("op-REUSE", "op-ADAPT") and op.kind in _EMITTERS:
            init_lines, apply_lines, sig = _EMITTERS[op.kind](op)
            init_blocks.append("\n".join(init_lines))
            apply_blocks.append("\n".join(apply_lines))
            palette.append(sig)
            pre_bound.append(
                {
                    "name": op.name,
                    "class": op.torch_class,
                    "kind": op.kind,
                    "ttnn_target": op.ttnn_target,
                    "helper": f"_apply_{_safe_attr(op.name)}",
                }
            )
        else:
            gaps.append(
                {
                    "name": op.name,
                    "class": op.torch_class,
                    "notes": op.notes,
                }
            )

    if not init_blocks:
        init_body = "        pass"
    else:
        init_body = indent("\n\n".join(init_blocks), "        ")

    apply_body = "\n\n".join(indent(block, "    ") for block in apply_blocks)

    palette_text = "\n".join(f"#   {line}" for line in palette) or "#   (none — every op was op-NEW)"
    gap_text = (
        "\n".join(f"#   - {g['name']}  ({g['class']})" + (f"  [{g['notes']}]" if g["notes"] else "") for g in gaps)
        or "#   (none — fully deterministic, no LLM_GAPs)"
    )

    summary = summarize_ops(ops)
    counts = summary["counts"]

    parts: List[str] = []
    parts.append(_SPDX_HEADER)
    parts.append(
        '"""Op-level partial TTNN port for `{name}` of {mid}.\n\n'
        "Generated deterministically by `tt_hw_planner op-synth`.\n"
        "Weight loading and op-REUSE/op-ADAPT helpers below are\n"
        "machine-emitted from the HF reference and DO NOT need LLM\n"
        "review. The `__call__` implementation falls back to HF\n"
        "torch so the bring-up smoke test passes immediately;\n"
        "the LLM's only remaining job is to rewrite `__call__`\n"
        "to call the pre-bound `_apply_*` helpers in the right\n"
        "order and fill any op-NEW gaps inline.\n\n"
        "Pre-bound deterministic helpers (op palette):\n"
        "{palette}\n\n"
        "LLM_GAPs (op-NEW — still need synthesis):\n"
        "{gaps}\n\n"
        "HF reference: {ref}\n"
        "Op counts: total={total}  op-REUSE={r}  op-ADAPT={a}  op-NEW={n}"
        '"""\n'.format(
            name=component_name,
            mid=model_id,
            palette=palette_text,
            gaps=gap_text,
            ref=hf_reference or "(none)",
            total=summary["total"],
            r=counts.get("op-REUSE", 0),
            a=counts.get("op-ADAPT", 0),
            n=counts.get("op-NEW", 0),
        )
    )
    parts.append("from __future__ import annotations\n")
    parts.append("import torch\n")
    parts.append("import ttnn\n")
    parts.append("import transformers\n\n")

    parts.append(f"HF_MODEL_ID = {model_id!r}\n")
    parts.append(f"_CANDIDATE_SUBMODULE_PATHS = {submodule_candidates!r}\n\n")
    parts.append(_MESH_KW_HELPER)
    parts.append(_RUNTIME_FALLBACK_HELPER)
    parts.append("_LLM_GAPS = [\n" + "".join(f"    {g!r},\n" for g in gaps) + "]\n\n")

    parts.append(
        "def _resolve(obj, dotted):\n"
        "    cur = obj\n"
        '    for tok in dotted.replace("[", ".").replace("]", "").split("."):\n'
        '        if tok == "":\n'
        "            continue\n"
        "        if tok.isdigit():\n"
        "            cur = cur[int(tok)]\n"
        "        else:\n"
        "            cur = getattr(cur, tok)\n"
        "    return cur\n\n"
    )

    from .bringup_loop import _FALLBACK_COERCE_TO_TORCH

    parts.append(_FALLBACK_COERCE_TO_TORCH + "\n")

    parts.append(f"class {cls_name}:\n")
    parts.append("    def __init__(self, device, torch_module):\n")
    parts.append("        self.device = device\n")
    parts.append("        self._torch_module = torch_module\n")
    parts.append("        sd = torch_module.state_dict()\n")
    parts.append(init_body + "\n\n")

    if apply_body:
        parts.append(apply_body + "\n\n")

    parts.append("    def __call__(self, *args, **kwargs):\n")
    parts.append("        # LLM_GAP: rewrite this method to use the pre-bound `_apply_*`\n")
    parts.append("        # helpers above and the ttnn primitives. Until then, fall\n")
    parts.append("        # back to the HF torch module so the smoke test still passes.\n")
    parts.append("        args = tuple(_coerce_to_torch(a) for a in args)\n")
    parts.append("        kwargs = {k: _coerce_to_torch(v) for k, v in kwargs.items()}\n")
    parts.append("        return self._torch_module(*args, **kwargs)\n\n")

    parts.append("def build(device, torch_module):\n")
    parts.append(f"    return {cls_name}(device, torch_module)\n\n")

    parts.append("_instance = None\n\n")
    parts.append(f"def {mod_name}(*args, **kwargs):\n")
    parts.append("    global _instance\n")
    parts.append("    if _instance is None:\n")
    parts.append(
        "        model = transformers.AutoModel.from_pretrained("
        'HF_MODEL_ID, trust_remote_code=True, torch_dtype="bfloat16", low_cpu_mem_usage=True)\n'
    )
    parts.append("        model.eval()\n")
    parts.append("        torch_sub = None\n")
    parts.append("        for path in _CANDIDATE_SUBMODULE_PATHS:\n")
    parts.append("            try:\n")
    parts.append("                torch_sub = _resolve(model, path)\n")
    parts.append("                break\n")
    parts.append("            except (AttributeError, IndexError, KeyError, TypeError):\n")
    parts.append("                continue\n")
    parts.append("        if torch_sub is None:\n")
    parts.append(f"            raise RuntimeError('partial-stub: could not resolve `{component_name}`')\n")
    parts.append("        _instance = build(ttnn.open_device(device_id=0), torch_sub)\n")
    parts.append("    return _instance(*args, **kwargs)\n")

    manifest = {
        "component": component_name,
        "class_name": cls_name,
        "module_function": mod_name,
        "pre_bound": pre_bound,
        "llm_gaps": gaps,
        "palette": palette,
        "counts": counts,
    }
    return "".join(parts), manifest
