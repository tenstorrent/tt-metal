# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gate helpers shared by the e2e test and the demo.

Gate 1 (native): no torch compute op in the forward code of a routed graduated stub or of the chain.
Weight preparation (__init__ / build / weight-layout helpers) and host input metadata are exempt; the
runtime counterpart is host_op_selftest(), which must see zero non-benign aten ops in the forward.
Gate 2 (invoked): every graduated module's counter moved during the real forward.
"""
from __future__ import annotations

import ast
import inspect

# torch compute (strict TT-only contract, section B); shape / dtype prep is allowed
FORBIDDEN_TORCH = {
    "matmul",
    "mm",
    "bmm",
    "einsum",
    "softmax",
    "log_softmax",
    "layer_norm",
    "rms_norm",
    "batch_norm",
    "group_norm",
    "embedding",
    "embedding_bag",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "scaled_dot_product_attention",
    "relu",
    "gelu",
    "silu",
    "tanh",
    "sigmoid",
    "leaky_relu",
    "argmax",
    "topk",
    "multinomial",
    "dropout",
}
# functions that run once at build time or encode inputs on host (not the forward)
EXEMPT_FUNCS = {
    "__init__",
    "build",
    "_prepare_torch_state",
    "around",
    "around_resample",
    "resident",
    "attach_block_ports",
    "walk",
    "rope_tables",
    "mask",
    "consts",
    "_vision_metadata",
    "_one_hot",
    "block_mask",
    "pad_rows",
    "text_attention_mask",
    "mrope_tables",
    "rotate_half_matrix",
    "upload",
    "upload_rows",
    "_replicated",
    "_sharded",
    "_as_tt",
    "_to_tt",
    "_t",
    "_pad_cols",
    "_pad_rows",
    "_symmetric_padding",
    "prepare",
    "_position_embeddings",
    "_ensure_w32",
    "_fused",
    "_b",
    "lin",
    "col",
    "w_col",
    "w_row",
    "norm_w",
    "_act",
    "_mesh_shape",
    "mesh_shape",
    "shard_mapper",
    "replicate",
    "hifi4_config",
    "pad_to_tile",
}


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def torch_compute_calls(module):
    """[(function, line, call)] of forbidden torch compute calls in a module's forward code."""
    src = inspect.getsource(module)
    tree = ast.parse(src)
    hits = []

    def visit(fn, stack):
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                name = _dotted(node.func)
                head, _, tail = name.rpartition(".")
                if not head:
                    continue
                if head in ("torch", "torch.nn.functional", "F", "torch.nn.functional") and tail in FORBIDDEN_TORCH:
                    hits.append((stack, node.lineno, name))
                elif head in ("F", "torch.nn.functional"):
                    hits.append((stack, node.lineno, name))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name not in EXEMPT_FUNCS:
            visit(node, node.name)
    # de-duplicate (nested functions are walked by their parents too)
    return sorted(set(hits), key=lambda h: h[1])


def gate1_native(stub_modules, chain_modules=()):
    """{module name: [violations]} over the graduated stub modules and the chain modules."""
    out = {}
    for m in list(stub_modules) + list(chain_modules):
        v = torch_compute_calls(m)
        if v:
            out[m.__name__] = v
    return out


def gate2_invoked(tracker, names):
    counts = tracker.snapshot()
    return {n: int(counts.get(n, 0)) for n in names}, [n for n in names if counts.get(n, 0) == 0]
